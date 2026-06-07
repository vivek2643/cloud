"""
Chat-turn runner: the single canonical implementation of the conversational
edit pipeline. Both the synchronous /chat endpoint and the async streaming
path call run_chat_turn().

It emits coarse progress phases and checks a cancel predicate between steps,
so the same code powers a blocking request OR a streamed, cancellable turn.

Phases (current single-Claude-call flow):
    retrieving -> reasoning -> persisting -> rendering -> done

The phase vocabulary is forward-compatible: when a turn grows into multiple
Claude calls (e.g. plan -> draft -> critique -> revise), each sub-step just
emits its own `reasoning` phase with a finer label and an interpolated pct.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from app.services import audit_log
from app.services.edl import renders_store
from app.services.edl import store as edl_store
from app.services.l3 import claude_editor
from app.services.l3.query_executor import (
    fetch_candidates_by_shot_ids,
    retrieve_for_chat,
)
from app.services.l3.xml_builder import build_fcp7_xml
from app.services.r2 import generate_presigned_get

logger = logging.getLogger(__name__)


class TurnCancelled(Exception):
    """Raised when a cancel was requested at a cooperative checkpoint."""


@dataclass
class ChatTurnInput:
    messages: List[Dict[str, Any]]
    file_ids: Optional[List[str]] = None
    folder_id: Optional[str] = None
    sequence_name: str = "AI Rough Cut"
    fps: int = 24
    # Chronological catalog budget. Larger than the old top-K default so the
    # editor can read the whole story when footage fits; downsampled above this.
    catalog_size: int = 120
    duration_target_s: Optional[int] = None


# emit(phase, pct, label) ; should_cancel() -> bool ; on_lineage(project_id, edl_version_id, render_id)
EmitFn = Callable[[str, int, str], None]
CancelFn = Callable[[], bool]
LineageFn = Callable[[Optional[str], Optional[str], Optional[str]], None]


def _noop_emit(phase: str, pct: int, label: str) -> None:  # pragma: no cover
    return None


def _never_cancel() -> bool:  # pragma: no cover
    return False


def _pathurl_for(clip) -> str:
    """Presigned HTTPS GET for the source media (NLE re-link friendly)."""
    key = clip.file_r2_proxy_key or clip.file_r2_key
    return generate_presigned_get(key, expires_in=86400)


def _file_ids_from_candidates(candidates) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for c in candidates:
        if c.file_id and c.file_id not in seen:
            seen.add(c.file_id)
            out.append(c.file_id)
    return out


def _file_ids_from_timeline(timeline: List[Dict[str, Any]]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for c in timeline:
        fid = c.get("file_id")
        if fid and fid not in seen:
            seen.add(fid)
            out.append(str(fid))
    return out


def _serialize_candidates(cands: list) -> list[dict]:
    return [
        {
            "shot_id": c.shot_id,
            "file_id": c.file_id,
            "file_name": c.file_name,
            "shot_index": c.shot_index,
            "start_ms": c.start_ms,
            "end_ms": c.end_ms,
            "duration_ms": c.end_ms - c.start_ms,
            "score": c.score,
            "intra_shot_variance": getattr(c, "intra_shot_variance", None),
            "peak_motion_ms": getattr(c, "peak_motion_ms", None),
            "blur_min": getattr(c, "blur_min", None),
        }
        for c in cands[:50]
    ]


def _serialize_clip(t) -> dict:
    return {
        "file_name": t.file_name,
        "source_in_ms": t.source_in_ms,
        "source_out_ms": t.source_out_ms,
        "duration_ms": t.source_out_ms - t.source_in_ms,
        "timeline_start_ms": t.timeline_start_ms,
        "timeline_end_ms": t.timeline_end_ms,
        "score": t.score,
        "trimmed_around_ms": getattr(t, "trimmed_around_ms", None),
    }


def _enqueue_render(render_id: str) -> bool:
    """Defer a render task to the procrastinate worker. Marks the render failed
    (so the UI stops polling) if enqueue is impossible."""
    try:
        from app.services.jobs import app as proc_app, register_tasks
        register_tasks()
        with proc_app.open():
            proc_app.tasks["render_edl"].defer(render_id=render_id)
        return True
    except Exception:
        logger.exception("Could not enqueue render %s", render_id)
        try:
            renders_store.update_status(
                render_id,
                status="failed",
                error="Worker unavailable: could not enqueue render job.",
            )
        except Exception:
            logger.exception("Also failed to mark render %s as failed", render_id)
        return False


def _check_cancel(should_cancel: CancelFn) -> None:
    if should_cancel():
        raise TurnCancelled()


def run_chat_turn(
    inp: ChatTurnInput,
    user_id: str,
    emit: EmitFn = _noop_emit,
    should_cancel: CancelFn = _never_cancel,
    on_lineage: Optional[LineageFn] = None,
) -> Dict[str, Any]:
    """
    Run one conversational edit turn. Returns a dict shaped like ChatResponse.
    Raises TurnCancelled if a cancel was requested at a checkpoint.
    """
    if not inp.messages or inp.messages[-1].get("role") != "user":
        raise ValueError("The last message in the history must be a user message.")
    latest = (inp.messages[-1].get("content") or "").strip()
    if not latest:
        raise ValueError("Latest user message is empty.")

    file_ids = list(dict.fromkeys([str(fid) for fid in (inp.file_ids or []) if fid]))

    audit = audit_log.open_edit_log(latest)
    audit.set("user_id", user_id)
    audit.set("folder_id", inp.folder_id)
    audit.set("file_ids", file_ids or None)
    audit.set("mode", "chat")
    audit.set("turns_in_history", len(inp.messages))

    try:
        _check_cancel(should_cancel)

        # 1-3. Director: plan -> fill (recipes/composer) -> critique -> re-plan
        emit("retrieving", 8, "Loading footage")
        from app.services.l3 import director as director_mod

        dres = director_mod.direct_edit(
            user_id=user_id,
            messages=inp.messages,
            file_ids=file_ids or None,
            folder_id=inp.folder_id,
            duration_target_s=inp.duration_target_s,
            fps=inp.fps,
            emit=emit,
        )
        audit.stage("director_plan", dres.plan)
        audit.stage("director_critiques", dres.critiques)
        audit.stage("director_sections", dres.sections)
        audit.stage("director_perception", dres.raw.get("perception"))
        audit.stage("editor_reasoning", dres.reasoning)
        audit.stage("editor_warnings", dres.warnings)

        timeline = dres.timeline
        has_av = bool(dres.edl and (dres.edl.get("video_track") or dres.edl.get("audio_track")))
        if not has_av:
            audit.succeed()
            empty = {
                "timeline": [],
                "fcp7_xml": "",
                "total_duration_ms": 0,
                "reasoning": dres.reasoning or "No timeline could be built from the available footage.",
                "warnings": dres.warnings or ["Empty timeline."],
                "catalog_size": 0,
                "sections": [],
                "project_id": None,
                "edl_version_id": None,
                "render_id": None,
            }
            emit("done", 100, "No usable footage")
            return empty

        actual_ms = dres.total_ms
        audit.stage("timeline", timeline)
        audit.stage("timeline_summary", {"clip_count": len(timeline), "actual_duration_ms": actual_ms})
        emit("persisting", 80, f"{len(timeline)} clips")

        _check_cancel(should_cancel)

        # Persist EDL v2 + enqueue render ------------------------------
        project_id: Optional[str] = None
        edl_version_id: Optional[str] = None
        render_id: Optional[str] = None
        try:
            project = edl_store.find_or_create_default_project(
                user_id=user_id,
                source_file_ids=file_ids if file_ids else _file_ids_from_timeline(timeline),
            )
            project_id = project["id"]
            parent = edl_store.get_latest_edl_version(project_id)
            new_version = edl_store.write_edl_version(
                project_id=project_id,
                edl_json=dres.edl,
                author_kind="claude",
                parent_id=parent["id"] if parent else None,
                commit_msg=latest[:200],
            )
            edl_version_id = new_version["id"]
            audit.set("project_id", project_id)
            audit.set("edl_version_id", edl_version_id)

            if timeline:
                emit("rendering", 90, "Starting render")
                preset = "preview_vertical" if dres.vertical else "preview"
                render_row = renders_store.create_render(edl_version_id, preset=preset)
                render_id = render_row["id"]
                _enqueue_render(render_id)
                audit.set("render_id", render_id)
            if on_lineage:
                on_lineage(project_id, edl_version_id, render_id)
        except Exception as e:
            logger.exception("EDL persistence / render enqueue failed; turn still returns.")
            audit.set("edl_persist_error", f"{type(e).__name__}: {e}")

        audit.succeed()

        payload = {
            "timeline": timeline,
            "fcp7_xml": "",
            "total_duration_ms": actual_ms,
            "reasoning": dres.reasoning,
            "warnings": dres.warnings,
            "catalog_size": len(timeline),
            "sections": dres.sections,
            "project_id": project_id,
            "edl_version_id": edl_version_id,
            "render_id": render_id,
        }
        emit("done", 100, "Done")
        return payload

    except TurnCancelled:
        audit.set("cancelled", True)
        audit.fail("cancelled")
        raise
    except Exception as e:
        logger.exception("Chat turn failed")
        audit.fail(f"{type(e).__name__}: {e}")
        raise
