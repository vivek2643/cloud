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

        # 1. Retrieval --------------------------------------------------
        emit("retrieving", 8, "Searching your footage")
        primary = retrieve_for_chat(
            user_id=user_id,
            prompt=latest,
            folder_id=inp.folder_id,
            file_ids=file_ids or None,
            cap=inp.catalog_size,
        )
        prior_shot_ids: List[str] = []
        for m in inp.messages:
            if m.get("role") != "assistant" or not m.get("timeline"):
                continue
            for c in m["timeline"]:
                sid = c.get("shot_id") if isinstance(c, dict) else None
                if sid:
                    prior_shot_ids.append(str(sid))
        prior_shot_ids = list(dict.fromkeys(prior_shot_ids))
        pinned = (
            fetch_candidates_by_shot_ids(
                user_id=user_id, shot_ids=prior_shot_ids, file_ids=file_ids or None
            )
            if prior_shot_ids
            else []
        )

        seen: set[str] = set()
        candidates = []
        for c in list(pinned) + list(primary):
            if c.shot_id in seen:
                continue
            seen.add(c.shot_id)
            candidates.append(c)
        audit.set("catalog_size", len(candidates))
        audit.stage("candidates_l1_only", _serialize_candidates(candidates))
        emit("retrieving", 28, f"{len(candidates)} candidate shots")

        if not candidates:
            audit.succeed()
            empty = {
                "timeline": [],
                "fcp7_xml": build_fcp7_xml(inp.sequence_name, [], _pathurl_for, inp.fps),
                "total_duration_ms": 0,
                "reasoning": "No indexed shots are available yet -- upload and index footage first.",
                "warnings": ["Empty catalog."],
                "catalog_size": 0,
                "project_id": None,
                "edl_version_id": None,
                "render_id": None,
            }
            emit("done", 100, "No footage available")
            return empty

        _check_cancel(should_cancel)

        # 2. Claude reasoning ------------------------------------------
        emit("reasoning", 35, "Editor is thinking")
        result = claude_editor.compile_timeline_chat(
            history=inp.messages,
            candidates=candidates,
            duration_target_s=inp.duration_target_s,
            emit=emit,
        )
        audit.stage("editor_user_message", result.user_text)
        audit.stage("editor_raw_response", result.raw_response)
        audit.stage("editor_reasoning", result.reasoning)
        audit.stage("editor_warnings", result.warnings)
        audit.stage("editor_post_processing", result.post_processing)

        timeline = result.timeline
        actual_ms = timeline[-1].timeline_end_ms if timeline else 0
        audit.stage("timeline", [_serialize_clip(t) for t in timeline])
        audit.stage(
            "timeline_summary",
            {
                "clip_count": len(timeline),
                "actual_duration_ms": actual_ms,
                "actual_duration_s": round(actual_ms / 1000.0, 2),
            },
        )
        emit("reasoning", 68, f"{len(timeline)} clips selected")

        fcp7 = build_fcp7_xml(inp.sequence_name, timeline, _pathurl_for, inp.fps)
        audit.set("fcp7_xml_chars", len(fcp7))

        _check_cancel(should_cancel)

        # 3. Persist EDL version + 4. enqueue render -------------------
        project_id: Optional[str] = None
        edl_version_id: Optional[str] = None
        render_id: Optional[str] = None
        try:
            emit("persisting", 78, "Saving timeline")
            project = edl_store.find_or_create_default_project(
                user_id=user_id,
                source_file_ids=file_ids if file_ids else _file_ids_from_candidates(candidates),
            )
            project_id = project["id"]
            parent = edl_store.get_latest_edl_version(project_id)
            edl_json = edl_store.edl_from_timeline_clips(timeline, fps=30)
            new_version = edl_store.write_edl_version(
                project_id=project_id,
                edl_json=edl_json,
                author_kind="claude",
                parent_id=parent["id"] if parent else None,
                commit_msg=latest[:200],
            )
            edl_version_id = new_version["id"]
            audit.set("project_id", project_id)
            audit.set("edl_version_id", edl_version_id)

            if timeline:
                emit("rendering", 88, "Starting render")
                render_row = renders_store.create_render(edl_version_id, preset="preview")
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
            "timeline": [
                {
                    "file_id": t.file_id,
                    "file_name": t.file_name,
                    "source_in_ms": t.source_in_ms,
                    "source_out_ms": t.source_out_ms,
                    "timeline_start_ms": t.timeline_start_ms,
                    "timeline_end_ms": t.timeline_end_ms,
                    "score": t.score,
                    "shot_id": getattr(t, "shot_id", None),
                    "role_in_edit": getattr(t, "role_in_edit", None),
                    "why": getattr(t, "why", None),
                }
                for t in timeline
            ],
            "fcp7_xml": fcp7,
            "total_duration_ms": actual_ms,
            "reasoning": result.reasoning,
            "warnings": result.warnings,
            "catalog_size": len(candidates),
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
