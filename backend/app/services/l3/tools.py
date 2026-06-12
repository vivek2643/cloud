"""
The orchestrator's tool belt: neutral tool specs + the executor that binds them
to one editing session.

Contract (the two-brain split): every tool here is deterministic. The model
chooses WHICH tool with WHAT intent; the engine decides exact frames and
reports objective numbers back. Tool results are compact JSON strings -- they
re-enter the model's context every iteration, so brevity is a cost feature.

`ask_user` and `finalize` are *terminal* tools: the loop runner watches for
them and ends the run (pausing or completing the thread) instead of looping.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.l3 import engine, score_span
from app.services.l3.catalog import ClipSummary
from app.services.l3.engine import ClipGrids
from app.services.l3.takes import TakeGroup

logger = logging.getLogger(__name__)

TERMINAL_TOOLS = {"ask_user", "finalize"}


# --------------------------------------------------------------------------
# Session: the working state one agent run mutates
# --------------------------------------------------------------------------

@dataclass
class EditSession:
    thread_id: str
    file_ids: List[str]
    catalog: List[ClipSummary]
    document: Dict[str, Any] = field(default_factory=dict)
    take_groups: List[TakeGroup] = field(default_factory=list)
    _grids: Dict[str, ClipGrids] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.document.setdefault("brief", {})
        self.document.setdefault("spine", None)
        self.document.setdefault("outline", [])
        self.document.setdefault("timeline", [])
        self.document.setdefault("open_questions", [])
        self.document.setdefault("diagnostics", {})

    def grids(self, file_id: str) -> ClipGrids:
        if file_id not in self._grids:
            self._grids[file_id] = engine.load_grids(file_id)
        return self._grids[file_id]

    def grids_by_file(self) -> Dict[str, ClipGrids]:
        for fid in {s["file_id"] for s in self.document["timeline"]}:
            self.grids(fid)
        return self._grids

    def find_segment(self, seg_id: str) -> Optional[dict]:
        for s in self.document["timeline"]:
            if s["seg_id"] == seg_id:
                return s
        return None


# --------------------------------------------------------------------------
# Tool specs (neutral; match Anthropic's tool shape)
# --------------------------------------------------------------------------

def _spec(name: str, description: str, properties: dict, required: List[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


_AXIS = {
    "type": "string",
    "enum": ["speech", "action", "music", "visual", "any"],
    "description": "which cut-cost channels dominate at this seam",
}

TOOL_SPECS: List[dict] = [
    _spec(
        "read_clip",
        "Full footage log for one clip: the L2 perception document (events, "
        "persons, reactions, gaze, camera craft, speaking spans) plus duration. "
        "Call this before using a clip's material; the catalog is only a teaser.",
        {"file_id": {"type": "string"}},
        ["file_id"],
    ),
    _spec(
        "query_seams",
        "Ranked clean cut candidates near a timestamp in one clip, from the "
        "deterministic cost grids (0=ideal..1=forbidden, dirty>0.45). Use to "
        "scout where a beat can start/end cleanly before adding a segment.",
        {
            "file_id": {"type": "string"},
            "around_ms": {"type": "integer"},
            "axis": _AXIS,
            "window_ms": {"type": "integer", "description": "search radius, default 2000"},
        },
        ["file_id", "around_ms"],
    ),
    _spec(
        "set_brief",
        "Record your interpretation of the user's brief (goal, target duration, "
        "tone, platform, constraints). Defaults you assumed belong in "
        "`assumptions` so they can become questions.",
        {
            "goal": {"type": "string"},
            "target_duration_s": {"type": "number"},
            "tone": {"type": "string"},
            "platform": {"type": "string"},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "assumptions": {"type": "array", "items": {"type": "string"}},
        },
        ["goal"],
    ),
    _spec(
        "set_spine",
        "Declare the EDIT SPINE before building the timeline: the load-bearing "
        "through-line every other choice serves. It names, per time-ordered "
        "region, which channel is LOCKED (irreplaceable) vs FREE (coverable / "
        "scoreable). Decoupling A/V is the privileged move -- default to "
        "kind='sync' (both locked, atomic) unless the brief or footage justifies "
        "freeing a channel. kinds: dialogue (audio locked, video free -> B-roll "
        "covers picture, cut on dialogue seams); music (music bed locked, video "
        "free -> coverage cut to beats/sections); visual (VIDEO locked -- "
        "on-screen text / demo / reveal / performance -- audio free to score, cut "
        "on action/visual); sync (BOTH locked -- punchline+face, sync-sound; do "
        "not split); other (escape hatch -- set label + locked_channels). Mark "
        "do-not-cover spans (on-screen text, key reveals) as protected_windows. "
        "One region for most edits; multiple only when the edit shifts mode "
        "(e.g. montage hook -> testimonial).",
        {
            "regions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["dialogue", "music", "visual", "sync", "other"],
                        },
                        "label": {
                            "type": "string",
                            "description": "human label; REQUIRED when kind='other'",
                        },
                        "locked_channels": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["video", "audio"]},
                            "description": "irreplaceable channels. dialogue/music=[audio]; "
                                           "visual=[video]; sync=[video,audio]",
                        },
                        "source_file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "clip(s)/track forming this region's spine "
                                           "(the VO/interview clip, or the music file)",
                        },
                        "protected_windows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_id": {"type": "string"},
                                    "start_ms": {"type": "integer"},
                                    "end_ms": {"type": "integer"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["file_id", "start_ms", "end_ms"],
                            },
                            "description": "do-not-cover spans inside an otherwise-free region",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "why this spine, citing the brief + footage evidence",
                        },
                    },
                    "required": ["kind", "locked_channels", "rationale"],
                },
            }
        },
        ["regions"],
    ),
    _spec(
        "set_outline",
        "Replace the beat outline (the narrative skeleton). Keep 2-6 beats; "
        "each maps to >=1 timeline segments later.",
        {
            "beats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "beat_id": {"type": "string"},
                        "purpose": {"type": "string", "description": "hook/setup/payoff/outro/..."},
                        "intent": {"type": "string"},
                        "target_s": {"type": "number"},
                    },
                    "required": ["beat_id", "purpose", "intent"],
                },
            }
        },
        ["beats"],
    ),
    _spec(
        "add_segment",
        "Add a span of one clip to the timeline. You give rough in/out; the "
        "engine snaps both ends to the cleanest nearby seams and returns exact "
        "frames + costs. position=-1 appends.",
        {
            "file_id": {"type": "string"},
            "in_ms": {"type": "integer", "description": "rough in point; will be snapped"},
            "out_ms": {"type": "integer", "description": "rough out point; will be snapped"},
            "axis": _AXIS,
            "beat_id": {"type": "string"},
            "content": {"type": "string", "description": "what is on screen in this span"},
            "rationale": {"type": "string", "description": "why this material, here"},
            "priority": {"type": "integer", "description": "1=never trim .. 5=trim first (default 3)"},
            "position": {"type": "integer", "description": "index in timeline; -1/omit = append"},
        },
        ["file_id", "in_ms", "out_ms"],
    ),
    _spec(
        "update_segment",
        "Re-cut an existing segment (new rough in/out get re-snapped) and/or "
        "update its metadata.",
        {
            "seg_id": {"type": "string"},
            "in_ms": {"type": "integer"},
            "out_ms": {"type": "integer"},
            "content": {"type": "string"},
            "rationale": {"type": "string"},
            "priority": {"type": "integer"},
            "beat_id": {"type": "string"},
        },
        ["seg_id"],
    ),
    _spec(
        "remove_segment",
        "Delete a segment from the timeline.",
        {"seg_id": {"type": "string"}},
        ["seg_id"],
    ),
    _spec(
        "move_segment",
        "Reorder: move a segment to a new index in the timeline.",
        {"seg_id": {"type": "string"}, "new_index": {"type": "integer"}},
        ["seg_id", "new_index"],
    ),
    _spec(
        "timeline_status",
        "Objective health report of the current timeline: total duration, "
        "per-segment durations and seam costs, jump-cut/short-segment warnings. "
        "Call after edits and before finalize.",
        {},
        [],
    ),
    _spec(
        "fit_duration",
        "Deterministically trim the timeline to a target duration: shrinks "
        "lowest-priority segments first, moving out-points onto clean seams. "
        "Only shrinks -- if under target, add material yourself.",
        {
            "target_s": {"type": "number"},
            "tolerance_ms": {"type": "integer", "description": "default 500"},
        },
        ["target_s"],
    ),
    _spec(
        "compare_takes",
        "When the same content was delivered more than once (see TAKE GROUPS in "
        "the catalog), get an objective, span-level scorecard for every competing "
        "take so you can choose the best one. Returns per-take metrics (word pace, "
        "fillers, pauses, gaze-to-camera, loudness) plus the perception's localized "
        "quality notes (energy/fluency/naturalness/technical). The choice is YOURS: "
        "weight the metrics by the brief (polished ad -> fluency; raw/authentic -> "
        "energy, stop penalizing imperfection), then add_segment the winner's span.",
        {"group_id": {"type": "string", "description": "a take-group id from the catalog, e.g. 'tg1'"}},
        ["group_id"],
    ),
    _spec(
        "ask_user",
        "Pause and ask the user. Use for genuine forks (length, ending, tone, "
        "include/exclude) -- not for things the footage answers. ALWAYS have a "
        "complete draft on the timeline before calling this; every question "
        "needs a default so the draft stands if the user never answers.",
        {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "q_id": {"type": "string"},
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "default": {"type": "string"},
                        "why": {"type": "string", "description": "what hinges on this"},
                    },
                    "required": ["q_id", "question", "default"],
                },
            }
        },
        ["questions"],
    ),
    _spec(
        "finalize",
        "Complete this run: attach a human-readable summary of the plan and "
        "what you chose/assumed. The current timeline becomes the new document "
        "version shown to the user.",
        {
            "summary": {"type": "string"},
            "notes": {"type": "array", "items": {"type": "string"},
                      "description": "caveats, weak spots, suggested next tweaks"},
        },
        ["summary"],
    ),
]


# --------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------

def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _read_clip(session: EditSession, file_id: str) -> str:
    from app.services.l3.catalog import _pg_conn

    if file_id not in session.file_ids:
        return _j({"error": "file_id not in this thread's scope"})
    with _pg_conn() as conn:
        row = conn.execute(
            """
            select cp.perception, coalesce(f.duration_seconds, 0), f.name
              from files f
              left join clip_perception cp on cp.file_id = f.id
             where f.id = %s
            """,
            (file_id,),
        ).fetchone()
    if not row:
        return _j({"error": "clip not found"})
    perception, duration_s, name = row
    doc = perception if isinstance(perception, dict) else (
        json.loads(perception) if perception else None
    )
    if not doc:
        return _j({
            "file_id": file_id, "name": name, "duration_s": float(duration_s),
            "note": "no L2 perception for this clip; only cost grids available",
        })

    # Strip nulls/empties recursively: the full doc is verbose and re-enters
    # context on every later iteration.
    def slim(v):
        if isinstance(v, dict):
            return {k: slim(x) for k, x in v.items() if x not in (None, [], {}, "")}
        if isinstance(v, list):
            return [slim(x) for x in v]
        return v

    return _j({
        "file_id": file_id, "name": name, "duration_s": float(duration_s),
        "perception": slim(doc),
    })


def execute_tool(session: EditSession, name: str, args: Dict[str, Any]) -> str:
    """Run one tool against the session; returns the JSON string fed back to
    the model. Raises nothing: errors return as structured results so the
    agent can correct course."""
    try:
        return _execute(session, name, args)
    except Exception as e:  # noqa: BLE001 - the agent handles its own errors
        logger.exception("L3 tool %s failed", name)
        return _j({"error": f"{type(e).__name__}: {e}"})


def _execute(session: EditSession, name: str, args: Dict[str, Any]) -> str:
    doc = session.document

    if name == "read_clip":
        return _read_clip(session, args["file_id"])

    if name == "query_seams":
        grids = session.grids(args["file_id"])
        seams = engine.query_seams(
            grids, int(args["around_ms"]),
            args.get("axis", "any"), int(args.get("window_ms", 2000)),
        )
        return _j({"seams": seams})

    if name == "set_brief":
        doc["brief"] = {k: v for k, v in args.items() if v is not None}
        return _j({"ok": True, "brief": doc["brief"]})

    if name == "set_spine":
        regions = args.get("regions") or []
        doc["spine"] = {"regions": regions}
        return _j({
            "ok": True,
            "region_count": len(regions),
            "spine": [
                {"kind": r.get("kind"), "locked": r.get("locked_channels", [])}
                for r in regions
            ],
        })

    if name == "set_outline":
        doc["outline"] = args["beats"]
        return _j({"ok": True, "beat_count": len(doc["outline"])})

    if name == "add_segment":
        grids = session.grids(args["file_id"])
        seg = engine.make_segment(
            grids,
            int(args["in_ms"]), int(args["out_ms"]),
            args.get("axis", "any"),
            beat_id=args.get("beat_id"),
            content=args.get("content"),
            rationale=args.get("rationale"),
            priority=int(args.get("priority", 3)),
        )
        pos = int(args.get("position", -1))
        if pos < 0 or pos >= len(doc["timeline"]):
            doc["timeline"].append(seg)
        else:
            doc["timeline"].insert(pos, seg)
        return _j({"segment": seg, "total_s": engine.timeline_status(doc["timeline"])["total_s"]})

    if name == "update_segment":
        seg = session.find_segment(args["seg_id"])
        if seg is None:
            return _j({"error": "unknown seg_id"})
        grids = session.grids(seg["file_id"])
        axis = seg.get("axis", "any")
        if "in_ms" in args and args["in_ms"] is not None:
            snapped = engine.snap_cut(grids, int(args["in_ms"]), axis)
            seg["in_ms"], seg["cut_in_cost"] = snapped["ts_ms"], snapped["cost"]
        if "out_ms" in args and args["out_ms"] is not None:
            snapped = engine.snap_cut(grids, int(args["out_ms"]), axis)
            seg["out_ms"], seg["cut_out_cost"] = snapped["ts_ms"], snapped["cost"]
        for k in ("content", "rationale", "priority", "beat_id"):
            if args.get(k) is not None:
                seg[k] = args[k]
        return _j({"segment": seg})

    if name == "remove_segment":
        before = len(doc["timeline"])
        doc["timeline"] = [s for s in doc["timeline"] if s["seg_id"] != args["seg_id"]]
        if len(doc["timeline"]) == before:
            return _j({"error": "unknown seg_id"})
        return _j({"ok": True, "remaining": len(doc["timeline"])})

    if name == "move_segment":
        seg = session.find_segment(args["seg_id"])
        if seg is None:
            return _j({"error": "unknown seg_id"})
        doc["timeline"].remove(seg)
        idx = max(0, min(int(args["new_index"]), len(doc["timeline"])))
        doc["timeline"].insert(idx, seg)
        return _j({"ok": True, "order": [s["seg_id"] for s in doc["timeline"]]})

    if name == "compare_takes":
        gid = args.get("group_id")
        group = next((g for g in session.take_groups if g.group_id == gid), None)
        if group is None:
            return _j({
                "error": "unknown group_id",
                "available": [g.group_id for g in session.take_groups],
            })
        sources = score_span.load_sources(list({a.file_id for a in group.attempts}))
        takes_out = []
        for a in group.attempts:
            src = sources.get(a.file_id)
            takes_out.append({
                "attempt_id": a.attempt_id,
                "file_id": a.file_id,
                "in_ms": a.start_ms,
                "out_ms": a.end_ms,
                "is_retry": a.is_restart,
                "text": a.text,
                "metrics": score_span.score_span(src, a.start_ms, a.end_ms) if src else {},
                "quality_notes": score_span.quality_events_in(src, a.start_ms, a.end_ms) if src else [],
            })
        return _j({"group_id": gid, "content_key": group.content_key, "takes": takes_out})

    if name == "timeline_status":
        return _j(engine.timeline_status(doc["timeline"]))

    if name == "fit_duration":
        fitted, report = engine.fit_duration(
            doc["timeline"], session.grids_by_file(),
            int(float(args["target_s"]) * 1000),
            int(args.get("tolerance_ms", 500)),
        )
        doc["timeline"] = fitted
        return _j({"report": report, "status": engine.timeline_status(fitted)})

    if name == "ask_user":
        doc["open_questions"] = args["questions"]
        return _j({"ok": True, "paused": True})

    if name == "finalize":
        doc["summary"] = args["summary"]
        doc["notes"] = args.get("notes", [])
        doc["open_questions"] = doc.get("open_questions", [])
        doc["diagnostics"] = engine.timeline_status(doc["timeline"])
        return _j({"ok": True, "finalized": True})

    return _j({"error": f"unknown tool {name!r}"})
