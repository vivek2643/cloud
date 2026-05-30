"""
Cut-only EDL editing agent (Phase 3).

Claude is given the current EDL plus a focused, cut-only tool set and edits a
WORKING COPY of the timeline through tool calls. Nothing is committed here:
the agent returns a PROPOSED EDL + a structured diff. The user reviews the
diff and applies it (which commits a new edl_version with author_kind='claude'
and renders) -- or discards it.

Tool set (all cut-only):
  read:    search_shots, get_shot_metadata, read_transcript_window,
           find_silences, list_timeline
  mutate:  trim_clip, move_clip, delete_clip, insert_clip
  control: done

Design notes:
  * The loop runs in a background thread (driven by the async-turn machinery),
    emitting a progress event per tool call so the SSE UI shows live steps.
  * Mutations operate on an in-memory clip list of minimal clips
    {id, shot_id, source_in_ms, source_out_ms}. timeline positions are derived
    only at the end (cut-only = sequential concat), exactly like manual edits.
  * The diff is computed by comparing the base clip list to the working copy.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings
from app.services import prompts
from app.services.edl import store as edl_store
from app.services.l3.anthropic_client import _client as _anthropic_client
from app.services.l3.query_executor import retrieve_top_k

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 20
MIN_CLIP_MS = 200


class AgentCancelled(Exception):
    pass


EmitFn = Callable[[str, int, str], None]
CancelFn = Callable[[], bool]


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


# ---------------------------------------------------------------------------
# Tool schemas (Anthropic tool-use format)
# ---------------------------------------------------------------------------

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "list_timeline",
        "description": "Return the current working timeline: ordered clips with their clip_id, shot_id, source in/out (ms) and duration. Call this whenever you need to re-check the state after edits.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "search_shots",
        "description": "Semantic search over the project's footage for shots matching a description. Returns candidate shots with shot_id, file_name, natural start/end (ms) and a transcript snippet. Use this to find material to insert.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, e.g. 'wide establishing shot of the room'."},
                "k": {"type": "integer", "description": "Max results (1-20).", "default": 10},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_shot_metadata",
        "description": "Get detailed metadata for one shot: file, natural start/end (ms), motion magnitude, brightness, focus score, framing, narrative role, and the transcript text overlapping the shot.",
        "input_schema": {
            "type": "object",
            "properties": {"shot_id": {"type": "string"}},
            "required": ["shot_id"],
        },
    },
    {
        "name": "read_transcript_window",
        "description": "Read the spoken words in a file between start_ms and end_ms, with per-word timings and filler-word flags. Use this to find exact cut points (e.g. trim to the end of a sentence, or remove an 'um').",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "start_ms": {"type": "integer"},
                "end_ms": {"type": "integer"},
            },
            "required": ["file_id", "start_ms", "end_ms"],
        },
    },
    {
        "name": "find_silences",
        "description": "Return silent intervals (start_ms, end_ms) detected in a file's audio, optionally filtered by a minimum duration. Useful for tightening pauses or finding clean cut points.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "min_ms": {"type": "integer", "description": "Only return silences at least this long.", "default": 300},
            },
            "required": ["file_id"],
        },
    },
    {
        "name": "trim_clip",
        "description": "Set a clip's source in and/or out point (absolute ms within the source file). Omit a field to leave it unchanged. The clip's source range must stay valid and at least 200ms long.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clip_id": {"type": "string"},
                "source_in_ms": {"type": "integer"},
                "source_out_ms": {"type": "integer"},
            },
            "required": ["clip_id"],
        },
    },
    {
        "name": "move_clip",
        "description": "Move a clip to a new zero-based position in the timeline order.",
        "input_schema": {
            "type": "object",
            "properties": {
                "clip_id": {"type": "string"},
                "to_index": {"type": "integer"},
            },
            "required": ["clip_id", "to_index"],
        },
    },
    {
        "name": "delete_clip",
        "description": "Remove a clip from the timeline.",
        "input_schema": {
            "type": "object",
            "properties": {"clip_id": {"type": "string"}},
            "required": ["clip_id"],
        },
    },
    {
        "name": "insert_clip",
        "description": "Insert a new clip from a shot into the timeline at an optional position (defaults to the end). source_in_ms/source_out_ms default to the shot's natural bounds when omitted.",
        "input_schema": {
            "type": "object",
            "properties": {
                "shot_id": {"type": "string"},
                "source_in_ms": {"type": "integer"},
                "source_out_ms": {"type": "integer"},
                "at_index": {"type": "integer"},
            },
            "required": ["shot_id"],
        },
    },
    {
        "name": "done",
        "description": "Call when the timeline satisfies the user's instruction. Provide a one-sentence summary of the changes you made.",
        "input_schema": {
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
        },
    },
]


# ---------------------------------------------------------------------------
# Working timeline + DB helpers
# ---------------------------------------------------------------------------

class _Timeline:
    """In-memory ordered list of minimal clips the tools mutate."""

    def __init__(self, clips: List[Dict[str, Any]]):
        # Keep only the cut-only fields; preserve order.
        self.clips: List[Dict[str, Any]] = [
            {
                "id": str(c["id"]),
                "shot_id": str(c["shot_id"]),
                "source_in_ms": int(c["source_in_ms"]),
                "source_out_ms": int(c["source_out_ms"]),
            }
            for c in clips
        ]

    def find(self, clip_id: str) -> Optional[int]:
        for i, c in enumerate(self.clips):
            if c["id"] == clip_id:
                return i
        return None

    def as_view(self) -> List[Dict[str, Any]]:
        out = []
        for i, c in enumerate(self.clips):
            out.append({
                "index": i,
                "clip_id": c["id"],
                "shot_id": c["shot_id"],
                "source_in_ms": c["source_in_ms"],
                "source_out_ms": c["source_out_ms"],
                "duration_ms": c["source_out_ms"] - c["source_in_ms"],
            })
        return out


def _shot_row(shot_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    with _pg() as conn:
        cur = conn.execute(
            """
            select s.id as shot_id, s.start_ms, s.end_ms, s.motion_magnitude,
                   s.brightness, s.focus_score, s.framing_scale, s.narrative_role,
                   f.id as file_id, f.name as file_name
            from shots s join files f on f.id = s.file_id
            where s.id = %s and f.user_id = %s
            """,
            (shot_id, user_id),
        )
        return cur.fetchone()


def _file_id_for_shot(shot_id: str, user_id: str) -> Optional[str]:
    row = _shot_row(shot_id, user_id)
    return str(row["file_id"]) if row else None


def _transcript_window(file_id: str, start_ms: int, end_ms: int, user_id: str) -> Dict[str, Any]:
    with _pg() as conn:
        cur = conn.execute(
            """
            select t.segments
            from transcripts t join files f on f.id = t.file_id
            where t.file_id = %s and f.user_id = %s
            """,
            (file_id, user_id),
        )
        row = cur.fetchone()
    if not row or not row["segments"]:
        return {"words": [], "text": ""}
    segs = row["segments"] if isinstance(row["segments"], list) else json.loads(row["segments"])
    words: List[Dict[str, Any]] = []
    for seg in segs:
        for w in seg.get("words", []) or []:
            ws, we = w.get("start_ms", 0), w.get("end_ms", 0)
            if we >= start_ms and ws <= end_ms:
                words.append({
                    "text": w.get("text", ""),
                    "start_ms": ws,
                    "end_ms": we,
                    "is_filler": bool(w.get("is_filler")),
                })
    text = " ".join(w["text"] for w in words)
    return {"words": words[:200], "text": text[:1500]}


def _silences(file_id: str, min_ms: int, user_id: str) -> List[Dict[str, int]]:
    with _pg() as conn:
        cur = conn.execute(
            """
            select af.silence_intervals
            from audio_features af join files f on f.id = af.file_id
            where af.file_id = %s and f.user_id = %s
            """,
            (file_id, user_id),
        )
        row = cur.fetchone()
    if not row or not row["silence_intervals"]:
        return []
    si = row["silence_intervals"] if isinstance(row["silence_intervals"], list) else json.loads(row["silence_intervals"])
    out = [
        {"start_ms": int(s["start_ms"]), "end_ms": int(s["end_ms"])}
        for s in si
        if int(s.get("end_ms", 0)) - int(s.get("start_ms", 0)) >= min_ms
    ]
    return out[:100]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

def _exec_tool(
    name: str,
    args: Dict[str, Any],
    tl: _Timeline,
    user_id: str,
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """
    Execute one tool. Returns (result_payload, is_done, done_summary).
    Tool errors are returned as {"error": ...} so the model can recover
    instead of crashing the loop.
    """
    try:
        if name == "list_timeline":
            return {"clips": tl.as_view()}, False, None

        if name == "search_shots":
            q = (args.get("query") or "").strip()
            k = max(1, min(int(args.get("k", 10)), 20))
            if not q:
                return {"error": "query is required"}, False, None
            cands = retrieve_top_k(user_id=user_id, prompt=q, k=k)
            return {
                "results": [
                    {
                        "shot_id": c.shot_id,
                        "file_id": c.file_id,
                        "file_name": c.file_name,
                        "start_ms": c.start_ms,
                        "end_ms": c.end_ms,
                        "duration_ms": c.end_ms - c.start_ms,
                        "score": round(float(c.score), 4),
                        "transcript_snippet": (c.transcript_text or "")[:160],
                    }
                    for c in cands
                ]
            }, False, None

        if name == "get_shot_metadata":
            row = _shot_row(str(args.get("shot_id")), user_id)
            if not row:
                return {"error": "shot not found"}, False, None
            tw = _transcript_window(str(row["file_id"]), row["start_ms"], row["end_ms"], user_id)
            return {
                "shot_id": str(row["shot_id"]),
                "file_id": str(row["file_id"]),
                "file_name": row["file_name"],
                "start_ms": row["start_ms"],
                "end_ms": row["end_ms"],
                "duration_ms": row["end_ms"] - row["start_ms"],
                "motion_magnitude": row.get("motion_magnitude"),
                "brightness": row.get("brightness"),
                "focus_score": row.get("focus_score"),
                "framing_scale": row.get("framing_scale"),
                "narrative_role": row.get("narrative_role"),
                "transcript_text": tw["text"],
            }, False, None

        if name == "read_transcript_window":
            return _transcript_window(
                str(args.get("file_id")),
                int(args.get("start_ms", 0)),
                int(args.get("end_ms", 0)),
                user_id,
            ), False, None

        if name == "find_silences":
            return {
                "silences": _silences(str(args.get("file_id")), int(args.get("min_ms", 300)), user_id)
            }, False, None

        if name == "trim_clip":
            idx = tl.find(str(args.get("clip_id")))
            if idx is None:
                return {"error": "clip_id not found"}, False, None
            c = tl.clips[idx]
            new_in = int(args["source_in_ms"]) if args.get("source_in_ms") is not None else c["source_in_ms"]
            new_out = int(args["source_out_ms"]) if args.get("source_out_ms") is not None else c["source_out_ms"]
            new_in = max(0, new_in)
            if new_out - new_in < MIN_CLIP_MS:
                return {"error": f"resulting clip would be shorter than {MIN_CLIP_MS}ms"}, False, None
            c["source_in_ms"], c["source_out_ms"] = new_in, new_out
            return {"ok": True, "clip": c}, False, None

        if name == "move_clip":
            idx = tl.find(str(args.get("clip_id")))
            if idx is None:
                return {"error": "clip_id not found"}, False, None
            to = max(0, min(int(args.get("to_index", idx)), len(tl.clips) - 1))
            c = tl.clips.pop(idx)
            tl.clips.insert(to, c)
            return {"ok": True, "clips": tl.as_view()}, False, None

        if name == "delete_clip":
            idx = tl.find(str(args.get("clip_id")))
            if idx is None:
                return {"error": "clip_id not found"}, False, None
            tl.clips.pop(idx)
            return {"ok": True, "clips": tl.as_view()}, False, None

        if name == "insert_clip":
            shot_id = str(args.get("shot_id"))
            row = _shot_row(shot_id, user_id)
            if not row:
                return {"error": "shot not found"}, False, None
            in_ms = int(args["source_in_ms"]) if args.get("source_in_ms") is not None else int(row["start_ms"])
            out_ms = int(args["source_out_ms"]) if args.get("source_out_ms") is not None else int(row["end_ms"])
            in_ms = max(0, in_ms)
            if out_ms - in_ms < MIN_CLIP_MS:
                return {"error": f"clip would be shorter than {MIN_CLIP_MS}ms"}, False, None
            new_clip = {
                "id": str(uuid.uuid4()),
                "shot_id": shot_id,
                "source_in_ms": in_ms,
                "source_out_ms": out_ms,
            }
            at = args.get("at_index")
            if at is None:
                tl.clips.append(new_clip)
            else:
                tl.clips.insert(max(0, min(int(at), len(tl.clips))), new_clip)
            return {"ok": True, "clip_id": new_clip["id"], "clips": tl.as_view()}, False, None

        if name == "done":
            return {"ok": True}, True, str(args.get("summary") or "Done")

        return {"error": f"unknown tool {name}"}, False, None
    except Exception as e:
        logger.exception("tool %s failed", name)
        return {"error": f"{type(e).__name__}: {e}"}, False, None


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------

def _compute_diff(
    base: List[Dict[str, Any]],
    work: List[Dict[str, Any]],
) -> Dict[str, Any]:
    base_by_id = {c["id"]: c for c in base}
    work_by_id = {c["id"]: c for c in work}
    base_order = [c["id"] for c in base]
    work_order = [c["id"] for c in work]

    added = [c for c in work if c["id"] not in base_by_id]
    removed = [c for c in base if c["id"] not in work_by_id]

    trimmed = []
    for c in work:
        b = base_by_id.get(c["id"])
        if b and (b["source_in_ms"] != c["source_in_ms"] or b["source_out_ms"] != c["source_out_ms"]):
            trimmed.append({
                "clip_id": c["id"],
                "from": {"source_in_ms": b["source_in_ms"], "source_out_ms": b["source_out_ms"]},
                "to": {"source_in_ms": c["source_in_ms"], "source_out_ms": c["source_out_ms"]},
            })

    # Moves: ids present in both whose relative position changed. Compare the
    # filtered order (ignoring added/removed) so a delete doesn't flag
    # everything after it as "moved".
    common_base = [i for i in base_order if i in work_by_id]
    common_work = [i for i in work_order if i in base_by_id]
    moved = []
    if common_base != common_work:
        pos_base = {cid: i for i, cid in enumerate(common_base)}
        pos_work = {cid: i for i, cid in enumerate(common_work)}
        for cid in common_work:
            if pos_base.get(cid) != pos_work.get(cid):
                moved.append({
                    "clip_id": cid,
                    "from_index": pos_base.get(cid),
                    "to_index": pos_work.get(cid),
                })

    return {
        "added": added,
        "removed": removed,
        "trimmed": trimmed,
        "moved": moved,
        "changed": bool(added or removed or trimmed or moved),
    }


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _noop_emit(phase: str, pct: int, label: str) -> None:
    return None


def _never_cancel() -> bool:
    return False


def run_agent(
    project_id: str,
    user_id: str,
    instruction: str,
    base_version_id: Optional[str] = None,
    emit: EmitFn = _noop_emit,
    should_cancel: CancelFn = _never_cancel,
) -> Dict[str, Any]:
    """
    Run the cut-only agent against a project's EDL. Returns a proposal dict:
        {
          base_version_id, instruction, summary, reasoning,
          proposed_clips: [{id, shot_id, source_in_ms, source_out_ms}],
          diff: {...}, tool_log: [...]
        }
    Does NOT commit. Raises AgentCancelled on cooperative cancel.
    """
    # Resolve base version.
    if base_version_id:
        version = edl_store.get_edl_version(base_version_id)
    else:
        version = edl_store.get_latest_edl_version(project_id)
    base_clips = (version["edl_json"].get("clips") if version else []) or []
    base_version_id = version["id"] if version else None

    tl = _Timeline(base_clips)
    emit("planning", 5, "Reading the timeline")

    system_prompt = prompts.load("edl_agent")
    initial_user = (
        f"USER INSTRUCTION:\n{instruction.strip()}\n\n"
        f"CURRENT TIMELINE ({len(tl.clips)} clips):\n"
        f"{json.dumps(tl.as_view(), indent=2)}\n\n"
        "Make the requested cut-only changes using the tools, then call done."
    )
    messages: List[Dict[str, Any]] = [{"role": "user", "content": initial_user}]

    client = _anthropic_client()
    settings = get_settings()
    tool_log: List[Dict[str, Any]] = []
    summary = ""
    final_text_bits: List[str] = []

    for it in range(MAX_ITERATIONS):
        if should_cancel():
            raise AgentCancelled()

        emit("reasoning", min(10 + it * 4, 80), f"Thinking (step {it + 1})")
        msg = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=1536,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        # Collect any assistant text + tool_use blocks.
        assistant_content: List[Dict[str, Any]] = []
        tool_uses = []
        for block in msg.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                final_text_bits.append(block.text)
                assistant_content.append({"type": "text", "text": block.text})
            elif btype == "tool_use":
                tool_uses.append(block)
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        messages.append({"role": "assistant", "content": assistant_content})

        if msg.stop_reason != "tool_use" or not tool_uses:
            # Model stopped without (more) tools -> treat as finished.
            break

        tool_results = []
        is_done = False
        for tu in tool_uses:
            args = tu.input if isinstance(tu.input, dict) else {}
            label = _tool_label(tu.name, args)
            emit("editing", min(15 + it * 4, 85), label)
            result, done, done_summary = _exec_tool(tu.name, args, tl, user_id)
            tool_log.append({"tool": tu.name, "input": args, "result_keys": list(result.keys())})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result)[:6000],
            })
            if done:
                is_done = True
                summary = done_summary or "Done"

        messages.append({"role": "user", "content": tool_results})
        if is_done:
            break

    if not summary:
        summary = " ".join(final_text_bits).strip()[:300] or "Proposed timeline edits."

    work_clips = tl.clips
    diff = _compute_diff(base_clips, work_clips)
    emit("done", 100, "Proposal ready")

    return {
        "project_id": project_id,
        "base_version_id": base_version_id,
        "instruction": instruction,
        "summary": summary,
        "reasoning": " ".join(final_text_bits).strip()[:2000],
        "proposed_clips": work_clips,
        "diff": diff,
        "tool_log": tool_log,
    }


def _tool_label(name: str, args: Dict[str, Any]) -> str:
    if name == "search_shots":
        return f"Searching: {args.get('query', '')[:40]}"
    if name == "read_transcript_window":
        return "Reading transcript"
    if name == "find_silences":
        return "Finding silences"
    if name == "get_shot_metadata":
        return "Inspecting shot"
    if name == "trim_clip":
        return "Trimming a clip"
    if name == "move_clip":
        return "Reordering"
    if name == "delete_clip":
        return "Removing a clip"
    if name == "insert_clip":
        return "Inserting a clip"
    if name == "list_timeline":
        return "Checking the timeline"
    if name == "done":
        return "Finalizing"
    return name
