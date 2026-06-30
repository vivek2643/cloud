"""
Paged arranger: the fallback path for libraries too large to hold resident.

The default arranger (``arrange._resident_arrange``) drops the WHOLE footage map
into one context and reasons over it. When that map exceeds the context budget
(``arranger_resident_char_budget``) we don't blindly truncate -- we page, exactly
like skimming a big codebase: the model gets a COMPACT one-line-per-moment index
and pulls the full detail for the handful of real candidates on demand, then
commits. Same neutral LLM client, same downstream seam (placements validated
through ``arrange._MapIndex`` / ``_coerce_placements``).

Everything here is provider-neutral: tools are declared with ``tool_spec`` and
results returned as neutral ``tool_result`` blocks; no vendor SDK is touched.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.services.llm import tool_result_block, tool_spec, user_message

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Tool declarations (neutral)
# --------------------------------------------------------------------------

_TOOLS = [
    tool_spec(
        "inspect_moment",
        "Get the full record for one moment: its complete text, every energy "
        "variant span, every atom, and the parent clip's context. Use before "
        "committing a candidate when the compact index is not enough.",
        {
            "type": "object",
            "properties": {
                "file_id": {"type": "string", "description": "The clip's file id."},
                "moment_id": {"type": "string", "description": "Full moment id, e.g. ab12cd34:m07."},
            },
            "required": ["file_id", "moment_id"],
        },
    ),
    tool_spec(
        "inspect_dup_group",
        "List every member of a same-beat group (the same line delivered more "
        "than once, as a retake or another angle) with full text, score, whether "
        "the speaker is on camera, delivery, and whether it's an abandoned retry "
        "-- so you can compare and choose which to use.",
        {
            "type": "object",
            "properties": {
                "dup_group": {"type": "string", "description": "Group id, e.g. tg4."},
            },
            "required": ["dup_group"],
        },
    ),
    tool_spec(
        "submit_timeline",
        "Commit the FINAL edit. Call this exactly once when you are done.",
        {
            "type": "object",
            "properties": {
                "timeline": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "ref": {"type": "string"},
                            "level": {"type": "string"},
                            "track": {"type": "integer"},
                            "reason": {"type": "string"},
                        },
                        "required": ["ref"],
                    },
                },
                "notes": {"type": "string"},
            },
            "required": ["timeline"],
        },
    ),
]


_PAGED_SYSTEM = (
    "You are the EDITOR of a video. The footage library is large, so you are given "
    "a COMPACT index: one line per moment with a short gist, its energy LEVELS, and "
    "duplicate-group markers. The full text of any moment is one tool call away.\n\n"
    "Work like an editor skimming all the footage: scan the index, INSPECT the "
    "handful of moments you are seriously considering (and inspect_dup_group for "
    "any 'dup:tgN' you might use, to compare the takes/angles and choose), form a "
    "story, then commit with submit_timeline.\n\n"
    "Index notation: '<clip8>:<m##> channel.subject speaker .score [in-out] \"gist\" · "
    "nrg:levels (+N atoms) · dup:tgN'. Refer to moments by FULL id. 'dup:tgN' = "
    "the same spoken beat as others in group N (a retake or angle; 'retry' = "
    "abandoned). No take is pre-picked -- choose by text and score. Don't repeat "
    "the same line on the main line, but you MAY use another member as a silent "
    "reaction/cutaway over that line's audio.\n\n"
    "Working principles (let the BRIEF and the footage decide the style -- no "
    "fixed format): cut slates, mic-checks, false starts, and anything that "
    "doesn't serve the brief; avoid unintentional repetition (don't repeat the "
    "same line unless the brief wants it or footage is limited); EVERY moment is "
    "an equal candidate for the main line -- never rank one category over another; "
    "choose energy level per cut for pacing; honor the target length. Track 0 is "
    "the main line (back-to-back, no gaps); overlays on higher tracks are optional "
    "and rare. Consecutive main-line picks from the SAME clip whose source times "
    "touch/overlap are auto-welded into one continuous segment (no visible cut), "
    "so adjacent slices build a longer beat; pick non-adjacent slices to force a "
    "hard cut within a clip.\n\n"
    "Be efficient: do not inspect everything -- only real candidates. Then call "
    "submit_timeline once."
)


# --------------------------------------------------------------------------
# Tool dispatch
# --------------------------------------------------------------------------

def _moments_in_group(map_struct: Dict[str, Any], dup_group: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for clip in (map_struct or {}).get("clips", []) or []:
        for m in clip.get("moments", []) or []:
            if m.get("dup_group") == dup_group:
                q = m.get("quality") or {}
                out.append({
                    "moment_id": m["moment_id"],
                    "file_id": m["file_id"],
                    "speaker": m.get("speaker"),
                    "score": float(m.get("score", 0.0)),
                    "on_camera": q.get("on_camera"),
                    "delivery": q.get("delivery"),
                    "retry": bool(m.get("dup_restart")),
                    "text": (m.get("gist") or "").strip(),
                    "in_ms": m.get("in_ms"),
                    "out_ms": m.get("out_ms"),
                    "levels": list((m.get("variants") or {}).keys()),
                })
    return out


def _dispatch(name: str, args: Dict[str, Any], *, index, map_struct: Dict[str, Any]
              ) -> Tuple[str, Optional[list]]:
    """Run one tool. Returns (result_text, placements_or_None). The placements
    are non-None only for submit_timeline (the terminal tool)."""
    from app.services.l3 import arrange, footage_map

    if name == "inspect_moment":
        detail = footage_map.moment_detail(
            str(args.get("file_id") or ""), str(args.get("moment_id") or ""))
        if detail is None:
            return ("not found -- check the ids against the index", None)
        return (json.dumps(detail, default=str), None)

    if name == "inspect_dup_group":
        moments = _moments_in_group(map_struct, str(args.get("dup_group") or ""))
        if not moments:
            return ("no such dup group", None)
        return (json.dumps(moments, default=str), None)

    if name == "submit_timeline":
        doc = {"timeline": args.get("timeline") or [], "notes": args.get("notes") or ""}
        placements = arrange._coerce_placements(doc, index)
        return (f"submitted {len(placements)} cuts", placements)

    return (f"unknown tool: {name}", None)


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def run_paged(brief: str, map_struct: Dict[str, Any], plan, *,
              llm, current_timeline: Optional[str] = None) -> List:
    """Paged reasoning loop. Returns validated placements (possibly [])."""
    from app.services.l3 import arrange, footage_map

    settings = get_settings()
    max_turns = int(getattr(settings, "arranger_max_turns", 12))
    effort = getattr(settings, "arranger_paged_effort", None) or "medium"
    max_tokens = settings.autoedit_max_output_tokens

    index = arrange._MapIndex(map_struct)
    file_ids = [c["file_id"] for c in (map_struct or {}).get("clips", []) or []]
    # Compact index from the same source (cached trees); fail-open to nothing.
    try:
        compact = footage_map.assemble_map(file_ids, compact=True).get("text") or ""
    except Exception:
        logger.exception("paged arranger: compact map build failed")
        return []

    messages = [user_message(arrange._arranger_prompt(brief, plan, compact, current_timeline))]
    submitted: Optional[list] = None

    for turn in range(max_turns):
        resp = llm.run(system=_PAGED_SYSTEM, messages=messages, tools=_TOOLS,
                       max_tokens=max_tokens, effort=effort, cache_system=True)
        messages.append(resp.assistant_message)

        if not resp.tool_calls:
            # No tool: maybe it emitted the JSON directly. Accept it, else nudge.
            from app.services.l3.auto_edit import _parse_json
            doc = _parse_json(resp.text)
            placements = arrange._coerce_placements(doc, index) if doc else []
            if placements:
                return placements
            if turn < max_turns - 1:
                messages.append(user_message(
                    "Call submit_timeline with your final timeline to finish."))
            continue

        results: List[Dict[str, Any]] = []
        for tc in resp.tool_calls:
            text, placements = _dispatch(tc.name, tc.input or {}, index=index,
                                         map_struct=map_struct)
            results.append(tool_result_block(tc.id, text))
            if placements is not None:
                submitted = placements
        messages.append(user_message(results))
        if submitted is not None:
            return submitted

    if submitted is None:
        logger.warning("paged arranger: hit turn cap (%d) with no submission", max_turns)
    return submitted or []
