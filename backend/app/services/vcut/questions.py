"""
vcut_pass2_rich.plan.md section 4 -- the closed generic question bank. Pass
1 (pass1.py) selects a per-clip subset of ids while it's already looking at
the frames (no extra call); Pass 2 (pass2.py) answers exactly that subset
against the shared frame cache. Closed: selection can only choose ids that
exist here, never mint new ones -- bounded, cheap, parseable, generic
across monuments/events/actions/people.
"""
from __future__ import annotations

from typing import Dict, List, NamedTuple


class Question(NamedTuple):
    id: str
    prompt: str          # what it asks
    output_hint: str      # expected output shape, for the prompt text


BANK: Dict[str, Question] = {
    q.id: q for q in (
        Question("subject", "the main subject of the moment", "short string"),
        Question("action", "what is happening (the motion/event)", "short string"),
        Question("moment_type", "this cut's role",
                 "one of: establishing, build, peak, aftermath, transition"),
        Question("setting", "where / environment", "short string"),
        Question("on_screen_text", "visible text (signage/captions/UI), verbatim",
                 "string, or empty if none"),
        Question("notable_object", "a prominent object / product / landmark",
                 "short string, or empty if none"),
        Question("count", "rough count of a repeated key element",
                 "integer, or null if not applicable"),
        Question("motion_quality", "energy of the shot", "one of: static, subtle, dynamic"),
        Question("continuity_cue", "does it visually continue/precede another shown moment",
                 "short hint, or empty if none"),
    )
}

# Applied when a clip selects no valid question -- always cheap and almost
# always relevant, per section 4's own "when relevant" column.
DEFAULT_QUESTION_IDS = ("subject", "action", "moment_type")

ALL_IDS = tuple(BANK.keys())


def bank_prompt_lines() -> str:
    """The bank rendered as prompt text (ids + one-liners) for both Pass 1's
    selection step and Pass 2's answering step."""
    return "\n".join(f"  {q.id}: {q.prompt} ({q.output_hint})" for q in BANK.values())


def validate_question_ids(ids: List[str]) -> List[str]:
    """Whitelist against the closed bank (drop unknown ids), dedupe (order-
    preserving), and fall back to DEFAULT_QUESTION_IDS if nothing valid
    survives (section 5: "If a clip selects none, apply a small default
    set")."""
    seen: List[str] = []
    for qid in ids or []:
        if qid in BANK and qid not in seen:
            seen.append(qid)
    return seen or list(DEFAULT_QUESTION_IDS)
