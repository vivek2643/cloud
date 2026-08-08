"""
vcut_pass2_video_specifics.plan.md section 4 -- the question planner: ONE
text-only Gemini call per run (no video, no frames -> nearly free) that
judges the project's genre and picks a small, content-aware subset of
questions.BANK ids (+ up to params.QPLAN_MAX_CUSTOM free-text probes) for
EACH moment flag, so Pass 2 asks what's actually relevant to THIS footage
instead of the same fixed default set for everything. Replaces pass2.
_question_ids_by_file's Phase-2-interim stopgap.

Writes onto MomentFlag.question_ids/custom_questions and MomentPlan.genre
(section 7.1/4.4) -- geometry (t_ms/shape) and Pass 1's semantics
(summary) are untouched; this module only ever adds planning metadata.

Never imports anything from app.services.l3 (principle 7's isolation
constraint) -- the local transcript context it reads is passed IN by the
caller (orchestrate.py, already loading it for the speech channel) as a
plain ``{file_id: [(start_ms, end_ms, text), ...]}`` shape, not any L3/
speech-module dataclass, so this module has zero dependency on how that
data was produced.

Fails open per section 1's own principle ("No hard failures for cost
optimizations"): ANY problem (empty project, model/schema error) leaves
every moment on DEFAULT_QUESTION_IDS (today's pre-planner behavior) and
genre at "" -- a planner miss is a quality gap, never a run failure.

THIS MODULE SPENDS REAL MONEY once invoked (one Gemini call per project,
same disclosure as pass1.run_pass1's own docstring) -- but a cheap,
text-only one.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

from app.services.vcut import questions
from app.services.vcut.params import QPLAN_CTX_MS, QPLAN_MAX_CUSTOM
from app.services.vcut.resolve import FilePlan, MomentPlan

logger = logging.getLogger(__name__)

# section 4.2 step 1: the closed genre vocabulary -- one call, one genre for
# the whole project (not per-file; a project is normally one shoot/edit).
_GENRES = ("performance", "tutorial", "vlog", "product_demo", "interview", "event", "broll", "other")

# section 4.2 step 2: a genre's usual STARTING set of question ids -- prompt
# guidance the model refines per moment from, never enforced server-side (a
# moment can legitimately need something outside its genre's usual set).
_GENRE_TEMPLATES: Dict[str, Tuple[str, ...]] = {
    "performance": ("subject", "action", "moment_type", "motion_quality", "energy_emotion", "shot_size"),
    "tutorial": ("subject", "action", "on_screen_text", "notable_object", "moment_type"),
    "vlog": ("subject", "action", "setting", "energy_emotion", "hook_potential"),
    "product_demo": ("subject", "notable_object", "on_screen_text", "action", "shot_size"),
    "interview": ("subject", "action", "moment_type", "energy_emotion"),
    "event": ("subject", "action", "setting", "moment_type", "count"),
    "broll": ("subject", "action", "setting", "shot_size", "camera_move"),
    "other": tuple(questions.DEFAULT_QUESTION_IDS),
}

# A single project-wide call whose output scales with moment count (like
# Pass 1's own per-project call once did, before the truncation fix that
# forced it per-file -- pass1_video_input.plan.md's own history). Kept as
# ONE call per the plan's "nearly free" design; if a very large project
# ever truncates this, per-file fan-out (mirroring Pass 1's own fix) is the
# next lever, not attempted here since it isn't validated as needed yet.
_MAX_TOKENS = 32000


class _CustomProbe(BaseModel):
    key: str = ""
    prompt: str = ""


class _MomentPlanOut(BaseModel):
    file_id: str
    moment_index: int
    question_ids: List[Literal[questions.ALL_IDS]] = []
    custom_questions: List[_CustomProbe] = []


class _QPlanSchema(BaseModel):
    genre: Literal[_GENRES] = "other"
    moments: List[_MomentPlanOut] = []


def _bank_lines() -> str:
    return questions.bank_prompt_lines()


def _templates_lines() -> str:
    return "\n".join(f"  {genre}: {', '.join(ids)}" for genre, ids in _GENRE_TEMPLATES.items())


def _system_prompt() -> str:
    return f"""You are planning which questions a later pass should ask about each MOMENT
in a video editing project, so that pass can answer them by actually watching
the footage.

You are shown every file's moments (file_id, moment index, its one-line
summary, and any speech happening nearby), in time order.

Your job, in two steps:

1. GENRE -- judge the whole project's genre from all the summaries and
   nearby speech: one of {", ".join(_GENRES)}.

2. PER-MOMENT QUESTIONS -- for each moment, pick which of the bank's
   question ids are actually worth answering for THAT moment, not every
   field for every moment (a piano recital doesn't need on_screen_text; a
   product demo usually does). Start from your genre's usual set below,
   then adjust per moment using its own summary and any nearby speech. Pick
   a small, relevant set (typically 2-5 ids) -- never the whole bank by
   default. You may also add up to {QPLAN_MAX_CUSTOM} CUSTOM probes per
   moment for something genuinely unusual this moment needs that the bank
   doesn't cover -- each is {{"key": a short snake_case name, "prompt": what
   to ask}}.

Question bank:
{_bank_lines()}

Genre -> usual starting set (a guide, not a rule -- refine per moment):
{_templates_lines()}

Cover EVERY moment shown -- a moment you don't mention falls back to a
generic default set, losing the tailoring this pass exists to provide.
"""


def _local_context(t_ms: int, words: List[Tuple[int, int, str]], ctx_ms: int) -> str:
    """The words whose span overlaps [t_ms-ctx_ms, t_ms+ctx_ms], joined in
    order -- "what's being said near this moment," the transcript-context
    input section 4.1 asks for. Empty when the file has no transcript at
    all (b-roll) or nothing falls in range."""
    lo, hi = t_ms - ctx_ms, t_ms + ctx_ms
    return " ".join(text for start, end, text in words if end >= lo and start <= hi and text)


def _moments_listing(plan: MomentPlan, words_by_file: Dict[str, List[Tuple[int, int, str]]]) -> str:
    lines: List[str] = []
    for fp in plan.files:
        words = words_by_file.get(fp.file_id) or []
        for i, flag in enumerate(fp.flags):
            ctx = _local_context(flag.t_ms, words, QPLAN_CTX_MS)
            ctx_part = f" nearby_speech={ctx!r}" if ctx else ""
            lines.append(f"- file={fp.file_id} moment={i} t={flag.t_ms}ms summary={flag.summary!r}{ctx_part}")
    return "\n".join(lines)


def _apply_defaults(plan: MomentPlan) -> MomentPlan:
    """Every flag falls back to DEFAULT_QUESTION_IDS, no custom probes --
    today's pre-planner behavior, used on any planner failure."""
    files = [
        FilePlan(file_id=fp.file_id, flags=[
            replace(flag, question_ids=list(questions.DEFAULT_QUESTION_IDS), custom_questions=[])
            for flag in fp.flags
        ])
        for fp in plan.files
    ]
    return MomentPlan(files=files, genre=plan.genre)


def _apply_plan(plan: MomentPlan, parsed: _QPlanSchema) -> MomentPlan:
    """Write the planner's per-moment output onto a NEW MomentPlan (plan.py
    stays pure/no mutation, matching resolve.py's own style). A moment the
    model didn't mention at all falls back to DEFAULT_QUESTION_IDS -- never
    left unplanned. question_ids are re-validated against the closed bank
    defensively (never trust the model even with a Literal constraint);
    custom_questions are capped at QPLAN_MAX_CUSTOM and malformed/blank
    entries are dropped."""
    by_key: Dict[Tuple[str, int], _MomentPlanOut] = {
        (m.file_id, m.moment_index): m for m in parsed.moments
    }
    files: List[FilePlan] = []
    for fp in plan.files:
        new_flags = []
        for i, flag in enumerate(fp.flags):
            out = by_key.get((fp.file_id, i))
            if out is None:
                new_flags.append(replace(flag, question_ids=list(questions.DEFAULT_QUESTION_IDS),
                                         custom_questions=[]))
                continue
            qids = questions.validate_question_ids(list(out.question_ids))
            custom = [
                {"key": c.key, "prompt": c.prompt}
                for c in out.custom_questions[:QPLAN_MAX_CUSTOM] if c.key and c.prompt
            ]
            new_flags.append(replace(flag, question_ids=qids, custom_questions=custom))
        files.append(FilePlan(file_id=fp.file_id, flags=new_flags))
    return MomentPlan(files=files, genre=parsed.genre)


def plan_questions(
    plan: MomentPlan,
    words_by_file: Dict[str, List[Tuple[int, int, str]]],
    *,
    model: Optional[str] = None,
) -> Tuple[MomentPlan, Dict[str, int]]:
    """ONE text-only Gemini call (section 4) -> a NEW MomentPlan with every
    flag's question_ids/custom_questions filled in and plan.genre set.
    ``words_by_file``: {file_id: [(start_ms, end_ms, text), ...]} -- the
    speech channel's own transcript words, passed in by the caller (not
    re-loaded here). NEVER raises: an empty project is a no-op; any model/
    schema failure logs and falls back to DEFAULT_QUESTION_IDS for every
    moment (section 1: "no hard failures for cost optimizations"). Returns
    (plan, usage) -- usage is {} when no call was made (nothing to plan, or
    the call failed before any usage was recorded)."""
    from app.config import get_settings
    from app.services.llm.base import text_block
    from app.services.llm.ingest_gemini import complete_gemini

    total_moments = sum(len(fp.flags) for fp in plan.files)
    if total_moments == 0:
        return plan, {}

    settings = get_settings()
    resolved_model = model or settings.vcut_qplan_model or settings.vcut_pass1_model

    try:
        task_text = _moments_listing(plan, words_by_file)
        completion = complete_gemini(
            _system_prompt(), [text_block(task_text)], _QPlanSchema,
            model=resolved_model, max_tokens=_MAX_TOKENS, thinking="low",
        )
        parsed = _QPlanSchema.model_validate(completion.data)
    except Exception:
        logger.exception("vcut qplan: planner call failed for %d moment(s) -- "
                         "every moment falls back to DEFAULT_QUESTION_IDS", total_moments)
        return _apply_defaults(plan), {}

    return _apply_plan(plan, parsed), completion.usage
