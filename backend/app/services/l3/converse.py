"""
Conversational L3: a chat-first, AGENTIC editor over a project's footage + edit.

Design note (why it's shaped this way): a strong model edits best with a NEAR-
EMPTY frame and room to think -- the same way it does when you just hand it the
footage and ask. So the brain works like a coding agent over a repo: it SEES the
whole shoot (footage map) and the current edit, and it has TOOLS -- deterministic
SENSES (``observe``: read_state / predict / validate / diagnose / affordances)
and edit VERBS (``act``: place / trim / remove / move / set_audio / tighten).

Each user turn runs a bounded perceive -> act -> re-perceive loop (``tools``):
the brain looks, edits the WORKING document, checks its work, and ends with a
prose reply. There is no propose->confirm round-trip anymore -- edits apply
directly (the user sees the timeline update + can undo via version history),
exactly like a coding agent editing files. ``respond`` returns the prose reply +
the mutated document; the caller persists it as a new version when it changed.

Fails OPEN: any LLM/tool error degrades to a plain chat reply with no document
change, so a turn never hard-fails.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from app.config import get_settings
from app.services.l3 import arrange, footage_map, observe, store, tools
from app.services.llm import LLMClient, get_llm

logger = logging.getLogger(__name__)

# The chat brain gets the WHOLE footage map (every clip + moment) so it has the
# same total awareness the arranger does -- like handing a coding agent the full
# repo map. The cap is only a runaway guard for pathologically large libraries
# (~35k chars for 7 clips / 263 moments; ~9k tokens), well within context.
_MAP_CHAR_CAP = 200_000


@dataclass
class ConverseResult:
    reply: str
    # The mutated working document + whether this turn changed it. The caller
    # persists a new version when ``changed`` is True.
    document: Optional[dict] = None
    changed: bool = False
    # When the editor asked the user to decide (ask_user), the turn paused: these
    # questions are surfaced and the user's next message is the answer.
    questions: List[dict] = field(default_factory=list)
    awaiting_user: bool = False


# The agentic frame: you're an editor working ON an edit with TOOLS. Minimal
# taste rules -- give it its senses + verbs, the channel model, and the way to
# refer to footage, then get out of its way.
_LOOP_SYSTEM = (
    "You are EDSO, a sharp video editor. You are working ON an edit with real "
    "TOOLS, like a coding agent editing a repo. Below you can see the WHOLE shoot "
    "-- every clip and every usable moment, with the COMPLETE line of what is said "
    "-- and the CURRENT TIMELINE of the edit so far.\n\n"
    "Each moment reads like: clip8:m07 speech .82 [0:14-0:21] \"the full line\" · "
    "nrg:calm|balanced · dup:tg4*\n"
    "  - refer to a moment by its full id (e.g. ab12cd34:m07).\n"
    "  - the quoted text is the complete line -- judge from it.\n"
    "  - nrg = the energy LEVELS (broad = the whole thing .. sharp = tightest).\n"
    "  - dup:tgN = the same content as others in that group (another take/angle); "
    "use only one.\n\n"
    "HOW YOU WORK. When the user just wants to talk or asks a question, answer in "
    "prose and DON'T touch the edit. When they want a change (or to build the "
    "edit), use your tools: OBSERVE first (read_state / diagnose / affordances / "
    "predict), then ACT (place / trim / remove / move / set_audio / tighten / "
    "split_screen), then read_state again to check your work. Your edits apply "
    "DIRECTLY -- the user "
    "watches the timeline update and can undo -- so don't ask for confirmation or "
    "say 'I will'; just do it, then tell them what you did in a sentence or two.\n\n"
    "CHANNELS. The main line is V1 video + A1 audio, playing in sequence -- that's "
    "`place` with channel V1. A silent video cutaway laid over the ongoing A1 audio "
    "rides V2 (`place` channel V2). A music/SFX bed rides A2. Default is just "
    "V1/A1; only add a cutaway or bed when it earns its place. Showing V1 and a "
    "second source at the SAME time (side-by-side, stacked, or picture-in-picture) "
    "is `split_screen` over a program window. Use only ids that appear below.\n\n"
    "ASKING. Most calls are yours to make -- just make them. But when a choice is "
    "genuinely the USER's (a split-screen or PiP layout, the delivery aspect / "
    "framing, a big pacing tradeoff), use `ask_user` with 2+ concrete options "
    "BEFORE acting -- that ends your turn and you resume when they answer. Don't "
    "ask about things you can reasonably decide."
)

def _context_block(file_ids: List[str], document: Optional[dict]) -> str:
    parts: List[str] = []
    try:
        text = (footage_map.assemble_map(file_ids).get("text") or "") if file_ids else ""
        if len(text) > _MAP_CHAR_CAP:
            text = text[:_MAP_CHAR_CAP] + "\n…(truncated)"
        if text:
            parts.append("FOOTAGE MAP:\n" + text)
    except Exception:
        logger.exception("converse: map build failed (continuing without it)")
    overlay = arrange.timeline_overlay(document)
    parts.append("CURRENT TIMELINE:\n" + overlay if overlay
                 else "CURRENT TIMELINE: (empty -- no edit drafted yet)")
    return "\n\n".join(parts)


def _seed_document(file_ids: List[str]) -> dict:
    """An empty Edit Document the agentic loop builds onto (place/... verbs).
    Mirrors the document shape the rest of the system reads so preview / render
    read it identically once resolved (via ``observe.resolve_doc``)."""
    return {
        "brief": {"goal": None, "aspect": "landscape", "target_duration_s": None, "assumptions": []},
        "format": {"aspect": "landscape"},
        "spine": {"regions": []},
        "outline": [],
        "timeline": [],
        "operations": [],
        "open_questions": [],
        "summary": "",
        "notes": [],
        "diagnostics": {"engine": "agentic_loop"},
    }


def respond(thread_id: str, *, llm: Optional[LLMClient] = None) -> ConverseResult:
    """Run one agentic turn on a thread.

    The editor SEES the footage map + current edit and drives a bounded tool loop
    (``tools.run_edit_loop``): observe -> act -> re-observe, mutating a WORKING
    copy of the Edit Document, then replies in prose. Returns the reply + the
    mutated document + whether it changed; the caller persists a new version when
    it did. Fails OPEN -- any error degrades to a plain reply, no doc change."""
    settings = get_settings()
    if llm is None:
        llm = get_llm(provider=settings.autoedit_provider or None,
                      model=settings.autoedit_model or None)

    thread = store.get_thread(thread_id)
    file_ids = (thread or {}).get("file_ids") or []
    document, _ = store.latest_document(thread_id)
    messages = store.load_messages(thread_id)
    if not messages:
        return ConverseResult(reply="Tell me what you'd like to do with these clips.")

    working = document if isinstance(document, dict) else _seed_document(file_ids)
    system = _LOOP_SYSTEM + "\n\n" + _context_block(file_ids, document)
    max_tokens = settings.autoedit_max_output_tokens
    try:
        ctx = observe.build_context(file_ids)
        result = tools.run_edit_loop(llm, system=system, messages=messages,
                                     ctx=ctx, document=working, max_tokens=max_tokens)
    except Exception:
        logger.exception("converse: agentic loop failed for thread %s", thread_id)
        return ConverseResult(reply="Sorry -- I hit an error there. Mind trying again?")

    reply = (result.reply or "").strip() or "…"
    if result.changed:
        try:
            observe.resolve_doc(result.document, ctx)
        except Exception:
            logger.exception("converse: resolve after edit failed for thread %s", thread_id)
    return ConverseResult(reply=reply, document=result.document, changed=result.changed,
                          questions=result.questions, awaiting_user=result.awaiting_user)
