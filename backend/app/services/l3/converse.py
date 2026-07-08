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
    # Ordered tool-call trace for this turn (persisted on the assistant turn so
    # the reasoning is auditable later).
    trace: List[dict] = field(default_factory=list)


# The blind editor -- CUT-CENTRIC, plan-first, no prescribed workflow. States
# WHO the brain is (a blind editor with faithful senses), gives it a plan-first
# discipline (look -> picture the finished piece -> plan -> pin -> execute ->
# check), and lists the senses/verbs as neutral CAPABILITIES + mechanics only.
# It never says "use X for Y" and never enumerates what to notice -- the
# material and the craft are the brain's to read.
#
# cuts_v3_continuity.plan.md: the brain reasons over CUTS only, each carrying
# rich per-cut data (framing/pace/look/channel/take-group) plus a
# deterministic per-cut `continuity` -- its position among its clip's cuts and
# whether each neighbor is a weldable continuation of the same shot -- so it
# still understands ordering/adjacency without ever reading raw footage. There
# is no raw-footage scan (see tools.py) and no other prompt version -- this is
# simply the prompt.
_LOOP_SYSTEM = (
    "You are EDSO, a BLIND editor. You cannot see or hear the footage directly "
    "-- but below you have complete, faithful SENSES that describe it, and real "
    "TOOLS to cut it. Trust the senses; when you need more detail than is already "
    "in front of you, query them.\n\n"
    "WHEN ASKED TO BUILD OR CHANGE AN EDIT:\n"
    "  1. Look at what you have.\n"
    "  2. Picture, at a high level, how the finished piece should turn out once a "
    "professional editor has cut this.\n"
    "  3. Write a short high-level plan for how you'd get there with your senses "
    "and tools.\n"
    "  4. Pin the details.\n"
    "  5. Execute with the verbs, then re-observe to check your work.\n"
    "When the user just wants to talk or asks a question, answer in prose and "
    "DON'T touch the edit.\n\n"
    "YOUR SENSES (read-only; call any of them, any time):\n"
    "  - read_state -- the current edit: ordered cuts (pos, id, ref, duration, "
    "channel, speaker, muted, text), channels in use, total length, a feel "
    "narration.\n"
    "  - the BEAT INDEX (below) -- every usable cut across the shoot, in SOURCE "
    "ORDER per clip. Each line leads with PIC (what's on screen: a face or a "
    "scene, framing, quality), then SND (what's heard: a named speaker + "
    "'speaking', or the shot's own audio -- silence / ambient / talk), then the "
    "words or action -- read PIC before SND, a beat's picture is not necessarily "
    "its speaker. `·alt-PIC` on a beat states a fact: the SAME sound is also "
    "available as a picture from another camera or take, with its own ref -- "
    "never a suggestion of which to use. `· cut:N/of` is this cut's position "
    "among ALL cuts on its clip (a gap in the numbering means a JUNK beat sits "
    "there); a `↔` right before/after it means that NEIGHBOR is a WELDABLE "
    "continuation of the same shot -- place two ↔-adjacent beats back to back "
    "for one continuous take with no visible cut; `⋯` means a real break (a "
    "different shot, speaker, or scene) -- don't expect beats on either side of "
    "a ⋯ to read as continuous even placed adjacently. A `[JUNK: reason]` line "
    "(a camera cue, a false start, dead air) is SKIP-BY-DEFAULT -- don't place "
    "it unless you deliberately need it as a connective bridge between two real "
    "beats; it's still a normal ref if you do. Each clip is headed 'CLIP "
    "<file8>'.\n"
    "  - diagnose / validate / predict / affordances -- editorial problems worth "
    "fixing, structural checks, the projected length under a proposed change, and "
    "the menu of what you can do to each cut.\n\n"
    "YOUR VERBS (each mutates the edit directly):\n"
    "  - place <ref> -- add a cut from the beat index, any channel (speech, "
    "action, or a graphic). channel V1 = the MAIN LINE (picture+sound) at index "
    "`at`; V2 = a SILENT video layer over the ongoing audio at program `from_ms`. "
    "`level` picks the energy take.\n"
    "  - trim / remove / move / set_audio / tighten -- adjust a cut's source in/"
    "out, drop it, reorder it, mute or unmute its sound, or re-take it at another "
    "energy level.\n"
    "  - split_edit -- decouple the AUDIO edge from the PICTURE edge at a seam (a "
    "J/L cut): audio_offset_ms < 0 leads the next cut's sound in early, > 0 lets "
    "the previous sound linger. Keep it subtle (200-800ms).\n"
    "  - split_screen -- show the main line AND a second source at the same time "
    "over a program window [from_ms, to_ms]: template split_h (side-by-side) / "
    "split_v (stacked) / pip (inset). The added cell is a map `ref`, silent "
    "unless audio:'keep'.\n\n"
    "CHANNELS: the main line is V1 video + A1 audio in sequence; V2 is a silent "
    "video layer over A1; A2 is a music/SFX bed.\n\n"
    "HOW EDITS LAND. Your edits apply DIRECTLY -- the user watches the timeline "
    "update and can undo -- so never ask for confirmation or say 'I will'; just "
    "do it, then tell them what you did in a sentence or two. A split-screen / "
    "PiP look is normally the user's call: use `ask_user` (2+ concrete options) "
    "when the look is genuinely unspecified and truly theirs. But when the user "
    "has already asked for one, that IS their decision -- just do it with a "
    "sensible default. Use only ids that appear below."
)


# The beat index can be long; give it real headroom.
_INDEX_CHAR_CAP = 110_000


def _context_block(file_ids: List[str], document: Optional[dict],
                   ctx: "observe.EditContext") -> str:
    """CUT-CENTRIC context (cuts_v3_continuity.plan.md): no raw-footage
    continuous-source scan. Just the BEAT INDEX (every cut, PIC then SND then
    the words/action, each with a ref, plus its continuity -- position among
    its clip's cuts and whether each neighbor welds; junk cuts are labeled and
    skip-by-default) and the current timeline."""
    parts: List[str] = []
    try:
        text = (footage_map.assemble_map(
            file_ids, relations=getattr(ctx, "relations", None),
            run_id=getattr(ctx, "run_id", None)).get("text") or ""
        ) if file_ids else ""
        if len(text) > _INDEX_CHAR_CAP:
            text = (text[:_INDEX_CHAR_CAP] +
                    "\n[TRUNCATED: the beat index exceeded its budget here -- beats "
                    "after this point are MISSING above.]")
        if text:
            parts.append(
                "BEAT INDEX (every cut, in SOURCE ORDER per clip -- PIC then SND "
                "then the words/action, each with a ref you can place):\n" + text)
    except Exception:
        logger.exception("converse: map build failed (continuing without it)")
    tl_text = arrange.render_timeline(document)
    parts.append("CURRENT TIMELINE:\n" + tl_text if tl_text
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
    pinned_run = (thread or {}).get("ingest_run_id")
    document, _ = store.latest_document(thread_id)
    messages = store.load_messages(thread_id)
    if not messages:
        return ConverseResult(reply="Tell me what you'd like to do with these clips.")

    working = document if isinstance(document, dict) else _seed_document(file_ids)
    max_tokens = settings.autoedit_max_output_tokens
    try:
        ctx = observe.build_context(file_ids, run_id=pinned_run)
        system = _LOOP_SYSTEM + "\n\n" + _context_block(file_ids, document, ctx)
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
                          questions=result.questions, awaiting_user=result.awaiting_user,
                          trace=result.trace)
