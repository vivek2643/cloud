"""
Conversational L3: a chat-first, AGENTIC editor over a project's footage + edit.

Design note (why it's shaped this way): a strong model edits best with a NEAR-
EMPTY frame and room to think -- the same way it does when you just hand it the
footage and ask. So the brain works like a coding agent over a repo: it SEES the
whole shoot (footage map) and the current edit, and it has TOOLS -- deterministic
SENSES (``observe``: read_state / predict / validate / diagnose / affordances)
and edit VERBS (``act``: place / trim / remove / move / set_audio / tighten /
retime).

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
import os
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


# Edso -- the blind editor. Identity + workflow are the user's own draft; the
# general CRAFT is inline here (universal, so it belongs to who the editor IS,
# not to a reference), while format-specific STYLES live in the guidance doc
# (appended below at prompt-build time). Senses/verbs are neutral capabilities +
# the mechanics needed to read a beat line; it never enumerates what to notice.
_LOOP_SYSTEM = (
    "You are EDSO, a BLIND video editor. You cannot see or hear the footage -- "
    "you make the best call at every turn from faithful text SENSES. We have "
    "divided all the raw clips into CUTS and given each a rich description; your "
    "job is to understand what the user needs and use those cuts to edit.\n\n"
    "WORKFLOW (when asked to build or change an edit):\n"
    "  1. Understand what the user wants. If it's genuinely ambiguous and truly "
    "theirs to decide, ask; otherwise proceed on a sensible read.\n"
    "  2. Read the PROJECT OVERVIEW -- the high-level summary of the raw clips.\n"
    "  3. Make a rough plan for how to get there with the cuts you have. Consult "
    "FORMAT GUIDANCE for the style that fits this material (blend/override when "
    "the footage or the user says otherwise).\n"
    "  4. Go through the cuts (BEAT INDEX) and edit to realize the plan with the "
    "verbs; re-observe to check your work.\n"
    "When the user just wants to talk or asks a question, answer in prose and "
    "DON'T touch the edit.\n\n"
    "CRAFT (how good work reads):\n"
    "  - Open on the strongest hook; earn attention in the first beat. End with "
    "intent (a payoff/button), don't just stop.\n"
    "  - Keep one clear through-line; every cut earns its place. Cut what doesn't.\n"
    "  - Respect continuity: place ↔-weldable neighbours together for a seamless "
    "take; treat ⋯ as a real break and don't expect it to read continuous.\n"
    "  - Two pacing axes, kept separate: `tighten` = how much of a beat you KEEP "
    "(space); `retime` = the PLAYBACK pace. Default to natural pace and retime "
    "ON PURPOSE (match energy across a montage, lift a slow stretch) -- bad "
    "retiming is the fastest way to look amateur, and SPEECH is never sped, only "
    "trimmed of dead-air.\n"
    "  - You are blind: never invent a face or speaker you aren't told about. "
    "When unsure, make the safe honest choice and say what you assumed.\n\n"
    "READING A BEAT LINE. The BEAT INDEX lists every usable cut in SOURCE ORDER "
    "per clip (each clip headed 'CLIP <file8>'). A line leads with PIC (what's on "
    "screen: a face or scene + framing + quality), then SND (what's heard: a "
    "named speaker 'speaking', or the shot's own audio -- silence/ambient/talk) "
    "-- read PIC before SND, a beat's picture is not necessarily its speaker. "
    "Then the words/action, then tags: `nrg:` the energy takes you can `tighten` "
    "to; `pace:LO-HIx` a video cut's playback-speed room / `trim<=Xs` a speech "
    "cut's removable dead-air budget (what `retime` can do); `cut:N/of` this "
    "cut's position among ALL its clip's cuts (a gap in the numbering means a "
    "JUNK beat sits there), with `↔` = that neighbour welds into one continuous "
    "shot and `⋯` = a real break; `·alt-PIC` = the SAME sound is also available "
    "as a picture from another camera/take (its own ref) -- a fact, not a "
    "suggestion. A `[JUNK: reason]` line (camera cue, false start, dead air) is "
    "SKIP-BY-DEFAULT -- place it only as a deliberate connective bridge.\n"
    "Also call read_state / affordances / diagnose / validate / predict any time "
    "to see the current edit, the menu per cut, problems, checks, or a projected "
    "length.\n\n"
    "YOUR VERBS (each mutates the edit directly):\n"
    "  - place <ref> -- add a cut from the beat index. channel V1 = the MAIN LINE "
    "(picture+sound) at index `at`; V2 = a SILENT video layer over the ongoing "
    "audio at program `from_ms`. `level` picks the energy take.\n"
    "  - tighten -- re-take cut(s) at another ENERGY level (how much of the beat "
    "you keep).\n"
    "  - retime -- set PLAYBACK pace: a video cut plays faster/slower (recorded, "
    "not yet baked into the export length); a speech cut is never sped -- "
    "'faster' shaves its dead-air/fillers instead.\n"
    "  - trim / remove / move / set_audio -- nudge a cut's source in/out, drop "
    "it, reorder it, mute or unmute its sound.\n"
    "  - split_edit -- decouple the AUDIO edge from the PICTURE edge at a seam "
    "(J/L cut): audio_offset_ms < 0 leads the next sound in early, > 0 lets the "
    "previous sound linger. Keep it subtle (200-800ms).\n"
    "  - split_screen -- show the main line AND a second source at once over "
    "[from_ms, to_ms]: split_h / split_v / pip. Silent unless audio:'keep'.\n\n"
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


# --------------------------------------------------------------------------
# Format guidance (reference-only style doc, cached with the system prompt)
# --------------------------------------------------------------------------

_GUIDANCE_PATH = os.path.join(os.path.dirname(__file__), "guidance_doc.md")
_guidance_cache: Optional[str] = None


def _load_guidance() -> str:
    """The format-guidance doc (guidance_doc.md), read once and cached. It's a
    reference of format-specific editing STYLES the brain consults while
    planning -- appended to the system prompt (so it's part of the cached
    prefix). HTML comments (the authoring notes) are stripped so only the
    guidance itself reaches the model. Empty/missing -> no guidance block."""
    global _guidance_cache
    if _guidance_cache is not None:
        return _guidance_cache
    text = ""
    try:
        with open(_GUIDANCE_PATH, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        logger.exception("converse: guidance doc read failed (continuing without)")
    import re
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL).strip()
    _guidance_cache = text
    return text


def _guidance_block() -> str:
    doc = _load_guidance()
    return ("\n\nFORMAT GUIDANCE (reference -- pick/blend the style that fits "
            "this material; it never overrides the user or the footage):\n" + doc) if doc else ""


# The beat index can be long; give it real headroom.
_INDEX_CHAR_CAP = 110_000


def _fmt_dur(ms: int) -> str:
    s = max(0, int(ms)) // 1000
    return f"{s // 60}m{s % 60:02d}s" if s >= 60 else f"{s}s"


def _project_overview(ctx: "observe.EditContext") -> str:
    """The high-level summary of the raw clips (workflow step 2), synthesized
    deterministically from each clip tree's header -- what KIND of material this
    is, how much of it, who's in it, and a one-line logline per clip -- so the
    brain plans against the shoot as a whole before reading individual beats.
    '' when nothing is ingested yet."""
    clips = (ctx.map_struct or {}).get("clips") or []
    if not clips:
        return ""
    total = sum(int(c.get("duration_ms") or 0) for c in clips)
    kinds: List[str] = []
    people: List[str] = []
    for c in clips:
        ct = c.get("content_type")
        if ct and ct not in kinds:
            kinds.append(ct)
        for p in c.get("people") or []:
            if p not in people:
                people.append(p)
    head = f"PROJECT OVERVIEW ({len(clips)} clip{'s' if len(clips) != 1 else ''}, ~{_fmt_dur(total)} total"
    if kinds:
        head += f"; {', '.join(kinds)}"
    if people:
        head += f"; people: {', '.join(people[:8])}"
    head += "):"
    lines = [head]
    for c in clips:
        bits = [_fmt_dur(int(c.get("duration_ms") or 0))]
        if c.get("content_type"):
            bits.append(c["content_type"])
        if c.get("primary_axis"):
            bits.append(f"axis:{c['primary_axis']}")
        n = c.get("moment_count") or len(c.get("moments") or [])
        bits.append(f"{n} cuts")
        logline = (c.get("logline") or "").strip().replace("\n", " ")
        tail = f" -- {logline}" if logline else ""
        lines.append(f"  CLIP {c['file_id'][:8]} \"{c.get('name') or c['file_id'][:8]}\" · "
                     + " · ".join(bits) + tail)
    return "\n".join(lines)


def _context_block(file_ids: List[str], document: Optional[dict],
                   ctx: "observe.EditContext") -> str:
    """CUT-CENTRIC context (cuts_v3_continuity.plan.md): no raw-footage
    continuous-source scan. A PROJECT OVERVIEW (the high-level clip summary the
    workflow reads first), then the BEAT INDEX (every cut, PIC then SND then
    the words/action, each with a ref, its pacing room + continuity -- position
    among its clip's cuts and whether each neighbor welds; junk cuts are labeled
    and skip-by-default) and the current timeline."""
    parts: List[str] = []
    try:
        overview = _project_overview(ctx)
        if overview:
            parts.append(overview)
    except Exception:
        logger.exception("converse: project overview failed (continuing without it)")
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
        system = (_LOOP_SYSTEM + _guidance_block()
                  + "\n\n" + _context_block(file_ids, document, ctx))
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
