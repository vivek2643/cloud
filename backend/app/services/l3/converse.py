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
import re
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


# Edso -- the blind editor. The prompt is deliberately LEAN: identity, the
# factual mechanics needed to read a beat line, and how the loop operates. It
# carries essentially NO editorial craft and NO usage guidance -- the model
# decides how to edit -- with ONE deliberate exception: a firm default to keep
# each cut's audio with its own picture (avoid split_edit), since detached A/V
# is rarely wanted and easy to get subtly wrong. Format-specific craft still
# lives only in the guidance doc (appended at build time), which is about
# GUESSING under incomplete perception, not craft.
_LOOP_SYSTEM = (
    "You are EDSO, a BLIND video editor. You cannot see or hear the footage -- "
    "you work entirely from faithful text SENSES. The raw clips have been divided "
    "into CUTS, each with a rich description; your job is to understand what the "
    "user wants and assemble those cuts into an edit.\n\n"
    "Figure out what the user is asking for. If they just want to talk or ask a "
    "question, answer in prose and don't touch the edit. If a choice is genuinely "
    "theirs and you can't reasonably settle it, use ask_user; otherwise proceed. "
    "When you do ask, SUGGEST rather than just asking -- set `recommended` to your "
    "pick and `why` to one short reason; reserve ask_user for genuinely user-owned "
    "or materially ambiguous choices, never to offload a guess you could make. "
    "Your edits apply DIRECTLY -- the user watches the timeline update and can "
    "undo -- so don't ask for confirmation or say 'I will'; make the edit, then "
    "tell them what you did in a sentence or two. Use only ids that appear below.\n\n"
    "THE ASK IS YOUR CONTRACT. On a turn that changes the edit, open by restating "
    "in ONE line how you read the goal -- the length if it was given, the must-haves, "
    "and the tone -- then edit to THAT. If the ask is materially ambiguous, or you'd "
    "need a rough length you cannot infer, use ask_user before editing rather than "
    "guessing big. Match ambition to the ask: make the SIMPLEST edit that satisfies "
    "the goal, and don't add layers, cutaways, effects, or pacing moves the ask "
    "didn't call for.\n\n"
    "READING A BEAT LINE. When the shoot's cast was reconciled, a CAST line "
    "lists the shoot's named persons (Px) once, each a short description plus "
    "which voice(s) are confirmed theirs -- read it once to know who's who; "
    "everyone else recognised but not cast-table-worthy is listed by id under "
    "'other'. The BEAT INDEX then lists every usable cut in SOURCE ORDER per "
    "clip (each clip headed 'CLIP <file8>'). A line has PIC (who/what's on "
    "screen: one or more Px ids joined with '+', or a scene/object + framing + "
    "quality) and SND (who's heard: a Px id tagged ON-CAM or OFF-CAM against "
    "THIS cut's OWN picture, or the shot's own audio -- silence/ambient/talk); "
    "a beat's picture is not necessarily its speaker -- SND names the true "
    "speaker even when PIC shows someone else (OFF-CAM), and a voice with no "
    "confident person match renders OFF-CAM with no id, never a guessed name. "
    "Then the quoted text: for a speech beat this is now the VERBATIM words "
    "spoken (not a paraphrase) -- read it to choose dialogue/takes; `vis:\"...\"` "
    "alongside it (when present) is the visual note for what's on screen / how "
    "it looks, and `aud:` is that line's delivery quality (crispness+loudness). "
    "For an action beat the quoted text is still the visual description. "
    "Then tags: "
    "`nrg:` the energy takes `tighten` accepts; `pace:LO-HIx` a video cut's "
    "playback-speed room / `trim<=Xs` a speech cut's removable dead-air budget "
    "(what `retime` reaches); `cam:` the shot's camera move; `cut:N/of` this "
    "cut's position among ALL its clip's cuts (a gap in the numbering means a "
    "JUNK beat sits there), with `↔` = that neighbour welds into one continuous "
    "shot and `⋯` = a real break; `peak:+Xs` (when present) = this cut's single "
    "strongest INSTANT, code-computed, as an offset from the cut's own start -- "
    "lean on it for emphasis / punch-in / hold timing; `·alt-PIC` = the same "
    "sound is also available as a picture from another camera/take (its own "
    "ref). A `[JUNK: reason]` line "
    "(camera cue, false start, dead air) marks a cut flagged as junk; it stays "
    "out of the edit unless you place it.\n\n"
    "GUESS FROM CONTEXT. Read each beat from ALL of its senses at once -- the "
    "words, the picture and sound, the cut's own description, and the signals -- "
    "as ONE reading, not a ranking; lean on whichever is richest at that moment. "
    "The beat words, in SOURCE ORDER, narrate the footage continuously: wherever "
    "a cut's own description is thin, generic, or non-speech, INFER what it most "
    "likely shows and where its key moment falls from the surrounding beats -- "
    "what is being talked about predicts what is on screen, and emphasis ('watch "
    "this', 'look', 'and then--') flags where something matters even when you "
    "can't see it. Reason from that inference rather than treating the gap as "
    "empty or the cut as generic; place, time, and order cuts on the reading the "
    "senses together best support. Guess IN PROPORTION to the evidence: commit "
    "where the senses converge, stay literal where they're thin, and never "
    "assert a specific detail the senses don't support. Guess confidently when "
    "the context points somewhere; only ask when it genuinely underdetermines a "
    "choice that is the user's to make.\n\n"
    "CHANNELS: the main line is V1 video + A1 audio in sequence; V2 is a silent "
    "video layer over A1; A2 is a music/SFX bed. Keep each cut's audio joined to "
    "its own picture by default -- mostly AVOID split_edit (decoupling the A1 "
    "audio edge from the V1 video edge, a J/L cut); only do it when a specific "
    "need clearly calls for it, and be extremely careful when you do. Your senses "
    "(read_state, predict, validate, diagnose, affordances) and edit verbs are "
    "described in the tools; call them as you need."
    "\n\nFINISHING. When you stop editing you'll get one AUTOMATIC CHECK of the edit "
    "against the contract. Never finish with STRUCTURAL problems -- fix them. If "
    "you're over a target length, either trim to it or say in one line why the "
    "current length is right. The rest (speaker runs, low-energy stretches, "
    "redundant takes) is advisory -- act on what serves the goal, ignore the rest."
)


# --------------------------------------------------------------------------
# Format guidance (reference-only style doc, cached with the system prompt)
# --------------------------------------------------------------------------

_GUIDANCE_PATH = os.path.join(os.path.dirname(__file__), "guidance_doc.md")
_guidance_cache: Optional[str] = None


def _load_guidance() -> str:
    """The guidance doc (guidance_doc.md), read once and cached. It's the ONLY
    editorial reference the brain gets -- how to GUESS when the senses leave a
    gap -- appended to the system prompt (so it's part of the cached prefix).
    HTML comments (the authoring notes) are stripped so only the guidance itself
    reaches the model. Empty/missing -> no guidance block."""
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
    return ("\n\nGUIDANCE (how to make good guesses when the senses leave a gap; "
            "blend or override per the user and the footage):\n" + doc) if doc else ""


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
            file_ids, run_id=getattr(ctx, "run_id", None)).get("text") or ""
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


_DUR_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>s(?:ec(?:onds?)?)?|m(?:in(?:ute)?s?)?)\b",
    re.IGNORECASE)
_WORD_MIN_RE = re.compile(r"\b(?:a|one)\s+minute\b", re.IGNORECASE)


def _extract_target_s(text: str) -> Optional[float]:
    """Best-effort target LENGTH (seconds) parsed from the user's OWN words -- e.g.
    '60s', '90 seconds', '2 min', 'a minute', '30-45s' (upper bound). Returns None
    when no explicit length is stated: we never INVENT a target (design choice B)."""
    if not text:
        return None
    best: Optional[float] = None
    for m in _DUR_RE.finditer(text):
        num = float(m.group("num"))
        secs = num * 60.0 if m.group("unit").lower().startswith("m") else num
        best = secs if best is None else max(best, secs)   # a range -> upper bound
    if best is None and _WORD_MIN_RE.search(text):
        best = 60.0
    return best


def _latest_user_text(messages: List[dict]) -> str:
    """The newest user message as plain text (content may be a bare string or a
    list of blocks -- see store.load_messages)."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(b.get("text", "") for b in c
                            if isinstance(b, dict) and b.get("type") == "text")
    return ""


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
    # Anchor the contract: capture an EXPLICIT target length from the user's latest
    # words into the brief so diagnose + the done-gate have something to check
    # against. Only ever SET from a stated number -- never cleared, never invented.
    _target_s = _extract_target_s(_latest_user_text(messages))
    if _target_s is not None:
        working.setdefault("brief", {})["target_duration_s"] = _target_s
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
