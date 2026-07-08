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
    # Ordered tool-call trace for this turn (persisted on the assistant turn so
    # the reasoning is auditable later).
    trace: List[dict] = field(default_factory=list)


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


# --- v2: continuous-source arranger ----------------------------------------
# The clip is no longer a bag of pre-scored speech "moments". It is a
# CONTINUOUS, fully-addressable timeline: the brain sees change-point lanes
# (present / speaking / gaze / shot / action over the whole clock), the clean
# seams, the impact peaks, and a scored cut INDEX -- and it can place ANY span,
# not just a pre-baked said-line. The speech-cut index stays as a fast way to
# lay the spoken spine, but reactions / listening beats / cutaways / b-roll come
# from `source_awareness` (+ `scan_source`) then `place_span`.
_LOOP_SYSTEM_V2 = (
    "You are EDSO, a sharp video editor working ON an edit with real TOOLS, like "
    "a coding agent editing a repo. Below you see the shoot as a CONTINUOUS "
    "SOURCE -- each clip is a fully-addressable timeline with change-point lanes "
    "(who is present / who is speaking on camera / gaze / shot size / action over "
    "the whole clock), its cleanest seams, its impact/reveal PEAKS, and a scored "
    "CUT INDEX -- plus a SPEECH-CUT INDEX of pre-scored said-lines, and the "
    "CURRENT TIMELINE of the edit so far.\n\n"
    "TWO WAYS TO CUT -- USE BOTH.\n"
    "  1. The spoken spine: lay the said-lines fast with `place` (channel V1) "
    "using a speech-cut ref (e.g. ab12cd34:m07). The quoted text is the complete "
    "line -- judge from it. dup:tgN links a line to a COVERAGE GROUP (below): the "
    "same beat delivered more than once, each member showing who and how well -- "
    "read the facts and decide which to show (and you can reuse another delivery "
    "as a cutaway).\n"
    "  2. Anything that ISN'T a clean said-line -- a person's SILENT reaction, a "
    "listening beat, a face just before/after their line, a held action, b-roll -- "
    "is NOT in the speech index. To use it: read `source_awareness` to see the "
    "lanes/peaks, `scan_source` to locate the exact window (e.g. lane "
    "'presence:p2' match {state:'on'} to find where p2 is on screen, then read "
    "each hit's facets to see if they're silent), then `place_span` that window "
    "(the 'CLIP <file8>' id + source in/out ms). Boundaries are AUTO-SNAPPED to "
    "the nearest clean seam (word gap / silence / impact, never mid-word) within "
    "~400ms -- nominate approximate windows and read the tool's `snap` field. An "
    "edge you place deliberately (a match-cut, a mid-motion edge) stays put: the "
    "snapper never moves an edge further than that cap (it only SUGGESTS the "
    "seam back), and snap:'off' skips it entirely.\n"
    "A strong cut breathes: it doesn't just staple talking-head lines back to "
    "back. When one speaker runs long, cut to the LISTENER's reaction; open or "
    "close a beat on a face or an action, not always on a word. Reach for "
    "`place_span` whenever the moment you want to show isn't a spoken line.\n\n"
    "PEOPLE ACROSS CLIPS. The PEOPLE OF THE SHOOT block names each person once "
    "with a global id (G1, G2, ...) that holds across every clip; the speech "
    "index and coverage groups use those ids too. To work with one person "
    "anywhere -- e.g. find a window where they listen silently -- scan the whole "
    "shoot in one call: `scan_source` with file:'*' and a lane like 'presence:G2' "
    "(the global id resolves to each clip's local person). To find a plausible "
    "reaction near a specific beat, take that beat's coverage member window and "
    "pass it as `within_ms` to a scan of another clip.\n\n"
    "CHANNELS. The main line is V1 video + A1 audio, in sequence -- `place` / "
    "`place_span` channel V1. A SILENT video cutaway over the ongoing A1 audio "
    "rides V2 (channel V2 at a program `from_ms`). A music/SFX bed rides A2. "
    "Showing V1 and a second source at the SAME time (side-by-side, stacked, or "
    "PiP) is `split_screen` over a program window. The audio edge at a seam can "
    "LEAD or LAG the picture edge with `split_edit` (a J-cut -- hearing the next "
    "speaker ~300-500ms before seeing them -- makes dialogue flow; use it on "
    "conversational seams, not every cut).\n\n"
    "HOW YOU WORK. When the user just wants to talk, answer in prose and DON'T "
    "touch the edit. When they want a change, use your tools and OBSERVE as you "
    "go (read_state / source_awareness / diagnose / validate), then ACT. Your "
    "edits apply DIRECTLY -- the user watches the timeline update and can undo -- "
    "so never ask for confirmation or say 'I will'; just do it, then tell them "
    "what you did in a sentence or two.\n\n"
    "ACTING ON A LOOK. A split-screen / PiP is normally the user's call. BUT when "
    "the user has already TOLD you to use one (e.g. 'use split screen on the "
    "short exchanges', 'PiP the reactions'), that request IS their decision -- "
    "just DO it with a sensible default (short two-person exchanges -> split_h "
    "side-by-side; a reaction over a talker -> pip), across the beats they meant. "
    "Only use `ask_user` when the look is genuinely unspecified and the choice is "
    "truly theirs (which template, which aspect/framing, a big pacing tradeoff), "
    "with 2+ concrete options. Never turn an explicit instruction back into a "
    "question. Use only ids that appear below."
)


# --- v3: the blind editor -- 100% awareness + tools, plan-first, NO workflow ---
# v2 prescribed WHEN to reach for each verb ("way 1: place the spoken spine; way
# 2: place_span reactions") -- a workflow that biased every edit into a talking
# -head spine with bolted-on garnish. v3 removes the prescription entirely: it
# states WHO the brain is (a blind editor with faithful senses), gives it a plan
# -first discipline (look -> picture the finished piece -> plan -> pin -> execute
# -> check), and lists the senses/verbs as neutral CAPABILITIES + mechanics only.
# It never says "use X for Y" and never enumerates what to notice -- the material
# and the craft are the brain's to read.
_LOOP_SYSTEM_V3 = (
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
    "  - source_awareness -- each clip as a fully-addressable timeline: change"
    "-point lanes (who is present / who is speaking on camera / gaze / shot size "
    "/ action over the whole clock), its cleanest seams, its impact/reveal PEAKS, "
    "and a scored cut INDEX. Each clip is headed 'CLIP <file8>'.\n"
    "  - the BEAT INDEX (below) -- every usable beat across the shoot. Each line "
    "leads with PIC (what's on screen: a face or a scene, framing, quality), then "
    "SND (what's heard: a named speaker + 'speaking', or the shot's own audio -- "
    "silence / ambient / talk), then the words or action -- read PIC before SND, "
    "a beat's picture is not necessarily its speaker. `·alt-PIC` on a beat states "
    "a fact: the SAME sound is also available as a picture from another camera or "
    "take, with its own ref -- never a suggestion of which to use.\n"
    "  - scan_source -- query that timeline for spans matching a facet: a `lane` "
    "(e.g. 'presence:G2', 'speaking', 'shot', 'speech', 'action') + optional "
    "`match` (e.g. {state:'on'}); each hit carries its full facets at its "
    "midpoint. `file` is a 'CLIP <file8>' id or '*' to scan EVERY clip at once; a "
    "global person id (G1, G2, ...) in the lane/match resolves per clip; optional "
    "`within_ms` ([a,b]) limits hits to a window. It is forgiving: a wrong lane "
    "name returns `lanes_available` and a guessed match key is ignored and "
    "reported with the lane's real `lane_vocab` -- read the result and re-query "
    "rather than assuming a token.\n"
    "  - diagnose / validate / predict / affordances -- editorial problems worth "
    "fixing, structural checks, the projected length under a proposed change, and "
    "the menu of what you can do to each cut.\n\n"
    "YOUR VERBS (each mutates the edit directly):\n"
    "  - place <ref> -- add a pre-scored beat from the beat index, any channel "
    "(speech, action, or a graphic). channel V1 = the MAIN LINE (picture+sound) "
    "at index `at`; V2 = a SILENT video layer over the ongoing audio at program "
    "`from_ms`. `level` picks the energy take.\n"
    "  - place_span <file, in_ms, out_ms> -- add ANY source window (the SOURCE ms "
    "into a 'CLIP <file8>'), not just a said-line. Same channels. Edges auto-snap "
    "to the nearest clean seam (word gap / silence / impact; never mid-word) "
    "within ~400ms -- nominate an approximate window and read the result's `snap` "
    "field; an edge whose nearest seam is further stays put, and snap:'off' "
    "places it raw.\n"
    "  - trim / remove / move / set_audio / tighten -- adjust a cut's source in/"
    "out, drop it, reorder it, mute or unmute its sound, or re-take it at another "
    "energy level.\n"
    "  - split_edit -- decouple the AUDIO edge from the PICTURE edge at a seam (a "
    "J/L cut): audio_offset_ms < 0 leads the next cut's sound in early, > 0 lets "
    "the previous sound linger. Keep it subtle (200-800ms).\n"
    "  - split_screen -- show the main line AND a second source at the same time "
    "over a program window [from_ms, to_ms]: template split_h (side-by-side) / "
    "split_v (stacked) / pip (inset). The added cell is a map `ref` OR a raw "
    "`file`+`in_ms`+`out_ms` window (seam-snaps like place_span), silent unless "
    "audio:'keep'.\n\n"
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
    tl_text = arrange.render_timeline(document)
    parts.append("CURRENT TIMELINE:\n" + tl_text if tl_text
                 else "CURRENT TIMELINE: (empty -- no edit drafted yet)")
    return "\n\n".join(parts)


# Split the char budget between the two substrates so neither starves the other
# (the continuous digest is compact; the speech index can be long).
_AWARE_CHAR_CAP = 90_000
_INDEX_CHAR_CAP = 110_000


def _assemble_source_context(file_ids: List[str], document: Optional[dict],
                             ctx: "observe.EditContext", *,
                             aware_header: str, index_header: str) -> str:
    """The shared continuous-source context: the CONTINUOUS SOURCE digest
    (lanes/seams/peaks/cut index), the speech-cut index, and the current
    timeline. The two block HEADERS are supplied by the caller so v2 (workflow
    -framed) and v3 (neutral) can present the identical awareness differently.
    Falls back gracefully if either projection is unavailable."""
    parts: List[str] = []
    try:
        aware = observe.source_awareness(ctx) if file_ids else ""
        if aware and not aware.lstrip().startswith("("):  # skip "(no ... available)" notices
            if len(aware) > _AWARE_CHAR_CAP:
                # Loud, instructive cut -- the brain must know awareness is
                # partial and how to recover it (never silently).
                aware = (aware[:_AWARE_CHAR_CAP] +
                         "\n[TRUNCATED: the digest exceeded its budget here. Clips "
                         "after this point are MISSING above -- call source_awareness "
                         "/ scan_source to read any clip before cutting from it.]")
            parts.append(aware_header + "\n" + aware)
    except Exception:
        logger.exception("converse: source_awareness build failed (continuing)")
    try:
        text = (footage_map.assemble_map(
            file_ids, relations=getattr(ctx, "relations", None),
            run_id=getattr(ctx, "run_id", None)).get("text") or ""
        ) if file_ids else ""
        if len(text) > _INDEX_CHAR_CAP:
            text = (text[:_INDEX_CHAR_CAP] +
                    "\n[TRUNCATED: the speech index exceeded its budget here -- lines "
                    "after this point are MISSING; use source_awareness / scan_source "
                    "(lane 'speech') on the later clips instead of assuming they are empty.]")
        if text:
            parts.append(index_header + "\n" + text)
    except Exception:
        logger.exception("converse: map build failed (continuing without it)")
    tl_text = arrange.render_timeline(document)
    parts.append("CURRENT TIMELINE:\n" + tl_text if tl_text
                 else "CURRENT TIMELINE: (empty -- no edit drafted yet)")
    return "\n\n".join(parts)


def _context_block_v2(file_ids: List[str], document: Optional[dict],
                      ctx: "observe.EditContext") -> str:
    """v2 context: workflow-framed headers (spoken spine + place_span garnish)."""
    return _assemble_source_context(
        file_ids, document, ctx,
        aware_header=("CONTINUOUS SOURCE (each clip as a fully-addressable timeline "
                      "-- lanes, seams, peaks, and a scored cut index; place ANY span "
                      "with place_span):"),
        index_header=("SPEECH-CUT INDEX (pre-scored said-lines -- lay the spoken "
                      "spine fast with `place <ref>`):"))


def _context_block_v3(file_ids: List[str], document: Optional[dict],
                      ctx: "observe.EditContext") -> str:
    """v3 context: identical awareness, NEUTRAL headers -- they describe what each
    block IS and how to reference it, with no 'lay the spine' / 'garnish' framing."""
    return _assemble_source_context(
        file_ids, document, ctx,
        aware_header=("CONTINUOUS SOURCE (each clip as a fully-addressable timeline "
                      "-- change-point lanes, cleanest seams, impact peaks, and a "
                      "scored cut index):"),
        index_header=("BEAT INDEX (every usable beat -- PIC then SND then the "
                      "words/action -- each with a ref you can place):"))


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
    version = (settings.autoedit_arranger_version or "v3").strip().lower()
    try:
        ctx = observe.build_context(file_ids, run_id=pinned_run)
        if version == "v3":
            system = _LOOP_SYSTEM_V3 + "\n\n" + _context_block_v3(file_ids, document, ctx)
        elif version == "v2":
            system = _LOOP_SYSTEM_V2 + "\n\n" + _context_block_v2(file_ids, document, ctx)
        else:
            system = _LOOP_SYSTEM + "\n\n" + _context_block(file_ids, document)
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
