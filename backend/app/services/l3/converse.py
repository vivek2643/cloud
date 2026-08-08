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
    "THINK, THEN ACT. Before your first edit, work out your WHOLE approach in your "
    "reasoning -- from the ask and the beat index, what this piece is for, roughly "
    "what to keep and in what order, whose angle to favor, where any requested "
    "feature (a split screen, a bed, a target length) will land, and the rough "
    "length you're aiming for. Then execute that approach directly: decide with "
    "predict/your senses before you place, place the selection you already settled "
    "on, and adjust from there -- don't discover the edit by placing, removing, "
    "and re-placing as you go; that's a sign you skipped the plan.\n\n"
    "PRECEDENCE, WHEN YOU MUST GUESS: the user's ask first, then the guidance "
    "defaults below, then your own judgment. Don't let a guess override a guidance "
    "default unless the user's ask (or a clear material reality) calls for it.\n\n"
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
    "`spec:\"...\"` (when present) is a sharper, footage-derived specific for "
    "the same beat, additive alongside the generic text, never a replacement "
    "for it -- lean on it once it's there. "
    "Then tags: "
    "`nrg:` the energy takes `tighten` accepts; `pace:LO-HIx` a video cut's "
    "playback-speed room / `trim<=Xs` a speech cut's removable dead-air budget "
    "(what `retime` reaches); `cam:` the shot's camera move; `cut:N/of` this "
    "cut's position among ALL its clip's cuts (a gap in the numbering means a "
    "JUNK beat sits there), with `↔` = that neighbour welds into one continuous "
    "shot and `⋯` = a real break; `peak:+Xs` (when present) = this cut's single "
    "strongest INSTANT, code-computed, as an offset from the cut's own start -- "
    "lean on it for emphasis / punch-in / hold timing; `sig:` (when present) = "
    "counts of interior action hits / audio dynamics / silence gaps / internal "
    "shot cuts this cut has (e.g. `sig:act3,shot1`) -- call `inspect_cut` on its "
    "ref for the full offsets/curves behind the count; `·alt-PIC` = the same "
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
    "COMPOSITING MODEL. Think of the edit as stacked tracks, like layers in a "
    "photo or video editor. The bottom track is the main line -- it plays start "
    "to finish and sets the total running time. A clip on any track above covers "
    "what's below it for as long as it's there: if it fills the frame it hides "
    "the track underneath; if it's an inset or side-by-side, you see both. All "
    "sound tracks play at once and mix together; when two overlap, the more "
    "important one stays up front and the other automatically dips beneath it. "
    "Everything lines up on one clock -- positions are in program time (where it "
    "lands in the finished video). Concretely: the main line is V1 video + A1 "
    "audio in sequence; V2 is a silent video layer over A1; A2 is a music/SFX "
    "bed. Keep each cut's audio joined to its own picture by default -- mostly "
    "AVOID split_edit (decoupling the A1 audio edge from the V1 video edge, a "
    "J/L cut); only do it when a specific need clearly calls for it, and be "
    "extremely careful when you do. Your senses (read_state, predict, validate, "
    "diagnose, affordances, audio_state, review) and edit verbs are described in "
    "the tools; call them as you need."
    "\n\nFINISHING. When you stop editing you'll get AUTOMATIC CHECKS of the edit, "
    "in order. Never finish with STRUCTURAL problems -- fix them. If you're over a "
    "target length, either trim to it or say in one line why the current length is "
    "right. Then, in order:\n"
    "(1) FIT TO INTENT -- did a feature you named (a split screen, a bed) actually "
    "land; act on what serves the ask or finish.\n"
    "(2) FIT TO CRAFT -- forget the ask entirely and judge the result the way you'd "
    "judge any finished video handed to you cold: does it look and sound like a "
    "real, high-quality piece of work? Fix what falls short. If you finish anyway "
    "with a known flaw, you must name -- in ONE line -- which of exactly two "
    "reasons applies: the user's ask required it, or the material can't support "
    "better. No other reason ('looks fine anyway') is enough to let it pass; that's "
    "rubber-stamping. Equally, don't invent a fix the material can't support just "
    "to avoid saying (b) -- naming the ceiling honestly is the point.\n"
    "(3) SPECIFIC FLAGS -- the rest (speaker runs, low-energy stretches, redundant "
    "takes, rough heads/tails, overlay fit, audio gaps, loudness balance) is "
    "advisory -- act on what serves the goal, ignore the rest."
)


# brain_perception_upgrade.plan.md Change 2 (base) + cut_structure_and_
# scene_specificity.plan.md Part 4 (REQUIRED update, kept in sync in the
# same PR as Part 1/Part 3 changes -- see that plan for the source of truth
# on cut formation + scene specificity): a distinct, detailed provenance
# section describing HOW everything the brain reads was produced --
# segmentation -> cuts -> scoring -> takes/outlooks -> tags -> identity ->
# scene specificity -> beat index / program map. Purely descriptive: HOW
# something is produced and its known mechanism/limits, never a command to
# trust one signal over another -- the brain reasons about reliability
# itself from that description. A static constant, so it stays inside the
# cached prefix alongside _LOOP_SYSTEM/_guidance_block (see the assembly in
# respond()).
_PROVENANCE = (
    "\n\nHOW YOUR SENSES ARE PRODUCED. Everything below describes how the "
    "text you read was made from the footage.\n\n"
    "1. FROM FOOTAGE TO CUTS. One GPU pass (L1) derives per-file signals "
    "from the proxy and the audio track: motion (action energy, camera "
    "motion, blur, stability), audio (loudness envelope, silence, onsets), "
    "and scene/shot/composition boundaries; a transcript is produced with "
    "word timings and speaker diarization. Speech cuts and video cuts are "
    "formed differently. A speech cut comes from a first text pass that "
    "groups the transcript into cuts along word and shot edges. A video "
    "cut's LOCATION is decided entirely by code, content-first, from those "
    "same L1 signals: an isolated burst of action or audio novelty, a "
    "sustained run of above-baseline action energy (split into separate "
    "moments wherever it genuinely lulls), a scene/composition change, or "
    "-- for repetitive footage (reps, a turntable, a conveyor) -- one "
    "representative cycle, each anchor where a cut begins. Camera motion "
    "and blur only REFINE an edge the content layer already chose: a "
    "stability change, a deliberate move starting or ending, a blur "
    "spike, or an existing shot/scene-transition seam can snap a nearby "
    "edge to a cleaner frame when doing so trims only a sliver, but never "
    "invents a cut on its own, and is suppressed entirely when it falls "
    "inside an ongoing content-bearing move, action run, or "
    "composition-continuous stretch. A stretch that is purely static, "
    "silent, and motionless -- including a stretch of otherwise-generous "
    "padding that itself carries no energy -- is dropped entirely rather "
    "than becoming a cut of its own, or padded into, at all; nothing "
    "manufactures a cut out of dead footage. A vision pass then describes "
    "each already-located cut from sampled frames; the VLM never decides "
    "WHERE a cut is, only what it shows and, for a video cut, which side "
    "of its key moment carries more value (`shape`). Assembly snaps every "
    "boundary to a word edge (speech) or the segmenter's own structural "
    "edge (video).\n\n"
    "2. CUT SPANS AND HERO FRAME. A speech cut's src_in_ms/src_out_ms are "
    "set by snapping to word edges. A video cut's span comes directly from "
    "the content-first segmenter described above -- a best-effort read of "
    "where a moment visually begins and ends from motion/audio/scene "
    "signals, not a judgment about what the moment MEANS; never an "
    "LLM-emitted millisecond either way. hero_ts_ms -- the still shown for "
    "the cut -- is chosen in order: an anchor timestamp when the cut has "
    "one, else the sharpest (least motion-blurred) frame in the span, else "
    "the span's midpoint.\n\n"
    "3. SCORING. speech_quality (a speech cut's delivery) is computed from "
    "how much of the span is clean speech (word timings, minus removable "
    "dead-air/filler) blended with loudness, normalized against that "
    "clip's own range. total_quality blends speech_quality with the "
    "visual score (on-camera presence, framing, sharpness, look) for a "
    "speech cut, or is the visual score alone for a video cut. PIC's "
    "q.XX is that visual score.\n\n"
    "4. TAKES & OUTLOOKS. Cuts sharing the same words and the same setting "
    "(a retry of one shot) are grouped into a take; the group's highest "
    "total_quality member is code-crowned the winner, the rest render as "
    "'take'. Cuts sharing the same words but a different camera "
    "(simultaneous angles of one moment) are grouped into an outlook; "
    "outlook members share one authoritative audio track and are never "
    "ranked against each other.\n\n"
    "5. SALIENCE / PEAK. peak:+Xs is the argmax of a curve fused from "
    "normalized action energy, normalized loudness, and a flat bump at "
    "any onset or word-anchor instant inside the cut's span, expressed as "
    "an offset from the cut's own start.\n\n"
    "6. CAMERA / ENERGY / PACE. cam: is read from the per-hop signed "
    "camera velocity model (pan/tilt/zoom rate + coherence) fit at L1. "
    "nrg: levels and the pace tag (pace:LO-HIx for video, trim<=Xs for "
    "speech) come from the pace envelope -- a set of playback-speed rungs "
    "for a video cut, or the removable dead-air/filler budget for a "
    "speech cut, both derived from the same motion/word signals. The dial "
    "only subdivides or fuses the events already found inside an "
    "already-located cut into tighter or broader pieces as energy rises "
    "or falls -- it has no mechanism to invent a boundary the content "
    "layer above didn't find, or to recover one a dead stretch caused to "
    "be dropped.\n\n"
    "7. CONTINUITY. cut:N/of numbers a cut among ALL of its clip's cuts in "
    "source order, including junk -- a gap in the numbering marks where a "
    "junk beat sits. The ↔ mark toward a neighbor means the seam "
    "classifier found the two cuts weldable into one continuous shot; "
    "⋯ means it found a real break (a shot change, a speaker change, "
    "or a flagged gap).\n\n"
    "8. THE PER-CUT SIGNAL BREADCRUMB (sig:). At ingest, each cut's L1 "
    "signals are scanned for interior structure: local peaks in action "
    "energy or motion-impact points (act), rises/falls in the loudness "
    "envelope (adx), silence gaps (sil), and internal shot or composition "
    "cuts (shot) -- counted only when they fall clearly inside the cut's "
    "span, not right at its edges. sig:act3,shot1 reports the COUNT on "
    "each channel that has any; a channel with nothing interior is simply "
    "absent from the tag.\n\n"
    "9. ON-DEMAND CUT INSPECTION (inspect_cut). Calling inspect_cut on a "
    "cut returns, computed fresh from the same L1 arrays: a downsampled "
    "action-energy curve plus the offsets of its strongest hits, a "
    "downsampled loudness-envelope curve plus rise/fall change offsets "
    "and silence-gap offsets, and the offsets of any internal shot or "
    "composition cuts -- all measured from the cut's own start and "
    "sampled at the L1 hop. When the cut has been vision-enriched it also "
    "returns specifics: the complete scene detail (count, notable_object, "
    "continuity_cue, setting, custom-probe answers, and the full per-"
    "moment shot-list for a merged loose cut) beyond what the beat line's "
    "compact spec: tag shows -- pull it when a beat's spec: line looks "
    "promising and you need the full picture before committing to it.\n\n"
    "10. IDENTITY / CAST. Diarization labels each word with a per-file "
    "speaker id; those voiceprints are clustered across every clip into "
    "global voices. A speaking voice is bound to a person by intersecting "
    "diarized turns against active-speaker intervals detected from face "
    "tracks. On-screen persons come from clustering those same face "
    "tracks across clips. The CAST line lists the shoot's named persons "
    "(Px ids) once, each with the voice(s) confirmed theirs; everyone "
    "else recognized but not cast-table-worthy is listed by id under "
    "'other'.\n\n"
    "11. THE BEAT INDEX AND PROGRAM MAP. The BEAT INDEX lists every usable "
    "cut in source order per clip, each with a placeable ref. The PROGRAM "
    "MAP is rendered from the resolved layer stack -- main line plus any "
    "V2/coverage layers and audio beds -- laid out on the shared program "
    "clock, the same clock read_state and the render use.\n\n"
    "12. SCENE SPECIFICITY (spec:, PROJECT DOMAIN, taxonomy). A cut's "
    "label/summary come from the first, generic vision pass, which names "
    "whatever concrete specifics it can see -- objects, on-screen text, "
    "part numbers, the literal action -- even where it cannot interpret "
    "them. In the background, AFTER cuts are already shown, a text pass "
    "reads every cut together and infers the project's DOMAIN (e.g. 'CNC "
    "machine shop', 'Indian wedding (Gujarati)') and, ONLY when a "
    "genuinely closed answer set exists for that domain, a TAXONOMY of "
    "named categories -- it also writes targeted, footage-derived "
    "questions for cuts whose generic description would benefit from a "
    "closer look. A second, targeted vision pass then answers those "
    "questions from the cut's own frame and writes one short, specific "
    "line: `spec:` in the beat line, ADDITIVE alongside the generic "
    "label/summary, never replacing it. This is a best-effort, model-"
    "inferred derivation -- domain, taxonomy entries, and specifics can be "
    "wrong or approximate, and 'other'/'unsure'/'unknown' are the honest, "
    "correct answer when the evidence does not clearly resolve to "
    "something sharper, not a failure. A cut this background pass has not "
    "reached yet, or a project whose evidence never converges on a "
    "domain, simply has no `spec:` tag and no PROJECT DOMAIN block -- the "
    "generic label/summary alone is exactly today's behavior, not a "
    "degraded one."
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
    return ("\n\nGUIDANCE (binding defaults for guessing when the senses leave a "
            "gap -- follow them unless the user's ask or a clear material reality "
            "calls for otherwise):\n" + doc) if doc else ""


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


def _scene_domain_block(ctx: "observe.EditContext") -> str:
    """PROJECT DOMAIN + closed-set TAXONOMY (cut_structure_and_scene_
    specificity.plan.md Part 3's middle text layer), when it has run -- a
    short, one-time orientation so the brain reads per-cut `spec:` tags
    against a stated frame of reference. '' when enrichment hasn't reached
    this run yet (a normal, common state -- it runs in the background after
    cuts are shown) or found the footage too mixed for a domain to emerge."""
    taxonomy = getattr(ctx, "scene_taxonomy", None)
    if not taxonomy:
        return ""
    domain = (taxonomy.get("domain") or "").strip()
    if not domain or domain == "unknown/mixed":
        return ""
    lines = [f"PROJECT DOMAIN: {domain} (confidence: {taxonomy.get('confidence', 'low')})"]
    evidence = [e for e in (taxonomy.get("evidence") or []) if e]
    if evidence:
        lines.append("  evidence: " + "; ".join(evidence[:5]))
    entries = taxonomy.get("taxonomy") or []
    if entries:
        items = [str(e.get("id")) + (f" ({e['definition']})" if e.get("definition") else "")
                for e in entries if e.get("id")]
        if items:
            lines.append("  known categories: " + ", ".join(items))
    return "\n".join(lines)


def _voiceover_block(ctx: "observe.EditContext") -> str:
    """VOICEOVER / NARRATION SCRIPTS -- every uploaded audio asset that has a
    transcript, shown verbatim with per-sentence timings (voiceover-as-spine).
    Always-on so the brain can read a narration script and sync visual cuts to
    its sentences and the pauses between them, rather than reporting it has "no
    transcript I can read". Iterates ALL audio assets (not just unplaced ones),
    so the script stays visible even after the VO is placed as a bed. Empty
    string when no audio asset carries a transcript (music/SFX only, etc.)."""
    lines: List[str] = []
    for a in getattr(ctx, "audio_assets", []) or []:
        t = a.get("transcript")
        segs = (t or {}).get("segments") or []
        if not segs:
            continue
        kind = "music/lyrics" if a.get("is_musical") else "voiceover/narration"
        lines.append(
            f'"{a.get("name")}" ({kind}, {a.get("dur_ms")}ms) '
            f'[file {observe._fid8(a["file_id"])}]:')
        for s in segs:
            lines.append(f'  [{s.get("start_ms")}-{s.get("end_ms")}ms] {s.get("text", "")}')
    if not lines:
        return ""
    return (
        "VOICEOVER / NARRATION SCRIPTS (verbatim transcript with per-sentence "
        "word-timed boundaries -- the exact words in each uploaded audio track. "
        "Use these sentence boundaries, and the gaps between one sentence's end "
        "and the next one's start, to place/pace/sync visual cuts to the "
        "narration):\n" + "\n".join(lines))


def _context_block(file_ids: List[str], document: Optional[dict],
                   ctx: "observe.EditContext") -> str:
    """CUT-CENTRIC context (cuts_v3_continuity.plan.md): no raw-footage
    continuous-source scan. A PROJECT OVERVIEW (the high-level clip summary the
    workflow reads first), the BEAT INDEX (every cut, PIC then SND then the
    words/action, each with a ref, its pacing room + continuity -- position
    among its clip's cuts and whether each neighbor welds; junk cuts are
    labeled and skip-by-default) -- the Footage Map (sources AVAILABLE) -- then
    the PROGRAM MAP (edso_pacing_audit_timing.plan.md item 2): the assembled
    edit ITSELF, every layer with a stable id, program window, and layout, so
    stacking/overlap is visible from the shared clock + z alone."""
    parts: List[str] = []
    try:
        overview = _project_overview(ctx)
        if overview:
            parts.append(overview)
    except Exception:
        logger.exception("converse: project overview failed (continuing without it)")
    try:
        domain_block = _scene_domain_block(ctx)
        if domain_block:
            parts.append(domain_block)
    except Exception:
        logger.exception("converse: scene domain block failed (continuing without it)")
    try:
        vo = _voiceover_block(ctx)
        if vo:
            if len(vo) > _INDEX_CHAR_CAP:
                vo = vo[:_INDEX_CHAR_CAP] + (
                    "\n[TRUNCATED: voiceover scripts exceeded budget here.]")
            parts.append(vo)
    except Exception:
        logger.exception("converse: voiceover block failed (continuing without it)")
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
    try:
        # Program Map (edso_pacing_audit_timing.plan.md item 2): the ASSEMBLED
        # edit as two small, time-aligned tables built from the fully-resolved
        # layer stack -- pure/cheap, so always-on like everything else here.
        pm_text = arrange.render_program_map(document, durations=ctx.durations,
                                             audio_features=ctx.audio_features)
    except Exception:
        logger.exception("converse: program map render failed (continuing without it)")
        pm_text = ""
    parts.append("CURRENT " + pm_text if pm_text
                 else "CURRENT PROGRAM MAP: (empty -- no edit drafted yet)")
    return "\n\n".join(parts)


def _seed_document(file_ids: List[str]) -> dict:
    """An empty Edit Document the agentic loop builds onto (place/... verbs).
    Mirrors the document shape the rest of the system reads so preview / render
    read it identically once resolved (via ``observe.resolve_doc``)."""
    return {
        "brief": {"goal": None, "aspect": "landscape", "target_duration_s": None, "assumptions": []},
        "format": {"aspect": "landscape"},
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
        ctx = observe.build_context(file_ids, run_id=pinned_run, thread_id=thread_id)
        system = (_LOOP_SYSTEM + _guidance_block() + _PROVENANCE
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
        try:
            from app.services.l3.grade.job import maybe_enqueue
            maybe_enqueue(thread_id, result.document)
        except Exception:
            logger.exception("converse: grade job enqueue failed for thread %s", thread_id)
    return ConverseResult(reply=reply, document=result.document, changed=result.changed,
                          questions=result.questions, awaiting_user=result.awaiting_user,
                          trace=result.trace)
