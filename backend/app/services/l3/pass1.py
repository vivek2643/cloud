"""
Cuts v3, Pass 1: text-only ingest over ALL clips in a project at once.

One call decides MEANING for the whole project (plan North Star #2): the
final speech grouping, cross-clip take candidates, tentative video groupings,
junk suspects, and a project + per-clip summary. Boundaries stay code-derived
-- the model emits WORD INDICES (speech) and ATOM IDS (video), never a
millisecond; ``lattice.snap_word_edge`` / the atom table resolve those to
precise ms afterward (North Star #1).

Pure-ish core: ``run_pass1(file_rows)`` takes already-loaded ``Lattice``
objects and makes exactly one ``llm.client.complete`` call (mockable, see
``scripts/test_pass1.py``); ``run_pass1_for_project`` is the DB-loading
convenience wrapper for real callers.
"""
from __future__ import annotations

import logging
import re
import statistics
from typing import Any, Dict, List, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

from app.services.l3 import lattice as lt
from app.services.l3.lattice import Lattice
from app.services.l3.seam import BREAK_BOUNDARY_REASONS, Seam, classify_seam
from app.services.llm import client as ic
from app.services.llm.base import text_block

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Output schema (persisted raw on ingest_runs.pass1_output for pass 2 + audit)
# --------------------------------------------------------------------------

class SpeechCut(BaseModel):
    file_id: str
    word_span: Tuple[int, int]
    label: str
    speaker_ids: List[str] = Field(default_factory=list)
    # Consecutive speech cuts the model judges to be ONE continuous spoken
    # beat (a single podcast answer, or a line that resumes after a brief
    # demonstrated action) share a beat_id. It is the model's SEMANTIC call
    # only -- code merges same-beat neighbours deterministically, and ONLY
    # across a weldable seam (no shot change / transition / speaker change
    # between them; see seam.classify_seam). No beat_id -> the cut stands
    # alone. Purely an ingest-time grouping hint: never persisted downstream.
    beat_id: str | None = None


class TakeMember(BaseModel):
    file_id: str
    word_span: Tuple[int, int]


class TakeCandidate(BaseModel):
    group_id: str
    members: List[TakeMember]


class VideoTentativeGroup(BaseModel):
    file_id: str
    atom_ids: List[int]


class JunkSuspect(BaseModel):
    file_id: str
    word_span: Tuple[int, int] | None = None
    atom_ids: List[int] | None = None
    reason: str


class ClipSummary(BaseModel):
    file_id: str
    summary: str


class Pass1Output(BaseModel):
    # Every field below has a safe default, so a model response that wraps
    # its real answer under an unexpected top-level key (observed once in
    # the wild: the whole payload nested under a literal "$PARAMETER_NAME"
    # key) would otherwise validate cleanly as an empty-but-"valid" result --
    # silently discarding everything. extra="forbid" turns that into a loud
    # schema violation instead, so it goes through the normal re-ask path.
    model_config = ConfigDict(extra="forbid")

    speech_cuts: List[SpeechCut] = Field(default_factory=list)
    take_candidates: List[TakeCandidate] = Field(default_factory=list)
    video_tentative_groups: List[VideoTentativeGroup] = Field(default_factory=list)
    junk_suspects: List[JunkSuspect] = Field(default_factory=list)
    project_summary: str = ""
    clip_summaries: List[ClipSummary] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

_SYSTEM = (
    "You are the editing brain preparing a raw-footage shoot for an editor. You "
    "see the WHOLE project at once -- every clip and everything the machine "
    "measured -- so you understand what this footage is and what it is for.\n\n"
    "Given per clip:\n"
    "  - TRANSCRIPT: the full word-timed, diarized transcript. Words are numbered; "
    "[Sx] is the diarized speaker. Diarization is only a rough guide -- it often "
    "fails to split a quick off-camera call-out ('go', 'action', 'cut') from the "
    "on-camera line, so trust the WORDS and their MEANING over the speaker tag.\n"
    "  - VIDEO ATOMS: contiguous non-speech spans, each with raw signals -- "
    "act / peak = mean / peak subject-motion energy (0..1), mot = camera-motion "
    "magnitude (0..1), coh = camera coherence (0..1), anchors = timestamps of "
    "detected motion impacts. Read these to judge what each shot is doing.\n"
    "  - HINTS: where speaker changes and long pauses fall (informational).\n\n"
    "Produce, across all clips:\n"
    "  - speech_cuts: coherent spoken beats, as word-index ranges. A speech cut "
    "is a COMPLETE, DELIVERABLE THOUGHT -- a full sentence or clause, or a "
    "coherent multi-sentence beat -- that an editor could place on its own. It "
    "is NEVER a fragment, a connector, or a trailing tail: do NOT emit a "
    "1-3 word runt ('yeah', 'so', 'and then...', a trailing '...right?') as "
    "its own cut -- absorb it into the adjacent thought it belongs to (the "
    "same range as the line it leads into or trails off from). A held pause "
    "or a demonstrated action in the MIDDLE of one continuous thought is NOT a "
    "boundary -- keep 'to inform you ... there's a catch' as ONE range even if "
    "the speaker pauses dramatically in the middle; never split mid-thought on "
    "a dramatic pause. When one continuous moment is split across consecutive "
    "cuts anyway -- a single podcast-style answer, a sentence that resumes "
    "after a beat -- tag those cuts with the SAME beat_id (your call, from "
    "meaning: what is one delivered thought?). Code merges same-beat cuts "
    "ONLY when the footage between is continuous (no shot change, transition, "
    "or speaker change), so tag freely -- the deterministic guard blocks a "
    "wrong merge (e.g. it will NOT let a false start you flag as junk fuse "
    "into the real line). Code also SPLITS a cut at any real speaker change "
    "(a listener's 'yeah'/'okay'/'right, exactly' stays folded in), so group "
    "by meaning and don't fret about lumping a Q and its answer -- the turn "
    "boundary is recovered for you. Leave beat_id empty for a cut that stands "
    "alone.\n"
    "  - take_candidates: near-identical retakes of the same line (within or across "
    "clips); each member is one whole speech_cut.\n"
    "  - video_tentative_groups: visual moments worth keeping, each a group of "
    "atom_ids that belong together as one continuous shot or action.\n"
    "  - junk_suspects: spans that are NOT part of the delivered piece. Be "
    "aggressive about the one closed class you alone can catch from meaning: spoken "
    "PRODUCTION CUES and counts aimed at the crew, not the audience (e.g. go / "
    "action / cut / rolling / set / reset / 3-2-1), asides to an off-camera "
    "operator, and clear FALSE STARTS the speaker then re-delivers (e.g. 'sorry', "
    "'let me redo that', 'take two'). Outside that class keep the bar HIGH -- if "
    "there is any doubt a span might be wanted, do NOT flag it.\n"
    "  - project_summary, and a one-line summary per clip.\n\n"
    "You are the editor's judgment: how to group, what belongs together, what is a "
    "cue versus a real line -- that is yours to decide from the evidence. Output "
    "ONLY categories, indices, ids and text: word ranges, atom ids, group ids, "
    "labels, reasons. NEVER emit a score, confidence, threshold, or any number you "
    "invented -- every measurement and every millisecond comes from code. Emit WORD "
    "INDICES (speech) and ATOM IDS (video), never timestamps."
)


def _render_clip_block(
    file_id: str, name: str, duration_ms: int, lattice: Lattice, sync_hint: str | None = None,
) -> str:
    lines = [f"=== CLIP {file_id} \"{name}\" ({duration_ms / 1000:.1f}s) ==="]
    if sync_hint:
        lines.append(sync_hint)
    if lattice.words:
        lines.append("TRANSCRIPT (word_idx:word, [Sx] marks a speaker change):")
        parts: List[str] = []
        prev_spk: Any = object()
        for i, w in enumerate(lattice.words):
            spk = w.get("speaker")
            tag = f"[{spk or 'S?'}]" if spk != prev_spk else ""
            parts.append(f"{tag}{i}:{(w.get('text') or '').strip()}")
            prev_spk = spk
        lines.append(" ".join(parts))
        if lattice.hints:
            lines.append("HINTS: " + "; ".join(lattice.hints))
    else:
        lines.append("(no speech)")
    if lattice.atoms:
        lines.append("ATOMS:")
        lines.append(lt.render_atom_table(lattice.atoms))
    return "\n".join(lines)


def build_pass1_blocks(
    file_rows: List[Tuple[str, str, int, Lattice]], sync_hints: Dict[str, str] | None = None,
) -> List[Dict[str, Any]]:
    """One text block per clip. ``file_rows`` = [(file_id, name, duration_ms,
    lattice)]. All clips ride in ONE user turn so pass 1 reasons across the
    whole project at once (cross-clip takes need this). ``sync_hints``
    (audio_sync.plan.md SS7.3): optional file_id -> hint line, feeding pass 1
    the KNOWN simultaneity of a synced multicam group's members explicitly,
    instead of leaving it to guess from (now byte-identical, re-based)
    transcript overlap alone."""
    sync_hints = sync_hints or {}
    return [
        text_block(_render_clip_block(fid, name, dur, lat, sync_hints.get(fid)))
        for fid, name, dur, lat in file_rows
    ]


# --------------------------------------------------------------------------
# Semantic checks (cross-object invariants pydantic's schema can't express;
# fed back through the client's re-ask loop, same pattern as pass2a's)
# --------------------------------------------------------------------------

def _word_spans_in_range(output: Pass1Output, lattices: Dict[str, Lattice]) -> str | None:
    """The one thing about speech grouping code can't repair itself: a
    word_span whose indices don't exist in the transcript at all. Everything
    else (crossing an atom-owned gap, take-member alignment) is fixed
    deterministically by ``enforce_lattice_partition`` -- but a garbage index
    has no right answer, so it goes back through the re-ask loop."""
    for i, sc in enumerate(output.speech_cuts):
        lattice = lattices.get(sc.file_id)
        if lattice is None:
            continue
        a, b = sc.word_span
        if not (0 <= a <= b < len(lattice.words)):
            return (f"speech_cut[{i}] word_span [{a}-{b}] is out of range for "
                    f"{sc.file_id} ({len(lattice.words)} words)")
    return None


def _no_overlapping_speech_cuts(output: Pass1Output) -> str | None:
    """Two speech cuts in the same file with overlapping word ranges. Since
    pass 2a stopped echoing word_spans (backfill_locators copies pass 1's
    verbatim), pass 1 is the only place an overlap can originate -- so it
    must be caught here, in pass 1's own re-ask loop."""
    by_file: Dict[str, List[Tuple[int, int, int]]] = {}
    for i, sc in enumerate(output.speech_cuts):
        by_file.setdefault(sc.file_id, []).append((sc.word_span[0], sc.word_span[1], i))
    for file_id, spans in by_file.items():
        spans.sort()
        for (a0, b0, i0), (a1, b1, i1) in zip(spans, spans[1:]):
            if a1 <= b0:
                return (f"speech_cut[{i0}] words[{a0}-{b0}] and speech_cut[{i1}] "
                        f"words[{a1}-{b1}] overlap in {file_id} -- speech cuts must "
                        f"partition non-overlapping word ranges")
    return None


def _no_speech_cut_swallows_atoms(output: Pass1Output, lattices: Dict[str, Lattice]) -> str | None:
    """A speech cut whose word range contains an atom that is ALSO a member of
    some video_tentative_group would make two final cuts (this speech beat and
    that video cut) claim the same instant -- a real overlap. So the check is
    against GROUPED atoms only. An atom the beat legitimately ABSORBED across a
    weldable seam (``_seam_split``) belongs to no group and produces no video
    cut, so it may sit inside the beat's span -- that is the whole point of the
    speech-to-speech bridge (cuts_v3_speech_bridge.plan.md). This is
    ``enforce_lattice_partition``'s post-condition (asserted there), kept
    separate so tests can probe it directly; ``post._validate_no_overlap`` is
    the final ms-level guard."""
    grouped: Dict[str, set] = {}
    for vg in output.video_tentative_groups:
        grouped.setdefault(vg.file_id, set()).update(vg.atom_ids)
    for i, sc in enumerate(output.speech_cuts):
        lattice = lattices.get(sc.file_id)
        if lattice is None or not lattice.words:
            continue
        a, b = sc.word_span
        if not (0 <= a <= b < len(lattice.words)):
            return (f"speech_cut[{i}] word_span [{a}-{b}] is out of range for "
                    f"{sc.file_id} ({len(lattice.words)} words)")
        span_s = int(lattice.words[a].get("start_ms", 0))
        span_e = int(lattice.words[b].get("end_ms", 0))
        gset = grouped.get(sc.file_id, set())
        for atom in lattice.atoms:
            if atom.atom_id in gset and atom.start_ms >= span_s and atom.end_ms <= span_e:
                return (f"speech_cut[{i}] \"{sc.label}\" words[{a}-{b}] swallows GROUPED "
                        f"video atom {atom.atom_id} [{atom.start_ms}-{atom.end_ms}]ms in "
                        f"{sc.file_id} -- that atom would also become a video cut; split "
                        f"the beat, or the seam should have absorbed the atom out of its "
                        f"group")
    return None


# --------------------------------------------------------------------------
# Deterministic lattice enforcement. Observed against the real API (683s
# single-clip pitch rehearsal with a stutter section: dozens of tiny speech
# fragments separated by long atom-filled pauses): the model repeatedly
# grouped words ACROSS a pause even when the re-ask named the exact violation
# -- burning both attempts and failing the whole run. But where those splits
# fall is not a judgment call at all: it's fully determined by the lattice
# (an atom in the gap means the gap is video territory, period). So code
# does it -- the model owns MEANING (which words go together), the lattice
# owns BOUNDARIES, and this is precisely a boundary. North Star #1.
# --------------------------------------------------------------------------

# A voice must own at least this share of the span's spoken TIME to count as
# a real speaker of the beat. Below it (a "yeah"/"mm-hmm" backchannel, a
# one-word interjection, a diarization glitch) it's dropped -- otherwise a
# 70-second answer that happens to contain a two-word nod gets stamped as a
# two-voice cut and (worse) can end up attributed to the wrong person. The
# dominant speaker is ALWAYS kept regardless of share, so a beat never loses
# its speaker.
_MINOR_VOICE_SHARE = 0.15


def _speaker_ids_for_span(words: List[dict], span: Tuple[int, int]) -> List[str]:
    """The diarized speakers of a word span, DOMINANT-FIRST (most spoken time
    first) with sub-threshold voices dropped -- the deterministic ground truth
    for "who is talking this beat" (voice_first_identity.plan.md: never the
    model's job to guess -- word-level diarization already knows).

    Two jobs, both deterministic:
      * ORDER by spoken time so the first id is the beat's dominant speaker.
        Downstream (`identity/apply._rewrite_cuts`) attributes `speaker_person`
        to this first voice, so a long answer with a brief interjection is
        credited to the person who actually holds the floor, not whoever the
        diarizer happened to emit first.
      * DROP any non-dominant speaker below `_MINOR_VOICE_SHARE` of the span's
        spoken time (backchannels / one-word nods / diar glitches), so a
        single-speaker answer stays a single-voice cut. The top speaker is
        always kept, so this never empties a real beat.

    Spoken time (summed word durations) is the weight; when the lattice has no
    word timings at all it falls back to raw word counts. Unset/None words
    contribute nothing."""
    a, b = span
    ms: Dict[str, int] = {}
    cnt: Dict[str, int] = {}
    for i in range(max(0, a), min(len(words), b + 1)):
        spk = words[i].get("speaker")
        if not spk:
            continue
        cnt[spk] = cnt.get(spk, 0) + 1
        ms[spk] = ms.get(spk, 0) + max(0, int(words[i].get("end_ms", 0)) - int(words[i].get("start_ms", 0)))
    if not cnt:
        return []
    weight = ms if sum(ms.values()) > 0 else cnt
    total = sum(weight.values())
    # Dominant first; label as a stable tiebreak so equal-weight speakers order
    # deterministically (never flap run-to-run).
    ranked = sorted(weight, key=lambda s: (-weight[s], s))
    kept = [ranked[0]]
    kept.extend(s for s in ranked[1:] if total > 0 and weight[s] / total >= _MINOR_VOICE_SHARE)
    return kept


# Short acknowledgment/affirmation tokens a listener drops into whoever holds
# the floor ("yeah", "okay", "right", "mm-hmm"). A run of ONLY these by the
# OTHER speaker is TRANSPARENT to the speaker-change split below -- it stays
# folded into the surrounding beat instead of forcing a cut. This is the whole
# point of the split: a speaker change should cut ~always, EXCEPT a trivial
# backchannel. Interjections that carry real content ("no", "wait", a question,
# a real answer) are deliberately NOT here, so they DO cut. Hesitation fillers
# (um/uh/mm) are already stripped upstream (l1/transcript.is_filler) and never
# reach the lattice, so this is only the affirmations that survive as real
# words. Mirrors l1/dialogue_segments.BACKCHANNEL_WORDS (kept local so pass 1's
# enforcement has no cross-lens import).
_BACKCHANNEL_WORDS = {
    "yeah", "yea", "yeahyeah", "yep", "yup", "yes", "ya", "uhhuh", "mhm",
    "mmhm", "mmhmm", "mm", "hmm", "huh", "okay", "ok", "kay", "right", "sure",
    "exactly", "totally", "definitely", "absolutely", "correct", "gotcha",
    "wow", "nice", "cool", "oh", "ah", "aha", "true",
}
# A same-speaker interjection at most this many words long, made up ENTIRELY of
# backchannel tokens, is treated as a backchannel (not a turn). Longer runs, or
# any run containing a content word, are a real turn and force a cut.
_BACKCHANNEL_MAX_WORDS = 3


def _bc_norm(text: str) -> str:
    """Lowercase, letters/digits only -- 'mm-hmm' -> 'mmhmm', 'Yeah,' ->
    'yeah', 'okay.' -> 'okay' -- so punctuation and hyphenation never hide a
    backchannel token from ``_BACKCHANNEL_WORDS``."""
    return "".join(_TAKE_WORD_RE.findall((text or "").lower()))


def _is_backchannel_run(words: List[dict], lo: int, hi: int) -> bool:
    """True if words[lo..hi] is a short run made up ENTIRELY of backchannel
    tokens (``_BACKCHANNEL_WORDS``) -- an acknowledging "yeah"/"right, exactly"
    a listener drops in, not a turn of their own."""
    if hi - lo + 1 > _BACKCHANNEL_MAX_WORDS:
        return False
    toks = [_bc_norm(words[k].get("text") or "") for k in range(lo, hi + 1)]
    return any(toks) and all(t in _BACKCHANNEL_WORDS for t in toks if t)


def _split_at_speaker_changes(words: List[dict], span: Tuple[int, int]) -> List[Tuple[int, int]]:
    """Split a word span into one piece per SUBSTANTIVE speaker turn: a real
    speaker change becomes a cut boundary, but a short backchannel run
    ("yeah"/"okay"/"right, exactly") by the OTHER speaker is TRANSPARENT -- it
    stays folded into the surrounding beat rather than forcing a boundary.

    Word-level diarization already knows exactly where the speaker changed, so
    this is deterministic ground truth; the model tends to lump a Q+A or an
    interjected turn into one span, and this recovers the turn boundary the
    editor actually wants to cut on. Always returns >= 1 piece covering exactly
    the same words, in order.

    Keys ONLY on per-word ``speaker`` labels, which are shared by index across
    a synced outlook group's angles (replicate_outlook_speech mirrors the
    authoritative audio's words), so every angle splits identically -- the
    byte-identical-span invariant group_outlooks relies on is preserved."""
    a, b = span
    lo = max(0, a)
    hi = min(len(words) - 1, b)
    if hi <= lo:
        return [(a, b)]
    # Contiguous runs of one speaker.
    runs: List[Tuple[int, int, "str | None"]] = []
    rs = lo
    rspk = words[lo].get("speaker")
    for i in range(lo + 1, hi + 1):
        spk = words[i].get("speaker")
        if spk != rspk:
            runs.append((rs, i - 1, rspk))
            rs, rspk = i, spk
    runs.append((rs, hi, rspk))
    if len(runs) == 1:
        return [(a, b)]
    # Walk runs; a backchannel run is transparent (doesn't change the current
    # speaker or open a piece), a substantive run of a NEW speaker opens a cut.
    pieces: List[Tuple[int, int]] = []
    cur_start = a
    cur_spk: "str | None" = None
    for r_lo, r_hi, spk in runs:
        if _is_backchannel_run(words, r_lo, r_hi):
            continue
        if cur_spk is None:
            cur_spk = spk
        elif spk != cur_spk:
            pieces.append((cur_start, r_lo - 1))
            cur_start = r_lo
            cur_spk = spk
    pieces.append((cur_start, b))
    return pieces


def _span_pieces(words: List[dict], atoms: List[Any], span: Tuple[int, int]) -> List[Tuple[int, int]]:
    """``span`` split at every inter-word gap that contains an atom (by
    midpoint -- robust to edge snapping either side). Always returns >= 1
    piece covering the same words."""
    a, b = span
    pieces: List[Tuple[int, int]] = []
    cur = a
    for i in range(a, b):
        gap_lo = int(words[i].get("end_ms", 0))
        gap_hi = int(words[i + 1].get("start_ms", 0))
        if gap_hi <= gap_lo:
            continue
        if any(gap_lo <= (at.start_ms + at.end_ms) // 2 < gap_hi for at in atoms):
            pieces.append((cur, i))
            cur = i + 1
    pieces.append((cur, b))
    return pieces


def _gap_seam(
    words: List[dict], atoms: List[Any], i: int, junk_atom_ids: set,
    beat_lo_ms: int, beat_hi_ms: int, *, synced: bool = False,
):
    """Classify the atom-owned gap between word ``i`` and word ``i+1`` for a beat
    spanning [``beat_lo_ms``, ``beat_hi_ms``]. Returns ``(verdict, gap_atoms)``;
    ``(None, [])`` when the gap holds no atom (a bare inter-word pause -- never a
    boundary on its own). The break-edge test reads the atoms' OWN boundary
    reasons (shot cut / transition strictly inside the gap); an R_ACTION energy
    edge is continuous footage, not a break.

    ``synced`` (audio_sync.plan.md SS7.6, recommended option (a)): this file
    belongs to a synced multicam group, so its own picture is decoupled from
    the speech beat -- a shot boundary in ITS atoms must NOT block an audio
    weld (the brain can still cut to a clean angle on top; welding is now an
    audio/semantic decision only). Skips the break-edge test entirely rather
    than just forcing it False, so a synced gap's verdict never depends on
    which angle happened to supply the atoms."""
    gap_lo = int(words[i].get("end_ms", 0))
    gap_hi = int(words[i + 1].get("start_ms", 0))
    if gap_hi <= gap_lo:
        return None, []
    gap_atoms = [at for at in atoms if gap_lo <= (at.start_ms + at.end_ms) // 2 < gap_hi]
    if not gap_atoms:
        return None, []
    has_break_edge = False if synced else any(
        (at.state_in in BREAK_BOUNDARY_REASONS and gap_lo < at.start_ms < gap_hi)
        or (at.state_out in BREAK_BOUNDARY_REASONS and gap_lo < at.end_ms < gap_hi)
        for at in gap_atoms
    )
    verdict = classify_seam(Seam(
        same_clip=True,
        same_speaker=(words[i].get("speaker") == words[i + 1].get("speaker")),
        gap_ms=gap_hi - gap_lo,
        bridged_speech_ms=max(0, gap_lo - beat_lo_ms) + max(0, beat_hi_ms - gap_hi),
        has_scene_or_transition=has_break_edge,
        has_flagged_break=any(at.atom_id in junk_atom_ids for at in gap_atoms),
    ))
    return verdict, gap_atoms


def _seam_split(
    words: List[dict], atoms: List[Any], span: Tuple[int, int], junk_atom_ids: set,
    *, synced: bool = False,
) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Seam-aware split of a single LLM word range (cuts_v3_speech_bridge.plan.md).
    Like ``_span_pieces``, but an atom-owned gap INSIDE the range is split ONLY
    when the seam is a HARD break (``_gap_seam``); a WELDABLE gap is ABSORBED --
    the beat keeps spanning it and the gap's atoms are returned as ``absorbed``
    (they leave the video pool and play inside the spoken beat). Coverage-fill /
    recovered cuts do NOT use this -- they keep splitting at every gap via
    ``_span_pieces``.

    Returns ``(pieces, absorbed_atom_ids)``; ``pieces`` always covers the same
    words (>= 1 piece)."""
    a, b = span
    beat_lo = int(words[a].get("start_ms", 0))
    beat_hi = int(words[b].get("end_ms", 0))
    pieces: List[Tuple[int, int]] = []
    absorbed: List[int] = []
    cur = a
    for i in range(a, b):
        verdict, gap_atoms = _gap_seam(words, atoms, i, junk_atom_ids, beat_lo, beat_hi, synced=synced)
        if verdict is None:
            continue
        if verdict.weldable:
            absorbed.extend(at.atom_id for at in gap_atoms)
        else:
            pieces.append((cur, i))
            cur = i + 1
    pieces.append((cur, b))
    return pieces, absorbed


def _merge_beats(
    cuts: List[SpeechCut], lattice: Lattice, junk_atom_ids: set, *, synced: bool = False,
) -> Tuple[List[SpeechCut], List[int]]:
    """Merge consecutive same-file speech cuts the model tagged with the SAME
    beat_id into one continuous beat -- but ONLY across a weldable seam (the
    deterministic guard; see seam.classify_seam). This is the LLM-proposes /
    code-disposes half of cuts_v3_speech_bridge.plan.md: the model's beat_id is
    a semantic 'these belong together' call, and code merges only where the
    footage is genuinely continuous (no shot change / transition / speaker
    change; a bare wordless pause with no atom merges when same-speaker).

    Two cuts are candidates only when word-adjacent (``prev.end + 1 ==
    next.start``) -- any words between them belong to another cut, so they are
    not one beat. Absorbed atoms (in a welded gap) are returned to leave the
    video pool. Returns ``(merged_cuts, absorbed_atom_ids)``."""
    words = lattice.words
    atoms = lattice.atoms
    ordered = sorted(cuts, key=lambda s: s.word_span[0])
    out: List[SpeechCut] = []
    absorbed: List[int] = []
    for sc in ordered:
        prev = out[-1] if out else None
        if (prev is not None and prev.beat_id and sc.beat_id
                and prev.beat_id == sc.beat_id
                and prev.word_span[1] + 1 == sc.word_span[0]
                and prev.word_span[1] < len(words) and sc.word_span[0] < len(words)):
            beat_lo = int(words[prev.word_span[0]].get("start_ms", 0))
            beat_hi = int(words[sc.word_span[1]].get("end_ms", 0))
            verdict, gap_atoms = _gap_seam(words, atoms, prev.word_span[1], junk_atom_ids,
                                           beat_lo, beat_hi, synced=synced)
            # No atom in the gap -> a bare short pause; weld iff same speaker
            # (there is no break to cross). An atom present -> trust the full
            # seam verdict (which already checks the speaker).
            if verdict is None:
                same_spk = (words[prev.word_span[1]].get("speaker")
                            == words[sc.word_span[0]].get("speaker"))
                weld = same_spk
            else:
                weld = verdict.weldable
            if weld:
                out[-1] = SpeechCut(file_id=prev.file_id,
                                    word_span=(prev.word_span[0], sc.word_span[1]),
                                    label=prev.label, speaker_ids=list(prev.speaker_ids),
                                    beat_id=prev.beat_id)
                absorbed.extend(at.atom_id for at in gap_atoms)
                continue
        out.append(sc)
    return out, absorbed


def _fold_silent_speech_cuts(cuts: List[SpeechCut], lattice: Lattice) -> List[SpeechCut]:
    """Fold any speech cut whose words carry ZERO audible duration into a
    word-adjacent same-speaker neighbour. A zero-duration transcript word (a
    trailing 'you.' timed end==start) that ends up alone in a cut has no audible
    span AND sits flush against the next atom, so its resolved (src_in, src_out)
    collapses to a point -- a degenerate cut the DB rejects (src_out > src_in).
    Rather than invent time an atom already owns, we re-attach the silent word to
    the spoken beat it belongs to (previous cut if word-touching, else next); an
    isolated silent word with no neighbour is dropped (no audio, nothing lost).
    Deterministic + clip-relative: 'audible' is the word's own end>start, never a
    tuned floor."""
    words = lattice.words
    if not words:
        return cuts

    def _silent(sc: SpeechCut) -> bool:
        a, b = sc.word_span
        return all(int(words[k].get("end_ms", 0)) <= int(words[k].get("start_ms", 0))
                   for k in range(max(0, a), min(len(words), b + 1)))

    atoms = lattice.atoms

    def _bare_gap(i: int) -> bool:
        """No atom sits in the inter-word gap between words i and i+1 -- a bare
        pause/touch across which folding cannot swallow a video atom."""
        gap_lo = int(words[i].get("end_ms", 0))
        gap_hi = int(words[i + 1].get("start_ms", 0))
        if gap_hi <= gap_lo:
            return True
        return not any(gap_lo <= (at.start_ms + at.end_ms) // 2 < gap_hi for at in atoms)

    ordered = sorted(cuts, key=lambda s: s.word_span[0])
    out: List[SpeechCut] = []
    for pos, sc in enumerate(ordered):
        if not _silent(sc):
            out.append(sc)
            continue
        a, b = sc.word_span
        spk = words[a].get("speaker") if 0 <= a < len(words) else None
        prev = out[-1] if out else None
        if (prev is not None and prev.word_span[1] + 1 == a
                and (words[prev.word_span[1]].get("speaker") == spk)
                and _bare_gap(prev.word_span[1])):
            out[-1] = SpeechCut(file_id=prev.file_id, word_span=(prev.word_span[0], b),
                                label=prev.label, speaker_ids=list(prev.speaker_ids),
                                beat_id=prev.beat_id)
            logger.info("pass1 enforce: folded silent words[%d-%d] into preceding beat %r",
                        a, b, prev.label)
            continue
        nxt = ordered[pos + 1] if pos + 1 < len(ordered) else None
        if (nxt is not None and b + 1 == nxt.word_span[0]
                and (words[nxt.word_span[0]].get("speaker") == spk)
                and _bare_gap(b)):
            ordered[pos + 1] = SpeechCut(file_id=nxt.file_id, word_span=(a, nxt.word_span[1]),
                                         label=nxt.label, speaker_ids=list(nxt.speaker_ids),
                                         beat_id=nxt.beat_id)
            logger.info("pass1 enforce: folded silent words[%d-%d] into following beat %r",
                        a, b, nxt.label)
            continue
        logger.info("pass1 enforce: dropping isolated silent words[%d-%d] (no audible span)",
                    a, b)
    return out


# A speech cut this short (word count) AND below the clip's own median is a
# RUNT -- a stray connector or trailing tail, not a deliverable thought (see
# the _SYSTEM speech_cuts bullet's "1-3 word runt" examples). Word-count based,
# never a hardcoded ms: mirrors the model-facing rule with a code-owned bound.
_RUNT_MAX_WORDS = 3


def _is_speech_runt(sc: SpeechCut, median_words: float) -> bool:
    n = sc.word_span[1] - sc.word_span[0] + 1
    return n <= _RUNT_MAX_WORDS and n < median_words


def _absorb_runt_speech_cuts(
    cuts: List[SpeechCut], lattice: Lattice, junk_atom_ids: set,
    claimed_words: set, junk_word_spans: List[Tuple[int, int]], *, synced: bool = False,
) -> Tuple[List[SpeechCut], List[int]]:
    """Deterministically fold a RUNT speech cut -- short by THIS clip's OWN
    word-count distribution, never a hardcoded ms (perception_upgrade.plan.md
    Part E2) -- into the adjacent thought it belongs to. Mirrors
    ``_merge_beats``'s propose/dispose contract, except the 'propose' half is
    code (a cut IS a runt iff ``_is_speech_runt``), not the model's beat_id.

    Absorption requires ALL of: same file + word-contiguous with the
    neighbour, a weldable seam between them (``seam.classify_seam`` -- no shot
    change / transition / speaker change, gap not longer than what it
    bridges), and neither cut is a take/outlook member (``claimed_words``) or
    flagged junk (``junk_word_spans``). Any doubt -> leave it split. Tries
    absorbing BACKWARD (into the preceding cut) first, then any surviving
    runt FORWARD (into the following cut), so a run of consecutive runts
    chains onto whichever real thought is adjacent. Returns
    ``(cuts, absorbed_atom_ids)`` -- absorbed atoms must leave the video pool,
    same as ``_merge_beats``."""
    words = lattice.words
    atoms = lattice.atoms
    ordered = sorted(cuts, key=lambda s: s.word_span[0])
    if len(ordered) < 2 or not words:
        return ordered, []
    median_words = statistics.median(sc.word_span[1] - sc.word_span[0] + 1 for sc in ordered)

    def _claimed_or_junk(sc: SpeechCut) -> bool:
        a, b = sc.word_span
        if any(a <= i <= b for i in claimed_words):
            return True
        return any(a <= ja and jb <= b for ja, jb in junk_word_spans)

    def _try_weld(prev: SpeechCut, nxt: SpeechCut) -> Tuple[bool, List[int]]:
        if prev.file_id != nxt.file_id or prev.word_span[1] + 1 != nxt.word_span[0]:
            return False, []
        if prev.word_span[1] >= len(words) or nxt.word_span[0] >= len(words):
            return False, []
        beat_lo = int(words[prev.word_span[0]].get("start_ms", 0))
        beat_hi = int(words[nxt.word_span[1]].get("end_ms", 0))
        verdict, gap_atoms = _gap_seam(words, atoms, prev.word_span[1], junk_atom_ids,
                                       beat_lo, beat_hi, synced=synced)
        if verdict is None:
            same_spk = (words[prev.word_span[1]].get("speaker") == words[nxt.word_span[0]].get("speaker"))
            return same_spk, []
        return verdict.weldable, [at.atom_id for at in gap_atoms]

    absorbed_atoms: List[int] = []
    n_absorbed = 0

    # Pass 1: absorb a runt BACKWARD into the preceding (already-placed) cut.
    stage1: List[SpeechCut] = []
    for sc in ordered:
        prev = stage1[-1] if stage1 else None
        if (prev is not None and _is_speech_runt(sc, median_words)
                and not _claimed_or_junk(sc) and not _claimed_or_junk(prev)):
            weld, gap_atoms = _try_weld(prev, sc)
            if weld:
                stage1[-1] = SpeechCut(file_id=prev.file_id, word_span=(prev.word_span[0], sc.word_span[1]),
                                       label=prev.label, speaker_ids=list(prev.speaker_ids),
                                       beat_id=prev.beat_id)
                absorbed_atoms.extend(gap_atoms)
                n_absorbed += 1
                continue
        stage1.append(sc)

    # Pass 2: absorb any SURVIVING runt FORWARD into the following cut.
    out: List[SpeechCut] = []
    for sc in reversed(stage1):
        nxt = out[0] if out else None
        if (nxt is not None and _is_speech_runt(sc, median_words)
                and not _claimed_or_junk(sc) and not _claimed_or_junk(nxt)):
            weld, gap_atoms = _try_weld(sc, nxt)
            if weld:
                out[0] = SpeechCut(file_id=nxt.file_id, word_span=(sc.word_span[0], nxt.word_span[1]),
                                   label=nxt.label, speaker_ids=list(nxt.speaker_ids),
                                   beat_id=nxt.beat_id)
                absorbed_atoms.extend(gap_atoms)
                n_absorbed += 1
                continue
        out.insert(0, sc)
    if n_absorbed:
        logger.info("pass1 enforce: runt guard absorbed %d speech cut(s) in %s",
                    n_absorbed, ordered[0].file_id if ordered else "?")
    return out, absorbed_atoms


# Take-identity text match: a take group is "the same LINE said again", so its
# members' transcripts must actually share content words. These two govern that
# SAME-LINE text test only (not cut boundaries): keep a member when at least
# half of the shorter line's distinct content words are shared with the group's
# fullest line, and at least two words overlap (one common word like "business"
# can't glue two different lines). Calibrated on real over-grouped groups: a
# "digital-marketing pitch" vs an "architecture print-shop idea" share ~0.1;
# five takes of one question share 0.5-1.0.
_TAKE_CONTAINMENT_MIN = 0.5
_TAKE_MIN_SHARED = 2
_TAKE_WORD_RE = re.compile(r"[a-z0-9']+")
# Fillers/function words carry no line identity -- stripped so "the the product"
# and "product" match, and so two lines don't bond on "and/you/to".
_TAKE_STOP = {
    "um", "uh", "er", "ah", "like", "you", "know", "so", "the", "a", "an", "and",
    "to", "of", "i", "it", "is", "we", "that", "this", "be", "if", "in", "on",
    "for", "at", "or", "but", "as", "my", "your", "me", "was", "are", "do",
}


def _take_tokens(lattice: Lattice, span: Tuple[int, int]) -> frozenset:
    """Distinct content words in a member's word span (fillers/function words
    dropped) -- the surface the same-line test compares."""
    a, b = span
    out: set = set()
    for k in range(max(0, a), min(len(lattice.words), b + 1)):
        for t in _TAKE_WORD_RE.findall((lattice.words[k].get("text") or "").lower()):
            if t not in _TAKE_STOP:
                out.add(t)
    return frozenset(out)


def _same_line(a: frozenset, b: frozenset) -> bool:
    """True when two members are plausibly the SAME spoken line: enough of the
    shorter one's content words appear in the other (containment, not full-string
    -- survives transcription drift and boundary bleed)."""
    if not a or not b:
        return False
    shared = len(a & b)
    return shared >= _TAKE_MIN_SHARED and shared / min(len(a), len(b)) >= _TAKE_CONTAINMENT_MIN


def _take_content_cluster(
    members: List["TakeMember"], lattices: Dict[str, Lattice]
) -> List["TakeMember"]:
    """Keep only the members that actually say the same line (over-grouping
    guard: the model sometimes lumps different content into one take group).
    Returns the LARGEST set of mutually same-line members; [] when no two agree
    (the caller then drops the group). Deterministic, clip-relative -- no cut
    numbers, just a text-identity check on the members' own transcripts."""
    toks: List[frozenset] = []
    for m in members:
        lat = lattices.get(m.file_id)
        toks.append(_take_tokens(lat, tuple(m.word_span)) if lat else frozenset())
    best: List[int] = []
    for i in range(len(members)):
        cluster = [i] + [j for j in range(len(members))
                         if j != i and _same_line(toks[i], toks[j])]
        if len(cluster) > len(best):
            best = cluster
    return [members[i] for i in sorted(best)] if len(best) >= 2 else []


def _contiguous_atom_runs(lattice: Lattice, atom_ids: List[int]) -> List[List[int]]:
    """``atom_ids`` split into time-contiguous runs (next.start_ms ==
    cur.end_ms). Unknown ids are dropped. A video group must be one
    continuous stretch of timeline: observed against the real API, pass 1
    grouped atoms from BOTH SIDES of a speech span into one group, whose
    resolved (min..max) span then bracketed the speech -- an unfixable
    cross-kind overlap downstream."""
    by_id = {a.atom_id: a for a in lattice.atoms}
    members = [by_id[i] for i in sorted(set(atom_ids)) if i in by_id]
    runs: List[List[int]] = []
    for a in members:
        if runs and by_id[runs[-1][-1]].end_ms == a.start_ms:
            runs[-1].append(a.atom_id)
        else:
            runs.append([a.atom_id])
    return runs


# --------------------------------------------------------------------------
# Outlook grouping. An OUTLOOK group's members are alternate cameras of the
# SAME moment (the user declared them). The offset alignment already knows --
# from the cross-correlation -- how they line up, so the outlook grouping is
# CODE's to produce, not the model's to guess (it consistently fails to, and
# the outlook hint tells it not to try).
#
#   replicate_outlook_speech: BEFORE enforcement, mirror the authoritative
#     angle's speech beats onto every angle in the group. Every angle already
#     carries the byte-identical (re-based) authoritative word list
#     (lattice_merge.authoritative_view), so a beat's word-INDEX span transfers
#     verbatim regardless of alignment confidence. Enforcement then runs per
#     angle with synced=True (a shot boundary never splits a beat; interior
#     atoms are absorbed), yielding byte-identical final speech cuts across the
#     group.
#   group_outlooks: AFTER enforcement, link those aligned per-angle cuts into
#     take_candidates -- one per beat. Pass 2a resolves each as an OUTLOOK
#     (same words, different setting -> NO crowned winner), which is exactly the
#     structure footage_map/observe's alt-PIC angle switching consumes.
#
# ALL declared members are grouped (confidence is metadata, never an
# exclusion): a member is an outlook of the others by definition, and demoting
# a low-confidence one to an independent clip is exactly what let the model
# mis-group it as a TAKE before. Takes and outlooks are distinct -- a declared
# group's members are never takes of each other (see the take-remap filter in
# enforce_lattice_partition).
# --------------------------------------------------------------------------

def _canonical_beats(
    owner_fid: str, cuts_by_file: Dict[str, List[SpeechCut]], owner_lattice: Lattice,
) -> List[Tuple[Tuple[int, int], str, List[str], "str | None"]]:
    """The full, gap-free beat set for an outlook group, defined ONCE on
    ``owner_fid`` (the authoritative angle) so every angle can mirror the same
    spans. The owner's model speech cuts are kept verbatim; any transcript word
    they leave uncovered is recovered EXACTLY as ``enforce_lattice_partition``'s
    speech coverage-fill would (``_span_pieces`` at every atom-owned gap, label
    "(recovered)") -- so once each mirrored angle passes through enforcement its
    coverage-fill finds everything already covered and adds nothing angle-local
    (that per-angle recovery is what produced misaligned cuts before this).
    Returned sorted by word start."""
    words = owner_lattice.words
    beats: List[Tuple[Tuple[int, int], str, List[str], "str | None"]] = []
    covered = [False] * len(words)
    for c in sorted(cuts_by_file.get(owner_fid, []), key=lambda s: s.word_span[0]):
        a, b = c.word_span
        for k in range(max(0, a), min(len(covered), b + 1)):
            covered[k] = True
        beats.append((tuple(c.word_span), c.label, list(c.speaker_ids), c.beat_id))
    i = 0
    while i < len(covered):
        if covered[i]:
            i += 1
            continue
        j = i
        while j < len(covered) and not covered[j]:
            j += 1
        for pa, pb in _span_pieces(words, owner_lattice.atoms, (i, j - 1)):
            beats.append(((pa, pb), "(recovered)", [], None))
        i = j
    beats.sort(key=lambda x: x[0][0])
    return beats


def replicate_outlook_speech(
    output: Pass1Output, lattices: Dict[str, Lattice],
    groups: Dict[str, Dict[str, Any]],
) -> Pass1Output:
    """Mirror each outlook group's authoritative-angle speech beats onto EVERY
    angle in the group, BEFORE ``enforce_lattice_partition``. Each angle ends up
    with byte-identical word spans, so enforcement -- run per angle with
    ``synced=True`` -- yields identical final speech cuts across the group,
    ready to link as outlooks (``group_outlooks``). Angles NOT in a group are
    untouched (their own model cuts + normal coverage-fill stand)."""
    if not groups:
        return output
    cuts_by_file: Dict[str, List[SpeechCut]] = {}
    for sc in output.speech_cuts:
        cuts_by_file.setdefault(sc.file_id, []).append(sc)
    handled: Set[str] = set()
    replicas: List[SpeechCut] = []
    for grp in groups.values():
        angles = sorted(f for f in grp["members"] if f in lattices and lattices[f].words)
        if len(angles) < 2:
            continue
        auth = grp["auth"]
        owner = auth if (auth in angles and cuts_by_file.get(auth)) else max(
            angles, key=lambda f: len(cuts_by_file.get(f, [])))
        beats = _canonical_beats(owner, cuts_by_file, lattices[owner])
        if not beats:
            continue
        for f in angles:
            # This angle's footage-clipped authoritative words (lattice_merge.
            # authoritative_view): a shorter/later-starting angle holds only a
            # prefix of the owner's words. Indices are shared with the owner up
            # to that prefix, so a beat is mirrored by indexing straight into
            # this angle's words -- clamped to the last word it actually has.
            nw = len(lattices[f].words)
            emitted = 0
            for span, label, spk, beat_id in beats:
                a, b = span
                if a >= nw:
                    continue  # this angle's footage ends before the beat starts
                replicas.append(SpeechCut(file_id=f, word_span=(a, min(b, nw - 1)),
                                          label=label, speaker_ids=list(spk), beat_id=beat_id))
                emitted += 1
            # Only take over an angle we actually placed beats on; an angle that
            # overlaps none of the owner's beats keeps its own model cuts.
            if emitted:
                handled.add(f)
    if not handled:
        return output
    kept = [sc for sc in output.speech_cuts if sc.file_id not in handled]
    return output.model_copy(update={"speech_cuts": kept + replicas})


def group_outlooks(
    output: Pass1Output, groups: Dict[str, Dict[str, Any]],
) -> Pass1Output:
    """Link the (now enforcement-aligned) speech cuts of each outlook group's
    angles into take_candidates -- one per beat, its members the
    identical-word-span cut on each angle. Runs AFTER
    ``enforce_lattice_partition`` (so members reference final cut spans exactly,
    needing none of enforce's remap/same-line repair) and BEFORE pass 2a (which
    resolves each as an OUTLOOK from the pixels -> no crowned winner)."""
    if not groups:
        return output
    spans_by_file: Dict[str, Set[Tuple[int, int]]] = {}
    for sc in output.speech_cuts:
        spans_by_file.setdefault(sc.file_id, set()).add(tuple(sc.word_span))
    new_takes: List[TakeCandidate] = []
    for gid, grp in groups.items():
        angles = sorted(f for f in grp["members"] if f in spans_by_file)
        if len(angles) < 2:
            continue
        span_members: Dict[Tuple[int, int], List[str]] = {}
        for f in angles:
            for span in spans_by_file[f]:
                span_members.setdefault(span, []).append(f)
        for span in sorted(span_members):
            files = span_members[span]
            if len(files) < 2:
                continue
            members = [TakeMember(file_id=f, word_span=span) for f in sorted(files)]
            new_takes.append(TakeCandidate(
                group_id=f"outlook:{gid[:8]}:{span[0]}-{span[1]}", members=members))
    if not new_takes:
        return output
    return output.model_copy(update={"take_candidates": output.take_candidates + new_takes})


def enforce_lattice_partition(
    output: Pass1Output, lattices: Dict[str, Lattice], outlook_file_ids: "set | None" = None,
    outlook_group_of: "dict | None" = None,
) -> Pass1Output:
    """Deterministically repair pass 1's grouping against the lattice, and
    guarantee TOTAL COVERAGE -- the load-bearing rule of deterministic-keep
    (cuts_v3_deterministic_keep.plan.md): the model chooses MEANING (grouping,
    takes, what is junk), but it can never silently drop a word or an atom.
    Anything it leaves out of every group is surfaced as a recovered candidate;
    the ONLY way something disappears is an explicit, recoverable junk label.

    ``outlook_file_ids``: file_ids belonging to an OUTLOOK group (alternate
    cameras of one moment), where a video shot boundary must NOT block an audio
    weld (see `_gap_seam`'s ``synced`` param). None/empty -> identical to
    today's behavior for every file (the no-regression guarantee).

    ``outlook_group_of`` (``{file_id: group_id}``): used to DROP any model
    take-candidate whose members all live in one outlook group -- those clips
    are alternate angles (outlooks), never retakes of each other, so code owns
    their grouping (``group_outlooks``) and the model's take verdict is void.

      1. Split any speech_cut at atom-owned gaps that are a HARD seam
         (``_seam_split`` / ``seam.classify_seam``); a WELDABLE gap is ABSORBED
         -- the beat spans it as one continuous cut and the gap's atoms leave
         the video pool (cuts_v3_speech_bridge.plan.md).
      2. COVERAGE FILL (speech): every transcript word the model left out of
         all speech_cuts is re-added as a recovered cut (split at EVERY
         atom-owned gap -- recovered spans were never claimed as a beat, so
         they don't bridge).
      2b. BEAT MERGE: fuse consecutive same-beat_id speech cuts into one beat,
         but ONLY across a weldable seam (``_merge_beats`` / ``seam``). The
         model proposes the grouping (beat_id); code merges only where the
         footage is continuous. Welded gaps' atoms are absorbed too.
      3. Remap every take member onto the (possibly split) speech_cut that
         contains most of its words -- so member == whole cut always holds
         downstream (pass 2a source_refs, image_plan joins). Groups left
         with fewer than two distinct members are dropped.
      4. Split any video_tentative_group at time discontinuities (see
         ``_contiguous_atom_runs``) -- a group must be one continuous
         stretch, never a bridge across speech.
      5. COVERAGE FILL (video): every atom the model left out of all groups
         (and not absorbed into a beat in step 1) is re-added, contiguous
         ungrouped atoms folded into one group.

    Post-condition asserted: no speech cut swallows a GROUPED atom (an absorbed
    one legitimately plays inside its beat)."""
    # Atoms a pass-1 junk suspect flags -- a flagged break inside a gap makes
    # its seam hard (seam.classify_seam), so a beat never bridges across a cue.
    junk_atom_ids: Dict[str, set] = {}
    for js in output.junk_suspects:
        if js.atom_ids:
            junk_atom_ids.setdefault(js.file_id, set()).update(js.atom_ids)

    # Word-level junk spans + take/outlook-claimed words, gathered up front for
    # the runt-absorption guard (step 2c below): a runt never absorbs across a
    # flagged span, and never touches a take/outlook member's word_span (its
    # identity must stay exactly what pass 2a / group_outlooks expect).
    junk_word_spans: Dict[str, List[Tuple[int, int]]] = {}
    for js in output.junk_suspects:
        if js.word_span is not None:
            junk_word_spans.setdefault(js.file_id, []).append(tuple(js.word_span))
    claimed_words: Dict[str, set] = {}
    for tc in output.take_candidates:
        for m in tc.members:
            claimed_words.setdefault(m.file_id, set()).update(range(m.word_span[0], m.word_span[1] + 1))

    # Atoms a beat ABSORBED across a weldable seam: they leave the video pool
    # (below) and play inside the spoken beat instead of becoming a video cut.
    absorbed_atoms: Dict[str, set] = {}
    new_cuts: List[SpeechCut] = []
    synced_ids = outlook_file_ids or set()
    group_of = outlook_group_of or {}
    for sc in output.speech_cuts:
        lattice = lattices.get(sc.file_id)
        if lattice is None or not lattice.words:
            new_cuts.append(sc)
            continue
        pieces, absorbed = _seam_split(lattice.words, lattice.atoms, tuple(sc.word_span),
                                       junk_atom_ids.get(sc.file_id, set()),
                                       synced=sc.file_id in synced_ids)
        if absorbed:
            absorbed_atoms.setdefault(sc.file_id, set()).update(absorbed)
            logger.info("pass1 enforce: beat %r words[%d-%d] absorbed %d atom(s) across "
                        "weldable seam(s)", sc.label, sc.word_span[0], sc.word_span[1], len(absorbed))
        if len(pieces) == 1:
            new_cuts.append(SpeechCut(file_id=sc.file_id, word_span=pieces[0],
                                      label=sc.label, speaker_ids=list(sc.speaker_ids),
                                      beat_id=sc.beat_id))
        else:
            logger.info("pass1 enforce: splitting speech_cut %r words[%d-%d] into %d pieces "
                        "at hard seams", sc.label, sc.word_span[0], sc.word_span[1], len(pieces))
            for j, (pa, pb) in enumerate(pieces, start=1):
                new_cuts.append(SpeechCut(file_id=sc.file_id, word_span=(pa, pb),
                                          label=f"{sc.label} ({j}/{len(pieces)})",
                                          speaker_ids=list(sc.speaker_ids), beat_id=sc.beat_id))

    # Coverage fill (speech): the model can flag a word junk, but it can't
    # silently drop it. Any word covered by no speech_cut becomes a recovered
    # cut, split at atom-owned gaps so it never swallows an atom.
    for file_id, lattice in lattices.items():
        if not lattice.words:
            continue
        covered = [False] * len(lattice.words)
        for sc in new_cuts:
            if sc.file_id != file_id:
                continue
            a, b = sc.word_span
            for k in range(max(0, a), min(len(covered), b + 1)):
                covered[k] = True
        i = 0
        while i < len(covered):
            if covered[i]:
                i += 1
                continue
            j = i
            while j < len(covered) and not covered[j]:
                j += 1
            logger.info("pass1 enforce: recovering uncovered words[%d-%d] in %s", i, j - 1, file_id)
            for pa, pb in _span_pieces(lattice.words, lattice.atoms, (i, j - 1)):
                new_cuts.append(SpeechCut(file_id=file_id, word_span=(pa, pb),
                                          label="(recovered)", speaker_ids=[]))
            i = j

    # Speaker-change split: a real speaker change is a cut boundary. The model
    # routinely lumps a Q+A or an interjected turn into one span; word-level
    # diarization already knows exactly where the speaker changed, so split
    # there deterministically -- EXCEPT across a trivial backchannel
    # ("yeah"/"okay"/"right, exactly") by the other speaker, which stays folded
    # into the beat (``_split_at_speaker_changes``). Runs AFTER coverage fill
    # (every word is now in some cut) and BEFORE beat-merge/runt-absorb: a
    # speaker change is a HARD seam, so neither of those re-welds these pieces.
    # Purely additive (only splits an existing cut, never merges two), and keyed
    # only on per-word speaker labels -- shared by index across a synced outlook
    # group's angles -- so it's safe to run for synced files too (every angle
    # splits identically).
    split_cuts: List[SpeechCut] = []
    for sc in new_cuts:
        lattice = lattices.get(sc.file_id)
        if lattice is None or not lattice.words:
            split_cuts.append(sc)
            continue
        pieces = _split_at_speaker_changes(lattice.words, tuple(sc.word_span))
        if len(pieces) == 1:
            split_cuts.append(sc.model_copy(update={"word_span": pieces[0]}))
            continue
        logger.info("pass1 enforce: splitting speech_cut %r words[%d-%d] into %d "
                    "turn(s) at speaker changes", sc.label, sc.word_span[0],
                    sc.word_span[1], len(pieces))
        for j, (pa, pb) in enumerate(pieces, start=1):
            label = f"{sc.label} ({j}/{len(pieces)})" if sc.label else sc.label
            split_cuts.append(sc.model_copy(update={"word_span": (pa, pb), "label": label}))
    new_cuts = split_cuts

    # Beat merge: fuse same-beat_id neighbours across a WELDABLE seam (the model
    # proposes with beat_id, code disposes via the deterministic seam guard).
    # Runs after coverage fill so recovered cuts (beat_id=None) are present but
    # never merge; absorbed atoms leave the video pool below.
    by_file_cuts: Dict[str, List[SpeechCut]] = {}
    for sc in new_cuts:
        by_file_cuts.setdefault(sc.file_id, []).append(sc)
    merged_cuts: List[SpeechCut] = []
    for file_id, file_cuts in by_file_cuts.items():
        lattice = lattices.get(file_id)
        if lattice is not None and lattice.words:
            file_cuts, merged_absorbed = _merge_beats(file_cuts, lattice,
                                                      junk_atom_ids.get(file_id, set()),
                                                      synced=file_id in synced_ids)
            if merged_absorbed:
                absorbed_atoms.setdefault(file_id, set()).update(merged_absorbed)
                logger.info("pass1 enforce: merged same-beat neighbours in %s, absorbing "
                            "%d atom(s)", file_id, len(merged_absorbed))
            # Fold any zero-audible-duration cut into its neighbour so no speech
            # cut resolves to a degenerate (src_out <= src_in) span downstream.
            file_cuts = _fold_silent_speech_cuts(file_cuts, lattice)
            # 2c. RUNT GUARD: absorb a cut that is short by this clip's OWN
            # word-count distribution into the weldable neighbour it belongs
            # to (perception_upgrade.plan.md Part E2). Skipped for outlook/
            # synced files -- group_outlooks (run after this function) links
            # angles by BYTE-IDENTICAL word spans, an invariant this file's
            # own local seam/junk pattern could break if run per-angle.
            if file_id not in synced_ids:
                file_cuts, runt_absorbed = _absorb_runt_speech_cuts(
                    file_cuts, lattice, junk_atom_ids.get(file_id, set()),
                    claimed_words.get(file_id, set()), junk_word_spans.get(file_id, []),
                    synced=False,
                )
                if runt_absorbed:
                    absorbed_atoms.setdefault(file_id, set()).update(runt_absorbed)
        merged_cuts.extend(file_cuts)
    new_cuts = merged_cuts

    def _dominant_cut_span(file_id: str, span: Tuple[int, int]) -> Tuple[int, int] | None:
        best: Tuple[int, int] | None = None
        best_overlap = 0
        for sc in new_cuts:
            if sc.file_id != file_id:
                continue
            overlap = min(span[1], sc.word_span[1]) - max(span[0], sc.word_span[0]) + 1
            if overlap > best_overlap:
                best_overlap, best = overlap, tuple(sc.word_span)
        return best

    new_takes: List[TakeCandidate] = []
    for tc in output.take_candidates:
        # Outlook guard: a candidate whose members all sit in ONE outlook group
        # is alternate angles of the same moment, not retakes -- drop it whole.
        # Code re-groups those beats as outlooks (`group_outlooks`) so they get
        # no crowned winner; keeping the model's take verdict would fabricate a
        # false winner + pull in that angle's non-authoritative audio.
        member_groups = {group_of.get(m.file_id) for m in tc.members}
        if len(tc.members) >= 2 and len(member_groups) == 1 and None not in member_groups:
            logger.info("pass1 enforce: dropping take group %r -- members are one "
                        "outlook group (angles, not takes)", tc.group_id)
            continue
        seen: set = set()
        members: List[TakeMember] = []
        for m in tc.members:
            mapped = _dominant_cut_span(m.file_id, tuple(m.word_span))
            if mapped is None or (m.file_id, mapped) in seen:
                continue
            seen.add((m.file_id, mapped))
            members.append(TakeMember(file_id=m.file_id, word_span=mapped))
        # Same-line guard: drop members the model lumped in that don't actually
        # say this line (over-grouping -> otherwise pass 2 is forced to "resolve"
        # different content as one take and crowns two winners).
        kept = _take_content_cluster(members, lattices)
        if len(kept) >= 2:
            new_takes.append(TakeCandidate(group_id=tc.group_id, members=kept))
        else:
            logger.info("pass1 enforce: dropping take group %r (%d member(s) remain "
                        "after same-line guard; was %d)", tc.group_id, len(kept), len(members))

    new_groups: List[VideoTentativeGroup] = []
    for vg in output.video_tentative_groups:
        lattice = lattices.get(vg.file_id)
        if lattice is None:
            new_groups.append(vg)
            continue
        # Contiguity split only (structural: a group must be one time-continuous
        # run, else its resolved span brackets a speech beat -- an unfixable
        # cross-kind overlap). NOT action isolation: grouping/merging is the
        # model's job now (signal-judge), so a swing and its follow-through can
        # ride in one group if the model says so.
        # Drop any atom a speech beat absorbed across a weldable seam -- it now
        # plays inside that beat, so it must not ALSO become a video cut.
        absorbed_here = absorbed_atoms.get(vg.file_id, set())
        kept_ids = [i for i in vg.atom_ids if i not in absorbed_here]
        if not kept_ids:
            continue
        runs = _contiguous_atom_runs(lattice, kept_ids)
        if len(runs) == 1 and runs[0] == kept_ids:
            new_groups.append(VideoTentativeGroup(file_id=vg.file_id, atom_ids=kept_ids))
        else:
            logger.info("pass1 enforce: splitting video group atoms=%s into %d "
                        "contiguous run(s)", kept_ids, len(runs))
            for run in runs:
                new_groups.append(VideoTentativeGroup(file_id=vg.file_id, atom_ids=run))

    # Coverage fill (video): every atom must land in some group. Grouping is
    # the model's call, but an atom it left out of every group is not lost --
    # it's re-added, with contiguous ungrouped atoms folded into one recovered
    # group (coverage without confetti). Junk stays the model's recoverable
    # label; nothing vanishes for merely being ungrouped.
    grouped_ids: Dict[str, set] = {}
    for vg in new_groups:
        grouped_ids.setdefault(vg.file_id, set()).update(vg.atom_ids)
    for file_id, lattice in lattices.items():
        # Grouped OR absorbed atoms are already spoken for; only genuinely
        # ungrouped, un-absorbed atoms need a recovered group.
        have = grouped_ids.get(file_id, set()) | absorbed_atoms.get(file_id, set())
        ungrouped = sorted((a for a in lattice.atoms if a.atom_id not in have),
                           key=lambda a: a.start_ms)
        if not ungrouped:
            continue
        run: List[Any] = []
        for a in ungrouped:
            if run and run[-1].end_ms == a.start_ms:
                run.append(a)
            else:
                if run:
                    new_groups.append(VideoTentativeGroup(
                        file_id=file_id, atom_ids=[x.atom_id for x in run]))
                run = [a]
        if run:
            new_groups.append(VideoTentativeGroup(
                file_id=file_id, atom_ids=[x.atom_id for x in run]))
        logger.info("pass1 enforce: recovered %d ungrouped atom(s) in %s",
                    len(ungrouped), file_id)

    # Deterministic per-beat voice (voice_first_identity.plan.md): "who is
    # talking this beat" was never the model's job to guess -- word-level
    # diarization already knows exactly who spoke each word. Stamp
    # speaker_ids from the FINAL (post-split/merge/absorb) word span,
    # overwriting whatever (always empty -- the model is never asked for
    # this field) the raw cut carried in.
    new_cuts = [
        sc.model_copy(update={"speaker_ids": _speaker_ids_for_span(lattices[sc.file_id].words, sc.word_span)})
        if sc.file_id in lattices and lattices[sc.file_id].words else sc
        for sc in new_cuts
    ]

    enforced = output.model_copy(update={"speech_cuts": new_cuts, "take_candidates": new_takes,
                                         "video_tentative_groups": new_groups})
    leftover = _no_speech_cut_swallows_atoms(enforced, lattices)
    if leftover:
        raise ValueError(f"enforce_lattice_partition post-condition failed: {leftover}")
    return enforced


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_pass1(
    file_rows: List[Tuple[str, str, int, Lattice]], sync_hints: Dict[str, str] | None = None,
) -> ic.Completion:
    """The ingest's first model call: everything, all clips at once. Makes
    exactly one ``llm.client.complete`` call over already-loaded lattices --
    no DB write here (the caller persists the result onto ``ingest_runs``).
    ``sync_hints``: see `build_pass1_blocks`."""
    if not file_rows:
        raise ValueError("run_pass1: no files to ingest")
    blocks = build_pass1_blocks(file_rows, sync_hints)
    lattices = {fid: lat for fid, _name, _dur, lat in file_rows}
    def _checks(output: Pass1Output) -> str | None:
        return (_word_spans_in_range(output, lattices)
                or _no_overlapping_speech_cuts(output))

    return ic.complete("pass1", _SYSTEM, blocks, Pass1Output, max_tokens=24000,
                       extra_check=_checks)


def _pg_conn():
    import psycopg
    from app.config import get_settings
    return psycopg.connect(get_settings().database_url, autocommit=True)


def load_project_file_rows(project_id: str) -> List[Tuple[str, str, int, Lattice]]:
    """Load (file_id, name, duration_ms, lattice) for every file in a
    project, skipping any file with no duration yet (still uploading) or no
    lattice (L1 not ready)."""
    with _pg_conn() as conn:
        row = conn.execute(
            "select source_file_ids from projects where id = %s", (project_id,)
        ).fetchone()
        file_ids = list(row[0] or []) if row else []
        if not file_ids:
            return []
        rows = conn.execute(
            "select id::text, name, duration_seconds from files where id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()

    out: List[Tuple[str, str, int, Lattice]] = []
    for fid, name, dur_s in rows:
        if not dur_s:
            continue
        lattice = lt.load_lattice(fid)
        if lattice is None:
            continue
        out.append((fid, name or fid, int(float(dur_s) * 1000), lattice))
    return out


def run_pass1_for_project(project_id: str) -> ic.Completion:
    """Convenience: load every ready file in a project and run pass 1.
    Raises ``ValueError`` if the project has no ingest-ready files yet."""
    file_rows = load_project_file_rows(project_id)
    if not file_rows:
        raise ValueError(f"project {project_id} has no ingest-ready files yet")
    logger.info("pass1: project %s -> %d clips", project_id, len(file_rows))
    return run_pass1(file_rows)


# --------------------------------------------------------------------------
# Rendering pass 1's own output back into text, for pass 2's cached prefix.
# Refs here (``speech_cut[i]`` / ``video_group[i]`` / ``take[group_id]``)
# match ``image_plan.py``'s ``PlannedFrame.ref`` exactly -- pass 2's numbered
# images ("IMG 12 = clip 3, 41.2s, speech_cut 7") point back at these lines.
# --------------------------------------------------------------------------

def render_pass1_output(pass1: Pass1Output, keep_refs: set | None = None) -> str:
    """Render the pass-1 result for a pass-2 call. When ``keep_refs`` is given
    (a set of "speech_cut[i]" / "video_group[gi]" strings), the render is
    SCOPED to just those cuts -- ORIGINAL indices are preserved so a ref never
    renumbers -- and junk suspects and clip summaries are trimmed to the same
    footprint. Scoping means a batch only ever sees (and so only ever emits)
    its own refs. ``keep_refs=None`` renders everything.

    Take candidates are deliberately NOT rendered here (pass2_merge.plan.md):
    take-grouping moved fully to deterministic code (``pass2.apply_take_groups``,
    fed by pass 1's own ``take_candidates`` post-hoc) -- the model never
    resolves takes anymore, and batches no longer co-locate a take's members,
    so the old TAKE CANDIDATES section would render near-empty in most
    batches with no instruction telling the model what to do with it."""
    def keep_speech(i: int) -> bool:
        return keep_refs is None or f"speech_cut[{i}]" in keep_refs

    def keep_video(gi: int) -> bool:
        return keep_refs is None or f"video_group[{gi}]" in keep_refs

    kept_files: set = set()
    lines = ["=== PASS 1 RESULT (your own prior output -- final unless revised below) ==="]

    lines.append("SPEECH CUTS:")
    for i, sc in enumerate(pass1.speech_cuts):
        if not keep_speech(i):
            continue
        kept_files.add(sc.file_id)
        speakers = ",".join(sc.speaker_ids) or "?"
        lines.append(f"  speech_cut[{i}]: file={sc.file_id} words[{sc.word_span[0]}-{sc.word_span[1]}] "
                     f"label=\"{sc.label}\" speakers={speakers}")

    lines.append("VIDEO TENTATIVE GROUPS:")
    for gi, vg in enumerate(pass1.video_tentative_groups):
        if not keep_video(gi):
            continue
        kept_files.add(vg.file_id)
        atoms = ",".join(str(a) for a in vg.atom_ids)
        lines.append(f"  video_group[{gi}]: file={vg.file_id} atoms=[{atoms}]")

    lines.append("JUNK SUSPECTS:")
    for js in pass1.junk_suspects:
        if keep_refs is not None and js.file_id not in kept_files:
            continue
        where = f"words[{js.word_span[0]}-{js.word_span[1]}]" if js.word_span else f"atoms={js.atom_ids}"
        lines.append(f"  file={js.file_id} {where} reason=\"{js.reason}\"")

    lines.append(f"PROJECT SUMMARY: {pass1.project_summary}")
    lines.append("CLIP SUMMARIES:")
    for cs in pass1.clip_summaries:
        if keep_refs is not None and cs.file_id not in kept_files:
            continue
        lines.append(f"  file={cs.file_id}: {cs.summary}")

    return "\n".join(lines)
