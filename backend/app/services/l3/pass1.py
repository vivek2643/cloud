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
from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from app.services.l3 import lattice as lt
from app.services.l3.lattice import Lattice
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
    "You are ingesting raw footage for a video editor. For every clip in "
    "this project you are given its full word-timed, diarized transcript, a "
    "table of VIDEO ATOMS (deterministic boundaries: shot cuts, disturbances, "
    "transitions, and action beats; each atom is also labelled with its "
    "dominant camera behavior -- hold / pan / handheld -- which is a LABEL, "
    "not a boundary), and HINTS about where long pauses and speaker changes "
    "fall in the transcript.\n\n"
    "Your job, across ALL clips at once:\n"
    "1. GROUP the words into speech_cuts -- each one coherent spoken beat (a "
    "sentence, a thought, an exchange). Decide the boundaries yourself using "
    "word indices; the hints inform you, they never bind you. A speaker "
    "change almost always starts a new cut. HARD CONSTRAINT: a speech_cut "
    "must lie within ONE continuous stretch of speech -- it must never span "
    "across a 'long pause' hint, because the video atoms own that silent "
    "territory (grouping across one makes the cut's span swallow video "
    "atoms, which is structurally invalid). If the same thought resumes "
    "after a long pause, emit separate speech_cuts on each side of it.\n"
    "2. Find TAKE CANDIDATES: near-identical spoken lines recurring across "
    "clips (a re-take) or within one clip. HARD CONSTRAINT: every take "
    "member must be exactly one whole speech_cut (identical file_id and "
    "word_span) -- a take boundary is always a hard cut boundary, so if a "
    "repeated line currently sits inside a bigger group, split the group so "
    "the take stands alone as its own speech_cut.\n"
    "3. SELECT and GROUP video atoms (by atom_id) into the shots worth keeping. "
    "This is a SELECTION, not a partition -- you need NOT place every atom in a "
    "group. Group atoms that read as one continuous, USABLE moment (a real held "
    "composition, a clean b-roll shot). LEAVE OUT pure connective tissue -- "
    "pre-roll before the first word, brief transitional holds between beats, "
    "dead-air gaps: those atoms are simply DROPPED from the edit, not tiled "
    "(coverage is not required). An atom tagged ACTION is a subject-motion "
    "PAYOFF (a hit, catch, jump, swing); it already carries its own wind-up / "
    "follow-through, so keep each ACTION atom as its OWN video_tentative_group, "
    "payoff intact, and never flag it junk.\n"
    "4. Flag JUNK. You have the WHOLE project's transcript and full context, "
    "so you can tell what is part of the delivered message from what is not. "
    "Apply TWO different bars:\n"
    "   (a) PRODUCTION NOISE -- words that are NOT part of the intended "
    "message: spoken direction/cues used to start, stop, reset or count into a "
    "take, acknowledgements exchanged with an off-camera director/operator, and "
    "clear self-corrections, restarts or abandoned false starts where the "
    "speaker resets and re-delivers the line. DIG for these AGGRESSIVELY -- "
    "this text-only pass is the ONE place with the context to catch them, so "
    "err toward removing them. Pay special attention to the very START and END "
    "of each clip and to short stray utterances sitting next to an action beat, "
    "which is where these cues almost always live. When a beat opens or closes "
    "with such noise, SPLIT it: emit the noise as its OWN short speech_cut AND "
    "list that exact span in junk_suspects with a short reason; the real beat "
    "is a separate speech_cut. These are clearly unusable -- flag them.\n"
    "   (b) BORDERLINE material -- a slightly awkward but genuinely delivered "
    "sentence, ambient dead air, plain footage: stay CONSERVATIVE. If in doubt, "
    "leave it UNFLAGGED and visible. When something IS recognizable junk, FLAG "
    "it (so it stays recoverable) rather than silently dropping it; never flag "
    "an ACTION atom as junk.\n"
    "5. Write a one-paragraph project_summary and a one-line summary per "
    "clip.\n\n"
    "Boundaries you emit are WORD INDICES (speech_cuts, take word_spans) or "
    "ATOM IDS (video_tentative_groups) -- NEVER a millisecond timestamp. "
    "Exact timing is derived from those, by code, afterward."
)


def _render_clip_block(file_id: str, name: str, duration_ms: int, lattice: Lattice) -> str:
    lines = [f"=== CLIP {file_id} \"{name}\" ({duration_ms / 1000:.1f}s) ==="]
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


def build_pass1_blocks(file_rows: List[Tuple[str, str, int, Lattice]]) -> List[Dict[str, Any]]:
    """One text block per clip. ``file_rows`` = [(file_id, name, duration_ms,
    lattice)]. All clips ride in ONE user turn so pass 1 reasons across the
    whole project at once (cross-clip takes need this)."""
    return [text_block(_render_clip_block(fid, name, dur, lat)) for fid, name, dur, lat in file_rows]


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
    """A speech cut whose word range fully contains any atom's span would
    make the cut's resolved ms span swallow that atom -- atoms and speech
    cuts must partition the timeline. ``enforce_lattice_partition`` splits
    such cuts deterministically; this check is its post-condition (and is
    asserted there), kept as a separate function so tests can probe it
    directly."""
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
        for atom in lattice.atoms:
            if atom.start_ms >= span_s and atom.end_ms <= span_e:
                return (f"speech_cut[{i}] \"{sc.label}\" words[{a}-{b}] spans a long "
                        f"silent gap that contains video atom {atom.atom_id} "
                        f"[{atom.start_ms}-{atom.end_ms}]ms in {sc.file_id} -- a speech "
                        f"cut must never cross a 'long pause' hint; split it into "
                        f"separate cuts on each side of the pause")
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


def _isolate_action_atoms(lattice: Lattice, run: List[int]) -> List[List[int]]:
    """A contiguous atom run split so every ACTION atom (a carved subject-motion
    payoff, section C) stands ALONE as its own group -- a payoff is a
    first-class candidate cut and must never be swallowed by an adjacent
    hold/pan group, whatever pass 1 grouped. The model owns which calm atoms
    belong together; the lattice owns that an action is its own beat."""
    by_id = {a.atom_id: a for a in lattice.atoms}
    out: List[List[int]] = []
    cur: List[int] = []
    for aid in run:
        atom = by_id.get(aid)
        if atom is not None and atom.is_action:
            if cur:
                out.append(cur)
                cur = []
            out.append([aid])
        else:
            cur.append(aid)
    if cur:
        out.append(cur)
    return out


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


def enforce_lattice_partition(output: Pass1Output, lattices: Dict[str, Lattice]) -> Pass1Output:
    """Deterministically repair pass 1's grouping against the lattice:

      1. Split any speech_cut at gaps that contain a video atom (that
         territory belongs to the atoms; see block comment above).
      2. Remap every take member onto the (possibly split) speech_cut that
         contains most of its words -- so member == whole cut always holds
         downstream (pass 2a source_refs, image_plan joins). Groups left
         with fewer than two distinct members are dropped.
      3. Split any video_tentative_group at time discontinuities (see
         ``_contiguous_atom_runs``) -- a group must be one continuous
         stretch, never a bridge across speech.

    Post-condition asserted: no speech cut swallows an atom."""
    new_cuts: List[SpeechCut] = []
    for sc in output.speech_cuts:
        lattice = lattices.get(sc.file_id)
        if lattice is None or not lattice.words:
            new_cuts.append(sc)
            continue
        pieces = _span_pieces(lattice.words, lattice.atoms, tuple(sc.word_span))
        if len(pieces) == 1:
            new_cuts.append(sc)
        else:
            logger.info("pass1 enforce: splitting speech_cut %r words[%d-%d] into %d pieces "
                        "at atom-owned gaps", sc.label, sc.word_span[0], sc.word_span[1], len(pieces))
            for j, (pa, pb) in enumerate(pieces, start=1):
                new_cuts.append(SpeechCut(file_id=sc.file_id, word_span=(pa, pb),
                                          label=f"{sc.label} ({j}/{len(pieces)})",
                                          speaker_ids=list(sc.speaker_ids)))

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
        seen: set = set()
        members: List[TakeMember] = []
        for m in tc.members:
            mapped = _dominant_cut_span(m.file_id, tuple(m.word_span))
            if mapped is None or (m.file_id, mapped) in seen:
                continue
            seen.add((m.file_id, mapped))
            members.append(TakeMember(file_id=m.file_id, word_span=mapped))
        if len(members) >= 2:
            new_takes.append(TakeCandidate(group_id=tc.group_id, members=members))
        else:
            logger.info("pass1 enforce: dropping take group %r (%d distinct member(s) "
                        "after remap)", tc.group_id, len(members))

    new_groups: List[VideoTentativeGroup] = []
    for vg in output.video_tentative_groups:
        lattice = lattices.get(vg.file_id)
        if lattice is None:
            new_groups.append(vg)
            continue
        runs = _contiguous_atom_runs(lattice, vg.atom_ids)
        pieces = [p for run in runs for p in _isolate_action_atoms(lattice, run)]
        if len(pieces) == 1 and pieces[0] == list(vg.atom_ids):
            new_groups.append(vg)
        else:
            logger.info("pass1 enforce: splitting video group atoms=%s into %d "
                        "piece(s) (contiguity + action isolation)", vg.atom_ids, len(pieces))
            for run in pieces:
                new_groups.append(VideoTentativeGroup(file_id=vg.file_id, atom_ids=run))

    # boundaries-v2: grouping is now a SELECTION -- the model may (and should)
    # drop connective tissue by leaving it ungrouped. But an ACTION atom is a
    # payoff that must ALWAYS surface, so we don't leave that to the model's
    # whim: any is_action atom not already in some group is re-added as its own
    # group. Deterministic guarantee, independent of what pass 1 chose to keep.
    grouped_ids: Dict[str, set] = {}
    for vg in new_groups:
        grouped_ids.setdefault(vg.file_id, set()).update(vg.atom_ids)
    for file_id, lattice in lattices.items():
        have = grouped_ids.get(file_id, set())
        for atom in lattice.atoms:
            if atom.is_action and atom.atom_id not in have:
                logger.info("pass1 enforce: re-adding dropped ACTION atom %d in %s as its own group",
                            atom.atom_id, file_id)
                new_groups.append(VideoTentativeGroup(file_id=file_id, atom_ids=[atom.atom_id]))

    enforced = output.model_copy(update={"speech_cuts": new_cuts, "take_candidates": new_takes,
                                         "video_tentative_groups": new_groups})
    leftover = _no_speech_cut_swallows_atoms(enforced, lattices)
    if leftover:
        raise ValueError(f"enforce_lattice_partition post-condition failed: {leftover}")
    return enforced


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_pass1(file_rows: List[Tuple[str, str, int, Lattice]]) -> ic.Completion:
    """The ingest's first model call: everything, all clips at once. Makes
    exactly one ``llm.client.complete`` call over already-loaded lattices --
    no DB write here (the caller persists the result onto ``ingest_runs``)."""
    if not file_rows:
        raise ValueError("run_pass1: no files to ingest")
    blocks = build_pass1_blocks(file_rows)
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

def render_pass1_output(pass1: Pass1Output) -> str:
    lines = ["=== PASS 1 RESULT (your own prior output -- final unless revised below) ==="]

    lines.append("SPEECH CUTS:")
    for i, sc in enumerate(pass1.speech_cuts):
        speakers = ",".join(sc.speaker_ids) or "?"
        lines.append(f"  speech_cut[{i}]: file={sc.file_id} words[{sc.word_span[0]}-{sc.word_span[1]}] "
                     f"label=\"{sc.label}\" speakers={speakers}")

    lines.append("TAKE CANDIDATES:")
    for tc in pass1.take_candidates:
        members = "; ".join(f"{m.file_id} words[{m.word_span[0]}-{m.word_span[1]}]" for m in tc.members)
        lines.append(f"  take[{tc.group_id}]: {members}")

    lines.append("VIDEO TENTATIVE GROUPS:")
    for gi, vg in enumerate(pass1.video_tentative_groups):
        atoms = ",".join(str(a) for a in vg.atom_ids)
        lines.append(f"  video_group[{gi}]: file={vg.file_id} atoms=[{atoms}]")

    lines.append("JUNK SUSPECTS:")
    for js in pass1.junk_suspects:
        where = f"words[{js.word_span[0]}-{js.word_span[1]}]" if js.word_span else f"atoms={js.atom_ids}"
        lines.append(f"  file={js.file_id} {where} reason=\"{js.reason}\"")

    lines.append(f"PROJECT SUMMARY: {pass1.project_summary}")
    lines.append("CLIP SUMMARIES:")
    for cs in pass1.clip_summaries:
        lines.append(f"  file={cs.file_id}: {cs.summary}")

    return "\n".join(lines)
