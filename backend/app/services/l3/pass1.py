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
    "  - speech_cuts: coherent spoken beats, as word-index ranges.\n"
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
    """Deterministically repair pass 1's grouping against the lattice, and
    guarantee TOTAL COVERAGE -- the load-bearing rule of deterministic-keep
    (cuts_v3_deterministic_keep.plan.md): the model chooses MEANING (grouping,
    takes, what is junk), but it can never silently drop a word or an atom.
    Anything it leaves out of every group is surfaced as a recovered candidate;
    the ONLY way something disappears is an explicit, recoverable junk label.

      1. Split any speech_cut at gaps that contain a video atom (that
         territory belongs to the atoms; see block comment above).
      2. COVERAGE FILL (speech): every transcript word the model left out of
         all speech_cuts is re-added as a recovered cut (split at atom-owned
         gaps).
      3. Remap every take member onto the (possibly split) speech_cut that
         contains most of its words -- so member == whole cut always holds
         downstream (pass 2a source_refs, image_plan joins). Groups left
         with fewer than two distinct members are dropped.
      4. Split any video_tentative_group at time discontinuities (see
         ``_contiguous_atom_runs``) -- a group must be one continuous
         stretch, never a bridge across speech.
      5. COVERAGE FILL (video): every atom the model left out of all groups
         is re-added, contiguous ungrouped atoms folded into one group.

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
        # Contiguity split only (structural: a group must be one time-continuous
        # run, else its resolved span brackets a speech beat -- an unfixable
        # cross-kind overlap). NOT action isolation: grouping/merging is the
        # model's job now (signal-judge), so a swing and its follow-through can
        # ride in one group if the model says so.
        runs = _contiguous_atom_runs(lattice, vg.atom_ids)
        if len(runs) == 1 and runs[0] == list(vg.atom_ids):
            new_groups.append(vg)
        else:
            logger.info("pass1 enforce: splitting video group atoms=%s into %d "
                        "contiguous run(s)", vg.atom_ids, len(runs))
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
        have = grouped_ids.get(file_id, set())
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
