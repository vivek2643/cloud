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
    "table of VIDEO ATOMS (deterministic boundaries: shot cuts, camera "
    "moves/settles, disturbances, transitions), and HINTS about where long "
    "pauses and speaker changes fall in the transcript.\n\n"
    "Your job, across ALL clips at once:\n"
    "1. GROUP the words into speech_cuts -- each one coherent spoken beat (a "
    "sentence, a thought, an exchange). Decide the boundaries yourself using "
    "word indices; the hints inform you, they never bind you. A speaker "
    "change almost always starts a new cut.\n"
    "2. Find TAKE CANDIDATES: near-identical spoken lines recurring across "
    "clips (a re-take) or within one clip.\n"
    "3. Tentatively GROUP video atoms (by atom_id) into visually-homogeneous "
    "runs that read as one continuous moment -- bounded by shot cuts and "
    "composition drift already, so a group is visually coherent by "
    "construction; you decide which adjacent atoms belong together.\n"
    "4. Flag JUNK SUSPECTS: speech or video spans that look like false "
    "starts, dead air, or unusable footage -- flag, never omit.\n"
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
# Orchestration
# --------------------------------------------------------------------------

def run_pass1(file_rows: List[Tuple[str, str, int, Lattice]]) -> ic.Completion:
    """The ingest's first model call: everything, all clips at once. Makes
    exactly one ``llm.client.complete`` call over already-loaded lattices --
    no DB write here (the caller persists the result onto ``ingest_runs``)."""
    if not file_rows:
        raise ValueError("run_pass1: no files to ingest")
    blocks = build_pass1_blocks(file_rows)
    return ic.complete("pass1", _SYSTEM, blocks, Pass1Output, max_tokens=24000)


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
