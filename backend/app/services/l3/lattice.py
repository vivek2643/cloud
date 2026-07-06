"""
Cuts v3 -- the LATTICE: the deterministic substrate the LLM ingest reasons
over. ``base_cuts.py`` evolves into this module (kept untouched itself until
cutover -- see cuts_v3.plan.md, section 2).

Two sides, two different shapes, because they need different things from the
model:

  * SPEECH has no atoms. The lattice IS the word list -- Whisper word timings
    are already millisecond-precise, so there is nothing to detect. Turns,
    long pauses and speaker changes are exported as PROMPT HINTS (informational
    text), never boundaries: pass 1 picks the actual ``word_span`` grouping
    itself. This is a deliberate reversal from ``base_cuts.py``, which turned
    speech turns into fixed cuts -- v3 wants the LLM's judgment there, not a
    deterministic turn-merge.
  * VIDEO gets ATOMS: the same robust boundary sources ``base_cuts.py``
    trusts unconditionally (hard shot cut, camera move/settle, disturbance
    edges), PLUS ``transition_points`` (wipe/degenerate) as additional edges
    -- but ONLY over the NON-SPEECH remainder of the clip. A video atom
    overlapping speech never happens by construction: atoms are built inside
    the gaps between speech turns, not against the whole timeline (unchanged
    rule -- never cut under speech).

Atoms are deliberately fine (over-segmenting bias is safe -- pass 1/2 merge
them back into ``video_tentative_groups`` / final cuts; under-splitting, which
would hide a real boundary forever, is the only fatal error).

Pure core: ``build_atoms(...)`` / ``speech_hints(...)`` / the pure
``_snap_word_edge`` take already-loaded artifacts and do no DB call, so they
are trivially testable -- see ``scripts/test_lattice.py``. ``load_lattice``
and ``snap_word_edge`` are the DB-loading convenience wrappers for real callers.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.base_cuts import (
    R_CLIP,
    R_DISTURB,
    R_MOVE,
    R_SETTLE,
    R_SHOT,
    _camera_marks,
    _disturbance_marks,
    _shot_marks,
)
from app.services.l3.base_cuts_params import LONG_PAUSE_MS, SNAP_MS
from app.services.l3.diarize import Turn
from app.services.l3.lattice_params import (
    CAMERA_HOLD_MOTION_MAX,
    CAMERA_PAN_COHERENCE_MIN,
    CAMERA_PAN_STABILITY_MIN,
    MIN_ATOM_MS,
)

logger = logging.getLogger(__name__)

# cuts-v3 transition-point reasons (not in base_cuts.py -- that module predates
# transition_points). Ranked alongside base_cuts' reasons for the same
# "several signals, one instant -> report the strongest" collapsing.
R_WIPE = "wipe"
R_DEGENERATE = "degenerate"
R_SPEECH_EDGE = "speech_edge"   # this atom's own edge borders a speech turn

_REASON_RANK = {
    R_SHOT: 0, R_DISTURB: 1, R_WIPE: 1, R_DEGENERATE: 1,
    R_MOVE: 2, R_SETTLE: 2, R_SPEECH_EDGE: 3, R_CLIP: 4,
}


# --------------------------------------------------------------------------
# The Atom shape
# --------------------------------------------------------------------------

@dataclass
class Atom:
    atom_id: int
    file_id: str
    start_ms: int
    end_ms: int
    state_in: str          # why this atom's LEFT edge exists
    state_out: str         # why this atom's RIGHT edge exists
    action_energy: float   # mean subject-motion energy over the span, 0..1
    camera_desc: str       # "hold" | "pan" | "handheld" -- coarse, for the prompt
    coherence: float       # mean camera_coherence over the span, 0..1
    anchor_ms: List[int] = field(default_factory=list)   # action_points inside this atom

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atom_id": self.atom_id, "file_id": self.file_id,
            "start_ms": self.start_ms, "end_ms": self.end_ms,
            "state_in": self.state_in, "state_out": self.state_out,
            "action_energy": self.action_energy, "camera_desc": self.camera_desc,
            "coherence": self.coherence, "anchor_ms": list(self.anchor_ms),
        }


@dataclass
class Lattice:
    """Everything one clip's ingest reasons over: the raw word list + prompt
    hints (speech side) and the video atoms (non-speech side)."""
    file_id: str
    duration_ms: int
    words: List[dict] = field(default_factory=list)
    turns: List[Turn] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    atoms: List[Atom] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id, "duration_ms": self.duration_ms,
            "words": self.words, "turns": [list(t) for t in self.turns],
            "hints": self.hints, "atoms": [a.to_dict() for a in self.atoms],
        }


# --------------------------------------------------------------------------
# Video side: atoms over the non-speech remainder
# --------------------------------------------------------------------------

def _transition_marks(motion: Optional[dict]) -> List[Tuple[int, str]]:
    """``transition_points`` (Phase B) as atom-edge marks -- a wipe or a
    degenerate span onset is exactly as trusted a boundary as a shot cut or a
    camera settle."""
    out: List[Tuple[int, str]] = []
    for p in ((motion or {}).get("transition_points") or []):
        kind = p.get("kind")
        ts = p.get("ts_ms")
        if ts is None or kind not in ("wipe", "degenerate"):
            continue
        out.append((int(ts), R_WIPE if kind == "wipe" else R_DEGENERATE))
    return out


def _subtract(span: Tuple[int, int], claimed: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """``span`` minus the union of ``claimed`` intervals -> free fragments, in
    time order (the non-speech remainder of the clip)."""
    free = [span]
    for cs, ce in claimed:
        nxt: List[Tuple[int, int]] = []
        for fs, fe in free:
            if ce <= fs or cs >= fe:
                nxt.append((fs, fe))
                continue
            if cs > fs:
                nxt.append((fs, cs))
            if ce < fe:
                nxt.append((ce, fe))
        free = nxt
        if not free:
            break
    return free


def _snap_collect(marks: List[Tuple[int, str]], lo: int, hi: int) -> Dict[int, str]:
    """{ts: reason} for marks strictly inside (lo, hi), snapping any two
    within SNAP_MS onto the earlier one and keeping the strongest reason."""
    at: Dict[int, str] = {}
    for ts, reason in sorted((t, r) for t, r in marks if lo < t < hi):
        near = next((k for k in at if abs(k - ts) <= SNAP_MS), None)
        key = near if near is not None else ts
        if key not in at or _REASON_RANK.get(reason, 99) < _REASON_RANK.get(at[key], 99):
            at[key] = reason
    return at


def _mean(xs: List[float], lo_i: int, hi_i: int) -> float:
    seg = xs[max(0, lo_i):max(lo_i + 1, hi_i)]
    return sum(seg) / len(seg) if seg else 0.0


def _camera_desc(motion: Optional[dict], s_ms: int, e_ms: int) -> str:
    m = motion or {}
    hop = int(m.get("hop_ms") or 0)
    if not hop:
        return "hold"
    lo_i, hi_i = s_ms // hop, e_ms // hop
    m_cam = _mean(m.get("camera_motion") or [], lo_i, hi_i)
    if m_cam < CAMERA_HOLD_MOTION_MAX:
        return "hold"
    m_coh = _mean(m.get("camera_coherence") or [], lo_i, hi_i)
    m_stab = _mean(m.get("camera_stability") or [], lo_i, hi_i)
    if m_coh >= CAMERA_PAN_COHERENCE_MIN and m_stab >= CAMERA_PAN_STABILITY_MIN:
        return "pan"
    return "handheld"


def _anchors_in(motion: Optional[dict], s_ms: int, e_ms: int) -> List[int]:
    pts = (motion or {}).get("action_points") or []
    return sorted(int(p["ts_ms"]) for p in pts if s_ms <= int(p.get("ts_ms", -1)) < e_ms)


def build_atoms(file_id: str, duration_ms: int, motion: Optional[dict],
                scene: Optional[dict], turns: List[Turn]) -> List[Atom]:
    """Pure core: video ATOMS over the non-speech remainder of
    [0, duration_ms]. Speech gets no atoms at all -- atoms are only ever
    built inside the gaps between speech turns, so a video atom overlapping
    speech cannot happen by construction (not a filter, a structural
    guarantee)."""
    if duration_ms <= 0:
        return []
    speech_spans = sorted((s, e) for (s, e, _spk) in turns if e > s)
    non_speech = _subtract((0, duration_ms), speech_spans)
    all_marks = (
        _camera_marks(motion) + _disturbance_marks(motion)
        + _shot_marks(scene) + _transition_marks(motion)
    )

    m = motion or {}
    hop = int(m.get("hop_ms") or 0)
    action = m.get("action_energy") or []
    coherence = m.get("camera_coherence") or []

    atoms: List[Atom] = []
    for seg_s, seg_e in non_speech:
        # NOT skipped when shorter than MIN_ATOM_MS: this whole segment is
        # everything between two speech turns (or the file's edge) with
        # nowhere else for its span to go -- dropping it would leave a real
        # coverage gap. MIN_ATOM_MS only governs merging INTERNAL boundary
        # marks below (over-splitting one long segment), never whether the
        # segment gets an atom at all. Confirmed via a real ingest run: a
        # 20ms trailing sliver was silently vanishing before this fix.
        at = _snap_collect(all_marks, seg_s, seg_e)
        bounds = sorted({seg_s, seg_e} | set(at))

        # Merge sub-MIN_ATOM_MS slivers forward so the segment's own coverage
        # stays total (over-split is safe; a too-short shard just isn't worth
        # its own atom -- its reason folds into the next real boundary).
        kept = [bounds[0]]
        for b in bounds[1:-1]:
            if b - kept[-1] >= MIN_ATOM_MS:
                kept.append(b)
        kept.append(bounds[-1])

        for a, b in zip(kept, kept[1:]):
            if b <= a:
                continue
            reason_in = at.get(a, R_CLIP if a == 0 else R_SPEECH_EDGE)
            reason_out = at.get(b, R_CLIP if b == duration_ms else R_SPEECH_EDGE)
            lo_i, hi_i = (a // hop, b // hop) if hop else (0, 0)
            atoms.append(Atom(
                atom_id=len(atoms), file_id=file_id, start_ms=a, end_ms=b,
                state_in=reason_in, state_out=reason_out,
                action_energy=round(_mean(action, lo_i, hi_i), 3) if hop else 0.0,
                camera_desc=_camera_desc(motion, a, b),
                coherence=round(_mean(coherence, lo_i, hi_i), 3) if hop else 0.0,
                anchor_ms=_anchors_in(motion, a, b),
            ))
    return atoms


def render_atom_table(atoms: List[Atom]) -> str:
    """The compact numbered text block for prompts -- motion enters as
    NUMBERS, pixels as stills (pass 2). One line per atom:
    ``ATOM 7 [12300-15800] move->settle act=0.70 cam=pan coh=0.90 anchors@13100``
    """
    lines = []
    for a in atoms:
        anchors = f" anchors@{','.join(str(x) for x in a.anchor_ms)}" if a.anchor_ms else ""
        lines.append(
            f"ATOM {a.atom_id} [{a.start_ms}-{a.end_ms}] {a.state_in}->{a.state_out} "
            f"act={a.action_energy:.2f} cam={a.camera_desc} coh={a.coherence:.2f}{anchors}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Speech side: the word list itself + prompt hints (never boundaries)
# --------------------------------------------------------------------------

def speech_hints(words: List[dict], turn_gap_ms: int = LONG_PAUSE_MS) -> List[str]:
    """Human-readable PROMPT HINTS for pass 1 -- where turns/speaker changes/
    long pauses fall in the word stream. Informational only: pass 1 decides
    the actual ``word_span`` grouping, these never constrain it."""
    hints: List[str] = []
    for i in range(1, len(words)):
        prev, cur = words[i - 1], words[i]
        prev_spk = prev.get("speaker") or "S?"
        cur_spk = cur.get("speaker") or "S?"
        gap = int(cur.get("start_ms", 0)) - int(prev.get("end_ms", 0))
        if cur_spk != prev_spk:
            hints.append(f"speaker change after word {i - 1} ({prev_spk} -> {cur_spk})")
        elif gap >= turn_gap_ms:
            hints.append(f"long pause after word {i - 1} ({gap / 1000:.1f}s)")
    return hints


def _snap_word_edge(words: List[dict], word_idx: int, silences: List[dict]) -> int:
    """The precise ms boundary immediately BEFORE ``words[word_idx]`` --
    snapped into the inter-word silence so a cut edge never lands mid-word.
    ``word_idx`` may equal ``len(words)`` (the edge after the last word)."""
    if not words:
        return 0
    if word_idx <= 0:
        return int(words[0].get("start_ms", 0))
    if word_idx >= len(words):
        return int(words[-1].get("end_ms", 0))
    prev_end = int(words[word_idx - 1].get("end_ms", 0))
    cur_start = int(words[word_idx].get("start_ms", 0))
    if cur_start <= prev_end:
        return cur_start   # words touch/overlap -- no gap to snap into
    for s in silences:
        s0, s1 = int(s.get("start_ms", 0)), int(s.get("end_ms", 0))
        if s0 < cur_start and s1 > prev_end:
            lo, hi = max(s0, prev_end), min(s1, cur_start)
            if hi > lo:
                return (lo + hi) // 2
    return (prev_end + cur_start) // 2


# --------------------------------------------------------------------------
# DB loaders + convenience wrappers (not pure -- real callers only)
# --------------------------------------------------------------------------

def _pg_conn():
    import psycopg
    from app.config import get_settings
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _load_words(file_id: str) -> List[dict]:
    """Flat, filler-free, time-ordered words for one file. A small, separate
    flattening from ``diarize.load_turns`` (which discards the flat list and
    is shared by other L2/L3 consumers -- not ours to change)."""
    with _pg_conn() as conn:
        row = conn.execute(
            "select segments from transcripts where file_id = %s", (file_id,)
        ).fetchone()
    if not row or not row[0]:
        return []
    segments = row[0] if isinstance(row[0], list) else json.loads(row[0])
    words = [w for seg in segments for w in (seg.get("words") or []) if not w.get("is_filler")]
    words.sort(key=lambda w: w.get("start_ms", 0))
    return words


def _load_silences(file_id: str) -> List[dict]:
    with _pg_conn() as conn:
        row = conn.execute(
            "select silence_intervals from audio_features where file_id = %s", (file_id,)
        ).fetchone()
    return (row[0] or []) if row else []


def snap_word_edge(file_id: str, word_idx: int) -> int:
    """The precise ms boundary immediately before word ``word_idx`` in
    ``file_id`` -- snapped into inter-word silence. See ``_snap_word_edge``."""
    words = _load_words(file_id)
    silences = _load_silences(file_id)
    return _snap_word_edge(words, word_idx, silences)


def load_lattice(file_id: str) -> Optional[Lattice]:
    """Load one file's full lattice (words + hints + video atoms). None when
    the file has no duration yet."""
    from app.services.l3.diarize import load_turns

    with _pg_conn() as conn:
        row = conn.execute(
            "select duration_seconds from files where id = %s", (file_id,)
        ).fetchone()
        if not row or not row[0]:
            return None
        duration_ms = int(float(row[0]) * 1000)

        m = conn.execute(
            """select hop_ms, camera_stability, camera_coherence, camera_motion,
                      blur, camera_cut_cost, action_energy, action_points,
                      transition_points
                 from motion_dynamics where file_id = %s""",
            (file_id,),
        ).fetchone()
        motion = None
        if m:
            motion = {
                "hop_ms": m[0], "camera_stability": m[1] or [],
                "camera_coherence": m[2] or [], "camera_motion": m[3] or [],
                "blur": m[4] or [], "camera_cut_cost": m[5] or [],
                "action_energy": m[6] or [], "action_points": m[7] or [],
                "transition_points": m[8] or [],
            }

        s = conn.execute(
            "select shot_points from scene_cuts where file_id = %s", (file_id,)
        ).fetchone()
        scene = {"shot_points": s[0] or []} if s else None

    words = _load_words(file_id)
    _text, _speakers, turns = load_turns(file_id, turn_gap_ms=LONG_PAUSE_MS)
    atoms = build_atoms(file_id, duration_ms, motion, scene, turns)
    hints = speech_hints(words)
    return Lattice(file_id=file_id, duration_ms=duration_ms, words=words,
                   turns=turns, hints=hints, atoms=atoms)
