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
  * VIDEO gets ATOMS: carved ONLY at boundaries we trust as genuine scene
    changes -- hard shot cut, disturbance edges, ``transition_points``
    (wipe/degenerate) -- over the NON-SPEECH remainder of the clip. Camera
    move/settle are DELIBERATELY NOT boundaries here (editorial pass, see
    cuts_v3_editorial.plan.md section B): a pan is one coherent atom labeled
    "pan", not [hold][move][settle] shredded into slivers -- camera motion is
    a LABEL and a selection handle, not a boundary that shreds the timeline.
    A video atom overlapping speech never happens by construction: atoms are
    built inside the gaps between speech turns, not against the whole timeline
    (unchanged rule -- never cut under speech).

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
    R_SHOT,
    _shot_marks,
)
from app.services.l3.base_cuts_params import LONG_PAUSE_MS, SNAP_MS
from app.services.l3.diarize import Turn
from app.services.l3.lattice_params import (
    ACTION_ANCHOR_MERGE_MS,
    ACTION_PAD_FRAC,
    CAMERA_HOLD_MOTION_MAX,
    CAMERA_MOVE_FRAC_MIN,
    CAMERA_PAN_COHERENCE_MIN,
    CAMERA_PAN_STABILITY_MIN,
    MIN_ATOM_MS,
    PERCEPTUAL_FLOOR_MS,
)
from app.services.l3.video_segments import MOVE, _confirm_hysteresis, _hop_states

logger = logging.getLogger(__name__)

# cuts-v3 transition-point reasons (not in base_cuts.py -- that module predates
# transition_points). Ranked alongside base_cuts' reasons for the same
# "several signals, one instant -> report the strongest" collapsing.
R_WIPE = "wipe"
R_DEGENERATE = "degenerate"
R_ACTION = "action"             # this edge bounds a subject-motion payoff span
R_SPEECH_EDGE = "speech_edge"   # this atom's own edge borders a speech turn

# Only GROUNDED events are atom boundaries now (boundaries-v2): hard shot cut,
# wipe/degenerate transition, ACTION window edge, and the speech/clip edges
# bounding the remainder. Camera move/settle (section B) and disturbance jitter
# (boundaries-v2) are LABELS, not edges. R_DISTURB is retained in the rank only
# so any legacy/base_cuts caller that still passes one collapses sanely; the
# lattice itself no longer emits disturbance marks.
_REASON_RANK = {
    R_SHOT: 0, R_DISTURB: 1, R_WIPE: 1, R_DEGENERATE: 1,
    R_ACTION: 2, R_SPEECH_EDGE: 3, R_CLIP: 4,
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
    is_action: bool = False   # this atom is a carved subject-motion payoff (section C)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atom_id": self.atom_id, "file_id": self.file_id,
            "start_ms": self.start_ms, "end_ms": self.end_ms,
            "state_in": self.state_in, "state_out": self.state_out,
            "action_energy": self.action_energy, "camera_desc": self.camera_desc,
            "coherence": self.coherence, "anchor_ms": list(self.anchor_ms),
            "is_action": self.is_action,
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


def _action_marks(motion: Optional[dict]) -> List[Tuple[int, str]]:
    """Action-span edges (section C). ``action_points`` are discrete L1-detected
    impacts; a cluster of them (anchors within ACTION_ANCHOR_MERGE_MS) is ONE
    motion payoff, and its padded edges are atom boundaries so the payoff is
    carved into its own atom rather than absorbed into a neighboring hold/pan.
    Only anchors landing in the NON-speech remainder matter -- ``_snap_collect``
    already drops marks outside a segment, so an anchor fired while the subject
    was talking never cuts under speech (plan rule)."""
    pts = sorted(int(p["ts_ms"]) for p in (motion or {}).get("action_points") or []
                 if isinstance(p, dict) and "ts_ms" in p)
    if not pts:
        return []
    clusters: List[List[int]] = [[pts[0]]]
    for t in pts[1:]:
        if t - clusters[-1][-1] <= ACTION_ANCHOR_MERGE_MS:
            clusters[-1].append(t)
        else:
            clusters.append([t])
    out: List[Tuple[int, str]] = []
    for members in clusters:
        cs, ce = members[0], members[-1]
        pad = _action_pad_ms(members)
        out.append((max(0, cs - pad), R_ACTION))
        out.append((ce + pad, R_ACTION))
    return out


def _action_pad_ms(anchors: List[int]) -> int:
    """Anchor-relative wind-up / follow-through pad (boundaries-v2). Scales with
    the cluster's OWN rhythm -- median gap between its impacts -- so a fast
    flurry stays tight and a lone slow swing breathes, without a per-clip
    magic-ms. Floored at the perceptual floor; a single-anchor cluster (no gap)
    takes the floor directly."""
    gaps = [b - a for a, b in zip(anchors, anchors[1:]) if b > a]
    if not gaps:
        return PERCEPTUAL_FLOOR_MS
    gaps.sort()
    median_gap = gaps[len(gaps) // 2]
    return max(PERCEPTUAL_FLOOR_MS, int(ACTION_PAD_FRAC * median_gap))


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
    """Classify an atom's DOMINANT camera behavior -- "hold" | "pan" |
    "handheld". Since a whole pan is now one atom (moves are no longer
    boundaries), the mean camera_motion over the span is diluted by any held
    lead-in/wind-down, so classification runs off the confirmed hold/move
    STATE MACHINE instead: what fraction of the span is actually moving. A
    move that stays coherent and stable is a deliberate "pan"; a jittery one
    is "handheld"; a mostly-still span is "hold"."""
    m = motion or {}
    hop = int(m.get("hop_ms") or 0)
    if not hop:
        return "hold"
    lo_i, hi_i = s_ms // hop, max(s_ms // hop + 1, e_ms // hop)
    states = _confirm_hysteresis(_hop_states(m))[lo_i:hi_i]
    if not states:
        # No state samples (span past the motion track) -- fall back to the
        # mean-motion test so a short trailing atom still gets a sane label.
        return "hold" if _mean(m.get("camera_motion") or [], lo_i, hi_i) < CAMERA_HOLD_MOTION_MAX else "handheld"
    if states.count(MOVE) / len(states) < CAMERA_MOVE_FRAC_MIN:
        return "hold"
    m_coh = _mean(m.get("camera_coherence") or [], lo_i, hi_i)
    m_stab = _mean(m.get("camera_stability") or [], lo_i, hi_i)
    if m_coh >= CAMERA_PAN_COHERENCE_MIN and m_stab >= CAMERA_PAN_STABILITY_MIN:
        return "pan"
    return "handheld"


def _anchors_in(motion: Optional[dict], s_ms: int, e_ms: int) -> List[int]:
    pts = (motion or {}).get("action_points") or []
    return sorted(int(p["ts_ms"]) for p in pts if s_ms <= int(p.get("ts_ms", -1)) < e_ms)


def _word_claim_spans(words: List[dict], merge_gap_ms: int = LONG_PAUSE_MS) -> List[Tuple[int, int]]:
    """Time spans claimed by the words themselves, adjacent words merged
    when their gap is below ``merge_gap_ms`` (a sub-long-pause gap belongs
    to the speech around it, and per-word confetti spans would shred the
    remainder into useless atom slivers). Needed because diarization TURNS
    and Whisper word timings are two different oracles: observed against
    real data (a long stutter section), a run of transcript words fell
    outside every diarized turn -- atoms then got built OVER those words,
    and any speech cut using them overlapped atom territory by
    construction. Words are ground truth for where speech is; turns alone
    are not sufficient."""
    spans: List[Tuple[int, int]] = []
    for w in sorted(words, key=lambda w: int(w.get("start_ms", 0))):
        s, e = int(w.get("start_ms", 0)), int(w.get("end_ms", 0))
        if e <= s:
            continue
        if spans and s - spans[-1][1] < merge_gap_ms:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
    return spans


def build_atoms(file_id: str, duration_ms: int, motion: Optional[dict],
                scene: Optional[dict], turns: List[Turn],
                words: Optional[List[dict]] = None) -> List[Atom]:
    """Pure core: video ATOMS over the non-speech remainder of
    [0, duration_ms]. Speech gets no atoms at all -- atoms are only ever
    built inside the gaps between speech turns AND word-claimed spans (see
    ``_word_claim_spans`` for why turns alone aren't enough), so a video
    atom overlapping speech cannot happen by construction (not a filter, a
    structural guarantee)."""
    if duration_ms <= 0:
        return []
    claimed = [(s, e) for (s, e, _spk) in turns if e > s]
    claimed += _word_claim_spans(words or [])
    speech_spans = sorted(claimed)
    non_speech = _subtract((0, duration_ms), speech_spans)
    # Boundaries-v2 (cuts_v3_boundaries_v2.plan.md): only GROUNDED events carve
    # atoms -- ones that correspond to something real in the footage. Camera
    # move/settle stopped being edges in section B; disturbance (handheld
    # jitter) stops being an edge here -- it fired boundaries where nothing
    # happened (a 60ms sliver on a shaky pan). Jitter is still captured, as a
    # LABEL (_camera_desc reads "handheld"), never a cut. What remains: hard
    # shot cut, wipe/degenerate transition, action-anchor windows -- plus the
    # speech/clip edges bounding the non-speech remainder itself.
    all_marks = _shot_marks(scene) + _transition_marks(motion) + _action_marks(motion)

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
            anchors = _anchors_in(motion, a, b)
            atoms.append(Atom(
                atom_id=len(atoms), file_id=file_id, start_ms=a, end_ms=b,
                state_in=reason_in, state_out=reason_out,
                action_energy=round(_mean(action, lo_i, hi_i), 3) if hop else 0.0,
                camera_desc=_camera_desc(motion, a, b),
                coherence=round(_mean(coherence, lo_i, hi_i), 3) if hop else 0.0,
                anchor_ms=anchors,
                # Carved by the action marks above -> any atom holding an anchor
                # IS a motion payoff (a calm neighbor never carries one).
                is_action=bool(anchors),
            ))
    return atoms


def render_atom_table(atoms: List[Atom]) -> str:
    """The compact numbered text block for prompts -- motion enters as
    NUMBERS, pixels as stills (pass 2). One line per atom:
    ``ATOM 7 [12300-15800] shot_cut->speech_edge act=0.70 cam=pan coh=0.90 anchors@13100``
    """
    lines = []
    for a in atoms:
        anchors = f" anchors@{','.join(str(x) for x in a.anchor_ms)}" if a.anchor_ms else ""
        tag = " ACTION" if a.is_action else ""
        lines.append(
            f"ATOM {a.atom_id} [{a.start_ms}-{a.end_ms}] {a.state_in}->{a.state_out} "
            f"act={a.action_energy:.2f} cam={a.camera_desc} coh={a.coherence:.2f}{anchors}{tag}"
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


def resolve_speech_span_ms(
    words: List[dict], atoms: List[Atom], word_span: Tuple[int, int], silences: List[dict],
) -> Tuple[int, int]:
    """A speech cut's (start_ms, end_ms), snapped into inter-word silence
    like ``_snap_word_edge`` but then CLAMPED so the cushioning padding
    never intrudes into a neighboring atom's already-claimed territory.

    Needed because an inter-word gap and an atom-filled gap aren't always
    the same boundary: atoms are carved from the non-speech remainder using
    diarization TURNS, while a speech cut's own word_span/silence snap can
    independently reach partway into that same gap for a natural cut point.
    Observed against real data: a 2.8s pause between two words, almost
    entirely consumed by 3 video atoms, still let the silence-midpoint snap
    land ~1000ms inside the atoms' own span -- a real, reproducible
    coverage overlap, not model noise. Atoms and speech together must
    partition the timeline; this is the one adjustment enforcing that at
    the boundary the two disagree about."""
    s = _snap_word_edge(words, word_span[0], silences)
    e = _snap_word_edge(words, word_span[1] + 1, silences)
    if not words or not atoms:
        return s, e
    idx0, idx1 = word_span[0], word_span[1]
    first_word_start = int(words[idx0].get("start_ms", s)) if 0 <= idx0 < len(words) else s
    last_word_end = int(words[idx1].get("end_ms", e)) if 0 <= idx1 < len(words) else e
    following = [a.start_ms for a in atoms if a.start_ms >= last_word_end]
    if following:
        e = min(e, min(following))
    preceding = [a.end_ms for a in atoms if a.end_ms <= first_word_start]
    if preceding:
        s = max(s, max(preceding))
    return s, e


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
    atoms = build_atoms(file_id, duration_ms, motion, scene, turns, words=words)
    hints = speech_hints(words)
    return Lattice(file_id=file_id, duration_ms=duration_ms, words=words,
                   turns=turns, hints=hints, atoms=atoms)
