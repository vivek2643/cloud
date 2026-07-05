"""
Cuts v2 -- the BASE partition.

The robust, deterministic skeleton every later stage sits on. A clip is sliced
ONLY at cut points we trust unconditionally, and NOTHING else -- no energy dial,
no action/motion beats, no done/shown tags, no junk filtering (all deferred).

Base boundaries (each a real, deterministic L1 signal):

  1. HARD SHOT CUT      -- a real cut inside the file (scene detection). Top
                           priority: two different shots must never share a cut.
  2. SPEAKER CHANGE     -- diarization: a new person speaking is always a cut.
  3. SPEECH EDGE        -- talking starts / stops (turn onset & offset): the
                           boundary between a spoken cut and a silent visual one.
  4. INTENTIONAL PAUSE  -- a long deliberate silence WITHIN one speaker's speech
                           (a real breath/stop, gap > LONG_PAUSE_MS).
  5. CAMERA MOVE/SETTLE -- every confirmed HOLD<->MOVE transition (cut when the
                           camera starts moving, and again where it settles).
  6. DISTURBANCE        -- the edges of a bad-camera span (whip/jerk/incoherent).

The result is ONE contiguous, non-overlapping, FULL-COVERAGE sequence of base
cuts: every millisecond of the clip lands in exactly one cut (slivers merge into
a neighbour -- nothing is ever dropped at the base). Each cut records WHY its
edges exist (``reason_in``/``reason_out``) so boundary quality is inspectable.

Pure core: ``partition_base(...)`` does no DB/model call; ``build_base_cuts``
loads + partitions for real callers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from app.services.l3 import vocab
from app.services.l3.base_cuts_params import (
    DIST_BRIDGE_MS,
    DIST_COHERENCE_MAX,
    DIST_MIN_MS,
    DIST_STABILITY_MAX,
    LONG_PAUSE_MS,
    MIN_CUT_MS,
    SNAP_MS,
)
from app.services.l3.video_segments import HOLD, _confirm_hysteresis, _hop_states, _runs

logger = logging.getLogger(__name__)

# Boundary reason tags (why a cut edge exists). Ordered by how strongly we trust
# them, so when several signals mark the same instant we report the strongest.
R_CLIP = "clip_edge"
R_SHOT = "shot_cut"
R_SPEAKER = "speaker_change"
R_SPEECH = "speech_edge"
R_PAUSE = "long_pause"
R_MOVE = "camera_move"
R_SETTLE = "settle"
R_DISTURB = "disturbance"

_REASON_RANK = {
    R_SHOT: 0, R_SPEAKER: 1, R_SPEECH: 2, R_PAUSE: 3,
    R_DISTURB: 4, R_MOVE: 5, R_SETTLE: 6, R_CLIP: 7,
}


@dataclass
class BaseCut:
    file_id: str
    start_ms: int
    end_ms: int
    kind: str                    # "speech" | "video"
    speaker: Optional[str]
    reason_in: str               # why this cut's LEFT edge exists
    reason_out: str              # why this cut's RIGHT edge exists

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    def to_dict(self) -> Dict:
        return {
            "file_id": self.file_id,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "kind": self.kind,
            "speaker": self.speaker,
            "reason_in": self.reason_in,
            "reason_out": self.reason_out,
        }


Turn = Tuple[int, int, str]      # (start_ms, end_ms, speaker)


# --------------------------------------------------------------------------
# Boundary sources
# --------------------------------------------------------------------------

def _speech_marks(turns: List[Turn]) -> List[Tuple[int, str]]:
    """Speech boundaries from merged speaker turns: every turn's onset/offset is
    a SPEECH EDGE; the seam between two adjacent turns is a SPEAKER CHANGE (new
    person) or, for the same speaker, an INTENTIONAL PAUSE (the merge already
    split them, so the gap cleared LONG_PAUSE_MS)."""
    marks: List[Tuple[int, str]] = []
    prev: Optional[Turn] = None
    for (s, e, spk) in turns:
        seam_reason = None
        if prev is not None:
            seam_reason = R_SPEAKER if spk != prev[2] else R_PAUSE
        marks.append((s, seam_reason or R_SPEECH))
        marks.append((e, R_SPEECH))
        prev = (s, e, spk)
    return marks


def _camera_marks(motion: Optional[dict]) -> List[Tuple[int, str]]:
    """Camera boundaries: every confirmed HOLD<->MOVE transition (a move onset
    or a settle)."""
    m = motion or {}
    hop = int(m.get("hop_ms") or 0)
    if not hop:
        return []
    runs = _runs(_confirm_hysteresis(_hop_states(m)), hop)
    out: List[Tuple[int, str]] = []
    for i, (state, s, _e) in enumerate(runs):
        if i == 0:
            continue                         # nothing precedes the first run
        out.append((s, R_SETTLE if state == HOLD else R_MOVE))
    return out


def _disturbance_marks(motion: Optional[dict]) -> List[Tuple[int, str]]:
    """Disturbance boundaries: the edges of a span where the camera is unstable
    or its motion is incoherent (a whip/jerk/focus-hunt) for at least
    DIST_MIN_MS. Absolute thresholds -- these signals aren't per-clip
    normalized."""
    m = motion or {}
    hop = int(m.get("hop_ms") or 0)
    if not hop:
        return []
    stab = m.get("camera_stability") or []
    coh = m.get("camera_coherence") or []
    n = max(len(stab), len(coh))
    bad = [
        ((stab[i] if i < len(stab) else 1.0) < DIST_STABILITY_MAX)
        or ((coh[i] if i < len(coh) else 1.0) < DIST_COHERENCE_MAX)
        for i in range(n)
    ]
    # Bad runs, then bridge those separated by < DIST_BRIDGE_MS (a momentary dip
    # under threshold in one continuous shake) so a pervasively shaky clip is one
    # disturbance span, not a string of fragments.
    runs: List[List[int]] = []
    i = 0
    while i < n:
        if not bad[i]:
            i += 1
            continue
        j = i
        while j < n and bad[j]:
            j += 1
        if runs and i * hop - runs[-1][1] < DIST_BRIDGE_MS:
            runs[-1][1] = j * hop
        else:
            runs.append([i * hop, j * hop])
        i = j
    out: List[Tuple[int, str]] = []
    for a, b in runs:
        if b - a >= DIST_MIN_MS:
            out.append((a, R_DISTURB))
            out.append((b, R_DISTURB))
    return out


def _shot_marks(scene: Optional[dict]) -> List[Tuple[int, str]]:
    return [(int(p["ts_ms"]), R_SHOT)
            for p in ((scene or {}).get("shot_points") or [])
            if isinstance(p, dict) and "ts_ms" in p]


# --------------------------------------------------------------------------
# Compose: collect marks -> snap -> slice -> classify
# --------------------------------------------------------------------------

def _collect(marks: List[Tuple[int, str]], duration_ms: int) -> Dict[int, Set[str]]:
    """Merge marks into a {ts: {reasons}} map, snapping any two within SNAP_MS
    onto the earlier one (one event, several signals)."""
    at: Dict[int, Set[str]] = {0: {R_CLIP}, duration_ms: {R_CLIP}}
    for ts, reason in sorted(marks):
        if not (0 < ts < duration_ms):
            continue
        near = next((k for k in at if abs(k - ts) <= SNAP_MS), None)
        at.setdefault(near if near is not None else ts, set()).add(reason)
    return at


def _reason(reasons: Set[str]) -> str:
    return min(reasons, key=lambda r: _REASON_RANK.get(r, 99))


def _turn_at(turns: List[Turn], mid_ms: int) -> Optional[Turn]:
    for (s, e, spk) in turns:
        if s <= mid_ms < e:
            return (s, e, spk)
    return None


def partition_base(file_id: str, duration_ms: int, turns: List[Turn],
                   motion: Optional[dict], scene: Optional[dict]) -> List[BaseCut]:
    """Pure core: slice [0, duration_ms] at the base boundaries into a
    contiguous, non-overlapping, full-coverage sequence of BaseCuts."""
    if duration_ms <= 0:
        return []

    # Speech is protected: it's carved by speaker-change / pause / speech-edge
    # ONLY. Camera moves and disturbances subdivide the NON-speech remainder --
    # we never chop a talking span just because the camera panned mid-sentence
    # ("never cut speech"). So drop any camera/disturbance mark inside a turn.
    speech_spans = [(s, e) for (s, e, _spk) in turns]

    def _in_speech(ts: int) -> bool:
        return any(s < ts < e for (s, e) in speech_spans)

    video_marks = [(ts, r) for (ts, r) in (_camera_marks(motion) + _disturbance_marks(motion))
                   if not _in_speech(ts)]
    at = _collect(_shot_marks(scene) + _speech_marks(turns) + video_marks, duration_ms)
    bounds = sorted(at)

    # Merge sub-MIN_CUT_MS slivers forward so coverage stays total: a boundary
    # that would open a too-short cut is dropped (its reason folds into the next
    # real boundary). The clip end is never dropped.
    kept = [bounds[0]]
    for b in bounds[1:-1]:
        if b - kept[-1] >= MIN_CUT_MS:
            kept.append(b)
    kept.append(bounds[-1])

    cuts: List[BaseCut] = []
    for a, b in zip(kept, kept[1:]):
        turn = _turn_at(turns, (a + b) // 2)
        cuts.append(BaseCut(
            file_id=file_id, start_ms=a, end_ms=b,
            kind="speech" if turn else "video",
            speaker=turn[2] if turn else None,
            reason_in=_reason(at.get(a, {R_CLIP})),
            reason_out=_reason(at.get(b, {R_CLIP})),
        ))
    return cuts


# --------------------------------------------------------------------------
# DB loader + convenience wrapper
# --------------------------------------------------------------------------

def build_base_cuts(file_id: str) -> List[BaseCut]:
    """Load one file's L1 artifacts and compute its base cuts. Empty when the
    file has no duration yet."""
    import psycopg
    from app.config import get_settings
    from app.services.l3.diarize import load_turns

    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        row = conn.execute(
            "select duration_seconds from files where id = %s", (file_id,)
        ).fetchone()
        if not row or not row[0]:
            return []
        duration_ms = int(float(row[0]) * 1000)

        m = conn.execute(
            """select hop_ms, camera_stability, camera_coherence, camera_motion, blur
                 from motion_dynamics where file_id = %s""",
            (file_id,),
        ).fetchone()
        motion = None
        if m:
            motion = {"hop_ms": m[0], "camera_stability": m[1] or [],
                      "camera_coherence": m[2] or [], "camera_motion": m[3] or [],
                      "blur": m[4] or []}

        s = conn.execute(
            "select shot_points from scene_cuts where file_id = %s", (file_id,)
        ).fetchone()
        scene = {"shot_points": s[0] or []} if s else None

    _text, _speakers, turns = load_turns(file_id, turn_gap_ms=LONG_PAUSE_MS)
    return partition_base(file_id, duration_ms, turns, motion, scene)
