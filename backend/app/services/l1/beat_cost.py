"""
L1 derived signal: beat/music cut-cost grid.

Pure, free derivation over signals the audio_features stage already persists
(librosa beat onsets + bpm + musicality flag). No new compute, no model.

Editors cut ON the beat, so this is a "hit" channel: the cost curve dips to 0 at
each beat onset and ramps back to 1 between beats (a triangular well of width
+/- BEAT_TOL_MS). "Safe to cut" = 1 - cost. Non-musical files yield an empty
grid (``has_beat = False``).

Discrete ``beat_points`` mirror the onsets so a later stage can snap a cut to an
exact downbeat instead of a quantized hop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from app.services.l1.cut_grid_common import hit_cost_curve
from app.services.l1.cut_grid_params import BEAT_HOP_MS, BEAT_TOL_MS


@dataclass
class BeatGrid:
    has_beat: bool = False
    hop_ms: int = BEAT_HOP_MS
    bpm: float = 0.0
    cost: List[float] = field(default_factory=list)
    points: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "has_beat": self.has_beat,
            "hop_ms": self.hop_ms,
            "bpm": self.bpm,
            "cost": self.cost,
            "points": self.points,
        }


def compute_beat_grid(
    *,
    is_musical: bool,
    bpm: float,
    onsets_ms: List[int],
    duration_ms: int,
    hop_ms: int = BEAT_HOP_MS,
    tol_ms: int = BEAT_TOL_MS,
) -> BeatGrid:
    onsets = sorted(int(o) for o in (onsets_ms or []) if o is not None)
    if not is_musical or not onsets or duration_ms <= 0:
        return BeatGrid(has_beat=False, hop_ms=hop_ms, bpm=float(bpm or 0.0))

    cost = hit_cost_curve(onsets, duration_ms, hop_ms, tol_ms)
    points = [
        {"ts_ms": o, "kind": "beat", "score": 1.0}
        for o in onsets
        if 0 <= o <= duration_ms
    ]
    return BeatGrid(
        has_beat=True,
        hop_ms=hop_ms,
        bpm=float(bpm or 0.0),
        cost=cost,
        points=points,
    )
