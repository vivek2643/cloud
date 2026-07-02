"""
The PROGRAM clock's cut field -- the second of the edit's two clocks.

An edit has two independent clocks and both need to be first-class:

  * the SOURCE clock (inside each clip)   -- where is it clean/impactful to cut
    a source window. That lives in the per-clip fused seam field
    (``l1.fused_seams`` retained on the ``ClipTimeline``) and is what
    ``place_span`` snapping consults.
  * the PROGRAM clock (the edit itself)   -- where on the program timeline
    should edges LAND. Music beats, section boundaries, a rhythm the cut should
    breathe with. Nothing populates it until a program-side source exists
    (a music bed's beat grid is the first real one), but the CONCEPT and the
    consultation point exist now, so adding music later is data, not a rework.

This module deliberately reuses ``FusedField`` -- a program field is the same
"dense cost + discrete seams" shape, just over the program duration instead of
a clip. ``build_program_field`` returns ``None`` when there are no program-side
sources (the common case today), and every consumer treats ``None`` as
"no opinion": placement times pass through untouched.
"""
from __future__ import annotations

from typing import List, Optional

from app.services.l1.fused_seams import FusedField, FusedSeam, snap_point

# How far a program-side attractor may pull an anchor (same sovereignty
# principle as source-side snapping: the field assists, it never vetoes).
PROGRAM_SNAP_WIN_MS = 400

_HOP_MS = 100


def build_program_field(*, duration_ms: int,
                        beats_ms: Optional[List[int]] = None,
                        downbeats_ms: Optional[List[int]] = None) -> Optional[FusedField]:
    """Fuse the program-side attractors into one field over the program clock.

    Today the only recognized sources are a beat grid / downbeat grid (supplied
    when a music bed lands on A2). No sources -> ``None`` -- the program clock
    has no opinion and placement is purely the brain's.
    """
    beats = sorted(set(int(b) for b in (beats_ms or []) if 0 <= int(b) <= duration_ms))
    downs = sorted(set(int(b) for b in (downbeats_ms or []) if 0 <= int(b) <= duration_ms))
    if duration_ms <= 0 or (not beats and not downs):
        return None
    n = max(1, duration_ms // _HOP_MS + 1)
    cost = [1.0] * n  # baseline: nothing special anywhere
    seams: List[FusedSeam] = []
    for ts in beats:
        j = min(n - 1, ts // _HOP_MS)
        cost[j] = min(cost[j], 0.3)
        seams.append(FusedSeam(ts_ms=ts, q=0.7, kind="beat", sources=["music"]))
    for ts in downs:  # downbeats dominate plain beats
        j = min(n - 1, ts // _HOP_MS)
        cost[j] = min(cost[j], 0.05)
        seams.append(FusedSeam(ts_ms=ts, q=0.95, kind="beat", sources=["music"]))
    seams.sort(key=lambda s: -s.q)
    return FusedField(hop_ms=_HOP_MS, cost=cost, seams=seams)


def snap_program_ms(field: Optional[FusedField], rough_ms: Optional[int], *,
                    win_ms: int = PROGRAM_SNAP_WIN_MS) -> Optional[int]:
    """Snap a program-clock anchor (a V2 ``from_ms``, a split window edge) to the
    program field, moving at most ``win_ms``. No field / no anchor -> unchanged.
    """
    if field is None or rough_ms is None:
        return rough_ms
    rough = max(0, int(rough_ms))
    return snap_point(field, rough, max(0, rough - win_ms), rough + win_ms)
