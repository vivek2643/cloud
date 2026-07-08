"""
Sample-index helpers shared by the peak-frame pickers (``post.pick_hero_ts_ms``,
``image_plan``). The rest of this module (the cuts-v2 camera-move-state video
segmenter) was retired in cleanup.plan.md B3.
"""
from __future__ import annotations

from typing import List, Optional, Tuple


def _seg_bounds(n: int, hop_ms: int, s: int, e: int) -> Optional[Tuple[int, int]]:
    """[s, e) (a segment's OWN span, exclusive of its end) -> inclusive sample
    index bounds, clamped to the array. `e // hop_ms` alone would include the
    NEXT segment's first sample when `e` lands exactly on a hop boundary --
    verified against a synthetic two-beat clip, where a middle segment's
    "strongest instant" search was silently picking up its neighbor's peak."""
    lo, hi = max(0, s // hop_ms), min(n - 1, (e - 1) // hop_ms)
    return None if hi < lo else (lo, hi)


def _sharpest_ms(blur: List[float], hop_ms: int, s: int, e: int, default_ms: int) -> int:
    """The least-blurred instant in [s, e) -- the thumbnail-worthy frame for a
    held (shown) segment. Falls back to ``default_ms`` when blur isn't
    available."""
    if not blur or hop_ms <= 0:
        return default_ms
    bounds = _seg_bounds(len(blur), hop_ms, s, e)
    if bounds is None:
        return default_ms
    lo, hi = bounds
    return min(range(lo, hi + 1), key=lambda i: blur[i]) * hop_ms
