"""
Tests for the sample-index helpers video_segments.py still carries
(``_seg_bounds``/``_sharpest_ms``) after the cuts-v2 camera-move-state
segmenter was retired (cleanup.plan.md B3). Pure, no DB. Run:
    .venv/bin/python scripts/test_video_segments.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import video_segments as vs  # noqa: E402


def test_seg_bounds_clamps_to_array_and_excludes_neighbor():
    # e lands exactly on a hop boundary -> must NOT include the next sample.
    assert vs._seg_bounds(10, 100, 0, 500) == (0, 4)
    # clamps to the array when e exceeds it.
    assert vs._seg_bounds(5, 100, 0, 10_000) == (0, 4)
    # empty/inverted span -> None.
    assert vs._seg_bounds(10, 100, 500, 500) is None
    print("ok  _seg_bounds clamps + excludes the neighbor's first sample")


def test_sharpest_ms_picks_least_blurred_instant():
    blur = [0.9, 0.9, 0.9, 0.05, 0.9]
    assert vs._sharpest_ms(blur, hop_ms=100, s=0, e=500, default_ms=999) == 300
    print("ok  _sharpest_ms picks the least-blurred sample in range")


def test_sharpest_ms_falls_back_without_blur_data():
    assert vs._sharpest_ms([], hop_ms=100, s=0, e=500, default_ms=250) == 250
    assert vs._sharpest_ms([0.1, 0.2], hop_ms=0, s=0, e=500, default_ms=250) == 250
    print("ok  _sharpest_ms falls back to default_ms with no usable blur data")


def main():
    test_seg_bounds_clamps_to_array_and_excludes_neighbor()
    test_sharpest_ms_picks_least_blurred_instant()
    test_sharpest_ms_falls_back_without_blur_data()
    print("\nall video-segments tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
