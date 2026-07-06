"""
Tests for cuts-v3 transition-point detection (``motion_dynamics._wipe_points`` /
``_degenerate_points``) -- no DB, no ffmpeg/opencv (pure arrays, mirroring how
the rest of this codebase tests local-maxima-based signal logic). End-to-end
validation against real optical flow was done by hand against synthetic
clips (a genuine frame-filling sweep, a sustained defocus, and a realistic
steady pan over static texture -- zero false positives); this suite locks in
that behavior at the array level so it doesn't need ffmpeg/opencv to run.

Run:  .venv/bin/python scripts/test_transition_points.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l1 import motion_dynamics as md  # noqa: E402


def test_wipe_fires_on_a_high_area_low_coherence_spike():
    """A genuine wipe: swept-area fraction spikes above the floor AND
    coherence collapses at the same instant -- fires, at the peak instant."""
    hop = 100
    wipe_frac = [0.0] * 20 + [0.1, 0.2, 0.45, 0.25, 0.1] + [0.0] * 20
    coherence = [0.95] * 20 + [0.7, 0.5, 0.25, 0.5, 0.7] + [0.95] * 20
    points = md._wipe_points(wipe_frac, coherence, hop)
    assert len(points) == 1, points
    assert points[0]["kind"] == "wipe"
    assert points[0]["ts_ms"] == 22 * hop, points
    print("ok  test_wipe_fires_on_a_high_area_low_coherence_spike")


def test_wipe_suppressed_when_coherence_stays_high():
    """The SAME area spike, but coherence never collapses (a clean fast pan,
    not chaotic near-field motion) -- never a wipe."""
    hop = 100
    wipe_frac = [0.0] * 20 + [0.1, 0.2, 0.45, 0.25, 0.1] + [0.0] * 20
    coherence = [0.95] * 45   # stays high throughout
    points = md._wipe_points(wipe_frac, coherence, hop)
    assert points == [], points
    print("ok  test_wipe_suppressed_when_coherence_stays_high")


def test_wipe_suppressed_when_it_never_recovers():
    """A sustained chaotic stretch that never drops back down is a
    disturbance (handled elsewhere), not a quick occlusion sweep."""
    hop = 100
    wipe_frac = [0.0] * 20 + [0.4] * 15   # spikes and STAYS high, no recovery
    coherence = [0.95] * 20 + [0.2] * 15
    points = md._wipe_points(wipe_frac, coherence, hop)
    assert points == [], points
    print("ok  test_wipe_suppressed_when_it_never_recovers")


def test_wipe_below_area_floor_never_fires():
    """Ordinary subject motion (a person gesturing) never sweeps enough of the
    grid to look like a wipe, even if coherence dips a little."""
    hop = 100
    wipe_frac = [0.05] * 40
    coherence = [0.6] * 40
    points = md._wipe_points(wipe_frac, coherence, hop)
    assert points == [], points
    print("ok  test_wipe_below_area_floor_never_fires")


def test_degenerate_fires_on_a_sustained_blur_run():
    """A sustained maxed-out blur run marks its ONSET as degenerate."""
    hop = 100
    blur = [0.1] * 20 + [0.95] * 10 + [0.1] * 10   # 1000ms sustained at 100ms hop
    points = md._degenerate_points(blur, hop)
    assert len(points) == 1, points
    assert points[0]["ts_ms"] == 20 * hop, points
    assert points[0]["kind"] == "degenerate"
    print("ok  test_degenerate_fires_on_a_sustained_blur_run")


def test_degenerate_ignores_a_brief_blur_blip():
    """One or two soft frames from fast motion (not sustained) never fires --
    below DEGENERATE_MIN_MS."""
    hop = 100
    blur = [0.1] * 20 + [0.95] * 2 + [0.1] * 20   # only 200ms
    points = md._degenerate_points(blur, hop)
    assert points == [], points
    print("ok  test_degenerate_ignores_a_brief_blur_blip")


def test_degenerate_below_threshold_never_fires():
    """High but not-maxed blur (ordinary motion blur, not a true degenerate
    frame) never fires."""
    hop = 100
    blur = [0.7] * 40
    points = md._degenerate_points(blur, hop)
    assert points == [], points
    print("ok  test_degenerate_below_threshold_never_fires")


def test_empty_inputs_are_safe_noops():
    assert md._wipe_points([], [], 100) == []
    assert md._degenerate_points([], 100) == []
    print("ok  test_empty_inputs_are_safe_noops")


def test_transition_points_field_present_and_sorted_on_no_motion():
    """Best-effort: no motion data still yields a well-shaped (empty)
    transition_points list, never a crash or a missing key."""
    r = md.MotionDynamics(has_motion=False)
    d = r.to_dict()
    assert d["transition_points"] == [], d
    print("ok  test_transition_points_field_present_and_sorted_on_no_motion")


def main():
    test_wipe_fires_on_a_high_area_low_coherence_spike()
    test_wipe_suppressed_when_coherence_stays_high()
    test_wipe_suppressed_when_it_never_recovers()
    test_wipe_below_area_floor_never_fires()
    test_degenerate_fires_on_a_sustained_blur_run()
    test_degenerate_ignores_a_brief_blur_blip()
    test_degenerate_below_threshold_never_fires()
    test_empty_inputs_are_safe_noops()
    test_transition_points_field_present_and_sorted_on_no_motion()
    print("\nall transition-point tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
