"""
Tests for cuts-v2 camera-move-state video segmentation + hard shot cuts
(Phases C1 + C3a of cuts_v2_boundaries.plan.md) -- no DB. Run:
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
from app.services.l3 import vocab  # noqa: E402


def _flat_motion(n, hop=100, stability=0.95, camera_motion=0.02, action=0.05, blur=0.4):
    return {
        "hop_ms": hop,
        "camera_stability": [stability] * n,
        "camera_motion": [camera_motion] * n,
        "action_energy": [action] * n,
        "blur": [blur] * n,
    }


def _bell(center_i, values):
    """{index: value} around a center, for a rise-peak-fall bump (never a
    flat plateau, which would defeat local-maxima -- see the module's own
    dynamic-range guards)."""
    return dict(zip(range(center_i - len(values) // 2, center_i - len(values) // 2 + len(values)), values))


def test_hysteresis_absorbs_a_single_noisy_hop():
    """One stray 'move' hop in the middle of a long hold must NOT flip the
    confirmed state (that would flap the segmentation on sensor noise)."""
    raw = ["hold"] * 20
    raw[10] = "move"
    confirmed = vs._confirm_hysteresis(raw)
    assert confirmed == ["hold"] * 20, confirmed
    print("ok  test_hysteresis_absorbs_a_single_noisy_hop")


def test_hysteresis_confirms_a_sustained_move():
    """A move sustained for >= HYSTERESIS_HOPS hops DOES flip the confirmed
    state (after the hysteresis delay)."""
    raw = ["hold"] * 10 + ["move"] * 10 + ["hold"] * 10
    confirmed = vs._confirm_hysteresis(raw)
    assert "move" in confirmed, confirmed
    # It flips only after HYSTERESIS_HOPS consecutive new-state hops.
    first_move = confirmed.index("move")
    assert first_move >= 10, first_move
    print("ok  test_hysteresis_confirms_a_sustained_move")


def test_settle_point_lands_at_the_hold_after_a_move():
    """A hold -> move -> hold sequence yields exactly one settle point, at
    (approximately) the start of the SECOND hold -- never mid-move."""
    n = 100
    motion = _flat_motion(n)
    motion["camera_stability"] = [0.9] * 30 + [0.2] * 10 + [0.9] * 60
    motion["camera_motion"] = [0.05] * 30 + [0.6] * 10 + [0.05] * 60
    segs = vs.segment_video(motion, 10_000, energy=0.1)  # Broad: settle-only
    assert len(segs) == 2, segs
    boundary = segs[0].end_ms
    assert 3900 <= boundary <= 4300, boundary   # ~4000ms, hysteresis-delayed
    assert segs[1].start_ms == boundary
    print("ok  test_settle_point_lands_at_the_hold_after_a_move")


def test_broad_merges_a_short_mid_pan_steadying():
    """A brief steadying hold in the middle of continuous movement (shorter
    than BROAD_HOLD_MIN_MS but longer than HOLD_MIN_MS) is a real boundary at
    Calm+ but merges away at Broad -- fewer, longer segments."""
    n = 100
    motion = _flat_motion(n)
    # hold(0-2000) move(2000-3000) BRIEF hold(3000-4000, 1000ms) move(4000-5000) hold(5000-10000)
    motion["camera_stability"] = [0.9]*20 + [0.2]*10 + [0.9]*10 + [0.2]*10 + [0.9]*50
    motion["camera_motion"]    = [0.05]*20 + [0.6]*10 + [0.05]*10 + [0.6]*10 + [0.05]*50
    broad = vs.segment_video(motion, 10_000, energy=0.1)
    calm = vs.segment_video(motion, 10_000, energy=0.3)
    assert len(broad) < len(calm), (broad, calm)
    print("ok  test_broad_merges_a_short_mid_pan_steadying")


def test_granularity_adds_beat_split_only_from_balanced_up():
    """The same hold-move-hold + one clean subject beat: Broad/Calm never
    sub-split; Balanced/Tight/Sharp add the beat as a split point."""
    n = 100
    motion = _flat_motion(n)
    motion["camera_stability"] = [0.9] * 30 + [0.2] * 10 + [0.9] * 60
    motion["camera_motion"] = [0.05] * 30 + [0.6] * 10 + [0.05] * 60
    for i, v in _bell(60, [0.3, 0.5, 0.75, 0.9, 0.98, 0.85, 0.6, 0.35]).items():
        motion["action_energy"][i] = v

    for energy in (0.1, 0.3):
        segs = vs.segment_video(motion, 10_000, energy=energy)
        assert len(segs) == 2, (energy, segs)
    for energy in (0.5, 0.7, 0.9):
        segs = vs.segment_video(motion, 10_000, energy=energy)
        assert len(segs) == 3, (energy, segs)
    print("ok  test_granularity_adds_beat_split_only_from_balanced_up")


def test_flat_action_energy_never_false_splits():
    """A dead-flat action-energy signal (no real beat anywhere) must not
    trigger any sub-split at ANY granularity -- verified regression: a bare
    percentile floor over a flat array collapses to the flat value itself and
    fires local-maxima on nearly every hop."""
    n = 100
    motion = _flat_motion(n, action=0.12)
    for energy in (0.1, 0.5, 0.9):
        segs = vs.segment_video(motion, 10_000, energy=energy)
        assert len(segs) == 1, (energy, segs)
        assert segs[0].tag == vocab.CHANNEL_SHOWN, segs
    print("ok  test_flat_action_energy_never_false_splits")


def test_mostly_flat_with_one_bump_splits_exactly_once():
    """A long calm segment with ONE genuine embedded beat: the bare-percentile
    trap (elevated samples are a small minority) must still resolve to exactly
    one split, not a cut every few hundred ms."""
    n = 100
    motion = _flat_motion(n, stability=0.9, camera_motion=0.05, action=0.1)
    for i, v in _bell(60, [0.3, 0.5, 0.75, 0.9, 0.98, 0.85, 0.6, 0.35]).items():
        motion["action_energy"][i] = v
    segs = vs.segment_video(motion, 10_000, energy=0.9)   # Sharp: loosest floor
    assert len(segs) == 2, segs
    print("ok  test_mostly_flat_with_one_bump_splits_exactly_once")


def test_locked_off_shot_falls_back_to_subject_beats():
    """No camera move ANYWHERE (one giant hold) still segments -- via the SAME
    mechanism, not a separate code path -- on two isolated subject beats, at
    Balanced+; Broad keeps it as one whole cut."""
    n = 100
    motion = _flat_motion(n, stability=0.95, camera_motion=0.02, action=0.05)
    for i, v in _bell(22, [0.3, 0.6, 0.95, 0.6, 0.3]).items():
        motion["action_energy"][i] = v
    for i, v in _bell(72, [0.3, 0.6, 0.9, 0.55, 0.3]).items():
        motion["action_energy"][i] = v

    broad = vs.segment_video(motion, 10_000, energy=0.1)
    assert len(broad) == 1, broad

    sharp = vs.segment_video(motion, 10_000, energy=0.9)
    assert len(sharp) == 3, sharp
    print("ok  test_locked_off_shot_falls_back_to_subject_beats")


def test_no_neighbor_peak_bleed_across_a_split():
    """A segment's own 'strongest instant' search must never spill into the
    NEXT segment's first sample (a boundary-inclusivity regression: `e //
    hop_ms` alone lands exactly on the neighbor's first index)."""
    n = 100
    motion = _flat_motion(n, stability=0.95, camera_motion=0.02, action=0.05)
    for i, v in _bell(22, [0.3, 0.6, 0.95, 0.6, 0.3]).items():
        motion["action_energy"][i] = v
    for i, v in _bell(72, [0.3, 0.6, 0.9, 0.55, 0.3]).items():
        motion["action_energy"][i] = v
    segs = vs.segment_video(motion, 10_000, energy=0.9)
    middle = segs[1]
    # The middle segment's peak must be its OWN content, not the next
    # segment's peak bleeding across the boundary at its exact end.
    assert middle.peak_ms < segs[2].start_ms, (middle, segs[2])
    print("ok  test_no_neighbor_peak_bleed_across_a_split")


def test_no_motion_data_yields_one_whole_clip_shown_segment():
    """Best-effort: missing motion data entirely still yields full coverage,
    never a crash or a gap."""
    segs = vs.segment_video(None, 5000, energy=0.5)
    assert len(segs) == 1, segs
    assert segs[0].start_ms == 0 and segs[0].end_ms == 5000
    assert segs[0].tag == vocab.CHANNEL_SHOWN
    print("ok  test_no_motion_data_yields_one_whole_clip_shown_segment")


def test_shot_cut_always_splits_regardless_of_energy():
    """A hard shot cut (Phase C3a) splits a would-otherwise-single hold at
    EVERY granularity -- Broad included -- never merged away like a settle
    can be. Two visually-similar shots stitched together must never read as
    one continuous segment."""
    motion = _flat_motion(100)
    scene = {"shot_points": [{"ts_ms": 5000, "kind": "shot_cut", "score": 1.0}]}
    for energy in (0.1, 0.5, 0.9):
        without = vs.segment_video(motion, 10_000, energy)
        with_scene = vs.segment_video(motion, 10_000, energy, scene=scene)
        assert len(without) == 1, (energy, without)
        assert len(with_scene) == 2, (energy, with_scene)
        assert with_scene[0].end_ms == 5000 and with_scene[1].start_ms == 5000, with_scene
    print("ok  test_shot_cut_always_splits_regardless_of_energy")


def test_shot_cut_splits_even_with_no_motion_data():
    """A hard shot cut still splits the clip even when there's no camera/
    motion data at all -- best-effort still respects the top-priority
    boundary, never silently drops it."""
    scene = {"shot_points": [{"ts_ms": 3000, "kind": "shot_cut", "score": 1.0}]}
    segs = vs.segment_video(None, 8000, energy=0.5, scene=scene)
    assert len(segs) == 2, segs
    assert segs[0].end_ms == 3000 and segs[1].start_ms == 3000, segs
    assert segs[0].start_ms == 0 and segs[-1].end_ms == 8000, segs
    print("ok  test_shot_cut_splits_even_with_no_motion_data")


def test_no_scene_data_is_a_safe_noop():
    """Omitting scene entirely (or an empty dict) behaves exactly like
    before -- no accidental split introduced."""
    motion = _flat_motion(100)
    segs_none = vs.segment_video(motion, 10_000, 0.5)
    segs_empty = vs.segment_video(motion, 10_000, 0.5, scene={})
    assert len(segs_none) == len(segs_empty) == 1, (segs_none, segs_empty)
    print("ok  test_no_scene_data_is_a_safe_noop")


def test_full_coverage_and_non_overlap_across_energies():
    """Segments always tile [0, duration_ms] exactly, at every energy."""
    n = 100
    motion = _flat_motion(n)
    motion["camera_stability"] = [0.9] * 20 + [0.2] * 10 + [0.9] * 15 + [0.2] * 10 + [0.9] * 45
    motion["camera_motion"] = [0.05] * 20 + [0.6] * 10 + [0.05] * 15 + [0.6] * 10 + [0.05] * 45
    for i, v in _bell(55, [0.3, 0.6, 0.95, 0.6, 0.3]).items():
        motion["action_energy"][i] = v
    for energy in (0.05, 0.25, 0.5, 0.75, 0.95):
        segs = vs.segment_video(motion, 10_000, energy=energy)
        assert segs[0].start_ms == 0 and segs[-1].end_ms == 10_000, (energy, segs)
        for a, b in zip(segs, segs[1:]):
            assert a.end_ms == b.start_ms, (energy, a, b)
    print("ok  test_full_coverage_and_non_overlap_across_energies")


def main():
    test_hysteresis_absorbs_a_single_noisy_hop()
    test_hysteresis_confirms_a_sustained_move()
    test_settle_point_lands_at_the_hold_after_a_move()
    test_broad_merges_a_short_mid_pan_steadying()
    test_granularity_adds_beat_split_only_from_balanced_up()
    test_flat_action_energy_never_false_splits()
    test_mostly_flat_with_one_bump_splits_exactly_once()
    test_locked_off_shot_falls_back_to_subject_beats()
    test_no_neighbor_peak_bleed_across_a_split()
    test_no_motion_data_yields_one_whole_clip_shown_segment()
    test_shot_cut_always_splits_regardless_of_energy()
    test_shot_cut_splits_even_with_no_motion_data()
    test_no_scene_data_is_a_safe_noop()
    test_full_coverage_and_non_overlap_across_energies()
    print("\nall video-segment tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
