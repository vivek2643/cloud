"""
Tests for the cuts-v2 unified priority partition (no DB).

Exercises the core deterministic claim algorithm against hand-built clip
artifacts: non-overlap, simultaneity-as-tags, priority demotion vs trim, full
contiguous coverage, and same-primary merge across a scene-cut break. Run:
  .venv/bin/python scripts/test_partition.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import partition as pt  # noqa: E402
from app.services.l3 import vocab  # noqa: E402
from app.services.l3.thought_segments import Span, Thought  # noqa: E402


def _thought(speaker, in_ms, out_ms, text, punch=None, strength=0.7):
    p = punch or (in_ms, out_ms)
    return Thought(
        speaker=speaker,
        thought=Span(in_ms, out_ms, text),
        core=Span(in_ms, out_ms, text),
        punch=Span(p[0], p[1], text),
        setup=None,
        strength=strength,
    )


def _motion(duration_ms, hop_ms=100, elevated=(), points=(), blur=None):
    n = duration_ms // hop_ms + 1
    energy = [0.05] * n
    for lo, hi, val in elevated:
        for i in range(lo // hop_ms, min(n, hi // hop_ms + 1)):
            energy[i] = val
    return {
        "hop_ms": hop_ms, "action_energy": energy,
        "action_points": [{"ts_ms": t, "kind": "action_impact", "score": 1.0} for t in points],
        "camera_cut_cost": [0.0] * n,
        "blur": blur or [0.5] * n,
    }


def _scene(shot_points=(), composition_points=()):
    return {
        "hop_ms": 200,
        "shot_points": [{"ts_ms": t, "kind": "shot_cut", "score": 1.0} for t in shot_points],
        "composition_points": [{"ts_ms": t, "kind": "composition_change", "score": 1.0}
                               for t in composition_points],
    }


def _clip(duration_ms, thoughts=(), motion=None, scene=None, audio=None):
    return pt.ClipArtifacts(file_id="f1", duration_ms=duration_ms, thoughts=list(thoughts),
                            motion=motion, scene=scene, audio=audio)


def test_non_overlap_invariant_holds_under_contention():
    """A said span, a fully-overlapping done window, and full-clip shown
    coverage -- classic three-way contention -- never produces overlapping cuts."""
    clip = _clip(
        10_000,
        thoughts=[_thought("S0", 2000, 6000, "we shipped the whole thing today")],
        motion=_motion(10_000, elevated=[(3000, 4000, 0.9)], points=[3500]),
        scene=_scene(shot_points=[7000]),
    )
    cuts = pt.partition_clip(clip)
    assert len(cuts) >= 2, cuts
    ordered = sorted(cuts, key=lambda c: c.src_in_ms)
    for a, b in zip(ordered, ordered[1:]):
        assert a.src_out_ms <= b.src_in_ms, (a, b)
    # Full contiguous coverage: first cut starts at 0, last ends at duration.
    assert ordered[0].src_in_ms == 0, ordered[0]
    assert ordered[-1].src_out_ms == 10_000, ordered[-1]
    print("ok  test_non_overlap_invariant_holds_under_contention")


def test_talking_while_gesturing_is_one_tagged_cut_not_two():
    """A done window fully inside a said span is Fact #2's textbook case:
    simultaneity is a TAG, never a parallel overlapping cut."""
    clip = _clip(
        8000,
        thoughts=[_thought("S0", 1000, 6000, "and then i just pointed at it like this")],
        motion=_motion(8000, elevated=[(2500, 3500, 0.9)], points=[3000]),
    )
    cuts = pt.partition_clip(clip)
    said_cuts = [c for c in cuts if c.primary == vocab.CHANNEL_SAID]
    assert len(said_cuts) == 1, cuts
    sc = said_cuts[0]
    assert sc.tags == [vocab.CHANNEL_SAID, vocab.CHANNEL_DONE], sc.tags
    # No separate done cut carved out of the said span.
    assert not any(c.primary == vocab.CHANNEL_DONE
                   and c.src_in_ms >= sc.src_in_ms and c.src_out_ms <= sc.src_out_ms
                   for c in cuts), cuts
    print("ok  test_talking_while_gesturing_is_one_tagged_cut_not_two")


def test_mostly_covered_done_candidate_demotes_to_tag():
    """A done window 90%+ covered by an already-claimed said span is absorbed
    as a tag (overlap >= 60% of the shorter span), not trimmed into a sliver cut."""
    said = [(1000, 1000, 5000)]  # placeholder, unused
    clip = _clip(
        8000,
        thoughts=[_thought("S0", 1000, 5000, "a full complete thought right here")],
        # Window mostly inside [1000,5000] but pokes 100ms past the end -- still
        # >=60% covered, so it must demote, not spawn a 100ms sliver cut.
        motion=_motion(8000, elevated=[(1200, 5100, 0.9)], points=[3000]),
    )
    cuts = pt.partition_clip(clip)
    assert not any(c.primary == vocab.CHANNEL_DONE for c in cuts), cuts
    said_cut = next(c for c in cuts if c.primary == vocab.CHANNEL_SAID)
    assert vocab.CHANNEL_DONE in said_cut.tags, said_cut.tags
    print("ok  test_mostly_covered_done_candidate_demotes_to_tag")


def test_partially_overlapping_done_trims_to_free_remainder():
    """A done window that only PARTIALLY overlaps a said span (< 60% covered)
    keeps its own cut, trimmed to the free remainder -- not absorbed as a tag."""
    clip = _clip(
        8000,
        thoughts=[_thought("S0", 3000, 4000, "quick line")],
        # 3s window [2000,5000): only 1000ms (1/3) overlaps the 1000ms said span
        # -> covered_frac ~= 1000/3000 = 33% < 60% tag threshold -> own cut(s).
        motion=_motion(8000, elevated=[(2000, 5000, 0.9)], points=[2300]),
    )
    cuts = pt.partition_clip(clip)
    done_cuts = [c for c in cuts if c.primary == vocab.CHANNEL_DONE]
    assert done_cuts, cuts
    said_cut = next(c for c in cuts if c.primary == vocab.CHANNEL_SAID)
    for d in done_cuts:
        assert d.src_out_ms <= said_cut.src_in_ms or d.src_in_ms >= said_cut.src_out_ms, (d, said_cut)
    print("ok  test_partially_overlapping_done_trims_to_free_remainder")


def test_shown_candidates_cover_the_whole_clip_with_no_said_or_done():
    """A clip with nothing but held footage still partitions -- contiguous
    shown coverage, no gaps -- and a scene cut splits it into two cuts."""
    clip = _clip(6000, scene=_scene(shot_points=[3000]))
    cuts = pt.partition_clip(clip)
    assert all(c.primary == vocab.CHANNEL_SHOWN for c in cuts), cuts
    ordered = sorted(cuts, key=lambda c: c.src_in_ms)
    assert ordered[0].src_in_ms == 0 and ordered[-1].src_out_ms == 6000, ordered
    # The shot cut at 3000 must be a boundary between two cuts, not inside one.
    assert any(c.src_out_ms == 3000 for c in ordered), ordered
    assert any(c.src_in_ms == 3000 for c in ordered), ordered
    print("ok  test_shown_candidates_cover_the_whole_clip_with_no_said_or_done")


def test_no_scene_data_yields_one_whole_clip_shown_cut():
    """No scene detection at all (best-effort miss) still yields full,
    single-cut coverage -- never a gap."""
    clip = _clip(5000)
    cuts = pt.partition_clip(clip)
    assert len(cuts) == 1, cuts
    assert cuts[0].src_in_ms == 0 and cuts[0].src_out_ms == 5000, cuts[0]
    assert cuts[0].primary == vocab.CHANNEL_SHOWN
    print("ok  test_no_scene_data_yields_one_whole_clip_shown_cut")


def test_adjacent_done_windows_merge_across_no_scene_cut():
    """Two done candidates that land back-to-back (no scene cut, no said
    between them) merge into ONE continuous cut, not two hairline-adjacent ones."""
    clip = _clip(
        6000,
        motion=_motion(6000, elevated=[(1000, 2000, 0.9), (2000, 3000, 0.9)],
                      points=[1500, 2500]),
    )
    cuts = pt.partition_clip(clip)
    done_cuts = [c for c in cuts if c.primary == vocab.CHANNEL_DONE]
    # The two impacts sit in one contiguous elevated band -> _done_candidates
    # itself already merges them; assert there's exactly one done cut spanning
    # both, not two.
    assert len(done_cuts) == 1, done_cuts
    print("ok  test_adjacent_done_windows_merge_across_no_scene_cut")


def test_scene_cut_between_done_windows_keeps_them_separate():
    """The SAME back-to-back done geometry, but with a hard scene cut sitting
    between the two impacts, must never merge across it."""
    clip = _clip(
        6000,
        motion=_motion(6000, elevated=[(1000, 1900, 0.9), (2100, 3000, 0.9)],
                      points=[1500, 2500]),
        scene=_scene(shot_points=[2000]),
    )
    cuts = pt.partition_clip(clip)
    done_cuts = sorted((c for c in cuts if c.primary == vocab.CHANNEL_DONE),
                       key=lambda c: c.src_in_ms)
    assert len(done_cuts) == 2, done_cuts
    assert done_cuts[0].src_out_ms <= 2000 <= done_cuts[1].src_in_ms, done_cuts
    print("ok  test_scene_cut_between_done_windows_keeps_them_separate")


def test_said_span_is_never_invaded_by_a_neighboring_cut():
    """A said cut's claimed span is a hard boundary for its shown/done
    neighbors -- the snap/clamp logic never lets a neighbor's edge cross it."""
    clip = _clip(
        10_000,
        thoughts=[_thought("S0", 4000, 6000, "the exact same words every single time")],
    )
    cuts = pt.partition_clip(clip)
    said_cut = next(c for c in cuts if c.primary == vocab.CHANNEL_SAID)
    for c in cuts:
        if c is said_cut:
            continue
        assert c.src_out_ms <= said_cut.src_in_ms or c.src_in_ms >= said_cut.src_out_ms, (c, said_cut)
    print("ok  test_said_span_is_never_invaded_by_a_neighboring_cut")


def test_multiple_thoughts_yield_multiple_said_cuts_with_speakers():
    """Two thoughts (different speakers) yield two distinct said cuts, each
    carrying its own speaker -- no cross-speaker merge."""
    clip = _clip(
        8000,
        thoughts=[
            _thought("S0", 500, 2000, "what made you start this company"),
            _thought("S1", 2200, 6000, "honestly it was kind of an accident"),
        ],
    )
    cuts = pt.partition_clip(clip)
    said_cuts = sorted((c for c in cuts if c.primary == vocab.CHANNEL_SAID),
                       key=lambda c: c.src_in_ms)
    assert len(said_cuts) == 2, said_cuts
    assert said_cuts[0].speaker == "S0" and said_cuts[1].speaker == "S1"
    print("ok  test_multiple_thoughts_yield_multiple_said_cuts_with_speakers")


def main():
    test_non_overlap_invariant_holds_under_contention()
    test_talking_while_gesturing_is_one_tagged_cut_not_two()
    test_mostly_covered_done_candidate_demotes_to_tag()
    test_partially_overlapping_done_trims_to_free_remainder()
    test_shown_candidates_cover_the_whole_clip_with_no_said_or_done()
    test_no_scene_data_yields_one_whole_clip_shown_cut()
    test_adjacent_done_windows_merge_across_no_scene_cut()
    test_scene_cut_between_done_windows_keeps_them_separate()
    test_said_span_is_never_invaded_by_a_neighboring_cut()
    test_multiple_thoughts_yield_multiple_said_cuts_with_speakers()
    print("\nall partition tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
