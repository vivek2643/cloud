"""
Tests for the cuts-v2 unified priority partition (no DB).

Two layers:
  * LOW-LEVEL: ``_claim`` / ``_merge_continuous`` tested directly against
    hand-built ``_Candidate`` / ``_Placed`` objects -- the tag-vs-trim /
    overlap-threshold logic and same-primary merge, independent of where
    candidates come from.
  * END-TO-END: ``partition_clip`` against real ``ClipArtifacts`` (thoughts +
    a locked-off-camera motion fixture) -- said still claims first, tightness
    applies, multiple speakers, full contiguous coverage. Camera-move-state
    video segmentation itself has its own dedicated tests,
    ``scripts/test_video_segments.py``.

Run:  .venv/bin/python scripts/test_partition.py
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


def _cand(channel, s, e, peak=None, label="x", speaker=None):
    return pt._Candidate(channel, s, e, peak if peak is not None else (s + e) // 2, label, speaker)


# --------------------------------------------------------------------------
# Low-level: _claim / _merge_continuous
# --------------------------------------------------------------------------

def test_claim_talking_while_gesturing_is_one_tagged_cut():
    """A done candidate fully inside a said candidate is Fact #2's textbook
    case: simultaneity is a TAG, never a parallel overlapping cut."""
    said = _cand(vocab.CHANNEL_SAID, 1000, 6000, label="line")
    done = _cand(vocab.CHANNEL_DONE, 2500, 3500, label="action")
    placed = pt._claim([said, done], field=None, duration_ms=8000)
    assert len(placed) == 1, placed
    assert placed[0].tags == [vocab.CHANNEL_SAID, vocab.CHANNEL_DONE], placed[0].tags
    print("ok  test_claim_talking_while_gesturing_is_one_tagged_cut")


def test_claim_mostly_covered_candidate_demotes_to_tag():
    """A candidate ~97% covered by an already-claimed said span is absorbed as
    a tag (overlap >= 60% of the shorter span), not trimmed into a sliver cut."""
    said = _cand(vocab.CHANNEL_SAID, 1000, 5000)
    done = _cand(vocab.CHANNEL_DONE, 1200, 5100)   # pokes 100ms past said's end
    placed = pt._claim([said, done], field=None, duration_ms=8000)
    assert len(placed) == 1, placed
    assert vocab.CHANNEL_DONE in placed[0].tags, placed[0].tags
    print("ok  test_claim_mostly_covered_candidate_demotes_to_tag")


def test_claim_partial_overlap_trims_to_free_remainder():
    """A candidate that only PARTIALLY overlaps a said span (< 60% covered)
    keeps its own cut(s), trimmed to the free remainder(s) -- not a tag."""
    said = _cand(vocab.CHANNEL_SAID, 3000, 4000)
    done = _cand(vocab.CHANNEL_DONE, 2000, 5000)   # 1000/3000 = 33% covered
    placed = pt._claim([said, done], field=None, duration_ms=8000)
    done_placed = [p for p in placed if p.primary == vocab.CHANNEL_DONE]
    assert done_placed, placed
    said_placed = next(p for p in placed if p.primary == vocab.CHANNEL_SAID)
    for d in done_placed:
        assert d.end_ms <= said_placed.start_ms or d.start_ms >= said_placed.end_ms, (d, said_placed)
    print("ok  test_claim_partial_overlap_trims_to_free_remainder")


def test_claim_two_disjoint_remainders_both_survive_as_cuts():
    """A catch-all candidate spanning a whole clip with a said cut carved out
    of its MIDDLE leaves TWO disjoint remainders -- both become their own
    cuts; the candidate must NOT collapse into a single tag (see partition.py's
    `_claim` docstring on why multi-remainder candidates skip the aggregate
    overlap check)."""
    said = _cand(vocab.CHANNEL_SAID, 1000, 6000)
    shown = _cand(vocab.CHANNEL_SHOWN, 0, 8000)
    placed = pt._claim([said, shown], field=None, duration_ms=8000)
    shown_placed = [p for p in placed if p.primary == vocab.CHANNEL_SHOWN]
    assert len(shown_placed) == 2, placed
    print("ok  test_claim_two_disjoint_remainders_both_survive_as_cuts")


def test_claim_non_overlap_holds_under_three_way_contention():
    said = _cand(vocab.CHANNEL_SAID, 2000, 6000)
    done = _cand(vocab.CHANNEL_DONE, 3000, 4000)
    shown = _cand(vocab.CHANNEL_SHOWN, 0, 10_000)
    placed = pt._claim([said, done, shown], field=None, duration_ms=10_000)
    ordered = sorted(placed, key=lambda p: p.start_ms)
    for a, b in zip(ordered, ordered[1:]):
        assert a.end_ms <= b.start_ms, (a, b)
    assert ordered[0].start_ms == 0 and ordered[-1].end_ms == 10_000, ordered
    print("ok  test_claim_non_overlap_holds_under_three_way_contention")


def test_merge_continuous_same_primary_no_boundary_between():
    p1 = pt._Placed(1000, 2000, [vocab.CHANNEL_DONE], vocab.CHANNEL_DONE, "a", None, 1500)
    p2 = pt._Placed(2000, 3000, [vocab.CHANNEL_DONE], vocab.CHANNEL_DONE, "b", None, 2500)
    merged = pt._merge_continuous([p1, p2], scene=None)
    assert len(merged) == 1, merged
    assert merged[0].start_ms == 1000 and merged[0].end_ms == 3000, merged[0]
    print("ok  test_merge_continuous_same_primary_no_boundary_between")


def test_merge_stops_at_a_scene_cut_boundary():
    """The SAME touching geometry, but with a hard scene cut sitting between
    the two pieces, must never merge across it."""
    p1 = pt._Placed(1000, 1990, [vocab.CHANNEL_DONE], vocab.CHANNEL_DONE, "a", None, 1500)
    p2 = pt._Placed(2010, 3000, [vocab.CHANNEL_DONE], vocab.CHANNEL_DONE, "b", None, 2500)
    scene = {"shot_points": [{"ts_ms": 2000, "kind": "shot_cut", "score": 1.0}]}
    merged = pt._merge_continuous([p1, p2], scene=scene)
    assert len(merged) == 2, merged
    print("ok  test_merge_stops_at_a_scene_cut_boundary")


def test_merge_never_crosses_a_speaker_change():
    p1 = pt._Placed(1000, 2000, [vocab.CHANNEL_SAID], vocab.CHANNEL_SAID, "a", "S0", 1500)
    p2 = pt._Placed(2000, 3000, [vocab.CHANNEL_SAID], vocab.CHANNEL_SAID, "b", "S1", 2500)
    merged = pt._merge_continuous([p1, p2], scene=None)
    assert len(merged) == 2, merged
    print("ok  test_merge_never_crosses_a_speaker_change")


# --------------------------------------------------------------------------
# End-to-end: partition_clip
# --------------------------------------------------------------------------

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


def _locked_motion(duration_ms, hop_ms=100, bumps=()):
    """A camera that never moves -- drives ``video_segments``' static-shot
    fallback (subject-beat sub-split) rather than its camera-move-state
    machinery (already covered by ``scripts/test_video_segments.py``).
    ``bumps`` is a list of (center_ms, peak_value) rise-peak-fall bells."""
    n = duration_ms // hop_ms + 1
    energy = [0.05] * n
    for center_ms, peak_val in bumps:
        ci = center_ms // hop_ms
        for off, frac in zip(range(-2, 3), (0.3, 0.7, 1.0, 0.7, 0.3)):
            i = ci + off
            if 0 <= i < n:
                energy[i] = max(energy[i], peak_val * frac)
    return {
        "hop_ms": hop_ms, "action_energy": energy,
        "camera_stability": [0.95] * n, "camera_motion": [0.02] * n,
        "blur": [0.5] * n,
    }


def _clip(duration_ms, thoughts=(), motion=None, scene=None, audio=None):
    return pt.ClipArtifacts(file_id="f1", duration_ms=duration_ms, thoughts=list(thoughts),
                            motion=motion, scene=scene, audio=audio)


def test_partition_clip_non_overlap_and_full_coverage():
    clip = _clip(
        10_000,
        thoughts=[_thought("S0", 2000, 6000, "we shipped the whole thing today")],
        motion=_locked_motion(10_000, bumps=[(7500, 0.9)]),
    )
    cuts = pt.partition_clip(clip, energy=0.5)
    ordered = sorted(cuts, key=lambda c: c.src_in_ms)
    for a, b in zip(ordered, ordered[1:]):
        assert a.src_out_ms <= b.src_in_ms, (a, b)
    assert ordered[0].src_in_ms == 0 and ordered[-1].src_out_ms == 10_000, ordered
    print("ok  test_partition_clip_non_overlap_and_full_coverage")


def test_partition_clip_said_wins_over_video_and_is_never_invaded():
    clip = _clip(
        10_000,
        thoughts=[_thought("S0", 4000, 6000, "the exact same words every single time")],
        motion=_locked_motion(10_000),
    )
    cuts = pt.partition_clip(clip, energy=0.5)
    said_cut = next(c for c in cuts if c.primary == vocab.CHANNEL_SAID)
    for c in cuts:
        if c is said_cut:
            continue
        assert c.src_out_ms <= said_cut.src_in_ms or c.src_in_ms >= said_cut.src_out_ms, (c, said_cut)
    print("ok  test_partition_clip_said_wins_over_video_and_is_never_invaded")


def test_partition_clip_no_motion_yields_one_whole_clip_shown_cut():
    """Best-effort: no motion data at all still yields full, single-cut
    coverage -- never a gap, never a crash."""
    clip = _clip(5000)
    cuts = pt.partition_clip(clip, energy=0.5)
    assert len(cuts) == 1, cuts
    assert cuts[0].src_in_ms == 0 and cuts[0].src_out_ms == 5000, cuts[0]
    assert cuts[0].primary == vocab.CHANNEL_SHOWN
    print("ok  test_partition_clip_no_motion_yields_one_whole_clip_shown_cut")


def test_partition_clip_respects_a_hard_shot_cut_even_at_broad():
    """Phase C3a: a scene shot cut splits the video into two cuts regardless
    of energy -- two visually-similar shots stitched together must never read
    as one continuous cut just because Broad wants fewer, longer segments.
    At high energy, TIGHTNESS may inset each cut away from the shot-cut
    boundary (a separate, legitimate effect) -- the invariant that matters
    here is the COUNT and that they never invade one another, not the exact
    boundary position."""
    clip = _clip(10_000, motion=_locked_motion(10_000),
                 scene={"shot_points": [{"ts_ms": 5000, "kind": "shot_cut", "score": 1.0}]})
    for energy in (0.1, 0.5, 0.9):
        cuts = pt.partition_clip(clip, energy=energy)
        assert len(cuts) == 2, (energy, cuts)
        ordered = sorted(cuts, key=lambda c: c.src_in_ms)
        assert ordered[0].src_out_ms <= ordered[1].src_in_ms, (energy, ordered)
        assert ordered[0].src_in_ms < 5000 <= ordered[1].src_out_ms, (energy, ordered)
    # At Balanced (no tightness inset), the boundary sits exactly on the cut.
    balanced = sorted(pt.partition_clip(clip, energy=0.5), key=lambda c: c.src_in_ms)
    assert balanced[0].src_out_ms == 5000 and balanced[1].src_in_ms == 5000, balanced
    print("ok  test_partition_clip_respects_a_hard_shot_cut_even_at_broad")


def test_partition_clip_multiple_thoughts_with_speakers():
    clip = _clip(
        8000,
        thoughts=[
            _thought("S0", 500, 2000, "what made you start this company"),
            _thought("S1", 2200, 6000, "honestly it was kind of an accident"),
        ],
        motion=_locked_motion(8000),
    )
    cuts = pt.partition_clip(clip, energy=0.5)
    said_cuts = sorted((c for c in cuts if c.primary == vocab.CHANNEL_SAID),
                       key=lambda c: c.src_in_ms)
    assert len(said_cuts) == 2, said_cuts
    assert said_cuts[0].speaker == "S0" and said_cuts[1].speaker == "S1"
    print("ok  test_partition_clip_multiple_thoughts_with_speakers")


def test_partition_clip_tightens_video_more_at_sharp_than_balanced():
    """The dial's TIGHTNESS half: the segment containing a subject beat insets
    tighter at Sharp than at Balanced (Balanced's done/shown core_frac is None
    -- no inset at all)."""
    motion = _locked_motion(10_000, bumps=[(6000, 0.95)])
    clip = _clip(10_000, motion=motion)
    balanced = pt.partition_clip(clip, energy=0.5)
    sharp = pt.partition_clip(clip, energy=0.95)

    def containing(cuts, ts):
        return next(c for c in cuts if c.src_in_ms <= ts < c.src_out_ms)

    b = containing(balanced, 6000)
    s = containing(sharp, 6000)
    assert (s.src_out_ms - s.src_in_ms) < (b.src_out_ms - b.src_in_ms), (b, s)
    print("ok  test_partition_clip_tightens_video_more_at_sharp_than_balanced")


def test_partition_clip_tightness_keeps_an_impact_near_the_edge():
    """Anchor-aware tightness: a segment whose only payoff (an L1 action
    impact) sits near its END must still contain that impact after tightening
    -- even though the flat energy/blur would otherwise inset toward the clip
    START (the adversarial 'peak is nowhere near the moment that matters'
    case that used to drop the ball-hit)."""
    n = 10_000 // 100 + 1
    motion = {
        "hop_ms": 100,
        "action_energy": [0.05] * n,          # flat -> one whole-clip `shown` segment
        "camera_stability": [0.95] * n, "camera_motion": [0.02] * n,
        "blur": [0.5] * n,                     # flat -> `shown` peak lands at ts 0
        "action_points": [{"ts_ms": 9000, "kind": "action_impact", "score": 1.0}],
    }
    clip = _clip(10_000, motion=motion)
    cuts = pt.partition_clip(clip, energy=0.9)          # Sharp -> tightness on
    hit = [c for c in cuts if c.src_in_ms <= 9000 <= c.src_out_ms]
    assert hit, ("impact at 9000 fell into no cut", cuts)
    c = hit[0]
    assert c.src_out_ms - c.src_in_ms < 10_000, ("cut was not tightened at all", c)
    print("ok  test_partition_clip_tightness_keeps_an_impact_near_the_edge")


def test_partition_clip_all_action_keeps_nearly_the_whole_span():
    """'If the clip is all action, show (almost) the whole thing.' Impacts
    spread across the clip can't tighten without dropping one, so the anchor
    envelope keeps a span covering every impact."""
    n = 10_000 // 100 + 1
    motion = {
        "hop_ms": 100,
        "action_energy": [0.05] * n,
        "camera_stability": [0.95] * n, "camera_motion": [0.02] * n,
        "blur": [0.5] * n,
        "action_points": [{"ts_ms": t, "kind": "action_impact", "score": 1.0}
                         for t in (1000, 3000, 5000, 7000, 9000)],
    }
    clip = _clip(10_000, motion=motion)
    cuts = pt.partition_clip(clip, energy=0.9)
    span = [c for c in cuts if c.src_in_ms <= 5000 <= c.src_out_ms][0]
    assert span.src_in_ms <= 1000 and span.src_out_ms >= 9000, ("dropped an impact", span)
    print("ok  test_partition_clip_all_action_keeps_nearly_the_whole_span")


def test_audio_onset_is_an_anchor_flow_misses():
    """An audio transient (a 'crack') with no motion signal at all is still an
    anchor tightness must keep -- the case flow alone can't see."""
    anchors = pt._video_anchors(
        motion={"hop_ms": 100, "action_energy": [], "action_points": []},
        audio={"prosody_hop_ms": 100,
               "rms_db": [-40.0] * 50 + [-30.0] + [-40.0] * 49},  # +10 dB jump at ts 5000
        s=0, e=10_000,
    )
    assert 5000 in anchors, anchors
    print("ok  test_audio_onset_is_an_anchor_flow_misses")


def test_partition_clip_is_deterministic():
    """Same artifacts, same energy -> identical cuts every time (no hidden
    randomness/ordering dependence)."""
    clip = _clip(
        8000,
        thoughts=[_thought("S0", 1000, 4000, "a repeatable thought")],
        motion=_locked_motion(8000, bumps=[(5500, 0.9)]),
    )
    a = [c.to_dict() for c in pt.partition_clip(clip, energy=0.6)]
    b = [c.to_dict() for c in pt.partition_clip(clip, energy=0.6)]
    assert a == b, (a, b)
    print("ok  test_partition_clip_is_deterministic")


def main():
    test_claim_talking_while_gesturing_is_one_tagged_cut()
    test_claim_mostly_covered_candidate_demotes_to_tag()
    test_claim_partial_overlap_trims_to_free_remainder()
    test_claim_two_disjoint_remainders_both_survive_as_cuts()
    test_claim_non_overlap_holds_under_three_way_contention()
    test_merge_continuous_same_primary_no_boundary_between()
    test_merge_stops_at_a_scene_cut_boundary()
    test_merge_never_crosses_a_speaker_change()
    test_partition_clip_non_overlap_and_full_coverage()
    test_partition_clip_said_wins_over_video_and_is_never_invaded()
    test_partition_clip_no_motion_yields_one_whole_clip_shown_cut()
    test_partition_clip_respects_a_hard_shot_cut_even_at_broad()
    test_partition_clip_multiple_thoughts_with_speakers()
    test_partition_clip_tightens_video_more_at_sharp_than_balanced()
    test_partition_clip_tightness_keeps_an_impact_near_the_edge()
    test_partition_clip_all_action_keeps_nearly_the_whole_span()
    test_audio_onset_is_an_anchor_flow_misses()
    test_partition_clip_is_deterministic()
    print("\nall partition tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
