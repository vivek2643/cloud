#!/usr/bin/env python3
"""Tests for the feel simulator (pure; no DB/LLM).

Run:  PYTHONPATH=. python scripts/test_feel.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import feel  # noqa: E402


def _seg(seg_id, file_id, in_ms, out_ms, content="", axis="speech", ref=None):
    return {"seg_id": seg_id, "file_id": file_id, "in_ms": in_ms, "out_ms": out_ms,
            "content": content, "axis": axis, "ref": ref}


def test_empty_timeline():
    r = feel.simulate([])
    assert r.cuts == [] and r.total_ms == 0
    assert "empty" in r.narrate().lower()
    print("ok  empty timeline narrates gracefully")


def test_pace_and_totals():
    # 4 words over 2s -> 2.0 wps; 10 words over 2s -> 5.0 wps.
    tl = [
        _seg("a", "f1", 0, 2000, content="one two three four", ref="f1:m00"),
        _seg("b", "f1", 2000, 4000, content="a b c d e f g h i j", ref="f1:m01"),
    ]
    r = feel.simulate(tl)
    assert r.total_ms == 4000, r.total_ms
    assert r.cuts[0].pace_wps == 2.0 and r.cuts[1].pace_wps == 5.0, r.cuts
    assert r.avg_pace == 3.5, r.avg_pace
    print("ok  pace + totals computed from the timeline alone")


def test_video_cut_has_no_pace():
    tl = [_seg("a", "f1", 0, 3000, content="", axis="any", ref="f1:m00")]
    r = feel.simulate(tl, meta_by_ref={"f1:m00": {"channel": "shown"}})
    c = r.cuts[0]
    assert c.pace_wps == 0.0 and not c.is_speech and c.channel == "shown"
    print("ok  silent video cut carries no pace, channel from map")


def test_fast_run_flagged():
    # Five back-to-back sub-2s cuts -> a fast burst spanning cuts 1-5.
    tl = [_seg(f"s{i}", "f1", i * 1000, i * 1000 + 900,
               content="quick", ref=f"f1:m{i:02d}") for i in range(5)]
    r = feel.simulate(tl)
    txt = r.narrate().lower()
    assert "race" in txt and "1-5" in txt, txt
    print("ok  fast run of short cuts flagged with anchors")


def test_same_speaker_jump_cut_risk():
    tl = [
        _seg("a", "f1", 0, 3000, content="hello there friend", ref="f1:m00"),
        _seg("b", "f1", 3000, 6000, content="and one more thing", ref="f1:m01"),
        _seg("c", "f2", 6000, 9000, content="different person now", ref="f2:m00"),
    ]
    meta = {"f1:m00": {"speaker_person": "P0"}, "f1:m01": {"speaker_person": "P0"},
            "f2:m00": {"speaker_person": "P1"}}
    r = feel.simulate(tl, meta_by_ref=meta)
    txt = r.narrate().lower()
    assert "jump-cut risk" in txt and "1-2" in txt, txt
    print("ok  adjacent same-speaker cuts flagged as jump-cut risk")


# --------------------------------------------------------------------------
# brain_mirror_readside.plan.md section 2: join_continuity -- the frame-
# grounded, content-relative seam classifier. file_id is a fast-path only
# (band-aid B retired); there is no fixed Hamming cutoff (band-aid A
# retired) -- verdicts are graded through a per-timeline scale.
# --------------------------------------------------------------------------

def _h(n):
    """A 64-bit int with exactly n low bits set -- hamming64(0, _h(n)) == n,
    so test fixtures can dial in an EXACT Hamming distance from a 0 baseline."""
    return (1 << n) - 1 if n > 0 else 0


def _desc(**by_file):
    """{file_id: [(t_ms, phash_int), ...]} -- load_descriptors' own shape."""
    return dict(by_file)


def test_join_seamless_same_clip_contiguous_no_descriptor_lookup_needed():
    tl = [
        _seg("a", "f1", 0, 10000, content="", axis="any", ref="f1:m00"),
        _seg("b", "f1", 10100, 15000, content="", axis="any", ref="f1:m01"),
    ]
    r = feel.simulate(tl)
    joins = feel.join_continuity(r.cuts)   # no descriptors at all
    assert len(joins) == 1
    assert joins[0]["kind"] == "seamless" and joins[0]["pic_bits"] is None, joins[0]
    print("ok  same-clip contiguous join classified seamless, no pixel lookup")


def test_join_same_clip_noncontiguous_similar_frames_is_a_jump_cut():
    # The clip93 pattern: backward same-file seam, SIMILAR seam frames
    # (small Hamming) -- one continuous framing, source played out of order.
    tl = [
        _seg("a", "f1", 0, 408000, content="", axis="any", ref="f1:m00"),
        _seg("b", "f1", 209000, 209500, content="", axis="any", ref="f1:m01"),
    ]
    # load_descriptors always returns samples sorted ascending by t_ms.
    descriptors = _desc(f1=[(209000, _h(6)), (408000, _h(0))])   # dist=6
    r = feel.simulate(tl, descriptors=descriptors)
    j = feel.join_continuity(r.cuts, r.descriptors)[0]
    assert j["kind"] == "jump-cut" and j["backward"] is True and j["pic_bits"] == 6, j
    print("ok  same-clip non-contiguous + similar frames -> jump-cut (clip93 pattern)")


def test_join_same_clip_noncontiguous_different_frames_is_a_cut_no_false_jump():
    # The talking-head -> slide pattern: one screen-recording file, non-
    # contiguous, but the PICTURE genuinely changed -- must NOT read jump-cut.
    tl = [
        _seg("a", "f1", 0, 10000, content="", axis="any", ref="f1:m00"),
        _seg("b", "f1", 45000, 50000, content="", axis="any", ref="f1:m01"),
    ]
    descriptors = _desc(f1=[(10000, _h(0)), (45000, _h(36))])   # dist=36, "clearly different"
    r = feel.simulate(tl, descriptors=descriptors)
    j = feel.join_continuity(r.cuts, r.descriptors)[0]
    assert j["kind"] == "cut" and j["pic_bits"] == 36, j
    print("ok  same-clip non-contiguous + different frames -> cut, no false jump-cut")


def test_join_cross_file_similar_frames_is_seamless_not_blocked_by_file_id():
    # Proves band-aid B is gone: a cross-file match-cut can read seamless.
    tl = [
        _seg("a", "f1", 0, 3000, content="", axis="any", ref="f1:m00"),
        _seg("b", "f2", 0, 3000, content="", axis="any", ref="f2:m00"),
    ]
    descriptors = _desc(f1=[(3000, _h(0))], f2=[(0, _h(0))])   # dist=0
    r = feel.simulate(tl, descriptors=descriptors)
    j = feel.join_continuity(r.cuts, r.descriptors)[0]
    assert j["kind"] == "seamless" and j["same_clip"] is False and j["pic_bits"] == 0, j
    print("ok  cross-file, pixel-identical seam classified seamless (no file_id gate)")


def test_join_cross_file_mid_band_is_a_cut_never_a_jump_cut():
    # Cross-file in the middle band is a deliberate match-adjacent join, a
    # "cut", never "jump-cut" -- jump-cut requires same_clip.
    tl = [
        _seg("a", "f1", 0, 3000, content="", axis="any", ref="f1:m00"),
        _seg("b", "f2", 0, 3000, content="", axis="any", ref="f2:m00"),
    ]
    descriptors = _desc(f1=[(3000, _h(0))], f2=[(0, _h(6))])   # dist=6 -- mid band
    r = feel.simulate(tl, descriptors=descriptors)
    j = feel.join_continuity(r.cuts, r.descriptors)[0]
    assert j["kind"] == "cut", j
    print("ok  cross-file mid-band seam classified cut, never jump-cut")


def test_join_missing_descriptor_fails_open_to_cut():
    tl = [
        _seg("a", "f1", 0, 10000, content="", axis="any", ref="f1:m00"),
        _seg("b", "f1", 45000, 50000, content="", axis="any", ref="f1:m01"),
    ]
    # No descriptors for f1 at all -- both lookups miss.
    r = feel.simulate(tl, descriptors={})
    j = feel.join_continuity(r.cuts, r.descriptors)[0]
    assert j["kind"] == "cut" and j["pic_bits"] is None, j
    print("ok  missing descriptor on either side fails open to cut, never a false jump")


def test_join_content_relative_scale_same_bits_different_verdict():
    # band-aid A retired: the SAME absolute Hamming distance (10) at the test
    # seam reads differently depending on how far-apart "different" content
    # reads elsewhere in THIS project.
    def _three_seam_timeline(other_dist):
        tl = [
            _seg("a", "f1", 0, 1000, content="", axis="any", ref="f1:m00"),
            _seg("b", "f1", 20000, 21000, content="", axis="any", ref="f1:m01"),
            _seg("c", "f1", 40000, 41000, content="", axis="any", ref="f1:m02"),
            _seg("d", "f1", 60000, 61000, content="", axis="any", ref="f1:m03"),
        ]
        descriptors = _desc(f1=[
            (1000, _h(0)), (20000, _h(other_dist)),      # seam 1: dist == other_dist
            (21000, _h(0)), (40000, _h(other_dist)),     # seam 2: dist == other_dist
            (41000, _h(0)), (60000, _h(10)),              # seam 3 (test seam): dist == 10, FIXED
        ])
        return feel.simulate(tl, descriptors=descriptors)

    clean = _three_seam_timeline(4)     # median([4, 4, 10]) == 4 -> narrow scale
    noisy = _three_seam_timeline(34)    # median([34, 34, 10]) == 34 -> wide scale
    j_clean = feel.join_continuity(clean.cuts, clean.descriptors)[-1]
    j_noisy = feel.join_continuity(noisy.cuts, noisy.descriptors)[-1]
    assert j_clean["pic_bits"] == 10 and j_noisy["pic_bits"] == 10, (j_clean, j_noisy)
    assert j_clean["kind"] == "cut", j_clean            # narrow band -> dist 10 reads as a real cut
    assert j_noisy["kind"] == "jump-cut", j_noisy        # wide band -> the SAME dist reads as a jump-cut
    assert j_clean["pic_scale"] < j_noisy["pic_scale"], (j_clean, j_noisy)
    print("ok  content-relative scale: same bits, different verdict (clean vs noisy project)")


def test_join_regression_multi_content_take_yields_zero_jump_cuts():
    # Regression: one screen-recording file with several genuinely different
    # segments (logo / talking-head / slides / demo) -- must yield ZERO
    # jump-cuts (the false-positive flood this plan retires).
    tl = [
        _seg("a", "f1", 0, 1000, content="", axis="any", ref="f1:m00"),
        _seg("b", "f1", 14000, 14500, content="", axis="any", ref="f1:m01"),
        _seg("c", "f1", 209000, 209500, content="", axis="any", ref="f1:m02"),
        _seg("d", "f1", 408000, 408500, content="", axis="any", ref="f1:m03"),
    ]
    descriptors = _desc(f1=[
        (1000, _h(0)), (14000, _h(40)),
        (14500, _h(0)), (209000, _h(40)),
        (209500, _h(0)), (408000, _h(40)),
    ])
    r = feel.simulate(tl, descriptors=descriptors)
    joins = feel.join_continuity(r.cuts, r.descriptors)
    assert all(j["kind"] != "jump-cut" for j in joins), joins
    print("ok  regression: a multi-content take (genuinely different frames) yields zero jump-cuts")


def test_join_regression_clip93_backward_seam_is_a_jump_cut():
    tl = [
        _seg("a", "f1", 0, 408000, content="", axis="any", ref="f1:m00"),
        _seg("b", "f1", 209000, 209500, content="", axis="any", ref="f1:m01"),
        _seg("c", "f1", 14000, 14500, content="", axis="any", ref="f1:m02"),
    ]
    descriptors = _desc(f1=[
        (14000, _h(5)), (209000, _h(5)),
        (209500, _h(0)), (408000, _h(0)),
    ])
    r = feel.simulate(tl, descriptors=descriptors)
    joins = feel.join_continuity(r.cuts, r.descriptors)
    assert all(j["kind"] == "jump-cut" for j in joins), joins
    print("ok  regression: clip93's 408->209->14 backward seams all classify jump-cut")


def test_narrate_reports_jump_cuts_in_editing_terms():
    tl = [
        _seg("a", "f1", 0, 408000, content="", axis="any", ref="f1:m00"),
        _seg("b", "f1", 209000, 209500, content="", axis="any", ref="f1:m01"),
    ]
    descriptors = _desc(f1=[(209000, _h(6)), (408000, _h(0))])
    r = feel.simulate(tl, descriptors=descriptors)
    txt = r.narrate().lower()
    assert "jump-cut" in txt and "1" in txt and "2" in txt, txt
    assert "out of source order" not in txt   # reworded in editing terms, not source-order jargon
    print("ok  narrate reports a jump-cut anchored at its positions, in editing terms")


def main():
    test_empty_timeline()
    test_pace_and_totals()
    test_video_cut_has_no_pace()
    test_fast_run_flagged()
    test_same_speaker_jump_cut_risk()
    test_join_seamless_same_clip_contiguous_no_descriptor_lookup_needed()
    test_join_same_clip_noncontiguous_similar_frames_is_a_jump_cut()
    test_join_same_clip_noncontiguous_different_frames_is_a_cut_no_false_jump()
    test_join_cross_file_similar_frames_is_seamless_not_blocked_by_file_id()
    test_join_cross_file_mid_band_is_a_cut_never_a_jump_cut()
    test_join_missing_descriptor_fails_open_to_cut()
    test_join_content_relative_scale_same_bits_different_verdict()
    test_join_regression_multi_content_take_yields_zero_jump_cuts()
    test_join_regression_clip93_backward_seam_is_a_jump_cut()
    test_narrate_reports_jump_cuts_in_editing_terms()
    print("\nall feel tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
