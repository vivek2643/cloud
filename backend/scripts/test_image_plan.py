"""
Pure unit tests for the cuts-v3 deterministic image plan
(``app.services.l3.image_plan``) -- no DB, no ffmpeg, no model calls.

Run:  .venv/bin/python scripts/test_image_plan.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import image_plan as ip  # noqa: E402
from app.services.l3.lattice import Atom, Lattice  # noqa: E402
from app.services.l3.pass1 import (  # noqa: E402
    Pass1Output, SpeechCut, TakeCandidate, TakeMember, VideoTentativeGroup,
)


def _words(n: int, gap_ms: int = 300, dur_ms: int = 200):
    out = []
    t = 0
    for i in range(n):
        out.append({"start_ms": t, "end_ms": t + dur_ms, "text": f"w{i}"})
        t += dur_ms + gap_ms
    return out


def _lattice_with_atoms(file_id: str, n_words: int, atoms):
    return Lattice(file_id=file_id, duration_ms=100000, words=_words(n_words), turns=[], hints=[], atoms=atoms)


def _atom(atom_id, s, e, anchors=None):
    return Atom(atom_id=atom_id, file_id="f1", start_ms=s, end_ms=e, state_in="x", state_out="y",
                action_energy=0.1, coherence=0.9, anchor_ms=anchors or [])


def test_speech_cut_frame_uses_sharpest_ms_in_span():
    lat = _lattice_with_atoms("f1", 5, [])
    # words 0..4 at t=0,500,1000,1500,2000 (200ms dur, 300ms gap)
    pass1 = Pass1Output(speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 2), label="x")])
    hop = 100
    blur = [1.0] * 30
    # word_span (0,2) -> ms range covers roughly [0, snap-before-word3). Put the sharpest
    # (lowest blur) sample near the middle of that range.
    for i in range(5, 10):
        blur[i] = 0.05
    motion = {"f1": {"hop_ms": hop, "blur": blur, "action_energy": [0.0] * 30}}
    frames = ip.build_image_plan(pass1, {"f1": lat}, motion, {}, {})
    assert len(frames) == 1, frames
    f = frames[0]
    assert f.reason == ip.REASON_SPEECH_CUT
    assert f.ref == "speech_cut[0]"
    assert 500 <= f.ts_ms <= 1000, f.ts_ms
    print("ok  test_speech_cut_frame_uses_sharpest_ms_in_span")


def test_composition_drift_extra_added_inside_span_only():
    lat = _lattice_with_atoms("f1", 3, [])
    pass1 = Pass1Output(speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 2), label="x")])
    scene = {"f1": {"composition_points": [
        {"ts_ms": 700},   # inside word span (0..~1700ms) -> kept
        {"ts_ms": 50000},  # way outside -> dropped
    ]}}
    frames = ip.build_image_plan(pass1, {"f1": lat}, {}, scene, {})
    drift = [f for f in frames if f.reason == ip.REASON_COMPOSITION_DRIFT]
    assert len(drift) == 1, frames
    assert drift[0].ts_ms == 700, drift[0]
    assert drift[0].ref == "speech_cut[0]"
    print("ok  test_composition_drift_extra_added_inside_span_only")


def test_every_pass1_unit_gets_a_frame_even_over_budget():
    """One frame per unit is MANDATORY -- the budget can never starve a unit
    of its only frame (a cut pass 2 never saw pixels for can't be judged;
    observed as a real 'no images resolved' pass-2b failure)."""
    lat = _lattice_with_atoms("f1", 5, [])
    pass1 = Pass1Output(
        speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 4), label="x")],
        video_tentative_groups=[VideoTentativeGroup(file_id="f1") for _ in range(6)],
    )
    v4_meta = {f"video_group[{gi}]": {"src_in_ms": gi * 1000, "src_out_ms": (gi + 1) * 1000}
              for gi in range(6)}
    orig_budget = ip.FRAME_BUDGET_PER_CLIP
    ip.FRAME_BUDGET_PER_CLIP = 2   # far fewer than the 7 units
    try:
        frames = ip.build_image_plan(pass1, {"f1": lat}, {}, {}, {}, v4_meta_by_ref=v4_meta)
    finally:
        ip.FRAME_BUDGET_PER_CLIP = orig_budget
    refs = {f.ref for f in frames}
    assert "speech_cut[0]" in refs
    for gi in range(6):
        assert f"video_group[{gi}]" in refs, (gi, refs)
    print("ok  test_every_pass1_unit_gets_a_frame_even_over_budget")


def test_video_group_unanchored_uses_calm_and_sharp_fallback():
    lat = _lattice_with_atoms("f1", 0, [])
    pass1 = Pass1Output(video_tentative_groups=[VideoTentativeGroup(file_id="f1")])
    v4_meta = {"video_group[0]": {"src_in_ms": 0, "src_out_ms": 1000}}
    hop = 100
    n = 10
    action = [0.8] * n
    blur = [0.8] * n
    action[4] = 0.05
    blur[4] = 0.05  # a clearly calmest+sharpest sample at i=4 -> ts=400
    motion = {"f1": {"hop_ms": hop, "action_energy": action, "blur": blur}}
    frames = ip.build_image_plan(pass1, {"f1": lat}, motion, {}, {}, v4_meta_by_ref=v4_meta)
    assert len(frames) == 1, frames
    assert frames[0].reason == ip.REASON_VIDEO_GROUP_CALM
    assert frames[0].ts_ms == 400, frames[0]
    print("ok  test_video_group_unanchored_uses_calm_and_sharp_fallback")


def test_video_group_with_no_v4_meta_is_skipped():
    lat = _lattice_with_atoms("f1", 0, [])
    pass1 = Pass1Output(video_tentative_groups=[VideoTentativeGroup(file_id="f1")])
    frames = ip.build_image_plan(pass1, {"f1": lat}, {}, {}, {})
    assert frames == [], frames
    print("ok  test_video_group_with_no_v4_meta_is_skipped")


def test_take_member_always_kept_even_at_tiny_budget():
    lat = _lattice_with_atoms("f1", 5, [])
    pass1 = Pass1Output(
        take_candidates=[TakeCandidate(group_id="tg1", members=[
            TakeMember(file_id="f1", word_span=(0, 1)),
            TakeMember(file_id="f1", word_span=(2, 3)),
        ])],
        speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 4), label="x")],
    )
    orig_budget = ip.FRAME_BUDGET_PER_CLIP
    ip.FRAME_BUDGET_PER_CLIP = 2
    try:
        frames = ip.build_image_plan(pass1, {"f1": lat}, {}, {}, {})
    finally:
        ip.FRAME_BUDGET_PER_CLIP = orig_budget
    # every unit's frame is mandatory: both take members AND the speech cut
    # ship even though the budget (2) is smaller than the plan (3).
    takes = [f for f in frames if f.reason == ip.REASON_TAKE_MEMBER]
    assert len(takes) == 2, frames
    assert any(f.reason == ip.REASON_SPEECH_CUT for f in frames), frames
    print("ok  test_take_member_always_kept_even_at_tiny_budget")


def test_budget_truncation_second_moment_outranks_drift():
    # A video group + a drift point inside the speech cut, no motion data
    # (hop_ms=0 -> every non-trivial span clears the runt guard, see
    # _is_runt_span). Tier order is mandatory > second_moment > drift: with
    # only 1 slot of budget beyond the 3 mandatory frames, it goes to a
    # 2nd-moment candidate (take's "late"), not the drift point.
    lat = _lattice_with_atoms("f1", 5, [])
    pass1 = Pass1Output(
        take_candidates=[TakeCandidate(group_id="tg1", members=[TakeMember(file_id="f1", word_span=(0, 1))])],
        speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 4), label="x")],
        video_tentative_groups=[VideoTentativeGroup(file_id="f1")],
    )
    v4_meta = {"video_group[0]": {"src_in_ms": 3000, "src_out_ms": 6000}}
    scene = {"f1": {"composition_points": [{"ts_ms": 700}]}}
    orig_budget = ip.FRAME_BUDGET_PER_CLIP
    ip.FRAME_BUDGET_PER_CLIP = 4   # 3 mandatory + room for exactly 1 extra
    try:
        frames = ip.build_image_plan(pass1, {"f1": lat}, {}, scene, {}, v4_meta_by_ref=v4_meta)
    finally:
        ip.FRAME_BUDGET_PER_CLIP = orig_budget
    assert len(frames) == 4, frames
    reasons = sorted(f.reason for f in frames)
    # 1 take (x2, early+late) + 1 speech (mandatory only) + 1 video
    # (mandatory only) -- the drift point lost out to the take's 2nd-moment
    # frame.
    assert reasons == [ip.REASON_SPEECH_CUT, ip.REASON_TAKE_MEMBER,
                       ip.REASON_TAKE_MEMBER, ip.REASON_VIDEO_GROUP_CALM], frames
    takes = [f for f in frames if f.reason == ip.REASON_TAKE_MEMBER]
    assert {f.phase for f in takes} == {"early", "late"}, takes
    # speech/video kept only their mandatory frame -- and since their would-be
    # 2nd-moment partner never survived the budget, they're relabeled "only"
    # (never a dangling "early" with no "late" to pair with).
    non_take = [f for f in frames if f.reason != ip.REASON_TAKE_MEMBER]
    assert all(f.phase == "only" for f in non_take), non_take
    print("ok  test_budget_truncation_second_moment_outranks_drift")


def test_budget_truncation_drift_wins_once_second_moment_is_covered():
    # Same fixture, but with enough budget to satisfy every mandatory AND
    # 2nd-moment candidate -- confirms drift (the lowest tier) still gets
    # the leftover slot once nothing higher-priority is competing for it.
    lat = _lattice_with_atoms("f1", 5, [])
    pass1 = Pass1Output(
        take_candidates=[TakeCandidate(group_id="tg1", members=[TakeMember(file_id="f1", word_span=(0, 1))])],
        speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 4), label="x")],
        video_tentative_groups=[VideoTentativeGroup(file_id="f1")],
    )
    v4_meta = {"video_group[0]": {"src_in_ms": 3000, "src_out_ms": 6000}}
    scene = {"f1": {"composition_points": [{"ts_ms": 700}]}}
    orig_budget = ip.FRAME_BUDGET_PER_CLIP
    # 3 mandatory + 3 second-moment (take/speech/video each get a "late") + 1.
    ip.FRAME_BUDGET_PER_CLIP = 7
    try:
        frames = ip.build_image_plan(pass1, {"f1": lat}, {}, scene, {}, v4_meta_by_ref=v4_meta)
    finally:
        ip.FRAME_BUDGET_PER_CLIP = orig_budget
    assert len(frames) == 7, frames
    video_frames = [f for f in frames if f.ref == "video_group[0]"]
    assert {f.phase for f in video_frames} == {"early", "late"}, video_frames
    assert any(f.reason == ip.REASON_COMPOSITION_DRIFT for f in frames), frames
    print("ok  test_budget_truncation_drift_wins_once_second_moment_is_covered")


def test_long_speech_cut_gets_early_and_late_frames():
    # A long span relative to a short baseline unit in the same clip, with
    # real motion data (hop_ms>0) so the runt guard's distribution check is
    # meaningfully exercised (not vacuously true the way hop_ms=0 makes it).
    lat = _lattice_with_atoms("f1", 20, [])
    short = SpeechCut(file_id="f1", word_span=(0, 1), label="short")   # ~700ms
    long_cut = SpeechCut(file_id="f1", word_span=(2, 15), label="long")  # much bigger
    pass1 = Pass1Output(speech_cuts=[short, long_cut])
    hop = 50
    n = 400
    blur = [0.9] * n
    blur[20] = 0.05    # sharp sample early in the long cut's span
    blur[200] = 0.05   # sharp sample late in the long cut's span
    motion = {"f1": {"hop_ms": hop, "blur": blur, "action_energy": [0.0] * n}}
    frames = ip.build_image_plan(pass1, {"f1": lat}, motion, {}, {})
    long_frames = [f for f in frames if f.ref == "speech_cut[1]"]
    assert len(long_frames) == 2, long_frames
    assert {f.phase for f in long_frames} == {"early", "late"}, long_frames
    early = next(f for f in long_frames if f.phase == "early")
    late = next(f for f in long_frames if f.phase == "late")
    assert early.ts_ms < late.ts_ms, (early, late)
    print("ok  test_long_speech_cut_gets_early_and_late_frames")


def test_runt_speech_cut_stays_single_frame():
    # Two units of similar (short) size in the same clip -- neither is short
    # relative to the OTHER, so this alone doesn't prove the median check;
    # combined with a real hop_ms it confirms two near-identical candidate
    # instants collapse via the proximity guard, not just default to 2 frames.
    lat = _lattice_with_atoms("f1", 3, [])
    pass1 = Pass1Output(speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 1), label="x")])
    hop = 500   # coarse hop relative to the ~700ms span -> few/no distinct samples
    motion = {"f1": {"hop_ms": hop, "blur": [0.5, 0.5], "action_energy": [0.0, 0.0]}}
    frames = ip.build_image_plan(pass1, {"f1": lat}, motion, {}, {})
    assert len(frames) == 1, frames
    assert frames[0].phase == "only", frames[0]
    print("ok  test_runt_speech_cut_stays_single_frame")



def test_v4_point_salience_straddles_the_peak():
    atoms = [_atom(0, 0, 5000, anchors=[])]
    lat = _lattice_with_atoms("f1", 0, atoms)
    pass1 = Pass1Output(video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[0])])
    v4_meta = {"video_group[0]": {"src_in_ms": 1000, "src_out_ms": 3000,
                                  "salience": {"peak_ms": 2000, "score": 0.9, "kind": "point", "span_ms": None}}}
    frames = ip.build_image_plan(pass1, {"f1": lat}, {}, {}, {}, v4_meta_by_ref=v4_meta)
    straddle = sorted(frames, key=lambda f: f.ts_ms)
    assert len(straddle) == 2, frames
    assert all(f.reason == ip.REASON_SHAPE_STRADDLE for f in straddle), frames
    assert straddle[0].phase == "early" and straddle[1].phase == "late", straddle
    assert straddle[0].ts_ms < 2000 < straddle[1].ts_ms, straddle
    # Frames stay inside the segmenter's own tight span, not the atom's [0,5000).
    assert 1000 <= straddle[0].ts_ms and straddle[1].ts_ms <= 3000, straddle
    print("ok  test_v4_point_salience_straddles_the_peak")


def test_v4_multi_event_cluster_yields_at_most_two_frames():
    """v4_cluster_read_act.plan.md Part A: a multi-event cluster is framed
    exactly like a single cut -- NOT one frame per event. Per-piece info the
    brain reads is 100% code-derived (salience.events), so no extra pixels
    are needed regardless of how many events a cluster holds."""
    atoms = [_atom(0, 0, 10000, anchors=[])]
    lat = _lattice_with_atoms("f1", 0, atoms)
    pass1 = Pass1Output(video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[0])])
    events = [
        {"peak_ms": 1000, "score": 0.5, "kind": "point", "onset_ms": 700, "settle_ms": 1300, "span_ms": None},
        {"peak_ms": 5000, "score": 1.0, "kind": "span", "onset_ms": 4600, "settle_ms": 5400,
         "span_ms": [4600, 5400]},
        {"peak_ms": 9000, "score": 0.4, "kind": "point", "onset_ms": 8700, "settle_ms": 9400, "span_ms": None},
    ]
    # primary is the span event -> top-level kind="span", falls through to the
    # ordinary unanchored (early/late) branch, not the point straddle.
    v4_meta = {"video_group[0]": {"src_in_ms": 0, "src_out_ms": 10000,
                                  "salience": {"peak_ms": 5000, "score": 1.0, "kind": "span",
                                             "span_ms": [4600, 5400], "events": events, "primary": 1}}}
    frames = ip.build_image_plan(pass1, {"f1": lat}, {}, {}, {}, v4_meta_by_ref=v4_meta)
    assert len(frames) <= 2, frames
    print("ok  test_v4_multi_event_cluster_yields_at_most_two_frames")


def test_v4_multi_event_cluster_with_point_primary_keeps_the_straddle():
    """A multi-event cluster whose PRIMARY event is a point still gets the
    ordinary peak-straddle treatment (2 frames around the primary's own
    peak) -- unaffected by however many other events the cluster holds."""
    atoms = [_atom(0, 0, 10000, anchors=[])]
    lat = _lattice_with_atoms("f1", 0, atoms)
    pass1 = Pass1Output(video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[0])])
    events = [
        {"peak_ms": 1000, "score": 0.5, "kind": "point", "onset_ms": 700, "settle_ms": 1300, "span_ms": None},
        {"peak_ms": 5000, "score": 1.0, "kind": "point", "onset_ms": 4600, "settle_ms": 5400, "span_ms": None},
        {"peak_ms": 9000, "score": 0.4, "kind": "point", "onset_ms": 8700, "settle_ms": 9400, "span_ms": None},
    ]
    v4_meta = {"video_group[0]": {"src_in_ms": 0, "src_out_ms": 10000,
                                  "salience": {"peak_ms": 5000, "score": 1.0, "kind": "point",
                                             "span_ms": None, "events": events, "primary": 1}}}
    frames = ip.build_image_plan(pass1, {"f1": lat}, {}, {}, {}, v4_meta_by_ref=v4_meta)
    assert len(frames) == 2, frames
    assert all(f.reason == ip.REASON_SHAPE_STRADDLE for f in frames), frames
    ts = sorted(f.ts_ms for f in frames)
    assert ts[0] < 5000 < ts[1], ts
    print("ok  test_v4_multi_event_cluster_with_point_primary_keeps_the_straddle")


def test_v4_none_kind_uses_segmenter_span_not_atom_bounds():
    """A V4 cut's frame timestamps must fall inside the segmenter's own
    (tighter) span, never the wider atom-membership bounding box the ref's
    atom_ids happen to resolve to."""
    atoms = [_atom(0, 0, 10000, anchors=[])]
    lat = _lattice_with_atoms("f1", 0, atoms)
    pass1 = Pass1Output(video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[0])])
    v4_meta = {"video_group[0]": {"src_in_ms": 4000, "src_out_ms": 4800,
                                  "salience": {"peak_ms": 4400, "score": 0.0, "kind": "none", "span_ms": None}}}
    frames = ip.build_image_plan(pass1, {"f1": lat}, {}, {}, {}, v4_meta_by_ref=v4_meta)
    assert frames, "must still get at least one frame"
    for f in frames:
        assert 4000 <= f.ts_ms <= 4800, f
    print("ok  test_v4_none_kind_uses_segmenter_span_not_atom_bounds")


def test_unknown_file_id_skipped_gracefully():
    pass1 = Pass1Output(
        speech_cuts=[SpeechCut(file_id="missing", word_span=(0, 1), label="x")],
        video_tentative_groups=[VideoTentativeGroup(file_id="missing", atom_ids=[0])],
        take_candidates=[TakeCandidate(group_id="tg1", members=[TakeMember(file_id="missing", word_span=(0, 1))])],
    )
    frames = ip.build_image_plan(pass1, {}, {}, {}, {})
    assert frames == [], frames
    print("ok  test_unknown_file_id_skipped_gracefully")


def test_no_pass1_content_yields_no_frames():
    lat = _lattice_with_atoms("f1", 3, [])
    frames = ip.build_image_plan(Pass1Output(), {"f1": lat}, {}, {}, {})
    assert frames == []
    print("ok  test_no_pass1_content_yields_no_frames")


def test_planned_frame_to_dict():
    f = ip.PlannedFrame("f1", 123, ip.REASON_SPEECH_CUT, "speech_cut[0]")
    assert f.to_dict() == {"file_id": "f1", "ts_ms": 123, "reason": "speech_cut",
                           "ref": "speech_cut[0]", "phase": "only"}
    f2 = ip.PlannedFrame("f1", 123, ip.REASON_SPEECH_CUT, "speech_cut[0]", "early")
    assert f2.to_dict()["phase"] == "early"
    print("ok  test_planned_frame_to_dict")


def main():
    test_speech_cut_frame_uses_sharpest_ms_in_span()
    test_composition_drift_extra_added_inside_span_only()
    test_every_pass1_unit_gets_a_frame_even_over_budget()
    test_video_group_unanchored_uses_calm_and_sharp_fallback()
    test_video_group_with_no_v4_meta_is_skipped()
    test_take_member_always_kept_even_at_tiny_budget()
    test_budget_truncation_second_moment_outranks_drift()
    test_budget_truncation_drift_wins_once_second_moment_is_covered()
    test_long_speech_cut_gets_early_and_late_frames()
    test_runt_speech_cut_stays_single_frame()
    test_v4_point_salience_straddles_the_peak()
    test_v4_multi_event_cluster_yields_at_most_two_frames()
    test_v4_multi_event_cluster_with_point_primary_keeps_the_straddle()
    test_v4_none_kind_uses_segmenter_span_not_atom_bounds()
    test_unknown_file_id_skipped_gracefully()
    test_no_pass1_content_yields_no_frames()
    test_planned_frame_to_dict()
    print("\nall image-plan tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
