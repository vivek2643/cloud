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
                action_energy=0.1, camera_desc="hold", coherence=0.9, anchor_ms=anchors or [])


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


def test_video_group_anchor_yields_one_frame_per_anchor():
    atoms = [_atom(0, 0, 1000, anchors=[200, 600]), _atom(1, 1000, 2000, anchors=[1500])]
    lat = _lattice_with_atoms("f1", 0, atoms)
    pass1 = Pass1Output(video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[0, 1])])
    frames = ip.build_image_plan(pass1, {"f1": lat}, {}, {}, {})
    anchor_frames = sorted(f.ts_ms for f in frames if f.reason == ip.REASON_VIDEO_GROUP_ANCHOR)
    assert anchor_frames == [200, 600, 1500], frames
    assert all(f.ref == "video_group[0]" for f in frames)
    print("ok  test_video_group_anchor_yields_one_frame_per_anchor")


def test_every_pass1_unit_gets_a_frame_even_over_budget():
    """One frame per unit is MANDATORY -- the budget can never starve a unit
    of its only frame (a cut pass 2 never saw pixels for can't be judged;
    observed as a real 'no images resolved' pass-2b failure)."""
    atoms = [_atom(i, i * 1000, (i + 1) * 1000) for i in range(6)]
    lat = _lattice_with_atoms("f1", 5, atoms)
    pass1 = Pass1Output(
        speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 4), label="x")],
        video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[i]) for i in range(6)],
    )
    orig_budget = ip.FRAME_BUDGET_PER_CLIP
    ip.FRAME_BUDGET_PER_CLIP = 2   # far fewer than the 7 units
    try:
        frames = ip.build_image_plan(pass1, {"f1": lat}, {}, {}, {})
    finally:
        ip.FRAME_BUDGET_PER_CLIP = orig_budget
    refs = {f.ref for f in frames}
    assert "speech_cut[0]" in refs
    for gi in range(6):
        assert f"video_group[{gi}]" in refs, (gi, refs)
    print("ok  test_every_pass1_unit_gets_a_frame_even_over_budget")


def test_video_group_unanchored_uses_calm_and_sharp_fallback():
    atoms = [_atom(0, 0, 1000, anchors=[])]
    lat = _lattice_with_atoms("f1", 0, atoms)
    pass1 = Pass1Output(video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[0])])
    hop = 100
    n = 10
    action = [0.8] * n
    blur = [0.8] * n
    action[4] = 0.05
    blur[4] = 0.05  # a clearly calmest+sharpest sample at i=4 -> ts=400
    motion = {"f1": {"hop_ms": hop, "action_energy": action, "blur": blur}}
    frames = ip.build_image_plan(pass1, {"f1": lat}, motion, {}, {})
    assert len(frames) == 1, frames
    assert frames[0].reason == ip.REASON_VIDEO_GROUP_CALM
    assert frames[0].ts_ms == 400, frames[0]
    print("ok  test_video_group_unanchored_uses_calm_and_sharp_fallback")


def test_video_group_with_no_resolvable_atoms_is_skipped():
    atoms = [_atom(0, 0, 1000)]
    lat = _lattice_with_atoms("f1", 0, atoms)
    pass1 = Pass1Output(video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[99])])
    frames = ip.build_image_plan(pass1, {"f1": lat}, {}, {}, {})
    assert frames == [], frames
    print("ok  test_video_group_with_no_resolvable_atoms_is_skipped")


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


def test_budget_truncation_drops_extras_not_mandatory_frames():
    # A group with 3 anchors + a drift point inside the speech cut: the
    # mandatory floor (take, speech cut, group's FIRST anchor) always ships;
    # the budget then trims extra anchors before drift, lowest tier first.
    atoms = [_atom(0, 3000, 6000, anchors=[3500, 4200, 5100])]
    lat = _lattice_with_atoms("f1", 5, atoms)
    pass1 = Pass1Output(
        take_candidates=[TakeCandidate(group_id="tg1", members=[TakeMember(file_id="f1", word_span=(0, 1))])],
        speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 4), label="x")],
        video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[0])],
    )
    scene = {"f1": {"composition_points": [{"ts_ms": 700}]}}
    orig_budget = ip.FRAME_BUDGET_PER_CLIP
    ip.FRAME_BUDGET_PER_CLIP = 4   # 3 mandatory + room for exactly 1 extra
    try:
        frames = ip.build_image_plan(pass1, {"f1": lat}, {}, scene, {})
    finally:
        ip.FRAME_BUDGET_PER_CLIP = orig_budget
    reasons = sorted(f.reason for f in frames)
    # 1 take + 1 speech + first anchor (mandatory) + 1 extra anchor; drift dropped.
    assert reasons == [ip.REASON_SPEECH_CUT, ip.REASON_TAKE_MEMBER,
                       ip.REASON_VIDEO_GROUP_ANCHOR, ip.REASON_VIDEO_GROUP_ANCHOR], frames
    anchor_ts = sorted(f.ts_ms for f in frames if f.reason == ip.REASON_VIDEO_GROUP_ANCHOR)
    assert anchor_ts == [3500, 4200], anchor_ts
    print("ok  test_budget_truncation_drops_extras_not_mandatory_frames")


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
    assert f.to_dict() == {"file_id": "f1", "ts_ms": 123, "reason": "speech_cut", "ref": "speech_cut[0]"}
    print("ok  test_planned_frame_to_dict")


def main():
    test_speech_cut_frame_uses_sharpest_ms_in_span()
    test_composition_drift_extra_added_inside_span_only()
    test_video_group_anchor_yields_one_frame_per_anchor()
    test_every_pass1_unit_gets_a_frame_even_over_budget()
    test_video_group_unanchored_uses_calm_and_sharp_fallback()
    test_video_group_with_no_resolvable_atoms_is_skipped()
    test_take_member_always_kept_even_at_tiny_budget()
    test_budget_truncation_drops_extras_not_mandatory_frames()
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
