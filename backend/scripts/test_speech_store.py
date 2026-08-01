"""
Pure unit tests for app.services.vcut.speech.store (speech_cuts_pipeline
.plan.md section 12: CutRecord assembly + outlook fan-out). No I/O --
SyncGroup/ResolvedBeat/FaceTrackLite are all plain dataclasses.

Run:  .venv/bin/python scripts/test_speech_store.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.speech.inputs import FaceTrackLite  # noqa: E402
from app.services.vcut.speech.outlooks import SyncGroup, SyncMember  # noqa: E402
from app.services.vcut.speech.segment_llm import _BeatOut  # noqa: E402
from app.services.vcut.speech.store import (  # noqa: E402
    ResolvedBeat, VisualFields, _build_record, build_speech_cut_records, cut_key,
    hero_ts_for_span, is_on_camera,
)


def _beat(id_, file_id, span=(0, 3), gist="hello there"):
    return _BeatOut(id=id_, file_id=file_id, word_span=span, gist=gist)


def _rb(beat, in_ms=1000, out_ms=3000, take_group_key="tg0", take_role="winner", quality=0.8):
    return ResolvedBeat(beat=beat, in_ms=in_ms, out_ms=out_ms, take_group_key=take_group_key,
                        take_role=take_role, speech_quality=quality)


def _group(group_id, auth_id, members_spec):
    g = SyncGroup(group_id=group_id, authoritative_file_id=auth_id)
    for fid, (off, role, conf) in members_spec.items():
        g.members[fid] = SyncMember(file_id=fid, offset_ms=off, role=role, confidence=conf)
    return g


# --------------------------------------------------------------------------
# hero_ts_for_span / is_on_camera
# --------------------------------------------------------------------------

def test_hero_ts_picks_the_highest_scoring_overlapping_speaking_interval():
    tracks = [
        FaceTrackLite(track_id=0, best_crop_ms=1500, speaking=[(1000, 2000, 0.4)]),
        FaceTrackLite(track_id=1, best_crop_ms=2500, speaking=[(2000, 3000, 0.9)]),
    ]
    ts = hero_ts_for_span(tracks, 1000, 3000)
    assert ts == 2500, ts
    print("ok  test_hero_ts_picks_the_highest_scoring_overlapping_speaking_interval")


def test_hero_ts_falls_back_to_midpoint_with_no_overlap():
    tracks = [FaceTrackLite(track_id=0, best_crop_ms=9000, speaking=[(9000, 9500, 0.9)])]
    ts = hero_ts_for_span(tracks, 1000, 3000)
    assert ts == 2000, ts  # midpoint of [1000,3000]
    print("ok  test_hero_ts_falls_back_to_midpoint_with_no_overlap")


def test_hero_ts_falls_back_to_midpoint_when_best_crop_lands_outside_the_span():
    tracks = [FaceTrackLite(track_id=0, best_crop_ms=50, speaking=[(1000, 3000, 0.9)])]
    ts = hero_ts_for_span(tracks, 1000, 3000)
    assert ts == 2000, ts
    print("ok  test_hero_ts_falls_back_to_midpoint_when_best_crop_lands_outside_the_span")


def test_is_on_camera_true_with_overlapping_speaking_interval():
    tracks = [FaceTrackLite(track_id=0, best_crop_ms=1500, speaking=[(1000, 2000, 0.5)])]
    assert is_on_camera(tracks, 1000, 3000) is True
    print("ok  test_is_on_camera_true_with_overlapping_speaking_interval")


def test_is_on_camera_false_with_no_face_tracks_or_no_overlap():
    assert is_on_camera([], 1000, 3000) is False
    tracks = [FaceTrackLite(track_id=0, best_crop_ms=9000, speaking=[(9000, 9500, 0.9)])]
    assert is_on_camera(tracks, 1000, 3000) is False
    print("ok  test_is_on_camera_false_with_no_face_tracks_or_no_overlap")


# --------------------------------------------------------------------------
# build_speech_cut_records -- fan-out + audio routing
# --------------------------------------------------------------------------

def test_ungrouped_winner_is_a_single_cut_with_no_audio_routing():
    rb = _rb(_beat("b0", "f1"))
    records, specifics = build_speech_cut_records([rb], {}, {}, {})
    assert len(records) == 1
    r = records[0]
    assert r.file_id == "f1" and r.sync_group_id is None
    assert r.audio_file_id == "" and r.audio_offset_ms == 0
    assert r.kind == "speech" and r.channel == "said"
    print("ok  test_ungrouped_winner_is_a_single_cut_with_no_audio_routing")


def test_winning_take_fans_out_to_every_video_angle():
    group = _group("g1", "hero", {
        "hero": (0, "video_angle", None), "angle2": (200, "video_angle", 0.8),
        "boom": (0, "audio", 0.95),
    })
    rb = _rb(_beat("b0", "hero"), in_ms=5000, out_ms=7000)
    records, _specifics = build_speech_cut_records([rb], {"g1": group}, {}, {})
    file_ids = sorted(r.file_id for r in records)
    assert file_ids == ["angle2", "hero"], file_ids  # "boom" excluded -- audio-only, no picture
    for r in records:
        assert r.sync_group_id == "g1"
        assert r.audio_file_id == "hero"
        # Every fanned angle is an equal outlook of the same audio -- never
        # "winner" -- so "Best takes" collapses only real retakes, never angles.
        assert r.take_role == "outlook", r.take_role
    print("ok  test_winning_take_fans_out_to_every_video_angle")


def test_angle_offset_applied_correctly_to_non_hero_cuts():
    group = _group("g1", "hero", {"hero": (100, "video_angle", None), "angle2": (400, "video_angle", 0.7)})
    rb = _rb(_beat("b0", "hero"), in_ms=5000, out_ms=7000)
    records, _s = build_speech_cut_records([rb], {"g1": group}, {}, {})
    by_file = {r.file_id: r for r in records}
    # angle_ms(5000, hero_offset=100, angle_offset=400) = 5000+100-400 = 4700
    assert by_file["angle2"].src_in_ms == 4700, by_file["angle2"].src_in_ms
    assert by_file["angle2"].src_out_ms == 6700, by_file["angle2"].src_out_ms
    assert by_file["angle2"].audio_offset_ms == 400 - 100 == 300
    assert by_file["angle2"].audio_align_confidence == 0.7
    assert by_file["hero"].src_in_ms == 5000  # identity: hero offset relative to itself is 0
    print("ok  test_angle_offset_applied_correctly_to_non_hero_cuts")


def test_alternate_never_fans_out_even_inside_a_sync_group():
    group = _group("g1", "hero", {"hero": (0, "video_angle", None), "angle2": (0, "video_angle", None)})
    rb = _rb(_beat("b1", "hero"), take_role="take")
    records, _s = build_speech_cut_records([rb], {"g1": group}, {}, {})
    assert len(records) == 1
    assert records[0].file_id == "hero"
    assert records[0].sync_group_id is None
    assert records[0].take_role == "take"
    print("ok  test_alternate_never_fans_out_even_inside_a_sync_group")


def test_winner_on_a_non_authoritative_file_does_not_fan_out():
    # a winning beat whose file is a member of a group but NOT the hero --
    # shouldn't happen from a correctly-collapsed take instance, but must
    # degrade to a single cut rather than crash or double-fan-out.
    group = _group("g1", "hero", {"hero": (0, "video_angle", None), "angle2": (0, "video_angle", None)})
    rb = _rb(_beat("b0", "angle2"))
    records, _s = build_speech_cut_records([rb], {"g1": group}, {}, {})
    assert len(records) == 1 and records[0].file_id == "angle2"
    print("ok  test_winner_on_a_non_authoritative_file_does_not_fan_out")


def test_visual_fields_attach_via_cut_key_scene_specifics_carries_angle_type():
    group = _group("g1", "hero", {"hero": (0, "video_angle", None), "angle2": (0, "video_angle", None)})
    rb = _rb(_beat("b0", "hero"))
    visual = {
        cut_key("b0", "hero"): VisualFields(on_camera=True, angle_type="medium",
                                            scene_specifics={"subject": "a person"}),
    }
    records, specifics = build_speech_cut_records([rb], {"g1": group}, {}, visual)
    by_file = {r.file_id: (r, s) for r, s in zip(records, specifics)}
    hero_record, hero_spec = by_file["hero"]
    assert hero_record.on_camera is True
    assert hero_spec == {"subject": "a person", "angle_type": "medium"}, hero_spec
    angle_record, angle_spec = by_file["angle2"]
    assert angle_record.on_camera is None  # no visual entry for this angle -> defaults
    assert angle_spec is None
    print("ok  test_visual_fields_attach_via_cut_key_scene_specifics_carries_angle_type")


def test_speech_quality_and_take_fields_carried_through():
    rb = _rb(_beat("b0", "f1"), take_group_key="tg-xyz", take_role="winner", quality=0.73)
    records, _s = build_speech_cut_records([rb], {}, {}, {})
    r = records[0]
    assert r.take_group_id == "tg-xyz"
    assert r.take_role == "winner"
    assert abs(r.speech_quality - 0.73) < 1e-9
    assert abs(r.total_quality - 0.73) < 1e-9
    print("ok  test_speech_quality_and_take_fields_carried_through")


def test_pace_envelope_never_speed_ramps():
    rb = _rb(_beat("b0", "f1"), in_ms=1000, out_ms=4000)
    records, _s = build_speech_cut_records([rb], {}, {}, {})
    r = records[0]
    assert r.pace.min_ms == r.pace.natural_ms == r.pace.max_ms == 3000
    print("ok  test_pace_envelope_never_speed_ramps")


# --------------------------------------------------------------------------
# junk threading -- speech_noise_gate.plan.md Improvement C: _build_record no
# longer hardcodes junk=False; it threads whatever the caller passes through
# onto the CutRecord.
# --------------------------------------------------------------------------

def test_build_record_default_call_is_not_junk():
    rb = _rb(_beat("b0", "f1"))
    record, _spec = _build_record(
        "f1", 1000, 3000, rb, [], None,
        sync_group_id=None, audio_file_id="", audio_offset_ms=0, audio_align_confidence=None,
        take_role="winner",
    )
    assert record.junk is False
    assert record.junk_reason == ""
    print("ok  test_build_record_default_call_is_not_junk")


def test_build_record_threads_junk_flag_and_reason_onto_the_cut_record():
    rb = _rb(_beat("b0", "f1"), take_role="take", quality=0.0)
    record, _spec = _build_record(
        "f1", 1000, 3000, rb, [], None,
        sync_group_id=None, audio_file_id="", audio_offset_ms=0, audio_align_confidence=None,
        take_role="take", junk=True, junk_reason="non_speech_energy",
    )
    assert record.junk is True
    assert record.junk_reason == "non_speech_energy"
    print("ok  test_build_record_threads_junk_flag_and_reason_onto_the_cut_record")


def test_build_speech_cut_records_carries_junk_through_from_resolved_beat():
    rb = ResolvedBeat(beat=_beat("b0", "f1"), in_ms=1000, out_ms=3000, take_group_key="tg0",
                      take_role="take", speech_quality=0.0, junk=True, junk_reason="non_speech_llm")
    records, _s = build_speech_cut_records([rb], {}, {}, {})
    assert len(records) == 1
    assert records[0].junk is True
    assert records[0].junk_reason == "non_speech_llm"
    print("ok  test_build_speech_cut_records_carries_junk_through_from_resolved_beat")


def test_resolved_beat_defaults_to_not_junk():
    rb = _rb(_beat("b0", "f1"))
    assert rb.junk is False
    assert rb.junk_reason == ""
    print("ok  test_resolved_beat_defaults_to_not_junk")


def main():
    test_hero_ts_picks_the_highest_scoring_overlapping_speaking_interval()
    test_hero_ts_falls_back_to_midpoint_with_no_overlap()
    test_hero_ts_falls_back_to_midpoint_when_best_crop_lands_outside_the_span()
    test_is_on_camera_true_with_overlapping_speaking_interval()
    test_is_on_camera_false_with_no_face_tracks_or_no_overlap()
    test_ungrouped_winner_is_a_single_cut_with_no_audio_routing()
    test_winning_take_fans_out_to_every_video_angle()
    test_angle_offset_applied_correctly_to_non_hero_cuts()
    test_alternate_never_fans_out_even_inside_a_sync_group()
    test_winner_on_a_non_authoritative_file_does_not_fan_out()
    test_visual_fields_attach_via_cut_key_scene_specifics_carries_angle_type()
    test_speech_quality_and_take_fields_carried_through()
    test_pace_envelope_never_speed_ramps()
    test_build_record_default_call_is_not_junk()
    test_build_record_threads_junk_flag_and_reason_onto_the_cut_record()
    test_build_speech_cut_records_carries_junk_through_from_resolved_beat()
    test_resolved_beat_defaults_to_not_junk()
    print("\nall speech store tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
