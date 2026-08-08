"""
Pure unit tests for app.services.vcut.speech.outlooks (speech_cuts_pipeline
.plan.md section 7). load_sync_groups itself needs a live DB and is
exercised via the live smoke run instead (this codebase's convention).

Run:  .venv/bin/python scripts/test_speech_outlooks.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.speech.outlooks import (  # noqa: E402
    SyncGroup, SyncMember, angle_ms, collapse_to_take_instances, group_for_file,
    video_angle_files,
)


def _group(group_id, auth_id, members_spec):
    """members_spec: {file_id: (offset_ms, role, confidence)}."""
    g = SyncGroup(group_id=group_id, authoritative_file_id=auth_id)
    for fid, (off, role, conf) in members_spec.items():
        g.members[fid] = SyncMember(file_id=fid, offset_ms=off, role=role, confidence=conf)
    return g


# --------------------------------------------------------------------------
# angle_ms -- the offset formula
# --------------------------------------------------------------------------

def test_angle_ms_zero_offset_both_sides_is_identity():
    assert angle_ms(auth_ms=5000, auth_offset_ms=0, angle_offset_ms=0) == 5000
    print("ok  test_angle_ms_zero_offset_both_sides_is_identity")


def test_angle_ms_applies_relative_offset():
    # auth started 200ms into the group clock; angle started 500ms in ->
    # angle is 300ms BEHIND auth on its own timeline.
    result = angle_ms(auth_ms=5000, auth_offset_ms=200, angle_offset_ms=500)
    assert result == 5000 + 200 - 500 == 4700, result
    print("ok  test_angle_ms_applies_relative_offset")


def test_angle_ms_is_symmetric_round_trip():
    # mapping auth->angle and back with swapped offsets recovers auth_ms.
    a = angle_ms(auth_ms=8000, auth_offset_ms=100, angle_offset_ms=400)
    back = angle_ms(auth_ms=a, auth_offset_ms=400, angle_offset_ms=100)
    assert back == 8000, back
    print("ok  test_angle_ms_is_symmetric_round_trip")


# --------------------------------------------------------------------------
# collapse_to_take_instances
# --------------------------------------------------------------------------

def test_ungrouped_file_is_its_own_take_instance():
    out = collapse_to_take_instances(["f1", "f2"], {})
    assert out == ["f1", "f2"], out
    print("ok  test_ungrouped_file_is_its_own_take_instance")


def test_sync_group_collapses_to_just_the_authoritative_file():
    groups = {"g1": _group("g1", "f1", {
        "f1": (0, "video_angle", None), "f2": (300, "video_angle", None),
    })}
    out = collapse_to_take_instances(["f1", "f2"], groups)
    assert out == ["f1"], out
    print("ok  test_sync_group_collapses_to_just_the_authoritative_file")


def test_mixed_grouped_and_ungrouped_files():
    groups = {"g1": _group("g1", "f2", {
        "f2": (0, "video_angle", None), "f3": (150, "video_angle", None),
    })}
    out = collapse_to_take_instances(["f1", "f2", "f3", "f4"], groups)
    assert out == ["f1", "f4", "f2"], out
    print("ok  test_mixed_grouped_and_ungrouped_files")


def test_authoritative_file_not_duplicated_if_already_ungrouped_somehow():
    # defensive: authoritative id appearing in file_ids AND already grouped
    # (normal case) never produces a duplicate entry.
    groups = {"g1": _group("g1", "f1", {"f1": (0, "video_angle", None), "f2": (0, "video_angle", None)})}
    out = collapse_to_take_instances(["f1", "f2"], groups)
    assert out.count("f1") == 1
    print("ok  test_authoritative_file_not_duplicated_if_already_ungrouped_somehow")


# --------------------------------------------------------------------------
# video_angle_files / group_for_file
# --------------------------------------------------------------------------

def test_video_angle_files_excludes_pure_audio_members():
    g = _group("g1", "f1", {
        "f1": (0, "video_angle", None), "f2": (100, "video_angle", None),
        "boom": (50, "audio", 0.9),
    })
    out = sorted(video_angle_files(g))
    assert out == ["f1", "f2"], out
    print("ok  test_video_angle_files_excludes_pure_audio_members")


def test_group_for_file_finds_the_right_group():
    groups = {
        "g1": _group("g1", "f1", {"f1": (0, "video_angle", None)}),
        "g2": _group("g2", "f9", {"f9": (0, "video_angle", None), "f10": (0, "video_angle", None)}),
    }
    assert group_for_file(groups, "f10").group_id == "g2"
    assert group_for_file(groups, "unknown") is None
    print("ok  test_group_for_file_finds_the_right_group")


def main():
    test_angle_ms_zero_offset_both_sides_is_identity()
    test_angle_ms_applies_relative_offset()
    test_angle_ms_is_symmetric_round_trip()
    test_ungrouped_file_is_its_own_take_instance()
    test_sync_group_collapses_to_just_the_authoritative_file()
    test_mixed_grouped_and_ungrouped_files()
    test_authoritative_file_not_duplicated_if_already_ungrouped_somehow()
    test_video_angle_files_excludes_pure_audio_members()
    test_group_for_file_finds_the_right_group()
    print("\nall speech outlooks tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
