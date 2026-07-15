"""
Tests for voice->face binding via Active Speaker Detection
(app.services.l3.identity.bind_asd) -- no DB, no network, no model call,
synthetic turns/tracks only.

Run:  .venv/bin/python scripts/test_identity_bind_asd.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l1.active_speaker import FaceTrack, SpeakingInterval  # noqa: E402
from app.services.l3.identity import bind_asd as ba  # noqa: E402


def _track(track_id, speaking):
    return FaceTrack(track_id=track_id, speaking=[SpeakingInterval(start_ms=s, end_ms=e, score=1.0)
                                                   for s, e in speaking])


# --------------------------------------------------------------------------
# _overlap_ms / _speaking_overlap_ms
# --------------------------------------------------------------------------

def test_overlap_ms_partial_overlap():
    assert ba._overlap_ms(0, 1000, 500, 1500) == 500
    print("ok  test_overlap_ms_partial_overlap")


def test_overlap_ms_no_overlap_is_zero():
    assert ba._overlap_ms(0, 100, 200, 300) == 0
    print("ok  test_overlap_ms_no_overlap_is_zero")


def test_speaking_overlap_ms_sums_across_intervals():
    track = _track(0, [(0, 200), (500, 800)])
    assert ba._speaking_overlap_ms(track, 0, 1000) == 200 + 300
    print("ok  test_speaking_overlap_ms_sums_across_intervals")


# --------------------------------------------------------------------------
# _local_speaker_owner
# --------------------------------------------------------------------------

def test_local_speaker_owner_picks_the_most_overlapping_track():
    tracks = {"f1": [_track(0, [(0, 900)]), _track(1, [(0, 100)])]}
    track_to_person = {("f1", 0): "P0", ("f1", 1): "P1"}
    owner = ba._local_speaker_owner("f1", [(0, 1000)], tracks, track_to_person)
    assert owner == "P0", owner
    print("ok  test_local_speaker_owner_picks_the_most_overlapping_track")


def test_local_speaker_owner_sums_overlap_across_multiple_turns():
    # Track 1 wins on total overlap across BOTH turns even though track 0
    # wins the first turn alone.
    tracks = {"f1": [_track(0, [(0, 600)]), _track(1, [(0, 400), (1000, 1900)])]}
    track_to_person = {("f1", 0): "P0", ("f1", 1): "P1"}
    owner = ba._local_speaker_owner("f1", [(0, 1000), (1000, 2000)], tracks, track_to_person)
    assert owner == "P1", owner
    print("ok  test_local_speaker_owner_sums_overlap_across_multiple_turns")


def test_local_speaker_owner_none_when_no_track_overlaps():
    tracks = {"f1": [_track(0, [(5000, 6000)])]}
    track_to_person = {("f1", 0): "P0"}
    owner = ba._local_speaker_owner("f1", [(0, 1000)], tracks, track_to_person)
    assert owner is None, owner
    print("ok  test_local_speaker_owner_none_when_no_track_overlaps")


def test_local_speaker_owner_none_for_untracked_file():
    owner = ba._local_speaker_owner("f-missing", [(0, 1000)], {}, {})
    assert owner is None, owner
    print("ok  test_local_speaker_owner_none_for_untracked_file")


# --------------------------------------------------------------------------
# bind (end to end)
# --------------------------------------------------------------------------

def test_bind_binds_two_clean_single_camera_voices():
    turns_by_file = {"f1": [(0, 1000, "S0"), (2000, 3000, "S1")]}
    voice_of = {("f1", "S0"): "V0", ("f1", "S1"): "V1"}
    face_tracks_by_file = {"f1": [_track(0, [(0, 1000)]), _track(1, [(2000, 3000)])]}
    track_to_person = {("f1", 0): "P0", ("f1", 1): "P1"}

    owner_by_voice, off_camera = ba.bind(turns_by_file, voice_of, face_tracks_by_file, track_to_person)
    assert owner_by_voice == {"V0": "P0", "V1": "P1"}, owner_by_voice
    assert off_camera == set(), off_camera
    print("ok  test_bind_binds_two_clean_single_camera_voices")


def test_bind_unbound_when_a_voice_conflicts_across_files():
    # V0 = (f1, S0) says P0, but ALSO (f2, S0) says P1 -- a genuine
    # multicam disagreement, majority+margin refuses to guess.
    turns_by_file = {
        "f1": [(0, 1000, "S0")],
        "f2": [(0, 1000, "S0")],
    }
    voice_of = {("f1", "S0"): "V0", ("f2", "S0"): "V0"}
    face_tracks_by_file = {
        "f1": [_track(0, [(0, 1000)])],
        "f2": [_track(0, [(0, 1000)])],
    }
    track_to_person = {("f1", 0): "P0", ("f2", 0): "P1"}

    owner_by_voice, off_camera = ba.bind(turns_by_file, voice_of, face_tracks_by_file, track_to_person)
    assert owner_by_voice == {"V0": None}, owner_by_voice
    assert off_camera == {"V0"}, off_camera
    print("ok  test_bind_unbound_when_a_voice_conflicts_across_files")


def test_bind_narrator_voice_has_no_vote_at_all():
    turns_by_file = {"f1": [(0, 1000, "S0")]}
    voice_of = {("f1", "S0"): "V0"}
    face_tracks_by_file = {"f1": []}   # no faces at all in this file
    owner_by_voice, off_camera = ba.bind(turns_by_file, voice_of, face_tracks_by_file, {})
    assert owner_by_voice == {}, owner_by_voice
    assert off_camera == set(), off_camera
    print("ok  test_bind_narrator_voice_has_no_vote_at_all")


def test_bind_ignores_turns_with_no_voice_mapping():
    turns_by_file = {"f1": [(0, 1000, "S0"), (2000, 3000, None)]}
    voice_of = {("f1", "S0"): "V0"}
    face_tracks_by_file = {"f1": [_track(0, [(0, 1000)])]}
    track_to_person = {("f1", 0): "P0"}
    owner_by_voice, _off_camera = ba.bind(turns_by_file, voice_of, face_tracks_by_file, track_to_person)
    assert owner_by_voice == {"V0": "P0"}, owner_by_voice
    print("ok  test_bind_ignores_turns_with_no_voice_mapping")


def test_bind_majority_wins_when_one_file_disagrees_among_several():
    # V0 = 3 (file, speaker) pairs voting P0, one voting P1 -- clear
    # majority + margin, binds P0.
    turns_by_file = {f"f{i}": [(0, 1000, "S0")] for i in range(4)}
    voice_of = {(f"f{i}", "S0"): "V0" for i in range(4)}
    face_tracks_by_file = {f"f{i}": [_track(0, [(0, 1000)])] for i in range(4)}
    track_to_person = {(f"f{i}", 0): ("P0" if i < 3 else "P1") for i in range(4)}

    owner_by_voice, off_camera = ba.bind(turns_by_file, voice_of, face_tracks_by_file, track_to_person)
    assert owner_by_voice == {"V0": "P0"}, owner_by_voice
    assert off_camera == set(), off_camera
    print("ok  test_bind_majority_wins_when_one_file_disagrees_among_several")


def main():
    test_overlap_ms_partial_overlap()
    test_overlap_ms_no_overlap_is_zero()
    test_speaking_overlap_ms_sums_across_intervals()
    test_local_speaker_owner_picks_the_most_overlapping_track()
    test_local_speaker_owner_sums_overlap_across_multiple_turns()
    test_local_speaker_owner_none_when_no_track_overlaps()
    test_local_speaker_owner_none_for_untracked_file()
    test_bind_binds_two_clean_single_camera_voices()
    test_bind_unbound_when_a_voice_conflicts_across_files()
    test_bind_narrator_voice_has_no_vote_at_all()
    test_bind_ignores_turns_with_no_voice_mapping()
    test_bind_majority_wins_when_one_file_disagrees_among_several()
    print("\nall identity-bind-asd tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
