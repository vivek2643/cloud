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
    # V0 = 3 files' overlap points at P0, one at P1 -- P0's cumulative overlap
    # (3000ms) clears the margin over P1 (1000ms), binds P0.
    turns_by_file = {f"f{i}": [(0, 1000, "S0")] for i in range(4)}
    voice_of = {(f"f{i}", "S0"): "V0" for i in range(4)}
    face_tracks_by_file = {f"f{i}": [_track(0, [(0, 1000)])] for i in range(4)}
    track_to_person = {(f"f{i}", 0): ("P0" if i < 3 else "P1") for i in range(4)}

    owner_by_voice, off_camera = ba.bind(turns_by_file, voice_of, face_tracks_by_file, track_to_person)
    assert owner_by_voice == {"V0": "P0"}, owner_by_voice
    assert off_camera == set(), off_camera
    print("ok  test_bind_majority_wins_when_one_file_disagrees_among_several")


def test_bind_magnitude_beats_equal_vote_tie_in_multicam_podcast():
    # The real 2-person, single-face-per-camera podcast failure mode. Two
    # cameras per person; each mic records BOTH voices, so each lone on-camera
    # face is ASD-"speaking" over its whole clip and thus overlaps BOTH voices'
    # turns. An equal per-file vote would tie every voice 2-2 (each voice draws
    # one "owner" vote from P0's cameras and one from P1's) -> everything
    # unbound. Magnitude aggregation instead sums overlap ms, so the true
    # speaker wins: P0 accrues 200s on V0 vs 100s on V1; P1 the mirror.
    def secs(*pairs):
        return [(s * 1000, e * 1000) for s, e in pairs]

    turns_by_file = {
        # P0's cameras: V0 (S0) dominates the on-camera speaking.
        "fA": [(0, 100_000, "S0"), (100_000, 190_000, "S1")],
        "fB": [(0, 100_000, "S0"), (100_000, 110_000, "S1")],
        # P1's cameras: V1 (S1) dominates.
        "fC": [(0, 90_000, "S0"), (90_000, 190_000, "S1")],
        "fD": [(0, 10_000, "S0"), (10_000, 110_000, "S1")],
    }
    voice_of = {(f, "S0"): "V0" for f in ("fA", "fB", "fC", "fD")}
    voice_of.update({(f, "S1"): "V1" for f in ("fA", "fB", "fC", "fD")})
    # One face per file, "speaking" over the WHOLE clip (mic carries both
    # voices) -- so it overlaps both voices' turns, the exact ambiguity.
    face_tracks_by_file = {
        "fA": [_track(0, secs((0, 190)))],
        "fB": [_track(0, secs((0, 110)))],
        "fC": [_track(0, secs((0, 190)))],
        "fD": [_track(0, secs((0, 110)))],
    }
    track_to_person = {
        ("fA", 0): "P0", ("fB", 0): "P0",
        ("fC", 0): "P1", ("fD", 0): "P1",
    }
    owner_by_voice, off_camera = ba.bind(turns_by_file, voice_of, face_tracks_by_file, track_to_person)
    assert owner_by_voice == {"V0": "P0", "V1": "P1"}, owner_by_voice
    assert off_camera == set(), off_camera
    print("ok  test_bind_magnitude_beats_equal_vote_tie_in_multicam_podcast")


def main():
    test_overlap_ms_partial_overlap()
    test_overlap_ms_no_overlap_is_zero()
    test_speaking_overlap_ms_sums_across_intervals()
    test_bind_binds_two_clean_single_camera_voices()
    test_bind_unbound_when_a_voice_conflicts_across_files()
    test_bind_narrator_voice_has_no_vote_at_all()
    test_bind_ignores_turns_with_no_voice_mapping()
    test_bind_majority_wins_when_one_file_disagrees_among_several()
    test_bind_magnitude_beats_equal_vote_tie_in_multicam_podcast()
    print("\nall identity-bind-asd tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
