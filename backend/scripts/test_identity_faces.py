"""
Tests for cross-file face-track clustering (app.services.l3.identity.faces)
-- no DB, no network, no model call, synthetic FaceTrack data only.

Run:  .venv/bin/python scripts/test_identity_faces.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l1.active_speaker import FaceFrame, FaceTrack  # noqa: E402
from app.services.l3 import pass2  # noqa: E402
from app.services.l3.identity import faces as fc  # noqa: E402
from app.services.l3.lattice import Lattice  # noqa: E402


def _track(track_id, embedding, frames):
    return FaceTrack(track_id=track_id, embedding=embedding,
                     frames=[FaceFrame(t_ms=t, box=box) for t, box in frames])


# --------------------------------------------------------------------------
# cluster
# --------------------------------------------------------------------------

def test_cluster_merges_similar_embeddings_across_files():
    face_tracks_by_file = {
        "f1": [_track(0, [1.0, 0.0, 0.0], [(0, (0, 0, 10, 10))])],
        "f2": [_track(0, [0.99, 0.01, 0.0], [(0, (0, 0, 10, 10))])],
    }
    track_to_person, persons = fc.cluster(face_tracks_by_file)
    assert track_to_person[("f1", 0)] == track_to_person[("f2", 0)], track_to_person
    assert len(persons) == 1, persons
    print("ok  test_cluster_merges_similar_embeddings_across_files")


def test_cluster_keeps_dissimilar_embeddings_separate():
    face_tracks_by_file = {
        "f1": [_track(0, [1.0, 0.0], [(0, (0, 0, 10, 10))])],
        "f2": [_track(0, [0.0, 1.0], [(0, (0, 0, 10, 10))])],
    }
    track_to_person, persons = fc.cluster(face_tracks_by_file)
    assert track_to_person[("f1", 0)] != track_to_person[("f2", 0)], track_to_person
    assert len(persons) == 2, persons
    print("ok  test_cluster_keeps_dissimilar_embeddings_separate")


def test_cluster_never_merges_two_tracks_in_the_same_file():
    # Identical embeddings, but same file -- two distinct people who happen
    # to look alike is far more likely than a face-tracker duplicating one
    # person into two tracks within its OWN file.
    face_tracks_by_file = {
        "f1": [
            _track(0, [1.0, 0.0], [(0, (0, 0, 10, 10))]),
            _track(1, [1.0, 0.0], [(0, (100, 100, 10, 10))]),
        ],
    }
    track_to_person, persons = fc.cluster(face_tracks_by_file)
    assert track_to_person[("f1", 0)] != track_to_person[("f1", 1)], track_to_person
    assert len(persons) == 2, persons
    print("ok  test_cluster_never_merges_two_tracks_in_the_same_file")


def test_cluster_track_with_no_embedding_stays_its_own_person():
    face_tracks_by_file = {"f1": [_track(0, [], [(0, (0, 0, 10, 10))])]}
    track_to_person, persons = fc.cluster(face_tracks_by_file)
    assert len(persons) == 1, persons
    print("ok  test_cluster_track_with_no_embedding_stays_its_own_person")


def test_cluster_appearance_count_sums_merged_tracks_frames():
    face_tracks_by_file = {
        "f1": [_track(0, [1.0, 0.0], [(0, (0, 0, 10, 10)), (200, (0, 0, 10, 10))])],
        "f2": [_track(0, [1.0, 0.0], [(0, (0, 0, 10, 10))])],
    }
    _track_to_person, persons = fc.cluster(face_tracks_by_file)
    assert len(persons) == 1, persons
    only = next(iter(persons.values()))
    assert only["appearance_count"] == 3, only
    assert only["is_major"] is True
    assert only["owned_voices"] == []
    print("ok  test_cluster_appearance_count_sums_merged_tracks_frames")


def test_cluster_empty_input_is_empty():
    track_to_person, persons = fc.cluster({})
    assert track_to_person == {} and persons == {}
    print("ok  test_cluster_empty_input_is_empty")


# --------------------------------------------------------------------------
# visible_persons_by_cut
# --------------------------------------------------------------------------

def _lat():
    words = [{"start_ms": i * 100, "end_ms": i * 100 + 90} for i in range(50)]
    return Lattice(file_id="f1", duration_ms=10000, words=words, turns=[], hints=[], atoms=[])


def _cut(ref, word_span):
    return pass2.Pass2Cut(source_ref=ref, kind="speech", file_id="f1", word_span=word_span,
                          label="x", summary="y")


def test_visible_persons_by_cut_includes_a_track_with_a_frame_in_span():
    # speech_cut[0], word_span (0,19) -> roughly [0, ~1990ms).
    face_tracks_by_file = {"f1": [_track(0, [1.0, 0.0], [(500, (0, 0, 10, 10))])]}
    track_to_person = {("f1", 0): "P0"}
    lattices = {"f1": _lat()}
    cuts = [_cut("speech_cut[0]", (0, 19))]
    vis = fc.visible_persons_by_cut(track_to_person, face_tracks_by_file, cuts, lattices)
    assert vis[("f1", "speech_cut[0]")] == ["P0"], vis
    print("ok  test_visible_persons_by_cut_includes_a_track_with_a_frame_in_span")


def test_visible_persons_by_cut_excludes_a_track_with_no_frame_in_span():
    # Track's only frame is at 5000ms, far outside speech_cut[0]'s ~[0,1990) span.
    face_tracks_by_file = {"f1": [_track(0, [1.0, 0.0], [(5000, (0, 0, 10, 10))])]}
    track_to_person = {("f1", 0): "P0"}
    lattices = {"f1": _lat()}
    cuts = [_cut("speech_cut[0]", (0, 19))]
    vis = fc.visible_persons_by_cut(track_to_person, face_tracks_by_file, cuts, lattices)
    assert ("f1", "speech_cut[0]") not in vis, vis
    print("ok  test_visible_persons_by_cut_excludes_a_track_with_no_frame_in_span")


def test_visible_persons_by_cut_caps_a_crowd_to_the_most_prominent():
    # 8 distinct persons all visible in one cut -- only the top
    # MAX_VISIBLE_PER_CUT (ranked by mean face-box area) survive.
    tracks = []
    track_to_person = {}
    for i in range(8):
        area_side = 10 + i   # later tracks have a bigger (more prominent) box
        tracks.append(_track(i, [float(i), 1.0], [(500, (0, 0, area_side, area_side))]))
        track_to_person[("f1", i)] = f"P{i}"
    face_tracks_by_file = {"f1": tracks}
    lattices = {"f1": _lat()}
    cuts = [_cut("speech_cut[0]", (0, 19))]
    vis = fc.visible_persons_by_cut(track_to_person, face_tracks_by_file, cuts, lattices)
    assert len(vis[("f1", "speech_cut[0]")]) == fc.MAX_VISIBLE_PER_CUT
    # The most prominent (largest-box) persons are P7, P6, ... -- the smallest
    # (P0) must have been dropped.
    assert "P0" not in vis[("f1", "speech_cut[0]")], vis
    print("ok  test_visible_persons_by_cut_caps_a_crowd_to_the_most_prominent")


def test_visible_persons_by_cut_unresolvable_span_is_skipped():
    # A video cut with no atom_ids resolves no span at all.
    cut = pass2.Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1",
                         label="x", summary="y")
    vis = fc.visible_persons_by_cut({}, {}, [cut], {"f1": _lat()})
    assert vis == {}, vis
    print("ok  test_visible_persons_by_cut_unresolvable_span_is_skipped")


def main():
    test_cluster_merges_similar_embeddings_across_files()
    test_cluster_keeps_dissimilar_embeddings_separate()
    test_cluster_never_merges_two_tracks_in_the_same_file()
    test_cluster_track_with_no_embedding_stays_its_own_person()
    test_cluster_appearance_count_sums_merged_tracks_frames()
    test_cluster_empty_input_is_empty()
    test_visible_persons_by_cut_includes_a_track_with_a_frame_in_span()
    test_visible_persons_by_cut_excludes_a_track_with_no_frame_in_span()
    test_visible_persons_by_cut_caps_a_crowd_to_the_most_prominent()
    test_visible_persons_by_cut_unresolvable_span_is_skipped()
    print("\nall identity-faces tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
