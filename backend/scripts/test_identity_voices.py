"""
Tests for cross-clip voice clustering (app.services.l3.identity.voices) --
no DB, no network, synthetic voiceprints only.

Run:  .venv/bin/python scripts/test_identity_voices.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3.identity import voices  # noqa: E402

A = [1.0, 0.0, 0.0]
A_NEAR = [0.98, 0.05, 0.0]     # cosine ~0.999 with A -- same voice
B = [0.0, 1.0, 0.0]            # orthogonal to A -- cosine 0.0, different voice


def test_cluster_voices_merges_similar_voiceprints_across_files():
    emb = {"f1": {"S0": A}, "f2": {"S0": A_NEAR}}
    voice_of = voices.cluster_voices(emb, {})
    assert voice_of[("f1", "S0")] == voice_of[("f2", "S0")], voice_of
    print("ok  test_cluster_voices_merges_similar_voiceprints_across_files")


def test_cluster_voices_keeps_dissimilar_voiceprints_separate():
    emb = {"f1": {"S0": A}, "f2": {"S0": B}}
    voice_of = voices.cluster_voices(emb, {})
    assert voice_of[("f1", "S0")] != voice_of[("f2", "S0")], voice_of
    print("ok  test_cluster_voices_keeps_dissimilar_voiceprints_separate")


def test_cluster_voices_never_merges_two_speakers_in_the_same_file():
    # Same file, two local speakers, IDENTICAL embeddings (contrived) --
    # must never merge two people captured on the same mic.
    emb = {"f1": {"S0": A, "S1": A}}
    voice_of = voices.cluster_voices(emb, {})
    assert voice_of[("f1", "S0")] != voice_of[("f1", "S1")], voice_of
    print("ok  test_cluster_voices_never_merges_two_speakers_in_the_same_file")


def test_cluster_voices_unifies_outlook_group_members_by_shared_label():
    # f1 and f2 are two angles of ONE outlook group, sharing local speaker
    # label "S0" on the (re-based) authoritative audio -- unified by group
    # membership alone, no embedding comparison needed (f2 has none at all).
    emb = {"f1": {"S0": A}}
    groups = {"g1": {"auth": "f1", "members": {"f1", "f2"}}}
    # f2's "S0" has no embedding at all -- group unification must still work
    # off the ROSTER (all_speakers_by_file), not the embedding-bearing subset.
    all_speakers = {"f1": ["S0"], "f2": ["S0"]}
    full = voices.assign_voices(emb, groups, all_speakers)
    assert full[("f1", "S0")] == full[("f2", "S0")], full
    print("ok  test_cluster_voices_unifies_outlook_group_members_by_shared_label")


def test_assign_voices_gives_singleton_to_unembedded_speaker():
    emb = {"f1": {"S0": A}}
    all_speakers = {"f1": ["S0"], "f2": ["S0"]}   # f2's S0 has NO embedding
    full = voices.assign_voices(emb, {}, all_speakers)
    assert full[("f1", "S0")] != full[("f2", "S0")], full
    assert len(set(full.values())) == 2, full
    print("ok  test_assign_voices_gives_singleton_to_unembedded_speaker")


def test_voice_ids_stable_ordering_by_minimum_key():
    emb = {"fz": {"S0": B}, "fa": {"S0": A}}
    voice_of = voices.cluster_voices(emb, {})
    # "fa" < "fz" lexicographically -- its cluster must be V0.
    assert voice_of[("fa", "S0")] == "V0", voice_of
    assert voice_of[("fz", "S0")] == "V1", voice_of
    print("ok  test_voice_ids_stable_ordering_by_minimum_key")


def test_cosine_helper_handles_degenerate_vectors():
    assert voices._cosine([], [1.0]) == -1.0
    assert voices._cosine([1.0, 2.0], [1.0]) == -1.0
    assert voices._cosine([0.0, 0.0], [1.0, 0.0]) == -1.0
    assert abs(voices._cosine([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
    print("ok  test_cosine_helper_handles_degenerate_vectors")


def main():
    test_cluster_voices_merges_similar_voiceprints_across_files()
    test_cluster_voices_keeps_dissimilar_voiceprints_separate()
    test_cluster_voices_never_merges_two_speakers_in_the_same_file()
    test_cluster_voices_unifies_outlook_group_members_by_shared_label()
    test_assign_voices_gives_singleton_to_unembedded_speaker()
    test_voice_ids_stable_ordering_by_minimum_key()
    test_cosine_helper_handles_degenerate_vectors()
    print("\nall identity-voices tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
