"""
Tests for the cuts-v3 pass-2a module (``app.services.l3.pass2a``) -- IDENTITY
+ take resolution. No DB, NO REAL API CALLS. Reuses the fake-SDK-client
pattern from ``test_ingest_client.py``.

Run:  .venv/bin/python scripts/test_pass2a.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import pass2a  # noqa: E402
from app.services.l3.image_plan import PlannedFrame  # noqa: E402
from app.services.l3.lattice import Lattice  # noqa: E402
from app.services.l3.pass1 import (  # noqa: E402
    Pass1Output, SpeechCut, TakeCandidate, TakeMember, VideoTentativeGroup,
)
from app.services.llm import client as ic  # noqa: E402
from test_ingest_client import FakeBlock, FakeClient, FakeResponse  # noqa: E402


def _with_fake_client(responses):
    fake = FakeClient(responses)
    orig = ic._sdk_client
    ic._sdk_client = lambda: fake
    return fake, orig


def _lat(file_id):
    return Lattice(file_id=file_id, duration_ms=10000, words=[], turns=[], hints=[], atoms=[])


# --------------------------------------------------------------------------
# IdentityCut / IdentityOutput schema
# --------------------------------------------------------------------------

def test_locators_are_optional_at_parse_time_and_backfilled():
    # word_span/atom_ids are deliberately NOT required by the schema -- the
    # model no longer echoes them (that echo was the single biggest
    # output-complexity failure); backfill_locators derives them from pass 1.
    p1 = Pass1Output(
        speech_cuts=[SpeechCut(file_id="f1", word_span=(3, 9), label="s")],
        video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[4, 5, 6])],
    )
    out = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          label="x", summary="y"),
        pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                          label="x", summary="y"),
    ])
    filled = pass2a.backfill_locators(out, p1)
    assert filled.cuts[0].word_span == (3, 9), filled.cuts[0]
    assert filled.cuts[1].atom_ids == [4, 5, 6], filled.cuts[1]
    assert pass2a._locators_resolved(filled) is None
    print("ok  test_locators_are_optional_at_parse_time_and_backfilled")


def test_kind_alias_is_normalized_at_parse_time():
    # The model intermittently echoes pass 1's unit name into the kind enum
    # ("video_tentative_group" instead of "video"). It's unambiguous, so it's
    # normalized rather than burning a re-ask (observed twice-in-a-row on a
    # real Reel-trail shard).
    c = pass2a.IdentityCut(source_ref="video_group[8]", kind="video_tentative_group",
                           file_id="f1", label="x", summary="y")
    assert c.kind == "video", c
    c2 = pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech_cut",
                            file_id="f1", label="x", summary="y")
    assert c2.kind == "speech", c2
    print("ok  test_kind_alias_is_normalized_at_parse_time")


def test_backfill_leaves_split_video_groups_to_the_model():
    # A video group split into two cuts: backfill must NOT overwrite the
    # pieces' own atom_ids (that split IS the model's judgment); the
    # partition check validates them instead.
    p1 = Pass1Output(
        video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[0, 1, 2])],
    )
    good = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0], label="a", summary="a"),
        pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[1, 2], label="b", summary="b"),
    ])
    filled = pass2a.backfill_locators(good, p1)
    assert filled.cuts[0].atom_ids == [0] and filled.cuts[1].atom_ids == [1, 2]
    assert pass2a._split_groups_partition_atoms(filled, p1) is None

    lost_atom = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0], label="a", summary="a"),
        pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[2], label="b", summary="b"),
    ])
    err = pass2a._split_groups_partition_atoms(lost_atom, p1)
    assert err is not None and "[1]" in err, err

    missing_ids = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0, 1, 2], label="a", summary="a"),
        pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                          label="b", summary="b"),   # split piece with no atom_ids
    ])
    filled2 = pass2a.backfill_locators(missing_ids, p1)
    err2 = pass2a._locators_resolved(filled2)
    assert err2 is not None and "atom_ids" in err2, err2
    print("ok  test_backfill_leaves_split_video_groups_to_the_model")


def test_invalid_take_role_rejected():
    try:
        pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          word_span=(0, 1), label="x", summary="y", take_role="runner-up")
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ok  test_invalid_take_role_rejected")


def test_take_role_aliases_normalize_to_take():
    # Real model output has used "alt" where the schema says "take" --
    # a reasonable synonym, normalized rather than burning a re-ask on it.
    for alias in ("alt", "ALT", "alternate", "loser", "Other"):
        cut = pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                                 word_span=(0, 1), label="x", summary="y", take_role=alias)
        assert cut.take_role == "take", (alias, cut.take_role)
    print("ok  test_take_role_aliases_normalize_to_take")


def test_valid_cuts_round_trip():
    speech = pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                                word_span=(0, 3), label="intro", summary="says hello",
                                take_role="winner", take_group_id="tg1")
    video = pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                               atom_ids=[0, 1], label="pan", summary="pans across desk")
    out = pass2a.IdentityOutput(cuts=[speech, video])
    assert len(out.cuts) == 2
    assert out.cuts[0].take_role == "winner"
    print("ok  test_valid_cuts_round_trip")


def test_identity_output_rejects_an_unexpected_wrapper_key():
    wrapped = {"$PARAMETER_NAME": {"cuts": [
        {"source_ref": "speech_cut[0]", "kind": "speech", "file_id": "f1",
         "word_span": [0, 1], "label": "x", "summary": "y"},
    ]}}
    try:
        pass2a.IdentityOutput.model_validate(wrapped)
        assert False, "expected a validation error"
    except Exception:
        pass
    print("ok  test_identity_output_rejects_an_unexpected_wrapper_key")


def test_no_duplicate_atoms_passes_when_every_atom_is_used_once():
    out = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0, 1], label="a", summary="a"),
        pass2a.IdentityCut(source_ref="video_group[1]", kind="video", file_id="f1",
                          atom_ids=[2, 3], label="b", summary="b"),
    ])
    assert pass2a._no_duplicate_atoms(out) is None
    print("ok  test_no_duplicate_atoms_passes_when_every_atom_is_used_once")


def test_no_duplicate_atoms_catches_an_atom_split_across_two_cuts():
    out = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0, 1, 2], label="a", summary="a"),
        pass2a.IdentityCut(source_ref="video_group[0b]", kind="video", file_id="f1",
                          atom_ids=[2, 3], label="b", summary="b"),   # atom 2 double-counted
    ])
    err = pass2a._no_duplicate_atoms(out)
    assert err is not None and "atom_id 2" in err, err
    print("ok  test_no_duplicate_atoms_catches_an_atom_split_across_two_cuts")


def test_no_duplicate_atoms_ignores_speech_cuts():
    out = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          word_span=(0, 1), label="a", summary="a"),
        pass2a.IdentityCut(source_ref="speech_cut[1]", kind="speech", file_id="f1",
                          word_span=(2, 3), label="b", summary="b"),
    ])
    assert pass2a._no_duplicate_atoms(out) is None
    print("ok  test_no_duplicate_atoms_ignores_speech_cuts")


def test_kind_matches_source_ref_passes_when_consistent():
    out = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          word_span=(0, 1), label="a", summary="a"),
        pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0, 1], label="b", summary="b"),
    ])
    assert pass2a._kind_matches_source_ref(out) is None
    print("ok  test_kind_matches_source_ref_passes_when_consistent")


def test_kind_matches_source_ref_catches_a_speech_ref_typed_as_video():
    # Observed against the real API: a cut kept its "speech_cut[10]" ref
    # name but was emitted with kind="video" -- word_span/atom_ids then
    # don't resolve, surfacing as a much less actionable error downstream.
    out = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[10]", kind="video", file_id="f1",
                          atom_ids=[3], label="a", summary="a"),
    ])
    err = pass2a._kind_matches_source_ref(out)
    assert err is not None and "speech_cut[10]" in err and "kind='video'" in err, err
    print("ok  test_kind_matches_source_ref_catches_a_speech_ref_typed_as_video")


def test_kind_matches_source_ref_catches_a_video_ref_typed_as_speech():
    out = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="video_group[2]", kind="speech", file_id="f1",
                          word_span=(0, 1), label="a", summary="a"),
    ])
    err = pass2a._kind_matches_source_ref(out)
    assert err is not None and "video_group[2]" in err, err
    print("ok  test_kind_matches_source_ref_catches_a_video_ref_typed_as_speech")


def test_no_overlapping_word_spans_passes_when_disjoint():
    out = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          word_span=(0, 4), label="a", summary="a"),
        pass2a.IdentityCut(source_ref="speech_cut[1]", kind="speech", file_id="f1",
                          word_span=(5, 12), label="b", summary="b"),
    ])
    assert pass2a._no_overlapping_word_spans(out) is None
    print("ok  test_no_overlapping_word_spans_passes_when_disjoint")


def test_no_overlapping_word_spans_catches_a_duplicate_span():
    # Observed against the real API: two speech cuts with the identical
    # word_span in the same file, surfacing downstream as a raw ms overlap
    # with no indication of which cuts or why.
    out = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          word_span=(0, 4), label="a", summary="a"),
        pass2a.IdentityCut(source_ref="speech_cut[1]", kind="speech", file_id="f1",
                          word_span=(0, 4), label="b", summary="b"),
    ])
    err = pass2a._no_overlapping_word_spans(out)
    assert err is not None and "speech_cut[0]" in err and "speech_cut[1]" in err, err
    print("ok  test_no_overlapping_word_spans_catches_a_duplicate_span")


def test_no_overlapping_word_spans_ignores_different_files():
    out = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          word_span=(0, 4), label="a", summary="a"),
        pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f2",
                          word_span=(0, 4), label="b", summary="b"),
    ])
    assert pass2a._no_overlapping_word_spans(out) is None
    print("ok  test_no_overlapping_word_spans_ignores_different_files")


def _pass1_with(n_speech: int = 4, n_video: int = 4) -> Pass1Output:
    """A Pass1Output with enough refs for every source_ref the semantic-check
    tests use -- _source_refs_exist validates refs against these counts."""
    return Pass1Output(
        speech_cuts=[SpeechCut(file_id="f1", word_span=(i, i), label=f"s{i}")
                     for i in range(n_speech)],
        video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[i])
                                for i in range(n_video)],
    )


def test_pass2a_semantic_checks_combines_all_checks():
    p1 = _pass1_with()
    kind_mismatch = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[0]", kind="video", file_id="f1",
                          atom_ids=[0], label="a", summary="a"),
    ])
    assert pass2a._pass2a_semantic_checks(kind_mismatch, p1, {}, {"f1"}) is not None

    # duplicate atoms are only possible within a SPLIT group now (backfill
    # overwrites an un-split group's atom_ids from pass 1 wholesale)
    dup_atoms = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0], label="a", summary="a"),
        pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0], label="b", summary="b"),
    ])
    assert pass2a._pass2a_semantic_checks(dup_atoms, p1, {}, {"f1"}) is not None

    # overlapping word spans can only originate in pass 1 now (backfill
    # copies pass 1's spans verbatim) -- caught here too, post-backfill
    p1_overlap = Pass1Output(speech_cuts=[
        SpeechCut(file_id="f1", word_span=(0, 4), label="a"),
        SpeechCut(file_id="f1", word_span=(2, 6), label="b"),
    ])
    dup_words = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          label="a", summary="a"),
        pass2a.IdentityCut(source_ref="speech_cut[1]", kind="speech", file_id="f1",
                          label="b", summary="b"),
    ])
    assert pass2a._pass2a_semantic_checks(dup_words, p1_overlap, {}, {"f1"}) is not None

    clean = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          word_span=(0, 1), label="a", summary="a"),
    ])
    assert pass2a._pass2a_semantic_checks(clean, p1, {}, {"f1"}) is None

    # a cut for a clip OUTSIDE this shard is FILTERED OUT (not an error) --
    # its own shard emits it, so the stray here is a pure duplicate. p1b has
    # speech_cut[1] on f2; a shard of {f1} must silently drop it.
    p1b = Pass1Output(speech_cuts=[
        SpeechCut(file_id="f1", word_span=(0, 1), label="a"),
        SpeechCut(file_id="f2", word_span=(0, 1), label="b"),
    ])
    out_of_scope = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[1]", kind="speech", file_id="f2",
                          word_span=(0, 1), label="b", summary="b"),
    ])
    assert pass2a._pass2a_semantic_checks(out_of_scope, p1b, {}, {"f1"}) is None
    print("ok  test_pass2a_semantic_checks_combines_all_checks")


def test_drop_out_of_shard_cuts_filters_strays_and_fixes_file_id():
    p1 = Pass1Output(speech_cuts=[
        SpeechCut(file_id="f1", word_span=(0, 1), label="a"),
        SpeechCut(file_id="f2", word_span=(0, 1), label="b"),
    ])
    out = pass2a.IdentityOutput(cuts=[
        # in-shard, correct
        pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          word_span=(0, 1), label="a", summary="a"),
        # out-of-shard (source_ref resolves to f2) -> dropped
        pass2a.IdentityCut(source_ref="speech_cut[1]", kind="speech", file_id="f2",
                          word_span=(0, 1), label="b", summary="b"),
        # in-shard but model mislabeled file_id -> kept, file_id corrected to f1
        pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f9",
                          word_span=(0, 1), label="c", summary="c"),
    ])
    filtered, dropped = pass2a._drop_out_of_shard_cuts(out, p1, {"f1"})
    assert dropped == 1, dropped
    assert [c.source_ref for c in filtered.cuts] == ["speech_cut[0]", "speech_cut[0]"]
    assert all(c.file_id == "f1" for c in filtered.cuts), [c.file_id for c in filtered.cuts]
    print("ok  test_drop_out_of_shard_cuts_filters_strays_and_fixes_file_id")


def test_source_refs_exist_rejects_an_invented_ref():
    p1 = _pass1_with(n_speech=2, n_video=1)
    invented = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="take[intro_greeting]_take1", kind="speech",
                          file_id="f1", word_span=(0, 6), label="a", summary="a"),
    ])
    err = pass2a._source_refs_exist(invented, p1)
    assert err is not None and "take[intro_greeting]_take1" in err, err

    out_of_range = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[7]", kind="speech", file_id="f1",
                          word_span=(0, 1), label="a", summary="a"),
    ])
    err2 = pass2a._source_refs_exist(out_of_range, p1)
    assert err2 is not None and "speech_cut[7]" in err2, err2

    ok = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[1]", kind="speech", file_id="f1",
                          word_span=(0, 1), label="a", summary="a"),
        pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0], label="b", summary="b"),
    ])
    assert pass2a._source_refs_exist(ok, p1) is None
    print("ok  test_source_refs_exist_rejects_an_invented_ref")


def _lattice_for_cross_kind_test():
    from app.services.l3.lattice import Atom, Lattice
    words = [
        {"start_ms": 0, "end_ms": 200, "text": "a"},
        {"start_ms": 300, "end_ms": 500, "text": "b"},
        {"start_ms": 600, "end_ms": 800, "text": "c"},
    ]
    atoms = [Atom(atom_id=0, file_id="f1", start_ms=400, end_ms=1000, state_in="x", state_out="y",
                 action_energy=0.1, coherence=0.9)]
    return Lattice(file_id="f1", duration_ms=1000, words=words, turns=[], hints=[], atoms=atoms)


def test_no_cross_kind_ms_overlap_passes_when_disjoint():
    lattices = {"f1": _lattice_for_cross_kind_test()}
    out = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          word_span=(0, 0), label="a", summary="a"),
    ])
    assert pass2a._no_cross_kind_ms_overlap(out, lattices) is None
    print("ok  test_no_cross_kind_ms_overlap_passes_when_disjoint")


def test_no_cross_kind_ms_overlap_catches_a_speech_and_video_cut_overlapping():
    # Observed against the real API: a speech cut and a video cut in the
    # same file resolve to overlapping ms spans -- neither the atom-id nor
    # the word-index check catches this (different kinds), only resolving
    # actual ms spans does.
    lattices = {"f1": _lattice_for_cross_kind_test()}
    out = pass2a.IdentityOutput(cuts=[
        pass2a.IdentityCut(source_ref="speech_cut[1]", kind="speech", file_id="f1",
                          word_span=(1, 2), label="a", summary="a"),   # words[1:2] -> ms [300, 800)
        pass2a.IdentityCut(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0], label="b", summary="b"),        # atom 0 -> ms [400, 1000)
    ])
    err = pass2a._no_cross_kind_ms_overlap(out, lattices)
    assert err is not None and "speech_cut[1]" in err and "video_group[0]" in err, err
    print("ok  test_no_cross_kind_ms_overlap_catches_a_speech_and_video_cut_overlapping")


# --------------------------------------------------------------------------
# Shard building
# --------------------------------------------------------------------------

def test_single_file_one_shard():
    frames = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f1", 200, "speech_cut", "speech_cut[1]")]
    shards = pass2a.build_identity_shards(Pass1Output(), frames)
    assert shards == [["f1"]], shards
    print("ok  test_single_file_one_shard")


def test_unrelated_files_pack_into_one_shard_when_small():
    frames = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f2", 100, "speech_cut", "speech_cut[0]")]
    shards = pass2a.build_identity_shards(Pass1Output(), frames)
    assert len(shards) == 1, shards
    assert set(shards[0]) == {"f1", "f2"}, shards
    print("ok  test_unrelated_files_pack_into_one_shard_when_small")


def test_take_group_forces_co_location_across_files():
    frames = [PlannedFrame("f1", 100, "take_member", "take[tg1]"),
             PlannedFrame("f2", 100, "take_member", "take[tg1]")]
    pass1 = Pass1Output(take_candidates=[TakeCandidate(group_id="tg1", members=[
        TakeMember(file_id="f1", word_span=(0, 1)),
        TakeMember(file_id="f2", word_span=(0, 1)),
    ])])
    shards = pass2a.build_identity_shards(pass1, frames)
    assert len(shards) == 1 and set(shards[0]) == {"f1", "f2"}, shards
    print("ok  test_take_group_forces_co_location_across_files")


def test_bin_packing_splits_when_over_budget():
    orig = pass2a.MAX_IMAGES_PER_SHARD
    pass2a.MAX_IMAGES_PER_SHARD = 2
    try:
        frames = ([PlannedFrame("f1", i, "speech_cut", f"speech_cut[{i}]") for i in range(2)] +
                 [PlannedFrame("f2", i, "speech_cut", f"speech_cut[{i}]") for i in range(2)])
        shards = pass2a.build_identity_shards(Pass1Output(), frames)
    finally:
        pass2a.MAX_IMAGES_PER_SHARD = orig
    assert len(shards) == 2, shards
    all_files = {f for shard in shards for f in shard}
    assert all_files == {"f1", "f2"}
    print("ok  test_bin_packing_splits_when_over_budget")


def test_bin_packing_splits_on_cut_count_even_when_images_fit():
    # Two unrelated files, plenty of shared image budget, but together they
    # have too many cuts for one shard -- added after real ingest runs
    # showed the model getting unreliable on very large single-call outputs
    # (~40-80 cuts) independent of image count.
    orig = pass2a.MAX_CUTS_PER_SHARD
    pass2a.MAX_CUTS_PER_SHARD = 2
    try:
        frames = [PlannedFrame("f1", 0, "speech_cut", "speech_cut[0]"),
                 PlannedFrame("f2", 0, "speech_cut", "speech_cut[0]")]
        pass1 = Pass1Output(speech_cuts=[
            SpeechCut(file_id="f1", word_span=(0, 1), label="a"),
            SpeechCut(file_id="f1", word_span=(2, 3), label="b"),
            SpeechCut(file_id="f2", word_span=(0, 1), label="c"),
            SpeechCut(file_id="f2", word_span=(2, 3), label="d"),
        ])
        shards = pass2a.build_identity_shards(pass1, frames)
    finally:
        pass2a.MAX_CUTS_PER_SHARD = orig
    assert len(shards) == 2, shards
    assert {f for shard in shards for f in shard} == {"f1", "f2"}
    print("ok  test_bin_packing_splits_on_cut_count_even_when_images_fit")


def test_oversized_take_cluster_is_not_split():
    orig = pass2a.MAX_IMAGES_PER_SHARD
    pass2a.MAX_IMAGES_PER_SHARD = 1
    try:
        frames = [PlannedFrame("f1", 0, "take_member", "take[tg1]"),
                 PlannedFrame("f2", 0, "take_member", "take[tg1]")]
        pass1 = Pass1Output(take_candidates=[TakeCandidate(group_id="tg1", members=[
            TakeMember(file_id="f1", word_span=(0, 1)),
            TakeMember(file_id="f2", word_span=(0, 1)),
        ])])
        shards = pass2a.build_identity_shards(pass1, frames)
    finally:
        pass2a.MAX_IMAGES_PER_SHARD = orig
    assert len(shards) == 1, shards
    assert set(shards[0]) == {"f1", "f2"}
    print("ok  test_oversized_take_cluster_is_not_split")


def test_empty_frames_yield_no_shards():
    assert pass2a.build_identity_shards(Pass1Output(), []) == []
    print("ok  test_empty_frames_yield_no_shards")


# --------------------------------------------------------------------------
# Shard block rendering + orchestration
# --------------------------------------------------------------------------

def test_shard_blocks_skip_unresolved_images():
    frames = [PlannedFrame("f1", 200, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f1", 100, "speech_cut", "speech_cut[1]")]
    images = {("f1", 100): "ZmFrZQ=="}   # only one of the two resolved
    blocks = pass2a.build_identity_shard_blocks(frames, images)
    assert len(blocks) == 2, blocks   # one [caption, image] pair only
    assert blocks[0]["type"] == "text" and "IMG 1" in blocks[0]["text"]
    assert "0.1s" in blocks[0]["text"], blocks[0]
    assert blocks[1]["type"] == "image"
    print("ok  test_shard_blocks_skip_unresolved_images")


def test_shard_blocks_ordered_by_file_then_ts():
    frames = [PlannedFrame("f2", 50, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f1", 200, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f1", 100, "speech_cut", "speech_cut[1]")]
    images = {("f2", 50): "a", ("f1", 200): "b", ("f1", 100): "c"}
    blocks = pass2a.build_identity_shard_blocks(frames, images)
    captions = [b["text"] for b in blocks if b["type"] == "text"]
    assert "clip f1, 0.1s" in captions[0], captions
    assert "clip f1, 0.2s" in captions[1], captions
    assert "clip f2, 0.1s" in captions[2], captions
    print("ok  test_shard_blocks_ordered_by_file_then_ts")


def test_run_identity_shard_raises_when_no_images_resolve():
    frames = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]")]
    try:
        pass2a.run_identity_shard([("f1", "a.mp4", 10000, _lat("f1"))], Pass1Output(), frames, {})
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ok  test_run_identity_shard_raises_when_no_images_resolve")


def test_run_identity_shard_calls_complete_with_pass2_stage_and_cached_prefix():
    good = {"cuts": [{
        "source_ref": "speech_cut[0]", "kind": "speech", "file_id": "f1",
        "word_span": [0, 2], "label": "intro", "summary": "hello there",
    }]}
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)])])
    frames = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]")]
    pass1_output = Pass1Output(speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 2), label="intro")])
    try:
        result = pass2a.run_identity_shard(
            [("f1", "a.mp4", 10000, _lat("f1"))], pass1_output, frames, {("f1", 100): "ZmFrZQ=="},
        )
    finally:
        ic._sdk_client = orig

    call = fake.messages.calls[0]
    assert call["tools"][0]["input_schema"] == pass2a.IdentityOutput.model_json_schema()
    content = call["messages"][0]["content"]
    # cached prefix (pass1 blocks + rendered pass1 output) ends with the cache
    # breakpoint; the image caption/image pair rides after it, uncached.
    assert content[-3]["cache_control"] == {"type": "ephemeral"}, content
    assert "cache_control" not in content[-2]
    assert "cache_control" not in content[-1]
    assert content[-2]["type"] == "text" and "IMG 1" in content[-2]["text"]
    assert content[-1]["type"] == "image"
    parsed = pass2a.IdentityOutput.model_validate(result.data)
    assert parsed.cuts[0].source_ref == "speech_cut[0]"
    print("ok  test_run_identity_shard_calls_complete_with_pass2_stage_and_cached_prefix")


def main():
    test_locators_are_optional_at_parse_time_and_backfilled()
    test_kind_alias_is_normalized_at_parse_time()
    test_backfill_leaves_split_video_groups_to_the_model()
    test_invalid_take_role_rejected()
    test_take_role_aliases_normalize_to_take()
    test_valid_cuts_round_trip()
    test_identity_output_rejects_an_unexpected_wrapper_key()
    test_no_duplicate_atoms_passes_when_every_atom_is_used_once()
    test_no_duplicate_atoms_catches_an_atom_split_across_two_cuts()
    test_no_duplicate_atoms_ignores_speech_cuts()
    test_kind_matches_source_ref_passes_when_consistent()
    test_kind_matches_source_ref_catches_a_speech_ref_typed_as_video()
    test_kind_matches_source_ref_catches_a_video_ref_typed_as_speech()
    test_no_overlapping_word_spans_passes_when_disjoint()
    test_no_overlapping_word_spans_catches_a_duplicate_span()
    test_no_overlapping_word_spans_ignores_different_files()
    test_pass2a_semantic_checks_combines_all_checks()
    test_drop_out_of_shard_cuts_filters_strays_and_fixes_file_id()
    test_source_refs_exist_rejects_an_invented_ref()
    test_no_cross_kind_ms_overlap_passes_when_disjoint()
    test_no_cross_kind_ms_overlap_catches_a_speech_and_video_cut_overlapping()
    test_single_file_one_shard()
    test_unrelated_files_pack_into_one_shard_when_small()
    test_take_group_forces_co_location_across_files()
    test_bin_packing_splits_when_over_budget()
    test_bin_packing_splits_on_cut_count_even_when_images_fit()
    test_oversized_take_cluster_is_not_split()
    test_empty_frames_yield_no_shards()
    test_shard_blocks_skip_unresolved_images()
    test_shard_blocks_ordered_by_file_then_ts()
    test_run_identity_shard_raises_when_no_images_resolve()
    test_run_identity_shard_calls_complete_with_pass2_stage_and_cached_prefix()
    print("\nall pass2a tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
