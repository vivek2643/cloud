"""
Tests for the cuts-v3 pass-2 module (``app.services.l3.pass2``) -- the
MERGED per-cut vision call (identity + full visual judgment in one call,
pass2_merge.plan.md). No DB, NO REAL API CALLS. Reuses the fake-SDK-client
pattern from ``test_ingest_client.py``.

Folded from the old ``test_pass2a.py``/``test_pass2b.py`` (both deleted --
their modules ``pass2a.py``/``pass2b.py`` were merged into ``pass2.py``).
Tests specific to the old take-comparison co-location machinery
(``build_identity_shards``' union-find bundling) are dropped -- that
machinery no longer exists, since take-grouping moved fully to deterministic
code (``apply_take_groups``) and batching is now pure size-based chunking.

Run:  .venv/bin/python scripts/test_pass2.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.config import get_settings  # noqa: E402
from app.services.l3 import pass2  # noqa: E402
from app.services.l3 import post  # noqa: E402
from app.services.l3.image_plan import PlannedFrame  # noqa: E402
from app.services.l3.lattice import Lattice  # noqa: E402
from app.services.l3.pass1 import (  # noqa: E402
    JunkSuspect, Pass1Output, SpeechCut, TakeCandidate, TakeMember, VideoTentativeGroup,
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
# Pass2Cut / Pass2Output (the final merged record post.py consumes)
# --------------------------------------------------------------------------

def test_pass2cut_constructs_with_all_fields():
    cut = pass2.Pass2Cut(
        source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 3),
        label="intro", summary="says hello", speaker="S0", on_camera=True,
        junk=False, junk_reason="", framing=pass2.Framing(rotation_deg=90.0),
        look=pass2.Look(graded=True), caption_zones=[(0.1, 0.1, 0.2, 0.1)],
        taste_fences=pass2.TasteFences(max_tasteful_speed=1.5), readability_ms=800,
        natural_sound=True, take_group_id="tg1", take_role="winner",
    )
    assert cut.framing.rotation_deg == 90.0
    assert cut.look.graded is True
    assert cut.take_role == "winner"
    print("ok  test_pass2cut_constructs_with_all_fields")


def test_apply_junk_suspects_hides_a_contained_speech_cut():
    # A leading camera cue the coverage-fill surfaced as its own recovered
    # speech_cut and pass 1 listed as a junk_suspect -> marked junk (binary,
    # recoverable; hidden into the Discarded tray).
    p2 = pass2.Pass2Output(cuts=[
        pass2.Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 1),
                       label="and go", summary="cue"),
        pass2.Pass2Cut(source_ref="speech_cut[1]", kind="speech", file_id="f1", word_span=(2, 8),
                       label="the real line", summary="content"),
    ])
    p1 = Pass1Output(junk_suspects=[JunkSuspect(file_id="f1", word_span=(0, 1), reason="camera cue")])
    out = pass2.apply_junk_suspects(p2, p1)
    assert out.cuts[0].junk is True, out.cuts[0]
    assert out.cuts[0].junk_reason == "camera cue"
    assert out.cuts[1].junk is False, out.cuts[1]   # real content untouched
    print("ok  test_apply_junk_suspects_hides_a_contained_speech_cut")


def test_apply_junk_suspects_ignores_partial_overlap():
    # A suspect that only partially overlaps a cut must NOT hide it (clipping
    # real content is exactly the over-removal the "keep the bar high" rule
    # forbids).
    p2 = pass2.Pass2Output(cuts=[
        pass2.Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 8),
                       label="mixed", summary="cue then content"),
    ])
    p1 = Pass1Output(junk_suspects=[JunkSuspect(file_id="f1", word_span=(0, 1), reason="camera cue")])
    out = pass2.apply_junk_suspects(p2, p1)
    assert out.cuts[0].junk is False, out.cuts[0]
    print("ok  test_apply_junk_suspects_ignores_partial_overlap")


# --------------------------------------------------------------------------
# apply_take_groups (pass2_merge.plan.md Phase 1 -- generalized from the old
# apply_outlook_roles; take-grouping is now fully deterministic code)
# --------------------------------------------------------------------------

def test_apply_take_groups_outlook_path_matches_old_apply_outlook_roles():
    # An "outlook:"-prefixed group -> every member forced to take_role
    # "outlook" with the shared group_id, same override behavior the old
    # apply_outlook_roles had (never a winner among alternate angles).
    p1 = Pass1Output(take_candidates=[TakeCandidate(group_id="outlook:beat0", members=[
        TakeMember(file_id="f1", word_span=(0, 5)),
        TakeMember(file_id="f2", word_span=(0, 5)),
    ])])
    p2 = pass2.Pass2Output(cuts=[
        pass2.Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 5),
                       label="a", summary="a"),
        pass2.Pass2Cut(source_ref="speech_cut[1]", kind="speech", file_id="f2", word_span=(0, 5),
                       label="b", summary="b"),
    ])
    out = pass2.apply_take_groups(p2, p1)
    assert all(c.take_role == "outlook" and c.take_group_id == "outlook:beat0" for c in out.cuts), out.cuts
    print("ok  test_apply_take_groups_outlook_path_matches_old_apply_outlook_roles")


def test_apply_take_groups_non_outlook_group_gets_take_role():
    # A plain (non-"outlook:") group -> every member gets take_role="take"
    # (never pre-selected "winner" -- the model doesn't emit take fields at
    # all anymore; post._enforce_take_winner alone crowns the winner).
    p1 = Pass1Output(take_candidates=[TakeCandidate(group_id="tg-retake-1", members=[
        TakeMember(file_id="f1", word_span=(0, 5)),
        TakeMember(file_id="f1", word_span=(10, 15)),
    ])])
    p2 = pass2.Pass2Output(cuts=[
        pass2.Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 5),
                       label="a", summary="a"),
        pass2.Pass2Cut(source_ref="speech_cut[1]", kind="speech", file_id="f1", word_span=(10, 15),
                       label="b", summary="b"),
    ])
    out = pass2.apply_take_groups(p2, p1)
    assert all(c.take_role == "take" and c.take_group_id == "tg-retake-1" for c in out.cuts), out.cuts
    print("ok  test_apply_take_groups_non_outlook_group_gets_take_role")


def test_apply_take_groups_then_enforce_take_winner_crowns_the_best():
    # End-to-end: apply_take_groups stamps "take" on a synthetic non-outlook
    # group, then post._enforce_take_winner (untouched by this plan) crowns
    # the highest total_quality as the winner -- confirms the full handoff
    # this plan's Phase 1 depends on still works.
    p1 = Pass1Output(take_candidates=[TakeCandidate(group_id="tg1", members=[
        TakeMember(file_id="f1", word_span=(0, 5)),
        TakeMember(file_id="f1", word_span=(10, 15)),
    ])])
    p2 = pass2.Pass2Output(cuts=[
        pass2.Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 5),
                       label="a", summary="a"),
        pass2.Pass2Cut(source_ref="speech_cut[1]", kind="speech", file_id="f1", word_span=(10, 15),
                       label="b", summary="b"),
    ])
    stamped = pass2.apply_take_groups(p2, p1)
    lattice = _lat("f1")
    records = post.assemble_cut_records(stamped, {"f1": lattice}, {}, {})
    # give the second member a strictly higher quality by construction
    best = max(records, key=lambda r: r.word_span[0])
    for r in records:
        r.total_quality = 0.9 if r is best else 0.1
    post._enforce_take_winner(records)
    assert best.take_role == "winner", [(r.word_span, r.take_role) for r in records]
    others = [r for r in records if r is not best]
    assert all(r.take_role == "take" for r in others), others
    print("ok  test_apply_take_groups_then_enforce_take_winner_crowns_the_best")


def test_apply_take_groups_untouched_cut_stays_unset():
    p1 = Pass1Output(take_candidates=[TakeCandidate(group_id="tg1", members=[
        TakeMember(file_id="f1", word_span=(0, 5)),
        TakeMember(file_id="f1", word_span=(10, 15)),
    ])])
    p2 = pass2.Pass2Output(cuts=[
        pass2.Pass2Cut(source_ref="speech_cut[5]", kind="speech", file_id="f1", word_span=(20, 25),
                       label="unrelated", summary="c"),
    ])
    out = pass2.apply_take_groups(p2, p1)
    assert out.cuts[0].take_group_id is None and out.cuts[0].take_role is None
    print("ok  test_apply_take_groups_untouched_cut_stays_unset")


def test_apply_take_groups_empty_take_candidates_is_a_noop():
    p2 = pass2.Pass2Output(cuts=[
        pass2.Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 1),
                       label="a", summary="a"),
    ])
    out = pass2.apply_take_groups(p2, Pass1Output())
    assert out is p2
    print("ok  test_apply_take_groups_empty_take_candidates_is_a_noop")


# --------------------------------------------------------------------------
# CutJudgment schema (the model's per-call response type)
# --------------------------------------------------------------------------

def test_cutjudgment_has_no_take_fields():
    # D2 (pass2_merge.plan.md): take_group_id/take_role are dropped from the
    # model schema entirely -- the model can no longer emit them.
    assert "take_group_id" not in pass2.CutJudgment.model_fields
    assert "take_role" not in pass2.CutJudgment.model_fields
    print("ok  test_cutjudgment_has_no_take_fields")


def test_locators_are_optional_at_parse_time_and_backfilled():
    # word_span/atom_ids are deliberately NOT required by the schema -- the
    # model no longer echoes them (that echo was the single biggest
    # output-complexity failure); backfill_locators derives them from pass 1.
    p1 = Pass1Output(
        speech_cuts=[SpeechCut(file_id="f1", word_span=(3, 9), label="s")],
        video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[4, 5, 6])],
    )
    out = pass2.Pass2BatchOutput(cuts=[
        pass2.CutJudgment(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          label="x", summary="y"),
        pass2.CutJudgment(source_ref="video_group[0]", kind="video", file_id="f1",
                          label="x", summary="y"),
    ])
    filled = pass2.backfill_locators(out, p1)
    assert filled.cuts[0].word_span == (3, 9), filled.cuts[0]
    assert filled.cuts[1].atom_ids == [4, 5, 6], filled.cuts[1]
    assert pass2._locators_resolved(filled) is None
    print("ok  test_locators_are_optional_at_parse_time_and_backfilled")


def test_kind_alias_is_normalized_at_parse_time():
    # The model intermittently echoes pass 1's unit name into the kind enum
    # ("video_tentative_group" instead of "video"). It's unambiguous, so it's
    # normalized rather than burning a re-ask (observed twice-in-a-row on a
    # real Reel-trail shard).
    c = pass2.CutJudgment(source_ref="video_group[8]", kind="video_tentative_group",
                          file_id="f1", label="x", summary="y")
    assert c.kind == "video", c
    c2 = pass2.CutJudgment(source_ref="speech_cut[0]", kind="speech_cut",
                           file_id="f1", label="x", summary="y")
    assert c2.kind == "speech", c2
    print("ok  test_kind_alias_is_normalized_at_parse_time")


def test_channel_alias_is_normalized_at_parse_time():
    c = pass2.CutJudgment(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0], label="x", summary="y", channel="demo")
    assert c.channel == "done", c
    c2 = pass2.CutJudgment(source_ref="video_group[1]", kind="video", file_id="f1",
                           atom_ids=[1], label="x", summary="y", channel="b-roll")
    assert c2.channel == "shown", c2
    print("ok  test_channel_alias_is_normalized_at_parse_time")


def test_backfill_leaves_split_video_groups_to_the_model():
    # A video group split into two cuts: backfill must NOT overwrite the
    # pieces' own atom_ids (that split IS the model's judgment); the
    # partition check validates them instead.
    p1 = Pass1Output(
        video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[0, 1, 2])],
    )
    good = pass2.Pass2BatchOutput(cuts=[
        pass2.CutJudgment(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0], label="a", summary="a"),
        pass2.CutJudgment(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[1, 2], label="b", summary="b"),
    ])
    filled = pass2.backfill_locators(good, p1)
    assert filled.cuts[0].atom_ids == [0] and filled.cuts[1].atom_ids == [1, 2]
    assert pass2._split_groups_partition_atoms(filled, p1) is None

    lost_atom = pass2.Pass2BatchOutput(cuts=[
        pass2.CutJudgment(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0], label="a", summary="a"),
        pass2.CutJudgment(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[2], label="b", summary="b"),
    ])
    err = pass2._split_groups_partition_atoms(lost_atom, p1)
    assert err is not None and "[1]" in err, err

    missing_ids = pass2.Pass2BatchOutput(cuts=[
        pass2.CutJudgment(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0, 1, 2], label="a", summary="a"),
        pass2.CutJudgment(source_ref="video_group[0]", kind="video", file_id="f1",
                          label="b", summary="b"),   # split piece with no atom_ids
    ])
    filled2 = pass2.backfill_locators(missing_ids, p1)
    err2 = pass2._locators_resolved(filled2)
    assert err2 is not None and "atom_ids" in err2, err2
    print("ok  test_backfill_leaves_split_video_groups_to_the_model")


def test_valid_cuts_round_trip():
    speech = pass2.CutJudgment(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                               word_span=(0, 3), label="intro", summary="says hello")
    video = pass2.CutJudgment(source_ref="video_group[0]", kind="video", file_id="f1",
                              atom_ids=[0, 1], label="pan", summary="pans across desk")
    out = pass2.Pass2BatchOutput(cuts=[speech, video])
    assert len(out.cuts) == 2
    print("ok  test_valid_cuts_round_trip")


def test_pass2_batch_output_rejects_an_unexpected_wrapper_key():
    wrapped = {"$PARAMETER_NAME": {"cuts": [
        {"source_ref": "speech_cut[0]", "kind": "speech", "file_id": "f1",
         "word_span": [0, 1], "label": "x", "summary": "y"},
    ]}}
    try:
        pass2.Pass2BatchOutput.model_validate(wrapped)
        assert False, "expected a validation error"
    except Exception:
        pass
    print("ok  test_pass2_batch_output_rejects_an_unexpected_wrapper_key")


def test_cutjudgment_defaults_are_safe():
    j = pass2.CutJudgment(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          word_span=(0, 1), label="x", summary="y")
    assert j.framing.rotation_deg == 0.0
    assert j.look.graded is False
    assert j.caption_zones == []
    assert j.taste_fences.max_tasteful_speed == 1.0
    assert j.readability_ms == 0
    assert j.people == []
    print("ok  test_cutjudgment_defaults_are_safe")


def test_no_duplicate_atoms_passes_when_every_atom_is_used_once():
    out = pass2.Pass2BatchOutput(cuts=[
        pass2.CutJudgment(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0, 1], label="a", summary="a"),
        pass2.CutJudgment(source_ref="video_group[1]", kind="video", file_id="f1",
                          atom_ids=[2, 3], label="b", summary="b"),
    ])
    assert pass2._no_duplicate_atoms(out) is None
    print("ok  test_no_duplicate_atoms_passes_when_every_atom_is_used_once")


def test_no_duplicate_atoms_catches_an_atom_split_across_two_cuts():
    out = pass2.Pass2BatchOutput(cuts=[
        pass2.CutJudgment(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0, 1, 2], label="a", summary="a"),
        pass2.CutJudgment(source_ref="video_group[0b]", kind="video", file_id="f1",
                          atom_ids=[2, 3], label="b", summary="b"),   # atom 2 double-counted
    ])
    err = pass2._no_duplicate_atoms(out)
    assert err is not None and "atom_id 2" in err, err
    print("ok  test_no_duplicate_atoms_catches_an_atom_split_across_two_cuts")


def test_kind_is_derived_from_source_ref_not_the_models_field():
    # Flash-Lite routinely confuses kind (structural) with channel (semantic) and
    # emits kind="said"/"shown" on an otherwise valid ref. kind is DERIVED from
    # the ref prefix at parse time, so the mismatch is silently corrected --
    # never a re-ask, never a hard-fail (regression guard for two observed run
    # failures: video_group[..] kind="shown", speech_cut[..] kind="said").
    speech = pass2.CutJudgment(source_ref="speech_cut[10]", kind="said", file_id="f1",
                               word_span=(0, 1), label="a", summary="a")
    assert speech.kind == "speech", speech.kind
    video = pass2.CutJudgment(source_ref="video_group[3]", kind="shown", file_id="f1",
                              atom_ids=[3], label="a", summary="a")
    assert video.kind == "video", video.kind
    # so the defensive ref/kind check can no longer fire on real cuts
    out = pass2.Pass2BatchOutput(cuts=[speech, video])
    assert pass2._kind_matches_source_ref(out) is None
    print("ok  test_kind_is_derived_from_source_ref_not_the_models_field")


def test_no_overlapping_word_spans_catches_a_duplicate_span():
    out = pass2.Pass2BatchOutput(cuts=[
        pass2.CutJudgment(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          word_span=(0, 4), label="a", summary="a"),
        pass2.CutJudgment(source_ref="speech_cut[1]", kind="speech", file_id="f1",
                          word_span=(0, 4), label="b", summary="b"),
    ])
    err = pass2._no_overlapping_word_spans(out)
    assert err is not None and "speech_cut[0]" in err and "speech_cut[1]" in err, err
    print("ok  test_no_overlapping_word_spans_catches_a_duplicate_span")


def _pass1_with(n_speech: int = 4, n_video: int = 4) -> Pass1Output:
    """A Pass1Output with enough refs for every source_ref the semantic-check
    tests use -- _source_refs_exist validates refs against these counts."""
    return Pass1Output(
        speech_cuts=[SpeechCut(file_id="f1", word_span=(i, i), label=f"s{i}")
                     for i in range(n_speech)],
        video_tentative_groups=[VideoTentativeGroup(file_id="f1", atom_ids=[i])
                                for i in range(n_video)],
    )


def test_pass2_semantic_checks_combines_all_checks():
    p1 = _pass1_with()
    # (kind/ref mismatch is no longer a semantic error -- kind is derived from
    # the ref at parse time; see test_kind_is_derived_from_source_ref_*.)
    dup_atoms = pass2.Pass2BatchOutput(cuts=[
        pass2.CutJudgment(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0], label="a", summary="a"),
        pass2.CutJudgment(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0], label="b", summary="b"),
    ])
    assert pass2._pass2_semantic_checks(dup_atoms, p1, {}, {"video_group[0]"}) is not None

    clean = pass2.Pass2BatchOutput(cuts=[
        pass2.CutJudgment(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          word_span=(0, 1), label="a", summary="a"),
    ])
    assert pass2._pass2_semantic_checks(clean, p1, {}, {"speech_cut[0]"}) is None

    # a cut whose ref is OUTSIDE this batch is FILTERED OUT (not an error) --
    # its own batch emits it, so the stray here is a pure duplicate.
    out_of_scope = pass2.Pass2BatchOutput(cuts=[
        pass2.CutJudgment(source_ref="speech_cut[1]", kind="speech", file_id="f1",
                          word_span=(1, 2), label="b", summary="b"),
    ])
    assert pass2._pass2_semantic_checks(out_of_scope, p1, {}, {"speech_cut[0]"}) is None
    print("ok  test_pass2_semantic_checks_combines_all_checks")


def test_drop_out_of_batch_cuts_filters_strays_and_fixes_file_id():
    p1 = Pass1Output(speech_cuts=[
        SpeechCut(file_id="f1", word_span=(0, 1), label="a"),
        SpeechCut(file_id="f2", word_span=(0, 1), label="b"),
    ])
    out = pass2.Pass2BatchOutput(cuts=[
        pass2.CutJudgment(source_ref="speech_cut[0]", kind="speech", file_id="f1",
                          word_span=(0, 1), label="a", summary="a"),
        pass2.CutJudgment(source_ref="speech_cut[1]", kind="speech", file_id="f2",
                          word_span=(0, 1), label="b", summary="b"),
        pass2.CutJudgment(source_ref="speech_cut[0]", kind="speech", file_id="f9",
                          word_span=(0, 1), label="c", summary="c"),
    ])
    filtered, dropped = pass2._drop_out_of_batch_cuts(out, p1, {"speech_cut[0]"})
    assert dropped == 1, dropped
    assert [c.source_ref for c in filtered.cuts] == ["speech_cut[0]", "speech_cut[0]"]
    assert all(c.file_id == "f1" for c in filtered.cuts), [c.file_id for c in filtered.cuts]
    print("ok  test_drop_out_of_batch_cuts_filters_strays_and_fixes_file_id")


def test_source_refs_exist_rejects_an_invented_ref():
    p1 = _pass1_with(n_speech=2, n_video=1)
    invented = pass2.Pass2BatchOutput(cuts=[
        pass2.CutJudgment(source_ref="take[intro_greeting]_take1", kind="speech",
                          file_id="f1", word_span=(0, 6), label="a", summary="a"),
    ])
    err = pass2._source_refs_exist(invented, p1)
    assert err is not None and "take[intro_greeting]_take1" in err, err
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


def test_no_cross_kind_ms_overlap_catches_a_speech_and_video_cut_overlapping():
    lattices = {"f1": _lattice_for_cross_kind_test()}
    out = pass2.Pass2BatchOutput(cuts=[
        pass2.CutJudgment(source_ref="speech_cut[1]", kind="speech", file_id="f1",
                          word_span=(1, 2), label="a", summary="a"),   # words[1:2] -> ms [300, 800)
        pass2.CutJudgment(source_ref="video_group[0]", kind="video", file_id="f1",
                          atom_ids=[0], label="b", summary="b"),        # atom 0 -> ms [400, 1000)
    ])
    err = pass2._no_cross_kind_ms_overlap(out, lattices)
    assert err is not None and "speech_cut[1]" in err and "video_group[0]" in err, err
    print("ok  test_no_cross_kind_ms_overlap_catches_a_speech_and_video_cut_overlapping")


# --------------------------------------------------------------------------
# to_pass2_cuts (replaces the old merge_identity_and_visual -- a single
# call's judgments need no merging, just a direct conversion)
# --------------------------------------------------------------------------

def test_to_pass2_cuts_converts_every_field():
    j = pass2.CutJudgment(
        source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 3),
        label="intro", summary="says hello", speaker="S0", on_camera=True,
        junk=False, natural_sound=True, channel="said",
        framing=pass2.Framing(rotation_deg=90.0), look=pass2.Look(graded=True),
        caption_zones=[(0.1, 0.1, 0.2, 0.1)],
        taste_fences=pass2.TasteFences(max_tasteful_speed=1.5), readability_ms=800,
        people=[pass2.PersonLook(description="a person", position="left", speaking=True)],
    )
    cuts = pass2.to_pass2_cuts([j])
    assert len(cuts) == 1
    cut = cuts[0]
    assert cut.source_ref == "speech_cut[0]" and cut.kind == "speech" and cut.word_span == (0, 3)
    assert cut.label == "intro" and cut.summary == "says hello" and cut.speaker == "S0"
    assert cut.on_camera is True and cut.natural_sound is True and cut.channel == "said"
    assert cut.framing.rotation_deg == 90.0
    assert cut.look.graded is True
    assert cut.caption_zones == [(0.1, 0.1, 0.2, 0.1)]
    assert cut.taste_fences.max_tasteful_speed == 1.5
    assert cut.readability_ms == 800
    # take fields start unset -- apply_take_groups is the only thing that sets them
    assert cut.take_group_id is None and cut.take_role is None
    # people flattened to plain dicts, same shape the old merge produced
    assert cut.people == [{"description": "a person", "appearance": pass2.Appearance().model_dump(),
                           "position": "left", "speaking": True}]
    print("ok  test_to_pass2_cuts_converts_every_field")


def test_to_pass2_cuts_empty_is_a_noop():
    assert pass2.to_pass2_cuts([]) == []
    print("ok  test_to_pass2_cuts_empty_is_a_noop")


# --------------------------------------------------------------------------
# Batching (pure size-based chunking, no co-location -- pass2_merge.plan.md
# Phase 1 moved take-grouping to code, so a take's members no longer need to
# share a batch)
# --------------------------------------------------------------------------

def test_build_pass2_batches_chunks_by_size():
    frames = [PlannedFrame("f1", i, "speech_cut", f"speech_cut[{i}]") for i in range(5)]
    batches = pass2.build_pass2_batches(Pass1Output(), frames, max_per_batch=2)
    assert batches == [["speech_cut[0]", "speech_cut[1]"], ["speech_cut[2]", "speech_cut[3]"],
                       ["speech_cut[4]"]], batches
    print("ok  test_build_pass2_batches_chunks_by_size")


def test_build_pass2_batches_groups_unique_refs_in_stable_order():
    frames = [PlannedFrame("f2", 50, "speech_cut", "speech_cut[1]"),
             PlannedFrame("f1", 200, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]")]  # 2 frames, same ref
    batches = pass2.build_pass2_batches(Pass1Output(), frames, max_per_batch=10)
    assert batches == [["speech_cut[0]", "speech_cut[1]"]], batches   # (file,ts) order, deduped by ref
    print("ok  test_build_pass2_batches_groups_unique_refs_in_stable_order")


def test_build_pass2_batches_no_longer_co_locates_take_members():
    # The key behavior change: even with a take_candidate linking two refs
    # across files, they are free to land in different batches -- there is
    # no co-location constraint anymore (take-grouping is deterministic code).
    frames = [PlannedFrame("f1", 0, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f2", 0, "speech_cut", "speech_cut[1]")]
    p1 = Pass1Output(
        speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 1), label="a"),
                     SpeechCut(file_id="f2", word_span=(0, 1), label="b")],
        take_candidates=[TakeCandidate(group_id="tg1", members=[
            TakeMember(file_id="f1", word_span=(0, 1)),
            TakeMember(file_id="f2", word_span=(0, 1)),
        ])])
    batches = pass2.build_pass2_batches(p1, frames, max_per_batch=1)
    assert batches == [["speech_cut[0]"], ["speech_cut[1]"]], batches
    print("ok  test_build_pass2_batches_no_longer_co_locates_take_members")


def test_build_pass2_batches_empty_frames_yield_no_batches():
    assert pass2.build_pass2_batches(Pass1Output(), []) == []
    print("ok  test_build_pass2_batches_empty_frames_yield_no_batches")


def test_build_pass2_batches_ignores_non_cut_refs():
    frames = [PlannedFrame("f1", 0, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f1", 5, "take", "take[tg1]")]   # e.g. a take-comparison frame reason
    batches = pass2.build_pass2_batches(Pass1Output(), frames)
    assert batches == [["speech_cut[0]"]], batches
    print("ok  test_build_pass2_batches_ignores_non_cut_refs")


# --------------------------------------------------------------------------
# Batch block rendering + orchestration
# --------------------------------------------------------------------------

def test_batch_blocks_skip_unresolved_images():
    # ref[0]'s frame is resolved, ref[1]'s is not -- sorted by (file, ref,
    # ts) the resolved one is IMG 1, the unresolved one is silently skipped
    # (not sent blank) and never gets a number at all.
    frames = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f1", 200, "speech_cut", "speech_cut[1]")]
    images = {("f1", 100): "ZmFrZQ=="}   # only one of the two resolved
    blocks = pass2.build_pass2_batch_blocks(frames, images)
    assert len(blocks) == 2, blocks   # one [caption, image] pair only
    assert blocks[0]["type"] == "text" and "IMG 1" in blocks[0]["text"]
    assert "0.1s" in blocks[0]["text"], blocks[0]
    assert blocks[1]["type"] == "image"
    print("ok  test_batch_blocks_skip_unresolved_images")


def test_batch_blocks_ordered_by_file_then_ref_then_ts():
    # perception_upgrade.plan.md Part B: sorted by (file_id, ref, ts_ms) --
    # NOT just (file_id, ts_ms) -- so a ref's early/late pair always lands
    # adjacent in the sequence, never interleaved with another ref's frame
    # that merely happens to fall between them in time.
    frames = [PlannedFrame("f2", 50, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f1", 200, "speech_cut", "speech_cut[0]"),
             PlannedFrame("f1", 100, "speech_cut", "speech_cut[1]")]
    images = {("f2", 50): "a", ("f1", 200): "b", ("f1", 100): "c"}
    blocks = pass2.build_pass2_batch_blocks(frames, images)
    captions = [b["text"] for b in blocks if b["type"] == "text"]
    # f1/speech_cut[0] (200ms) sorts before f1/speech_cut[1] (100ms) --
    # grouped by REF first, not by timestamp -- then f2/speech_cut[0].
    assert "clip f1, 0.2s" in captions[0], captions
    assert "clip f1, 0.1s" in captions[1], captions
    assert "clip f2, 0.1s" in captions[2], captions
    print("ok  test_batch_blocks_ordered_by_file_then_ref_then_ts")


def test_batch_blocks_label_early_late_phase_only_only_gets_no_suffix():
    frames = [
        PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]", "early"),
        PlannedFrame("f1", 300, "speech_cut", "speech_cut[0]", "late"),
        PlannedFrame("f1", 500, "speech_cut", "speech_cut[1]", "only"),
    ]
    images = {("f1", 100): "a", ("f1", 300): "b", ("f1", 500): "c"}
    blocks = pass2.build_pass2_batch_blocks(frames, images)
    captions = [b["text"] for b in blocks if b["type"] == "text"]
    assert captions[0].endswith("speech_cut[0] (early)"), captions
    assert captions[1].endswith("speech_cut[0] (late)"), captions
    assert captions[2].endswith("speech_cut[1]"), captions   # no suffix for "only"
    print("ok  test_batch_blocks_label_early_late_phase_only_only_gets_no_suffix")


def test_run_pass2_batch_raises_when_no_images_resolve():
    frames = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]")]
    try:
        pass2.run_pass2_batch([("f1", "a.mp4", 10000, _lat("f1"))], Pass1Output(), frames, {})
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ok  test_run_pass2_batch_raises_when_no_images_resolve")


def test_run_pass2_batch_calls_complete_with_pass2_stage_and_cached_prefix():
    # Forces the anthropic provider explicitly -- perception_upgrade.plan.md
    # Part A made "gemini" the default, and this test exercises the
    # Anthropic-specific SDK client/cache_control wire shape (see
    # test_ingest_gemini.py for the gemini-provider equivalent).
    good = {"cuts": [{
        "source_ref": "speech_cut[0]", "kind": "speech", "file_id": "f1",
        "word_span": [0, 2], "label": "intro", "summary": "hello there",
    }]}
    fake, orig = _with_fake_client([FakeResponse([FakeBlock("tool_use", ic._TOOL_NAME, good)])])
    frames = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]")]
    pass1_output = Pass1Output(speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 2), label="intro")])
    settings = get_settings()
    orig_provider = settings.ingest_pass2_provider
    settings.ingest_pass2_provider = "anthropic"
    try:
        result = pass2.run_pass2_batch(
            [("f1", "a.mp4", 10000, _lat("f1"))], pass1_output, frames, {("f1", 100): "ZmFrZQ=="},
        )
    finally:
        ic._sdk_client = orig
        settings.ingest_pass2_provider = orig_provider

    call = fake.messages.calls[0]
    assert call["tools"][0]["input_schema"] == pass2.Pass2BatchOutput.model_json_schema()
    content = call["messages"][0]["content"]
    # cached prefix (pass1 blocks + rendered pass1 output) ends with the cache
    # breakpoint; the image caption/image pair rides after it, uncached.
    assert content[-3]["cache_control"] == {"type": "ephemeral"}, content
    assert "cache_control" not in content[-2]
    assert "cache_control" not in content[-1]
    assert content[-2]["type"] == "text" and "IMG 1" in content[-2]["text"]
    assert content[-1]["type"] == "image"
    parsed = pass2.Pass2BatchOutput.model_validate(result.data)
    assert parsed.cuts[0].source_ref == "speech_cut[0]"
    print("ok  test_run_pass2_batch_calls_complete_with_pass2_stage_and_cached_prefix")


def test_render_pass1_output_omits_take_candidates_section():
    # pass2_merge.plan.md: take-grouping is code-owned now, so the model no
    # longer needs (or gets) TAKE CANDIDATES context.
    from app.services.l3.pass1 import render_pass1_output
    p1 = Pass1Output(
        speech_cuts=[SpeechCut(file_id="f1", word_span=(0, 1), label="a")],
        take_candidates=[TakeCandidate(group_id="tg1", members=[
            TakeMember(file_id="f1", word_span=(0, 1)),
        ])],
    )
    text = render_pass1_output(p1)
    assert "TAKE CANDIDATES" not in text, text
    assert "tg1" not in text, text
    print("ok  test_render_pass1_output_omits_take_candidates_section")


def main():
    test_pass2cut_constructs_with_all_fields()
    test_apply_junk_suspects_hides_a_contained_speech_cut()
    test_apply_junk_suspects_ignores_partial_overlap()
    test_apply_take_groups_outlook_path_matches_old_apply_outlook_roles()
    test_apply_take_groups_non_outlook_group_gets_take_role()
    test_apply_take_groups_then_enforce_take_winner_crowns_the_best()
    test_apply_take_groups_untouched_cut_stays_unset()
    test_apply_take_groups_empty_take_candidates_is_a_noop()
    test_cutjudgment_has_no_take_fields()
    test_locators_are_optional_at_parse_time_and_backfilled()
    test_kind_alias_is_normalized_at_parse_time()
    test_channel_alias_is_normalized_at_parse_time()
    test_backfill_leaves_split_video_groups_to_the_model()
    test_valid_cuts_round_trip()
    test_pass2_batch_output_rejects_an_unexpected_wrapper_key()
    test_cutjudgment_defaults_are_safe()
    test_no_duplicate_atoms_passes_when_every_atom_is_used_once()
    test_no_duplicate_atoms_catches_an_atom_split_across_two_cuts()
    test_kind_is_derived_from_source_ref_not_the_models_field()
    test_no_overlapping_word_spans_catches_a_duplicate_span()
    test_pass2_semantic_checks_combines_all_checks()
    test_drop_out_of_batch_cuts_filters_strays_and_fixes_file_id()
    test_source_refs_exist_rejects_an_invented_ref()
    test_no_cross_kind_ms_overlap_catches_a_speech_and_video_cut_overlapping()
    test_to_pass2_cuts_converts_every_field()
    test_to_pass2_cuts_empty_is_a_noop()
    test_build_pass2_batches_chunks_by_size()
    test_build_pass2_batches_groups_unique_refs_in_stable_order()
    test_build_pass2_batches_no_longer_co_locates_take_members()
    test_build_pass2_batches_empty_frames_yield_no_batches()
    test_build_pass2_batches_ignores_non_cut_refs()
    test_batch_blocks_skip_unresolved_images()
    test_batch_blocks_ordered_by_file_then_ref_then_ts()
    test_batch_blocks_label_early_late_phase_only_only_gets_no_suffix()
    test_run_pass2_batch_raises_when_no_images_resolve()
    test_run_pass2_batch_calls_complete_with_pass2_stage_and_cached_prefix()
    test_render_pass1_output_omits_take_candidates_section()
    print("\nall pass2 (merged) tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
