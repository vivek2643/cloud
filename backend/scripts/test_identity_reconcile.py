"""
Tests for cross-cut person unification (app.services.l3.identity.reconcile)
-- no DB, no network, synthetic occurrences only.

Run:  .venv/bin/python scripts/test_identity_reconcile.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3.identity import reconcile as rc  # noqa: E402


def _app(**kw):
    base = {"apparent_gender": None, "apparent_age_band": None, "hair": None,
           "hair_color": None, "facial_hair": None, "glasses": None,
           "skin_tone": None, "build": None}
    base.update(kw)
    return base


def _occ(file_id, ref, idx, appearance, description="a person", is_speech=True, area=0.1):
    return rc.Occurrence(file_id=file_id, source_ref=ref, person_index=idx,
                         appearance=appearance, description=description,
                         is_speech_cut=is_speech, subject_box_area=area)


def test_build_fingerprint_majority_vote_and_tie_leaves_unset():
    apps = [_app(hair="bald"), _app(hair="bald"), _app(hair="short")]
    fp = rc.build_fingerprint(apps)
    assert fp["hair"] == "bald", fp
    tie = [_app(build="slim"), _app(build="heavy")]
    fp2 = rc.build_fingerprint(tie)
    assert "build" not in fp2, fp2
    fp3 = rc.build_fingerprint([_app(glasses="unsure"), _app(glasses=None)])
    assert "glasses" not in fp3, fp3
    print("ok  test_build_fingerprint_majority_vote_and_tie_leaves_unset")


def test_cluster_occurrences_merges_zero_disagreement_enough_shared():
    fps = {
        ("f1", "speech_cut[0]", 0): {"apparent_gender": "male", "hair": "bald", "facial_hair": "beard"},
        ("f2", "speech_cut[0]", 0): {"apparent_gender": "male", "hair": "bald", "facial_hair": "beard"},
    }
    occ_person = rc.cluster_occurrences(fps)
    assert occ_person[("f1", "speech_cut[0]", 0)] == occ_person[("f2", "speech_cut[0]", 0)], occ_person
    print("ok  test_cluster_occurrences_merges_zero_disagreement_enough_shared")


def test_cluster_occurrences_keeps_disagreeing_occurrences_separate():
    fps = {
        ("f1", "speech_cut[0]", 0): {"apparent_gender": "male", "hair": "bald", "facial_hair": "beard"},
        ("f2", "speech_cut[0]", 0): {"apparent_gender": "female", "hair": "long", "facial_hair": "none"},
    }
    occ_person = rc.cluster_occurrences(fps)
    assert occ_person[("f1", "speech_cut[0]", 0)] != occ_person[("f2", "speech_cut[0]", 0)], occ_person
    print("ok  test_cluster_occurrences_keeps_disagreeing_occurrences_separate")


def test_cluster_occurrences_keeps_sparse_occurrences_separate():
    # Only 2 shared fields (< MIN_SHARED_FIELDS=3) even though zero disagreement.
    fps = {
        ("f1", "speech_cut[0]", 0): {"apparent_gender": "male", "hair": "bald"},
        ("f2", "speech_cut[0]", 0): {"apparent_gender": "male", "hair": "bald"},
    }
    occ_person = rc.cluster_occurrences(fps)
    assert occ_person[("f1", "speech_cut[0]", 0)] != occ_person[("f2", "speech_cut[0]", 0)], occ_person
    print("ok  test_cluster_occurrences_keeps_sparse_occurrences_separate")


def test_build_persons_marks_every_reconciled_person_a_cast_member():
    occs = []
    for i in range(3):    # person A: 3 appearances
        occs.append(_occ("f1", f"speech_cut[{i}]", 0,
                         _app(apparent_gender="male", hair="bald", facial_hair="beard")))
    for i in range(2):    # person B: 2 appearances
        occs.append(_occ("f1", f"speech_cut[{i + 10}]", 0,
                         _app(apparent_gender="female", hair="long", facial_hair="none")))
    for i in range(2):    # person C: 2 appearances
        occs.append(_occ("f1", f"speech_cut[{i + 20}]", 0,
                         _app(apparent_gender="male", hair="short", facial_hair="moustache")))
    # person D: 1 appearance -- the least prominent, but with an uncapped cast
    # table it is STILL a full cast member (no top-N cut).
    occs.append(_occ("f1", "speech_cut[30]", 0,
                     _app(apparent_gender="female", hair="short", facial_hair="stubble")))

    fps = {(o.file_id, o.source_ref, o.person_index): rc.build_fingerprint([o.appearance]) for o in occs}
    occ_person = rc.cluster_occurrences(fps)
    persons = rc.build_persons(occ_person, occs)
    assert len(persons) == 4, persons
    majors = {pid for pid, p in persons.items() if p.is_major}
    assert majors == set(persons.keys()), majors
    print("ok  test_build_persons_marks_every_reconciled_person_a_cast_member")


def test_visible_persons_by_cut_groups_distinct_people_per_cut():
    occ_person = {
        ("f1", "speech_cut[0]", 0): "P0",
        ("f1", "speech_cut[0]", 1): "P1",
        ("f1", "speech_cut[1]", 0): "P0",
    }
    vis = rc.visible_persons_by_cut(occ_person)
    assert vis[("f1", "speech_cut[0]")] == ["P0", "P1"], vis
    assert vis[("f1", "speech_cut[1]")] == ["P0"], vis
    print("ok  test_visible_persons_by_cut_groups_distinct_people_per_cut")


def test_reconcile_excludes_crowd_cut_occurrences():
    occs = [_occ("f1", "speech_cut[0]", i,
                _app(apparent_gender="male", hair="bald", facial_hair="beard"))
           for i in range(rc.CROWD_SIZE + 1)]
    result = rc.reconcile(occs)
    assert result["persons"] == [], result["persons"]
    assert result["visible_persons"] == {}, result["visible_persons"]
    print("ok  test_reconcile_excludes_crowd_cut_occurrences")


def test_reconcile_end_to_end_two_people_in_one_cut():
    occs = [
        _occ("f1", "speech_cut[0]", 0, _app(apparent_gender="male", hair="bald", facial_hair="beard")),
        _occ("f1", "speech_cut[0]", 1, _app(apparent_gender="female", hair="long", facial_hair="none")),
    ]
    result = rc.reconcile(occs)
    assert len(result["persons"]) == 2, result["persons"]
    vis = result["visible_persons"][("f1", "speech_cut[0]")]
    assert len(vis) == 2, vis
    print("ok  test_reconcile_end_to_end_two_people_in_one_cut")


def test_reconcile_sparse_occurrence_stays_its_own_unmerged_person():
    # An occurrence with too little evidence to compare against anyone still
    # gets its OWN person id -- over-split is the safe failure, never dropped.
    occs = [_occ("f1", "speech_cut[0]", 0, _app())]   # nothing set at all
    result = rc.reconcile(occs)
    assert len(result["persons"]) == 1, result["persons"]
    print("ok  test_reconcile_sparse_occurrence_stays_its_own_unmerged_person")


def main():
    test_build_fingerprint_majority_vote_and_tie_leaves_unset()
    test_cluster_occurrences_merges_zero_disagreement_enough_shared()
    test_cluster_occurrences_keeps_disagreeing_occurrences_separate()
    test_cluster_occurrences_keeps_sparse_occurrences_separate()
    test_build_persons_marks_every_reconciled_person_a_cast_member()
    test_visible_persons_by_cut_groups_distinct_people_per_cut()
    test_reconcile_excludes_crowd_cut_occurrences()
    test_reconcile_end_to_end_two_people_in_one_cut()
    test_reconcile_sparse_occurrence_stays_its_own_unmerged_person()
    print("\nall identity-reconcile tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
