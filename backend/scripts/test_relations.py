#!/usr/bin/env python3
"""Tests for cross-clip relations (l3.relations) and the program clock field
(l3.program_clock) -- the shoot-level substrate: co-temporal offsets + global
person identity derived from existing take groups / cast, and the program-side
cut field. Pure (no DB / no VLM). Run:
    PYTHONPATH=. .venv/bin/python scripts/test_relations.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import program_clock as pc  # noqa: E402
from app.services.l3 import relations as rel  # noqa: E402
from app.services.l3.takes import Attempt, TakeGroup  # noqa: E402

FA = "aaaaaaaa-0000-0000-0000-000000000000"
FB = "bbbbbbbb-0000-0000-0000-000000000000"


def _att(fid, start, end, speaker="S0", text="the same line here"):
    return Attempt(attempt_id=f"{fid[:8]}:u:{start}", file_id=fid, unit_id="u",
                   start_ms=start, end_ms=end, kind="speech",
                   content_key=text, text=text, speaker=speaker,
                   tokens=frozenset(text.split()))


def _group(gid, attempts):
    return TakeGroup(group_id=gid, content_key="k", attempts=attempts)


# --------------------------------------------------------------------------
# TIME: co-temporal offsets
# --------------------------------------------------------------------------

def test_offsets_agreeing_deltas_mean_co_temporal():
    """Three matched lines all ~12.4s apart -> the clips are the SAME live
    moment at that offset."""
    groups = [
        _group("tg1", [_att(FA, 1000, 3000), _att(FB, 13400, 15400)]),
        _group("tg2", [_att(FA, 20000, 22000), _att(FB, 32500, 34500)]),
        _group("tg3", [_att(FA, 40000, 42000), _att(FB, 52300, 54300)]),
    ]
    offs = rel.derive_offsets(groups)
    assert len(offs) == 1, offs
    o = offs[0]
    assert (o["file_a"], o["file_b"]) == (FA, FB), o
    assert abs(o["offset_ms"] - 12400) <= rel.OFFSET_AGREE_MS, o
    assert o["matches"] == 3 and o["confidence"] > 0.5, o
    print("ok  agreeing start-time deltas yield a co-temporal offset")


def test_offsets_scattered_deltas_are_retakes_not_co_temporal():
    """The same line delivered again minutes apart (true retakes) must NOT
    produce a time relation -- that case belongs to the dup groups."""
    groups = [
        _group("tg1", [_att(FA, 1000, 3000), _att(FB, 5000, 7000)]),      # +4s
        _group("tg2", [_att(FA, 20000, 22000), _att(FB, 95000, 97000)]),  # +75s
    ]
    assert rel.derive_offsets(groups) == [], rel.derive_offsets(groups)
    print("ok  scattered deltas (retakes) produce no offset relation")


def test_offsets_need_two_matches():
    groups = [_group("tg1", [_att(FA, 1000, 3000), _att(FB, 13400, 15400)])]
    assert rel.derive_offsets(groups) == []
    print("ok  a single matched line is not enough to claim co-temporal")


# --------------------------------------------------------------------------
# IDENTITY: one human across files
# --------------------------------------------------------------------------

def _clips():
    """Two clips: FA has p1 visibly speaking (voice S0); FB has p2 visibly
    speaking (voice S1). The same line matched across them = same human."""
    perc_a = {"file_id": FA,
              "persons": [{"local_id": "p1", "role": "host",
                           "canonical_description": "tall man, beard"}],
              "speaking": [{"start_ms": 1000, "end_ms": 3000, "subject": "p1"}]}
    words_a = [{"start_ms": 1000, "end_ms": 3000, "text": "the same line here",
                "speaker": "S0"}]
    perc_b = {"file_id": FB,
              "persons": [{"local_id": "p2", "role": "host"}],
              "speaking": [{"start_ms": 13400, "end_ms": 15400, "subject": "p2"}]}
    words_b = [{"start_ms": 13400, "end_ms": 15400, "text": "the same line here",
                "speaker": "S1"}]
    return {FA: (perc_a, words_a), FB: (perc_b, words_b)}


def test_identity_links_voices_and_dresses_with_cast():
    groups = [_group("tg1", [_att(FA, 1000, 3000, speaker="S0"),
                             _att(FB, 13400, 15400, speaker="S1")])]
    idents = rel.derive_identities(groups, _clips())
    assert len(idents) == 1, idents
    members = {(m["file"], m["voice"], m["person"]) for m in idents[0]["members"]}
    assert (FA, "S0", "p1") in members and (FB, "S1", "p2") in members, members
    assert idents[0]["description"] == "tall man, beard", idents[0]
    print("ok  matched voices across files become ONE identity with a face+role")


def test_identity_requires_cross_file_evidence():
    """Two voices matched only WITHIN one file never form a (cross-clip)
    identity."""
    groups = [_group("tg1", [_att(FA, 1000, 3000, speaker="S0"),
                             _att(FA, 50000, 52000, speaker="S0")])]
    assert rel.derive_identities(groups, _clips()) == []
    print("ok  same-file matches alone form no cross-clip identity")


def test_render_relations_reads_clean():
    groups = [
        _group("tg1", [_att(FA, 1000, 3000), _att(FB, 13400, 15400, speaker="S1")]),
        _group("tg2", [_att(FA, 20000, 22000), _att(FB, 32400, 34400, speaker="S1")]),
    ]
    relations = {"offsets": rel.derive_offsets(groups),
                 "identities": rel.derive_identities(groups, _clips())}
    text = rel.render_relations(relations)
    # Identity-only digest, no time/offset ontology anywhere.
    assert "PEOPLE OF THE SHOOT" in text and "G1" in text, text
    assert "co-temporal" not in text and "offset" not in text.lower(), text
    assert "bbbbbbbb" in text and "aaaaaaaa" in text, text
    assert rel.render_relations({}) == ""
    print("ok  relations render into a clean identity-only digest")
    print("--- sample relations ---\n" + text + "\n------------------------")


def test_identity_trait_fallback_links_silent_lookalikes():
    """Two clips that share NO spoken line still link one person when the VLM
    appearance descriptions overlap -- as a LOW-confidence, trait-based id."""
    perc_a = {"file_id": FA, "persons": [
        {"local_id": "p1", "role": "host",
         "canonical_description": "bald head gray beard dark jacket"}]}
    perc_b = {"file_id": FB, "persons": [
        {"local_id": "p3", "role": "host",
         "canonical_description": "gray beard bald head dark jacket glasses"}]}
    clips = {FA: (perc_a, []), FB: (perc_b, [])}
    idents = rel.derive_identities([], clips)      # no take groups at all
    assert len(idents) == 1, idents
    assert idents[0]["basis"] == "traits", idents[0]
    assert idents[0]["confidence"] < 0.5, idents[0]
    files = {m["file"] for m in idents[0]["members"]}
    assert files == {FA, FB}, idents[0]
    text = rel.render_relations({"identities": idents})
    assert "appearance-matched" in text and "low confidence" in text, text
    print("ok  trait fallback links silent look-alikes at low confidence")


def test_identity_registry_lookups():
    groups = [_group("tg1", [_att(FA, 1000, 3000, speaker="S0"),
                             _att(FB, 13400, 15400, speaker="S1")])]
    relations = {"identities": rel.derive_identities(groups, _clips())}
    assert rel.global_id_of(relations, FA, "S0") == "G1"
    assert rel.global_id_of(relations, FB, "p2") == "G1"   # by person handle too
    assert rel.global_id_of(relations, FA, "S9") is None
    loc = rel.local_of(relations, FB, "G1")
    assert loc and loc["voice"] == "S1", loc
    print("ok  registry aliases voice/person <-> global id both directions")


# --------------------------------------------------------------------------
# PROGRAM CLOCK: the second field
# --------------------------------------------------------------------------

def test_program_field_none_without_sources():
    assert pc.build_program_field(duration_ms=60000) is None
    assert pc.snap_program_ms(None, 1234) == 1234       # no opinion -> pass-through
    assert pc.snap_program_ms(None, None) is None
    print("ok  no program-side sources -> no field, anchors pass through")


def test_program_field_snaps_to_downbeat():
    fld = pc.build_program_field(duration_ms=10000,
                                 beats_ms=[500, 1000, 1500, 2000],
                                 downbeats_ms=[2000])
    assert fld is not None
    # 1800 is 200ms from the 2000 downbeat -> pulled onto it
    assert pc.snap_program_ms(fld, 1800) == 2000
    # far from any beat (win 400) -> stays near where it was
    assert abs(pc.snap_program_ms(fld, 5000) - 5000) <= pc.PROGRAM_SNAP_WIN_MS
    print("ok  program anchors snap to the beat grid within the window")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall relations + program-clock tests passed")
