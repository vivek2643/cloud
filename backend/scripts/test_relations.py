#!/usr/bin/env python3
"""Tests for cross-clip identity (l3.relations) and the program clock field
(l3.program_clock). Identity is derived from matched SPEECH alone -- lines are
content-clustered across clips (label-agnostic), voted into per-pair one-to-one
voice correspondences, and unioned under the hard same-clip-distinctness
constraint (no appearance/offset fallbacks). Pure (no DB / no VLM). Run:
    PYTHONPATH=. .venv/bin/python scripts/test_relations.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import program_clock as pc  # noqa: E402
from app.services.l3 import relations as rel  # noqa: E402
from app.services.l3.takes import Attempt  # noqa: E402

FA = "aaaaaaaa-0000-0000-0000-000000000000"
FB = "bbbbbbbb-0000-0000-0000-000000000000"
FC = "cccccccc-0000-0000-0000-000000000000"

LINE1 = "the first shared sentence here today"
LINE2 = "another common line we all spoke"


def _att(fid, start, end, speaker="S0", text=LINE1):
    return Attempt(attempt_id=f"{fid[:8]}:u:{start}", file_id=fid, unit_id="u",
                   start_ms=start, end_ms=end, kind="speech",
                   content_key=text, text=text, speaker=speaker,
                   tokens=frozenset(text.split()))


# --------------------------------------------------------------------------
# IDENTITY: one human across files, from matched speech (>= MIN_LINK_VOTES lines)
# --------------------------------------------------------------------------

def _clips():
    """Two clips: FA has p1 visibly speaking (voice S0); FB has p2 visibly
    speaking (voice S1). The same line matched across them = same human."""
    perc_a = {"file_id": FA,
              "persons": [{"local_id": "p1", "role": "host",
                           "canonical_description": "tall man, beard"}],
              "speaking": [{"start_ms": 1000, "end_ms": 3000, "subject": "p1"}]}
    words_a = [{"start_ms": 1000, "end_ms": 3000, "text": LINE1, "speaker": "S0"}]
    perc_b = {"file_id": FB,
              "persons": [{"local_id": "p2", "role": "host"}],
              "speaking": [{"start_ms": 13400, "end_ms": 15400, "subject": "p2"}]}
    words_b = [{"start_ms": 13400, "end_ms": 15400, "text": LINE1, "speaker": "S1"}]
    return {FA: (perc_a, words_a), FB: (perc_b, words_b)}


def test_identity_links_voices_and_dresses_with_cast():
    """Two shared lines both delivered by FA:S0 and FB:S1 -> one global person,
    dressed from the per-clip cast (person/role + appearance)."""
    attempts = [
        _att(FA, 1000, 3000, "S0", LINE1), _att(FB, 13400, 15400, "S1", LINE1),
        _att(FA, 20000, 22000, "S0", LINE2), _att(FB, 32400, 34400, "S1", LINE2),
    ]
    idents = rel.derive_identities(attempts, _clips())
    assert len(idents) == 1, idents
    members = {(m["file"], m["voice"], m["person"]) for m in idents[0]["members"]}
    assert (FA, "S0", "p1") in members and (FB, "S1", "p2") in members, members
    assert idents[0]["description"] == "tall man, beard", idents[0]
    print("ok  matched voices across files become ONE identity with a face+role")


def test_identity_needs_corroborating_lines():
    """A SINGLE loose content match is not enough to invent a cross-clip link."""
    attempts = [_att(FA, 1000, 3000, "S0", LINE1),
                _att(FB, 13400, 15400, "S1", LINE1)]
    assert rel.derive_identities(attempts, _clips()) == []
    print("ok  one shared line is below the corroboration threshold")


def test_identity_requires_cross_file_evidence():
    """Two voices matched only WITHIN one file never form a (cross-clip)
    identity."""
    attempts = [_att(FA, 1000, 3000, "S0", LINE1),
                _att(FA, 50000, 52000, "S0", LINE1),
                _att(FA, 60000, 62000, "S0", LINE2)]
    assert rel.derive_identities(attempts, _clips()) == []
    print("ok  same-file matches alone form no cross-clip identity")


def test_identity_never_merges_two_voices_of_one_clip():
    """The hard constraint: two distinct voices in the SAME clip must land in
    DIFFERENT global people. Two people talk in every clip and a third clip
    SWAPS the diarization labels -- the collapse trap -- yet they stay apart."""
    a1, a2 = LINE1, "morning everyone welcome back to the studio now"
    b1, b2 = LINE2, "thanks so much for having me here today"
    perc = lambda f: {"file_id": f, "persons": [
        {"local_id": "p1", "role": "host"}, {"local_id": "p2", "role": "guest"}]}
    words = lambda: [
        {"start_ms": 0, "end_ms": 2000, "text": a1, "speaker": "S0"},
        {"start_ms": 3000, "end_ms": 5000, "text": b1, "speaker": "S1"}]
    clips = {FA: (perc(FA), words()), FB: (perc(FB), words()), FC: (perc(FC), words())}
    attempts = []
    # Person A speaks a1+a2 in all three clips; person B speaks b1+b2. Voice
    # labels are SWAPPED in the noisy clip C (A=S1, B=S0) on purpose.
    for f, sA, sB in [(FA, "S0", "S1"), (FB, "S0", "S1"), (FC, "S1", "S0")]:
        attempts += [_att(f, 0, 2000, sA, a1), _att(f, 100000, 102000, sA, a2),
                     _att(f, 3000, 5000, sB, b1), _att(f, 103000, 105000, sB, b2)]
    idents = rel.derive_identities(attempts, clips)
    # Exactly two people, and no identity mixes two voices from the same clip.
    assert len(idents) == 2, idents
    for ident in idents:
        seen = {}
        for m in ident["members"]:
            assert m["file"] not in seen, ("collapsed two voices of one clip", ident)
            seen[m["file"]] = m["voice"]
    print("ok  two voices of one clip never collapse into one person")


def test_render_relations_reads_clean():
    attempts = [
        _att(FA, 1000, 3000, "S0", LINE1), _att(FB, 13400, 15400, "S1", LINE1),
        _att(FA, 20000, 22000, "S0", LINE2), _att(FB, 32400, 34400, "S1", LINE2),
    ]
    relations = {"identities": rel.derive_identities(attempts, _clips())}
    text = rel.render_relations(relations)
    assert "PEOPLE OF THE SHOOT" in text and "G1" in text, text
    assert "co-temporal" not in text and "offset" not in text.lower(), text
    assert "bbbbbbbb" in text and "aaaaaaaa" in text, text
    assert rel.render_relations({}) == ""
    print("ok  relations render into a clean identity-only digest")
    print("--- sample relations ---\n" + text + "\n------------------------")


def test_identity_registry_lookups():
    attempts = [
        _att(FA, 1000, 3000, "S0", LINE1), _att(FB, 13400, 15400, "S1", LINE1),
        _att(FA, 20000, 22000, "S0", LINE2), _att(FB, 32400, 34400, "S1", LINE2),
    ]
    relations = {"identities": rel.derive_identities(attempts, _clips())}
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
