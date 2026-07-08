#!/usr/bin/env python3
"""Tests for the reconciled shoot cast (l3.relations).

Identity fuses TWO signals, each doing its own job: SPEECH clusters voices
(label-agnostic line matching -> per-pair one-to-one, dominance-gated), APPEARANCE
clusters faces, and a HIGH-CONFIDENCE per-clip A/V link bridges a face cluster to
a voice cluster. `on_camera` is DERIVED (a voice is on camera in a clip iff its
person owns that clip's on-camera face), so one clip's wrong-but-confident A/V
link cannot glue two people together -- the correct face that owns the voice
elsewhere wins, and the bad clip's true on-camera voice is recovered by
elimination. Pure (no DB / no VLM). Run:
    PYTHONPATH=. .venv/bin/python scripts/test_relations.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import relations as rel  # noqa: E402
from app.services.l3.takes import Attempt  # noqa: E402

# Two co-temporal camera pairs (segment A + segment B), exactly the real shape.
A_CURLY = "aaaaaaaa-0000-0000-0000-000000000001"   # curly on camera, seg A
A_BALD = "aaaaaaaa-0000-0000-0000-000000000002"    # bald  on camera, seg A
B_CURLY = "bbbbbbbb-0000-0000-0000-000000000001"   # curly on camera, seg B
B_BALD = "bbbbbbbb-0000-0000-0000-000000000002"    # bald  on camera, seg B (inflated link)

CURLY_FACE = "a man with dark curly hair and a thick moustache"
BALD_FACE = "a man with a bald head and a full grey beard"

# Distinct multi-token lines (>= MIN_SHARED_TOKENS=4, near-zero cross-line overlap).
A1 = "so tell me about your very first startup venture"
A2 = "we launched the whole product during a snowy winter"
A3 = "honestly that must have been incredibly difficult back then"
A4 = "did the early investors believe your crazy vision immediately"
B1 = "how did the second funding round actually come together"
B2 = "our biggest customer nearly walked away that whole year"
B3 = "what advice would you give much younger founders today"
B4 = "which single decision changed the entire trajectory completely afterwards"


def _att(fid, start, end, voice, text):
    return Attempt(attempt_id=f"{fid[:8]}:u:{start}", file_id=fid, unit_id="u",
                   start_ms=start, end_ms=end, kind="speech",
                   content_key=text, text=text, speaker=voice,
                   tokens=frozenset(text.split()))


def _clip(fid, desc, lines, speaking_spans):
    """(perception, words) for one clip. `lines` = [(start, end, text, voice)];
    `speaking_spans` = the VLM's on-camera speaking spans for the sole person p1
    (controls the A/V link -- set them over the OTHER voice to simulate an
    inflated 'visibly speaking' sensor)."""
    perc = {"file_id": fid,
            "persons": [{"local_id": "p1", "role": "host",
                         "canonical_description": desc}],
            "speaking": [{"start_ms": a, "end_ms": b, "subject": "p1"}
                         for a, b in speaking_spans]}
    words = [{"start_ms": s, "end_ms": e, "text": t, "speaker": v}
             for s, e, t, v in lines]
    return (perc, words)


def _atts(fid, lines):
    return [_att(fid, s, e, v, t) for s, e, t, v in lines]


def _shoot():
    """Build the four-clip shoot. curly speaks a1,a2 / b1,b2; bald speaks a3,a4 /
    b3,b4. Voice labels DIFFER per clip (identity must not lean on them). B_BALD's
    A/V link is INFLATED -- its on-camera bald face links to the loud off-camera
    curly voice (a real sensor failure)."""
    # Segment A -----------------------------------------------------------
    a_curly_lines = [(1000, 3000, A1, "X0"), (4000, 6000, A2, "X0"),
                     (7000, 9000, A3, "X1"), (10000, 12000, A4, "X1")]
    a_bald_lines = [(1000, 3000, A1, "S1"), (4000, 6000, A2, "S1"),
                    (7000, 9000, A3, "S0"), (10000, 12000, A4, "S0")]
    clips = {
        A_CURLY: _clip(A_CURLY, CURLY_FACE, a_curly_lines,
                       [(1000, 3000), (4000, 6000)]),          # curly on-cam
        A_BALD: _clip(A_BALD, BALD_FACE, a_bald_lines,
                      [(7000, 9000), (10000, 12000)]),         # bald on-cam
    }
    attempts = _atts(A_CURLY, a_curly_lines) + _atts(A_BALD, a_bald_lines)
    # Segment B -----------------------------------------------------------
    b_curly_lines = [(1000, 3000, B1, "P0"), (4000, 6000, B2, "P0"),
                     (7000, 9000, B3, "P1"), (10000, 12000, B4, "P1")]
    b_bald_lines = [(1000, 3000, B1, "S0"), (4000, 6000, B2, "S0"),
                    (7000, 9000, B3, "S1"), (10000, 12000, B4, "S1")]
    clips[B_CURLY] = _clip(B_CURLY, CURLY_FACE, b_curly_lines,
                           [(1000, 3000), (4000, 6000)])       # curly on-cam
    # INFLATED: bald's "speaking" spans cover curly's b1,b2 (loud) + only part of
    # his own b3 -> dominant voice is S0 (curly), so the link is bald->S0 at ~0.67
    # confidence: confident, and WRONG.
    clips[B_BALD] = _clip(B_BALD, BALD_FACE, b_bald_lines, [(1000, 8000)])
    attempts += _atts(B_CURLY, b_curly_lines) + _atts(B_BALD, b_bald_lines)
    return attempts, clips


# --------------------------------------------------------------------------
# RECONCILIATION
# --------------------------------------------------------------------------

def _by_gid(idents):
    """{description-word 'curly'/'bald' -> identity} for readable assertions."""
    out = {}
    for i in idents:
        d = (i.get("description") or "").lower()
        key = "curly" if "curly" in d else "bald" if "bald" in d else i["global_id"]
        out[key] = i
    return out


def _oncam_files(ident):
    return {m["file"] for m in ident["members"] if m.get("on_camera") is True}


def _offcam_files(ident):
    return {m["file"] for m in ident["members"] if m.get("on_camera") is False}


def test_reconcile_two_people_across_four_clips():
    """Two cameras of a two-person conversation, per-clip labels swapped: exactly
    two global people, each on camera in their OWN clips and heard off-camera in
    the other's -- with the right faces."""
    attempts, clips = _shoot()
    idents = rel.derive_identities(attempts, clips)
    assert len(idents) == 2, idents
    g = _by_gid(idents)
    assert set(g) == {"curly", "bald"}, g
    assert _oncam_files(g["curly"]) == {A_CURLY, B_CURLY}, g["curly"]
    assert _oncam_files(g["bald"]) == {A_BALD, B_BALD}, g["bald"]
    assert _offcam_files(g["curly"]) == {A_BALD, B_BALD}
    assert _offcam_files(g["bald"]) == {A_CURLY, B_CURLY}
    print("ok  two cameras of two people -> two people, right faces on right clips")


def test_inflated_link_is_overridden_not_collapsed():
    """B_BALD's confident-but-wrong A/V link (bald face -> curly's loud voice) must
    NOT fuse the two people. The correct face that owns that voice elsewhere wins;
    B_BALD's true on-camera voice is recovered by elimination; a NOTE is raised."""
    attempts, clips = _shoot()
    recon = rel._reconcile(attempts, clips)
    idents, warnings = recon["identities"], recon["warnings"]
    assert len(idents) == 2, idents                     # NOT collapsed into one
    g = _by_gid(idents)
    # bald is on camera in B_BALD (its face) even though the sensor mislinked it.
    assert B_BALD in _oncam_files(g["bald"]), g["bald"]
    # curly is only HEARD in B_BALD (its loud off-camera voice), never shown there.
    assert B_BALD in _offcam_files(g["curly"]), g["curly"]
    assert any("bbbbbbbb" in w and "another person" in w for w in warnings), warnings
    print("ok  a wrong-but-confident A/V link is overridden, flagged, not collapsed")


def test_faces_carry_appearance_from_the_vlm():
    attempts, clips = _shoot()
    g = _by_gid(rel.derive_identities(attempts, clips))
    assert "curly" in (g["curly"]["description"] or "").lower()
    assert "bald" in (g["bald"]["description"] or "").lower()
    print("ok  each global person carries its face's canonical appearance")


def test_ambiguous_cross_clip_noise_forms_no_voice_link():
    """Two clips that merely share stray filler tie across pairings (no dominant
    correspondence) -> the dominance gate forms NO voice link (appearance would
    link such people, not a coin-flip vote)."""
    from collections import Counter
    cnt = Counter({("S0", "S1"): 2, ("S1", "S0"): 2, ("S1", "S1"): 2})
    assert rel._match_pair(cnt) == [], rel._match_pair(cnt)
    clear = Counter({("S1", "S1"): 9, ("S1", "S0"): 2, ("S0", "S0"): 5})
    edges = rel._match_pair(clear)
    assert (9, "S1", "S1") in edges and (5, "S0", "S0") in edges, edges
    print("ok  tied cross-clip votes drop; a dominant correspondence is kept")


def test_one_shared_line_is_below_threshold():
    """A single matched line is not enough to link voices across clips."""
    clips = {A_CURLY: _clip(A_CURLY, CURLY_FACE, [(1000, 3000, A1, "X0")], []),
             A_BALD: _clip(A_BALD, BALD_FACE, [(1000, 3000, A1, "S1")], [])}
    attempts = _atts(A_CURLY, [(1000, 3000, A1, "X0")]) + \
        _atts(A_BALD, [(1000, 3000, A1, "S1")])
    # No cross-clip voice link and no bridge -> no cross-clip identity.
    assert rel.derive_identities(attempts, clips) == []
    print("ok  one shared line is below the corroboration threshold")


# --------------------------------------------------------------------------
# ORIENTATION HEADER + REGISTRY + INVARIANTS
# --------------------------------------------------------------------------

def test_orientation_header_reads_clean():
    attempts, clips = _shoot()
    relations = rel.build_relations  # noqa: F841 (documents the entry point)
    recon = rel._reconcile(attempts, clips)
    text = rel.render_relations(recon)
    assert "PEOPLE OF THE SHOOT" in text, text
    assert "on camera in" in text and "heard off-camera in" in text, text
    assert "NOTE" in text, text                          # the honest contradiction
    assert "offset" not in text.lower() and "co-temporal" not in text, text
    assert rel.render_relations({}) == ""
    print("ok  orientation header names who is on which camera, honestly")
    print("--- sample orientation ---\n" + text + "\n--------------------------")


def test_registry_and_oncam_lookups():
    attempts, clips = _shoot()
    relations = rel._reconcile(attempts, clips)
    g = _by_gid(relations["identities"])
    curly_gid = g["curly"]["global_id"]
    # A per-clip voice aliases to its global id (both segments, both labels).
    assert rel.global_id_of(relations, A_CURLY, "X0") == curly_gid
    assert rel.global_id_of(relations, A_BALD, "S1") == curly_gid   # off-cam, label-swapped
    assert rel.global_id_of(relations, A_CURLY, "nope") is None
    # oncam_global_by_file answers "whose face shows in this clip?"
    oncam = rel.oncam_global_by_file(relations)
    assert oncam[A_CURLY] == curly_gid and oncam[B_CURLY] == curly_gid
    assert oncam[A_BALD] == g["bald"]["global_id"]
    print("ok  registry aliases voices and reports the on-camera face per clip")


def test_validate_catches_structural_corruption():
    good = {"identities": [
        {"global_id": "G1", "members": [{"file": A_CURLY, "voice": "X0",
                                         "on_camera": True}]},
        {"global_id": "G2", "members": [{"file": A_BALD, "voice": "S0",
                                         "on_camera": True}]}]}
    assert rel.validate(good) == [], rel.validate(good)
    # Two people both claim the SAME clip's face on camera -> caught.
    bad_cam = {"identities": [
        {"global_id": "G1", "members": [{"file": A_CURLY, "on_camera": True}]},
        {"global_id": "G2", "members": [{"file": A_CURLY, "on_camera": True}]}]}
    assert any("on-camera for both" in p for p in rel.validate(bad_cam))
    # One voice in two people -> caught.
    bad_voice = {"identities": [
        {"global_id": "G1", "members": [{"file": A_CURLY, "voice": "X0"}]},
        {"global_id": "G2", "members": [{"file": A_CURLY, "voice": "X0"}]}]}
    assert any("belongs to both" in p for p in rel.validate(bad_voice))
    print("ok  invariant check catches two-faces and shared-voice corruption")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall relations tests passed")
