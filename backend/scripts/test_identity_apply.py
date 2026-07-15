"""
End-to-end tests for the identity assembly step (app.services.l3.identity.
apply) -- no DB, no R2, no model call, no CV. asd_identity.plan.md moved
ALL the real work upstream into identity/faces.py (clustering) and
identity/bind_asd.py (voice binding), both of which ingest.py now calls
directly; apply.run is pure assembly, taking their already-computed output.

Run:  .venv/bin/python scripts/test_identity_apply.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import pass2  # noqa: E402
from app.services.l3.identity import apply as identity_apply  # noqa: E402


def _cut(ref, voice_ids, visible=(), position="center", subject_box=(0.2, 0.2, 0.3, 0.3)):
    return pass2.Pass2Cut(
        source_ref=ref, kind="speech", file_id="f1", word_span=(0, 1),
        label="x", summary="y", voice_ids=voice_ids,
        people=[{"description": "someone", "appearance": {}, "position": position}] if visible else [],
        framing=pass2.Framing(subject_box=subject_box),
    )


def _person(pid):
    return {"person_id": pid, "appearance_count": 3, "is_major": True, "owned_voices": []}


def test_run_binds_two_speakers_and_derives_on_camera():
    cuts = [_cut("speech_cut[0]", ["V0"], visible=True), _cut("speech_cut[1]", ["V1"], visible=True)]
    pass2_output = pass2.Pass2Output(cuts=cuts)
    voice_of = {("f1", "S0"): "V0", ("f1", "S1"): "V1"}
    persons = {"P0": _person("P0"), "P1": _person("P1")}
    visible_persons = {("f1", "speech_cut[0]"): ["P0"], ("f1", "speech_cut[1]"): ["P1"]}
    owner_by_voice = {"V0": "P0", "V1": "P1"}

    new_output, payload = identity_apply.run(
        pass2_output, voice_of, persons, visible_persons, owner_by_voice, set())

    assert len(payload["persons"]) == 2, payload["persons"]
    assert payload["voice_owner"] == {"V0": "P0", "V1": "P1"}, payload["voice_owner"]
    assert payload["off_camera_voices"] == [], payload["off_camera_voices"]
    owned = {p["person_id"]: p["owned_voices"] for p in payload["persons"]}
    assert owned == {"P0": ["V0"], "P1": ["V1"]}, owned

    c0, c1 = new_output.cuts
    assert c0.speaker_person == "P0" and c0.on_camera is True, (c0.speaker_person, c0.on_camera)
    assert c1.speaker_person == "P1" and c1.on_camera is True, (c1.speaker_person, c1.on_camera)
    assert c0.visible_persons == ["P0"] and c1.visible_persons == ["P1"]
    print("ok  test_run_binds_two_speakers_and_derives_on_camera")


def test_run_leaves_a_narrator_voice_unbound_and_off_camera():
    # V0 has voice_ids on its cut but bind_asd never resolved an owner for it
    # (pure narration, no face ever ASD-tracks its speech) -- speaker_person/
    # on_camera must stay unset, never guessed.
    cuts = [_cut("speech_cut[0]", ["V0"], visible=False)]
    pass2_output = pass2.Pass2Output(cuts=cuts)
    voice_of = {("f1", "S0"): "V0"}

    new_output, payload = identity_apply.run(pass2_output, voice_of, {}, {}, {}, {"V0"})

    assert payload["persons"] == [], payload["persons"]
    assert payload["off_camera_voices"] == ["V0"], payload["off_camera_voices"]
    assert new_output.cuts[0].speaker_person is None
    assert new_output.cuts[0].on_camera is None
    print("ok  test_run_leaves_a_narrator_voice_unbound_and_off_camera")


def test_run_marks_a_voice_with_zero_asd_votes_off_camera_via_full_roster():
    # V1 is in voice_of (the full project roster) but bind_asd.bind never
    # produced ANY vote for it (absent from owner_by_voice entirely, not
    # even as an explicit None) -- it must still land in off_camera_voices,
    # not silently vanish.
    cuts = [_cut("speech_cut[0]", ["V0"], visible=True)]
    pass2_output = pass2.Pass2Output(cuts=cuts)
    voice_of = {("f1", "S0"): "V0", ("f1", "S1"): "V1"}
    persons = {"P0": _person("P0")}
    visible_persons = {("f1", "speech_cut[0]"): ["P0"]}
    owner_by_voice = {"V0": "P0"}

    _new_output, payload = identity_apply.run(
        pass2_output, voice_of, persons, visible_persons, owner_by_voice, set())

    assert "V1" in payload["off_camera_voices"], payload["off_camera_voices"]
    assert "V1" not in payload["voice_owner"], payload["voice_owner"]
    print("ok  test_run_marks_a_voice_with_zero_asd_votes_off_camera_via_full_roster")


def test_run_dominant_voice_only_no_fallthrough_to_a_bound_secondary():
    # A cut's voice_ids[0] (dominant) is unbound; voice_ids[1] IS bound. The
    # cut must still show speaker_person=None -- never falls through to a
    # minor interjector's owner.
    cuts = [_cut("speech_cut[0]", ["V0", "V1"], visible=True)]
    pass2_output = pass2.Pass2Output(cuts=cuts)
    voice_of = {("f1", "S0"): "V0", ("f1", "S1"): "V1"}
    persons = {"P1": _person("P1")}
    visible_persons = {("f1", "speech_cut[0]"): ["P1"]}
    owner_by_voice = {"V0": None, "V1": "P1"}

    new_output, _payload = identity_apply.run(
        pass2_output, voice_of, persons, visible_persons, owner_by_voice, {"V0"})

    assert new_output.cuts[0].speaker_person is None
    assert new_output.cuts[0].on_camera is None
    print("ok  test_run_dominant_voice_only_no_fallthrough_to_a_bound_secondary")


def main():
    test_run_binds_two_speakers_and_derives_on_camera()
    test_run_leaves_a_narrator_voice_unbound_and_off_camera()
    test_run_marks_a_voice_with_zero_asd_votes_off_camera_via_full_roster()
    test_run_dominant_voice_only_no_fallthrough_to_a_bound_secondary()
    print("\nall identity-apply tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
