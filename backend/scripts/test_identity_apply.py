"""
End-to-end tests for the identity orchestrator (app.services.l3.identity.
apply) -- no DB, no R2, no model call. voice_id_pass.plan.md moved the
model call (and clip extraction) upstream into Track A/B of ingest.py, so
apply.run is now pure code: it takes Track B's already-computed clip
verdicts directly and reconciles them against Track A's face clustering.

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
from app.services.l3.identity import voice_id as vid  # noqa: E402
from app.services.l3.lattice import Lattice  # noqa: E402


def _lat():
    words = [{"start_ms": i * 100, "end_ms": i * 100 + 90} for i in range(50)]
    return Lattice(file_id="f1", duration_ms=10000, words=words, turns=[], hints=[], atoms=[])


def _cut(ref, word_span, voice_ids, appearance, position="center", subject_box=(0.2, 0.2, 0.3, 0.3)):
    return pass2.Pass2Cut(
        source_ref=ref, kind="speech", file_id="f1", word_span=word_span,
        label="x", summary="y", voice_ids=voice_ids,
        people=[{"description": "someone", "appearance": appearance, "position": position}],
        framing=pass2.Framing(subject_box=subject_box),
    )


def _app(**kw):
    base = {"apparent_gender": None, "apparent_age_band": None, "hair": None,
           "hair_color": None, "facial_hair": None, "glasses": None,
           "skin_tone": None, "build": None}
    base.update(kw)
    return base


def _verdict(voice, clip_id, verdict, center_ms):
    return vid.ClipVerdict(voice=voice, clip_id=clip_id, file_id="f1", verdict=verdict, center_ms=center_ms)


def test_run_binds_two_speakers_and_derives_on_camera():
    cuts = [
        _cut("speech_cut[0]", (0, 19), ["V0"],
            _app(apparent_gender="male", hair="bald", facial_hair="beard")),
        _cut("speech_cut[1]", (25, 44), ["V1"],
            _app(apparent_gender="female", hair="long", facial_hair="none")),
    ]
    pass2_output = pass2.Pass2Output(cuts=cuts)
    lattices = {"f1": _lat()}
    voice_of = {("f1", "S0"): "V0", ("f1", "S1"): "V1"}
    verdicts = [
        _verdict("V0", "V0:c0:f1", "speaking", 50),      # inside speech_cut[0]'s span
        _verdict("V1", "V1:c0:f1", "speaking", 2550),    # inside speech_cut[1]'s span
    ]

    new_output, payload = identity_apply.run(pass2_output, lattices, voice_of, verdicts)

    assert len(payload["persons"]) == 2, payload["persons"]
    assert payload["voice_owner"] == {"V0": "P0", "V1": "P1"}, payload["voice_owner"]
    assert payload["off_camera_voices"] == [], payload["off_camera_voices"]

    c0, c1 = new_output.cuts
    assert c0.speaker_person == "P0" and c0.on_camera is True, (c0.speaker_person, c0.on_camera)
    assert c1.speaker_person == "P1" and c1.on_camera is True, (c1.speaker_person, c1.on_camera)
    assert c0.visible_persons == ["P0"] and c1.visible_persons == ["P1"]
    print("ok  test_run_binds_two_speakers_and_derives_on_camera")


def test_run_leaves_a_narrator_voice_unbound_and_off_camera():
    # V0 has voice_ids on its cut but NO visible person anywhere (pure
    # narration) -- speaker_person/on_camera must stay unset, never guessed.
    cuts = [_cut("speech_cut[0]", (0, 19), ["V0"], _app())]
    cuts[0] = cuts[0].model_copy(update={"people": []})
    pass2_output = pass2.Pass2Output(cuts=cuts)
    lattices = {"f1": _lat()}
    voice_of = {("f1", "S0"): "V0"}

    new_output, payload = identity_apply.run(pass2_output, lattices, voice_of, [])

    assert payload["persons"] == [], payload["persons"]
    assert payload["off_camera_voices"] == ["V0"], payload["off_camera_voices"]
    assert new_output.cuts[0].speaker_person is None
    assert new_output.cuts[0].on_camera is None
    print("ok  test_run_leaves_a_narrator_voice_unbound_and_off_camera")


def test_run_marks_a_voice_with_zero_verdicts_off_camera_via_full_roster():
    # V1 is in voice_of (the full project roster) but never produced a
    # single clip verdict (e.g. Track A had no clean window for it) -- it
    # must still land in off_camera_voices, not silently vanish.
    cuts = [_cut("speech_cut[0]", (0, 19), ["V0"],
                _app(apparent_gender="male", hair="bald", facial_hair="beard"))]
    pass2_output = pass2.Pass2Output(cuts=cuts)
    lattices = {"f1": _lat()}
    voice_of = {("f1", "S0"): "V0", ("f1", "S1"): "V1"}
    verdicts = [_verdict("V0", "V0:c0:f1", "speaking", 50)]

    _new_output, payload = identity_apply.run(pass2_output, lattices, voice_of, verdicts)

    assert "V1" in payload["off_camera_voices"], payload["off_camera_voices"]
    assert "V1" not in payload["voice_owner"], payload["voice_owner"]
    print("ok  test_run_marks_a_voice_with_zero_verdicts_off_camera_via_full_roster")


def main():
    test_run_binds_two_speakers_and_derives_on_camera()
    test_run_leaves_a_narrator_voice_unbound_and_off_camera()
    test_run_marks_a_voice_with_zero_verdicts_off_camera_via_full_roster()
    print("\nall identity-apply tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
