"""
End-to-end tests for the identity orchestrator (app.services.l3.identity.
apply) -- no DB, no R2, NO REAL API CALLS. Mocks frame extraction and the
Gemini SDK client (same pattern test_identity_speaker_pass.py uses).

Run:  .venv/bin/python scripts/test_identity_apply.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import pass2  # noqa: E402
from app.services.l3.identity import apply as identity_apply  # noqa: E402
from app.services.l3.lattice import Lattice  # noqa: E402
from app.services.llm import ingest_gemini as ig  # noqa: E402


def _types():
    from google.genai import types
    return types


class _FakeUsageMD:
    prompt_token_count = 100
    candidates_token_count = 20
    cached_content_token_count = 0
    thoughts_token_count = 0


class _FakeGeminiResp:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = _FakeUsageMD()
        self.candidates = []


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
    turns_by_file = {"f1": [(0, 2000, "S0"), (2500, 4500, "S1")]}
    audio_by_file = {"f1": {"rms_db": [-20.0] * 100, "hop_ms": 50}}
    motion_by_file = {"f1": {"blur": [0.3] * 100, "hop_ms": 50}}
    proxy_key_by_file = {"f1": "proxy-key"}

    responses = iter([
        '{"votes": [{"window_id": "V0:w0", "speaking_person": "P0"}]}',
        '{"votes": [{"window_id": "V1:w0", "speaking_person": "P1"}]}',
    ])

    def fake_generate_content(model, contents, config):
        return _FakeGeminiResp(next(responses))

    fake_client = mock.Mock()
    fake_client.models.generate_content = fake_generate_content

    def fake_extract(planned_frames, proxy_key_by_file, width=768):
        return {(f.file_id, f.ts_ms): "eA==" for f in planned_frames}   # valid b64 for "x"

    with mock.patch.object(ig, "_sdk", return_value=(fake_client, _types())), \
         mock.patch.object(identity_apply.fr, "extract_for_planned_frames", side_effect=fake_extract):
        new_output, payload = identity_apply.run(
            pass2_output, lattices, voice_of, turns_by_file,
            audio_by_file, motion_by_file, proxy_key_by_file,
        )

    assert len(payload["persons"]) == 2, payload["persons"]
    assert payload["voice_owner"] == {"V0": "P0", "V1": "P1"}, payload["voice_owner"]
    assert payload["off_camera_voices"] == [], payload["off_camera_voices"]

    c0, c1 = new_output.cuts
    assert c0.speaker_person == "P0" and c0.on_camera is True, (c0.speaker_person, c0.on_camera)
    assert c1.speaker_person == "P1" and c1.on_camera is True, (c1.speaker_person, c1.on_camera)
    assert c0.visible_persons == ["P0"] and c1.visible_persons == ["P1"]
    print("ok  test_run_binds_two_speakers_and_derives_on_camera")


def test_run_leaves_a_narrator_voice_unbound_and_off_camera():
    # V0 has a turn but NO covering cut/visible person anywhere -> pure
    # narration. speaker_person/on_camera must stay unset, never guessed.
    cuts = [_cut("speech_cut[0]", (0, 19), ["V0"], _app())]
    # Strip the person off this cut entirely -- nothing to see.
    cuts[0] = cuts[0].model_copy(update={"people": []})
    pass2_output = pass2.Pass2Output(cuts=cuts)
    lattices = {"f1": _lat()}
    voice_of = {("f1", "S0"): "V0"}
    turns_by_file = {"f1": [(0, 2000, "S0")]}
    audio_by_file = {"f1": {"rms_db": [-20.0] * 100, "hop_ms": 50}}
    motion_by_file = {"f1": {"blur": [0.3] * 100, "hop_ms": 50}}
    proxy_key_by_file = {"f1": "proxy-key"}

    with mock.patch.object(identity_apply.fr, "extract_for_planned_frames") as mocked_extract:
        new_output, payload = identity_apply.run(
            pass2_output, lattices, voice_of, turns_by_file,
            audio_by_file, motion_by_file, proxy_key_by_file,
        )
    mocked_extract.assert_not_called()   # no bursts planned -> nothing to extract
    assert payload["persons"] == [], payload["persons"]
    assert payload["off_camera_voices"] == ["V0"], payload["off_camera_voices"]
    assert new_output.cuts[0].speaker_person is None
    assert new_output.cuts[0].on_camera is None
    print("ok  test_run_leaves_a_narrator_voice_unbound_and_off_camera")


def main():
    test_run_binds_two_speakers_and_derives_on_camera()
    test_run_leaves_a_narrator_voice_unbound_and_off_camera()
    print("\nall identity-apply tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
