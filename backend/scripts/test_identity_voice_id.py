"""
Tests for the voice-ID pass (app.services.l3.identity.voice_id,
voice_id_pass.plan.md) -- no DB, NO REAL API CALLS (Gemini SDK mocked the
same way test_ingest_gemini.py does). Covers:

  - select_clips: Part A's pure-code planning step -- clean-window ranking,
    cross-camera fan-out via outlook groups, the max_clips ceiling.
  - run_voice_id_pass: one Gemini call per voice, verdict parsing, the
    "no bytes -> skipped" / "no verdict returned -> no_face" fail-open
    contracts.
  - bind_from_verdicts: Part B's majority+margin vote aggregation, ported
    from the old identity/speaker_pass.aggregate_votes over clip votes
    instead of window votes.

Run:  .venv/bin/python scripts/test_identity_voice_id.py
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
from app.services.l3.identity import voice_id as vid  # noqa: E402
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


# --------------------------------------------------------------------------
# select_clips
# --------------------------------------------------------------------------

def test_clip_window_stays_clamped_inside_the_turn():
    # Peak near the very start of a short turn -- a centered CLIP_MS window
    # would reach before turn_s and past turn_e; it must shrink to fit.
    lo, hi = vid._clip_window(1000, 1800, t_star=1050, clip_ms=3000)
    assert lo == 1000 and hi == 1800, (lo, hi)
    # Peak comfortably inside a long turn -- the window centers on it and
    # stays within [turn_s, turn_e).
    lo, hi = vid._clip_window(0, 10000, t_star=5000, clip_ms=3000)
    assert lo == 3500 and hi == 6500, (lo, hi)
    assert 0 <= lo and hi <= 10000
    print("ok  test_clip_window_stays_clamped_inside_the_turn")


def test_select_clips_single_camera_one_clip_per_kept_turn():
    turns_by_file = {"f1": [(0, 2000, "S0")]}
    voice_of = {("f1", "S0"): "V0"}
    audio_by_file = {"f1": {"rms_db": [-20.0] * 40, "hop_ms": 50}}
    reqs = vid.select_clips(turns_by_file, voice_of, {}, audio_by_file)
    assert len(reqs) == 1, reqs
    r = reqs[0]
    assert r.voice == "V0" and r.file_id == "f1"
    assert r.start_ms < r.end_ms
    print("ok  test_select_clips_single_camera_one_clip_per_kept_turn")


def test_select_clips_fans_out_across_outlook_group_cameras():
    # f1/f2 are declared outlook-group members whose (already re-based)
    # turns carry the IDENTICAL span for the same moment -> one moment,
    # two ClipRequests (one per camera).
    turns_by_file = {"f1": [(0, 2000, "S0")], "f2": [(0, 2000, "S0")]}
    voice_of = {("f1", "S0"): "V0", ("f2", "S0"): "V0"}
    groups = {"g0": {"auth": "f1", "members": {"f1", "f2"}}}
    reqs = vid.select_clips(turns_by_file, voice_of, groups)
    assert len(reqs) == 2, reqs
    assert {r.file_id for r in reqs} == {"f1", "f2"}
    print("ok  test_select_clips_fans_out_across_outlook_group_cameras")


def test_select_clips_does_not_merge_ungrouped_files_with_coincidental_span():
    # f1/f2 are NOT declared outlook members -- even if their spans happen
    # to coincide, each is its own singleton "group" and gets counted as a
    # SEPARATE moment (so top-K ranks them independently), not fanned out
    # as one moment.
    turns_by_file = {"f1": [(0, 2000, "S0")], "f2": [(0, 2000, "S0")]}
    voice_of = {("f1", "S0"): "V0", ("f2", "S0"): "V0"}
    reqs = vid.select_clips(turns_by_file, voice_of, {}, k=1)
    # k=1 -- only one MOMENT kept, but since f1/f2 are distinct (ungrouped)
    # moments, only one file's clip should appear, not both.
    assert len(reqs) == 1, reqs
    print("ok  test_select_clips_does_not_merge_ungrouped_files_with_coincidental_span")


def test_select_clips_keeps_only_top_k_longest_moments():
    turns_by_file = {"f1": [(0, 500, "S0"), (1000, 3000, "S0"), (4000, 4600, "S0")]}
    voice_of = {("f1", "S0"): "V0"}
    reqs = vid.select_clips(turns_by_file, voice_of, {}, k=2)
    # Only the two longest clean windows (~2000ms, ~600ms) should survive;
    # the shortest (~500ms) is dropped.
    assert len(reqs) == 2, reqs
    print("ok  test_select_clips_keeps_only_top_k_longest_moments")


def test_select_clips_empty_for_a_voice_with_no_clean_window():
    # A turn fully swallowed by another voice's guard zone -> no windows.
    turns_by_file = {"f1": [(1000, 1500, "S0"), (1500, 1600, "S1")]}
    voice_of = {("f1", "S0"): "V0", ("f1", "S1"): "V1"}
    reqs = vid.select_clips(turns_by_file, voice_of, {})
    assert all(r.voice != "V0" for r in reqs), reqs
    print("ok  test_select_clips_empty_for_a_voice_with_no_clean_window")


def test_select_clips_respects_max_clips_ceiling():
    turns_by_file = {"f1": [(i * 10000, i * 10000 + 3000, "S0") for i in range(10)]}
    voice_of = {("f1", "S0"): "V0"}
    reqs = vid.select_clips(turns_by_file, voice_of, {}, k=10, max_clips=3)
    assert len(reqs) == 3, reqs
    print("ok  test_select_clips_respects_max_clips_ceiling")


# --------------------------------------------------------------------------
# run_voice_id_pass
# --------------------------------------------------------------------------

def test_run_voice_id_pass_returns_empty_with_no_clip_bytes():
    reqs = [vid.ClipRequest(voice="V0", clip_id="V0:c0:f1", file_id="f1", start_ms=0, end_ms=3000)]
    assert vid.run_voice_id_pass(reqs, {}) == []
    print("ok  test_run_voice_id_pass_returns_empty_with_no_clip_bytes")


def test_run_voice_id_pass_parses_verdicts_and_defaults_missing_to_no_face():
    reqs = [
        vid.ClipRequest(voice="V0", clip_id="V0:c0:f1", file_id="f1", start_ms=0, end_ms=3000),
        vid.ClipRequest(voice="V0", clip_id="V0:c1:f1", file_id="f1", start_ms=5000, end_ms=8000),
    ]
    clips_b64 = {"V0:c0:f1": "ZmFrZQ==", "V0:c1:f1": "ZmFrZQ=="}
    # The model only returns a verdict for c0 -- c1 must default to no_face.
    payload = '{"verdicts": [{"clip_id": "V0:c0:f1", "verdict": "speaking"}]}'

    def fake_generate_content(model, contents, config):
        return _FakeGeminiResp(payload)

    fake_client = mock.Mock()
    fake_client.models.generate_content = fake_generate_content
    with mock.patch.object(ig, "_sdk", return_value=(fake_client, _types())):
        verdicts = vid.run_voice_id_pass(reqs, clips_b64)
    by_clip = {v.clip_id: v.verdict for v in verdicts}
    assert by_clip == {"V0:c0:f1": "speaking", "V0:c1:f1": "no_face"}, by_clip
    print("ok  test_run_voice_id_pass_parses_verdicts_and_defaults_missing_to_no_face")


def test_run_voice_id_pass_makes_one_call_per_voice():
    reqs = [
        vid.ClipRequest(voice="V0", clip_id="V0:c0:f1", file_id="f1", start_ms=0, end_ms=3000),
        vid.ClipRequest(voice="V1", clip_id="V1:c0:f1", file_id="f1", start_ms=5000, end_ms=8000),
    ]
    clips_b64 = {"V0:c0:f1": "ZmFrZQ==", "V1:c0:f1": "ZmFrZQ=="}
    calls = []

    def fake_generate_content(model, contents, config):
        calls.append(1)
        return _FakeGeminiResp('{"verdicts": []}')

    fake_client = mock.Mock()
    fake_client.models.generate_content = fake_generate_content
    with mock.patch.object(ig, "_sdk", return_value=(fake_client, _types())):
        vid.run_voice_id_pass(reqs, clips_b64)
    assert len(calls) == 2, calls
    print("ok  test_run_voice_id_pass_makes_one_call_per_voice")


# --------------------------------------------------------------------------
# bind_from_verdicts
# --------------------------------------------------------------------------

def _lat():
    words = [{"start_ms": i * 100, "end_ms": i * 100 + 90} for i in range(100)]
    return Lattice(file_id="f1", duration_ms=10000, words=words, turns=[], hints=[], atoms=[])


def _cut(ref, word_span):
    return pass2.Pass2Cut(source_ref=ref, kind="speech", file_id="f1", word_span=word_span,
                          label="x", summary="y")


def _verdict(voice, clip_id, verdict, center_ms, file_id="f1"):
    return vid.ClipVerdict(voice=voice, clip_id=clip_id, file_id=file_id, verdict=verdict, center_ms=center_ms)


def test_bind_from_verdicts_binds_a_clear_majority():
    # cut[0] covers [0, 1000) and shows P0; 2 "speaking" clips resolve there,
    # 1 "not_speaking" clip is an abstention that must not dilute the count.
    cuts_by_file = {"f1": [_cut("speech_cut[0]", (0, 9))]}
    lattices = {"f1": _lat()}
    visible_persons = {("f1", "speech_cut[0]"): ["P0"]}
    verdicts = [
        _verdict("V0", "c0", "speaking", 100),
        _verdict("V0", "c1", "speaking", 200),
        _verdict("V0", "c2", "not_speaking", 300),
    ]
    owner_by_voice, off_camera = vid.bind_from_verdicts(verdicts, visible_persons, cuts_by_file, lattices)
    assert owner_by_voice == {"V0": "P0"}, owner_by_voice
    assert off_camera == set(), off_camera
    print("ok  test_bind_from_verdicts_binds_a_clear_majority")


def test_bind_from_verdicts_unbound_on_a_split_tie():
    cuts_by_file = {"f1": [_cut("speech_cut[0]", (0, 9)), _cut("speech_cut[1]", (10, 19))]}
    lattices = {"f1": _lat()}
    visible_persons = {("f1", "speech_cut[0]"): ["P0"], ("f1", "speech_cut[1]"): ["P1"]}
    verdicts = [_verdict("V0", "c0", "speaking", 100), _verdict("V0", "c1", "speaking", 1100)]
    owner_by_voice, off_camera = vid.bind_from_verdicts(verdicts, visible_persons, cuts_by_file, lattices)
    assert owner_by_voice == {"V0": None}, owner_by_voice
    assert off_camera == {"V0"}, off_camera
    print("ok  test_bind_from_verdicts_unbound_on_a_split_tie")


def test_bind_from_verdicts_unbound_when_nobody_ever_resolves_a_person():
    cuts_by_file = {"f1": [_cut("speech_cut[0]", (0, 9))]}
    lattices = {"f1": _lat()}
    verdicts = [_verdict("V0", "c0", "not_speaking", 100), _verdict("V0", "c1", "no_face", 200)]
    owner_by_voice, off_camera = vid.bind_from_verdicts(verdicts, {}, cuts_by_file, lattices)
    assert owner_by_voice == {"V0": None}, owner_by_voice
    assert off_camera == {"V0"}, off_camera
    print("ok  test_bind_from_verdicts_unbound_when_nobody_ever_resolves_a_person")


def test_bind_from_verdicts_multi_person_cut_votes_for_every_visible_person():
    cuts_by_file = {"f1": [_cut("speech_cut[0]", (0, 9))]}
    lattices = {"f1": _lat()}
    visible_persons = {("f1", "speech_cut[0]"): ["P0", "P1"]}
    # A single "speaking" clip in a two-person cut votes for BOTH -- with
    # nothing else to break the tie, that alone stays unbound (no margin).
    verdicts = [_verdict("V0", "c0", "speaking", 100)]
    owner_by_voice, off_camera = vid.bind_from_verdicts(verdicts, visible_persons, cuts_by_file, lattices)
    assert owner_by_voice == {"V0": None}, owner_by_voice
    print("ok  test_bind_from_verdicts_multi_person_cut_votes_for_every_visible_person")


def test_bind_from_verdicts_voice_absent_entirely_is_not_in_the_output():
    # V1 never appears in verdicts at all (Track A never got a clip for it)
    # -- bind_from_verdicts leaves it out entirely; the caller (identity/
    # apply.py) is responsible for reconciling against the full voice roster.
    cuts_by_file = {"f1": [_cut("speech_cut[0]", (0, 9))]}
    lattices = {"f1": _lat()}
    visible_persons = {("f1", "speech_cut[0]"): ["P0"]}
    verdicts = [_verdict("V0", "c0", "speaking", 100)]
    owner_by_voice, off_camera = vid.bind_from_verdicts(verdicts, visible_persons, cuts_by_file, lattices)
    assert "V1" not in owner_by_voice, owner_by_voice
    print("ok  test_bind_from_verdicts_voice_absent_entirely_is_not_in_the_output")


def main():
    test_clip_window_stays_clamped_inside_the_turn()
    test_select_clips_single_camera_one_clip_per_kept_turn()
    test_select_clips_fans_out_across_outlook_group_cameras()
    test_select_clips_does_not_merge_ungrouped_files_with_coincidental_span()
    test_select_clips_keeps_only_top_k_longest_moments()
    test_select_clips_empty_for_a_voice_with_no_clean_window()
    test_select_clips_respects_max_clips_ceiling()
    test_run_voice_id_pass_returns_empty_with_no_clip_bytes()
    test_run_voice_id_pass_parses_verdicts_and_defaults_missing_to_no_face()
    test_run_voice_id_pass_makes_one_call_per_voice()
    test_bind_from_verdicts_binds_a_clear_majority()
    test_bind_from_verdicts_unbound_on_a_split_tie()
    test_bind_from_verdicts_unbound_when_nobody_ever_resolves_a_person()
    test_bind_from_verdicts_multi_person_cut_votes_for_every_visible_person()
    test_bind_from_verdicts_voice_absent_entirely_is_not_in_the_output()
    print("\nall identity-voice-id tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
