"""
Pure unit tests for app.services.vcut.speech.orchestrate._resolve_take_group
-- speech_noise_gate.plan.md section 3: a gated beat (LLM- or energy-flagged)
is routed to its own junk=True ResolvedBeat instead of silently dropped, and
never enters group_delivery_scores/select_winner. No I/O -- FileSpeechInputs/
_BeatOut are plain in-memory objects; boundaries/delivery/select underneath
are themselves pure.

Run:  .venv/bin/python scripts/test_speech_orchestrate.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.speech.inputs import FileSpeechInputs, Word  # noqa: E402
from app.services.vcut.speech.orchestrate import _resolve_take_group  # noqa: E402
from app.services.vcut.speech.segment_llm import _BeatOut  # noqa: E402

# voiced/floor refs with a real spread (40dB) so the energy gate is ACTIVE
# for every test file below: threshold = floor + 0.30*(voiced-floor) = -38dB.
_VOICED_DB = -10.0
_FLOOR_DB = -50.0


def _words(n, gap_ms=1000):
    return [Word(idx=i, start_ms=i * gap_ms, end_ms=i * gap_ms + 400, text=f"w{i}") for i in range(n)]


def _fi(file_id, n_words=4, rms_db_value=-15.0, rms_hop_ms=200, n_rms=50, duration_ms=10000):
    return FileSpeechInputs(
        file_id=file_id, duration_ms=duration_ms, words=_words(n_words),
        rms_db=[rms_db_value] * n_rms, rms_hop_ms=rms_hop_ms, silences=[],
        rms_voiced_db=_VOICED_DB, rms_floor_db=_FLOOR_DB,
    )


def _beat(id_, file_id, span=(0, 1), gist="g", fluency=0.5, flags=None, is_speech=True):
    return _BeatOut(id=id_, file_id=file_id, word_span=span, gist=gist, fluency=fluency,
                    flags=flags or [], is_speech=is_speech)


_LOUD_DB = -15.0   # clears the -38dB threshold -> would pass the energy gate
_QUIET_DB = -48.0  # sits near the floor -> fails the energy gate


def test_llm_flagged_beat_is_junk_non_speech_llm_even_with_loud_energy():
    beat = _beat("b0", "f1", is_speech=False)
    fi = _fi("f1", rms_db_value=_LOUD_DB)  # energy alone would pass
    resolved = _resolve_take_group([beat], {"f1": fi})
    assert len(resolved) == 1
    rb = resolved[0]
    assert rb.junk is True
    assert rb.junk_reason == "non_speech_llm"
    assert rb.take_role == "take"
    print("ok  test_llm_flagged_beat_is_junk_non_speech_llm_even_with_loud_energy")


def test_near_floor_energy_beat_is_junk_non_speech_energy():
    beat = _beat("b0", "f1", is_speech=True)
    fi = _fi("f1", rms_db_value=_QUIET_DB)
    resolved = _resolve_take_group([beat], {"f1": fi})
    assert len(resolved) == 1
    rb = resolved[0]
    assert rb.junk is True
    assert rb.junk_reason == "non_speech_energy"
    assert rb.take_role == "take"
    print("ok  test_near_floor_energy_beat_is_junk_non_speech_energy")


def test_real_beat_is_not_junk_and_wins_as_a_singleton():
    beat = _beat("b0", "f1", is_speech=True)
    fi = _fi("f1", rms_db_value=_LOUD_DB)
    resolved = _resolve_take_group([beat], {"f1": fi})
    assert len(resolved) == 1
    rb = resolved[0]
    assert rb.junk is False and rb.junk_reason == ""
    assert rb.take_role == "winner"
    print("ok  test_real_beat_is_not_junk_and_wins_as_a_singleton")


def test_group_of_real_and_junk_real_wins_junk_never_competes():
    real = _beat("b0", "f1", span=(0, 1), is_speech=True)
    junk = _beat("b1", "f1", span=(2, 3), is_speech=False)
    fi = _fi("f1", n_words=4, rms_db_value=_LOUD_DB)
    resolved = _resolve_take_group([real, junk], {"f1": fi})
    by_id = {rb.beat.id: rb for rb in resolved}
    assert len(resolved) == 2
    assert by_id["b0"].take_role == "winner" and by_id["b0"].junk is False
    assert by_id["b1"].take_role == "take" and by_id["b1"].junk is True
    assert by_id["b1"].junk_reason == "non_speech_llm"
    print("ok  test_group_of_real_and_junk_real_wins_junk_never_competes")


def test_group_all_junk_yields_only_junk_no_winner_no_crash():
    b0 = _beat("b0", "f1", span=(0, 1), is_speech=False)
    b1 = _beat("b1", "f1", span=(2, 3), is_speech=False)
    fi = _fi("f1", n_words=4, rms_db_value=_LOUD_DB)
    resolved = _resolve_take_group([b0, b1], {"f1": fi})
    assert len(resolved) == 2
    assert all(rb.junk for rb in resolved)
    assert all(rb.take_role == "take" for rb in resolved)
    assert {rb.junk_reason for rb in resolved} == {"non_speech_llm"}
    print("ok  test_group_all_junk_yields_only_junk_no_winner_no_crash")


def test_beat_for_a_file_with_no_loaded_inputs_is_dropped_defensively():
    beat = _beat("b0", "ghost_file")
    resolved = _resolve_take_group([beat], {})
    assert resolved == []
    print("ok  test_beat_for_a_file_with_no_loaded_inputs_is_dropped_defensively")


def main():
    test_llm_flagged_beat_is_junk_non_speech_llm_even_with_loud_energy()
    test_near_floor_energy_beat_is_junk_non_speech_energy()
    test_real_beat_is_not_junk_and_wins_as_a_singleton()
    test_group_of_real_and_junk_real_wins_junk_never_competes()
    test_group_all_junk_yields_only_junk_no_winner_no_crash()
    test_beat_for_a_file_with_no_loaded_inputs_is_dropped_defensively()
    print("\nall speech orchestrate tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
