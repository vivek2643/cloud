"""
Pure unit tests for the non-model parts of app.services.vcut.speech.
segment_llm -- span validation/overlap resolution and take-group building
with singleton completion (speech_cuts_pipeline.plan.md section 8). The
actual Gemini call (run_segment_llm) is real-money and exercised via the
live smoke run instead.

Run:  .venv/bin/python scripts/test_speech_segment_llm.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.vcut.speech.inputs import FileSpeechInputs, Word  # noqa: E402
from app.services.vcut.speech.segment_llm import (  # noqa: E402
    SegmentSchema, _BeatOut, _TakeGroupOut, _validate_beats, build_task_text,
    build_take_groups,
)


def _words(n):
    return [Word(idx=i, start_ms=i * 200, end_ms=i * 200 + 150, text=f"w{i}") for i in range(n)]


def _inputs(file_id, n_words):
    return FileSpeechInputs(file_id=file_id, duration_ms=n_words * 200, words=_words(n_words))


def _beat(id_, file_id, span, gist="g", fluency=0.5, flags=None):
    return _BeatOut(id=id_, file_id=file_id, word_span=span, gist=gist, fluency=fluency,
                    flags=flags or [])


# --------------------------------------------------------------------------
# _validate_beats
# --------------------------------------------------------------------------

def test_valid_nonoverlapping_beats_survive():
    inputs = {"f1": _inputs("f1", 10)}
    beats = [_beat("b0", "f1", (0, 3)), _beat("b1", "f1", (4, 8))]
    out = _validate_beats(beats, inputs)
    assert {b.id for b in out} == {"b0", "b1"}
    print("ok  test_valid_nonoverlapping_beats_survive")


def test_out_of_range_span_is_dropped():
    inputs = {"f1": _inputs("f1", 5)}
    beats = [_beat("b0", "f1", (0, 10))]  # word 10 doesn't exist (only 0..4)
    out = _validate_beats(beats, inputs)
    assert out == []
    print("ok  test_out_of_range_span_is_dropped")


def test_overlapping_beats_first_by_start_wins():
    inputs = {"f1": _inputs("f1", 10)}
    beats = [_beat("b0", "f1", (0, 5)), _beat("b1", "f1", (3, 8))]
    out = _validate_beats(beats, inputs)
    assert {b.id for b in out} == {"b0"}, {b.id for b in out}
    print("ok  test_overlapping_beats_first_by_start_wins")


def test_beat_for_unknown_file_is_dropped():
    inputs = {"f1": _inputs("f1", 10)}
    beats = [_beat("b0", "ghost_file", (0, 2))]
    out = _validate_beats(beats, inputs)
    assert out == []
    print("ok  test_beat_for_unknown_file_is_dropped")


def test_beats_across_different_files_never_conflict():
    inputs = {"f1": _inputs("f1", 5), "f2": _inputs("f2", 5)}
    beats = [_beat("b0", "f1", (0, 3)), _beat("b1", "f2", (0, 3))]
    out = _validate_beats(beats, inputs)
    assert {b.id for b in out} == {"b0", "b1"}
    print("ok  test_beats_across_different_files_never_conflict")


# --------------------------------------------------------------------------
# build_take_groups
# --------------------------------------------------------------------------

def test_grouped_beats_form_one_cluster():
    beats = [_beat("b0", "f1", (0, 2)), _beat("b1", "f1", (5, 7))]
    groups = build_take_groups(beats, [_TakeGroupOut(beat_ids=["b0", "b1"])])
    assert len(groups) == 1
    assert {b.id for b in groups[0]} == {"b0", "b1"}
    print("ok  test_grouped_beats_form_one_cluster")


def test_unclaimed_beat_becomes_its_own_singleton():
    beats = [_beat("b0", "f1", (0, 2)), _beat("b1", "f1", (5, 7))]
    groups = build_take_groups(beats, [_TakeGroupOut(beat_ids=["b0"])])
    # b0 is its own group (an LLM-declared group of one is still respected),
    # b1 is never mentioned -> becomes its own singleton.
    ids_per_group = sorted(tuple(sorted(b.id for b in g)) for g in groups)
    assert ids_per_group == [("b0",), ("b1",)], ids_per_group
    print("ok  test_unclaimed_beat_becomes_its_own_singleton")


def test_dangling_beat_id_reference_is_dropped():
    beats = [_beat("b0", "f1", (0, 2))]
    groups = build_take_groups(beats, [_TakeGroupOut(beat_ids=["b0", "ghost"])])
    assert len(groups) == 1 and {b.id for b in groups[0]} == {"b0"}
    print("ok  test_dangling_beat_id_reference_is_dropped")


def test_group_referencing_only_dangling_ids_is_skipped_entirely():
    beats = [_beat("b0", "f1", (0, 2))]
    groups = build_take_groups(beats, [_TakeGroupOut(beat_ids=["ghost1", "ghost2"])])
    # the empty group vanishes; b0 still surfaces as its own singleton.
    assert len(groups) == 1 and {b.id for b in groups[0]} == {"b0"}
    print("ok  test_group_referencing_only_dangling_ids_is_skipped_entirely")


def test_a_beat_claimed_by_an_earlier_group_is_not_double_claimed():
    beats = [_beat("b0", "f1", (0, 2)), _beat("b1", "f1", (5, 7)), _beat("b2", "f1", (10, 12))]
    groups = build_take_groups(beats, [
        _TakeGroupOut(beat_ids=["b0", "b1"]), _TakeGroupOut(beat_ids=["b1", "b2"]),
    ])
    all_ids = [b.id for g in groups for b in g]
    assert all_ids.count("b1") == 1, all_ids
    print("ok  test_a_beat_claimed_by_an_earlier_group_is_not_double_claimed")


def test_no_take_groups_at_all_every_beat_is_a_singleton():
    beats = [_beat("b0", "f1", (0, 2)), _beat("b1", "f1", (5, 7))]
    groups = build_take_groups(beats, [])
    assert len(groups) == 2
    assert all(len(g) == 1 for g in groups)
    print("ok  test_no_take_groups_at_all_every_beat_is_a_singleton")


# --------------------------------------------------------------------------
# build_task_text / schema sanity
# --------------------------------------------------------------------------

def test_build_task_text_includes_every_file_and_word():
    inputs = {"f1": _inputs("f1", 3)}
    text = build_task_text(inputs)
    assert "FILE f1" in text
    assert "0: w0" in text and "2: w2" in text
    print("ok  test_build_task_text_includes_every_file_and_word")


def test_schema_rejects_invalid_flag_values():
    from pydantic import ValidationError
    try:
        SegmentSchema.model_validate({"beats": [{"id": "b0", "file_id": "f1", "word_span": [0, 1],
                                                  "flags": ["not_a_real_flag"]}]})
        raise AssertionError("expected a ValidationError for an invalid flag")
    except ValidationError:
        pass
    print("ok  test_schema_rejects_invalid_flag_values")


# --------------------------------------------------------------------------
# _BeatOut.is_speech -- speech_noise_gate.plan.md Improvement A: fail-open
# default (True), and it round-trips both ways through raw JSON.
# --------------------------------------------------------------------------

def test_beat_out_is_speech_defaults_to_true_when_omitted():
    parsed = SegmentSchema.model_validate({"beats": [{"id": "b0", "file_id": "f1", "word_span": [0, 1]}]})
    assert parsed.beats[0].is_speech is True
    print("ok  test_beat_out_is_speech_defaults_to_true_when_omitted")


def test_beat_out_is_speech_false_round_trips():
    parsed = SegmentSchema.model_validate({
        "beats": [{"id": "b0", "file_id": "f1", "word_span": [0, 1], "is_speech": False}],
    })
    assert parsed.beats[0].is_speech is False
    print("ok  test_beat_out_is_speech_false_round_trips")


def test_beat_out_constructed_directly_defaults_is_speech_true():
    b = _beat("b0", "f1", (0, 1))
    assert b.is_speech is True
    print("ok  test_beat_out_constructed_directly_defaults_is_speech_true")


def main():
    test_valid_nonoverlapping_beats_survive()
    test_out_of_range_span_is_dropped()
    test_overlapping_beats_first_by_start_wins()
    test_beat_for_unknown_file_is_dropped()
    test_beats_across_different_files_never_conflict()
    test_grouped_beats_form_one_cluster()
    test_unclaimed_beat_becomes_its_own_singleton()
    test_dangling_beat_id_reference_is_dropped()
    test_group_referencing_only_dangling_ids_is_skipped_entirely()
    test_a_beat_claimed_by_an_earlier_group_is_not_double_claimed()
    test_no_take_groups_at_all_every_beat_is_a_singleton()
    test_build_task_text_includes_every_file_and_word()
    test_schema_rejects_invalid_flag_values()
    test_beat_out_is_speech_defaults_to_true_when_omitted()
    test_beat_out_is_speech_false_round_trips()
    test_beat_out_constructed_directly_defaults_is_speech_true()
    print("\nall speech segment_llm tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
