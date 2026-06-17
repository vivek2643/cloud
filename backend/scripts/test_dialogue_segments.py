"""
Regression tests for the Dialogues lens segmenter (l1/dialogue_segments.py).

Pure-Python, no DB / no audio file needed: snapping is tested with a synthetic
Envelope. Run:  python backend/scripts/test_dialogue_segments.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l1 import dialogue_segments as d  # noqa: E402


def _w(start, end, text, speaker="S0", filler=False):
    return {"start_ms": start, "end_ms": end, "text": text,
            "is_filler": filler, "speaker": speaker}


def test_phrases_split_on_speaker_and_gap():
    words = [
        _w(0, 300, "Hello", "S0"),
        _w(330, 700, "there.", "S0"),          # same speaker, tiny gap -> same phrase
        _w(2000, 2300, "Hi", "S1"),            # speaker change -> new phrase
        _w(2350, 2600, "back.", "S1"),
        _w(4000, 4300, "Again", "S1"),         # same speaker, big gap -> new phrase
    ]
    phrases = d.build_phrases(words)
    assert len(phrases) == 3, [[(w["text"]) for w in p] for p in phrases]
    assert [p[0]["speaker"] for p in phrases] == ["S0", "S1", "S1"]
    print("ok: phrases split on speaker change + gap")


def test_sentences_break_on_punctuation_and_speaker():
    words = [
        _w(0, 300, "Hello", "S0"),
        _w(330, 700, "there.", "S0"),          # sentence-final -> boundary after
        _w(750, 1100, "How", "S0"),
        _w(1130, 1500, "are", "S0"),
        _w(1530, 1900, "you?", "S0"),          # sentence-final
        _w(3000, 3300, "Good.", "S1"),         # speaker change
    ]
    sents = d.merge_sentences(words)
    texts = [s.text for s in sents]
    assert texts == ["Hello there.", "How are you?", "Good."], texts
    assert [s.speaker for s in sents] == ["S0", "S0", "S1"]
    print("ok: sentences break on punctuation + speaker change")


def test_backchannel_detection_and_topic_bridging():
    words = [
        # interviewer question
        _w(0, 400, "What", "S0"), _w(420, 900, "happened?", "S0"),
        # answer part 1
        _w(1500, 1900, "We", "S1"), _w(1930, 2600, "won.", "S1"),
        # interviewer backchannel in the middle
        _w(2700, 2900, "mhm", "S0"),
        # answer part 2 (same speaker, should join the topic)
        _w(3100, 3500, "And", "S1"), _w(3530, 4200, "celebrated.", "S1"),
        # interviewer new turn after a gap (distinct speaker -> new topic)
        _w(6000, 6300, "Alright.", "S0"), _w(6330, 6900, "Next.", "S0"),
    ]
    sents = d.merge_sentences(words)
    # the lone "mhm" sentence must be flagged a backchannel
    bc = [s for s in sents if s.is_backchannel]
    assert len(bc) == 1 and "mhm" in bc[0].text.lower(), [s.text for s in sents]

    topics = d.build_topics(sents)
    # S0 question | S1 answer (bridged across mhm) | S0 next turn
    assert len(topics) == 3, [(t.speaker, t.text) for t in topics]
    answer = topics[1]
    assert answer.speaker == "S1"
    assert "won." in answer.text and "celebrated." in answer.text, answer.text
    print("ok: backchannel detected + topic bridges across it")


def test_snap_to_silence_trough_and_noisy_fallback():
    # 2000ms @ 10ms hop = 200 frames. Speech ~ -5 dB; a clean dip to -40 at 900ms.
    rms = [-5.0] * 200
    rms[90] = -40.0  # trough at 900ms
    env = d.Envelope(rms, hop_ms=10, speech_ref=-5.0)

    t, _v, clean = env.trough(800, 1000)
    assert clean and t == 900, (t, clean)

    # snap_out near 850 should land on the 900ms trough
    src_out, noisy = d._snap_out(env, raw_out=850, next_in=None)
    assert src_out == 900 and not noisy, (src_out, noisy)

    # a window with no dip -> noisy fallback uses a fixed handle
    flat = d.Envelope([-5.0] * 200, hop_ms=10, speech_ref=-5.0)
    src_out2, noisy2 = d._snap_out(flat, raw_out=850, next_in=None)
    assert noisy2 and src_out2 == 850 + d.DEFAULT_HANDLE_MS, (src_out2, noisy2)
    print("ok: snapping hits silence trough; noisy gap falls back to handle")


def test_overlap_flag_on_crosstalk():
    segs = [
        {"speaker": "S0", "raw_in_ms": 0, "raw_out_ms": 1200, "flags": []},
        {"speaker": "S1", "raw_in_ms": 1000, "raw_out_ms": 2000, "flags": []},  # overlaps S0
    ]
    d._flag_overlaps(segs)
    assert "overlap" in segs[0]["flags"] and "overlap" in segs[1]["flags"]
    print("ok: cross-talk overlap flagged on both clips")


def test_end_to_end_silent_envelope():
    words = [
        _w(0, 300, "Hello", "S0"), _w(330, 700, "there.", "S0"),
        _w(2000, 2400, "Yes", "S1"), _w(2430, 2900, "indeed.", "S1"),
    ]
    out = d.build_dialogue_segments(words, wav_path=None)
    assert set(out.keys()) == {"sentence", "topic"}
    assert len(out["sentence"]) == 2 and len(out["topic"]) == 2
    s0 = out["sentence"][0]
    # contract: required keys present, src window valid, topic linkage stamped
    for k in ("seg_id", "level", "speaker", "text", "src_in_ms", "src_out_ms",
              "raw_in_ms", "raw_out_ms", "fade_in_ms", "flags", "topic_id"):
        assert k in s0, k
    assert s0["src_out_ms"] > s0["src_in_ms"]
    assert out["topic"][0]["child_seg_ids"], out["topic"][0]
    assert out["sentence"][0]["topic_id"] == 0
    print("ok: end-to-end build returns linked sentence + topic selects")


def main():
    test_phrases_split_on_speaker_and_gap()
    test_sentences_break_on_punctuation_and_speaker()
    test_backchannel_detection_and_topic_bridging()
    test_snap_to_silence_trough_and_noisy_fallback()
    test_overlap_flag_on_crosstalk()
    test_end_to_end_silent_envelope()
    print("\nALL DIALOGUE SEGMENT TESTS PASSED")


if __name__ == "__main__":
    main()
