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


def test_production_cue_flagged_and_kept_out_of_topics():
    # An isolated "Go" at the very start (crew cue) then a real line by S0.
    words = [
        _w(0, 250, "Go", "S0"),                   # isolated edge cue
        _w(1500, 1900, "Today", "S0"),            # real speech, >600ms gap after cue
        _w(1930, 2300, "we're", "S0"),
        _w(2330, 2800, "cooking.", "S0"),
    ]
    out = d.build_dialogue_segments(words, wav_path=None)
    cue = [s for s in out["sentence"] if s["text"].strip().lower() == "go"]
    assert cue and "production_cue" in cue[0]["flags"], out["sentence"]
    # The cue must not appear inside any topic (kept clean).
    assert all("Go" not in t["text"].split()[:1] for t in out["topic"]), out["topic"]
    # A mid-sentence "go" must NOT be flagged.
    words2 = [
        _w(0, 300, "Let's", "S0"), _w(330, 600, "go", "S0"),
        _w(630, 1000, "there.", "S0"),
    ]
    out2 = d.build_dialogue_segments(words2, wav_path=None)
    assert all("production_cue" not in s["flags"] for s in out2["sentence"]), out2["sentence"]
    print("ok: isolated crew cue flagged + excluded from topics; mid-sentence 'go' kept")


def test_production_cue_survives_short_forward_merge():
    # "Cut" sits only 150ms before the real line -- old forward-merge would glue
    # them; merge-immunity keeps the cue atomic so it can be flagged + dropped.
    words = [
        _w(0, 250, "Cut", "S0"),
        _w(400, 800, "Today", "S0"),
        _w(830, 1200, "we're", "S0"),
        _w(1230, 1600, "cooking.", "S0"),
    ]
    out = d.build_dialogue_segments(words, wav_path=None)
    cue = [s for s in out["sentence"] if s["text"].strip().lower() == "cut"]
    assert cue and "production_cue" in cue[0]["flags"], out["sentence"]
    real = [s for s in out["sentence"] if "cooking" in s["text"]]
    assert real and "cut" not in real[0]["text"].lower().split()[:1], real[0]["text"]
    print("ok: production cue stays separate from following speech when gap < merge threshold")


def test_offscreen_loudness_flag():
    # speech_ref -5 dB; an isolated quiet word (-25 dB) at the start is off-mic.
    rms = [-5.0] * 400
    for f in range(0, 30):  # 0..300ms quiet
        rms[f] = -25.0
    env = d.Envelope(rms, hop_ms=10, speech_ref=-5.0)
    units = [
        d._Unit(speaker="S0", raw_in_ms=0, raw_out_ms=250, text="hey"),
        d._Unit(speaker="S0", raw_in_ms=1500, raw_out_ms=2500, text="the real line"),
    ]
    d._mark_offscreen_units(units, env, clip_start=0, clip_end=2500)
    assert "offscreen" in units[0].flags and "offscreen" not in units[1].flags, [u.flags for u in units]
    print("ok: isolated off-mic (quiet) speech flagged offscreen")


def test_diarization_smooths_phantom_speaker_blips():
    from app.services.l1 import diarization as dz
    # Trailing word flipped to a phantom S1 -> folds back into S0 ("matters").
    words = [
        {"start_ms": 0, "end_ms": 400}, {"start_ms": 450, "end_ms": 900},
        {"start_ms": 950, "end_ms": 1400}, {"start_ms": 2000, "end_ms": 2400},
    ]
    spk = ["S0", "S0", "S0", "S1"]
    dz._smooth_speakers(words, spk)
    assert spk == ["S0", "S0", "S0", "S0"], spk
    # A sandwiched 1-word other-speaker blip also collapses.
    words2 = [{"start_ms": i * 500, "end_ms": i * 500 + 400} for i in range(5)]
    spk2 = ["S0", "S0", "S1", "S0", "S0"]
    dz._smooth_speakers(words2, spk2)
    assert spk2 == ["S0"] * 5, spk2
    # A genuine, long second-speaker turn is NOT smoothed away.
    words3 = [{"start_ms": i * 600, "end_ms": i * 600 + 500} for i in range(6)]
    spk3 = ["S0", "S0", "S1", "S1", "S1", "S0"]
    dz._smooth_speakers(words3, spk3)
    assert spk3 == ["S0", "S0", "S1", "S1", "S1", "S0"], spk3
    print("ok: diarization folds phantom trailing/sandwiched blips, keeps real turns")


def test_speaker_embeddings_maps_labels_and_drops_nan():
    from app.services.l1 import diarization as dz

    class _FakeAnnotation:
        def labels(self):
            return ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"]

    def fake_pipe(wav_path, return_embeddings=False, **kwargs):
        assert return_embeddings is True
        return _FakeAnnotation(), [
            [0.1, 0.2, 0.3],
            [float("nan"), 0.5, 0.6],   # dropped -- NaN
            [0.7, 0.8],
        ]

    annotation, by_label = dz._speaker_embeddings(fake_pipe, "x.wav", {})
    assert isinstance(annotation, _FakeAnnotation)
    assert by_label == {"SPEAKER_00": [0.1, 0.2, 0.3], "SPEAKER_02": [0.7, 0.8]}, by_label
    print("ok: _speaker_embeddings maps pyannote labels to vectors, drops NaN")


def test_speaker_embeddings_falls_back_on_typeerror():
    from app.services.l1 import diarization as dz

    class _FakeAnnotation:
        def labels(self):
            return []

    def fake_pipe(wav_path, **kwargs):
        # Simulates an older pyannote build that doesn't accept the kwarg.
        if "return_embeddings" in kwargs:
            raise TypeError("unexpected keyword argument 'return_embeddings'")
        return _FakeAnnotation()

    annotation, by_label = dz._speaker_embeddings(fake_pipe, "x.wav", {})
    assert isinstance(annotation, _FakeAnnotation)
    assert by_label == {}, by_label
    print("ok: _speaker_embeddings falls back cleanly on an older pyannote build")


def test_rejoin_trailing_fragment():
    # "...what actually" (unfinished) + a 0.5s "matters" tail after a ~700ms pause.
    words = [
        _w(2000, 2400, "what", "S0"), _w(2430, 3600, "actually", "S0"),
        _w(4300, 4800, "matters", "S0"),
    ]
    out = d.build_dialogue_segments(words, wav_path=None)
    texts = [s["text"] for s in out["sentence"]]
    assert len(texts) == 1 and "actually matters" in texts[0], texts
    # But a finished previous line (terminal punctuation) does NOT absorb a tail.
    words2 = [
        _w(2000, 2400, "Done", "S0"), _w(2430, 3600, "already.", "S0"),
        _w(4300, 4800, "Matters", "S0"),
    ]
    out2 = d.build_dialogue_segments(words2, wav_path=None)
    assert len(out2["sentence"]) == 2, [s["text"] for s in out2["sentence"]]
    print("ok: short tail rejoins an unfinished line; a finished line stays split")


def main():
    test_phrases_split_on_speaker_and_gap()
    test_sentences_break_on_punctuation_and_speaker()
    test_backchannel_detection_and_topic_bridging()
    test_snap_to_silence_trough_and_noisy_fallback()
    test_overlap_flag_on_crosstalk()
    test_end_to_end_silent_envelope()
    test_production_cue_flagged_and_kept_out_of_topics()
    test_production_cue_survives_short_forward_merge()
    test_offscreen_loudness_flag()
    test_diarization_smooths_phantom_speaker_blips()
    test_speaker_embeddings_maps_labels_and_drops_nan()
    test_speaker_embeddings_falls_back_on_typeerror()
    test_rejoin_trailing_fragment()
    print("\nALL DIALOGUE SEGMENT TESTS PASSED")


if __name__ == "__main__":
    main()
