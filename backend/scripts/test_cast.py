"""
Regression tests for the cast / speaker map (no DB, no VLM).

Exercises the A/V fusion: link a visible person to the voice that dominates
their on-camera speech, mark unlinked voices off-camera / unknown (never drop),
and resolve display labels + on-camera ratios. Run:
  .venv/bin/python scripts/test_cast.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import cast as cst  # noqa: E402


def _w(text, s, e, spk):
    return {"text": text, "start_ms": s, "end_ms": e, "speaker": spk, "is_filler": False}


def test_links_person_to_dominant_voice():
    """Two on-camera people, two voices: each person links to the voice that
    speaks while they're visibly speaking, with full confidence + region."""
    words = [
        _w("hello", 0, 500, "S0"), _w("there", 500, 1000, "S0"),
        _w("hi", 2000, 2500, "S1"), _w("back", 2500, 3000, "S1"),
    ]
    perception = {
        "persons": [
            {"local_id": "p1", "role": "interviewer", "frame_region": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.5}},
            {"local_id": "p2", "role": "main subject", "frame_region": {"x": 0.6, "y": 0.1, "w": 0.3, "h": 0.5}},
        ],
        "speaking": [
            {"subject": "p1", "start_ms": 0, "end_ms": 1000},
            {"subject": "p2", "start_ms": 2000, "end_ms": 3000},
        ],
    }
    cast = cst.build_cast(perception, words)
    assert cast.has_visible_speakers
    m0 = cast.resolve("S0")
    m1 = cast.resolve("S1")
    assert m0 and m0.person_id == "p1" and m0.role == "interviewer", m0
    assert m0.on_camera is True and m0.av_link_confidence == 1.0, m0
    assert m0.region == {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.5}, m0.region
    assert m1 and m1.person_id == "p2", m1
    assert cast.display_label("S0") == "interviewer"
    assert cast.display_label("S1") == "main subject"
    print("ok  test_links_person_to_dominant_voice")


def test_offcamera_voice_kept_not_dropped():
    """A voice with no visible speaker is kept as an OFF-camera member (the clip
    has visible speakers elsewhere, so we can tell it's off)."""
    words = [
        _w("on", 0, 500, "S0"), _w("camera", 500, 1000, "S0"),
        _w("a", 5000, 5300, "S9"), _w("voice", 5300, 5800, "S9"), _w("offscreen", 5800, 6200, "S9"),
    ]
    perception = {
        "persons": [{"local_id": "p1", "role": "host"}],
        "speaking": [{"subject": "p1", "start_ms": 0, "end_ms": 1000}],
    }
    cast = cst.build_cast(perception, words)
    off = cast.resolve("S9")
    assert off is not None, "off-camera voice must not be discarded"
    assert off.person_id is None and off.on_camera is False, off
    assert cast.display_label("S9") == "S9"   # falls back to the raw voice id
    print("ok  test_offcamera_voice_kept_not_dropped")


def test_no_visible_speaking_is_unknown_not_offcamera():
    """With no speaking spans at all (e.g. audio-only clip), camera state is
    UNKNOWN (None), never a false 'off-camera'."""
    words = [_w("just", 0, 400, "S0"), _w("audio", 400, 900, "S0")]
    cast = cst.build_cast({"persons": []}, words)
    m = cast.resolve("S0")
    assert m is not None and m.on_camera is None, m
    assert not cast.has_visible_speakers
    print("ok  test_no_visible_speaking_is_unknown_not_offcamera")


def test_on_camera_ratio():
    """on_camera_ratio measures how much of a window the owner is visibly
    speaking -- the signal the take picker uses to prefer on-screen takes."""
    words = [_w("x", 0, 1000, "S0")]
    perception = {
        "persons": [{"local_id": "p1"}],
        "speaking": [{"subject": "p1", "start_ms": 0, "end_ms": 1000}],
    }
    m = cst.build_cast(perception, words).resolve("S0")
    assert m.on_camera_ratio(0, 1000) == 1.0
    assert m.on_camera_ratio(0, 2000) == 0.5
    assert m.on_camera_at(500) is True and m.on_camera_at(1500) is False
    print("ok  test_on_camera_ratio")


def test_silent_visible_person_carried():
    """A visible person who never links to a voice is still carried (so labels /
    reframing can reach them), as a voiceless member."""
    words = [_w("speaking", 0, 500, "S0"), _w("now", 500, 1000, "S0")]
    perception = {
        "persons": [
            {"local_id": "p1"},
            {"local_id": "p2", "role": "guest", "frame_region": {"x": 0.5, "y": 0.0, "w": 0.4, "h": 0.6}},
        ],
        "speaking": [{"subject": "p1", "start_ms": 0, "end_ms": 1000}],
    }
    cast = cst.build_cast(perception, words)
    voiceless = [m for m in cast.members if m.person_id == "p2"]
    assert len(voiceless) == 1 and voiceless[0].voice_speaker_id is None, voiceless
    assert voiceless[0].role == "guest"
    print("ok  test_silent_visible_person_carried")


def main():
    test_links_person_to_dominant_voice()
    test_offcamera_voice_kept_not_dropped()
    test_no_visible_speaking_is_unknown_not_offcamera()
    test_on_camera_ratio()
    test_silent_visible_person_carried()
    print("\nall cast tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
