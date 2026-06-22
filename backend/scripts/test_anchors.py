#!/usr/bin/env python3
"""Tests for the anchor layer (no DB). Run: python scripts/test_anchors.py"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import anchors as an  # noqa: E402


def _sentence(seg_id, text, a, b, speaker="S0", flags=None):
    return {"seg_id": seg_id, "text": text, "src_in_ms": a, "src_out_ms": b,
            "raw_in_ms": a, "raw_out_ms": b, "speaker": speaker, "flags": flags or []}


def test_gathers_every_kind():
    """Each stored track contributes its anchor; off-camera speech is dropped."""
    dialogue = {"sentence": [
        _sentence("s0", "this is a clean usable on camera line here", 1000, 4000),
        _sentence("s1", "go",  100, 400, flags=["production_cue"]),   # crew cue -> dropped
    ]}
    perception = {
        "content_units": [{"unit_id": "u1", "kind": "action", "label": "drops and misses",
                           "start_ms": 6000, "end_ms": 7500}],
        "cutaways": [
            {"start_ms": 7500, "end_ms": 8000, "kind": "reaction", "affordance": "reaction",
             "subject": "p1", "label": "smile · the miss", "intensity": 0.7, "trigger": "the miss"},
            {"start_ms": 0, "end_ms": 5000, "kind": "broll_hold", "affordance": "broll",
             "label": "p1 face"},
            {"start_ms": 22000, "end_ms": 23000, "kind": "reveal", "affordance": "insert",
             "label": "slide changes to the insight"},
        ],
        "take_quality_events": [],
    }
    motion = {"hop_ms": 100, "action_energy": [0.8] * 100,
              "action_points": [{"ts_ms": 6800, "score": 1.0}]}

    al = an.gather_anchors(duration_ms=30000, dialogue=dialogue,
                           perception=perception, motion=motion)
    affs = {a.affordance for a in al}
    assert an.AFF_SPEECH in affs and an.AFF_ACTION in affs and an.AFF_REACTION in affs
    assert an.AFF_BROLL in affs and an.AFF_INSERT in affs, affs
    # off-camera "go" never becomes an anchor
    assert all(a.text != "go" for a in al)
    # time sorted
    assert [a.start_ms for a in al] == sorted(a.start_ms for a in al)
    print("ok  gathers every kind, drops off-camera")


def test_action_ts_is_the_impact():
    """The action anchor's representative instant snaps to the strongest motion
    impact inside the beat (the contact frame), not the midpoint."""
    perception = {"content_units": [{"unit_id": "u1", "kind": "action",
                                     "start_ms": 6000, "end_ms": 7500, "label": "swing"}],
                  "take_quality_events": []}
    motion = {"hop_ms": 100, "action_energy": [0.7] * 100,
              "action_points": [{"ts_ms": 6800, "score": 1.0}, {"ts_ms": 6200, "score": 0.4}]}
    al = an.gather_anchors(duration_ms=10000, perception=perception, motion=motion)
    act = [a for a in al if a.affordance == an.AFF_ACTION]
    assert len(act) == 1 and act[0].ts_ms == 6800, act
    assert act[0].start_ms == 6000 and act[0].end_ms == 7500   # core extent preserved
    print("ok  action ts = strongest impact, core extent kept")


def test_overlay_kinds_surface_from_cutaways():
    """Performance content_units -> action (sync). The sparse cutaways track
    carries the full overlay vocabulary: a gaze departure -> reaction, an
    interaction -> insert."""
    perception = {
        "content_units": [{"unit_id": "perf1", "kind": "performance",
                           "label": "sings the chorus", "start_ms": 1000, "end_ms": 5000}],
        "cutaways": [
            {"start_ms": 6000, "end_ms": 7200, "kind": "gaze", "affordance": "reaction",
             "subject": "p1", "label": "looks at the prototype"},
            {"start_ms": 11000, "end_ms": 12500, "kind": "interaction", "affordance": "insert",
             "label": "handshake · p1 and p2 shake hands"},
        ],
        "take_quality_events": [],
    }
    al = an.gather_anchors(duration_ms=30000, perception=perception)
    # The fixture's content_unit is a performance -> kind preserved, action bucket.
    perf = [a for a in al if a.kind == "performance"]
    assert perf and perf[0].affordance == an.AFF_ACTION, al
    gz = [a for a in al if a.kind == "gaze"]
    assert len(gz) == 1 and gz[0].affordance == an.AFF_REACTION, gz
    it = [a for a in al if a.kind == "interaction"]
    assert len(it) == 1 and it[0].affordance == an.AFF_INSERT, it
    print("ok  overlay kinds surface from cutaways (performance/gaze/interaction)")


def test_audio_event_is_sound_minus_speech():
    """A loud, non-speech region (laughter/applause) becomes an audio_event
    anchor; loud regions covered by speech do NOT (that's just speech)."""
    hop = 100
    n = 100                       # 10s envelope
    rms = [-40.0] * n             # quiet floor
    for i in range(10, 30):       # speech 1.0-3.0s, loud
        rms[i] = -12.0
    for i in range(60, 75):       # non-speech burst 6.0-7.5s, loud (a laugh)
        rms[i] = -10.0
    audio = {"rms_db": rms, "prosody_hop_ms": hop}
    sentences = [{"src_in_ms": 1000, "src_out_ms": 3000}]
    al = an.gather_anchors(duration_ms=10000, dialogue={"sentence": sentences}, audio=audio)
    ev = [a for a in al if a.kind == "audio_event"]
    assert len(ev) == 1, ev                      # the burst, not the speech
    assert 5500 <= ev[0].ts_ms <= 8000, ev[0].ts_ms
    print("ok  audio event = sound minus speech")


def test_salience_from_intensity():
    """Reaction salience tracks the cutaway's intensity."""
    p = {"cutaways": [
        {"start_ms": 0, "end_ms": 500, "kind": "reaction", "affordance": "reaction",
         "subject": "p1", "label": "nod", "intensity": 0.2},
        {"start_ms": 1000, "end_ms": 1500, "kind": "reaction", "affordance": "reaction",
         "subject": "p1", "label": "laugh", "intensity": 0.9},
    ]}
    al = an.gather_anchors(duration_ms=5000, perception=p)
    rx = sorted([a for a in al if a.affordance == an.AFF_REACTION], key=lambda a: a.salience)
    assert rx[0].salience < rx[-1].salience and rx[-1].salience >= 0.85, [a.salience for a in rx]
    print("ok  salience from intensity")


def test_overlays_come_only_from_cutaways():
    """Overlays are read solely from the sparse cutaways track; the raw L2
    reactions / camera_craft tracks are never minted into anchors."""
    perception = {
        "cutaways": [
            {"start_ms": 1000, "end_ms": 2000, "kind": "reaction", "affordance": "reaction",
             "subject": "p2", "label": "listener laugh", "intensity": 0.8},
        ],
        "reactions": [
            {"start_ms": 1000, "end_ms": 2000, "subject": "p1", "type": "nod", "intensity": 0.3},
        ],
        "camera_craft": [{"start_ms": 0, "end_ms": 8000, "movement": "static"}],
    }
    al = an.gather_anchors(duration_ms=10000, perception=perception)
    rx = [a for a in al if a.affordance == an.AFF_REACTION]
    assert len(rx) == 1 and "listener laugh" in rx[0].text, rx
    assert an.AFF_BROLL not in {a.affordance for a in al}
    print("ok  overlays come only from cutaways")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall anchor tests passed")
