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


def test_mapping_gaps_now_surface():
    """The three previously-ignored-but-stored tracks now become anchors:
    performance content_units -> action (sync); gaze -> reaction (overlay);
    interactions -> insert (overlay). 'conversation' interactions stay in speech."""
    perception = {
        "content_units": [{"unit_id": "perf1", "kind": "performance",
                           "label": "sings the chorus", "start_ms": 1000, "end_ms": 5000}],
        "gaze": [
            # baseline eyeline = to_camera (longest) -> not a cutaway
            {"start_ms": 0, "end_ms": 4000, "subject": "p1", "direction": "to_camera"},
            # a held DEPARTURE to an object -> surfaces
            {"start_ms": 6000, "end_ms": 7200, "subject": "p1",
             "direction": "at_object", "target": "the prototype"},
            # a micro-dart departure -> too short, dropped
            {"start_ms": 8000, "end_ms": 8100, "subject": "p1", "direction": "off_camera"},
        ],
        "interactions": [
            {"id": "i1", "kind": "handshake", "start_ms": 11000, "end_ms": 12500,
             "participants": ["p1", "p2"], "description": "p1 and p2 shake hands"},
            {"id": "i2", "kind": "conversation", "start_ms": 0, "end_ms": 30000,
             "participants": ["p1", "p2"]},  # just talking -> skipped
        ],
        "take_quality_events": [],
    }
    al = an.gather_anchors(duration_ms=30000, perception=perception)
    perf = [a for a in al if a.kind == "action_beat"]
    assert perf and perf[0].affordance == an.AFF_ACTION and perf[0].audio_role == an.SYNC, al
    gz = [a for a in al if a.kind == "gaze"]
    assert len(gz) == 1 and gz[0].affordance == an.AFF_REACTION and gz[0].audio_role == an.OVERLAY, gz
    assert "at object" in gz[0].text, gz[0].text   # the departure, not the baseline
    it = [a for a in al if a.kind == "interaction"]
    assert len(it) == 1 and it[0].affordance == an.AFF_INSERT and it[0].audio_role == an.OVERLAY, it
    assert all("conversation" not in (a.text or "") for a in it)
    print("ok  mapping gaps surface (performance/gaze/interaction)")


def test_audio_event_is_sound_minus_speech():
    """A loud, non-speech region (laughter/applause) becomes a SYNC audio_event
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
    assert ev[0].audio_role == an.SYNC, ev[0].audio_role
    assert 5500 <= ev[0].ts_ms <= 8000, ev[0].ts_ms
    print("ok  audio event = sound minus speech (sync)")


def test_audio_role_is_a_field_not_a_constant():
    """audio_role defaults from affordance but can be overridden per-anchor, so an
    audible reaction is sync while a silent reaction stays overlay."""
    silent = an.Anchor(0, 0, 1, "expression", an.AFF_REACTION)
    audible = an.Anchor(0, 0, 1, "audio_event", an.AFF_REACTION, audio_role=an.SYNC)
    assert silent.audio_role == an.OVERLAY and audible.audio_role == an.SYNC
    print("ok  audio_role is a per-anchor field")


def test_audio_role_split():
    """speech/action carry sync audio; reaction/broll/insert are silent overlays."""
    assert an.Anchor(0, 0, 1, "speech", an.AFF_SPEECH).audio_role == an.SYNC
    assert an.Anchor(0, 0, 1, "action_beat", an.AFF_ACTION).audio_role == an.SYNC
    assert an.Anchor(0, 0, 1, "expression", an.AFF_REACTION).audio_role == an.OVERLAY
    assert an.Anchor(0, 0, 1, "hold", an.AFF_BROLL).audio_role == an.OVERLAY
    assert an.Anchor(0, 0, 1, "reveal", an.AFF_INSERT).audio_role == an.OVERLAY
    print("ok  audio role split (sync vs overlay)")


def test_salience_from_intensity_and_words():
    """Reaction salience tracks intensity; speech salience tracks word count."""
    p = {"reactions": [
        {"start_ms": 0, "end_ms": 500, "subject": "p1", "type": "nod", "intensity": 0.2},
        {"start_ms": 1000, "end_ms": 1500, "subject": "p1", "type": "laugh", "intensity": 0.9},
    ]}
    al = an.gather_anchors(duration_ms=5000, perception=p)
    rx = sorted([a for a in al if a.affordance == an.AFF_REACTION], key=lambda a: a.salience)
    assert rx[0].salience < rx[-1].salience and rx[-1].salience >= 0.85, [a.salience for a in rx]
    print("ok  salience from intensity / words")


def test_cutaway_path_preferred_over_legacy():
    """When cutaways is populated, legacy reactions/broll/insert tracks are ignored."""
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
    print("ok  cutaway path preferred over legacy")


def test_legacy_overlay_fallback():
    """Old L2 JSON without cutaways still surfaces overlays from legacy tracks."""
    perception = {
        "reactions": [{"start_ms": 0, "end_ms": 500, "subject": "p1", "type": "nod", "intensity": 0.5}],
        "camera_craft": [{"start_ms": 0, "end_ms": 5000, "movement": "static", "subject_focus": "wide"}],
    }
    al = an.gather_anchors(duration_ms=10000, perception=perception)
    assert an.AFF_REACTION in {a.affordance for a in al}
    assert an.AFF_BROLL in {a.affordance for a in al}
    print("ok  legacy overlay fallback")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall anchor tests passed")
