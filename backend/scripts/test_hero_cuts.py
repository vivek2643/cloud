"""
Regression tests for the hero-cuts assembly engine (no DB).

Exercises the pure logic: speech candidates from the dialogue lens (with
off-camera filtering + energy-driven granularity), action candidates snapped to
the motion grid, and take stacking (repeats collapse into one hero, best in
front). Run:  .venv/bin/python scripts/test_hero_cuts.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import hero_cuts as hc  # noqa: E402
from app.services.l3 import score_span as ss  # noqa: E402


def _src(file_id: str, duration_ms: int, words):
    return ss.SpanSource(
        file_id=file_id, duration_ms=duration_ms, words=words,
        fillers=[], silences=[], gaze=[], quality_events=[],
    )


def _seg(seg_id, level, text, in_ms, out_ms, speaker="S0", flags=None):
    return {
        "seg_id": seg_id, "level": level, "text": text,
        "src_in_ms": in_ms, "src_out_ms": out_ms,
        "raw_in_ms": in_ms, "raw_out_ms": out_ms,
        "speaker": speaker, "flags": flags or [],
    }


def _words(spec):
    """spec: list of (text, start_ms, end_ms) -> word dicts."""
    return [{"text": t, "start_ms": s, "end_ms": e, "is_filler": False} for t, s, e in spec]


def test_speech_drops_offcamera_and_short():
    """Off-camera/production-cue/backchannel selects and sub-min-word fragments
    are not surfaced as heroes."""
    words = _words([
        ("this", 0, 300), ("is", 300, 500), ("a", 500, 600),
        ("real", 600, 900), ("usable", 900, 1300), ("line", 1300, 1700),
        ("action", 5000, 5300),  # crew cue
    ])
    clip = hc._ClipInputs(
        file_id="aaaaaaaa-1", duration_ms=8000,
        dialogue={"topic": [
            _seg("t0", "topic", "this is a real usable line", 0, 1700),
            _seg("t1", "topic", "action", 5000, 5300, flags=["production_cue"]),
            _seg("t2", "topic", "yeah", 6000, 6200, flags=["backchannel"]),
        ], "sentence": []},
        perception=None, motion=None,
    )
    heroes = hc._speech_candidates(clip, _src("aaaaaaaa-1", 8000, words), energy=0.0)
    assert len(heroes) == 1, [h.label for h in heroes]
    assert heroes[0].label == "this is a real usable line"
    assert heroes[0].modality == "speech"
    print("ok  test_speech_drops_offcamera_and_short")


def test_energy_selects_granularity():
    """Low energy -> topic spans; high energy -> sentence spans."""
    clip = hc._ClipInputs(
        file_id="bbbbbbbb-1", duration_ms=8000,
        dialogue={
            "topic": [_seg("t0", "topic", "one big complete thought here now", 0, 4000)],
            "sentence": [
                _seg("s0", "sentence", "one big complete", 0, 2000),
                _seg("s1", "sentence", "thought here now", 2000, 4000),
            ],
        },
        perception=None, motion=None,
    )
    words = _words([("one", 0, 400), ("big", 400, 800), ("complete", 800, 1400),
                    ("thought", 2000, 2500), ("here", 2500, 2900), ("now", 2900, 3400)])
    src = _src("bbbbbbbb-1", 8000, words)
    low = hc._speech_candidates(clip, src, energy=0.0)
    high = hc._speech_candidates(clip, src, energy=1.0)
    assert len(low) == 1, low
    assert len(high) == 2, high
    print("ok  test_energy_selects_granularity")


def test_take_stacking_collapses_repeats(monkeypatch=None):
    """A take group (from l3.takes) of two deliveries collapses into ONE hero
    with take_count=2, higher-scoring delivery in front, loser as an alt. An
    unrelated hero stays solo. Stacking maps the group's attempts onto heroes by
    time-overlap, so `build_take_groups` is stubbed to isolate the fold logic."""
    from app.services.l3 import takes as tk

    a = hc.HeroCut("h1", "fff", "speech", "the product changes everything", 1000, 2000, score=0.6)
    b = hc.HeroCut("h2", "fff", "speech", "the product changes everything", 5000, 6000, score=0.9)
    c = hc.HeroCut("h3", "fff", "speech", "a different sentence about pricing", 8000, 9000, score=0.7)

    group = tk.TakeGroup(group_id="tg1", content_key="product changes everything", attempts=[
        tk.Attempt("fff:u1:0", "fff", "u1", 1000, 2000, "speech", "product changes everything", "...", False),
        tk.Attempt("fff:u2:0", "fff", "u2", 5000, 6000, "speech", "product changes everything", "...", False),
    ])
    orig = hc.build_take_groups
    hc.build_take_groups = lambda file_ids: [group]
    try:
        stacked = hc._stack_takes([a, b, c], ["fff"])
    finally:
        hc.build_take_groups = orig

    assert len(stacked) == 2, [h.label for h in stacked]
    repeated = next(h for h in stacked if "product" in h.label)
    assert repeated.take_count == 2
    assert repeated.score == 0.9            # best in front
    assert len(repeated.alt_takes) == 1 and repeated.alt_takes[0].score == 0.6
    assert any("pricing" in h.label and h.take_count == 1 for h in stacked)
    print("ok  test_take_stacking_collapses_repeats")


def test_action_snaps_to_calm_motion_seam():
    """Fallback path (no fused field): an action content_unit is snapped to the
    calmest (lowest camera_cut_cost) frame near each raw boundary."""
    # hop=100ms; calm (0.05) dips sit at 800ms (idx8) and 2100ms (idx21).
    cost = [0.9] * 30
    cost[8] = 0.05    # calm just before the action -> in-point
    cost[21] = 0.05   # calm just after -> out-point
    motion = {
        "hop_ms": 100,
        "action_energy": [0.7] * 30,
        "action_cut_cost": [0.0] * 30,
        "camera_cut_cost": cost,
        "action_points": [],
    }
    clip = hc._ClipInputs(
        file_id="cccccccc-1", duration_ms=3000,
        dialogue={"topic": [], "sentence": []},
        perception={"content_units": [
            {"unit_id": "u1", "kind": "action", "label": "hits the ball",
             "start_ms": 1000, "end_ms": 2000},
        ], "take_quality_events": []},
        motion=motion,
    )
    heroes = hc._action_candidates(clip, energy=0.0, field=None)
    assert len(heroes) == 1, heroes
    h = heroes[0]
    assert h.modality == "action" and h.label == "hits the ball"
    assert h.src_in_ms == 800, h.src_in_ms
    assert h.src_out_ms == 2100, h.src_out_ms
    print("ok  test_action_snaps_to_calm_motion_seam")


def test_action_fused_avoids_speech():
    """With a fused field present, an action unit whose raw out-point sits inside
    speech is pulled OUT of the spoken region instead of bleeding into it."""
    n = 30
    dlg = [0.0] * n
    for i in range(18, 26):       # speech (dialogue veto) over 1.8s..2.6s
        dlg[i] = 1.0
    action_cost = [1.0] * n
    action_cost[10] = 0.0          # motion impact at 1.0s (attractor for the in-point)
    motion = {
        "hop_ms": 100, "action_energy": [0.7] * n,
        "action_cut_cost": action_cost, "camera_cut_cost": [0.2] * n, "action_points": [],
    }
    clip = hc._ClipInputs(
        file_id="eeeeeeee-1", duration_ms=3000,
        dialogue={"topic": [], "sentence": []},
        perception={"content_units": [
            {"unit_id": "u1", "kind": "action", "label": "swing",
             "start_ms": 1000, "end_ms": 2000},  # raw out=2.0s is INSIDE speech
        ], "take_quality_events": []},
        motion=motion,
        audio={"dialogue_cut_cost": dlg, "dialogue_cut_hop_ms": 100, "dialogue_cut_points": [],
               "beat_cut_cost": [], "beat_cut_hop_ms": 100, "beat_cut_points": []},
    )
    field = hc._build_field(clip, energy=0.5)
    assert field is not None
    heroes = hc._action_candidates(clip, energy=0.5, field=field)
    assert len(heroes) == 1, heroes
    h = heroes[0]
    assert not (1800 < h.src_out_ms < 2600), h.src_out_ms   # not inside the speech
    assert h.src_out_ms <= 1800, h.src_out_ms               # trimmed to before it
    print("ok  test_action_fused_avoids_speech")


def test_action_skipped_without_motion():
    """No motion grid -> no action heroes (no deterministic boundary to snap)."""
    clip = hc._ClipInputs(
        file_id="dddddddd-1", duration_ms=3000,
        dialogue={"topic": [], "sentence": []},
        perception={"content_units": [
            {"unit_id": "u1", "kind": "action", "label": "x", "start_ms": 0, "end_ms": 500},
        ]},
        motion=None,
    )
    assert hc._action_candidates(clip, energy=0.5, field=None) == []
    print("ok  test_action_skipped_without_motion")


def main():
    test_speech_drops_offcamera_and_short()
    test_energy_selects_granularity()
    test_take_stacking_collapses_repeats()
    test_action_snaps_to_calm_motion_seam()
    test_action_fused_avoids_speech()
    test_action_skipped_without_motion()
    print("\nall hero-cuts tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
