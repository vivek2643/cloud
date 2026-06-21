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
    heroes = hc._speech_candidates(clip, _src("aaaaaaaa-1", 8000, words),
                                   None, hc.energy_to_params(0.0))
    assert len(heroes) == 1, [h.label for h in heroes]
    assert heroes[0].label == "this is a real usable line"
    assert heroes[0].modality == "speech"
    print("ok  test_speech_drops_offcamera_and_short")


def test_energy_selects_granularity():
    """Low energy clusters adjacent sentences into one answer; high energy keeps
    them as separate sentence heroes."""
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
    low = hc._speech_candidates(clip, src, None, hc.energy_to_params(0.0))
    high = hc._speech_candidates(clip, src, None, hc.energy_to_params(1.0))
    assert len(low) == 1, low                 # merged into one answer
    assert len(high) == 2, high               # kept as two sentences
    print("ok  test_energy_selects_granularity")


def test_clustering_gradient():
    """Granularity is monotonic in energy: rising energy never yields FEWER
    speech heroes (answers -> sentences -> clauses), and a long pause is never
    merged across even at low energy."""
    # Three sentences: s0,s1 close together (0.2s gap), s2 after a 3s pause.
    clip = hc._ClipInputs(
        file_id="gggggggg-1", duration_ms=12000,
        dialogue={"topic": [], "sentence": [
            _seg("s0", "sentence", "first part of the idea", 0, 1500),
            _seg("s1", "sentence", "second part of the idea", 1700, 3200),
            _seg("s2", "sentence", "a separate later point entirely", 6200, 7800),
        ]},
        perception=None, motion=None,
    )
    words = _words([
        ("first", 0, 400), ("part", 400, 800), ("of", 800, 1000),
        ("the", 1000, 1200), ("idea", 1200, 1500),
        ("second", 1700, 2100), ("part", 2100, 2500), ("of", 2500, 2700),
        ("the", 2700, 2900), ("idea", 2900, 3200),
        ("a", 6200, 6400), ("separate", 6400, 6900), ("later", 6900, 7300),
        ("point", 7300, 7600), ("entirely", 7600, 7800),
    ])
    src = _src("gggggggg-1", 12000, words)
    counts = [len(hc._speech_candidates(clip, src, None, hc.energy_to_params(e)))
              for e in (0.0, 0.5, 0.8)]
    assert counts == sorted(counts), counts        # non-decreasing with energy
    assert counts[0] == 2, counts   # low energy: s0+s1 merge, s2 stays apart (3s pause)
    assert counts[-1] == 3, counts  # sentence level: all three distinct
    print("ok  test_clustering_gradient")


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
    heroes = hc._action_candidates(clip, hc.energy_to_params(0.0), None)
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
    field = hc._build_field(clip, 0.5)
    assert field is not None
    heroes = hc._action_candidates(clip, hc.energy_to_params(0.5), field)
    assert len(heroes) == 1, heroes
    h = heroes[0]
    # Core-preservation: the boundary never lands INSIDE the spoken word...
    assert not (1800 < h.src_out_ms < 2600), h.src_out_ms
    # ...and the action core (ends at 2000) is never clipped -- the out lands at a
    # clean seam at/after the core (here, just after the speech ends ~2600).
    assert h.src_out_ms >= 2000, h.src_out_ms
    print("ok  test_action_fused_avoids_speech")


def _clip_multi():
    """A clip with speech + a reaction + a held shot + an action beat + audio and
    motion grids, so a real fused field can be built."""
    n = 120  # 12s at 100ms
    dlg = [0.0] * n
    for i in range(10, 30):     # speech 1.0-3.0s
        dlg[i] = 1.0
    cam = [0.05] * n
    for i in range(60, 78):     # busy motion during the action 6.0-7.8s
        cam[i] = 0.8
    action_cost = [1.0] * n
    action_cost[68] = 0.0       # impact ~6.8s
    motion = {"hop_ms": 100, "action_energy": [0.7] * n,
              "action_cut_cost": action_cost, "camera_cut_cost": cam,
              "action_points": [{"ts_ms": 6800, "score": 1.0}]}
    audio = {"dialogue_cut_cost": dlg, "dialogue_cut_hop_ms": 100, "dialogue_cut_points": [],
             "beat_cut_cost": [], "beat_cut_hop_ms": 100, "beat_cut_points": []}
    perception = {
        "content_type": "vlog",
        "editability": {"primary_axis": "action"},
        "content_units": [{"unit_id": "u1", "kind": "action", "label": "drops and misses",
                           "start_ms": 6000, "end_ms": 7800}],
        "reactions": [{"start_ms": 8200, "end_ms": 9100, "subject": "p1",
                       "type": "smile", "intensity": 0.7, "trigger": "the miss"}],
        "camera_craft": [{"start_ms": 9000, "end_ms": 11500, "movement": "static",
                          "subject_focus": "wide room"}],
        "take_quality_events": [],
    }
    clip = hc._ClipInputs(
        file_id="ffffffff-1", duration_ms=12000,
        dialogue={"topic": [], "sentence": [
            _seg("s0", "sentence", "here is a clean on camera spoken line", 1000, 3000)]},
        perception=perception, motion=motion, audio=audio)
    return clip


def test_reaction_and_broll_surface_as_cutaways():
    """The whole cutaway vocabulary (reactions, held b-roll) now appears in the
    feed alongside action -- the thing that was completely invisible before."""
    clip = _clip_multi()
    field = hc._build_field(clip, 0.5)
    anchors = hc.anc.gather_anchors(duration_ms=clip.duration_ms, dialogue=clip.dialogue,
                                    perception=clip.perception, motion=clip.motion)
    beats = hc._beat_segments(clip, field, hc.energy_to_params(0.5), anchors)
    mods = {b.modality for b in beats}
    assert hc.anc.AFF_REACTION in mods and hc.anc.AFF_BROLL in mods and hc.anc.AFF_ACTION in mods, mods
    print("ok  test_reaction_and_broll_surface_as_cutaways")


def test_action_core_preserved():
    """Core-preservation: the action segment always contains the whole beat
    (6000-7800) at Broad/Balanced; Sharp may split but payoff still covers the end."""
    clip = _clip_multi()
    for e in (0.0, 0.5):
        field = hc._build_field(clip, e)
        anchors = hc.anc.gather_anchors(duration_ms=clip.duration_ms, dialogue=clip.dialogue,
                                        perception=clip.perception, motion=clip.motion)
        beats = hc._beat_segments(clip, field, hc.energy_to_params(e), anchors)
        acts = [b for b in beats if b.modality == hc.anc.AFF_ACTION]
        assert acts
        assert min(a.src_in_ms for a in acts) <= 6000, (e, acts)
        assert max(a.src_out_ms for a in acts) >= 7800, (e, acts)
    print("ok  test_action_core_preserved")


def test_action_split_at_sharp():
    """Sharp band: editorial split at impact (windup + payoff), even when fused
    seam quality at impact is poor -- outer edges snap, hinge does not."""
    clip = _clip_multi()
    field = hc._build_field(clip, 0.85)
    params = hc.energy_to_params(0.85)
    assert params.action_split_at_impact
    heroes = hc._action_candidates(clip, params, field)
    acts = [h for h in heroes if h.modality == hc.anc.AFF_ACTION]
    assert len(acts) == 2, [(h.label, h.src_in_ms, h.src_out_ms) for h in acts]
    assert min(h.src_in_ms for h in acts) <= 6000
    assert max(h.src_out_ms for h in acts) >= 7800
    assert any("windup" in h.label for h in acts)
    assert any("payoff" in h.label for h in acts)
    print("ok  test_action_split_at_sharp")


def test_coverage_every_anchor_in_a_segment():
    """The coverage guarantee: every (non-trivial) anchor lands inside some
    produced segment -- so there is no usable moment only reachable in raw."""
    clip = _clip_multi()
    field = hc._build_field(clip, 0.5)
    src = _src("ffffffff-1", 12000, _words([
        ("here", 1000, 1300), ("is", 1300, 1500), ("a", 1500, 1600), ("clean", 1600, 2000),
        ("on", 2000, 2200), ("camera", 2200, 2600), ("spoken", 2600, 2900), ("line", 2900, 3000)]))
    params = hc.energy_to_params(0.75)
    anchors = hc.anc.gather_anchors(duration_ms=clip.duration_ms, dialogue=clip.dialogue,
                                    perception=clip.perception, motion=clip.motion)
    segs = hc._speech_candidates(clip, src, field, params) + hc._beat_segments(clip, field, params, anchors)
    for a in anchors:
        covered = any(hc._overlap_ms(s.src_in_ms, s.src_out_ms, a.start_ms, a.end_ms) > 0 for s in segs)
        assert covered, f"anchor {a.kind} {a.start_ms}-{a.end_ms} not covered by any segment"
    print("ok  test_coverage_every_anchor_in_a_segment")


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
    assert hc._action_candidates(clip, hc.energy_to_params(0.5), None) == []
    print("ok  test_action_skipped_without_motion")


def main():
    test_speech_drops_offcamera_and_short()
    test_energy_selects_granularity()
    test_clustering_gradient()
    test_take_stacking_collapses_repeats()
    test_action_snaps_to_calm_motion_seam()
    test_action_fused_avoids_speech()
    test_action_skipped_without_motion()
    test_reaction_and_broll_surface_as_cutaways()
    test_action_core_preserved()
    test_action_split_at_sharp()
    test_coverage_every_anchor_in_a_segment()
    print("\nall hero-cuts tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
