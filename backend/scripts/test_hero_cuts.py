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
from app.services.l3.thought_segments import Span, Thought  # noqa: E402


def _action_beats(clip, params, field):
    """Action/visual/insert hero cuts: gather anchors, keep the sync+overlay
    affordances, run the beat engine. (Mirrors the production path in
    ``_file_heroes``; a test-only convenience.)"""
    anchors = hc.anc.gather_anchors(
        duration_ms=clip.duration_ms,
        perception=clip.perception, motion=clip.motion)
    action = [a for a in anchors
              if a.affordance in (hc.anc.AFF_ACTION, hc.anc.AFF_BROLL, hc.anc.AFF_INSERT)]
    return hc._beat_segments(clip, field, params, action)


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
    """Granularity zooms through the hierarchy: rising energy never yields FEWER
    speech heroes. Broad merges a same-speaker run into one block (gaps under the
    'huge' threshold); sentence level splits them all apart."""
    # Three same-speaker sentences spanning sub-'huge' gaps (0.2s, 3.0s).
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
    assert counts[0] == 1, counts   # broad: one same-speaker block (gaps < huge)
    assert counts[-1] == 3, counts  # sentence level: all three distinct
    print("ok  test_clustering_gradient")


def test_sharp_breath_removal_edit_list():
    """Sharp band excises an internal breath into a jump-cut edit-list: same
    sentence, dead air deleted. Tight keeps it contiguous (no keep_spans)."""
    clip = hc._ClipInputs(
        file_id="hhhhhhhh-1", duration_ms=6000,
        dialogue={"topic": [], "sentence": [
            _seg("s0", "sentence", "the product really changes everything today", 0, 3500),
        ]},
        perception=None, motion=None,
    )
    # One long internal breath (900ms) between 'really' and 'changes'.
    words = _words([
        ("the", 0, 300), ("product", 300, 800), ("really", 800, 1300),
        ("changes", 2200, 2700), ("everything", 2700, 3200), ("today", 3200, 3500),
    ])
    src = _src("hhhhhhhh-1", 6000, words)
    tight = hc._speech_candidates(clip, src, None, hc.energy_to_params(0.7))
    sharp = hc._speech_candidates(clip, src, None, hc.energy_to_params(1.0))
    assert len(tight) == 1 and tight[0].keep_spans is None, tight   # with breath
    assert len(sharp) == 1, sharp
    ks = sharp[0].keep_spans
    assert ks is not None and len(ks) == 2, ks                      # breath excised
    assert ks[0][1] <= 1300 and ks[1][0] >= 2200, ks               # cut spans the gap
    assert sharp[0].play_ms() < sharp[0].src_out_ms - sharp[0].src_in_ms
    d = sharp[0].to_dict()
    assert d["play_ms"] == sharp[0].play_ms() and d["play_ms"] < d["duration_ms"]
    assert len(d["keep_spans"]) == 2 and "in_ms" in d["keep_spans"][0]
    print("ok  test_sharp_breath_removal_edit_list")


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


def test_best_take_prefers_on_camera_then_delivery():
    """Best-take is deterministic: the take where the speaker is ON CAMERA wins
    over a higher-SCORED but off-camera take; delivery breaks ties."""
    from app.services.l3 import takes as tk

    # b is lower-scored but on camera with cleaner delivery -> should lead.
    a = hc.HeroCut("h1", "fff", "speech", "the product changes everything", 1000, 2000,
                   score=0.9, quality={"on_camera": 0.0, "delivery": 0.5})
    b = hc.HeroCut("h2", "fff", "speech", "the product changes everything", 5000, 6000,
                   score=0.6, quality={"on_camera": 1.0, "delivery": 0.85})
    group = tk.TakeGroup(group_id="tg1", content_key="product changes everything", attempts=[
        tk.Attempt("fff:u1:0", "fff", "u1", 1000, 2000, "speech", "product changes everything", "...", False),
        tk.Attempt("fff:u2:0", "fff", "u2", 5000, 6000, "speech", "product changes everything", "...", False),
    ])
    orig = hc.build_take_groups
    hc.build_take_groups = lambda file_ids: [group]
    try:
        stacked = hc._stack_takes([a, b], ["fff"])
    finally:
        hc.build_take_groups = orig
    assert len(stacked) == 1, stacked
    front = stacked[0]
    assert front.hero_id == "h2", "on-camera take must lead despite lower score"
    assert front.take_count == 2
    # _take_rank: on-camera beats off-camera regardless of score.
    assert hc._take_rank(b) > hc._take_rank(a)
    print("ok  test_best_take_prefers_on_camera_then_delivery")


def _span(in_ms, out_ms, text, si, sj):
    return Span(raw_in_ms=in_ms, raw_out_ms=out_ms, text=text, start_word=si, end_word=sj)


def test_thought_bands_select_hierarchy():
    """The five energy bands cut the SAME thought at different zoom levels:
    turn/setup -> the run-up + idea, balanced -> the idea, tight -> the core
    sentence, sharp -> the punchline clause (with matching in/out + text)."""
    words = _words([
        ("so", 0, 400), ("anyway", 400, 900),
        ("we", 1000, 1300), ("almost", 1300, 1700), ("shut", 1700, 2100),
        ("down", 2100, 2500), ("last", 2500, 2900), ("year", 2900, 3300),
    ])
    th = Thought(
        speaker="S0",
        thought=_span(1000, 3300, "we almost shut down last year", 2, 7),
        core=_span(1000, 2500, "we almost shut down", 2, 5),
        punch=_span(1300, 2500, "almost shut down", 3, 5),
        setup=_span(0, 900, "so anyway", 0, 1),
        strength=0.8,
    )
    clip = hc._ClipInputs(
        file_id="tttttttt-1", duration_ms=5000,
        dialogue={"topic": [], "sentence": []},
        perception=None, motion=None, thoughts=[th])
    src = _src("tttttttt-1", 5000, words)

    def one(energy):
        hs = hc._speech_candidates(clip, src, None, hc.energy_to_params(energy))
        assert len(hs) == 1, (energy, [h.label for h in hs])
        return hs[0]

    balanced = one(0.5)
    assert (balanced.src_in_ms, balanced.src_out_ms) == (1000, 3300), balanced
    assert balanced.label == "we almost shut down last year"

    tight = one(0.7)
    assert (tight.src_in_ms, tight.src_out_ms) == (1000, 2500), tight
    assert tight.label == "we almost shut down"

    sharp = one(1.0)
    assert (sharp.src_in_ms, sharp.src_out_ms) == (1300, 2500), sharp
    assert sharp.label == "almost shut down"

    calm = one(0.3)
    assert calm.src_in_ms == 0 and calm.src_out_ms == 3300, calm   # setup + thought
    assert calm.label.startswith("so anyway we almost"), calm.label
    print("ok  test_thought_bands_select_hierarchy")


def test_thought_turn_merge_at_broad():
    """Broad merges consecutive same-speaker thoughts into one turn; Balanced
    keeps them as separate thought cuts."""
    words = _words([
        ("so", 0, 400), ("anyway", 400, 900),
        ("we", 1000, 1300), ("almost", 1300, 1700), ("shut", 1700, 2100), ("down", 2100, 2500),
        ("one", 2600, 2900), ("customer", 2900, 3300), ("changed", 3300, 3700), ("everything", 3700, 4100),
    ])
    t1 = Thought(
        speaker="S0",
        thought=_span(1000, 2500, "we almost shut down", 2, 5),
        core=_span(1000, 2500, "we almost shut down", 2, 5),
        punch=_span(1300, 2500, "almost shut down", 3, 5),
        setup=_span(0, 900, "so anyway", 0, 1), strength=0.8)
    t2 = Thought(
        speaker="S0",
        thought=_span(2600, 4100, "one customer changed everything", 6, 9),
        core=_span(2600, 4100, "one customer changed everything", 6, 9),
        punch=_span(3300, 4100, "changed everything", 8, 9), setup=None, strength=0.7)
    clip = hc._ClipInputs(
        file_id="uuuuuuuu-1", duration_ms=6000,
        dialogue={"topic": [], "sentence": []},
        perception=None, motion=None, thoughts=[t1, t2])
    src = _src("uuuuuuuu-1", 6000, words)

    broad = hc._speech_candidates(clip, src, None, hc.energy_to_params(0.0))
    assert len(broad) == 1, [h.label for h in broad]          # one merged turn
    assert (broad[0].src_in_ms, broad[0].src_out_ms) == (0, 4100), broad[0]
    balanced = hc._speech_candidates(clip, src, None, hc.energy_to_params(0.5))
    assert len(balanced) == 2, [h.label for h in balanced]    # two thoughts
    print("ok  test_thought_turn_merge_at_broad")


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
    heroes = _action_beats(clip, hc.energy_to_params(0.0), None)
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
    heroes = _action_beats(clip, hc.energy_to_params(0.5), field)
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
        # Overlay moments live in the sparse cutaways track (the single source the
        # anchor layer reads): a listener reaction + a held b-roll shot.
        "cutaways": [
            {"start_ms": 8200, "end_ms": 9100, "kind": "reaction", "affordance": "reaction",
             "subject": "p1", "label": "smile", "trigger": "the miss", "intensity": 0.7},
            {"start_ms": 9000, "end_ms": 11500, "kind": "broll_hold", "affordance": "broll",
             "label": "wide room"},
        ],
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


def test_action_overlay_cuts_own_ladder_and_framing():
    """Action + overlay cuts now carry an owned broad..sharp ladder, the actor
    in `people`, and camera framing from camera_craft."""
    clip = _clip_multi()
    clip.perception["camera_craft"] = [
        {"start_ms": 0, "end_ms": 12000, "shot_size": "wide", "angle": "eye_level",
         "movement": "static", "subject_focus": "the room"},
    ]
    field = hc._build_field(clip, 0.5)
    anchors = hc.anc.gather_anchors(duration_ms=clip.duration_ms, dialogue=clip.dialogue,
                                    perception=clip.perception, motion=clip.motion)
    beats = hc._beat_segments(clip, field, hc.energy_to_params(0.5), anchors)
    act = next(b for b in beats if b.modality == hc.anc.AFF_ACTION)
    assert [r.level for r in act.ladder] == ["broad", "calm", "balanced", "tight", "sharp"], act.ladder
    bal = next(r for r in act.ladder if r.level == "balanced")
    assert (bal.in_ms(), bal.out_ms()) == (act.src_in_ms, act.src_out_ms)
    assert act.framing and act.framing["shot_size"] == "wide"
    # Round-trips through the cache with facets intact.
    back = hc.HeroCut.from_cache(act.to_dict())
    assert len(back.ladder) == 5 and back.framing["movement"] == "static"
    print("ok  test_action_overlay_cuts_own_ladder_and_framing")


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
    heroes = _action_beats(clip, params, field)
    acts = [h for h in heroes if h.modality == hc.anc.AFF_ACTION]
    assert len(acts) == 2, [(h.label, h.src_in_ms, h.src_out_ms) for h in acts]
    assert min(h.src_in_ms for h in acts) <= 6000
    assert max(h.src_out_ms for h in acts) >= 7800
    assert any("windup" in h.label for h in acts)
    assert any("payoff" in h.label for h in acts)
    print("ok  test_action_split_at_sharp")


def test_action_core_caps_and_performance_exempt():
    """Sharp band: a long action beat is core-capped impact-forward (negative
    padding), while a performance keeps its full duration (never trimmed)."""
    params = hc.energy_to_params(1.0)            # Sharp: split on, core 1800
    motion = {"hop_ms": 100, "action_energy": [0.7] * 200}
    act = hc.anc.Anchor(ts_ms=5000, start_ms=2000, end_ms=12000,
                        kind="action_beat", affordance=hc.anc.AFF_ACTION, salience=0.8)
    pieces = hc._action_pieces(act, motion, params, None, None)
    payoff = [p for p in pieces if "payoff" in p[2]]
    assert payoff, pieces
    pin, pout, _ = payoff[0]
    assert pout - pin <= params.action_core_ms + 5, (pin, pout)
    perf = hc.anc.Anchor(ts_ms=15000, start_ms=13000, end_ms=19000,
                        kind="performance", affordance=hc.anc.AFF_ACTION, salience=0.8)
    assert hc._action_pieces(perf, motion, params, None, None) == [(13000, 19000, "")]
    print("ok  test_action_core_caps_and_performance_exempt")


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
    assert _action_beats(clip, hc.energy_to_params(0.5), None) == []
    print("ok  test_action_skipped_without_motion")


def _wspk(spec):
    """spec: list of (text, start_ms, end_ms, speaker) -> diarized word dicts."""
    return [{"text": t, "start_ms": s, "end_ms": e, "speaker": spk, "is_filler": False}
            for t, s, e, spk in spec]


def _one_thought_clip(file_id, perception):
    from app.services.l3 import cast as cst
    words = _wspk([
        ("we", 1000, 1300, "S0"), ("almost", 1300, 1700, "S0"), ("shut", 1700, 2100, "S0"),
        ("down", 2100, 2500, "S0"), ("last", 2500, 2900, "S0"), ("year", 2900, 3300, "S0"),
    ])
    th = Thought(
        speaker="S0",
        thought=_span(1000, 3300, "we almost shut down last year", 0, 5),
        core=_span(1000, 2500, "we almost shut down", 0, 3),
        punch=_span(1300, 2500, "almost shut down", 1, 3),
        setup=None, strength=0.8)
    clip = hc._ClipInputs(
        file_id=file_id, duration_ms=5000,
        dialogue={"topic": [], "sentence": []},
        perception=perception, motion=None, thoughts=[th])
    clip.cast = cst.build_cast(perception, words)
    return clip, _src(file_id, 5000, words)


def test_speech_cut_owns_ladder_and_resolves_speaker():
    """A speech cut carries its full broad..sharp ladder, the selected rung
    matches the flat span, and the speaker is the cast-resolved person."""
    perception = {
        "persons": [{"local_id": "p1", "role": "main subject",
                     "frame_region": {"x": 0.3, "y": 0.1, "w": 0.4, "h": 0.6}}],
        "speaking": [{"subject": "p1", "start_ms": 1000, "end_ms": 3300}],
        "take_quality_events": [],
    }
    clip, src = _one_thought_clip("llllllll-1", perception)
    hs = hc._speech_candidates(clip, src, None, hc.energy_to_params(0.5))
    assert len(hs) == 1, hs
    h = hs[0]
    levels = [r.level for r in h.ladder]
    assert levels == ["broad", "calm", "balanced", "tight", "sharp"], levels
    # Selected (balanced) rung agrees with the flat span that plays.
    bal = next(r for r in h.ladder if r.level == "balanced")
    assert (bal.in_ms(), bal.out_ms()) == (h.src_in_ms, h.src_out_ms) == (1000, 3300)
    # Speaker resolved to the on-screen person; people + framing + quality set.
    assert h.speaker == "main subject", h.speaker
    assert h.people and h.people[0]["person_id"] == "p1" and h.people[0]["on_camera"] is True
    assert h.framing and h.framing["region"]["w"] == 0.4
    assert h.quality and "delivery" in h.quality and h.quality["on_camera"] == 1.0
    # Round-trips through the cache.
    back = hc.HeroCut.from_cache(h.to_dict())
    assert [r.level for r in back.ladder] == levels
    print("ok  test_speech_cut_owns_ladder_and_resolves_speaker")


def test_offcamera_speech_flagged_not_dropped():
    """Off-frame voice (audio with no visible speaker) is KEPT and flagged
    'offscreen', never discarded -- the off-camera interviewer survives."""
    from app.services.l3 import cast as cst
    # p1 is on camera early; the question at 5s is a different, off-screen voice.
    words = _wspk([
        ("so", 1000, 1300, "S0"), ("here", 1300, 1700, "S0"), ("we", 1700, 2100, "S0"),
        ("are", 2100, 2500, "S0"),
        ("what", 5000, 5300, "S9"), ("made", 5300, 5700, "S9"), ("you", 5700, 6000, "S9"),
        ("start", 6000, 6400, "S9"), ("this", 6400, 6800, "S9"),
    ])
    th_on = Thought(speaker="S0",
                    thought=_span(1000, 2500, "so here we are", 0, 3),
                    core=_span(1000, 2500, "so here we are", 0, 3),
                    punch=_span(1300, 2500, "here we are", 1, 3), setup=None, strength=0.6)
    th_off = Thought(speaker="S9",
                     thought=_span(5000, 6800, "what made you start this", 4, 8),
                     core=_span(5000, 6800, "what made you start this", 4, 8),
                     punch=_span(5300, 6800, "made you start this", 5, 8), setup=None, strength=0.7)
    perception = {
        "persons": [{"local_id": "p1"}],
        "speaking": [{"subject": "p1", "start_ms": 1000, "end_ms": 2500}],
        "take_quality_events": [],
    }
    clip = hc._ClipInputs(file_id="oooooooo-1", duration_ms=8000,
                          dialogue={"topic": [], "sentence": []},
                          perception=perception, motion=None, thoughts=[th_on, th_off])
    clip.cast = cst.build_cast(perception, words)
    hs = hc._speech_candidates(clip, _src("oooooooo-1", 8000, words), None,
                               hc.energy_to_params(0.5))
    # Both survive (never discarded); the off-screen question is flagged.
    labels = {h.label: h for h in hs}
    assert any("here we are" in L for L in labels), labels
    off = next((h for L, h in labels.items() if "made you start" in L), None)
    assert off is not None, "off-camera question must not be dropped"
    assert "offscreen" in off.flags, off.flags
    print("ok  test_offcamera_speech_flagged_not_dropped")


def test_relation_links_reaction_to_line():
    """The VLM's typed `responds_to` edge (a listener reaction -> the line it
    answers) is mapped onto the two cuts by time: both stay first-class cards, get
    the directional edge, and share a moment cluster."""
    s = hc.HeroCut("z:sp0", "zzzz", "speech", "that was incredible", 1000, 3000, score=0.7,
                   speaker="host", affordances=["speech"],
                   people=[{"person_id": "p1", "on_camera": True}])
    r = hc.HeroCut("z:rea0", "zzzz", "reaction", "p2 grins", 2500, 3500, score=0.5,
                   affordances=["reaction"], people=[{"person_id": "p2", "on_camera": True}])
    perception = {
        "content_units": [{"unit_id": "u1", "start_ms": 1000, "end_ms": 3000, "kind": "speech"}],
        "reactions": [{"id": "rx1", "start_ms": 2500, "end_ms": 3500, "subject": "p2"}],
        "relations": [{"type": "responds_to", "from_id": "rx1", "to_id": "u1",
                       "note": "grins at the line"}],
    }
    clip = hc._ClipInputs(file_id="zzzzzzzz-1", duration_ms=40000,
                          dialogue={"topic": [], "sentence": []}, perception=perception, motion=None)
    out = hc._annotate_moments(clip, [s, r])
    assert len(out) == 2, [(h.hero_id, h.modality) for h in out]    # nothing dropped
    assert s.moment_id is not None and s.moment_id == r.moment_id    # one cluster
    # directional edge recorded on both ends
    assert any(e["type"] == "responds_to" and e["dir"] == "out" and e["other"] == "z:sp0"
               for e in r.relations), r.relations
    assert any(e["type"] == "responds_to" and e["dir"] == "in" and e["other"] == "z:rea0"
               for e in s.relations), s.relations
    assert s.is_moment() and r.is_moment()
    print("ok  test_relation_links_reaction_to_line")


def test_relation_take_of_does_not_form_moment():
    """A `take_of` edge (two deliveries of the same line) is recorded as an edge
    but does NOT bundle the cuts into a moment -- alternates are one slot."""
    a = hc.HeroCut("z:sp0", "zzzz", "speech", "take one", 1000, 3000, score=0.7,
                   speaker="host", affordances=["speech"])
    b = hc.HeroCut("z:sp1", "zzzz", "speech", "take two", 9000, 11000, score=0.6,
                   speaker="host", affordances=["speech"])
    perception = {
        "content_units": [
            {"unit_id": "u1", "start_ms": 1000, "end_ms": 3000, "kind": "speech"},
            {"unit_id": "u2", "start_ms": 9000, "end_ms": 11000, "kind": "speech"},
        ],
        "relations": [{"type": "take_of", "from_id": "u1", "to_id": "u2"}],
    }
    clip = hc._ClipInputs(file_id="zzzzzzzz-4", duration_ms=20000,
                          dialogue={"topic": [], "sentence": []}, perception=perception, motion=None)
    out = hc._annotate_moments(clip, [a, b])
    assert len(out) == 2
    assert a.relations and a.relations[0]["type"] == "take_of"     # edge recorded
    assert a.moment_id is None and b.moment_id is None             # but no cluster
    print("ok  test_relation_take_of_does_not_form_moment")


def test_roles_assigned_from_l2_and_listening_flag():
    """Each cut takes its narrative role from the best-overlapping VLM role; a
    synthesized listening shot (no L2 row) gets 'listener' from its flag."""
    hook = hc.HeroCut("z:sp0", "zzzz", "speech", "the wild opener", 1000, 3000,
                      score=0.7, speaker="host", affordances=["speech"])
    mid = hc.HeroCut("z:sp1", "zzzz", "speech", "ordinary middle", 5000, 7000,
                     score=0.6, speaker="host", affordances=["speech"])
    listen = hc.HeroCut("z:rea0", "zzzz", "reaction", "p2 listens", 5200, 6800,
                        score=0.5, affordances=["reaction"], flags=["listening"])
    perception = {"content_units": [
        {"unit_id": "u1", "start_ms": 1000, "end_ms": 3000, "kind": "speech", "role": "hook"},
        {"unit_id": "u2", "start_ms": 5000, "end_ms": 7000, "kind": "speech"},  # no role
    ]}
    clip = hc._ClipInputs(file_id="zzzzzzzz-7", duration_ms=10000,
                          dialogue={"topic": [], "sentence": []}, perception=perception, motion=None)
    hc._assign_roles(clip, [hook, mid, listen])
    assert hook.role == "hook", hook.role
    assert mid.role is None                       # ordinary middle stays unmarked
    assert listen.role == "listener", listen.role  # from the listening flag
    print("ok  test_roles_assigned_from_l2_and_listening_flag")


def test_no_relations_no_moments():
    """With no relation graph (old cache / independent lines) nothing is a moment
    -- the flat default. A podcast of plain lines yields zero clusters."""
    a = hc.HeroCut("z:sp0", "zzzz", "speech", "line a", 1000, 3000, score=0.7, speaker="host")
    b = hc.HeroCut("z:sp1", "zzzz", "speech", "line b", 9000, 11000, score=0.6, speaker="host")
    clip = hc._ClipInputs(file_id="zzzzzzzz-5", duration_ms=20000,
                          dialogue={"topic": [], "sentence": []}, perception={}, motion=None)
    out = hc._annotate_moments(clip, [a, b])
    assert len(out) == 2 and a.moment_id is None and b.moment_id is None
    assert not a.is_moment()
    print("ok  test_no_relations_no_moments")


def test_relation_illustrates_chains_into_one_moment():
    """A line, its reaction, and the b-roll that illustrates it chain (via
    moment-forming edges) into ONE connected cluster -- a rich reel moment."""
    sp = hc.HeroCut("z:sp0", "zzzz", "speech", "look at this view", 1000, 3000, score=0.7,
                    speaker="host", affordances=["speech"])
    rea = hc.HeroCut("z:rea0", "zzzz", "reaction", "p2 gasps", 2500, 3500, score=0.5,
                     affordances=["reaction"])
    bro = hc.HeroCut("z:bro0", "zzzz", "broll", "the mountain", 1200, 2800, score=0.5,
                     affordances=["broll"])
    perception = {
        "content_units": [{"unit_id": "u1", "start_ms": 1000, "end_ms": 3000, "kind": "speech"}],
        "reactions": [{"id": "rx1", "start_ms": 2500, "end_ms": 3500, "subject": "p2"}],
        "cutaways": [{"id": "cx1", "start_ms": 1200, "end_ms": 2800, "kind": "broll_hold",
                      "affordance": "broll", "label": "mountain"}],
        "relations": [
            {"type": "responds_to", "from_id": "rx1", "to_id": "u1"},
            {"type": "illustrates", "from_id": "cx1", "to_id": "u1"},
        ],
    }
    clip = hc._ClipInputs(file_id="zzzzzzzz-6", duration_ms=10000,
                          dialogue={"topic": [], "sentence": []}, perception=perception, motion=None)
    out = hc._annotate_moments(clip, [sp, rea, bro])
    assert len(out) == 3
    mids = {sp.moment_id, rea.moment_id, bro.moment_id}
    assert len(mids) == 1 and None not in mids, (sp.moment_id, rea.moment_id, bro.moment_id)
    print("ok  test_relation_illustrates_chains_into_one_moment")


def test_behavior_anchors_from_event_timeline():
    """The VLM event timeline (incidental physical business) becomes ACTION
    anchors (kind='behavior') -- the thing that was previously dropped. There is
    no separate 'behavior' affordance: a coffee sip IS action."""
    perception = {"events": [
        {"id": "e1", "start_ms": 1000, "end_ms": 2500, "actor": "p1",
         "description": "p1 sips coffee"},
        {"id": "e2", "start_ms": 9000, "end_ms": 9200, "actor": "p1",
         "description": "p1 blinks"},   # too short -> gated out
    ]}
    anchors = hc.anc.gather_anchors(duration_ms=12000, perception=perception, motion=None)
    beh = [a for a in anchors if a.kind == "behavior"]
    assert len(beh) == 1, [(a.text, a.start_ms, a.end_ms) for a in beh]
    assert beh[0].affordance == hc.anc.AFF_ACTION       # folded into action
    assert beh[0].actor == "p1" and "coffee" in beh[0].text
    assert (beh[0].start_ms, beh[0].end_ms) == (1000, 2500)
    print("ok  test_behavior_anchors_from_event_timeline")


def test_behavior_continuity_grouping():
    """Consecutive same-actor events with a short inter-gap stitch into ONE
    continuous behavior; a different actor / a long gap starts a new one."""
    perception = {"events": [
        {"id": "e1", "start_ms": 1000, "end_ms": 1800, "actor": "p1", "description": "p1 stands"},
        {"id": "e2", "start_ms": 2200, "end_ms": 3000, "actor": "p1", "description": "p1 walks"},
        {"id": "e3", "start_ms": 3300, "end_ms": 4200, "actor": "p1", "description": "p1 opens door"},
        {"id": "e4", "start_ms": 4400, "end_ms": 5400, "actor": "p2", "description": "p2 waves"},
        {"id": "e5", "start_ms": 20000, "end_ms": 21000, "actor": "p1", "description": "p1 sits"},
    ]}
    anchors = hc.anc.gather_anchors(duration_ms=30000, perception=perception, motion=None)
    beh = sorted((a for a in anchors if a.kind == "behavior"),
                 key=lambda a: a.start_ms)
    # p1's three adjacent beats -> one span (1000-4200); p2 separate; p1 late separate.
    assert len(beh) == 3, [(a.actor, a.start_ms, a.end_ms, a.text) for a in beh]
    assert (beh[0].actor, beh[0].start_ms, beh[0].end_ms) == ("p1", 1000, 4200), beh[0]
    assert "\u2192" in beh[0].text     # stitched description
    assert (beh[1].actor, beh[1].start_ms) == ("p2", 4400)
    assert (beh[2].actor, beh[2].start_ms) == ("p1", 20000)
    print("ok  test_behavior_continuity_grouping")


def test_behavior_surfaces_as_action_cut():
    """A behavior anchor flows through the beat engine into an ACTION hero cut
    (the closed vocabulary -- behavior is not its own bucket)."""
    clip = _clip_multi()
    clip.perception["events"] = [
        {"id": "e1", "start_ms": 4000, "end_ms": 5500, "actor": "p1",
         "description": "p1 gestures emphatically"},
    ]
    field = hc._build_field(clip, 0.5)
    anchors = hc.anc.gather_anchors(duration_ms=clip.duration_ms, dialogue=clip.dialogue,
                                    perception=clip.perception, motion=clip.motion)
    beats = hc._beat_segments(clip, field, hc.energy_to_params(0.5), anchors)
    gestures = [b for b in beats if b.modality == hc.anc.AFF_ACTION and "gestures" in b.label]
    assert gestures, [(b.modality, b.label) for b in beats]
    assert gestures[0].affordances == [hc.anc.AFF_ACTION]
    print("ok  test_behavior_surfaces_as_action_cut")


def test_listening_anchor_from_speaking_inverse():
    """A sustained turn by one speaker synthesizes a held LISTENING reaction for
    the other on-camera person; a short turn earns none."""
    perception = {
        "persons": [
            {"local_id": "p1", "frame_region": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.5}},
            {"local_id": "p2", "frame_region": {"x": 0.6, "y": 0.1, "w": 0.3, "h": 0.5}},
        ],
        "speaking": [
            {"subject": "p1", "start_ms": 1000, "end_ms": 11000},   # long turn
            {"subject": "p2", "start_ms": 11500, "end_ms": 12000},  # short
        ],
    }
    anchors = hc.anc.gather_anchors(duration_ms=13000, perception=perception, motion=None)
    listen = [a for a in anchors if "listening" in a.flags]
    assert len(listen) == 1, [(a.actor, a.start_ms, a.end_ms) for a in listen]
    assert listen[0].actor == "p2" and listen[0].affordance == hc.anc.AFF_REACTION
    assert listen[0].salience >= 0.9, listen[0].salience      # 10s turn -> full warrant
    assert listen[0].region["w"] == 0.3
    assert listen[0].end_ms - listen[0].start_ms <= hc.anc.LISTEN_MAX_MS
    print("ok  test_listening_anchor_from_speaking_inverse")


def test_listening_deduped_against_logged_reaction():
    """When the VLM already logged a reaction for the SAME listener over the SAME
    stretch, the synthesized listening shot is suppressed (its explicit one wins)."""
    perception = {
        "persons": [
            {"local_id": "p1", "frame_region": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.5}},
            {"local_id": "p2", "frame_region": {"x": 0.6, "y": 0.1, "w": 0.3, "h": 0.5}},
        ],
        "speaking": [{"subject": "p1", "start_ms": 1000, "end_ms": 11000}],
        "cutaways": [
            {"start_ms": 3000, "end_ms": 9000, "kind": "reaction", "affordance": "reaction",
             "subject": "p2", "label": "nods along", "intensity": 0.6},
        ],
    }
    anchors = hc.anc.gather_anchors(duration_ms=13000, perception=perception, motion=None)
    listen = [a for a in anchors if "listening" in a.flags]
    assert listen == [], [(a.actor, a.start_ms, a.end_ms) for a in listen]
    # the explicit VLM reaction survives
    assert any(a.affordance == hc.anc.AFF_REACTION and a.actor == "p2" for a in anchors)
    print("ok  test_listening_deduped_against_logged_reaction")


def test_facet_record_round_trips():
    """The Cut facet record (ladder rungs + people/framing/quality) survives the
    cache round-trip, and a rung's keep_spans/play_ms reflect a split."""
    ladder = [
        hc.Rung(level="balanced", spans=[(1000, 3300)], text="the whole thought", score=0.7),
        hc.Rung(level="sharp", spans=[(1300, 1800), (2200, 2500)], text="punch", score=0.8),
    ]
    h = hc.HeroCut(
        "z:sp0", "zzzz", "speech", "the whole thought", 1000, 3300, score=0.7,
        speaker="interviewer", affordances=["speech"], ladder=ladder,
        people=[{"voice_speaker_id": "S0", "person_id": "p1", "role": "interviewer",
                 "on_camera": True, "region": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.5},
                 "av_link_confidence": 1.0}],
        framing={"shot_size": "medium", "angle": "eye_level", "movement": "static"},
        quality={"delivery": 0.82, "vlm": 0.6},
    )
    # Rung helpers
    sharp = ladder[1]
    assert sharp.in_ms() == 1300 and sharp.out_ms() == 2500
    assert sharp.play_ms() == 800 and sharp.keep_spans() == [(1300, 1800), (2200, 2500)]
    assert ladder[0].keep_spans() is None        # single span plays contiguously

    back = hc.HeroCut.from_cache(h.to_dict())
    assert len(back.ladder) == 2 and back.ladder[1].level == "sharp"
    assert back.ladder[1].spans == [(1300, 1800), (2200, 2500)]
    assert back.people[0]["person_id"] == "p1" and back.people[0]["on_camera"] is True
    assert back.framing["shot_size"] == "medium"
    assert back.quality["delivery"] == 0.82
    # A cut with no facets emits null (compact) and rehydrates empty.
    plain = hc.HeroCut("z:sp1", "zzzz", "speech", "x", 0, 100, score=0.1)
    d = plain.to_dict()
    assert d["ladder"] is None and d["people"] is None
    assert hc.HeroCut.from_cache(d).ladder == []
    print("ok  test_facet_record_round_trips")


def main():
    test_behavior_anchors_from_event_timeline()
    test_behavior_continuity_grouping()
    test_behavior_surfaces_as_action_cut()
    test_listening_anchor_from_speaking_inverse()
    test_listening_deduped_against_logged_reaction()
    test_facet_record_round_trips()
    test_relation_links_reaction_to_line()
    test_relation_take_of_does_not_form_moment()
    test_roles_assigned_from_l2_and_listening_flag()
    test_no_relations_no_moments()
    test_relation_illustrates_chains_into_one_moment()
    test_speech_cut_owns_ladder_and_resolves_speaker()
    test_offcamera_speech_flagged_not_dropped()
    test_speech_drops_offcamera_and_short()
    test_energy_selects_granularity()
    test_clustering_gradient()
    test_thought_bands_select_hierarchy()
    test_thought_turn_merge_at_broad()
    test_sharp_breath_removal_edit_list()
    test_take_stacking_collapses_repeats()
    test_best_take_prefers_on_camera_then_delivery()
    test_action_snaps_to_calm_motion_seam()
    test_action_fused_avoids_speech()
    test_action_skipped_without_motion()
    test_reaction_and_broll_surface_as_cutaways()
    test_action_overlay_cuts_own_ladder_and_framing()
    test_action_core_preserved()
    test_action_split_at_sharp()
    test_action_core_caps_and_performance_exempt()
    test_coverage_every_anchor_in_a_segment()
    print("\nall hero-cuts tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
