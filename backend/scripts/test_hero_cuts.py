"""
Regression tests for the hero-cuts assembly engine (no DB).

Exercises the pure logic on the channel model (said/done/shown): speech
candidates from the dialogue lens (with off-camera filtering + energy-driven
granularity), the thought-ladder bands, take stacking, and HeroCut
serialization. Run:  .venv/bin/python scripts/test_hero_cuts.py
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


def _wspk(spec):
    """spec: list of (text, start_ms, end_ms, speaker) -> diarized word dicts."""
    return [{"text": t, "start_ms": s, "end_ms": e, "speaker": spk, "is_filler": False}
            for t, s, e, spk in spec]


def _span(in_ms, out_ms, text, si, sj):
    return Span(raw_in_ms=in_ms, raw_out_ms=out_ms, text=text, start_word=si, end_word=sj)


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
    assert heroes[0].channel == "said"
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


def test_dead_air_floor_excises_long_hole_at_every_band():
    """The dead-air FLOOR excises a long internal silence (>= _DEAD_AIR_FLOOR_MS)
    even at BROAD, where the Sharp breath pass is off -- so no speech cut ever
    plays a big hole (e.g. a merged turn spanning the gap between two thoughts).
    Below the floor, a short breath is kept contiguous."""
    clip = hc._ClipInputs(
        file_id="dddddddd-1", duration_ms=8000,
        dialogue={"topic": [], "sentence": [
            _seg("s0", "sentence", "the product really changes everything today", 0, 4300),
        ]},
        perception=None, motion=None,
    )
    # A 2.0s dead hole (>= 1.2s floor) between 'really' and 'changes'.
    words = _words([
        ("the", 0, 300), ("product", 300, 800), ("really", 800, 1300),
        ("changes", 3300, 3800), ("everything", 3800, 4200), ("today", 4200, 4300),
    ])
    src = _src("dddddddd-1", 8000, words)
    broad = hc._speech_candidates(clip, src, None, hc.energy_to_params(0.0))
    assert len(broad) == 1, broad
    ks = broad[0].keep_spans
    assert ks is not None and len(ks) == 2, ks                 # hole excised at Broad
    assert ks[0][1] <= 1300 and ks[1][0] >= 3300, ks           # cut spans the 2s hole
    assert broad[0].play_ms() < broad[0].src_out_ms - broad[0].src_in_ms
    print("ok  test_dead_air_floor_excises_long_hole_at_every_band")


def test_take_stacking_collapses_repeats(monkeypatch=None):
    """A take group (from l3.takes) of two deliveries collapses into ONE hero
    with take_count=2, higher-scoring delivery in front, loser as an alt. An
    unrelated hero stays solo. Stacking maps the group's attempts onto heroes by
    time-overlap, so `build_take_groups` is stubbed to isolate the fold logic."""
    from app.services.l3 import takes as tk

    a = hc.HeroCut("h1", "fff", "said", "the product changes everything", 1000, 2000, score=0.6)
    b = hc.HeroCut("h2", "fff", "said", "the product changes everything", 5000, 6000, score=0.9)
    c = hc.HeroCut("h3", "fff", "said", "a different sentence about pricing", 8000, 9000, score=0.7)

    group = tk.TakeGroup(group_id="tg1", content_key="product changes everything", attempts=[
        tk.Attempt("fff:u1:0", "fff", "u1", 1000, 2000, "said", "product changes everything", "...", False),
        tk.Attempt("fff:u2:0", "fff", "u2", 5000, 6000, "said", "product changes everything", "...", False),
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
    a = hc.HeroCut("h1", "fff", "said", "the product changes everything", 1000, 2000,
                   score=0.9, quality={"on_camera": 0.0, "delivery": 0.5})
    b = hc.HeroCut("h2", "fff", "said", "the product changes everything", 5000, 6000,
                   score=0.6, quality={"on_camera": 1.0, "delivery": 0.85})
    group = tk.TakeGroup(group_id="tg1", content_key="product changes everything", attempts=[
        tk.Attempt("fff:u1:0", "fff", "u1", 1000, 2000, "said", "product changes everything", "...", False),
        tk.Attempt("fff:u2:0", "fff", "u2", 5000, 6000, "said", "product changes everything", "...", False),
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


def test_facet_record_round_trips():
    """The Cut facet record (ladder rungs + people/framing/quality) survives the
    cache round-trip, and a rung's keep_spans/play_ms reflect a split."""
    ladder = [
        hc.Rung(level="balanced", spans=[(1000, 3300)], text="the whole thought", score=0.7),
        hc.Rung(level="sharp", spans=[(1300, 1800), (2200, 2500)], text="punch", score=0.8),
    ]
    h = hc.HeroCut(
        "z:sp0", "zzzz", "said", "the whole thought", 1000, 3300, score=0.7,
        speaker="interviewer", ladder=ladder,
        people=[{"voice_speaker_id": "S0", "person_id": "p1", "role": "interviewer",
                 "on_camera": True, "region": {"x": 0.1, "y": 0.1, "w": 0.3, "h": 0.5},
                 "av_link_confidence": 1.0}],
        framing={"shot_size": "medium", "angle": "eye_level", "movement": "static"},
        quality={"delivery": 0.82, "vlm": 0.6},
        summary="revenue up 40% in Q3",
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
    assert back.summary == "revenue up 40% in Q3"
    # A cut with no facets emits null (compact) and rehydrates empty.
    plain = hc.HeroCut("z:sp1", "zzzz", "said", "x", 0, 100, score=0.1)
    d = plain.to_dict()
    assert d["ladder"] is None and d["people"] is None
    assert hc.HeroCut.from_cache(d).ladder == []
    print("ok  test_facet_record_round_trips")


def main():
    test_speech_drops_offcamera_and_short()
    test_energy_selects_granularity()
    test_clustering_gradient()
    test_sharp_breath_removal_edit_list()
    test_dead_air_floor_excises_long_hole_at_every_band()
    test_take_stacking_collapses_repeats()
    test_best_take_prefers_on_camera_then_delivery()
    test_thought_bands_select_hierarchy()
    test_thought_turn_merge_at_broad()
    test_speech_cut_owns_ladder_and_resolves_speaker()
    test_offcamera_speech_flagged_not_dropped()
    test_facet_record_round_trips()
    print("\nall hero-cuts tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
