"""
Pure unit tests for app.services.vcut.salience -- the salience bridge
(brain_cut_salience_parity.plan.md). No DB, no model call; also exercises
the READ-side machinery it lights up (cutrecord_map.synth_ladder/_video_rung,
footage_map.piece_breakdown/_landmarks_tag/_specific_tag) directly against
synthesized salience/landmarks, since that machinery-lighting-up IS the
parity being tested, not just the bridge's own output shape.

Run:  .venv/bin/python scripts/test_vcut_salience.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import cutrecord_map as cm  # noqa: E402
from app.services.l3 import footage_map as fm  # noqa: E402
from app.services.vcut.resolve import FilePlan, MomentFlag, MomentPlan, ResolvedCut, resolve_cuts  # noqa: E402
from app.services.vcut.salience import build_landmarks, build_salience  # noqa: E402


def _moment(peak_ms, shape="both", summary="", specifics=None):
    return {"peak_ms": peak_ms, "shape": shape, "summary": summary, "specifics": specifics or {}}


def _cut(file_id="f1", in_ms=0, out_ms=10000, peak_ms=5000, tag="both", moments=None):
    return ResolvedCut(file_id=file_id, in_ms=in_ms, out_ms=out_ms, peak_ms=peak_ms, tag=tag,
                       summary="", moments=moments if moments is not None else [_moment(peak_ms, tag)])


def _row(**over):
    row = {
        "id": "cut-1", "file_id": "f1", "kind": "video",
        "src_in_ms": 0, "src_out_ms": 10000, "hero_ts_ms": 5000,
        "pace": {"min_ms": 800, "natural_ms": 10000, "max_ms": 10000, "natural_sound": True},
    }
    row.update(over)
    return row


# --------------------------------------------------------------------------
# section 10.1: bridge emits correct events from flags
# --------------------------------------------------------------------------

def test_build_salience_emits_one_event_per_moment_with_floored_clamped_windows():
    moments = [_moment(1000, "both"), _moment(5000, "both"), _moment(9000, "both")]
    cut = _cut(in_ms=0, out_ms=10000, peak_ms=5000, tag="both", moments=moments)
    action_energy = [0.0] * 101
    action_energy[10] = 0.2   # bin for 1000ms (hop_ms=100)
    action_energy[50] = 1.0   # bin for 5000ms -- strongest
    action_energy[90] = 0.5   # bin for 9000ms
    sal = build_salience(cut, hop_ms=100, action_energy=action_energy, mean_ae=0.3)

    events = sal["events"]
    assert [e["peak_ms"] for e in events] == [1000, 5000, 9000]
    assert all(e["kind"] == "point" and e["span_ms"] is None for e in events)
    # RUN_UP_FLOOR_MS=300, FOLLOW_THROUGH_FLOOR_MS=500, clamped into [0,10000].
    assert (events[0]["onset_ms"], events[0]["settle_ms"]) == (700, 1500)
    assert (events[1]["onset_ms"], events[1]["settle_ms"]) == (4700, 5500)
    assert (events[2]["onset_ms"], events[2]["settle_ms"]) == (8700, 9500)
    # min-max normalized against THIS cut's own three events: lo=0.2, hi=1.0.
    assert events[0]["score"] == 0.0
    assert events[1]["score"] == 1.0
    assert events[2]["score"] == 0.375
    assert sal["primary"] == 1   # the event matching cut.peak_ms=5000
    assert sal["peak_ms"] == 5000 and sal["score"] == 1.0 and sal["kind"] == "point"
    print("ok  test_build_salience_emits_one_event_per_moment_with_floored_clamped_windows")


def test_build_salience_onset_settle_clamp_to_cut_bounds():
    cut = _cut(in_ms=0, out_ms=1000, peak_ms=100, tag="both", moments=[_moment(100, "both")])
    sal = build_salience(cut, hop_ms=100, action_energy=[], mean_ae=0.0)
    ev = sal["events"][0]
    assert ev["onset_ms"] == 0     # 100-300 clamps to cut.in_ms
    assert ev["settle_ms"] == 600  # 100+500=600, already inside [0,1000]
    print("ok  test_build_salience_onset_settle_clamp_to_cut_bounds")


def test_build_salience_no_signal_all_events_score_equal():
    moments = [_moment(1000, "both"), _moment(5000, "both")]
    cut = _cut(in_ms=0, out_ms=10000, peak_ms=1000, tag="both", moments=moments)
    sal = build_salience(cut, hop_ms=100, action_energy=[], mean_ae=0.0)
    assert all(e["score"] == 1.0 for e in sal["events"])
    print("ok  test_build_salience_no_signal_all_events_score_equal")


def test_build_salience_rides_summary_and_specifics_onto_events():
    moments = [_moment(1000, "build", summary="barista grabs cup", specifics={"subject": "barista"})]
    cut = _cut(in_ms=0, out_ms=2000, peak_ms=1000, tag="build", moments=moments)
    sal = build_salience(cut, hop_ms=100, action_energy=[], mean_ae=0.0)
    ev = sal["events"][0]
    assert ev["summary"] == "barista grabs cup"
    assert ev["specifics"] == {"subject": "barista"}
    print("ok  test_build_salience_rides_summary_and_specifics_onto_events")


def test_build_salience_primary_tie_break_is_deterministic_and_max_scoring():
    # section 11's degenerate case: two DISTINCT moments refined onto the
    # SAME peak_ms (both match cut.peak_ms -- and, since score is derived
    # purely from that shared peak's own action_energy bin, both events
    # necessarily tie in score too). The multi-match branch must still run
    # cleanly and land on a genuine max-score event -- never an unvalidated
    # index-0 guess -- and do so deterministically (same input -> same
    # primary every time).
    moments = [
        {"peak_ms": 5000, "shape": "both", "summary": "a", "specifics": {}},
        {"peak_ms": 5000, "shape": "both", "summary": "b", "specifics": {}},
    ]
    cut = _cut(in_ms=0, out_ms=10000, peak_ms=5000, tag="both", moments=moments)
    action_energy = [0.0] * 101
    action_energy[50] = 1.0
    first = build_salience(cut, hop_ms=100, action_energy=action_energy, mean_ae=0.3)
    second = build_salience(cut, hop_ms=100, action_energy=action_energy, mean_ae=0.3)
    assert first["primary"] == second["primary"], "must be deterministic across identical calls"
    assert first["events"][first["primary"]]["score"] == max(e["score"] for e in first["events"])
    print("ok  test_build_salience_primary_tie_break_is_deterministic_and_max_scoring")


def test_build_salience_density_is_clamped_mean_action_energy():
    cut = _cut()
    assert build_salience(cut, 100, [], mean_ae=1.4)["density"] == 1.0
    assert build_salience(cut, 100, [], mean_ae=-0.2)["density"] == 0.0
    assert build_salience(cut, 100, [], mean_ae=0.42)["density"] == 0.42
    print("ok  test_build_salience_density_is_clamped_mean_action_energy")


# --------------------------------------------------------------------------
# section 10.2 + 10.4: shape mapping and the resulting directional trim via
# cutrecord_map._video_rung (that rung's own before/after/center math is
# already proven correct elsewhere -- test_cutrecord_map.py; this only
# checks build_salience feeds it the right shape).
# --------------------------------------------------------------------------

def test_shape_mapping_build_settle_both_and_unknown():
    for tag, expected in (("build", "before"), ("settle", "after"), ("both", "center"), ("unknown", "center")):
        cut = _cut(peak_ms=5000, tag=tag, moments=[_moment(5000, tag)])
        sal = build_salience(cut, hop_ms=100, action_energy=[], mean_ae=0.0)
        assert sal["shape"] == expected, (tag, sal["shape"])
    print("ok  test_shape_mapping_build_settle_both_and_unknown")


def test_before_shape_routes_to_before_rung_head_trims():
    cut = _cut(in_ms=0, out_ms=10000, peak_ms=9600, tag="build", moments=[_moment(9600, "build")])
    sal = build_salience(cut, hop_ms=100, action_energy=[], mean_ae=0.0)
    row = _row(hero_ts_ms=9999, salience=sal)
    low = cm._video_rung(row, energy=0.1, level="broad", score=0.5)
    high = cm._video_rung(row, energy=1.0, level="sharp", score=0.5)
    assert abs(low["out_ms"] - high["out_ms"]) < 500, (low, high)
    assert high["in_ms"] - low["in_ms"] > 5000, (low, high)
    print("ok  test_before_shape_routes_to_before_rung_head_trims")


def test_after_shape_routes_to_after_rung_tail_trims():
    cut = _cut(in_ms=0, out_ms=10000, peak_ms=400, tag="settle", moments=[_moment(400, "settle")])
    sal = build_salience(cut, hop_ms=100, action_energy=[], mean_ae=0.0)
    row = _row(hero_ts_ms=1, salience=sal)
    low = cm._video_rung(row, energy=0.1, level="broad", score=0.5)
    high = cm._video_rung(row, energy=1.0, level="sharp", score=0.5)
    assert abs(low["in_ms"] - high["in_ms"]) < 500, (low, high)
    assert low["out_ms"] - high["out_ms"] > 5000, (low, high)
    print("ok  test_after_shape_routes_to_after_rung_tail_trims")


# --------------------------------------------------------------------------
# section 10.3: the ladder fragments a multi-flag cut
# --------------------------------------------------------------------------

def test_ladder_fragments_a_multi_flag_cut_broad_stays_whole():
    # The two OUTER events tie for strongest (score 1.0 each); the middle one
    # is comparatively weak (min-max normalizes it to 0.0) and gets pruned at
    # the sharp band -- the two survivors, far apart in time, fragment into
    # separate pieces.
    moments = [_moment(1000, "both"), _moment(5000, "both"), _moment(9000, "both")]
    cut = _cut(in_ms=0, out_ms=10000, peak_ms=5000, tag="both", moments=moments)
    action_energy = [0.0] * 101
    action_energy[10], action_energy[50], action_energy[90] = 1.0, 0.5, 1.0
    sal = build_salience(cut, hop_ms=100, action_energy=action_energy, mean_ae=0.3)
    row = _row(salience=sal)
    ladder = cm.synth_ladder(row, cm._score_for(row))
    by_level = {r["level"]: r for r in ladder}
    assert len(by_level["broad"]["spans"]) == 1
    assert by_level["broad"]["in_ms"] == 0 and by_level["broad"]["out_ms"] == 10000
    assert len(by_level["sharp"]["spans"]) >= 2
    print("ok  test_ladder_fragments_a_multi_flag_cut_broad_stays_whole")


# --------------------------------------------------------------------------
# section 10.7: landmarks.act
# --------------------------------------------------------------------------

def test_build_landmarks_act_channel_counts_interior_moments():
    moments = [_moment(1000, "both"), _moment(5000, "both"), _moment(9000, "both")]
    cut = _cut(in_ms=0, out_ms=10000, peak_ms=5000, tag="both", moments=moments)
    assert build_landmarks(cut, {}) == {"act": {"n": 3, "hits": [1000, 5000, 9000]}}
    print("ok  test_build_landmarks_act_channel_counts_interior_moments")


def test_build_landmarks_empty_when_single_moment_pinned_to_edge():
    cut = _cut(in_ms=1000, out_ms=2000, peak_ms=1000, tag="both", moments=[_moment(1000, "both")])
    assert build_landmarks(cut, {}) == {}
    print("ok  test_build_landmarks_empty_when_single_moment_pinned_to_edge")


def test_build_landmarks_caps_hits_at_five_but_n_is_the_true_count():
    moments = [_moment(ms, "both") for ms in (1000, 2000, 3000, 4000, 5000, 6000)]
    cut = _cut(in_ms=0, out_ms=7000, peak_ms=1000, tag="both", moments=moments)
    landmarks = build_landmarks(cut, {})
    assert landmarks["act"]["n"] == 6
    assert len(landmarks["act"]["hits"]) == 5
    print("ok  test_build_landmarks_caps_hits_at_five_but_n_is_the_true_count")


def test_landmarks_tag_renders_sig_actn_from_bridge_output():
    moments = [_moment(1000, "both"), _moment(5000, "both"), _moment(9000, "both")]
    cut = _cut(in_ms=0, out_ms=10000, peak_ms=5000, tag="both", moments=moments)
    m = {"landmarks": build_landmarks(cut, {})}
    assert fm._landmarks_tag(m) == " sig:act3", fm._landmarks_tag(m)
    print("ok  test_landmarks_tag_renders_sig_actn_from_bridge_output")


# --------------------------------------------------------------------------
# full-landmark-parity extension: adx/sil/shot channels from the widened
# seam cache (rms_db / silence_intervals / shot_points+composition_points),
# reusing the SHARED l3.landmarks builders. Each channel's exact shape is
# asserted against post.py's own contract:
#   adx  = {"n", "changes":[{"off","dir"}]}
#   sil  = {"n", "gaps":[{"off","dur"}]}
#   shot = {"n", "cuts":[{"off","hard"}]}
# --------------------------------------------------------------------------

def _landmark_seam(rms_db=None, rms_hop_ms=100, silence_intervals=None,
                   shot_points=None, composition_points=None):
    return {
        "rms_db": rms_db if rms_db is not None else [],
        "rms_hop_ms": rms_hop_ms,
        "silence_intervals": silence_intervals if silence_intervals is not None else [],
        "shot_points": shot_points if shot_points is not None else [],
        "composition_points": composition_points if composition_points is not None else [],
    }


def _edge_pinned_cut():
    # single moment pinned to the head -> no interior act channel, so a
    # channel assertion isn't polluted by the moment-driven act channel.
    return _cut(in_ms=0, out_ms=10000, peak_ms=100, tag="both", moments=[_moment(100, "both")])


def test_build_landmarks_adx_channel_from_seam_rms_db():
    # A single clip-relative step-up at bin 50 (rms_hop_ms=100 -> 5000ms),
    # interior to [0,10000] (edge_guard=1000). series_lohi over the whole
    # rms_db gives lo=-40/hi=-10, so the step normalizes to a +1.0 delta,
    # well past the _ADX_MIN_DELTA=0.12 floor.
    rms = [-40.0] * 50 + [-10.0] * 51
    lm = build_landmarks(_edge_pinned_cut(), _landmark_seam(rms_db=rms, rms_hop_ms=100))
    assert "act" not in lm, lm
    assert lm["adx"] == {"n": 1, "changes": [{"off": 5000, "dir": "up"}]}, lm["adx"]
    print("ok  test_build_landmarks_adx_channel_from_seam_rms_db")


def test_build_landmarks_sil_channel_from_seam_silence_intervals():
    seam = _landmark_seam(rms_hop_ms=100, silence_intervals=[{"start_ms": 4000, "end_ms": 5000}])
    lm = build_landmarks(_edge_pinned_cut(), seam)
    assert lm["sil"] == {"n": 1, "gaps": [{"off": 4000, "dur": 1000}]}, lm["sil"]
    print("ok  test_build_landmarks_sil_channel_from_seam_silence_intervals")


def test_build_landmarks_shot_channel_hard_and_soft_from_seam():
    seam = _landmark_seam(shot_points=[{"ts_ms": 3000}], composition_points=[{"ts_ms": 6000}])
    lm = build_landmarks(_edge_pinned_cut(), seam)
    assert lm["shot"] == {"n": 2, "cuts": [{"off": 3000, "hard": True}, {"off": 6000, "hard": False}]}, lm["shot"]
    print("ok  test_build_landmarks_shot_channel_hard_and_soft_from_seam")


def test_build_landmarks_all_four_channels_and_full_sig_breadcrumb():
    moments = [_moment(5000, "both")]  # one interior act hit
    cut = _cut(in_ms=0, out_ms=10000, peak_ms=5000, tag="both", moments=moments)
    seam = _landmark_seam(
        rms_db=[-40.0] * 50 + [-10.0] * 51, rms_hop_ms=100,
        silence_intervals=[{"start_ms": 7000, "end_ms": 8000}],
        shot_points=[{"ts_ms": 3000}], composition_points=[{"ts_ms": 6000}],
    )
    lm = build_landmarks(cut, seam)
    assert set(lm) == {"act", "adx", "sil", "shot"}, set(lm)
    assert lm["act"]["n"] == 1 and lm["adx"]["n"] == 1 and lm["sil"]["n"] == 1 and lm["shot"]["n"] == 2
    assert fm._landmarks_tag({"landmarks": lm}) == " sig:act1,adx1,sil1,shot2", fm._landmarks_tag({"landmarks": lm})
    print("ok  test_build_landmarks_all_four_channels_and_full_sig_breadcrumb")


def test_build_landmarks_absent_signal_omits_channel_not_fabricated_zero():
    # Silent footage / no scene cuts -> empty seam signals. adx/sil/shot must
    # be ABSENT entirely (an honest per-file gap), never a fabricated {"n":0}.
    cut = _cut(in_ms=0, out_ms=10000, peak_ms=5000, tag="both", moments=[_moment(5000, "both")])
    lm = build_landmarks(cut, _landmark_seam())
    assert set(lm) == {"act"}, lm
    assert "adx" not in lm and "sil" not in lm and "shot" not in lm
    print("ok  test_build_landmarks_absent_signal_omits_channel_not_fabricated_zero")


def test_build_landmarks_present_signal_no_interior_structure_still_absent():
    # A shot cut pinned inside the edge guard (500ms < 1000ms guard) carries
    # no INTERIOR structure -> the channel is correctly omitted even though
    # the signal IS present (not a silent {} degrade of a real signal, just
    # nothing interior to report -- identical to post._shot_cuts' None).
    seam = _landmark_seam(shot_points=[{"ts_ms": 500}])
    lm = build_landmarks(_edge_pinned_cut(), seam)
    assert lm == {}, lm
    print("ok  test_build_landmarks_present_signal_no_interior_structure_still_absent")


# --------------------------------------------------------------------------
# section 10.5 + 10.6: footage_map's moments-list recompose (section 4 step
# 2) and beat-line parity between a vcut-synthesized salience block and the
# old pipeline's own literal shape.
# --------------------------------------------------------------------------

def _tree_cut(hero_id, in_ms, out_ms, salience=None, landmarks=None, scene_specifics=None):
    d = {
        "hero_id": hero_id, "file_id": "ffffffff-1111", "channel": "shown",
        "subject": "object", "label": "b-roll", "src_in_ms": in_ms, "src_out_ms": out_ms,
        "play_ms": out_ms - in_ms, "keep_spans": None, "score": 0.7, "speaker": None,
        "flags": [], "take_count": 1, "ladder": None,
    }
    if salience is not None:
        d["salience"] = salience
    if landmarks is not None:
        d["landmarks"] = landmarks
    if scene_specifics is not None:
        d["scene_specifics"] = scene_specifics
    return d


def test_specifics_recompose_per_rung_drops_pruned_events_at_sharpest_band():
    moments = [
        _moment(1000, "both", summary="grabs cup"),
        _moment(5000, "both", summary="pours latte"),
        _moment(9000, "both", summary="hands it over"),
    ]
    cut = _cut(in_ms=0, out_ms=10000, peak_ms=5000, tag="both", moments=moments)
    action_energy = [0.0] * 101
    action_energy[10], action_energy[50], action_energy[90] = 1.0, 0.5, 1.0
    sal = build_salience(cut, hop_ms=100, action_energy=action_energy, mean_ae=0.3)

    # A legacy frozen scene_specifics.moments snapshot -- the LIVE list
    # (derived from surviving salience.events) must win over it.
    spec = {"subject": "barista", "moments": [
        {"t_ms": 1000, "summary": "OLD FROZEN grabs cup"},
        {"t_ms": 5000, "summary": "OLD FROZEN pours latte"},
        {"t_ms": 9000, "summary": "OLD FROZEN hands it over"},
    ]}
    m = {"salience": sal, "in_ms": 0, "out_ms": 10000}
    live = fm._live_moments(spec, m)
    live_summaries = {mo["summary"] for mo in live}
    assert live_summaries == {"grabs cup", "hands it over"}, live_summaries   # the pruned survivors
    assert not any("OLD FROZEN" in s for s in live_summaries)
    print("ok  test_specifics_recompose_per_rung_drops_pruned_events_at_sharpest_band")


def test_live_moments_falls_back_to_frozen_snapshot_for_single_event_cut():
    # A single-flag cut has no moments list of its own (matches
    # _composed_specifics' "no moments key at all" contract) -- falls back
    # to whatever spec["moments"] holds (normally empty too).
    cut = _cut(in_ms=0, out_ms=2000, peak_ms=1000, tag="both", moments=[_moment(1000, "both")])
    sal = build_salience(cut, hop_ms=100, action_energy=[], mean_ae=0.0)
    m = {"salience": sal, "in_ms": 0, "out_ms": 2000}
    assert fm._live_moments({"moments": []}, m) == []
    print("ok  test_live_moments_falls_back_to_frozen_snapshot_for_single_event_cut")


def test_live_moments_falls_back_to_frozen_snapshot_when_events_carry_no_payload():
    # A legacy multi-event salience block with no per-event summary/specifics
    # (an un-re-ingested run predating section 4's "ride specifics on
    # events" change) -- falls back to the frozen scene_specifics.moments.
    events = [
        {"peak_ms": 1000, "score": 1.0, "kind": "point", "onset_ms": 700, "settle_ms": 1500, "span_ms": None},
        {"peak_ms": 5000, "score": 1.0, "kind": "point", "onset_ms": 4700, "settle_ms": 5500, "span_ms": None},
    ]
    m = {"salience": {"events": events}, "in_ms": 0, "out_ms": 6000}
    spec = {"moments": [{"t_ms": 1000, "summary": "frozen a"}, {"t_ms": 5000, "summary": "frozen b"}]}
    assert fm._live_moments(spec, m) == spec["moments"]
    print("ok  test_live_moments_falls_back_to_frozen_snapshot_when_events_carry_no_payload")


def test_parity_vcut_salience_renders_identical_beat_line_to_old_pipeline_shape():
    """A vcut-synthesized salience block and a hand-built old-pipeline-shape
    block (same peaks/scores, no additive summary/specifics keys on its
    events) must render IDENTICAL range:/peak:/sig: tags -- footage_map's
    read side is generic over which pipeline produced the block."""
    moments = [_moment(1000, "both"), _moment(5000, "both"), _moment(9000, "both")]
    cut = _cut(in_ms=0, out_ms=10000, peak_ms=5000, tag="both", moments=moments)
    action_energy = [0.0] * 101
    action_energy[10], action_energy[50], action_energy[90] = 1.0, 0.5, 1.0
    vcut_sal = build_salience(cut, hop_ms=100, action_energy=action_energy, mean_ae=0.3)
    vcut_lm = build_landmarks(cut, {})

    old_sal = {
        "peak_ms": vcut_sal["peak_ms"], "score": vcut_sal["score"], "kind": "point",
        "shape": vcut_sal["shape"], "span_ms": None, "primary": vcut_sal["primary"],
        "density": vcut_sal["density"],
        "events": [
            {"peak_ms": e["peak_ms"], "score": e["score"], "kind": "point",
             "onset_ms": e["onset_ms"], "settle_ms": e["settle_ms"], "span_ms": None}
            for e in vcut_sal["events"]
        ],
    }

    vcut_tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 12000},
                                   [_tree_cut("f:v", 0, 10000, salience=vcut_sal, landmarks=vcut_lm)])
    old_tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 12000},
                                  [_tree_cut("f:o", 0, 10000, salience=old_sal, landmarks=vcut_lm)])

    vcut_line = fm._moment_line(vcut_tree["moments"][0])
    old_line = fm._moment_line(old_tree["moments"][0])

    import re

    def _tag(line, pattern):
        match = re.search(pattern, line)
        return match.group() if match else None

    assert _tag(vcut_line, r"peak:\+[\d.]+s") == _tag(old_line, r"peak:\+[\d.]+s")
    assert _tag(vcut_line, r"sig:\S+") == _tag(old_line, r"sig:\S+")

    vcut_pieces = fm._piece_lines(vcut_tree["moments"][0])
    old_pieces = fm._piece_lines(old_tree["moments"][0])
    assert vcut_pieces == old_pieces, (vcut_pieces, old_pieces)
    print("ok  test_parity_vcut_salience_renders_identical_beat_line_to_old_pipeline_shape")


# --------------------------------------------------------------------------
# section 10.8: hard-fail -- no silent fallbacks
# --------------------------------------------------------------------------

def test_build_salience_raises_on_malformed_moment_missing_peak_ms():
    cut = _cut(moments=[{"shape": "both", "summary": "x", "specifics": {}}])
    try:
        build_salience(cut, hop_ms=100, action_energy=[], mean_ae=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ok  test_build_salience_raises_on_malformed_moment_missing_peak_ms")


def test_build_salience_raises_on_malformed_moment_missing_shape():
    cut = _cut(moments=[{"peak_ms": 1000, "summary": "x", "specifics": {}}])
    try:
        build_salience(cut, hop_ms=100, action_energy=[], mean_ae=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ok  test_build_salience_raises_on_malformed_moment_missing_shape")


def test_build_salience_raises_on_empty_moments():
    cut = _cut(moments=[])
    try:
        build_salience(cut, hop_ms=100, action_energy=[], mean_ae=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
    print("ok  test_build_salience_raises_on_empty_moments")


# --------------------------------------------------------------------------
# section 10.9: energy-invariance / re-resolve
# --------------------------------------------------------------------------

def test_energy_invariance_salience_regenerates_and_moment_peak_stable():
    n = 141
    flags = [MomentFlag(t_ms=ms, shape="both") for ms in (5000, 7000, 9000)]
    plan = MomentPlan(files=[FilePlan(file_id="f1", flags=flags)])
    ae = [0.0] * n
    for ms in (5000, 7000, 9000):
        ae[ms // 100] = 1.0
    S = [0.1] * n
    S[5] = 1.0  # spike well away from the flags -- no strong-seam wall
    seam = {"f1": {"hop_ms": 100, "S": S, "action_energy": ae, "frame_diff": [0.0] * n}}

    low = resolve_cuts(plan, seam, energy=0.0)    # fuses into ONE cut
    high = resolve_cuts(plan, seam, energy=0.7)   # falls apart into three

    assert len(low) == 1
    sal_low = build_salience(low[0], 100, ae, 0.3)
    assert len(sal_low["events"]) == 3

    assert len(high) == 3
    for cut in high:
        sal = build_salience(cut, 100, ae, 0.3)
        assert len(sal["events"]) == 1
        assert sal["events"][0]["peak_ms"] == cut.peak_ms
    print("ok  test_energy_invariance_salience_regenerates_and_moment_peak_stable")


# --------------------------------------------------------------------------
# section 10.10: speech untouched
# --------------------------------------------------------------------------

def test_speech_cut_record_leaves_salience_and_landmarks_at_default():
    from app.services.l3.post import CutRecord

    record = CutRecord(
        file_id="f1", src_in_ms=0, src_out_ms=1000, kind="speech", word_span=None, atom_ids=None,
        label="x", summary="x", on_camera=None, junk=False, junk_reason="",
        framing={}, look={}, caption_zones=[], hero_ts_ms=500,
        pace=None, take_group_id=None, take_role=None, channel="said",
        speech_quality=0.5, total_quality=0.5,
    )
    assert record.salience == {} and record.landmarks == {}
    print("ok  test_speech_cut_record_leaves_salience_and_landmarks_at_default")


def test_speech_row_routes_to_speech_rung_not_video_ladder():
    row = _row(kind="speech", pace={"min_ms": 500, "natural_ms": 2000, "max_ms": 2000,
                                    "natural_sound": True, "remove_spans": [[400, 900]]})
    ladder = cm.synth_ladder(row, 0.5)
    for rung in ladder:
        assert "spans" in rung   # the speech/cluster shape, not a single-window video rung
    print("ok  test_speech_row_routes_to_speech_rung_not_video_ladder")


def main():
    test_build_salience_emits_one_event_per_moment_with_floored_clamped_windows()
    test_build_salience_onset_settle_clamp_to_cut_bounds()
    test_build_salience_no_signal_all_events_score_equal()
    test_build_salience_rides_summary_and_specifics_onto_events()
    test_build_salience_primary_tie_break_is_deterministic_and_max_scoring()
    test_build_salience_density_is_clamped_mean_action_energy()
    test_shape_mapping_build_settle_both_and_unknown()
    test_before_shape_routes_to_before_rung_head_trims()
    test_after_shape_routes_to_after_rung_tail_trims()
    test_ladder_fragments_a_multi_flag_cut_broad_stays_whole()
    test_build_landmarks_act_channel_counts_interior_moments()
    test_build_landmarks_empty_when_single_moment_pinned_to_edge()
    test_build_landmarks_caps_hits_at_five_but_n_is_the_true_count()
    test_landmarks_tag_renders_sig_actn_from_bridge_output()
    test_build_landmarks_adx_channel_from_seam_rms_db()
    test_build_landmarks_sil_channel_from_seam_silence_intervals()
    test_build_landmarks_shot_channel_hard_and_soft_from_seam()
    test_build_landmarks_all_four_channels_and_full_sig_breadcrumb()
    test_build_landmarks_absent_signal_omits_channel_not_fabricated_zero()
    test_build_landmarks_present_signal_no_interior_structure_still_absent()
    test_specifics_recompose_per_rung_drops_pruned_events_at_sharpest_band()
    test_live_moments_falls_back_to_frozen_snapshot_for_single_event_cut()
    test_live_moments_falls_back_to_frozen_snapshot_when_events_carry_no_payload()
    test_parity_vcut_salience_renders_identical_beat_line_to_old_pipeline_shape()
    test_build_salience_raises_on_malformed_moment_missing_peak_ms()
    test_build_salience_raises_on_malformed_moment_missing_shape()
    test_build_salience_raises_on_empty_moments()
    test_energy_invariance_salience_regenerates_and_moment_peak_stable()
    test_speech_cut_record_leaves_salience_and_landmarks_at_default()
    test_speech_row_routes_to_speech_rung_not_video_ladder()
    print("\nall vcut salience tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
