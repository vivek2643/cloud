"""
Pure unit tests for cuts-v3 post-compute assembly (``app.services.l3.post``)
-- no DB, no ffmpeg, no model calls.

Run:  .venv/bin/python scripts/test_post.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import post  # noqa: E402
from app.services.l3.lattice import Atom, Lattice  # noqa: E402
from app.services.l3.pass1 import JunkSuspect  # noqa: E402
from app.services.l3.pass2 import Pass2Cut, Pass2Output  # noqa: E402


# --------------------------------------------------------------------------
# hero_ts_ms
# --------------------------------------------------------------------------

def test_hero_ts_prefers_anchor_over_sharp():
    ts = post.pick_hero_ts_ms([1500], blur=[0.9] * 10, hop_ms=100, s=0, e=2000)
    assert ts == 1500
    print("ok  test_hero_ts_prefers_anchor_over_sharp")


def test_hero_ts_falls_back_to_sharpest():
    blur = [0.9] * 10
    blur[3] = 0.05
    ts = post.pick_hero_ts_ms([], blur, hop_ms=100, s=0, e=1000)
    assert ts == 300, ts
    print("ok  test_hero_ts_falls_back_to_sharpest")


def test_hero_ts_falls_back_to_midpoint_with_no_blur():
    ts = post.pick_hero_ts_ms([], [], hop_ms=0, s=0, e=1000)
    assert ts == 500, ts
    print("ok  test_hero_ts_falls_back_to_midpoint_with_no_blur")


# --------------------------------------------------------------------------
# pace envelope
# --------------------------------------------------------------------------

def test_speech_pace_envelope_is_native_speed_only():
    pace = post.compute_pace_envelope(
        kind="speech", s=1000, e=3000, readability_ms=500, anchors=[],        action_energy=[0.1] * 40, hop_ms=100, next_cut_start_ms=3000,
        max_tasteful_speed=2.0, min_tasteful_speed=0.5, natural_sound=True,
    )
    assert pace.natural_ms == 2000, pace
    assert pace.min_ms == 2000, pace
    assert pace.max_ms == 2000, pace
    assert pace.levels == [1.0] * 5, pace
    assert pace.natural_sound is True
    print("ok  test_speech_pace_envelope_is_native_speed_only")


def test_video_pace_min_ms_from_anchor_span():
    pace = post.compute_pace_envelope(
        kind="video", s=1000, e=5000, readability_ms=0, anchors=[2000, 2400],        action_energy=[0.0] * 60, hop_ms=100, next_cut_start_ms=5000,
        max_tasteful_speed=2.0, min_tasteful_speed=0.5, natural_sound=False,
    )
    assert pace.min_ms == 900, pace   # (2400-2000) + 2*250
    print("ok  test_video_pace_min_ms_from_anchor_span")


def test_video_pace_max_ms_extends_through_flatline():
    pace = post.compute_pace_envelope(
        kind="video", s=1000, e=2000, readability_ms=0, anchors=[],        action_energy=[0.1] * 100, hop_ms=100, next_cut_start_ms=8000,
        max_tasteful_speed=2.0, min_tasteful_speed=0.5, natural_sound=False,
    )
    assert pace.max_ms == 7000, pace   # flat all the way to the ceiling (8000-1000)
    print("ok  test_video_pace_max_ms_extends_through_flatline")


def test_video_pace_max_ms_stops_at_departure_from_flatline():
    ae = [0.1] * 100
    ae[30] = 0.9   # departs the flat band at ts=3000
    pace = post.compute_pace_envelope(
        kind="video", s=1000, e=2000, readability_ms=0, anchors=[],        action_energy=ae, hop_ms=100, next_cut_start_ms=8000,
        max_tasteful_speed=2.0, min_tasteful_speed=0.5, natural_sound=False,
    )
    assert pace.max_ms == 2000, pace   # 3000 - s(1000)
    print("ok  test_video_pace_max_ms_stops_at_departure_from_flatline")


def test_pace_levels_partial_saturation_repeats_nearest_reachable():
    pace = post.compute_pace_envelope(
        kind="video", s=0, e=1000, readability_ms=0, anchors=[],        action_energy=[1.0] * 10, hop_ms=100, next_cut_start_ms=1000,
        max_tasteful_speed=1.2, min_tasteful_speed=0.25, natural_sound=False,
    )
    assert pace.levels == [0.5, 0.8, 1.0, 1.2, 1.2], pace.levels
    assert pace.levels == sorted(pace.levels), "levels must stay monotonic"
    print("ok  test_pace_levels_partial_saturation_repeats_nearest_reachable")


def test_pace_levels_zero_intrinsic_velocity_maxes_every_level():
    pace = post.compute_pace_envelope(
        kind="video", s=0, e=1000, readability_ms=0, anchors=[],        action_energy=[0.0] * 10, hop_ms=100, next_cut_start_ms=1000,
        max_tasteful_speed=2.0, min_tasteful_speed=0.5, natural_sound=False,
    )
    assert pace.levels == [2.0] * 5, pace.levels
    print("ok  test_pace_levels_zero_intrinsic_velocity_maxes_every_level")


def test_energy_grade_bands():
    assert post._energy_grade(0.05) == "calm"
    assert post._energy_grade(0.3) == "active"
    assert post._energy_grade(0.9) == "high"
    print("ok  test_energy_grade_bands")


# --------------------------------------------------------------------------
# salience (perception_upgrade.plan.md Part D): a single code-owned "strongest
# instant" per cut, fused from action_energy + rms loudness + onset/anchor
# bumps, all normalized clip-relative (no absolute constants).
# --------------------------------------------------------------------------

def test_salience_no_signal_falls_back_to_hero_ts():
    sal = post._salience([], 100, 0, 1000, None, None, [], 0, None, None, [], [], 500)
    assert sal == {"peak_ms": 500, "score": 0.0}, sal
    print("ok  test_salience_no_signal_falls_back_to_hero_ts")


def test_salience_peaks_at_the_loudest_action_energy_hop():
    ae = [0.1] * 10
    ae[5] = 0.9
    lo, hi = post._series_lohi(ae)
    sal = post._salience(ae, 100, 0, 1000, lo, hi, [], 0, None, None, [], [], 999)
    assert sal["peak_ms"] == 500, sal
    assert sal["score"] == 1.0, sal
    print("ok  test_salience_peaks_at_the_loudest_action_energy_hop")


def test_salience_onset_bump_dominates_flat_energy():
    # A perfectly flat action_energy series is DEGENERATE under clip-relative
    # normalization (lo == hi -> _norm_in_clip is None everywhere) -- it
    # contributes nothing. An onset inside the span is then the only signal.
    ae = [0.5] * 10
    lo, hi = post._series_lohi(ae)
    sal = post._salience(ae, 100, 0, 1000, lo, hi, [], 0, None, None, [], [300], 999)
    assert sal["peak_ms"] == 300, sal
    assert sal["score"] == 1.0, sal
    print("ok  test_salience_onset_bump_dominates_flat_energy")


def test_salience_onset_outside_span_is_ignored():
    sal = post._salience([], 100, 1000, 2000, None, None, [], 0, None, None, [], [500], 1500)
    assert sal == {"peak_ms": 1500, "score": 0.0}, sal
    print("ok  test_salience_onset_outside_span_is_ignored")


def test_salience_uses_rms_loudness_when_action_energy_flat():
    ae = [0.2] * 10   # degenerate -> no contribution, isolates the rms term
    rms = [-40.0, -40.0, -10.0, -40.0, -40.0]   # rms_hop_ms=200ms over [0, 1000)
    rms_lo, rms_hi = post._series_lohi(rms)
    sal = post._salience(ae, 100, 0, 1000, 0.2, 0.2, rms, 200, rms_lo, rms_hi, [], [], 999)
    assert sal["peak_ms"] == 400, sal   # rms bin 2 (-10dB, loudest) covers ms [400, 600)
    assert sal["score"] == 1.0, sal
    print("ok  test_salience_uses_rms_loudness_when_action_energy_flat")


def _cam_motion(dx=None, dy=None, dz=None, action=None, coherence=None, hop_ms=100, n=10):
    z = [0.0] * n
    return {
        "hop_ms": hop_ms,
        "camera_dx": dx if dx is not None else list(z),
        "camera_dy": dy if dy is not None else list(z),
        "camera_zoom": dz if dz is not None else list(z),
        "action_energy": action if action is not None else list(z),
        "camera_coherence": coherence if coherence is not None else [1.0] * n,
    }


def test_classify_camera_move():
    span = (0, 1000)  # 10 hops @ 100ms -> dur 1.0s
    # No signal at all -> unknown (never fabricate a move).
    assert post.classify_camera_move({}, *span) == "unknown"
    # Flat -> static.
    assert post.classify_camera_move(_cam_motion(), *span) == "static"
    # Sign convention: +dx = scene right = camera pans LEFT; -dx = pans RIGHT.
    assert post.classify_camera_move(_cam_motion(dx=[0.02] * 10), *span) == "pan left"
    assert post.classify_camera_move(_cam_motion(dx=[-0.02] * 10), *span) == "pan right"
    # +dy = scene down = tilt UP; -dy = tilt DOWN.
    assert post.classify_camera_move(_cam_motion(dy=[0.02] * 10), *span) == "tilt up"
    assert post.classify_camera_move(_cam_motion(dy=[-0.02] * 10), *span) == "tilt down"
    # +zoom = scene expands = zoom IN; - = OUT.
    assert post.classify_camera_move(_cam_motion(dz=[0.01] * 10), *span) == "zoom in"
    assert post.classify_camera_move(_cam_motion(dz=[-0.01] * 10), *span) == "zoom out"
    # A pan tracking a busy subject in a coherent frame reads as following.
    assert post.classify_camera_move(
        _cam_motion(dx=[0.02] * 10, action=[0.6] * 10, coherence=[0.9] * 10), *span
    ) == "follow subject"
    # Big per-hop jitter, near-zero net, incoherent -> shaky, not static.
    jitter = [0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05, 0.05, -0.05]
    assert post.classify_camera_move(
        _cam_motion(dx=jitter, coherence=[0.2] * 10), *span
    ) == "shaky"
    print("ok  test_classify_camera_move")


# --------------------------------------------------------------------------
# coverage/overlap invariant
# --------------------------------------------------------------------------

def test_validate_no_overlap_passes_exact_coverage():
    post._validate_no_overlap("f1", [(0, 1000), (1000, 2000)], 2000)
    print("ok  test_validate_no_overlap_passes_exact_coverage")


def test_validate_no_overlap_allows_start_gap():
    # boundaries-v2: cuts are a selection, gaps are legal. Pre-roll dropped.
    post._validate_no_overlap("f1", [(100, 2000)], 2000)
    print("ok  test_validate_no_overlap_allows_start_gap")


def test_validate_no_overlap_allows_end_gap():
    post._validate_no_overlap("f1", [(0, 1900)], 2000)
    print("ok  test_validate_no_overlap_allows_end_gap")


def test_validate_no_overlap_raises_on_overlap():
    try:
        post._validate_no_overlap("f1", [(0, 1100), (1000, 2000)], 2000)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "overlap" in str(e)
    print("ok  test_validate_no_overlap_raises_on_overlap")


def test_validate_no_overlap_allows_middle_gap():
    # A dropped connective hold between two kept cuts -- legal now.
    post._validate_no_overlap("f1", [(0, 900), (1000, 2000)], 2000)
    print("ok  test_validate_no_overlap_allows_middle_gap")


def test_validate_no_overlap_allows_no_cuts():
    # An all-junk / all-dead-air file contributes zero cuts -- legal, not fatal.
    post._validate_no_overlap("f1", [], 2000)
    print("ok  test_validate_no_overlap_allows_no_cuts")


# --------------------------------------------------------------------------
# assemble_cut_records (end to end over a small synthetic file)
# --------------------------------------------------------------------------

def _words():
    return [
        {"start_ms": 0, "end_ms": 200, "text": "hey"},
        {"start_ms": 300, "end_ms": 500, "text": "there"},
        {"start_ms": 600, "end_ms": 800, "text": "friend"},
    ]


def _atoms():
    return [
        Atom(atom_id=0, file_id="f1", start_ms=800, end_ms=1400, state_in="speech_edge",
             state_out="shot", action_energy=0.2, coherence=0.9),
        Atom(atom_id=1, file_id="f1", start_ms=1400, end_ms=2000, state_in="shot",
             state_out="clip_edge", action_energy=0.4, coherence=0.8),
    ]


def _lattice():
    return Lattice(file_id="f1", duration_ms=2000, words=_words(), turns=[], hints=[], atoms=_atoms())


def _motion():
    return {"hop_ms": 100, "blur": [0.5] * 20, "action_energy": [0.2] * 20,
           "action_points": [{"ts_ms": 1700}]}


def test_assemble_cut_records_end_to_end():
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 2),
                label="intro", summary="says hi", readability_ms=300),
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0, 1],
                label="pan shot", summary="pans across the desk"),
    ])
    records = post.assemble_cut_records(p2, {"f1": _lattice()}, {"f1": _motion()}, {})
    assert len(records) == 2, records
    speech, video = records
    assert (speech.src_in_ms, speech.src_out_ms) == (0, 800), speech
    assert (video.src_in_ms, video.src_out_ms) == (800, 2000), video
    assert video.hero_ts_ms == 1700, video   # anchor inside the video span wins
    assert video.pace.min_ms == 500, video.pace   # anchor-span floor: 0 + 2*250 (single anchor)
    assert video.pace.max_ms == 1200, video.pace   # last cut in file -> capped at natural span
    assert speech.pace.levels == [1.0] * 5
    # Continuity: the two cuts touch (gap 0), same (unset) speaker on both
    # sides, no break-type atom boundary at the shared edge -> weldable, and
    # each edge case (first/last) reads False.
    assert speech.continuity == {
        "clip": "f1", "cut_no": 1, "of": 2,
        "prev_contiguous": False, "next_contiguous": True,
        "seam_reason_prev": None, "seam_reason_next": "continuous take",
    }, speech.continuity
    assert video.continuity == {
        "clip": "f1", "cut_no": 2, "of": 2,
        "prev_contiguous": True, "next_contiguous": False,
        "seam_reason_prev": "continuous take", "seam_reason_next": None,
    }, video.continuity
    print("ok  test_assemble_cut_records_end_to_end")


# --------------------------------------------------------------------------
# V4 (cuts_v4_segmentation.plan.md): v4_meta_by_ref overrides span/salience,
# density scales min_ms instead of the anchor-span formula.
# --------------------------------------------------------------------------

def test_v4_meta_by_ref_overrides_span_and_salience_and_stamps_shape():
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0, 1],
                label="pan shot", summary="pans across the desk", shape="after"),
    ])
    v4_meta = {"video_group[0]": {
        "src_in_ms": 900, "src_out_ms": 1100,
        "salience": {"peak_ms": 950, "score": 0.8, "kind": "point", "span_ms": None},
        "density": 0.6,
    }}
    records = post.assemble_cut_records(p2, {"f1": _lattice()}, {"f1": _motion()}, {},
                                        v4_meta_by_ref=v4_meta)
    rec = records[0]
    # The segmenter's own span wins over the (much wider) atom_ids bounding
    # box ([0,1] resolves to [800,2000) via _lattice()'s atoms).
    assert (rec.src_in_ms, rec.src_out_ms) == (900, 1100), rec
    assert rec.salience == {"peak_ms": 950, "score": 0.8, "kind": "point",
                            "span_ms": None, "shape": "after"}, rec.salience
    # density-scaled min_ms (post_params.V4_MIN_MS_FLOOR + density * V4_MIN_MS_DENSE_BONUS),
    # not the V3 anchor-span formula.
    from app.services.l3.post_params import V4_MIN_MS_DENSE_BONUS, V4_MIN_MS_FLOOR
    assert rec.pace.min_ms == round(V4_MIN_MS_FLOOR + 0.6 * V4_MIN_MS_DENSE_BONUS), rec.pace
    print("ok  test_v4_meta_by_ref_overrides_span_and_salience_and_stamps_shape")


def test_v3_video_cut_unaffected_when_v4_meta_by_ref_absent():
    """No v4_meta_by_ref (or a ref not present in it) -> byte-identical to
    today: span from atom membership, salience from post._salience, no
    density in the pace envelope."""
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0, 1],
                label="pan shot", summary="pans across the desk"),
    ])
    records = post.assemble_cut_records(p2, {"f1": _lattice()}, {"f1": _motion()}, {}, v4_meta_by_ref={})
    rec = records[0]
    assert (rec.src_in_ms, rec.src_out_ms) == (800, 2000), rec
    assert "kind" not in rec.salience, rec.salience
    print("ok  test_v3_video_cut_unaffected_when_v4_meta_by_ref_absent")


def test_density_scales_min_ms_sparse_vs_dense():
    from app.services.l3.post_params import V4_MIN_MS_DENSE_BONUS, V4_MIN_MS_FLOOR
    kwargs = dict(kind="video", s=0, e=10000, readability_ms=0, anchors=[],
                 action_energy=[0.1] * 100, hop_ms=100, next_cut_start_ms=10000,
                 max_tasteful_speed=2.0, min_tasteful_speed=0.5, natural_sound=False)
    sparse = post.compute_pace_envelope(density=0.0, **kwargs)
    dense = post.compute_pace_envelope(density=1.0, **kwargs)
    assert sparse.min_ms == V4_MIN_MS_FLOOR, sparse
    assert dense.min_ms == V4_MIN_MS_FLOOR + V4_MIN_MS_DENSE_BONUS, dense
    assert dense.min_ms > sparse.min_ms, "a dense span must hold more room than a sparse one"
    print("ok  test_density_scales_min_ms_sparse_vs_dense")


# --------------------------------------------------------------------------
# A/V coupling (av_coupling_authoritative.plan.md)
# --------------------------------------------------------------------------

def test_solo_cut_couples_to_its_own_file_at_zero_offset():
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 2),
                label="intro", summary="says hi"),
    ])
    records = post.assemble_cut_records(p2, {"f1": _lattice()}, {"f1": _motion()}, {})
    rec = records[0]
    assert rec.audio_file_id == "f1", rec.audio_file_id
    assert rec.audio_offset_ms == 0, rec.audio_offset_ms
    assert rec.audio_align_confidence is None, rec.audio_align_confidence
    print("ok  test_solo_cut_couples_to_its_own_file_at_zero_offset")


def test_synced_cut_couples_to_authoritative_file_with_refined_offset():
    # f1 (picture) is grouped with f2 (authoritative audio); the group's
    # globally-solved offset only gets to +100ms, but the two files' own
    # loudness envelopes carry a matching spike 200ms apart -- the local
    # cross-correlation refinement must find that true total offset.
    f1_rms = [-40.0] * 20
    f1_rms[3] = -5.0     # spike at 300-400ms, inside the resolved cut span [0,800)
    f2_rms = [-40.0] * 20
    f2_rms[5] = -5.0     # same spike, 200ms later
    audio_by_file = {
        "f1": {"rms_db": f1_rms, "hop_ms": 100},
        "f2": {"rms_db": f2_rms, "hop_ms": 100},
    }
    sync_info_by_file = {
        "f1": {"authoritative_audio_file_id": "f2",
              "members": {"f1": {"offset_ms": 100}, "f2": {"offset_ms": 0}}},
    }
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 2),
                label="intro", summary="says hi"),
    ])
    records = post.assemble_cut_records(
        p2, {"f1": _lattice()}, {"f1": _motion()}, {},
        audio_by_file=audio_by_file, sync_info_by_file=sync_info_by_file,
    )
    rec = records[0]
    assert rec.audio_file_id == "f2", rec.audio_file_id
    assert rec.audio_offset_ms == 200, rec.audio_offset_ms
    assert rec.audio_align_confidence is not None and rec.audio_align_confidence > 0.9, \
        rec.audio_align_confidence
    print("ok  test_synced_cut_couples_to_authoritative_file_with_refined_offset")


def test_synced_cut_falls_back_to_global_delta_with_no_authoritative_envelope():
    # The authoritative file was never independently ingested (e.g. no
    # transcript/audio_features row) -- refine_offset's guard must keep the
    # unrefined global delta rather than raise or fabricate a lag.
    sync_info_by_file = {
        "f1": {"authoritative_audio_file_id": "f2",
              "members": {"f1": {"offset_ms": 250}, "f2": {"offset_ms": 0}}},
    }
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 2),
                label="intro", summary="says hi"),
    ])
    records = post.assemble_cut_records(
        p2, {"f1": _lattice()}, {"f1": _motion()}, {}, sync_info_by_file=sync_info_by_file,
    )
    rec = records[0]
    assert rec.audio_file_id == "f2", rec.audio_file_id
    assert rec.audio_offset_ms == 250, rec.audio_offset_ms
    assert rec.audio_align_confidence is None, rec.audio_align_confidence
    print("ok  test_synced_cut_falls_back_to_global_delta_with_no_authoritative_envelope")


# --------------------------------------------------------------------------
# continuity (cuts_v3_continuity.plan.md)
# --------------------------------------------------------------------------

def test_continuity_numbers_all_cuts_including_junk():
    """cut_no/of count over ALL cuts on the clip, junk included -- a gap in the
    numbering IS the signal a junk beat sits there once junk is filtered out
    downstream."""
    atoms = [Atom(atom_id=0, file_id="f1", start_ms=1000, end_ms=1500, state_in="clip_edge",
                  state_out="clip_edge", action_energy=0.1, coherence=0.9)]
    lat = Lattice(file_id="f1", duration_ms=3000, words=[], turns=[], hints=[], atoms=atoms)
    motion = {"hop_ms": 100, "blur": [0.5] * 30, "action_energy": [0.1] * 30, "action_points": []}
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0],
                label="a", summary="a"),
    ])
    # A second, separate clip's cuts must never leak into f1's numbering.
    atoms2 = [Atom(atom_id=1, file_id="f2", start_ms=0, end_ms=500, state_in="clip_edge",
                   state_out="clip_edge", action_energy=0.1, coherence=0.9)]
    lat2 = Lattice(file_id="f2", duration_ms=500, words=[], turns=[], hints=[], atoms=atoms2)
    p2.cuts.append(Pass2Cut(source_ref="video_group[1]", kind="video", file_id="f2",
                            atom_ids=[1], label="b", summary="b"))
    records = post.assemble_cut_records(p2, {"f1": lat, "f2": lat2},
                                        {"f1": motion, "f2": motion}, {})
    f1 = next(r for r in records if r.file_id == "f1")
    f2 = next(r for r in records if r.file_id == "f2")
    assert f1.continuity["cut_no"] == 1 and f1.continuity["of"] == 1, f1.continuity
    assert f2.continuity["cut_no"] == 1 and f2.continuity["of"] == 1, f2.continuity
    print("ok  test_continuity_numbers_all_cuts_including_junk")


def test_continuity_hard_break_on_shot_cut_boundary():
    """A break-type atom boundary (shot_cut) exactly at the shared edge between
    two adjacent cuts makes the seam HARD, even though nothing else about it
    looks like a break (same speaker, zero gap)."""
    atoms = [
        Atom(atom_id=0, file_id="f1", start_ms=0, end_ms=1000, state_in="clip_edge",
             state_out="shot_cut", action_energy=0.1, coherence=0.9),
        Atom(atom_id=1, file_id="f1", start_ms=1000, end_ms=2000, state_in="shot_cut",
             state_out="clip_edge", action_energy=0.1, coherence=0.9),
    ]
    lat = Lattice(file_id="f1", duration_ms=2000, words=[], turns=[], hints=[], atoms=atoms)
    motion = {"hop_ms": 100, "blur": [0.5] * 20, "action_energy": [0.1] * 20, "action_points": []}
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0],
                label="a", summary="a"),
        Pass2Cut(source_ref="video_group[1]", kind="video", file_id="f1", atom_ids=[1],
                label="b", summary="b"),
    ])
    records = post.assemble_cut_records(p2, {"f1": lat}, {"f1": motion}, {})
    first, second = records
    assert first.continuity["next_contiguous"] is False, first.continuity
    assert "shot" in first.continuity["seam_reason_next"], first.continuity
    assert second.continuity["prev_contiguous"] is False, second.continuity
    assert first.continuity["seam_reason_next"] == second.continuity["seam_reason_prev"]
    print("ok  test_continuity_hard_break_on_shot_cut_boundary")


def test_continuity_hard_break_on_speaker_change():
    words = [
        {"start_ms": 0, "end_ms": 500, "text": "hello", "speaker": "S0"},
        {"start_ms": 500, "end_ms": 1000, "text": "hi", "speaker": "S1"},
    ]
    lat = Lattice(file_id="f1", duration_ms=1000, words=words, turns=[], hints=[], atoms=[])
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 0),
                label="a", summary="a", voice_ids=["V0"]),
        Pass2Cut(source_ref="speech_cut[1]", kind="speech", file_id="f1", word_span=(1, 1),
                label="b", summary="b", voice_ids=["V1"]),
    ])
    records = post.assemble_cut_records(p2, {"f1": lat}, {"f1": {}}, {})
    first, second = records
    assert first.continuity["next_contiguous"] is False, first.continuity
    assert "speaker" in first.continuity["seam_reason_next"], first.continuity
    print("ok  test_continuity_hard_break_on_speaker_change")


def test_continuity_flagged_junk_suspect_in_the_gap_is_hard():
    """A pass-1 junk suspect (e.g. a camera cue) sitting in the dropped
    connective tissue between two cuts hard-splits the seam, even with no
    atom-level break and the same speaker either side."""
    words = [
        {"start_ms": 0, "end_ms": 500, "text": "hello", "speaker": "S0"},
        {"start_ms": 600, "end_ms": 900, "text": "cut", "speaker": "S0"},   # the cue itself
        {"start_ms": 1000, "end_ms": 1500, "text": "again", "speaker": "S0"},
    ]
    lat = Lattice(file_id="f1", duration_ms=1500, words=words, turns=[], hints=[], atoms=[])
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 0),
                label="a", summary="a", voice_ids=["V0"]),
        Pass2Cut(source_ref="speech_cut[1]", kind="speech", file_id="f1", word_span=(2, 2),
                label="b", summary="b", voice_ids=["V0"]),
    ])
    suspects = [JunkSuspect(file_id="f1", word_span=(1, 1), reason="camera cue")]
    records = post.assemble_cut_records(p2, {"f1": lat}, {"f1": {}}, {}, junk_suspects=suspects)
    first, second = records
    assert first.continuity["next_contiguous"] is False, first.continuity
    assert "flagged" in first.continuity["seam_reason_next"], first.continuity
    print("ok  test_continuity_flagged_junk_suspect_in_the_gap_is_hard")


def test_continuity_first_and_last_edges_are_false():
    atoms = [
        Atom(atom_id=0, file_id="f1", start_ms=0, end_ms=1000, state_in="clip_edge",
             state_out="energy_shift", action_energy=0.1, coherence=0.9),
        Atom(atom_id=1, file_id="f1", start_ms=1000, end_ms=2000, state_in="energy_shift",
             state_out="clip_edge", action_energy=0.1, coherence=0.9),
    ]
    lat = Lattice(file_id="f1", duration_ms=2000, words=[], turns=[], hints=[], atoms=atoms)
    motion = {"hop_ms": 100, "blur": [0.5] * 20, "action_energy": [0.1] * 20, "action_points": []}
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0],
                label="a", summary="a"),
        Pass2Cut(source_ref="video_group[1]", kind="video", file_id="f1", atom_ids=[1],
                label="b", summary="b"),
    ])
    records = post.assemble_cut_records(p2, {"f1": lat}, {"f1": motion}, {})
    first, second = records
    assert first.continuity["prev_contiguous"] is False and first.continuity["seam_reason_prev"] is None
    assert second.continuity["next_contiguous"] is False and second.continuity["seam_reason_next"] is None
    # An energy-regime edge (not a break-type reason) is continuous footage.
    assert first.continuity["next_contiguous"] is True, first.continuity
    print("ok  test_continuity_first_and_last_edges_are_false")


def test_junk_flag_is_preserved_verbatim():
    # Deterministic-keep: post no longer second-guesses the model's semantic
    # junk call with a hardcoded energy/anchor threshold. Whatever the model
    # (or pass-1 suspects) decided is carried through verbatim -- junk is a
    # recoverable label shown in the Discarded tray, never a code override.
    atoms = [Atom(atom_id=0, file_id="f1", start_ms=0, end_ms=2000, state_in="clip_edge",
                  state_out="clip_edge", action_energy=0.6, coherence=0.5)]
    lat = Lattice(file_id="f1", duration_ms=2000, words=[], turns=[], hints=[], atoms=atoms)
    motion = {"hop_ms": 100, "blur": [0.5] * 20, "action_energy": [0.6] * 20,
              "action_points": [{"ts_ms": 500}, {"ts_ms": 1200}, {"ts_ms": 1600}]}
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0],
                 label="cue", summary="and go", junk=True, junk_reason="production cue"),
    ])
    rec = post.assemble_cut_records(p2, {"f1": lat}, {"f1": motion}, {})[0]
    assert rec.junk is True and rec.junk_reason == "production cue", rec
    print("ok  test_junk_flag_is_preserved_verbatim")


def test_non_junk_stays_non_junk():
    atoms = [Atom(atom_id=0, file_id="f1", start_ms=0, end_ms=2000, state_in="clip_edge",
                  state_out="clip_edge", action_energy=0.05, coherence=0.95)]
    lat = Lattice(file_id="f1", duration_ms=2000, words=[], turns=[], hints=[], atoms=atoms)
    motion = {"hop_ms": 100, "blur": [0.5] * 20, "action_energy": [0.05] * 20, "action_points": []}
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0],
                 label="hold", summary="a quiet hold", junk=False),
    ])
    rec = post.assemble_cut_records(p2, {"f1": lat}, {"f1": motion}, {})[0]
    assert rec.junk is False, rec
    print("ok  test_non_junk_stays_non_junk")


def test_assemble_raises_on_unknown_file_id():
    p2 = Pass2Output(cuts=[Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="missing",
                                    word_span=(0, 1), label="x", summary="y")])
    try:
        post.assemble_cut_records(p2, {}, {}, {})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "unknown file_id" in str(e)
    print("ok  test_assemble_raises_on_unknown_file_id")


def test_assemble_raises_on_unresolvable_atom_ids():
    p2 = Pass2Output(cuts=[Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1",
                                    atom_ids=[99], label="x", summary="y")])
    try:
        post.assemble_cut_records(p2, {"f1": _lattice()}, {"f1": _motion()}, {})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "no resolvable atoms" in str(e)
    print("ok  test_assemble_raises_on_unresolvable_atom_ids")


def test_assemble_raises_on_overlap_between_cuts():
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0],
                label="a", summary="a"),
        Pass2Cut(source_ref="video_group[1]", kind="video", file_id="f1", atom_ids=[0, 1],
                label="b", summary="b"),   # overlaps the first (both include atom 0)
    ])
    try:
        post.assemble_cut_records(p2, {"f1": _lattice()}, {"f1": _motion()}, {})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "overlap" in str(e) or "gap" in str(e)
    print("ok  test_assemble_raises_on_overlap_between_cuts")


def test_assemble_allows_a_project_file_with_zero_cuts():
    # boundaries-v2: two files in the project, pass 2 reported cuts for only
    # one. An all-junk / all-dead-air clip contributing nothing is now a valid
    # outcome (a warning, not a raise); the run still succeeds with f1's cuts.
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0, 1],
                label="a", summary="a"),
    ])
    lattices = {"f1": _lattice(), "f2": _lattice()}
    motion = {"f1": _motion(), "f2": _motion()}
    records = post.assemble_cut_records(p2, lattices, motion, {})
    assert records, "expected f1's cut to still be assembled"
    assert all(r.file_id == "f1" for r in records), [r.file_id for r in records]
    print("ok  test_assemble_allows_a_project_file_with_zero_cuts")


def test_remove_spans_shaves_edges():
    words = [
        {"start_ms": 0, "end_ms": 300, "text": "Um,"},
        {"start_ms": 300, "end_ms": 600, "text": "so"},
        {"start_ms": 600, "end_ms": 1200, "text": "the"},
        {"start_ms": 1200, "end_ms": 1800, "text": "product"},
        {"start_ms": 1800, "end_ms": 2200, "text": "ships."},
        {"start_ms": 2200, "end_ms": 2500, "text": "you"},
        {"start_ms": 2500, "end_ms": 2800, "text": "know"},
    ]
    spans = post.compute_speech_remove_spans(words, (0, 6), 0, 2800)
    # leading "Um, so" + silence -> [0, 600]; trailing "you know" -> [2200, 2800]
    assert (0, 600) in spans, spans
    assert (2200, 2800) in spans, spans
    print("ok  test_remove_spans_shaves_edges")


def test_remove_spans_interior_filler_and_pause():
    # contiguous words (100ms gaps as rhythm), one interior "um", one long pause
    words = [
        {"start_ms": 0, "end_ms": 400, "text": "The"},
        {"start_ms": 500, "end_ms": 900, "text": "ball"},          # gap 100 (rhythm)
        {"start_ms": 1000, "end_ms": 1300, "text": "um"},          # interior filler
        {"start_ms": 1400, "end_ms": 1800, "text": "flies"},       # gap 100
        {"start_ms": 3800, "end_ms": 4200, "text": "far."},        # gap 2000 -> big pause
    ]
    spans = post.compute_speech_remove_spans(words, (0, 4), 0, 4200)
    # interior filler "um" removed
    assert (1000, 1300) in spans, spans
    # a chunk of the 2s pause between "flies" and "far." is removable (excess
    # over the ~100ms median rhythm), taken from the middle
    assert any(1800 < a and b < 3800 and (b - a) > 1000 for a, b in spans), spans
    print("ok  test_remove_spans_interior_filler_and_pause")


def test_remove_spans_none_when_clean_and_tight():
    words = [{"start_ms": 0, "end_ms": 500, "text": "Hello"},
             {"start_ms": 500, "end_ms": 1000, "text": "world."}]
    assert post.compute_speech_remove_spans(words, (0, 1), 0, 1000) == []
    print("ok  test_remove_spans_none_when_clean_and_tight")


def test_remove_spans_all_filler_left_whole():
    words = [{"start_ms": 0, "end_ms": 300, "text": "um"},
             {"start_ms": 300, "end_ms": 600, "text": "uh"}]
    assert post.compute_speech_remove_spans(words, (0, 1), 0, 600) == []
    print("ok  test_remove_spans_all_filler_left_whole")


def test_speech_quality_fluency_and_loudness():
    # A clean, tight beat with nothing removable and no audio -> pure fluency 1.
    assert post.compute_speech_quality([], 0, 0, 1000, 0, None, None) == 1.0
    # Half the span removable (dead air/fillers) halves fluency.
    assert post.compute_speech_quality([], 0, 0, 1000, 500, None, None) == 0.5
    # Loudness blends in, normalised against the clip's own rms range: a -30dB
    # cut sits halfway in a [-40,-20] clip -> loudness 0.5, fluency 1.0 -> 0.75.
    rms = [-30.0] * 12  # hop 100ms across the [0,1000) window
    q = post.compute_speech_quality(rms, 100, 0, 1000, 0, -40.0, -20.0)
    assert abs(q - 0.75) < 1e-6, q
    print("ok  test_speech_quality_fluency_and_loudness")


def test_visual_score_prefers_oncam_closeup():
    hi = post.compute_visual_score(
        True, {"shot_size": "close_up"}, {"graded": True, "exposure_flags": []},
        [], 0, 0, 1000, None, None)
    lo = post.compute_visual_score(
        False, {"shot_size": "wide"}, {"graded": False, "exposure_flags": []},
        [], 0, 0, 1000, None, None)
    assert hi is not None and lo is not None and hi > lo, (hi, lo)
    # Nothing visual known at all -> None (so total_quality can fall back).
    assert post.compute_visual_score(None, {}, {}, [], 0, 0, 1000, None, None) is None
    print("ok  test_visual_score_prefers_oncam_closeup")


def test_total_quality_blends_speech_and_video_is_visual_only():
    # Speech blends the two equally.
    assert post.compute_total_quality("speech", 0.8, 0.4) == 0.6
    # A speech cut with no visual falls back to speech alone.
    assert post.compute_total_quality("speech", 0.8, None) == 0.8
    # Video is visual-only (speech_quality is None for it).
    assert post.compute_total_quality("video", None, 0.5) == 0.5
    print("ok  test_total_quality_blends_speech_and_video_is_visual_only")


def _tg_rec(fid, s, e, role, tq):
    return post.CutRecord(
        file_id=fid, src_in_ms=s, src_out_ms=e, kind="speech", word_span=(0, 4),
        atom_ids=None, label="line", summary="", on_camera=None,
        junk=False, junk_reason=None, framing={}, look={}, caption_zones=[],
        hero_ts_ms=s, pace=None, take_group_id="tg1", take_role=role, channel="said",
        speech_quality=None, total_quality=tq)


def test_take_winner_is_highest_total_quality():
    # Same-setting cluster (two takes) -> highest total_quality wins regardless
    # of length; the outlook angle is never touched, never a winner.
    recs = [_tg_rec("f1", 0, 3000, "winner", 0.4), _tg_rec("f2", 0, 2000, "take", 0.8),
            _tg_rec("f3", 0, 2500, "outlook", 0.9)]
    post._enforce_take_winner(recs)
    roles = {r.file_id: r.take_role for r in recs}
    assert roles == {"f1": "take", "f2": "winner", "f3": "outlook"}, roles
    print("ok  test_take_winner_is_highest_total_quality")


def test_no_winner_for_outlook_only_group():
    # A podcast beat filmed by 3 cameras: pass 2 crowned one "winner" + 2
    # "outlook", but there's no same-setting RETRY -> no winner is named; the
    # lone same-setting member joins the peer angles as an outlook.
    recs = [_tg_rec("cam1", 0, 3000, "winner", 0.9), _tg_rec("cam2", 0, 3000, "outlook", 0.7),
            _tg_rec("cam3", 0, 3000, "outlook", 0.5)]
    post._enforce_take_winner(recs)
    roles = {r.file_id: r.take_role for r in recs}
    assert roles == {"cam1": "outlook", "cam2": "outlook", "cam3": "outlook"}, roles
    assert not any(r.take_role == "winner" for r in recs)
    print("ok  test_no_winner_for_outlook_only_group")


def main():
    test_remove_spans_shaves_edges()
    test_remove_spans_interior_filler_and_pause()
    test_remove_spans_none_when_clean_and_tight()
    test_remove_spans_all_filler_left_whole()
    test_speech_quality_fluency_and_loudness()
    test_visual_score_prefers_oncam_closeup()
    test_total_quality_blends_speech_and_video_is_visual_only()
    test_take_winner_is_highest_total_quality()
    test_no_winner_for_outlook_only_group()
    test_hero_ts_prefers_anchor_over_sharp()
    test_hero_ts_falls_back_to_sharpest()
    test_hero_ts_falls_back_to_midpoint_with_no_blur()
    test_speech_pace_envelope_is_native_speed_only()
    test_video_pace_min_ms_from_anchor_span()
    test_video_pace_max_ms_extends_through_flatline()
    test_video_pace_max_ms_stops_at_departure_from_flatline()
    test_pace_levels_partial_saturation_repeats_nearest_reachable()
    test_pace_levels_zero_intrinsic_velocity_maxes_every_level()
    test_energy_grade_bands()
    test_salience_no_signal_falls_back_to_hero_ts()
    test_salience_peaks_at_the_loudest_action_energy_hop()
    test_salience_onset_bump_dominates_flat_energy()
    test_salience_onset_outside_span_is_ignored()
    test_salience_uses_rms_loudness_when_action_energy_flat()
    test_classify_camera_move()
    test_validate_no_overlap_passes_exact_coverage()
    test_validate_no_overlap_allows_start_gap()
    test_validate_no_overlap_allows_end_gap()
    test_validate_no_overlap_raises_on_overlap()
    test_validate_no_overlap_allows_middle_gap()
    test_validate_no_overlap_allows_no_cuts()
    test_assemble_cut_records_end_to_end()
    test_v4_meta_by_ref_overrides_span_and_salience_and_stamps_shape()
    test_v3_video_cut_unaffected_when_v4_meta_by_ref_absent()
    test_density_scales_min_ms_sparse_vs_dense()
    test_solo_cut_couples_to_its_own_file_at_zero_offset()
    test_synced_cut_couples_to_authoritative_file_with_refined_offset()
    test_synced_cut_falls_back_to_global_delta_with_no_authoritative_envelope()
    test_junk_flag_is_preserved_verbatim()
    test_non_junk_stays_non_junk()
    test_assemble_raises_on_unknown_file_id()
    test_assemble_raises_on_unresolvable_atom_ids()
    test_assemble_raises_on_overlap_between_cuts()
    test_assemble_allows_a_project_file_with_zero_cuts()
    test_continuity_numbers_all_cuts_including_junk()
    test_continuity_hard_break_on_shot_cut_boundary()
    test_continuity_hard_break_on_speaker_change()
    test_continuity_flagged_junk_suspect_in_the_gap_is_hard()
    test_continuity_first_and_last_edges_are_false()
    print("\nall post tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
