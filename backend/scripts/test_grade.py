#!/usr/bin/env python3
"""Tests for the color_grading_upgrade.plan.md Phase 1 stack -- tone/working
space, span measurement, percentile correct, sequence match, the v1 job, and
layers.py's read-path. No DB / ffmpeg / R2: DB-touching functions in
grade/job.py and grade/measure_span.py are exercised via mock.patch on their
I/O helpers (mirrors test_tools_loop.py's scripted-fake pattern), never a
live connection -- consistent with the rest of this test suite's "no DB"
convention (this module has ZERO prior test coverage, so this file is also
the first).

Run:  .venv/bin/python scripts/test_grade.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import layers  # noqa: E402
from app.services.l3.grade import job as grade_job  # noqa: E402
from app.services.l3.grade import tone  # noqa: E402
from app.services.l3.grade.cdl import Grade, apply_cdl, identity_grade_json  # noqa: E402
from app.services.l3.grade.correct import solve_correct_grade  # noqa: E402
from app.services.l3.grade.lut_bake import bake_cube_text  # noqa: E402
from app.services.l3.grade.match import ShotStats, group_neighbors, solve_sequence_match  # noqa: E402
from app.services.l3.grade.resolver import resolve_clip_grade  # noqa: E402


# --------------------------------------------------------------------------
# Step 1.1: tone.py (working space + filmic shoulder)
# --------------------------------------------------------------------------

def test_tone_legacy_is_exact_identity():
    import numpy as np
    x = np.array([0.0, 0.18, 0.5, 0.8, 1.0], dtype=np.float32)
    assert np.array_equal(tone.to_working(x, "rec709_legacy"), x)
    assert np.array_equal(tone.from_working(x, "rec709_legacy"), x)
    assert np.array_equal(tone.to_working(x, "rec709"), x)   # any non-v1 value -> identity
    print("ok  tone: legacy/unrecognized working_space is exact identity")


def test_tone_v1_black_stays_black():
    import numpy as np
    lin = tone.to_working(np.array([0.0], dtype=np.float32), tone.WORKING_SPACE_V1)
    out = tone.from_working(lin, tone.WORKING_SPACE_V1)
    assert abs(float(out[0])) < 1e-5, out
    print("ok  tone: v1 black point stays exactly black")


def test_tone_v1_never_exceeds_one():
    import numpy as np
    for x in (0.0, 0.5, 0.8, 1.0, 3.0, 50.0):
        out = tone.from_working(np.array([x], dtype=np.float32), tone.WORKING_SPACE_V1)
        assert out[0] <= 1.0 + 1e-6, (x, out)
    print("ok  tone: v1 shoulder never exceeds 1.0 regardless of input")


def test_tone_v1_midgray_barely_moves_shadows_untouched():
    """Below the shoulder, from_working(to_working(x)) round-trips to
    within float noise -- shadows/midtones are exact identity, only
    highlights compress. Catches the exact bug an HDR-calibrated curve
    (e.g. a naive Hable/Uncharted2 port) would introduce: a global darkening
    of everything, not just a highlight rolloff."""
    import numpy as np
    mid_display = np.array([0.46], dtype=np.float32)
    lin = tone.to_working(mid_display, tone.WORKING_SPACE_V1)
    out = tone.from_working(lin, tone.WORKING_SPACE_V1)
    assert abs(float(out[0] - mid_display[0])) < 0.01, (mid_display, out)
    print("ok  tone: v1 midtones/shadows are untouched (only highlights compress)")


def test_tone_v1_monotonic():
    import numpy as np
    sweep = np.linspace(0, 1, 200).astype(np.float32)
    lin = tone.to_working(sweep, tone.WORKING_SPACE_V1)
    out = tone.from_working(lin, tone.WORKING_SPACE_V1)
    assert bool(np.all(np.diff(out) >= -1e-6))
    print("ok  tone: v1 curve is monotonically non-decreasing")


# --------------------------------------------------------------------------
# Step 1.1: lut_bake.py -- direct-compute vs baked-cube parity, legacy parity
# --------------------------------------------------------------------------

def test_lut_bake_legacy_unaffected_by_working_space_param():
    """Adding the `working_space` param to bake_cube_text must not change
    ANY existing caller's output -- the default ("rec709") is legacy/identity,
    so baking with no working_space arg at all is byte-identical to baking
    with working_space="rec709" explicitly, and both equal a direct
    apply_cdl (the pre-Step-1.1 behavior exactly)."""
    grade = Grade(slope=(1.1, 1.0, 0.95), offset=(0.01, 0.0, -0.01))
    default_bake = bake_cube_text(grade, size=5)
    explicit_legacy_bake = bake_cube_text(grade, size=5, working_space="rec709")
    assert default_bake == explicit_legacy_bake, "legacy default must match explicit legacy"
    print("ok  lut_bake: legacy default is unaffected by the new working_space param")


def test_lut_bake_v1_parity_direct_vs_baked_cube():
    """Step 1.1 §3's acceptance test: sampling the SAME RGB through
    apply_cdl+tone directly must closely match trilinearly sampling the
    baked cube, within tolerance."""
    import numpy as np
    from app.services.l3.grade.lut_bake import _sample_lut_trilinear, parse_cube_text

    grade = Grade(slope=(1.15, 1.05, 0.9), offset=(0.02, 0.0, -0.01))
    size = 33
    cube_text = bake_cube_text(grade, size=size, working_space=tone.WORKING_SPACE_V1)
    grid, parsed_size = parse_cube_text(cube_text)
    assert parsed_size == size

    probes = np.array([
        [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5],
        [0.18, 0.18, 0.18], [0.9, 0.4, 0.2], [0.05, 0.6, 0.95],
    ], dtype=np.float32)
    direct = tone.from_working(apply_cdl(tone.to_working(probes, tone.WORKING_SPACE_V1), grade),
                               tone.WORKING_SPACE_V1)
    sampled = _sample_lut_trilinear(grid, probes)
    max_err = float(np.max(np.abs(direct - sampled)))
    assert max_err < 0.02, f"direct-vs-baked-cube parity exceeded tolerance: {max_err}"
    print(f"ok  lut_bake: v1 direct-compute vs baked-cube parity (max err {max_err:.4f})")


def test_lut_bake_v1_differs_from_legacy_for_same_grade():
    """Sanity: v1's working-space wrapper must actually DO something --
    baking the SAME grade under legacy vs v1 must NOT produce identical
    bytes (otherwise Step 1.1 shipped a no-op)."""
    grade = Grade()  # even the identity grade should differ (the tone curve alone)
    legacy = bake_cube_text(grade, size=9, working_space="rec709")
    v1 = bake_cube_text(grade, size=9, working_space=tone.WORKING_SPACE_V1)
    assert legacy != v1
    print("ok  lut_bake: v1 working space actually changes the baked cube vs legacy")


# --------------------------------------------------------------------------
# Step 1.3: correct.py percentile-based v1 levels
# --------------------------------------------------------------------------

def _cs(**kw):
    base = {"black_point": 0.02, "white_point": 0.97, "mid_gray": 0.5,
           "clip_shadow_pct": 0.0, "clip_highlight_pct": 0.0,
           "wb_gray_world": [1.0, 1.0, 1.0], "wb_white_patch": [1.0, 1.0, 1.0]}
    base.update(kw)
    return base


def test_correct_legacy_untouched_by_pipeline_param():
    cs = _cs(mid_gray=0.28)
    default_call = solve_correct_grade(cs)
    explicit_legacy = solve_correct_grade(cs, pipeline="legacy")
    assert default_call == explicit_legacy
    print("ok  correct: legacy path unaffected by the new pipeline param")


def test_correct_v1_nudges_mid_gray_toward_target_bounded():
    cs = _cs(mid_gray=0.36)   # within the clamp's reach of TARGET_MID_GRAY (0.42)
    g = solve_correct_grade(cs, pipeline="v1")
    projected = 0.36 * g.slope[0] + g.offset[0]
    assert abs(projected - 0.42) < 0.01, projected
    print("ok  correct: v1 nudges mid-gray toward target (small gap -> lands close)")


def test_correct_v1_never_worse_on_already_correct_footage():
    cs = _cs(mid_gray=0.42)
    g = solve_correct_grade(cs, pipeline="v1")
    assert abs(g.slope[0] - 1.0) < 0.05, g.slope
    print("ok  correct: v1 barely moves already-correctly-exposed footage")


# --------------------------------------------------------------------------
# Step 1.4: match.py solve_sequence_match (neighbor-only)
# --------------------------------------------------------------------------

def test_match_two_camera_interview_matches_across_the_cut():
    shots = [
        ShotStats(key="s0", file_id="camA",
                  stats={"black_point": 0.02, "white_point": 0.9, "mid_gray": 0.35,
                        "rgb_mean": [0.5, 0.48, 0.46]}, quality=0.5),
        ShotStats(key="s1", file_id="camB",
                  stats={"black_point": 0.05, "white_point": 0.8, "mid_gray": 0.3,
                        "rgb_mean": [0.46, 0.47, 0.5]}, quality=0.8),
    ]
    groups = group_neighbors(shots)
    assert groups == [[0, 1]], groups
    deltas = solve_sequence_match(shots)
    assert "s1" not in deltas   # s1 is the anchor (higher quality)
    assert "s0" in deltas and deltas["s0"].slope != (1.0, 1.0, 1.0)
    print("ok  match: a two-camera interview matches across the cut")


def test_match_never_groups_non_adjacent_shots():
    """Two RGB-identical but non-adjacent shots must never be dragged
    together -- the whole point of replacing global clustering."""
    a = {"black_point": 0.0, "white_point": 1.0, "mid_gray": 0.5, "rgb_mean": [0.9, 0.1, 0.1]}
    b = {"black_point": 0.0, "white_point": 1.0, "mid_gray": 0.5, "rgb_mean": [0.1, 0.9, 0.1]}
    shots = [
        ShotStats(key="a", file_id="f1", stats=a),
        ShotStats(key="b", file_id="f2", stats=b),
        ShotStats(key="c", file_id="f3", stats=a),   # identical to 'a', but NOT adjacent to it
    ]
    assert group_neighbors(shots) == [[0], [1], [2]]
    assert solve_sequence_match(shots) == {}
    print("ok  match: non-adjacent (even identical) shots are never grouped")


def test_match_same_file_always_groups_regardless_of_rgb():
    shots = [
        ShotStats(key="a", file_id="f1", stats={"black_point": 0.0, "white_point": 1.0,
                                                 "mid_gray": 0.5, "rgb_mean": [0.9, 0.1, 0.1]}, quality=1.0),
        ShotStats(key="b", file_id="f1", stats={"black_point": 0.0, "white_point": 1.0,
                                                 "mid_gray": 0.5, "rgb_mean": [0.1, 0.9, 0.1]}, quality=0.1),
    ]
    assert group_neighbors(shots) == [[0, 1]]
    deltas = solve_sequence_match(shots)
    assert "b" in deltas   # 'a' (higher quality) is anchor; 'b' gets nudged despite being far in RGB
    print("ok  match: adjacent same-file shots always group, regardless of RGB distance")


# --------------------------------------------------------------------------
# Step 1.5/1.1/1.3/1.7: resolver.py pipeline plumbing
# --------------------------------------------------------------------------

def test_resolver_legacy_default_working_space_unchanged():
    g = resolve_clip_grade({}, color_stats=None)
    assert g["working_space"] == "rec709", g["working_space"]
    print("ok  resolver: legacy default working_space is unchanged ('rec709')")


def test_resolver_v1_sets_v1_working_space():
    g = resolve_clip_grade({}, color_stats=None, pipeline="v1")
    assert g["working_space"] == tone.WORKING_SPACE_V1, g["working_space"]
    print("ok  resolver: pipeline='v1' selects the v1 working space")


def test_resolver_explicit_working_space_overrides_pipeline_default():
    g = resolve_clip_grade({"working_space": "custom_space"}, color_stats=None, pipeline="v1")
    assert g["working_space"] == "custom_space", g["working_space"]
    print("ok  resolver: an item's explicit working_space wins over the pipeline default")


def test_resolver_reference_transfer_v1_does_not_crash_or_blow_up():
    """Step 1.5: dropping a reference under v1 still composes without
    double-stretching into an extreme grade."""
    color_stats = _cs(rgb_mean=[0.3, 0.3, 0.3], rgb_std=[0.15, 0.15, 0.15])
    sequence_look = {
        "mode": "reference",
        "reference_stats": {"rgb_mean": [0.6, 0.5, 0.4], "rgb_std": [0.2, 0.2, 0.2]},
        "match_strength": 0.6,
    }
    g_dict = resolve_clip_grade({}, color_stats=color_stats, sequence_look=sequence_look, pipeline="v1")
    cdl = Grade.from_dict(g_dict["cdl"])
    for s in cdl.slope:
        assert 0.3 < s < 3.0, cdl.slope   # bounded, not blown out
    print("ok  resolver: v1 reference-transfer composes without blowing out")


def test_resolver_subject_box_seam_carries_through_no_visual_change_by_default():
    """Step 1.7: subject_box rides on soft_local end-to-end when a vignette
    is requested; with NO vignette requested (the common case), soft_local
    stays None regardless of subject_box -- no visual change yet."""
    g_no_vignette = resolve_clip_grade({"subject_box": [0.3, 0.2, 0.4, 0.4]}, color_stats=None)
    assert g_no_vignette["soft_local"] is None

    g_with_vignette = resolve_clip_grade(
        {"subject_box": [0.3, 0.2, 0.4, 0.4]}, color_stats=None,
        sequence_look={"vignette_strength": 0.3},
    )
    assert g_with_vignette["soft_local"]["subject_box"] == [0.3, 0.2, 0.4, 0.4]
    assert g_with_vignette["soft_local"]["vignette"]["cx"] == 0.5   # box center: 0.3+0.4/2
    print("ok  resolver: subject_box carries end-to-end (resolve->hash->bake seam), inert without a vignette")


# --------------------------------------------------------------------------
# Step 1.6: temporal stability invariant
# --------------------------------------------------------------------------

def test_one_grade_per_shot_no_intra_shot_variance():
    """Each timeline segment resolves to exactly ONE grade_hash across its
    whole duration -- formalizes the existing (by construction) invariant
    that a shot never varies its grade frame-to-frame."""
    doc = {"timeline": [
        {"seg_id": "s0", "file_id": "f1", "in_ms": 0, "out_ms": 4000},
        {"seg_id": "s1", "file_id": "f1", "in_ms": 4000, "out_ms": 9000},
    ], "operations": []}
    resolved = layers.resolve(doc, {}, {"f1": _cs(rgb_mean=[0.4, 0.5, 0.6])})
    hashes = [v.grade.get("grade_hash") for v in resolved.video_layers if v.kind == "spine"]
    assert len(hashes) == 2 and len(set(hashes)) <= 2   # one hash PER segment, stable across its span
    for v in resolved.video_layers:
        assert v.grade.get("grade_hash"), "every spine layer must resolve to exactly one grade"
    print("ok  temporal stability: one resolved grade per timeline segment, no intra-shot variance")


# --------------------------------------------------------------------------
# Step 1.0: layers.py's v1 read-path (grade_lookup)
# --------------------------------------------------------------------------

def _doc_one_seg():
    return {"timeline": [{"seg_id": "s0", "file_id": "f1", "in_ms": 0, "out_ms": 2000}], "operations": []}


def test_layers_legacy_default_is_byte_identical_to_before():
    """Calling resolve() with NO grade_pipeline/grade_lookup args (every
    existing caller) must produce the exact same grade resolve_clip_grade
    would inline -- the core 'legacy reproduces today's bytes' contract."""
    doc = _doc_one_seg()
    cs = {"f1": _cs(rgb_mean=[0.4, 0.5, 0.6])}
    resolved = layers.resolve(doc, {}, cs)
    expected = resolve_clip_grade(doc["timeline"][0], color_stats=cs["f1"], sequence_look=None, match_delta=None)
    assert resolved.video_layers[0].grade == expected
    print("ok  layers: default resolve() call is byte-identical to inline resolve_clip_grade")


def test_layers_v1_reads_grade_lookup_hit():
    doc = _doc_one_seg()
    fake_grade = identity_grade_json(tone.WORKING_SPACE_V1)
    fake_grade["cdl"] = Grade(slope=(1.3, 1.0, 1.0)).to_dict()   # a distinctive, obviously-not-computed value
    resolved = layers.resolve(doc, {}, {}, grade_pipeline="v1", grade_lookup={"s0": fake_grade})
    assert resolved.video_layers[0].grade == fake_grade
    print("ok  layers: v1 reads the pre-fetched grade_lookup hit verbatim (never recomputes)")


def test_layers_v1_falls_back_to_identity_on_miss():
    """A shot missing from grade_lookup (the job hasn't produced it yet)
    must render as identity, never an error and never a stale inline
    computation -- preview stays responsive while the job catches up."""
    doc = _doc_one_seg()
    cs = {"f1": _cs(rgb_mean=[0.9, 0.1, 0.1])}   # would normally correct heavily
    resolved = layers.resolve(doc, {}, cs, grade_pipeline="v1", grade_lookup={})
    cdl = Grade.from_dict(resolved.video_layers[0].grade["cdl"])
    assert cdl == Grade(), cdl   # identity -- color_stats is NOT used to compute anything under v1
    print("ok  layers: v1 falls back to identity (never computes inline) when grade_lookup misses")


# --------------------------------------------------------------------------
# Step 1.0: job.py -- compute_input_hash (pure) + ordered_shots
# --------------------------------------------------------------------------

def _grade_doc(spans, look=None):
    return {
        "timeline": [{"seg_id": f"s{i}", "file_id": fid, "in_ms": a, "out_ms": b}
                    for i, (fid, a, b) in enumerate(spans)],
        "operations": [], "look": look or {},
    }


def test_input_hash_stable_for_identical_documents():
    doc1 = _grade_doc([("f1", 0, 2000), ("f1", 2000, 5000)])
    doc2 = _grade_doc([("f1", 0, 2000), ("f1", 2000, 5000)])
    assert grade_job.compute_input_hash(doc1) == grade_job.compute_input_hash(doc2)
    print("ok  job: compute_input_hash is stable for identical documents")


def test_input_hash_changes_when_a_span_trims():
    """The user's explicit callout: input_hash MUST include timeline spans,
    not just the look -- trimming a cut changes both its own span stats and
    its neighbors' matching window."""
    doc_before = _grade_doc([("f1", 0, 2000), ("f1", 2000, 5000)])
    doc_trimmed = _grade_doc([("f1", 0, 1800), ("f1", 2000, 5000)])   # s0's out_ms trimmed
    assert grade_job.compute_input_hash(doc_before) != grade_job.compute_input_hash(doc_trimmed)
    print("ok  job: compute_input_hash changes when a cut's span trims (not just the look)")


def test_input_hash_changes_when_look_changes():
    doc_a = _grade_doc([("f1", 0, 2000)], look={"mode": "preset", "preset_id": "warm"})
    doc_b = _grade_doc([("f1", 0, 2000)], look={"mode": "preset", "preset_id": "cool"})
    assert grade_job.compute_input_hash(doc_a) != grade_job.compute_input_hash(doc_b)
    print("ok  job: compute_input_hash changes when the look changes")


def test_input_hash_unaffected_by_shot_reorder_being_a_real_change():
    """Order is semantically part of the hash (neighbor grouping depends on
    it) -- swapping two shots' order must also change the hash."""
    doc_a = _grade_doc([("f1", 0, 2000), ("f2", 0, 2000)])
    doc_b = _grade_doc([("f2", 0, 2000), ("f1", 0, 2000)])
    assert grade_job.compute_input_hash(doc_a) != grade_job.compute_input_hash(doc_b)
    print("ok  job: compute_input_hash reflects shot ORDER (neighbor grouping depends on it)")


def test_ordered_shots_covers_spine_and_place_video_ops_in_order():
    doc = {
        "timeline": [{"seg_id": "s0", "file_id": "f1", "in_ms": 0, "out_ms": 1000}],
        "operations": [
            {"type": "place_video", "op_id": "ov1", "source_file_id": "f2",
             "src_in_ms": 0, "src_out_ms": 500, "from_ms": 0, "to_ms": 500},
            {"type": "place_audio", "op_id": "pa1", "source_file_id": "f3"},  # not gradeable -- excluded
        ],
    }
    shots = grade_job.ordered_shots(doc)
    assert [s.key for s in shots] == ["s0", "ov1"], [s.key for s in shots]
    print("ok  job: ordered_shots covers spine segs + place_video ops, excludes place_audio")


# --------------------------------------------------------------------------
# Step 1.0: run_grade_job, fully mocked (no DB/ffmpeg/R2) -- exercises the
# real control flow: hash, measure, match, resolve, persist, cube-cache-by-hash.
# --------------------------------------------------------------------------

def test_run_grade_job_end_to_end_mocked():
    doc = {
        "timeline": [
            {"seg_id": "s0", "file_id": "f1", "in_ms": 0, "out_ms": 2000},
            {"seg_id": "s1", "file_id": "f1", "in_ms": 2000, "out_ms": 4000},
        ],
        "operations": [], "look": {},
    }
    upserted_rows = []
    status_calls = []

    def fake_measure_span(file_id, in_ms, out_ms, *, hero_ts_ms=None):
        return _cs(rgb_mean=[0.4, 0.5, 0.6], mid_gray=0.3)

    with mock.patch("app.services.l3.grade.job.get_job_state", return_value=None), \
         mock.patch("app.services.l3.grade.job._upsert_job_status",
                    side_effect=lambda tid, **kw: status_calls.append(kw)), \
         mock.patch("app.services.l3.grade.job._upsert_grade_row",
                    side_effect=lambda tid, key, h, gj, cube: upserted_rows.append((key, h, gj, cube))), \
         mock.patch("app.services.l3.grade.job.fetch_color_stats", return_value={}), \
         mock.patch("app.services.l3.grade.job.measure_span", side_effect=fake_measure_span), \
         mock.patch("app.services.l3.grade.job.ensure_cube_file", return_value="/tmp/fake.cube"), \
         mock.patch("app.services.l3.store.latest_document", return_value=(doc, 1), create=True):
        grade_job.run_grade_job("thread-1")

    assert len(upserted_rows) == 2, upserted_rows
    keys = {r[0] for r in upserted_rows}
    assert keys == {"s0", "s1"}, keys
    for _key, _h, gj, cube in upserted_rows:
        assert gj.get("grade_hash")
        assert cube == "/tmp/fake.cube"
    # progress must have advanced monotonically to the final total
    done_values = [c["done"] for c in status_calls if "done" in c]
    assert done_values == sorted(done_values), done_values
    assert done_values[-1] == 2, done_values
    states = [c["state"] for c in status_calls if "state" in c]
    assert states[0] == "grading" and states[-1] == "done", states
    print("ok  job: run_grade_job (mocked) grades every shot, advances progress, marks done")


def test_run_grade_job_records_error_never_crashes():
    doc = {"timeline": [{"seg_id": "s0", "file_id": "f1", "in_ms": 0, "out_ms": 1000}], "operations": []}
    status_calls = []
    with mock.patch("app.services.l3.grade.job.get_job_state", return_value=None), \
         mock.patch("app.services.l3.grade.job._upsert_job_status",
                    side_effect=lambda tid, **kw: status_calls.append(kw)), \
         mock.patch("app.services.l3.grade.job.fetch_color_stats", side_effect=RuntimeError("boom")), \
         mock.patch("app.services.l3.store.latest_document", return_value=(doc, 1), create=True):
        grade_job.run_grade_job("thread-2")   # must not raise
    errors = [c["error"] for c in status_calls if "error" in c]
    assert errors and "boom" in errors[-1], errors
    print("ok  job: run_grade_job records an error and never crashes the worker")


def test_run_grade_job_skips_when_already_done_for_current_hash():
    doc = {"timeline": [{"seg_id": "s0", "file_id": "f1", "in_ms": 0, "out_ms": 1000}], "operations": []}
    h = grade_job.compute_input_hash(doc)
    with mock.patch("app.services.l3.grade.job.get_job_state",
                    return_value={"state": "done", "input_hash": h}), \
         mock.patch("app.services.l3.grade.job._upsert_job_status") as upsert_status, \
         mock.patch("app.services.l3.store.latest_document", return_value=(doc, 1), create=True):
        grade_job.run_grade_job("thread-3")
    upsert_status.assert_not_called()
    print("ok  job: run_grade_job no-ops when already done for the current input_hash")


def main():
    test_tone_legacy_is_exact_identity()
    test_tone_v1_black_stays_black()
    test_tone_v1_never_exceeds_one()
    test_tone_v1_midgray_barely_moves_shadows_untouched()
    test_tone_v1_monotonic()
    test_lut_bake_legacy_unaffected_by_working_space_param()
    test_lut_bake_v1_parity_direct_vs_baked_cube()
    test_lut_bake_v1_differs_from_legacy_for_same_grade()
    test_correct_legacy_untouched_by_pipeline_param()
    test_correct_v1_nudges_mid_gray_toward_target_bounded()
    test_correct_v1_never_worse_on_already_correct_footage()
    test_match_two_camera_interview_matches_across_the_cut()
    test_match_never_groups_non_adjacent_shots()
    test_match_same_file_always_groups_regardless_of_rgb()
    test_resolver_legacy_default_working_space_unchanged()
    test_resolver_v1_sets_v1_working_space()
    test_resolver_explicit_working_space_overrides_pipeline_default()
    test_resolver_reference_transfer_v1_does_not_crash_or_blow_up()
    test_resolver_subject_box_seam_carries_through_no_visual_change_by_default()
    test_one_grade_per_shot_no_intra_shot_variance()
    test_layers_legacy_default_is_byte_identical_to_before()
    test_layers_v1_reads_grade_lookup_hit()
    test_layers_v1_falls_back_to_identity_on_miss()
    test_input_hash_stable_for_identical_documents()
    test_input_hash_changes_when_a_span_trims()
    test_input_hash_changes_when_look_changes()
    test_input_hash_unaffected_by_shot_reorder_being_a_real_change()
    test_ordered_shots_covers_spine_and_place_video_ops_in_order()
    test_run_grade_job_end_to_end_mocked()
    test_run_grade_job_records_error_never_crashes()
    test_run_grade_job_skips_when_already_done_for_current_hash()
    print("\nall grade tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
