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

import math
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
from app.services.l3.grade.cdl import Grade, apply_cdl, grade_hash, identity_grade_json  # noqa: E402
from app.services.l3.grade.colorspace import lab_to_srgb, srgb_to_lab  # noqa: E402
from app.services.l3.grade.correct import (  # noqa: E402
    SAT_BOOST_MAX, SKIN_LOCUS_DEG, TARGET_CHROMA, WB_MULTIPLIER_CLAMP,
    _skin_multiplier, _solve_wb, solve_correct_grade,
)
from app.services.l3.grade.leveling import (  # noqa: E402
    SILHOUETTE_RATIO, ShotLevelInput, solve_exposure_leveling, solve_leveling, solve_tonal_leveling,
)
from app.services.l3.grade.look_engine import (  # noqa: E402
    LOOKS, LookSpec, _apply_hue_rotate, _apply_hue_sat, _apply_split_tone, _rgb_to_hsv,
    build_look_grid, get_engine_look, list_engine_looks, resolve_look_spec,
)
from app.services.l3.grade.lut_bake import _identity_grid, _sample_lut_trilinear, bake_cube_text, parse_cube_text  # noqa: E402
from app.services.l3.grade.match import ShotStats, group_neighbors, solve_sequence_match  # noqa: E402
from app.services.l3.grade.measure_span import _measure_subject_lab, _measure_subject_luma  # noqa: E402
from app.services.l3.grade.presets import list_presets  # noqa: E402
from app.services.l3.grade.reference import GroupReference, compute_group_reference  # noqa: E402
from app.services.l3.grade.resolver import resolve_clip_grade  # noqa: E402
from app.services.l3.grade.scene_group import ShotSceneMeta, group_shots_semantically  # noqa: E402
from app.services.l3.grade.scene_meta import ShotCutMeta, lookup_shot_cut_meta  # noqa: E402
from app.services.l3.grade.softlocal import grain_ffmpeg_filter, halation_ffmpeg_subgraph  # noqa: E402
from app.services.render.compositor import _grade_key, _transform_vf  # noqa: E402


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
# color_phase1.plan.md Part 2: WORKING_SPACE_LOG_V1 -- golden byte-identical
# regression on rec709_v1 (the hard requirement) + the new log-decode curve
# --------------------------------------------------------------------------

def test_tone_v1_golden_byte_identical_after_log_v1_added():
    """HARD REQUIREMENT (color_phase1.plan.md 2.2): adding the log_v1 branch
    must not perturb rec709_v1 by even one bit. Recomputes the inverse sRGB
    EOTF / filmic-shoulder+OETF formulas independently (not by calling
    tone.py) against fixed probe values, so this test would catch a
    regression even if someone edited the v1 branch itself, not just the
    new log_v1 one."""
    import numpy as np
    probes = np.array([0.0, 0.04045, 0.18, 0.435, 0.8, 0.9, 1.0], dtype=np.float32)

    lo = probes / 12.92
    hi = np.power((probes + tone._SRGB_A) / (1.0 + tone._SRGB_A), 2.4)
    expected_lin = np.where(probes <= tone._SRGB_DISPLAY_THRESH, lo, hi).astype(np.float32)
    got_lin = tone.to_working(probes, tone.WORKING_SPACE_V1)
    assert np.allclose(got_lin, expected_lin, atol=1e-7), (got_lin, expected_lin)

    headroom = 1.0 - tone._SHOULDER_START
    over = np.clip(expected_lin - tone._SHOULDER_START, 0.0, None)
    toned = np.where(expected_lin <= tone._SHOULDER_START, expected_lin,
                      tone._SHOULDER_START + headroom * over / (headroom + over))
    toned = np.clip(toned, 0.0, 1.0)
    lo2 = toned * 12.92
    hi2 = (1.0 + tone._SRGB_A) * np.power(toned, 1.0 / 2.4) - tone._SRGB_A
    expected_display = np.where(toned <= tone._SRGB_LINEAR_THRESH, lo2, hi2).astype(np.float32)
    got_display = tone.from_working(got_lin, tone.WORKING_SPACE_V1)
    assert np.allclose(got_display, expected_display, atol=1e-7), (got_display, expected_display)
    print("ok  tone: rec709_v1 is byte-identical to its independently-recomputed formula")


def test_tone_log_v1_endpoints_pinned_and_bounded():
    import numpy as np
    x = np.array([0.0, 0.38, 1.0], dtype=np.float32)
    lin = tone.to_working(x, tone.WORKING_SPACE_LOG_V1)
    assert abs(float(lin[0])) < 1e-6, lin
    assert abs(float(lin[-1]) - 1.0) < 1e-6, lin
    assert bool(np.all(lin >= 0.0)) and bool(np.all(lin <= 1.0))
    print("ok  tone: log_v1 decode is endpoint-pinned (0->0, 1->1) and stays in [0,1]")


def test_tone_log_v1_midgray_decodes_near_true_scene_linear():
    """The whole point of the decode: a log profile packs the SAME
    scene-linear mid-gray (0.18) into a LOWER display code value (~0.32-0.42)
    than sRGB would (~0.461) -- log footage looks flatter/darker un-decoded,
    preserving highlight headroom. So decoding that code value must produce
    a HIGHER linear result than naively running it through the sRGB EOTF
    would (which under-shoots to ~0.12): the log curve is deliberately
    gentler (GAMMA=1.8 < sRGB's ~2.4) to correct for that compression."""
    import numpy as np
    x = np.array([0.38], dtype=np.float32)
    log_decoded = float(tone.to_working(x, tone.WORKING_SPACE_LOG_V1)[0])
    srgb_decoded = float(tone.to_working(x, tone.WORKING_SPACE_V1)[0])
    assert log_decoded > srgb_decoded, (log_decoded, srgb_decoded)
    assert abs(log_decoded - 0.18) < 0.05, log_decoded
    print(f"ok  tone: log_v1 mid-gray (0.38) decodes to {log_decoded:.3f}, near scene-linear 0.18")


def test_tone_log_v1_monotonic():
    import numpy as np
    sweep = np.linspace(0, 1, 200).astype(np.float32)
    lin = tone.to_working(sweep, tone.WORKING_SPACE_LOG_V1)
    assert bool(np.all(np.diff(lin) >= -1e-6))
    print("ok  tone: log_v1 decode curve is monotonically non-decreasing")


def test_tone_log_v1_from_working_matches_v1_reencode():
    """from_working must treat log_v1 identically to v1 -- same tonemap +
    re-encode to display Rec.709, only the INPUT decode differs."""
    import numpy as np
    lin = np.array([0.0, 0.18, 0.5, 0.9], dtype=np.float32)
    out_log = tone.from_working(lin, tone.WORKING_SPACE_LOG_V1)
    out_v1 = tone.from_working(lin, tone.WORKING_SPACE_V1)
    assert np.array_equal(out_log, out_v1), (out_log, out_v1)
    print("ok  tone: log_v1 from_working is identical to v1's (same re-encode, decode differs)")


def test_tone_log_v1_black_stays_black():
    import numpy as np
    lin = tone.to_working(np.array([0.0], dtype=np.float32), tone.WORKING_SPACE_LOG_V1)
    out = tone.from_working(lin, tone.WORKING_SPACE_LOG_V1)
    assert abs(float(out[0])) < 1e-5, out
    print("ok  tone: log_v1 black point stays exactly black")


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
    """v1 solves the levels CDL in WORKING space (the space the bake applies
    it in), so the target check must be the actual round-trip -- linearize the
    display mid, apply the CDL, re-encode -- not `mid*slope+offset` on display
    values (that mixed a display input with a working-space slope/offset and is
    exactly the dark-crush bug this fix removes)."""
    # 0.40 is within MID_GRAY_EXTRA_CLAMP's reach of TARGET_MID_GRAY (0.42) in
    # WORKING space (the ~1.11x linear nudge fits under the 1.2 clamp). The old
    # test used 0.36, which only fit under the clamp in the buggy display-space
    # math -- in the corrected linear space that gap needs ~1.38x and the clamp
    # (correctly) bounds it short, so 0.40 is the faithful "small gap" case.
    cs = _cs(mid_gray=0.40)
    g = solve_correct_grade(cs, pipeline="v1")
    projected = _roundtrip_v1(0.40, g)
    assert abs(projected - 0.42) < 0.02, projected
    print("ok  correct: v1 nudges mid-gray toward target (working-space round-trip lands close)")


def test_correct_v1_never_worse_on_already_correct_footage():
    cs = _cs(mid_gray=0.42)
    g = solve_correct_grade(cs, pipeline="v1")
    assert abs(g.slope[0] - 1.0) < 0.05, g.slope
    print("ok  correct: v1 barely moves already-correctly-exposed footage")


# --------------------------------------------------------------------------
# color_phase1.plan.md Part 2: LOG_FLAT_PRE_LIFT gated off under the real
# log decode (WORKING_SPACE_LOG_V1), kept as a fallback otherwise
# --------------------------------------------------------------------------

def test_correct_pre_lift_applies_by_default_for_is_log_flat():
    """Unchanged fallback behavior: no explicit working_space -> the
    pipeline-derived WORKING_SPACE_V1 is used, so the crude pre-lift still
    fires for an is_log_flat clip (pre-Part-2 behavior, byte-identical)."""
    # Modest stretch (not already pinned to LEVELS_SLOPE_MAX) so the pre-
    # lift's extra multiplier is actually visible in the result -- an
    # aggressive black/white gap saturates the ceiling either way and the
    # re-anchoring clamp erases the difference, which would be a false pass.
    cs = _cs(black_point=0.15, white_point=0.70, mid_gray=0.40, is_log_flat=True)
    with_lift = solve_correct_grade(cs, pipeline="v1")
    without_flag = solve_correct_grade({**cs, "is_log_flat": False}, pipeline="v1")
    assert with_lift.slope != without_flag.slope or with_lift.offset != without_flag.offset
    print("ok  correct: is_log_flat still applies the crude pre-lift when no explicit working_space is given")


def test_correct_pre_lift_gated_off_under_log_working_space():
    """The real fix: once WORKING_SPACE_LOG_V1 is explicitly selected (as
    resolver.py now does for an is_log_flat clip), the crude 1.06x pre-lift
    must NOT also fire -- the log decode already compensates; double-
    applying would over-lift. Compare the SAME color_stats solved under v1
    (pre-lift fires) vs explicitly under log_v1 (pre-lift gated off) -- they
    must differ, proving the gate actually took effect."""
    cs = _cs(black_point=0.15, white_point=0.70, mid_gray=0.40, is_log_flat=True)
    g_v1 = solve_correct_grade(cs, pipeline="v1", working_space=tone.WORKING_SPACE_V1)
    g_log = solve_correct_grade(cs, pipeline="v1", working_space=tone.WORKING_SPACE_LOG_V1)
    assert g_v1.slope != g_log.slope or g_v1.offset != g_log.offset, (g_v1, g_log)
    print("ok  correct: LOG_FLAT_PRE_LIFT is gated off when working_space is explicitly log_v1")


def test_correct_pre_lift_gate_is_a_no_op_for_non_log_footage():
    """The gate only touches the is_log_flat branch -- non-log footage
    solved under either working_space value (hypothetically) would still
    differ only because of the differing decode, never because of the
    pre-lift (which never fires without is_log_flat)."""
    cs = _cs(black_point=0.05, white_point=0.85, mid_gray=0.42, is_log_flat=False)
    g_default = solve_correct_grade(cs, pipeline="v1")
    g_explicit_v1 = solve_correct_grade(cs, pipeline="v1", working_space=tone.WORKING_SPACE_V1)
    assert g_default == g_explicit_v1, (g_default, g_explicit_v1)
    print("ok  correct: explicitly passing WORKING_SPACE_V1 reproduces the pipeline-derived default exactly")


def _roundtrip_v1(display_value, grade):
    """Push a DISPLAY scalar through the full v1 bake path for one channel:
    to_working -> apply_cdl(grade) -> from_working -> display."""
    import numpy as np
    lin = float(tone.to_working(np.array([display_value], dtype=np.float32), tone.WORKING_SPACE_V1)[0])
    rgb = np.full(3, lin, dtype=np.float32)
    return float(tone.from_working(apply_cdl(rgb, grade), tone.WORKING_SPACE_V1)[0])


def test_v1_grade_does_not_crush_midtones_or_shadows():
    """Regression for the "everything too dark" bug: the correct/match layers
    used to SOLVE their CDL in display space but the bake APPLIES it in linear
    working space, so a display-space negative black-offset (~-0.2) was
    subtracted from a linearized midtone (~0.07-0.21) -> zeroed shadows,
    halved midtones (real DB data: display mid 0.5 landed at 0.03-0.57).

    With the fix, resolve a v1 grade for a representative daylight clip, then
    push a display mid-gray (0.5) and a shadow (0.15) through the actual v1
    bake round-trip -- the mid must stay a plausible midtone and the shadow
    must NOT be crushed to black. (For this exact clip the OLD display-space
    solve crushed the 0.15 shadow to 0.0 and the fix lands it near ~0.15.)
    Thresholds are loose/generic on purpose."""
    cs = _cs(black_point=0.06, white_point=0.85, mid_gray=0.38)
    g_dict = resolve_clip_grade({}, color_stats=cs)
    cdl = Grade.from_dict(g_dict["cdl"])

    mid_out = _roundtrip_v1(0.5, cdl)
    shadow_out = _roundtrip_v1(0.15, cdl)
    assert 0.35 <= mid_out <= 0.6, f"mid 0.5 -> {mid_out} (crushed or blown)"
    assert shadow_out > 0.02, f"shadow 0.15 -> {shadow_out} (crushed to black)"
    print(f"ok  correct: v1 round-trip doesn't crush (mid 0.5 -> {mid_out:.3f}, shadow 0.15 -> {shadow_out:.3f})")


def test_v1_composite_slope_and_offset_are_bounded():
    """Fixes 2 & 3: the FINAL composed v1 CDL is clamped to a composite slope
    ceiling and a negative-offset floor, so stacked layers can't over-contrast
    or crush a nominal mid-gray to black regardless of how they combine."""
    from app.services.l3.grade.resolver import (
        COMPOSITE_MID_FLOOR, COMPOSITE_SLOPE_MAX, _clamp_composite_v1,
    )
    # A deliberately extreme composed grade (steeper + more negative than any
    # single layer would emit) must come back inside the composite bounds.
    hot = Grade(slope=(2.65, 2.65, 2.65), offset=(-0.4, -0.4, -0.4))
    clamped = _clamp_composite_v1(hot)
    assert all(s <= COMPOSITE_SLOPE_MAX + 1e-6 for s in clamped.slope), clamped.slope
    mid_lin = _roundtrip_mid_linear()
    for c in range(3):
        assert mid_lin * clamped.slope[c] + clamped.offset[c] >= COMPOSITE_MID_FLOOR - 1e-6, clamped.offset
    print(f"ok  resolver: composite slope ceiling ({COMPOSITE_SLOPE_MAX}) + offset floor bound the final CDL")


def _roundtrip_mid_linear():
    import numpy as np
    return float(tone.to_working(np.array([0.5], dtype=np.float32), tone.WORKING_SPACE_V1)[0])


def test_v1_composite_offset_floor_protects_a_modest_shadow_crush():
    """Follow-up fix to the composite guardrail bug: COMPOSITE_MID_FLOOR
    alone only protects the mid-gray ANCHOR point -- it does nothing for a
    genuine shadow below it, so a shot needing only a MODEST negative
    offset (nowhere near extreme enough to trip a mid-gray-only floor) can
    still crush real shadow detail to pure black. This is the exact
    real-world case observed live (a Siri-reel shot's Balance+Match delta,
    slope~0.99, offset~-0.022): mid-gray was already safely above the
    floor, so the OLD floor never engaged, and a display 0.15 shadow
    crushed to exactly 0."""
    from app.services.l3.grade.resolver import _clamp_composite_v1

    modest = Grade(slope=(0.995, 0.993, 0.927), offset=(-0.0223, -0.0223, -0.0223))
    clamped = _clamp_composite_v1(modest)
    shadow_out = _roundtrip_v1(0.15, clamped)
    assert shadow_out > 0.02, shadow_out
    # essentially unchanged from its own input -- the floor should barely
    # need to nudge this specific case, not visibly re-grade it.
    assert abs(shadow_out - 0.15) < 0.02, shadow_out
    print(f"ok  resolver: the shadow floor fixes a modest-offset crush that the "
         f"mid-gray-only floor missed (0.15 -> {shadow_out:.3f}, was 0.0)")


def test_v1_composite_offset_floor_does_not_raise_true_black():
    """The shadow floor must NOT lift the whole toe: a genuine near-black
    (well below the shadow probe) stays free to reach ~0 -- this is a
    shadow-DETAIL guard, not a black-point lift. Checked against both a
    realistic modest-offset grade and the existing extreme 'hot' fixture
    (whose slope gets clamped to COMPOSITE_SLOPE_MAX, the harder case for
    accidentally lifting blacks)."""
    from app.services.l3.grade.resolver import _clamp_composite_v1

    modest = Grade(slope=(0.995, 0.993, 0.927), offset=(-0.0223, -0.0223, -0.0223))
    black_out = _roundtrip_v1(0.05, _clamp_composite_v1(modest))
    # stays close to its OWN input (0.05), nowhere near the protected shadow
    # probe (0.15) -- the floor barely moves an already-modest offset, it
    # doesn't lift near-black toward shadow-floor territory.
    assert black_out < 0.10, black_out

    hot = Grade(slope=(2.65, 2.65, 2.65), offset=(-0.4, -0.4, -0.4))
    black_out_hot = _roundtrip_v1(0.05, _clamp_composite_v1(hot))
    assert black_out_hot < 0.01, black_out_hot
    print(f"ok  resolver: the shadow floor doesn't lift true black (modest -> "
         f"{black_out:.4f}, extreme-slope case -> {black_out_hot:.4f})")


def test_v1_composite_offset_floor_preserves_mid_gray_and_slope_ceiling():
    """Regression guard: the pre-existing composite protections (slope
    ceiling, mid-gray floor) are unchanged by adding the shadow floor."""
    from app.services.l3.grade.resolver import (
        COMPOSITE_MID_FLOOR, COMPOSITE_SLOPE_MAX, _clamp_composite_v1,
    )

    modest = Grade(slope=(0.995, 0.993, 0.927), offset=(-0.0223, -0.0223, -0.0223))
    mid_out = _roundtrip_v1(0.5, _clamp_composite_v1(modest))
    assert 0.46 <= mid_out <= 0.65, mid_out

    hot = Grade(slope=(2.65, 2.65, 2.65), offset=(-0.4, -0.4, -0.4))
    clamped = _clamp_composite_v1(hot)
    assert all(s <= COMPOSITE_SLOPE_MAX + 1e-6 for s in clamped.slope), clamped.slope
    mid_lin = _roundtrip_mid_linear()
    for c in range(3):
        assert mid_lin * clamped.slope[c] + clamped.offset[c] >= COMPOSITE_MID_FLOOR - 1e-6, clamped.offset
    print("ok  resolver: mid-gray floor and slope ceiling still hold with the shadow floor added")


def test_v1_composite_offset_floor_respects_power():
    """A per-channel power (e.g. from a manual override) is respected: the
    floor solves for the PRE-power value needed so the POST-power output
    still clears the shadow floor, not the pre-power value directly."""
    from app.services.l3.grade.resolver import _clamp_composite_v1

    graded = Grade(slope=(1.0, 1.0, 1.0), offset=(-0.5, -0.5, -0.5), power=(1.0, 1.5, 0.7))
    clamped = _clamp_composite_v1(graded)
    for c in range(3):
        single_channel = Grade(slope=(clamped.slope[c],) * 3, offset=(clamped.offset[c],) * 3,
                               power=(clamped.power[c],) * 3)
        shadow_out = _roundtrip_v1(0.15, single_channel)
        assert shadow_out > 0.02, (c, shadow_out)
    print("ok  resolver: the shadow floor respects per-channel power, not just slope/offset")


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
    ref = compute_group_reference([s.stats for s in shots])
    deltas = solve_sequence_match(shots, references={0: ref})
    # grade_pipeline_standardize.plan.md: no member is exempt as "the
    # anchor" -- every member converges toward the group's reference.
    assert "s0" in deltas and "s1" in deltas, deltas
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
    ref = compute_group_reference([s.stats for s in shots])
    deltas = solve_sequence_match(shots, references={0: ref})
    assert "a" in deltas and "b" in deltas   # both converge, despite being far apart in RGB
    print("ok  match: adjacent same-file shots always group, regardless of RGB distance")


# --------------------------------------------------------------------------
# Step 1.5/1.1/1.3/1.7: resolver.py pipeline plumbing
# --------------------------------------------------------------------------

def test_resolver_v1_sets_v1_working_space():
    g = resolve_clip_grade({}, color_stats=None)
    assert g["working_space"] == tone.WORKING_SPACE_V1, g["working_space"]
    print("ok  resolver: default working_space is the v1 working space")


def test_resolver_explicit_working_space_overrides_pipeline_default():
    g = resolve_clip_grade({"working_space": "custom_space"}, color_stats=None)
    assert g["working_space"] == "custom_space", g["working_space"]
    print("ok  resolver: an item's explicit working_space wins over the default")


# --------------------------------------------------------------------------
# color_phase1.plan.md Part 2: per-clip log working-space selection
# --------------------------------------------------------------------------

def test_resolver_is_log_flat_selects_log_working_space():
    g = resolve_clip_grade({}, color_stats=_cs(is_log_flat=True))
    assert g["working_space"] == tone.WORKING_SPACE_LOG_V1, g["working_space"]
    print("ok  resolver: is_log_flat color_stats selects WORKING_SPACE_LOG_V1")


def test_resolver_non_log_stays_on_v1_working_space():
    g = resolve_clip_grade({}, color_stats=_cs(is_log_flat=False))
    assert g["working_space"] == tone.WORKING_SPACE_V1, g["working_space"]
    print("ok  resolver: is_log_flat=False (or absent) stays on WORKING_SPACE_V1")


def test_resolver_no_color_stats_stays_on_v1_working_space():
    """color_stats=None must not crash the `.get("is_log_flat")` read."""
    g = resolve_clip_grade({}, color_stats=None)
    assert g["working_space"] == tone.WORKING_SPACE_V1, g["working_space"]
    print("ok  resolver: color_stats=None is treated as non-log, no crash")


def test_resolver_explicit_working_space_wins_over_is_log_flat():
    g = resolve_clip_grade({"working_space": "custom_space"}, color_stats=_cs(is_log_flat=True))
    assert g["working_space"] == "custom_space", g["working_space"]
    print("ok  resolver: an item's explicit working_space still wins even when is_log_flat is set")


def test_resolver_log_clip_produces_a_finite_bounded_cdl():
    """The log decode + composite clamp must still produce a sane CDL --
    not NaN/inf/wildly out of range -- for a representative log/flat
    color_stats row (low native contrast, mid_gray reading dim)."""
    cs = _cs(black_point=0.05, white_point=0.55, mid_gray=0.25, is_log_flat=True)
    g_dict = resolve_clip_grade({}, color_stats=cs)
    cdl = Grade.from_dict(g_dict["cdl"])
    for s in cdl.slope:
        assert math.isfinite(s) and 0.0 < s < 5.0, cdl.slope
    for o in cdl.offset:
        assert math.isfinite(o), cdl.offset
    print("ok  resolver: a log-flat clip resolves to a finite, bounded CDL")


def test_resolver_non_log_golden_grade_hash_unchanged_by_part2():
    """HARD REQUIREMENT (color_phase1.plan.md 2.2): adding per-clip log
    working-space selection must be byte-identical for a non-log clip -- the
    golden hash below was captured by running this exact resolve BEFORE
    Part 2's tone.py/correct.py/resolver.py changes (git stash + rerun), not
    guessed. If this ever changes, something in the non-log path moved."""
    g = resolve_clip_grade({}, color_stats=_cs())
    assert g["working_space"] == tone.WORKING_SPACE_V1, g["working_space"]
    assert g["grade_hash"] == "362cad69961b690a70e6280c1f21507e", g["grade_hash"]
    assert g["cdl"] == {
        "slope": [0.8333333333333334, 0.8333333333333334, 0.8333333333333334],
        "offset": [0.003661125103632607, 0.003661125103632607, 0.003661125103632607],
        "power": [1.0, 1.0, 1.0], "sat": 1.0,
    }, g["cdl"]
    print("ok  resolver: non-log clip's grade_hash is byte-identical to pre-Part-2")


def test_resolver_log_flat_grade_hash_differs_from_treating_it_as_v1():
    """The whole point of Part 2: a log-tagged clip must actually get a
    DIFFERENT (distinct cache key / distinct baked cube) grade than the
    same measurement would under WORKING_SPACE_V1 -- otherwise the log
    decode isn't actually being applied."""
    cs_log = _cs(black_point=0.05, white_point=0.55, mid_gray=0.25, is_log_flat=True)
    cs_plain = _cs(black_point=0.05, white_point=0.55, mid_gray=0.25, is_log_flat=False)
    g_log = resolve_clip_grade({}, color_stats=cs_log)
    g_plain = resolve_clip_grade({}, color_stats=cs_plain)
    assert g_log["working_space"] != g_plain["working_space"]
    assert g_log["grade_hash"] != g_plain["grade_hash"]
    print("ok  resolver: a log-flat clip's grade_hash differs from the same clip graded as v1")


def test_resolver_reference_transfer_v1_does_not_crash_or_blow_up():
    """Step 1.5: dropping a reference still composes without double-
    stretching into an extreme grade."""
    color_stats = _cs(rgb_mean=[0.3, 0.3, 0.3], rgb_std=[0.15, 0.15, 0.15])
    sequence_look = {
        "mode": "reference",
        "reference_stats": {"rgb_mean": [0.6, 0.5, 0.4], "rgb_std": [0.2, 0.2, 0.2]},
        "match_strength": 0.6,
    }
    g_dict = resolve_clip_grade({}, color_stats=color_stats, sequence_look=sequence_look)
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
# Step 1.0: layers.py's read-path (grade_lookup)
# --------------------------------------------------------------------------

def _doc_one_seg():
    return {"timeline": [{"seg_id": "s0", "file_id": "f1", "in_ms": 0, "out_ms": 2000}], "operations": []}


def test_layers_no_grade_lookup_falls_back_to_identity():
    """Calling resolve() with NO grade_lookup arg (every caller that hasn't
    fetched one) must render identity -- layers.resolve never computes a
    grade inline; grading happens exclusively in run_grade_job."""
    doc = _doc_one_seg()
    cs = {"f1": _cs(rgb_mean=[0.4, 0.5, 0.6])}
    resolved = layers.resolve(doc, {}, cs)
    cdl = Grade.from_dict(resolved.video_layers[0].grade["cdl"])
    assert cdl == Grade(), cdl
    print("ok  layers: no grade_lookup -> identity (never computes inline)")


def test_layers_v1_reads_grade_lookup_hit():
    doc = _doc_one_seg()
    fake_grade = identity_grade_json(tone.WORKING_SPACE_V1)
    fake_grade["cdl"] = Grade(slope=(1.3, 1.0, 1.0)).to_dict()   # a distinctive, obviously-not-computed value
    resolved = layers.resolve(doc, {}, {}, grade_lookup={"s0": fake_grade})
    assert resolved.video_layers[0].grade == fake_grade
    print("ok  layers: reads the pre-fetched grade_lookup hit verbatim (never recomputes)")


def test_layers_v1_falls_back_to_identity_on_miss():
    """A shot missing from grade_lookup (the job hasn't produced it yet)
    must render as identity, never an error and never a stale inline
    computation -- preview stays responsive while the job catches up."""
    doc = _doc_one_seg()
    cs = {"f1": _cs(rgb_mean=[0.9, 0.1, 0.1])}   # would normally correct heavily
    resolved = layers.resolve(doc, {}, cs, grade_lookup={})
    cdl = Grade.from_dict(resolved.video_layers[0].grade["cdl"])
    assert cdl == Grade(), cdl   # identity -- color_stats is NOT used to compute anything
    print("ok  layers: falls back to identity (never computes inline) when grade_lookup misses")


def test_layers_split_screen_region_preserves_spine_grade():
    """Regression (found live while verifying grade_pipeline_standardize.plan.md
    against a real split-screen thread): a layout_region with a "spine" cell
    slices the spine's video layers via _slice_video (layers.py::
    _dest_spine_window/_apply_layout_regions) to stamp the split-screen dest
    rect -- _slice_video was dropping `.grade` entirely (defaulted to `{}`,
    the dataclass field default), so every split-screen document silently
    rendered its spine ungraded. The sliced layer(s) must carry the SAME
    grade the un-sliced layer had."""
    doc = {
        "timeline": [{"seg_id": "s0", "file_id": "f1", "in_ms": 0, "out_ms": 4000}],
        "operations": [{
            "type": "place_video", "op_id": "op0", "source_file_id": "f2",
            "src_in_ms": 0, "src_out_ms": 4000, "from_ms": 0, "to_ms": 4000,
        }],
        "layout_regions": [{
            "region_id": "lr0", "template": "split_h", "from_ms": 0, "to_ms": 4000,
            "cells": {"left": {"layer": "spine"}, "right": {"layer": "op0"}},
        }],
    }
    fake_grade = identity_grade_json(tone.WORKING_SPACE_V1)
    fake_grade["cdl"] = Grade(slope=(1.3, 1.0, 1.0)).to_dict()
    resolved = layers.resolve(doc, {}, {}, grade_lookup={"s0": fake_grade, "op0": fake_grade})
    spine_layers = [v for v in resolved.video_layers if v.kind == "spine"]
    assert spine_layers, "the region must actually slice at least one spine layer"
    for v in spine_layers:
        assert v.grade == fake_grade, (v.layer_id, v.grade)
    print("ok  layers: a split-screen layout_region's sliced spine layers keep their grade")


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

    def fake_measure_span(file_id, in_ms, out_ms, *, hero_ts_ms=None, subject_box=None):
        return _cs(rgb_mean=[0.4, 0.5, 0.6], mid_gray=0.3)

    with mock.patch("app.services.l3.grade.job.get_job_state", return_value=None), \
         mock.patch("app.services.l3.grade.job._upsert_job_status",
                    side_effect=lambda tid, **kw: status_calls.append(kw)), \
         mock.patch("app.services.l3.grade.job._upsert_grade_row",
                    side_effect=lambda tid, key, h, gj, cube: upserted_rows.append((key, h, gj, cube))), \
         mock.patch("app.services.l3.grade.job.fetch_color_stats", return_value={}), \
         mock.patch("app.services.l3.grade.job.measure_span", side_effect=fake_measure_span), \
         mock.patch("app.services.l3.grade.job.ensure_cube_file", return_value="/tmp/fake.cube"), \
         mock.patch("app.services.l3.grade.scene_meta.lookup_shot_cut_meta", return_value={}), \
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


# --------------------------------------------------------------------------
# Phase 2: grade/leveling.py (exposure + tonal-placement leveling)
# --------------------------------------------------------------------------

def test_leveling_flattens_jittery_brightness():
    jittery = [ShotLevelInput(key=f"s{i}", mid_gray=mg, black_point=0.02, white_point=0.95)
              for i, mg in enumerate([0.4, 0.55, 0.3, 0.5, 0.35, 0.45, 0.32])]
    deltas = solve_exposure_leveling(jittery)
    assert len(deltas) >= 5, deltas   # most shots need SOME nudge in a jittery sequence
    print("ok  leveling: flattens a jittery-brightness montage")


def test_leveling_preserves_an_intentional_arc():
    arc = [ShotLevelInput(key=f"s{i}", mid_gray=mg, black_point=0.02, white_point=0.95)
          for i, mg in enumerate([0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15])]
    deltas = solve_exposure_leveling(arc)
    projected = [s.mid_gray * (deltas[s.key].slope[0] if s.key in deltas else 1.0) for s in arc]
    assert projected[0] > projected[-1] + 0.2, projected   # the overall day->night trend survives
    print("ok  leveling: an intentional day->night arc survives (the smooth target follows it)")


def test_leveling_exposure_gain_is_bounded():
    spike = [ShotLevelInput(key=f"s{i}", mid_gray=0.4, black_point=0.02, white_point=0.95) for i in range(5)]
    spike[2] = ShotLevelInput(key="s2", mid_gray=0.05, black_point=0.02, white_point=0.95)
    deltas = solve_exposure_leveling(spike)
    cap = 2.0 ** 0.5   # EXPOSURE_CAP_STOPS
    assert abs(deltas["s2"].slope[0] - cap) < 1e-6, deltas["s2"].slope
    print("ok  leveling: exposure gain never exceeds the stop cap")


def test_leveling_tonal_converges_low_contrast_and_punchy():
    scene = [
        ShotLevelInput(key="low1", mid_gray=0.5, black_point=0.15, white_point=0.75),
        ShotLevelInput(key="punchy", mid_gray=0.5, black_point=0.01, white_point=0.99),
        ShotLevelInput(key="low2", mid_gray=0.5, black_point=0.13, white_point=0.77),
    ]
    deltas = solve_tonal_leveling(scene)
    assert "low1" in deltas and "low2" in deltas
    print("ok  leveling: low-contrast and punchy shots in one scene converge")


def test_leveling_tonal_skips_cross_scene_outlier():
    outlier_scene = [
        ShotLevelInput(key="a", black_point=0.1, white_point=0.8, mid_gray=0.5),
        ShotLevelInput(key="b", black_point=0.1, white_point=0.8, mid_gray=0.5),
        ShotLevelInput(key="weird", black_point=0.45, white_point=0.55, mid_gray=0.5),
        ShotLevelInput(key="c", black_point=0.1, white_point=0.8, mid_gray=0.5),
        ShotLevelInput(key="d", black_point=0.1, white_point=0.8, mid_gray=0.5),
    ]
    deltas = solve_tonal_leveling(outlier_scene)
    assert "weird" not in deltas, "a genuinely different scene must not be forced to fit"
    print("ok  leveling: a cross-scene outlier is skipped, not forced to fit")


def test_leveling_tonal_never_pushes_toward_clipping():
    near_full = [ShotLevelInput(key=f"s{i}", black_point=0.0, white_point=1.0, mid_gray=0.5) for i in range(4)]
    deltas = solve_tonal_leveling(near_full)
    for k, g in deltas.items():
        proj_b, proj_w = 0.0 * g.slope[0] + g.offset[0], 1.0 * g.slope[0] + g.offset[0]
        assert -0.011 <= proj_b and proj_w <= 1.011, (k, proj_b, proj_w)
    print("ok  leveling: tonal alignment never pushes a shot toward clipping")


def test_leveling_never_crashes_on_a_true_black_point():
    """Regression: black_point == 0.0 exactly (a common, legitimate value)
    must not break the outlier check (a ratio-based check on black_point
    itself would divide by ~0)."""
    mixed = [ShotLevelInput(key="z0", black_point=0.0, white_point=0.9, mid_gray=0.4),
            ShotLevelInput(key="z1", black_point=0.05, white_point=0.85, mid_gray=0.4),
            ShotLevelInput(key="z2", black_point=0.0, white_point=0.92, mid_gray=0.4)]
    solve_tonal_leveling(mixed)   # must not raise
    print("ok  leveling: a true black_point of 0.0 never crashes the outlier check")


def test_leveling_subject_luma_used_when_not_a_silhouette():
    """Step 3.1: exposure leveling targets subject_luma (not whole-frame
    mid_gray) when a usable one is present."""
    shots = [ShotLevelInput(key=f"s{i}", mid_gray=0.5, black_point=0.02, white_point=0.95,
                            subject_luma=sl)
            for i, sl in enumerate([0.3, 0.5, 0.28, 0.48, 0.32])]
    deltas = solve_exposure_leveling(shots)
    # subject_luma jitters (0.3/0.5 alternating) while mid_gray is FLAT at
    # 0.5 -- if subject_luma weren't being used, nothing would need leveling.
    assert len(deltas) >= 2, deltas
    print("ok  leveling: subject-aware exposure targets subject_luma, not whole-frame mid_gray")


def test_leveling_subject_luma_ignored_when_silhouette():
    """Step 3.1's gate: a subject_luma far enough from the frame's own
    mid_gray (a deliberate silhouette/backlit shot) is NOT treated as a
    wrong exposure -- falls back to whole-frame mid_gray, which is already
    flat/consistent here, so nothing should move."""
    shots = [ShotLevelInput(key=f"s{i}", mid_gray=0.5, black_point=0.02, white_point=0.95,
                            subject_luma=0.05)   # a silhouette: subject WAY darker than the frame
            for i in range(5)]
    deltas = solve_exposure_leveling(shots)
    assert deltas == {}, deltas
    print("ok  leveling: a deliberate silhouette's subject_luma is ignored (falls back to mid_gray)")


def test_leveling_composed_result_includes_both_stages():
    shots = [
        ShotLevelInput(key="a", mid_gray=0.3, black_point=0.1, white_point=0.8),
        ShotLevelInput(key="b", mid_gray=0.5, black_point=0.02, white_point=0.95),
        ShotLevelInput(key="c", mid_gray=0.3, black_point=0.1, white_point=0.8),
    ]
    composed = solve_leveling(shots)
    exposure_only = solve_exposure_leveling(shots)
    tonal_only = solve_tonal_leveling(shots)
    assert set(composed) == set(exposure_only) | set(tonal_only)
    print("ok  leveling: solve_leveling composes exposure + tonal into one delta per shot")


# --------------------------------------------------------------------------
# Step 3.1: measure_span's subject-luma crop (pure -- synthetic frame, no ffmpeg)
# --------------------------------------------------------------------------

def test_measure_subject_luma_reads_the_box_not_the_whole_frame():
    import numpy as np
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = [20, 20, 20]         # dark background
    frame[40:60, 40:60] = [230, 230, 230]   # a bright subject box, center 40%x40%..60%x60%
    luma = _measure_subject_luma(frame, (0.4, 0.4, 0.2, 0.2))
    assert luma is not None and luma > 0.8, luma   # reads the bright box, not the dark background
    whole_frame_luma = _measure_subject_luma(frame, (0.0, 0.0, 1.0, 1.0))
    assert whole_frame_luma < luma   # the whole-frame average is dragged down by the dark background
    print("ok  measure_span: subject_luma reads inside the box, not the whole frame")


def test_measure_subject_luma_none_for_degenerate_box():
    import numpy as np
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert _measure_subject_luma(frame, (0.5, 0.5, 0.0, 0.0)) is None
    assert _measure_subject_luma(frame, (1.5, 1.5, 0.1, 0.1)) is None
    print("ok  measure_span: subject_luma is None for a degenerate/out-of-frame box")


# --------------------------------------------------------------------------
# Step 3.2: grade/scene_group.py
# --------------------------------------------------------------------------

def test_scene_group_same_speaker_different_file_groups():
    meta = [
        ShotSceneMeta(key="s0", file_id="f1", speaker_person="P1", on_camera=True),
        ShotSceneMeta(key="s1", file_id="f2", speaker_person="P1", on_camera=True),
    ]
    assert group_shots_semantically(meta) == [[0, 1]]
    print("ok  scene_group: same speaker across different files groups")


def test_scene_group_no_shared_signal_does_not_group():
    meta = [
        ShotSceneMeta(key="a", file_id="f1", speaker_person="P1"),
        ShotSceneMeta(key="b", file_id="f2", speaker_person="P2", label="kitchen"),
    ]
    assert group_shots_semantically(meta) == [[0], [1]]
    print("ok  scene_group: no shared trusted signal -> no group")


def test_scene_group_overrides_rgb_grouping_in_solve_sequence_match():
    """Step 3.2's acceptance: shots from one setup grade together even when
    a transient skews their RGB (RGB-based grouping alone would miss it)."""
    shots = [
        ShotStats(key="s0", file_id="f1",
                  stats={"black_point": 0.02, "white_point": 0.9, "mid_gray": 0.35,
                        "rgb_mean": [0.9, 0.1, 0.1]}, quality=0.5),
        ShotStats(key="s1", file_id="f2",
                  stats={"black_point": 0.05, "white_point": 0.8, "mid_gray": 0.3,
                        "rgb_mean": [0.1, 0.9, 0.1]}, quality=0.8),
    ]
    assert solve_sequence_match(shots) == {}, "RGB-only grouping should NOT match these"
    ref = compute_group_reference([s.stats for s in shots])
    forced = solve_sequence_match(shots, groups=[[0, 1]], references={0: ref})
    assert "s0" in forced, "semantic groups must override the default RGB grouping"
    print("ok  scene_group: semantic groups align a same-setup pair RGB alone would miss")


# --------------------------------------------------------------------------
# color_shot_matching.plan.md: graceful grouping fallback, a robust
# per-group reference, and the new Balance layer (Phases 1-4)
# --------------------------------------------------------------------------

def test_shot_match_falls_back_to_rgb_when_semantic_is_all_singletons():
    """Phase 1 (SS2 "primary cause"): a real multi-file reel with NO
    speaker_person/on_camera/label/summary set (the common real-data case)
    degrades semantic grouping to all-singletons -- job.py's
    _has_real_groups must flag this so the caller falls back to RGB
    adjacency instead of silently letting Match do nothing."""
    shots = [
        ShotStats(key=f"s{i}", file_id=f"f{i}",
                  stats={"black_point": 0.02, "white_point": 0.9,
                        "mid_gray": 0.3 + 0.01 * i,
                        "rgb_mean": [0.4 + 0.005 * i, 0.4, 0.4]})
        for i in range(8)
    ]
    rgb_groups = group_neighbors(shots)
    assert any(len(g) >= 2 for g in rgb_groups), rgb_groups   # RGB adjacency finds structure

    scene_meta = [ShotSceneMeta(key=f"s{i}", file_id=f"f{i}") for i in range(8)]
    semantic_groups = group_shots_semantically(scene_meta)
    assert semantic_groups == [[i] for i in range(8)], semantic_groups   # all singletons

    assert grade_job._has_real_groups(semantic_groups) is False
    assert grade_job._has_real_groups(rgb_groups) is True
    print("ok  shot_match: all-singleton semantic result correctly signals a fallback to RGB adjacency")


def _synthetic_reel_group():
    """5-shot synthetic scene-group -- color_shot_matching.plan.md's own
    suggested mids, plus varied black/white/rgb_mean casts -- the
    acceptance-criteria fixture for the spread/shadow tests below."""
    return [
        ("g0", 0.30, 0.10, 0.75, [0.35, 0.30, 0.25]),
        ("g1", 0.55, 0.02, 0.95, [0.55, 0.50, 0.45]),
        ("g2", 0.32, 0.12, 0.70, [0.30, 0.32, 0.30]),
        ("g3", 0.50, 0.03, 0.92, [0.52, 0.48, 0.44]),
        ("g4", 0.35, 0.09, 0.78, [0.38, 0.34, 0.30]),
    ]


def _balance_and_match_composed(specs):
    """Mirrors job.py's Phase 2c wiring for ONE scene-group: project to
    working space, compute the robust median reference, solve Balance,
    then project EACH shot's stats through its OWN Balance delta
    (grade.job._corrected_display_stats) before solving Match against them
    (+ a reference recomputed from the corrected stats). Returns
    (key -> raw display stats, key -> the composed Balance-then-Match
    Grade, each already run through resolver._clamp_composite_v1, matching
    what the real v1 bake applies).

    Solving Match against the RAW stats instead of the corrected ones is
    NOT equivalent to this -- it double-corrects and overshoots (see
    grade.job._corrected_display_stats's docstring for the verified
    numbers)."""
    from app.services.l3.grade.balance import solve_balance
    from app.services.l3.grade.resolver import _clamp_composite_v1

    keys = [s[0] for s in specs]
    display_stats = {
        key: {"mid_gray": mid, "black_point": black, "white_point": white, "rgb_mean": rgb}
        for key, mid, black, white, rgb in specs
    }
    ordered_stats = [display_stats[k] for k in keys]
    ws_stats = [grade_job._ws_stats(st) for st in ordered_stats]
    groups = [list(range(len(specs)))]

    balance_ref = compute_group_reference(ws_stats)
    balance_deltas = solve_balance(ws_stats, groups, {0: balance_ref}, keys)

    corrected_stats = [
        grade_job._corrected_display_stats(ws_stats[i], balance_deltas[k]) if k in balance_deltas
        else ordered_stats[i]
        for i, k in enumerate(keys)
    ]
    match_ref = compute_group_reference(corrected_stats)
    match_shots = [ShotStats(key=k, file_id="f", stats=st) for k, st in zip(keys, corrected_stats)]
    match_deltas = solve_sequence_match(
        match_shots, groups=groups, working_space=tone.WORKING_SPACE_V1, references={0: match_ref},
    )

    from app.services.l3.grade.cdl import compose

    composed = {}
    for k in keys:
        stack = balance_deltas.get(k, Grade())
        if k in match_deltas:
            stack = compose(stack, match_deltas[k], 1.0)
        composed[k] = _clamp_composite_v1(stack)
    return display_stats, composed


def _roundtrip_rgb_v1(rgb_display, grade):
    """Like _roundtrip_v1 but for a full RGB triple (WB checks need the
    per-channel result, not just channel 0)."""
    import numpy as np
    lin = tone.to_working(np.array(rgb_display, dtype=np.float32), tone.WORKING_SPACE_V1)
    out = tone.from_working(apply_cdl(lin, grade), tone.WORKING_SPACE_V1)
    return [float(x) for x in out]


def test_shot_match_cross_shot_spread_shrinks_after_balance_and_match():
    """color_shot_matching.plan.md SS9 acceptance: exposure, contrast, and
    white-balance spread across a scene-group all shrink substantially
    after Balance+Match (exposure < 0.04 absolute and >= 40% smaller;
    contrast spread >= 25% smaller -- lower than the plan's original "~40%"
    because the resolver.py shadow-floor follow-up fix (see
    _clamp_composite_v1's COMPOSITE_SHADOW_PROBE) now also nudges some of
    these shots' offsets to protect a genuine shadow, which trades a little
    contrast-convergence precision for never crushing shadow detail --
    verified at ~29% for this fixture, was ~38% before that fix)."""
    import statistics

    specs = _synthetic_reel_group()
    display_stats, composed = _balance_and_match_composed(specs)
    keys = [s[0] for s in specs]

    pre_mids = [display_stats[k]["mid_gray"] for k in keys]
    post_mids = [_roundtrip_v1(display_stats[k]["mid_gray"], composed[k]) for k in keys]
    pre_mid_sd, post_mid_sd = statistics.pstdev(pre_mids), statistics.pstdev(post_mids)
    assert post_mid_sd < 0.04, post_mid_sd
    assert post_mid_sd <= pre_mid_sd * 0.6, (pre_mid_sd, post_mid_sd)

    pre_c = [display_stats[k]["white_point"] - display_stats[k]["black_point"] for k in keys]
    post_c = [
        _roundtrip_v1(display_stats[k]["white_point"], composed[k])
        - _roundtrip_v1(display_stats[k]["black_point"], composed[k])
        for k in keys
    ]
    pre_c_sd, post_c_sd = statistics.pstdev(pre_c), statistics.pstdev(post_c)
    assert post_c_sd <= pre_c_sd * 0.75, (pre_c_sd, post_c_sd)

    pre_rg = [display_stats[k]["rgb_mean"][0] / display_stats[k]["rgb_mean"][1] for k in keys]
    post_rgb = [_roundtrip_rgb_v1(display_stats[k]["rgb_mean"], composed[k]) for k in keys]
    post_rg = [o[0] / o[1] for o in post_rgb]
    pre_rg_sd, post_rg_sd = statistics.pstdev(pre_rg), statistics.pstdev(post_rg)
    assert post_rg_sd <= pre_rg_sd * 0.5, (pre_rg_sd, post_rg_sd)

    print(f"ok  shot_match: cross-shot spread shrinks after Balance+Match (mid stdev "
         f"{pre_mid_sd:.3f}->{post_mid_sd:.3f}, contrast {pre_c_sd:.3f}->{post_c_sd:.3f}, "
         f"R/G {pre_rg_sd:.3f}->{post_rg_sd:.3f})")


def test_shot_match_no_shadow_crush_after_balance_and_match():
    """No shadow-crush regression (the already-shipped darkness fix must
    survive the new layers): for the dullest shot in the synthetic group, a
    display 0.15 shadow is not crushed and a display 0.5 mid lands in the
    plausible-midtone range after the FULL composed+clamped grade."""
    specs = _synthetic_reel_group()
    display_stats, composed = _balance_and_match_composed(specs)
    dullest_key = min(display_stats, key=lambda k: display_stats[k]["mid_gray"])

    shadow_out = _roundtrip_v1(0.15, composed[dullest_key])
    mid_out = _roundtrip_v1(0.5, composed[dullest_key])
    assert shadow_out > 0.02, shadow_out
    assert 0.46 <= mid_out <= 0.65, mid_out
    print(f"ok  shot_match: no shadow-crush for the dullest shot after Balance+Match "
         f"(shadow 0.15->{shadow_out:.3f}, mid 0.5->{mid_out:.3f})")


def test_balance_lifts_capped_dull_shot_and_survives_composite_slope_clamp():
    """Guardrail-interaction test: a very dull shot (mid 0.20) matched
    against a much brighter reference (mid ~0.50) needs a pre-clamp slope
    that EXCEEDS resolver.COMPOSITE_SLOPE_MAX -- so this only means
    anything if the clamp is actually exercised. After clamping, the shot's
    mid must still rise materially (not be capped back down to ~identity).

    Note: verified empirically that the resulting offset is small and
    slightly NEGATIVE for a realistic (near-zero) black point, not positive
    as color_shot_matching.plan.md's prose describes -- pivoting exactly at
    `black` (out(black)=black) is algebraically incompatible with a
    positive offset whenever slope > 1 and black > 0. The lift comes
    overwhelmingly from the (pre-clamp) SLOPE term, not the offset; this
    test asserts the verified, actually-true property (material lift
    survives the clamp) rather than the offset's sign."""
    from app.services.l3.grade.balance import solve_balance
    from app.services.l3.grade.resolver import COMPOSITE_SLOPE_MAX, _clamp_composite_v1

    dull = {"mid_gray": 0.20, "black_point": 0.02, "white_point": 0.85, "rgb_mean": [0.2, 0.2, 0.2]}
    bright = {"mid_gray": 0.55, "black_point": 0.02, "white_point": 0.95, "rgb_mean": [0.55, 0.55, 0.55]}
    ws_stats = [grade_job._ws_stats(dull), grade_job._ws_stats(bright)]
    ref = compute_group_reference(ws_stats)
    deltas = solve_balance(ws_stats, [[0, 1]], {0: ref}, ["dull", "bright"])

    g = deltas["dull"]
    assert g.slope[0] > COMPOSITE_SLOPE_MAX, g.slope   # genuinely exercises the clamp
    clamped = _clamp_composite_v1(g)
    assert clamped.slope[0] == COMPOSITE_SLOPE_MAX

    mid_after = _roundtrip_v1(0.20, clamped)
    assert mid_after - 0.20 >= 0.05, mid_after
    print(f"ok  balance: a capped-dull shot still lifts materially after the composite "
         f"slope clamp (0.20 -> {mid_after:.3f})")


def test_shot_match_references_override_matches_every_member_not_just_anchor():
    """An explicit `references` override matches EVERY member toward the
    reference -- no member is exempt as "the anchor" (grade_pipeline_
    standardize.plan.md: the only path since the anchor fallback was
    removed)."""
    shots = [
        ShotStats(key="s0", file_id="camA",
                  stats={"black_point": 0.02, "white_point": 0.9, "mid_gray": 0.35,
                        "rgb_mean": [0.5, 0.48, 0.46]}, quality=0.5),
        ShotStats(key="s1", file_id="camB",
                  stats={"black_point": 0.05, "white_point": 0.8, "mid_gray": 0.3,
                        "rgb_mean": [0.46, 0.47, 0.5]}, quality=0.8),
    ]
    ref = GroupReference(mid_gray=0.4, black_point=0.03, white_point=0.85, rgb_mean=[0.48, 0.475, 0.48])
    forced = solve_sequence_match(shots, references={0: ref})
    assert "s0" in forced and "s1" in forced, forced   # no member exempt

    no_references = solve_sequence_match(shots)
    assert no_references == {}   # no reference for the group -> skipped, not an anchor fallback
    print("ok  shot_match: an explicit reference matches every member, none exempt as 'the anchor'")


# --------------------------------------------------------------------------
# color_scene_grouping.plan.md: real cut_records metadata + an always-on
# RGB base folded into the semantic grouper itself
# --------------------------------------------------------------------------

def test_scene_grouping_rgb_base_prevents_all_singletons():
    """Phase 2c: 8 shots, different file_id, no speaker/on_camera/take/sync
    signals, but adjacent rgb_mean within SCENE_RGB_DIST_MAX -- the RGB base
    must chain them into real groups. Without rgb_mean (today's pre-this-
    plan behavior), the same shots are all-singletons."""
    from app.services.l3.grade.scene_group import SCENE_RGB_DIST_MAX

    meta_with_rgb = [
        ShotSceneMeta(key=f"s{i}", file_id=f"f{i}", rgb_mean=[0.4 + 0.01 * i, 0.4, 0.4])
        for i in range(8)
    ]
    groups = group_shots_semantically(meta_with_rgb)
    assert any(len(g) >= 2 for g in groups), groups

    meta_without_rgb = [ShotSceneMeta(key=f"s{i}", file_id=f"f{i}") for i in range(8)]
    assert group_shots_semantically(meta_without_rgb) == [[i] for i in range(8)]
    print(f"ok  scene_grouping: the RGB base (< {SCENE_RGB_DIST_MAX}) chains an "
         f"otherwise all-singleton reel")


def test_scene_grouping_sync_group_id_groups_across_files():
    meta = [
        ShotSceneMeta(key="s0", file_id="f1", sync_group_id="sync-A"),
        ShotSceneMeta(key="s1", file_id="f2", sync_group_id="sync-A"),
    ]
    assert group_shots_semantically(meta) == [[0, 1]]
    print("ok  scene_grouping: same sync_group_id groups across files (multicam outlook)")


def test_scene_grouping_take_group_id_groups_across_files():
    meta = [
        ShotSceneMeta(key="s0", file_id="f1", take_group_id="take-A"),
        ShotSceneMeta(key="s1", file_id="f2", take_group_id="take-A"),
    ]
    assert group_shots_semantically(meta) == [[0, 1]]
    print("ok  scene_grouping: same take_group_id groups across files (retakes)")


def test_scene_grouping_voice_ids_overlap_groups_across_files():
    meta = [
        ShotSceneMeta(key="s0", file_id="f1", voice_ids=["V1", "V2"]),
        ShotSceneMeta(key="s1", file_id="f2", voice_ids=["V2", "V3"]),
    ]
    assert group_shots_semantically(meta) == [[0, 1]]
    meta_no_overlap = [
        ShotSceneMeta(key="s0", file_id="f1", voice_ids=["V1"]),
        ShotSceneMeta(key="s1", file_id="f2", voice_ids=["V9"]),
    ]
    assert group_shots_semantically(meta_no_overlap) == [[0], [1]]
    print("ok  scene_grouping: overlapping voice_ids groups across files, no overlap does not")


def test_scene_meta_join_picks_max_overlap_cut_record():
    """scene_meta.lookup_shot_cut_meta: among cut_records for the same file,
    picks the one with the LARGEST time-overlap with the shot's span. A
    shot with no overlapping cut_record is absent from the result (no
    crash, no fabricated metadata)."""
    rows = [
        {"file_id": "f1", "src_in_ms": 0, "src_out_ms": 2000,
         "label": "first cut", "summary": "", "on_camera": None,
         "speaker_person": None, "voice_ids": [], "take_group_id": None, "sync_group_id": None},
        {"file_id": "f1", "src_in_ms": 1800, "src_out_ms": 5000,
         "label": "second cut", "summary": "", "on_camera": None,
         "speaker_person": None, "voice_ids": [], "take_group_id": None, "sync_group_id": None},
    ]
    with mock.patch("app.services.l3.cuts_v3_read.latest_run_for_files", return_value="run-1"), \
         mock.patch("app.services.l3.cuts_v3_read.rows_for_run", return_value=rows):
        result = lookup_shot_cut_meta([
            ("shot-a", "f1", 2200, 4800),   # overlaps 2nd cut [1800,5000] far more than 1st [0,2000]
            ("shot-b", "f1", 9000, 9500),   # no overlap with either cut
        ])
    assert result["shot-a"].label == "second cut", result
    assert "shot-b" not in result, result
    print("ok  scene_meta: the (file_id, span) join picks the max-overlap cut_record")


def test_run_grade_job_groups_multi_file_reel_via_scene_join():
    """Acceptance: a 3-shot, 3-file doc where the mocked cut_records join
    returns a shared sync_group_id for two of the three shots -- Balance/
    Match must act on the grouped pair (non-identity deltas)."""
    doc = {
        "timeline": [
            {"seg_id": "s0", "file_id": "f0", "in_ms": 0, "out_ms": 1000},
            {"seg_id": "s1", "file_id": "f1", "in_ms": 0, "out_ms": 1000},
            {"seg_id": "s2", "file_id": "f2", "in_ms": 0, "out_ms": 1000},
        ],
        "operations": [], "look": {},
    }

    def fake_measure_span(file_id, in_ms, out_ms, *, hero_ts_ms=None, subject_box=None):
        return {"black_point": 0.02, "white_point": 0.9,
               "mid_gray": {"f0": 0.30, "f1": 0.55, "f2": 0.32}[file_id],
               "rgb_mean": [0.4, 0.4, 0.4], "rgb_std": [0.1, 0.1, 0.1]}

    joined = {
        "s0": ShotCutMeta(sync_group_id="sync-X"),
        "s1": ShotCutMeta(sync_group_id="sync-X"),
        # s2 has no covering cut_record -- absent from the join result
    }
    rows = {}
    with mock.patch("app.services.l3.grade.job.get_job_state", return_value=None), \
         mock.patch("app.services.l3.grade.job._upsert_job_status"), \
         mock.patch("app.services.l3.grade.job._upsert_grade_row",
                    side_effect=lambda tid, key, h, gj, cube: rows.__setitem__(key, gj)), \
         mock.patch("app.services.l3.grade.job.fetch_color_stats", return_value={}), \
         mock.patch("app.services.l3.grade.job.measure_span", side_effect=fake_measure_span), \
         mock.patch("app.services.l3.grade.job.ensure_cube_file", return_value=None), \
         mock.patch("app.services.l3.grade.scene_meta.lookup_shot_cut_meta", return_value=joined), \
         mock.patch("app.services.l3.store.latest_document", return_value=(doc, 1), create=True):
        grade_job.run_grade_job("thread-scene-join")

    assert set(rows) == {"s0", "s1", "s2"}
    non_identity = {k for k, gj in rows.items() if Grade.from_dict(gj["cdl"]) != Grade()}
    assert {"s0", "s1"} & non_identity, rows
    print("ok  job: run_grade_job groups a multi-file reel via the mocked cut_records join")


def test_run_grade_job_scene_join_does_not_over_group_unrelated_shots():
    """With the join returning empty (no covering cut_record for any shot)
    and RGB genuinely far apart, grouping must NOT force a match --
    verifies an empty join doesn't silently over-group."""
    doc = {
        "timeline": [
            {"seg_id": "s0", "file_id": "f0", "in_ms": 0, "out_ms": 1000},
            {"seg_id": "s1", "file_id": "f1", "in_ms": 0, "out_ms": 1000},
        ],
        "operations": [], "look": {},
    }

    def fake_measure_span(file_id, in_ms, out_ms, *, hero_ts_ms=None, subject_box=None):
        return {"black_point": 0.02, "white_point": 0.9, "mid_gray": 0.3,
               "rgb_mean": [0.9, 0.1, 0.1] if file_id == "f0" else [0.1, 0.9, 0.1],
               "rgb_std": [0.1, 0.1, 0.1]}

    rows = {}
    with mock.patch("app.services.l3.grade.job.get_job_state", return_value=None), \
         mock.patch("app.services.l3.grade.job._upsert_job_status"), \
         mock.patch("app.services.l3.grade.job._upsert_grade_row",
                    side_effect=lambda tid, key, h, gj, cube: rows.__setitem__(key, gj)), \
         mock.patch("app.services.l3.grade.job.fetch_color_stats", return_value={}), \
         mock.patch("app.services.l3.grade.job.measure_span", side_effect=fake_measure_span), \
         mock.patch("app.services.l3.grade.job.ensure_cube_file", return_value=None), \
         mock.patch("app.services.l3.grade.scene_meta.lookup_shot_cut_meta", return_value={}), \
         mock.patch("app.services.l3.store.latest_document", return_value=(doc, 1), create=True):
        grade_job.run_grade_job("thread-no-over-group")

    assert set(rows) == {"s0", "s1"}
    assert rows["s0"]["cdl"] == rows["s1"]["cdl"], rows   # no Match/Balance contribution for either
    print("ok  job: an empty join + RGB-far shots does not over-group (no forced match)")


def test_run_grade_job_scene_join_fail_open_on_db_error():
    """Mirrors test_run_grade_job_records_error_never_crashes's spirit but
    expects SUCCESS: the join must fail open (lookup_shot_cut_meta itself
    never raises -- this guards run_grade_job in case a future change to
    it does), and the RGB base still groups shots that are close in color
    even with the join unavailable."""
    doc = {
        "timeline": [
            {"seg_id": "s0", "file_id": "f0", "in_ms": 0, "out_ms": 1000},
            {"seg_id": "s1", "file_id": "f1", "in_ms": 0, "out_ms": 1000},
        ],
        "operations": [], "look": {},
    }

    def fake_measure_span(file_id, in_ms, out_ms, *, hero_ts_ms=None, subject_box=None):
        return {"black_point": 0.02, "white_point": 0.9,
               "mid_gray": 0.3 if file_id == "f0" else 0.5,
               "rgb_mean": [0.4, 0.4, 0.4], "rgb_std": [0.1, 0.1, 0.1]}   # close RGB, both shots

    rows = {}
    with mock.patch("app.services.l3.grade.job.get_job_state", return_value=None), \
         mock.patch("app.services.l3.grade.job._upsert_job_status"), \
         mock.patch("app.services.l3.grade.job._upsert_grade_row",
                    side_effect=lambda tid, key, h, gj, cube: rows.__setitem__(key, gj)), \
         mock.patch("app.services.l3.grade.job.fetch_color_stats", return_value={}), \
         mock.patch("app.services.l3.grade.job.measure_span", side_effect=fake_measure_span), \
         mock.patch("app.services.l3.grade.job.ensure_cube_file", return_value=None), \
         mock.patch("app.services.l3.cuts_v3_read.latest_run_for_files", side_effect=RuntimeError("db down")), \
         mock.patch("app.services.l3.store.latest_document", return_value=(doc, 1), create=True):
        grade_job.run_grade_job("thread-fail-open")   # must not raise, must succeed

    assert set(rows) == {"s0", "s1"}
    non_identity = {k for k, gj in rows.items() if Grade.from_dict(gj["cdl"]) != Grade()}
    assert non_identity, rows
    print("ok  job: the cut_records join fails open on a DB error -- RGB base still groups, job still succeeds")


# --------------------------------------------------------------------------
# color_subject_exposure.plan.md: subject_box join + Leveling subject
# convergence (test 1 -- synthetic frame + degenerate box -- is already
# covered by test_measure_subject_luma_reads_the_box_not_the_whole_frame /
# test_measure_subject_luma_none_for_degenerate_box above; measure_span.py
# is unchanged by this plan)
# --------------------------------------------------------------------------

def test_two_subjects_converge_after_leveling():
    """Two shots, equal whole-frame mid_gray but subject_luma far apart
    (0.25 vs 0.55, same group) must converge toward their working-space
    median under solve_exposure_leveling, bounded by EXPOSURE_CAP_STOPS."""
    from app.services.l3.grade.leveling import EXPOSURE_CAP_STOPS

    target = (0.25 + 0.55) / 2
    a = ShotLevelInput(key="a", mid_gray=0.4, black_point=0.02, white_point=0.9,
                       subject_luma=0.25, target_subject_luma=target)
    b = ShotLevelInput(key="b", mid_gray=0.4, black_point=0.02, white_point=0.9,
                       subject_luma=0.55, target_subject_luma=target)
    deltas = solve_exposure_leveling([a, b])
    assert "a" in deltas and "b" in deltas, deltas

    post_a = a.subject_luma * deltas["a"].slope[0]
    post_b = b.subject_luma * deltas["b"].slope[0]
    pre_spread = abs(b.subject_luma - a.subject_luma)
    post_spread = abs(post_b - post_a)
    assert post_spread < pre_spread, (pre_spread, post_spread)
    cap = 2.0 ** EXPOSURE_CAP_STOPS
    assert 1.0 / cap - 1e-6 <= deltas["a"].slope[0] <= cap + 1e-6
    assert 1.0 / cap - 1e-6 <= deltas["b"].slope[0] <= cap + 1e-6
    print(f"ok  leveling: two far-apart subjects converge (spread {pre_spread:.3f} -> {post_spread:.3f}, "
         f"bounded by the {EXPOSURE_CAP_STOPS}-stop cap)")


def test_no_subject_target_identical_to_today():
    """A shot with no subject_luma/target_subject_luma at all behaves
    exactly as before this plan (target_mid_gray, else the smooth target) --
    and resolve_clip_grade with no subject_box produces the same grade_hash
    whether or not grade_subject_exposure logic exists in the codebase
    (since balance_delta/match_delta/leveling_delta are just None here)."""
    plain = [ShotLevelInput(key=f"s{i}", mid_gray=mg, black_point=0.02, white_point=0.95)
            for i, mg in enumerate([0.4, 0.55, 0.3, 0.5, 0.35])]
    deltas_a = solve_leveling(plain)
    deltas_b = solve_leveling(plain)   # deterministic -- re-running must match exactly
    assert deltas_a == deltas_b

    cs = _cs(black_point=0.06, white_point=0.85, mid_gray=0.38)
    g1 = resolve_clip_grade({}, color_stats=cs)
    g2 = resolve_clip_grade({}, color_stats=cs)   # no subject_box either call
    assert g1["grade_hash"] == g2["grade_hash"]
    print("ok  leveling/resolver: no subject signal at all -> identical, deterministic output")


def test_silhouette_subject_ignores_target_subject_luma_too():
    """The SILHOUETTE_RATIO gate governs regardless of whether an explicit
    target_subject_luma is set -- a silhouette shot must still fall back to
    target_mid_gray/the smooth target, never leveled toward a subject
    target computed from a signal the gate itself distrusts."""
    silhouette = ShotLevelInput(
        key="s0", mid_gray=0.5, black_point=0.02, white_point=0.95,
        subject_luma=0.5 / (SILHOUETTE_RATIO * 2),   # well past the gate
        target_subject_luma=0.5, target_mid_gray=0.5,
    )
    deltas = solve_exposure_leveling([silhouette, ShotLevelInput(
        key="s1", mid_gray=0.5, black_point=0.02, white_point=0.95, target_mid_gray=0.5,
    )])
    # both shots are already AT their target_mid_gray (0.5) -- if the
    # silhouette's subject_luma were (wrongly) used, it would need a large
    # gain toward 0.5; since it isn't, neither shot needs a nudge.
    assert "s0" not in deltas, deltas
    print("ok  leveling: SILHOUETTE_RATIO gate overrides an explicit target_subject_luma too")


def test_scene_meta_join_empty_result_has_no_subject_box():
    """No covering run at all (the DB join fails open) -- lookup_shot_cut_meta
    returns an empty dict, never raises; a caller's `.get(key)` is None for
    every shot, so no subject_box reaches measure_span."""
    with mock.patch("app.services.l3.cuts_v3_read.latest_run_for_files", return_value=None):
        result = lookup_shot_cut_meta([("s0", "f0", 0, 1000), ("s1", "f1", 0, 1000)])
    assert result == {}
    assert result.get("s0") is None and result.get("s1") is None
    print("ok  scene_meta: no covering run -> empty result, no subject_box, no crash")


def test_scene_meta_invalid_subject_box_rejected():
    """A malformed framing.subject_box (wrong length, non-finite, or
    degenerate w/h) is rejected -- the shot gets subject_box=None rather
    than a value that would crash _measure_subject_luma downstream."""
    rows = [
        {"file_id": "f0", "src_in_ms": 0, "src_out_ms": 1000, "label": "", "summary": "",
         "on_camera": None, "speaker_person": None, "voice_ids": [], "take_group_id": None,
         "sync_group_id": None, "framing": {"subject_box": [0.2, 0.3, 0.0, 0.5]}},   # w=0
        {"file_id": "f0", "src_in_ms": 1000, "src_out_ms": 2000, "label": "", "summary": "",
         "on_camera": None, "speaker_person": None, "voice_ids": [], "take_group_id": None,
         "sync_group_id": None, "framing": {"subject_box": [0.2, 0.3, 0.5]}},         # wrong length
        {"file_id": "f0", "src_in_ms": 2000, "src_out_ms": 3000, "label": "", "summary": "",
         "on_camera": None, "speaker_person": None, "voice_ids": [], "take_group_id": None,
         "sync_group_id": None, "framing": {"subject_box": [0.2, 0.3, 0.4, 0.5]}},    # valid
    ]
    with mock.patch("app.services.l3.cuts_v3_read.latest_run_for_files", return_value="run-1"), \
         mock.patch("app.services.l3.cuts_v3_read.rows_for_run", return_value=rows):
        result = lookup_shot_cut_meta([
            ("bad-w", "f0", 0, 1000), ("bad-len", "f0", 1000, 2000), ("ok", "f0", 2000, 3000),
        ])
    assert result["bad-w"].subject_box is None, result["bad-w"]
    assert result["bad-len"].subject_box is None, result["bad-len"]
    assert result["ok"].subject_box == [0.2, 0.3, 0.4, 0.5], result["ok"]
    print("ok  scene_meta: a malformed framing.subject_box is rejected, fail-open")


def test_scene_meta_join_populates_hero_ts_ms():
    """cut_records.hero_ts_ms (100% populated, source-time axis) is joined
    alongside subject_box -- verified live that NO real timeline seg carries
    its own hero_ts_ms, so without this the whole subject_luma chain would
    stay inert regardless of a valid box."""
    rows = [
        {"file_id": "f0", "src_in_ms": 0, "src_out_ms": 2000, "label": "", "summary": "",
         "on_camera": None, "speaker_person": None, "voice_ids": [], "take_group_id": None,
         "sync_group_id": None, "framing": {"subject_box": [0.2, 0.3, 0.4, 0.5]}, "hero_ts_ms": 900},
    ]
    with mock.patch("app.services.l3.cuts_v3_read.latest_run_for_files", return_value="run-1"), \
         mock.patch("app.services.l3.cuts_v3_read.rows_for_run", return_value=rows):
        result = lookup_shot_cut_meta([("s0", "f0", 0, 2000)])
    assert result["s0"].hero_ts_ms == 900, result["s0"]
    print("ok  scene_meta: cut_records.hero_ts_ms is joined alongside subject_box")


def test_run_grade_job_hero_ts_ms_fallback_only_when_box_resolves():
    """job.py's hero_ts_ms fallback (joined from cut_records) must ONLY
    apply when a subject_box is actually being resolved for that shot --
    never for a shot with no box, so a document not using this feature
    never has measure_span's SAMPLE POINTS shifted (hero_ts_ms reorders
    timestamps[0] even without a box)."""
    doc = {
        "timeline": [
            {"seg_id": "s0", "file_id": "f0", "in_ms": 0, "out_ms": 2000},
            {"seg_id": "s1", "file_id": "f1", "in_ms": 0, "out_ms": 2000},
        ],
        "operations": [], "look": {},
    }
    joined = {
        "s0": ShotCutMeta(subject_box=[0.3, 0.3, 0.4, 0.4], hero_ts_ms=900),
        "s1": ShotCutMeta(subject_box=None, hero_ts_ms=900),   # no box -- fallback must NOT fire
    }
    seen = {}

    def fake_measure_span(file_id, in_ms, out_ms, *, hero_ts_ms=None, subject_box=None):
        seen[file_id] = hero_ts_ms
        return _cs(mid_gray=0.4)

    with mock.patch("app.services.l3.grade.job.get_job_state", return_value=None), \
         mock.patch("app.services.l3.grade.job._upsert_job_status"), \
         mock.patch("app.services.l3.grade.job._upsert_grade_row"), \
         mock.patch("app.services.l3.grade.job.fetch_color_stats", return_value={}), \
         mock.patch("app.services.l3.grade.job.measure_span", side_effect=fake_measure_span), \
         mock.patch("app.services.l3.grade.job.ensure_cube_file", return_value=None), \
         mock.patch("app.services.l3.grade.scene_meta.lookup_shot_cut_meta", return_value=joined), \
         mock.patch("app.services.l3.store.latest_document", return_value=(doc, 1), create=True):
        grade_job.run_grade_job("thread-hero-fallback")

    assert seen["f0"] == 900, seen        # box resolved -> fallback fires
    assert seen["f1"] is None, seen       # no box -> fallback must not fire, sampling unchanged
    print("ok  job: the joined hero_ts_ms fallback only fires when a subject_box actually resolves")


def test_run_grade_job_subject_exposure_converges_grouped_subjects():
    """End-to-end (mocked): a joined subject_box reaches measure_span, and
    two grouped shots' far-apart subject lumas converge toward each other."""
    doc = {
        "timeline": [
            {"seg_id": "s0", "file_id": "f0", "in_ms": 0, "out_ms": 1000},
            {"seg_id": "s1", "file_id": "f1", "in_ms": 0, "out_ms": 1000},
        ],
        "operations": [], "look": {},
    }
    joined = {
        "s0": ShotCutMeta(sync_group_id="sync-X", subject_box=[0.3, 0.3, 0.4, 0.4]),
        "s1": ShotCutMeta(sync_group_id="sync-X", subject_box=[0.3, 0.3, 0.4, 0.4]),
    }
    boxes_seen = []

    def fake_measure_span(file_id, in_ms, out_ms, *, hero_ts_ms=None, subject_box=None):
        boxes_seen.append(subject_box)
        return {"black_point": 0.02, "white_point": 0.9, "mid_gray": 0.5,
               "rgb_mean": [0.4, 0.4, 0.4], "rgb_std": [0.1, 0.1, 0.1],
               "subject_luma": 0.35 if file_id == "f0" else 0.62}

    rows = {}
    with mock.patch("app.services.l3.grade.job.get_job_state", return_value=None), \
         mock.patch("app.services.l3.grade.job._upsert_job_status"), \
         mock.patch("app.services.l3.grade.job._upsert_grade_row",
                    side_effect=lambda tid, key, h, gj, cube: rows.__setitem__(key, gj)), \
         mock.patch("app.services.l3.grade.job.fetch_color_stats", return_value={}), \
         mock.patch("app.services.l3.grade.job.measure_span", side_effect=fake_measure_span), \
         mock.patch("app.services.l3.grade.job.ensure_cube_file", return_value=None), \
         mock.patch("app.services.l3.grade.scene_meta.lookup_shot_cut_meta", return_value=joined), \
         mock.patch("app.services.l3.store.latest_document", return_value=(doc, 1), create=True):
        grade_job.run_grade_job("thread-subject-e2e")

    def roundtrip(dv, grade):
        import numpy as np
        lin = float(tone.to_working(np.array([dv], dtype=np.float32), tone.WORKING_SPACE_V1)[0])
        rgb = np.full(3, lin, dtype=np.float32)
        return float(tone.from_working(apply_cdl(rgb, grade), tone.WORKING_SPACE_V1)[0])

    assert all(b is not None for b in boxes_seen), boxes_seen
    post_a = roundtrip(0.35, Grade.from_dict(rows["s0"]["cdl"]))
    post_b = roundtrip(0.62, Grade.from_dict(rows["s1"]["cdl"]))
    assert abs(post_b - post_a) < abs(0.62 - 0.35), (post_a, post_b)
    print(f"ok  job: grouped subjects converge (0.35/0.62 -> {post_a:.3f}/{post_b:.3f})")


# --------------------------------------------------------------------------
# Phase 2/3: run_grade_job exercises leveling + semantic grouping
# --------------------------------------------------------------------------

def test_run_grade_job_applies_leveling_and_semantic_grouping_when_flagged():
    """color_scene_grouping.plan.md: speaker_person/on_camera live on the
    JOINED cut_record now, never on the raw timeline seg (that's the bug
    this plan fixes) -- so grouping-by-speaker must be exercised by mocking
    `scene_meta.lookup_shot_cut_meta`, not by setting fields on the seg
    dict (those are read into compute_input_hash's payload only, never by
    grouping)."""
    doc = {
        "timeline": [
            {"seg_id": "s0", "file_id": "f1", "in_ms": 0, "out_ms": 2000},
            {"seg_id": "s1", "file_id": "f2", "in_ms": 0, "out_ms": 2000},
        ],
        "operations": [], "look": {},
    }
    call_log = []

    def fake_measure_span(file_id, in_ms, out_ms, *, hero_ts_ms=None, subject_box=None):
        # very different RGB (so RGB-based grouping would isolate them) but
        # jittery mid_gray (so leveling has something to do), same speaker
        # per the mocked cut_records join (so semantic grouping should
        # force a match despite the RGB gap).
        return {"black_point": 0.02, "white_point": 0.9,
               "mid_gray": 0.3 if file_id == "f1" else 0.55,
               "rgb_mean": [0.9, 0.1, 0.1] if file_id == "f1" else [0.1, 0.9, 0.1],
               "rgb_std": [0.1, 0.1, 0.1]}

    joined_meta = {
        "s0": ShotCutMeta(speaker_person="P1", on_camera=True),
        "s1": ShotCutMeta(speaker_person="P1", on_camera=True),
    }

    with mock.patch("app.services.l3.grade.job.get_job_state", return_value=None), \
         mock.patch("app.services.l3.grade.job._upsert_job_status", side_effect=lambda tid, **kw: call_log.append(("status", kw))), \
         mock.patch("app.services.l3.grade.job._upsert_grade_row",
                    side_effect=lambda tid, key, h, gj, cube: call_log.append(("row", key, gj))), \
         mock.patch("app.services.l3.grade.job.fetch_color_stats", return_value={}), \
         mock.patch("app.services.l3.grade.job.measure_span", side_effect=fake_measure_span), \
         mock.patch("app.services.l3.grade.job.ensure_cube_file", return_value=None), \
         mock.patch("app.services.l3.grade.scene_meta.lookup_shot_cut_meta", return_value=joined_meta), \
         mock.patch("app.services.l3.store.latest_document", return_value=(doc, 1), create=True):
        grade_job.run_grade_job("thread-flags")

    rows = {r[1]: r[2] for r in call_log if r[0] == "row"}
    assert set(rows) == {"s0", "s1"}
    # both shots ended up graded (leveling + semantic-forced match both ran
    # without crashing and produced a real, non-identity result somewhere).
    non_identity = [k for k, gj in rows.items() if Grade.from_dict(gj["cdl"]) != Grade()]
    assert non_identity, rows
    print("ok  job: run_grade_job runs leveling + semantic grouping")


# --------------------------------------------------------------------------
# color_skin_vibrance.plan.md: bounded global vibrance + skin-anchored tint
# correction (both v1-only, gated on skin_vibrance)
# --------------------------------------------------------------------------

def test_vibrance_boosts_low_chroma_bounded():
    g = solve_correct_grade(_cs(chroma_mean=10.0), pipeline="v1", skin_vibrance=True)
    assert 1.0 < g.sat <= SAT_BOOST_MAX, g.sat
    g_very_low = solve_correct_grade(_cs(chroma_mean=0.5), pipeline="v1", skin_vibrance=True)
    assert g_very_low.sat == SAT_BOOST_MAX, g_very_low.sat   # hard cap, not an uncapped ratio
    print("ok  correct: vibrance boosts low-chroma footage, bounded by SAT_BOOST_MAX")


def test_vibrance_no_desaturation_on_vivid_footage():
    at_target = solve_correct_grade(_cs(chroma_mean=TARGET_CHROMA), pipeline="v1", skin_vibrance=True)
    assert at_target.sat == 1.0, at_target.sat
    above_target = solve_correct_grade(_cs(chroma_mean=40.0), pipeline="v1", skin_vibrance=True)
    assert above_target.sat == 1.0, above_target.sat
    print("ok  correct: vibrance never desaturates already-vivid footage (sat==1.0 at/above target chroma)")


def test_vibrance_missing_chroma_is_identity():
    g = solve_correct_grade(_cs(), pipeline="v1", skin_vibrance=True)
    assert g.sat == 1.0, g.sat
    print("ok  correct: vibrance is identity (sat=1.0) when chroma_mean is missing (fail-open)")


def test_vibrance_flag_off_and_legacy_sat_is_one():
    g_off = solve_correct_grade(_cs(chroma_mean=5.0), pipeline="v1", skin_vibrance=False)
    assert g_off.sat == 1.0, g_off.sat
    g_legacy = solve_correct_grade(_cs(chroma_mean=5.0), pipeline="legacy", skin_vibrance=True)
    assert g_legacy.sat == 1.0, g_legacy.sat
    print("ok  correct: vibrance is inert with the flag off or under the legacy pipeline")


def _on_locus(L, r):
    """A skin Lab sample exactly ON the skin locus (d_perp=0) at radius `r`
    -- the "no cast, just warmth+saturation" fixture the tint tests build on."""
    theta = math.radians(SKIN_LOCUS_DEG)
    return [L, r * math.cos(theta), r * math.sin(theta)]


def _push_perp(lab, d):
    """Displace a Lab sample by `d` along the axis PERPENDICULAR to the skin
    locus -- injects a controlled tint cast without touching along-locus
    warmth or L*."""
    theta = math.radians(SKIN_LOCUS_DEG)
    L, a, b = lab
    return [L, a - d * math.sin(theta), b + d * math.cos(theta)]


def test_skin_tint_corrects_green_cast_bounded():
    skin_on_locus = _on_locus(60.0, 20.0)
    skin_green = _push_perp(skin_on_locus, 12.0)   # a confident, in-gate tint cast
    m = _skin_multiplier(skin_green)
    assert m is not None, "a confident skin sample must not be gated out"
    # Removing a green cast means turning DOWN the green channel relative to
    # red/blue -- the direction a fluorescent/mixed-light green tint needs.
    assert m[1] < m[0] and m[1] < m[2], m
    for v in m:
        assert 1.0 / WB_MULTIPLIER_CLAMP <= v <= WB_MULTIPLIER_CLAMP, m
    print("ok  correct: skin tint correction pulls a green cast toward the locus, bounded by WB_MULTIPLIER_CLAMP")


def test_skin_tint_preserves_warmth_and_tone():
    warm_on_locus = _on_locus(55.0, 35.0)    # a saturated, warm (golden-hour) skin sample
    dark_on_locus = _on_locus(30.0, 15.0)    # a darker skin tone, same locus
    m_warm = _skin_multiplier(warm_on_locus)
    m_dark = _skin_multiplier(dark_on_locus)
    for m in (m_warm, m_dark):
        assert m is not None
        for v in m:
            assert abs(v - 1.0) < 1e-6, m   # on-locus -> no correction, regardless of warmth or tone
    print("ok  correct: on-locus skin (any warmth, any tone) is left unchanged -- no privileging")


def test_skin_tint_skips_non_skin():
    baseline = _solve_wb(_cs(), None, skin_lab=None)
    too_dark = _solve_wb(_cs(), None, skin_lab=[10.0, 10.0, 10.0])          # L below SKIN_L_MIN
    too_bright = _solve_wb(_cs(), None, skin_lab=[95.0, 10.0, 10.0])        # L above SKIN_L_MAX
    near_neutral = _solve_wb(_cs(), None, skin_lab=[60.0, 1.0, 1.0])        # chroma below SKIN_MIN_CHROMA
    colored_object = _solve_wb(_cs(), None, skin_lab=_push_perp(_on_locus(60.0, 20.0), 30.0))  # d_perp too big
    assert too_dark == baseline
    assert too_bright == baseline
    assert near_neutral == baseline
    assert colored_object == baseline
    print("ok  correct: a non-skin sample (bad lightness/near-neutral/too-far-off-locus) casts no WB vote")


def test_skin_prefers_subject_lab_over_center_proxy():
    skin_center = _push_perp(_on_locus(60.0, 20.0), 12.0)     # center-proxy: pushed one way
    subject_lab = _push_perp(_on_locus(60.0, 20.0), -12.0)    # face-region: pushed the OTHER way
    both = solve_correct_grade(_cs(skin_lab=skin_center, subject_lab=subject_lab), pipeline="v1", skin_vibrance=True)
    subject_only = solve_correct_grade(_cs(subject_lab=subject_lab), pipeline="v1", skin_vibrance=True)
    skin_only = solve_correct_grade(_cs(skin_lab=skin_center), pipeline="v1", skin_vibrance=True)
    assert both.slope == subject_only.slope, (both.slope, subject_only.slope)
    assert both.slope != skin_only.slope, "subject_lab must win over the center-proxy skin_lab, not average with it"
    print("ok  correct: when both are present, the face-region subject_lab drives the correction, not skin_lab")


def test_lab_to_srgb_round_trips():
    colors = [
        (0.5, 0.5, 0.5), (0.8, 0.6, 0.5), (0.3, 0.2, 0.15),   # incl. two skin-ish tones
        (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.1, 0.1, 0.9),
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0),
    ]
    for rgb in colors:
        back = lab_to_srgb(srgb_to_lab(rgb))
        for a, b in zip(rgb, back):
            assert abs(a - b) < 1e-4, (rgb, back)
    print("ok  colorspace: lab_to_srgb(srgb_to_lab(rgb)) round-trips within tolerance")


def test_white_reference_still_wins_over_skin():
    white_ref = (0.5, 0.5, 0.5)   # perfectly neutral, verified
    skin_lab = _push_perp(_on_locus(60.0, 20.0), 12.0)
    with_skin = solve_correct_grade(
        _cs(skin_lab=skin_lab), pipeline="v1", skin_vibrance=True, white_reference_rgb=white_ref,
    )
    without_skin = solve_correct_grade(_cs(), pipeline="v1", white_reference_rgb=white_ref)
    assert with_skin.slope == without_skin.slope, (with_skin.slope, without_skin.slope)
    print("ok  correct: a verified white_reference still overrides the skin vote")


def test_measure_subject_lab_reads_the_box_not_the_whole_frame():
    import numpy as np
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = [20, 20, 20]              # dark neutral background
    frame[40:60, 40:60] = [200, 150, 120]   # a warm, skin-ish box
    lab = _measure_subject_lab(frame, (0.4, 0.4, 0.2, 0.2))
    assert lab is not None
    whole_frame_lab = _measure_subject_lab(frame, (0.0, 0.0, 1.0, 1.0))
    assert abs(lab[1] - whole_frame_lab[1]) > 1.0 or abs(lab[2] - whole_frame_lab[2]) > 1.0, (lab, whole_frame_lab)
    print("ok  measure_span: subject_lab reads inside the box, not the whole frame")


def test_measure_subject_lab_none_for_degenerate_box():
    import numpy as np
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert _measure_subject_lab(frame, (0.5, 0.5, 0.0, 0.0)) is None
    assert _measure_subject_lab(frame, (1.5, 1.5, 0.1, 0.1)) is None
    print("ok  measure_span: subject_lab is None for a degenerate/out-of-frame box")


# --------------------------------------------------------------------------
# color_tone_contrast.plan.md: filmic contrast S-curve in tone.from_working
# --------------------------------------------------------------------------

def test_tone_contrast_zero_is_exact_identity():
    import numpy as np
    x = np.array([0.0, 0.18, 0.5, 0.8, 1.0], dtype=np.float32)
    default_call = tone.from_working(x, tone.WORKING_SPACE_V1)
    explicit_zero = tone.from_working(x, tone.WORKING_SPACE_V1, contrast=0.0)
    assert np.array_equal(default_call, explicit_zero)
    print("ok  tone: contrast=0.0 is bit-for-bit identical to the pre-plan from_working")


def test_tone_contrast_endpoints_pinned():
    """`_contrast_pivot` itself (a pure function over DISPLAY 0..1) is exactly
    endpoint-pinned: f(0)=0, f(1)=1 regardless of strength. `from_working`'s
    COMPOSED output is separately checked for never exceeding [0,1] --
    it can't reach display=1.0 for any finite linear input even with
    contrast=0 (the pre-existing shoulder is only asymptotic), so pinning is
    a property of the curve, not of the full round-trip."""
    import numpy as np
    x = np.array([0.0, 1.0], dtype=np.float32)
    for g in (1.3, 1.9, 2.5):
        out = tone._contrast_pivot(x, g)
        assert abs(float(out[0])) < 1e-6, (g, out)
        assert abs(float(out[1]) - 1.0) < 1e-6, (g, out)
    for c in (0.3, 0.9, 1.5):
        vals = tone.from_working(
            np.array([0.0, 0.5, 0.8, 1.0, 3.0, 50.0], dtype=np.float32), tone.WORKING_SPACE_V1, contrast=c,
        )
        assert float(vals.min()) >= -1e-6 and float(vals.max()) <= 1.0 + 1e-6, (c, vals)
    print("ok  tone: contrast curve is exactly endpoint-pinned; from_working never clips past [0,1]")


def test_tone_contrast_monotonic():
    import numpy as np
    sweep = np.linspace(0, 1, 500).astype(np.float32)
    for c in (0.3, 0.9, 1.5):
        out = tone.from_working(sweep, tone.WORKING_SPACE_V1, contrast=c)
        assert bool(np.all(np.diff(out) >= -1e-6)), c
    print("ok  tone: contrast curve is monotonic non-decreasing at every strength")


def test_tone_contrast_increases_midtone_slope():
    """A real S-shape: pick LINEAR (working-space) inputs whose contrast=0
    DISPLAY output lands clearly below/above TONE_PIVOT (the shoulder+OETF
    between from_working's linear input and its display output means a
    linear value equal to TONE_PIVOT does NOT itself land on the pivot --
    verified via the off/off fixture-sanity asserts below), then verify
    contrast=0.9 pushes the below-pivot point further DOWN and the
    above-pivot point further UP. The pivot-fixed invariant itself is
    checked directly on `_contrast_pivot` (a pure display-space function),
    not through this round-trip."""
    import numpy as np
    x = np.array([0.05, 0.5], dtype=np.float32)
    off = tone.from_working(x, tone.WORKING_SPACE_V1, contrast=0.0)
    on = tone.from_working(x, tone.WORKING_SPACE_V1, contrast=0.9)
    assert off[0] < tone.TONE_PIVOT < off[1], off   # fixture sanity
    assert on[0] < off[0], (on[0], off[0])     # below pivot moves down
    assert on[1] > off[1], (on[1], off[1])     # above pivot moves up
    for g in (1.3, 1.9, 2.5):
        at_pivot = tone._contrast_pivot(np.array([tone.TONE_PIVOT], dtype=np.float32), g)
        assert abs(float(at_pivot[0]) - tone.TONE_PIVOT) < 1e-5, (g, at_pivot)   # pivot itself is fixed
    print("ok  tone: contrast increases midtone slope (below moves down, above moves up, pivot fixed)")


def test_tone_contrast_legacy_and_nonv1_identity():
    import numpy as np
    x = np.array([0.0, 0.18, 0.5, 0.8, 1.0], dtype=np.float32)
    for ws in ("rec709_legacy", "rec709"):
        out = tone.from_working(x, ws, contrast=0.9)
        assert np.array_equal(out, x), (ws, out)
    print("ok  tone: contrast is inert for legacy/non-v1 working_space, regardless of strength")


def test_bake_parity_with_contrast():
    """Step 1.1's parity acceptance test, extended: direct-compute must still
    match the trilinearly-sampled baked cube WITH the contrast curve on."""
    import numpy as np
    from app.services.l3.grade.lut_bake import _sample_lut_trilinear, parse_cube_text

    grade = Grade(slope=(1.15, 1.05, 0.9), offset=(0.02, 0.0, -0.01))
    size = 33
    cube_text = bake_cube_text(grade, size=size, working_space=tone.WORKING_SPACE_V1, tone_contrast=0.9)
    grid, parsed_size = parse_cube_text(cube_text)
    assert parsed_size == size

    probes = np.array([
        [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5],
        [0.18, 0.18, 0.18], [0.9, 0.4, 0.2], [0.05, 0.6, 0.95],
    ], dtype=np.float32)
    direct = tone.from_working(apply_cdl(tone.to_working(probes, tone.WORKING_SPACE_V1), grade),
                               tone.WORKING_SPACE_V1, contrast=0.9)
    sampled = _sample_lut_trilinear(grid, probes)
    max_err = float(np.max(np.abs(direct - sampled)))
    assert max_err < 0.02, f"direct-vs-baked-cube parity (with contrast) exceeded tolerance: {max_err}"
    print(f"ok  lut_bake: v1 direct-vs-baked-cube parity holds WITH the contrast curve (max err {max_err:.4f})")


def test_resolver_flag_off_byte_identical():
    cs = _cs(black_point=0.06, white_point=0.85, mid_gray=0.38)
    default_call = resolve_clip_grade({}, color_stats=cs)
    explicit_zero = resolve_clip_grade({}, color_stats=cs, tone_contrast=0.0)
    assert default_call == explicit_zero
    assert default_call["grade_hash"] == explicit_zero["grade_hash"]
    print("ok  resolver: tone_contrast=0.0 (default) is byte-identical, same grade_hash")


def test_grade_hash_changes_with_tone_contrast():
    g = Grade(slope=(1.1, 1.0, 0.95))
    h0 = grade_hash(g, working_space=tone.WORKING_SPACE_V1, tone_contrast=0.0)
    h1 = grade_hash(g, working_space=tone.WORKING_SPACE_V1, tone_contrast=0.9)
    assert h0 != h1, "grade_hash must change when tone_contrast changes, or the cube never rebakes"
    print("ok  cdl: grade_hash changes with tone_contrast (cube correctly rebakes)")


def test_compositor_grade_key_distinguishes_tone_contrast_from_identity():
    """Regression for a real gap found while implementing this plan:
    render/compositor.py's segment-cache identity-collapse only checked the
    CDL, not tone_contrast -- an identity-CDL shot with the contrast curve on
    would collapse to the SAME '' cache token as a truly-identity shot
    rendered before the flag existed, silently reusing a stale segment clip
    that never got the curve. tone_contrast>0 must break the collapse."""
    identity_cdl = Grade().to_dict()
    no_contrast = {"cdl": identity_cdl, "working_space": tone.WORKING_SPACE_V1,
                   "tone_contrast": 0.0, "grade_hash": "aaa"}
    with_contrast = {"cdl": identity_cdl, "working_space": tone.WORKING_SPACE_V1,
                      "tone_contrast": 0.9, "grade_hash": "bbb"}
    assert _grade_key(no_contrast) == ""
    assert _grade_key(with_contrast) != ""
    assert _grade_key(with_contrast) != _grade_key(no_contrast)
    print("ok  compositor: _grade_key does not collapse an identity-CDL + tone_contrast>0 grade to ''")


# --------------------------------------------------------------------------
# color_response_engine.plan.md: parametric Look engine (mode == "engine")
# --------------------------------------------------------------------------

# color_look_library.plan.md dropped the engine_punchy/engine_film
# validation-only catalog entries in favor of the real library -- resolved
# from the real catalog (not hardcoded) so these tests never drift from
# whatever punchy_vibrant/kodak_2383 actually tune to.
_PUNCHY_SPEC = get_engine_look("punchy_vibrant").spec


def test_look_identity_spec_is_identity_grid():
    grid, size = build_look_grid(LookSpec(), size=17)
    ident = _identity_grid(17)
    assert size == 17
    assert bool((grid == ident).all()), float((grid - ident).max())
    print("ok  look_engine: an empty LookSpec builds the EXACT identity grid")


def test_look_grid_deterministic():
    g1, _ = build_look_grid(_PUNCHY_SPEC, size=17)
    g2, _ = build_look_grid(_PUNCHY_SPEC, size=17)
    assert g1.tobytes() == g2.tobytes()
    print("ok  look_engine: the same spec builds a byte-identical grid twice")


def test_look_grid_no_clip_no_nan():
    import numpy as np

    strong = LookSpec(
        contrast=0.5,
        shadow_tint=(-0.08, 0.02, 0.09), mid_tint=(0.02, -0.01, 0.0), highlight_tint=(0.09, 0.03, -0.07),
        hue_rotate=((30.0, 40.0, 40.0), (200.0, 60.0, -30.0)),
        hue_sat=((30.0, 40.0, 1.4), (140.0, 50.0, 0.4)),
        sat=1.3,
    )
    grid, size = build_look_grid(strong, size=33)
    assert np.isfinite(grid).all(), "NaN/inf in a strong look's grid"
    assert grid.min() >= 0.0 and grid.max() <= 1.0, (grid.min(), grid.max())
    diag = np.array([grid[i, i, i] for i in range(size)])   # neutral ramp (r=g=b)
    luma = diag[:, 0] * 0.2126 + diag[:, 1] * 0.7152 + diag[:, 2] * 0.0722
    assert bool(np.all(np.diff(luma) >= -1e-6)), luma
    print("ok  look_engine: a strong spec never clips/NaNs; the neutral ramp stays monotonic")


def test_look_split_tone_directional():
    import numpy as np

    dark_gray = np.array([[0.05, 0.05, 0.05]], dtype=np.float32)
    light_gray = np.array([[0.95, 0.95, 0.95]], dtype=np.float32)
    shadow_spec = LookSpec(shadow_tint=(0.0, 0.0, 0.08))
    d_shift = float(_apply_split_tone(dark_gray, shadow_spec)[0, 2] - dark_gray[0, 2])
    l_shift = float(_apply_split_tone(light_gray, shadow_spec)[0, 2] - light_gray[0, 2])
    assert d_shift > 0.05, d_shift          # dark gray bluer
    assert l_shift < 0.001, l_shift          # light gray ~unchanged (w_shadow ~0 there)

    highlight_spec = LookSpec(highlight_tint=(0.0, 0.0, 0.08))
    d2_shift = float(_apply_split_tone(dark_gray, highlight_spec)[0, 2] - dark_gray[0, 2])
    l2_shift = float(_apply_split_tone(light_gray, highlight_spec)[0, 2] - light_gray[0, 2])
    assert l2_shift > 0.05, l2_shift         # the inverse: light gray bluer
    assert d2_shift < 0.001, d2_shift        # dark gray ~unchanged
    print("ok  look_engine: split-tone tints are zone-directional (shadow vs highlight, inverse)")


def test_look_hue_rotate_only_targets_band():
    import numpy as np

    orange = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)    # hue ~30
    blue = np.array([[0.0, 0.2, 1.0]], dtype=np.float32)       # hue ~228, well outside the band
    gray = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    spec = LookSpec(hue_rotate=((30.0, 40.0, 25.0),))

    h_before, _, _ = _rgb_to_hsv(orange)
    h_after, _, _ = _rgb_to_hsv(_apply_hue_rotate(orange, spec))
    assert abs(float(h_after[0] - h_before[0]) - 25.0) < 0.5, (h_before, h_after)   # band center: full rotate_deg

    hb_before, _, _ = _rgb_to_hsv(blue)
    hb_after, _, _ = _rgb_to_hsv(_apply_hue_rotate(blue, spec))
    assert abs(float(hb_after[0] - hb_before[0])) < 0.5, (hb_before, hb_after)

    gray_out = _apply_hue_rotate(gray, spec)
    assert np.array_equal(gray_out, gray), (gray, gray_out)   # achromatic guard: EXACT, not approximate
    print("ok  look_engine: hue_rotate only moves the targeted band; grays are exactly unchanged")


def test_look_hue_sat_only_targets_band():
    import numpy as np

    green = np.array([[0.1, 0.8, 0.1]], dtype=np.float32)     # hue ~113, inside a (140,50) band
    orange = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)     # hue ~30, well outside
    gray = np.array([[0.5, 0.5, 0.5]], dtype=np.float32)
    spec = LookSpec(hue_sat=((140.0, 50.0, 0.5),))

    _, sg_before, _ = _rgb_to_hsv(green)
    _, sg_after, _ = _rgb_to_hsv(_apply_hue_sat(green, spec))
    assert float(sg_after[0]) < float(sg_before[0]) * 0.9, (sg_before, sg_after)   # meaningfully desaturated

    _, so_before, _ = _rgb_to_hsv(orange)
    _, so_after, _ = _rgb_to_hsv(_apply_hue_sat(orange, spec))
    assert abs(float(so_after[0] - so_before[0])) < 0.01, (so_before, so_after)

    gray_out = _apply_hue_sat(gray, spec)
    assert np.array_equal(gray_out, gray), (gray, gray_out)   # achromatic guard: EXACT
    print("ok  look_engine: hue_sat only scales saturation in the targeted band; grays are exactly unchanged")


def test_look_engine_bake_parity():
    """Proves preview == export for engine looks: sampling the grid DIRECTLY
    must match trilinearly sampling the SAME grid round-tripped through
    bake_cube_text -> parse_cube_text (the real preview/export path)."""
    import numpy as np

    grid, size = build_look_grid(_PUNCHY_SPEC, size=33)
    cube_text = bake_cube_text(Grade(), size=size, creative_lut_grid=(grid, size), working_space="rec709")
    parsed_grid, parsed_size = parse_cube_text(cube_text)
    assert parsed_size == size

    probes = np.array([
        [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5],
        [0.9, 0.4, 0.2], [0.05, 0.6, 0.95],
    ], dtype=np.float32)
    direct = _sample_lut_trilinear(grid, probes)
    baked = _sample_lut_trilinear(parsed_grid, probes)
    max_err = float(np.max(np.abs(direct - baked)))
    assert max_err < 0.01, f"engine-look bake parity exceeded tolerance: {max_err}"
    print(f"ok  look_engine: baked-cube trilinear sample matches the direct grid (max err {max_err:.5f})")


def test_resolver_engine_sets_look_engine_and_changes_hash():
    cs = _cs(black_point=0.06, white_point=0.85, mid_gray=0.38)
    baseline = resolve_clip_grade({}, color_stats=cs)

    seq = {"mode": "engine", "look_id": "punchy_vibrant"}
    on = resolve_clip_grade({}, color_stats=cs, sequence_look=seq)
    assert on.get("look_engine"), on.get("look_engine")
    assert on["grade_hash"] != baseline["grade_hash"]
    assert on.get("creative_lut_ref") is None   # engine + uploaded LUT are mutually exclusive

    # the identity catalog entry stays inert -- an empty spec skips the grid
    # entirely (byte-identical to no look at all).
    seq_identity = {"mode": "engine", "look_id": "engine_identity"}
    identity_on = resolve_clip_grade({}, color_stats=cs, sequence_look=seq_identity)
    assert identity_on.get("look_engine") is None
    assert identity_on["grade_hash"] == baseline["grade_hash"]
    print("ok  resolver: mode='engine' sets look_engine + rehashes; the identity look stays inert")


def test_grade_hash_look_engine_in_payload():
    h0 = grade_hash(Grade(), working_space=tone.WORKING_SPACE_V1, look_engine=None)
    h1 = grade_hash(Grade(), working_space=tone.WORKING_SPACE_V1, look_engine=_PUNCHY_SPEC.to_dict())
    assert h0 != h1, "grade_hash must change when look_engine changes, or the cube never rebakes per-look"
    print("ok  cdl: grade_hash changes with look_engine (cube correctly rebakes per look)")


def test_compositor_grade_key_distinguishes_look_engine_from_identity():
    """Same class of gap as tone_contrast's: an identity CDL with a
    non-identity look_engine dict is NOT a no-op bake."""
    identity_cdl = Grade().to_dict()
    no_engine = {"cdl": identity_cdl, "working_space": "rec709", "look_engine": None, "grade_hash": "aaa"}
    with_engine = {"cdl": identity_cdl, "working_space": "rec709",
                   "look_engine": _PUNCHY_SPEC.to_dict(), "grade_hash": "bbb"}
    assert _grade_key(no_engine) == ""
    assert _grade_key(with_engine) != ""
    assert _grade_key(with_engine) != _grade_key(no_engine)
    print("ok  compositor: _grade_key does not collapse an identity-CDL + look_engine grade to ''")


def test_resolve_look_spec_prefers_catalog_over_inline_params():
    catalog = resolve_look_spec({"look_id": "punchy_vibrant"})
    assert catalog == _PUNCHY_SPEC

    inline = resolve_look_spec({"look_params": {"sat": 1.2}})
    assert inline is not None and abs(inline.sat - 1.2) < 1e-9

    both = resolve_look_spec({"look_id": "punchy_vibrant", "look_params": {"sat": 1.2}})
    assert both == _PUNCHY_SPEC   # a valid catalog look_id wins over inline params

    unknown = resolve_look_spec({"look_id": "not_a_real_look"})
    assert unknown is None

    nothing = resolve_look_spec({})
    assert nothing is None
    print("ok  look_engine: resolve_look_spec prefers a valid catalog look_id, falls back to look_params, else None")


# --------------------------------------------------------------------------
# halation_grain.plan.md: look-scoped spatial film texture (halation + grain)
# --------------------------------------------------------------------------

def test_lookspec_texture_roundtrip():
    spec = LookSpec(halation=0.4, grain=0.1)
    d = spec.to_dict()
    assert d["halation"] == 0.4 and d["grain"] == 0.1
    back = LookSpec.from_dict(d)
    assert back == spec
    assert not spec.is_identity()
    print("ok  look_engine: halation/grain round-trip to_dict/from_dict; a texture-only spec is not identity")


def test_build_look_grid_ignores_texture():
    import numpy as np

    with_texture, _ = build_look_grid(LookSpec(halation=0.4, grain=0.1, sat=1.2), size=17)
    without_texture, _ = build_look_grid(LookSpec(sat=1.2), size=17)
    assert np.array_equal(with_texture, without_texture)
    print("ok  look_engine: build_look_grid ignores halation/grain -- texture never bakes into the color grid")


def test_resolver_film_texture_populates_soft_local_when_look_declares_it():
    """grade_pipeline_standardize.plan.md: film texture is look-scoped, not
    a global toggle -- it fires purely on the active look declaring nonzero
    halation/grain."""
    cs = _cs(black_point=0.06, white_point=0.85, mid_gray=0.38)
    no_look = resolve_clip_grade({}, color_stats=cs)
    assert no_look.get("soft_local") is None

    seq_film = {"mode": "engine", "look_id": "kodak_2383"}
    on = resolve_clip_grade({}, color_stats=cs, sequence_look=seq_film)
    assert on["soft_local"] == {"halation": {"strength": 0.25}, "grain": {"strength": 0.04}}
    assert on["grade_hash"] != no_look["grade_hash"]

    # a look with color but zero texture (punchy_vibrant) -> soft_local
    # stays None (the look's own params, not a flag, gate this).
    seq_punchy = {"mode": "engine", "look_id": "punchy_vibrant"}
    punchy_on = resolve_clip_grade({}, color_stats=cs, sequence_look=seq_punchy)
    assert punchy_on.get("soft_local") is None
    print("ok  resolver: soft_local.halation/.grain populate exactly when the active look declares them")


def test_grain_ffmpeg_filter():
    assert grain_ffmpeg_filter(None) is None
    assert grain_ffmpeg_filter({"strength": 0.0}) is None
    low = grain_ffmpeg_filter({"strength": 0.1})
    high = grain_ffmpeg_filter({"strength": 0.8})
    assert low is not None and high is not None
    assert low.startswith("noise=alls=") and "allf=t+u" in low

    def _alls(clause):
        return int(clause.split("alls=")[1].split(":")[0])

    assert _alls(high) > _alls(low)   # monotonic in strength
    print("ok  softlocal: grain_ffmpeg_filter is None at zero, a noise=... clause otherwise, monotonic in strength")


def test_halation_subgraph_shape():
    assert halation_ffmpeg_subgraph(None) is None
    assert halation_ffmpeg_subgraph({"strength": 0.0}) is None
    frag = halation_ffmpeg_subgraph({"strength": 0.3}, frame_height=1080)
    assert frag is not None
    assert "split" in frag and "gblur" in frag and "blend=all_mode=screen" in frag
    assert "[hbase]" in frag and "[hglow]" in frag
    # frame-height scaling: a 4K frame blurs more (larger sigma) than a
    # 240p one, both taken from a fixed reference constant.
    small = halation_ffmpeg_subgraph({"strength": 0.3}, frame_height=240)
    large = halation_ffmpeg_subgraph({"strength": 0.3}, frame_height=2160)

    def _sigma(clause):
        return float(clause.split("gblur=sigma=")[1].split("[")[0])

    assert _sigma(large) > _sigma(small)
    print("ok  softlocal: halation_ffmpeg_subgraph is None at zero, a split/gblur/screen-blend graph otherwise")


def test_grade_hash_changes_with_texture():
    h0 = grade_hash(Grade(), working_space=tone.WORKING_SPACE_V1, soft_local=None)
    h1 = grade_hash(Grade(), working_space=tone.WORKING_SPACE_V1,
                    soft_local={"halation": {"strength": 0.25}, "grain": {"strength": 0.04}})
    assert h0 != h1, "grade_hash must change when soft_local texture changes, or nothing re-bakes/re-renders"
    print("ok  cdl: grade_hash changes with soft_local halation/grain (re-bakes/re-renders)")


def test_grade_key_not_collapsed_with_texture():
    """Verifies the plan's own claim that _grade_key's existing identity
    collapse ALREADY covers this (a non-empty soft_local is already
    truthy) -- no compositor.py change was needed for this specific gap,
    unlike tone_contrast/look_engine which each needed a dedicated fix."""
    identity_cdl = Grade().to_dict()
    no_texture = {"cdl": identity_cdl, "working_space": "rec709", "soft_local": None, "grade_hash": "aaa"}
    with_texture = {"cdl": identity_cdl, "working_space": "rec709",
                    "soft_local": {"halation": {"strength": 0.25}}, "grade_hash": "bbb"}
    assert _grade_key(no_texture) == ""
    assert _grade_key(with_texture) != ""
    assert _grade_key(with_texture) != _grade_key(no_texture)
    print("ok  compositor: _grade_key does not collapse an identity-CDL + soft_local.halation grade to ''")


def test_transform_vf_texture_order_and_off_is_untouched():
    """Order (both sides): LUT -> vignette -> halation -> grain. Flag/texture
    off -> byte-identical -vf chain to before this plan (no dedicated flag
    threads into _transform_vf itself; None args are the off state)."""
    cfg = {"width": 640, "height": 360, "fps": 30}
    off = _transform_vf(cfg, None, cube_path=None, vignette_filter=None)
    assert "vignette" not in off and "split" not in off and "noise" not in off

    vignette = "vignette=angle=0.5:x0=w*0.5:y0=h*0.5"
    halation = halation_ffmpeg_subgraph({"strength": 0.3}, frame_height=360)
    grain = grain_ffmpeg_filter({"strength": 0.1})
    on = _transform_vf(cfg, None, cube_path=None, vignette_filter=vignette,
                       halation_filter=halation, grain_filter=grain)
    assert on.index(vignette) < on.index("split=2") < on.index("noise=")
    print("ok  compositor: _transform_vf appends halation then grain after the vignette; off is untouched")


# --------------------------------------------------------------------------
# color_look_library.plan.md: black_lift + negative contrast + the real
# 17-look catalog (16 authored + engine_identity)
# --------------------------------------------------------------------------

def test_black_lift_raises_floor_keeps_white():
    import numpy as np

    grid, size = build_look_grid(LookSpec(black_lift=0.08), size=17)
    assert abs(float(grid.min()) - 0.08) < 1e-4, grid.min()
    assert abs(float(grid.max()) - 1.0) < 1e-4, grid.max()
    diag = np.array([grid[i, i, i] for i in range(size)])
    luma = diag[:, 0] * 0.2126 + diag[:, 1] * 0.7152 + diag[:, 2] * 0.0722
    assert bool(np.all(np.diff(luma) >= -1e-6)), luma
    print("ok  look_engine: black_lift raises the floor to black_lift, keeps white at 1.0, stays monotonic")


def test_negative_contrast_softens_monotonic():
    import numpy as np

    identity_grid, size = build_look_grid(LookSpec(), size=33)
    soft_grid, _ = build_look_grid(LookSpec(contrast=-0.1), size=33)

    def _diag_luma(grid):
        diag = np.array([grid[i, i, i] for i in range(size)])
        return diag[:, 0] * 0.2126 + diag[:, 1] * 0.7152 + diag[:, 2] * 0.0722

    identity_luma = _diag_luma(identity_grid)
    soft_luma = _diag_luma(soft_grid)
    mid = size // 2
    identity_slope = identity_luma[mid + 2] - identity_luma[mid - 2]
    soft_slope = soft_luma[mid + 2] - soft_luma[mid - 2]
    assert soft_slope < identity_slope, (soft_slope, identity_slope)   # softened midtone slope
    assert bool(np.all(np.diff(soft_luma) >= -1e-6)), soft_luma        # still monotonic
    assert abs(float(soft_luma[0])) < 1e-4 and abs(float(soft_luma[-1]) - 1.0) < 1e-4   # endpoints pinned
    print("ok  look_engine: negative contrast softens the midtone slope, stays monotonic, endpoints pinned")


def test_identity_still_exact_with_new_fields():
    grid, size = build_look_grid(LookSpec(), size=17)
    ident = _identity_grid(17)
    assert bool((grid == ident).all()), float((grid - ident).max())
    print("ok  look_engine: LookSpec() (black_lift=0, contrast=0 defaults) still builds the EXACT identity grid")


def test_catalog_all_looks_valid():
    import numpy as np

    ids = [look.look_id for look in LOOKS]
    assert len(ids) == len(set(ids)), f"duplicate look_id in LOOKS: {ids}"
    assert len(LOOKS) >= 17, len(LOOKS)   # engine_identity + 6 creator + 6 film + 4 ad

    for look in LOOKS:
        grid, _ = build_look_grid(look.spec, size=17)
        assert np.isfinite(grid).all(), f"{look.look_id}: non-finite grid"
        assert grid.min() >= 0.0 and grid.max() <= 1.0, f"{look.look_id}: unclamped grid"

    listing = list_engine_looks()
    assert len(listing) == len(LOOKS)
    for entry in listing:
        assert entry["mode"] == "engine", entry
        assert entry["family"] in {"creator", "film", "ad"}, entry
    print(f"ok  look_engine: all {len(LOOKS)} catalog looks are finite/clamped, tagged, no duplicate ids")


def test_bake_parity_with_new_knobs():
    import numpy as np

    spec = LookSpec(contrast=-0.1, black_lift=0.06, sat=0.9)
    grid, size = build_look_grid(spec, size=33)
    cube_text = bake_cube_text(Grade(), size=size, creative_lut_grid=(grid, size), working_space="rec709")
    parsed_grid, parsed_size = parse_cube_text(cube_text)
    assert parsed_size == size

    probes = np.array([
        [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5],
        [0.9, 0.4, 0.2], [0.05, 0.6, 0.95],
    ], dtype=np.float32)
    direct = _sample_lut_trilinear(grid, probes)
    baked = _sample_lut_trilinear(parsed_grid, probes)
    max_err = float(np.max(np.abs(direct - baked)))
    assert max_err < 0.01, f"bake parity (black_lift + negative contrast) exceeded tolerance: {max_err}"
    print(f"ok  look_engine: bake parity holds with black_lift + negative contrast (max err {max_err:.5f})")


def test_film_looks_carry_texture():
    for look in LOOKS:
        has_texture = look.spec.halation > 0.0 or look.spec.grain > 0.0
        if look.family == "film":
            assert has_texture, f"{look.look_id} is family=film but carries no halation/grain"
        elif look.family in ("creator", "ad") and look.look_id != "engine_identity":
            assert look.spec.halation == 0.0, f"{look.look_id} ({look.family}) unexpectedly carries halation"
    print("ok  look_engine: every family=='film' look carries halation/grain; creator/ad looks carry no halation")


# --------------------------------------------------------------------------
# frontend_look_gallery.plan.md: gallery listing carries look_params (for
# live thumbnails) + family; legacy CDL presets stay unchanged
# --------------------------------------------------------------------------

def test_list_engine_looks_has_params_and_family():
    engine_listing = list_engine_looks()
    assert len(engine_listing) == len(LOOKS)
    for entry, look in zip(engine_listing, LOOKS):
        assert entry["mode"] == "engine"
        assert entry["family"] in {"creator", "film", "ad"}, entry
        assert entry["look_id"] == look.look_id
        assert "look_params" in entry, entry
        # round-trips through LookSpec.from_dict without crashing and
        # reproduces the exact spec it was built from
        spec = LookSpec.from_dict(entry["look_params"])
        assert spec == look.spec, (entry["look_id"], spec, look.spec)

    preset_listing = list_presets()
    assert len(preset_listing) > 0
    for entry in preset_listing:
        assert entry["mode"] == "preset"
        assert "family" not in entry, entry          # regression: CDL presets stay unchanged
        assert "look_params" not in entry, entry
    print("ok  look_engine/presets: every engine entry carries look_params + family; CDL presets carry neither")


def main():
    test_tone_legacy_is_exact_identity()
    test_tone_v1_black_stays_black()
    test_tone_v1_never_exceeds_one()
    test_tone_v1_midgray_barely_moves_shadows_untouched()
    test_tone_v1_monotonic()
    test_tone_v1_golden_byte_identical_after_log_v1_added()
    test_tone_log_v1_endpoints_pinned_and_bounded()
    test_tone_log_v1_midgray_decodes_near_true_scene_linear()
    test_tone_log_v1_monotonic()
    test_tone_log_v1_from_working_matches_v1_reencode()
    test_tone_log_v1_black_stays_black()
    test_lut_bake_legacy_unaffected_by_working_space_param()
    test_lut_bake_v1_parity_direct_vs_baked_cube()
    test_lut_bake_v1_differs_from_legacy_for_same_grade()
    test_correct_legacy_untouched_by_pipeline_param()
    test_correct_v1_nudges_mid_gray_toward_target_bounded()
    test_correct_v1_never_worse_on_already_correct_footage()
    test_correct_pre_lift_applies_by_default_for_is_log_flat()
    test_correct_pre_lift_gated_off_under_log_working_space()
    test_correct_pre_lift_gate_is_a_no_op_for_non_log_footage()
    test_v1_grade_does_not_crush_midtones_or_shadows()
    test_v1_composite_slope_and_offset_are_bounded()
    test_v1_composite_offset_floor_protects_a_modest_shadow_crush()
    test_v1_composite_offset_floor_does_not_raise_true_black()
    test_v1_composite_offset_floor_preserves_mid_gray_and_slope_ceiling()
    test_v1_composite_offset_floor_respects_power()
    test_match_two_camera_interview_matches_across_the_cut()
    test_match_never_groups_non_adjacent_shots()
    test_match_same_file_always_groups_regardless_of_rgb()
    test_resolver_v1_sets_v1_working_space()
    test_resolver_explicit_working_space_overrides_pipeline_default()
    test_resolver_is_log_flat_selects_log_working_space()
    test_resolver_non_log_stays_on_v1_working_space()
    test_resolver_no_color_stats_stays_on_v1_working_space()
    test_resolver_explicit_working_space_wins_over_is_log_flat()
    test_resolver_log_clip_produces_a_finite_bounded_cdl()
    test_resolver_non_log_golden_grade_hash_unchanged_by_part2()
    test_resolver_log_flat_grade_hash_differs_from_treating_it_as_v1()
    test_resolver_reference_transfer_v1_does_not_crash_or_blow_up()
    test_resolver_subject_box_seam_carries_through_no_visual_change_by_default()
    test_one_grade_per_shot_no_intra_shot_variance()
    test_layers_no_grade_lookup_falls_back_to_identity()
    test_layers_v1_reads_grade_lookup_hit()
    test_layers_v1_falls_back_to_identity_on_miss()
    test_layers_split_screen_region_preserves_spine_grade()
    test_input_hash_stable_for_identical_documents()
    test_input_hash_changes_when_a_span_trims()
    test_input_hash_changes_when_look_changes()
    test_input_hash_unaffected_by_shot_reorder_being_a_real_change()
    test_ordered_shots_covers_spine_and_place_video_ops_in_order()
    test_run_grade_job_end_to_end_mocked()
    test_run_grade_job_records_error_never_crashes()
    test_run_grade_job_skips_when_already_done_for_current_hash()
    test_leveling_flattens_jittery_brightness()
    test_leveling_preserves_an_intentional_arc()
    test_leveling_exposure_gain_is_bounded()
    test_leveling_tonal_converges_low_contrast_and_punchy()
    test_leveling_tonal_skips_cross_scene_outlier()
    test_leveling_tonal_never_pushes_toward_clipping()
    test_leveling_never_crashes_on_a_true_black_point()
    test_leveling_subject_luma_used_when_not_a_silhouette()
    test_leveling_subject_luma_ignored_when_silhouette()
    test_leveling_composed_result_includes_both_stages()
    test_measure_subject_luma_reads_the_box_not_the_whole_frame()
    test_measure_subject_luma_none_for_degenerate_box()
    test_scene_group_same_speaker_different_file_groups()
    test_scene_group_no_shared_signal_does_not_group()
    test_scene_group_overrides_rgb_grouping_in_solve_sequence_match()
    test_shot_match_falls_back_to_rgb_when_semantic_is_all_singletons()
    test_shot_match_cross_shot_spread_shrinks_after_balance_and_match()
    test_shot_match_no_shadow_crush_after_balance_and_match()
    test_balance_lifts_capped_dull_shot_and_survives_composite_slope_clamp()
    test_shot_match_references_override_matches_every_member_not_just_anchor()
    test_scene_grouping_rgb_base_prevents_all_singletons()
    test_scene_grouping_sync_group_id_groups_across_files()
    test_scene_grouping_take_group_id_groups_across_files()
    test_scene_grouping_voice_ids_overlap_groups_across_files()
    test_scene_meta_join_picks_max_overlap_cut_record()
    test_run_grade_job_groups_multi_file_reel_via_scene_join()
    test_run_grade_job_scene_join_does_not_over_group_unrelated_shots()
    test_run_grade_job_scene_join_fail_open_on_db_error()
    test_two_subjects_converge_after_leveling()
    test_no_subject_target_identical_to_today()
    test_silhouette_subject_ignores_target_subject_luma_too()
    test_scene_meta_join_empty_result_has_no_subject_box()
    test_scene_meta_invalid_subject_box_rejected()
    test_scene_meta_join_populates_hero_ts_ms()
    test_run_grade_job_hero_ts_ms_fallback_only_when_box_resolves()
    test_run_grade_job_subject_exposure_converges_grouped_subjects()
    test_run_grade_job_applies_leveling_and_semantic_grouping_when_flagged()
    test_vibrance_boosts_low_chroma_bounded()
    test_vibrance_no_desaturation_on_vivid_footage()
    test_vibrance_missing_chroma_is_identity()
    test_vibrance_flag_off_and_legacy_sat_is_one()
    test_skin_tint_corrects_green_cast_bounded()
    test_skin_tint_preserves_warmth_and_tone()
    test_skin_tint_skips_non_skin()
    test_skin_prefers_subject_lab_over_center_proxy()
    test_lab_to_srgb_round_trips()
    test_white_reference_still_wins_over_skin()
    test_measure_subject_lab_reads_the_box_not_the_whole_frame()
    test_measure_subject_lab_none_for_degenerate_box()
    test_tone_contrast_zero_is_exact_identity()
    test_tone_contrast_endpoints_pinned()
    test_tone_contrast_monotonic()
    test_tone_contrast_increases_midtone_slope()
    test_tone_contrast_legacy_and_nonv1_identity()
    test_bake_parity_with_contrast()
    test_resolver_flag_off_byte_identical()
    test_grade_hash_changes_with_tone_contrast()
    test_compositor_grade_key_distinguishes_tone_contrast_from_identity()
    test_look_identity_spec_is_identity_grid()
    test_look_grid_deterministic()
    test_look_grid_no_clip_no_nan()
    test_look_split_tone_directional()
    test_look_hue_rotate_only_targets_band()
    test_look_hue_sat_only_targets_band()
    test_look_engine_bake_parity()
    test_resolver_engine_sets_look_engine_and_changes_hash()
    test_grade_hash_look_engine_in_payload()
    test_compositor_grade_key_distinguishes_look_engine_from_identity()
    test_resolve_look_spec_prefers_catalog_over_inline_params()
    test_lookspec_texture_roundtrip()
    test_build_look_grid_ignores_texture()
    test_resolver_film_texture_populates_soft_local_when_look_declares_it()
    test_grain_ffmpeg_filter()
    test_halation_subgraph_shape()
    test_grade_hash_changes_with_texture()
    test_grade_key_not_collapsed_with_texture()
    test_transform_vf_texture_order_and_off_is_untouched()
    test_black_lift_raises_floor_keeps_white()
    test_negative_contrast_softens_monotonic()
    test_identity_still_exact_with_new_fields()
    test_catalog_all_looks_valid()
    test_bake_parity_with_new_knobs()
    test_film_looks_carry_texture()
    test_list_engine_looks_has_params_and_family()
    print("\nall grade tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
