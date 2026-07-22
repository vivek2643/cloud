"""color_qa_harness.plan.md Part A.4: the failure-taxonomy metrics.

Pure functions over decoded RGB frames (float32, 0..1, HxWx3) plus optional
subject_box / group membership -- no DB, no network, no ffmpeg subprocess
calls. `_diag_qa_score.py` is the only caller that touches the DB/filesystem;
everything here just crunches arrays, so it's directly unit-testable
(see `scripts/test_qa_metrics.py`).

Luma uses Rec.709 coefficients (`cdl.LUMA_R/G/B` -- the SAME constants
`look_engine.py` uses, and numerically identical to `color_stats._aggregate`'s
hardcoded `0.2126/0.7152/0.0722`). Lab uses OpenCV's 8-bit packing
(L*x100/255, a*/b*-128) via `cv2.cvtColor(..., COLOR_RGB2LAB)` on a
0..255 uint8 frame -- the exact convention `color_stats._aggregate` and
`measure_span._measure_subject_lab`/`_measure_subject_luma` use, so numbers
here are directly comparable to what the grade solver saw. The skin locus
constants are imported from `grade.correct` (not re-declared) so the harness
always measures against the exact axis the solver corrects on -- they can
never silently drift apart.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from app.services.l3.grade.cdl import LUMA_B, LUMA_G, LUMA_R
from app.services.l3.grade.correct import (
    SKIN_L_MAX,
    SKIN_L_MIN,
    SKIN_LOCUS_DEG,
    SKIN_MIN_CHROMA,
    TARGET_MID_GRAY,
)

Verdict = str  # "pass" | "warn" | "fail" | "na"

# --------------------------------------------------------------------------
# Bands (A.4) -- starting values, calibrate against a hand-labeled set
# before trusting them as absolute truth (Risk 1 in the plan).
# --------------------------------------------------------------------------

CLIP_LOW_8BIT = 2 / 255.0        # color_stats.CLIP_LOW_8BIT
CLIP_HIGH_8BIT = 253 / 255.0     # color_stats.CLIP_HIGH_8BIT

CRUSHED_BLACK_PASS, CRUSHED_BLACK_WARN = 0.02, 0.05
CLIPPED_HIGHLIGHT_PASS, CLIPPED_HIGHLIGHT_WARN = 0.02, 0.05
EXPOSURE_BAND_PASS = (0.30, 0.60)
EXPOSURE_BAND_WARN = (0.22, 0.72)

NEUTRAL_CHROMA_MASK_MAX = 6.0     # |ab| below this -> "near neutral" pixel
NEUTRAL_MASK_MIN_FRACTION = 0.01  # below this coverage, deviation is N/A
NEUTRAL_AB_PASS, NEUTRAL_AB_WARN = 3.0, 6.0

SKIN_PERP_PASS, SKIN_PERP_WARN = 6.0, 12.0

GROUP_LUMA_STD_PASS, GROUP_LUMA_STD_WARN = 0.03, 0.06
GROUP_CHROMA_STD_PASS, GROUP_CHROMA_STD_WARN = 2.5, 5.0
GROUP_SUBJECT_LUMA_STD_PASS, GROUP_SUBJECT_LUMA_STD_WARN = 0.03, 0.06

SATURATION_PASS = (12.0, 40.0)
SATURATION_WARN = (8.0, 55.0)
CHROMA_INCREASE_FAIL_RATIO = 2.0

BANDING_PASS, BANDING_WARN = 0.15, 0.30

LOOK_FIDELITY_COS_PASS, LOOK_FIDELITY_COS_WARN = 0.8, 0.5

_VERDICT_RANK = {"pass": 0, "warn": 1, "fail": 2}


def worst_verdict(verdicts: Sequence[Verdict]) -> Verdict:
    """The worst of a set of verdicts, ignoring N/A -- N/A never contributes
    a failure, it just means "no reading." All-N/A -> "na" (nothing scored)."""
    real = [v for v in verdicts if v in _VERDICT_RANK]
    if not real:
        return "na"
    return max(real, key=lambda v: _VERDICT_RANK[v])


def _band_upper(value: float, pass_max: float, warn_max: float) -> Verdict:
    """Lower is better: PASS below `pass_max`, WARN below `warn_max`, else FAIL."""
    if value < pass_max:
        return "pass"
    if value < warn_max:
        return "warn"
    return "fail"


def _band_lower(value: float, pass_min: float, warn_min: float) -> Verdict:
    """Higher is better: PASS above `pass_min`, WARN above `warn_min`, else FAIL."""
    if value > pass_min:
        return "pass"
    if value > warn_min:
        return "warn"
    return "fail"


def _band_range(value: float, pass_range: tuple, warn_range: tuple) -> Verdict:
    lo_p, hi_p = pass_range
    lo_w, hi_w = warn_range
    if lo_p <= value <= hi_p:
        return "pass"
    if lo_w <= value <= hi_w:
        return "warn"
    return "fail"


@dataclass
class MetricResult:
    name: str
    value: Optional[float]
    verdict: Verdict
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "value": self.value, "verdict": self.verdict, **self.extra}


# --------------------------------------------------------------------------
# Array primitives
# --------------------------------------------------------------------------

def luma01(rgb01: np.ndarray) -> np.ndarray:
    """Rec.709 luma (0..1) of an (H,W,3) RGB 0..1 float array."""
    return rgb01[..., 0] * LUMA_R + rgb01[..., 1] * LUMA_G + rgb01[..., 2] * LUMA_B


def to_lab(rgb01: np.ndarray) -> np.ndarray:
    """(H,W,3) RGB 0..1 float -> (H,W,3) Lab, L in 0..100, a*/b* roughly
    -128..127 -- OpenCV's 8-bit packing, un-scaled to the conventional Lab
    range (same convention `color_stats._aggregate` uses)."""
    import cv2

    u8 = np.clip(rgb01 * 255.0, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    a = lab[..., 1] - 128.0
    b = lab[..., 2] - 128.0
    return np.stack([L, a, b], axis=-1)


def crop_box(rgb01: np.ndarray, box: Optional[Sequence[float]]) -> Optional[np.ndarray]:
    """Crop the normalized (x,y,w,h) `box` out of an (H,W,3) frame -- same
    pixel-mapping `measure_span._measure_subject_luma`/`_measure_subject_lab`
    use. None when the box is missing/malformed or resolves to zero pixels
    (mirrors those functions' fail-open contract)."""
    if not box:
        return None
    try:
        x, y, w, h = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    fh, fw = rgb01.shape[0], rgb01.shape[1]
    x0, y0 = int(max(0.0, x) * fw), int(max(0.0, y) * fh)
    x1, y1 = int(min(1.0, x + w) * fw), int(min(1.0, y + h) * fh)
    if x1 <= x0 or y1 <= y0:
        return None
    return rgb01[y0:y1, x0:x1, :]


# --------------------------------------------------------------------------
# 1. Exposure
# --------------------------------------------------------------------------

def exposure_metrics(graded01: np.ndarray, raw01: Optional[np.ndarray] = None) -> Dict[str, MetricResult]:
    l = luma01(graded01)
    crushed = float(np.mean(l <= CLIP_LOW_8BIT))
    clipped = float(np.mean(l >= CLIP_HIGH_8BIT))
    mean_l, median_l = float(l.mean()), float(np.median(l))

    crushed_extra: Dict[str, Any] = {}
    clipped_extra: Dict[str, Any] = {}
    if raw01 is not None:
        raw_l = luma01(raw01)
        raw_crushed = float(np.mean(raw_l <= CLIP_LOW_8BIT))
        raw_clipped = float(np.mean(raw_l >= CLIP_HIGH_8BIT))
        crushed_extra = {"raw": raw_crushed, "delta": crushed - raw_crushed}
        clipped_extra = {"raw": raw_clipped, "delta": clipped - raw_clipped}

    exposure_verdict = worst_verdict([
        _band_range(mean_l, EXPOSURE_BAND_PASS, EXPOSURE_BAND_WARN),
        _band_range(median_l, EXPOSURE_BAND_PASS, EXPOSURE_BAND_WARN),
    ])

    return {
        "crushed_black_fraction": MetricResult(
            "crushed_black_fraction", crushed,
            _band_upper(crushed, CRUSHED_BLACK_PASS, CRUSHED_BLACK_WARN), crushed_extra,
        ),
        "clipped_highlight_fraction": MetricResult(
            "clipped_highlight_fraction", clipped,
            _band_upper(clipped, CLIPPED_HIGHLIGHT_PASS, CLIPPED_HIGHLIGHT_WARN), clipped_extra,
        ),
        "exposure_band": MetricResult(
            "exposure_band", median_l, exposure_verdict,
            {"mean": mean_l, "median": median_l, "target": TARGET_MID_GRAY},
        ),
    }


# --------------------------------------------------------------------------
# 2. White balance / color cast
# --------------------------------------------------------------------------

def neutral_axis_deviation(graded01: np.ndarray) -> MetricResult:
    lab = to_lab(graded01)
    a, b = lab[..., 1], lab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    mask = chroma < NEUTRAL_CHROMA_MASK_MAX
    frac = float(mask.mean())
    if frac < NEUTRAL_MASK_MIN_FRACTION:
        return MetricResult("neutral_axis_deviation", None, "na", {"mask_fraction": frac})
    mean_a, mean_b = float(a[mask].mean()), float(b[mask].mean())
    dev = float(math.hypot(mean_a, mean_b))
    return MetricResult(
        "neutral_axis_deviation", dev, _band_upper(dev, NEUTRAL_AB_PASS, NEUTRAL_AB_WARN),
        {"mask_fraction": frac, "mean_a": mean_a, "mean_b": mean_b},
    )


def skin_perp_residual(graded01: np.ndarray, subject_box: Optional[Sequence[float]]) -> MetricResult:
    """Off-locus (green<->magenta tint) residual of the subject-box mean Lab,
    measured on the exact axis `correct._skin_multiplier` corrects
    (`SKIN_LOCUS_DEG`, imported not re-derived). N/A when there's no box, or
    the box's mean Lab doesn't gate as a confident skin read (same
    `SKIN_L_MIN/MAX`/`SKIN_MIN_CHROMA` guards `_skin_multiplier` uses) -- we
    never score a non-skin box."""
    crop = crop_box(graded01, subject_box)
    if crop is None or crop.size == 0:
        return MetricResult("skin_perp_residual", None, "na")
    lab = to_lab(crop)
    L = float(lab[..., 0].mean())
    a = float(lab[..., 1].mean())
    b = float(lab[..., 2].mean())
    if not (SKIN_L_MIN <= L <= SKIN_L_MAX) or math.hypot(a, b) < SKIN_MIN_CHROMA:
        return MetricResult("skin_perp_residual", None, "na", {"L": L, "a": a, "b": b})
    theta = math.radians(SKIN_LOCUS_DEG)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    r_par = a * cos_t + b * sin_t          # position along the locus (warmth) -- diagnostic only
    d_perp = -a * sin_t + b * cos_t        # off-locus tint -- the scored quantity
    return MetricResult(
        "skin_perp_residual", abs(d_perp), _band_upper(abs(d_perp), SKIN_PERP_PASS, SKIN_PERP_WARN),
        {"L": L, "a": a, "b": b, "warmth_r_par": r_par},
    )


# --------------------------------------------------------------------------
# 3. Shot-to-shot consistency (operates on PER-SHOT summaries, not frames --
# grouping/membership is the caller's job, this is pure stdev + banding).
# --------------------------------------------------------------------------

@dataclass
class ShotSummary:
    """One shot's per-frame aggregate, cheap to carry around for group
    rollups without keeping full-resolution arrays in memory."""
    median_luma: float
    mean_luma: float
    mean_a: float
    mean_b: float
    black_point: float
    white_point: float
    subject_luma: Optional[float] = None


def summarize_shot(rgb01: np.ndarray, subject_box: Optional[Sequence[float]] = None) -> ShotSummary:
    l = luma01(rgb01)
    flat = l.reshape(-1)
    lab = to_lab(rgb01)
    subject_luma = None
    if subject_box:
        crop = crop_box(rgb01, subject_box)
        if crop is not None and crop.size:
            subject_luma = float(luma01(crop).mean())
    return ShotSummary(
        median_luma=float(np.median(flat)),
        mean_luma=float(flat.mean()),
        mean_a=float(lab[..., 1].mean()),
        mean_b=float(lab[..., 2].mean()),
        black_point=float(np.percentile(flat, 0.5)),
        white_point=float(np.percentile(flat, 99.5)),
        subject_luma=subject_luma,
    )


def _stdev(values: Sequence[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return None
    return float(np.std(np.array(clean, dtype=np.float64)))


def group_consistency_metrics(
    graded: List[ShotSummary], raw: Optional[List[ShotSummary]] = None,
) -> Dict[str, MetricResult]:
    """Per-group (2+ members) consistency, reported GRADED (scored) plus RAW
    side-by-side when given -- the grade should REDUCE spread; a group whose
    GRADED std >= RAW std means matching was a no-op or harmful there."""
    if len(graded) < 2:
        return {}
    luma_std = _stdev([s.median_luma for s in graded])
    a_std = _stdev([s.mean_a for s in graded])
    b_std = _stdev([s.mean_b for s in graded])
    chroma_std = max(a_std or 0.0, b_std or 0.0) if (a_std is not None or b_std is not None) else None
    black_std = _stdev([s.black_point for s in graded])
    white_std = _stdev([s.white_point for s in graded])

    out: Dict[str, MetricResult] = {}
    if luma_std is not None:
        extra: Dict[str, Any] = {}
        if raw and len(raw) == len(graded):
            raw_std = _stdev([s.median_luma for s in raw])
            if raw_std is not None:
                extra = {"raw": raw_std, "improved": luma_std < raw_std}
        out["intra_group_luma_std"] = MetricResult(
            "intra_group_luma_std", luma_std,
            _band_upper(luma_std, GROUP_LUMA_STD_PASS, GROUP_LUMA_STD_WARN), extra,
        )
    if chroma_std is not None:
        extra = {}
        if raw and len(raw) == len(graded):
            raw_a_std = _stdev([s.mean_a for s in raw])
            raw_b_std = _stdev([s.mean_b for s in raw])
            if raw_a_std is not None or raw_b_std is not None:
                raw_chroma_std = max(raw_a_std or 0.0, raw_b_std or 0.0)
                extra = {"raw": raw_chroma_std, "improved": chroma_std < raw_chroma_std}
        out["intra_group_chroma_std"] = MetricResult(
            "intra_group_chroma_std", chroma_std,
            _band_upper(chroma_std, GROUP_CHROMA_STD_PASS, GROUP_CHROMA_STD_WARN), extra,
        )
    if black_std is not None:
        out["intra_group_black_std"] = MetricResult("intra_group_black_std", black_std, "na")
    if white_std is not None:
        out["intra_group_white_std"] = MetricResult("intra_group_white_std", white_std, "na")
    return out


def group_subject_exposure_metrics(
    graded: List[ShotSummary], raw: Optional[List[ShotSummary]] = None,
) -> Dict[str, MetricResult]:
    """#6 exposure-evenness: the same stdev idea as group_consistency_metrics
    but on SUBJECT-BOX luma specifically -- ties `even_lighting`/
    `subject_exposure`'s `target_subject_luma` convergence to a number.
    Members with no subject_luma (no box) are excluded, not FAILed."""
    g_subj = [s.subject_luma for s in graded if s.subject_luma is not None]
    if len(g_subj) < 2:
        return {}
    subj_std = _stdev(g_subj)
    if subj_std is None:
        return {}
    extra: Dict[str, Any] = {}
    if raw:
        r_subj = [s.subject_luma for s in raw if s.subject_luma is not None]
        if len(r_subj) >= 2:
            raw_std = _stdev(r_subj)
            if raw_std is not None:
                extra = {"raw": raw_std, "convergence_delta": raw_std - subj_std}
    return {
        "intra_group_subject_luma_std": MetricResult(
            "intra_group_subject_luma_std", subj_std,
            _band_upper(subj_std, GROUP_SUBJECT_LUMA_STD_PASS, GROUP_SUBJECT_LUMA_STD_WARN), extra,
        )
    }


# --------------------------------------------------------------------------
# 4. Over-processing
# --------------------------------------------------------------------------

def saturation_band(graded01: np.ndarray, raw01: Optional[np.ndarray] = None) -> MetricResult:
    lab = to_lab(graded01)
    a, b = lab[..., 1], lab[..., 2]
    chroma_mean = float(np.sqrt(a * a + b * b).mean())
    verdict = _band_range(chroma_mean, SATURATION_PASS, SATURATION_WARN)
    extra: Dict[str, Any] = {}
    if raw01 is not None:
        raw_lab = to_lab(raw01)
        ra, rb = raw_lab[..., 1], raw_lab[..., 2]
        raw_chroma = float(np.sqrt(ra * ra + rb * rb).mean())
        extra["raw_chroma_mean"] = raw_chroma
        if raw_chroma > 1e-6:
            ratio = chroma_mean / raw_chroma
            extra["chroma_increase_ratio"] = ratio
            if ratio > CHROMA_INCREASE_FAIL_RATIO:
                verdict = "fail"
    return MetricResult("saturation_band", chroma_mean, verdict, extra)


def banding_score(graded01: np.ndarray) -> MetricResult:
    """Posterization heuristic: blur the luma channel (suppress single-pixel
    noise), histogram into 256 bins, and measure the fraction of EMPTY bins
    between the lowest and highest occupied bin. A smooth gradient fills
    every bin in its range; an aggressively-quantized one leaves gaps.
    WARN-only signal per the plan -- the contact sheet is the real judge."""
    import cv2

    l = luma01(graded01)
    l_blur = cv2.GaussianBlur(l.astype(np.float32), (5, 5), 0)
    l_u8 = np.clip(l_blur * 255.0, 0, 255).astype(np.uint8)
    hist, _ = np.histogram(l_u8, bins=256, range=(0, 255))
    occupied = np.nonzero(hist)[0]
    if len(occupied) < 2:
        return MetricResult("banding_score", 0.0, "pass")
    lo, hi = int(occupied[0]), int(occupied[-1])
    span = hi - lo + 1
    empty = int((hist[lo:hi + 1] == 0).sum())
    frac = empty / span if span > 0 else 0.0
    return MetricResult("banding_score", float(frac), _band_upper(frac, BANDING_PASS, BANDING_WARN))


# --------------------------------------------------------------------------
# 5. Look fidelity
# --------------------------------------------------------------------------

def _shift_vector(a01: np.ndarray, b01: np.ndarray) -> np.ndarray:
    """5-dim (per-channel RGB mean shift + Lab a*/b* mean shift) direction
    vector of `a - b`, at whole-frame resolution -- the "did this transform
    push color the same way" signal look_fidelity compares."""
    mean_shift = a01.reshape(-1, 3).mean(axis=0) - b01.reshape(-1, 3).mean(axis=0)
    lab_a, lab_b = to_lab(a01), to_lab(b01)
    chroma_shift = np.array([
        lab_a[..., 1].mean() - lab_b[..., 1].mean(),
        lab_a[..., 2].mean() - lab_b[..., 2].mean(),
    ], dtype=np.float64)
    return np.concatenate([mean_shift.astype(np.float64), chroma_shift])


def look_fidelity_metric(graded01: np.ndarray, raw01: np.ndarray, look_only01: np.ndarray) -> MetricResult:
    """Cosine similarity of (graded-raw)'s shift vector against (look_only-raw)'s
    -- direction/shape, not pixel-equality (graded also carries correction).
    Low cosine means the applied look is pulling a different direction than
    the look alone intends -- a compositing/order bug, not a subtle taste
    difference."""
    v_graded = _shift_vector(graded01, raw01)
    v_look = _shift_vector(look_only01, raw01)
    n1, n2 = float(np.linalg.norm(v_graded)), float(np.linalg.norm(v_look))
    if n1 < 1e-6 or n2 < 1e-6:
        return MetricResult("look_fidelity_cosine", None, "na", {"graded_norm": n1, "look_norm": n2})
    cos = float(np.dot(v_graded, v_look) / (n1 * n2))
    return MetricResult(
        "look_fidelity_cosine", cos, _band_lower(cos, LOOK_FIDELITY_COS_PASS, LOOK_FIDELITY_COS_WARN),
    )
