"""
L1 keyframe extractor: Anchor + PeakMotion + Variance.

Produces three temporally-distinct keyframes per shot, used by:
  - L1 SigLIP embeddings (3 vectors per shot for richer retrieval)
  - L1 blur telemetry (Laplacian variance per keyframe)
  - L1 intra_shot_variance (cosine distance between anchor & motion vectors)
  - L2 Stage A (DINOv2), Stage B (faces), Stage D (VLM narrative)

Anchor is the MIDPOINT sample of the shot (the most representative frame and
also the long-standing "thumbnail" semantic). PeakMotion is the sample with
the largest optical-flow magnitude vs. its predecessor sample. Variance is
the sample with the largest histogram distance from anchor.

The extractor returns absolute video timestamps (ms) for each pick so that
downstream code (L3 sub-clip trimming) can target the chaotic moment.

This file used to live under l2/; the move is intentional. Keyframes are
indexed at L1 time and reused by L2 (downloaded from R2 to skip re-decoding).
"""
from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2  # type: ignore
import numpy as np

logger = logging.getLogger(__name__)

# Internal sampling resolution -- only used to PICK timestamps; the JPEGs we
# write to disk are full-resolution from ffmpeg.
DOWNSCALE_WIDTH = 320
DOWNSCALE_HEIGHT = 180
SAMPLE_FRAMES = 12  # per shot


# --- Adaptive coverage tuning (Layer A) ----------------------------------
# Roughly one coverage frame per this many seconds of shot, scaled down for
# visually static shots and capped so one long shot can't explode the budget.
COVERAGE_SECONDS_PER_FRAME = 4.0
COVERAGE_MAX_TOTAL = 8           # hard cap on keyframes per shot (incl. base 3)
COVERAGE_NOVELTY_MIN = 0.12      # stop adding frames once the farthest sample is
                                 # this close (1 - hist-correlation) to what we have


@dataclass
class ThreeKeyframes:
    anchor_path: Optional[str]
    motion_path: Optional[str]
    variance_path: Optional[str]
    anchor_ts_ms: Optional[int]
    motion_ts_ms: Optional[int]
    variance_ts_ms: Optional[int]
    # Peak optical-flow magnitude across the sampled frames. Computed here as a
    # by-product of picking the motion keyframe, so the shots stage can reuse it
    # instead of re-opening the video for a second flow pass.
    motion_mag: Optional[float] = None
    # Dominant (magnitude-weighted mean) screen-space motion direction of the
    # shot, in px/frame at the 320x180 sampling resolution.
    motion_dx: Optional[float] = None
    motion_dy: Optional[float] = None


@dataclass
class CoverageFrame:
    ts_ms: int
    path: str


@dataclass
class AdaptiveKeyframes(ThreeKeyframes):
    """The legacy anchor/motion/variance triple PLUS a variable number of extra
    coverage frames chosen by farthest-point sampling. `coverage` excludes the
    base three so callers can keep treating those specially."""
    coverage: List[CoverageFrame] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.coverage is None:
            self.coverage = []


def _hist(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h.flatten()


def _flow_vec(prev_gray: np.ndarray, cur_gray: np.ndarray) -> Tuple[float, float, float]:
    """Dense optical flow between two frames -> (mean magnitude, mean dx, mean dy).

    The mean (dx, dy) is the dominant screen-space motion direction of this
    transition (px/frame at the 320x180 sampling resolution). We used to throw
    it away and keep only magnitude; the direction is what lets the editor make
    motion-continuous match cuts and avoid jarring direction reversals.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag, _ = cv2.cartToPolar(fx, fy)
    return float(mag.mean()), float(fx.mean()), float(fy.mean())


def _sample_shot(
    video_path: str, start_ms: int, end_ms: int, n_samples: int,
) -> List[Tuple[float, np.ndarray]]:
    """Decode the shot span ONCE and return up to `n_samples` evenly-spaced
    (timestamp_ms, downscaled_bgr) pairs.

    One seek + sequential reads decodes each shot span at most once. The old
    code did a backward CAP_PROP_POS_MSEC seek per sample (each re-decodes from
    the prior keyframe), which dominated the L1 cost on long clips.
    """
    cap = cv2.VideoCapture(video_path)
    try:
        span = end_ms - start_ms
        if n_samples > 1:
            targets = [start_ms + span * i / (n_samples - 1) for i in range(n_samples)]
        else:
            targets = [(start_ms + end_ms) / 2.0]

        cap.set(cv2.CAP_PROP_POS_MSEC, max(start_ms - 1.0, 0.0))
        samples: List[Tuple[float, np.ndarray]] = []
        ti = 0
        reads = 0
        max_reads = n_samples * 400  # safety bound vs. runaway decode
        while ti < len(targets) and reads < max_reads:
            ok, frame = cap.read()
            reads += 1
            if not ok or frame is None:
                break
            cur = cap.get(cv2.CAP_PROP_POS_MSEC)
            small = None
            while ti < len(targets) and cur + 1e-3 >= targets[ti]:
                if small is None:
                    small = cv2.resize(frame, (DOWNSCALE_WIDTH, DOWNSCALE_HEIGHT))
                samples.append((cur, small))
                ti += 1
            if cur >= end_ms:
                break
        return samples
    finally:
        cap.release()


def _pick_base(samples: List[Tuple[float, np.ndarray]]) -> dict:
    """Pick anchor (midpoint), peak-motion and peak-variance from the samples.
    Returns the picks plus per-sample histograms (reused by coverage selection)."""
    anchor_idx = len(samples) // 2
    anchor_ts, anchor_frame = samples[anchor_idx]
    anchor_gray = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY)
    anchor_hist = _hist(anchor_frame)

    hists: List[np.ndarray] = [None] * len(samples)  # type: ignore[list-item]
    hists[anchor_idx] = anchor_hist

    best_motion = (anchor_ts, -1.0)
    best_var = (anchor_ts, -1.0)
    best_var_idx = anchor_idx
    # Accumulate a magnitude-weighted mean flow vector over the shot -> its
    # dominant motion direction (used for match-cut continuity).
    sum_dx = 0.0
    sum_dy = 0.0
    sum_w = 0.0
    prev_gray = anchor_gray
    for j, (ts, frame) in enumerate(samples):
        if j == anchor_idx:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mag, dx, dy = _flow_vec(prev_gray, gray)
        sum_dx += dx * mag
        sum_dy += dy * mag
        sum_w += mag
        if mag > best_motion[1]:
            best_motion = (ts, mag)
        hist = _hist(frame)
        hists[j] = hist
        dist = float(1 - cv2.compareHist(anchor_hist, hist, cv2.HISTCMP_CORREL))
        if dist > best_var[1]:
            best_var = (ts, dist)
            best_var_idx = j
        prev_gray = gray

    if best_motion[1] < 0:
        best_motion = (anchor_ts, 0.0)
    if best_var[1] < 0:
        best_var = (anchor_ts, 0.0)

    motion_dx = (sum_dx / sum_w) if sum_w > 0 else 0.0
    motion_dy = (sum_dy / sum_w) if sum_w > 0 else 0.0

    return {
        "anchor_idx": anchor_idx,
        "anchor_ts": anchor_ts,
        "best_motion": best_motion,
        "best_var": best_var,
        "best_var_idx": best_var_idx,
        "motion_dx": motion_dx,
        "motion_dy": motion_dy,
        "hists": hists,
    }


def _hist_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


def _select_coverage(
    samples: List[Tuple[float, np.ndarray]],
    hists: List[np.ndarray],
    base_indices: List[int],
    target_total: int,
    novelty_min: float,
) -> List[int]:
    """Farthest-point sampling in histogram space: starting from the base
    frames, repeatedly add the sample most visually distinct from everything
    chosen so far. Returns the EXTRA indices (excluding base) added, stopping at
    `target_total` total frames or when nothing novel remains."""
    n = len(samples)
    selected = list(base_indices)
    if len(selected) >= target_total or n == 0:
        return []

    # Ensure every sample has a histogram (base pass leaves some None on very
    # short shots where samples were skipped).
    for i in range(n):
        if hists[i] is None:
            hists[i] = _hist(samples[i][1])

    min_d = [
        min((_hist_dist(hists[i], hists[s]) for s in selected), default=0.0)
        for i in range(n)
    ]
    extra: List[int] = []
    while len(selected) < target_total:
        cand, cand_d = -1, -1.0
        for i in range(n):
            if i in selected:
                continue
            if min_d[i] > cand_d:
                cand_d, cand = min_d[i], i
        if cand < 0 or cand_d < novelty_min:
            break
        selected.append(cand)
        extra.append(cand)
        for i in range(n):
            d = _hist_dist(hists[i], hists[cand])
            if d < min_d[i]:
                min_d[i] = d
    return extra


def extract_three(
    video_path: str,
    start_ms: int,
    end_ms: int,
    out_dir: str,
    prefix: str,
) -> ThreeKeyframes:
    """Anchor (midpoint) + peak-motion + peak-variance keyframes for a shot.
    Writes 3 full-resolution JPEGs; caller downscales to 224x224 for R2."""
    none = ThreeKeyframes(None, None, None, None, None, None)
    if end_ms <= start_ms:
        return none
    samples = _sample_shot(video_path, start_ms, end_ms, SAMPLE_FRAMES)
    if not samples:
        return none

    base = _pick_base(samples)
    anchor_ts = base["anchor_ts"]
    best_motion = base["best_motion"]
    best_var = base["best_var"]

    anchor_path = os.path.join(out_dir, f"{prefix}_anchor.jpg")
    motion_path = os.path.join(out_dir, f"{prefix}_motion.jpg")
    variance_path = os.path.join(out_dir, f"{prefix}_variance.jpg")
    ok_a, ok_m, ok_v = _dump_batch(
        video_path,
        [
            (anchor_ts / 1000.0, anchor_path),
            (best_motion[0] / 1000.0, motion_path),
            (best_var[0] / 1000.0, variance_path),
        ],
    )
    return ThreeKeyframes(
        anchor_path=anchor_path if ok_a else None,
        motion_path=motion_path if ok_m else None,
        variance_path=variance_path if ok_v else None,
        anchor_ts_ms=int(anchor_ts) if ok_a else None,
        motion_ts_ms=int(best_motion[0]) if ok_m else None,
        variance_ts_ms=int(best_var[0]) if ok_v else None,
        motion_mag=float(best_motion[1]) if best_motion[1] >= 0 else 0.0,
        motion_dx=float(base.get("motion_dx", 0.0)),
        motion_dy=float(base.get("motion_dy", 0.0)),
    )


def _adaptive_target_total(duration_s: float, variance: float, n_base: int) -> int:
    """How many keyframes this shot should get. More for long, visually busy
    shots; clamped to [n_base, COVERAGE_MAX_TOTAL]. `variance` is the peak
    intra-shot histogram distance (~0..1) from _pick_base."""
    # Static shots (low variance) shrink toward the base; busy shots reach full.
    variance_factor = max(0.4, min(1.0, variance / 0.3))
    raw = round(duration_s / COVERAGE_SECONDS_PER_FRAME * variance_factor)
    return int(max(n_base, min(COVERAGE_MAX_TOTAL, raw)))


def extract_adaptive(
    video_path: str,
    start_ms: int,
    end_ms: int,
    out_dir: str,
    prefix: str,
) -> AdaptiveKeyframes:
    """Adaptive coverage: the anchor/motion/variance triple plus a variable
    number of farthest-point coverage frames, scaled by shot length and how much
    the shot changes visually. Writes all frames as full-res JPEGs."""
    none = AdaptiveKeyframes(None, None, None, None, None, None)
    if end_ms <= start_ms:
        return none

    duration_s = (end_ms - start_ms) / 1000.0
    # Sample more densely for longer shots so coverage has frames to choose from.
    n_samples = int(max(SAMPLE_FRAMES, min(40, round(duration_s * 2))))
    samples = _sample_shot(video_path, start_ms, end_ms, n_samples)
    if not samples:
        return none

    base = _pick_base(samples)
    anchor_idx = base["anchor_idx"]
    anchor_ts = base["anchor_ts"]
    best_motion = base["best_motion"]
    best_var = base["best_var"]
    best_var_idx = base["best_var_idx"]
    hists = base["hists"]

    # Base indices we must keep: anchor + variance pick (motion is timestamp-only
    # since its sample index isn't tracked; that's fine -- coverage just avoids
    # re-picking anchor/variance frames it can see).
    base_indices = sorted({anchor_idx, best_var_idx})
    target_total = _adaptive_target_total(duration_s, best_var[1], n_base=3)
    extra_idx = _select_coverage(
        samples, hists, base_indices, target_total, COVERAGE_NOVELTY_MIN,
    )

    # Build the dump plan: base 3 first (stable filenames), then coverage.
    plan: List[Tuple[float, str]] = [
        (anchor_ts / 1000.0, os.path.join(out_dir, f"{prefix}_anchor.jpg")),
        (best_motion[0] / 1000.0, os.path.join(out_dir, f"{prefix}_motion.jpg")),
        (best_var[0] / 1000.0, os.path.join(out_dir, f"{prefix}_variance.jpg")),
    ]
    cov_meta: List[Tuple[float, str]] = []
    for k, idx in enumerate(extra_idx):
        ts = samples[idx][0]
        path = os.path.join(out_dir, f"{prefix}_cov{k:02d}.jpg")
        plan.append((ts / 1000.0, path))
        cov_meta.append((ts, path))

    oks = _dump_batch(video_path, plan)
    ok_a, ok_m, ok_v = oks[0], oks[1], oks[2]
    coverage = [
        CoverageFrame(ts_ms=int(ts), path=path)
        for (ts, path), ok in zip(cov_meta, oks[3:]) if ok
    ]

    return AdaptiveKeyframes(
        anchor_path=plan[0][1] if ok_a else None,
        motion_path=plan[1][1] if ok_m else None,
        variance_path=plan[2][1] if ok_v else None,
        anchor_ts_ms=int(anchor_ts) if ok_a else None,
        motion_ts_ms=int(best_motion[0]) if ok_m else None,
        variance_ts_ms=int(best_var[0]) if ok_v else None,
        motion_mag=float(best_motion[1]) if best_motion[1] >= 0 else 0.0,
        motion_dx=float(base.get("motion_dx", 0.0)),
        motion_dy=float(base.get("motion_dy", 0.0)),
        coverage=coverage,
    )


def _dump_batch(
    video_path: str,
    targets: List[Tuple[float, str]],
) -> List[bool]:
    """
    Extract multiple frames in a single ffmpeg invocation.

    Each `(timestamp_s, output_path)` becomes one `-ss <ts> -i <video>
    -frames:v 1 -q:v 3 <out>` block. Using input-side `-ss` (placed BEFORE
    `-i`) makes ffmpeg jump to the nearest keyframe before decoding, so we
    pay only ONE process-startup cost regardless of how many frames we
    need. For 3 frames this is ~3x faster than 3 separate invocations.
    """
    if not targets:
        return []

    cmd: list[str] = ["ffmpeg", "-y", "-loglevel", "error"]
    for ts_s, _out in targets:
        cmd += ["-ss", f"{ts_s:.3f}", "-i", video_path]
    for i, (_ts, out_path) in enumerate(targets):
        cmd += ["-map", f"{i}:v:0", "-frames:v", "1", "-q:v", "3", out_path]

    try:
        subprocess.run(cmd, check=False, capture_output=True, timeout=60)
    except Exception:
        logger.exception("ffmpeg batch dump failed for %s", video_path)
        return [False] * len(targets)

    return [os.path.exists(out) and os.path.getsize(out) > 0 for _ts, out in targets]


def downscale_for_storage(src_path: str, dst_path: str, target_size: int = 224, quality: int = 85) -> bool:
    """
    Convert a full-resolution keyframe to a 224x224 JPEG suitable for R2.
    Uses aspect-preserving fit + center-crop so the subject stays centered.
    Returns True on success.
    """
    if not src_path or not os.path.exists(src_path):
        return False
    try:
        from PIL import Image
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = target_size / min(w, h)
            new_w, new_h = int(round(w * scale)), int(round(h * scale))
            im = im.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - target_size) // 2
            top = (new_h - target_size) // 2
            im = im.crop((left, top, left + target_size, top + target_size))
            im.save(dst_path, "JPEG", quality=quality, optimize=True)
        return os.path.exists(dst_path) and os.path.getsize(dst_path) > 0
    except Exception:
        logger.exception("downscale_for_storage failed for %s", src_path)
        return False


def laplacian_blur_score(image_path: str) -> Optional[float]:
    """
    Laplacian-variance blur metric. Lower = blurrier.

    Standard heuristic: variance < ~50 is generally blurry, > 100 is sharp,
    but absolute thresholds depend on resolution and content. We store the
    raw value and let downstream logic compare relatively.
    """
    if not image_path or not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
