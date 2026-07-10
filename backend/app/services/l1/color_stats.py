"""
L1 derived signal: deterministic per-file color statistics -- the foundation
every downstream color-grading layer (correct/match/look/arc) measures off.
See color_grading.plan.md SS2.2.

How it works
------------
Same decode philosophy as motion_dynamics.py/scene_cuts.py: ffmpeg decodes
frames straight into numpy, then all measurement happens in Python
(opencv/numpy), never by parsing ffmpeg's own filter text output. Unlike
those two (which build a per-hop TIME SERIES over the whole clip),
color_stats is a per-FILE AGGREGATE -- it fast-seeks to a handful of evenly
spaced timestamps (skipping the first/last 5%, where fades and slates live)
and decodes exactly one frame each, which is far cheaper than a full decode
pass for a statistic that doesn't need per-hop resolution.

From the sampled frames we compute: a luma histogram, black/white points,
mid-gray, per-channel RGB mean/median, a mean Lab a*/b* cast, a gray-world
+ white-patch white-balance estimate, highlight/shadow clipping %,
a log/flat heuristic (low luma spread + compressed range), a "skin" Lab
sample, and a k-means dominant palette.

Skin sample caveat: there is no face detector anywhere in this codebase, and
L1 runs at ingest time -- before any cut or VLM pass exists, so
`cut_records.framing.subject_box` isn't available yet either. The skin
sample is therefore a plain geometric proxy (a center-weighted region,
where a talking-head subject usually sits), not an actual face region. It's
the same "best-effort, never-worse" trade the whole grading system makes
(docs/color_grading.md SS11): good enough to anchor the correct layer's
auto-WB, not a claim of precision.

Best-effort, CPU-only, bounded by the L1 duration cap: any decode/opencv
failure returns an empty (`has_stats=False`) result, never fails L1 --
mirrors motion_dynamics.compute_motion_dynamics's failure semantics.

Runs off whatever proxy `_track_motion` already fetched (the same source
motion_dynamics/scene_cuts use) -- no new proxy fetch, no new ingest cost.
On the fast client-proxy path that source can be as small as 160x90; we
accept that trade rather than add a new download, the same call scene_cuts
already makes for its own color-sensitive (HSV histogram) work.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Bump when the measurement logic/shape changes so cached rows recompute even
# if the underlying proxy did not.
SCHEMA_VERSION = 1

COLOR_STATS_W = 320
COLOR_STATS_H = 180
COLOR_STATS_MAX_FRAMES = 12
LUMA_BINS = 32
CLIP_LOW_8BIT = 2          # luma <= this (0..255) counts as shadow-clipped
CLIP_HIGH_8BIT = 253       # luma >= this (0..255) counts as highlight-clipped
LOG_FLAT_STD_MAX = 0.16    # normalized luma std-dev below this reads as flat
LOG_FLAT_RANGE_MAX = 0.75  # (p99.5 - p0.5) luma range below this reads as flat
PALETTE_K = 5
PALETTE_SAMPLE_MAX = 20000


@dataclass
class ColorStats:
    has_stats: bool = False
    frames_sampled: int = 0
    luma_hist: List[float] = field(default_factory=list)        # LUMA_BINS bins, L1-normalized
    black_point: float = 0.0                                    # 0..1 luma, p0.5
    white_point: float = 1.0                                    # 0..1 luma, p99.5
    mid_gray: float = 0.5                                       # 0..1 luma, median
    rgb_mean: List[float] = field(default_factory=list)         # [r,g,b] 0..1
    rgb_median: List[float] = field(default_factory=list)       # [r,g,b] 0..1
    lab_ab_cast: List[float] = field(default_factory=list)      # [a*, b*] mean, CIE Lab
    wb_gray_world: List[float] = field(default_factory=list)    # [r,g,b] multipliers to neutralize
    wb_white_patch: List[float] = field(default_factory=list)   # [r,g,b] multipliers from brightest patch
    clip_shadow_pct: float = 0.0
    clip_highlight_pct: float = 0.0
    is_log_flat: bool = False
    skin_lab: Optional[List[float]] = None                      # [L*, a*, b*], center-weighted proxy region
    palette: List[List[float]] = field(default_factory=list)    # up to PALETTE_K dominant [r,g,b] (0..1), by prevalence

    def to_dict(self) -> Dict:
        return {
            "has_stats": self.has_stats,
            "frames_sampled": self.frames_sampled,
            "luma_hist": self.luma_hist,
            "black_point": self.black_point,
            "white_point": self.white_point,
            "mid_gray": self.mid_gray,
            "rgb_mean": self.rgb_mean,
            "rgb_median": self.rgb_median,
            "lab_ab_cast": self.lab_ab_cast,
            "wb_gray_world": self.wb_gray_world,
            "wb_white_patch": self.wb_white_patch,
            "clip_shadow_pct": self.clip_shadow_pct,
            "clip_highlight_pct": self.clip_highlight_pct,
            "is_log_flat": self.is_log_flat,
            "skin_lab": self.skin_lab,
            "palette": self.palette,
        }


def _sample_timestamps(duration_s: float, n: int) -> List[float]:
    """N evenly spaced timestamps over the middle 90% of the clip (skips
    fades/slates/black frames that tend to sit at the very start/end)."""
    if duration_s <= 0:
        return [0.0]
    n = max(1, n)
    if n == 1:
        return [duration_s / 2.0]
    lo, hi = duration_s * 0.05, duration_s * 0.95
    if hi <= lo:
        return [duration_s / 2.0]
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


def _decode_rgb_frame_at(video_path: str, ts_s: float, w: int, h: int):
    """Fast-seek + decode exactly one RGB frame at ts_s, scaled to w x h."""
    import numpy as np

    cmd = [
        "ffmpeg", "-v", "error", "-ss", f"{max(0.0, ts_s):.3f}", "-i", video_path,
        "-frames:v", "1", "-vf", f"scale={w}:{h}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = w * h * 3
    buf = proc.stdout
    if len(buf) < frame_bytes:
        return None
    return np.frombuffer(buf[:frame_bytes], dtype=np.uint8).reshape(h, w, 3).copy()


def _dominant_palette(flat_rgb_u8) -> List[List[float]]:
    """k-means over a subsample of pixels, ordered by cluster prevalence."""
    import cv2
    import numpy as np

    n = flat_rgb_u8.shape[0]
    if n > PALETTE_SAMPLE_MAX:
        idx = np.random.default_rng(0).choice(n, size=PALETTE_SAMPLE_MAX, replace=False)
        sample = flat_rgb_u8[idx]
    else:
        sample = flat_rgb_u8
    sample32 = sample.astype(np.float32)
    k_eff = min(PALETTE_K, max(1, sample32.shape[0]))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5)
    _compactness, labels, centers = cv2.kmeans(
        sample32, k_eff, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    counts = np.bincount(labels.flatten(), minlength=k_eff)
    order = np.argsort(-counts)
    return [(centers[i] / 255.0).tolist() for i in order]


def _aggregate(frames) -> ColorStats:
    import cv2
    import numpy as np

    stacked_u8 = np.stack(frames, axis=0)                      # (N,H,W,3) RGB uint8
    stacked = stacked_u8.astype(np.float32) / 255.0             # (N,H,W,3) RGB 0..1

    luma = 0.2126 * stacked[..., 0] + 0.7152 * stacked[..., 1] + 0.0722 * stacked[..., 2]
    flat_luma = luma.reshape(-1)

    hist, _ = np.histogram(flat_luma, bins=LUMA_BINS, range=(0.0, 1.0))
    luma_hist = (hist / max(1, int(hist.sum()))).tolist()

    black_point = float(np.percentile(flat_luma, 0.5))
    white_point = float(np.percentile(flat_luma, 99.5))
    mid_gray = float(np.median(flat_luma))

    clip_shadow_pct = float(np.mean(flat_luma <= (CLIP_LOW_8BIT / 255.0)))
    clip_highlight_pct = float(np.mean(flat_luma >= (CLIP_HIGH_8BIT / 255.0)))

    flat_rgb = stacked.reshape(-1, 3)
    flat_rgb_u8 = stacked_u8.reshape(-1, 3)
    rgb_mean = flat_rgb.mean(axis=0).tolist()
    rgb_median = np.median(flat_rgb, axis=0).tolist()

    lab_frames = [cv2.cvtColor(f, cv2.COLOR_RGB2LAB) for f in frames]
    lab_stacked = np.stack(lab_frames, axis=0).astype(np.float32)
    # OpenCV 8-bit Lab packing: L in 0..255 (scale /255*100), a/b offset +128.
    a_mean = float(lab_stacked[..., 1].mean() - 128.0)
    b_mean = float(lab_stacked[..., 2].mean() - 128.0)

    eps = 1e-6
    r_mean, g_mean, b_mean_ch = rgb_mean
    gray_target = (r_mean + g_mean + b_mean_ch) / 3.0
    wb_gray_world = [
        float(gray_target / max(eps, r_mean)),
        float(gray_target / max(eps, g_mean)),
        float(gray_target / max(eps, b_mean_ch)),
    ]

    bright_thresh = float(np.percentile(flat_luma, 99.0))
    bright_mask = flat_luma >= bright_thresh
    if bright_mask.any():
        patch = flat_rgb[bright_mask].mean(axis=0)
        patch_max = float(patch.max())
        wb_white_patch = [
            float(patch_max / max(eps, patch[0])),
            float(patch_max / max(eps, patch[1])),
            float(patch_max / max(eps, patch[2])),
        ]
    else:
        wb_white_patch = [1.0, 1.0, 1.0]

    spread = float(np.std(flat_luma))
    is_log_flat = spread < LOG_FLAT_STD_MAX and (white_point - black_point) < LOG_FLAT_RANGE_MAX

    # Skin sample: center-weighted geometric proxy (no face detector -- see
    # module docstring). A talking-head subject is usually framed here.
    h, w = stacked.shape[1], stacked.shape[2]
    y0, y1 = int(h * 0.25), int(h * 0.75)
    x0, x1 = int(w * 0.35), int(w * 0.65)
    if y1 > y0 and x1 > x0:
        center_lab_frames = [
            cv2.cvtColor(f[y0:y1, x0:x1, :], cv2.COLOR_RGB2LAB) for f in frames
        ]
        center_lab = np.stack(center_lab_frames, axis=0).astype(np.float32)
        skin_lab = [
            float(center_lab[..., 0].mean() * (100.0 / 255.0)),
            float(center_lab[..., 1].mean() - 128.0),
            float(center_lab[..., 2].mean() - 128.0),
        ]
    else:
        skin_lab = None

    palette = _dominant_palette(flat_rgb_u8)

    return ColorStats(
        has_stats=True,
        frames_sampled=len(frames),
        luma_hist=luma_hist,
        black_point=black_point,
        white_point=white_point,
        mid_gray=mid_gray,
        rgb_mean=rgb_mean,
        rgb_median=rgb_median,
        lab_ab_cast=[a_mean, b_mean],
        wb_gray_world=wb_gray_world,
        wb_white_patch=wb_white_patch,
        clip_shadow_pct=clip_shadow_pct,
        clip_highlight_pct=clip_highlight_pct,
        is_log_flat=is_log_flat,
        skin_lab=skin_lab,
        palette=palette,
    )


def compute_color_stats(
    video_path: str,
    duration_ms: int,
    *,
    max_frames: int = COLOR_STATS_MAX_FRAMES,
    w: int = COLOR_STATS_W,
    h: int = COLOR_STATS_H,
) -> ColorStats:
    try:
        import cv2  # noqa: F401  (availability check)
        import numpy as np  # noqa: F401
    except Exception:
        logger.warning("opencv/numpy unavailable; skipping color stats.")
        return ColorStats(has_stats=False)

    duration_s = max(0.0, (duration_ms or 0) / 1000.0)
    timestamps = _sample_timestamps(duration_s, max_frames)

    frames = []
    try:
        for ts in timestamps:
            frame = _decode_rgb_frame_at(video_path, ts, w, h)
            if frame is not None:
                frames.append(frame)
    except Exception:
        logger.exception("Color stats decode pass failed for %s.", video_path)
        return ColorStats(has_stats=False)

    if not frames:
        return ColorStats(has_stats=False)

    try:
        return _aggregate(frames)
    except Exception:
        logger.exception("Color stats aggregation failed for %s.", video_path)
        return ColorStats(has_stats=False, frames_sampled=len(frames))
