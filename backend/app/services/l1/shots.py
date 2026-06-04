"""
L1 Stage 3: Shot detection + 3-keyframe extraction + visual telemetry.

Pipeline:
  1. PySceneDetect ContentDetector for raw boundaries.
  2. Form-factor-aware post-pass (Delta 1.A): bucket video by total duration,
     enforce per-bucket min/max shot length by merging or splitting.
  3. For each final shot, extract 3 keyframes via the L1 keyframes module
     (anchor=midpoint, peak motion, peak variance) in ONE ffmpeg call.
  4. Compute per-keyframe Laplacian-variance blur scores; the minimum is
     the shot's `blur_min` signal.
  5. Compute brightness on the anchor and motion magnitude on a 320x180
     start-vs-midpoint optical flow pass (legacy L1 telemetry).
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2  # type: ignore
import numpy as np

from app.services.l1 import keyframes as kf_mod

logger = logging.getLogger(__name__)


# --- Form-factor-aware constraints (Delta 1.A) ---------------------------
FORM_FACTORS = {
    "short":  {"max_total_s":   180.0, "min_s": 0.5, "max_s":  4.0},
    "medium": {"max_total_s":  1200.0, "min_s": 2.0, "max_s": 15.0},
    "long":   {"max_total_s": float("inf"), "min_s": 5.0, "max_s": 30.0},
}


def pick_form_factor(duration_s: float) -> dict:
    for name in ("short", "medium", "long"):
        cfg = FORM_FACTORS[name]
        if duration_s <= cfg["max_total_s"]:
            return {"name": name, **cfg}
    return {"name": "long", **FORM_FACTORS["long"]}


@dataclass
class Shot:
    index: int
    start_ms: int
    end_ms: int
    # Three keyframes (full-resolution local paths during indexing; later
    # downscaled to 224x224 for R2 upload).
    anchor_local_path: Optional[str] = None
    motion_local_path: Optional[str] = None
    variance_local_path: Optional[str] = None
    anchor_ts_ms: Optional[int] = None
    motion_ts_ms: Optional[int] = None
    variance_ts_ms: Optional[int] = None
    # R2 keys filled in by the orchestrator after upload.
    keyframe_r2_key: Optional[str] = None          # = anchor (legacy column reuse)
    r2_keyframe_motion_key: Optional[str] = None
    r2_keyframe_variance_key: Optional[str] = None
    # Telemetry
    focus_score: Optional[float] = None             # Laplacian variance of anchor (legacy)
    brightness: Optional[float] = None
    motion_magnitude: Optional[float] = None
    blur_min: Optional[float] = None                # min Laplacian variance across 3 keyframes


# --- PySceneDetect wrapper -----------------------------------------------

def detect_raw_shots(video_path: str, downscale: int = 1, frame_skip: int = 0) -> List[Tuple[int, int]]:
    """Return raw shot boundaries from PySceneDetect as a list of (start_ms, end_ms).

    `downscale` shrinks each frame before the content comparison (huge speed
    win, negligible accuracy loss). `frame_skip` analyses every Nth frame for
    a further ~Nx speedup on long clips at the cost of boundary precision.
    """
    from scenedetect import ContentDetector, open_video, SceneManager

    video = open_video(video_path)
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=27.0))
    if downscale > 1:
        manager.downscale = downscale
    manager.detect_scenes(video=video, show_progress=False, frame_skip=frame_skip)
    scenes = manager.get_scene_list()

    if not scenes:
        duration_s = video.duration.get_seconds()
        return [(0, int(duration_s * 1000))]

    out: List[Tuple[int, int]] = []
    for start, end in scenes:
        out.append((int(start.get_seconds() * 1000), int(end.get_seconds() * 1000)))
    return out


# --- Form-factor-aware post-pass (Delta 1.A) -----------------------------

def apply_form_factor_constraints(
    raw: List[Tuple[int, int]],
    duration_s: float,
) -> List[Tuple[int, int]]:
    """Merge sub-min shots; split super-max shots."""
    cfg = pick_form_factor(duration_s)
    min_ms = int(cfg["min_s"] * 1000)
    max_ms = int(cfg["max_s"] * 1000)

    merged: List[List[int]] = []
    for s, e in raw:
        if merged and (e - s) < min_ms:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    while len(merged) >= 2 and (merged[-1][1] - merged[-1][0]) < min_ms:
        merged[-2][1] = merged[-1][1]
        merged.pop()

    final: List[Tuple[int, int]] = []
    for s, e in merged:
        if (e - s) <= max_ms:
            final.append((s, e))
            continue
        cursor = s
        while cursor < e:
            chunk_end = min(cursor + max_ms, e)
            final.append((cursor, chunk_end))
            cursor = chunk_end
    return final


# --- Telemetry helpers ---------------------------------------------------

def _brightness(image_path: str) -> Optional[float]:
    img = cv2.imread(image_path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def _motion_magnitude(video_path: str, start_ms: int, mid_ms: int) -> float:
    """Optical-flow magnitude across start-of-shot vs midpoint frames."""
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(start_ms, 0))
        ok1, frame1 = cap.read()
        if not ok1:
            return 0.0
        cap.set(cv2.CAP_PROP_POS_MSEC, mid_ms)
        ok2, frame2 = cap.read()
        if not ok2:
            return 0.0
        g1 = cv2.cvtColor(cv2.resize(frame1, (320, 180)), cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(cv2.resize(frame2, (320, 180)), cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(mag.mean())
    finally:
        cap.release()


# --- High-level entry point ----------------------------------------------

def detect_shots(video_path: str, duration_s: float, output_dir: str) -> List[Shot]:
    """
    Detect shots, extract 3 keyframes per shot, compute telemetry.
    Returns Shot rows ready for the orchestrator to upload + persist.

    `video_path` should be the 1080p proxy when available: detection,
    keyframe extraction and optical-flow telemetry all decode this file, so
    running them against the proxy instead of a 4K raw is a large speedup with
    no loss (keyframes are downscaled to 224px for SigLIP anyway).
    """
    # Always downscale the detector input; skip frames on longer clips. With a
    # 1080p proxy, downscale=2 -> ~960x540 comparison frames, which is plenty
    # for ContentDetector and an order of magnitude cheaper than full-res.
    downscale = 3 if duration_s > 1800 else 2
    frame_skip = 1 if duration_s > 600 else 0
    raw = detect_raw_shots(video_path, downscale=downscale, frame_skip=frame_skip)
    bounded = apply_form_factor_constraints(raw, duration_s)

    shots: List[Shot] = []
    for idx, (s_ms, e_ms) in enumerate(bounded):
        prefix = f"shot_{idx:05d}"
        kfs = kf_mod.extract_three(video_path, s_ms, e_ms, output_dir, prefix)

        anchor = kfs.anchor_path
        # Per-keyframe blur
        blurs = [
            kf_mod.laplacian_blur_score(p)
            for p in (kfs.anchor_path, kfs.motion_path, kfs.variance_path)
            if p
        ]
        blur_min = float(min(b for b in blurs if b is not None)) if any(b is not None for b in blurs) else None

        # Legacy telemetry: focus = anchor Laplacian variance (= blur of anchor),
        # brightness = mean luminance of anchor, motion_magnitude = start-vs-mid.
        focus = float(kf_mod.laplacian_blur_score(anchor)) if anchor else None
        brightness = _brightness(anchor) if anchor else None
        mid_ms = (s_ms + e_ms) // 2
        motion = _motion_magnitude(video_path, s_ms, mid_ms) if anchor else 0.0

        shots.append(Shot(
            index=idx,
            start_ms=s_ms,
            end_ms=e_ms,
            anchor_local_path=kfs.anchor_path,
            motion_local_path=kfs.motion_path,
            variance_local_path=kfs.variance_path,
            anchor_ts_ms=kfs.anchor_ts_ms,
            motion_ts_ms=kfs.motion_ts_ms,
            variance_ts_ms=kfs.variance_ts_ms,
            focus_score=focus,
            brightness=brightness,
            motion_magnitude=motion,
            blur_min=blur_min,
        ))
    return shots
