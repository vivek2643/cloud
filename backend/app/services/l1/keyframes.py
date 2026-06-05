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


def _hist(frame_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h.flatten()


def _flow_mag(prev_gray: np.ndarray, cur_gray: np.ndarray) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(mag.mean())


def extract_three(
    video_path: str,
    start_ms: int,
    end_ms: int,
    out_dir: str,
    prefix: str,
) -> ThreeKeyframes:
    """
    Sample SAMPLE_FRAMES frames between start_ms and end_ms, then pick:
      - Anchor:    the MIDPOINT sample (representative frame)
      - PeakMotion: frame with highest flow magnitude vs. previous sample
                    (excludes anchor itself so motion frame is distinct)
      - Variance:  frame with largest histogram distance from anchor

    Writes the 3 frames as full-resolution JPEGs in `out_dir`. The caller
    is expected to downscale to 224x224 before R2 upload.
    """
    none = ThreeKeyframes(None, None, None, None, None, None)
    if end_ms <= start_ms:
        return none

    cap = cv2.VideoCapture(video_path)
    try:
        span = end_ms - start_ms
        if SAMPLE_FRAMES > 1:
            targets = [
                start_ms + span * i / (SAMPLE_FRAMES - 1)
                for i in range(SAMPLE_FRAMES)
            ]
        else:
            targets = [(start_ms + end_ms) / 2.0]

        # Seek ONCE to the shot start, then decode forward sequentially and grab
        # the frame nearest each target timestamp. The old code did a backward
        # CAP_PROP_POS_MSEC seek per sample (12 per shot); each such seek
        # re-decodes from the prior keyframe, so on a long clip the video was
        # effectively decoded dozens of times -- the L1 bottleneck. One seek +
        # sequential reads decodes each shot span at most once.
        cap.set(cv2.CAP_PROP_POS_MSEC, max(start_ms - 1.0, 0.0))
        samples: List[Tuple[float, np.ndarray]] = []
        ti = 0
        reads = 0
        max_reads = SAMPLE_FRAMES * 400  # safety bound vs. runaway decode
        while ti < len(targets) and reads < max_reads:
            ok, frame = cap.read()
            reads += 1
            if not ok or frame is None:
                break
            cur = cap.get(cv2.CAP_PROP_POS_MSEC)
            small = None
            # A single decoded frame may satisfy one or more pending targets
            # (very short shots); advance through all it has reached.
            while ti < len(targets) and cur + 1e-3 >= targets[ti]:
                if small is None:
                    small = cv2.resize(frame, (DOWNSCALE_WIDTH, DOWNSCALE_HEIGHT))
                samples.append((cur, small))
                ti += 1
            if cur >= end_ms:
                break

        if not samples:
            return none

        # Anchor = midpoint sample. The midpoint frame is what we use as the
        # shot thumbnail; it's also the most likely to contain the subject.
        anchor_idx = len(samples) // 2
        anchor_ts, anchor_frame = samples[anchor_idx]
        anchor_gray = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY)
        anchor_hist = _hist(anchor_frame)

        best_motion = (anchor_ts, -1.0)
        best_var = (anchor_ts, -1.0)
        prev_gray = anchor_gray
        for j, (ts, frame) in enumerate(samples):
            if j == anchor_idx:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mag = _flow_mag(prev_gray, gray)
            if mag > best_motion[1]:
                best_motion = (ts, mag)
            hist = _hist(frame)
            dist = float(1 - cv2.compareHist(anchor_hist, hist, cv2.HISTCMP_CORREL))
            if dist > best_var[1]:
                best_var = (ts, dist)
            prev_gray = gray

        # If we never found a non-anchor candidate, reuse anchor.
        if best_motion[1] < 0:
            best_motion = (anchor_ts, 0.0)
        if best_var[1] < 0:
            best_var = (anchor_ts, 0.0)

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
        )
    finally:
        cap.release()


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
