"""
L1 derived signal: scene / shot detection (cuts v2).

L2 used to assume "one continuous take" per clip -- fine for a single-camera
interview, dishonest for real multi-shot footage (an edited screen recording, a
sizzle reel, cutaway b-roll spliced into a raw file). This module supplies the
one genuinely NEW signal cuts-v2 needs: where the footage actually changes shot
or composition, so `shown` boundaries (l3.partition) land on real visual
changes instead of an arbitrary window.

How it works
------------
Mirrors ``motion_dynamics.py``'s pattern: ffmpeg decodes the proxy down to tiny
color frames piped straight into numpy, then the actual "is this a scene
change?" scoring happens in Python (opencv), not by parsing ffmpeg's internal
scene-filter text output (version-fragile). For each frame we take a coarse
Hue/Saturation 2D histogram (Value/brightness excluded so exposure ramps and
flashes don't read as cuts) and compare it to the previous frame's via
histogram correlation. DRIFT = 1 - correlation is high exactly where the shot
changes and low while the camera holds on the same scene.

Two outputs, same signal at two thresholds:
  * ``shot_points``        -- HARD cuts: strong, isolated drift spikes.
  * ``composition_points`` -- softer within-shot changes (a reframe, a subject
    entering/leaving) -- the same channel, a lower bar, with anything that
    coincides with a shot cut folded into that shot cut (not double-reported).

Best-effort, CPU-only, bounded by the L1 duration cap: any decode/opencv
failure returns an empty (``has_scenes=False``) result, never fails L1 --
mirrors ``motion_dynamics.compute_motion_dynamics``'s failure semantics.
"""
from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List

from app.services.l1.cut_grid_common import clamp01, local_maxima
from app.services.l1.scene_cuts_params import (
    COMPOSITION_DRIFT_FLOOR,
    COMPOSITION_MIN_GAP_MS,
    COMPOSITION_SHOT_MERGE_MS,
    SCENE_FPS,
    SCENE_H,
    SCENE_HUE_BINS,
    SCENE_SAT_BINS,
    SCENE_W,
    SHOT_DRIFT_FLOOR,
    SHOT_MIN_GAP_MS,
)

logger = logging.getLogger(__name__)

# Bump when the detection logic/shape changes so cached rows recompute even if
# the underlying proxy did not.
SCHEMA_VERSION = 1


@dataclass
class SceneCuts:
    has_scenes: bool = False
    hop_ms: int = 0
    shot_points: List[Dict] = field(default_factory=list)          # hard cuts
    composition_points: List[Dict] = field(default_factory=list)   # within-shot

    def to_dict(self) -> Dict:
        return {
            "has_scenes": self.has_scenes,
            "hop_ms": self.hop_ms,
            "shot_points": self.shot_points,
            "composition_points": self.composition_points,
        }


def _decode_bgr_frames(video_path: str, w: int, h: int, fps: int):
    """Yield consecutive (h, w, 3) uint8 BGR frames from ffmpeg at a fixed fps."""
    import numpy as np

    cmd = [
        "ffmpeg", "-v", "error", "-i", video_path,
        "-vf", f"scale={w}:{h},fps={fps}",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    frame_bytes = w * h * 3
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        proc.wait()


def _hs_hist(frame_bgr):
    """Coarse, L1-normalized Hue/Saturation 2D histogram for one frame."""
    import cv2

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [SCENE_HUE_BINS, SCENE_SAT_BINS],
                        [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def compute_scene_cuts(
    video_path: str,
    duration_ms: int,
    *,
    fps: int = SCENE_FPS,
    w: int = SCENE_W,
    h: int = SCENE_H,
) -> SceneCuts:
    try:
        import cv2  # noqa: F401  (imported for the side-effect-free availability check)
    except Exception:
        logger.warning("opencv unavailable; skipping scene detection.")
        return SceneCuts(has_scenes=False)

    hop_ms = int(round(1000 / fps))
    drift: List[float] = [0.0]
    prev_hist = None
    try:
        for frame in _decode_bgr_frames(video_path, w, h, fps):
            hist = _hs_hist(frame)
            if prev_hist is not None:
                import cv2
                corr = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
                drift.append(clamp01(1.0 - corr))
            prev_hist = hist
    except Exception:
        logger.exception("Scene detection decode/hist pass failed for %s.", video_path)
        return SceneCuts(has_scenes=False, hop_ms=hop_ms)

    if len(drift) < 2:
        return SceneCuts(has_scenes=False, hop_ms=hop_ms)

    shot_ts = local_maxima(drift, hop_ms, SHOT_DRIFT_FLOOR, SHOT_MIN_GAP_MS)
    comp_candidates = local_maxima(drift, hop_ms, COMPOSITION_DRIFT_FLOOR, COMPOSITION_MIN_GAP_MS)
    composition_ts = [
        t for t in comp_candidates
        if all(abs(t - s) > COMPOSITION_SHOT_MERGE_MS for s in shot_ts)
    ]

    return SceneCuts(
        has_scenes=True,
        hop_ms=hop_ms,
        shot_points=[{"ts_ms": t, "kind": "shot_cut", "score": 1.0} for t in shot_ts],
        composition_points=[
            {"ts_ms": t, "kind": "composition_change", "score": 1.0}
            for t in composition_ts
        ],
    )
