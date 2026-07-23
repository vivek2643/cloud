"""
L1 derived signal: motion dynamics -> ACTION + CAMERA/DISTORTION cut grids.

One cheap optical-flow pass over the proxy produces the raw per-hop signals, from
which two cut-cost channels are derived. CPU-only, bounded by the L1 1h cap.

How it works
------------
ffmpeg decodes the proxy down to a tiny gray, fixed-fps stream (MOTION_W x
MOTION_H @ MOTION_FPS) piped straight into numpy -- so decode cost lives in
ffmpeg, not the Python loop. For each consecutive frame pair we run Farneback
dense optical flow, then FIT A GLOBAL SIMILARITY TRANSFORM (translation + zoom +
roll) to the flow with RANSAC. That single fit is what lets us tell a deliberate
camera move apart from bad-to-cut motion:

    camera motion = mean displacement the fitted model induces across the frame
                    (combines pan/tilt + zoom + roll into one px/frame number)
    coherence     = RANSAC inlier ratio -- how much of the frame moves as ONE
                    rigid body. ~1 for a clean pan/zoom; ~0 for shake or a still
                    camera with a thrashing subject.
    action energy = mean flow RESIDUAL after removing the model = true non-camera
                    (subject) motion.
    stability     = how steady the model's velocity is over a short window. ~1
                    for a sustained dolly; ~0 for a whip-pan / bump (a transient).
    sharpness     = Laplacian variance (low => motion-blurred / distorted).

Derived channels (cost 0 = ideal seam .. 1 = avoid):

  * ACTION  -- a "hit" channel. Impacts = local maxima of action energy; the
    cost curve dips to 0 on each impact (cut on the hit) and is high mid-motion.
  * CAMERA/DISTORTION -- an "avoid" channel, but gated by motion QUALITY. A
    smooth, coherent, sustained move is cheap to cut; only incoherent motion,
    jerky transients (whip/bump) and blur are expensive. So a deliberate dolly or
    steady pan is NOT penalized -- only the kinds of camera motion you actually
    shouldn't cut through.

Best-effort: any ffmpeg/opencv failure returns an empty result (non-fatal).
"""
from __future__ import annotations

import logging
import math
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.services import limits
from app.services.l1.cut_grid_common import (
    clamp01,
    hit_cost_curve,
    local_maxima,
    normalize_pctl,
    percentile,
)
from app.services.l1.cut_grid_params import (
    ACTION_MIN_PEAK_GAP_MS,
    ACTION_PEAK_PCTL,
    ACTION_TOL_MS,
    CAMERA_BLUR_WEIGHT,
    CAMERA_CHAOS_WEIGHT,
    CAMERA_JERK_DEADBAND_PX,
    CAMERA_MIN_SPEED_PX,
    CAMERA_REL_JERK_FULL,
    CAMERA_STABILITY_WIN_MS,
    CAMERA_TRANSIENT_WEIGHT,
    MOTION_FPS,
    MOTION_GRID_STEP,
    MOTION_H,
    MOTION_NORM_PCTL,
    MOTION_RANSAC_PX,
    MOTION_W,
)
from app.services.l1.motion_params import (
    DEGENERATE_BLUR_MIN,
    DEGENERATE_MIN_MS,
    WIPE_AREA_FRAC,
    WIPE_COHERENCE_MAX,
    WIPE_MAG_PX,
    WIPE_MIN_GAP_MS,
    WIPE_RECOVERY_MS,
)

logger = logging.getLogger(__name__)


@dataclass
class MotionDynamics:
    has_motion: bool = False
    hop_ms: int = 0
    # Raw, file-normalized signals (0..1) -- kept for inspection / future tuning.
    action_energy: List[float] = field(default_factory=list)
    camera_motion: List[float] = field(default_factory=list)
    camera_coherence: List[float] = field(default_factory=list)  # 1 = rigid global move
    camera_stability: List[float] = field(default_factory=list)  # 1 = steady/sustained
    blur: List[float] = field(default_factory=list)              # 1 = fully blurred/distorted
    # SIGNED per-hop camera velocity (absolute, NOT file-normalized -- so a pan
    # is a pan regardless of the clip's own spread), used only for the direction
    # of a cut's camera-move label (post._classify_camera_move). Sign convention
    # is scene-flow: +dx = scene moves right = camera pans LEFT; +dy = scene
    # moves down = camera tilts UP; +zoom = scene expands = camera zooms IN.
    camera_dx: List[float] = field(default_factory=list)   # translation x, fraction of frame width / hop
    camera_dy: List[float] = field(default_factory=list)   # translation y, fraction of frame height / hop
    camera_zoom: List[float] = field(default_factory=list)  # scale-1 / hop (+ in, - out)
    # Derived cut-cost channels (0 = ideal seam .. 1 = avoid).
    action_cut_cost: List[float] = field(default_factory=list)
    camera_cut_cost: List[float] = field(default_factory=list)
    # Discrete subject-motion impacts (cut ON these).
    action_points: List[Dict] = field(default_factory=list)
    # cuts-v3: premium natural cut instants -- occlusion wipes (a near-field
    # blob sweeps the frame) and degenerate spans (the frame collapses to one
    # texture: over-zoom, lens blocked). [{ts_ms, kind: wipe|degenerate, strength}]
    transition_points: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "has_motion": self.has_motion,
            "hop_ms": self.hop_ms,
            "action_energy": self.action_energy,
            "camera_motion": self.camera_motion,
            "camera_coherence": self.camera_coherence,
            "camera_stability": self.camera_stability,
            "blur": self.blur,
            "camera_dx": self.camera_dx,
            "camera_dy": self.camera_dy,
            "camera_zoom": self.camera_zoom,
            "action_cut_cost": self.action_cut_cost,
            "camera_cut_cost": self.camera_cut_cost,
            "action_points": self.action_points,
            "transition_points": self.transition_points,
        }


def _decode_gray_frames(video_path: str, w: int, h: int, fps: int):
    """Yield consecutive (h, w) uint8 gray frames from ffmpeg at a fixed fps."""
    import numpy as np

    cmd = [
        "ffmpeg", "-v", "error", "-i", video_path,
        "-vf", f"scale={w}:{h},fps={fps},format=gray",
        "-f", "rawvideo", "-",
    ]
    # Held for the whole decode (not just the spawn) -- a live subprocess
    # streaming frames for as long as the caller keeps pulling, exactly the
    # resource FFMPEG_CONCURRENCY is meant to bound.
    with limits.ffmpeg_slot():
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        frame_bytes = w * h
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                yield np.frombuffer(buf, dtype=np.uint8).reshape(h, w)
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            proc.wait()


def _fit_camera_model(src, dst, ransac_px: float):
    """Fit a similarity transform (pan/tilt + zoom + roll) from flow vectors.

    Returns (camera_motion, coherence, residual_mean, param_vec) where
    ``param_vec = (tx, ty, R*theta, R*(scale-1))`` is the camera "velocity" in
    comparable px units (R = frame half-diagonal), used downstream for temporal
    stability. Falls back to a robust median-translation estimate if the
    RANSAC fit fails (which itself signals incoherent motion -> coherence 0).
    """
    import cv2
    import numpy as np

    n = len(src)
    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_px
    )
    R = math.hypot(MOTION_W, MOTION_H) / 2.0

    if M is None:
        # No global consensus: treat as incoherent. Use median translation so the
        # magnitude is still meaningful, but coherence collapses to 0.
        d = dst - src
        gx, gy = float(np.median(d[:, 0])), float(np.median(d[:, 1]))
        cam = math.hypot(gx, gy)
        resid = float(np.mean(np.hypot(d[:, 0] - gx, d[:, 1] - gy)))
        return cam, 0.0, resid, (gx, gy, 0.0, 0.0)

    a, b = float(M[0, 0]), float(M[1, 0])
    tx, ty = float(M[0, 2]), float(M[1, 2])
    scale = math.hypot(a, b)
    theta = math.atan2(b, a)

    # Camera motion = mean displacement the model induces over the sampled grid.
    pred = (src @ M[:, :2].T) + M[:, 2]
    cam = float(np.mean(np.hypot(pred[:, 0] - src[:, 0], pred[:, 1] - src[:, 1])))
    # Residual = observed flow the model can't explain = true subject motion.
    resid = float(np.mean(np.hypot(dst[:, 0] - pred[:, 0], dst[:, 1] - pred[:, 1])))
    coherence = float(inliers.sum()) / n if (inliers is not None and n) else 0.0

    param_vec = (tx, ty, R * theta, R * (scale - 1.0))
    return cam, clamp01(coherence), resid, param_vec


def compute_motion_dynamics(
    video_path: str,
    duration_ms: int,
    *,
    fps: int = MOTION_FPS,
    w: int = MOTION_W,
    h: int = MOTION_H,
) -> MotionDynamics:
    try:
        import cv2
        import numpy as np
    except Exception:
        logger.warning("opencv/numpy unavailable; skipping motion dynamics.")
        return MotionDynamics(has_motion=False)

    hop_ms = int(round(1000 / fps))

    # Subsample grid of source positions (built once; flow gives the dst offset).
    ys, xs = np.mgrid[0:h:MOTION_GRID_STEP, 0:w:MOTION_GRID_STEP]
    grid = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float32)
    gi_y = ys.ravel()
    gi_x = xs.ravel()

    action_raw: List[float] = []
    camera_raw: List[float] = []
    coherence: List[float] = []
    sharp_raw: List[float] = []
    # cuts-v3: fraction of the sampled grid sweeping at RAW flow magnitude above
    # WIPE_MAG_PX -- the occlusion-wipe signal (see transition_points below).
    # RAW (pre camera-model) on purpose: a near-field object crossing the lens
    # is exactly the large, chaotic flow the model fit doesn't explain away.
    wipe_frac_raw: List[float] = []
    params: List[Tuple[float, float, float, float]] = []
    # Per-hop normalized centroid (cx, cy in 0..1) of SUBJECT motion = where the
    # action is, so a reframe can follow it. Camera motion is removed first by
    # subtracting the global median flow, then we take the residual-weighted mean
    # grid position. (0.5, 0.5) when there is no subject motion to point at.
    centroids: List[Tuple[float, float]] = []

    prev = None
    try:
        for frame in _decode_gray_frames(video_path, w, h, fps):
            sharp_raw.append(float(cv2.Laplacian(frame, cv2.CV_64F).var()))
            if prev is None:
                action_raw.append(0.0)
                camera_raw.append(0.0)
                coherence.append(1.0)
                params.append((0.0, 0.0, 0.0, 0.0))
                centroids.append((0.5, 0.5))
                wipe_frac_raw.append(0.0)
            else:
                # Deep pyramid (5 levels) + wide window so a fast pan/whip (large
                # per-hop displacement) is still tracked instead of read as noise.
                flow = cv2.calcOpticalFlowFarneback(
                    prev, frame, None, 0.5, 5, 21, 3, 7, 1.5, 0
                )
                fx = flow[gi_y, gi_x, 0]
                fy = flow[gi_y, gi_x, 1]
                wipe_frac_raw.append(float(np.mean(np.hypot(fx, fy) > WIPE_MAG_PX)))
                # dst = where each sampled source pixel moved to.
                dst = grid + np.column_stack([fx, fy]).astype(np.float32)
                cam, coh, resid, pvec = _fit_camera_model(grid, dst, MOTION_RANSAC_PX)
                camera_raw.append(cam)
                coherence.append(coh)
                action_raw.append(resid)
                params.append(pvec)
                # Subject-motion centroid: residual flow after removing the global
                # (camera) median, weighted mean of grid positions.
                rx = fx - float(np.median(fx))
                ry = fy - float(np.median(fy))
                mag = np.hypot(rx, ry)
                msum = float(mag.sum())
                if msum > 1e-3:
                    cx = float((gi_x * mag).sum() / msum) / max(1, w)
                    cy = float((gi_y * mag).sum() / msum) / max(1, h)
                    centroids.append((round(clamp01(cx), 3), round(clamp01(cy), 3)))
                else:
                    centroids.append((0.5, 0.5))
            prev = frame
    except Exception:
        logger.exception("Motion-dynamics flow pass failed for %s.", video_path)
        return MotionDynamics(has_motion=False, hop_ms=hop_ms)

    if len(action_raw) < 2:
        return MotionDynamics(has_motion=False, hop_ms=hop_ms)

    action_n = normalize_pctl(action_raw, MOTION_NORM_PCTL)
    camera_n = normalize_pctl(camera_raw, MOTION_NORM_PCTL)

    # Temporal stability via RELATIVE jerk: |change in camera velocity| divided by
    # the current speed. A constant-velocity move (any speed) -> ~0 -> steady;
    # only accelerations (move onset/stop, whip, bump) -> high -> transient. This
    # is absolute (not file-normalized), so a clip that is entirely one smooth
    # move is correctly read as stable.
    def _speed(p) -> float:
        return math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2 + p[3] ** 2)

    rel_jerk = [0.0]
    for i in range(1, len(params)):
        p, q = params[i], params[i - 1]
        dv = math.sqrt(sum((p[k] - q[k]) ** 2 for k in range(4)))
        dv = max(0.0, dv - CAMERA_JERK_DEADBAND_PX)  # ignore sub-px fit jitter
        denom = max(CAMERA_MIN_SPEED_PX, 0.5 * (_speed(p) + _speed(q)))
        rel_jerk.append(dv / denom)
    win = max(1, int(round(CAMERA_STABILITY_WIN_MS / hop_ms)))
    rel_jerk_s = _moving_avg(rel_jerk, win)
    stability = [round(clamp01(1.0 - j / CAMERA_REL_JERK_FULL), 3) for j in rel_jerk_s]

    # Blur/distortion: low Laplacian variance => blurred. Normalize against the
    # file's own sharp frames, then invert so 1 = fully blurred.
    sharp_ref = percentile(sharp_raw, MOTION_NORM_PCTL)
    if sharp_ref <= 0:
        blur = [0.0] * len(sharp_raw)
    else:
        blur = [round(clamp01(1.0 - (s / sharp_ref)), 3) for s in sharp_raw]

    transition_points = sorted(
        _wipe_points(wipe_frac_raw, coherence, hop_ms) + _degenerate_points(blur, hop_ms),
        key=lambda p: p["ts_ms"],
    )

    # ACTION (hit channel): impacts = local maxima of subject motion.
    floor = percentile(action_n, ACTION_PEAK_PCTL)
    impacts = local_maxima(action_n, hop_ms, floor, ACTION_MIN_PEAK_GAP_MS)
    action_cut_cost = hit_cost_curve(impacts, duration_ms, hop_ms, ACTION_TOL_MS)
    n = len(action_cut_cost) if action_cut_cost else len(action_n)

    def _centroid_at(ts_ms: int) -> Tuple[float, float]:
        if not centroids:
            return (0.5, 0.5)
        i = int(round(ts_ms / hop_ms)) if hop_ms else 0
        return centroids[max(0, min(len(centroids) - 1, i))]

    action_points = [
        {"ts_ms": t, "kind": "action_impact", "score": 1.0,
         "centroid": list(_centroid_at(t))}
        for t in impacts
    ]

    # CAMERA/DISTORTION (avoid channel), gated by motion quality: a smooth,
    # coherent, sustained move stays cheap; only chaos / transients / blur cost.
    def _g(arr: List[float], i: int) -> float:
        return arr[i] if i < len(arr) else 0.0

    camera_cut_cost = [
        round(clamp01(
            CAMERA_CHAOS_WEIGHT     * _g(camera_n, i) * (1.0 - _g(coherence, i))
            + CAMERA_TRANSIENT_WEIGHT * _g(camera_n, i) * (1.0 - _g(stability, i))
            + CAMERA_BLUR_WEIGHT      * _g(blur, i)
        ), 3)
        for i in range(n)
    ]

    def _fit(arr: List[float]) -> List[float]:
        if len(arr) >= n:
            return arr[:n]
        return arr + [0.0] * (n - len(arr))

    # SIGNED camera velocity per hop, in frame-relative units, straight off the
    # fitted model params (tx, ty, R*theta, R*(scale-1)). Absolute (un-normalized)
    # so the move-direction classifier can use physical thresholds; see the
    # sign convention on the dataclass fields.
    R = math.hypot(MOTION_W, MOTION_H) / 2.0
    cam_dx = [round(p[0] / MOTION_W, 5) for p in params]
    cam_dy = [round(p[1] / MOTION_H, 5) for p in params]
    cam_zoom = [round(p[3] / R, 5) for p in params] if R > 0 else [0.0] * len(params)

    return MotionDynamics(
        has_motion=True,
        hop_ms=hop_ms,
        action_energy=_fit(action_n),
        camera_motion=_fit(camera_n),
        camera_coherence=_fit([round(c, 3) for c in coherence]),
        camera_stability=_fit(stability),
        blur=_fit(blur),
        camera_dx=_fit(cam_dx),
        camera_dy=_fit(cam_dy),
        camera_zoom=_fit(cam_zoom),
        action_cut_cost=action_cut_cost or [1.0] * n,
        camera_cut_cost=camera_cut_cost,
        action_points=action_points,
        transition_points=transition_points,
    )


def _moving_avg(xs: List[float], win: int) -> List[float]:
    if win <= 1 or len(xs) < 2:
        return list(xs)
    out: List[float] = []
    half = win // 2
    for i in range(len(xs)):
        lo = max(0, i - half)
        hi = min(len(xs), i + half + 1)
        seg = xs[lo:hi]
        out.append(sum(seg) / len(seg))
    return out


# --------------------------------------------------------------------------
# cuts-v3: transition points (premium natural cut instants)
# --------------------------------------------------------------------------

def _wipe_points(wipe_frac: List[float], coherence: List[float], hop_ms: int) -> List[Dict]:
    """Occlusion-wipe instants: a large near-field blob sweeps the frame --
    classic pass-by transition editors hunt for. A candidate is a local
    maximum of the swept-area FRACTION above WIPE_AREA_FRAC, gated on the
    camera model's coherence ALSO collapsing (a clean fast pan keeps high
    coherence at high magnitude -- never mistaken for a wipe) and on a quick
    RECOVERY afterward (a sustained chaotic stretch is a disturbance, handled
    elsewhere, not a wipe)."""
    if not wipe_frac:
        return []
    n = len(wipe_frac)
    recovery_hops = max(1, round(WIPE_RECOVERY_MS / hop_ms))
    candidates = local_maxima(wipe_frac, hop_ms, WIPE_AREA_FRAC, WIPE_MIN_GAP_MS)
    out: List[Dict] = []
    for ts in candidates:
        i = min(n - 1, ts // hop_ms)
        if i < len(coherence) and coherence[i] > WIPE_COHERENCE_MAX:
            continue  # coherent fast motion (a clean pan/zoom) -- not a wipe
        after = wipe_frac[i + 1:i + 1 + recovery_hops]
        if after and min(after) > WIPE_AREA_FRAC * 0.5:
            continue  # never recovers -- sustained chaos, not a quick sweep
        out.append({"ts_ms": ts, "kind": "wipe", "strength": round(float(wipe_frac[i]), 3)})
    return out


def _degenerate_points(blur: List[float], hop_ms: int) -> List[Dict]:
    """Degenerate spans: the frame collapses to one texture (over-zoom, lens
    blocked) -- a sustained run where `blur` stays maxed out, not just one
    soft frame from fast motion. Marks the ONSET (nothing past this point is
    worth watching)."""
    n = len(blur)
    out: List[Dict] = []
    i = 0
    while i < n:
        if blur[i] >= DEGENERATE_BLUR_MIN:
            j = i
            while j < n and blur[j] >= DEGENERATE_BLUR_MIN:
                j += 1
            if (j - i) * hop_ms >= DEGENERATE_MIN_MS:
                strength = sum(blur[i:j]) / (j - i)
                out.append({"ts_ms": i * hop_ms, "kind": "degenerate", "strength": round(strength, 3)})
            i = j
        else:
            i += 1
    return out
