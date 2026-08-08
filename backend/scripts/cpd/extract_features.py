"""
cpd_boundary_segmenter.plan.md Phase C -- feature extraction (LOCAL, CPU).

For each manifest clip: reuse the PRODUCTION extractor
(app.services.l1.motion_dynamics.compute_motion_dynamics) verbatim for
channels 1-5, add the new channel 6 (appearance_drift, Phase C.2) by
reusing scene_cuts.py's own HS-histogram/correlation logic but decoded on
the SAME 10 fps grid as motion (scene_cuts.py itself runs its production
pass at 5 fps -- fine for THRESHOLDED shot points, but we want a CONTINUOUS
curve aligned hop-for-hop with the other 8 channels here). Assemble a T x C
matrix and save data/features/<clip_id>.npz -- train == inference because
this is the exact function production calls, not a reimplementation.

Run:
  .venv/bin/python scripts/cpd/extract_features.py --split train
  .venv/bin/python scripts/cpd/extract_features.py --split train --limit 50 --workers 1
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from multiprocessing import Pool
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(os.path.dirname(HERE))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import CHANNEL_NAMES, HOP_MS, manifest_rows, save_clip_npz  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("extract_features")

DATA_DIR = os.path.join(HERE, "data")
FEATURES_DIR = os.path.join(DATA_DIR, "features")


def compute_appearance_drift(video_path: str, *, fps: int, w: int, h: int) -> List[float]:
    """Per-hop HS-histogram correlation drift (1 - correlation), decoded at
    (fps, w, h) instead of scene_cuts.py's own production (SCENE_FPS,
    SCENE_W, SCENE_H) -- same histogram machinery
    (app.services.l1.scene_cuts._decode_bgr_frames / _hs_hist,
    HISTCMP_CORREL), just re-run on the CPD's shared 10 fps hop grid so this
    channel lines up index-for-index with motion_dynamics' output. Unlike
    scene_cuts.py's production path, the full continuous curve is kept (no
    threshold/local-maxima step) -- that's the whole point of computing it
    here rather than reusing scene_cuts.py's public SceneCuts result, which
    only exposes thresholded points."""
    import cv2

    from app.services.l1.scene_cuts import _decode_bgr_frames, _hs_hist

    drift: List[float] = [0.0]
    prev_hist = None
    for frame in _decode_bgr_frames(video_path, w, h, fps):
        hist = _hs_hist(frame)
        if prev_hist is not None:
            corr = float(cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL))
            drift.append(max(0.0, 1.0 - corr))
        prev_hist = hist
    return drift


def _fit_len(arr: List[float], n: int) -> List[float]:
    if len(arr) >= n:
        return arr[:n]
    return arr + [0.0] * (n - len(arr))


def extract_one(video_path: str, duration_ms: int) -> Optional[Any]:
    """video_path, duration_ms -> T x C float32 matrix (channel order =
    common.CHANNEL_NAMES), or None if the production extractor couldn't
    read this clip (bad download, corrupt file -- best-effort, matches
    compute_motion_dynamics' own non-fatal failure semantics)."""
    import numpy as np

    from app.services.l1.cut_grid_params import MOTION_FPS
    from app.services.l1.motion_dynamics import compute_motion_dynamics
    from app.services.l1.scene_cuts_params import SCENE_H, SCENE_W

    motion = compute_motion_dynamics(video_path, duration_ms)
    if not motion.has_motion or not motion.action_energy:
        return None
    n = len(motion.action_energy)

    drift = compute_appearance_drift(video_path, fps=MOTION_FPS, w=SCENE_W, h=SCENE_H)
    drift = _fit_len(drift, n)

    columns = {
        "frame_diff": motion.frame_diff,
        "camera_dx": motion.camera_dx,
        "camera_dy": motion.camera_dy,
        "camera_zoom": motion.camera_zoom,
        "action_energy": motion.action_energy,
        "action_energy_raw": motion.action_energy_raw,
        "camera_coherence": motion.camera_coherence,
        "blur": motion.blur,
        "appearance_drift": drift,
    }
    matrix = np.stack([np.asarray(columns[c], dtype=np.float32) for c in CHANNEL_NAMES], axis=1)
    return matrix


def _process_row(row: Dict[str, str]) -> Dict[str, Any]:
    clip_id = row["clip_id"]
    video_path = os.path.join(DATA_DIR, row["path"])
    duration_ms = int(round(float(row["duration_sec"]) * 1000))
    out_path = os.path.join(FEATURES_DIR, f"{clip_id}.npz")
    if os.path.exists(out_path):
        return {"clip_id": clip_id, "status": "skipped (exists)"}
    if not os.path.exists(video_path):
        return {"clip_id": clip_id, "status": "missing video"}
    try:
        matrix = extract_one(video_path, duration_ms)
    except Exception as exc:  # noqa: BLE001 -- best-effort batch job, log and continue
        logger.exception("extraction failed for %s", clip_id)
        return {"clip_id": clip_id, "status": f"error: {exc}"}
    if matrix is None:
        return {"clip_id": clip_id, "status": "no motion signal"}
    save_clip_npz(out_path, matrix, HOP_MS, CHANNEL_NAMES, clip_id=clip_id)
    return {"clip_id": clip_id, "status": "ok", "T": matrix.shape[0]}


def run(split: Optional[str], limit: Optional[int], workers: int,
        manifest_name: str = "manifest.csv") -> None:
    os.makedirs(FEATURES_DIR, exist_ok=True)
    rows = manifest_rows(DATA_DIR, split, manifest_name)
    if limit is not None:
        rows = rows[:limit]
    logger.info("extracting features for %d clip(s) (workers=%d)", len(rows), workers)

    results: List[Dict[str, Any]]
    if workers <= 1:
        results = [_process_row(r) for r in rows]
    else:
        with Pool(workers) as pool:
            results = pool.map(_process_row, rows)

    ok = sum(1 for r in results if r["status"] == "ok")
    logger.info("done: %d/%d ok", ok, len(results))
    for r in results:
        if r["status"] != "ok":
            logger.info("  %s: %s", r["clip_id"], r["status"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=None,
                    help="e.g. train/val (GEBD) or a custom split name in --manifest-name")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--manifest-name", default="manifest.csv",
                    help="e.g. indomain_manifest.csv for the in-domain harness")
    args = ap.parse_args()
    run(args.split, args.limit, args.workers, args.manifest_name)


if __name__ == "__main__":
    main()
