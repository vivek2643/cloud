"""
cpd_boundary_segmenter.plan.md Phase G -- inference artifact.

The reusable entry point: ``predict(video_path, duration_ms=None)`` runs
the SAME Phase C extractor (extract_features.extract_one), the trained
Phase E model, a smoothing pass, and peak-picking -> boundary timestamps
in ms. This is what a later cuts-pipeline integration would import --
nothing here is wired into production; this plan ends at a validated,
reusable model + eval report (see the plan's explicit non-goals).

CLI:
  .venv/bin/python scripts/cpd/infer_cpd.py /path/to/clip.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(os.path.dirname(HERE))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import (  # noqa: E402
    MODEL_PATH_DEFAULT, augment_with_unsupervised_score, pick_peaks, smooth_curve, windowed_features,
)
from extract_features import extract_one  # noqa: E402


def probe_duration_ms(video_path: str) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=30,
    )
    return int(round(float(result.stdout.strip()) * 1000))


def predict(
    video_path: str, duration_ms: Optional[int] = None, *,
    model_path: str = MODEL_PATH_DEFAULT, min_gap_ms: int = 300, threshold: float = 0.5,
    smoothing_hops: int = 2,
) -> List[int]:
    """video_path -> boundary timestamps in ms, source-agnostic (GEBD or
    any of our own footage -- the extractor is the same one production
    uses). Raises FileNotFoundError if no trained model exists at
    ``model_path`` (run train_cpd.py first)."""
    import joblib

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"no trained model at {model_path} -- run train_cpd.py first")
    bundle = joblib.load(model_path)

    if duration_ms is None:
        duration_ms = probe_duration_ms(video_path)
    matrix = extract_one(video_path, duration_ms)
    if matrix is None:
        return []

    if bundle.get("early_fusion", False):
        matrix = augment_with_unsupervised_score(matrix, bundle.get("unsup_window_hops", 5))
    x = windowed_features(matrix, bundle["window_hops"])
    proba = bundle["model"].predict_proba(x)[:, 1]
    proba = smooth_curve(proba, window_hops=smoothing_hops)
    boundaries_sec = pick_peaks(proba, bundle["hop_ms"], min_gap_ms=min_gap_ms, min_score=threshold)
    return [int(round(t * 1000)) for t in boundaries_sec]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video_path")
    ap.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    ap.add_argument("--min-gap-ms", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    boundaries_ms = predict(
        args.video_path, model_path=args.model_path,
        min_gap_ms=args.min_gap_ms, threshold=args.threshold)
    print(boundaries_ms)


if __name__ == "__main__":
    main()
