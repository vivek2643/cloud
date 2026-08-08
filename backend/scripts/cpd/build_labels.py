"""
cpd_boundary_segmenter.plan.md Phase D -- label alignment.

Reads each manifest clip's raw per-annotator GEBD boundary marks (seconds,
relative to clip start -- written by fetch_gebd.py), builds a CONSENSUS
boundary set (common.consensus_boundaries_sec: a mark survives only when
>= --min-agree distinct annotators placed a boundary within
--merge-window-s of each other), converts consensus timestamps to a
per-hop binary target vector on the SAME grid extract_features.py already
wrote (common.hop_labels, tolerance = --tolerance-ms -- ~200ms, the GEBD
Rel.Dis protocol's typical value for these clip lengths), and re-saves the
clip's .npz with `labels` added. Boundaries are rare (class imbalance is
expected and handled in Phase E's training, not here).

Run:  .venv/bin/python scripts/cpd/build_labels.py --split train
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(os.path.dirname(HERE))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import (  # noqa: E402
    consensus_boundaries_sec, features_path, hop_labels, load_clip_npz, manifest_rows,
    save_clip_npz,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_labels")

DATA_DIR = os.path.join(HERE, "data")


def label_one(clip_id: str, duration_sec: float, annotator_boundaries_sec,
              *, min_agree: int, merge_window_s: float, tolerance_ms: float) -> Optional[int]:
    """Returns the number of positive (boundary) hops written, or None if
    this clip has no extracted features yet (run extract_features.py
    first -- labeling is a separate pass so re-labeling with a different
    consensus rule never requires re-extracting)."""
    npz_path = features_path(DATA_DIR, clip_id)
    if not os.path.exists(npz_path):
        return None
    data = load_clip_npz(npz_path)
    consensus = consensus_boundaries_sec(
        annotator_boundaries_sec, min_agree=min_agree, merge_window_s=merge_window_s)
    duration_ms = int(round(duration_sec * 1000))
    labels = hop_labels(consensus, duration_ms, data["hop_ms"], tolerance_ms)
    t = data["features"].shape[0]
    # Align to the feature matrix's own length (hop_labels derives T from
    # duration_sec independently -- the two should already agree since both
    # come from the same clip, but a video whose real decoded frame count
    # differs slightly from its reported duration is possible; never let a
    # shape mismatch crash the batch).
    if len(labels) != t:
        if len(labels) > t:
            labels = labels[:t]
        else:
            import numpy as np
            labels = np.concatenate([labels, np.zeros(t - len(labels), dtype=labels.dtype)])
    save_clip_npz(npz_path, data["features"], data["hop_ms"], data["channel_names"],
                 labels=labels, clip_id=clip_id)
    return int(labels.sum())


def run(split: Optional[str], min_agree: int, merge_window_s: float, tolerance_ms: float,
        manifest_name: str = "manifest.csv") -> None:
    rows = manifest_rows(DATA_DIR, split, manifest_name)

    n_labeled = 0
    n_missing_features = 0
    total_positive_hops = 0
    total_hops = 0
    for row in rows:
        clip_id = row["clip_id"]
        annotator_boundaries = json.loads(row["annotator_boundaries_sec_json"])
        n_positive = label_one(
            clip_id, float(row["duration_sec"]), annotator_boundaries,
            min_agree=min_agree, merge_window_s=merge_window_s, tolerance_ms=tolerance_ms)
        if n_positive is None:
            n_missing_features += 1
            continue
        n_labeled += 1
        total_positive_hops += n_positive
        total_hops += load_clip_npz(features_path(DATA_DIR, clip_id))["features"].shape[0]

    logger.info("labeled %d clip(s), %d missing extracted features (run extract_features.py first)",
               n_labeled, n_missing_features)
    if total_hops:
        logger.info("positive-hop rate: %d/%d (%.1f%%) -- expected class imbalance for Phase E",
                   total_positive_hops, total_hops, 100.0 * total_positive_hops / total_hops)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=None,
                    help="e.g. train/val (GEBD) or a custom split name in --manifest-name")
    ap.add_argument("--min-agree", type=int, default=2,
                    help="use 1 for the single-annotator in-domain harness")
    ap.add_argument("--merge-window-s", type=float, default=0.3)
    ap.add_argument("--tolerance-ms", type=float, default=200.0)
    ap.add_argument("--manifest-name", default="manifest.csv",
                    help="e.g. indomain_manifest.csv for the in-domain harness")
    args = ap.parse_args()
    run(args.split, args.min_agree, args.merge_window_s, args.tolerance_ms, args.manifest_name)


if __name__ == "__main__":
    main()
