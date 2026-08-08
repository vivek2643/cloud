"""
cpd_boundary_segmenter.plan.md Phase E -- model + training.

Trains the SUPERVISED boundary model (Phase E.2): a gradient-boosted-tree
classifier (sklearn's HistGradientBoostingClassifier -- lightgbm isn't
installed in this env; the plan explicitly allows "LightGBM/sklearn" for
this tabular baseline) on a flattened sliding window of the 9 raw channels
around each hop (common.windowed_features), predicting P(boundary at this
hop). class_weight="balanced" handles the expected imbalance (boundaries
are rare) without hand-tuned up-sampling.

The UNSUPERVISED baseline (Phase E.1, common.unsupervised_boundary_score)
needs no training at all -- it's evaluated directly in eval_cpd.py, not
here.

Train/val split is BY VIDEO with zero leakage: GEBD's own train/val split
(different source videos per split, written by fetch_gebd.py) already
guarantees this -- train_cpd.py only ever fits on the train split.

--limit N (learning-curve support): trains on only the FIRST N labeled
train clips, in manifest order (= download order, deterministic) -- so
--limit 150/300/500 against a growing manifest are NESTED subsets of each
other, the standard shape for a clean learning curve (each larger point's
training set is a strict superset of the smaller one's, so the curve
isn't also absorbing which-clips-got-picked variance).

--pos-weight M (recall-lever experiment): up-weights positive (boundary)
samples by an EXTRA factor M on top of the standard class-balanced rate
(computed manually and passed as sample_weight, since
HistGradientBoostingClassifier's class_weight only accepts "balanced"/None,
not a custom ratio) -- M=1.0 is exactly equivalent to class_weight=
"balanced" (the default path below); M>1 pushes the model to fire on more
candidate boundaries at the cost of precision, the direct lever for testing
whether supervised's lower recall (vs the unsupervised baseline) is fixable
by reweighting rather than more data.

--early-fusion (last GEBD experiment): appends the UNSUPERVISED two-window
distance score (common.unsupervised_boundary_score) as an extra per-hop
input channel BEFORE windowing (common.augment_with_unsupervised_score),
so the tree sees the classical-CPD statistic directly as a feature and can
learn its own combination -- unlike the late-fusion "detect then snap"
ensemble (which failed to beat either standalone), this lets training
itself decide how much to trust each signal, per hop. Recorded in the
saved model bundle (early_fusion / unsup_window_hops) so eval_cpd.py/
infer_cpd.py reconstruct the IDENTICAL augmented input at inference time.

Run:  .venv/bin/python scripts/cpd/train_cpd.py --window-hops 5
      .venv/bin/python scripts/cpd/train_cpd.py --limit 150 --model-path models/cpd_boundary_n150.joblib
      .venv/bin/python scripts/cpd/train_cpd.py --pos-weight 3.0 --model-path models/cpd_boundary_posw3.joblib
      .venv/bin/python scripts/cpd/train_cpd.py --early-fusion --model-path models/cpd_boundary_earlyfusion.joblib
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(os.path.dirname(HERE))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import (  # noqa: E402
    CHANNEL_NAMES, DEFAULT_WINDOW_HOPS, HOP_MS, MODEL_PATH_DEFAULT, augment_with_unsupervised_score,
    load_labeled_clips, windowed_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_cpd")

DATA_DIR = os.path.join(HERE, "data")


def build_training_matrix(clips: List[dict], window_hops: int,
                          early_fusion: bool = False, unsup_window_hops: int = 5):
    import numpy as np

    xs, ys = [], []
    for clip in clips:
        features = clip["features"]
        if early_fusion:
            features = augment_with_unsupervised_score(features, unsup_window_hops)
        xs.append(windowed_features(features, window_hops))
        ys.append(clip["labels"])
    if not xs:
        raise SystemExit(
            "no labeled train clips found -- run extract_features.py then "
            "build_labels.py for --split train first.")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def train(window_hops: int, model_path: str, limit: Optional[int] = None,
          pos_weight: Optional[float] = None, early_fusion: bool = False,
          unsup_window_hops: int = 5) -> None:
    import joblib
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier

    clips = load_labeled_clips(DATA_DIR, "train")
    logger.info("loaded %d labeled train clip(s)", len(clips))
    if limit is not None:
        clips = clips[:limit]
        logger.info("--limit %d -> training on the first %d clip(s) (learning-curve subset)",
                   limit, len(clips))
    if early_fusion:
        logger.info("early fusion ON -- appending unsupervised_boundary_score "
                   "(window_hops=%d) as an extra input channel", unsup_window_hops)
    x, y = build_training_matrix(clips, window_hops, early_fusion, unsup_window_hops)
    logger.info("training matrix: %s, positive rate %.2f%%", x.shape, 100.0 * y.mean())

    if pos_weight is not None:
        n = len(y)
        n_pos = float(y.sum())
        n_neg = n - n_pos
        base_pos_w = n / (2.0 * n_pos) if n_pos > 0 else 1.0
        base_neg_w = n / (2.0 * n_neg) if n_neg > 0 else 1.0
        sample_weight = np.where(y == 1, base_pos_w * pos_weight, base_neg_w)
        logger.info("pos_weight=%.2f -> effective positive sample weight %.3f (balanced alone: %.3f)",
                   pos_weight, base_pos_w * pos_weight, base_pos_w)
        model = HistGradientBoostingClassifier(class_weight=None, random_state=0)
        model.fit(x, y, sample_weight=sample_weight)
    else:
        model = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
        model.fit(x, y)
    train_acc = model.score(x, y)
    logger.info("train-set accuracy (NOT a held-out metric -- see eval_cpd.py): %.3f", train_acc)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    channel_names = list(CHANNEL_NAMES) + (["unsupervised_score"] if early_fusion else [])
    joblib.dump({
        "model": model, "window_hops": window_hops, "hop_ms": HOP_MS,
        "channel_names": channel_names,
        "early_fusion": early_fusion, "unsup_window_hops": unsup_window_hops,
    }, model_path)
    logger.info("saved model to %s", model_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-hops", type=int, default=DEFAULT_WINDOW_HOPS)
    ap.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    ap.add_argument("--limit", type=int, default=None,
                    help="train on only the first N labeled train clips (learning-curve sweeps)")
    ap.add_argument("--pos-weight", type=float, default=None,
                    help="extra positive-class up-weight beyond balanced (1.0 == balanced; recall lever)")
    ap.add_argument("--early-fusion", action="store_true",
                    help="append the unsupervised score as an extra input channel before windowing")
    ap.add_argument("--unsup-window-hops", type=int, default=5,
                    help="window_hops for the early-fusion unsupervised_boundary_score channel")
    args = ap.parse_args()
    train(args.window_hops, args.model_path, args.limit, args.pos_weight,
         args.early_fusion, args.unsup_window_hops)


if __name__ == "__main__":
    main()
