"""
cpd_boundary_segmenter.plan.md Phase F -- evaluation (the scoreboard).

On the held-out VAL split, computes GEBD F1 with the Relative-Distance
protocol (common.f1_curve: F1@0.05 is the headline, plus the full
0.05->0.5 sweep) for three boundary sources:
  1. Unsupervised baseline (Phase E.1) -- no training, a pure two-sample
     statistic on the (z-scored) feature channels. If this alone tracks
     GEBD boundaries, the features carry the signal.
  2. Supervised model (Phase E.2) -- the trained tree classifier's smoothed
     P(boundary) curve, peak-picked.
  3. Current production segmenter (v4_segment.segment_video), as a SANITY
     CHECK only -- its cut edges are an editorial "keep vs discard" split,
     not a perceptual-boundary detector (cuts_content_first_segmentation
     .plan.md's segmenter DROPS dead footage rather than marking every
     transition), so a low score here is expected and not itself a
     failure; it's here so the two numbers are directly comparable, not to
     grade the segmenter. Best-effort -- skipped per-clip on any failure
     (missing video file, L1 compute error) rather than blocking the rest
     of the report.

This report is the deliverable that says whether to proceed to the
embedding escalation (Open decisions) or move on to the downstream
trim/energy layers -- see the plan's Phase F.3.

Run:  .venv/bin/python scripts/cpd/eval_cpd.py --model-path models/cpd_boundary.joblib
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(os.path.dirname(HERE))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import (  # noqa: E402
    MODEL_PATH_DEFAULT, REL_DIS_THRESHOLDS, augment_with_unsupervised_score, consensus_boundaries_sec,
    f1_at_threshold, f1_curve, find_peak_indices, load_labeled_clips, pick_peaks,
    precision_recall_f1_at_threshold, smooth_curve, unsupervised_boundary_score, windowed_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_cpd")

DATA_DIR = os.path.join(HERE, "data")


def unsupervised_predictions(clips: List[dict], window_hops: int, min_gap_ms: int) -> Dict[str, List[float]]:
    out = {}
    for clip in clips:
        score = unsupervised_boundary_score(clip["features"], window_hops)
        out[clip["clip_id"]] = pick_peaks(score, clip["hop_ms"], min_gap_ms=min_gap_ms,
                                          min_score=float(score.mean() + score.std())) \
            if score.size else []
    return out


def supervised_predictions(clips: List[dict], model, window_hops: int, min_gap_ms: int,
                           threshold: float = 0.5, early_fusion: bool = False,
                           unsup_window_hops: int = 5) -> Dict[str, List[float]]:
    """early_fusion/unsup_window_hops MUST match how this model was trained
    (train_cpd.py's --early-fusion) -- read straight from the saved model
    bundle by run() below, never hand-picked, so train==inference holds for
    the augmented channel too."""
    import numpy as np

    out = {}
    for clip in clips:
        features = clip["features"]
        if early_fusion:
            features = augment_with_unsupervised_score(features, unsup_window_hops)
        x = windowed_features(features, window_hops)
        proba = smooth_curve(model.predict_proba(x)[:, 1], window_hops=2)
        out[clip["clip_id"]] = pick_peaks(
            np.asarray(proba), clip["hop_ms"], min_gap_ms=min_gap_ms, min_score=threshold)
    return out


def ensemble_predictions(
    clips: List[dict], model, window_hops: int, min_gap_ms: int,
    *, candidate_min_score: Optional[float] = None, snap_radius_hops: int = 3,
    refine_threshold: float = 0.3,
) -> Dict[str, List[float]]:
    """Detect (unsupervised) -> refine (supervised) prototype: unsupervised_
    boundary_score's peaks are CANDIDATE generation cast with a wide net
    (candidate_min_score defaults to the curve's own MEAN with no +std term
    -- looser than the standalone unsupervised baseline's mean+std bar,
    since recall here only needs to be good enough that the true boundary
    is SOMEWHERE near a candidate, not that every candidate is already a
    correct answer). Each candidate then SNAPS to the nearest local max of
    the supervised model's smoothed P(boundary) within +/-snap_radius_hops
    (the tight-localization strength the standalone supervised model
    already has), and survives only if that snapped instant clears
    refine_threshold -- otherwise the candidate is a false-positive detect
    the refiner rejects. This is the complementary-strengths architecture:
    unsupervised supplies recall/coverage, supervised supplies precision/
    localization."""
    import numpy as np

    out: Dict[str, List[float]] = {}
    for clip in clips:
        features, hop_ms = clip["features"], clip["hop_ms"]
        unsup_score = unsupervised_boundary_score(features, window_hops)
        if not unsup_score.size:
            out[clip["clip_id"]] = []
            continue
        min_score = candidate_min_score
        if min_score is None:
            min_score = float(unsup_score.mean())
        candidate_idxs = find_peak_indices(unsup_score, hop_ms, min_gap_ms=min_gap_ms, min_score=min_score)

        x = windowed_features(features, window_hops)
        sup_proba = np.asarray(smooth_curve(model.predict_proba(x)[:, 1], window_hops=2))
        n = len(sup_proba)

        snapped: set = set()
        for c in candidate_idxs:
            lo, hi = max(0, c - snap_radius_hops), min(n, c + snap_radius_hops + 1)
            if hi <= lo:
                continue
            local_best = lo + int(np.argmax(sup_proba[lo:hi]))
            if sup_proba[local_best] >= refine_threshold:
                snapped.add(local_best)

        out[clip["clip_id"]] = sorted((i * hop_ms) / 1000.0 for i in snapped)
    return out


def current_segmenter_predictions(clips: List[dict]) -> Dict[str, List[float]]:
    """Best-effort v4_segment.segment_video sanity check -- see module
    docstring. Re-runs the FULL production L1 signals (motion + scene) on
    each val clip's raw video, since the CPD feature npz only carries the
    9-channel CPD subset, not everything segment_video needs (action_points,
    camera_stability, transition_points, shot/composition points)."""
    from app.services.l1.motion_dynamics import compute_motion_dynamics
    from app.services.l1.scene_cuts import compute_scene_cuts
    from app.services.l3.v4_segment import segment_video

    out: Dict[str, List[float]] = {}
    for clip in clips:
        clip_id = clip["clip_id"]
        video_path = os.path.join(DATA_DIR, "raw_videos", f"{clip_id}.mp4")
        if not os.path.exists(video_path):
            continue
        duration_ms = int(round(clip["duration_sec"] * 1000))
        try:
            motion = compute_motion_dynamics(video_path, duration_ms)
            scene = compute_scene_cuts(video_path, duration_ms)
            if not motion.has_motion:
                continue
            cuts = segment_video(
                file_id=clip_id, duration_ms=duration_ms, speech_spans=[],
                motion=motion.to_dict(), audio={}, scene=scene.to_dict())
        except Exception:
            logger.exception("current-segmenter sanity check failed for %s", clip_id)
            continue
        marks = sorted({c.src_in_ms / 1000.0 for c in cuts} | {c.src_out_ms / 1000.0 for c in cuts})
        out[clip_id] = marks
    return out


def report(clips: List[dict], predictions_by_source: Dict[str, Dict[str, List[float]]],
          min_agree: int = 2) -> None:
    """min_agree MUST match how this ground truth was hand-marked --
    GEBD's real multi-annotator consensus wants 2 (the default); the
    in-domain harness (a SINGLE annotator, me) needs 1, or
    consensus_boundaries_sec silently drops every mark (no combination of
    annotators can ever reach 2 when there's only 1)."""
    gt_by_clip = {c["clip_id"]: consensus_boundaries_sec(c["annotator_boundaries_sec"], min_agree=min_agree)
                 for c in clips}
    duration_by_clip = {c["clip_id"]: c["duration_sec"] for c in clips}

    header = (f"{'source':<28}{'P@.05':>7}{'R@.05':>7}"
             + "".join(f"{t:>7.2f}" for t in REL_DIS_THRESHOLDS) + f"{'mean':>8}")
    print(header)
    print("-" * len(header))
    for source, pred_by_clip in predictions_by_source.items():
        precision, recall, _f1 = precision_recall_f1_at_threshold(
            pred_by_clip, gt_by_clip, duration_by_clip, rel_threshold=0.05)
        curve = f1_curve(pred_by_clip, gt_by_clip, duration_by_clip)
        mean_f1 = sum(curve.values()) / len(curve)
        row = (f"{source:<28}{precision:>7.3f}{recall:>7.3f}"
              + "".join(f"{curve[t]:>7.3f}" for t in REL_DIS_THRESHOLDS) + f"{mean_f1:>8.3f}")
        print(row)
    print("\n(P@.05/R@.05 = precision/recall at the strict 0.05 Rel.Dis threshold; "
         "F1@0.05 = that column's blend; 'mean' = the 0.05->0.5 curve average)")
    n_gt_empty = sum(1 for v in gt_by_clip.values() if not v)
    print(f"{len(clips)} val clip(s), {n_gt_empty} with no consensus boundary at all.")


def report_by_regime(
    clips: List[dict], predictions_by_source: Dict[str, Dict[str, List[float]]],
    thresholds: Sequence[float] = (0.05, 0.02), min_agree: int = 2,
) -> None:
    """Per-regime F1 breakdown (the in-domain harness's own deliverable --
    GEBD clips all have regime="" and collapse into one "unknown" row, which
    is fine/expected there; this is meant for the in-domain manifest, where
    every clip carries a real regime label). See report()'s min_agree note."""
    gt_by_clip = {c["clip_id"]: consensus_boundaries_sec(c["annotator_boundaries_sec"], min_agree=min_agree)
                 for c in clips}
    duration_by_clip = {c["clip_id"]: c["duration_sec"] for c in clips}
    regimes = sorted({(c.get("regime") or "unknown") for c in clips})

    for source, pred_by_clip in predictions_by_source.items():
        print(f"\n=== {source} (per regime) ===")
        header = f"{'regime':<24}{'n':>4}" + "".join(f"{'F1@' + str(t):>10}" for t in thresholds)
        print(header)
        print("-" * len(header))
        for regime in regimes:
            clip_ids = [c["clip_id"] for c in clips if (c.get("regime") or "unknown") == regime]
            sub_pred = {cid: pred_by_clip.get(cid, []) for cid in clip_ids}
            sub_gt = {cid: gt_by_clip[cid] for cid in clip_ids}
            sub_dur = {cid: duration_by_clip[cid] for cid in clip_ids}
            row = f"{regime:<24}{len(clip_ids):>4}"
            for t in thresholds:
                row += f"{f1_at_threshold(sub_pred, sub_gt, sub_dur, t):>10.3f}"
            print(row)
        row = f"{'ALL':<24}{len(clips):>4}"
        for t in thresholds:
            row += f"{f1_at_threshold(pred_by_clip, gt_by_clip, duration_by_clip, t):>10.3f}"
        print(row)


def run(model_path: Optional[str], window_hops_unsup: int, min_gap_ms: int,
        with_segmenter_baseline: bool, threshold: float = 0.5, with_ensemble: bool = False,
        split: str = "val", manifest_name: str = "manifest.csv", per_regime: bool = False,
        min_agree: int = 2) -> None:
    import joblib

    clips = load_labeled_clips(DATA_DIR, split, manifest_name)
    if not clips:
        raise SystemExit(
            f"no labeled clips found for split={split!r} in {manifest_name} -- run "
            f"extract_features.py/build_labels.py for that split first.")
    logger.info("evaluating on %d %s clip(s)", len(clips), split)

    predictions: Dict[str, Dict[str, List[float]]] = {
        "unsupervised": unsupervised_predictions(clips, window_hops_unsup, min_gap_ms),
    }

    bundle = None
    if model_path and os.path.exists(model_path):
        bundle = joblib.load(model_path)
        predictions["supervised"] = supervised_predictions(
            clips, bundle["model"], bundle["window_hops"], min_gap_ms, threshold=threshold,
            early_fusion=bundle.get("early_fusion", False),
            unsup_window_hops=bundle.get("unsup_window_hops", 5))
    else:
        logger.warning("no trained model at %s -- skipping supervised row (run train_cpd.py)",
                       model_path)

    if with_ensemble and bundle is not None:
        predictions["ensemble (detect->refine)"] = ensemble_predictions(
            clips, bundle["model"], bundle["window_hops"], min_gap_ms)

    if with_segmenter_baseline:
        predictions["current_segmenter*"] = current_segmenter_predictions(clips)

    report(clips, predictions, min_agree=min_agree)
    if per_regime:
        report_by_regime(clips, predictions, min_agree=min_agree)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    ap.add_argument("--window-hops-unsupervised", type=int, default=5)
    ap.add_argument("--min-gap-ms", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="supervised peak-pick floor on smoothed P(boundary) -- the recall lever")
    ap.add_argument("--with-segmenter-baseline", action="store_true", default=True)
    ap.add_argument("--no-segmenter-baseline", dest="with_segmenter_baseline", action="store_false")
    ap.add_argument("--ensemble", dest="with_ensemble", action="store_true",
                    help="also report the detect(unsupervised)->refine(supervised) ensemble row")
    ap.add_argument("--split", default="val")
    ap.add_argument("--manifest-name", default="manifest.csv",
                    help="e.g. indomain_manifest.csv for the in-domain harness")
    ap.add_argument("--per-regime", action="store_true",
                    help="also print a per-regime F1@0.05/F1@0.02 breakdown (in-domain harness)")
    ap.add_argument("--min-agree", type=int, default=2,
                    help="consensus rule for ground truth -- 2 for GEBD (default), "
                        "1 for the single-annotator in-domain harness")
    args = ap.parse_args()
    run(args.model_path, args.window_hops_unsupervised, args.min_gap_ms,
        args.with_segmenter_baseline, args.threshold, args.with_ensemble,
        args.split, args.manifest_name, args.per_regime, args.min_agree)


if __name__ == "__main__":
    main()
