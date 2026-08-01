"""
Shared utilities for the CPD boundary-segmenter pipeline
(cpd_boundary_segmenter.plan.md). Every phase script in this directory
imports from here so the hop grid, channel layout, npz shape, consensus
rule, and Rel.Dis metric stay IDENTICAL across extraction, labeling,
training, evaluation, and inference -- "train == inference" and "the eval
number means what it says" both depend on every phase agreeing on this one
shared definition, not five independent reimplementations.

No app-specific import here beyond what's needed for typing -- this module
itself has no dependency on the FastAPI app, so it can be imported without
DB/env setup (only extract_features.py needs the real app.services.l1
extractors).
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Default sliding-window half-width (hops) for the supervised model's
# flattened-window input (Phase E.2) -- 5 hops = 500ms each side. The
# TRAINED model's own saved metadata is authoritative at eval/inference
# time (see train_cpd.py/infer_cpd.py); this is only the CLI default.
DEFAULT_WINDOW_HOPS = 5

MODEL_PATH_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "models", "cpd_boundary.joblib")

# 10 fps -- matches app.services.l1.cut_grid_params.MOTION_FPS exactly, so
# the motion channels need no resampling. appearance_drift (computed
# separately, see extract_features.py) is decoded on this SAME grid rather
# than scene_cuts.py's native 5 fps, so every channel in the feature matrix
# shares one hop.
HOP_MS = 100

# The channel layout every extracted feature matrix (T x C) uses, in column
# order -- cpd_boundary_segmenter.plan.md's locked channel table. dx/dy/zoom
# are the RAW signed camera vector (never a derived pan/scale-rate feature);
# action_energy is production's percentile-normalized channel, action_energy
# _raw is its un-normalized physical-unit twin (the plan's explicit "(+
# `_raw`)"); appearance_drift is the NEW histogram-correlation-drift curve
# (Phase C.2), kept in its own natural [0, ~1] scale like camera_coherence
# (neither is percentile-renormalized -- see scene_cuts_params.py's own
# comment on why correlation-based drift is already absolute-scaled).
CHANNEL_NAMES: Tuple[str, ...] = (
    "frame_diff", "camera_dx", "camera_dy", "camera_zoom",
    "action_energy", "action_energy_raw", "camera_coherence", "blur",
    "appearance_drift",
)


def manifest_rows(
    data_dir: str, split: Optional[str] = None, manifest_name: str = "manifest.csv",
) -> List[Dict[str, str]]:
    """<manifest_name> rows for ``split`` (or every row when None) -- the one
    place every phase script reads a manifest, so a column rename/addition
    only needs updating here. ``manifest_name`` defaults to the GEBD
    train/val manifest fetch_gebd.py writes; the in-domain harness (its own
    single-annotator clips, never mixed with GEBD rows) uses a separate
    file, e.g. indomain_manifest.csv, via this same reader."""
    path = os.path.join(data_dir, manifest_name)
    with open(path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if split is None or r["split"] == split]


def load_labeled_clips(
    data_dir: str, split: str, manifest_name: str = "manifest.csv",
) -> List[Dict[str, Any]]:
    """Every clip in ``split`` that has BOTH extracted features and labels
    on disk (build_labels.py already ran) -- {clip_id, duration_sec,
    features, hop_ms, labels, regime}. Silently skips a clip missing either
    (an incomplete/partial pipeline run), since train_cpd.py and eval_cpd.py
    both just want "whatever's ready", not a hard failure mid-batch.
    ``regime`` is "" when the manifest has no such column (GEBD clips) --
    only the in-domain manifest populates it, for per-regime reporting."""
    out: List[Dict[str, Any]] = []
    for row in manifest_rows(data_dir, split, manifest_name):
        npz_path = features_path(data_dir, row["clip_id"])
        if not os.path.exists(npz_path):
            continue
        data = load_clip_npz(npz_path)
        if "labels" not in data:
            continue
        out.append({
            "clip_id": row["clip_id"], "duration_sec": float(row["duration_sec"]),
            "annotator_boundaries_sec": json.loads(row["annotator_boundaries_sec_json"]),
            "regime": row.get("regime", ""),
            **data,
        })
    return out


def clip_id_for(video_id: str, start_sec: float, end_sec: float) -> str:
    """Kinetics-GEBD's own clip-id convention: <youtube_id>_<start>_<end>,
    zero-padded to 6 digits (matches how yt-dlp/ffmpeg-cut filenames and the
    manifest key each clip)."""
    return f"{video_id}_{int(round(start_sec)):06d}_{int(round(end_sec)):06d}"


# --------------------------------------------------------------------------
# Feature-matrix npz I/O -- ONE shape, used by extract/build_labels/train/eval.
# --------------------------------------------------------------------------

def features_path(data_dir: str, clip_id: str) -> str:
    return os.path.join(data_dir, "features", f"{clip_id}.npz")


def save_clip_npz(
    path: str, features: np.ndarray, hop_ms: int,
    channel_names: Sequence[str] = CHANNEL_NAMES,
    labels: Optional[np.ndarray] = None, clip_id: str = "",
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload: Dict[str, Any] = {
        "features": features.astype(np.float32),
        "hop_ms": np.int64(hop_ms),
        "channel_names": np.array(list(channel_names)),
        "clip_id": np.array(clip_id),
    }
    if labels is not None:
        payload["labels"] = labels.astype(np.float32)
    np.savez_compressed(path, **payload)


def load_clip_npz(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        out: Dict[str, Any] = {
            "features": z["features"],
            "hop_ms": int(z["hop_ms"]),
            "channel_names": [str(c) for c in z["channel_names"]],
            "clip_id": str(z["clip_id"]) if "clip_id" in z else "",
        }
        if "labels" in z:
            out["labels"] = z["labels"]
    return out


# --------------------------------------------------------------------------
# Phase D: GEBD consensus boundaries -> per-hop binary targets
# --------------------------------------------------------------------------

def consensus_boundaries_sec(
    annotator_boundaries_sec: Sequence[Sequence[float]],
    *, min_agree: int = 2, merge_window_s: float = 0.3,
) -> List[float]:
    """GEBD-style multi-annotator consensus. Every annotator's boundary
    marks are pooled and greedily chained into clusters (a mark joins the
    current cluster when it sits within merge_window_s of the cluster's
    OWN most recent mark -- clusters can't grow unboundedly wide one mark
    at a time, since consecutive real boundaries in GEBD's own ~5s clips
    are themselves usually seconds apart). A cluster survives only when
    marks from at least min_agree DISTINCT annotators contributed to it --
    a boundary only one annotator saw is noise, not consensus. Each
    surviving cluster's timestamp is the mean of its marks.

    Empty/degenerate annotator lists (a clip nobody marked, or every
    annotator list empty) simply yield no boundaries -- not an error."""
    marks: List[Tuple[float, int]] = []
    for annot_idx, boundaries in enumerate(annotator_boundaries_sec):
        for t in boundaries:
            marks.append((float(t), annot_idx))
    if not marks:
        return []
    marks.sort(key=lambda m: m[0])

    clusters: List[List[Tuple[float, int]]] = [[marks[0]]]
    for t, annot_idx in marks[1:]:
        if t - clusters[-1][-1][0] <= merge_window_s:
            clusters[-1].append((t, annot_idx))
        else:
            clusters.append([(t, annot_idx)])

    out: List[float] = []
    for cluster in clusters:
        distinct_annotators = {a for _t, a in cluster}
        if len(distinct_annotators) >= min_agree:
            out.append(sum(t for t, _a in cluster) / len(cluster))
    return sorted(out)


def hop_labels(
    consensus_sec: Sequence[float], duration_ms: int, hop_ms: int, tolerance_ms: float,
) -> np.ndarray:
    """Length-T float32 0/1 vector on the [0, duration_ms) hop grid: 1 at
    hop t iff its timestamp (t * hop_ms) is within tolerance_ms of ANY
    consensus boundary. T = ceil(duration_ms / hop_ms), matching how many
    hops the feature extractor actually produces for a clip of this
    duration."""
    n = max(0, -(-duration_ms // hop_ms))  # ceil div
    labels = np.zeros(n, dtype=np.float32)
    if not consensus_sec:
        return labels
    boundary_ms = sorted(t * 1000.0 for t in consensus_sec)
    for i in range(n):
        ts = i * hop_ms
        # boundary_ms is sorted and short (a handful per clip) -- linear
        # scan is simpler and fast enough than a binary search here.
        if any(abs(ts - b) <= tolerance_ms for b in boundary_ms):
            labels[i] = 1.0
    return labels


# --------------------------------------------------------------------------
# Phase F: GEBD F1 with the Relative-Distance (Rel.Dis) protocol.
#
# Rel.Dis is a FRACTION of the clip's own duration, not an absolute-time
# tolerance (short clips get a proportionally tighter window than long
# ones) -- the LOVEU/GEBD challenge's own definition. This is a from-spec
# reimplementation (the plan's Phase F.1 description), not a vendored copy
# of the official Challenge_eval_Code/eval.py; the matching rule below
# (greedy nearest-first, one-to-one) is the standard, documented approach
# for this kind of boundary-detection F1 and is what's reproduced here.
# --------------------------------------------------------------------------

def match_boundaries(
    pred_sec: Sequence[float], gt_sec: Sequence[float], threshold_sec: float,
) -> Tuple[int, int, int]:
    """Greedy nearest-first ONE-TO-ONE matching within threshold_sec.
    Returns (n_matched, n_pred, n_gt) so the caller can accumulate
    precision/recall across clips before dividing (matches GEBD's own
    micro-averaged-then-macro-averaged style loosely -- see f1_at_threshold,
    which macro-averages per-video F1, the more common reporting choice)."""
    if not pred_sec or not gt_sec:
        return 0, len(pred_sec), len(gt_sec)
    pairs = []
    for pi, p in enumerate(pred_sec):
        for gi, g in enumerate(gt_sec):
            d = abs(p - g)
            if d <= threshold_sec:
                pairs.append((d, pi, gi))
    pairs.sort(key=lambda x: x[0])
    used_pred: set = set()
    used_gt: set = set()
    matched = 0
    for _d, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matched += 1
    return matched, len(pred_sec), len(gt_sec)


def precision_recall_f1_at_threshold(
    pred_by_clip: Dict[str, Sequence[float]], gt_by_clip: Dict[str, Sequence[float]],
    duration_by_clip: Dict[str, float], rel_threshold: float,
) -> Tuple[float, float, float]:
    """Macro-averaged (precision, recall, F1) at one Rel.Dis threshold -- a
    clip with zero ground-truth AND zero predicted boundaries scores 1.0 on
    all three (correctly predicted "nothing here"); zero ground truth but a
    false positive prediction scores 0.0. Clips absent from duration_by_clip
    are skipped (can't compute an absolute threshold_sec from a relative one
    without a duration). Exposed separately (not just the blended F1) so a
    recall-vs-precision bottleneck is diagnosable directly, rather than
    inferred from F1 alone."""
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for clip_id, gt in gt_by_clip.items():
        duration = duration_by_clip.get(clip_id)
        if duration is None or duration <= 0:
            continue
        pred = pred_by_clip.get(clip_id, [])
        threshold_sec = rel_threshold * duration
        matched, n_pred, n_gt = match_boundaries(pred, gt, threshold_sec)
        if n_pred == 0 and n_gt == 0:
            precisions.append(1.0)
            recalls.append(1.0)
            f1s.append(1.0)
            continue
        precision = matched / n_pred if n_pred else 0.0
        recall = matched / n_gt if n_gt else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    n = len(f1s)
    if n == 0:
        return 0.0, 0.0, 0.0
    return sum(precisions) / n, sum(recalls) / n, sum(f1s) / n


def f1_at_threshold(
    pred_by_clip: Dict[str, Sequence[float]], gt_by_clip: Dict[str, Sequence[float]],
    duration_by_clip: Dict[str, float], rel_threshold: float,
) -> float:
    """Macro-averaged F1 alone -- see precision_recall_f1_at_threshold for
    the full breakdown and the matching rule."""
    _p, _r, f1 = precision_recall_f1_at_threshold(pred_by_clip, gt_by_clip, duration_by_clip, rel_threshold)
    return f1


REL_DIS_THRESHOLDS: Tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 11))  # 0.05..0.5


def f1_curve(
    pred_by_clip: Dict[str, Sequence[float]], gt_by_clip: Dict[str, Sequence[float]],
    duration_by_clip: Dict[str, float],
) -> Dict[float, float]:
    """{threshold: F1} across the standard 0.05..0.5 Rel.Dis sweep. F1@0.05
    is the strict headline number; mean(curve.values()) is the "0.05->0.5
    curve" summary the plan's Phase F.1 asks for."""
    return {t: f1_at_threshold(pred_by_clip, gt_by_clip, duration_by_clip, t)
            for t in REL_DIS_THRESHOLDS}


# --------------------------------------------------------------------------
# Phase E.1: unsupervised CPD baseline -- honest classical change-point
# detection, no training. If this already predicts GEBD boundaries, the
# features themselves carry the signal (the plan's own framing).
# --------------------------------------------------------------------------

def unsupervised_boundary_score(features: np.ndarray, window_hops: int) -> np.ndarray:
    """boundary_score(t) = Euclidean distance between the MEAN feature
    vector in the window immediately BEFORE t and the window immediately
    AFTER t -- a two-sample statistic on the multivariate series, the
    textbook classical CPD move. Each channel is z-scored against the
    WHOLE clip first (mean 0, std 1) so no single channel's raw scale
    (e.g. frame_diff's 0..1 vs camera_zoom's tiny per-hop deltas) dominates
    the distance purely by unit size -- this is the "per-clip normalized"
    step for THIS specific consumer, distinct from (and in addition to)
    each channel's own production normalization. The first/last
    window_hops instants (not enough history on one side) score 0."""
    t, c = features.shape
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    std[std < 1e-9] = 1.0
    z = (features - mean) / std

    score = np.zeros(t, dtype=np.float32)
    for i in range(window_hops, t - window_hops):
        before = z[i - window_hops:i].mean(axis=0)
        after = z[i:i + window_hops].mean(axis=0)
        score[i] = float(np.linalg.norm(after - before))
    return score


def augment_with_unsupervised_score(features: np.ndarray, unsup_window_hops: int) -> np.ndarray:
    """T x C -> T x (C+1): appends unsupervised_boundary_score as an EXTRA
    per-hop channel -- early fusion (the round-2 late-fusion "detect then
    snap" ensemble failed; this instead lets the supervised model see the
    unsupervised statistic as raw input and learn its own combination,
    rather than a fixed post-hoc rule). The SAME function is called at
    train and inference time (train_cpd.py / eval_cpd.py / infer_cpd.py),
    so train==inference holds for this augmented channel exactly like
    every other one."""
    score = unsupervised_boundary_score(features, unsup_window_hops)
    return np.concatenate([features, score.reshape(-1, 1).astype(features.dtype)], axis=1)


# --------------------------------------------------------------------------
# Phase E.2: supervised model's input -- a flattened window of raw channel
# values around each hop (shared by train_cpd.py and infer_cpd.py so the
# exact same preprocessing runs at train and inference time).
# --------------------------------------------------------------------------

def windowed_features(features: np.ndarray, window_hops: int) -> np.ndarray:
    """T x C -> T x ((2*window_hops+1)*C): each row is the flattened
    window of ``window_hops`` hops before/after (inclusive of) that hop,
    edge-padded by repeating the first/last row so every hop in the clip
    (including its very first/last) gets a full-width row -- a short clip
    is exactly where a real boundary might sit right at the edge, so it
    must not be dropped from training/inference."""
    t, c = features.shape
    pad = np.repeat(features[:1], window_hops, axis=0)
    pad_end = np.repeat(features[-1:], window_hops, axis=0)
    padded = np.concatenate([pad, features, pad_end], axis=0)
    rows = [padded[i:i + 2 * window_hops + 1].reshape(-1) for i in range(t)]
    return np.stack(rows, axis=0)


def smooth_curve(values: np.ndarray, window_hops: int) -> np.ndarray:
    """Centered moving average, edge-padded (replicate) so the output stays
    length T -- de-jitters an independent per-hop probability curve before
    peak-picking (Phase G's "smoothed probability -> peak-pick"; also
    applied in eval_cpd.py so the reported F1 matches what inference
    actually does, not an unsmoothed stand-in)."""
    if window_hops <= 0 or len(values) == 0:
        return values
    kernel = np.ones(2 * window_hops + 1) / (2 * window_hops + 1)
    pad = np.pad(values, (window_hops, window_hops), mode="edge")
    return np.convolve(pad, kernel, mode="valid")


# --------------------------------------------------------------------------
# Peak-picking -- shared by the unsupervised baseline (Phase E.1) and
# inference (Phase G): a continuous boundary score -> discrete timestamps.
# --------------------------------------------------------------------------

def find_peak_indices(
    score: np.ndarray, hop_ms: int, *, min_gap_ms: int = 300, min_score: float = 0.0,
) -> List[int]:
    """Local maxima of ``score`` at/above min_score, non-max-suppressed
    within +/-min_gap_ms so a wide bump yields one boundary, not a cluster
    -- same greedy strongest-first NMS shape as v4_segment._find_peaks.
    Returns HOP INDICES (not timestamps) so a caller can do further
    index-space work -- e.g. the detect->refine ensemble's snap-to-nearest-
    local-max-in-a-DIFFERENT-curve step -- before converting to seconds."""
    n = len(score)
    if n == 0:
        return []
    min_gap_hops = max(1, min_gap_ms // max(hop_ms, 1))
    candidates = sorted((i for i in range(n) if score[i] >= min_score), key=lambda i: -score[i])
    chosen: List[int] = []
    for i in candidates:
        if all(abs(i - c) > min_gap_hops for c in chosen):
            chosen.append(i)
    return sorted(chosen)


def pick_peaks(
    score: np.ndarray, hop_ms: int, *, min_gap_ms: int = 300, min_score: float = 0.0,
) -> List[float]:
    """find_peak_indices, converted to timestamps in SECONDS (GEBD's own
    unit) on the hop grid."""
    return [(i * hop_ms) / 1000.0 for i in find_peak_indices(score, hop_ms, min_gap_ms=min_gap_ms,
                                                              min_score=min_score)]
