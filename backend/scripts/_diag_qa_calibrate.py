#!/usr/bin/env python3
"""color_phase1.plan.md Part 1d: fit PASS bands against the human labels
from `_diag_qa_label_sheet.py`'s labels.json, validate on the held-out
split, and print a diff-ready block of proposed `metrics.py` constants.
Read-only: this script never edits metrics.py -- calibration PROPOSES, a
person commits.

Scope: only the metrics `_diag_qa_label_sheet.py` actually labels per-shot
readouts for -- `saturation_band`, `exposure_band`, `skin_perp_residual`.
The `GROUP_*_STD_*` consistency constants are also nominally "tunable" per
the plan, but the label sheet judges individual shots (raw vs graded
stills), not whole-group consistency, so there is no natural per-shot human
label to fit a group-level threshold against; calibrating them would need a
different (group-level) labeling mechanism and is deferred.

Fitting: every tunable metric here is judged as either
  - an "upper" bound (lower value = better: skin_perp_residual), or
  - a nested "window" (saturation_band's look-normalized ratio,
    exposure_band's luma vs target) where the INNER band is PASS.
Candidate cut points are the calibration set's own observed signal values
(decision-tree-style thresholding -- exact and sufficient at this sample
size). Per the plan: prefer the cut that yields ZERO human-labeled-bad
shots predicted "good" (a false-PASS), then maximize overall agreement
among the cuts that clear that bar; if no cut clears it, fall back to
maximizing agreement outright. The calibration set only ever picks the
cut; held-out agreement is reported but never used to pick it.

Usage: PYTHONPATH=. .venv/bin/python scripts/_diag_qa_calibrate.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from qa import metrics as m  # noqa: E402

OUT_ROOT = os.path.join(HERE, "_out", "qa")
SCOREBOARD_PATH = os.path.join(OUT_ROOT, "scoreboard.json")
LABELS_PATH = os.path.join(OUT_ROOT, "labels.json")

HELD_OUT_AGREEMENT_TARGET = 0.90
MIN_CALIBRATION_SAMPLES = 5

Pair = Tuple[float, bool]  # (signal, is_good)


def _saturation_signal(md: Dict[str, Any]) -> Optional[float]:
    v = md.get("normalized_ratio", md.get("value"))
    return float(v) if v is not None else None


def _exposure_signal(md: Dict[str, Any]) -> Optional[float]:
    v = md.get("median", md.get("value"))
    return float(v) if v is not None else None


def _plain_value_signal(md: Dict[str, Any]) -> Optional[float]:
    v = md.get("value")
    return float(v) if v is not None else None


# metric name -> (band shape, signal extractor, proposed metrics.py constant name)
TUNABLE_METRICS: Dict[str, Tuple[str, Callable[[Dict[str, Any]], Optional[float]], str]] = {
    "saturation_band": ("window", _saturation_signal, "SATURATION_RATIO_PASS"),
    "exposure_band": ("window", _exposure_signal, "EXPOSURE_BAND_PASS"),
    "skin_perp_residual": ("upper", _plain_value_signal, "SKIN_PERP_PASS"),
}


def _shot_uid(project_id: str, thread_id: str, shot_key: str) -> str:
    return f"{project_id}::{thread_id}::{shot_key}"


def _load_labeled_pairs(
    metric_name: str, extractor: Callable[[Dict[str, Any]], Optional[float]],
    labels: Dict[str, Any], scoreboard_by_uid: Dict[str, Any],
) -> Tuple[List[Pair], List[Pair]]:
    cal: List[Pair] = []
    held: List[Pair] = []
    for uid, label in labels.items():
        verdict = label.get("verdict")
        if verdict not in ("good", "bad"):
            continue
        shot = scoreboard_by_uid.get(uid)
        if shot is None:
            continue
        md = shot["metrics"].get(metric_name)
        if md is None or md.get("verdict") == "na":
            continue
        signal = extractor(md)
        if signal is None:
            continue
        pair: Pair = (signal, verdict == "good")
        (cal if label.get("split") == "calibration" else held).append(pair)
    return cal, held


def _agreement(pairs: List[Pair], predict: Callable[[float], bool]) -> float:
    if not pairs:
        return 0.0
    correct = sum(1 for signal, is_good in pairs if predict(signal) == is_good)
    return correct / len(pairs)


def _fit_upper(pairs: List[Pair]) -> Optional[Dict[str, Any]]:
    if not pairs:
        return None
    values = sorted({s for s, _ in pairs})
    candidates = [values[0] - 1e-6] + values

    scored = []
    for t in candidates:
        def predict(s: float, t: float = t) -> bool:
            return s <= t
        false_good_on_bad = sum(1 for s, good in pairs if not good and predict(s))
        scored.append((t, _agreement(pairs, predict), false_good_on_bad))

    zero_fp = [(t, a) for t, a, fp in scored if fp == 0]
    pool = zero_fp if zero_fp else [(t, a) for t, a, _fp in scored]
    best_t, best_agreement = max(pool, key=lambda ta: ta[1])
    return {
        "shape": "upper", "pass_threshold": best_t,
        "calibration_agreement": best_agreement, "zero_false_good_achieved": bool(zero_fp),
    }


def _fit_window(pairs: List[Pair]) -> Optional[Dict[str, Any]]:
    if not pairs:
        return None
    values = sorted({s for s, _ in pairs})
    lo_candidates = [values[0] - 1e-6] + values
    hi_candidates = values + [values[-1] + 1e-6]

    scored = []
    for lo in lo_candidates:
        for hi in hi_candidates:
            if hi < lo:
                continue

            def predict(s: float, lo: float = lo, hi: float = hi) -> bool:
                return lo <= s <= hi
            false_good_on_bad = sum(1 for s, good in pairs if not good and predict(s))
            scored.append((lo, hi, _agreement(pairs, predict), false_good_on_bad))

    zero_fp = [(lo, hi, a) for lo, hi, a, fp in scored if fp == 0]
    pool = zero_fp if zero_fp else [(lo, hi, a) for lo, hi, a, _fp in scored]
    best_lo, best_hi, best_agreement = max(pool, key=lambda t: t[2])
    return {
        "shape": "window", "pass_window": (best_lo, best_hi),
        "calibration_agreement": best_agreement, "zero_false_good_achieved": bool(zero_fp),
    }


def _validate(fit: Dict[str, Any], held_pairs: List[Pair]) -> Optional[float]:
    if not held_pairs:
        return None
    if fit["shape"] == "upper":
        t = fit["pass_threshold"]
        return _agreement(held_pairs, lambda s: s <= t)
    lo, hi = fit["pass_window"]
    return _agreement(held_pairs, lambda s: lo <= s <= hi)


def calibrate_metric(
    metric_name: str, kind: str, extractor: Callable[[Dict[str, Any]], Optional[float]],
    labels: Dict[str, Any], scoreboard_by_uid: Dict[str, Any],
) -> Dict[str, Any]:
    """Pure aggregation entry point (no printing) -- kept separate from
    main() so it's directly unit-testable."""
    cal, held = _load_labeled_pairs(metric_name, extractor, labels, scoreboard_by_uid)
    result: Dict[str, Any] = {
        "metric": metric_name, "calibration_n": len(cal), "held_out_n": len(held), "fit": None,
        "held_out_agreement": None, "adopt": False,
    }
    if len(cal) < MIN_CALIBRATION_SAMPLES:
        return result
    fit = _fit_window(cal) if kind == "window" else _fit_upper(cal)
    result["fit"] = fit
    held_agreement = _validate(fit, held) if fit else None
    result["held_out_agreement"] = held_agreement
    result["adopt"] = held_agreement is not None and held_agreement >= HELD_OUT_AGREEMENT_TARGET
    return result


def _format_proposal(metric_name: str, constant_name: str, fit: Dict[str, Any]) -> str:
    current = getattr(m, constant_name, None)
    if fit["shape"] == "upper":
        proposed = round(fit["pass_threshold"], 3)
    else:
        lo, hi = fit["pass_window"]
        proposed = (round(lo, 3), round(hi, 3))
    return f"{constant_name} = {proposed!r}  # was {current!r} ({metric_name})"


def main() -> None:
    if not os.path.exists(LABELS_PATH):
        raise SystemExit(f"no labels at {LABELS_PATH} -- run _diag_qa_label_sheet.py first")
    if not os.path.exists(SCOREBOARD_PATH):
        raise SystemExit(f"no scoreboard at {SCOREBOARD_PATH} -- run _diag_qa_score.py first")
    with open(LABELS_PATH) as f:
        labels = json.load(f)
    with open(SCOREBOARD_PATH) as f:
        scoreboard = json.load(f)
    scoreboard_by_uid = {_shot_uid(s["project_id"], s["thread_id"], s["shot_key"]): s for s in scoreboard["shots"]}

    n_labeled = sum(1 for label in labels.values() if label.get("verdict") in ("good", "bad"))
    if n_labeled == 0:
        print(f"0/{len(labels)} shots labeled yet in {LABELS_PATH}.")
        print("Open _out/qa/label_sheet.html, mark good/bad per shot, then paste the JSON preview")
        print("over labels.json (or hand-edit labels.json directly) and re-run this script.")
        return

    print(f"{n_labeled}/{len(labels)} shots labeled -- calibrating {len(TUNABLE_METRICS)} tunable metrics\n")

    adopted: List[Tuple[str, str, Dict[str, Any]]] = []
    for metric_name, (kind, extractor, constant_name) in TUNABLE_METRICS.items():
        r = calibrate_metric(metric_name, kind, extractor, labels, scoreboard_by_uid)
        if r["fit"] is None:
            print(f"{metric_name}: only {r['calibration_n']} calibration-set labels "
                  f"(need >= {MIN_CALIBRATION_SAMPLES}) -- skipped")
            continue
        fit = r["fit"]
        cal_agree = fit["calibration_agreement"]
        held_agree = r["held_out_agreement"]
        held_str = f"{held_agree:.1%}" if held_agree is not None else "n/a (no held-out labels)"
        verdict_str = "ADOPT" if r["adopt"] else "OVERFIT -- do not adopt, widen/simplify"
        print(f"{metric_name}: cal_n={r['calibration_n']} held_n={r['held_out_n']} "
              f"cal_agreement={cal_agree:.1%} held_agreement={held_str} "
              f"zero_false_good={fit['zero_false_good_achieved']} -> {verdict_str}")
        if r["adopt"]:
            adopted.append((metric_name, constant_name, fit))

    if adopted:
        print("\n# ---- diff-ready proposed metrics.py constants (paste by hand) ----")
        for metric_name, constant_name, fit in adopted:
            print(_format_proposal(metric_name, constant_name, fit))
    else:
        print("\nno metric cleared the held-out bar -- nothing to propose yet")


if __name__ == "__main__":
    main()
