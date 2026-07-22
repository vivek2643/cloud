#!/usr/bin/env python3
"""color_qa_harness.plan.md A.5: scoring -- reads samples.json (from
`_diag_qa_sample.py`), runs every A.4 metric per shot, rolls group-level
consistency/subject-exposure metrics up per scene group, then per project,
then into one global ranked scoreboard.

Per-shot metrics score the HERO frame specifically (the single deliberately
-chosen representative still, same anchor `measure_span`'s subject_luma/
subject_lab already use) -- the additional ~25/50/75% keyframes A.2 samples
back the contact sheets (`_diag_qa_sheets.py`) and are available in
samples.json for a future multi-frame-robustness pass, not averaged into the
v1 score.

Look-fidelity (#5) bakes the look's OWN thumbnail cube (identity CDL +
`build_look_grid`, exactly `_diag_look_thumbs.thumb_cube_output`'s recipe)
against the shot's own raw hero still, not the bundled look-gallery
reference image -- so the comparison is "this look, on this actual shot."

Writes backend/scripts/_out/qa/scoreboard.json and prints the ranked
failure-class table (`_grade_all_projects.py`'s SUMMARY style), including a
log_flat-vs-rec709 split (Part B.1's evidence: does the log/flat subset
cluster in WARN/FAIL enough to justify building the IDT).

Usage: PYTHONPATH=. .venv/bin/python scripts/_diag_qa_score.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from app.services.l3.grade.cdl import Grade  # noqa: E402
from app.services.l3.grade.look_engine import LookSpec, build_look_grid  # noqa: E402
from app.services.l3.grade.lut_bake import _sample_lut_trilinear, bake_cube_text, parse_cube_text  # noqa: E402
from qa import metrics as m  # noqa: E402

OUT_ROOT = os.path.join(HERE, "_out", "qa")
SAMPLES_PATH = os.path.join(OUT_ROOT, "samples.json")
SCOREBOARD_PATH = os.path.join(OUT_ROOT, "scoreboard.json")

_SEVERITY = {"pass": 0, "warn": 1, "fail": 2, "na": 0}


def _load_rgb01(path: str) -> np.ndarray:
    import cv2

    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not read {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _hero_frame(shot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for f in shot.get("frames") or []:
        if f.get("is_hero"):
            return f
    frames = shot.get("frames") or []
    return frames[0] if frames else None


def _look_only_still(look_engine: dict, raw01: np.ndarray) -> Optional[np.ndarray]:
    """The look's OWN color transform on this shot's raw still -- identity
    CDL + the look's grid, exactly `_diag_look_thumbs.thumb_cube_output`."""
    try:
        spec = LookSpec.from_dict(look_engine)
        grid = build_look_grid(spec)
        cube_text = bake_cube_text(Grade(), working_space="rec709_v1", creative_lut_grid=grid, tone_contrast=0.0)
        lut_grid, _size = parse_cube_text(cube_text)
        return np.clip(_sample_lut_trilinear(lut_grid, raw01), 0.0, 1.0)
    except Exception as e:
        print(f"    !! look-only bake failed: {e}")
        return None


def score_shot(shot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    hero = _hero_frame(shot)
    if hero is None:
        return None
    raw_path = os.path.join(OUT_ROOT, hero["raw_path"])
    graded_path = os.path.join(OUT_ROOT, hero["graded_path"])
    try:
        raw01 = _load_rgb01(raw_path)
        graded01 = _load_rgb01(graded_path)
    except Exception as e:
        print(f"    !! {shot['shot_key']}: failed to load frames: {e}")
        return None

    results: Dict[str, m.MetricResult] = {}
    results.update(m.exposure_metrics(graded01, raw01=raw01))
    results["neutral_axis_deviation"] = m.neutral_axis_deviation(graded01)
    results["skin_perp_residual"] = m.skin_perp_residual(graded01, shot.get("subject_box"))
    results["saturation_band"] = m.saturation_band(graded01, raw01=raw01)
    results["banding_score"] = m.banding_score(graded01)

    if shot.get("look_engine"):
        look_only01 = _look_only_still(shot["look_engine"], raw01)
        if look_only01 is not None:
            results["look_fidelity_cosine"] = m.look_fidelity_metric(graded01, raw01, look_only01)

    summary_graded = m.summarize_shot(graded01, subject_box=shot.get("subject_box"))
    summary_raw = m.summarize_shot(raw01, subject_box=shot.get("subject_box"))

    shot_verdict = m.worst_verdict([r.verdict for r in results.values()])
    return {
        "project_id": shot["project_id"], "project_label": shot["project_label"],
        "thread_id": shot["thread_id"], "shot_key": shot["shot_key"],
        "group_id": shot.get("group_id"), "has_grade": shot.get("has_grade"),
        "is_log_flat": shot.get("is_log_flat"), "verdict": shot_verdict,
        "metrics": {k: v.to_dict() for k, v in results.items()},
        "summary_graded": summary_graded, "summary_raw": summary_raw,
    }


def score_groups(shot_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Roll consistency + subject-exposure metrics up per (thread, group)."""
    by_group: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    for s in shot_scores:
        if s.get("group_id") is None:
            continue
        by_group[(s["thread_id"], s["group_id"])].append(s)

    out: List[Dict[str, Any]] = []
    for (thread_id, group_id), members in by_group.items():
        if len(members) < 2:
            continue
        graded_summaries = [s["summary_graded"] for s in members]
        raw_summaries = [s["summary_raw"] for s in members]
        consistency = m.group_consistency_metrics(graded_summaries, raw=raw_summaries)
        subject_exposure = m.group_subject_exposure_metrics(graded_summaries, raw=raw_summaries)
        if not consistency and not subject_exposure:
            continue
        out.append({
            "thread_id": thread_id, "group_id": group_id,
            "project_id": members[0]["project_id"], "project_label": members[0]["project_label"],
            "member_shot_keys": [s["shot_key"] for s in members],
            "metrics": {k: v.to_dict() for k, v in {**consistency, **subject_exposure}.items()},
        })
    return out


def _rank_failure_classes(shot_scores: List[Dict[str, Any]], group_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    agg: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"pass": 0, "warn": 0, "fail": 0, "na": 0, "severity_sum": 0})
    for shot in shot_scores:
        for name, md in shot["metrics"].items():
            v = md["verdict"]
            agg[name][v] += 1
            agg[name]["severity_sum"] += _SEVERITY.get(v, 0)
    for grp in group_scores:
        for name, md in grp["metrics"].items():
            v = md["verdict"]
            agg[name][v] += 1
            agg[name]["severity_sum"] += _SEVERITY.get(v, 0)

    ranked = []
    for name, counts in agg.items():
        total = counts["pass"] + counts["warn"] + counts["fail"]
        mean_severity = counts["severity_sum"] / total if total else 0.0
        ranked.append({
            "metric": name, "pass": counts["pass"], "warn": counts["warn"], "fail": counts["fail"],
            "na": counts["na"], "total_scored": total, "mean_severity": round(mean_severity, 3),
            "rank_score": counts["fail"] * 2 + counts["warn"],
        })
    ranked.sort(key=lambda r: -r["rank_score"])
    return ranked


def _project_rollup(shot_scores: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_project: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in shot_scores:
        by_project[s["project_id"]].append(s)
    out = []
    for pid, shots in by_project.items():
        counts = {"pass": 0, "warn": 0, "fail": 0, "na": 0}
        for s in shots:
            counts[s["verdict"]] += 1
        n = len(shots)
        worst = [f["metric"] for f in _rank_failure_classes(shots, [])[:3] if f["rank_score"] > 0]
        out.append({
            "project_id": pid, "project_label": shots[0]["project_label"],
            "shot_count": n, "pass": counts["pass"], "warn": counts["warn"], "fail": counts["fail"],
            "na": counts["na"], "pass_pct": round(100.0 * counts["pass"] / n, 1) if n else 0.0,
            "worst_metric_classes": worst,
        })
    out.sort(key=lambda p: p["pass_pct"])
    return out


def main() -> None:
    if not os.path.exists(SAMPLES_PATH):
        raise SystemExit(f"no samples manifest at {SAMPLES_PATH} -- run _diag_qa_sample.py first")
    with open(SAMPLES_PATH) as f:
        samples = json.load(f)

    shot_scores: List[Dict[str, Any]] = []
    for i, shot in enumerate(samples):
        score = score_shot(shot)
        if score is not None:
            shot_scores.append(score)
        if (i + 1) % 10 == 0:
            print(f"  scored {i + 1}/{len(samples)} shots...", flush=True)

    group_scores = score_groups(shot_scores)
    ranked = _rank_failure_classes(shot_scores, group_scores)
    projects = _project_rollup(shot_scores)

    # ShotSummary (a dataclass) was needed as-is for score_groups' grouping
    # math above; make the per-shot records JSON-safe before they go in the
    # scoreboard.
    for s in shot_scores:
        s["summary_graded"] = dataclasses.asdict(s["summary_graded"])
        s["summary_raw"] = dataclasses.asdict(s["summary_raw"])

    log_flat_shots = [s for s in shot_scores if s.get("is_log_flat")]
    rec709_shots = [s for s in shot_scores if not s.get("is_log_flat")]
    log_flat_ranked = _rank_failure_classes(log_flat_shots, []) if log_flat_shots else []
    rec709_ranked = _rank_failure_classes(rec709_shots, []) if rec709_shots else []

    overall = {"pass": 0, "warn": 0, "fail": 0, "na": 0}
    for s in shot_scores:
        overall[s["verdict"]] += 1
    n = len(shot_scores)

    scoreboard = {
        "shot_count": n,
        "group_count": len(group_scores),
        "overall": overall,
        "overall_pass_pct": round(100.0 * overall["pass"] / n, 1) if n else 0.0,
        "failure_classes_ranked": ranked,
        "log_flat_shot_count": len(log_flat_shots),
        "rec709_shot_count": len(rec709_shots),
        "failure_classes_ranked_log_flat": log_flat_ranked,
        "failure_classes_ranked_rec709": rec709_ranked,
        "projects": projects,
        "shots": shot_scores,
        "groups": group_scores,
    }
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(SCOREBOARD_PATH, "w") as f:
        json.dump(scoreboard, f, indent=2)

    print(f"\n==== SCOREBOARD ({n} shots, {len(group_scores)} scored groups) ====")
    print(f"overall: pass={overall['pass']} warn={overall['warn']} fail={overall['fail']} na={overall['na']} "
         f"({scoreboard['overall_pass_pct']}% pass)")

    print(f"\n{'metric':32} {'pass':>5} {'warn':>5} {'fail':>5} {'na':>5} {'mean_sev':>9} {'rank':>6}")
    for r in ranked:
        print(f"{r['metric']:32} {r['pass']:5} {r['warn']:5} {r['fail']:5} {r['na']:5} "
             f"{r['mean_severity']:9.3f} {r['rank_score']:6}")

    print(f"\n==== log_flat subset ({len(log_flat_shots)} shots) vs rec709 subset ({len(rec709_shots)} shots) ====")
    print(f"{'metric':32} {'log_flat rank':>14} {'rec709 rank':>12}")
    names = {r["metric"] for r in log_flat_ranked} | {r["metric"] for r in rec709_ranked}
    lf_by_name = {r["metric"]: r for r in log_flat_ranked}
    r7_by_name = {r["metric"]: r for r in rec709_ranked}
    for name in sorted(names, key=lambda n: -(lf_by_name.get(n, {}).get("rank_score", 0))):
        print(f"{name:32} {lf_by_name.get(name, {}).get('rank_score', 0):14} "
             f"{r7_by_name.get(name, {}).get('rank_score', 0):12}")

    print("\n==== PROJECT ROLLUP (worst-first) ====")
    print(f"{'project':22} {'shots':>5} {'pass':>5} {'warn':>5} {'fail':>5} {'pass%':>7}  worst classes")
    for p in projects:
        print(f"{p['project_label'][:22]:22} {p['shot_count']:5} {p['pass']:5} {p['warn']:5} {p['fail']:5} "
             f"{p['pass_pct']:6.1f}%  {', '.join(p['worst_metric_classes']) or '-'}")

    print(f"\nwrote {SCOREBOARD_PATH}")


if __name__ == "__main__":
    main()
