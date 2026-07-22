#!/usr/bin/env python3
"""color_qa_harness.plan.md A.3: contact sheets + a global index -- the
human-eyeball backstop for what the A.4 metrics can't fully capture.

Per project: one PNG, an hstacked RAW | GRADED row per shot's hero frame
(labeled with shot_key + its pass/warn/fail badge, same ffmpeg
hstack/vstack/drawtext recipe as `_grade_v1_frames.py`), vstacked. A global
backend/scripts/_out/qa/index.html thumbnails every project sheet annotated
with that project's rollup score, sorted worst-first.

Reads samples.json (frame paths) and scoreboard.json (verdicts + rollup),
joined by (thread_id, shot_key).

Usage: PYTHONPATH=. .venv/bin/python scripts/_diag_qa_sheets.py
"""
from __future__ import annotations

import html
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

OUT_ROOT = os.path.join(HERE, "_out", "qa")
SAMPLES_PATH = os.path.join(OUT_ROOT, "samples.json")
SCOREBOARD_PATH = os.path.join(OUT_ROOT, "scoreboard.json")
SHEETS_DIR = os.path.join(OUT_ROOT, "sheets")
INDEX_PATH = os.path.join(OUT_ROOT, "index.html")
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
VERDICT_COLOR = {"pass": "0x2ecc71", "warn": "0xf1c40f", "fail": "0xe74c3c", "na": "0x95a5a6"}


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "project"


def _hero_paths(shot: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    for f in shot.get("frames") or []:
        if f.get("is_hero"):
            return f["raw_path"], f["graded_path"]
    frames = shot.get("frames") or []
    if frames:
        return frames[0]["raw_path"], frames[0]["graded_path"]
    return None, None


def _label(img_path: str, text: str, color: str, out_path: str) -> None:
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", img_path, "-vf",
             f"drawtext=fontfile={FONT}:text='{text}':x=8:y=8:fontsize=20:"
             f"fontcolor=white:box=1:boxcolor={color}@0.75:boxborderw=6",
             out_path],
            check=True, capture_output=True,
        )
    except Exception:
        subprocess.run(["cp", img_path, out_path], check=True)


def build_project_sheet(shots: List[Dict[str, Any]], out_path: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="edso_qa_sheet_") as tmp:
        row_imgs = []
        for i, shot in enumerate(shots):
            raw_rel, graded_rel = _hero_paths(shot)
            if not raw_rel or not graded_rel:
                continue
            raw_path = os.path.join(OUT_ROOT, raw_rel)
            graded_path = os.path.join(OUT_ROOT, graded_rel)
            if not (os.path.exists(raw_path) and os.path.exists(graded_path)):
                continue
            verdict = shot.get("verdict", "na")
            color = VERDICT_COLOR.get(verdict, "0x95a5a6")
            rawL = os.path.join(tmp, f"raw_{i}.jpg")
            gradedL = os.path.join(tmp, f"graded_{i}.jpg")
            _label(raw_path, f"{shot['shot_key']} RAW", "0x333333", rawL)
            _label(graded_path, f"{shot['shot_key']} GRADED [{verdict.upper()}]", color, gradedL)
            row = os.path.join(tmp, f"row_{i}.jpg")
            try:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", rawL, "-i", gradedL, "-filter_complex", "hstack=inputs=2", row],
                    check=True, capture_output=True,
                )
                row_imgs.append(row)
            except Exception as e:
                print(f"    !! row failed for {shot['shot_key']}: {e}")
        if not row_imgs:
            return False
        inputs = []
        for r in row_imgs:
            inputs += ["-i", r]
        subprocess.run(
            ["ffmpeg", "-y", *inputs, "-filter_complex", f"vstack=inputs={len(row_imgs)}", out_path],
            check=True, capture_output=True,
        )
        return True


def _write_index(rows: List[Dict[str, Any]]) -> None:
    cards = []
    for r in rows:
        classes = ", ".join(r["worst_metric_classes"]) or "-"
        cards.append(f"""
        <section class="card">
          <h2>{html.escape(r['label'])} <span class="pct">{r['pass_pct']}% pass</span></h2>
          <p>{r['shot_count']} shots -- pass {r['pass']} / warn {r['warn']} / fail {r['fail']}
             -- worst: {html.escape(classes)}</p>
          <img src="{html.escape(r['sheet_path'])}" loading="lazy" />
        </section>""")
    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Color QA scoreboard</title>
<style>
body {{ font-family: -apple-system, sans-serif; background:#111; color:#eee; margin:24px; }}
h1 {{ font-weight:600; }}
.card {{ margin-bottom:32px; border-bottom:1px solid #333; padding-bottom:16px; }}
.card h2 {{ font-size:16px; font-weight:600; }}
.pct {{ color:#9ad; font-weight:400; margin-left:8px; }}
.card p {{ color:#aaa; font-size:13px; }}
.card img {{ max-width:100%; border-radius:4px; }}
</style></head>
<body>
<h1>Color QA -- project contact sheets (worst-first)</h1>
{"".join(cards)}
</body></html>"""
    with open(INDEX_PATH, "w") as f:
        f.write(doc)


def main() -> None:
    if not (os.path.exists(SAMPLES_PATH) and os.path.exists(SCOREBOARD_PATH)):
        raise SystemExit(
            f"missing {SAMPLES_PATH} / {SCOREBOARD_PATH} -- run _diag_qa_sample.py and _diag_qa_score.py first"
        )
    with open(SAMPLES_PATH) as f:
        samples = json.load(f)
    with open(SCOREBOARD_PATH) as f:
        scoreboard = json.load(f)

    verdict_by_key = {(s["thread_id"], s["shot_key"]): s["verdict"] for s in scoreboard["shots"]}
    for shot in samples:
        shot["verdict"] = verdict_by_key.get((shot["thread_id"], shot["shot_key"]), "na")

    by_project: Dict[str, List[Dict[str, Any]]] = {}
    for shot in samples:
        by_project.setdefault(shot["project_id"], []).append(shot)

    os.makedirs(SHEETS_DIR, exist_ok=True)
    project_rollup = {p["project_id"]: p for p in scoreboard["projects"]}
    sheet_rows = []
    for pid, shots in by_project.items():
        label = shots[0]["project_label"]
        slug = _slugify(f"{label}_{pid[:8]}")
        out_path = os.path.join(SHEETS_DIR, f"{slug}.png")
        print(f"[sheet] {label} ({pid[:8]}) -- {len(shots)} shots")
        ok = build_project_sheet(shots, out_path)
        if not ok:
            print(f"    !! no rows rendered for {label}")
            continue
        rollup = project_rollup.get(pid, {})
        sheet_rows.append({
            "project_id": pid, "label": label, "sheet_path": os.path.relpath(out_path, OUT_ROOT),
            "pass_pct": rollup.get("pass_pct", 0.0), "shot_count": rollup.get("shot_count", len(shots)),
            "pass": rollup.get("pass", 0), "warn": rollup.get("warn", 0), "fail": rollup.get("fail", 0),
            "worst_metric_classes": rollup.get("worst_metric_classes", []),
        })

    sheet_rows.sort(key=lambda r: r["pass_pct"])
    _write_index(sheet_rows)
    print(f"\nwrote {len(sheet_rows)} contact sheets + {INDEX_PATH}")


if __name__ == "__main__":
    main()
