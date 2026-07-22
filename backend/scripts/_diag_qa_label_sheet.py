#!/usr/bin/env python3
"""color_phase1.plan.md Part 1d: a human-labeling artifact.

Selects ~20-30 representative shots spanning the LOG subset, a spread of
Rec.709 shots (dark/mid/bright), a couple of intentional-look shots (one
mono/desaturated, one warm/saturated), and the demo-trail synthetic project
-- reusing the already-sampled hero-frame stills from samples.json and the
already-computed metric readouts from scoreboard.json (read-only: this
script re-extracts nothing, re-scores nothing).

Writes:
  backend/scripts/_out/qa/label_sheet.html -- raw | graded | metrics | a
    good/bad radio + "intentional look?" checkbox, per shot. Radios feed a
    live JSON preview textarea (select-all + paste over labels.json).
  backend/scripts/_out/qa/labels.json -- pre-seeded skeleton (verdict: null,
    intentional: false) the user can also hand-edit directly, one row per
    selected shot, each tagged with its deterministic calibration/held_out
    split (seeded by a hash of shot_key, ~70/30, reproducible).

The ACTUAL LABELING IS THE USER'S JOB -- this script only builds the page
and the skeleton; it must never fabricate good/bad verdicts.

Usage: PYTHONPATH=. .venv/bin/python scripts/_diag_qa_label_sheet.py
"""
from __future__ import annotations

import hashlib
import html
import json
import os
import sys
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from qa import metrics as m  # noqa: E402

OUT_ROOT = os.path.join(HERE, "_out", "qa")
SAMPLES_PATH = os.path.join(OUT_ROOT, "samples.json")
SCOREBOARD_PATH = os.path.join(OUT_ROOT, "scoreboard.json")
LABEL_SHEET_PATH = os.path.join(OUT_ROOT, "label_sheet.html")
LABELS_PATH = os.path.join(OUT_ROOT, "labels.json")

MAX_SHOTS = 28
CALIBRATION_FRACTION = 0.70
# The metrics whose readouts are worth a human's eyes on this page -- the
# ones Part 1 calibrates + the structural Tier-A signals for context.
DISPLAY_METRICS = [
    "exposure_band", "crushed_black_fraction", "clipped_highlight_fraction",
    "neutral_axis_deviation", "saturation_band", "skin_perp_residual",
    "look_fidelity_cosine", "banding_score",
]


def _shot_uid(shot: Dict[str, Any]) -> str:
    """`shot_key` (e.g. "a000") is scoped to a single thread's document, NOT
    globally unique -- this corpus alone has 83 sampled shots collapsing to
    just 17 distinct shot_key strings across different projects/threads (and
    even within one thread, when a project row is duplicated in corpus.json).
    Every dict keyed "per shot" in this script must use this composite id,
    never bare shot_key, or unrelated shots silently collide."""
    return f"{shot['project_id']}::{shot['thread_id']}::{shot['shot_key']}"


def _split_for(uid: str) -> str:
    """Deterministic, reproducible ~70/30 split seeded by the shot's
    composite id -- never random, so re-running this script doesn't
    reshuffle which shots were already used to pick thresholds vs held out
    to validate them."""
    digest = hashlib.sha256(uid.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return "calibration" if bucket < CALIBRATION_FRACTION * 100 else "held_out"


def _hero_frame(shot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for f in shot.get("frames") or []:
        if f.get("is_hero"):
            return f
    frames = shot.get("frames") or []
    return frames[0] if frames else None


def _select_shots(samples: List[Dict[str, Any]], scoreboard_by_uid: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Stratified, deterministic (sorted by composite id, not random) picks
    across the categories 1d asks for. Each candidate must have a scored
    hero frame (present in scoreboard.json) so the metric readouts on the
    page are real, not placeholders."""
    candidates = []
    for s in samples:
        hero = _hero_frame(s)
        score = scoreboard_by_uid.get(_shot_uid(s))
        if hero is None or score is None:
            continue
        candidates.append((s, hero, score))
    candidates.sort(key=lambda t: _shot_uid(t[0]))

    picked: Dict[str, tuple] = {}

    def take(pool, n):
        for item in pool:
            if len(picked) >= MAX_SHOTS:
                return
            uid = _shot_uid(item[0])
            if uid not in picked and n > 0:
                picked[uid] = item
                n -= 1

    log_pool = [c for c in candidates if c[0].get("is_log_flat")]
    take(log_pool, 6)

    synthetic_pool = [c for c in candidates if c[2].get("content_type") == "synthetic"]
    take(synthetic_pool, 2)

    look_intents = [
        (c, m.look_saturation_intent(c[0].get("look_engine")))
        for c in candidates if c[0].get("look_engine")
    ]
    look_intents = [(c, li) for c, li in look_intents if li is not None]
    look_intents.sort(key=lambda t: t[1])
    if look_intents:
        take([look_intents[0][0]], 1)   # most mono/desaturated
        take([look_intents[-1][0]], 1)  # most saturated/warm

    photographic_rec709 = [
        c for c in candidates
        if not c[0].get("is_log_flat") and c[2].get("content_type") != "synthetic"
    ]
    dark = [c for c in photographic_rec709 if (c[2]["metrics"].get("exposure_band") or {}).get("median", 0.5) < 0.30]
    bright = [c for c in photographic_rec709 if (c[2]["metrics"].get("exposure_band") or {}).get("median", 0.5) > 0.60]
    mid = [c for c in photographic_rec709 if c not in dark and c not in bright]
    remaining = MAX_SHOTS - len(picked)
    take(dark, max(2, remaining // 3))
    take(bright, max(2, remaining // 3))
    take(mid, MAX_SHOTS - len(picked))

    return [picked[k] for k in sorted(picked.keys())]


def _metric_cell(name: str, md: Optional[Dict[str, Any]]) -> str:
    if md is None:
        return f'<div class="metric metric-missing">{html.escape(name)}: --</div>'
    verdict = md.get("verdict", "?")
    value = md.get("value")
    value_str = f"{value:.3f}" if isinstance(value, (int, float)) else str(value)
    return (
        f'<div class="metric metric-{html.escape(verdict)}">'
        f'{html.escape(name)}: {value_str} <b>[{html.escape(verdict)}]</b></div>'
    )


def _render_row(shot: Dict[str, Any], hero: Dict[str, Any], score: Dict[str, Any]) -> str:
    uid = _shot_uid(shot)
    row_id = html.escape(uid, quote=True)
    metrics_html = "".join(_metric_cell(n, score["metrics"].get(n)) for n in DISPLAY_METRICS)
    look_engine = shot.get("look_engine") or {}
    look_name = html.escape(str(look_engine.get("name") or look_engine.get("look_id") or "-"))
    tags = []
    if shot.get("is_log_flat"):
        tags.append("LOG")
    if score.get("content_type") == "synthetic":
        tags.append("SYNTHETIC")
    if look_engine:
        tags.append(f"look={look_name}")
    tags_html = " ".join(f'<span class="tag">{t}</span>' for t in tags)
    split = _split_for(uid)
    return f"""
    <tr id="row-{row_id}" data-shot-key="{row_id}" data-split="{split}">
      <td class="thumbs">
        <div><img src="{html.escape(hero['raw_path'])}" loading="lazy"><div class="cap">raw</div></div>
        <div><img src="{html.escape(hero['graded_path'])}" loading="lazy"><div class="cap">graded</div></div>
      </td>
      <td class="meta">
        <div class="shotkey">{html.escape(shot['shot_key'])}</div>
        <div class="project">{html.escape(shot['project_label'])}</div>
        <div class="tags">{tags_html}</div>
        <div class="split">split: {split}</div>
      </td>
      <td class="metrics">{metrics_html}</td>
      <td class="verdict">
        <label><input type="radio" name="verdict-{row_id}" value="good" onchange="updateJson()"> good</label>
        <label><input type="radio" name="verdict-{row_id}" value="bad" onchange="updateJson()"> bad</label>
        <label class="intentional"><input type="checkbox" id="intentional-{row_id}" onchange="updateJson()"> intentional look?</label>
      </td>
    </tr>"""


_PAGE_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>QA label sheet</title>
<style>
  body {{ font-family: -apple-system, sans-serif; margin: 16px; background: #111; color: #eee; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ border-bottom: 1px solid #333; padding: 8px; vertical-align: top; text-align: left; }}
  .thumbs {{ display: flex; gap: 8px; }}
  .thumbs img {{ width: 160px; height: auto; display: block; border: 1px solid #444; }}
  .cap {{ font-size: 11px; color: #999; text-align: center; }}
  .shotkey {{ font-weight: bold; }}
  .project {{ color: #aaa; font-size: 12px; }}
  .tags {{ margin-top: 4px; }}
  .tag {{ background: #333; border-radius: 3px; padding: 1px 6px; font-size: 11px; margin-right: 4px; }}
  .split {{ font-size: 11px; color: #888; margin-top: 4px; }}
  .metric {{ font-size: 12px; font-family: monospace; }}
  .metric-fail {{ color: #ff6b6b; }}
  .metric-warn {{ color: #ffd166; }}
  .metric-pass {{ color: #6bcB77; }}
  .metric-na {{ color: #777; }}
  .verdict label {{ display: block; font-size: 13px; margin-bottom: 2px; }}
  .intentional {{ margin-top: 6px; color: #aaa; }}
  #jsonOut {{ width: 100%; height: 220px; font-family: monospace; font-size: 12px; margin-top: 12px; }}
  h1 {{ font-size: 18px; }}
  .hint {{ color: #999; font-size: 13px; }}
</style>
</head>
<body>
<h1>color_phase1.plan.md Part 1d -- QA label sheet ({count} shots)</h1>
<p class="hint">Mark each shot good/bad (does the GRADED still look correct/professional -- not "do I like the look").
Check "intentional look?" when a bad-looking-by-the-numbers shot is a deliberate creative choice (e.g. mono/desaturated).
The textarea below live-updates as you click; select-all + copy it over <code>_out/qa/labels.json</code> when done
(or hand-edit the pre-seeded <code>labels.json</code> directly instead -- either path works).</p>
<table>
<thead><tr><th>stills</th><th>shot</th><th>metrics</th><th>label</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
<h2>labels.json preview</h2>
<textarea id="jsonOut" readonly></textarea>
<script>
function updateJson() {{
  const rows = document.querySelectorAll('tr[data-shot-key]');
  const out = {{}};
  rows.forEach(row => {{
    const key = row.dataset.shotKey;
    const split = row.dataset.split;
    const checked = row.querySelector(`input[name="verdict-${{CSS.escape(key)}}"]:checked`);
    const intentional = row.querySelector(`#intentional-${{CSS.escape(key)}}`);
    out[key] = {{
      verdict: checked ? checked.value : null,
      intentional: intentional ? intentional.checked : false,
      split: split,
    }};
  }});
  document.getElementById('jsonOut').value = JSON.stringify(out, null, 2);
}}
updateJson();
</script>
</body>
</html>
"""


def main() -> None:
    if not os.path.exists(SAMPLES_PATH):
        raise SystemExit(f"no samples manifest at {SAMPLES_PATH} -- run _diag_qa_sample.py first")
    if not os.path.exists(SCOREBOARD_PATH):
        raise SystemExit(f"no scoreboard at {SCOREBOARD_PATH} -- run _diag_qa_score.py first")
    with open(SAMPLES_PATH) as f:
        samples = json.load(f)
    with open(SCOREBOARD_PATH) as f:
        scoreboard = json.load(f)

    scoreboard_by_uid = {_shot_uid(s): s for s in scoreboard["shots"]}
    selected = _select_shots(samples, scoreboard_by_uid)
    if not selected:
        raise SystemExit("no candidate shots found (samples/scoreboard empty or mismatched?)")

    rows_html = "".join(_render_row(shot, hero, score) for shot, hero, score in selected)
    page = _PAGE_TEMPLATE.format(count=len(selected), rows=rows_html)
    with open(LABEL_SHEET_PATH, "w") as f:
        f.write(page)

    labels_skeleton = {}
    for shot, _hero, _score in selected:
        uid = _shot_uid(shot)
        labels_skeleton[uid] = {
            "verdict": None, "intentional": False, "split": _split_for(uid),
            "shot_key": shot["shot_key"], "project_label": shot["project_label"],
            "is_log_flat": bool(shot.get("is_log_flat")),
        }
    with open(LABELS_PATH, "w") as f:
        json.dump(labels_skeleton, f, indent=2)

    n_cal = sum(1 for v in labels_skeleton.values() if v["split"] == "calibration")
    n_held = len(labels_skeleton) - n_cal
    print(f"selected {len(selected)} shots ({n_cal} calibration / {n_held} held_out)")
    print(f"wrote {LABEL_SHEET_PATH}")
    print(f"wrote {LABELS_PATH} (pre-seeded, verdict: null -- fill in by hand or via the page)")


if __name__ == "__main__":
    main()
