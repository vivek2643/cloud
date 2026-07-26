"""READ-ONLY: (A) detect whether composite guardrails are actively clamping
persisted rec709 CDLs, and (B) demonstrate how much the linear working-space
application + re-encode mutes a CDL's DISPLAY effect vs applying it in display
space. No re-grade."""
import json
from collections import defaultdict

import numpy as np
import psycopg
from app.config import get_settings
from app.services.l3.grade.cdl import Grade, apply_cdl
from app.services.l3.grade.resolver import (
    COMPOSITE_MID_FLOOR, COMPOSITE_SHADOW_PROBE, COMPOSITE_SLOPE_MAX,
    _clamp_composite_v1, _floor_for_probe,
)
from app.services.l3.grade.tone import (
    WORKING_SPACE_LOG_V1, WORKING_SPACE_V1, from_working, to_working,
)

SB = json.load(open("scripts/_out/qa/scoreboard.json"))
meta = {}
for sh in SB["shots"]:
    if sh["is_log_flat"]:
        b = "log"
    elif sh["content_type"] == "synthetic":
        b = "synthetic"
    else:
        b = "rec709"
    meta[(sh["thread_id"], sh["shot_key"])] = b

thread_ids = sorted({t for (t, _k) in meta})
rows = []
with psycopg.connect(get_settings().database_url) as c:
    for tid in thread_ids:
        got = c.execute(
            "select distinct on (shot_key) shot_key, grade_json from resolved_grades "
            "where thread_id::text=%s order by shot_key, updated_at desc",
            (tid,),
        ).fetchall()
        for shot_key, gj in got:
            gj = gj if isinstance(gj, dict) else json.loads(gj)
            rows.append((tid, shot_key, gj))

# ---- (A) guardrail-active detection ----
print("=" * 70)
print("(A) ARE THE COMPOSITE GUARDRAILS ACTIVELY CLAMPING? (persisted CDLs)")
print(f"    COMPOSITE_SLOPE_MAX={COMPOSITE_SLOPE_MAX}  MID_FLOOR={COMPOSITE_MID_FLOOR}  "
      f"SHADOW_PROBE={COMPOSITE_SHADOW_PROBE}")
print("=" * 70)
EPS = 1e-4
counts = defaultdict(lambda: {"n": 0, "slope_at_ceiling": 0, "offset_at_floor": 0})
for tid, shot_key, gj in rows:
    b = meta.get((tid, shot_key))
    if not b:
        continue
    cdl = gj.get("cdl") or {}
    ws = gj.get("working_space") or WORKING_SPACE_V1
    slope = cdl.get("slope") or [1, 1, 1]
    offset = cdl.get("offset") or [0, 0, 0]
    power = cdl.get("power") or [1, 1, 1]
    counts[b]["n"] += 1
    if any(abs(s - COMPOSITE_SLOPE_MAX) < EPS for s in slope):
        counts[b]["slope_at_ceiling"] += 1
    # offset floor: recompute the two probe floors for this CDL's channels
    mid_lin = float(to_working(np.array([0.5], np.float32), ws)[0])
    sh_lin = float(to_working(np.array([COMPOSITE_SHADOW_PROBE], np.float32), ws)[0])
    at_floor = False
    for cch in range(3):
        f1 = _floor_for_probe(mid_lin, min(slope[cch], COMPOSITE_SLOPE_MAX), power[cch], COMPOSITE_MID_FLOOR)
        f2 = _floor_for_probe(sh_lin, min(slope[cch], COMPOSITE_SLOPE_MAX), power[cch], COMPOSITE_MID_FLOOR)
        if abs(offset[cch] - max(f1, f2)) < EPS and max(f1, f2) > offset[cch] - EPS:
            # offset sits exactly on the floor -> clamp bound it
            at_floor = True
    if at_floor:
        counts[b]["offset_at_floor"] += 1

for b in ("log", "rec709", "synthetic"):
    c = counts[b]
    print(f"  {b:10s}: n={c['n']:3d}  slope_at_ceiling(2.0)={c['slope_at_ceiling']:3d}  "
          f"offset_at_floor={c['offset_at_floor']:3d}")
print()
print("  NOTE: persisted CDLs are already post-clamp; 'at ceiling/floor' means the")
print("  clamp is the binding constraint for that shot (pre-clamp exceeded it).")
print()

# ---- (B) display effect of applying the CDL in linear space vs display space ----
print("=" * 70)
print("(B) HOW MUCH DOES LINEAR-SPACE APPLICATION MUTE THE DISPLAY EFFECT?")
print("=" * 70)
# probe patches: a neutral mid-gray + a moderately saturated skin-ish/red patch
mid_disp = np.array([[[0.46, 0.46, 0.46]]], np.float32)
red_disp = np.array([[[0.55, 0.35, 0.30]]], np.float32)  # warm/skin-ish

def luma(x):
    return float(x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722)

def chroma(x):
    # crude a*b* proxy via rgb spread
    r, g, bl = float(x[...,0]), float(x[...,1]), float(x[...,2])
    return ((r-g)**2 + (g-bl)**2 + (r-bl)**2) ** 0.5

agg = defaultdict(lambda: defaultdict(list))
for tid, shot_key, gj in rows:
    b = meta.get((tid, shot_key))
    if not b:
        continue
    cdl = gj.get("cdl") or {}
    g = Grade.from_dict(cdl)
    ws = gj.get("working_space") or WORKING_SPACE_V1

    for name, patch in (("mid", mid_disp), ("warm", red_disp)):
        # pipeline-faithful: decode -> apply_cdl -> re-encode
        lin = to_working(patch, ws)
        graded_lin = apply_cdl(lin, g)
        graded_disp = from_working(graded_lin, ws)
        # counterfactual: apply the SAME cdl directly in display space
        graded_display_space = apply_cdl(patch, g)

        agg[b][f"{name}_dLuma_pipeline"].append(abs(luma(graded_disp) - luma(patch)))
        agg[b][f"{name}_dLuma_displayspace"].append(abs(luma(graded_display_space) - luma(patch)))
        if name == "warm":
            agg[b]["warm_dChroma_pipeline"].append(chroma(graded_disp) - chroma(patch))
            agg[b]["warm_dChroma_displayspace"].append(chroma(graded_display_space) - chroma(patch))

import statistics as st
def m(v):
    v = [x for x in v if x is not None]
    return f"mean={st.mean(v):.4f} median={st.median(v):.4f}" if v else "(none)"

for b in ("log", "rec709", "synthetic"):
    print(f"[{b}]")
    for f in ["mid_dLuma_pipeline", "mid_dLuma_displayspace",
              "warm_dLuma_pipeline", "warm_dLuma_displayspace",
              "warm_dChroma_pipeline", "warm_dChroma_displayspace"]:
        print(f"    {f:28s}: {m(agg[b][f])}")
    print()
