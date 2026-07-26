"""READ-ONLY: quantify how much the composite guardrails actually change a
rec709 grade. Reconstruct correct-alone (pre composite-clamp; balance/match/
leveling are ~no-op on these mostly-singleton timelines) and compare to the
clamped result -- so we can say whether the clamp is THE throttle or a
backstop that barely moves real rec709 grades."""
import json
import statistics as st
from collections import defaultdict

import numpy as np
import psycopg
from app.config import get_settings
from app.services.l3 import store as edit_store
from app.services.l3.grade.cdl import Grade
from app.services.l3.grade.correct import LEVELS_SLOPE_MAX, solve_correct_grade
from app.services.l3.grade.measure import fetch_color_stats
from app.services.l3.grade.resolver import _clamp_composite_v1
from app.services.l3.grade.tone import WORKING_SPACE_V1

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
        doc, _v = edit_store.latest_document(tid)
        file_of = {}
        for seg in (doc.get("timeline") or []):
            if seg.get("seg_id"):
                file_of[str(seg["seg_id"])] = str(seg.get("file_id"))
        for op in (doc.get("operations") or []):
            if op.get("op_id"):
                file_of[str(op["op_id"])] = str(op.get("source_file_id"))
        for shot_key, gj in got:
            gj = gj if isinstance(gj, dict) else json.loads(gj)
            rows.append((tid, shot_key, gj, file_of.get(shot_key)))

all_files = list({r[3] for r in rows if r[3]})
cs = fetch_color_stats(all_files)

agg = defaultdict(lambda: defaultdict(list))
n_slope_capped_correct = defaultdict(int)
n_clamp_changed = defaultdict(int)
n_total = defaultdict(int)
for tid, shot_key, gj, fid in rows:
    b = meta.get((tid, shot_key))
    stats = cs.get(fid) if fid else None
    if not b or not stats:
        continue
    ws = gj.get("working_space") or WORKING_SPACE_V1
    correct = solve_correct_grade(stats, pipeline="v1", skin_vibrance=True, working_space=ws)
    clamped = _clamp_composite_v1(correct, ws)
    n_total[b] += 1

    # did correct hit its OWN slope cap?
    if any(abs(s - LEVELS_SLOPE_MAX * max(correct.slope) / max(correct.slope)) < 1 for s in [0]):
        pass
    luma_slope = max(correct.slope)  # slope = wb*luma_slope; luma_slope ~ max channel-ish
    # better: luma component = geometric-ish; just check if any channel near a cap
    # correct's luma_slope is capped at LEVELS_SLOPE_MAX; wb multiplies it (<=1.5).
    # Detect undershoot proxy instead below.

    # how much did the composite clamp move slope/offset?
    dslope = sum(abs(clamped.slope[i] - correct.slope[i]) for i in range(3)) / 3
    doff = sum(abs(clamped.offset[i] - correct.offset[i]) for i in range(3)) / 3
    agg[b]["clamp_dslope"].append(dslope)
    agg[b]["clamp_doffset"].append(doff)
    agg[b]["correct_offset_avg"].append(sum(correct.offset) / 3)
    if dslope > 1e-4 or doff > 1e-4:
        n_clamp_changed[b] += 1


def m(v):
    v = [x for x in v if x is not None]
    return f"n={len(v)} mean={st.mean(v):.5f} median={st.median(v):.5f} max={max(v):.5f}" if v else "(none)"


print(f"LEVELS_SLOPE_MAX (per-correct-layer cap) = {LEVELS_SLOPE_MAX}")
print()
print("How often / how much does the COMPOSITE clamp change the correct-alone grade?")
print("(correct-alone ~= pre-clamp composite on these singleton timelines)")
print()
for b in ("log", "rec709", "synthetic"):
    print(f"===== {b}  (n={n_total[b]}) =====")
    print(f"   composite clamp changed grade at all: {n_clamp_changed[b]}/{n_total[b]}")
    print(f"   clamp_dslope (avg |slope change|)   : {m(agg[b]['clamp_dslope'])}")
    print(f"   clamp_doffset (avg |offset change|) : {m(agg[b]['clamp_doffset'])}")
    print(f"   correct_offset_avg (pre-clamp)      : {m(agg[b]['correct_offset_avg'])}")
    print()
