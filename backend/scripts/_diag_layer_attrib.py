"""READ-ONLY layer attribution: for each persisted shot, re-solve the Correct
layer alone from its file color_stats and compare to the final persisted
composite CDL, to attribute 'how much of the (small) rec709 grade is Correct vs
Balance/Match/Leveling/Look'. Also reports whether any sequence look is active
and the graded-vs-target mid-gray undershoot. No re-grade of pixels."""
import json
import statistics as st
from collections import defaultdict

import psycopg
from app.config import get_settings
from app.services.l3.grade.correct import TARGET_MID_GRAY, solve_correct_grade
from app.services.l3.grade.measure import fetch_color_stats
from app.services.l3.grade.tone import WORKING_SPACE_LOG_V1, WORKING_SPACE_V1

SB = json.load(open("scripts/_out/qa/scoreboard.json"))
meta, sb_by = {}, {}
for sh in SB["shots"]:
    if sh["is_log_flat"]:
        b = "log"
    elif sh["content_type"] == "synthetic":
        b = "synthetic"
    else:
        b = "rec709"
    meta[(sh["thread_id"], sh["shot_key"])] = b
    sb_by[(sh["thread_id"], sh["shot_key"])] = sh

thread_ids = sorted({t for (t, _k) in meta})

# We need file_id per (thread, shot_key). resolved_grades doesn't store it, but
# the edit document's timeline does. Simpler: pull the latest document per
# thread and map seg_id/op_id -> file_id, then fetch whole-file color_stats.
from app.services.l3 import store as edit_store

rows = []
look_active = defaultdict(int)
look_total = defaultdict(int)
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
        look = doc.get("look") or {}
        for shot_key, gj in got:
            gj = gj if isinstance(gj, dict) else json.loads(gj)
            rows.append((tid, shot_key, gj, file_of.get(shot_key), look))

# batch fetch color_stats
all_files = list({r[3] for r in rows if r[3]})
cs = fetch_color_stats(all_files)


def dist(g):
    slope_dev = sum(abs(s - 1) for s in g.slope) / 3
    off = sum(abs(o) for o in g.offset) / 3
    return slope_dev, off, abs(g.sat - 1), (max(g.slope) - min(g.slope))


agg = defaultdict(lambda: defaultdict(list))
for tid, shot_key, gj, fid, look in rows:
    b = meta.get((tid, shot_key))
    if not b:
        continue
    look_total[b] += 1
    mode = look.get("mode")
    if mode in ("preset", "reference", "engine", "lut"):
        look_active[b] += 1
    stats = cs.get(fid) if fid else None
    if not stats:
        continue
    ws = gj.get("working_space") or WORKING_SPACE_V1
    correct = solve_correct_grade(stats, pipeline="v1", skin_vibrance=True, working_space=ws)
    from app.services.l3.grade.cdl import Grade
    final = Grade.from_dict(gj.get("cdl"))

    c_sd, c_off, c_sat, c_wb = dist(correct)
    f_sd, f_off, f_sat, f_wb = dist(final)
    agg[b]["correct_avg|slope-1|"].append(c_sd)
    agg[b]["final_avg|slope-1|"].append(f_sd)
    agg[b]["correct_|sat-1|"].append(c_sat)
    agg[b]["final_|sat-1|"].append(f_sat)
    agg[b]["correct_wb_spread"].append(c_wb)
    agg[b]["final_wb_spread"].append(f_wb)
    # mean slope contributed by correct vs final
    agg[b]["correct_mean_slope"].append(sum(correct.slope) / 3)
    agg[b]["final_mean_slope"].append(sum(final.slope) / 3)

    # undershoot: graded median vs target 0.42 (display) from scoreboard
    sh = sb_by[(tid, shot_key)]
    eb = (sh.get("metrics") or {}).get("exposure_band") or {}
    if eb.get("median") is not None:
        agg[b]["graded_median_luma"].append(eb["median"])
        agg[b]["undershoot_vs_0.42"].append(TARGET_MID_GRAY - eb["median"])
        if eb.get("raw_median") is not None:
            agg[b]["raw_median_luma"].append(eb["raw_median"])


def m(v):
    v = [x for x in v if x is not None]
    return f"n={len(v)} mean={st.mean(v):.4f} median={st.median(v):.4f}" if v else "(none)"


print("Look mode active per bucket:", {b: f"{look_active[b]}/{look_total[b]}" for b in ("log", "rec709", "synthetic")})
print()
fields = ["correct_mean_slope", "final_mean_slope", "correct_avg|slope-1|", "final_avg|slope-1|",
          "correct_|sat-1|", "final_|sat-1|", "correct_wb_spread", "final_wb_spread",
          "raw_median_luma", "graded_median_luma", "undershoot_vs_0.42"]
for b in ("log", "rec709", "synthetic"):
    print(f"===== {b} =====")
    for f in fields:
        print(f"   {f:24s}: {m(agg[b][f])}")
    print()
