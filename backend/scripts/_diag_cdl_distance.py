"""READ-ONLY: pull persisted resolved_grades CDLs, join to scoreboard content
type, and measure composite CDL 'distance from identity' by bucket."""
import json
import statistics as st
from collections import defaultdict

import psycopg
from app.config import get_settings

SB = json.load(open("scripts/_out/qa/scoreboard.json"))
# (thread_id, shot_key) -> (bucket, is_log_flat, content_type)
meta = {}
for sh in SB["shots"]:
    if sh["is_log_flat"]:
        b = "log"
    elif sh["content_type"] == "synthetic":
        b = "synthetic"
    else:
        b = "rec709"
    meta[(sh["thread_id"], sh["shot_key"])] = (b, sh["content_type"])

thread_ids = sorted({t for (t, _k) in meta})

rows = []
with psycopg.connect(get_settings().database_url) as c:
    for tid in thread_ids:
        got = c.execute(
            "select distinct on (shot_key) shot_key, grade_json "
            "from resolved_grades where thread_id::text=%s "
            "order by shot_key, updated_at desc",
            (tid,),
        ).fetchall()
        for shot_key, gj in got:
            gj = gj if isinstance(gj, dict) else json.loads(gj)
            rows.append((tid, shot_key, gj))

print(f"pulled {len(rows)} resolved_grades rows (latest per shot) across {len(thread_ids)} threads")


def summ(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return "  (none)"
    vals_s = sorted(vals)
    return (f"n={len(vals)} mean={st.mean(vals):.4f} median={st.median(vals):.4f} "
            f"p90={vals_s[int(0.9*(len(vals)-1))]:.4f} max={max(vals):.4f}")


buckets = defaultdict(lambda: defaultdict(list))
identity_count = defaultdict(int)
bucket_count = defaultdict(int)
matched = 0

for tid, shot_key, gj in rows:
    m = meta.get((tid, shot_key))
    if not m:
        continue
    matched += 1
    b, _ct = m
    bucket_count[b] += 1
    cdl = gj.get("cdl") or {}
    slope = cdl.get("slope") or [1, 1, 1]
    offset = cdl.get("offset") or [0, 0, 0]
    power = cdl.get("power") or [1, 1, 1]
    sat = cdl.get("sat", 1.0)
    ws = gj.get("working_space")

    # distance-from-identity measures
    mean_slope = sum(slope) / 3.0
    slope_dev = sum(abs(s - 1.0) for s in slope) / 3.0     # avg |slope-1|
    max_slope_dev = max(abs(s - 1.0) for s in slope)
    offset_mag = sum(abs(o) for o in offset) / 3.0          # avg |offset|
    max_offset_mag = max(abs(o) for o in offset)
    power_dev = sum(abs(p - 1.0) for p in power) / 3.0
    sat_dev = abs(sat - 1.0)
    # WB imbalance: spread of slope channels (r vs b) = white balance work
    wb_spread = max(slope) - min(slope)

    buckets[b]["mean_slope"].append(mean_slope)
    buckets[b]["avg_|slope-1|"].append(slope_dev)
    buckets[b]["max_|slope-1|"].append(max_slope_dev)
    buckets[b]["avg_|offset|"].append(offset_mag)
    buckets[b]["max_|offset|"].append(max_offset_mag)
    buckets[b]["avg_|power-1|"].append(power_dev)
    buckets[b]["|sat-1|"].append(sat_dev)
    buckets[b]["wb_slope_spread"].append(wb_spread)

    # near-identity CDL?
    is_ident = (slope_dev < 0.02 and offset_mag < 0.005 and power_dev < 0.01
                and sat_dev < 0.02 and wb_spread < 0.03)
    if is_ident:
        identity_count[b] += 1

print(f"matched {matched} rows to scoreboard meta")
print("bucket counts (with a persisted CDL):", dict(bucket_count))
print("near-identity CDL counts:", dict(identity_count))
print()

order = ["log", "rec709", "synthetic"]
fields = ["mean_slope", "avg_|slope-1|", "max_|slope-1|", "avg_|offset|",
          "max_|offset|", "avg_|power-1|", "|sat-1|", "wb_slope_spread"]
for f in fields:
    print(f"--- {f} ---")
    for b in order:
        print(f"  {b:10s}: {summ(buckets[b][f])}")
    print()
