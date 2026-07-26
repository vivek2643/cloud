"""READ-ONLY diagnostic: quantify raw->graded change magnitude by content type
from the QA scoreboard. No DB, no re-render."""
import json
import math
import statistics as st
from collections import defaultdict

SB = json.load(open("scripts/_out/qa/scoreboard.json"))
shots = SB["shots"]


def bucket(sh):
    if sh["is_log_flat"]:
        return "log"
    if sh["content_type"] == "synthetic":
        return "synthetic"
    return "rec709"


def summ(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": round(st.mean(vals), 4),
        "median": round(st.median(vals), 4),
        "p90": round(sorted(vals)[int(0.9 * (len(vals) - 1))], 4),
        "max": round(max(vals), 4),
    }


groups = defaultdict(lambda: defaultdict(list))

for sh in shots:
    b = bucket(sh)
    g = sh.get("summary_graded") or {}
    r = sh.get("summary_raw") or {}
    m = sh.get("metrics") or {}

    # luma shift (abs)
    if g.get("median_luma") is not None and r.get("median_luma") is not None:
        groups[b]["abs_dluma_median"].append(abs(g["median_luma"] - r["median_luma"]))
        groups[b]["dluma_median_signed"].append(g["median_luma"] - r["median_luma"])
    if g.get("mean_luma") is not None and r.get("mean_luma") is not None:
        groups[b]["abs_dluma_mean"].append(abs(g["mean_luma"] - r["mean_luma"]))

    # chroma / saturation shift from saturation_band metric
    sat = m.get("saturation_band") or {}
    if sat.get("value") is not None and sat.get("raw_chroma_mean") is not None:
        groups[b]["abs_dchroma"].append(abs(sat["value"] - sat["raw_chroma_mean"]))
        if sat.get("raw_chroma_mean", 0) > 1e-6:
            groups[b]["chroma_ratio"].append(sat["value"] / sat["raw_chroma_mean"])

    # WB / neutral-axis shift: compare graded vs raw mean_a/mean_b (ab hypot)
    if all(g.get(k) is not None for k in ("mean_a", "mean_b")) and all(
        r.get(k) is not None for k in ("mean_a", "mean_b")
    ):
        dab = math.hypot(g["mean_a"] - r["mean_a"], g["mean_b"] - r["mean_b"])
        groups[b]["ab_shift"].append(dab)

    # black/white point shift
    if g.get("black_point") is not None and r.get("black_point") is not None:
        groups[b]["abs_dblack"].append(abs(g["black_point"] - r["black_point"]))
    if g.get("white_point") is not None and r.get("white_point") is not None:
        groups[b]["abs_dwhite"].append(abs(g["white_point"] - r["white_point"]))

    # subject luma shift
    if g.get("subject_luma") is not None and r.get("subject_luma") is not None:
        groups[b]["abs_dsubjluma"].append(abs(g["subject_luma"] - r["subject_luma"]))


print("=" * 70)
print("RAW->GRADED CHANGE MAGNITUDE BY CONTENT TYPE (from scoreboard.json)")
print("=" * 70)
counts = defaultdict(int)
for sh in shots:
    counts[bucket(sh)] += 1
print("shot counts:", dict(counts))
print()

metrics = [
    "abs_dluma_median", "abs_dluma_mean", "dluma_median_signed",
    "abs_dsubjluma", "abs_dchroma", "chroma_ratio", "ab_shift",
    "abs_dblack", "abs_dwhite",
]
for metric in metrics:
    print(f"--- {metric} ---")
    for b in ("log", "rec709", "synthetic"):
        s = summ(groups[b][metric])
        print(f"  {b:10s}: {s}")
    print()
