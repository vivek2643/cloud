"""READ-ONLY: distribution of raw exposure & per-shot raw->graded luma move for
rec709, to separate 'already well-exposed -> needs little' from 'dark ->
clamped/undershoot'."""
import json
import statistics as st

SB = json.load(open("scripts/_out/qa/scoreboard.json"))

rec = []
for sh in SB["shots"]:
    if sh["is_log_flat"] or sh["content_type"] == "synthetic":
        continue
    eb = (sh.get("metrics") or {}).get("exposure_band") or {}
    rm = eb.get("raw_median")
    gm = eb.get("median")
    if rm is None or gm is None:
        continue
    rec.append((sh["project_label"], sh["shot_key"], rm, gm, gm - rm))

rec.sort(key=lambda r: r[2])
print(f"{len(rec)} rec709 shots, sorted by raw median luma")
print(f"{'proj':14s} {'shot':6s} {'raw_med':>8s} {'grd_med':>8s} {'dLuma':>7s}")
for p, k, rm, gm, d in rec:
    print(f"{p[:14]:14s} {k:6s} {rm:8.3f} {gm:8.3f} {d:+7.3f}")

print()
# buckets by raw exposure
already_ok = [r for r in rec if 0.30 <= r[2] <= 0.55]
dark = [r for r in rec if r[2] < 0.30]
bright = [r for r in rec if r[2] > 0.55]
print(f"raw already in [0.30,0.55] (needs little): {len(already_ok)}  "
      f"-> median |dLuma|={st.median([abs(r[4]) for r in already_ok]):.3f}" if already_ok else "none")
print(f"raw dark (<0.30):                          {len(dark)}  "
      f"-> median |dLuma|={st.median([abs(r[4]) for r in dark]):.3f}" if dark else "none")
print(f"raw bright (>0.55):                        {len(bright)}  "
      f"-> median |dLuma|={st.median([abs(r[4]) for r in bright]):.3f}" if bright else "none")
print()
near_identity = [r for r in rec if abs(r[4]) < 0.02]
print(f"rec709 shots with |dLuma| < 0.02 (visually ~identical exposure): "
      f"{len(near_identity)}/{len(rec)} = {100*len(near_identity)/len(rec):.0f}%")
