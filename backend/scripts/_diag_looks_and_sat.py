"""READ-ONLY: (A) what look is set per thread (mode/preset/engine params), since
engine/lut looks bake into the creative LUT grid, not the CDL; (B) clean demo of
how a sat multiplier applied in LINEAR working space produces a much smaller
DISPLAY chroma change than the same multiplier in display space."""
import json
from collections import Counter

import numpy as np
import psycopg
from app.config import get_settings
from app.services.l3 import store as edit_store
from app.services.l3.grade.cdl import Grade, apply_cdl
from app.services.l3.grade.tone import WORKING_SPACE_V1, from_working, to_working

SB = json.load(open("scripts/_out/qa/scoreboard.json"))
thread_ids = sorted({sh["thread_id"] for sh in SB["shots"]})
label_of = {sh["thread_id"]: sh["project_label"] for sh in SB["shots"]}

print("=" * 70)
print("(A) LOOK CONFIG PER THREAD (looks bake into LUT grid, NOT the CDL)")
print("=" * 70)
for tid in thread_ids:
    doc, _v = edit_store.latest_document(tid)
    look = (doc or {}).get("look") or {}
    mode = look.get("mode")
    extra = ""
    if mode == "engine":
        extra = f" look_id={look.get('look_id')} intensity={look.get('intensity')} params={ {k:v for k,v in look.items() if k not in ('mode',)} }"
    elif mode == "preset":
        extra = f" preset_id={look.get('preset_id')}"
    elif mode == "reference":
        extra = f" match_strength={look.get('match_strength')}"
    print(f"  {label_of.get(tid,'?')[:18]:18s} {tid[:8]}  mode={mode}{extra}")

print()
print("=" * 70)
print("(B) SAT MULTIPLIER: display effect when applied in LINEAR vs DISPLAY space")
print("=" * 70)
# a moderately saturated patch (display)
patches = {
    "skin":  np.array([[[0.62, 0.44, 0.36]]], np.float32),
    "red":   np.array([[[0.60, 0.20, 0.18]]], np.float32),
    "teal":  np.array([[[0.20, 0.45, 0.48]]], np.float32),
}


def lab_chroma(disp):
    import cv2
    u8 = np.clip(disp * 255, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB).astype(np.float32)
    a = lab[..., 1] - 128
    b = lab[..., 2] - 128
    return float(np.sqrt(a * a + b * b).mean())


for sat in (1.15, 1.25, 1.4):
    g = Grade(sat=sat)
    print(f"\n  sat={sat}")
    for name, p in patches.items():
        base_c = lab_chroma(p)
        # pipeline: decode->apply sat in linear->reencode
        lin = to_working(p, WORKING_SPACE_V1)
        pipe = from_working(apply_cdl(lin, g), WORKING_SPACE_V1)
        pipe_c = lab_chroma(pipe)
        # display-space application
        disp = apply_cdl(p, g)
        disp_c = lab_chroma(disp)
        print(f"    {name:5s}: base_chroma={base_c:6.2f}  "
              f"linear->display ratio={pipe_c/base_c:5.3f}  "
              f"display-space ratio={disp_c/base_c:5.3f}")
