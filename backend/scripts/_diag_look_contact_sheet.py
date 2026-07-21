#!/usr/bin/env python3
"""Visual iteration loop for the LOOK layer (not a test -- a tuning tool).

Renders ONE real frame from a project's proxy through:
  - the raw source,
  - the shipped v1 Correct layer (skin + vibrance on),
  - a PROTOTYPE cinematic contrast/tone S-curve on top of Correct (candidate
    for tone.py::from_working),
  - the current CDL presets (as they render today), and
  - two PROTOTYPE "real LUT" looks (per-hue split-tone + tone curve) that CDL
    slope/offset/power fundamentally cannot express -- the target quality bar.

All variants are composed into one labeled grid PNG so we can compare and tune
by eye. This is the taste-validation loop presets.py never had.

Usage:
  .venv/bin/python scripts/_diag_look_contact_sheet.py [FILE_ID] [TS_SECONDS]
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import psycopg  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.l1 import color_stats as cs  # noqa: E402
from app.services.l3.grade import presets as presets_mod  # noqa: E402
from app.services.l3.grade import tone  # noqa: E402
from app.services.l3.grade.cdl import apply_cdl, compose  # noqa: E402
from app.services.l3.grade.correct import solve_correct_grade  # noqa: E402
from app.services.l3.grade.look_engine import LOOKS, build_look_grid  # noqa: E402
from app.services.l3.grade.lut_bake import _sample_lut_trilinear  # noqa: E402
from app.services.l3.grade.measure_span import _fetch_proxy_path  # noqa: E402
from app.services.l3.grade.softlocal import HALATION_SIGMA, HALATION_SIGMA_REF_H, HALATION_THRESHOLD, HALATION_TINT  # noqa: E402

WS = tone.WORKING_SPACE_V1
TILE_H = 360


# --------------------------------------------------------------------------
# Prototype cinematic contrast S-curve (candidate for tone.py::from_working).
# Symmetric sigmoid through (0,0),(0.5,0.5),(1,1) in DISPLAY space -- adds the
# midtone contrast the current shoulder-only curve never touches.
# --------------------------------------------------------------------------
def contrast_scurve(rgb_display: np.ndarray, k: float = 3.2) -> np.ndarray:
    def sig(x):
        return 1.0 / (1.0 + np.exp(-k * (x - 0.5)))
    s0, s1 = sig(0.0), sig(1.0)
    return np.clip((sig(rgb_display) - s0) / (s1 - s0), 0.0, 1.0)


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


# --------------------------------------------------------------------------
# Prototype "real LUT" looks: per-pixel ops (split-tone by luma, hue-selective
# saturation) -- exactly what a baked 3D .cube would encode, and exactly what
# CDL slope/offset/power CANNOT do. These are the quality target.
# --------------------------------------------------------------------------
def _hsv_sat_by_hue(rgb: np.ndarray, boost_center_deg: float, boost: float,
                    suppress_center_deg: float, suppress: float) -> np.ndarray:
    import cv2
    hsv = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    h = hsv[..., 0] * 2.0  # OpenCV H is 0..179 -> degrees
    s = hsv[..., 1] / 255.0

    def bell(center):
        d = np.abs(((h - center + 180) % 360) - 180)
        return np.exp(-(d ** 2) / (2 * 35.0 ** 2))
    s = s * (1.0 + boost * bell(boost_center_deg)) * (1.0 - suppress * bell(suppress_center_deg))
    hsv[..., 1] = np.clip(s, 0, 1) * 255.0
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0


def look_teal_orange_real(rgb: np.ndarray) -> np.ndarray:
    y = luma(rgb)[..., None]
    shadow_tone = np.array([-0.02, 0.01, 0.06], dtype=np.float32)   # teal in shadows
    highlight_tone = np.array([0.06, 0.015, -0.03], dtype=np.float32)  # warm in highlights
    out = rgb + shadow_tone * (1.0 - y) + highlight_tone * y
    out = np.clip(out, 0, 1)
    out = _hsv_sat_by_hue(out, boost_center_deg=30.0, boost=0.35,      # pop skin/orange
                          suppress_center_deg=140.0, suppress=0.35)    # calm greens
    return contrast_scurve(out, k=3.4)


def look_warm_film_real(rgb: np.ndarray) -> np.ndarray:
    y = luma(rgb)[..., None]
    out = rgb + np.array([0.03, 0.012, -0.02], dtype=np.float32) * (0.4 + 0.6 * y)  # warm
    out = out * 0.92 + 0.035          # faded/lifted toe (film blacks aren't pure black)
    out = np.clip(out, 0, 1)
    out = _hsv_sat_by_hue(out, boost_center_deg=30.0, boost=0.12,
                          suppress_center_deg=220.0, suppress=0.2)
    return np.clip(contrast_scurve(out, k=2.4) * 0.98 + 0.01, 0, 1)


def look_natural_plus(rgb: np.ndarray) -> np.ndarray:
    """Just the base tone curve + a touch of clean saturation -- a safe default."""
    out = _hsv_sat_by_hue(rgb, boost_center_deg=30.0, boost=0.1, suppress_center_deg=999, suppress=0.0)
    return contrast_scurve(out, k=2.9)


def look_golden_real(rgb: np.ndarray) -> np.ndarray:
    """Warm afternoon light -- controlled (not the CDL blowout): warm highlights,
    neutral-ish shadows, skin popped, sky not over-yellowed."""
    y = luma(rgb)[..., None]
    out = rgb + np.array([0.05, 0.02, -0.03], dtype=np.float32) * y            # warm the highlights only
    out = out + np.array([0.01, 0.0, 0.0], dtype=np.float32) * (1 - y)         # tiny warm in shadows
    out = np.clip(out, 0, 1)
    out = _hsv_sat_by_hue(out, boost_center_deg=35.0, boost=0.28, suppress_center_deg=210.0, suppress=0.15)
    return contrast_scurve(out, k=2.7)


def look_moody_real(rgb: np.ndarray) -> np.ndarray:
    """Cool, desaturated, deeper contrast, lifted toe -- somber/editorial."""
    y = luma(rgb)[..., None]
    out = rgb + np.array([-0.02, 0.0, 0.03], dtype=np.float32) * (1 - y)       # cool shadows
    out = out * 0.94 + 0.02
    out = np.clip(out, 0, 1)
    out = _hsv_sat_by_hue(out, boost_center_deg=30.0, boost=0.0, suppress_center_deg=120.0, suppress=0.45)
    # global slight desat toward luma
    out = np.clip(luma(out)[..., None] + 0.78 * (out - luma(out)[..., None]), 0, 1)
    return contrast_scurve(out, k=3.4)


def look_clean_punch_real(rgb: np.ndarray) -> np.ndarray:
    """Neutral commercial look: crisp contrast, clean whites, tasteful sat -- no cast."""
    out = _hsv_sat_by_hue(rgb, boost_center_deg=30.0, boost=0.22, suppress_center_deg=999, suppress=0.0)
    return contrast_scurve(out, k=3.6)


def look_cool_cine_real(rgb: np.ndarray) -> np.ndarray:
    """Modern cool blockbuster: teal midtones, retained warm skin, punchy."""
    y = luma(rgb)[..., None]
    out = rgb + np.array([-0.03, 0.0, 0.05], dtype=np.float32) * (0.5 + 0.5 * (1 - y))
    out = np.clip(out, 0, 1)
    out = _hsv_sat_by_hue(out, boost_center_deg=28.0, boost=0.4, suppress_center_deg=150.0, suppress=0.3)
    return contrast_scurve(out, k=3.5)


NEW_GALLERY = [
    ("Natural+", look_natural_plus),
    ("Teal & Orange", look_teal_orange_real),
    ("Warm Film", look_warm_film_real),
    ("Golden Hour", look_golden_real),
    ("Cool Cinematic", look_cool_cine_real),
    ("Moody Editorial", look_moody_real),
    ("Clean Punch", look_clean_punch_real),
]


# --------------------------------------------------------------------------
def render_grade(frame_display: np.ndarray, grade, tone_curve: bool) -> np.ndarray:
    """to_working -> apply_cdl -> from_working (the exact baked-cube pipeline),
    optionally with the prototype contrast curve on the display output."""
    working = tone.to_working(frame_display, WS)
    graded = apply_cdl(working, grade)
    out = tone.from_working(graded, WS)
    return contrast_scurve(out) if tone_curve else out


def label(img: np.ndarray, text: str) -> np.ndarray:
    import cv2
    bar = np.full((34, img.shape[1], 3), 0.08, dtype=np.float32)
    cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (1, 1, 1), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


# --------------------------------------------------------------------------
# color_look_library.plan.md S4: render the REAL catalog (build_look_grid,
# the exact same trilinear-sample path bake_cube_text/preview/export use),
# not the prototype functions above -- so this sheet shows exactly what
# ships. Film-family looks additionally get a NUMPY approximation of the
# halation/grain spatial pass (not the shipped ffmpeg/WebGL implementation
# -- see softlocal.py for that) so texture is tuned in context, per S4
# point 4's "reuse the halation/grain apply math or approximate in the
# sheet" note.
# --------------------------------------------------------------------------

def _approx_halation(rgb_display: np.ndarray, strength: float, frame_height: int) -> np.ndarray:
    import cv2

    if strength <= 0.0:
        return rgb_display
    luma = rgb_display[..., 0] * 0.2126 + rgb_display[..., 1] * 0.7152 + rgb_display[..., 2] * 0.0722
    mask = np.where(luma[..., None] > HALATION_THRESHOLD, rgb_display, 0.0).astype(np.float32)
    tint = np.array(HALATION_TINT, dtype=np.float32)
    tinted = mask * tint
    sigma = max(1.0, HALATION_SIGMA * (frame_height / float(HALATION_SIGMA_REF_H)))
    ksize = int(sigma * 3) | 1  # odd kernel size, ~3 sigma radius
    blurred = cv2.GaussianBlur(tinted, (ksize, ksize), sigma)
    screened = 1.0 - (1.0 - rgb_display) * (1.0 - blurred)
    return np.clip(rgb_display * (1.0 - strength) + screened * strength, 0.0, 1.0)


def _approx_grain(rgb_display: np.ndarray, strength: float, seed: int = 0) -> np.ndarray:
    if strength <= 0.0:
        return rgb_display
    rng = np.random.default_rng(seed)
    noise = rng.uniform(-0.5, 0.5, size=rgb_display.shape[:2] + (1,)).astype(np.float32)
    return np.clip(rgb_display + noise * strength, 0.0, 1.0)


def render_catalog_look(base_display: np.ndarray, spec, frame_height: int) -> np.ndarray:
    """The exact shipped color path (build_look_grid -> trilinear sample,
    the same one bake_cube_text/preview/export use) plus the numpy
    halation/grain approximation above for in-context texture tuning."""
    grid, size = build_look_grid(spec, size=33)
    out = _sample_lut_trilinear(grid, base_display)
    if spec.halation > 0.0:
        out = _approx_halation(out, spec.halation, frame_height)
    if spec.grain > 0.0:
        out = _approx_grain(out, spec.grain)
    return out


def render_catalog_sheet(file_id: str, ts_frac: float, con) -> None:
    row = con.execute(
        "select r2_proxy_key, duration_seconds, width, height, name from files where id=%s", (file_id,)
    ).fetchone()
    if not row:
        print(f"no file {file_id}")
        return
    dur, w, h, name = row[1], row[2], row[3], row[4]
    ts = max(0.5, (dur or 4) * ts_frac)
    tw = int(round(TILE_H * (w / h))) if w and h else 640

    with tempfile.TemporaryDirectory(prefix="edso_sheet_") as tmp:
        path = _fetch_proxy_path(file_id, tmp)
        if not path:
            print(f"proxy download failed for {file_id}")
            return
        frame_u8 = cs._decode_rgb_frame_at(path, ts, tw, TILE_H)
        stats = cs._aggregate([cs._decode_rgb_frame_at(path, ts, cs.COLOR_STATS_W, cs.COLOR_STATS_H)]).to_dict()

    if frame_u8 is None:
        print(f"decode failed for {file_id}")
        return
    frame = frame_u8.astype(np.float32) / 255.0
    correct_grade = solve_correct_grade(stats, pipeline="v1", skin_vibrance=True)
    base = render_grade(frame, correct_grade, False)

    tiles = [("Original (raw)", frame), ("v1 Correct", base)]
    for family in ("creator", "film", "ad"):
        for look in LOOKS:
            if look.family != family or look.look_id == "engine_identity":
                continue
            out = render_catalog_look(base, look.spec, TILE_H)
            tiles.append((f"[{family}] {look.label}", out))

    cols = 4
    labeled = [label(np.clip(img, 0, 1), txt) for txt, img in tiles]
    while len(labeled) % cols:
        labeled.append(np.zeros_like(labeled[0]))
    rows_img = [np.hstack(labeled[i:i + cols]) for i in range(0, len(labeled), cols)]
    sheet = np.vstack(rows_img)

    import cv2
    out_path = os.path.join(BACKEND, "logs", f"look_catalog_{file_id[:8]}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor((sheet * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"file: {name}  @ {ts:.1f}s  ({w}x{h})")
    print(f"wrote {out_path}  ({sheet.shape[1]}x{sheet.shape[0]})")


def main() -> None:
    file_id_arg = sys.argv[1] if len(sys.argv) > 1 else "93e94ec3-313f-4d44-bf66-f11bf2502853"
    con = psycopg.connect(get_settings().database_url, autocommit=True)
    if file_id_arg == "catalog":
        # color_look_library.plan.md S4: the 3 real representative frames --
        # a talking-head (Siri Reel, skin check), a highlight-heavy clip
        # (Food Reel, halation check), a flat/low-light clip (podcast trial
        # 3, grain + black-lift check).
        for fid, ts_frac in (
            ("abce4178-c03e-489e-92a8-383b348e71e4", 0.45),   # Siri Reel (talking-head)
            ("5b861488-d10d-44b3-bb92-d2084221d706", 0.5),    # Food Reel (highlight-heavy)
            ("1aedb093-9259-4deb-aa45-c8d5fba6def0", 0.0006), # podcast trial 3 (flat/low-light)
        ):
            render_catalog_sheet(fid, ts_frac, con)
        return
    file_id = file_id_arg
    row = con.execute(
        "select r2_proxy_key, duration_seconds, width, height, name from files where id=%s", (file_id,)
    ).fetchone()
    if not row:
        print(f"no file {file_id}"); sys.exit(1)
    dur, w, h, name = row[1], row[2], row[3], row[4]
    ts = float(sys.argv[2]) if len(sys.argv) > 2 else max(0.5, (dur or 4) * 0.45)
    tw = int(round(TILE_H * (w / h))) if w and h else 640

    with tempfile.TemporaryDirectory(prefix="edso_sheet_") as tmp:
        path = _fetch_proxy_path(file_id, tmp)
        if not path:
            print("proxy download failed"); sys.exit(1)
        frame_u8 = cs._decode_rgb_frame_at(path, ts, tw, TILE_H)
        stats = cs._aggregate([cs._decode_rgb_frame_at(path, ts, cs.COLOR_STATS_W, cs.COLOR_STATS_H)]).to_dict()

    frame = frame_u8.astype(np.float32) / 255.0
    correct = solve_correct_grade(stats, pipeline="v1", skin_vibrance=True)

    preset = {p.preset_id: p for p in presets_mod.PRESETS}
    tiles = []
    base = render_grade(frame, correct, False)
    mode = sys.argv[3] if len(sys.argv) > 3 else "compare"
    if mode == "gallery":
        tiles.append(("Original (raw)", frame))
        tiles.append(("v1 Correct", base))
        for label_txt, fn in NEW_GALLERY:
            tiles.append((f"{label_txt} (NEW real look)", fn(base)))
    else:
        tiles.append(("Original (raw)", frame))
        tiles.append(("v1 Correct (skin+vibrance)", base))
        tiles.append(("Correct + PROTOTYPE tone S-curve", contrast_scurve(base)))
        for pid in ("cinematic_teal_orange", "warm_film", "moody_desaturated", "golden_hour", "blue_hour"):
            g = compose(correct, preset[pid].grade, 1.0)
            tiles.append((f"{preset[pid].label} (current CDL)", render_grade(frame, g, False)))
        tiles.append(("Teal & Orange (PROTOTYPE real LUT)", look_teal_orange_real(base)))
        tiles.append(("Warm Film (PROTOTYPE real LUT)", look_warm_film_real(base)))

    cols = 4
    labeled = [label(np.clip(img, 0, 1), txt) for txt, img in tiles]
    while len(labeled) % cols:
        labeled.append(np.zeros_like(labeled[0]))
    rows_img = [np.hstack(labeled[i:i + cols]) for i in range(0, len(labeled), cols)]
    sheet = np.vstack(rows_img)

    import cv2
    out_path = os.path.join(BACKEND, "logs", f"look_sheet_{file_id[:8]}_{mode}.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor((sheet * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"file: {name}  @ {ts:.1f}s  ({w}x{h})")
    print(f"wrote {out_path}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
