"""Render raw | legacy | v1 still comparisons for a graded thread, so the v1
color grade can be eyeballed without flipping any production flag.

For a few shots of the thread: extract the hero still off the proxy, bake the
legacy grade cube and read the persisted v1 cube, apply each with ffmpeg lut3d,
and hstack raw|legacy|v1 (labeled), then vstack the shots into one PNG.
"""
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("GRADE_PIPELINE", "v1")

import psycopg  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.services.l3 import store as edit_store  # noqa: E402
from app.services.l3.frames import extract_still  # noqa: E402
from app.services.l3.grade import job as grade_job  # noqa: E402
from app.services.l3.grade.cache import ensure_cube_file  # noqa: E402
from app.services.l3.grade.resolver import resolve_clip_grade  # noqa: E402
from app.services.l3.grade.measure import fetch_color_stats  # noqa: E402
from app.services.processing import _download_from_r2  # noqa: E402

s = get_settings()
thread_id = sys.argv[1]
label = sys.argv[2]
out_png = sys.argv[3]
cube_dir = os.path.join(tempfile.gettempdir(), "edso_grade_cubes")
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
W = 640

doc, _ = edit_store.latest_document(thread_id)
shots = grade_job.ordered_shots(doc)
# take up to 3 shots spread across the timeline
picks = shots[:3] if len(shots) <= 3 else [shots[0], shots[len(shots)//2], shots[-1]]

v1_rows = grade_job.fetch_latest_grades(thread_id, [sh.key for sh in picks])
cstats = fetch_color_stats(list({sh.file_id for sh in picks}))

with psycopg.connect(s.database_url, autocommit=True) as c:
    proxy = {}
    for sh in picks:
        r = c.execute("select r2_proxy_key from files where id=%s", (sh.file_id,)).fetchone()
        proxy[sh.file_id] = r[0] if r else None


def apply_cube(frame, cube_path, out):
    if not cube_path:
        subprocess.run(["cp", frame, out], check=True)
        return
    subprocess.run(["ffmpeg", "-y", "-i", frame, "-vf", f"lut3d=file='{cube_path}'", out],
                   check=True, capture_output=True)


def labeled(img, text, out):
    try:
        subprocess.run(["ffmpeg", "-y", "-i", img, "-vf",
                        f"drawtext=fontfile={FONT}:text='{text}':x=10:y=10:fontsize=28:"
                        f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=6", out],
                       check=True, capture_output=True)
    except Exception:
        subprocess.run(["cp", img, out], check=True)


with tempfile.TemporaryDirectory() as tmp:
    row_imgs = []
    dl_cache = {}
    for i, sh in enumerate(picks):
        pkey = proxy.get(sh.file_id)
        if not pkey:
            print(f"skip {sh.key}: no proxy"); continue
        if pkey not in dl_cache:
            p = os.path.join(tmp, f"proxy_{i}.mp4"); _download_from_r2(pkey, p); dl_cache[pkey] = p
        ppath = dl_cache[pkey]
        ts = sh.hero_ts_ms if sh.hero_ts_ms is not None else (sh.in_ms + sh.out_ms) // 2
        raw = os.path.join(tmp, f"raw_{i}.jpg")
        extract_still(ppath, int(ts), raw, width=W)

        v1_cube = ensure_cube_file(v1_rows.get(sh.key), cube_dir)
        legacy_grade = resolve_clip_grade(sh.item, color_stats=cstats.get(sh.file_id),
                                          sequence_look=doc.get("look"))
        legacy_cube = ensure_cube_file(legacy_grade, cube_dir)

        leg = os.path.join(tmp, f"leg_{i}.jpg"); apply_cube(raw, legacy_cube, leg)
        v1 = os.path.join(tmp, f"v1_{i}.jpg"); apply_cube(raw, v1_cube, v1)
        rawL = os.path.join(tmp, f"rawL_{i}.jpg"); labeled(raw, f"{sh.key} RAW", rawL)
        legL = os.path.join(tmp, f"legL_{i}.jpg"); labeled(leg, "LEGACY", legL)
        v1L = os.path.join(tmp, f"v1L_{i}.jpg"); labeled(v1, "V1", v1L)

        row = os.path.join(tmp, f"row_{i}.jpg")
        subprocess.run(["ffmpeg", "-y", "-i", rawL, "-i", legL, "-i", v1L,
                        "-filter_complex", "hstack=inputs=3", row], check=True, capture_output=True)
        row_imgs.append(row)

    if not row_imgs:
        print("no rows rendered"); sys.exit(1)
    inputs = []
    for r in row_imgs:
        inputs += ["-i", r]
    subprocess.run(["ffmpeg", "-y", *inputs,
                    "-filter_complex", f"vstack=inputs={len(row_imgs)}", out_png],
                   check=True, capture_output=True)
    print(f"{label}: wrote {out_png} ({len(row_imgs)} shots)")
