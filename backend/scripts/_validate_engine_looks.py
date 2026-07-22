"""Validate the engine-look Look mode (grade_pipeline_standardize.plan.md:
the pipeline has no more dev flags -- every capability below is always on,
matching run_grade_job exactly).

Replicates run_grade_job's delta pipeline (balance/match/leveling) IN MEMORY
for a thread -- with zero DB writes -- then resolves each shot's grade for a
CLEAN baseline (look=None) and for several engine looks (tone_contrast stays
hardwired to 0.0, matching the shipped pipeline).

Because the engine look contributes an IDENTITY CDL delta (resolver._solve_look
returns Grade() for mode=="engine") and the balance/match/leveling deltas are
independent of the look, the only thing that changes across looks is the baked
creative_lut_grid -- so this is byte-identical to what run_grade_job would
persist with that look selected, without touching the DB.

Renders raw | v1(no look) | <look>... per shot into one PNG per shot and a
combined sheet, and prints numeric diffs proving the looks differ.

Usage: PYTHONPATH=. .venv/bin/python scripts/_validate_engine_looks.py [THREAD_ID]
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import psycopg

from app.config import get_settings
from app.services.l3 import store as edit_store
from app.services.l3.frames import extract_still
from app.services.l3.grade import job as GJ
from app.services.l3.grade.cache import ensure_cube_file
from app.services.l3.grade.cdl import Grade, compose as cdl_compose
from app.services.l3.grade.leveling import ShotLevelInput, solve_leveling
from app.services.l3.grade.balance import solve_balance
from app.services.l3.grade.match import ShotStats, group_neighbors, solve_sequence_match
from app.services.l3.grade.measure import fetch_color_stats
from app.services.l3.grade.measure_span import measure_span
from app.services.l3.grade.reference import GroupReference, compute_group_reference
from app.services.l3.grade.resolver import resolve_clip_grade
from app.services.l3.grade.scene_group import ShotSceneMeta as SceneMeta
from app.services.l3.grade.scene_group import group_shots_semantically
from app.services.l3.grade.tone import WORKING_SPACE_V1
from app.services.processing import _download_from_r2

THREAD = sys.argv[1] if len(sys.argv) > 1 else "3bfe3db3-0dce-4fc6-bc97-42fb4ec08bad"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_out", "engine_look_validation")
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
W = 480

# Visually distinct looks, one per family (creator/film/ad) + an extra strongly
# different creator look, to prove different picks render differently.
LOOKS_TO_TEST = ["punchy_vibrant", "moody_cinematic", "kodak_2383", "tech_sleek"]


def compute_deltas(thread_id: str):
    """Mirror run_grade_job's balance/match/leveling delta computation, but in
    memory. Returns (shots, shot_stats, balance_deltas, match_deltas,
    leveling_deltas). Deltas are look-independent, so we compute them once and
    reuse for every look below."""
    document, _ = edit_store.latest_document(thread_id)
    shots = GJ.ordered_shots(document)
    file_ids = list({s.file_id for s in shots})
    color_stats = fetch_color_stats(file_ids)

    from app.services.l3.grade.scene_meta import lookup_shot_cut_meta
    cut_meta = lookup_shot_cut_meta([(s.key, s.file_id, s.in_ms, s.out_ms) for s in shots])
    subject_boxes = {k: m.subject_box for k, m in cut_meta.items() if m.subject_box}

    shot_stats: Dict[str, ShotStats] = {}
    for s in shots:
        subject_box = s.item.get("subject_box") or subject_boxes.get(s.key)
        hero_ts_ms = s.hero_ts_ms
        if hero_ts_ms is None and subject_box is not None:
            cm = cut_meta.get(s.key)
            if cm is not None and cm.hero_ts_ms is not None:
                hero_ts_ms = cm.hero_ts_ms
        stats = measure_span(s.file_id, s.in_ms, s.out_ms, hero_ts_ms=hero_ts_ms, subject_box=subject_box)
        if stats is None:
            stats = color_stats.get(s.file_id)
        shot_stats[s.key] = ShotStats(key=s.key, file_id=s.file_id, stats=stats)

    scene_meta = []
    for s in shots:
        cm = cut_meta.get(s.key)
        span_rgb = (shot_stats[s.key].stats or {}).get("rgb_mean")
        scene_meta.append(SceneMeta(
            key=s.key, file_id=s.file_id,
            speaker_person=(cm.speaker_person if cm else None),
            on_camera=(cm.on_camera if cm else None),
            label=(cm.label if cm else ""),
            summary=(cm.summary if cm else ""),
            voice_ids=(cm.voice_ids if cm else []),
            take_group_id=(cm.take_group_id if cm else None),
            sync_group_id=(cm.sync_group_id if cm else None),
            rgb_mean=list(span_rgb) if span_rgb else None,
        ))
    semantic_groups = group_shots_semantically(scene_meta)

    ordered = [shot_stats[s.key] for s in shots]
    if not GJ._has_real_groups(semantic_groups):
        semantic_groups = None
    groups = semantic_groups if semantic_groups is not None else group_neighbors(ordered)

    balance_references: Dict[int, GroupReference] = {}
    display_stats = [o.stats or {} for o in ordered]
    ws_stats: List[Dict[str, Any]] = [GJ._ws_stats(o.stats) for o in ordered]
    for gi, idxs in enumerate(groups):
        b_ref = compute_group_reference([ws_stats[i] for i in idxs])
        if b_ref is not None:
            balance_references[gi] = b_ref
    balance_deltas: Dict[str, Grade] = solve_balance(ws_stats, groups, balance_references, [s.key for s in shots])
    balanced_display_stats = [
        display_stats[i] if o.key not in balance_deltas
        else GJ._corrected_display_stats(ws_stats[i], balance_deltas[o.key])
        for i, o in enumerate(ordered)
    ]
    match_references: Dict[int, GroupReference] = {}
    for gi, idxs in enumerate(groups):
        m_ref = compute_group_reference([balanced_display_stats[i] for i in idxs])
        if m_ref is not None:
            match_references[gi] = m_ref
    balanced_shots = [
        ShotStats(key=o.key, file_id=o.file_id, stats=balanced_display_stats[i])
        for i, o in enumerate(ordered)
    ]
    match_deltas: Dict[str, Grade] = solve_sequence_match(
        balanced_shots, groups=groups, working_space=WORKING_SPACE_V1, references=match_references)

    group_idx_by_shot = [
        next((gi for gi, idxs in enumerate(groups) if i in idxs), None) for i in range(len(shots))
    ]
    level_inputs: List[ShotLevelInput] = []
    for i, s in enumerate(shots):
        stats = shot_stats[s.key].stats or {}
        subj = stats.get("subject_luma")
        subject_luma = GJ._to_working_scalar(subj, None) if subj is not None else None
        group_idx = group_idx_by_shot[i]
        ref = balance_references.get(group_idx) if group_idx is not None else None
        if ref is not None and ws_stats:
            bm_grade = cdl_compose(balance_deltas.get(s.key, Grade()), match_deltas.get(s.key, Grade()), 1.0)
            mid_gray = GJ._apply_working_scalar(ws_stats[i].get("mid_gray"), bm_grade, 1)
            if mid_gray is None:
                mid_gray = 0.5
            black = GJ._apply_working_scalar(ws_stats[i]["black_point"], bm_grade, 1)
            white = GJ._apply_working_scalar(ws_stats[i]["white_point"], bm_grade, 1)
            level_inputs.append(ShotLevelInput(
                key=s.key, mid_gray=mid_gray, black_point=black, white_point=white,
                subject_luma=subject_luma, target_mid_gray=ref.mid_gray,
                target_black_point=ref.black_point, target_white_point=ref.white_point))
        else:
            mid_gray = GJ._to_working_scalar(stats.get("mid_gray"), 0.5)
            black = GJ._to_working_scalar(stats.get("black_point"), 0.0)
            white = GJ._to_working_scalar(stats.get("white_point"), 1.0)
            level_inputs.append(ShotLevelInput(
                key=s.key, mid_gray=mid_gray, black_point=black, white_point=white,
                subject_luma=subject_luma))
    from statistics import median
    from app.services.l3.grade.leveling import _usable_subject_luma
    subject_by_group: Dict[int, List[float]] = {}
    for i, gi in enumerate(group_idx_by_shot):
        if gi is None:
            continue
        usable = _usable_subject_luma(level_inputs[i].subject_luma, level_inputs[i].mid_gray)
        if usable is not None:
            subject_by_group.setdefault(gi, []).append(usable)
    for i, gi in enumerate(group_idx_by_shot):
        members = subject_by_group.get(gi) if gi is not None else None
        if members and len(members) >= 2:
            level_inputs[i].target_subject_luma = median(members)
    leveling_deltas: Dict[str, Grade] = solve_leveling(level_inputs)

    return shots, shot_stats, balance_deltas, match_deltas, leveling_deltas


def resolve_for_look(shot, stats, balance_deltas, match_deltas, leveling_deltas, look: Optional[dict]):
    """Resolve one shot's grade for a given look -- tone_contrast stays
    hardwired to 0.0, matching the shipped pipeline."""
    return resolve_clip_grade(
        shot.item, color_stats=stats, sequence_look=look,
        balance_delta=balance_deltas.get(shot.key),
        match_delta=match_deltas.get(shot.key),
        leveling_delta=leveling_deltas.get(shot.key),
        tone_contrast=0.0,
    )


def apply_cube(frame, cube_path, out):
    if not cube_path:
        subprocess.run(["cp", frame, out], check=True)
        return
    subprocess.run(["ffmpeg", "-y", "-i", frame, "-vf", f"lut3d=file='{cube_path}'", out],
                   check=True, capture_output=True)


def labeled(img, text, out):
    try:
        subprocess.run(["ffmpeg", "-y", "-i", img, "-vf",
                        f"drawtext=fontfile={FONT}:text='{text}':x=8:y=8:fontsize=22:"
                        f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5", out],
                       check=True, capture_output=True)
    except Exception:
        subprocess.run(["cp", img, out], check=True)


def frame_stats(path):
    import cv2
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    s = get_settings()
    print(f"THREAD {THREAD}")
    print(f"looks under test (tone_contrast hardwired to 0.0): {LOOKS_TO_TEST}\n")

    cube_dir = tempfile.mkdtemp(prefix="edso_validate_cubes_")
    shots, shot_stats, bd, md, ld = compute_deltas(THREAD)

    # pick up to 3 shots spread across the timeline, from distinct files where possible
    if len(shots) <= 3:
        picks = shots
    else:
        picks = [shots[0], shots[len(shots) // 2], shots[-1]]

    with psycopg.connect(s.database_url, autocommit=True) as c:
        proxy = {}
        for sh in picks:
            r = c.execute("select r2_proxy_key from files where id=%s", (sh.file_id,)).fetchone()
            proxy[sh.file_id] = r[0] if r else None

    columns = ["none"] + LOOKS_TO_TEST

    def look_dict(name):
        if name == "none":
            return None
        return {"mode": "engine", "look_id": name}

    # sanity: print resolved descriptors for the first shot
    print("== resolved look_engine descriptors (shot 0) ==")
    for name in columns:
        g = resolve_for_look(picks[0], shot_stats[picks[0].key].stats, bd, md, ld, look_dict(name))
        le = g.get("look_engine")
        print(f"  {name:16} hash={g['grade_hash'][:12]} look_engine={'identity/None' if not le else 'set'} "
              f"cdl_sat={round(g['cdl']['sat'],3)}")
    print()

    with tempfile.TemporaryDirectory() as tmp:
        row_imgs = []
        dl_cache = {}
        per_look_rgb: Dict[str, list] = {name: [] for name in columns}
        for i, sh in enumerate(picks):
            pkey = proxy.get(sh.file_id)
            if not pkey:
                print(f"skip {sh.key}: no proxy")
                continue
            if pkey not in dl_cache:
                p = os.path.join(tmp, f"proxy_{i}.mp4")
                _download_from_r2(pkey, p)
                dl_cache[pkey] = p
            ppath = dl_cache[pkey]
            ts = sh.hero_ts_ms if sh.hero_ts_ms is not None else (sh.in_ms + sh.out_ms) // 2
            raw = os.path.join(tmp, f"raw_{i}.jpg")
            extract_still(ppath, int(ts), raw, width=W)

            tiles = []
            rawL = os.path.join(tmp, f"rawL_{i}.jpg")
            labeled(raw, f"{sh.key} RAW", rawL)
            tiles.append(rawL)

            for name in columns:
                g = resolve_for_look(sh, shot_stats[sh.key].stats, bd, md, ld, look_dict(name))
                cube = ensure_cube_file(g, cube_dir)
                graded = os.path.join(tmp, f"{name}_{i}.jpg")
                apply_cube(raw, cube, graded)
                rgb = frame_stats(graded)
                if rgb is not None:
                    per_look_rgb[name].append(rgb)
                lbl = "v1 (no look)" if name == "none" else name
                gradedL = os.path.join(tmp, f"{name}L_{i}.jpg")
                labeled(graded, lbl, gradedL)
                tiles.append(gradedL)

            # also save the individual graded frames to OUT_DIR
            for name in columns:
                g = resolve_for_look(sh, shot_stats[sh.key].stats, bd, md, ld, look_dict(name))
                cube = ensure_cube_file(g, cube_dir)
                dst = os.path.join(OUT_DIR, f"{sh.key}_{name}.jpg")
                apply_cube(raw, cube, dst)

            row = os.path.join(tmp, f"row_{i}.jpg")
            inputs = []
            for t in tiles:
                inputs += ["-i", t]
            subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex",
                            f"hstack=inputs={len(tiles)}", row], check=True, capture_output=True)
            row_imgs.append(row)
            print(f"rendered shot {sh.key} ({i+1}/{len(picks)})")

        if row_imgs:
            sheet = os.path.join(OUT_DIR, "sheet.png")
            inputs = []
            for r in row_imgs:
                inputs += ["-i", r]
            subprocess.run(["ffmpeg", "-y", *inputs, "-filter_complex",
                            f"vstack=inputs={len(row_imgs)}", sheet], check=True, capture_output=True)
            print(f"\nwrote combined sheet: {sheet}")

    # numeric proof the looks differ from each other and from baseline
    print("\n== mean per-pixel abs difference vs 'v1 (no look)' (0..1 scale, averaged over shots) ==")
    base = per_look_rgb.get("none")
    for name in LOOKS_TO_TEST:
        arr = per_look_rgb.get(name)
        if not arr or not base:
            continue
        diffs = [float(np.mean(np.abs(a - b))) for a, b in zip(arr, base)]
        print(f"  {name:16} mean_abs_diff={np.mean(diffs):.4f}")
    print("\n== pairwise mean abs difference between looks (must be > ~0.01 to be visibly distinct) ==")
    names = LOOKS_TO_TEST
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a = per_look_rgb.get(names[x]); b = per_look_rgb.get(names[y])
            if not a or not b:
                continue
            d = np.mean([float(np.mean(np.abs(u - v))) for u, v in zip(a, b)])
            print(f"  {names[x]:16} vs {names[y]:16} = {d:.4f}")
    # brightness / clipping sanity per look
    print("\n== per-look brightness + clipping sanity (averaged over shots) ==")
    for name in columns:
        arr = per_look_rgb.get(name)
        if not arr:
            continue
        means = [float(a.mean()) for a in arr]
        blk = [float((a < 0.02).mean()) for a in arr]   # frac near-black
        wht = [float((a > 0.98).mean()) for a in arr]    # frac near-white
        lbl = "v1 (no look)" if name == "none" else name
        print(f"  {lbl:16} mean_luma~{np.mean(means):.3f}  crushed<0.02={np.mean(blk):.3f}  blown>0.98={np.mean(wht):.3f}")


if __name__ == "__main__":
    main()
