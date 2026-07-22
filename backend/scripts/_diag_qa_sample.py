#!/usr/bin/env python3
"""color_qa_harness.plan.md A.2: RAW-vs-GRADED still sampling.

For every shot in every corpus thread (backend/scripts/_out/qa/corpus.json,
written by `_diag_qa_corpus.py`): resolve up to 4 frame timestamps (the hero
frame + ~25/50/75% of the span), extract a RAW still from the proxy, and put
it through the shot's ACTUAL baked `.cube` (ffmpeg `lut3d`, not a re-render)
to get the GRADED still -- the exact bytes preview/export produce
(color_grading.plan.md SS4 "Fork A" parity). A shot with no persisted grade
row is recorded `graded == raw` (identity) so it still scores, flagged
`has_grade=False`.

Also recomputes each thread's scene grouping EXACTLY as `grade/job.py` does
(semantic groups, falling back to RGB-adjacency when the semantic result is
all-singletons) so `_diag_qa_score.py`'s consistency metrics can roll up by
group without re-deriving it, and records each shot's `is_log_flat`/
`chroma_mean` (Part B.1's evidence: which subset would the IDT even touch).

Writes stills under backend/scripts/_out/qa/<project_label>/ and a manifest
backend/scripts/_out/qa/samples.json.

Usage: PYTHONPATH=. .venv/bin/python scripts/_diag_qa_sample.py
       [--projects <label-substring>[,<label-substring>...]] [--frames-per-shot N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import psycopg  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.l1 import color_stats as color_stats_mod  # noqa: E402
from app.services.l3 import store as edit_store  # noqa: E402
from app.services.l3.frames import extract_still  # noqa: E402
from app.services.l3.grade import job as grade_job  # noqa: E402
from app.services.l3.grade.cache import ensure_cube_file  # noqa: E402
from app.services.l3.grade.measure_span import SPAN_MAX_FRAMES, measure_span  # noqa: E402
from app.services.l3.grade.scene_group import ShotSceneMeta as SceneMeta  # noqa: E402
from app.services.l3.grade.scene_group import group_shots_semantically  # noqa: E402
from app.services.l3.grade.scene_meta import lookup_shot_cut_meta  # noqa: E402
from app.services.l3.grade.match import ShotStats, group_neighbors  # noqa: E402
from app.services.processing import _download_from_r2  # noqa: E402

CORPUS_PATH = os.path.join(HERE, "_out", "qa", "corpus.json")
OUT_ROOT = os.path.join(HERE, "_out", "qa")
SAMPLES_PATH = os.path.join(OUT_ROOT, "samples.json")
CUBE_DIR = os.path.join(tempfile.gettempdir(), "edso_qa_cubes")
STILL_WIDTH = 640


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return slug or "project"


def apply_cube(frame_path: str, cube_path: Optional[str], out_path: str) -> None:
    """Same recipe as `_grade_v1_frames.py::apply_cube` -- no cube = a plain copy."""
    if not cube_path:
        subprocess.run(["cp", frame_path, out_path], check=True)
        return
    subprocess.run(
        ["ffmpeg", "-y", "-i", frame_path, "-vf", f"lut3d=file='{cube_path}'", out_path],
        check=True, capture_output=True,
    )


def _proxy_paths(file_ids: List[str], tmp_dir: str) -> Dict[str, Optional[str]]:
    """Download each file's proxy (or original) once -- same `r2_proxy_key`
    -preferred lookup `measure_span._fetch_proxy_path` uses."""
    if not file_ids:
        return {}
    with _pg() as c:
        rows = c.execute(
            "select id::text, r2_proxy_key, r2_key from files where id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()
    out: Dict[str, Optional[str]] = {}
    for fid, proxy_key, r2_key in rows:
        key = proxy_key or r2_key
        if not key:
            out[fid] = None
            continue
        ext = os.path.splitext(key)[1] or ".mp4"
        local = os.path.join(tmp_dir, f"{fid}{ext}")
        try:
            _download_from_r2(key, local)
            out[fid] = local
        except Exception as e:
            print(f"    !! download failed for file {fid}: {e}")
            out[fid] = None
    return out


def _resolve_group_ids(shots, shot_stats: Dict[str, Optional[dict]], cut_meta: Dict[str, Any]) -> Dict[str, int]:
    """EXACTLY grade/job.py::run_grade_job's grouping: semantic groups from
    the (already-joined) cut_record metadata + span rgb_mean as the RGB
    base, falling back to RGB-adjacency (`group_neighbors`) when the
    semantic result is all-singletons. Returns shot_key -> group index."""
    scene_meta = []
    for s in shots:
        cm = cut_meta.get(s.key)
        span_rgb = (shot_stats.get(s.key) or {}).get("rgb_mean")
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
    if not grade_job._has_real_groups(semantic_groups):
        semantic_groups = None
    if semantic_groups is not None:
        groups = semantic_groups
        ordered_keys = [s.key for s in shots]
    else:
        ordered = [ShotStats(key=s.key, file_id=s.file_id, stats=shot_stats.get(s.key)) for s in shots]
        groups = group_neighbors(ordered)
        ordered_keys = [s.key for s in shots]

    out: Dict[str, int] = {}
    for gi, idxs in enumerate(groups):
        for i in idxs:
            out[ordered_keys[i]] = gi
    return out


def sample_thread(entry: Dict[str, Any], frames_per_shot: int) -> List[Dict[str, Any]]:
    thread_id = entry["thread_id"]
    label = entry["project_label"]
    slug = _slugify(f"{label}_{entry['project_id'][:8]}")
    project_dir = os.path.join(OUT_ROOT, slug)
    os.makedirs(project_dir, exist_ok=True)

    doc, _version = edit_store.latest_document(thread_id)
    if not doc:
        print(f"  !! {label}: document vanished, skipping")
        return []
    shots = grade_job.ordered_shots(doc)
    if not shots:
        return []

    grades = grade_job.fetch_latest_grades(thread_id, [s.key for s in shots])
    cut_meta = lookup_shot_cut_meta([(s.key, s.file_id, s.in_ms, s.out_ms) for s in shots])

    # Span stats per shot -- same measure_span call job.py makes (cached in
    # cut_color_stats, so a rerun over an unchanged span is cheap), feeding
    # both the RGB grouping base and the is_log_flat/chroma_mean tags.
    shot_stats: Dict[str, Optional[dict]] = {}
    for s in shots:
        subject_box = s.item.get("subject_box") or (cut_meta.get(s.key).subject_box if cut_meta.get(s.key) else None)
        hero_ts_ms = s.hero_ts_ms
        if hero_ts_ms is None and subject_box is not None:
            cm = cut_meta.get(s.key)
            if cm is not None and cm.hero_ts_ms is not None:
                hero_ts_ms = cm.hero_ts_ms
        shot_stats[s.key] = measure_span(s.file_id, s.in_ms, s.out_ms, hero_ts_ms=hero_ts_ms, subject_box=subject_box)

    group_ids = _resolve_group_ids(shots, shot_stats, cut_meta)

    file_ids = sorted({s.file_id for s in shots})
    records: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="edso_qa_sample_") as tmp:
        proxies = _proxy_paths(file_ids, tmp)
        for s in shots:
            proxy_path = proxies.get(s.file_id)
            if not proxy_path:
                print(f"    skip {s.key}: no proxy/original for file {s.file_id}")
                continue

            cm = cut_meta.get(s.key)
            subject_box = s.item.get("subject_box") or (cm.subject_box if cm else None)
            hero_ts_ms = s.hero_ts_ms
            if hero_ts_ms is None and cm is not None and cm.hero_ts_ms is not None:
                hero_ts_ms = cm.hero_ts_ms
            if hero_ts_ms is None or not (s.in_ms <= hero_ts_ms <= s.out_ms):
                hero_ts_ms = (s.in_ms + s.out_ms) // 2

            span_s = max(0.0, (s.out_ms - s.in_ms) / 1000.0)
            n_keyframes = max(0, min(frames_per_shot, SPAN_MAX_FRAMES) - 1)
            keyframe_offsets = color_stats_mod._sample_timestamps(span_s, n_keyframes) if n_keyframes else []
            keyframe_ts = [int(s.in_ms + o * 1000) for o in keyframe_offsets]
            ts_list = sorted({int(hero_ts_ms), *keyframe_ts})[:max(1, frames_per_shot)]

            grade_json = grades.get(s.key)
            has_grade = grade_json is not None
            cube_path = ensure_cube_file(grade_json, CUBE_DIR) if has_grade else None
            look_engine = (grade_json or {}).get("look_engine")
            # color_phase1.plan.md Part 2: the shot's ACTUAL working_space
            # (rec709_v1 or, for a log-tagged clip, log_v1) -- _diag_qa_score.
            # py's look-fidelity bake must use the SAME space the real grade
            # composed in, or the comparison baseline is measuring the wrong
            # decode entirely.
            working_space = (grade_json or {}).get("working_space")

            frame_records = []
            for ts_ms in ts_list:
                raw_path = os.path.join(project_dir, f"{s.key}_{ts_ms}_raw.jpg")
                graded_path = os.path.join(project_dir, f"{s.key}_{ts_ms}_graded.jpg")
                try:
                    extract_still(proxy_path, ts_ms, raw_path, width=STILL_WIDTH)
                except Exception as e:
                    print(f"    !! {s.key}@{ts_ms}: extract_still failed: {e}")
                    continue
                try:
                    apply_cube(raw_path, cube_path, graded_path)
                except Exception as e:
                    print(f"    !! {s.key}@{ts_ms}: apply_cube failed: {e}")
                    continue
                frame_records.append({
                    "ts_ms": ts_ms, "is_hero": ts_ms == int(hero_ts_ms),
                    "raw_path": os.path.relpath(raw_path, OUT_ROOT),
                    "graded_path": os.path.relpath(graded_path, OUT_ROOT),
                })

            if not frame_records:
                continue
            stats = shot_stats.get(s.key) or {}
            records.append({
                "project_id": entry["project_id"], "project_label": label,
                "thread_id": thread_id, "shot_key": s.key, "file_id": s.file_id,
                "group_id": group_ids.get(s.key),
                "subject_box": subject_box,
                "has_grade": has_grade,
                "look_engine": look_engine,
                "working_space": working_space,
                "is_log_flat": bool(stats.get("is_log_flat")),
                "chroma_mean": stats.get("chroma_mean"),
                "frames": frame_records,
            })
        print(f"  {label}: sampled {len(records)}/{len(shots)} shots")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", default=None, help="comma-separated project-label substrings to include")
    parser.add_argument("--frames-per-shot", type=int, default=4, help="max frames sampled per shot (default 4)")
    args = parser.parse_args()

    if not os.path.exists(CORPUS_PATH):
        raise SystemExit(f"no corpus manifest at {CORPUS_PATH} -- run _diag_qa_corpus.py first")
    with open(CORPUS_PATH) as f:
        corpus = json.load(f)

    if args.projects:
        needles = [p.strip().lower() for p in args.projects.split(",") if p.strip()]
        corpus = [e for e in corpus if any(n in e["project_label"].lower() for n in needles)]

    all_records: List[Dict[str, Any]] = []
    for entry in corpus:
        print(f"[sample] {entry['project_label']} ({entry['project_id'][:8]}) thread={entry['thread_id'][:8]}")
        all_records.extend(sample_thread(entry, args.frames_per_shot))

    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(SAMPLES_PATH, "w") as f:
        json.dump(all_records, f, indent=2)
    total_frames = sum(len(r["frames"]) for r in all_records)
    print(f"\nwrote {SAMPLES_PATH}: {len(all_records)} shots, {total_frames} frame pairs")


if __name__ == "__main__":
    main()
