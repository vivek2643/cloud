"""
cpd_boundary_segmenter.plan.md Phase A -- data acquisition.

Two independent steps:
  1. Annotations: Kinetics-GEBD's release pickle (per-video multi-annotator
     boundary timestamps). The official release is hosted on Google Drive
     (github.com/StanLei52/GEBD -- there is no stable, direct-curl URL), so
     the supported path is: download it yourself in a browser and pass
     --annotations-path, OR pass --gdrive-file-id to try an automated
     best-effort fetch via `gdown` (installed on demand) -- Drive can still
     throttle/interstitial large automated downloads, so treat this as
     opportunistic, not guaranteed.
  2. Videos: yt-dlp, one `--download-sections` call per clip so we only
     pull the labeled time range (not the whole source video), at low
     resolution -- our own extractor runs on a tiny proxy anyway.

Writes data/manifest.csv: clip_id, video_id, split, path, duration_sec,
annotator_boundaries_sec_json (one list of boundary seconds per annotator,
relative to the clip's own start -- Phase D's consensus rule consumes this
directly; nothing is pre-flattened here so no consensus decision is baked
in before build_labels.py runs).

Fallback (TAPOS or any other (video, boundary-timestamps) source): write
data/manifest.csv by hand or a small adapter script in this same shape --
everything downstream (Phase C onward) only depends on the manifest, never
on Kinetics/YouTube specifically.

Run:
  .venv/bin/python scripts/cpd/fetch_gebd.py \\
      --annotations-path /path/to/k400_mr345_train_min_change_duration0.3.pkl \\
      --split train --limit 500
  .venv/bin/python scripts/cpd/fetch_gebd.py \\
      --annotations-path /path/to/..._val_....pkl --split val --limit 200
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import pickle
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(os.path.dirname(HERE))
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import clip_id_for  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fetch_gebd")

DATA_DIR = os.path.join(HERE, "data")
RAW_VIDEOS_DIR = os.path.join(DATA_DIR, "raw_videos")
ANNOTATIONS_DIR = os.path.join(DATA_DIR, "annotations")
MANIFEST_PATH = os.path.join(DATA_DIR, "manifest.csv")


# --------------------------------------------------------------------------
# Annotations
# --------------------------------------------------------------------------

def _try_gdown_fetch(gdrive_file_id: str, dest_path: str) -> bool:
    try:
        import gdown  # type: ignore
    except ImportError:
        logger.warning(
            "gdown not installed -- run `.venv/bin/pip install gdown` to enable "
            "the automated Drive fetch, or pass --annotations-path to a file you "
            "downloaded manually.")
        return False
    try:
        gdown.download(id=gdrive_file_id, output=dest_path, quiet=False)
        return os.path.exists(dest_path) and os.path.getsize(dest_path) > 0
    except Exception:
        logger.exception("gdown fetch failed for file id %s", gdrive_file_id)
        return False


def _extract_timestamp(entry: Any) -> Optional[float]:
    """One raw GEBD boundary mark -> its timestamp in seconds. The REAL
    k400_{train,val}_raw_annotation.pkl release (verified against the
    actual downloaded file -- see fetch: {"start_time", "end_time",
    "label"} per mark; a "Timestamp" mark has start==end, a
    "Range"/gradual-change mark has start<end, midpointed here) is the
    primary shape; a bare number / [timestamp, label] pair / a dict with a
    plain timestamp-like key are kept as a defensive fallback in case a
    different release (TAPOS, a future GEBD version) uses one of those
    instead. Returns None (never raises) on a shape this doesn't recognize
    -- the caller logs and skips rather than mis-parsing silently."""
    if isinstance(entry, dict):
        if "start_time" in entry and "end_time" in entry:
            return (float(entry["start_time"]) + float(entry["end_time"])) / 2.0
        for key in ("timestamp", "time", "t"):
            if key in entry and isinstance(entry[key], (int, float)):
                return float(entry[key])
        return None
    if isinstance(entry, (int, float)):
        return float(entry)
    if isinstance(entry, (list, tuple)) and entry and isinstance(entry[0], (int, float)):
        return float(entry[0])
    return None


def _parse_path_video(path_video: str) -> Optional[Tuple[str, float, float]]:
    """<kinetics_label>/<youtube_id>_<start_sec>_<end_sec>.mp4 (Kinetics'
    own clip naming, seconds zero-padded to 6 digits) -> (video_id,
    start_sec, end_sec). This is the verified real source of a raw
    annotation entry's clip identity -- the pickle's own top-level dict key
    is JUST the bare youtube_id, NOT `<id>_<start>_<end>` as earlier GEBD
    documentation snippets could be read to suggest."""
    stem, _ext = os.path.splitext(os.path.basename(path_video))
    parts = stem.rsplit("_", 2)
    if len(parts) != 3:
        return None
    video_id, start_s, end_s = parts
    try:
        return video_id, float(start_s), float(end_s)
    except ValueError:
        return None


def load_annotations(path: str) -> Dict[str, Dict[str, Any]]:
    """Load the GEBD release pickle into {clip_id: {"video_id", "start_sec",
    "end_sec", "duration_sec", "annotator_boundaries_sec": [[...], ...]}},
    ``clip_id`` built via common.clip_id_for(video_id, start_sec, end_sec)
    (see _parse_path_video -- the pickle's own dict key is not the clip
    identity, ``path_video`` is).

    Schema-flexible on purpose (see module docstring): grounded in the
    REAL k400_raw_annotation.pkl fields, verified against the actual
    downloaded release (num_frames, path_video, fps, video_duration,
    f1_consis, substages_timestamps -- a list of per-annotator lists of
    {start_time, end_time, label} dicts). If the first entry has neither
    'substages_timestamps' nor 'path_video', this raises with the actual
    keys seen so a human can adjust the parser in one place rather than
    the pipeline silently training on an empty label set.
    """
    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a dict keyed by clip id, got {type(raw)}")

    out: Dict[str, Dict[str, Any]] = {}
    checked_shape = False
    n_unparseable_path = 0
    for _key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        if not checked_shape:
            checked_shape = True
            if "substages_timestamps" not in entry or "path_video" not in entry:
                raise ValueError(
                    f"{path}: first entry is missing 'substages_timestamps' or "
                    f"'path_video'. Actual keys: {sorted(entry.keys())}. Update "
                    f"load_annotations()/_parse_path_video() with the real field "
                    f"names before proceeding.")

        parsed = _parse_path_video(entry.get("path_video") or "")
        if parsed is None:
            n_unparseable_path += 1
            continue
        video_id, start_sec, end_sec = parsed
        duration_sec = float(entry.get("video_duration") or (end_sec - start_sec))

        annotator_boundaries_sec: List[List[float]] = []
        for annotator_marks in (entry.get("substages_timestamps") or []):
            marks_sec = [t for t in (_extract_timestamp(m) for m in (annotator_marks or []))
                        if t is not None]
            annotator_boundaries_sec.append(marks_sec)

        clip_id = clip_id_for(video_id, start_sec, end_sec)
        out[clip_id] = {
            "video_id": video_id, "start_sec": start_sec, "end_sec": end_sec,
            "duration_sec": duration_sec,
            "annotator_boundaries_sec": annotator_boundaries_sec,
        }
    if n_unparseable_path:
        logger.warning("skipped %d entries with an unparseable path_video", n_unparseable_path)
    logger.info("loaded annotations for %d clips from %s", len(out), path)
    return out


# --------------------------------------------------------------------------
# Videos
# --------------------------------------------------------------------------

def _yt_dlp_bin() -> str:
    """Resolve the yt-dlp executable robustly: PATH first, else the venv
    directory sys.executable itself lives in (a subprocess spawned from
    `.venv/bin/python` does NOT automatically get `.venv/bin` on its own
    PATH, so a bare "yt-dlp" can 404 even though `pip install yt-dlp` put
    it right next to the interpreter)."""
    import shutil
    found = shutil.which("yt-dlp")
    if found:
        return found
    candidate = os.path.join(os.path.dirname(sys.executable), "yt-dlp")
    return candidate if os.path.exists(candidate) else "yt-dlp"


def download_clip(video_id: str, start_sec: float, end_sec: float, dest_path: str) -> bool:
    """One yt-dlp call, time-ranged so only the labeled window downloads.
    Returns False (never raises) on any failure -- link rot / region locks /
    removed videos are expected at this scale; the caller logs coverage.

    --extractor-args youtube:player_client=android: as of late 2025,
    YouTube's default (web/tv) extraction path fails almost universally
    with "The page needs to be reloaded" / SABR-streaming errors unless
    yt-dlp has a PO-token provider configured (github.com/yt-dlp/yt-dlp
    issue #12482) -- verified live against this environment (both a
    Kinetics clip and a known-public video 403'd on every client except
    android, which yt-dlp's own player-response formats still work
    through). If YouTube closes this path too, the fix is the same one
    line -- swap the client name or add a PO-token provider -- not a
    pipeline redesign."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        return True
    url = f"https://www.youtube.com/watch?v={video_id}"
    cmd = [
        _yt_dlp_bin(), "-q", "--no-warnings",
        "--extractor-args", "youtube:player_client=android",
        "-f", "bv*[height<=256]+ba/b[height<=256]/worst",
        "--download-sections", f"*{start_sec}-{end_sec}",
        "--force-keyframes-at-cuts",
        "-o", dest_path,
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        logger.warning("timeout downloading %s", video_id)
        return False
    if result.returncode != 0 or not os.path.exists(dest_path):
        logger.debug("yt-dlp failed for %s: %s", video_id, (result.stderr or "")[-300:])
        return False
    return True


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _existing_manifest_clip_ids() -> set:
    if not os.path.exists(MANIFEST_PATH):
        return set()
    with open(MANIFEST_PATH, newline="") as fh:
        return {r["clip_id"] for r in csv.DictReader(fh)}


def run(annotations_path: str, split: str, limit: Optional[int]) -> None:
    os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)
    os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

    annotations = load_annotations(annotations_path)
    already_done = _existing_manifest_clip_ids()
    clip_ids = [c for c in annotations.keys() if c not in already_done]
    if limit is not None:
        clip_ids = clip_ids[:limit]
    if already_done:
        logger.info("%d clip(s) already in manifest.csv -- skipping, fetching up to %s more",
                   len(already_done), limit if limit is not None else "all remaining")

    rows: List[Dict[str, Any]] = []
    n_ok = 0
    for i, clip_id in enumerate(clip_ids):
        info = annotations[clip_id]
        dest_path = os.path.join(RAW_VIDEOS_DIR, f"{clip_id}.mp4")

        ok = download_clip(info["video_id"], info["start_sec"], info["end_sec"], dest_path)
        if ok:
            n_ok += 1
            rows.append({
                "clip_id": clip_id, "video_id": info["video_id"], "split": split,
                "path": os.path.relpath(dest_path, DATA_DIR),
                "duration_sec": info["duration_sec"],
                "annotator_boundaries_sec_json": json.dumps(info["annotator_boundaries_sec"]),
            })
        if (i + 1) % 25 == 0:
            logger.info("progress: %d/%d attempted, %d downloaded", i + 1, len(clip_ids), n_ok)

    logger.info("done: %d/%d clips downloaded for split=%s (%.0f%% coverage)",
               n_ok, len(clip_ids), split, 100.0 * n_ok / max(1, len(clip_ids)))
    _append_manifest(rows)


def _append_manifest(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    already_done = _existing_manifest_clip_ids()
    rows = [r for r in rows if r["clip_id"] not in already_done]
    if not rows:
        return
    fieldnames = ["clip_id", "video_id", "split", "path", "duration_sec",
                 "annotator_boundaries_sec_json"]
    write_header = not os.path.exists(MANIFEST_PATH)
    with open(MANIFEST_PATH, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
    logger.info("appended %d row(s) to %s", len(rows), MANIFEST_PATH)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations-path", help="local path to a GEBD release .pkl")
    ap.add_argument("--gdrive-file-id", help="best-effort: fetch the pkl via gdown first")
    ap.add_argument("--split", required=True, choices=["train", "val"])
    ap.add_argument("--limit", type=int, default=None, help="cap the number of clips attempted")
    args = ap.parse_args()

    annotations_path = args.annotations_path
    if not annotations_path and args.gdrive_file_id:
        annotations_path = os.path.join(ANNOTATIONS_DIR, f"gebd_{args.split}.pkl")
        os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
        if not _try_gdown_fetch(args.gdrive_file_id, annotations_path):
            raise SystemExit(
                "gdown fetch failed -- download the annotation pkl manually from "
                "https://drive.google.com/drive/folders/1AlPr63Q9D-HAGc5bOUNTzjCiWOC1a3xo "
                "and re-run with --annotations-path.")
    if not annotations_path:
        raise SystemExit("pass --annotations-path (or --gdrive-file-id for a best-effort fetch)")

    run(annotations_path, args.split, args.limit)


if __name__ == "__main__":
    main()
