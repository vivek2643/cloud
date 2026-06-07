#!/usr/bin/env python3
"""
Layer B evaluator (READ-ONLY).

The hybrid keyframe plan has three layers:

  Layer A  index-time adaptive coverage  -> how many frames we *store* per shot
  Layer B  edit-time adaptive selection  -> which stored frames we *send* to the
                                            multimodal editor, given an intent
                                            and a hard image budget
  Layer C  the edit path actually using those frames

This script exercises **Layer B against real stored data** without touching the
pipeline, the DB (no writes), or the edit path. For a given file and an edit
query it:

  1. loads every stored keyframe (anchor / peak-motion / peak-variance) for the
     file, with its SigLIP vector,
  2. scores each frame against the query (SigLIP text<->image cosine),
  3. runs MMR selection (relevance + visual diversity) under a frame budget and
     a per-shot cap -- i.e. exactly the set of images Layer C would hand to the
     vision model,
  4. prints the selection, coverage stats, and a naive top-K baseline so you can
     eyeball whether the adaptive picker beats "just take the highest-scoring
     anchors",
  5. optionally downloads the chosen JPEGs so you can *look* at them.

Run from the backend/ directory:

    .venv/bin/python scripts/eval_keyframe_select.py --file-id <uuid> \
        --query "the wide establishing drone shot of the coastline" \
        --budget 12 --download-dir /tmp/kf

    # list candidate files for a user
    .venv/bin/python scripts/eval_keyframe_select.py --user-id <uuid> --list

    # query-less "visual coverage" mode (pure diversity -> storyboard)
    .venv/bin/python scripts/eval_keyframe_select.py --file-id <uuid> --coverage
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# Make `app...` importable when run as `python scripts/eval_keyframe_select.py`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

from app.config import get_settings  # noqa: E402


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

@dataclass
class Frame:
    shot_index: int
    kind: str            # anchor | motion | variance
    ts_ms: int
    r2_key: str
    vec: np.ndarray      # L2-normalized SigLIP image embedding (768,)
    rel: float = 0.0     # cosine to the query (filled per-query)


@dataclass
class ShotRow:
    shot_index: int
    start_ms: int
    end_ms: int
    anchor_key: Optional[str]
    motion_key: Optional[str]
    variance_key: Optional[str]
    peak_motion_ms: Optional[int]
    peak_variance_ms: Optional[int]
    emb_anchor: Optional[np.ndarray]
    emb_motion: Optional[np.ndarray]
    emb_variance: Optional[np.ndarray]
    frames: List[Frame] = field(default_factory=list)


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


def _parse_halfvec(raw) -> Optional[np.ndarray]:
    """pgvector halfvec comes back as a '[a,b,c]' string (or already a list)."""
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        v = np.asarray(raw, dtype=np.float32)
    else:
        s = str(raw).strip().lstrip("[").rstrip("]")
        if not s:
            return None
        v = np.fromstring(s, sep=",", dtype=np.float32)
    if v.size == 0:
        return None
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n > 1e-8 else v


def _fmt_ts(ms: Optional[int]) -> str:
    if ms is None:
        return "--:--"
    s = int(ms) // 1000
    return f"{s // 60:02d}:{s % 60:02d}"


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def list_files(user_id: str) -> None:
    with _pg() as conn:
        rows = conn.execute(
            """
            select f.id, f.name, f.duration_seconds, f.l1_status,
                   count(s.id) as n_shots
              from files f
              left join shots s on s.file_id = f.id
             where f.user_id = %s
             group by f.id, f.name, f.duration_seconds, f.l1_status
             order by f.created_at desc
             limit 50
            """,
            (user_id,),
        ).fetchall()
    if not rows:
        print(f"No files for user {user_id}")
        return
    print(f"{'file_id':38}  {'shots':>5}  {'dur':>7}  {'l1':9}  name")
    for r in rows:
        dur = f"{(r['duration_seconds'] or 0):.0f}s"
        print(f"{str(r['id']):38}  {r['n_shots']:>5}  {dur:>7}  {str(r['l1_status']):9}  {r['name']}")


def _load_from_shot_keyframes(file_id: str) -> List[ShotRow]:
    """Layer A path: variable-count adaptive frames from shot_keyframes."""
    sql = """
        select s.shot_index, s.start_ms, s.end_ms,
               sk.frame_index, sk.kind, sk.ts_ms, sk.r2_key,
               sk.embedding::text as emb
          from shots s
          join shot_keyframes sk on sk.shot_id = s.id
         where s.file_id = %s and sk.embedding is not null
         order by s.shot_index, sk.frame_index
    """
    with _pg() as conn:
        rows = conn.execute(sql, (file_id,)).fetchall()

    by_shot: dict = {}
    for r in rows:
        vec = _parse_halfvec(r["emb"])
        if vec is None:
            continue
        sr = by_shot.get(r["shot_index"])
        if sr is None:
            sr = ShotRow(
                shot_index=r["shot_index"], start_ms=r["start_ms"], end_ms=r["end_ms"],
                anchor_key=None, motion_key=None, variance_key=None,
                peak_motion_ms=None, peak_variance_ms=None,
                emb_anchor=None, emb_motion=None, emb_variance=None,
            )
            by_shot[r["shot_index"]] = sr
        sr.frames.append(Frame(r["shot_index"], r["kind"], int(r["ts_ms"]), r["r2_key"], vec))
    return [by_shot[k] for k in sorted(by_shot)]


def load_shots(file_id: str) -> List[ShotRow]:
    adaptive = _load_from_shot_keyframes(file_id)
    if adaptive:
        print("(source: shot_keyframes / Layer A adaptive coverage)")
        return adaptive
    print("(source: legacy anchor/motion/variance triple)")

    sql = """
        select s.shot_index, s.start_ms, s.end_ms,
               s.keyframe_r2_key            as anchor_key,
               s.r2_keyframe_motion_key     as motion_key,
               s.r2_keyframe_variance_key   as variance_key,
               s.peak_motion_ms, s.peak_variance_ms,
               se.embedding::text           as emb_anchor,
               se.embedding_motion::text    as emb_motion,
               se.embedding_variance::text  as emb_variance
          from shots s
          left join shot_embeddings se on se.shot_id = s.id
         where s.file_id = %s
         order by s.shot_index
    """
    with _pg() as conn:
        rows = conn.execute(sql, (file_id,)).fetchall()

    shots: List[ShotRow] = []
    for r in rows:
        anchor_ts = (int(r["start_ms"]) + int(r["end_ms"])) // 2
        sr = ShotRow(
            shot_index=r["shot_index"],
            start_ms=r["start_ms"],
            end_ms=r["end_ms"],
            anchor_key=r["anchor_key"],
            motion_key=r["motion_key"],
            variance_key=r["variance_key"],
            peak_motion_ms=r["peak_motion_ms"],
            peak_variance_ms=r["peak_variance_ms"],
            emb_anchor=_parse_halfvec(r["emb_anchor"]),
            emb_motion=_parse_halfvec(r["emb_motion"]),
            emb_variance=_parse_halfvec(r["emb_variance"]),
        )
        for kind, key, ts, vec in (
            ("anchor", sr.anchor_key, anchor_ts, sr.emb_anchor),
            ("motion", sr.motion_key, sr.peak_motion_ms if sr.peak_motion_ms is not None else anchor_ts, sr.emb_motion),
            ("variance", sr.variance_key, sr.peak_variance_ms if sr.peak_variance_ms is not None else anchor_ts, sr.emb_variance),
        ):
            if key and vec is not None:
                sr.frames.append(Frame(sr.shot_index, kind, int(ts), key, vec))
        shots.append(sr)
    return shots


def all_frames(shots: List[ShotRow]) -> List[Frame]:
    return [fr for s in shots for fr in s.frames]


# --------------------------------------------------------------------------- #
# Selection (Layer B core)
# --------------------------------------------------------------------------- #

def mmr_select(
    frames: List[Frame],
    budget: int,
    per_shot_max: int,
    lam: float,
) -> List[int]:
    """Maximal-marginal-relevance pick over `frames`.

    score(i) = lam * relevance(i) - (1 - lam) * max_sim(i, already_selected)

    lam=1 -> pure relevance (collapses to top-K by score).
    lam=0 -> pure diversity  (visual coverage / storyboard).
    Respects a per-shot frame cap so one busy shot can't eat the budget.
    """
    if not frames:
        return []
    mat = np.stack([f.vec for f in frames])           # (N, 768)
    rel = np.array([f.rel for f in frames], dtype=np.float32)

    selected: List[int] = []
    shot_counts: dict[int, int] = {}
    remaining = set(range(len(frames)))
    # running max similarity of each candidate to the selected set
    max_sim = np.zeros(len(frames), dtype=np.float32)

    while remaining and len(selected) < budget:
        best_i, best_score = -1, -1e9
        for i in remaining:
            if shot_counts.get(frames[i].shot_index, 0) >= per_shot_max:
                continue
            score = lam * rel[i] - (1.0 - lam) * max_sim[i]
            if score > best_score:
                best_score, best_i = score, i
        if best_i < 0:
            break  # everything left is blocked by the per-shot cap
        selected.append(best_i)
        remaining.discard(best_i)
        shot_counts[frames[best_i].shot_index] = shot_counts.get(frames[best_i].shot_index, 0) + 1
        sims = mat @ mat[best_i]
        max_sim = np.maximum(max_sim, sims)
    return selected


def naive_topk(frames: List[Frame], budget: int) -> List[int]:
    """Baseline: highest-scoring anchor frames, one per shot, ignoring diversity."""
    anchors = [i for i, f in enumerate(frames) if f.kind == "anchor"]
    anchors.sort(key=lambda i: frames[i].rel, reverse=True)
    return anchors[:budget]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def coverage_stats(frames: List[Frame], idxs: List[int], n_shots: int) -> dict:
    if not idxs:
        return {"frames": 0}
    sel = [frames[i] for i in idxs]
    shots = {f.shot_index for f in sel}
    ts = np.array([f.ts_ms for f in sel], dtype=np.float64)
    rels = np.array([f.rel for f in sel], dtype=np.float64)
    # pairwise visual redundancy among selected (mean off-diagonal cosine)
    mat = np.stack([f.vec for f in sel])
    sim = mat @ mat.T
    n = len(sel)
    redundancy = (sim.sum() - np.trace(sim)) / max(n * (n - 1), 1)
    return {
        "frames": n,
        "shots_covered": len(shots),
        "shot_coverage_pct": 100.0 * len(shots) / max(n_shots, 1),
        "timeline_span": f"{_fmt_ts(int(ts.min()))}-{_fmt_ts(int(ts.max()))}",
        "mean_rel": float(rels.mean()),
        "min_rel": float(rels.min()),
        "mean_redundancy": float(redundancy),
    }


def print_selection(title: str, frames: List[Frame], idxs: List[int]) -> None:
    print(f"\n  {title}")
    print(f"    {'#':>2}  {'shot':>4}  {'kind':8}  {'ts':>6}  {'rel':>6}  r2_key")
    for rank, i in enumerate(idxs, 1):
        f = frames[i]
        print(f"    {rank:>2}  {f.shot_index:>4}  {f.kind:8}  {_fmt_ts(f.ts_ms):>6}  "
              f"{f.rel:6.3f}  {f.r2_key}")


def print_stats(label: str, stats: dict) -> None:
    if stats.get("frames", 0) == 0:
        print(f"    [{label}] no frames")
        return
    print(f"    [{label}] frames={stats['frames']}  "
          f"shots={stats['shots_covered']} ({stats['shot_coverage_pct']:.0f}% of video)  "
          f"span={stats['timeline_span']}  "
          f"mean_rel={stats['mean_rel']:.3f} min_rel={stats['min_rel']:.3f}  "
          f"redundancy={stats['mean_redundancy']:.3f}")


def download_frames(frames: List[Frame], idxs: List[int], out_dir: str, tag: str) -> None:
    from app.services.processing import _download_from_r2
    sub = os.path.join(out_dir, tag)
    os.makedirs(sub, exist_ok=True)
    for rank, i in enumerate(idxs, 1):
        f = frames[i]
        dest = os.path.join(sub, f"{rank:02d}_shot{f.shot_index:04d}_{f.kind}.jpg")
        try:
            _download_from_r2(f.r2_key, dest)
        except Exception as e:  # noqa: BLE001
            print(f"    ! download failed for {f.r2_key}: {e}")
    print(f"    saved {len(idxs)} frames -> {sub}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def run_query(
    shots: List[ShotRow],
    frames: List[Frame],
    query: Optional[str],
    budget: int,
    per_shot_max: int,
    lam: float,
    out_dir: Optional[str],
) -> None:
    n_shots = len(shots)
    header = f'query="{query}"' if query else "COVERAGE (query-less, pure diversity)"
    print(f"\n{'=' * 78}\n{header}\n{'=' * 78}")

    if query:
        from app.services.l1 import embeddings as emb_mod
        qv = emb_mod.embed_text(query)
        qv = qv / (np.linalg.norm(qv) + 1e-8)
        mat = np.stack([f.vec for f in frames])
        rels = mat @ qv
        for f, r in zip(frames, rels):
            f.rel = float(r)
        effective_lam = lam
    else:
        for f in frames:
            f.rel = 0.0
        effective_lam = 0.0  # ignore (uniform) relevance -> pure coverage

    sel = mmr_select(frames, budget, per_shot_max, effective_lam)
    print_selection(f"ADAPTIVE (MMR lam={effective_lam}, budget={budget}, per_shot_max={per_shot_max})",
                    frames, sel)
    print_stats("adaptive", coverage_stats(frames, sel, n_shots))

    if query:
        base = naive_topk(frames, budget)
        print_stats("baseline(top-K anchors)", coverage_stats(frames, base, n_shots))

    if out_dir:
        tag = "coverage" if not query else re.sub(r"[^a-z0-9]+", "_", query.lower())[:40]
        download_frames(frames, sel, out_dir, tag)


def main() -> int:
    ap = argparse.ArgumentParser(description="Read-only Layer B keyframe-selection evaluator")
    ap.add_argument("--file-id", help="file UUID to evaluate")
    ap.add_argument("--user-id", help="user UUID (with --list to enumerate files)")
    ap.add_argument("--list", action="store_true", help="list this user's files and exit")
    ap.add_argument("--query", action="append", default=[],
                    help="edit-intent query (repeatable for multiple intents)")
    ap.add_argument("--coverage", action="store_true",
                    help="also run query-less visual-coverage selection")
    ap.add_argument("--budget", type=int, default=12, help="max frames to select")
    ap.add_argument("--per-shot-max", type=int, default=1, help="max frames from one shot")
    ap.add_argument("--lam", type=float, default=0.6,
                    help="MMR lambda: 1=pure relevance, 0=pure diversity")
    ap.add_argument("--download-dir", help="save selected JPEGs here for visual inspection")
    args = ap.parse_args()

    if args.list:
        if not args.user_id:
            ap.error("--list requires --user-id")
        list_files(args.user_id)
        return 0

    if not args.file_id:
        ap.error("--file-id is required (or use --user-id --list)")

    shots = load_shots(args.file_id)
    frames = all_frames(shots)
    print(f"file {args.file_id}: {len(shots)} shots, {len(frames)} stored keyframes "
          f"({sum(1 for f in frames if f.kind == 'anchor')} anchor / "
          f"{sum(1 for f in frames if f.kind == 'motion')} motion / "
          f"{sum(1 for f in frames if f.kind == 'variance')} variance)")
    if not frames:
        print("No keyframes/embeddings stored for this file -- has L1 finished?")
        return 1

    if not args.query and not args.coverage:
        # default to coverage so the script does something useful with no query
        args.coverage = True

    for q in args.query:
        run_query(shots, frames, q, args.budget, args.per_shot_max, args.lam, args.download_dir)

    if args.coverage:
        run_query(shots, frames, None, args.budget, args.per_shot_max, args.lam, args.download_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
