"""
Coverage substrate: alternate angles + b-roll for any moment on the spine.

This is the general mechanism that makes multicam camera-switching and b-roll
cutaways the SAME operation instead of two special features. For a moment on
the spine we ask "what else could overlay this?" and score candidates by:

  - audio simultaneity  -> near-identical audio == the same moment from another
    camera. Detected by correlating the fixed-hop sync envelopes; the time
    offset falls out for free.
  - (future) visual similarity (SigLIP) and topic overlap for non-simultaneous
    b-roll. Left as hooks here; v1 leans on simultaneity + same-corpus visuals.

Pure functions over the loaded analyses/units -- no DB, no model calls -- so the
director/composer can call this cheaply and it stays unit-testable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Correlation above this == the two files captured the same audio (multicam).
SIMULTANEITY_MIN_SCORE = 0.65
# Don't bother correlating envelopes shorter than this many samples.
MIN_SYNC_SAMPLES = 8


def align_files(
    env_a: Sequence[float], env_b: Sequence[float], hop_ms: int
) -> Tuple[int, float]:
    """Cross-correlate two normalized energy envelopes sharing a time grid.

    Returns (offset_ms, score) where offset_ms is how far file B lags file A
    (B_time = A_time - offset_ms), and score in [-1, 1] is the peak normalized
    correlation. High score == same recording.
    """
    import numpy as np

    if hop_ms <= 0 or len(env_a) < MIN_SYNC_SAMPLES or len(env_b) < MIN_SYNC_SAMPLES:
        return 0, 0.0
    a = np.asarray(env_a, dtype=np.float64)
    b = np.asarray(env_b, dtype=np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(float(np.sum(a * a)) * float(np.sum(b * b))))
    if denom <= 1e-9:
        return 0, 0.0
    corr = np.correlate(a, b, mode="full")
    best = int(np.argmax(corr))
    lag = best - (len(b) - 1)          # >0 => b starts later than a
    score = float(corr[best]) / denom
    return int(lag * hop_ms), score


@dataclass
class SimCluster:
    """A group of files that captured the same moment. `offsets[file_id]` is the
    ms to add to that file's local time to reach the cluster's shared clock."""
    ref_file_id: str
    offsets: Dict[str, int] = field(default_factory=dict)
    members: List[str] = field(default_factory=list)


def simultaneity_map(analyses) -> Dict[str, SimCluster]:
    """Cluster files by shared audio. Returns {file_id -> SimCluster}.

    Single-camera corpora (or files with no audio) yield singleton clusters, so
    callers can treat everything uniformly.
    """
    items = list(analyses.values()) if isinstance(analyses, dict) else list(analyses)
    file_ids = [fa.file_id for fa in items]
    env = {
        fa.file_id: (fa.audio.sync_env, fa.audio.sync_hop_ms)
        for fa in items
        if getattr(fa, "audio", None) and fa.audio.sync_env and fa.audio.sync_hop_ms
    }

    # Union-find over pairwise alignment above threshold.
    parent = {fid: fid for fid in file_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    pair_offsets: Dict[Tuple[str, str], int] = {}
    ids_with_env = list(env.keys())
    for i in range(len(ids_with_env)):
        for j in range(i + 1, len(ids_with_env)):
            fa_i, fa_j = ids_with_env[i], ids_with_env[j]
            (ea, hop_a), (eb, hop_b) = env[fa_i], env[fa_j]
            if hop_a != hop_b:
                continue
            offset_ms, score = align_files(ea, eb, hop_a)
            if score >= SIMULTANEITY_MIN_SCORE:
                pair_offsets[(fa_i, fa_j)] = offset_ms
                union(fa_i, fa_j)

    # Build clusters; assign per-member offset relative to a chosen reference.
    groups: Dict[str, List[str]] = {}
    for fid in file_ids:
        groups.setdefault(find(fid), []).append(fid)

    out: Dict[str, SimCluster] = {}
    for root, members in groups.items():
        cluster = SimCluster(ref_file_id=root, members=members)
        cluster.offsets[root] = 0
        # Resolve each member's offset to the reference via known pair offsets.
        for m in members:
            if m == root:
                continue
            off = pair_offsets.get((root, m))
            if off is None:
                rev = pair_offsets.get((m, root))
                off = -rev if rev is not None else 0
            cluster.offsets[m] = int(off)
        for m in members:
            out[m] = cluster
    return out


def coverage_candidates(
    spine_unit,
    all_units,
    sim: Dict[str, SimCluster],
    *,
    max_candidates: int = 6,
) -> List:
    """Coverage units (alternate angles + b-roll) that could overlay a spine unit.

    Priority:
      1. Simultaneous angles -- coverage-lane units from OTHER files in the same
         simultaneity cluster whose (offset-corrected) time overlaps the spine.
      2. Same-file cutaways -- coverage-lane visual units overlapping in time.
      3. Topical b-roll -- remaining coverage-lane units, highest quality first.
    Returns up to `max_candidates`, de-duplicated, best first.
    """
    cluster = sim.get(spine_unit.file_id)
    spine_off = cluster.offsets.get(spine_unit.file_id, 0) if cluster else 0
    s_lo = spine_unit.in_ms + spine_off
    s_hi = spine_unit.out_ms + spine_off

    simultaneous: List = []
    same_file: List = []
    broll: List = []
    for u in all_units:
        if u.lane != "coverage" or u.id == spine_unit.id:
            continue
        u_cluster = sim.get(u.file_id)
        same_cluster = bool(cluster and u_cluster and u_cluster.ref_file_id == cluster.ref_file_id)
        if same_cluster and u.file_id != spine_unit.file_id:
            u_off = u_cluster.offsets.get(u.file_id, 0)
            if _overlaps(u.in_ms + u_off, u.out_ms + u_off, s_lo, s_hi):
                simultaneous.append(u)
                continue
        if u.file_id == spine_unit.file_id and _overlaps(u.in_ms, u.out_ms, spine_unit.in_ms, spine_unit.out_ms):
            same_file.append(u)
        else:
            broll.append(u)

    simultaneous.sort(key=lambda u: u.quality, reverse=True)
    same_file.sort(key=lambda u: u.quality, reverse=True)
    broll.sort(key=lambda u: u.quality, reverse=True)

    ordered: List = []
    seen = set()
    for group in (simultaneous, same_file, broll):
        for u in group:
            if u.id in seen:
                continue
            seen.add(u.id)
            ordered.append(u)
            if len(ordered) >= max_candidates:
                return ordered
    return ordered


def _overlaps(a_lo: int, a_hi: int, b_lo: int, b_hi: int) -> bool:
    return a_lo < b_hi and b_lo < a_hi
