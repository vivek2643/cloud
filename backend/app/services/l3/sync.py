"""
L3 synchronized-source detection (multicam, single frame).

Multiple recordings of the SAME moment (camera angles, a screen-rec + a webcam,
a separate audio recorder) can be aligned in time by the fact that they captured
the same sound. We align them by cross-correlating the per-file `sync_env` that
L1 already stores on audio_features -- a fixed-hop (500 ms), normalized energy
envelope built for exactly this ("same recording from two cameras -> near
identical sync_env").

Pipeline:
  S1 coarse  -- pairwise normalized cross-correlation of sync_env -> best lag
                (multiple of the 500 ms hop) + a confidence (the peak Pearson
                correlation). Files whose confidence clears a threshold are the
                same moment.
  S2 refine  -- parabolic interpolation around the correlation peak gives a
                sub-hop offset (well under 500 ms) with no extra data. A broad/
                low peak is flagged low-confidence (possible clock drift); true
                sample-accuracy is a later pass.
  S3 group   -- union-find clusters synced files into a group sharing one master
                time, with each member's offset relative to an anchor and a
                suggested hero (cleanest) audio source.

Source switching itself needs NO new machinery: an angle switch is just a
`place_video` of another group member, and the engine derives its source time
from the group offset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.config import get_settings

logger = logging.getLogger(__name__)

# Min peak correlation (0..1) to call two clips the same moment.
SYNC_CONF_THRESHOLD = 0.6
# A clip must overlap the other by at least this many hops to be comparable.
MIN_OVERLAP_HOPS = 6  # ~3 s at the 500 ms sync hop


@dataclass
class SyncMember:
    file_id: str
    offset_ms: int           # this clip's start in the group's master time
    confidence: float        # correlation to the anchor


@dataclass
class SyncGroup:
    group_id: str
    members: List[SyncMember] = field(default_factory=list)
    anchor_file_id: str = ""
    hero_file_id: str = ""   # suggested cleanest-audio source (overridable)
    drift_warning: bool = False

    def offset_of(self, file_id: str) -> Optional[int]:
        for m in self.members:
            if m.file_id == file_id:
                return m.offset_ms
        return None


def _pg_conn():
    import psycopg
    return psycopg.connect(get_settings().database_url, autocommit=True)


def load_sync_envs(file_ids: List[str]) -> Dict[str, Tuple[int, List[float], Optional[float]]]:
    """file_id -> (sync_hop_ms, sync_env, integrated_lufs). Missing/empty skipped."""
    if not file_ids:
        return {}
    out: Dict[str, Tuple[int, List[float], Optional[float]]] = {}
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select file_id, sync_hop_ms, sync_env, integrated_lufs
              from audio_features
             where file_id = any(%s)
            """,
            (list(file_ids),),
        ).fetchall()
    for fid, hop, env, lufs in rows:
        env = env if isinstance(env, list) else []
        if hop and env:
            out[str(fid)] = (int(hop), [float(x) for x in env], lufs)
    return out


def _best_lag(a: List[float], b: List[float]) -> Tuple[int, float, float]:
    """Normalized cross-correlation of two envelopes.

    Returns (best_lag_hops, peak_corr, sub_hop_frac). A positive lag means `b`
    starts `lag` hops AFTER `a`. sub_hop_frac in (-0.5,0.5) refines the peak via
    parabolic interpolation.
    """
    import numpy as np

    va, vb = np.asarray(a, float), np.asarray(b, float)
    na, nb = va.size, vb.size
    if na == 0 or nb == 0:
        return 0, 0.0, 0.0

    best_lag, best_corr = 0, -2.0
    corr_at: Dict[int, float] = {}
    # Slide b across a; require a real overlap so a tiny tail can't score high.
    for lag in range(-(nb - 1), na):
        a0 = max(0, lag)
        b0 = max(0, -lag)
        n = min(na - a0, nb - b0)
        if n < MIN_OVERLAP_HOPS:
            continue
        sa = va[a0:a0 + n]
        sb = vb[b0:b0 + n]
        sa = sa - sa.mean()
        sb = sb - sb.mean()
        denom = (np.linalg.norm(sa) * np.linalg.norm(sb))
        c = float(np.dot(sa, sb) / denom) if denom > 0 else 0.0
        corr_at[lag] = c
        if c > best_corr:
            best_corr, best_lag = c, lag

    # Parabolic interpolation around the peak for a sub-hop estimate.
    frac = 0.0
    cm = corr_at.get(best_lag - 1)
    cp = corr_at.get(best_lag + 1)
    if cm is not None and cp is not None:
        denom = (cm - 2 * best_corr + cp)
        if denom != 0:
            frac = 0.5 * (cm - cp) / denom
            frac = max(-0.5, min(0.5, frac))
    return best_lag, best_corr, frac


def align_pair(
    env_a: Tuple[int, List[float], Optional[float]],
    env_b: Tuple[int, List[float], Optional[float]],
) -> Tuple[int, float]:
    """Returns (offset_ms of b relative to a, confidence). Positive => b starts
    after a."""
    hop_a, a, _ = env_a
    hop_b, b, _ = env_b
    # Envelopes share the fixed SYNC_HOP_MS grid by construction; guard anyway.
    if hop_a != hop_b:
        return 0, 0.0
    lag, corr, frac = _best_lag(a, b)
    offset_ms = int(round((lag + frac) * hop_a))
    return offset_ms, max(0.0, corr)


def build_sync_groups(file_ids: List[str], threshold: float = SYNC_CONF_THRESHOLD) -> List[SyncGroup]:
    """Cluster the in-scope clips into synchronized-source groups."""
    envs = load_sync_envs(file_ids)
    fids = list(envs.keys())
    if len(fids) < 2:
        return []

    # Pairwise alignment -> edges above threshold (union-find clusters).
    parent = {f: f for f in fids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    pair_offset: Dict[Tuple[str, str], Tuple[int, float]] = {}
    for i in range(len(fids)):
        for j in range(i + 1, len(fids)):
            off, conf = align_pair(envs[fids[i]], envs[fids[j]])
            if conf >= threshold:
                pair_offset[(fids[i], fids[j])] = (off, conf)
                union(fids[i], fids[j])

    clusters: Dict[str, List[str]] = {}
    for f in fids:
        clusters.setdefault(find(f), []).append(f)

    groups: List[SyncGroup] = []
    gi = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        gi += 1
        # Anchor = the member with the most/strongest links (first is fine here).
        anchor = members[0]
        sm: List[SyncMember] = []
        drift = False
        for f in members:
            if f == anchor:
                sm.append(SyncMember(file_id=f, offset_ms=0, confidence=1.0))
                continue
            key = (anchor, f) if (anchor, f) in pair_offset else (f, anchor)
            if key in pair_offset:
                off, conf = pair_offset[key]
                if key[0] != anchor:  # stored as (f, anchor) -> invert sign
                    off = -off
            else:
                off, conf = align_pair(envs[anchor], envs[f])
            if conf < threshold:
                drift = True
            sm.append(SyncMember(file_id=f, offset_ms=off, confidence=round(conf, 3)))
        # Hero = loudest integrated audio (rough proxy for "cleanest"; overridable).
        hero = max(members, key=lambda f: (envs[f][2] if envs[f][2] is not None else -120.0))
        groups.append(SyncGroup(
            group_id=f"sg{gi}", members=sm,
            anchor_file_id=anchor, hero_file_id=hero, drift_warning=drift,
        ))
    return groups


def find_group_for(groups: List[SyncGroup], file_id: str) -> Optional[SyncGroup]:
    for g in groups:
        if g.offset_of(file_id) is not None:
            return g
    return None


def render_sync_groups_text(groups: List[SyncGroup]) -> str:
    """Compact catalog block so Opus knows which clips are the same moment."""
    if not groups:
        return ""
    lines = ["SYNCHRONIZED-SOURCE GROUPS (same moment, multiple angles/sources):"]
    for g in groups:
        mem = ", ".join(
            f"{m.file_id}(+{m.offset_ms}ms)" + (" [hero]" if m.file_id == g.hero_file_id else "")
            for m in g.members
        )
        warn = "  ! low-confidence/drift" if g.drift_warning else ""
        lines.append(f"  {g.group_id}: {mem}{warn}")
    lines.append(
        "  -> switch angles with place_video of another member (its source time "
        "is derived from the offset automatically); keep audio on the [hero] source."
    )
    return "\n".join(lines)
