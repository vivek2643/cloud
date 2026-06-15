"""
On-demand audio alignment for synchronized sources (multicam / a separate
recorder / a screen-rec + webcam).

Two recordings of the SAME moment captured the same sound, so we can recover
their time offset by cross-correlating the per-file `sync_env` that L1 stores on
audio_features -- a fixed-hop (500 ms), normalized energy envelope built for
exactly this.

This is deliberately NOT a global precompute that partitions the library into
"sync groups" (that mislabeled distinct-content clips as duplicate angles and
made the model drop unique footage). It is an AIRTIGHT, UNIVERSAL, on-demand
query: `align_clips(a, b, span)` answers "are these two simultaneous, and if so
by what offset?" for the one pair a cut is actually considering, and returns an
honest `None` when they are not simultaneous (or one is silent).

Airtight = three gates a coincidental match cannot pass:
  1. SUBSTANTIAL OVERLAP -- the compared region must be long (>= ~25 s and a
     quarter of the shorter clip); a lucky 3-second tail is ineligible.
  2. PEAK PROMINENCE -- the best lag must beat the best lag far from it by a
     margin; a flat/ambiguous correlation surface is rejected.
  3. CONFIDENCE FLOOR -- the peak Pearson correlation must clear a threshold.
A sub-hop offset is refined by parabolic interpolation around the peak.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from app.config import get_settings

logger = logging.getLogger(__name__)

# Min peak Pearson correlation (0..1) to accept a match.
SYNC_CONF_THRESHOLD = 0.45
# Overlap must be at least this many hops AND this fraction of the shorter clip.
MIN_OVERLAP_HOPS_FLOOR = 50      # ~25 s at the 500 ms sync hop
MIN_OVERLAP_FRACTION = 0.25      # of the shorter envelope
# The best lag must beat the best lag outside +/- NEIGHBOR hops by this margin.
PEAK_PROMINENCE_MARGIN = 0.10
PROMINENCE_NEIGHBOR_HOPS = 6

Env = Tuple[int, List[float], Optional[float]]  # (hop_ms, sync_env, integrated_lufs)


@dataclass
class AlignResult:
    offset_ms: int        # b's start in a's clock; positive => b starts after a
    confidence: float     # peak correlation, 0..1
    overlap_ms: int       # length of the compared overlap at the peak
    prominence: float     # how far the peak stands above the far-field

    def to_dict(self) -> dict:
        return {
            "offset_ms": self.offset_ms,
            "confidence": round(self.confidence, 3),
            "overlap_ms": self.overlap_ms,
            "prominence": round(self.prominence, 3),
        }


def _pg_conn():
    import psycopg
    return psycopg.connect(get_settings().database_url, autocommit=True)


def load_sync_envs(file_ids: List[str]) -> Dict[str, Env]:
    """file_id -> (sync_hop_ms, sync_env, integrated_lufs). Missing/empty skipped."""
    if not file_ids:
        return {}
    out: Dict[str, Env] = {}
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


def _overlap_floor_hops(na: int, nb: int) -> int:
    """How many overlapping hops we demand before a lag is even eligible."""
    return max(MIN_OVERLAP_HOPS_FLOOR, int(MIN_OVERLAP_FRACTION * min(na, nb)))


def _best_lag(a: List[float], b: List[float]) -> Optional[dict]:
    """Hardened normalized cross-correlation of two envelopes.

    Only lags whose overlap clears the floor are eligible (so a tiny coincidental
    tail can't win). Returns the winning lag with its correlation, sub-hop
    refinement, overlap length, and prominence over the far-field -- or None if
    nothing is eligible (clips too short to be sure).
    """
    import numpy as np

    va, vb = np.asarray(a, float), np.asarray(b, float)
    na, nb = va.size, vb.size
    if na == 0 or nb == 0:
        return None

    floor = _overlap_floor_hops(na, nb)
    if floor > min(na, nb):
        return None  # the clips can't possibly overlap enough to be certain

    best_lag, best_corr, best_n = 0, -2.0, 0
    corr_at: Dict[int, float] = {}
    for lag in range(-(nb - 1), na):
        a0 = max(0, lag)
        b0 = max(0, -lag)
        n = min(na - a0, nb - b0)
        if n < floor:
            continue
        sa = va[a0:a0 + n]
        sb = vb[b0:b0 + n]
        sa = sa - sa.mean()
        sb = sb - sb.mean()
        denom = np.linalg.norm(sa) * np.linalg.norm(sb)
        c = float(np.dot(sa, sb) / denom) if denom > 0 else 0.0
        corr_at[lag] = c
        if c > best_corr:
            best_corr, best_lag, best_n = c, lag, n

    if not corr_at:
        return None

    # Prominence: the best correlation OUTSIDE a small neighborhood of the peak.
    far = [c for lg, c in corr_at.items() if abs(lg - best_lag) > PROMINENCE_NEIGHBOR_HOPS]
    far_best = max(far) if far else -1.0
    prominence = best_corr - far_best

    # Parabolic interpolation around the peak -> sub-hop fraction.
    frac = 0.0
    cm = corr_at.get(best_lag - 1)
    cp = corr_at.get(best_lag + 1)
    if cm is not None and cp is not None:
        denom = cm - 2 * best_corr + cp
        if denom != 0:
            frac = max(-0.5, min(0.5, 0.5 * (cm - cp) / denom))

    return {
        "lag": best_lag,
        "corr": best_corr,
        "frac": frac,
        "overlap_hops": best_n,
        "prominence": prominence,
    }


def align_envs(env_a: Env, env_b: Env) -> Optional[AlignResult]:
    """Align two loaded envelopes. None unless all three airtight gates pass."""
    hop_a, a, _ = env_a
    hop_b, b, _ = env_b
    if hop_a != hop_b or not a or not b:
        return None
    r = _best_lag(a, b)
    if r is None:
        return None
    corr = r["corr"]
    if corr < SYNC_CONF_THRESHOLD or r["prominence"] < PEAK_PROMINENCE_MARGIN:
        return None  # ambiguous or weak -> honest "not simultaneous"
    offset_ms = int(round((r["lag"] + r["frac"]) * hop_a))
    return AlignResult(
        offset_ms=offset_ms,
        confidence=max(0.0, min(1.0, corr)),
        overlap_ms=int(r["overlap_hops"] * hop_a),
        prominence=r["prominence"],
    )


# --------------------------------------------------------------------------
# Candidate surfacing: a CHEAP, genre-general pre-screen.
#
# Text overlap (content.py) only nominates synced pairs that share DIALOGUE, so
# action / music / mostly-silent multicam is invisible to it. This pre-screen
# nominates pairs by their AUDIO ENERGY alone (a decimated, gate-free coarse
# cross-correlation), so anything that recorded the same sound is surfaced. It
# is deliberately permissive -- it only says "these MIGHT be the same moment";
# align_clips remains the airtight confirm. Bounded for large projects by a
# top-K cap on the pairs handed to full verification.
# --------------------------------------------------------------------------

# Loose floor for nomination (well below SYNC_CONF_THRESHOLD; this only narrows
# the field for the airtight check, it does not accept anything).
CANDIDATE_CORR_FLOOR = 0.30
# Cap the samples the coarse pass correlates per envelope. Decimation only kicks
# in for pathologically long clips; striding destroys the sub-hop phase that
# carries a real offset, so normal-length clips run at full resolution.
CANDIDATE_MAX_SAMPLES = 2000
CANDIDATE_MIN_OVERLAP_HOPS = 20  # at the (possibly decimated) hop


@dataclass
class SyncCandidate:
    file_a: str
    file_b: str
    coarse: float


@dataclass
class VerifiedAngle:
    file_a: str          # the spine-side clip
    file_b: str          # the angle
    result: AlignResult


def _decim_for(*lengths: int) -> int:
    """Stride that keeps each envelope under CANDIDATE_MAX_SAMPLES (1 = none)."""
    n = max(lengths) if lengths else 0
    return max(1, -(-n // CANDIDATE_MAX_SAMPLES))  # ceil division


def _coarse_corr(a: List[float], b: List[float], decim: int = 1) -> float:
    """Best gate-free normalized correlation -- a fast screen, not a measurement.
    `decim` strides both envelopes (only for very long clips); returns 0 when the
    overlap is too small to mean anything."""
    import numpy as np

    va = np.asarray(a[::decim], float)
    vb = np.asarray(b[::decim], float)
    na, nb = va.size, vb.size
    if na < CANDIDATE_MIN_OVERLAP_HOPS or nb < CANDIDATE_MIN_OVERLAP_HOPS:
        return 0.0
    best = 0.0
    for lag in range(-(nb - 1), na):
        a0, b0 = max(0, lag), max(0, -lag)
        n = min(na - a0, nb - b0)
        if n < CANDIDATE_MIN_OVERLAP_HOPS:
            continue
        sa = va[a0:a0 + n] - va[a0:a0 + n].mean()
        sb = vb[b0:b0 + n] - vb[b0:b0 + n].mean()
        denom = np.linalg.norm(sa) * np.linalg.norm(sb)
        if denom > 0:
            best = max(best, float(np.dot(sa, sb) / denom))
    return best


def sync_candidates(file_ids: List[str], max_pairs: int = 24) -> List[SyncCandidate]:
    """Nominate pairs that MIGHT be the same moment, by audio energy alone.
    Cheap (decimated, gate-free); the result is the short list align_clips then
    confirms. Capped to `max_pairs` strongest so an n^2 scan stays bounded."""
    envs = load_sync_envs(file_ids)
    fids = [f for f in file_ids if f in envs]
    out: List[SyncCandidate] = []
    for i in range(len(fids)):
        hop_i, ei, _ = envs[fids[i]]
        for j in range(i + 1, len(fids)):
            hop_j, ej, _ = envs[fids[j]]
            if hop_i != hop_j or not ei or not ej:
                continue
            c = _coarse_corr(ei, ej, _decim_for(len(ei), len(ej)))
            if c >= CANDIDATE_CORR_FLOOR:
                out.append(SyncCandidate(fids[i], fids[j], round(c, 3)))
    out.sort(key=lambda p: p.coarse, reverse=True)
    return out[:max_pairs]


def discover_synced_angles(file_ids: List[str], max_pairs: int = 24) -> List[VerifiedAngle]:
    """Full pipeline: pre-screen candidates -> airtight-verify each with
    align_clips -> keep only confirmed synced pairs. This is what the
    orchestrator surfaces so Opus knows which clips are real second angles."""
    verified: List[VerifiedAngle] = []
    for cand in sync_candidates(file_ids, max_pairs):
        res = align_clips(cand.file_a, cand.file_b)
        if res is not None:
            verified.append(VerifiedAngle(cand.file_a, cand.file_b, res))
    verified.sort(key=lambda v: v.result.confidence, reverse=True)
    return verified


def align_clips(
    file_a: str,
    file_b: str,
    span: Optional[Tuple[int, int]] = None,
) -> Optional[AlignResult]:
    """On-demand: are `file_a` and `file_b` the same moment, and by what offset?

    Returns b's offset in a's clock (positive => b starts after a), a confidence,
    and the overlap length -- or None when they are not simultaneous, one has no
    audio envelope, or the comparable overlap is too short to be sure. `span`
    (ms in file_a) restricts the comparison to a region of a (e.g. where a cut is
    considered); the returned offset stays in file_a's absolute clock.
    """
    envs = load_sync_envs([file_a, file_b])
    ea, eb = envs.get(file_a), envs.get(file_b)
    if not ea or not eb:
        return None  # no audio envelope on one side (e.g. a silent clip)

    if span:
        hop, a_env, lufs = ea
        s0 = max(0, int(span[0]) // hop)
        s1 = min(len(a_env), max(s0 + 1, int(span[1]) // hop + 1))
        sliced = (hop, a_env[s0:s1], lufs)
        res = align_envs(sliced, eb)
        if res is None:
            return None
        # The slice's index 0 sits at s0 hops into file_a, so re-base the offset.
        return AlignResult(
            offset_ms=res.offset_ms + s0 * hop,
            confidence=res.confidence,
            overlap_ms=res.overlap_ms,
            prominence=res.prominence,
        )

    return align_envs(ea, eb)
