"""
L1 derived signal: the FUSED SEAM FIELD -- one cut-cost curve that composes
every per-channel grid into a single "where is it safe AND good to cut?" answer.

This is the core cut-placement primitive. Speech, action, visual and combined
heroes all snap their boundaries through this one field, so a cut can never
land somewhere one modality forbids (e.g. an action cut inside a spoken word).

Why not just add the channels?
------------------------------
The channels are two DIFFERENT kinds of signal and must be composed, not summed:

  * VETOES (constraints) -- "you must NOT cut here":
        dialogue_cut_cost   1 = mid-word (forbidden), 0 = clean gap
        camera_cut_cost     1 = whip/shake/blur, 0 = calm/coherent
    A single veto near 1 should kill the seam regardless of the others, so we
    multiply their safeties (Noisy-AND): a word boundary that is also blurry is
    still unsafe.

  * ATTRACTORS (rewards) -- "this is an ESPECIALLY good place to cut, but
    elsewhere is not bad":
        action_cut_cost     0 = on a motion impact, 1 = mid-motion
        beat_cost           0 = on the beat, 1 = off-beat
    These sit near 1 almost everywhere (only their few hits dip to 0), so adding
    them as cost would penalize clean silence. They may only REWARD at their
    hits, never penalize between them.

Composition (in safety space, safe = 1 - cost):

    S(t) = (1 - dialogue)·(1 - camera)·protect          # veto product
    r(t) = max(1 - action, 1 - beat)                    # best attractor, 0..1
    Q(t) = clamp01( S(t) · (1 + lambda·r(t)) )          # seam quality 0..1
    fused_cost(t) = 1 - Q(t)

`lambda` (attractor weight) is driven by the energy knob: calm edits cut on
clean seams; punchy edits snap hard onto impacts/beats. A missing channel is
neutral (veto safety 1 / attractor reward 0), so the field degrades gracefully
on silent / motionless / non-musical clips.

Pure-Python lists in/out (no numpy); all channels are assumed to share the
100 ms hop the L1 grids already use, and are resampled if not.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from app.services.l1.cut_grid_common import clamp01, n_hops

# Attractor weight as a function of energy (0 = calm .. 1 = punchy). At r=1 the
# boost factor is (1 + lambda), so a beat/impact in an otherwise-safe spot is
# pulled to the front of the candidate list -- gently at low energy, hard at high.
LAMBDA_MIN = 0.25
LAMBDA_MAX = 1.25

# Default boundary search half-windows (ms); they tighten as energy rises so
# punchy cuts hug the rough nomination instead of wandering to a far seam.
SNAP_BASE_WIN_MS = 1200
SNAP_TIGHT_WIN_MS = 400

# Two candidate seams closer than this collapse to one (keep the better Q).
SEAM_MERGE_MS = 60

# A candidate whose FUSED quality is below this is a vetoed point (e.g. a beat /
# impact / speaker mark that lands mid-word), not a usable cut -- drop it from the
# seam list. The dense grid still covers that region for snapping, so this only
# affects which discrete seams we advertise, never where we can cut.
SEAM_Q_FLOOR = 0.25

# An attractor (action impact / beat) may only BOOST a cut that has this much
# clean "room" around it -- i.e. the veto safety stays high across +/- this
# window. This stops a motion impact that lands in a ~140 ms breath between two
# words from scoring ~1 (a confident cut mid-sentence). It does NOT touch the
# dialogue veto, so genuine sentence-end / pause seams are unaffected; it only
# withholds the attractor bonus where there isn't space to cut cleanly.
ATTRACTOR_ROOM_MS = 250

GRID_HOP_MS = 100


@dataclass
class FusedSeam:
    ts_ms: int
    q: float                 # seam quality 0..1 (higher = cleaner/better cut)
    kind: str                # word_gap | sentence_end | speaker_change | beat | action_impact | rest
    sources: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"ts_ms": self.ts_ms, "q": round(self.q, 3),
                "kind": self.kind, "sources": self.sources}


@dataclass
class FusedField:
    hop_ms: int
    cost: List[float]                       # dense fused cost (1 - Q), 0..1
    seams: List[FusedSeam] = field(default_factory=list)  # discrete, best-first

    def q_at(self, ts_ms: int) -> float:
        if not self.cost:
            return 0.0
        j = max(0, min(len(self.cost) - 1, int(round(ts_ms / self.hop_ms))))
        return 1.0 - self.cost[j]


# --------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------

def _resample(cost: Optional[Sequence[float]], src_hop: int, n: int,
              hop_ms: int, fill: float) -> List[float]:
    """Map a channel onto the common (hop_ms, n) grid; `fill` is used where the
    channel is absent or out of range (its neutral value)."""
    if not cost:
        return [fill] * n
    src_hop = src_hop or hop_ms
    out: List[float] = []
    for i in range(n):
        j = int(round((i * hop_ms) / src_hop))
        out.append(float(cost[j]) if 0 <= j < len(cost) else fill)
    return out


def _erode(values: List[float], w: int) -> List[float]:
    """Min-filter (grayscale erosion): out[i] = min(values[i-w .. i+w]). Used to
    require an attractor to sit in a SUSTAINED safe window, not a 1-hop notch."""
    n = len(values)
    if n == 0 or w <= 0:
        return list(values)
    out = [0.0] * n
    for i in range(n):
        lo, hi = max(0, i - w), min(n - 1, i + w)
        out[i] = min(values[lo:hi + 1])
    return out


def _protect_mask(spans: Optional[Sequence[Tuple[int, int]]], n: int, hop_ms: int) -> List[float]:
    """1.0 everywhere except 0.0 inside protected spans (VLM reveal/peak/hold) --
    a hard veto layered on top of the signal vetoes."""
    mask = [1.0] * n
    for s in spans or []:
        a, b = int(s[0]), int(s[1])
        for i in range(max(0, a // hop_ms), min(n - 1, b // hop_ms) + 1):
            mask[i] = 0.0
    return mask


def _point_ts(points: Optional[Sequence]) -> List[Tuple[int, str]]:
    """Pull (ts_ms, kind) out of a channel's discrete points list."""
    out: List[Tuple[int, str]] = []
    for p in points or []:
        if isinstance(p, dict) and p.get("ts_ms") is not None:
            out.append((int(p["ts_ms"]), str(p.get("kind", "seam"))))
    return out


def _build_seams(cost: List[float], hop_ms: int,
                 candidates: List[Tuple[int, str, str]]) -> List[FusedSeam]:
    """Re-score every channel's discrete candidate by the FUSED quality at its
    instant, merge near-duplicates (keeping the better one, unioning sources),
    and return them best-first. Candidates are (ts_ms, kind, source)."""
    n = len(cost)
    scored: List[FusedSeam] = []
    for ts, kind, src in candidates:
        if ts < 0:
            continue
        j = max(0, min(n - 1, int(round(ts / hop_ms))))
        scored.append(FusedSeam(ts_ms=ts, q=1.0 - cost[j], kind=kind, sources=[src]))
    scored.sort(key=lambda s: (s.ts_ms, -s.q))

    merged: List[FusedSeam] = []
    for s in scored:
        if merged and s.ts_ms - merged[-1].ts_ms <= SEAM_MERGE_MS:
            keep = merged[-1]
            for src in s.sources:
                if src not in keep.sources:
                    keep.sources.append(src)
            if s.q > keep.q:
                keep.q, keep.ts_ms, keep.kind = s.q, s.ts_ms, s.kind
        else:
            merged.append(s)
    merged = [s for s in merged if s.q >= SEAM_Q_FLOOR]  # drop vetoed candidates
    merged.sort(key=lambda s: -s.q)
    return merged


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def compute_fused_field(
    *,
    duration_ms: int,
    energy: float = 0.5,
    hop_ms: int = GRID_HOP_MS,
    dialogue_cost: Optional[Sequence[float]] = None,
    dialogue_hop: int = GRID_HOP_MS,
    camera_cost: Optional[Sequence[float]] = None,
    action_cost: Optional[Sequence[float]] = None,
    motion_hop: int = GRID_HOP_MS,
    beat_cost: Optional[Sequence[float]] = None,
    beat_hop: int = GRID_HOP_MS,
    protected_spans: Optional[Sequence[Tuple[int, int]]] = None,
    dialogue_points: Optional[Sequence] = None,
    beat_points: Optional[Sequence] = None,
    action_points: Optional[Sequence] = None,
    attractor_room_ms: int = ATTRACTOR_ROOM_MS,
) -> FusedField:
    """Compose the per-channel grids into one fused seam field. See module docstring."""
    n = n_hops(duration_ms, hop_ms)
    if n == 0:
        return FusedField(hop_ms=hop_ms, cost=[], seams=[])

    # Missing-channel neutrals: vetoes default safe (cost 0); attractors default
    # no-reward (cost 1).
    d = _resample(dialogue_cost, dialogue_hop, n, hop_ms, 0.0)
    cam = _resample(camera_cost, motion_hop, n, hop_ms, 0.0)
    act = _resample(action_cost, motion_hop, n, hop_ms, 1.0)
    beat = _resample(beat_cost, beat_hop, n, hop_ms, 1.0)
    protect = _protect_mask(protected_spans, n, hop_ms)

    lam = LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * clamp01(energy)

    # "Room" gate: eroded safety so an attractor only earns its bonus where the
    # safe window is sustained (not a mid-sentence breath between two words).
    safety = [(1.0 - d[i]) * (1.0 - cam[i]) * protect[i] for i in range(n)]
    room = _erode(safety, max(1, round(attractor_room_ms / hop_ms)))

    cost: List[float] = []
    for i in range(n):
        reward = max(1.0 - act[i], 1.0 - beat[i]) * room[i]
        q = clamp01(safety[i] * (1.0 + lam * reward))
        cost.append(round(1.0 - q, 4))

    candidates: List[Tuple[int, str, str]] = []
    for ts, kind in _point_ts(dialogue_points):
        candidates.append((ts, kind, "dialogue"))
    for ts, kind in _point_ts(beat_points):
        candidates.append((ts, "beat", "beat"))
    for ts, kind in _point_ts(action_points):
        candidates.append((ts, "action_impact", "action"))

    return FusedField(hop_ms=hop_ms, cost=cost,
                      seams=_build_seams(cost, hop_ms, candidates))


def snap_point(field: FusedField, rough_ms: int, lo_ms: int, hi_ms: int) -> int:
    """Best instant to cut within [lo,hi]: maximize fused quality, breaking ties
    toward the rough nomination. Considers both exact channel seams and the hop
    grid, so we land on a sub-hop word/beat/impact edge when one is nearby."""
    n = len(field.cost)
    if n == 0 or hi_ms <= lo_ms:
        return rough_ms
    hop = field.hop_ms
    cands = {s.ts_ms for s in field.seams if lo_ms <= s.ts_ms <= hi_ms}
    i0, i1 = max(0, int(lo_ms // hop)), min(n - 1, int(hi_ms // hop))
    cands.update(i * hop for i in range(i0, i1 + 1))
    if not cands:
        return rough_ms
    return max(cands, key=lambda t: (round(field.q_at(t), 3), -abs(t - rough_ms)))


def snap_bounds(
    field: FusedField, raw_in: int, raw_out: int, *,
    energy: float = 0.5, duration_ms: int = 0,
    base_win_ms: int = SNAP_BASE_WIN_MS, tight_win_ms: int = SNAP_TIGHT_WIN_MS,
) -> Tuple[int, int]:
    """Snap a rough [in,out] nomination to the best fused seams. The in-point may
    pull a little earlier (pre-roll), the out-point a little later (post-roll);
    the window tightens as energy rises."""
    win = int(base_win_ms - (base_win_ms - tight_win_ms) * clamp01(energy))
    in_ms = snap_point(field, raw_in, max(0, raw_in - win), raw_in + win // 2)
    hi = min(duration_ms, raw_out + win) if duration_ms else raw_out + win
    out_ms = snap_point(field, raw_out, max(in_ms + 1, raw_out - win // 2), hi)
    if out_ms <= in_ms:
        out_ms = max(raw_out, in_ms + 1)
    return in_ms, out_ms
