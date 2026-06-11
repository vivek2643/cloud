"""
Deterministic cut-engine: the mechanical half of the L3 editor.

Everything in here is pure math over the L1 cut-cost grids -- no LLM, no
opinions. The orchestrator (Opus) proposes WHAT to cut and ROUGHLY where; this
module answers EXACTLY where (snap to the cheapest nearby seam), HOW CLEAN it
is (cost), and WHETHER the assembly obeys hard constraints (duration, segment
sanity). Because the two halves fail in opposite ways, the contract is strict:
the LLM never places frames, the engine never chooses material.

Grid model (from L1):
  * Dense per-hop cost arrays (0 = ideal seam .. 1 = forbidden), one per
    channel: dialogue / beat (audio_features) and action / camera
    (motion_dynamics). Hops are typically 100 ms.
  * Sparse "snap points" per channel: discrete, exact-timestamp seam
    candidates (word gaps, beats, action impacts) with their own scores.

The combined cost of cutting at time t is a per-axis weighted blend: e.g. for
a speech-driven clip the dialogue channel dominates but a chaotic camera
moment still penalizes the seam.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.config import get_settings

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tunables (same spirit as cut_grid_params: keep the knobs in one place)
# --------------------------------------------------------------------------

# How far snap() may move a proposed cut to find a clean seam.
DEFAULT_SNAP_WINDOW_MS = 700
# A seam costing more than this is reported but flagged as dirty.
DIRTY_SEAM_COST = 0.45
# Segments shorter than this are almost never usable on a timeline.
MIN_SEGMENT_MS = 400
# Adjacent segments from the same file separated by less than this read as a
# jump cut (same framing, tiny time skip).
JUMP_CUT_GAP_MS = 2000
# fit_duration may move a trim point this far from the arithmetic ideal to
# land on a clean seam (wider than the normal snap window: when shrinking,
# trimming slightly more/less than asked is much better than a dirty cut).
FIT_SNAP_WINDOW_MS = 1500

# Per-axis channel weights for the combined seam cost. `max` semantics: the
# weighted-worst channel decides, so one forbidden channel can't be averaged
# away by three idle ones.
AXIS_WEIGHTS: Dict[str, Dict[str, float]] = {
    "speech":  {"dialogue": 1.0, "camera": 0.5, "action": 0.25},
    "action":  {"action": 1.0, "camera": 0.6, "dialogue": 0.5},
    "music":   {"beat": 1.0, "camera": 0.4, "dialogue": 0.3},
    "visual":  {"camera": 1.0, "action": 0.4},
    # Safe default when the axis is unknown: respect everything.
    "any":     {"dialogue": 1.0, "beat": 0.5, "action": 0.7, "camera": 0.7},
}


# --------------------------------------------------------------------------
# Grid container + loading
# --------------------------------------------------------------------------

@dataclass
class ClipGrids:
    file_id: str
    duration_ms: int
    # channel -> (hop_ms, dense cost array). Missing channel = no signal.
    channels: Dict[str, Tuple[int, List[float]]] = field(default_factory=dict)
    # channel -> sparse snap points [{ts_ms, kind, score, ...}]
    points: Dict[str, List[dict]] = field(default_factory=dict)

    def cost_at(self, ms: int, channel: str) -> Optional[float]:
        got = self.channels.get(channel)
        if not got:
            return None
        hop_ms, arr = got
        if hop_ms <= 0 or not arr:
            return None
        idx = min(max(ms // hop_ms, 0), len(arr) - 1)
        return float(arr[idx])

    def combined_cost(self, ms: int, axis: str) -> float:
        """Weighted-worst-channel cost at ms. Channels without data are skipped;
        a clip with no grids at all costs 0 everywhere (nothing to violate)."""
        weights = AXIS_WEIGHTS.get(axis) or AXIS_WEIGHTS["any"]
        worst = 0.0
        for channel, w in weights.items():
            c = self.cost_at(ms, channel)
            if c is not None:
                worst = max(worst, w * c)
        return round(worst, 4)


def _pg_conn():
    import psycopg
    settings = get_settings()
    return psycopg.connect(settings.database_url, autocommit=True)


def _as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return json.loads(v)


def load_grids(file_id: str) -> ClipGrids:
    """Pull every cut-grid channel for one file out of the DB."""
    with _pg_conn() as conn:
        frow = conn.execute(
            "select coalesce(duration_seconds, 0) from files where id = %s",
            (file_id,),
        ).fetchone()
        duration_ms = int(float(frow[0]) * 1000) if frow else 0

        grids = ClipGrids(file_id=file_id, duration_ms=duration_ms)

        arow = conn.execute(
            """
            select dialogue_cut_cost, dialogue_cut_hop_ms, dialogue_cut_points,
                   beat_cut_cost, beat_cut_hop_ms, beat_cut_points
              from audio_features where file_id = %s
            """,
            (file_id,),
        ).fetchone()
        if arow:
            d_cost, d_hop, d_pts, b_cost, b_hop, b_pts = arow
            d_cost, b_cost = _as_list(d_cost), _as_list(b_cost)
            if d_cost and d_hop:
                grids.channels["dialogue"] = (int(d_hop), [float(x) for x in d_cost])
                grids.points["dialogue"] = _as_list(d_pts)
            if b_cost and b_hop:
                grids.channels["beat"] = (int(b_hop), [float(x) for x in b_cost])
                grids.points["beat"] = _as_list(b_pts)

        mrow = conn.execute(
            """
            select hop_ms, action_cut_cost, camera_cut_cost, action_points
              from motion_dynamics where file_id = %s
            """,
            (file_id,),
        ).fetchone()
        if mrow:
            hop, a_cost, c_cost, a_pts = mrow
            a_cost, c_cost = _as_list(a_cost), _as_list(c_cost)
            if a_cost and hop:
                grids.channels["action"] = (int(hop), [float(x) for x in a_cost])
                grids.points["action"] = _as_list(a_pts)
            if c_cost and hop:
                grids.channels["camera"] = (int(hop), [float(x) for x in c_cost])

    return grids


# --------------------------------------------------------------------------
# Seam queries + snapping
# --------------------------------------------------------------------------

def query_seams(
    grids: ClipGrids,
    around_ms: int,
    axis: str = "any",
    window_ms: int = 2000,
    limit: int = 8,
) -> List[dict]:
    """Ranked clean cut candidates near `around_ms`.

    Merges two sources: (a) the discrete snap points L1 emitted (word gaps,
    beats, impacts) and (b) local minima of the combined dense grid -- then
    scores each by combined cost plus a mild distance penalty so a marginally
    cleaner seam far away doesn't beat a clean seam right here.
    """
    lo = max(0, around_ms - window_ms)
    hi = min(grids.duration_ms or (around_ms + window_ms), around_ms + window_ms)
    candidates: Dict[int, dict] = {}

    # (a) discrete snap points in window
    for channel, pts in grids.points.items():
        for p in pts:
            ts = int(p.get("ts_ms", -1))
            if ts < lo or ts > hi:
                continue
            cost = grids.combined_cost(ts, axis)
            prev = candidates.get(ts)
            if prev is None or cost < prev["cost"]:
                candidates[ts] = {
                    "ts_ms": ts, "cost": cost,
                    "source": f"{channel}:{p.get('kind', 'point')}",
                }

    # (b) local minima of the dense combined grid (coarsest hop available)
    hops = [hop for hop, _ in grids.channels.values()]
    hop = min(hops) if hops else 100
    series = [(t, grids.combined_cost(t, axis)) for t in range(lo, hi + 1, hop)]
    for i in range(1, len(series) - 1):
        t, c = series[i]
        if c <= series[i - 1][1] and c <= series[i + 1][1]:
            if t not in candidates or c < candidates[t]["cost"]:
                candidates[t] = {"ts_ms": t, "cost": c, "source": "grid:minimum"}

    ranked = sorted(
        candidates.values(),
        key=lambda s: s["cost"] + 0.10 * (abs(s["ts_ms"] - around_ms) / max(window_ms, 1)),
    )
    for s in ranked:
        s["dirty"] = s["cost"] > DIRTY_SEAM_COST
        s["distance_ms"] = s["ts_ms"] - around_ms
    return ranked[:limit]


def snap_cut(
    grids: ClipGrids,
    proposed_ms: int,
    axis: str = "any",
    window_ms: int = DEFAULT_SNAP_WINDOW_MS,
) -> dict:
    """Move one proposed cut to the best nearby seam. Always returns a result;
    if nothing clean exists in the window, returns the best available with
    dirty=True + a warning so the orchestrator can decide to look elsewhere."""
    proposed_ms = min(max(0, proposed_ms), grids.duration_ms or proposed_ms)
    seams = query_seams(grids, proposed_ms, axis, window_ms, limit=1)
    if not seams:
        cost = grids.combined_cost(proposed_ms, axis)
        return {
            "ts_ms": proposed_ms, "cost": cost, "source": "unsnapped",
            "dirty": cost > DIRTY_SEAM_COST, "distance_ms": 0,
            "warning": "no seam candidates in window; cut left where proposed",
        }
    best = seams[0]
    if best["dirty"]:
        best["warning"] = (
            f"cleanest seam near {proposed_ms}ms still costs {best['cost']:.2f}; "
            "consider a different moment"
        )
    return best


# --------------------------------------------------------------------------
# Timeline operations (pure functions over the document's timeline array)
# --------------------------------------------------------------------------
#
# A segment is a plain dict (it lives inside the Edit Document JSON):
#   { seg_id, file_id, in_ms, out_ms, axis, beat_id?, content?, rationale?,
#     priority? (1=most protected), cut_in_cost, cut_out_cost, warnings: [] }

def make_segment(
    grids: ClipGrids,
    in_ms: int,
    out_ms: int,
    axis: str = "any",
    *,
    beat_id: Optional[str] = None,
    content: Optional[str] = None,
    rationale: Optional[str] = None,
    priority: int = 3,
    snap_window_ms: int = DEFAULT_SNAP_WINDOW_MS,
) -> dict:
    """Snap both ends of a proposed span and package it as a timeline segment."""
    snap_in = snap_cut(grids, in_ms, axis, snap_window_ms)
    snap_out = snap_cut(grids, out_ms, axis, snap_window_ms)
    warnings = [w for w in (snap_in.get("warning"), snap_out.get("warning")) if w]

    s_in, s_out = snap_in["ts_ms"], snap_out["ts_ms"]
    if s_out - s_in < MIN_SEGMENT_MS:
        warnings.append(
            f"segment is only {s_out - s_in}ms after snapping (< {MIN_SEGMENT_MS}ms)"
        )

    return {
        "seg_id": f"s{uuid.uuid4().hex[:6]}",
        "file_id": grids.file_id,
        "in_ms": s_in,
        "out_ms": s_out,
        "axis": axis,
        "beat_id": beat_id,
        "content": content,
        "rationale": rationale,
        "priority": priority,
        "cut_in_cost": snap_in["cost"],
        "cut_out_cost": snap_out["cost"],
        "warnings": warnings,
    }


def timeline_status(timeline: List[dict]) -> dict:
    """Objective health report: totals, per-segment durations, assembly cost,
    and deterministic continuity flags."""
    total_ms = 0
    seam_costs: List[float] = []
    warnings: List[str] = []

    for i, seg in enumerate(timeline):
        dur = seg["out_ms"] - seg["in_ms"]
        total_ms += dur
        seam_costs.extend([seg.get("cut_in_cost", 0.0), seg.get("cut_out_cost", 0.0)])
        if dur < MIN_SEGMENT_MS:
            warnings.append(f"{seg['seg_id']}: very short ({dur}ms)")
        for w in seg.get("warnings") or []:
            warnings.append(f"{seg['seg_id']}: {w}")
        if i > 0:
            prev = timeline[i - 1]
            if prev["file_id"] == seg["file_id"]:
                gap = seg["in_ms"] - prev["out_ms"]
                if 0 <= gap < JUMP_CUT_GAP_MS:
                    warnings.append(
                        f"{prev['seg_id']}->{seg['seg_id']}: likely jump cut "
                        f"(same clip, {gap}ms skipped)"
                    )

    return {
        "segment_count": len(timeline),
        "total_ms": total_ms,
        "total_s": round(total_ms / 1000.0, 2),
        "segments": [
            {
                "seg_id": s["seg_id"], "beat_id": s.get("beat_id"),
                "file_id": s["file_id"],
                "in_ms": s["in_ms"], "out_ms": s["out_ms"],
                "duration_ms": s["out_ms"] - s["in_ms"],
                "cut_costs": [s.get("cut_in_cost"), s.get("cut_out_cost")],
            }
            for s in timeline
        ],
        "mean_seam_cost": round(sum(seam_costs) / len(seam_costs), 4) if seam_costs else 0.0,
        "max_seam_cost": round(max(seam_costs), 4) if seam_costs else 0.0,
        "warnings": warnings,
    }


def fit_duration(
    timeline: List[dict],
    grids_by_file: Dict[str, ClipGrids],
    target_ms: int,
    tolerance_ms: int = 500,
) -> Tuple[List[dict], dict]:
    """Deterministically trim an over-long timeline down to target.

    Strategy (v1, greedy and predictable):
      * Only shrinks; if the cut is already under target it is returned as-is
        (choosing what to ADD is creative work -- the orchestrator's job).
      * Repeatedly takes the lowest-priority (highest number), longest segment
        and pulls its out-point back to the best seam near the needed position.
      * Protected ends: never trims a segment below MIN_SEGMENT_MS * 2.

    Returns (new_timeline, report).
    """
    timeline = [dict(s) for s in timeline]  # never mutate caller's copy
    report = {"moves": [], "fitted": False, "residual_ms": 0}

    def total() -> int:
        return sum(s["out_ms"] - s["in_ms"] for s in timeline)

    overshoot = total() - target_ms
    if overshoot <= tolerance_ms:
        report["fitted"] = True
        report["residual_ms"] = max(0, overshoot)
        return timeline, report

    # Most-trimmable first: low priority, then long duration.
    for _ in range(64):  # hard bound; each pass trims one segment
        overshoot = total() - target_ms
        if overshoot <= tolerance_ms:
            break
        order = sorted(
            timeline,
            key=lambda s: (-s.get("priority", 3), -(s["out_ms"] - s["in_ms"])),
        )
        moved = False
        for seg in order:
            dur = seg["out_ms"] - seg["in_ms"]
            slack = dur - MIN_SEGMENT_MS * 2
            if slack <= 0:
                continue
            want = min(slack, overshoot)
            grids = grids_by_file.get(seg["file_id"])
            desired_out = seg["out_ms"] - want
            lo_legal = seg["in_ms"] + MIN_SEGMENT_MS * 2
            hi_legal = seg["out_ms"] - 1
            if grids is not None:
                # Choose the cheapest *legal* seam near the ideal trim point
                # (snap_cut alone may return a seam outside the legal band or a
                # dirty one when a clean seam sits slightly farther away).
                axis = seg.get("axis", "any")
                cands = query_seams(grids, desired_out, axis,
                                    window_ms=FIT_SNAP_WINDOW_MS, limit=8)
                legal = [s for s in cands if lo_legal <= s["ts_ms"] <= hi_legal]
                if legal:
                    best = min(legal, key=lambda s: s["cost"])
                    new_out = best["ts_ms"]
                else:
                    new_out = max(lo_legal, min(desired_out, hi_legal))
                cost = grids.combined_cost(new_out, axis)
            else:
                new_out = max(lo_legal, min(desired_out, hi_legal))
                cost = None
            if new_out >= seg["out_ms"]:
                continue
            report["moves"].append({
                "seg_id": seg["seg_id"],
                "old_out_ms": seg["out_ms"],
                "new_out_ms": new_out,
                "trimmed_ms": seg["out_ms"] - new_out,
                "new_cut_cost": cost,
            })
            seg["out_ms"] = new_out
            if cost is not None:
                seg["cut_out_cost"] = cost
            moved = True
            break
        if not moved:
            break  # nothing left that can legally shrink

    residual = total() - target_ms
    report["fitted"] = residual <= tolerance_ms
    report["residual_ms"] = max(0, residual)
    if not report["fitted"]:
        report["note"] = (
            "could not reach target by trimming alone; consider dropping a "
            "whole segment (orchestrator decision)"
        )
    return timeline, report
