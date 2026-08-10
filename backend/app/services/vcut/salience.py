"""
brain_cut_salience_parity.plan.md -- the salience bridge: synthesize the old
L3/v4 pipeline's `salience`/`landmarks` cut_records columns from a vcut
ResolvedCut's own `moments` (one entry per absorbed MomentFlag), at STORE
time (vcut/store.build_cut_records). Populating these two fields lights up
cutrecord_map.py's EXISTING, generic read-side ladder machinery (content-
aware fragmentation, shape-driven directional trim, the `peak:`/`sig:` tags)
with zero new brain code -- see the plan's TL;DR.

Pure, deterministic, no I/O -- mirrors the geometric/statistical spirit of
l3.post._salience/_landmarks (a code-owned number, never an LLM one) without
importing l3.post itself (vcut's isolation contract: only the v4_segment_
params floors are reused here, mirroring how cutrecord_map.py itself imports
them; l3.post's private helpers are small enough to duplicate locally rather
than reach across the boundary -- store.py's own _energy_grade sets this
precedent).

NO SILENT FALLBACKS (plan section 9): a malformed ResolvedCut.moments entry
(missing peak_ms/shape) raises ValueError rather than writing a dark cut --
resolve.py always constructs this field with a fixed shape, so a validation
failure here means a real caller bug, not a shape of input worth tolerating.
"""
from __future__ import annotations

from typing import Any, Dict, List

from app.services.l3.landmarks import (
    _LANDMARK_ACT_CAP,
    _adx_changes,
    _interior_edge_guard,
    _shot_cuts,
    _sil_gaps,
    series_lohi,
)
from app.services.l3.v4_segment_params import (
    FOLLOW_THROUGH_FLOOR_MS as _FOLLOW_THROUGH_FLOOR_MS,
    RUN_UP_FLOOR_MS as _RUN_UP_FLOOR_MS,
)
from app.services.vcut.resolve import ResolvedCut

# section 3.4: vcut's shape vocabulary -> the ladder's directional-trim
# vocabulary. Confirmed both ends agree (plan section 3.4): vcut's own
# TAG_SPLIT (params.py) keeps MORE run-up for "build" (matches "before"'s
# shrinking lead-in) and more follow-through for "settle" (matches "after"'s
# tail-absorbs-the-shrink). `.get(shape, "center")` is the fallback for any
# unrecognized/legacy value.
_SHAPE_TO_LADDER = {"build": "before", "settle": "after", "both": "center"}

def _event_for(moment: Dict[str, Any], cut_in_ms: int, cut_out_ms: int) -> Dict[str, Any]:
    """One point-event dict (l3.v4_segment._point_event's own contract:
    peak_ms/score/kind/onset_ms/settle_ms/span_ms) from a single absorbed
    MomentFlag. `score` is left at 0.0 here -- _score_events fills it in
    once every event in the cut is known (min-max needs the whole set).
    section 4 step 1: also rides the moment's own summary/specifics onto
    the event -- additive keys the geometry readers (resolve_cluster/
    _prune_events) ignore, but footage_map's per-rung moments recompose
    (section 4 step 2) reads them."""
    if "peak_ms" not in moment or "shape" not in moment:
        raise ValueError(f"malformed ResolvedCut.moments entry (missing peak_ms/shape): {moment!r}")
    peak_ms = int(moment["peak_ms"])
    onset_ms = max(cut_in_ms, peak_ms - _RUN_UP_FLOOR_MS)
    settle_ms = min(cut_out_ms, peak_ms + _FOLLOW_THROUGH_FLOOR_MS)
    return {
        "peak_ms": peak_ms, "score": 0.0, "kind": "point",
        "onset_ms": onset_ms, "settle_ms": settle_ms, "span_ms": None,
        "summary": moment.get("summary") or "",
        "specifics": dict(moment.get("specifics") or {}),
    }


def _score_events(events: List[Dict[str, Any]], hop_ms: int, action_energy: List[float]) -> None:
    """In place: each event's score = its peak's action_energy height,
    min-max normalized across THIS CUT's own events so the strongest scores
    ~1.0 -- a within-cut relative magnitude, never a cross-cut absolute
    (section 3.5; mirrors l3.post._salience's own philosophy). Uses the same
    peak-ms -> hop-bin math as resolve._refine_peak_ms for consistency. No
    usable signal at all (empty action_energy/hop_ms) -> every event scores
    1.0 (equal salience) -- the one benign, documented degrade (mirrors old
    `no_signal` behavior): _prune_events then keeps everything and
    fragmentation still works off geometry alone, rather than fabricating a
    hierarchy from nothing."""
    if not events:
        return
    if not action_energy or hop_ms <= 0:
        for e in events:
            e["score"] = 1.0
        return
    last_i = len(action_energy) - 1
    raw = [action_energy[max(0, min(last_i, int(e["peak_ms"]) // hop_ms))] for e in events]
    lo, hi = min(raw), max(raw)
    for e, r in zip(events, raw):
        e["score"] = round((r - lo) / (hi - lo), 3) if hi > lo else 1.0


def build_salience(
    cut: ResolvedCut, hop_ms: int, action_energy: List[float], mean_ae: float,
) -> Dict[str, Any]:
    """section 3.7: the full salience block for one video cut -- one point
    event per absorbed moment flag, scored cut-relative, plus the
    representative flag's fields broadcast to the top level (kind/shape/
    peak_ms/score/primary), matching l3.v4_segment._build_salience's own
    "primary event broadcast to top level" contract so every existing
    single-anchor reader (footage_map._peak_tag, cutrecord_map._video_rung's
    non-cluster branch) keeps working unchanged. Raises ValueError if
    ``cut`` carries no moments (resolve.py's own invariant guarantees every
    resolved cut has at least one -- an empty list here means a caller
    bug, not a legitimate "no signal" cut) or if no event's peak_ms matches
    cut.peak_ms (the representative flag resolve.py already picked)."""
    if not cut.moments:
        raise ValueError(f"build_salience: cut {cut.file_id}@{cut.in_ms} has no moments")
    events = [_event_for(mo, cut.in_ms, cut.out_ms) for mo in cut.moments]
    _score_events(events, hop_ms, action_energy)

    matches = [i for i, e in enumerate(events) if e["peak_ms"] == cut.peak_ms]
    if not matches:
        raise ValueError(
            f"build_salience: no event matches cut.peak_ms={cut.peak_ms} for cut {cut.file_id}")
    # section 3.4/11: a degenerate tie (two flags refined onto the same ms)
    # breaks deterministically toward the higher-scoring event -- never a
    # silent index-0 guess.
    primary = matches[0] if len(matches) == 1 else max(matches, key=lambda i: events[i]["score"])
    prim = events[primary]

    return {
        "peak_ms": prim["peak_ms"], "score": prim["score"], "kind": "point",
        "shape": _SHAPE_TO_LADDER.get(cut.tag, "center"), "span_ms": None,
        "events": events, "primary": primary,
        "density": max(0.0, min(1.0, mean_ae)),
    }


def _act_from_moments(cut: ResolvedCut, span_ms: int, edge_guard: int) -> Dict[str, Any]:
    """The ``act`` channel, derived from vcut's OWN interior action structure
    -- the included moment flags -- rather than from action_energy local
    maxima the way l3.landmarks._act_hits does for the old pipeline. This is
    the ONE channel where vcut has a richer, more semantic signal than the
    raw energy curve (each moment is a model-judged beat), so it is kept
    moment-driven (plan section 3.8). Shape matches l3.landmarks._act_hits'
    own {"n", "hits"} contract exactly: a plain list of ms offsets from the
    cut's own start, capped at _LANDMARK_ACT_CAP, time-ordered; "n" is the
    FULL interior count (may exceed len(hits)). {} sub-result (channel
    omitted) when no moment's peak is interior."""
    offsets = sorted(
        off for off in (int(mo["peak_ms"]) - cut.in_ms for mo in cut.moments)
        if edge_guard <= off <= span_ms - edge_guard
    )
    if not offsets:
        return {}
    return {"act": {"n": len(offsets), "hits": offsets[:_LANDMARK_ACT_CAP]}}


def build_landmarks(cut: ResolvedCut, seam_entry: Dict[str, Any]) -> Dict[str, Any]:
    """brain_cut_salience_parity.plan.md full-landmark-parity extension: all
    FOUR interior-structure channels for one vcut cut over its own
    [in_ms, out_ms] span, matching the exact per-channel shapes
    l3.post/footage_map._landmarks_tag expect (verified against post.py):

    - ``act`` = {"n", "hits":[off_ms]} -- from this cut's moment flags
      (vcut's own action structure; see _act_from_moments).
    - ``adx`` = {"n", "changes":[{"off","dir"}]} -- audio-dynamics change
      points, from the seam entry's persisted rms_db (l3.landmarks._adx_changes).
    - ``sil`` = {"n", "gaps":[{"off","dur"}]} -- silence gaps, from the seam
      entry's persisted silence_intervals (l3.landmarks._sil_gaps).
    - ``shot`` = {"n", "cuts":[{"off","hard"}]} -- internal shot/composition
      cuts, from the seam entry's persisted shot_points/composition_points
      (l3.landmarks._shot_cuts).

    adx/sil/shot reuse the SHARED l3.landmarks channel builders (one source
    of truth with the old pipeline -- no copy-paste drift), fed the L1
    signals orchestrate._add_landmark_signals_to_seam_cache persisted onto
    the seam entry. A signal genuinely absent for THIS file (silent footage
    -> no rms_db/silence_intervals; no detected shot cuts -> no
    shot_points) makes that channel's builder return None and the channel is
    omitted -- an honest per-file absence identical to how the old pipeline
    treated empty inputs, NOT a fabricated zero. A channel whose signal IS
    present but carries no interior structure is likewise correctly absent.
    {} overall when no channel has interior structure (or the span is
    degenerate/zero-length)."""
    span_ms = cut.out_ms - cut.in_ms
    if span_ms <= 0:
        return {}
    edge_guard = _interior_edge_guard(span_ms)
    out: Dict[str, Any] = {}

    if cut.moments:
        out.update(_act_from_moments(cut, span_ms, edge_guard))

    seam_entry = seam_entry or {}
    rms_db = seam_entry.get("rms_db") or []
    rms_hop_ms = int(seam_entry.get("rms_hop_ms") or 0)
    silence_intervals = seam_entry.get("silence_intervals") or []
    shot_points = seam_entry.get("shot_points") or []
    composition_points = seam_entry.get("composition_points") or []

    rms_lo, rms_hi = series_lohi(rms_db)
    adx = _adx_changes(rms_db, rms_hop_ms, rms_lo, rms_hi, cut.in_ms, cut.out_ms, edge_guard)
    if adx:
        out["adx"] = adx
    sil = _sil_gaps(silence_intervals, rms_hop_ms, cut.in_ms, cut.out_ms, edge_guard)
    if sil:
        out["sil"] = sil
    shot = _shot_cuts(shot_points, composition_points, cut.in_ms, cut.out_ms, edge_guard)
    if shot:
        out["shot"] = shot

    return out
