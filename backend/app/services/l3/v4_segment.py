"""
Cuts V4 -- the deterministic video segmenter (cuts_v4_segmentation.plan.md,
v4_cuts_as_primitive.plan.md, v4_cluster_tree_cuts.plan.md).

Replaces the video half of pass 1's job (grouping atoms into
``VideoTentativeGroup``s) with a signal-driven extractor built on one
principle: a raw clip is mostly scrap -- find the small usable part(s) and
discard the rest. Default is to trim hard to the usable core, never "keep the
whole clip". Speech is untouched (pass 1 still owns speech grouping + junk);
this module only decides the NON-SPEECH remainder's cuts.

Three rules threaded throughout, per the plans:
  * Salience = contrast/novelty (how much a moment stands out from its LOCAL
    surroundings), not absolute level, and not requiring audio+motion
    consensus -- either channel alone can produce a point; agreement only
    raises confidence (they simply add).
  * The VLM (pass 2, elsewhere) decides *shape* (semantic); this module (code)
    always decides *where* -- location is deterministic.
  * A video cut is a CLUSTER, not a flat span: one continuous moment carrying
    every salient EVENT inside it (point and span kinds coexisting). The
    energy ladder (cutrecord_map.resolve_cluster) resolves that cluster into
    the right piece-set at every level -- broad = the whole moment as one
    cut, punchy = each event as its own tight piece. A cluster of exactly one
    event degenerates to exactly today's single-window V4 cut at every level
    (backward compatible by construction).

Pure core: ``segment_video(...)`` takes already-loaded signals (the same
``motion``/``scene`` shapes ``lattice.build_atoms`` consumes) and chooses
spans directly on the motion hop grid. V4 does NOT carve atoms and does not
map cuts onto them at all -- a V4 cut's span IS the primitive, carried as-is
to the brain; atoms remain only the SPEECH substrate (built elsewhere, in
``lattice.build_atoms``, untouched by this module). No model call, no DB
call -- see ``scripts/test_v4_segment.py``.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.lattice import _subtract
from app.services.l3.post import _mean, _norm_in_clip, _series_lohi, _span_slice
from app.services.l3.v4_segment_params import (
    CAMERA_MOVE_COHERENCE_MIN, CAMERA_MOVE_MAGNITUDE_MIN, CAMERA_MOVE_MIN_MS,
    CLUSTER_SEPARATION_MULTIPLIER, DECAY_FRACTION, DENSITY_PEAKS_PER_SEC_CAP,
    EDGE_SNAP_SEARCH_MS, EDGE_SNAP_STABILITY_MAX, FOLLOW_THROUGH_FLOOR_MS,
    MAX_CLUSTER_SEPARATION_MS, MAX_PAD_MS, MIN_CUT_DURATION_MS, MIN_CUT_GAP_MS,
    NOVELTY_ABSOLUTE_FLOOR, NOVELTY_BASELINE_RADIUS_MS, PEAK_MIN_GAP_MS,
    PEAK_PROMINENCE_RATIO, PERIODICITY_SCORE_THRESHOLD, REPRESENTATIVE_WINDOW_MS,
    RUN_UP_FLOOR_MS,
)


@dataclass
class VideoCut:
    file_id: str
    src_in_ms: int
    src_out_ms: int
    # Multi-peak, v4_cluster_tree_cuts.plan.md section 3:
    #   {"peak_ms", "score", "kind", "span_ms"}  -- events[primary]'s own
    #     fields, broadcast to the top level so every existing single-anchor
    #     reader (post.py, cutrecord_map._video_rung's single-event path,
    #     image_plan's straddle bias, the frontend dial) keeps working
    #     unchanged on a cluster of one.
    #   "events": [{"peak_ms","score","kind","onset_ms","settle_ms",
    #               "span_ms"}, ...] -- every salient event in this cluster,
    #     time-ordered. The merge tree/dendrogram is DELIBERATELY not
    #     materialized as a separate structure (section 4.3): it's fully
    #     recoverable from this ordered list's own inter-event gaps (sort
    #     ascending -> merge order), which is what resolve_cluster consumes.
    #   "primary": index into "events" of the strongest one (by score).
    #   "density": this cluster's own novelty-peak-rate stat, 0..1 (also
    #     carried as VideoCut.density below for post.compute_pace_envelope).
    salience: Dict[str, Any] = field(default_factory=dict)
    # Event density/novelty stat (0..1) feeding post.compute_pace_envelope's
    # content-aware min_ms: sparse/monotonous -> collapses hard; dense ->
    # holds more room at the same energy.
    density: float = 0.0
    # Transient (NOT persisted -- absent from to_dict): the working span this cut
    # was carved from. Lets _finalize_cuts weld a sub-floor sliver ONLY into a
    # same-shot/same-span neighbor, never across the speech (or shot) gap between
    # two spans -- a cross-gap union would engulf the content between them (the
    # video-cut-swallows-speech overlap).
    span_key: Optional[Tuple[int, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"file_id": self.file_id, "src_in_ms": self.src_in_ms,
                "src_out_ms": self.src_out_ms,
                "salience": dict(self.salience), "density": self.density}


# --------------------------------------------------------------------------
# Step 0: working units -- single-shot, non-speech
# --------------------------------------------------------------------------

def _shot_and_transition_marks(motion: Optional[dict], scene: Optional[dict]) -> List[int]:
    """Sorted, deduped ts_ms of every shot cut (scene.shot_points) + transition
    (motion.transition_points -- wipe/degenerate; lives on the motion signal,
    not scene, unlike the plan's shorthand "scene (shot/composition/
    transition)" suggests -- see lattice._transition_marks). These are the
    mechanical pre-split only; V4 never treats them as editorial choices."""
    marks = {int(p["ts_ms"]) for p in ((scene or {}).get("shot_points") or [])
             if isinstance(p, dict) and "ts_ms" in p}
    marks |= {int(p["ts_ms"]) for p in ((motion or {}).get("transition_points") or [])
              if isinstance(p, dict) and "ts_ms" in p}
    return sorted(marks)


def _working_spans(duration_ms: int, motion: Optional[dict], scene: Optional[dict],
                    speech_spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """[0, duration_ms) split at shot/transition marks into single-shot
    segments, minus every speech span -> the non-speech working spans V4
    operates on, one shot at a time."""
    if duration_ms <= 0:
        return []
    marks = [m for m in _shot_and_transition_marks(motion, scene) if 0 < m < duration_ms]
    bounds = sorted({0, duration_ms} | set(marks))
    shots = [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
    out: List[Tuple[int, int]] = []
    for s, e in shots:
        out.extend(_subtract((s, e), speech_spans))
    return [(s, e) for s, e in out if e > s]


# --------------------------------------------------------------------------
# Step 2: the novelty curve
# --------------------------------------------------------------------------

def _rolling_baseline(values: List[float], radius: int) -> List[float]:
    """Centered rolling-median baseline: value[i]'s "local neighborhood" is
    +/-radius samples around it. A sustained-high stretch tracks its own
    baseline (near-zero novelty); a burst out of calm doesn't."""
    n = len(values)
    return [statistics.median(values[max(0, i - radius):min(n, i + radius + 1)])
            for i in range(n)]


def _step_novelty(norm: List[float], radius: int) -> List[float]:
    """Two-sided mean-shift ('step') detector: |mean(after) - mean(before)|
    at each instant. A centered rolling-MEDIAN baseline tracks a smooth
    monotonic ramp almost exactly (median of a symmetric window over a linear
    trend ~= its center value), so it under-detects a RAMP INTO a new
    sustained level once the window's horizon reaches the new level on both
    sides -- this catches that level-shift directly, so a transition into a
    plateau still registers as an event, not just a return-to-calm impulse."""
    n = len(norm)
    out = [0.0] * n
    for i in range(n):
        before = norm[max(0, i - radius):i] or [norm[i]]
        after = norm[i:min(n, i + radius)] or [norm[i]]
        out[i] = abs(sum(after) / len(after) - sum(before) / len(before))
    return out


def _channel_novelty(values: List[float], lo: Optional[float], hi: Optional[float],
                      radius: int) -> List[float]:
    norm = [(_norm_in_clip(v, lo, hi) or 0.0) for v in values]
    baseline = _rolling_baseline(norm, radius)
    impulse = [max(0.0, n - b) for n, b in zip(norm, baseline)]
    step = _step_novelty(norm, radius)
    return [max(im, st) for im, st in zip(impulse, step)]


def _periodicity_score(values: List[float]) -> float:
    """0..1: best normalized autocorrelation of the signal's FIRST DIFFERENCE
    at a non-trivial lag -- a blinking light / wave / timelapse has a
    periodically repeating CHANGE pattern; a one-off burst or a ramp into a
    new sustained level has exactly one isolated change and does not
    correlate with a shifted copy of itself anywhere. Differencing first
    (rather than testing the raw levels) is what tells "oscillates
    repeatedly" apart from "holds one long constant/flat stretch", which
    would otherwise trivially self-correlate at every lag. Signal-only (no
    action_points needed), so it catches a periodic CONTINUOUS signal too,
    not just evenly-spaced discrete events."""
    if len(values) < 7:
        return 0.0
    values = [b - a for a, b in zip(values, values[1:])]
    n = len(values)
    mean = sum(values) / n
    centered = [v - mean for v in values]
    energy = sum(c * c for c in centered)
    if energy <= 1e-9:
        return 0.0
    max_lag = min(n - 1, n // 2)
    best = 0.0
    for lag in range(2, max_lag + 1):
        num = sum(centered[i] * centered[i + lag] for i in range(n - lag))
        corr = num / energy
        if corr > best:
            best = corr
    return max(0.0, min(1.0, best))


def _evenly_spaced(ts: List[int]) -> bool:
    """True when >=3 timestamps have near-uniform consecutive gaps (low
    coefficient of variation) -- the discrete-event half of the periodicity
    test (Step 2's "evenly-spaced repeated action_points" discount)."""
    if len(ts) < 3:
        return False
    gaps = [b - a for a, b in zip(ts, ts[1:]) if b > a]
    if len(gaps) < 2:
        return False
    mean_gap = sum(gaps) / len(gaps)
    if mean_gap <= 0:
        return False
    var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    cv = (var ** 0.5) / mean_gap
    return cv < 0.2


def _novelty_curve(
    span: Tuple[int, int], motion: dict, audio: dict, hop_ms: int,
    ae_lohi: Tuple[Optional[float], Optional[float]], rms_lohi: Tuple[Optional[float], Optional[float]],
) -> List[float]:
    """The fused, periodicity-discounted novelty curve over ``span``, on the
    motion hop grid -- Step 2. Either motion or audio novelty alone can drive
    a point (no hard consensus requirement); they simply add, so agreement
    naturally raises the combined value without gating on it."""
    s, e = span
    radius = max(1, NOVELTY_BASELINE_RADIUS_MS // max(hop_ms, 1))
    action = _span_slice(motion.get("action_energy") or [], hop_ms, s, e)
    ae_lo, ae_hi = ae_lohi
    motion_nov = _channel_novelty(action, ae_lo, ae_hi, radius)
    n = len(motion_nov)
    if n == 0:
        return []

    # Pre-resampled onto the motion hop grid by segment_video, so it can be
    # sliced with the SAME (hop_ms, s, e) as action -- kept span-aligned.
    rms = _span_slice(motion.get("_rms_at_motion_hop") or [], hop_ms, s, e)
    rms_lo, rms_hi = rms_lohi
    audio_nov = _channel_novelty(rms, rms_lo, rms_hi, radius) if rms else [0.0] * n
    if len(audio_nov) < n:
        audio_nov = audio_nov + [0.0] * (n - len(audio_nov))

    curve = [m + a for m, a in zip(motion_nov, audio_nov[:n])]

    anchors = [int(p["ts_ms"]) for p in (motion.get("action_points") or [])
               if s <= int(p.get("ts_ms", -1)) < e]
    if audio.get("is_musical"):
        anchors += [t for t in (audio.get("onsets_ms") or []) if s <= t < e]
    for t in anchors:
        i = (t - s) // hop_ms
        if 0 <= i < n and curve[i] > 0:
            curve[i] += 1.0

    # Continuous, not a fixed haircut: a near-perfect repeat (autocorrelation
    # / evenly-spaced-events score near 1) suppresses novelty almost entirely
    # (a blink is all "change", no "event"); a one-off burst (low score) is
    # untouched. Below the threshold, no discount at all -- incidental
    # autocorrelation in a short/noisy curve shouldn't nibble real events.
    periodicity = max(_periodicity_score(action), 1.0 if _evenly_spaced(sorted(anchors)) else 0.0)
    if periodicity >= PERIODICITY_SCORE_THRESHOLD:
        curve = [c * max(0.0, 1.0 - periodicity) for c in curve]
    return curve


# --------------------------------------------------------------------------
# Step 3: event detection (every kind coexists -- v4_cluster_tree_cuts.plan.md
# section 4.1/4.2)
# --------------------------------------------------------------------------

def _find_peaks(curve: List[float], min_gap_hops: int) -> List[int]:
    """Local maxima, non-max-suppressed within +/-min_gap_hops so a wide bump
    yields one peak, not a cluster."""
    candidates = sorted((i for i in range(len(curve)) if curve[i] > 0), key=lambda i: -curve[i])
    chosen: List[int] = []
    for i in candidates:
        if all(abs(i - c) > min_gap_hops for c in chosen):
            chosen.append(i)
    return sorted(chosen)


def _prominent_peaks(curve: List[float], hop_ms: int) -> List[int]:
    if not curve:
        return []
    lo, hi = min(curve), max(curve)
    # Relative prominence is scale-invariant (uniformly shrinking the whole
    # curve never changes whether its max clears a RELATIVE bar) -- the
    # absolute floor is what lets a periodicity-discounted curve actually
    # fall through to kind="none" instead of always finding SOME "peak".
    if hi - lo < 1e-9 or hi < NOVELTY_ABSOLUTE_FLOOR:
        return []
    thr = max(lo + PEAK_PROMINENCE_RATIO * (hi - lo), NOVELTY_ABSOLUTE_FLOOR)
    min_gap_hops = max(1, PEAK_MIN_GAP_MS // max(hop_ms, 1))
    return [i for i in _find_peaks(curve, min_gap_hops) if curve[i] >= thr]


def _true_runs(flags: List[bool]) -> List[Tuple[int, int]]:
    """[start_i, end_i) index runs where ``flags`` is True."""
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < n and flags[j]:
            j += 1
        runs.append((i, j))
        i = j
    return runs


def _camera_move_cores(motion: dict, span: Tuple[int, int], hop_ms: int) -> List[Tuple[int, int]]:
    """(start_ms, end_ms) of EVERY sustained, coherent camera move in
    ``span`` -- v4_cluster_tree_cuts.plan.md section 4.2 (the pan-loss bug):
    a SECOND good pan must not be silently dropped just because an earlier
    one happened to be longer. A hop counts as "moving" once its combined
    |dx|+|dy|+|zoom| clears CAMERA_MOVE_MAGNITUDE_MIN AND coherence clears
    CAMERA_MOVE_COHERENCE_MIN (deliberate, not shake); each run must sustain
    CAMERA_MOVE_MIN_MS to count as a real payload."""
    s, e = span
    dx = _span_slice(motion.get("camera_dx") or [], hop_ms, s, e)
    dy = _span_slice(motion.get("camera_dy") or [], hop_ms, s, e)
    dz = _span_slice(motion.get("camera_zoom") or [], hop_ms, s, e)
    coh = _span_slice(motion.get("camera_coherence") or [], hop_ms, s, e)
    n = max(len(dx), len(dy), len(dz))
    if n == 0:
        return []

    def _at(arr: List[float], i: int, default: float) -> float:
        return arr[i] if i < len(arr) else default

    moving = [
        (abs(_at(dx, i, 0.0)) + abs(_at(dy, i, 0.0)) + abs(_at(dz, i, 0.0))) >= CAMERA_MOVE_MAGNITUDE_MIN
        and _at(coh, i, 0.0) >= CAMERA_MOVE_COHERENCE_MIN
        for i in range(n)
    ]
    out: List[Tuple[int, int]] = []
    for start_i, end_i in _true_runs(moving):
        if (end_i - start_i) * hop_ms < CAMERA_MOVE_MIN_MS:
            continue
        out.append((s + start_i * hop_ms, min(e, s + end_i * hop_ms)))
    return out


def _representative_window(motion: dict, span: Tuple[int, int], hop_ms: int) -> Tuple[int, int]:
    """Modest window centered on the steadiest, sharpest instant -- Step 3.4,
    the "nothing stands out anywhere" fallback. Never the whole span."""
    s, e = span
    stability = _span_slice(motion.get("camera_stability") or [], hop_ms, s, e)
    blur = _span_slice(motion.get("blur") or [], hop_ms, s, e)
    n = max(len(stability), len(blur))
    half = min(REPRESENTATIVE_WINDOW_MS, e - s) // 2
    if n == 0:
        mid = (s + e) // 2
        return max(s, mid - half), min(e, mid + half)

    def _cost(i: int) -> float:
        st = stability[i] if i < len(stability) else 1.0
        bl = blur[i] if i < len(blur) else 0.0
        return bl - st

    best_i = min(range(n), key=_cost)
    center = s + best_i * hop_ms
    in_ms = max(s, min(center - half, e - 2 * half))
    out_ms = min(e, in_ms + 2 * half)
    return in_ms, out_ms


# --------------------------------------------------------------------------
# Step 4: edges. onset_ms/settle_ms are the RAW decay-walk bounds (no floor,
# no MAX_PAD, no quality snap) -- the event's own natural, content-derived
# reach, persisted so resolve_cluster can interpolate a per-event window at
# any energy. _broad_window_for_event separately derives the FLOOR/MAX_PAD-
# clamped, quality-snapped "broad" (energy=0) window from those raw bounds --
# byte-identical to the pre-cluster _point_edges formula, used for cluster
# grouping and the single-event fast path.
# --------------------------------------------------------------------------

def _decay_bound(curve: List[float], peak_i: int, direction: int, floor_i: int, ceil_i: int) -> int:
    """Walk from ``peak_i`` in ``direction`` (+1/-1) until the curve decays
    below DECAY_FRACTION of the peak's own height, bounded by [floor_i, ceil_i]."""
    thr = curve[peak_i] * DECAY_FRACTION
    i = peak_i
    while True:
        nxt = i + direction
        if direction > 0 and nxt > ceil_i:
            break
        if direction < 0 and nxt < floor_i:
            break
        i = nxt
        if curve[i] < thr:
            break
    return i


def _snap_to_quality_gate(motion: dict, hop_ms: int, ts_ms: int, lo_ms: int, hi_ms: int) -> int:
    """Nudge ``ts_ms`` to the nearest instant within EDGE_SNAP_SEARCH_MS whose
    camera_stability reads as a whip/bump (a clean place to cut) rather than a
    smooth in-progress move -- Step 4's camera-quality gate. A no-op when
    there's no stability signal, or nothing better is found nearby."""
    stability = motion.get("camera_stability") or []
    if not stability or hop_ms <= 0:
        return ts_ms
    lo_i = max(lo_ms // hop_ms, (ts_ms - EDGE_SNAP_SEARCH_MS) // hop_ms)
    hi_i = min((hi_ms - 1) // hop_ms if hi_ms > 0 else 0, (ts_ms + EDGE_SNAP_SEARCH_MS) // hop_ms)
    hi_i = min(hi_i, len(stability) - 1)
    if lo_i > hi_i:
        return ts_ms
    cur_i = min(max(ts_ms // hop_ms, 0), len(stability) - 1)
    if stability[cur_i] <= EDGE_SNAP_STABILITY_MAX:
        return ts_ms
    best_i = min(range(lo_i, hi_i + 1), key=lambda i: (stability[i], abs(i - cur_i)))
    return best_i * hop_ms if stability[best_i] <= EDGE_SNAP_STABILITY_MAX else ts_ms


def _score_at(curve: List[float], i: int) -> float:
    lo, hi = min(curve), max(curve)
    if hi - lo < 1e-9:
        return 1.0 if curve[i] > 0 else 0.0
    return max(0.0, min(1.0, (curve[i] - lo) / (hi - lo)))


def _point_event(curve: List[float], span: Tuple[int, int], hop_ms: int,
                  peak_i: int, peak_ms_override: Optional[int] = None) -> Dict[str, Any]:
    """One point event from a novelty-curve peak (or a transition seam, via
    ``peak_ms_override``): peak_ms/score plus the RAW decay-walk onset/settle
    (unclamped -- see module note above)."""
    s, _e = span
    n = len(curve)
    back_i = _decay_bound(curve, peak_i, -1, 0, n - 1)
    fwd_i = _decay_bound(curve, peak_i, +1, 0, n - 1)
    peak_ms = peak_ms_override if peak_ms_override is not None else s + peak_i * hop_ms
    onset_ms = min(s + back_i * hop_ms, peak_ms)
    settle_ms = max(s + fwd_i * hop_ms, peak_ms)
    return {"peak_ms": peak_ms, "score": _score_at(curve, peak_i), "kind": "point",
            "onset_ms": onset_ms, "settle_ms": settle_ms, "span_ms": None}


def _broad_window_for_event(event: Dict[str, Any], motion: dict, hop_ms: int,
                            span: Tuple[int, int]) -> Tuple[int, int]:
    """The event's own BROAD (energy=0) window -- floor/MAX_PAD-clamped and
    quality-gate-snapped. Byte-identical to the pre-cluster _point_edges
    formula for a point event (given the RAW onset/settle _point_event
    produces), so a cluster of one event reproduces today's V4 span exactly.
    A span event's own core, clamped to the working span; the representative-
    window fallback's own bounds verbatim (already inside the span)."""
    kind = event.get("kind")
    if kind == "span":
        s0, e0 = event["span_ms"]
        return max(span[0], int(s0)), min(span[1], int(e0))
    if kind == "none":
        return event["onset_ms"], event["settle_ms"]
    peak = event["peak_ms"]
    run_up = max(RUN_UP_FLOOR_MS, min(MAX_PAD_MS, peak - event["onset_ms"]))
    follow_through = max(FOLLOW_THROUGH_FLOOR_MS, min(MAX_PAD_MS, event["settle_ms"] - peak))
    in_ms = max(span[0], peak - run_up)
    out_ms = min(span[1], peak + follow_through)
    in_ms = _snap_to_quality_gate(motion, hop_ms, in_ms, span[0], peak)
    out_ms = _snap_to_quality_gate(motion, hop_ms, out_ms, peak, span[1])
    return min(in_ms, peak), max(out_ms, peak)


# --------------------------------------------------------------------------
# Step 3 (cont'd): collect EVERY event in a working span -- point and span
# kinds coexisting (v4_cluster_tree_cuts.plan.md section 4.1). The
# representative-window fallback fires only when the span produced no
# events of any kind at all.
# --------------------------------------------------------------------------

def _novelty_density(curve: List[float], hop_ms: int) -> float:
    if not curve or hop_ms <= 0:
        return 0.0
    min_gap_hops = max(1, PEAK_MIN_GAP_MS // hop_ms)
    peaks = _find_peaks(curve, min_gap_hops)
    dur_s = max(0.001, len(curve) * hop_ms / 1000.0)
    return max(0.0, min(1.0, len(peaks) / dur_s / DENSITY_PEAKS_PER_SEC_CAP))


def _dedupe_point_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge point events whose decay-walk windows overlap into ONE (the
    stronger by score) -- a noisy/wiggly curve near a single real burst can
    surface several nearby local maxima that all clear the prominence bar
    (e.g. a slight dip mid-rise, or on the way back down), and those are
    detections of the SAME moment, not separate events. Span events pass
    through untouched (§4.2 wants every sustained move kept, not deduped)."""
    points = sorted((ev for ev in events if ev["kind"] == "point"), key=lambda ev: ev["onset_ms"])
    others = [ev for ev in events if ev["kind"] != "point"]
    merged: List[Dict[str, Any]] = []
    for ev in points:
        if merged and ev["onset_ms"] <= merged[-1]["settle_ms"]:
            if ev["score"] > merged[-1]["score"]:
                merged[-1] = ev
        else:
            merged.append(ev)
    return merged + others


def _events_for_span(
    span: Tuple[int, int], motion: dict, audio: dict, hop_ms: int,
    ae_lohi: Tuple[Optional[float], Optional[float]], rms_lohi: Tuple[Optional[float], Optional[float]],
) -> Tuple[List[Dict[str, Any]], float]:
    """Every salient event in one working span, plus this span's own density
    stat. Point events (transition seams + novelty peaks) and span events
    (every sustained camera move) all coexist -- never a first-match
    early-exit. Falls back to one synthetic kind="none" representative-window
    event only when nothing else fired at all."""
    s, e = span
    curve = _novelty_curve(span, motion, audio, hop_ms, ae_lohi, rms_lohi)
    density = _novelty_density(curve, hop_ms)

    events: List[Dict[str, Any]] = []

    # Transition seams -- a premium natural seam, always a point event.
    transitions = [int(p["ts_ms"]) for p in (motion.get("transition_points") or [])
                   if s < int(p.get("ts_ms", -1)) < e]
    transition_idxs: set = set()
    if curve:
        for ts in transitions:
            i = min(max((ts - s) // hop_ms, 0), len(curve) - 1)
            transition_idxs.add(i)
            events.append(_point_event(curve, span, hop_ms, i, peak_ms_override=ts))

    # Novelty peaks clearing the span's own prominence bar -- point events.
    # Skip an index already emitted as a transition (same instant, not a
    # second event).
    if curve:
        for i in _prominent_peaks(curve, hop_ms):
            if i in transition_idxs:
                continue
            events.append(_point_event(curve, span, hop_ms, i))

    events = _dedupe_point_events(events)

    # EVERY sustained camera move -- span events (never just the longest).
    for core_s, core_e in _camera_move_cores(motion, span, hop_ms):
        coh = _span_slice(motion.get("camera_coherence") or [], hop_ms, core_s, core_e)
        score = max(0.0, min(1.0, _mean(coh) or 0.0))
        events.append({"peak_ms": core_s + (core_e - core_s) // 2, "score": score,
                       "kind": "span", "onset_ms": core_s, "settle_ms": core_e,
                       "span_ms": [core_s, core_e]})

    if not events:
        win_s, win_e = _representative_window(motion, span, hop_ms)
        events.append({"peak_ms": win_s + (win_e - win_s) // 2, "score": 0.0,
                       "kind": "none", "onset_ms": win_s, "settle_ms": win_e, "span_ms": None})

    return events, density


# --------------------------------------------------------------------------
# Step 5: cluster grouping -- events close enough to fuse (at the broadest
# window) belong to one cluster (one VideoCut); a big dead gap starts a new
# one (v4_cluster_tree_cuts.plan.md section 4.3). The merge tree is the
# ordered event list itself (sort inter-event gaps ascending -> merge order)
# -- deliberately not materialized as a separate structure; resolve_cluster
# (cutrecord_map.py) consumes the ordered events + their own onset/settle
# directly.
# --------------------------------------------------------------------------

def _cluster_separation_ms(gaps: List[int]) -> int:
    """The gap a working span's OWN event spacing has to clear to count as a
    genuine break, not just this burst's normal rhythm -- content-derived
    (a multiple of the span's own median inter-event gap), clamped to
    [MIN_CUT_GAP_MS, MAX_CLUSTER_SEPARATION_MS] so one outlier gap in a tiny
    sample can't blow the threshold out, and a very tight span still gets
    some separation floor. Degenerate (no positive gaps) -> the floor."""
    positive = [g for g in gaps if g > 0]
    if not positive:
        return MIN_CUT_GAP_MS
    return int(max(MIN_CUT_GAP_MS, min(MAX_CLUSTER_SEPARATION_MS,
                                       statistics.median(positive) * CLUSTER_SEPARATION_MULTIPLIER)))


def _cluster_events(events: List[Dict[str, Any]], motion: dict, hop_ms: int,
                    span: Tuple[int, int]) -> List[List[Dict[str, Any]]]:
    """Group one working span's events into clusters by their own BROAD
    (energy=0) window gaps -- the same window ``_broad_window_for_event``
    already computes for cluster-extent purposes, so "close enough to fuse
    at the broadest window" is judged on the exact windows that fusion would
    use. Always >= 1 cluster when ``events`` is non-empty."""
    if not events:
        return []
    windows = sorted(((_broad_window_for_event(ev, motion, hop_ms, span), ev) for ev in events),
                     key=lambda x: x[0][0])
    gaps = [windows[i + 1][0][0] - windows[i][0][1] for i in range(len(windows) - 1)]
    threshold = _cluster_separation_ms(gaps)
    clusters: List[List[Dict[str, Any]]] = [[windows[0][1]]]
    for i in range(1, len(windows)):
        gap = windows[i][0][0] - windows[i - 1][0][1]
        if gap > threshold:
            clusters.append([windows[i][1]])
        else:
            clusters[-1].append(windows[i][1])
    return clusters


def _build_salience(events: List[Dict[str, Any]], density: float) -> Dict[str, Any]:
    """The multi-peak salience dict (v4_cluster_tree_cuts.plan.md section 3):
    events + which one is strongest (primary) + this cluster's density,
    PLUS the primary event's own fields broadcast to the top level so every
    existing single-anchor reader keeps working unchanged."""
    primary = max(range(len(events)), key=lambda i: events[i].get("score", 0.0))
    prim = events[primary]
    return {
        "peak_ms": prim["peak_ms"], "score": prim["score"], "kind": prim["kind"],
        "span_ms": prim["span_ms"],
        "events": [dict(ev) for ev in events],
        "primary": primary,
        "density": density,
    }


# --------------------------------------------------------------------------
# Step 6: finalize -- geometry only, no atoms in this loop at all
# (v4_cuts_as_primitive.plan.md section 6).
# --------------------------------------------------------------------------

def _merge_pair(cuts: List[VideoCut], j: int) -> List[VideoCut]:
    """Merge ``cuts[j]`` into whichever SAME-WORKING-SPAN neighbor it sits closer
    to (by gap) -- the union's events are the UNION of both clusters' events
    (never just one side's salience wholesale, or a merged sliver's own event(s)
    would silently vanish), re-scored via _build_salience. Only same-span
    neighbors are eligible: welding across the gap between two spans (a speech
    span, or another shot) would produce a union that swallows the content
    between them. Caller guarantees at least one same-span neighbor exists."""
    same = cuts[j].span_key
    left = j - 1 if j - 1 >= 0 and cuts[j - 1].span_key == same else None
    right = j + 1 if j + 1 < len(cuts) and cuts[j + 1].span_key == same else None

    def _gap(x: int) -> int:
        return (cuts[j].src_in_ms - cuts[x].src_out_ms) if x < j else (cuts[x].src_in_ms - cuts[j].src_out_ms)

    pick = left if right is None else (right if left is None else (
        left if _gap(left) <= _gap(right) else right))
    a_, b_ = (pick, j) if pick < j else (j, pick)
    ca, cb = cuts[a_], cuts[b_]
    events = list(ca.salience.get("events") or []) + list(cb.salience.get("events") or [])
    density = max(ca.density, cb.density)
    merged = VideoCut(file_id=ca.file_id, src_in_ms=min(ca.src_in_ms, cb.src_in_ms),
                      src_out_ms=max(ca.src_out_ms, cb.src_out_ms),
                      salience=_build_salience(events, density), density=density,
                      span_key=same)
    return cuts[:a_] + [merged] + cuts[b_ + 1:]


def _finalize_cuts(cuts: List[VideoCut]) -> List[VideoCut]:
    """Enforce the two invariants per-cluster logic can't, over ALL of a
    file's cuts -- geometry only; no atoms in this loop at all:

    * DISJOINT, CLAMPED TO ITS OWN WORKING SPAN. Cluster grouping only
      dedupes WITHIN one working span; a cluster whose edge was extended
      (an event's own run-up/follow-through, or a camera move's settle)
      past its span can still collide with the next span's cluster. Clamp
      the earlier cut's out down to the later cut's in.
    * MIN-DURATION FLOOR. A cluster left too short by that clamp (or an
      unusually tight single-event anchor) isn't a distinct usable moment
      on its own -- merge it into whichever same-span neighbor it sits
      closer to. A cluster spanning multiple events is never below the
      floor by construction (events have gaps between them), so this only
      ever affects a degenerate single-event cluster -- unchanged behavior
      there from the pre-cluster V4."""
    if not cuts:
        return cuts
    cuts = sorted(cuts, key=lambda c: c.src_in_ms)
    for i in range(len(cuts) - 1):
        if cuts[i + 1].src_in_ms < cuts[i].src_out_ms:
            cuts[i].src_out_ms = cuts[i + 1].src_in_ms
    cuts = [c for c in cuts if c.src_out_ms > c.src_in_ms]

    while len(cuts) > 1:
        short = None
        for i, c in enumerate(cuts):
            if c.src_out_ms - c.src_in_ms >= MIN_CUT_DURATION_MS:
                continue
            # Weldable only into a SAME-SPAN neighbor. A sub-floor sliver with no
            # same-span neighbor (e.g. isolated between two speech spans) stays
            # as-is: a slightly-short but disjoint cut is acceptable, an overlap
            # (from welding across the speech between spans) is not.
            has_same = ((i > 0 and cuts[i - 1].span_key == c.span_key) or
                        (i < len(cuts) - 1 and cuts[i + 1].span_key == c.span_key))
            if has_same:
                short = i
                break
        if short is None:
            break
        cuts = _merge_pair(cuts, short)
    return cuts


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def segment_video(
    *, file_id: str, duration_ms: int, speech_spans: List[Tuple[int, int]],
    motion: Dict[str, Any], audio: Dict[str, Any], scene: Dict[str, Any],
) -> List[VideoCut]:
    """The non-speech remainder of one file -> a small set of tight, salient
    video CLUSTERS (never the whole span by default), each carrying every
    event inside it. Pure and deterministic: same signals always produce the
    same clusters."""
    motion = motion or {}
    audio = audio or {}
    scene = scene or {}
    hop_ms = int(motion.get("hop_ms") or 0)
    if hop_ms <= 0:
        return []

    spans = _working_spans(duration_ms, motion, scene, speech_spans)
    if not spans:
        return []

    action = motion.get("action_energy") or []
    ae_lohi = _series_lohi(action)
    rms = audio.get("rms_db") or []
    rms_hop_ms = int(audio.get("hop_ms") or 0)
    rms_lohi = _series_lohi(rms)
    # Resample rms onto the motion hop grid ONCE (spans share the same
    # file-wide grids) so _novelty_curve can treat both channels uniformly.
    motion = dict(motion)
    if rms and rms_hop_ms > 0:
        n_motion = (duration_ms // hop_ms) + 1
        motion["_rms_at_motion_hop"] = [
            rms[min(len(rms) - 1, (i * hop_ms) // rms_hop_ms)] for i in range(n_motion)
        ]

    cuts: List[VideoCut] = []
    for span in spans:
        events, density = _events_for_span(span, motion, audio, hop_ms, ae_lohi, rms_lohi)
        for cluster in _cluster_events(events, motion, hop_ms, span):
            windows = [_broad_window_for_event(ev, motion, hop_ms, span) for ev in cluster]
            in_ms = max(span[0], min(w[0] for w in windows))
            out_ms = min(span[1], max(w[1] for w in windows))
            if out_ms <= in_ms:
                continue
            cuts.append(VideoCut(file_id=file_id, src_in_ms=in_ms, src_out_ms=out_ms,
                                 salience=_build_salience(cluster, density), density=density,
                                 span_key=span))
    cuts.sort(key=lambda c: c.src_in_ms)
    return _finalize_cuts(cuts)
