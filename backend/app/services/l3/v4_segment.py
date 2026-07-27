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
    CAMERA_MOVE_COHERENCE_MIN, CAMERA_MOVE_MIN_MS,
    CLUSTER_SEPARATION_MULTIPLIER, DEAD_ENERGY_FLOOR, DECAY_FRACTION,
    DENSITY_PEAKS_PER_SEC_CAP, FOLLOW_THROUGH_FLOOR_MS,
    ACTION_RUN_BASELINE_FRACTION, ACTION_RUN_LEFTOVER_MIN_MS, ACTION_RUN_MIN_MS,
    LULL_LEVEL_FRACTION, LULL_MIN_MS,
    MAX_CLUSTER_SEPARATION_MS, MAX_PAD_MS, MIN_CUT_DURATION_MS, MIN_CUT_GAP_MS,
    MOVE_ABSOLUTE_FLOOR, MOVE_MAGNITUDE_PERCENTILE_HI, MOVE_MAGNITUDE_PERCENTILE_LO,
    MOVE_RELATIVE_FRACTION, MOVE_SPREAD_RATIO_MIN,
    NOVELTY_ABSOLUTE_FLOOR, NOVELTY_BASELINE_RADIUS_MS, PEAK_MIN_GAP_MS,
    PEAK_PROMINENCE_RATIO, PERIODICITY_SCORE_THRESHOLD, REGIME_BLUR_MAX,
    REGIME_COHERENCE_MOVE_MIN, REGIME_MAGNITUDE_MOVE_MIN,
    REGIME_STABILITY_TRANSIENT_MAX, REPRESENTATIVE_WINDOW_MS, RUN_UP_FLOOR_MS,
    SNAP_FRAC, SNAP_MS_FLOOR,
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


def _periodicity_score(values: List[float]) -> Tuple[float, int]:
    """(0..1 score, period_in_samples): best normalized autocorrelation of
    the signal's FIRST DIFFERENCE at a non-trivial lag, and the lag it was
    found at -- a blinking light / wave / timelapse has a periodically
    repeating CHANGE pattern; a one-off burst or a ramp into a new sustained
    level has exactly one isolated change and does not correlate with a
    shifted copy of itself anywhere. Differencing first (rather than testing
    the raw levels) is what tells "oscillates repeatedly" apart from "holds
    one long constant/flat stretch", which would otherwise trivially
    self-correlate at every lag -- and doesn't shift where a repeat lands,
    so the winning lag is the period of the ORIGINAL (undifferenced) signal
    too (cuts_content_first_segmentation.plan.md Part 6 reuses it exactly
    this way, in _representative_window). Signal-only (no action_points
    needed), so it catches a periodic CONTINUOUS signal too, not just
    evenly-spaced discrete events. period is 0 when nothing periodic was
    found (degenerate signal, or the search space was too short)."""
    if len(values) < 7:
        return 0.0, 0
    diffs = [b - a for a, b in zip(values, values[1:])]
    n = len(diffs)
    mean = sum(diffs) / n
    centered = [v - mean for v in diffs]
    energy = sum(c * c for c in centered)
    if energy <= 1e-9:
        return 0.0, 0
    max_lag = min(n - 1, n // 2)
    best = 0.0
    best_lag = 0
    for lag in range(2, max_lag + 1):
        num = sum(centered[i] * centered[i + lag] for i in range(n - lag))
        corr = num / energy
        if corr > best:
            best = corr
            best_lag = lag
    return max(0.0, min(1.0, best)), best_lag


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
    action_periodicity_score, _action_period_hops = _periodicity_score(action)
    periodicity = max(action_periodicity_score, 1.0 if _evenly_spaced(sorted(anchors)) else 0.0)
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


def _clip_move_threshold(motion: dict) -> float:
    """cuts_content_first_segmentation.plan.md Part 1: this clip's own
    clip-relative "deliberate move" magnitude threshold, replacing the flat
    REGIME_MAGNITUDE_MOVE_MIN (0.03) that aerial/drone footage's ~50x-
    smaller flow magnitude never clears. A robust HIGH percentile of the
    WHOLE clip's own per-hop |dx|+|dy|+|zoom| magnitude series, scaled down
    by MOVE_RELATIVE_FRACTION -- a hop doesn't have to be this clip's single
    highest instant to count, just solidly in its own upper range.

    Guarded on both sides: MOVE_ABSOLUTE_FLOOR keeps a literal near-zero
    clip from ever registering (roundoff, not motion), and the spread test
    (this clip's own high percentile vs its low percentile) keeps a
    genuinely locked/still clip's uniform sensor jitter from being promoted
    to "a move" just because it's this clip's own relative maximum -- that
    case falls back to the OLD absolute floor untouched, never lowering the
    bar for footage that was never moving in the first place."""
    dx = motion.get("camera_dx") or []
    dy = motion.get("camera_dy") or []
    dz = motion.get("camera_zoom") or []
    n = max(len(dx), len(dy), len(dz))
    if n < 4:
        return REGIME_MAGNITUDE_MOVE_MIN
    mags = sorted(
        (abs(dx[i]) if i < len(dx) else 0.0) + (abs(dy[i]) if i < len(dy) else 0.0)
        + (abs(dz[i]) if i < len(dz) else 0.0)
        for i in range(n)
    )
    lo = mags[int(MOVE_MAGNITUDE_PERCENTILE_LO * (n - 1))]
    hi = mags[int(MOVE_MAGNITUDE_PERCENTILE_HI * (n - 1))]
    if hi < MOVE_SPREAD_RATIO_MIN * max(lo, MOVE_ABSOLUTE_FLOOR):
        return REGIME_MAGNITUDE_MOVE_MIN
    return max(MOVE_ABSOLUTE_FLOOR, MOVE_RELATIVE_FRACTION * hi)


def _camera_move_cores(motion: dict, span: Tuple[int, int], hop_ms: int,
                       move_threshold: float = REGIME_MAGNITUDE_MOVE_MIN) -> List[Tuple[int, int]]:
    """(start_ms, end_ms) of EVERY sustained, coherent camera move in
    ``span`` -- v4_cluster_tree_cuts.plan.md section 4.2 (the pan-loss bug):
    a SECOND good pan must not be silently dropped just because an earlier
    one happened to be longer. A hop counts as "moving" once its combined
    |dx|+|dy|+|zoom| clears ``move_threshold`` (Part 1: this clip's own
    clip-relative gate, see ``_clip_move_threshold``) AND coherence clears
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
        (abs(_at(dx, i, 0.0)) + abs(_at(dy, i, 0.0)) + abs(_at(dz, i, 0.0))) >= move_threshold
        and _at(coh, i, 0.0) >= CAMERA_MOVE_COHERENCE_MIN
        for i in range(n)
    ]
    out: List[Tuple[int, int]] = []
    for start_i, end_i in _true_runs(moving):
        if (end_i - start_i) * hop_ms < CAMERA_MOVE_MIN_MS:
            continue
        out.append((s + start_i * hop_ms, min(e, s + end_i * hop_ms)))
    return out


def _peak_norm(values: List[float], lo: Optional[float], hi: Optional[float]) -> float:
    """Highest clip-normalized value in ``values``, or 0.0 when there's no
    usable signal (empty, or degenerate -- lo==hi, no clip-relative
    variation at all to normalize against)."""
    if not values or lo is None or hi is None or hi <= lo:
        return 0.0
    return max((_norm_in_clip(v, lo, hi) or 0.0) for v in values)


def _has_energy_in(
    motion: dict, hop_ms: int, s: int, e: int,
    ae_lohi: Tuple[Optional[float], Optional[float]], rms_lohi: Tuple[Optional[float], Optional[float]],
) -> bool:
    """True when [s, e) carries ANY clip-relative action/rms amplitude at or
    above DEAD_ENERGY_FLOOR, or ANY camera motion reaching REGIME_MAGNITUDE_
    MOVE_MIN -- the shared "is something actually happening in this stretch"
    amplitude test behind both ``_span_is_dead`` (a whole working span) and
    the padding floors' sub-region gate (Part 4, ``_broad_window_for_event``).
    Distinct from low NOVELTY -- a periodic/blinking signal has real
    amplitude but low novelty (see _novelty_curve's periodicity discount)."""
    if e <= s:
        return False
    action = _span_slice(motion.get("action_energy") or [], hop_ms, s, e)
    if _peak_norm(action, *ae_lohi) >= DEAD_ENERGY_FLOOR:
        return True
    rms = _span_slice(motion.get("_rms_at_motion_hop") or [], hop_ms, s, e)
    if _peak_norm(rms, *rms_lohi) >= DEAD_ENERGY_FLOOR:
        return True
    dx = _span_slice(motion.get("camera_dx") or [], hop_ms, s, e)
    dy = _span_slice(motion.get("camera_dy") or [], hop_ms, s, e)
    dz = _span_slice(motion.get("camera_zoom") or [], hop_ms, s, e)
    n = max(len(dx), len(dy), len(dz))
    for i in range(n):
        magnitude = ((abs(dx[i]) if i < len(dx) else 0.0) + (abs(dy[i]) if i < len(dy) else 0.0)
                    + (abs(dz[i]) if i < len(dz) else 0.0))
        if magnitude >= REGIME_MAGNITUDE_MOVE_MIN:
            return True
    return False


def _span_is_dead(
    motion: dict, span: Tuple[int, int], hop_ms: int,
    ae_lohi: Tuple[Optional[float], Optional[float]], rms_lohi: Tuple[Optional[float], Optional[float]],
) -> bool:
    """True when NOTHING salient could possibly be happening anywhere in
    ``span`` (see ``_has_energy_in``) -- feeds the camera-start-still fix: a
    dead span now produces no event at all rather than a fabricated one."""
    s, e = span
    return not _has_energy_in(motion, hop_ms, s, e, ae_lohi, rms_lohi)


def _representative_window(
    motion: dict, span: Tuple[int, int], hop_ms: int,
    ae_lohi: Tuple[Optional[float], Optional[float]], rms_lohi: Tuple[Optional[float], Optional[float]],
) -> Optional[Tuple[int, int]]:
    """Modest window centered on the highest-ENERGY instant (peak of
    clip-normalized action + rms + camera magnitude) -- Step 3.4, the
    "nothing else fired" fallback. Biased toward where something is
    actually HAPPENING, never toward stillness: picking the steadiest,
    sharpest instant is exactly what used to land this fallback on a
    locked-off camera-start still. Returns None when the whole span is dead
    (_span_is_dead) -- a genuinely empty span now produces NO event at all
    rather than a fabricated one.

    cuts_content_first_segmentation.plan.md Part 6 (lever A): when the
    span's action is periodic (reps, turntable, waves, conveyor -- the SAME
    autocorrelation _novelty_curve already runs to discount periodic
    novelty, reused here for its LAG instead), the window width is the
    detected PERIOD -- one clean cycle -- instead of the generic fixed
    REPRESENTATIVE_WINDOW_MS. A mis-locked period is no worse than today's
    arbitrary window (Risk: low)."""
    s, e = span
    if _span_is_dead(motion, span, hop_ms, ae_lohi, rms_lohi):
        return None

    action = _span_slice(motion.get("action_energy") or [], hop_ms, s, e)
    rms = _span_slice(motion.get("_rms_at_motion_hop") or [], hop_ms, s, e)
    dx = _span_slice(motion.get("camera_dx") or [], hop_ms, s, e)
    dy = _span_slice(motion.get("camera_dy") or [], hop_ms, s, e)
    dz = _span_slice(motion.get("camera_zoom") or [], hop_ms, s, e)
    n = max(len(action), len(rms), len(dx), len(dy), len(dz))
    periodicity, period_hops = _periodicity_score(action)
    if periodicity >= PERIODICITY_SCORE_THRESHOLD and period_hops > 0:
        width = max(MIN_CUT_DURATION_MS, min(period_hops * hop_ms, e - s))
    else:
        width = min(REPRESENTATIVE_WINDOW_MS, e - s)
    half = width // 2
    if n == 0:
        mid = (s + e) // 2
        return max(s, mid - half), min(e, mid + half)

    ae_lo, ae_hi = ae_lohi
    rms_lo, rms_hi = rms_lohi

    def _energy_at(i: int) -> float:
        a = _norm_in_clip(action[i], ae_lo, ae_hi) if i < len(action) else None
        r = _norm_in_clip(rms[i], rms_lo, rms_hi) if i < len(rms) else None
        magnitude = ((abs(dx[i]) if i < len(dx) else 0.0) + (abs(dy[i]) if i < len(dy) else 0.0)
                    + (abs(dz[i]) if i < len(dz) else 0.0))
        cam = min(1.0, magnitude / max(REGIME_MAGNITUDE_MOVE_MIN, 1e-6))
        return (a or 0.0) + (r or 0.0) + cam

    best_i = max(range(n), key=_energy_at)
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


# --------------------------------------------------------------------------
# cut_structure_and_scene_specificity.plan.md Part 1, Step 1: structure-first
# camera-regime classification -- WHERE it is visually clean to cut, computed
# from camera + blur ONLY, no content signal at all. Regime boundaries +
# premium seams (whips, blur spikes, move onset/offset, existing
# transition_points) are the candidate clean cut points. A seam alone never
# invents a cut -- Step 3 (_snap_edge_to_seam) only ever SNAPS an
# already-content-chosen edge to the nearest one.
# --------------------------------------------------------------------------

def _camera_regime_at(motion: dict, i: int,
                      move_threshold: float = REGIME_MAGNITUDE_MOVE_MIN) -> Tuple[str, str]:
    """(stability_regime, move_regime) at GLOBAL motion hop index ``i``:
      stability_regime: "steady" | "transient" (whip/bump -- low camera_stability)
      move_regime: "static-hold" | "coherent-move" | "shake"

    ``move_threshold`` is this clip's own clip-relative move-magnitude gate
    (Part 1, see ``_clip_move_threshold``) -- defaults to the old absolute
    REGIME_MAGNITUDE_MOVE_MIN so a caller that never computed one (e.g. a
    direct unit-test call) keeps the pre-Part-1 behavior exactly."""
    stability = motion.get("camera_stability") or []
    dx = motion.get("camera_dx") or []
    dy = motion.get("camera_dy") or []
    dz = motion.get("camera_zoom") or []
    coh = motion.get("camera_coherence") or []

    def _at(arr: List[float], default: float) -> float:
        return arr[i] if i < len(arr) else default

    stability_regime = "transient" if _at(stability, 1.0) <= REGIME_STABILITY_TRANSIENT_MAX else "steady"

    magnitude = abs(_at(dx, 0.0)) + abs(_at(dy, 0.0)) + abs(_at(dz, 0.0))
    if magnitude < move_threshold:
        move_regime = "static-hold"
    elif _at(coh, 0.0) >= REGIME_COHERENCE_MOVE_MIN:
        move_regime = "coherent-move"
    else:
        move_regime = "shake"
    return stability_regime, move_regime


def _move_is_content(
    move_core: Tuple[int, int], motion: dict, hop_ms: int,
    ae_lohi: Tuple[Optional[float], Optional[float]], rms_lohi: Tuple[Optional[float], Optional[float]],
) -> bool:
    """True when a camera-move core overlaps real action/rms energy (peak
    clip-normalized value >= DEAD_ENERGY_FLOOR anywhere inside it) -- a
    content-bearing move, not free structure. A coherent-move that's
    content is never treated as disposable padding the way a static,
    silent hold's edges are (below)."""
    s, e = move_core
    action = _span_slice(motion.get("action_energy") or [], hop_ms, s, e)
    if _peak_norm(action, *ae_lohi) >= DEAD_ENERGY_FLOOR:
        return True
    rms = _span_slice(motion.get("_rms_at_motion_hop") or [], hop_ms, s, e)
    return _peak_norm(rms, *rms_lohi) >= DEAD_ENERGY_FLOOR


def _structural_seams(
    motion: dict, scene: dict, span: Tuple[int, int], hop_ms: int,
    ae_lohi: Tuple[Optional[float], Optional[float]], rms_lohi: Tuple[Optional[float], Optional[float]],
    move_threshold: float = REGIME_MAGNITUDE_MOVE_MIN,
) -> List[int]:
    """Every candidate CLEAN cut point in ``span``: camera/blur regime-
    boundary instants, transient (whip/bump) instants, blur spikes, move
    onset/offset, and existing ``transition_points``. A transient/blur
    instant that falls INSIDE ongoing continuous content (see
    ``_inside_continuous_content``) is excluded -- that reads as motion blur
    from a real pan/rally/scene, not a clean edit point; a move's own
    onset/offset is still always a seam regardless (the moment right
    before/after a deliberate move genuinely is a clean cut point)."""
    s, e = span
    if hop_ms <= 0 or e <= s:
        return []
    lo_i, hi_i = s // hop_ms, (e - 1) // hop_ms
    n = hi_i - lo_i + 1
    if n <= 0:
        return []

    content_moves = [core for core in _camera_move_cores(motion, span, hop_ms, move_threshold)
                     if _move_is_content(core, motion, hop_ms, ae_lohi, rms_lohi)]
    action_runs = _action_runs_and_lulls(span, motion, hop_ms, ae_lohi)
    composition_ts = [int(p["ts_ms"]) for p in (scene.get("composition_points") or [])
                      if s <= int(p.get("ts_ms", -1)) < e]

    def _inside_continuous_content(ts: int) -> bool:
        """cuts_content_first_segmentation.plan.md Part 5: protects a
        candidate seam not just inside a content-bearing camera move (the
        original mechanism) but ALSO when action is sustained (Part 3's own
        run/lull structure -- inside a run means away from a lull edge) OR
        composition drift is low right here (no composition_point sits
        within SNAP_MS_FLOOR -- reuses the same "genuine sliver" scale
        _snap_edge_to_seam already uses, rather than inventing a new one;
        only engages when this file actually HAS composition points to
        check against, never a blanket protect-everything default)."""
        if any(c0 <= ts < c1 for c0, c1 in content_moves):
            return True
        if any(r0 <= ts < r1 for r0, r1 in action_runs):
            return True
        return bool(composition_ts) and not any(abs(ts - cp) <= SNAP_MS_FLOOR for cp in composition_ts)

    blur = motion.get("blur") or []
    seams: set = set()
    prev_stab: Optional[str] = None
    prev_move: Optional[str] = None
    for k in range(n):
        gi = lo_i + k
        ts = gi * hop_ms
        stab_regime, move_regime = _camera_regime_at(motion, gi, move_threshold)
        content_here = _inside_continuous_content(ts)
        if stab_regime == "transient" and not content_here:
            seams.add(ts)
        if prev_stab is not None and stab_regime != prev_stab and not content_here:
            seams.add(ts)
        if prev_move is not None and move_regime != prev_move:
            seams.add(ts)                                     # move onset/offset: always a seam
        if gi < len(blur) and blur[gi] >= REGIME_BLUR_MAX and not content_here:
            seams.add(ts)
        prev_stab, prev_move = stab_regime, move_regime

    for p in (motion.get("transition_points") or []):
        ts = int(p.get("ts_ms", -1))
        if s <= ts < e:
            seams.add(ts)

    return sorted(t for t in seams if s <= t < e)


def _snap_edge_to_seam(edge_ms: int, far_ms: int, seams: List[int]) -> int:
    """Step 3's 25%+ms-floor reconcile rule: among ``seams``, the one
    nearest ``edge_ms`` snaps it there IF that seam sits within SNAP_FRAC of
    the [edge_ms, far_ms] window's own duration from ``edge_ms`` AND the
    resulting trim is under SNAP_MS_FLOOR ms (a genuine sliver -- a clean
    trim that loses almost nothing). Otherwise the seam sits too far into
    real content -- ``edge_ms`` is left untouched; a cut is never forced at
    a seam that hasn't earned it."""
    if not seams:
        return edge_ms
    window_ms = abs(far_ms - edge_ms)
    if window_ms <= 0:
        return edge_ms
    nearest = min(seams, key=lambda t: abs(t - edge_ms))
    sliver_ms = abs(nearest - edge_ms)
    if sliver_ms == 0:
        return edge_ms
    if sliver_ms / window_ms < SNAP_FRAC and sliver_ms < SNAP_MS_FLOOR:
        return nearest
    return edge_ms


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


def _broad_window_for_event(
    event: Dict[str, Any], motion: dict, hop_ms: int, span: Tuple[int, int], seams: List[int],
    ae_lohi: Tuple[Optional[float], Optional[float]] = (None, None),
    rms_lohi: Tuple[Optional[float], Optional[float]] = (None, None),
) -> Tuple[int, int]:
    """The event's own BROAD (energy=0) window -- floor/MAX_PAD-clamped, then
    Step 3 reconciled (each edge snapped to the nearest structural seam only
    when it's a genuine sliver -- see ``_snap_edge_to_seam``). Backward
    compatible by construction for a cluster of one event when ``seams`` is
    empty (no camera/blur structure in this span at all): byte-identical to
    the pre-Part-1 ``_point_edges`` formula. A span event's own core,
    clamped to the working span (never snapped -- its onset/offset already
    ARE a structural seam by construction, see ``_structural_seams``); the
    representative-window fallback's own bounds verbatim (already inside
    the span).

    cut_content_first_segmentation.plan.md Part 4: RUN_UP_FLOOR_MS/
    FOLLOW_THROUGH_FLOOR_MS only apply across time that itself carries energy
    (``_has_energy_in``) -- the raw decay-walk bound (onset_ms/settle_ms) is
    always honored regardless (that reach IS the event, floor or not); it's
    only the FLOOR'S OWN extra reach *beyond* that raw bound (a sharp event
    whose natural decay is narrower than the floor) that's now conditional.
    A dead sub-region there (the "camera-start-still at the padding level"
    bug) no longer gets padded into -- the edge trims back to the raw bound."""
    kind = event.get("kind")
    if kind == "span":
        s0, e0 = event["span_ms"]
        return max(span[0], int(s0)), min(span[1], int(e0))
    if kind == "none":
        return event["onset_ms"], event["settle_ms"]
    peak = event["peak_ms"]
    raw_onset, raw_settle = event["onset_ms"], event["settle_ms"]
    run_up = max(RUN_UP_FLOOR_MS, min(MAX_PAD_MS, peak - raw_onset))
    follow_through = max(FOLLOW_THROUGH_FLOOR_MS, min(MAX_PAD_MS, raw_settle - peak))
    in_ms = max(span[0], peak - run_up)
    out_ms = min(span[1], peak + follow_through)
    # Floor-only reach: the stretch beyond the raw decay bound the floor
    # alone is responsible for. Non-empty only when the floor pushed past
    # what the curve itself decayed to.
    if in_ms < raw_onset and not _has_energy_in(motion, hop_ms, in_ms, raw_onset, ae_lohi, rms_lohi):
        in_ms = raw_onset
    if out_ms > raw_settle and not _has_energy_in(motion, hop_ms, raw_settle, out_ms, ae_lohi, rms_lohi):
        out_ms = raw_settle
    in_ms = _snap_edge_to_seam(in_ms, peak, seams)
    out_ms = _snap_edge_to_seam(out_ms, peak, seams)
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


def _composition_events_for_span(span: Tuple[int, int], scene: dict) -> List[Dict[str, Any]]:
    """cuts_content_first_segmentation.plan.md Part 2: scene.composition_points
    (soft within-shot content change -- HS-histogram drift, already
    thresholded/spaced by L1's scene_detect stage) as first-class point
    events, so a static-camera content change (a machine advancing to a new
    operation, a subject entering) can open a boundary even when action and
    camera are both quiet. Each survivor's onset/settle is set DIRECTLY to
    RUN_UP_FLOOR_MS/FOLLOW_THROUGH_FLOOR_MS (not a zero-width raw reach) --
    deliberately independent of local action/rms/camera amplitude (Part 4's
    energy gate): composition drift IS this event's own justification, and
    doesn't need a second, redundant proof from a channel it was
    specifically added to cover for. Subject to the SAME discrete
    periodicity discipline _novelty_curve applies to action_points/onsets
    (``_evenly_spaced``) -- an evenly-spaced run (a strobe/blinking reframe)
    is suppressed entirely rather than spamming one boundary per flicker.
    The underlying drift CURVE isn't persisted (only these thresholded
    points), so unlike action/audio this can't drive a prominence-ranked
    curve -- every surviving point is used as-is."""
    s, e = span
    points = [p for p in (scene.get("composition_points") or [])
             if isinstance(p, dict) and s < int(p.get("ts_ms", -1)) < e]
    if not points:
        return []
    ts_list = sorted(int(p["ts_ms"]) for p in points)
    if _evenly_spaced(ts_list):
        return []
    return [
        {"peak_ms": int(p["ts_ms"]), "score": float(p.get("score", 1.0)), "kind": "point",
         "onset_ms": int(p["ts_ms"]) - RUN_UP_FLOOR_MS, "settle_ms": int(p["ts_ms"]) + FOLLOW_THROUGH_FLOOR_MS,
         "span_ms": None}
        for p in points
    ]


def _action_runs_and_lulls(
    span: Tuple[int, int], motion: dict, hop_ms: int,
    ae_lohi: Tuple[Optional[float], Optional[float]],
) -> List[Tuple[int, int]]:
    """cuts_content_first_segmentation.plan.md Part 3: sustained action-
    energy STRUCTURE as a content anchor, alongside isolated novelty peaks.
    Constant-motion footage (machinery, a continuous rally) has no isolated
    peak to anchor on -- nothing "surprises" a flat, continuously-elevated
    signal, so the novelty curve stays near zero throughout. A contiguous
    clip-relative-elevated stretch (>= ACTION_RUN_MIN_MS) is one candidate
    moment; a drop below LULL_LEVEL_FRACTION of the RUN's OWN mean level
    (not the clip baseline) sustained for >= LULL_MIN_MS -- a lull -- splits
    it into separate moments. Returns raw (start_ms, end_ms) candidates,
    unfiltered against what the novelty-curve/camera-move events already
    cover -- the caller subtracts those (this mechanism only fills the gaps
    they leave, never double-counts a burst those already anchor)."""
    s, e = span
    action = _span_slice(motion.get("action_energy") or [], hop_ms, s, e)
    ae_lo, ae_hi = ae_lohi
    norm = [(_norm_in_clip(v, ae_lo, ae_hi) or 0.0) for v in action]
    n = len(norm)
    if n == 0:
        return []

    active = [v >= ACTION_RUN_BASELINE_FRACTION for v in norm]
    out: List[Tuple[int, int]] = []
    for run_s, run_e in _true_runs(active):
        if (run_e - run_s) * hop_ms < ACTION_RUN_MIN_MS:
            continue
        run_level = sum(norm[run_s:run_e]) / (run_e - run_s)
        if run_level <= 0:
            continue
        lull_thr = LULL_LEVEL_FRACTION * run_level
        low = [norm[i] < lull_thr for i in range(run_s, run_e)]
        lulls = [(run_s + a, run_s + b) for a, b in _true_runs(low)
                if (b - a) * hop_ms >= LULL_MIN_MS]
        cursor = run_s
        for lull_s, lull_e in lulls:
            if lull_s > cursor:
                out.append((s + cursor * hop_ms, s + lull_s * hop_ms))
            cursor = lull_e
        if cursor < run_e:
            out.append((s + cursor * hop_ms, s + run_e * hop_ms))
    return out


def _events_for_span(
    span: Tuple[int, int], motion: dict, audio: dict, scene: dict, hop_ms: int,
    ae_lohi: Tuple[Optional[float], Optional[float]], rms_lohi: Tuple[Optional[float], Optional[float]],
    move_threshold: float = REGIME_MAGNITUDE_MOVE_MIN,
) -> Tuple[List[Dict[str, Any]], float]:
    """Every salient event in one working span, plus this span's own density
    stat. Point events (transition seams + novelty peaks + composition
    changes) and span events (every sustained camera move) all coexist --
    never a first-match early-exit. Falls back to one synthetic kind="none"
    representative-window event only when nothing else fired at all AND the
    span isn't dead (_representative_window returns None on a dead span --
    see there); a genuinely dead span then contributes NO events, and so no
    cut, at all."""
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

    # Composition changes (Part 2) -- static-camera content boundaries
    # invisible to action/camera.
    events.extend(_composition_events_for_span(span, scene))

    events = _dedupe_point_events(events)

    # EVERY sustained camera move -- span events (never just the longest).
    for core_s, core_e in _camera_move_cores(motion, span, hop_ms, move_threshold):
        coh = _span_slice(motion.get("camera_coherence") or [], hop_ms, core_s, core_e)
        score = max(0.0, min(1.0, _mean(coh) or 0.0))
        events.append({"peak_ms": core_s + (core_e - core_s) // 2, "score": score,
                       "kind": "span", "onset_ms": core_s, "settle_ms": core_e,
                       "span_ms": [core_s, core_e]})

    # Part 3: action runs/lulls -- fills gaps the novelty-curve/camera-move
    # events above don't already cover (constant-motion content with no
    # isolated peak). Subtracted against every event claimed so far, so a
    # short isolated burst (already fully covered by its own point event's
    # onset/settle) never also spawns a redundant run event.
    ae_lo, ae_hi = ae_lohi
    claimed = [(ev["onset_ms"], ev["settle_ms"]) for ev in events]
    for run_s, run_e in _action_runs_and_lulls(span, motion, hop_ms, ae_lohi):
        for frag_s, frag_e in _subtract((run_s, run_e), claimed):
            if frag_e - frag_s < ACTION_RUN_LEFTOVER_MIN_MS:
                continue
            frag_action = _span_slice(motion.get("action_energy") or [], hop_ms, frag_s, frag_e)
            frag_norm = [(_norm_in_clip(v, ae_lo, ae_hi) or 0.0) for v in frag_action]
            score = max(0.0, min(1.0, _mean(frag_norm) or 0.0))
            events.append({"peak_ms": frag_s + (frag_e - frag_s) // 2, "score": score,
                           "kind": "span", "onset_ms": frag_s, "settle_ms": frag_e,
                           "span_ms": [frag_s, frag_e]})

    if not events:
        window = _representative_window(motion, span, hop_ms, ae_lohi, rms_lohi)
        if window is not None:
            win_s, win_e = window
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


def _cluster_events(
    events: List[Dict[str, Any]], motion: dict, hop_ms: int, span: Tuple[int, int], seams: List[int],
    ae_lohi: Tuple[Optional[float], Optional[float]] = (None, None),
    rms_lohi: Tuple[Optional[float], Optional[float]] = (None, None),
) -> List[List[Dict[str, Any]]]:
    """Group one working span's events into clusters by their own BROAD
    (energy=0) window gaps -- the same window ``_broad_window_for_event``
    already computes for cluster-extent purposes, so "close enough to fuse
    at the broadest window" is judged on the exact (seam-reconciled) windows
    that fusion would use. Always >= 1 cluster when ``events`` is non-empty."""
    if not events:
        return []
    windows = sorted(((_broad_window_for_event(ev, motion, hop_ms, span, seams, ae_lohi, rms_lohi), ev)
                      for ev in events), key=lambda x: x[0][0])
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
    move_threshold = _clip_move_threshold(motion)
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
        # Step 1 (structure) computed once per working span, independent of
        # content; Step 2 (content, _events_for_span) unchanged in spirit;
        # Step 3 (reconcile) happens inside _broad_window_for_event, which
        # every window below already routes through.
        seams = _structural_seams(motion, scene, span, hop_ms, ae_lohi, rms_lohi, move_threshold)
        events, density = _events_for_span(span, motion, audio, scene, hop_ms, ae_lohi, rms_lohi, move_threshold)
        for cluster in _cluster_events(events, motion, hop_ms, span, seams, ae_lohi, rms_lohi):
            windows = [_broad_window_for_event(ev, motion, hop_ms, span, seams, ae_lohi, rms_lohi)
                      for ev in cluster]
            in_ms = max(span[0], min(w[0] for w in windows))
            out_ms = min(span[1], max(w[1] for w in windows))
            if out_ms <= in_ms:
                continue
            cuts.append(VideoCut(file_id=file_id, src_in_ms=in_ms, src_out_ms=out_ms,
                                 salience=_build_salience(cluster, density), density=density,
                                 span_key=span))
    cuts.sort(key=lambda c: c.src_in_ms)
    return _finalize_cuts(cuts)
