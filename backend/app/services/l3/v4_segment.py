"""
Cuts V4 -- the deterministic video segmenter (cuts_v4_segmentation.plan.md).

Replaces the video half of pass 1's job (grouping atoms into
``VideoTentativeGroup``s) with a signal-driven extractor built on one
principle: a raw clip is mostly scrap -- find the small usable part(s) and
discard the rest. Default is to trim hard to the usable core, never "keep the
whole clip". Speech is untouched (pass 1 still owns speech grouping + junk);
this module only decides the NON-SPEECH remainder's cuts.

Two rules threaded throughout, per the plan:
  * Salience = contrast/novelty (how much a moment stands out from its LOCAL
    surroundings), not absolute level, and not requiring audio+motion
    consensus -- either channel alone can produce a point; agreement only
    raises confidence (they simply add).
  * The VLM (pass 2, elsewhere) decides *shape* (semantic); this module (code)
    always decides *where* -- location is deterministic.

Pure core: ``segment_video(...)`` takes already-loaded signals (the same
``motion``/``scene`` shapes ``lattice.build_atoms`` consumes) and a built
``Lattice`` (for atom_ids mapping only -- V4 does not carve its own atoms; it
chooses spans directly on the motion hop grid, independent of the atom
lattice's coarser shot/energy-regime boundaries). No model call, no DB call --
see ``scripts/test_v4_segment.py``.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.lattice import Lattice, _subtract
from app.services.l3.post import _mean, _norm_in_clip, _series_lohi, _span_slice
from app.services.l3.v4_segment_params import (
    CAMERA_MOVE_COHERENCE_MIN, CAMERA_MOVE_MAGNITUDE_MIN, CAMERA_MOVE_MIN_MS,
    DECAY_FRACTION, DENSITY_PEAKS_PER_SEC_CAP, EDGE_SNAP_SEARCH_MS,
    EDGE_SNAP_STABILITY_MAX, FOLLOW_THROUGH_FLOOR_MS, MAX_PAD_MS, MIN_CUT_GAP_MS,
    NOVELTY_ABSOLUTE_FLOOR, NOVELTY_BASELINE_RADIUS_MS, PEAK_MIN_GAP_MS,
    PEAK_PROMINENCE_RATIO, PERIODICITY_SCORE_THRESHOLD, REPRESENTATIVE_WINDOW_MS,
    RUN_UP_FLOOR_MS,
)


@dataclass
class VideoCut:
    file_id: str
    src_in_ms: int
    src_out_ms: int
    atom_ids: List[int] = field(default_factory=list)
    # {peak_ms, score, kind: "point"|"span"|"none", span_ms: [in,out]|None} --
    # see post._salience's shape for the V3 analogue; V4 emits this directly
    # (code-owned, never recomputed downstream -- section 4 of the plan).
    salience: Dict[str, Any] = field(default_factory=dict)
    # Event density/novelty stat (0..1) feeding post.compute_pace_envelope's
    # content-aware min_ms (section 6): sparse/monotonous -> collapses hard;
    # dense -> holds more room at the same energy.
    density: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"file_id": self.file_id, "src_in_ms": self.src_in_ms,
                "src_out_ms": self.src_out_ms, "atom_ids": list(self.atom_ids),
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
# Step 3: anchor selection
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


def _camera_move_core(motion: dict, span: Tuple[int, int], hop_ms: int) -> Optional[Tuple[int, int]]:
    """(start_ms, end_ms) of the dominant sustained, coherent camera move in
    ``span``, or None -- Step 3.3. A hop counts as "moving" once its combined
    |dx|+|dy|+|zoom| clears CAMERA_MOVE_MAGNITUDE_MIN AND coherence clears
    CAMERA_MOVE_COHERENCE_MIN (deliberate, not shake); the longest such run
    must sustain CAMERA_MOVE_MIN_MS to count as a real payload."""
    s, e = span
    dx = _span_slice(motion.get("camera_dx") or [], hop_ms, s, e)
    dy = _span_slice(motion.get("camera_dy") or [], hop_ms, s, e)
    dz = _span_slice(motion.get("camera_zoom") or [], hop_ms, s, e)
    coh = _span_slice(motion.get("camera_coherence") or [], hop_ms, s, e)
    n = max(len(dx), len(dy), len(dz))
    if n == 0:
        return None

    def _at(arr: List[float], i: int, default: float) -> float:
        return arr[i] if i < len(arr) else default

    moving = [
        (abs(_at(dx, i, 0.0)) + abs(_at(dy, i, 0.0)) + abs(_at(dz, i, 0.0))) >= CAMERA_MOVE_MAGNITUDE_MIN
        and _at(coh, i, 0.0) >= CAMERA_MOVE_COHERENCE_MIN
        for i in range(n)
    ]
    runs = _true_runs(moving)
    if not runs:
        return None
    start_i, end_i = max(runs, key=lambda r: r[1] - r[0])
    if (end_i - start_i) * hop_ms < CAMERA_MOVE_MIN_MS:
        return None
    return s + start_i * hop_ms, min(e, s + end_i * hop_ms)


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
# Step 4: edges
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


def _point_edges(curve: List[float], span: Tuple[int, int], hop_ms: int,
                  peak_i: int, motion: dict) -> Tuple[int, int]:
    s, _e = span
    n = len(curve)
    back_i = _decay_bound(curve, peak_i, -1, 0, n - 1)
    fwd_i = _decay_bound(curve, peak_i, +1, 0, n - 1)
    run_up = max(RUN_UP_FLOOR_MS, min(MAX_PAD_MS, (peak_i - back_i) * hop_ms))
    follow_through = max(FOLLOW_THROUGH_FLOOR_MS, min(MAX_PAD_MS, (fwd_i - peak_i) * hop_ms))
    peak_ms = s + peak_i * hop_ms
    in_ms = max(span[0], peak_ms - run_up)
    out_ms = min(span[1], peak_ms + follow_through)
    in_ms = _snap_to_quality_gate(motion, hop_ms, in_ms, span[0], peak_ms)
    out_ms = _snap_to_quality_gate(motion, hop_ms, out_ms, peak_ms, span[1])
    return min(in_ms, peak_ms), max(out_ms, peak_ms)


# --------------------------------------------------------------------------
# Step 5/6: per-working-span anchors -> candidate cuts, consolidated
# --------------------------------------------------------------------------

def _novelty_density(curve: List[float], hop_ms: int) -> float:
    if not curve or hop_ms <= 0:
        return 0.0
    min_gap_hops = max(1, PEAK_MIN_GAP_MS // hop_ms)
    peaks = _find_peaks(curve, min_gap_hops)
    dur_s = max(0.001, len(curve) * hop_ms / 1000.0)
    return max(0.0, min(1.0, len(peaks) / dur_s / DENSITY_PEAKS_PER_SEC_CAP))


def _score_at(curve: List[float], i: int) -> float:
    lo, hi = min(curve), max(curve)
    if hi - lo < 1e-9:
        return 1.0 if curve[i] > 0 else 0.0
    return max(0.0, min(1.0, (curve[i] - lo) / (hi - lo)))


def _candidates_for_span(
    span: Tuple[int, int], motion: dict, audio: dict, hop_ms: int,
    ae_lohi: Tuple[Optional[float], Optional[float]], rms_lohi: Tuple[Optional[float], Optional[float]],
) -> List[Tuple[int, int, Dict[str, Any]]]:
    """[(in_ms, out_ms, salience_dict), ...] for one working span, anchors
    chosen in preference order (Step 3), edges carved per anchor kind
    (Step 4). Always >= 1 candidate (the representative-window fallback)."""
    s, e = span
    density_curve = _novelty_curve(span, motion, audio, hop_ms, ae_lohi, rms_lohi)
    density = _novelty_density(density_curve, hop_ms)

    # 1. Transition points strictly inside the span -- a premium natural seam.
    transitions = [int(p["ts_ms"]) for p in (motion.get("transition_points") or [])
                   if s < int(p.get("ts_ms", -1)) < e]
    out: List[Tuple[int, int, Dict[str, Any]]] = []
    if transitions and density_curve:
        for ts in transitions:
            i = min(max((ts - s) // hop_ms, 0), len(density_curve) - 1)
            in_ms, out_ms = _point_edges(density_curve, span, hop_ms, i, motion)
            out.append((in_ms, out_ms, {"peak_ms": ts, "score": _score_at(density_curve, i),
                                         "kind": "point", "span_ms": None}))

    # 2. Novelty peaks clearing the span's own prominence bar.
    if not out and density_curve:
        peaks = _prominent_peaks(density_curve, hop_ms)
        for i in peaks:
            in_ms, out_ms = _point_edges(density_curve, span, hop_ms, i, motion)
            peak_ms = s + i * hop_ms
            out.append((in_ms, out_ms, {"peak_ms": peak_ms, "score": _score_at(density_curve, i),
                                         "kind": "point", "span_ms": None}))

    # 3. Camera-move payload -- a SPAN anchor, not a point.
    if not out:
        core = _camera_move_core(motion, span, hop_ms)
        if core is not None:
            core_s, core_e = core
            coh = _span_slice(motion.get("camera_coherence") or [], hop_ms, core_s, core_e)
            score = max(0.0, min(1.0, _mean(coh) or 0.0))
            out.append((core_s, core_e, {"peak_ms": core_s + (core_e - core_s) // 2, "score": score,
                                          "kind": "span", "span_ms": [core_s, core_e]}))

    # 4. Fallback: representative window, salience kind="none".
    if not out:
        win_s, win_e = _representative_window(motion, span, hop_ms)
        out.append((win_s, win_e, {"peak_ms": win_s + (win_e - win_s) // 2, "score": 0.0,
                                    "kind": "none", "span_ms": None}))

    for i in range(len(out)):
        s0, e0, sal = out[i]
        out[i] = (s0, e0, dict(sal, density=density))
    return out


def _consolidate(cuts: List[Tuple[int, int, Dict[str, Any]]]) -> List[Tuple[int, int, Dict[str, Any]]]:
    """Merge any two candidates whose gap (or overlap) is below
    MIN_CUT_GAP_MS, keeping the stronger anchor's salience, spanning the
    union -- Step 5. Guarantees zero overlap as a side effect (any actual
    overlap has a negative gap, which is always < the floor)."""
    ordered = sorted(cuts, key=lambda c: c[0])
    out: List[Tuple[int, int, Dict[str, Any]]] = []
    for c in ordered:
        if out and c[0] - out[-1][1] < MIN_CUT_GAP_MS:
            prev = out[-1]
            keep_sal = prev[2] if prev[2].get("score", 0.0) >= c[2].get("score", 0.0) else c[2]
            out[-1] = (min(prev[0], c[0]), max(prev[1], c[1]), keep_sal)
        else:
            out.append(c)
    return out


def _atom_ids_covering(lattice: Lattice, in_ms: int, out_ms: int) -> List[int]:
    """Atoms (from the pre-existing lattice, shared with pass 1) whose span
    overlaps [in_ms, out_ms) -- informational only (continuity/image_plan
    compatibility per section 2 of the plan); V4's own span, not these atoms'
    bounds, is what gets persisted -- see ingest.run_ingest's V4 branch. Atoms
    tile the WHOLE non-speech remainder with zero gaps, so this is normally
    non-empty for any real V4 span; falls back to the single nearest atom
    (never truly empty while the file has any atom at all) so a downstream
    consumer that requires >=1 atom_id per video cut (pass2.backfill_locators'
    "split group" check) never sees a spuriously empty list."""
    ids = sorted(a.atom_id for a in lattice.atoms if a.start_ms < out_ms and a.end_ms > in_ms)
    if ids or not lattice.atoms:
        return ids
    mid = (in_ms + out_ms) // 2
    nearest = min(lattice.atoms, key=lambda a: min(abs(a.start_ms - mid), abs(a.end_ms - mid)))
    return [nearest.atom_id]


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def segment_video(
    *, file_id: str, duration_ms: int, speech_spans: List[Tuple[int, int]],
    motion: Dict[str, Any], audio: Dict[str, Any], scene: Dict[str, Any], lattice: Lattice,
) -> List[VideoCut]:
    """The non-speech remainder of one file -> a small set of tight, salient
    video cuts (never the whole span by default). Pure and deterministic:
    same signals always produce the same cuts."""
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
        candidates = _candidates_for_span(span, motion, audio, hop_ms, ae_lohi, rms_lohi)
        for in_ms, out_ms, sal in _consolidate(candidates):
            if out_ms <= in_ms:
                continue
            density = sal.pop("density", 0.0)
            atom_ids = _atom_ids_covering(lattice, in_ms, out_ms)
            cuts.append(VideoCut(file_id=file_id, src_in_ms=in_ms, src_out_ms=out_ms,
                                 atom_ids=atom_ids, salience=sal, density=density))
    cuts.sort(key=lambda c: c.src_in_ms)
    return cuts
