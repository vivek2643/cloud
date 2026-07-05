"""
Cuts v2, Phases C1 + C3a: camera-move-state video segmentation + hard shot cuts.

Follow-on to ``cuts_v2.plan.md``'s ``partition.py`` baseline. Replaces the
action-impact-WINDOW ``done`` detector and the scene/shot-bounded ``shown``
detector (both "known-imperfect" per that plan) with ONE segmentation of a
clip's video track:

  * a hard SHOT CUT (Phase B1's scene detection, re-enabled in Phase C3) is a
    TOP-PRIORITY boundary at every granularity -- multi-shot footage must
    never merge two different shots into one segment, not even at Broad.
  * within one shot, a SETTLE (the start of a camera hold) is the next
    boundary -- "cut on the still, not mid-move" -- so a pan/push plays
    through to where it lands, not chopped mid-motion.
  * a subject-motion BEAT is an additional, dial-gated split source -- and the
    ONLY split source at all when a shot never moves the camera. The
    static/locked-off case therefore isn't a separate code path: a clip (or a
    shot within it) that never leaves HOLD state is just one big camera-
    segment that the same beat-splitting step then subdivides, same as any
    other segment.

The hold/move STATE MACHINE and shot-cut detection are both energy-
independent (real, deterministic detection). GRANULARITY -- how many of the
detected settles/beats the dial admits as extra split points -- is energy-
dependent, a deliberate departure from ``cuts_v2.plan.md``'s "detect once"
North Star (see that plan's "Honest risks" #1); a shot cut is never gated by
energy, unlike a settle or a beat. TAGGING (done vs shown) is PROVISIONAL:
real classification waits on the image pass (Phase C3, still deferred); this
only keeps the two tags meaningful meanwhile.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from app.services.l3 import vocab
from app.services.l1.cut_grid_common import local_maxima, percentile
from app.services.l3.partition_params import GRAN_SPLIT_MIN_MS, MIN_SUBUNIT_MS
from app.services.l3.video_segment_params import (
    BEAT_FLOOR_PCTL_BALANCED,
    BEAT_FLOOR_PCTL_SHARP,
    BEAT_FLOOR_PCTL_TIGHT,
    BEAT_MIN_DYNAMIC_RANGE,
    BEAT_MIN_GAP_MS,
    BROAD_HOLD_MIN_MS,
    DONE_TAG_MIN_DYNAMIC_RANGE,
    DONE_TAG_PCTL,
    HOLD_MIN_MS,
    HYSTERESIS_HOPS,
    MOTION_HOLD_MAX,
    STABILITY_HOLD_MIN,
)

HOLD, MOVE = "hold", "move"


@dataclass
class VideoSegment:
    start_ms: int
    end_ms: int
    peak_ms: int          # representative frame instant (thumbnail)
    tag: str              # vocab.CHANNEL_DONE | vocab.CHANNEL_SHOWN (provisional)


# --------------------------------------------------------------------------
# Hold / move state machine (energy-independent)
# --------------------------------------------------------------------------

def _hop_states(motion: dict) -> List[str]:
    """Per-hop RAW hold/move classification. A hop is a HOLD candidate when
    the camera is both steady (absolute stability, reliable across clips) and
    not moving much (file-relative camera_motion, a known-imperfect proxy)."""
    stability = motion.get("camera_stability") or []
    camera_motion = motion.get("camera_motion") or []
    n = max(len(stability), len(camera_motion))
    states: List[str] = []
    for i in range(n):
        stab = stability[i] if i < len(stability) else 0.0
        mot = camera_motion[i] if i < len(camera_motion) else 1.0
        states.append(HOLD if (stab >= STABILITY_HOLD_MIN and mot <= MOTION_HOLD_MAX) else MOVE)
    return states


def _confirm_hysteresis(raw_states: List[str]) -> List[str]:
    """A CONFIRMED state only flips once HYSTERESIS_HOPS consecutive hops
    agree on the new raw state -- stops a single noisy hop from flapping the
    segmentation. Tracks a streak of the RAW value itself (not "raw vs
    current"), so the streak actually accumulates while the confirmed state
    hasn't flipped yet."""
    if not raw_states:
        return []
    confirmed = [raw_states[0]] * len(raw_states)
    current = raw_states[0]
    streak_val = raw_states[0]
    streak_len = 1
    for i in range(1, len(raw_states)):
        if raw_states[i] == streak_val:
            streak_len += 1
        else:
            streak_val, streak_len = raw_states[i], 1
        if streak_val != current and streak_len >= HYSTERESIS_HOPS:
            current = streak_val
        confirmed[i] = current
    return confirmed


def _runs(states: List[str], hop_ms: int) -> List[Tuple[str, int, int]]:
    """Confirmed per-hop states -> run-length-encoded (state, start_ms, end_ms)."""
    if not states:
        return []
    out: List[Tuple[str, int, int]] = []
    cur, start = states[0], 0
    for i in range(1, len(states)):
        if states[i] != cur:
            out.append((cur, start * hop_ms, i * hop_ms))
            cur, start = states[i], i
    out.append((cur, start * hop_ms, len(states) * hop_ms))
    return out


def _settle_points(runs: List[Tuple[str, int, int]], *, broad: bool) -> List[int]:
    """Start-of-hold timestamps that count as a real SETTLE (a camera-state
    boundary): the hold must clear HOLD_MIN_MS (the stricter
    BROAD_HOLD_MIN_MS at the Broad band, so shorter holds merge into a longer
    segment). The very first run is never a boundary -- nothing precedes it
    to settle FROM."""
    floor = BROAD_HOLD_MIN_MS if broad else HOLD_MIN_MS
    return [
        s for i, (state, s, e) in enumerate(runs)
        if i > 0 and state == HOLD and (e - s) >= floor
    ]


# --------------------------------------------------------------------------
# Subject-motion-beat sub-split (granularity, Balanced and above)
# --------------------------------------------------------------------------

def _beat_floor_pctl(band: int) -> Optional[float]:
    """Percentile floor for a subject-motion-beat sub-split, by energy band.
    None (no sub-splitting at all) at Broad/Calm -- granularity only adds
    beats from Balanced upward, per the dial mapping."""
    return {2: BEAT_FLOOR_PCTL_BALANCED, 3: BEAT_FLOOR_PCTL_TIGHT,
           4: BEAT_FLOOR_PCTL_SHARP}.get(band)


def _beat_splits(action: List[float], hop_ms: int, s: int, e: int, floor_pctl: float) -> List[int]:
    """Strong, isolated local maxima of THIS segment's own action energy,
    strictly inside (s, e) -- additional split points. A segment shorter than
    GRAN_SPLIT_MIN_MS never sub-splits (too short to be worth it)."""
    if not action or hop_ms <= 0 or e - s < GRAN_SPLIT_MIN_MS:
        return []
    lo, hi = s // hop_ms, e // hop_ms
    window = action[lo:hi + 1]
    if len(window) < 3 or max(window) - min(window) < BEAT_MIN_DYNAMIC_RANGE:
        return []
    # A percentile-rank floor alone can still collapse toward the window's own
    # flat majority when only a small slice of it is genuinely elevated (a
    # short beat inside a long calm segment) -- verified against a synthetic
    # mostly-flat-with-one-bump window, which over-split every 500ms under a
    # pure rank-based floor. Enforce a minimum absolute margin above the
    # window's own baseline (median) too.
    floor = max(percentile(window, floor_pctl),
               percentile(window, 50.0) + BEAT_MIN_DYNAMIC_RANGE)
    local = local_maxima(window, hop_ms, floor, BEAT_MIN_GAP_MS)
    pts = [lo * hop_ms + t for t in local]
    return [t for t in pts if (t - s) >= MIN_SUBUNIT_MS and (e - t) >= MIN_SUBUNIT_MS]


# --------------------------------------------------------------------------
# Peak selection + tagging (provisional)
# --------------------------------------------------------------------------

def _value_at(arr: List[float], hop_ms: int, ts_ms: int) -> float:
    if not arr or hop_ms <= 0:
        return 0.0
    i = max(0, min(len(arr) - 1, ts_ms // hop_ms))
    return arr[i]


def _seg_bounds(n: int, hop_ms: int, s: int, e: int) -> Optional[Tuple[int, int]]:
    """[s, e) (a segment's OWN span, exclusive of its end) -> inclusive sample
    index bounds, clamped to the array. `e // hop_ms` alone would include the
    NEXT segment's first sample when `e` lands exactly on a hop boundary --
    verified against a synthetic two-beat clip, where a middle segment's
    "strongest instant" search was silently picking up its neighbor's peak."""
    lo, hi = max(0, s // hop_ms), min(n - 1, (e - 1) // hop_ms)
    return None if hi < lo else (lo, hi)


def _strongest_ms(arr: List[float], hop_ms: int, s: int, e: int, default_ms: int) -> int:
    if not arr or hop_ms <= 0:
        return default_ms
    bounds = _seg_bounds(len(arr), hop_ms, s, e)
    if bounds is None:
        return default_ms
    lo, hi = bounds
    return max(range(lo, hi + 1), key=lambda i: arr[i]) * hop_ms


def _sharpest_ms(blur: List[float], hop_ms: int, s: int, e: int, default_ms: int) -> int:
    """The least-blurred instant in [s, e) -- the thumbnail-worthy frame for a
    held (shown) segment. Falls back to ``default_ms`` when blur isn't
    available."""
    if not blur or hop_ms <= 0:
        return default_ms
    bounds = _seg_bounds(len(blur), hop_ms, s, e)
    if bounds is None:
        return default_ms
    lo, hi = bounds
    return min(range(lo, hi + 1), key=lambda i: blur[i]) * hop_ms


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def segment_video(motion: Optional[dict], duration_ms: int, energy: float,
                  scene: Optional[dict] = None) -> List[VideoSegment]:
    """The clip's video track as ONE non-overlapping, full-coverage sequence
    of tag-bearing segments, at a given ``energy``. Hold/move detection and
    shot-cut boundaries never change with energy; only how many subject-beat
    sub-splits the dial admits does.

    Best-effort: no motion data at all (or no hop) still yields shot-cut-
    bounded segments (or, with no scene data either, exactly one segment
    spanning the whole clip), tagged ``shown`` (nothing to judge)."""
    from app.services.l3.energy import energy_band  # local: keep this module import-light

    motion = motion or {}
    action = motion.get("action_energy") or []
    hop = int(motion.get("hop_ms") or 0)
    blur = motion.get("blur") or []

    # Hard shot cuts (Phase C3a): a TOP-PRIORITY boundary at every
    # granularity, never gated by energy -- multi-shot footage must never
    # merge two different shots into one segment, even at Broad.
    shot_cuts = {int(p.get("ts_ms", -1)) for p in ((scene or {}).get("shot_points") or [])
                if 0 < int(p.get("ts_ms", -1)) < duration_ms}

    if not motion or not hop:
        bounds = sorted({0, duration_ms} | shot_cuts)
        return [
            VideoSegment(s, e, (s + e) // 2, vocab.CHANNEL_SHOWN)
            for s, e in zip(bounds, bounds[1:]) if e > s
        ]

    band = energy_band(energy)
    raw_states = _hop_states(motion)
    confirmed = _confirm_hysteresis(raw_states)
    runs = _runs(confirmed, hop)
    settle = _settle_points(runs, broad=(band == 0)) if runs else []

    bounds = sorted({0, duration_ms} | {t for t in settle if 0 < t < duration_ms} | shot_cuts)

    floor_pctl = _beat_floor_pctl(band)
    if floor_pctl is not None and action:
        extra = set()
        for s, e in zip(bounds, bounds[1:]):
            extra.update(_beat_splits(action, hop, s, e, floor_pctl))
        if extra:
            bounds = sorted(set(bounds) | extra)

    # A clip whose action energy barely varies has nothing to distinguish --
    # never tag `done` off a flat baseline's own percentile (it trivially
    # "clears itself"); default everything `shown` instead. And, as with the
    # per-segment beat floor above, a handful of elevated samples in an
    # otherwise-flat clip can still leave the RANK-based percentile sitting on
    # the flat majority -- enforce the same minimum absolute margin above the
    # clip's own median.
    has_action_signal = bool(action) and (max(action) - min(action)) >= DONE_TAG_MIN_DYNAMIC_RANGE
    action_ref = (
        max(percentile(action, DONE_TAG_PCTL), percentile(action, 50.0) + DONE_TAG_MIN_DYNAMIC_RANGE)
        if has_action_signal else 0.0
    )
    segs: List[VideoSegment] = []
    for s, e in zip(bounds, bounds[1:]):
        if e <= s:
            continue
        strongest = _strongest_ms(action, hop, s, e, (s + e) // 2)
        is_done = has_action_signal and action_ref > 0 and _value_at(action, hop, strongest) >= action_ref
        if is_done:
            segs.append(VideoSegment(s, e, strongest, vocab.CHANNEL_DONE))
        else:
            peak = _sharpest_ms(blur, hop, s, e, (s + e) // 2)
            segs.append(VideoSegment(s, e, peak, vocab.CHANNEL_SHOWN))
    return segs
