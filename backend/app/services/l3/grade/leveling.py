"""
Leveling layer (color_grading_upgrade.plan.md Phase 2, bounded photometric
leveling -- Step 2.1/2.2; Step 3.1 extends it to subject-aware exposure).
Composed as a new stage BETWEEN Match and Look in `resolver.py`, gated
entirely on `settings.grade_even_lighting`; runs INSIDE `run_grade_job`
(Step 1.0), never inline.

Two generic-safe ideas, both a SMOOTH-TARGET + BOUNDED-GAIN pattern:
  * `solve_exposure_leveling` (2.1): nudge each shot's exposure (subject
    luma when a usable one exists -- Step 3.1 -- else whole-frame
    `mid_gray`) toward a LOW-PASS target across the sequence. The target
    itself follows a slow intended arc (day->night), so leveling flattens
    shot-to-shot flicker without erasing that arc; the gain is capped in
    "stops" so no single shot ever gets pushed further than the bound.
  * `solve_tonal_leveling` (2.2): the same smooth-target + bounded-gain
    pattern applied to shadow/highlight PLACEMENT (not just brightness), so
    lighting reads even rather than merely equally bright. Skips shots that
    are statistical outliers from their local target (more likely a
    genuinely different scene than jitter) and never pushes black/white
    toward clipping.

Both operate on WORKING-SPACE scalars (mean/black/white/subject luma
already projected through `tone.to_working` by the caller -- this module
stays pure numeric, no tone.py dependency) and emit plain slope/offset
`Grade`s the caller composes onto the stack.

Explicit non-goals (documented, not silently dropped): temporal WITHIN-shot
stabilization and local/intra-frame relight are OUT -- both need time-
varying or spatially-varying grades the compositor can't apply today (the
same limitation unrendered video speed has), and both need a flaw-vs-intent
signal this pass doesn't have. Revisit in a render-capabilities effort.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from app.services.l3.grade.cdl import Grade, compose

# A low-pass window this wide (in SHOTS, not ms) flattens shot-to-shot
# flicker while still tracking a real slow arc across a scene.
# color_shot_matching.plan.md Phase 4b: widened from 5 -- a narrower window
# let Leveling re-diverge what Match/Balance had just converged (it was
# targeting a different, LOCAL reference than their group reference). A
# wider low-pass tracks only a slow arc, so it stops fighting them.
LEVELING_WINDOW = 9

# Never move a shot's exposure/tonal placement by more than this many
# "stops" (2**stops multiplier in working-space linear terms) -- large
# INTENDED differences (a deliberate day/night cut) survive the bound.
EXPOSURE_CAP_STOPS = 0.5
TONAL_CAP_STOPS = 0.4

# A shot whose black/white point sits further than this from its LOCAL
# smooth target is more likely a genuinely different scene than lighting
# jitter -- skip tonal leveling for it entirely rather than forcing a fit.
OUTLIER_STOPS = 1.5

# A subject_luma this many times darker/brighter than the shot's own
# whole-frame mid_gray reads as a deliberate silhouette/backlit choice, not
# something evenness should "fix" -- fall back to whole-frame mid_gray.
SILHOUETTE_RATIO = 3.0


@dataclass
class ShotLevelInput:
    """One shot's WORKING-SPACE photometric facts -- what leveling nudges.
    `subject_luma` (Step 3.1, optional) is luma inside the subject's own box
    on the hero frame; when present and not a silhouette (see
    `_usable_subject_luma`), exposure leveling targets IT instead of the
    whole-frame `mid_gray` -- the perceptually correct evenness for people
    content (a consistent face brightness, not a consistent frame average).

    `target_*` (color_shot_matching.plan.md Phase 4b escalation, optional):
    an EXPLICIT target -- a scene-group's robust reference, already solved
    by Balance/Match -- to level toward INSTEAD OF the local smooth-target
    average. Without this, Leveling's own local window average can
    re-diverge what Balance/Match just converged toward a DIFFERENT
    (robust-reference) target (verified: on an adversarial synthetic
    multi-shot group, local-window leveling alone re-introduced most of the
    spread Balance+Match had just closed). `None` (default, and always for
    singleton/ungrouped shots) keeps today's smooth-target behavior."""
    key: str
    mid_gray: float
    black_point: float
    white_point: float
    subject_luma: Optional[float] = None
    target_mid_gray: Optional[float] = None
    target_black_point: Optional[float] = None
    target_white_point: Optional[float] = None
    # color_subject_exposure.plan.md Phase 2: a scene-group's median SUBJECT
    # luma (working space), for a shot whose exposure value came from a
    # usable subject_luma (see `_usable_subject_luma`). Without this, a
    # grouped subject shot's subject_luma was leveled toward
    # `target_mid_gray` -- the group's WHOLE-FRAME reference -- a mismatch
    # (a face's own brightness has no reason to equal the frame average).
    # `None` (default, and always for ungrouped/no-subject shots) keeps
    # today's behavior: `target_mid_gray`, then the local smooth target.
    target_subject_luma: Optional[float] = None


def _smooth_target(values: List[float]) -> List[float]:
    """Centered moving average, `LEVELING_WINDOW` wide, shrinking gracefully
    at the timeline's edges (no look-ahead/behind past what's actually
    there) -- the low-pass filter that lets a real arc survive while
    shot-to-shot jitter flattens."""
    n = len(values)
    half = LEVELING_WINDOW // 2
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        window = values[lo:hi]
        out.append(sum(window) / len(window))
    return out


def _capped_gain(current: float, target: float, cap_stops: float) -> float:
    """A multiplicative gain moving `current` toward `target`, capped at
    +/- `cap_stops`."""
    if current <= 1e-6:
        return 1.0
    full_gain = target / current
    cap = 2.0 ** cap_stops
    return max(1.0 / cap, min(cap, full_gain))


def _is_outlier(shot_range: float, target_range: float, threshold_stops: float) -> bool:
    """True when a shot's CONTRAST (white-black range) is too far from its
    local smooth target's range to trust as 'the same lighting, just needs
    a nudge' -- a punchy shot dropped into a flat sequence (or vice versa)
    reads as a genuinely different scene, not jitter. Range-ratio rather
    than checking black/white independently: black_point is legitimately
    0.0 often (a true black), which breaks a per-point ratio check; a
    range is never degenerate once `white_point > black_point` already
    holds (the caller's guard before this is ever reached)."""
    if shot_range <= 1e-6 or target_range <= 1e-6:
        return False
    ratio = target_range / shot_range
    return ratio > 2.0 ** threshold_stops or ratio < 2.0 ** (-threshold_stops)


def _usable_subject_luma(subject_luma: Optional[float], mid_gray: float) -> Optional[float]:
    """Step 3.1's gate: a subject_luma far enough from the shot's OWN
    whole-frame mid_gray reads as an intentional silhouette/backlit choice,
    not a wrong exposure to correct -- treat it as unusable (falls back to
    whole-frame leveling for that shot, same as genuine b-roll)."""
    if subject_luma is None:
        return None
    if mid_gray <= 1e-6:
        return subject_luma
    ratio = subject_luma / mid_gray
    if ratio > SILHOUETTE_RATIO or ratio < 1.0 / SILHOUETTE_RATIO:
        return None
    return subject_luma


def _exposure_value(s: ShotLevelInput) -> float:
    usable = _usable_subject_luma(s.subject_luma, s.mid_gray)
    return usable if usable is not None else s.mid_gray


def solve_exposure_leveling(ordered_shots: List[ShotLevelInput]) -> Dict[str, Grade]:
    """Step 2.1 (+ Step 3.1's subject-aware extension): nudge each shot's
    exposure toward a smooth target across the sequence, bounded. <2 shots
    -> nothing to level against. Target priority per shot: `target_subject_luma`
    (color_subject_exposure.plan.md Phase 2 -- only when this shot's own
    exposure value actually came from a USABLE subject_luma, never applied
    to a whole-frame value) -> `target_mid_gray` (Phase 4b) -> the local
    smooth-target average (today's behavior)."""
    if len(ordered_shots) < 2:
        return {}
    values = [_exposure_value(s) for s in ordered_shots]
    smooth_targets = _smooth_target(values)
    targets = []
    for s, t in zip(ordered_shots, smooth_targets):
        has_usable_subject = _usable_subject_luma(s.subject_luma, s.mid_gray) is not None
        if has_usable_subject and s.target_subject_luma is not None:
            targets.append(s.target_subject_luma)
        elif s.target_mid_gray is not None:
            targets.append(s.target_mid_gray)
        else:
            targets.append(t)
    out: Dict[str, Grade] = {}
    for s, value, target in zip(ordered_shots, values, targets):
        gain = _capped_gain(value, target, EXPOSURE_CAP_STOPS)
        if abs(gain - 1.0) < 1e-4:
            continue
        out[s.key] = Grade(slope=(gain, gain, gain))
    return out


def solve_tonal_leveling(ordered_shots: List[ShotLevelInput]) -> Dict[str, Grade]:
    """Step 2.2: align each shot's shadow/highlight PLACEMENT toward the
    sequence's smooth target -- a bounded levels-style remap, skipping
    statistical outliers and never pushing toward clipping. A shot with
    `target_black_point`/`target_white_point` set (Phase 4b) uses those
    EXPLICIT targets instead of the local smooth-target average."""
    if len(ordered_shots) < 2:
        return {}
    blacks = [s.black_point for s in ordered_shots]
    whites = [s.white_point for s in ordered_shots]
    smooth_black_targets = _smooth_target(blacks)
    smooth_white_targets = _smooth_target(whites)
    black_targets = [
        s.target_black_point if s.target_black_point is not None else t
        for s, t in zip(ordered_shots, smooth_black_targets)
    ]
    white_targets = [
        s.target_white_point if s.target_white_point is not None else t
        for s, t in zip(ordered_shots, smooth_white_targets)
    ]

    out: Dict[str, Grade] = {}
    for s, b_target, w_target in zip(ordered_shots, black_targets, white_targets):
        if s.white_point <= s.black_point:
            continue
        if _is_outlier(s.white_point - s.black_point, w_target - b_target, OUTLIER_STOPS):
            continue
        full_slope = (w_target - b_target) / max(1e-4, (s.white_point - s.black_point))
        cap = 2.0 ** TONAL_CAP_STOPS
        slope = max(1.0 / cap, min(cap, full_slope))
        target_black = max(0.0, min(1.0, s.black_point + (b_target - s.black_point)))
        offset = target_black - s.black_point * slope
        projected_black = s.black_point * slope + offset
        projected_white = s.white_point * slope + offset
        if projected_black < -0.01 or projected_white > 1.01:
            continue   # never-worse: don't push toward clipping
        if abs(slope - 1.0) < 1e-4 and abs(offset) < 1e-4:
            continue
        out[s.key] = Grade(slope=(slope,) * 3, offset=(offset,) * 3)
    return out


def solve_leveling(ordered_shots: List[ShotLevelInput]) -> Dict[str, Grade]:
    """Both leveling ideas composed into ONE delta per shot -- what
    `resolver.py`'s leveling stage (between Match and Look) actually
    applies: exposure first, tonal placement on top."""
    exposure = solve_exposure_leveling(ordered_shots)
    tonal = solve_tonal_leveling(ordered_shots)
    out: Dict[str, Grade] = {}
    for key in set(exposure) | set(tonal):
        out[key] = compose(exposure.get(key, Grade()), tonal.get(key, Grade()), 1.0)
    return out
