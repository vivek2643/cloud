"""
Per-modality quality scoring, all normalized to 0..1 (higher = better).

These turn the raw L1/L2 signals (blur, focus, motion, fillers, silence) into
comparable scores so recipes and take-selection can rank candidate segments
without each reinventing thresholds. The exact constants are heuristic and
deliberately bounded; they only need to rank segments sensibly relative to one
another, not be physically calibrated.
"""
from __future__ import annotations

from typing import List, Optional

from app.services.l3.primitives.loader import ShotRow, WordTok

# Laplacian-variance blur: below this a frame reads as soft/blurry.
BLUR_SHARP_REF = 120.0
# Motion magnitude that we treat as "full energy" (clamped above this).
MOTION_FULL_REF = 10.0


def _clamp01(x: float) -> float:
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def visual_quality(shot: ShotRow) -> float:
    """
    Combine sharpness (blur_min), focus, and intra-shot stability into one
    0..1 score. Missing signals are treated as neutral (0.5) so old rows
    aren't unfairly penalized.
    """
    # Sharpness: blur_min is min Laplacian variance across keyframes.
    if shot.blur_min is None:
        sharp = 0.5
    else:
        sharp = _clamp01(shot.blur_min / BLUR_SHARP_REF)

    # Focus score (already a quality-ish signal); normalize defensively.
    if shot.focus_score is None:
        focus = 0.5
    else:
        focus = _clamp01(shot.focus_score / BLUR_SHARP_REF) if shot.focus_score > 1.5 else _clamp01(shot.focus_score)

    # Stability: a very high intra-shot variance means the shot changes a lot
    # internally (often whip pans / chaos). Mild preference for stable shots.
    if shot.intra_shot_variance is None:
        stable = 0.7
    else:
        stable = _clamp01(1.0 - shot.intra_shot_variance)

    return _clamp01(0.5 * sharp + 0.25 * focus + 0.25 * stable)


def energy_score(shot: ShotRow) -> float:
    """Motion energy, 0..1. Used by montage/highlight/action recipes."""
    if shot.motion_magnitude is None:
        return 0.3
    return _clamp01(shot.motion_magnitude / MOTION_FULL_REF)


def speech_quality(words: List[WordTok]) -> float:
    """
    Quality of a spoken segment, 0..1, from the words it contains:
      - low filler density is good,
      - reasonable speaking rate (not dead air, not a firehose) is good,
      - a segment with no words scores 0.
    """
    if not words:
        return 0.0
    real = [w for w in words if not w.is_filler and w.text]
    if not real:
        return 0.05

    filler_density = 1.0 - (len(real) / max(1, len(words)))
    filler_ok = _clamp01(1.0 - filler_density)  # 1 when no fillers

    span_ms = max(1, words[-1].end_ms - words[0].start_ms)
    wps = len(real) / (span_ms / 1000.0)
    # Comfortable conversational rate ~2-4 words/sec; penalize extremes.
    if wps < 1.0:
        rate = _clamp01(wps)           # too sparse -> dead air
    elif wps > 5.0:
        rate = _clamp01(1.0 - (wps - 5.0) / 5.0)
    else:
        rate = 1.0

    return _clamp01(0.6 * filler_ok + 0.4 * rate)


def take_quality(
    *,
    visual: Optional[float] = None,
    speech: Optional[float] = None,
    energy: Optional[float] = None,
    valence: Optional[float] = None,
    weights: Optional[dict] = None,
) -> float:
    """
    Blend whichever sub-scores are present into a single 0..1 take quality.
    ``weights`` lets a recipe re-weight modalities (e.g. talking-head leans on
    speech; montage leans on energy/visual). Valence is mapped from [-1,1] to
    [0,1] and contributes a mild positive-content bias.
    """
    w = {"visual": 0.4, "speech": 0.4, "energy": 0.15, "valence": 0.05}
    if weights:
        w.update(weights)

    num = 0.0
    den = 0.0
    if visual is not None:
        num += w["visual"] * visual; den += w["visual"]
    if speech is not None:
        num += w["speech"] * speech; den += w["speech"]
    if energy is not None:
        num += w["energy"] * energy; den += w["energy"]
    if valence is not None:
        num += w["valence"] * _clamp01((valence + 1.0) / 2.0); den += w["valence"]

    return _clamp01(num / den) if den > 0 else 0.0
