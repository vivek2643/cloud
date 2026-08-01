"""
seam_function.plan.md §2 -- the S(t) math. A PURE function of SeamSignals:
no I/O, no cut decisions, no thresholding, no peak selection (see the
plan's "Explicit non-goals" -- the seam layer never decides whether/where a
cut exists, it only scores frame cleanliness). Unit-testable with synthetic
SeamSignals, the same shape as audio_features._detect_structure.

    S(t) = g_sharp(t) * g_gest(t) * [ w_vis*still(t) + w_aud(t)*audio(t) ]

g_sharp/g_gest are multiplicative GATES (can shrink a frame toward 0);
still/audio are additive ATTRACTORS (pull toward a good frame), blended by
w_vis (constant) and w_aud(t) = W_AUD_BASE * salience(t) (adaptive).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

from app.services.seam.params import AUDIO_SIGMA_MS, GEST_COEFF, W_AUD_BASE, W_VIS
from app.services.seam.signals import SeamSignals


@dataclass
class SeamCurve:
    hop_ms: int
    t_ms: List[int] = field(default_factory=list)
    S: List[float] = field(default_factory=list)
    # per-term contributions (debug / validation overlay):
    g_sharp: List[float] = field(default_factory=list)
    g_gest: List[float] = field(default_factory=list)
    still: List[float] = field(default_factory=list)
    audio: List[float] = field(default_factory=list)
    w_aud: List[float] = field(default_factory=list)   # W_AUD_BASE * salience(t) (varies over time)
    meta: Dict[str, Any] = field(default_factory=dict)  # weights used, sigma, provenance


def _gaussian_kernel_sum(
    t_ms: List[int], events_ms: List[int], strengths: List[float], sigma_ms: float,
) -> List[float]:
    """Σ_k strength_k * exp(-Δt_k² / (2σ²)) evaluated at every grid time --
    a continuous kernel sum over event timestamps (§2/§3): event
    timestamps are used directly inside the Gaussian, no resampling of
    events onto the grid. Returns all-zero when there are no events."""
    if not events_ms:
        return [0.0] * len(t_ms)
    two_sigma_sq = 2.0 * sigma_ms * sigma_ms
    out: List[float] = []
    for t in t_ms:
        total = 0.0
        for ek, sk in zip(events_ms, strengths):
            dt = t - ek
            total += sk * math.exp(-(dt * dt) / two_sigma_sq)
        out.append(total)
    return out


def _salience(is_musical: bool, onset_strength_at_t: List[float]) -> List[float]:
    """salience(t) = 1.0 when musical, else the clip's own scaled onset
    strength near t (0 when silent) -- so w_aud only outranks stillness
    when audio is present AND salient, adaptive with no footage-type
    branch (§2)."""
    if is_musical:
        return [1.0] * len(onset_strength_at_t)
    return [min(1.0, v) for v in onset_strength_at_t]


def compute_seam_curve(signals: SeamSignals) -> SeamCurve:
    """Evaluate S(t) on signals' own motion grid. Pure function -- no I/O,
    no cut decisions, no thresholding."""
    n = signals.n
    hop_ms = signals.hop_ms
    t_ms = [i * hop_ms for i in range(n)]

    def _at(arr: List[float], i: int) -> float:
        return arr[i] if i < len(arr) else 0.0

    g_sharp = [1.0 - _at(signals.blur_n, i) for i in range(n)]
    g_gest = [1.0 - GEST_COEFF * _at(signals.act_n, i) for i in range(n)]
    # still(t) deliberately EXCLUDES action_energy (already the g_gest gate) --
    # requires BOTH the camera vector and frame-diff to be quiet (max = either
    # one active kills stillness).
    still = [1.0 - max(_at(signals.cam_n, i), _at(signals.fd_n, i)) for i in range(n)]

    # audio(t): musical beats (strength 1.0 each -- SeamSignals carries no
    # separate beat-strength array, see §5's dataclass spec) + non-speech
    # onsets (their own recomputed, clip-normalized strength).
    events_ms = list(signals.beats_ms) + list(signals.onsets_ms)
    strengths = [1.0] * len(signals.beats_ms) + list(signals.onset_strength)
    audio = _gaussian_kernel_sum(t_ms, events_ms, strengths, AUDIO_SIGMA_MS)

    # scaled_onset_strength(t) for salience: the SAME kernel construction,
    # restricted to onsets only, so a beat elsewhere can't spuriously
    # inflate a non-musical clip's salience near t.
    onset_strength_at_t = _gaussian_kernel_sum(
        t_ms, list(signals.onsets_ms), list(signals.onset_strength), AUDIO_SIGMA_MS)
    salience = _salience(signals.is_musical, onset_strength_at_t)
    w_aud = [W_AUD_BASE * s for s in salience]

    S = [
        gs * gg * (W_VIS * st + wa * au)
        for gs, gg, st, wa, au in zip(g_sharp, g_gest, still, w_aud, audio)
    ]

    return SeamCurve(
        hop_ms=hop_ms, t_ms=t_ms, S=S, g_sharp=g_sharp, g_gest=g_gest,
        still=still, audio=audio, w_aud=w_aud,
        meta={
            "w_vis": W_VIS, "w_aud_base": W_AUD_BASE, "gest_coeff": GEST_COEFF,
            "audio_sigma_ms": AUDIO_SIGMA_MS, "is_musical": signals.is_musical,
            "provenance": dict(signals.meta),
        },
    )
