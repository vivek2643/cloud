"""
The single energy dial -> concrete, deterministic cut parameters.

One number (0 = relaxed/contextual .. 1 = punchy/isolated) drives modality-specific
behavior through five bands (Broad .. Sharp). Speech uses cluster / clause /
snap / pad; action uses merge / onset / impact anchor / optional impact split;
overlay uses merge thresholds, salience floors, and territory strictness.

Everything here is a pure function of the energy float.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# --- Speech (granularity + tightness) -----------------------------------------
CLUSTER_GAP_MAX_MS = 2500
CLUSTER_CURVE = 3.0
PURE_SENTENCE_ENERGY = 0.8
CLAUSE_GAP_HI_MS = 1000
CLAUSE_GAP_TOP_MS = 120

SNAP_WINDOW_LOOSE_MS = 1300
SNAP_WINDOW_TIGHT_MS = 350
PAD_IN_LOOSE_MS = 500
PAD_OUT_LOOSE_MS = 700

FUSE_MOMENTS_BELOW = 0.6

# --- Five bands (match UI: Broad, Calm, Balanced, Tight, Sharp) ---------------
BAND_EDGES = (0.2, 0.4, 0.6, 0.8)   # band i covers [edge[i-1], edge[i]) with 0 at start

# Action merge gap per band (0 = no merge)
_ACTION_MERGE = (2500, 1500, 0, 0, 0)
# unit | onset | impact
_ACTION_ANCHOR = ("unit", "unit", "onset", "impact", "impact")

# Reaction
# Energy does NOT change HOW MANY reactions surface -- relevance (the "warrant":
# reacting to an action, a strong expression, or a long preceding turn) decides
# that, so the floor is flat across bands. Energy only changes the CUT: how the
# reaction is grouped (merge) and how tight it is trimmed (core, below).
_REACTION_MERGE = (2000, 1200, 600, 0, 0)
_REACTION_MIN_WARRANT = (0.55, 0.55, 0.55, 0.55, 0.55)
_REACTION_MIN_DURATION = (1200, 1000, 800, 600, 500)
# Target handle length per band, inset around the expression peak at CUT time
# (negative padding). Broad = None = keep the full VLM span; higher energy trims
# to a punchy core, the same mechanism as b-roll.
_REACTION_CORE_MS = (None, 2200, 1600, 1100, 800)

# B-roll
_BROLL_MERGE = (4000, 2500, 0, 0, 0)
_BROLL_MIN_SALIENCE = (0.55, 0.45, 0.35, 0.25, 0.20)
_BROLL_LOW_SPEECH = (True, True, False, False, False)
# Target handle length per band, inset around the shot's peak/middle at CUT time
# (the VLM hands us the full end-to-end shot). Broad = None = keep the full shot
# (capped only by the anchor safety guard); higher energy trims to a punchy core.
_BROLL_CORE_MS = (None, 4000, 3000, 2000, 1500)

# Insert
_INSERT_COLLAPSE = (True, True, False, False, False)
_INSERT_MIN_SALIENCE = (0.50, 0.42, 0.35, 0.30, 0.25)

# Audible non-speech
_AUDIO_MIN_SALIENCE = (0.75, 0.65, 0.55, 0.45, 0.35)
_AUDIO_MERGE = (1000, 700, 400, 200, 0)

_TERRITORY_STRICT = (True, True, False, False, False)


@dataclass(frozen=True)
class EnergyParams:
    energy: float
    band: int                       # 0..4 Broad .. Sharp
    # speech
    cluster_gap_ms: int
    clause_gap_ms: int
    snap_window_ms: int
    pad_in_ms: int
    pad_out_ms: int
    fuse_moments: bool
    # action / performance
    action_merge_gap_ms: int
    action_anchor_mode: str         # unit | onset | impact
    action_split_at_impact: bool    # Sharp band (energy >= 0.8): windup + payoff
    # overlay — reaction
    reaction_merge_gap_ms: int
    reaction_min_warrant: float
    reaction_min_duration_ms: int
    reaction_core_ms: Optional[int]   # target handle length (None = full span)
    # overlay — b-roll
    broll_merge_gap_ms: int
    broll_min_salience: float
    broll_prefer_low_speech: bool
    broll_core_ms: Optional[int]    # target handle length (None = full shot)
    # overlay — insert
    insert_collapse_graphics: bool
    insert_min_salience: float
    # overlay — audio events
    audio_min_salience: float
    audio_merge_gap_ms: int
    # territory ranking
    territory_strict: bool


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _lerp(t: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) * t


def energy_band(energy: float) -> int:
    """Map 0..1 to band 0 (Broad) .. 4 (Sharp)."""
    e = _clamp01(energy)
    if e >= BAND_EDGES[3]:
        return 4
    if e >= BAND_EDGES[2]:
        return 3
    if e >= BAND_EDGES[1]:
        return 2
    if e >= BAND_EDGES[0]:
        return 1
    return 0


def energy_to_params(energy: float) -> EnergyParams:
    """Map the energy slider to concrete, deterministic cut parameters."""
    e = _clamp01(float(energy))
    band = energy_band(e)

    if e < PURE_SENTENCE_ENERGY:
        frac = e / PURE_SENTENCE_ENERGY
        cluster_gap = round(CLUSTER_GAP_MAX_MS * (1.0 - frac) ** CLUSTER_CURVE)
        clause_gap = 0
    else:
        cluster_gap = 0
        frac = (e - PURE_SENTENCE_ENERGY) / (1.0 - PURE_SENTENCE_ENERGY)
        clause_gap = round(_lerp(frac, CLAUSE_GAP_HI_MS, CLAUSE_GAP_TOP_MS))

    snap = round(_lerp(e, SNAP_WINDOW_LOOSE_MS, SNAP_WINDOW_TIGHT_MS))
    pad_factor = max(0.0, 1.0 - e / PURE_SENTENCE_ENERGY)

    return EnergyParams(
        energy=e,
        band=band,
        cluster_gap_ms=cluster_gap,
        clause_gap_ms=clause_gap,
        snap_window_ms=snap,
        pad_in_ms=round(PAD_IN_LOOSE_MS * pad_factor),
        pad_out_ms=round(PAD_OUT_LOOSE_MS * pad_factor),
        fuse_moments=e < FUSE_MOMENTS_BELOW,
        action_merge_gap_ms=_ACTION_MERGE[band],
        action_anchor_mode=_ACTION_ANCHOR[band],
        action_split_at_impact=e >= BAND_EDGES[3],
        reaction_merge_gap_ms=_REACTION_MERGE[band],
        reaction_min_warrant=_REACTION_MIN_WARRANT[band],
        reaction_min_duration_ms=_REACTION_MIN_DURATION[band],
        reaction_core_ms=_REACTION_CORE_MS[band],
        broll_merge_gap_ms=_BROLL_MERGE[band],
        broll_min_salience=_BROLL_MIN_SALIENCE[band],
        broll_prefer_low_speech=_BROLL_LOW_SPEECH[band],
        broll_core_ms=_BROLL_CORE_MS[band],
        insert_collapse_graphics=_INSERT_COLLAPSE[band],
        insert_min_salience=_INSERT_MIN_SALIENCE[band],
        audio_min_salience=_AUDIO_MIN_SALIENCE[band],
        audio_merge_gap_ms=_AUDIO_MERGE[band],
        territory_strict=_TERRITORY_STRICT[band],
    )
