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
# Granularity = which LEVEL of the THOUGHT hierarchy we emit, per band (not a
# silence threshold). A thought is one speaker's self-contained idea (see
# l3.thought_segments); each level zooms it in or out and the levels NEST:
#
#   turn    -- every consecutive thought by the speaker        (Broad)
#   setup   -- the thought + the speaker's own run-up into it   (Calm)
#   thought -- the complete idea proper                         (Balanced, pivot)
#   core    -- the one sentence that carries it                 (Tight)
#   punch   -- the tightest landing clause                      (Sharp, + breath)
#
# Balanced is the pivot (one complete thought); zoom OUT by adding the setup then
# merging same-speaker thoughts into a turn, zoom IN to the core sentence then
# the punchline. The Sharp band additionally loses its internal breaths (the
# progressive breath-removal step layered on later).
_SPEECH_UNIT = ("turn", "setup", "thought", "core", "punch")
# Thought-merge gap for the turn level (Broad only); 0 = emit the native level
# (one thought per unit). Calm..Sharp never merge across thoughts.
_SPEECH_MERGE_MS = (4000, 0, 0, 0, 0)

# Progressive breath removal (Sharp band only): excise internal silent gaps
# whose length >= the threshold, turning a sentence into a jump-cut edit-list.
# Active only above the Tight/Sharp edge -- Tight keeps its breath. Ramps from
# "long pauses only" at the Sharp onset down to "even short breaths" at full
# energy, so the dial removes progressively more dead air. 0 = no removal.
SPEECH_BREATH_HI_MS = 700      # at Sharp onset (0.8): only long pauses go
SPEECH_BREATH_LO_MS = 220      # at max energy (1.0): tight, snappy jump-cuts

# Padding is SYMMETRIC around the Balanced pivot (band 2): below it the cut is
# extended outward (positive padding = breathing room); at Balanced it sits on
# its natural span (pad 0); above it the core inset trims inward toward the peak
# (negative padding). So `pad_factor` ramps 1 -> 0 from energy 0 up to Balanced
# and the per-affordance `*_core_ms` tuples only bite at Tight/Sharp -- the two
# halves of one signed ladder, never overlapping.
PAD_PIVOT_ENERGY = 0.6         # = BAND_EDGES[2]; positive padding fades to zero here

SNAP_WINDOW_LOOSE_MS = 1300
SNAP_WINDOW_TIGHT_MS = 350
PAD_IN_LOOSE_MS = 500
PAD_OUT_LOOSE_MS = 700

FUSE_MOMENTS_BELOW = 0.6

# Deterministic relatedness FUSE↔ATOMIZE ladder (P4). The max time gap for two
# complementary cuts to read as ONE moment, per band. Energy is the dial: at
# Broad we fuse widely (a whole demo run is one moment); at Sharp we atomize
# (gap 0 -> only literally-overlapping complementary beats group, everything
# else stands as its own punchy cut). Pairs with the relatedness GATE in
# hero_cuts (shared actor/region still required -- this only sets the reach).
#
# Re-centered onto the 2-4 WORKING range: bands 1 (Broad) and 5 (Sharp) are
# rarely-used extremes, so the whole fuse->atomize transition is packed into
# Calm/Balanced/Tight. Atomize (gap 0) now arrives at TIGHT (band 4); Sharp adds
# only the breath-removal punch on top. A typical small moment thus exposes its
# peak member within the working range, not only at the extreme.
_FUSE_GAP_MS = (1500, 800, 350, 0, 0)

# --- Five bands (match UI: Broad, Calm, Balanced, Tight, Sharp) ---------------
BAND_EDGES = (0.2, 0.4, 0.6, 0.8)   # band i covers [edge[i-1], edge[i]) with 0 at start

# Action merge gap per band (0 = no merge). Smoothed so Balanced still lightly
# groups instead of a cliff to no-clustering above Calm.
_ACTION_MERGE = (2500, 1500, 800, 0, 0)
# unit | onset | impact
_ACTION_ANCHOR = ("unit", "unit", "onset", "impact", "impact")
# Target handle length per band, inset around the IMPACT at cut time (negative
# padding, impact-forward via lead_frac=0). Broad..Balanced = None = the full
# action (Broad/Calm keep the whole unit; Balanced drops the windup via the
# onset anchor but stays uncapped); only Tight/Sharp cap toward the impact.
# Performances are exempt (a song/dance keeps its full duration).
_ACTION_CORE_MS = (None, None, None, 2500, 1800)

# Reaction
# Energy does NOT change HOW MANY reactions surface -- relevance (the "warrant":
# reacting to an action, a strong expression, or a long preceding turn) decides
# that, so the floor is flat across bands. Energy only changes the CUT: how the
# reaction is grouped (merge) and how tight it is trimmed (core, below).
_REACTION_MERGE = (2000, 1200, 600, 0, 0)
_REACTION_MIN_WARRANT = (0.55, 0.55, 0.55, 0.55, 0.55)
_REACTION_MIN_DURATION = (1200, 1000, 800, 600, 500)
# Target handle length per band, inset around the expression peak at CUT time
# (negative padding). Broad..Balanced = None = keep the full VLM span (a long,
# held reaction / deep-listening shot lives here); only Tight/Sharp trim to a
# punchy core. Symmetric with the positive padding that pivots at Balanced.
_REACTION_CORE_MS = (None, None, None, 1100, 800)

# B-roll
_BROLL_MERGE = (4000, 2500, 0, 0, 0)
_BROLL_MIN_SALIENCE = (0.55, 0.45, 0.35, 0.25, 0.20)
_BROLL_LOW_SPEECH = (True, True, False, False, False)
# Target handle length per band, inset around the shot's peak/middle at CUT time
# (the VLM hands us the full end-to-end shot). Broad..Balanced = None = keep the
# full shot (capped only by the anchor safety guard); only Tight/Sharp trim to a
# punchy core. Symmetric with the positive padding that pivots at Balanced.
_BROLL_CORE_MS = (None, None, None, 2000, 1500)

# Insert
# Inserts are sparse and already meaningful (a reveal / title / interaction the
# VLM flagged), so we barely filter -- a flat low floor keeps nearly all. Energy
# only changes the CUT: dedup repeated graphics (collapse) + negative padding.
_INSERT_COLLAPSE = (True, True, False, False, False)
_INSERT_MIN_SALIENCE = (0.30, 0.30, 0.30, 0.30, 0.30)
# Target handle length per band, inset from the onset at CUT time (the insert is
# start-anchored, so this trims the tail). Broad..Balanced = None = full onset
# handle; only Tight/Sharp trim. Symmetric with the positive padding pivot.
_INSERT_CORE_MS = (None, None, None, 2000, 1500)

# Audible non-speech
_AUDIO_MIN_SALIENCE = (0.75, 0.65, 0.55, 0.45, 0.35)
_AUDIO_MERGE = (1000, 700, 400, 200, 0)

_TERRITORY_STRICT = (True, True, False, False, False)


@dataclass(frozen=True)
class EnergyParams:
    energy: float
    band: int                       # 0..4 Broad .. Sharp
    # speech
    speech_unit: str                # turn | setup | thought | core | punch (thought level)
    speech_merge_gap_ms: int        # thought-merge gap for the turn level (0 = native)
    speech_breath_gap_ms: int       # excise internal gaps >= this (0 = keep breath)
    snap_window_ms: int
    pad_in_ms: int
    pad_out_ms: int
    fuse_moments: bool
    fuse_gap_ms: int                # relatedness reach: Broad fuses wide, Sharp = 0 (atomize)
    # action / performance
    action_merge_gap_ms: int
    action_anchor_mode: str         # unit | onset | impact
    action_split_at_impact: bool    # Sharp band (energy >= 0.8): windup + payoff
    action_core_ms: Optional[int]   # target handle length (None = full unit)
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
    insert_core_ms: Optional[int]     # target handle length (None = full onset handle)
    # overlay — audio events
    audio_min_salience: float
    audio_merge_gap_ms: int
    # territory ranking
    territory_strict: bool


# Genre -> default energy CENTER (the slider's starting point, not a cap; the
# editor still dials the full range). Long-form/observational sits low so shots
# breathe (a podcast, an interview, scenery); short-form/punchy sits high (a
# product/action reel cuts tight). Instructional content sits just below
# Balanced (steady, follow the steps). Tunable; unknown genres -> Balanced.
_GENRE_DEFAULT_ENERGY = {
    "interview": 0.3,
    "talking_head": 0.3,
    "scenic": 0.3,
    "broll": 0.3,
    "tutorial": 0.4,
    "demo": 0.4,
    "screen_recording": 0.4,
    "vlog": 0.5,
    "event": 0.5,
    "other": 0.5,
    "performance": 0.6,
    "product": 0.7,
    "action": 0.7,
}
DEFAULT_ENERGY = 0.5


def default_energy_for(content_type: Optional[str]) -> float:
    """The slider's starting energy for a detected genre (see the table). The
    editor can still move anywhere; this only sets where the dial opens."""
    return _GENRE_DEFAULT_ENERGY.get((content_type or "").strip().lower(), DEFAULT_ENERGY)


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

    snap = round(_lerp(e, SNAP_WINDOW_LOOSE_MS, SNAP_WINDOW_TIGHT_MS))
    pad_factor = max(0.0, 1.0 - e / PAD_PIVOT_ENERGY)

    # Breath removal ramps in only across the Sharp band (energy >= top edge).
    if e >= BAND_EDGES[3]:
        frac = (e - BAND_EDGES[3]) / (1.0 - BAND_EDGES[3])
        breath_gap = round(_lerp(frac, SPEECH_BREATH_HI_MS, SPEECH_BREATH_LO_MS))
    else:
        breath_gap = 0

    return EnergyParams(
        energy=e,
        band=band,
        speech_unit=_SPEECH_UNIT[band],
        speech_merge_gap_ms=_SPEECH_MERGE_MS[band],
        speech_breath_gap_ms=breath_gap,
        snap_window_ms=snap,
        pad_in_ms=round(PAD_IN_LOOSE_MS * pad_factor),
        pad_out_ms=round(PAD_OUT_LOOSE_MS * pad_factor),
        fuse_moments=e < FUSE_MOMENTS_BELOW,
        fuse_gap_ms=_FUSE_GAP_MS[band],
        action_merge_gap_ms=_ACTION_MERGE[band],
        action_anchor_mode=_ACTION_ANCHOR[band],
        action_split_at_impact=e >= BAND_EDGES[3],
        action_core_ms=_ACTION_CORE_MS[band],
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
        insert_core_ms=_INSERT_CORE_MS[band],
        audio_min_salience=_AUDIO_MIN_SALIENCE[band],
        audio_merge_gap_ms=_AUDIO_MERGE[band],
        territory_strict=_TERRITORY_STRICT[band],
    )
