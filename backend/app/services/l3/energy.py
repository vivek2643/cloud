"""
The single energy dial -> concrete, deterministic cut parameters.

One number (0 = relaxed/contextual .. 1 = punchy/isolated) drives channel-specific
behavior through five bands (Broad .. Sharp). SAID uses the thought hierarchy
(turn / setup / thought / core / punch) + snap / pad / breath removal; the video
channels (DONE | SHOWN) share one uniform knob: a span-proportional negative-
padding handle that bites only at Tight/Sharp, plus an optional windup|payoff
split at the peak (Sharp).

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

# --- Video cuts (DONE | SHOWN): one uniform negative-padding knob --------------
# The handle a video beat keeps at each band, expressed as a FRACTION of the
# beat's OWN span -- so the trim scales with content length (a 10s drone hold
# keeps proportionally more than a 2s one; there are no fixed second-counts).
# Broad..Balanced = None = keep the full beat; only Tight/Sharp inset toward the
# peak (impact / reveal). Symmetric with the positive padding that pivots at
# Balanced. Done and Shown share the ladder; kept as two tuples so either channel
# can be tuned independently later.
_DONE_CORE_FRAC = (None, None, None, 0.6, 0.4)
_SHOWN_CORE_FRAC = (None, None, None, 0.6, 0.4)
# Floor so a short beat never insets below a usable, frame-safe handle.
CORE_FLOOR_MS = 600


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
    fuse_gap_ms: int                # relatedness reach: Broad fuses wide, Sharp = 0 (atomize)
    # video cuts (done | shown): negative-padding handle as a fraction of the
    # beat's own span (None below Tight = keep full), and whether a beat may
    # SPLIT windup|payoff at its peak (Sharp band only).
    done_core_frac: Optional[float]
    shown_core_frac: Optional[float]
    core_floor_ms: int
    split_at_peak: bool             # Sharp band (energy >= 0.8): windup|payoff at the peak


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
        fuse_gap_ms=_FUSE_GAP_MS[band],
        done_core_frac=_DONE_CORE_FRAC[band],
        shown_core_frac=_SHOWN_CORE_FRAC[band],
        core_floor_ms=CORE_FLOOR_MS,
        split_at_peak=e >= BAND_EDGES[3],
    )
