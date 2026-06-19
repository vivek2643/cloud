"""
The single energy dial -> concrete, deterministic cut parameters.

One number (0 = relaxed/contextual .. 1 = punchy/isolated) drives BOTH axes the
editor cares about, monotonically:

  * GRANULARITY (clustering)  -- low energy merges adjacent sentences into whole
    answers; rising energy stops merging (sentences); the top splits sentences
    into clauses at their internal pauses.
  * TIGHTNESS (cutness)       -- low energy cuts loose (wide fused-field search +
    pre/post-roll padding); high energy cuts close (narrow search, zero padding).

It also decides whether action+dialogue is presented FUSED (one moment, low
energy) or as separate atoms (high energy).

Everything here is a pure function of the energy float, so the same energy always
yields the same cuts. Tune the anchors below; they are intentionally the only
knobs.
"""
from __future__ import annotations

from dataclasses import dataclass

# --- Granularity (clustering) ---------------------------------------------
# Below PURE_SENTENCE_ENERGY we cluster sentences into answers (merge across
# gaps up to cluster_gap). At/above it, sentences stand alone; in the top band
# we sub-split sentences at internal pauses >= clause_gap.
CLUSTER_GAP_MAX_MS = 2500
# Most real inter-sentence gaps are short (~150-600ms), so a linear gap sweep
# would dump all the un-merging into the last sliver of the clustering band.
# Curve it (>1) so the threshold drops fast through the rare big gaps early and
# then lingers in the dense short-gap region -> merges release evenly.
CLUSTER_CURVE = 3.0
PURE_SENTENCE_ENERGY = 0.8
# Clause subsplit threshold. Starts high so PURE_SENTENCE_ENERGY is a clean
# "sentences" detent (only very long mid-sentence pauses split), then ramps down
# so the top of the dial fragments into clauses -- a smooth handoff, no cliff.
CLAUSE_GAP_HI_MS = 1000    # at energy = PURE_SENTENCE_ENERGY
CLAUSE_GAP_TOP_MS = 120    # at energy = 1.0

# --- Tightness (cutness) --------------------------------------------------
SNAP_WINDOW_LOOSE_MS = 1300   # fused-field search half-window at energy 0
SNAP_WINDOW_TIGHT_MS = 350    # ... at energy 1
PAD_IN_LOOSE_MS = 500         # pre-roll at energy 0 (vanishes by PURE_SENTENCE_ENERGY)
PAD_OUT_LOOSE_MS = 700        # post-roll at energy 0

# --- Moments --------------------------------------------------------------
FUSE_MOMENTS_BELOW = 0.6      # below this energy, emit the fused action+dialogue moment


@dataclass(frozen=True)
class EnergyParams:
    energy: float
    cluster_gap_ms: int   # merge adjacent same-speaker sentences whose gap < this (0 = never)
    clause_gap_ms: int    # >0: sub-split sentences at internal pauses >= this; 0 = off
    snap_window_ms: int   # fused-field boundary search half-window
    pad_in_ms: int        # pre-roll added at the in-point (loose -> 0)
    pad_out_ms: int       # post-roll added at the out-point
    fuse_moments: bool     # present action+dialogue as one fused moment


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _lerp(t: float, lo: float, hi: float) -> float:
    return lo + (hi - lo) * t


def energy_to_params(energy: float) -> EnergyParams:
    """Map the energy slider to the concrete, deterministic cut parameters."""
    e = _clamp01(float(energy))

    if e < PURE_SENTENCE_ENERGY:
        frac = e / PURE_SENTENCE_ENERGY                 # 0..1 across the clustering band
        cluster_gap = round(CLUSTER_GAP_MAX_MS * (1.0 - frac) ** CLUSTER_CURVE)
        clause_gap = 0
    else:
        cluster_gap = 0
        frac = (e - PURE_SENTENCE_ENERGY) / (1.0 - PURE_SENTENCE_ENERGY)  # 0..1 in subsplit band
        clause_gap = round(_lerp(frac, CLAUSE_GAP_HI_MS, CLAUSE_GAP_TOP_MS))

    snap = round(_lerp(e, SNAP_WINDOW_LOOSE_MS, SNAP_WINDOW_TIGHT_MS))
    pad_factor = max(0.0, 1.0 - e / PURE_SENTENCE_ENERGY)   # pads gone by PURE_SENTENCE_ENERGY
    return EnergyParams(
        energy=e,
        cluster_gap_ms=cluster_gap,
        clause_gap_ms=clause_gap,
        snap_window_ms=snap,
        pad_in_ms=round(PAD_IN_LOOSE_MS * pad_factor),
        pad_out_ms=round(PAD_OUT_LOOSE_MS * pad_factor),
        fuse_moments=e < FUSE_MOMENTS_BELOW,
    )
