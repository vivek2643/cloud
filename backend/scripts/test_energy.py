#!/usr/bin/env python3
"""Tests for the energy->params mapping (monotonic, single dial). Run:
    python scripts/test_energy.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3.energy import (  # noqa: E402
    energy_to_params, PAD_PIVOT_ENERGY,
    SPEECH_BREATH_HI_MS, SPEECH_BREATH_LO_MS,
)


def test_clamps():
    assert energy_to_params(-1).energy == 0.0
    assert energy_to_params(2).energy == 1.0
    print("ok  clamps")


def test_granularity_tiers():
    """The dial zooms through the THOUGHT hierarchy: turn -> setup -> thought ->
    core -> punch. Only the turn level carries a thought-merge gap."""
    units = [energy_to_params(i / 20).speech_unit for i in range(21)]
    order = {"turn": 0, "setup": 1, "thought": 2, "core": 3, "punch": 4}
    ranks = [order[u] for u in units]
    assert ranks == sorted(ranks), units                     # coarse -> fine, no regressions
    assert energy_to_params(0.0).speech_unit == "turn"       # broad: merge whole turn
    assert energy_to_params(0.5).speech_unit == "thought"    # balanced = one thought
    assert energy_to_params(1.0).speech_unit == "punch"      # sharp = punchline clause
    # Merge gap only applies to the turn level and is non-increasing.
    gaps = [energy_to_params(i / 20).speech_merge_gap_ms for i in range(21)]
    assert gaps == sorted(gaps, reverse=True), gaps
    assert energy_to_params(0.0).speech_merge_gap_ms > 0      # turn merges thoughts
    assert energy_to_params(0.5).speech_merge_gap_ms == 0     # thought level emits native
    assert energy_to_params(1.0).speech_merge_gap_ms == 0
    print("ok  granularity tiers")


def test_breath_removal_progressive():
    """Breath removal is off through Tight, then ramps in across the Sharp band:
    long-pauses-only at the onset (0.8) down to short breaths at full energy."""
    assert energy_to_params(0.5).speech_breath_gap_ms == 0     # topic: contiguous
    assert energy_to_params(0.7).speech_breath_gap_ms == 0     # Tight: keep breath
    assert energy_to_params(0.8).speech_breath_gap_ms == SPEECH_BREATH_HI_MS
    assert energy_to_params(1.0).speech_breath_gap_ms == SPEECH_BREATH_LO_MS
    # Within the Sharp band the threshold only tightens (removes progressively more).
    sharp = [energy_to_params(0.8 + i / 100).speech_breath_gap_ms for i in range(0, 21)]
    assert sharp == sorted(sharp, reverse=True), sharp
    assert SPEECH_BREATH_LO_MS < SPEECH_BREATH_HI_MS
    print("ok  breath removal progressive")


def test_tightness_monotonic():
    """snap window and padding both shrink as energy rises (loose->tight)."""
    es = [i / 20 for i in range(21)]
    wins = [energy_to_params(e).snap_window_ms for e in es]
    pads = [energy_to_params(e).pad_out_ms for e in es]
    assert wins == sorted(wins, reverse=True), wins
    assert pads == sorted(pads, reverse=True), pads
    assert energy_to_params(0.0).pad_out_ms > 0
    # Symmetric pivot: positive padding fades to zero by the Balanced pivot, not
    # only at full energy (above the pivot the negative core inset takes over).
    assert energy_to_params(PAD_PIVOT_ENERGY).pad_out_ms == 0
    assert energy_to_params(1.0).pad_out_ms == 0
    assert energy_to_params(1.0).snap_window_ms < energy_to_params(0.0).snap_window_ms
    print("ok  tightness monotonic")


def test_fuse_gap_low_wide_high_zero():
    assert energy_to_params(0.2).fuse_gap_ms > 0
    assert energy_to_params(0.9).fuse_gap_ms == 0
    print("ok  fuse gap wide low / atomized high")


def test_fuse_gap_ladder_widens_low_atomizes_high():
    """The relatedness reach is the fuse<->atomize dial, RE-CENTERED onto the
    2-4 working range: widest at Broad, shrinking through Calm/Balanced, and
    atomized (gap 0) by TIGHT -- Sharp stays 0 (it only adds breath-removal).
    Bands 1 (Broad) and 5 (Sharp) are saturated extremes, not where the
    transition lives."""
    broad, calm, balanced, tight, sharp = (
        energy_to_params(e).fuse_gap_ms for e in (0.0, 0.3, 0.5, 0.7, 1.0)
    )
    assert broad > calm > balanced > 0, (broad, calm, balanced)
    assert tight == 0, tight          # atomize arrives in the working range
    assert sharp == 0, sharp          # and stays atomized at the extreme
    print("ok  fuse-gap ladder re-centered (atomizes by Tight)")


def test_bands_and_video_cores():
    """Video channels (done|shown) share one uniform knob: a span-PROPORTIONAL
    negative-padding handle that is None (keep full) through Balanced and only
    bites at Tight/Sharp, plus a windup|payoff split at Tight (Sharp is the
    pure banger -- no split)."""
    broad = energy_to_params(0.0)
    calm = energy_to_params(0.3)
    balanced = energy_to_params(0.5)
    tight = energy_to_params(0.7)
    sharp = energy_to_params(1.0)
    assert broad.band == 0 and balanced.band == 2 and tight.band == 3
    # Windup|payoff split lives on the TIGHT band only; Sharp is a pure banger.
    assert tight.split_at_peak is True
    assert sharp.split_at_peak is False
    assert energy_to_params(0.6).split_at_peak is True       # Tight onset
    assert energy_to_params(0.59).split_at_peak is False     # Balanced
    assert energy_to_params(0.8).split_at_peak is False      # Sharp onset (banger)
    # Done/Shown handle fractions: full (None) through Balanced; Tight > Sharp > 0.
    for core in ("done_core_frac", "shown_core_frac"):
        assert getattr(broad, core) is None and getattr(calm, core) is None
        assert getattr(balanced, core) is None
        assert getattr(tight, core) > getattr(sharp, core) > 0
    # A floor in ms keeps a short beat frame-safe.
    assert sharp.core_floor_ms > 0
    print("ok  bands and video cores")


def test_video_core_fraction_scales_with_length():
    """The handle is a FRACTION of the beat span, so the trim varies with clip
    length -- a long shot keeps proportionally more than a short one."""
    sharp = energy_to_params(1.0)
    frac = sharp.shown_core_frac
    short_handle = max(sharp.core_floor_ms, int(frac * 2000))
    long_handle = max(sharp.core_floor_ms, int(frac * 10000))
    assert long_handle > short_handle, (short_handle, long_handle)
    print("ok  video core fraction scales with length")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall energy tests passed")
