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


def test_moments_fuse_low_split_high():
    assert energy_to_params(0.2).fuse_moments is True
    assert energy_to_params(0.9).fuse_moments is False
    print("ok  moments fuse low / split high")


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


def test_bands_and_action_modes():
    broad = energy_to_params(0.0)
    calm = energy_to_params(0.3)
    balanced = energy_to_params(0.5)
    tight = energy_to_params(0.7)
    sharp = energy_to_params(1.0)
    assert broad.band == 0 and broad.action_anchor_mode == "unit"
    assert broad.action_merge_gap_ms > 0
    assert balanced.band == 2 and balanced.action_anchor_mode == "onset"
    assert tight.band == 3 and tight.action_anchor_mode == "impact"
    assert sharp.action_split_at_impact is True
    assert energy_to_params(0.8).action_split_at_impact is True
    assert energy_to_params(0.79).action_split_at_impact is False
    # Action: smooth merge ramp (Balanced still lightly groups) + impact core.
    # Symmetric pivot: Broad..Balanced keep the full action (core None); only
    # Tight/Sharp cap impact-forward.
    assert broad.action_merge_gap_ms > calm.action_merge_gap_ms > balanced.action_merge_gap_ms
    assert balanced.action_merge_gap_ms > 0 and sharp.action_merge_gap_ms == 0
    assert broad.action_core_ms is None and calm.action_core_ms is None
    assert balanced.action_core_ms is None
    assert tight.action_core_ms > sharp.action_core_ms > 0
    # Reaction COUNT is energy-independent (flat warrant floor); only the CUT
    # tightens: Broad..Balanced keep the full span (a long held / listening
    # shot), Tight/Sharp trim to a punchy core.
    assert broad.reaction_min_warrant == sharp.reaction_min_warrant
    assert broad.reaction_core_ms is None and balanced.reaction_core_ms is None
    assert sharp.reaction_core_ms is not None
    assert broad.territory_strict and not sharp.territory_strict
    print("ok  bands and action modes")


def test_overlay_thresholds_monotonic():
    es = [0.0, 0.25, 0.5, 0.75, 1.0]
    br = [energy_to_params(e).broll_min_salience for e in es]
    # Reaction core length tightens monotonically with energy (Broad full = inf).
    rc = [energy_to_params(e).reaction_core_ms or 10**9 for e in es]
    assert rc == sorted(rc, reverse=True), rc
    assert br == sorted(br, reverse=True), br
    print("ok  overlay thresholds monotonic")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall energy tests passed")
