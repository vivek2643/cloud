#!/usr/bin/env python3
"""Tests for the energy->params mapping (monotonic, single dial). Run:
    python scripts/test_energy.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3.energy import energy_to_params, PURE_SENTENCE_ENERGY  # noqa: E402


def test_clamps():
    assert energy_to_params(-1).energy == 0.0
    assert energy_to_params(2).energy == 1.0
    print("ok  clamps")


def test_granularity_monotonic():
    """cluster_gap shrinks as energy rises (coarse->fine), hits 0 at sentences,
    then clause subsplit turns on at the top."""
    es = [i / 20 for i in range(21)]
    gaps = [energy_to_params(e).cluster_gap_ms for e in es]
    assert gaps == sorted(gaps, reverse=True), gaps          # non-increasing
    assert energy_to_params(0.0).cluster_gap_ms > 0          # answers merge at low energy
    assert energy_to_params(PURE_SENTENCE_ENERGY).cluster_gap_ms == 0
    assert energy_to_params(0.5).clause_gap_ms == 0          # no subsplit mid-range
    assert energy_to_params(1.0).clause_gap_ms > 0           # clauses at the top
    print("ok  granularity monotonic")


def test_tightness_monotonic():
    """snap window and padding both shrink as energy rises (loose->tight)."""
    es = [i / 20 for i in range(21)]
    wins = [energy_to_params(e).snap_window_ms for e in es]
    pads = [energy_to_params(e).pad_out_ms for e in es]
    assert wins == sorted(wins, reverse=True), wins
    assert pads == sorted(pads, reverse=True), pads
    assert energy_to_params(0.0).pad_out_ms > 0
    assert energy_to_params(1.0).pad_out_ms == 0
    assert energy_to_params(1.0).snap_window_ms < energy_to_params(0.0).snap_window_ms
    print("ok  tightness monotonic")


def test_moments_fuse_low_split_high():
    assert energy_to_params(0.2).fuse_moments is True
    assert energy_to_params(0.9).fuse_moments is False
    print("ok  moments fuse low / split high")


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
    # Action: smooth merge ramp (Balanced still lightly groups) + impact core
    # tightens with energy (Broad/Calm = full unit).
    assert broad.action_merge_gap_ms > calm.action_merge_gap_ms > balanced.action_merge_gap_ms
    assert balanced.action_merge_gap_ms > 0 and sharp.action_merge_gap_ms == 0
    assert broad.action_core_ms is None and calm.action_core_ms is None
    assert balanced.action_core_ms > tight.action_core_ms > sharp.action_core_ms > 0
    # Reaction COUNT is energy-independent (flat warrant floor); only the CUT
    # tightens: Broad keeps the full span, Sharp trims to a punchy core.
    assert broad.reaction_min_warrant == sharp.reaction_min_warrant
    assert broad.reaction_core_ms is None and sharp.reaction_core_ms is not None
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
