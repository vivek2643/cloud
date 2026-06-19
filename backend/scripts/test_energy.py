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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall energy tests passed")
