#!/usr/bin/env python3
"""Unit tests for the fused seam field (the core cut-placement primitive).

    python scripts/test_fused_seams.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l1 import fused_seams as fs  # noqa: E402

HOP = 100
DUR = 4000  # 40 hops


def _const(v):
    return [v] * fs.n_hops(DUR, HOP)


def test_dialogue_vetoes_cut():
    """A spoken word (dialogue cost 1) makes the seam unsafe no matter what."""
    dlg = _const(0.0)
    for i in range(10, 20):  # 1.0s..2.0s mid-speech
        dlg[i] = 1.0
    field = fs.compute_fused_field(duration_ms=DUR, dialogue_cost=dlg)
    assert field.q_at(1500) < 0.05, field.q_at(1500)   # vetoed inside the word
    assert field.q_at(500) > 0.95, field.q_at(500)      # clean gap is safe
    print("ok  dialogue veto")


def test_action_attracts_within_safe_region():
    """When safety is partial (e.g. mild camera motion), an impact scores higher
    than ordinary ground -- the attractor breaks the tie toward the hit. In
    perfectly clean audio both already clamp to 1, so we add a baseline veto."""
    camera = _const(0.5)            # mild motion everywhere -> safety 0.5
    action = _const(1.0)            # no reward everywhere
    for i in range(20, 23):         # impact well around 2.0..2.2s
        action[i] = 0.0
    field = fs.compute_fused_field(duration_ms=DUR, camera_cost=camera,
                                   action_cost=action, action_points=[{"ts_ms": 2100}])
    assert field.q_at(2100) > field.q_at(500), (field.q_at(2100), field.q_at(500))
    print("ok  action attractor")


def test_attractor_needs_room():
    """With base safety held EQUAL (0.5) at two impacts, the impact in open space
    gets the attractor boost while the one wedged between vetoes does not -- so an
    impact in a cramped breath can't manufacture a confident cut."""
    dlg = _const(0.0)
    dlg[13] = dlg[14] = dlg[16] = dlg[17] = 1.0   # vetoes hugging a 1-hop notch
    dlg[15] = 0.5                                  # cramped impact: base safety 0.5
    for i in range(28, 33):
        dlg[i] = 0.5                               # open impact: same base safety 0.5
    action = _const(1.0)
    action[15] = 0.0
    action[30] = 0.0
    field = fs.compute_fused_field(duration_ms=DUR, dialogue_cost=dlg, action_cost=action,
                                   action_points=[{"ts_ms": 1500}, {"ts_ms": 3000}])
    cramped, openq = field.q_at(1500), field.q_at(3000)
    assert openq > cramped, (openq, cramped)          # boost only where there's room
    assert abs(cramped - 0.5) < 0.02, cramped         # cramped impact got no boost
    print("ok  attractor needs room")


def test_veto_beats_attractor():
    """An impact landing ON a spoken word stays vetoed -- safety dominates."""
    dlg = _const(0.0)
    action = _const(1.0)
    for i in range(10, 20):
        dlg[i] = 1.0               # speech 1.0..2.0s
        action[i] = 0.0            # impact in the SAME region
    field = fs.compute_fused_field(duration_ms=DUR, dialogue_cost=dlg,
                                   action_cost=action, action_points=[{"ts_ms": 1500}])
    assert field.q_at(1500) < 0.05, field.q_at(1500)
    print("ok  veto beats attractor")


def test_energy_increases_attractor_pull():
    camera = _const(0.5)            # partial safety so the boost is visible
    action = _const(1.0)
    action[20] = 0.0
    lo = fs.compute_fused_field(duration_ms=DUR, energy=0.0,
                                camera_cost=camera, action_cost=action)
    hi = fs.compute_fused_field(duration_ms=DUR, energy=1.0,
                                camera_cost=camera, action_cost=action)
    assert hi.q_at(2000) > lo.q_at(2000), (hi.q_at(2000), lo.q_at(2000))
    print("ok  energy raises attractor weight")


def test_snap_moves_off_vetoed_point():
    dlg = _const(0.0)
    for i in range(15, 25):        # speech 1.5..2.5s
        dlg[i] = 1.0
    field = fs.compute_fused_field(duration_ms=DUR, dialogue_cost=dlg,
                                   dialogue_points=[{"ts_ms": 1000, "kind": "pause"}])
    snapped = fs.snap_point(field, rough_ms=2000, lo_ms=800, hi_ms=2200)
    assert field.q_at(snapped) > field.q_at(2000)
    assert dlg[snapped // HOP] < 1.0   # not inside the word
    print("ok  snap avoids vetoed region")


def test_snap_around_core_never_clips_core():
    """The core (e.g. the racquet swing + miss) sits in a vetoed/high-motion
    region; snap_around_core must land the in-point at/before core_in and the
    out-point at/after core_out -- never inside -- so the payoff is never cut."""
    # Core = 2.0..2.8s (a busy action beat). Calm rests just outside it.
    cam = _const(0.9)
    for i in range(0, 19):     # calm before
        cam[i] = 0.05
    for i in range(29, 40):    # calm after the beat
        cam[i] = 0.05
    field = fs.compute_fused_field(duration_ms=DUR, camera_cost=cam)
    in_ms, out_ms = fs.snap_around_core(field, 2000, 2800, win_ms=1200, duration_ms=DUR)
    assert in_ms <= 2000, in_ms          # never starts inside the core
    assert out_ms >= 2800, out_ms        # never ends inside the core
    assert in_ms >= 1500 and out_ms <= 3200    # but stays within the search window
    print("ok  snap_around_core never clips core")


def test_missing_channels_are_neutral():
    """No grids at all -> everything is safe (degrades gracefully)."""
    field = fs.compute_fused_field(duration_ms=DUR)
    assert all(c < 0.01 for c in field.cost)
    print("ok  missing channels neutral")


def test_protected_span_hard_veto():
    field = fs.compute_fused_field(duration_ms=DUR, protected_spans=[(1000, 2000)])
    assert field.q_at(1500) < 0.01
    assert field.q_at(3000) > 0.99
    print("ok  protected span veto")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall fused-seam tests passed")
