#!/usr/bin/env python3
"""Tests for territory rank multipliers. Run: python scripts/test_territory.py"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import anchors as an  # noqa: E402
from app.services.l3 import territory as terr  # noqa: E402


def test_listener_reaction_kept():
    a = an.Anchor(5000, 5000, 6000, "expression", an.AFF_REACTION,
                  actor="p2", salience=0.7)
    speaking = [{"subject": "p1", "start_ms": 0, "end_ms": 10000}]
    assert terr.territory_multiplier(a, speaking=speaking, strict=True) == 1.0
    print("ok  listener reaction kept")


def test_speaker_self_reaction_demoted():
    a = an.Anchor(2000, 2000, 3000, "expression", an.AFF_REACTION,
                  actor="p1", salience=0.8)
    speaking = [{"subject": "p1", "start_ms": 1000, "end_ms": 4000}]
    m = terr.territory_multiplier(a, speaking=speaking, strict=True)
    assert m < 0.5, m
    print("ok  speaker-self reaction demoted")


def test_broll_high_speech_occupation_demoted():
    a = an.Anchor(0, 0, 10000, "hold", an.AFF_BROLL, salience=0.5)
    speaking = [{"subject": "p1", "start_ms": 0, "end_ms": 10000}]
    m = terr.territory_multiplier(a, speaking=speaking, strict=True)
    assert m < 0.5, m
    print("ok  broll high speech occupation demoted")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall territory tests passed")
