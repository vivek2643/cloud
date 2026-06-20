#!/usr/bin/env python3
"""Tests for L2 cutaway thinning. Run: python scripts/test_l2_cutaways.py"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l2.cutaways import thin_cutaways  # noqa: E402
from app.services.l2.schema import (  # noqa: E402
    ClipPerception,
    CutawayAffordance,
    CutawayKind,
    CutawayMoment,
    SpeakingSpan,
)


def _c(**kw) -> CutawayMoment:
    defaults = dict(
        start_ms=0, end_ms=1000, kind=CutawayKind.reaction,
        affordance=CutawayAffordance.reaction, label="test",
    )
    defaults.update(kw)
    return CutawayMoment(**defaults)


def test_merge_adjacent_similar():
    p = ClipPerception(cutaways=[
        _c(start_ms=1000, end_ms=1500, subject="p2", label="smile", intensity=0.5),
        _c(start_ms=1600, end_ms=2200, subject="p2", label="smile", intensity=0.9),
    ])
    thin_cutaways(p)
    assert len(p.cutaways) == 1, p.cutaways
    assert p.cutaways[0].end_ms == 2200
    assert (p.cutaways[0].intensity or 0) >= 0.9
    print("ok  merge adjacent similar cutaways")


def test_drop_speaker_self_reaction():
    p = ClipPerception(
        speaking=[SpeakingSpan(start_ms=500, end_ms=2500, subject="p1")],
        cutaways=[_c(start_ms=1000, end_ms=2000, subject="p1", label="talking face")],
    )
    thin_cutaways(p)
    assert p.cutaways == [], p.cutaways
    print("ok  drop speaker-self reaction")


def test_drop_short_broll():
    p = ClipPerception(cutaways=[
        CutawayMoment(
            start_ms=0, end_ms=800, kind=CutawayKind.broll_hold,
            affordance=CutawayAffordance.broll, label="too short",
        ),
    ])
    thin_cutaways(p)
    assert p.cutaways == [], p.cutaways
    print("ok  drop short b-roll")


def main():
    test_merge_adjacent_similar()
    test_drop_speaker_self_reaction()
    test_drop_short_broll()
    print("\nall l2 cutaway tests passed")


if __name__ == "__main__":
    main()
