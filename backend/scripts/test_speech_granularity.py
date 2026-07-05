"""
Tests for cuts-v2 speech granularity + prosody grading (Phase C2 of
cuts_v2_boundaries.plan.md) -- no DB. Run:
  .venv/bin/python scripts/test_speech_granularity.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import partition as pt  # noqa: E402
from app.services.l3.thought_segments import Span, Thought  # noqa: E402


def _thought(speaker, setup, thought_span, core, punch, text="idea"):
    return Thought(
        speaker=speaker,
        setup=Span(setup[0], setup[1], "setup") if setup else None,
        thought=Span(thought_span[0], thought_span[1], text),
        core=Span(core[0], core[1], "core"),
        punch=Span(punch[0], punch[1], "punch"),
        strength=0.7,
    )


def _flat_thought(speaker, s, e, text="idea"):
    return _thought(speaker, None, (s, e), (s, e), (s, e), text)


def _audio(n, hop=100, f0=None, rms=None):
    return {
        "f0_hz": f0 if f0 is not None else [180.0] * n,
        "rms_db": rms if rms is not None else [-15.0] * n,
        "prosody_hop_ms": hop,
    }


def test_bands_read_progressively_narrower_spans_off_one_thought():
    """Broad/Calm -> setup+thought; Balanced -> thought; Tight -> core;
    Sharp -> punch -- all from the SAME thought's nested hierarchy."""
    t = _thought("S0", (900, 1000), (1000, 5000), (2000, 4000), (3500, 4000))
    spans = {
        label: [(c.start_ms, c.end_ms) for c in pt._said_candidates([t], None, energy)]
        for label, energy in [("broad", 0.1), ("calm", 0.3), ("balanced", 0.5),
                              ("tight", 0.7), ("sharp", 0.9)]
    }
    assert spans["broad"] == [(900, 5000)], spans
    assert spans["calm"] == [(900, 5000)], spans
    assert spans["balanced"] == [(1000, 5000)], spans
    assert spans["tight"] == [(2000, 4000)], spans
    assert spans["sharp"] == [(3500, 4000)], spans
    print("ok  test_bands_read_progressively_narrower_spans_off_one_thought")


def test_broad_merges_close_same_speaker_thoughts_without_pitch_signal():
    """No prosody data at all -> falls back to the old gap-length-only rule:
    a short gap between same-speaker thoughts merges into one turn at Broad,
    but each stays its OWN candidate at Balanced (no merging above Broad)."""
    t1 = _flat_thought("S0", 1000, 3000, "first")
    t2 = _flat_thought("S0", 4500, 6000, "second")
    broad = pt._said_candidates([t1, t2], None, energy=0.1)
    balanced = pt._said_candidates([t1, t2], None, energy=0.5)
    assert len(broad) == 1 and broad[0].start_ms == 1000 and broad[0].end_ms == 6000, broad
    assert len(balanced) == 2, balanced
    print("ok  test_broad_merges_close_same_speaker_thoughts_without_pitch_signal")


def test_speaker_change_never_merges_at_broad():
    t1 = _flat_thought("S0", 1000, 3000)
    t2 = _flat_thought("S1", 3200, 5000)
    broad = pt._said_candidates([t1, t2], None, energy=0.1)
    assert len(broad) == 2, broad
    print("ok  test_speaker_change_never_merges_at_broad")


def test_falling_pitch_and_energy_breaks_even_a_short_gap():
    """A short gap (well under the base merge threshold) still reads as a
    REAL break when the trailing pitch falls and energy drops -- the
    declarative-statement-ending shape."""
    t1 = _flat_thought("S0", 1000, 3000)
    t2 = _flat_thought("S0", 4500, 6000)
    n = 60
    f0 = [180.0] * n
    rms = [-15.0] * n
    for i in range(25, 31):   # tail of t1 (2500..3000ms): falling
        f0[i] = 180 - (i - 25) * 15
        rms[i] = -15 - (i - 25) * 2
    cands = pt._said_candidates([t1, t2], _audio(n, f0=f0, rms=rms), energy=0.1)
    assert len(cands) == 2, cands
    print("ok  test_falling_pitch_and_energy_breaks_even_a_short_gap")


def test_sustained_pitch_bridges_the_gap():
    """The same geometry, but flat (sustained) pitch/energy right up to the
    gap -- an intentional pause -- bridges into one turn."""
    t1 = _flat_thought("S0", 1000, 3000)
    t2 = _flat_thought("S0", 4500, 6000)
    cands = pt._said_candidates([t1, t2], _audio(60), energy=0.1)
    assert len(cands) == 1, cands
    assert cands[0].start_ms == 1000 and cands[0].end_ms == 6000, cands
    print("ok  test_sustained_pitch_bridges_the_gap")


def test_gap_beyond_the_absolute_ceiling_never_bridges():
    """Even with perfectly sustained pitch, a gap past MAX_BRIDGE_GAP_MS is
    never bridged -- the absolute safety ceiling."""
    t1 = _flat_thought("S0", 1000, 3000)
    t2 = _flat_thought("S0", 3000 + 7000, 3000 + 7000 + 2000)  # 7s gap
    n = 130
    cands = pt._said_candidates([t1, t2], _audio(n), energy=0.1)
    assert len(cands) == 2, cands
    print("ok  test_gap_beyond_the_absolute_ceiling_never_bridges")


def test_empty_thoughts_yields_no_candidates():
    assert pt._said_candidates([], None, 0.5) == []
    print("ok  test_empty_thoughts_yields_no_candidates")


def test_prosody_bridges_gap_helper_directly():
    """Unit-level check of the grader's tri-state return: real signal ->
    True/False; missing signal -> None (the fall-back-to-gap-length trigger)."""
    assert pt._prosody_bridges_gap([], [], 100, 3000) is None
    n = 40
    f0 = [180.0] * n
    rms = [-15.0] * n
    for i in range(25, 31):
        f0[i] = 180 - (i - 25) * 15
        rms[i] = -15 - (i - 25) * 2
    assert pt._prosody_bridges_gap(f0, rms, 100, 3000) is False   # real break
    assert pt._prosody_bridges_gap([180.0] * n, [-15.0] * n, 100, 3000) is True   # bridge
    print("ok  test_prosody_bridges_gap_helper_directly")


def main():
    test_bands_read_progressively_narrower_spans_off_one_thought()
    test_broad_merges_close_same_speaker_thoughts_without_pitch_signal()
    test_speaker_change_never_merges_at_broad()
    test_falling_pitch_and_energy_breaks_even_a_short_gap()
    test_sustained_pitch_bridges_the_gap()
    test_gap_beyond_the_absolute_ceiling_never_bridges()
    test_empty_thoughts_yields_no_candidates()
    test_prosody_bridges_gap_helper_directly()
    print("\nall speech-granularity tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
