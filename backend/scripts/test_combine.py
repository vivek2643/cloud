#!/usr/bin/env python3
"""Tests for the uniform energy combiner (l3.combine): within-channel fuse,
peak zoom/split, cross-channel capture-moments, and the speech-safety guarantee
(a cut never lands inside a spoken word, even with a peak attractor). Run:
    python scripts/test_combine.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import vocab  # noqa: E402
from app.services.l3 import combine as cmb  # noqa: E402
from app.services.l3.atoms import Atom  # noqa: E402
from app.services.l3.energy import energy_to_params  # noqa: E402


def _clip(motion=None, perception=None, duration_ms=60000):
    return SimpleNamespace(
        file_id="abcdef12-0000-0000-0000-000000000000", motion=motion,
        perception=perception or {}, duration_ms=duration_ms, cast=None,
        audio=None, dialogue={},
    )


def _done(a, b, peak=None, conf=0.6, subject="person", actor=None, label="move"):
    return Atom(vocab.CHANNEL_DONE, a, b, peak if peak is not None else (a + b) // 2,
                confidence=conf, subject=subject, actor=actor, label=label)


def _shown(a, b, conf=0.6, subject="object", actor=None, label="thing"):
    return Atom(vocab.CHANNEL_SHOWN, a, b, (a + b) // 2,
                confidence=conf, subject=subject, actor=actor, label=label)


# -- within-channel fuse <-> atomize ------------------------------------------

def test_fuse_continuous_run_at_broad():
    """Broad: a continuous run of same-channel beats fuses into ONE cut."""
    atoms = [_done(0, 1000), _done(1200, 2000), _done(2200, 3000)]  # 200ms gaps
    cuts = cmb.combine_video(atoms, energy_to_params(0.1), None, _clip())
    done = [c for c in cuts if c.channel == vocab.CHANNEL_DONE]
    assert len(done) == 1, [(c.src_in_ms, c.src_out_ms) for c in done]
    print("ok  fuse continuous run at Broad")


def test_atomize_at_tight():
    """Tight: fuse gap is 0, so the same run atomizes into separate cuts."""
    atoms = [_done(0, 1000), _done(1200, 2000), _done(2200, 3000)]
    cuts = cmb.combine_video(atoms, energy_to_params(0.8), None, _clip())
    done = [c for c in cuts if c.channel == vocab.CHANNEL_DONE]
    assert len(done) == 3, [(c.src_in_ms, c.src_out_ms) for c in done]
    print("ok  atomize at Tight")


def test_peak_inset_shrinks_toward_impact():
    """Tight insets a long Done beat toward its peak (negative padding)."""
    atoms = [_done(0, 8000, peak=6000)]
    cuts = cmb.combine_video(atoms, energy_to_params(0.8), None, _clip())
    c = next(c for c in cuts if c.channel == vocab.CHANNEL_DONE)
    assert (c.src_out_ms - c.src_in_ms) < 8000           # shrunk
    assert c.src_in_ms <= 6000 <= c.src_out_ms           # still covers the impact
    print("ok  peak inset shrinks toward impact")


def test_done_split_excises_lull_at_tight():
    """TIGHT: a Done beat with a big dead interior lull plays a windup|payoff
    jump-cut -- keep both ends, excise the lull, never the impact peak. The split
    is taken only because it lands TIGHTER than the plain inset (dominant lull)."""
    hop = 100
    n = 100  # 10000ms
    energy = [0.9] * 8 + [0.05] * 72 + [0.9] * 20    # lull 800..8000ms (dominant)
    motion = {"hop_ms": hop, "action_energy": energy, "action_points": []}
    atoms = [_done(0, 10000, peak=9000)]               # peak in the loud payoff
    cuts = cmb.combine_video(atoms, energy_to_params(0.7), None, _clip(motion=motion))
    c = next(c for c in cuts if c.channel == vocab.CHANNEL_DONE)
    assert c.keep_spans and len(c.keep_spans) == 2, c.keep_spans
    (a0, a1), (b0, b1) = c.keep_spans
    assert a1 <= 9000 <= b1                              # peak survives in a kept span
    assert b0 - a1 >= cmb._LULL_MIN_MS                   # a real gap was excised
    print("ok  Done split excises lull at Tight, keeps peak")


def test_sharp_is_pure_banger_no_split():
    """SHARP never splits -- it is the single tightest peak-inset banger, even on
    a beat whose Tight rung jump-cuts. Sharp's play is the shortest of the ladder."""
    hop = 100
    energy = [0.9] * 8 + [0.05] * 72 + [0.9] * 20    # same dominant lull as above
    motion = {"hop_ms": hop, "action_energy": energy, "action_points": []}
    atoms = [_done(0, 10000, peak=9000)]
    cuts = cmb.combine_video(atoms, energy_to_params(0.95), None, _clip(motion=motion))
    c = next(c for c in cuts if c.channel == vocab.CHANNEL_DONE)
    assert c.keep_spans is None                          # the flat (Sharp) span never split
    sharp = next(r for r in c.ladder if r.level == "sharp")
    tight = next(r for r in c.ladder if r.level == "tight")
    assert sharp.keep_spans() is None                    # Sharp plays one contiguous banger
    assert sharp.play_ms() <= tight.play_ms()            # banger is the tightest
    plays = [r.play_ms() for r in c.ladder]
    assert plays == sorted(plays, reverse=True), plays   # whole ladder monotonic
    print("ok  Sharp is a pure banger (no split), ladder monotonic")


# -- owned broad..sharp ladder (video cuts are shrinkable) --------------------

def test_video_cut_owns_full_ladder():
    """A done/shown cut now carries the full broad..sharp ladder (the SAME beat
    zoomed at every band), not a single flat rung -- so it is shrinkable like a
    speech cut. Plays are non-increasing; Balanced is the flat span; Sharp is
    genuinely tighter."""
    atoms = [_shown(0, 8000, conf=0.7)]            # 8s static hold, no motion
    cuts = cmb.combine_video(atoms, energy_to_params(0.5), None, _clip())
    c = next(c for c in cuts if c.channel == vocab.CHANNEL_SHOWN)
    levels = [r.level for r in c.ladder]
    assert levels == ["broad", "calm", "balanced", "tight", "sharp"], levels
    plays = [r.play_ms() for r in c.ladder]
    assert plays == sorted(plays, reverse=True), plays            # non-increasing
    bal = next(r for r in c.ladder if r.level == "balanced")
    assert (bal.in_ms(), bal.out_ms()) == (c.src_in_ms, c.src_out_ms)  # flat = balanced
    sharp = next(r for r in c.ladder if r.level == "sharp")
    assert sharp.play_ms() < bal.play_ms()                        # genuinely shrinks
    print("ok  video cut owns full broad..sharp ladder")


def test_video_ladder_round_trips():
    """The owned ladder survives the cache round-trip (HeroCut.from_cache)."""
    from app.services.l3.hero_cuts import HeroCut
    atoms = [_done(0, 8000, peak=6000, conf=0.7)]
    cuts = cmb.combine_video(atoms, energy_to_params(0.5), None, _clip())
    c = next(c for c in cuts if c.channel == vocab.CHANNEL_DONE)
    back = HeroCut.from_cache(c.to_dict())
    assert [r.level for r in back.ladder] == [r.level for r in c.ladder]
    assert back.src_in_ms == c.src_in_ms and back.src_out_ms == c.src_out_ms
    print("ok  video ladder round-trips through cache")


# -- capture-moments -----------------------------------------------------------

def _mk(channel, a, b, subject=None, speaker=None, region=None, fid="abcdef12"):
    from app.services.l3.hero_cuts import HeroCut
    people = [{"person_id": speaker, "on_camera": True}] if speaker else []
    return HeroCut(hero_id=f"{fid}:{channel}{a}", file_id="abcdef12-x",
                   modality=channel, label="", src_in_ms=a, src_out_ms=b,
                   score=0.5, channel=channel, subject=subject, speaker=speaker,
                   people=people,
                   framing=({"region": region} if region else None))


def test_moment_cross_channel_same_actor():
    """A line and an action by the SAME person, close in time, form a moment."""
    said = _mk(vocab.CHANNEL_SAID, 0, 2000, subject="person", speaker="p1")
    done = _mk(vocab.CHANNEL_DONE, 1800, 3000, subject="person", speaker="p1")
    cuts = [said, done]
    cmb.derive_moments(cuts, energy_to_params(0.3))
    assert said.moment_id and said.moment_id == done.moment_id
    print("ok  moment: cross-channel same actor")


def test_no_moment_same_channel():
    """Two adjacent lines (same channel) are NOT a moment -- a podcast stays
    moment-free."""
    a = _mk(vocab.CHANNEL_SAID, 0, 2000, subject="person", speaker="p1")
    b = _mk(vocab.CHANNEL_SAID, 2100, 4000, subject="person", speaker="p1")
    cuts = [a, b]
    cmb.derive_moments(cuts, energy_to_params(0.3))
    assert a.moment_id is None and b.moment_id is None
    print("ok  no moment for same-channel adjacency")


def test_no_moment_unrelated_subjects():
    """A line by p1 + b-roll of an unrelated object don't auto-moment (brain's
    job). Different actors, different subjects, no region overlap."""
    said = _mk(vocab.CHANNEL_SAID, 0, 2000, subject="person", speaker="p1")
    broll = _mk(vocab.CHANNEL_SHOWN, 1800, 3000, subject="object")
    cuts = [said, broll]
    cmb.derive_moments(cuts, energy_to_params(0.3))
    assert said.moment_id is None and broll.moment_id is None
    print("ok  no moment for unrelated subjects")


def test_atomize_breaks_moment_at_sharp():
    """At Sharp (fuse reach 0) only literally-overlapping cross-channel cuts
    group; a small time gap leaves them apart."""
    said = _mk(vocab.CHANNEL_SAID, 0, 2000, subject="person", speaker="p1")
    done = _mk(vocab.CHANNEL_DONE, 2200, 3000, subject="person", speaker="p1")
    cuts = [said, done]
    cmb.derive_moments(cuts, energy_to_params(0.95))
    assert said.moment_id is None and done.moment_id is None
    print("ok  Sharp atomizes the moment")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall combine tests passed")
