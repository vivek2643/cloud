#!/usr/bin/env python3
"""Tests for the uniform energy combiner (l3.combine): within-channel fuse,
peak zoom/split, and the owned broad..sharp ladder. (Cross-channel capture-
moments were retired -- grouping is the brain's job; the timeline weld is tested
in test_arrange.py.) Run:
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


# -- audio policy on video cuts (mute uncontrolled audio under a shot) ---------

def _src(words=None, silences=None):
    """A minimal SpanSource-like stub for the audio policy (.words + .silences)."""
    return SimpleNamespace(words=words or [], silences=silences or [])


def _word(a, b):
    return {"start_ms": a, "end_ms": b, "text": "x", "is_filler": False}


def _sil(a, b):
    return {"start_ms": a, "end_ms": b}


def test_video_audio_muted_when_speech_under_shot():
    """A shown/done cut with real talking under it (>=15% of the span) is tagged
    audio='speech' and muted by default -- b-roll shouldn't drag an out-of-context
    half-sentence onto the edit."""
    atoms = [_shown(0, 8000, conf=0.7)]
    src = _src(words=[_word(500, 2000), _word(2100, 3500)])   # ~3s talk over 8s -> 37%
    cuts = cmb.combine_video(atoms, energy_to_params(0.5), None, _clip(), source=src)
    c = next(c for c in cuts if c.channel == vocab.CHANNEL_SHOWN)
    assert c.audio == "speech" and c.mute is True, (c.audio, c.mute)
    assert c.to_dict()["mute"] is True
    print("ok  video cut mutes stray speech under the shot")


def test_video_audio_muted_when_untranscribed_sound():
    """No transcribed speech, but the shot is mostly NON-silence (off-mic voice /
    crew noise / an action's own sound) -> audio='sound', muted by default. This
    is the Reel-5 case: loud opener with an empty transcript."""
    atoms = [_shown(0, 8000, conf=0.7)]
    # Only 1s of logged silence over 8s -> 87% sound -> well over _VIDEO_SOUND_FRAC.
    src = _src(words=[], silences=[_sil(3000, 4000)])
    cuts = cmb.combine_video(atoms, energy_to_params(0.5), None, _clip(), source=src)
    c = next(c for c in cuts if c.channel == vocab.CHANNEL_SHOWN)
    assert c.audio == "sound" and c.mute is True, (c.audio, c.mute)
    assert c.to_dict()["mute"] is True
    print("ok  video cut mutes uncontrolled (untranscribed) sound")


def test_video_audio_silent_when_mostly_silence():
    """A shot that is essentially silence -> audio='silent', not muted (there is
    nothing to silence, and we don't assert a policy on quiet)."""
    atoms = [_shown(0, 8000, conf=0.7)]
    src = _src(words=[], silences=[_sil(0, 7800)])       # ~97% silence
    cuts = cmb.combine_video(atoms, energy_to_params(0.5), None, _clip(), source=src)
    c = next(c for c in cuts if c.channel == vocab.CHANNEL_SHOWN)
    assert c.audio == "silent" and c.mute is False, (c.audio, c.mute)
    assert c.to_dict()["mute"] is None                   # not emitted when false
    print("ok  video cut leaves near-silent audio alone")


def test_video_audio_none_without_audio_data():
    """No words AND no silence map -> we can't judge; leave the audio alone."""
    atoms = [_shown(0, 8000, conf=0.7)]
    cuts = cmb.combine_video(atoms, energy_to_params(0.5), None, _clip(), source=_src())
    c = next(c for c in cuts if c.channel == vocab.CHANNEL_SHOWN)
    assert c.audio is None and c.mute is False, (c.audio, c.mute)
    print("ok  video cut without audio data asserts no policy")


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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall combine tests passed")
