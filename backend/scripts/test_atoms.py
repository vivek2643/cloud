#!/usr/bin/env python3
"""Tests for the v2 atom substrate: per-channel confidence gate + the
dominant-channel fold (highest-risk piece, tested first). Run:
    python scripts/test_atoms.py
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import vocab  # noqa: E402
from app.services.l3.atoms import (  # noqa: E402
    Atom, build_atoms, _gate, _fold_redundant, _CHANNEL_FLOOR,
)


def _clip(perception=None, dialogue=None, motion=None, audio=None, duration_ms=60000):
    return SimpleNamespace(
        perception=perception or {}, dialogue=dialogue or {}, motion=motion,
        audio=audio, duration_ms=duration_ms, cast=None, thoughts=[],
    )


# -- gate ----------------------------------------------------------------------

def test_gate_per_channel_floor():
    atoms = [
        Atom(vocab.CHANNEL_DONE, 0, 1000, 500, confidence=0.20),   # below 0.30 -> drop
        Atom(vocab.CHANNEL_DONE, 0, 1000, 500, confidence=0.31),   # keep
        Atom(vocab.CHANNEL_SHOWN, 0, 1000, 500, confidence=0.29),  # below 0.30 -> drop
        Atom(vocab.CHANNEL_SHOWN, 0, 1000, 500, confidence=0.50),  # keep
        Atom(vocab.CHANNEL_HEARD, 0, 1000, 500, confidence=0.60),  # below 0.70 -> drop
        Atom(vocab.CHANNEL_HEARD, 0, 1000, 500, confidence=0.80),  # keep
        Atom(vocab.CHANNEL_SAID, 0, 1000, 500, confidence=0.0),    # said floor 0.0 -> keep
    ]
    kept = _gate(atoms)
    assert _CHANNEL_FLOOR[vocab.CHANNEL_HEARD] > _CHANNEL_FLOOR[vocab.CHANNEL_DONE]
    assert len(kept) == 4, [(a.channel, a.confidence) for a in kept]
    assert all(a.confidence >= _CHANNEL_FLOOR[a.channel] for a in kept)
    print("ok  gate per-channel floor")


# -- fold ----------------------------------------------------------------------

def test_fold_talking_head_collapses_to_one():
    """A Shown.person and a Done gesture, both same actor and co-extensive with a
    Said line, fold into the Said cut: a talking head is ONE card, not three."""
    said = Atom(vocab.CHANNEL_SAID, 0, 4000, 2000, confidence=1.0,
                subject=vocab.SUBJECT_PERSON, actor="p1", text="hello there")
    shown = Atom(vocab.CHANNEL_SHOWN, 100, 3900, 2000, confidence=0.6,
                 subject=vocab.SUBJECT_PERSON, actor="p1")
    gesture = Atom(vocab.CHANNEL_DONE, 200, 3800, 2000, confidence=0.5,
                   subject=vocab.SUBJECT_PERSON, actor="p1")
    kept = _fold_redundant([said, shown, gesture])
    assert kept == [said], [(a.channel, a.actor) for a in kept]
    assert "on_camera" in said.flags and "gesturing" in said.flags
    print("ok  fold talking head -> one cut")


def test_fold_keeps_other_subject_cutaway():
    """A held shot of a DIFFERENT subject (a listener, the product, the scenery)
    is a real cutaway and must survive the fold."""
    said = Atom(vocab.CHANNEL_SAID, 0, 4000, 2000, confidence=1.0,
                subject=vocab.SUBJECT_PERSON, actor="p1", text="look at this")
    listener = Atom(vocab.CHANNEL_SHOWN, 100, 3900, 2000, confidence=0.6,
                    subject=vocab.SUBJECT_PERSON, actor="p2")
    product = Atom(vocab.CHANNEL_SHOWN, 500, 3500, 2000, confidence=0.6,
                   subject=vocab.SUBJECT_OBJECT, actor=None)
    kept = _fold_redundant([said, listener, product])
    assert listener in kept and product in kept and said in kept
    print("ok  fold keeps other-subject cutaway")


def test_fold_needs_actor_and_overlap():
    """No fold when actors are unknown or the overlap is too small."""
    said = Atom(vocab.CHANNEL_SAID, 0, 4000, 2000, confidence=1.0,
                subject=vocab.SUBJECT_PERSON, actor="p1")
    no_actor = Atom(vocab.CHANNEL_SHOWN, 0, 4000, 2000, confidence=0.6,
                    subject=vocab.SUBJECT_PERSON, actor=None)
    brief = Atom(vocab.CHANNEL_SHOWN, 3800, 9000, 6000, confidence=0.6,
                 subject=vocab.SUBJECT_PERSON, actor="p1")  # overlaps only 200ms
    kept = _fold_redundant([said, no_actor, brief])
    assert no_actor in kept and brief in kept
    print("ok  fold requires shared actor + sufficient overlap")


def test_fold_does_not_drop_stronger_atom():
    """The weaker atom folds into the stronger; a higher-conf Shown is not eaten
    by a lower-conf Said-less Done."""
    strong = Atom(vocab.CHANNEL_DONE, 0, 4000, 2000, confidence=0.9,
                  subject=vocab.SUBJECT_PERSON, actor="p1")
    weak = Atom(vocab.CHANNEL_SHOWN, 0, 4000, 2000, confidence=0.4,
                subject=vocab.SUBJECT_PERSON, actor="p1")
    kept = _fold_redundant([strong, weak])
    assert strong in kept and weak not in kept
    assert "on_camera" in strong.flags
    print("ok  fold drops weaker, keeps stronger")


# -- build_atoms integration ---------------------------------------------------

def test_build_atoms_from_vlm():
    perception = {
        "speaking": [{"subject": "p1", "start_ms": 0, "end_ms": 4000}],
        "atoms": [
            {"channel": "shown", "subject": "person", "actor": "p1",
             "start_ms": 0, "end_ms": 4000, "confidence": 0.6},          # folds into said
            {"channel": "shown", "subject": "object", "start_ms": 5000,
             "end_ms": 8000, "confidence": 0.5, "label": "product"},     # survives
            {"channel": "done", "subject": "person", "start_ms": 9000,
             "end_ms": 11000, "confidence": 0.2, "label": "tiny move"},  # gated out
        ],
    }
    dialogue = {"sentence": [
        {"seg_id": "s0", "speaker": "S0", "text": "hi there", "flags": [],
         "src_in_ms": 0, "src_out_ms": 4000},
    ]}
    atoms = build_atoms(_clip(perception=perception, dialogue=dialogue))
    chans = sorted(a.channel for a in atoms)
    assert chans == [vocab.CHANNEL_SAID, vocab.CHANNEL_SHOWN], chans
    said = next(a for a in atoms if a.channel == vocab.CHANNEL_SAID)
    assert said.actor == "p1" and said.on_camera and "on_camera" in said.flags
    obj = next(a for a in atoms if a.channel == vocab.CHANNEL_SHOWN)
    assert obj.subject == vocab.SUBJECT_OBJECT
    print("ok  build_atoms from VLM atoms (gate+fold+said)")


def test_build_atoms_v1_fallback():
    """No `atoms` track -> derive from legacy content_units/cutaways so v2 runs
    on the existing corpus before L2 is re-run."""
    perception = {
        "content_units": [
            {"unit_id": "u0", "kind": "action", "primitive": "action",
             "start_ms": 1000, "end_ms": 3000, "confidence": 0.7, "label": "kick"},
        ],
        "cutaways": [
            {"id": "c0", "primitive": "place", "affordance": "broll",
             "start_ms": 5000, "end_ms": 8000, "salience_hint": 0.6},
        ],
    }
    atoms = build_atoms(_clip(perception=perception))
    by_ch = {a.channel for a in atoms}
    assert vocab.CHANNEL_DONE in by_ch and vocab.CHANNEL_SHOWN in by_ch
    place = next(a for a in atoms if a.channel == vocab.CHANNEL_SHOWN)
    assert place.subject == vocab.SUBJECT_PLACE
    print("ok  build_atoms v1 fallback")


def test_build_atoms_sorted_and_clamped():
    perception = {"atoms": [
        {"channel": "shown", "subject": "object", "start_ms": 50000,
         "end_ms": 999999, "confidence": 0.6},
        {"channel": "done", "subject": "object", "start_ms": 1000,
         "end_ms": 3000, "confidence": 0.6},
    ]}
    atoms = build_atoms(_clip(perception=perception, duration_ms=60000))
    assert [a.start_ms for a in atoms] == sorted(a.start_ms for a in atoms)
    assert all(a.end_ms <= 60000 for a in atoms)
    print("ok  build_atoms sorted + clamped to clip")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("\nall atom tests passed")
