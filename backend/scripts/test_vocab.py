"""
Tests for the cuts-v2 capture vocabulary: four CHANNELS (said/done/shown/heard)
+ an orthogonal SUBJECT tag. There is no affordance/primitive/view layer. Run:
    .venv/bin/python scripts/test_vocab.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import vocab as v  # noqa: E402


def test_channels_closed_and_surfaced():
    assert set(v.CHANNELS) == {v.CHANNEL_SAID, v.CHANNEL_DONE, v.CHANNEL_SHOWN, v.CHANNEL_HEARD}
    for c in v.CHANNELS:
        assert v.is_channel(c)
    # Heard is built but withheld from the editor/brain surface.
    assert set(v.SURFACED_CHANNELS) == {v.CHANNEL_SAID, v.CHANNEL_DONE, v.CHANNEL_SHOWN}
    assert v.is_surfaced_channel(v.CHANNEL_DONE)
    assert not v.is_surfaced_channel(v.CHANNEL_HEARD)
    print("ok  channels closed + surfaced set")


def test_subjects_closed():
    assert set(v.SUBJECTS) == {v.SUBJECT_PERSON, v.SUBJECT_PLACE, v.SUBJECT_OBJECT, v.SUBJECT_GRAPHIC}
    for s in v.SUBJECTS:
        assert v.is_subject(s)
    assert not v.is_subject(v.CHANNEL_DONE)   # a channel is not a subject
    print("ok  subjects closed")


def test_labels():
    assert v.channel_label(v.CHANNEL_SAID) == "said"
    assert v.channel_label("SHOWN") == "shown"        # case-insensitive
    assert v.channel_label("nonsense") == "nonsense"  # passthrough
    print("ok  channel labels")


def test_no_legacy_vocabulary():
    # The v1 affordance/primitive/view layer is fully gone.
    for gone in ("AFFORDANCES", "AFF_SPEECH", "PRIM_PERSON", "primitives_for",
                 "channel_for_affordance", "DERIVED_VIEWS", "SOURCE_AFFORDANCE"):
        assert not hasattr(v, gone), gone
    print("ok  no legacy affordance/primitive vocabulary remains")


def main():
    test_channels_closed_and_surfaced()
    test_subjects_closed()
    test_labels()
    test_no_legacy_vocabulary()
    print("\nall vocab tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
