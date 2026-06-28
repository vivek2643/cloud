"""
Tests for the editing vocabulary's layered model: capture primitives (intrinsic)
vs editor-facing affordances/views (derived). Run:
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


def test_primitive_and_view_sets_are_disjoint_and_closed():
    # The two visual-overlapping names (action) are intentionally shared in
    # spelling but the SETS are distinct concepts; assert membership is exact.
    assert v.is_capture_primitive(v.PRIM_PERSON)
    assert v.is_capture_primitive(v.PRIM_SPEECH)
    assert not v.is_capture_primitive(v.VIEW_REACTION)
    assert v.is_derived_view(v.VIEW_REACTION)
    assert v.is_derived_view(v.VIEW_BROLL)
    assert v.is_derived_view(v.VIEW_MOMENT)
    assert not v.is_derived_view(v.PRIM_PLACE)
    # Audio vs visual split.
    assert v.PRIM_SPEECH in v.AUDIO_PRIMITIVES
    assert v.PRIM_SPEECH not in v.VISUAL_PRIMITIVES
    assert set(v.CAPTURE_PRIMITIVES) == set(v.VISUAL_PRIMITIVES) | set(v.AUDIO_PRIMITIVES)
    print("ok  primitive/view sets closed and split")


def test_every_affordance_maps_to_a_primitive():
    for aff in v.AFFORDANCES:
        p = v.primitive_for_affordance(aff)
        assert v.is_capture_primitive(p), (aff, p)
    print("ok  every affordance maps to a primitive")


def test_reaction_is_a_person_shot_broll_is_place_insert_is_graphic():
    # The whole point of the re-grounding: reaction/b-roll are NOT capture
    # primitives -- they're a person shot / a place used supplementary.
    assert v.primitive_for_affordance(v.AFF_REACTION) == v.PRIM_PERSON
    assert v.primitive_for_affordance(v.AFF_BROLL) == v.PRIM_PLACE
    assert v.primitive_for_affordance(v.AFF_INSERT) == v.PRIM_GRAPHIC
    assert v.primitive_for_affordance(v.AFF_SPEECH) == v.PRIM_SPEECH
    assert v.primitive_for_affordance(v.AFF_ACTION) == v.PRIM_ACTION
    print("ok  reaction=person, broll=place, insert=graphic")


def test_primitives_for_dedupes_and_preserves_order():
    # A multi-affordance cut (speech + reaction) -> speech + person, deduped.
    assert v.primitives_for([v.AFF_SPEECH, v.AFF_REACTION]) == [v.PRIM_SPEECH, v.PRIM_PERSON]
    # reaction + action both present -> person + action.
    assert v.primitives_for([v.AFF_REACTION, v.AFF_ACTION]) == [v.PRIM_PERSON, v.PRIM_ACTION]
    # Duplicate primitive (broll twice) collapses.
    assert v.primitives_for([v.AFF_BROLL, v.AFF_BROLL]) == [v.PRIM_PLACE]
    assert v.primitives_for([]) == []
    print("ok  primitives_for dedupes + ordered")


def main():
    test_primitive_and_view_sets_are_disjoint_and_closed()
    test_every_affordance_maps_to_a_primitive()
    test_reaction_is_a_person_shot_broll_is_place_insert_is_graphic()
    test_primitives_for_dedupes_and_preserves_order()
    print("\nall vocab tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
