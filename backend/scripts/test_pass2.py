"""
Tests for the cuts-v3 pass-2 MERGED types module (``app.services.l3.pass2``)
-- ``Pass2Cut``/``Pass2Output`` (what post.py consumes) and
``merge_identity_and_visual`` (combining pass2a's identity output with
pass2b's visual judgments). No DB, no API calls -- the actual LLM-calling
logic lives in pass2a.py/pass2b.py and is tested there.

Run:  .venv/bin/python scripts/test_pass2.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import pass2  # noqa: E402
from app.services.l3.pass2a import IdentityCut, IdentityOutput  # noqa: E402
from app.services.l3.pass2b import Framing, Look, TasteFences, VisualJudgment  # noqa: E402


def test_pass2cut_constructs_with_all_fields():
    cut = pass2.Pass2Cut(
        source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 3),
        label="intro", summary="says hello", speaker="S0", on_camera=True,
        junk=False, junk_reason="", framing=Framing(rotation_deg=90.0),
        look=Look(graded=True), caption_zones=[(0.1, 0.1, 0.2, 0.1)],
        taste_fences=TasteFences(max_tasteful_speed=1.5), readability_ms=800,
        natural_sound=True, take_group_id="tg1", take_role="winner",
    )
    assert cut.framing.rotation_deg == 90.0
    assert cut.look.graded is True
    assert cut.take_role == "winner"
    print("ok  test_pass2cut_constructs_with_all_fields")


def test_merge_identity_and_visual_combines_fields_correctly():
    identity = IdentityOutput(cuts=[
        IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 3),
                   label="intro", summary="says hello", speaker="S0", on_camera=True,
                   junk=False, natural_sound=True, take_group_id="tg1", take_role="winner"),
    ])
    visual_by_index = {
        0: VisualJudgment(cut_index=0, framing=Framing(rotation_deg=90.0), look=Look(graded=True),
                         caption_zones=[(0.1, 0.1, 0.2, 0.1)],
                         taste_fences=TasteFences(max_tasteful_speed=1.5), readability_ms=800),
    }
    merged = pass2.merge_identity_and_visual(identity, visual_by_index)
    assert len(merged.cuts) == 1
    cut = merged.cuts[0]
    assert cut.source_ref == "speech_cut[0]" and cut.kind == "speech" and cut.word_span == (0, 3)
    assert cut.label == "intro" and cut.summary == "says hello" and cut.speaker == "S0"
    assert cut.on_camera is True and cut.natural_sound is True
    assert cut.take_group_id == "tg1" and cut.take_role == "winner"
    assert cut.framing.rotation_deg == 90.0
    assert cut.look.graded is True
    assert cut.caption_zones == [(0.1, 0.1, 0.2, 0.1)]
    assert cut.taste_fences.max_tasteful_speed == 1.5
    assert cut.readability_ms == 800
    print("ok  test_merge_identity_and_visual_combines_fields_correctly")


def test_merge_identity_and_visual_raises_when_a_judgment_is_missing():
    identity = IdentityOutput(cuts=[
        IdentityCut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 1),
                   label="a", summary="a"),
        IdentityCut(source_ref="speech_cut[1]", kind="speech", file_id="f1", word_span=(2, 3),
                   label="b", summary="b"),
    ])
    visual_by_index = {0: VisualJudgment(cut_index=0)}   # index 1 missing
    try:
        pass2.merge_identity_and_visual(identity, visual_by_index)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "1" in str(e), e
    print("ok  test_merge_identity_and_visual_raises_when_a_judgment_is_missing")


def test_merge_identity_and_visual_empty_is_a_noop():
    merged = pass2.merge_identity_and_visual(IdentityOutput(), {})
    assert merged.cuts == []
    print("ok  test_merge_identity_and_visual_empty_is_a_noop")


def main():
    test_pass2cut_constructs_with_all_fields()
    test_merge_identity_and_visual_combines_fields_correctly()
    test_merge_identity_and_visual_raises_when_a_judgment_is_missing()
    test_merge_identity_and_visual_empty_is_a_noop()
    print("\nall pass2 (merged types) tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
