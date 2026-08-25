"""
Tests for app.services.l3.converse's context-block assembly: the PROJECT
DOMAIN block (cut_structure_and_scene_specificity.plan.md Part 3) and the
Part 4 self-model/provenance section. No DB, no network -- EditContext is
hand-built the same way test_observe_act.py/test_tools_loop.py do.

Run:  .venv/bin/python scripts/test_converse_context.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import converse, observe  # noqa: E402
from app.services.l3.arrange import _MapIndex  # noqa: E402


def _ctx(scene_taxonomy=None):
    struct = {"clips": []}
    return observe.EditContext(
        file_ids=[], index=_MapIndex(struct), map_struct=struct,
        durations={}, dup_groups=[], scene_taxonomy=scene_taxonomy)


# --------------------------------------------------------------------------
# Part 3: PROJECT DOMAIN block
# --------------------------------------------------------------------------

def test_scene_domain_block_empty_when_no_taxonomy():
    assert converse._scene_domain_block(_ctx()) == ""
    print("ok  test_scene_domain_block_empty_when_no_taxonomy")


def test_scene_domain_block_empty_when_domain_unknown_mixed():
    ctx = _ctx(scene_taxonomy={"domain": "unknown/mixed", "confidence": "low"})
    assert converse._scene_domain_block(ctx) == ""
    print("ok  test_scene_domain_block_empty_when_domain_unknown_mixed")


def test_scene_domain_block_renders_domain_evidence_and_taxonomy():
    ctx = _ctx(scene_taxonomy={
        "domain": "CNC machine shop", "confidence": "med",
        "evidence": ["repeated lathe references"],
        "taxonomy": [{"id": "milling", "definition": "cutting with a rotating tool"}],
    })
    block = converse._scene_domain_block(ctx)
    assert "CNC machine shop" in block, block
    assert "repeated lathe references" in block, block
    assert "milling" in block and "cutting with a rotating tool" in block, block
    print("ok  test_scene_domain_block_renders_domain_evidence_and_taxonomy")


# --------------------------------------------------------------------------
# Part 4: self-model / provenance block (REQUIRED, kept in sync with Part 1
# cut-formation changes and Part 3 scene specificity in the same PR).
# --------------------------------------------------------------------------

def test_provenance_block_describes_content_first_cut_formation():
    """cuts_content_first_segmentation.plan.md: cut formation flipped from
    structure-first (camera/blur decide WHERE) to content-first (action
    novelty/runs/lulls, scene/composition change, or one representative
    cycle decide WHERE; camera/blur only refine/snap)."""
    p = converse._PROVENANCE
    assert "content-first" in p, p
    assert "Camera motion and blur only REFINE" in p, p
    assert "dropped entirely" in p, p
    assert "VLM never decides WHERE a cut is" in p, p
    print("ok  test_provenance_block_describes_content_first_cut_formation")


def test_provenance_block_describes_the_energy_dials_limit():
    """Part 4-style follow-through: the dial subdivides an already-located
    cut but cannot invent or recover a missed boundary -- descriptive of
    mechanism + limit, not a command to prefer one signal."""
    p = converse._PROVENANCE
    assert "invent a boundary the content layer" in p, p
    assert "recover one a dead stretch caused to be dropped" in p, p
    print("ok  test_provenance_block_describes_the_energy_dials_limit")


def test_provenance_block_describes_scene_specificity():
    p = converse._PROVENANCE
    assert "SCENE SPECIFICITY" in p, p
    assert "spec:" in p, p
    assert "PROJECT DOMAIN" in p, p
    assert "ADDITIVE" in p and "never replacing it" in p, p
    print("ok  test_provenance_block_describes_scene_specificity")


def test_provenance_block_has_no_trust_dictating_language():
    """Describes mechanism + limits, never a command to prefer one signal
    over another (cut_structure_and_scene_specificity.plan.md Part 4's own
    principle)."""
    p = converse._PROVENANCE.lower()
    for phrase in ("trust ", "always prefer", "more reliable than", "you should trust"):
        assert phrase not in p, phrase
    print("ok  test_provenance_block_has_no_trust_dictating_language")


def test_provenance_is_inside_the_cached_prefix_before_the_context_block():
    """respond()'s assembly: _LOOP_SYSTEM + _guidance_block() + _PROVENANCE,
    THEN the dynamic per-turn _context_block -- so the self-model stays
    static (cacheable) and the beat index remains the tail."""
    import inspect
    src = inspect.getsource(converse.respond)
    assert "_LOOP_SYSTEM + _guidance_block() + _PROVENANCE" in src.replace("\n", " ").replace("  ", " "), src
    print("ok  test_provenance_is_inside_the_cached_prefix_before_the_context_block")


# --------------------------------------------------------------------------
# brain_cut_index_fidelity.plan.md Stage 1: vocabulary (beat -> cut) and the
# quality-column honesty fixes (A5/B6), plus the stale nrg: legend drift.
# --------------------------------------------------------------------------

def test_loop_system_uses_cut_vocabulary_not_beat():
    """1.1: the addressable unit is a CUT everywhere the brain reads -- the
    legend must say CUT INDEX / cut line, never the retired BEAT wording."""
    s = converse._LOOP_SYSTEM
    assert "CUT INDEX" in s, s
    assert "READING A CUT LINE" in s, s
    assert "BEAT INDEX" not in s, s
    assert "beat" not in s.lower(), s
    print("ok  test_loop_system_uses_cut_vocabulary_not_beat")


def test_provenance_uses_cut_vocabulary_not_beat():
    p = converse._PROVENANCE
    assert "CUT INDEX" in p, p
    assert "BEAT INDEX" not in p, p
    assert "beat" not in p.lower(), p
    print("ok  test_provenance_uses_cut_vocabulary_not_beat")


def test_loop_system_legend_no_longer_claims_nrg():
    """The nrg: tag was renamed to energy:/levels: (brain_material_truth.
    plan.md Part 3); the legend describing it must not still say `nrg:`,
    and must separately explain both real tags."""
    s = converse._LOOP_SYSTEM
    assert "nrg:" not in s, s
    assert "`energy:`" in s, s
    assert "`levels:`" in s, s
    print("ok  test_loop_system_legend_no_longer_claims_nrg")


def test_provenance_section_6_no_longer_claims_nrg():
    p = converse._PROVENANCE
    assert "nrg:" not in p, p
    assert "energy: is this cut's own grade" in p, p
    assert "levels: and the pace tag" in p, p
    print("ok  test_provenance_section_6_no_longer_claims_nrg")


def test_provenance_scoring_section_no_longer_claims_qxx_is_visual_score():
    """1.3/B6: the old claim ('PIC's q.XX is that visual score') was false
    for a video cut on the vcut pipeline (it's mean cut-cleanliness, not a
    visual judgement). The new text must not make that claim, must use the
    new score. token, and must say cut-cleanliness != looks good."""
    p = converse._PROVENANCE
    assert "q.XX is that visual score" not in p, p
    assert "score.XX" in p, p
    assert "cut-cleanliness" in p, p
    assert "not how good it" in p or "not a visual-quality judgement" in p, p
    print("ok  test_provenance_scoring_section_no_longer_claims_qxx_is_visual_score")


def test_provenance_scoring_section_warns_scores_not_comparable_across_kind():
    """A5's whole point: after normalizing onto one scale the NUMBERS are
    comparable in magnitude, but what they measure still differs -- the
    text must say so, not imply a video score.80 and speech score.80 are
    the same kind of good."""
    p = converse._PROVENANCE
    assert "never in what they measure" in p, p
    print("ok  test_provenance_scoring_section_warns_scores_not_comparable_across_kind")


# --------------------------------------------------------------------------
# brain_cut_index_fidelity.plan.md Stage 2 (B1/B7): the piece/`range:` block
# was rendered on every multi-event cut but never explained anywhere in the
# prompt.
# --------------------------------------------------------------------------

def test_loop_system_explains_the_range_block():
    s = converse._LOOP_SYSTEM
    assert "`range:`" in s, s
    assert "core" in s, s
    assert "place(ref, piece=k)" in s, s
    print("ok  test_loop_system_explains_the_range_block")


def test_loop_system_range_explanation_has_no_beat_wording():
    s = converse._LOOP_SYSTEM
    assert "beat" not in s.lower(), s
    print("ok  test_loop_system_range_explanation_has_no_beat_wording")


def test_provenance_section_6_explains_pieces_derive_from_salience_events():
    p = converse._PROVENANCE
    assert "range:` block's PIECES are exactly these events" in p, p
    assert "does not create or destroy pieces" in p, p
    print("ok  test_provenance_section_6_explains_pieces_derive_from_salience_events")


# --------------------------------------------------------------------------
# brain_cut_index_fidelity.plan.md Stage 4 (B3/B4/B5): sig: now renders
# offsets (A3), so the three prompt sites that pointed the brain at
# inspect_cut for "the full offsets" behind a count must land in the SAME
# commit, or the prompt tells the brain to spend a turn on data already on
# the line.
# --------------------------------------------------------------------------

def test_loop_system_sig_legend_describes_offsets_not_counts():
    s = converse._LOOP_SYSTEM
    assert "sig:act+1.2s,+3.4s|shot+2.0s!" in s, s
    assert "act3,shot1" not in s, s
    assert "rhythm track" in s.lower(), s
    print("ok  test_loop_system_sig_legend_describes_offsets_not_counts")


def test_provenance_section_8_describes_offsets_not_counts():
    p = converse._PROVENANCE
    assert "reports the COUNT on" not in p, p
    assert "reports each channel's own OFFSETS" in p, p
    print("ok  test_provenance_section_8_describes_offsets_not_counts")


def test_provenance_section_9_says_inspect_cut_role_is_narrower():
    """B5: inspect_cut's marginal value shifts to curves/full specifics once
    offsets are resident -- the text must say so, not still frame it as the
    way to get "the full offsets"."""
    p = converse._PROVENANCE
    assert "already resident on every cut line" in p, p
    assert "role is narrower now" in p, p
    assert "not for offsets you can already read off sig:" in p, p
    print("ok  test_provenance_section_9_says_inspect_cut_role_is_narrower")


# --------------------------------------------------------------------------
# brain_cut_index_fidelity.plan.md Stage 5 (A7): the dup-audio: cross-
# reference between an overlapping shown/said pair sharing the same file
# needs a legend entry, or the brain has no idea what the tag means.
# --------------------------------------------------------------------------

def test_loop_system_explains_dup_audio_tag():
    s = converse._LOOP_SYSTEM
    assert "`dup-audio:`" in s, s
    assert "plays the identical words twice" in s, s
    print("ok  test_loop_system_explains_dup_audio_tag")


def main():
    test_scene_domain_block_empty_when_no_taxonomy()
    test_scene_domain_block_empty_when_domain_unknown_mixed()
    test_scene_domain_block_renders_domain_evidence_and_taxonomy()
    test_provenance_block_describes_content_first_cut_formation()
    test_provenance_block_describes_the_energy_dials_limit()
    test_provenance_block_describes_scene_specificity()
    test_provenance_block_has_no_trust_dictating_language()
    test_provenance_is_inside_the_cached_prefix_before_the_context_block()
    test_loop_system_uses_cut_vocabulary_not_beat()
    test_provenance_uses_cut_vocabulary_not_beat()
    test_loop_system_legend_no_longer_claims_nrg()
    test_provenance_section_6_no_longer_claims_nrg()
    test_provenance_scoring_section_no_longer_claims_qxx_is_visual_score()
    test_provenance_scoring_section_warns_scores_not_comparable_across_kind()
    test_loop_system_explains_the_range_block()
    test_loop_system_range_explanation_has_no_beat_wording()
    test_provenance_section_6_explains_pieces_derive_from_salience_events()
    test_loop_system_sig_legend_describes_offsets_not_counts()
    test_provenance_section_8_describes_offsets_not_counts()
    test_provenance_section_9_says_inspect_cut_role_is_narrower()
    test_loop_system_explains_dup_audio_tag()
    print("\nall converse-context tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
