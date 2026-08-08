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


def main():
    test_scene_domain_block_empty_when_no_taxonomy()
    test_scene_domain_block_empty_when_domain_unknown_mixed()
    test_scene_domain_block_renders_domain_evidence_and_taxonomy()
    test_provenance_block_describes_content_first_cut_formation()
    test_provenance_block_describes_the_energy_dials_limit()
    test_provenance_block_describes_scene_specificity()
    test_provenance_block_has_no_trust_dictating_language()
    test_provenance_is_inside_the_cached_prefix_before_the_context_block()
    print("\nall converse-context tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
