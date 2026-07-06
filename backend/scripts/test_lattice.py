"""
Tests for the cuts-v3 lattice builder (``lattice.py``) -- no DB. Run:
  .venv/bin/python scripts/test_lattice.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import lattice as lt  # noqa: E402
from app.services.l3.base_cuts import R_CLIP, R_DISTURB, R_MOVE, R_SETTLE, R_SHOT  # noqa: E402
from app.services.l3.lattice import Atom, R_SPEECH_EDGE  # noqa: E402


def _flat_motion(n, hop=100, stability=0.95, camera_motion=0.02, action=0.05, coherence=0.95):
    return {
        "hop_ms": hop,
        "camera_stability": [stability] * n,
        "camera_motion": [camera_motion] * n,
        "camera_coherence": [coherence] * n,
        "action_energy": [action] * n,
        "action_points": [],
        "transition_points": [],
    }


# --------------------------------------------------------------------------
# build_atoms
# --------------------------------------------------------------------------

def test_atoms_never_overlap_a_speech_turn():
    """Video atoms are built ONLY over the non-speech remainder -- never
    inside a turn, by construction (never cut under speech)."""
    motion = _flat_motion(100)
    turns = [(2000, 5000, "S0")]
    atoms = lt.build_atoms("f1", 10_000, motion, None, turns)
    for a in atoms:
        assert a.end_ms <= 2000 or a.start_ms >= 5000, a
    print("ok  test_atoms_never_overlap_a_speech_turn")


def test_atoms_cover_the_whole_non_speech_remainder():
    """Full coverage of [0,duration) minus the speech span, no gaps."""
    motion = _flat_motion(100)
    turns = [(3000, 6000, "S0")]
    atoms = lt.build_atoms("f1", 10_000, motion, None, turns)
    before = sorted([a for a in atoms if a.end_ms <= 3000], key=lambda a: a.start_ms)
    after = sorted([a for a in atoms if a.start_ms >= 6000], key=lambda a: a.start_ms)
    assert before[0].start_ms == 0 and before[-1].end_ms == 3000, before
    assert after[0].start_ms == 6000 and after[-1].end_ms == 10_000, after
    for grp in (before, after):
        for x, y in zip(grp, grp[1:]):
            assert x.end_ms == y.start_ms, (x, y)
    print("ok  test_atoms_cover_the_whole_non_speech_remainder")


def test_a_trailing_sliver_shorter_than_min_atom_ms_still_gets_covered():
    """Regression: a real ingest run hit a 20ms coverage gap at a file's
    very end -- the whole-segment MIN_ATOM_MS check was dropping any
    non-speech remainder shorter than the floor instead of still emitting
    it as (a short) atom. There's nowhere else for that span to go."""
    motion = _flat_motion(100)
    turns = [(0, 9980, "S0")]   # leaves only a 20ms non-speech tail
    atoms = lt.build_atoms("f1", 10_000, motion, None, turns)
    assert atoms, "expected the 20ms tail to still produce an atom"
    assert atoms[0].start_ms == 9980 and atoms[-1].end_ms == 10_000, atoms
    print("ok  test_a_trailing_sliver_shorter_than_min_atom_ms_still_gets_covered")


def test_a_leading_sliver_shorter_than_min_atom_ms_still_gets_covered():
    motion = _flat_motion(100)
    turns = [(20, 10_000, "S0")]   # leaves only a 20ms non-speech head
    atoms = lt.build_atoms("f1", 10_000, motion, None, turns)
    assert atoms, "expected the 20ms head to still produce an atom"
    assert atoms[0].start_ms == 0 and atoms[-1].end_ms == 20, atoms
    print("ok  test_a_leading_sliver_shorter_than_min_atom_ms_still_gets_covered")


def test_no_speech_at_all_yields_one_whole_clip_atom():
    motion = _flat_motion(50)
    atoms = lt.build_atoms("f1", 5000, motion, None, [])
    assert len(atoms) == 1, atoms
    assert atoms[0].start_ms == 0 and atoms[0].end_ms == 5000
    assert atoms[0].state_in == R_CLIP and atoms[0].state_out == R_CLIP, atoms[0]
    print("ok  test_no_speech_at_all_yields_one_whole_clip_atom")


def test_shot_cut_always_splits_an_atom():
    motion = _flat_motion(100)
    scene = {"shot_points": [{"ts_ms": 5000, "kind": "shot_cut", "score": 1.0}]}
    atoms = lt.build_atoms("f1", 10_000, motion, scene, [])
    assert len(atoms) == 2, atoms
    assert atoms[0].end_ms == 5000 and atoms[1].start_ms == 5000
    assert atoms[0].state_out == R_SHOT or atoms[1].state_in == R_SHOT, atoms
    print("ok  test_shot_cut_always_splits_an_atom")


def test_camera_move_settle_splits_an_atom():
    """A hold -> move -> hold sequence yields three atoms, with move/settle
    reasons at the internal boundaries."""
    n = 100
    motion = _flat_motion(n)
    motion["camera_stability"] = [0.9] * 30 + [0.2] * 10 + [0.9] * 60
    motion["camera_motion"] = [0.05] * 30 + [0.6] * 10 + [0.05] * 60
    atoms = lt.build_atoms("f1", 10_000, motion, None, [])
    assert len(atoms) >= 2, atoms
    reasons = {a.state_in for a in atoms} | {a.state_out for a in atoms}
    assert R_MOVE in reasons or R_SETTLE in reasons, atoms
    print("ok  test_camera_move_settle_splits_an_atom")


def test_disturbance_edges_split_an_atom():
    n = 100
    motion = _flat_motion(n)
    # Bad camera (high camera_cut_cost) with no subject payoff (low action) --
    # a genuine disturbance span in the middle.
    motion["camera_cut_cost"] = [0.05] * 30 + [0.6] * 20 + [0.05] * 50
    atoms = lt.build_atoms("f1", 10_000, motion, None, [])
    reasons = {a.state_in for a in atoms} | {a.state_out for a in atoms}
    assert R_DISTURB in reasons, atoms
    print("ok  test_disturbance_edges_split_an_atom")


def test_transition_point_wipe_splits_an_atom():
    motion = _flat_motion(100)
    motion["transition_points"] = [{"ts_ms": 4000, "kind": "wipe", "strength": 0.4}]
    atoms = lt.build_atoms("f1", 10_000, motion, None, [])
    assert any(a.end_ms == 4000 for a in atoms), atoms
    assert any(a.start_ms == 4000 for a in atoms), atoms
    boundary = next(a for a in atoms if a.end_ms == 4000)
    assert boundary.state_out == lt.R_WIPE, boundary
    print("ok  test_transition_point_wipe_splits_an_atom")


def test_transition_point_degenerate_splits_an_atom():
    motion = _flat_motion(100)
    motion["transition_points"] = [{"ts_ms": 6000, "kind": "degenerate", "strength": 1.0}]
    atoms = lt.build_atoms("f1", 10_000, motion, None, [])
    boundary = next(a for a in atoms if a.end_ms == 6000)
    assert boundary.state_out == lt.R_DEGENERATE, boundary
    print("ok  test_transition_point_degenerate_splits_an_atom")


def test_no_motion_or_scene_data_is_a_safe_noop():
    atoms = lt.build_atoms("f1", 5000, None, None, [])
    assert len(atoms) == 1, atoms
    assert atoms[0].camera_desc == "hold"
    assert atoms[0].action_energy == 0.0 and atoms[0].coherence == 0.0
    print("ok  test_no_motion_or_scene_data_is_a_safe_noop")


def test_anchor_ms_captures_action_points_inside_the_atom():
    motion = _flat_motion(100)
    motion["action_points"] = [{"ts_ms": 4200, "kind": "action_impact", "score": 1.0}]
    atoms = lt.build_atoms("f1", 10_000, motion, None, [])
    assert len(atoms) == 1, atoms
    assert atoms[0].anchor_ms == [4200], atoms[0]
    print("ok  test_anchor_ms_captures_action_points_inside_the_atom")


def test_camera_desc_pan_vs_handheld_vs_hold():
    n = 60
    hold = lt._camera_desc(_flat_motion(n, camera_motion=0.02), 0, n * 100)
    pan = lt._camera_desc(_flat_motion(n, camera_motion=0.5, coherence=0.9, stability=0.8), 0, n * 100)
    handheld = lt._camera_desc(_flat_motion(n, camera_motion=0.5, coherence=0.3, stability=0.3), 0, n * 100)
    assert hold == "hold", hold
    assert pan == "pan", pan
    assert handheld == "handheld", handheld
    print("ok  test_camera_desc_pan_vs_handheld_vs_hold")


# --------------------------------------------------------------------------
# render_atom_table
# --------------------------------------------------------------------------

def test_render_atom_table_format():
    atom = lt.Atom(atom_id=7, file_id="f1", start_ms=12300, end_ms=15800,
                   state_in=R_MOVE, state_out=R_SETTLE, action_energy=0.7,
                   camera_desc="pan", coherence=0.9, anchor_ms=[13100])
    text = lt.render_atom_table([atom])
    assert text == "ATOM 7 [12300-15800] camera_move->settle act=0.70 cam=pan coh=0.90 anchors@13100", text
    print("ok  test_render_atom_table_format")


def test_render_atom_table_omits_anchors_when_none():
    atom = lt.Atom(atom_id=0, file_id="f1", start_ms=0, end_ms=1000,
                   state_in=R_CLIP, state_out=R_CLIP, action_energy=0.0,
                   camera_desc="hold", coherence=0.0, anchor_ms=[])
    text = lt.render_atom_table([atom])
    assert "anchors@" not in text, text
    print("ok  test_render_atom_table_omits_anchors_when_none")


# --------------------------------------------------------------------------
# speech_hints
# --------------------------------------------------------------------------

def _word(start, end, text, speaker="S0"):
    return {"start_ms": start, "end_ms": end, "text": text, "speaker": speaker, "is_filler": False}


def test_speech_hints_flags_long_pause():
    words = [_word(0, 500, "hello"), _word(2500, 3000, "world")]   # 2000ms gap
    hints = lt.speech_hints(words, turn_gap_ms=1200)
    assert len(hints) == 1 and "long pause after word 0" in hints[0], hints
    print("ok  test_speech_hints_flags_long_pause")


def test_speech_hints_flags_speaker_change_not_pause():
    words = [_word(0, 500, "hello", "S0"), _word(600, 1000, "hi", "S1")]
    hints = lt.speech_hints(words, turn_gap_ms=1200)
    assert len(hints) == 1 and "speaker change after word 0" in hints[0], hints
    print("ok  test_speech_hints_flags_speaker_change_not_pause")


def test_speech_hints_ignores_short_natural_gaps():
    words = [_word(0, 500, "hello", "S0"), _word(600, 1000, "there", "S0")]
    hints = lt.speech_hints(words, turn_gap_ms=1200)
    assert hints == [], hints
    print("ok  test_speech_hints_ignores_short_natural_gaps")


# --------------------------------------------------------------------------
# _snap_word_edge
# --------------------------------------------------------------------------

def test_snap_word_edge_before_first_and_after_last():
    words = [_word(1000, 1500, "a"), _word(2000, 2500, "b")]
    assert lt._snap_word_edge(words, 0, []) == 1000
    assert lt._snap_word_edge(words, 2, []) == 2500
    print("ok  test_snap_word_edge_before_first_and_after_last")


def test_snap_word_edge_uses_silence_interval_midpoint():
    words = [_word(1000, 1500, "a"), _word(2000, 2500, "b")]
    silences = [{"start_ms": 1400, "end_ms": 2100}]
    edge = lt._snap_word_edge(words, 1, silences)
    # Midpoint of the overlap between the silence [1400,2100] and the raw gap [1500,2000].
    assert edge == (1500 + 2000) // 2, edge
    print("ok  test_snap_word_edge_uses_silence_interval_midpoint")


def test_snap_word_edge_falls_back_to_gap_midpoint_without_silence_data():
    words = [_word(1000, 1500, "a"), _word(2000, 2500, "b")]
    edge = lt._snap_word_edge(words, 1, [])
    assert edge == (1500 + 2000) // 2, edge
    print("ok  test_snap_word_edge_falls_back_to_gap_midpoint_without_silence_data")


def test_snap_word_edge_touching_words_has_no_gap():
    words = [_word(1000, 1500, "a"), _word(1500, 2000, "b")]
    edge = lt._snap_word_edge(words, 1, [{"start_ms": 0, "end_ms": 5000}])
    assert edge == 1500, edge
    print("ok  test_snap_word_edge_touching_words_has_no_gap")


def test_snap_word_edge_empty_words_is_safe():
    assert lt._snap_word_edge([], 0, []) == 0
    print("ok  test_snap_word_edge_empty_words_is_safe")


# --------------------------------------------------------------------------
# resolve_speech_span_ms
# --------------------------------------------------------------------------

def _reel_trail_regression_fixture():
    """The exact word/atom layout that produced a real, reproducible
    coverage overlap in cuts-v3 (file 93991c78...): a 2.76s pause between
    words 20 and 21, almost entirely carved into video atoms, where the
    raw gap-midpoint snap for a speech cut ending at word 20 landed at
    ms=8240 -- deep inside the atoms' own [6860-9620) span."""
    words = [
        {"start_ms": 0, "end_ms": 400, "text": "and"},
        {"start_ms": 400, "end_ms": 740, "text": "go"},
        {"start_ms": 740, "end_ms": 1640, "text": "brands"},
        {"start_ms": 6580, "end_ms": 6860, "text": "you"},
        {"start_ms": 9620, "end_ms": 10220, "text": "there's"},
        {"start_ms": 10220, "end_ms": 10300, "text": "a"},
    ]
    atoms = [
        Atom(atom_id=0, file_id="f1", start_ms=6860, end_ms=7700, state_in=R_SPEECH_EDGE,
             state_out=R_SETTLE, action_energy=0.1, camera_desc="hold", coherence=0.9),
        Atom(atom_id=1, file_id="f1", start_ms=7700, end_ms=9200, state_in=R_SETTLE,
             state_out=R_MOVE, action_energy=0.3, camera_desc="pan", coherence=0.8),
        Atom(atom_id=2, file_id="f1", start_ms=9200, end_ms=9620, state_in=R_MOVE,
             state_out=R_SPEECH_EDGE, action_energy=0.2, camera_desc="hold", coherence=0.9),
    ]
    return words, atoms


def test_resolve_speech_span_ms_clamps_end_to_the_next_atom():
    words, atoms = _reel_trail_regression_fixture()
    # word_span (0, 3): "and go brands ... you" -- ends at word 3 (index),
    # the raw gap-midpoint snap for word 4 ("there's") would land at
    # (6860+9620)//2 = 8240, deep inside atom 0/1's span.
    s, e = lt.resolve_speech_span_ms(words, atoms, (0, 3), [])
    assert e == 6860, e   # clamped to where atom 0 begins, not 8240
    assert s == 0, s
    print("ok  test_resolve_speech_span_ms_clamps_end_to_the_next_atom")


def test_resolve_speech_span_ms_clamps_start_to_the_preceding_atom():
    words, atoms = _reel_trail_regression_fixture()
    # word_span (4, 5): "there's a" -- starts right after the same atom run;
    # the raw gap-midpoint snap for the start would reach back toward 8240,
    # inside atom 1/2's span, unless clamped to atom 2's own end (9620).
    s, e = lt.resolve_speech_span_ms(words, atoms, (4, 5), [])
    assert s == 9620, s   # clamped to where atom 2 ends, not the raw midpoint
    print("ok  test_resolve_speech_span_ms_clamps_start_to_the_preceding_atom")


def test_resolve_speech_span_ms_is_a_noop_without_atoms():
    words, _atoms = _reel_trail_regression_fixture()
    s, e = lt.resolve_speech_span_ms(words, [], (0, 3), [])
    assert e == (6860 + 9620) // 2, e   # unclamped raw gap-midpoint
    print("ok  test_resolve_speech_span_ms_is_a_noop_without_atoms")


def main():
    test_atoms_never_overlap_a_speech_turn()
    test_atoms_cover_the_whole_non_speech_remainder()
    test_a_trailing_sliver_shorter_than_min_atom_ms_still_gets_covered()
    test_a_leading_sliver_shorter_than_min_atom_ms_still_gets_covered()
    test_no_speech_at_all_yields_one_whole_clip_atom()
    test_shot_cut_always_splits_an_atom()
    test_camera_move_settle_splits_an_atom()
    test_disturbance_edges_split_an_atom()
    test_transition_point_wipe_splits_an_atom()
    test_transition_point_degenerate_splits_an_atom()
    test_no_motion_or_scene_data_is_a_safe_noop()
    test_anchor_ms_captures_action_points_inside_the_atom()
    test_camera_desc_pan_vs_handheld_vs_hold()
    test_render_atom_table_format()
    test_render_atom_table_omits_anchors_when_none()
    test_speech_hints_flags_long_pause()
    test_speech_hints_flags_speaker_change_not_pause()
    test_speech_hints_ignores_short_natural_gaps()
    test_snap_word_edge_before_first_and_after_last()
    test_snap_word_edge_uses_silence_interval_midpoint()
    test_snap_word_edge_falls_back_to_gap_midpoint_without_silence_data()
    test_snap_word_edge_touching_words_has_no_gap()
    test_snap_word_edge_empty_words_is_safe()
    test_resolve_speech_span_ms_clamps_end_to_the_next_atom()
    test_resolve_speech_span_ms_clamps_start_to_the_preceding_atom()
    test_resolve_speech_span_ms_is_a_noop_without_atoms()
    print("\nall lattice tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
