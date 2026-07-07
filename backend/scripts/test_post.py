"""
Pure unit tests for cuts-v3 post-compute assembly (``app.services.l3.post``)
-- no DB, no ffmpeg, no model calls.

Run:  .venv/bin/python scripts/test_post.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import post  # noqa: E402
from app.services.l3.lattice import Atom, Lattice  # noqa: E402
from app.services.l3.pass2 import Pass2Cut, Pass2Output  # noqa: E402


# --------------------------------------------------------------------------
# hero_ts_ms
# --------------------------------------------------------------------------

def test_hero_ts_prefers_anchor_over_sharp():
    ts = post.pick_hero_ts_ms([1500], blur=[0.9] * 10, hop_ms=100, s=0, e=2000)
    assert ts == 1500
    print("ok  test_hero_ts_prefers_anchor_over_sharp")


def test_hero_ts_falls_back_to_sharpest():
    blur = [0.9] * 10
    blur[3] = 0.05
    ts = post.pick_hero_ts_ms([], blur, hop_ms=100, s=0, e=1000)
    assert ts == 300, ts
    print("ok  test_hero_ts_falls_back_to_sharpest")


def test_hero_ts_falls_back_to_midpoint_with_no_blur():
    ts = post.pick_hero_ts_ms([], [], hop_ms=0, s=0, e=1000)
    assert ts == 500, ts
    print("ok  test_hero_ts_falls_back_to_midpoint_with_no_blur")


# --------------------------------------------------------------------------
# pace envelope
# --------------------------------------------------------------------------

def test_speech_pace_envelope_is_native_speed_only():
    pace = post.compute_pace_envelope(
        kind="speech", s=1000, e=3000, readability_ms=500, anchors=[],        action_energy=[0.1] * 40, hop_ms=100, next_cut_start_ms=3000,
        max_tasteful_speed=2.0, min_tasteful_speed=0.5, natural_sound=True,
    )
    assert pace.natural_ms == 2000, pace
    assert pace.min_ms == 2000, pace
    assert pace.max_ms == 2000, pace
    assert pace.levels == [1.0] * 5, pace
    assert pace.natural_sound is True
    print("ok  test_speech_pace_envelope_is_native_speed_only")


def test_video_pace_min_ms_from_anchor_span():
    pace = post.compute_pace_envelope(
        kind="video", s=1000, e=5000, readability_ms=0, anchors=[2000, 2400],        action_energy=[0.0] * 60, hop_ms=100, next_cut_start_ms=5000,
        max_tasteful_speed=2.0, min_tasteful_speed=0.5, natural_sound=False,
    )
    assert pace.min_ms == 900, pace   # (2400-2000) + 2*250
    print("ok  test_video_pace_min_ms_from_anchor_span")


def test_video_pace_max_ms_extends_through_flatline():
    pace = post.compute_pace_envelope(
        kind="video", s=1000, e=2000, readability_ms=0, anchors=[],        action_energy=[0.1] * 100, hop_ms=100, next_cut_start_ms=8000,
        max_tasteful_speed=2.0, min_tasteful_speed=0.5, natural_sound=False,
    )
    assert pace.max_ms == 7000, pace   # flat all the way to the ceiling (8000-1000)
    print("ok  test_video_pace_max_ms_extends_through_flatline")


def test_video_pace_max_ms_stops_at_departure_from_flatline():
    ae = [0.1] * 100
    ae[30] = 0.9   # departs the flat band at ts=3000
    pace = post.compute_pace_envelope(
        kind="video", s=1000, e=2000, readability_ms=0, anchors=[],        action_energy=ae, hop_ms=100, next_cut_start_ms=8000,
        max_tasteful_speed=2.0, min_tasteful_speed=0.5, natural_sound=False,
    )
    assert pace.max_ms == 2000, pace   # 3000 - s(1000)
    print("ok  test_video_pace_max_ms_stops_at_departure_from_flatline")


def test_pace_levels_partial_saturation_repeats_nearest_reachable():
    pace = post.compute_pace_envelope(
        kind="video", s=0, e=1000, readability_ms=0, anchors=[],        action_energy=[1.0] * 10, hop_ms=100, next_cut_start_ms=1000,
        max_tasteful_speed=1.2, min_tasteful_speed=0.25, natural_sound=False,
    )
    assert pace.levels == [0.5, 0.8, 1.0, 1.2, 1.2], pace.levels
    assert pace.levels == sorted(pace.levels), "levels must stay monotonic"
    print("ok  test_pace_levels_partial_saturation_repeats_nearest_reachable")


def test_pace_levels_zero_intrinsic_velocity_maxes_every_level():
    pace = post.compute_pace_envelope(
        kind="video", s=0, e=1000, readability_ms=0, anchors=[],        action_energy=[0.0] * 10, hop_ms=100, next_cut_start_ms=1000,
        max_tasteful_speed=2.0, min_tasteful_speed=0.5, natural_sound=False,
    )
    assert pace.levels == [2.0] * 5, pace.levels
    print("ok  test_pace_levels_zero_intrinsic_velocity_maxes_every_level")


def test_energy_grade_bands():
    assert post._energy_grade(0.05) == "calm"
    assert post._energy_grade(0.3) == "active"
    assert post._energy_grade(0.9) == "high"
    print("ok  test_energy_grade_bands")


# --------------------------------------------------------------------------
# coverage/overlap invariant
# --------------------------------------------------------------------------

def test_validate_no_overlap_passes_exact_coverage():
    post._validate_no_overlap("f1", [(0, 1000), (1000, 2000)], 2000)
    print("ok  test_validate_no_overlap_passes_exact_coverage")


def test_validate_no_overlap_allows_start_gap():
    # boundaries-v2: cuts are a selection, gaps are legal. Pre-roll dropped.
    post._validate_no_overlap("f1", [(100, 2000)], 2000)
    print("ok  test_validate_no_overlap_allows_start_gap")


def test_validate_no_overlap_allows_end_gap():
    post._validate_no_overlap("f1", [(0, 1900)], 2000)
    print("ok  test_validate_no_overlap_allows_end_gap")


def test_validate_no_overlap_raises_on_overlap():
    try:
        post._validate_no_overlap("f1", [(0, 1100), (1000, 2000)], 2000)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "overlap" in str(e)
    print("ok  test_validate_no_overlap_raises_on_overlap")


def test_validate_no_overlap_allows_middle_gap():
    # A dropped connective hold between two kept cuts -- legal now.
    post._validate_no_overlap("f1", [(0, 900), (1000, 2000)], 2000)
    print("ok  test_validate_no_overlap_allows_middle_gap")


def test_validate_no_overlap_allows_no_cuts():
    # An all-junk / all-dead-air file contributes zero cuts -- legal, not fatal.
    post._validate_no_overlap("f1", [], 2000)
    print("ok  test_validate_no_overlap_allows_no_cuts")


# --------------------------------------------------------------------------
# assemble_cut_records (end to end over a small synthetic file)
# --------------------------------------------------------------------------

def _words():
    return [
        {"start_ms": 0, "end_ms": 200, "text": "hey"},
        {"start_ms": 300, "end_ms": 500, "text": "there"},
        {"start_ms": 600, "end_ms": 800, "text": "friend"},
    ]


def _atoms():
    return [
        Atom(atom_id=0, file_id="f1", start_ms=800, end_ms=1400, state_in="speech_edge",
             state_out="shot", action_energy=0.2, coherence=0.9),
        Atom(atom_id=1, file_id="f1", start_ms=1400, end_ms=2000, state_in="shot",
             state_out="clip_edge", action_energy=0.4, coherence=0.8),
    ]


def _lattice():
    return Lattice(file_id="f1", duration_ms=2000, words=_words(), turns=[], hints=[], atoms=_atoms())


def _motion():
    return {"hop_ms": 100, "blur": [0.5] * 20, "action_energy": [0.2] * 20,
           "action_points": [{"ts_ms": 1700}]}


def test_assemble_cut_records_end_to_end():
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="f1", word_span=(0, 2),
                label="intro", summary="says hi", readability_ms=300),
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0, 1],
                label="pan shot", summary="pans across the desk"),
    ])
    records = post.assemble_cut_records(p2, {"f1": _lattice()}, {"f1": _motion()}, {})
    assert len(records) == 2, records
    speech, video = records
    assert (speech.src_in_ms, speech.src_out_ms) == (0, 800), speech
    assert (video.src_in_ms, video.src_out_ms) == (800, 2000), video
    assert video.hero_ts_ms == 1700, video   # anchor inside the video span wins
    assert video.pace.min_ms == 500, video.pace   # anchor-span floor: 0 + 2*250 (single anchor)
    assert video.pace.max_ms == 1200, video.pace   # last cut in file -> capped at natural span
    assert speech.pace.levels == [1.0] * 5
    print("ok  test_assemble_cut_records_end_to_end")


def test_junk_flag_is_preserved_verbatim():
    # Deterministic-keep: post no longer second-guesses the model's semantic
    # junk call with a hardcoded energy/anchor threshold. Whatever the model
    # (or pass-1 suspects) decided is carried through verbatim -- junk is a
    # recoverable label shown in the Discarded tray, never a code override.
    atoms = [Atom(atom_id=0, file_id="f1", start_ms=0, end_ms=2000, state_in="clip_edge",
                  state_out="clip_edge", action_energy=0.6, coherence=0.5)]
    lat = Lattice(file_id="f1", duration_ms=2000, words=[], turns=[], hints=[], atoms=atoms)
    motion = {"hop_ms": 100, "blur": [0.5] * 20, "action_energy": [0.6] * 20,
              "action_points": [{"ts_ms": 500}, {"ts_ms": 1200}, {"ts_ms": 1600}]}
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0],
                 label="cue", summary="and go", junk=True, junk_reason="production cue"),
    ])
    rec = post.assemble_cut_records(p2, {"f1": lat}, {"f1": motion}, {})[0]
    assert rec.junk is True and rec.junk_reason == "production cue", rec
    print("ok  test_junk_flag_is_preserved_verbatim")


def test_non_junk_stays_non_junk():
    atoms = [Atom(atom_id=0, file_id="f1", start_ms=0, end_ms=2000, state_in="clip_edge",
                  state_out="clip_edge", action_energy=0.05, coherence=0.95)]
    lat = Lattice(file_id="f1", duration_ms=2000, words=[], turns=[], hints=[], atoms=atoms)
    motion = {"hop_ms": 100, "blur": [0.5] * 20, "action_energy": [0.05] * 20, "action_points": []}
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0],
                 label="hold", summary="a quiet hold", junk=False),
    ])
    rec = post.assemble_cut_records(p2, {"f1": lat}, {"f1": motion}, {})[0]
    assert rec.junk is False, rec
    print("ok  test_non_junk_stays_non_junk")


def test_assemble_raises_on_unknown_file_id():
    p2 = Pass2Output(cuts=[Pass2Cut(source_ref="speech_cut[0]", kind="speech", file_id="missing",
                                    word_span=(0, 1), label="x", summary="y")])
    try:
        post.assemble_cut_records(p2, {}, {}, {})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "unknown file_id" in str(e)
    print("ok  test_assemble_raises_on_unknown_file_id")


def test_assemble_raises_on_unresolvable_atom_ids():
    p2 = Pass2Output(cuts=[Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1",
                                    atom_ids=[99], label="x", summary="y")])
    try:
        post.assemble_cut_records(p2, {"f1": _lattice()}, {"f1": _motion()}, {})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "no resolvable atoms" in str(e)
    print("ok  test_assemble_raises_on_unresolvable_atom_ids")


def test_assemble_raises_on_overlap_between_cuts():
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0],
                label="a", summary="a"),
        Pass2Cut(source_ref="video_group[1]", kind="video", file_id="f1", atom_ids=[0, 1],
                label="b", summary="b"),   # overlaps the first (both include atom 0)
    ])
    try:
        post.assemble_cut_records(p2, {"f1": _lattice()}, {"f1": _motion()}, {})
        assert False, "expected ValueError"
    except ValueError as e:
        assert "overlap" in str(e) or "gap" in str(e)
    print("ok  test_assemble_raises_on_overlap_between_cuts")


def test_assemble_allows_a_project_file_with_zero_cuts():
    # boundaries-v2: two files in the project, pass 2 reported cuts for only
    # one. An all-junk / all-dead-air clip contributing nothing is now a valid
    # outcome (a warning, not a raise); the run still succeeds with f1's cuts.
    p2 = Pass2Output(cuts=[
        Pass2Cut(source_ref="video_group[0]", kind="video", file_id="f1", atom_ids=[0, 1],
                label="a", summary="a"),
    ])
    lattices = {"f1": _lattice(), "f2": _lattice()}
    motion = {"f1": _motion(), "f2": _motion()}
    records = post.assemble_cut_records(p2, lattices, motion, {})
    assert records, "expected f1's cut to still be assembled"
    assert all(r.file_id == "f1" for r in records), [r.file_id for r in records]
    print("ok  test_assemble_allows_a_project_file_with_zero_cuts")


def test_one_winner_per_take_group_backstop():
    # Two clips of one take, both crowned "winner" by pass 2 -> the longest stays
    # winner, the shorter is demoted to "take"; an outlook is untouched.
    def rec(fid, s, e, role):
        return post.CutRecord(
            file_id=fid, src_in_ms=s, src_out_ms=e, kind="speech", word_span=(0, 4),
            atom_ids=None, label="line", summary="", speaker=None, on_camera=None,
            junk=False, junk_reason=None, framing={}, look={}, caption_zones=[],
            hero_ts_ms=s, pace=None, take_group_id="tg1", take_role=role, channel="said")
    recs = [rec("f1", 0, 3000, "winner"), rec("f2", 0, 2000, "winner"),
            rec("f3", 0, 2500, "outlook")]
    post._enforce_one_winner_per_take_group(recs)
    roles = {r.file_id: r.take_role for r in recs}
    assert roles == {"f1": "winner", "f2": "take", "f3": "outlook"}, roles
    print("ok  test_one_winner_per_take_group_backstop")


def main():
    test_one_winner_per_take_group_backstop()
    test_hero_ts_prefers_anchor_over_sharp()
    test_hero_ts_falls_back_to_sharpest()
    test_hero_ts_falls_back_to_midpoint_with_no_blur()
    test_speech_pace_envelope_is_native_speed_only()
    test_video_pace_min_ms_from_anchor_span()
    test_video_pace_max_ms_extends_through_flatline()
    test_video_pace_max_ms_stops_at_departure_from_flatline()
    test_pace_levels_partial_saturation_repeats_nearest_reachable()
    test_pace_levels_zero_intrinsic_velocity_maxes_every_level()
    test_energy_grade_bands()
    test_validate_no_overlap_passes_exact_coverage()
    test_validate_no_overlap_allows_start_gap()
    test_validate_no_overlap_allows_end_gap()
    test_validate_no_overlap_raises_on_overlap()
    test_validate_no_overlap_allows_middle_gap()
    test_validate_no_overlap_allows_no_cuts()
    test_assemble_cut_records_end_to_end()
    test_junk_flag_is_preserved_verbatim()
    test_non_junk_stays_non_junk()
    test_assemble_raises_on_unknown_file_id()
    test_assemble_raises_on_unresolvable_atom_ids()
    test_assemble_raises_on_overlap_between_cuts()
    test_assemble_allows_a_project_file_with_zero_cuts()
    print("\nall post tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
