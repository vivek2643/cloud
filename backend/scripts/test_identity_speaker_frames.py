"""
Tests for deterministic close-burst frame planning
(app.services.l3.identity.speaker_frames) -- no DB, no network, no model
call, synthetic turns/signals only.

Run:  .venv/bin/python scripts/test_identity_speaker_frames.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import pass2  # noqa: E402
from app.services.l3.identity import speaker_frames as sf  # noqa: E402
from app.services.l3.lattice import Lattice  # noqa: E402


def test_merge_intervals_joins_overlapping_and_touching():
    merged = sf._merge_intervals([(100, 200), (150, 300), (400, 500)])
    assert merged == [(100, 300), (400, 500)], merged
    print("ok  test_merge_intervals_joins_overlapping_and_touching")


def test_largest_clean_subspan_picks_the_bigger_piece():
    # [0, 1000) with a forbidden zone [200, 600) -> pieces [0,200) (len 200)
    # and [600,1000) (len 400): the second is bigger.
    span = sf._largest_clean_subspan(0, 1000, [(200, 600)])
    assert span == (600, 1000), span
    print("ok  test_largest_clean_subspan_picks_the_bigger_piece")


def test_largest_clean_subspan_none_when_fully_covered():
    assert sf._largest_clean_subspan(100, 200, [(0, 300)]) is None
    print("ok  test_largest_clean_subspan_none_when_fully_covered")


def test_clean_windows_carves_around_a_nearby_other_voice_turn():
    # V's turn [1000, 3000); another voice speaks [2800, 3200) -- guard=150
    # forbids [2650, 3350), so the clean sub-span is [1000, 2650) (len 1650).
    turns = [("f1", 1000, 3000)]
    others = {"f1": [(2800, 3200)]}
    windows = sf.clean_windows(turns, others)
    assert windows == [("f1", 1000, 2650)], windows
    print("ok  test_clean_windows_carves_around_a_nearby_other_voice_turn")


def test_clean_windows_drops_a_turn_with_no_span_above_min_win_ms():
    # A 500ms turn entirely swallowed by a nearby other-voice guard zone.
    turns = [("f1", 1000, 1500)]
    others = {"f1": [(1500, 1600)]}   # guard 150 -> forbids [1350, 1750)
    windows = sf.clean_windows(turns, others)
    assert windows == [], windows
    print("ok  test_clean_windows_drops_a_turn_with_no_span_above_min_win_ms")


def test_loudness_peak_ms_finds_the_argmax_hop():
    rms = [-40.0, -40.0, -10.0, -40.0]
    assert sf._loudness_peak_ms(rms, 100, 0, 400) == 200
    print("ok  test_loudness_peak_ms_finds_the_argmax_hop")


def test_loudness_peak_ms_falls_back_to_midpoint_with_no_signal():
    assert sf._loudness_peak_ms([], 0, 1000, 2000) == 1500
    print("ok  test_loudness_peak_ms_falls_back_to_midpoint_with_no_signal")


def test_burst_offsets_scale_and_center_by_window_length():
    # N=5, D_MS=100: a roomy window fits all 5 centered on 0; shorter windows
    # shrink the count but stay symmetric; a tiny window collapses to [0].
    assert sf._burst_offsets(1000) == [-200, -100, 0, 100, 200]
    assert sf._burst_offsets(300) == [-150, -50, 50, 150]   # fits 4
    assert sf._burst_offsets(120) == [-50, 50]              # fits 2
    assert sf._burst_offsets(50) == [0]
    print("ok  test_burst_offsets_scale_and_center_by_window_length")


def test_burst_ts_stays_inside_the_window_and_snaps_to_sharpness():
    # A sharp (low-blur) instant sits right at the window edge; the burst
    # offset near it should snap there instead of the blurrier raw instant.
    blur = [0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
    ts = sf._burst_ts(t_star=500, s=0, e=1000, blur=blur, hop_ms=100)
    assert len(ts) == 5
    assert all(0 <= t < 1000 for t in ts), ts
    print("ok  test_burst_ts_stays_inside_the_window_and_snaps_to_sharpness")


def test_window_score_rewards_loud_sharp_isolated_prominent():
    quiet_blurry = sf._window_score(
        t_star=500, isolation_margin_ms=0,
        rms_db=[-40.0] * 10, rms_hop_ms=100, rms_lo=-40.0, rms_hi=-10.0,
        blur=[0.9] * 10, blur_hop_ms=100, blur_lo=0.1, blur_hi=0.9,
        subject_box_area=None, has_position=False,
    )
    loud_sharp = sf._window_score(
        t_star=500, isolation_margin_ms=2 * sf.GUARD_MS,
        rms_db=[-10.0] * 10, rms_hop_ms=100, rms_lo=-40.0, rms_hi=-10.0,
        blur=[0.1] * 10, blur_hop_ms=100, blur_lo=0.1, blur_hi=0.9,
        subject_box_area=0.3, has_position=True,
    )
    assert loud_sharp > quiet_blurry, (loud_sharp, quiet_blurry)
    print("ok  test_window_score_rewards_loud_sharp_isolated_prominent")


def _lat(words):
    return Lattice(file_id="f1", duration_ms=10000, words=words, turns=[], hints=[], atoms=[])


def _cut(ref, file_id, word_span, people, subject_box=None):
    return pass2.Pass2Cut(
        source_ref=ref, kind="speech", file_id=file_id, word_span=word_span,
        label="x", summary="y", people=people,
        framing=pass2.Framing(subject_box=subject_box),
    )


def test_covering_candidates_finds_the_visible_person_in_the_covering_cut():
    words = [{"start_ms": i * 200, "end_ms": i * 200 + 150} for i in range(10)]
    lattice = _lat(words)
    cuts = [_cut("speech_cut[0]", "f1", (0, 4), [{"position": "center"}], subject_box=(0.2, 0.2, 0.3, 0.3))]
    visible = {("f1", "speech_cut[0]"): ["P0"]}
    cands = sf._covering_candidates("f1", 0, 900, {"f1": cuts}, lattice, visible)
    assert cands == [("P0", "speech_cut[0]")], cands
    print("ok  test_covering_candidates_finds_the_visible_person_in_the_covering_cut")


def test_covering_candidates_empty_when_no_cut_covers_the_window():
    words = [{"start_ms": i * 200, "end_ms": i * 200 + 150} for i in range(10)]
    lattice = _lat(words)
    cuts = [_cut("speech_cut[0]", "f1", (0, 4), [{"position": "center"}])]
    visible = {("f1", "speech_cut[0]"): ["P0"]}
    cands = sf._covering_candidates("f1", 5000, 6000, {"f1": cuts}, lattice, visible)
    assert cands == [], cands
    print("ok  test_covering_candidates_empty_when_no_cut_covers_the_window")


def test_plan_bursts_end_to_end_two_speakers_one_narrator():
    # f1: P0 speaks [0, 2000), P1 speaks [2500, 4500) -- clean, well
    # separated turns, each covered by its own cut showing that person.
    # A third voice V2 (narrator) has a turn but NO covering cut/visible
    # person anywhere -> must come back off-camera.
    words = [{"start_ms": i * 100, "end_ms": i * 100 + 90} for i in range(50)]
    lattice = _lat(words)
    turns_by_file = {"f1": [(0, 2000, "S0"), (2500, 4500, "S1"), (5000, 7000, "S2")]}
    voice_of = {("f1", "S0"): "V0", ("f1", "S1"): "V1", ("f1", "S2"): "V2"}
    cuts = [
        _cut("speech_cut[0]", "f1", (0, 19), [{"position": "center"}], subject_box=(0.3, 0.2, 0.3, 0.4)),
        _cut("speech_cut[1]", "f1", (25, 44), [{"position": "left"}], subject_box=(0.1, 0.2, 0.25, 0.4)),
        # No cut covers V2's [5000,7000) window at all -- pure narration.
    ]
    visible_persons = {("f1", "speech_cut[0]"): ["P0"], ("f1", "speech_cut[1]"): ["P1"]}
    audio_by_file = {"f1": {"rms_db": [-20.0] * 100, "hop_ms": 50}}
    motion_by_file = {"f1": {"blur": [0.3] * 100, "hop_ms": 50}}

    bursts, off_camera = sf.plan_bursts(
        turns_by_file, voice_of, {"f1": cuts}, {"f1": lattice},
        visible_persons, audio_by_file, motion_by_file,
    )
    assert off_camera == {"V2"}, off_camera
    voices_with_bursts = {b.voice for b in bursts}
    assert voices_with_bursts == {"V0", "V1"}, voices_with_bursts
    v0_bursts = [b for b in bursts if b.voice == "V0"]
    assert all(b.candidate_person == "P0" for b in v0_bursts), v0_bursts
    assert all(len(b.ts_ms) == 5 for b in bursts), bursts
    print("ok  test_plan_bursts_end_to_end_two_speakers_one_narrator")


def test_plan_bursts_multi_candidate_window_frames_every_candidate():
    # One shared camera shows P0 AND P1 at once while V0 speaks -- both
    # candidates must get a burst at the SAME timestamps for that window.
    words = [{"start_ms": i * 100, "end_ms": i * 100 + 90} for i in range(30)]
    lattice = _lat(words)
    turns_by_file = {"f1": [(0, 2000, "S0")]}
    voice_of = {("f1", "S0"): "V0"}
    cuts = [_cut("speech_cut[0]", "f1", (0, 19),
                [{"position": "left"}, {"position": "right"}], subject_box=(0.1, 0.1, 0.6, 0.6))]
    visible_persons = {("f1", "speech_cut[0]"): ["P0", "P1"]}
    audio_by_file = {"f1": {"rms_db": [-20.0] * 100, "hop_ms": 50}}
    motion_by_file = {"f1": {"blur": [0.3] * 100, "hop_ms": 50}}

    bursts, off_camera = sf.plan_bursts(
        turns_by_file, voice_of, {"f1": cuts}, {"f1": lattice},
        visible_persons, audio_by_file, motion_by_file,
    )
    assert off_camera == set(), off_camera
    cands = {b.candidate_person for b in bursts}
    assert cands == {"P0", "P1"}, cands
    p0 = next(b for b in bursts if b.candidate_person == "P0")
    p1 = next(b for b in bursts if b.candidate_person == "P1")
    assert p0.ts_ms == p1.ts_ms, (p0.ts_ms, p1.ts_ms)
    print("ok  test_plan_bursts_multi_candidate_window_frames_every_candidate")


def main():
    test_merge_intervals_joins_overlapping_and_touching()
    test_largest_clean_subspan_picks_the_bigger_piece()
    test_largest_clean_subspan_none_when_fully_covered()
    test_clean_windows_carves_around_a_nearby_other_voice_turn()
    test_clean_windows_drops_a_turn_with_no_span_above_min_win_ms()
    test_loudness_peak_ms_finds_the_argmax_hop()
    test_loudness_peak_ms_falls_back_to_midpoint_with_no_signal()
    test_burst_offsets_scale_and_center_by_window_length()
    test_burst_ts_stays_inside_the_window_and_snaps_to_sharpness()
    test_window_score_rewards_loud_sharp_isolated_prominent()
    test_covering_candidates_finds_the_visible_person_in_the_covering_cut()
    test_covering_candidates_empty_when_no_cut_covers_the_window()
    test_plan_bursts_end_to_end_two_speakers_one_narrator()
    test_plan_bursts_multi_candidate_window_frames_every_candidate()
    print("\nall identity-speaker-frames tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
