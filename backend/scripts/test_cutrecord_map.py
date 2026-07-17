"""
Pure unit tests for `app.services.l3.cutrecord_map` -- the cuts-v3
`cut_records` -> clip-tree projection (no DB; the SQL-touching resolver
functions `latest_run_for_files`/`rows_for_run`/`cut_dicts_for_files`/
`signatures_for` are exercised against real Postgres, not here -- see
cuts_v3_to_brain.plan.md "Testing / verification").

Run:  .venv/bin/python scripts/test_cutrecord_map.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import cutrecord_map as cm  # noqa: E402
from app.services.l3 import footage_map as fm  # noqa: E402


def _row(**over):
    row = {
        "id": "cut-1", "file_id": "ffffffff-1111", "kind": "video",
        "src_in_ms": 1000, "src_out_ms": 5000, "label": "product reveal",
        "summary": None, "voice_ids": None, "speaker_person": None,
        "visible_persons": None, "on_camera": None,
        "take_group_id": None, "take_role": None, "channel": "shown",
        "junk": False, "framing": None, "look": None, "hero_ts_ms": 4200,
        "pace": {"min_ms": 800, "natural_ms": 4000, "max_ms": 4000, "natural_sound": True},
    }
    row.update(over)
    return row


def test_to_cut_dict_maps_exact_keys_build_clip_tree_reads():
    """The cut dict carries every key `build_clip_tree` (footage_map.py) reads
    off a cut, with the right derived values."""
    row = _row()
    d = cm._to_cut_dict(row)
    for key in ("hero_id", "file_id", "channel", "subject", "label", "summary",
                "voice_ids", "speaker_person", "visible_persons", "on_camera",
                "src_in_ms", "src_out_ms", "play_ms", "keep_spans",
                "score", "flags", "audio", "mute", "people", "framing", "quality",
                "ladder", "take_group_id", "take_role", "junk", "junk_reason", "continuity",
                "screen_text", "salience", "audio_file_id", "audio_offset_ms",
                "audio_align_confidence"):
        assert key in d, f"missing {key}"
    assert d["hero_id"] == "cut-1"
    assert d["channel"] == "shown"
    assert d["subject"] == "object"
    assert d["label"] == "product reveal"
    assert d["src_in_ms"] == 1000 and d["src_out_ms"] == 5000
    assert 0.0 <= d["score"] <= 1.0
    assert len(d["ladder"]) == 5
    assert [r["level"] for r in d["ladder"]] == ["broad", "calm", "balanced", "tight", "sharp"]
    print("ok  test_to_cut_dict_maps_exact_keys_build_clip_tree_reads")


def test_to_cut_dict_surfaces_screen_text_salience_and_av_coupling():
    row = _row(screen_text="Chapter 3", salience={"peak_ms": 4300, "score": 0.8},
               audio_file_id="ffffffff-2222", audio_offset_ms=150, audio_align_confidence=0.92)
    d = cm._to_cut_dict(row)
    assert d["screen_text"] == "Chapter 3", d["screen_text"]
    assert d["salience"] == {"peak_ms": 4300, "score": 0.8}, d["salience"]
    assert d["audio_file_id"] == "ffffffff-2222", d["audio_file_id"]
    assert d["audio_offset_ms"] == 150, d["audio_offset_ms"]
    assert d["audio_align_confidence"] == 0.92, d["audio_align_confidence"]
    print("ok  test_to_cut_dict_surfaces_screen_text_salience_and_av_coupling")


def test_to_cut_dict_legacy_row_couples_to_its_own_file():
    # A pre-migration row (audio_file_id column NULL) -- same-source coupling,
    # never a null/missing audio source downstream.
    row = _row(audio_file_id=None, audio_offset_ms=None, audio_align_confidence=None)
    d = cm._to_cut_dict(row)
    assert d["audio_file_id"] == row["file_id"], d["audio_file_id"]
    assert d["audio_offset_ms"] == 0, d["audio_offset_ms"]
    assert d["audio_align_confidence"] is None, d["audio_align_confidence"]
    assert d["screen_text"] == "" and d["salience"] == {}
    print("ok  test_to_cut_dict_legacy_row_couples_to_its_own_file")


def test_to_cut_dict_feeds_build_clip_tree_end_to_end():
    """A cut dict from a real-shaped row builds a valid moment with all 5
    variants -- the actual integration point Phase 1/2 rely on."""
    cuts = [cm._to_cut_dict(_row()),
            cm._to_cut_dict(_row(id="cut-2", kind="speech", channel="said",
                                 src_in_ms=5000, src_out_ms=8000,
                                 voice_ids=["V0"], speaker_person="P0",
                                 label="", summary=None,
                                 pace={"min_ms": 3000, "natural_ms": 3000, "max_ms": 3000,
                                       "natural_sound": True, "remove_spans": [[5000, 5200]]}))]
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, cuts)
    assert tree["moment_count"] == 2, tree["moment_count"]
    m0, m1 = tree["moments"]
    assert set(m0["variants"].keys()) == {"broad", "calm", "balanced", "tight", "sharp"}
    assert m1["channel"] == "said" and m1["subject"] == "person"
    print("ok  test_to_cut_dict_feeds_build_clip_tree_end_to_end")


def test_junk_and_continuity_ride_through_unfiltered():
    """cuts_v3_continuity.plan.md: junk is KEPT (labeled), not dropped, and its
    persisted continuity block rides straight through onto the cut dict."""
    cont = {"clip": "ffffffff-1111", "cut_no": 2, "of": 5,
            "prev_contiguous": True, "next_contiguous": False,
            "seam_reason_prev": "continuous take",
            "seam_reason_next": "a flagged production break (cue/reset/dead air) in the gap"}
    row = _row(junk=True, junk_reason="camera cue", continuity=cont)
    d = cm._to_cut_dict(row)
    assert d["junk"] is True
    assert d["junk_reason"] == "camera cue"
    assert d["continuity"] == cont
    # A non-junk row with no continuity yet (pre-migration backfill '{}')
    # degrades to an empty dict, never a crash.
    plain = cm._to_cut_dict(_row(junk=False, continuity=None))
    assert plain["junk"] is False and plain["continuity"] == {}
    print("ok  test_junk_and_continuity_ride_through_unfiltered")


def test_subject_derivation():
    assert cm._SUBJECT_BY_CHANNEL["said"] == "person"
    assert cm._SUBJECT_BY_CHANNEL["done"] == "person"
    assert cm._SUBJECT_BY_CHANNEL["shown"] == "object"
    print("ok  test_subject_derivation")


def test_audio_mute_rule():
    """Said cuts are never touched (audio IS the point); a video cut whose
    pace envelope says its sound isn't worth keeping is muted by default."""
    audio, mute, flags = cm._audio_mute_for("said", _row(pace={"natural_sound": False}))
    assert audio is None and mute is False and flags == []

    audio, mute, flags = cm._audio_mute_for("shown", _row(pace={"natural_sound": True}))
    assert audio == "sound" and mute is False and flags == []

    audio, mute, flags = cm._audio_mute_for("done", _row(pace={"natural_sound": False}))
    assert audio == "silent" and mute is True and flags == ["muted"]
    print("ok  test_audio_mute_rule")


def test_people_from_speaker():
    assert cm._people_for(_row(speaker_person=None)) == []
    people = cm._people_for(_row(speaker_person="P0", on_camera=True))
    assert people == [{"person_id": "P0", "voice_speaker_id": "P0", "on_camera": True,
                        "characteristics": []}]
    print("ok  test_people_from_speaker")


def test_score_prefers_longer_better_anchored_cuts():
    long_centered = cm._score_for(_row(src_in_ms=0, src_out_ms=10000, hero_ts_ms=5000))
    short_edge = cm._score_for(_row(src_in_ms=0, src_out_ms=500, hero_ts_ms=0))
    assert 0.0 <= short_edge < long_centered <= 1.0
    print("ok  test_score_prefers_longer_better_anchored_cuts")


def test_ladder_never_trims_past_the_anchor_or_the_source_span():
    """Fork A invariant: every video rung's window still contains hero_ts_ms
    (the anchor is never trimmed away) and never exceeds [src_in_ms, src_out_ms]."""
    row = _row(src_in_ms=0, src_out_ms=10000, hero_ts_ms=8500,
              pace={"min_ms": 1500, "natural_ms": 10000, "max_ms": 10000, "natural_sound": True})
    score = cm._score_for(row)
    ladder = cm.synth_ladder(row, score)
    plays = [r["play_ms"] for r in ladder]
    assert plays == sorted(plays, reverse=True), "should tighten monotonically broad->sharp"
    for r in ladder:
        assert 0 <= r["in_ms"] <= 8500 <= r["out_ms"] <= 10000, r
    print("ok  test_ladder_never_trims_past_the_anchor_or_the_source_span")


def test_ladder_clamps_to_min_ms_floor():
    """The tightest (sharp) rung never insets narrower than pace.min_ms."""
    row = _row(src_in_ms=0, src_out_ms=10000, hero_ts_ms=5000,
              pace={"min_ms": 2000, "natural_ms": 10000, "max_ms": 10000, "natural_sound": True})
    ladder = cm.synth_ladder(row, cm._score_for(row))
    assert ladder[-1]["play_ms"] >= 2000 - 1, ladder[-1]
    print("ok  test_ladder_clamps_to_min_ms_floor")


def test_speech_ladder_stays_full_span_with_no_removable_budget():
    """A clean spoken beat (no pace.remove_spans) plays whole at every level --
    speech never gets anchor-protected negative padding."""
    row = _row(kind="speech", channel="said", src_in_ms=0, src_out_ms=4000,
              pace={"min_ms": 4000, "natural_ms": 4000, "max_ms": 4000, "natural_sound": True})
    ladder = cm.synth_ladder(row, cm._score_for(row))
    for r in ladder:
        assert r["in_ms"] == 0 and r["out_ms"] == 4000 and r["play_ms"] == 4000, r
    print("ok  test_speech_ladder_stays_full_span_with_no_removable_budget")


def test_speech_ladder_threads_remove_spans_into_keep_spans():
    """Interior dead-air/fillers are progressively shaved via a multi-span
    keep-list as the rung's energy rises; the outer span never shrinks past
    what the kept content itself defines, and every span stays in [0, 6000]."""
    row = _row(kind="speech", channel="said", src_in_ms=0, src_out_ms=6000,
              pace={"min_ms": 6000, "natural_ms": 6000, "max_ms": 6000, "natural_sound": True,
                    "remove_spans": [[0, 400], [5800, 6000], [2500, 2900]]})
    ladder = cm.synth_ladder(row, cm._score_for(row))
    plays = [r["play_ms"] for r in ladder]
    assert plays == sorted(plays, reverse=True), "more gets shaved as energy rises"
    assert plays[0] < 6000, "even the lowest sampled energy (0.1) removes the biggest span"
    for r in ladder:
        assert r["in_ms"] >= 0 and r["out_ms"] <= 6000
        for sp in r["spans"]:
            assert 0 <= sp["in_ms"] < sp["out_ms"] <= 6000
    print("ok  test_speech_ladder_threads_remove_spans_into_keep_spans")


# --------------------------------------------------------------------------
# V4 shape-aware ladder (cuts_v4_segmentation.plan.md section 6)
# --------------------------------------------------------------------------

def test_v3_row_without_salience_kind_keeps_hero_centered_symmetric_shrink():
    """A row with no salience.kind at all (V3, or a pre-migration row) must
    ladder EXACTLY as before -- hero_ts_ms-centered, ignoring any stray
    salience/shape data that might otherwise be present."""
    row = _row(src_in_ms=0, src_out_ms=10000, hero_ts_ms=8500,
              salience={"peak_ms": 1000, "score": 0.9},  # present but no "kind" -> ignored
              pace={"min_ms": 1500, "natural_ms": 10000, "max_ms": 10000, "natural_sound": True})
    rung = cm._video_rung(row, energy=1.0, level="sharp", score=0.5)
    assert rung["in_ms"] <= 8500 <= rung["out_ms"], rung
    assert not (1000 - 100 <= (rung["in_ms"] + rung["out_ms"]) / 2 <= 1000 + 100), \
        "must center on hero_ts_ms, not the unrelated salience.peak_ms"
    print("ok  test_v3_row_without_salience_kind_keeps_hero_centered_symmetric_shrink")


def test_v4_both_shape_is_symmetric_around_salience_peak_not_hero():
    row = _row(src_in_ms=0, src_out_ms=10000, hero_ts_ms=2000,
              salience={"peak_ms": 7000, "score": 0.8, "kind": "point", "shape": "both", "span_ms": None},
              pace={"min_ms": 1000, "natural_ms": 10000, "max_ms": 10000, "natural_sound": True})
    rung = cm._video_rung(row, energy=1.0, level="sharp", score=0.5)
    center = (rung["in_ms"] + rung["out_ms"]) / 2
    assert abs(center - 7000) < 50, rung
    print("ok  test_v4_both_shape_is_symmetric_around_salience_peak_not_hero")


def test_v4_before_shape_out_edge_barely_moves_in_edge_absorbs_shrink():
    """shape="before": the impact is near the natural OUT edge -- as energy
    rises the window should end close to where it always did (near the
    peak/impact) while the IN edge does almost all the shrinking."""
    row = _row(src_in_ms=0, src_out_ms=10000, hero_ts_ms=9999,
              salience={"peak_ms": 9600, "score": 0.9, "kind": "point", "shape": "before", "span_ms": None},
              pace={"min_ms": 800, "natural_ms": 10000, "max_ms": 10000, "natural_sound": True})
    low = cm._video_rung(row, energy=0.1, level="broad", score=0.5)
    high = cm._video_rung(row, energy=1.0, level="sharp", score=0.5)
    # OUT barely moves (stays within a follow-through floor of the natural out);
    # IN moves a lot (absorbs almost the entire shrink).
    assert abs(low["out_ms"] - high["out_ms"]) < 500, (low, high)
    assert high["in_ms"] - low["in_ms"] > 5000, (low, high)
    # Never clips the impact: out must always land at/after the peak.
    assert high["out_ms"] >= 9600, high
    print("ok  test_v4_before_shape_out_edge_barely_moves_in_edge_absorbs_shrink")


def test_v4_after_shape_in_edge_barely_moves_out_edge_absorbs_shrink():
    """shape="after": the reveal is near the natural IN edge -- mirror of
    shape="before"."""
    row = _row(src_in_ms=0, src_out_ms=10000, hero_ts_ms=1,
              salience={"peak_ms": 400, "score": 0.9, "kind": "point", "shape": "after", "span_ms": None},
              pace={"min_ms": 800, "natural_ms": 10000, "max_ms": 10000, "natural_sound": True})
    low = cm._video_rung(row, energy=0.1, level="broad", score=0.5)
    high = cm._video_rung(row, energy=1.0, level="sharp", score=0.5)
    assert abs(low["in_ms"] - high["in_ms"]) < 500, (low, high)
    assert low["out_ms"] - high["out_ms"] > 5000, (low, high)
    # Never clips the lead-in: in must always land at/before the peak.
    assert high["in_ms"] <= 400, high
    print("ok  test_v4_after_shape_in_edge_barely_moves_out_edge_absorbs_shrink")


def test_v4_span_kind_trims_head_keeps_settle():
    """salience.kind="span" (camera move): OUT stays pinned at the cut's
    natural out (the move's settle) at every energy; IN moves toward the
    move's own dynamic core start but never past it."""
    # min_ms == exactly the core's own width (10000-3000): at max energy the
    # target-length window should reach precisely span_ms[0], never past it.
    row = _row(src_in_ms=0, src_out_ms=10000, hero_ts_ms=5000,
              salience={"peak_ms": 5000, "score": 0.7, "kind": "span", "shape": "center",
                       "span_ms": [3000, 10000]},
              pace={"min_ms": 7000, "natural_ms": 10000, "max_ms": 10000, "natural_sound": True})
    for energy in (0.0, 0.25, 0.5, 0.75, 1.0):
        rung = cm._video_rung(row, energy=energy, level="x", score=0.5)
        assert rung["out_ms"] == 10000, rung
    for energy in (0.25, 0.5, 0.75, 1.0):   # energy 0 keeps the full natural span (no trim yet)
        rung = cm._video_rung(row, energy=energy, level="x", score=0.5)
        assert rung["in_ms"] >= 3000, rung   # never trims into the move's own core
    tight = cm._video_rung(row, energy=1.0, level="sharp", score=0.5)
    assert tight["in_ms"] == 3000, "at max energy the head trims exactly to the core start"

    # A tighter floor than the core's width (min_ms < 7000) must still clamp
    # at span_ms[0] rather than trim past it into the ramp-in's own core.
    tiny_floor = _row(src_in_ms=0, src_out_ms=10000, hero_ts_ms=5000,
                      salience={"peak_ms": 5000, "score": 0.7, "kind": "span", "shape": "center",
                               "span_ms": [3000, 10000]},
                      pace={"min_ms": 500, "natural_ms": 10000, "max_ms": 10000, "natural_sound": True})
    clamped = cm._video_rung(tiny_floor, energy=1.0, level="sharp", score=0.5)
    assert clamped["in_ms"] == 9500 and clamped["out_ms"] == 10000, clamped
    print("ok  test_v4_span_kind_trims_head_keeps_settle")


def test_v4_punchy_rung_never_clips_the_peak():
    """At max energy, a point cut's window must still contain its own
    salience peak -- the whole point of the follow-through/lead floors."""
    for shape in ("before", "after", "both", "center"):
        row = _row(src_in_ms=0, src_out_ms=10000, hero_ts_ms=5000,
                  salience={"peak_ms": 5000, "score": 0.9, "kind": "point", "shape": shape,
                           "span_ms": None},
                  pace={"min_ms": 400, "natural_ms": 10000, "max_ms": 10000, "natural_sound": True})
        rung = cm._video_rung(row, energy=1.0, level="sharp", score=0.5)
        assert rung["in_ms"] <= 5000 <= rung["out_ms"], (shape, rung)
    print("ok  test_v4_punchy_rung_never_clips_the_peak")


def test_pinned_run_id_is_honored_without_live_resolve():
    """When a thread pins a run (migration 028), the projection reads THAT run
    and never falls back to `latest_run_for_files` -- so a re-ingest mid-thread
    can't swap the beat universe under an active edit."""
    from app.services.l3 import cuts_v3_read
    calls = {"latest": 0, "rows_run": None}

    def fake_latest(_fids):
        calls["latest"] += 1
        return "LATEST-RUN"

    def fake_rows(run_id, file_ids=None):
        calls["rows_run"] = run_id
        return [_row(id="c", file_id="ffffffff-1111")]

    orig_latest, orig_rows = cuts_v3_read.latest_run_for_files, cuts_v3_read.rows_for_run
    cuts_v3_read.latest_run_for_files, cuts_v3_read.rows_for_run = fake_latest, fake_rows
    try:
        cm.cut_dicts_for_files(["ffffffff-1111"], run_id="PINNED-RUN")
        assert calls["rows_run"] == "PINNED-RUN", calls
        assert calls["latest"] == 0, "pinned run must NOT trigger a live latest-run resolve"
        cm.signatures_for(["ffffffff-1111"], run_id="PINNED-RUN")
        assert calls["rows_run"] == "PINNED-RUN" and calls["latest"] == 0, calls
    finally:
        cuts_v3_read.latest_run_for_files, cuts_v3_read.rows_for_run = orig_latest, orig_rows
    print("ok  test_pinned_run_id_is_honored_without_live_resolve")


def test_unpinned_falls_back_to_latest_run():
    """No pin (older threads / pre-028) => resolve the latest covering run live,
    exactly today's behavior."""
    from app.services.l3 import cuts_v3_read
    calls = {"latest": 0, "rows_run": None}

    def fake_latest(_fids):
        calls["latest"] += 1
        return "LATEST-RUN"

    def fake_rows(run_id, file_ids=None):
        calls["rows_run"] = run_id
        return [_row(id="c", file_id="ffffffff-1111")]

    orig_latest, orig_rows = cuts_v3_read.latest_run_for_files, cuts_v3_read.rows_for_run
    cuts_v3_read.latest_run_for_files, cuts_v3_read.rows_for_run = fake_latest, fake_rows
    try:
        cm.cut_dicts_for_files(["ffffffff-1111"])
        assert calls["latest"] == 1 and calls["rows_run"] == "LATEST-RUN", calls
    finally:
        cuts_v3_read.latest_run_for_files, cuts_v3_read.rows_for_run = orig_latest, orig_rows
    print("ok  test_unpinned_falls_back_to_latest_run")


def main():
    test_to_cut_dict_maps_exact_keys_build_clip_tree_reads()
    test_to_cut_dict_surfaces_screen_text_salience_and_av_coupling()
    test_to_cut_dict_legacy_row_couples_to_its_own_file()
    test_to_cut_dict_feeds_build_clip_tree_end_to_end()
    test_junk_and_continuity_ride_through_unfiltered()
    test_subject_derivation()
    test_audio_mute_rule()
    test_people_from_speaker()
    test_score_prefers_longer_better_anchored_cuts()
    test_ladder_never_trims_past_the_anchor_or_the_source_span()
    test_ladder_clamps_to_min_ms_floor()
    test_speech_ladder_stays_full_span_with_no_removable_budget()
    test_speech_ladder_threads_remove_spans_into_keep_spans()
    test_v3_row_without_salience_kind_keeps_hero_centered_symmetric_shrink()
    test_v4_both_shape_is_symmetric_around_salience_peak_not_hero()
    test_v4_before_shape_out_edge_barely_moves_in_edge_absorbs_shrink()
    test_v4_after_shape_in_edge_barely_moves_out_edge_absorbs_shrink()
    test_v4_span_kind_trims_head_keeps_settle()
    test_v4_punchy_rung_never_clips_the_peak()
    test_pinned_run_id_is_honored_without_live_resolve()
    test_unpinned_falls_back_to_latest_run()
    print("\nall cutrecord-map tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
