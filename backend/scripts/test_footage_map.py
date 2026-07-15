"""
Tests for the footage moment-tree builder (no DB).

Exercises the owned-ladder breakdown: each cut carries its own zoom ladder, so a
moment reads its VARIANTS straight off the rungs (no cross-band re-matching, no
atoms; a split is a multi-span rung). Then the compact Tier-0 map text and
Tier-1 moment record. Run:  .venv/bin/python scripts/test_footage_map.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import footage_map as fm  # noqa: E402
from app.services.l3.arrange import _MapIndex  # noqa: E402

# This module is explicitly "no DB" (see module docstring): build_clip_tree
# now calls _said_text_for_span -> _sentences_for_file for every "said" cut
# (beat_transcript.plan.md), which would otherwise hit a real Postgres
# connection. Replace it process-wide with an empty-transcript stub -- the
# same observable result as "no dialogue_segments row for this file" -- so
# every existing test keeps its pre-transcript behavior (said_text="" ->
# falls back to the visual gist) with zero DB touches. Tests that exercise
# the new transcript rendering override it locally via mock.patch.object.
fm._sentences_for_file = lambda file_id: ()


def _rung(level, in_ms, out_ms, text="", score=0.5, spans=None):
    sp = spans or [(in_ms, out_ms)]
    return {
        "level": level,
        "spans": [{"in_ms": a, "out_ms": b} for a, b in sp],
        "in_ms": min(a for a, _ in sp), "out_ms": max(b for _, b in sp),
        "play_ms": sum(b - a for a, b in sp), "text": text, "score": score,
    }


def _cut(hero_id, in_ms, out_ms, label="", channel="said", subject="person",
         speaker="S0", score=0.5, play_ms=None, keep_spans=None, ladder=None, **extra):
    d = {
        "hero_id": hero_id, "file_id": "ffffffff-1111", "channel": channel,
        "subject": subject, "label": label, "src_in_ms": in_ms, "src_out_ms": out_ms,
        "play_ms": play_ms if play_ms is not None else (out_ms - in_ms),
        "keep_spans": keep_spans, "score": score, "speaker": speaker,
        "flags": [], "take_count": 1, "ladder": ladder,
    }
    d.update(extra)
    return d


def _thought_cut():
    """One thought cut whose ladder is the five nested zooms: the turn / run-up
    (broad/calm) -> the thought (balanced) -> core sentence (tight) -> punchline
    clause (sharp)."""
    return _cut("f:th", 1000, 4000, "we almost shut the company down", score=0.82,
                ladder=[
                    _rung("broad", 0, 8000, "so anyway we almost shut the company down", 0.8),
                    _rung("calm", 500, 4000, "so we almost shut the company down", 0.8),
                    _rung("balanced", 1000, 4000, "we almost shut the company down", 0.82),
                    _rung("tight", 1000, 3000, "we almost shut down", 0.7),
                    _rung("sharp", 1500, 3000, "shut down", 0.7),
                ])


def test_thought_levels_become_variants():
    """Every nested level (incl. tight=core) is a selectable VARIANT read off the
    cut's ladder; a single thought yields NO atoms."""
    tree = fm.build_clip_tree("ffffffff-1111",
                              {"name": "Take 2", "duration_ms": 8000,
                               "content_type": "interview", "primary_axis": "dialogue"},
                              [_thought_cut()])
    assert tree["moment_count"] == 1, tree["moment_count"]
    m = tree["moments"][0]
    assert m["moment_id"] == "ffffffff:m00", m["moment_id"]
    assert set(m["variants"].keys()) == {"broad", "calm", "balanced", "tight", "sharp"}, \
        m["variants"].keys()
    # The moment anchors on balanced (one complete thought per cut).
    assert (m["in_ms"], m["out_ms"]) == (1000, 4000), (m["in_ms"], m["out_ms"])
    assert m["variants"]["broad"]["out_ms"] == 8000
    assert m["variants"]["tight"]["out_ms"] == 3000
    assert m["variants"]["sharp"]["in_ms"] == 1500
    assert m["atoms"] == [], m["atoms"]
    print("ok  test_thought_levels_become_variants")


def test_split_rung_becomes_keep_spans():
    """A multi-span rung (a jump-cut / breath-excised split) surfaces as the
    variant's keep_spans, not separate atoms."""
    cut = _cut("f:sp", 1000, 4000, "the product really changes everything", score=0.7,
               ladder=[
                   _rung("balanced", 1000, 4000, "the product really changes everything", 0.7),
                   _rung("sharp", 1000, 4000, "really changes everything", 0.8,
                         spans=[(1000, 1800), (2600, 4000)]),
               ])
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    sharp = tree["moments"][0]["variants"]["sharp"]
    assert sharp["keep_spans"] == [[1000, 1800], [2600, 4000]], sharp["keep_spans"]
    assert sharp["in_ms"] == 1000 and sharp["out_ms"] == 4000
    assert tree["moments"][0]["variants"]["balanced"]["keep_spans"] is None
    print("ok  test_split_rung_becomes_keep_spans")


def test_no_ladder_uses_flat_span():
    """A legacy cut with no ladder still yields a moment (balanced variant from
    its flat span)."""
    cut = _cut("f:legacy", 0, 3000, "held wide shot", channel="shown", subject="place",
               speaker=None, ladder=None)
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "B", "duration_ms": 3000}, [cut])
    assert tree["moment_count"] == 1, tree["moment_count"]
    m = tree["moments"][0]
    assert m["channel"] == "shown" and m["subject"] == "place"
    assert m["variants"]["balanced"]["out_ms"] == 3000
    print("ok  test_no_ladder_uses_flat_span")


def test_facets_surface_on_moment():
    """People / framing / quality facets ride along onto the moment for the
    brain to read."""
    cut = _cut("f:th", 1000, 4000, "a clean line", score=0.8,
               ladder=[_rung("balanced", 1000, 4000, "a clean line", 0.8)],
               people=[{"voice_speaker_id": "S0", "person_id": "p1", "role": "host",
                        "on_camera": True}],
               framing={"shot_size": "medium", "region": {"x": 0.3}},
               quality={"delivery": 0.81, "on_camera": 1.0})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    assert m["people"][0]["person_id"] == "p1"
    assert m["framing"]["shot_size"] == "medium"
    assert m["quality"]["delivery"] == 0.81
    print("ok  test_facets_surface_on_moment")


def test_map_text_lists_variants_no_atoms():
    tree = fm.build_clip_tree("ffffffff-1111",
                              {"name": "Take 2", "duration_ms": 8000,
                               "content_type": "interview"}, [_thought_cut()])
    block = fm._clip_block(tree)
    lines = block.splitlines()
    assert lines[0].startswith('CLIP ffffffff "Take 2"'), lines[0]
    assert len(lines) == 1 + tree["moment_count"]
    assert "nrg:broad|calm|balanced|tight|sharp" in lines[1], lines[1]
    assert "atoms" not in lines[1], lines[1]
    print("ok  test_map_text_lists_variants_no_atoms")


def test_moment_line_flags_offcamera():
    """The brain's one-line index keys on CHANNEL.SUBJECT; an off-camera voice
    (an off-screen interviewer / voiceover -- never visually confirmed, so
    face clustering never saw them and the voice stayed unbound) renders an
    honest PIC:? -- never a fabricated face -- and SND:OFF-CAM with no id,
    never a guessed name."""
    cut = _cut("f:mo", 1000, 4000, "what made you start this", channel="said",
               subject="person", score=0.7,
               ladder=[_rung("balanced", 1000, 4000, "what made you start this", 0.7)],
               flags=["offscreen"], voice_ids=["V9"], speaker_person=None, visible_persons=[])
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "said.person" in line, line
    assert "PIC:?" in line, line
    assert "SND:OFF-CAM speaking" in line, line
    print("ok  test_moment_line_flags_offcamera")


def test_moment_line_shows_channel_subject_v2():
    """cuts-v2: the resident line keys on CHANNEL.SUBJECT."""
    cut = _cut("f:v0", 1000, 4000, "kicks the ball", channel="done", subject="person",
               speaker="p1", score=0.6,
               ladder=[_rung("balanced", 1000, 4000, "kicks the ball", 0.6)])
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    assert m["channel"] == "done" and m["subject"] == "person"
    assert fm._capture_tag(m) == "done.person", fm._capture_tag(m)
    print("ok  test_moment_line_shows_channel_subject_v2")


def test_source_contiguous_beats_form_a_run_channel_agnostic():
    """Beats back-to-back in source time form one run REGARDLESS of channel --
    the run is a purely temporal fact (same clip + adjacent time), like the weld.
    A real gap starts a new run; a lone far-away beat gets no tag."""
    cuts = [
        _cut("c:0", 0, 2000, "kick", channel="done", subject="person", speaker=None,
             ladder=[_rung("balanced", 0, 2000, "kick", 0.6)]),
        _cut("c:1", 500, 2500, "go go go", channel="said", subject="person", speaker="S1",
             ladder=[_rung("balanced", 500, 2500, "go go go", 0.6)]),
        _cut("c:2", 2100, 4000, "the scoreboard", channel="shown", subject="graphic",
             speaker=None, ladder=[_rung("balanced", 2100, 4000, "the scoreboard", 0.6)]),
        _cut("c:3", 4200, 6000, "shoot", channel="done", subject="person", speaker=None,
             ladder=[_rung("balanced", 4200, 6000, "shoot", 0.6)]),
        _cut("c:4", 30000, 32000, "later", channel="done", subject="person", speaker=None,
             ladder=[_rung("balanced", 30000, 32000, "later", 0.6)]),
    ]
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "Match", "duration_ms": 40000}, cuts)
    by_id = {m["moment_id"].split(":")[-1]: m for m in tree["moments"]}
    # m00..m03 are adjacent in source time (mixed channels) -> ONE run of 4.
    run0 = by_id["m00"].get("run_id")
    assert run0 is not None
    assert all(by_id[mid]["run_id"] == run0 for mid in ("m00", "m01", "m02", "m03")), by_id
    assert by_id["m00"]["run_len"] == 4, by_id["m00"]["run_len"]
    assert [by_id[mid]["run_pos"] for mid in ("m00", "m01", "m02", "m03")] == [0, 1, 2, 3]
    # The far-away beat (big gap) is its own lone run -> no tag.
    assert by_id["m04"].get("run_id") is None, "far-away beat starts no (multi) run"
    # Channel never entered it: a said and a shown sit inside the run too.
    assert by_id["m01"]["channel"] == "said" and by_id["m02"]["channel"] == "shown"
    assert "· run:" in fm._moment_line(by_id["m01"]), fm._moment_line(by_id["m01"])
    print("ok  test_source_contiguous_beats_form_a_run_channel_agnostic")


def test_reconciled_shows_face_and_cam_override():
    """PIC reports WHOSE FACE is visible in the clip (per-cut-occurrence face
    clustering, voice_first_identity.plan.md Phase D) -- and SND names who's
    heard, tagged OFF-CAM when the speaking voice's bound person isn't the
    one shown here (the listener-camera case: P1 talks, but this clip shows
    P0). PIC leads the line; SND trails it."""
    cut = _cut("48c93cef:m03", 1000, 4000, "and then we shipped it", channel="said",
               subject="person", score=0.6,
               ladder=[_rung("balanced", 1000, 4000, "and then we shipped it", 0.6)],
               voice_ids=["V1"], speaker_person="P1", visible_persons=["P0"], on_camera=False)
    tree = fm.build_clip_tree("48c93cef-aaa", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "PIC:P0" in line, line
    assert "SND:P1 OFF-CAM speaking" in line, line
    assert line.index("PIC:") < line.index("SND:"), line  # picture leads, never the speaker
    print("ok  test_reconciled_shows_face_and_cam_override")


def test_said_beat_on_listener_camera_reads_pic_first_with_alt_pic():
    """The exact podcast-bug shape: a said beat delivered on the LISTENER's
    camera renders PIC (who's shown) before SND (who's heard), so placing it can
    no longer be mistaken for showing the speaker -- and `alt-PIC` points at the
    camera that DOES show the speaker for the same words (Fact #2 folded onto
    the beat, not buried in a distant coverage block)."""
    cut = _cut("48c93cef:m24", 1000, 4000, "that freedom he gave you", channel="said",
               subject="person", score=0.6,
               ladder=[_rung("balanced", 1000, 4000, "that freedom he gave you", 0.6)],
               voice_ids=["V2"], speaker_person="P2", visible_persons=["P1"], on_camera=False,
               framing="med")
    tree = fm.build_clip_tree("48c93cef-aaa", {"name": "Cam A", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    # Folded on by `_annotate_dups` in the real path; set directly here to test
    # the render in isolation (mirrors how `_dups_block` fixtures worked before).
    m["alt_pic"] = [{"moment_id": "1aedb093:m13", "file": "1aedb093-bbb",
                     "visible_persons": ["P2"], "speaker_person": "P2",
                     "framing": "med", "score": 0.73, "restart": False}]
    line = fm._moment_line(m)
    assert "PIC:P1" in line, line
    assert "SND:P2 OFF-CAM speaking" in line, line
    assert line.index("PIC:") < line.index("SND:"), line
    assert "·alt-PIC:P2→1aedb093:m13" in line, line
    print("ok  test_said_beat_on_listener_camera_reads_pic_first_with_alt_pic")


def test_said_beat_with_transcript_quotes_verbatim_text_first():
    """beat_transcript.plan.md: a speech beat's PRIMARY quote is now the
    verbatim dialogue_segments text, not the vision-model gist -- but the
    gist still rides along as a short vis:"..." note, never dropped."""
    cut = _cut("f:tr0", 1000, 4000, "we shipped it fast", channel="said", score=0.6,
               ladder=[_rung("balanced", 1000, 4000, "we shipped it fast", 0.6)])
    sentences = ({"speaker": "S0", "text": "we actually shipped it in three days flat",
                 "src_in_ms": 1000, "src_out_ms": 4000},)
    with mock.patch.object(fm, "_sentences_for_file", return_value=sentences):
        tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    assert m["said_text"] == "we actually shipped it in three days flat", m["said_text"]
    line = fm._moment_line(m)
    assert '"we actually shipped it in three days flat"' in line, line
    assert 'vis:"we shipped it fast"' in line, line
    assert line.index('"we actually shipped') < line.index("vis:"), line
    print("ok  test_said_beat_with_transcript_quotes_verbatim_text_first")


def test_said_beat_shows_aud_tag_from_speech_quality():
    """speech_quality (delivery, camera-independent) surfaces as its own
    aud: tag alongside PIC's own q.XX (visual) score."""
    cut = _cut("f:tr1", 1000, 4000, "solid take", channel="said", score=0.6,
               ladder=[_rung("balanced", 1000, 4000, "solid take", 0.6)], speech_quality=0.73)
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "aud:0.73" in line, line
    print("ok  test_said_beat_shows_aud_tag_from_speech_quality")


def test_aud_tag_absent_without_speech_quality():
    cut = _cut("f:tr1b", 1000, 4000, "solid take", channel="said", score=0.6,
               ladder=[_rung("balanced", 1000, 4000, "solid take", 0.6)])
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "aud:" not in line, line
    print("ok  test_aud_tag_absent_without_speech_quality")


def test_action_beat_never_gets_said_text():
    """A done/shown beat's visual label stays primary, never overwritten by
    transcript text that happens to overlap it in TIME -- said_text is only
    ever computed for channel == 'said' cuts."""
    cut = _cut("f:tr2", 1000, 3000, "nods thoughtfully", channel="done", subject="person",
               score=0.6, ladder=[_rung("balanced", 1000, 3000, "nods thoughtfully", 0.6)])
    sentences = ({"speaker": "S0", "text": "narration that happens to overlap in time",
                 "src_in_ms": 1000, "src_out_ms": 3000},)
    with mock.patch.object(fm, "_sentences_for_file", return_value=sentences):
        tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    assert m["said_text"] == "", m["said_text"]
    line = fm._moment_line(m)
    assert '"nods thoughtfully"' in line, line
    assert "narration that happens to overlap" not in line, line
    print("ok  test_action_beat_never_gets_said_text")


def test_said_beat_transcript_truncates_in_compact_mode():
    """compact (paged) mode truncates the quote like today's gist; resident
    mode shows it in full. The moment itself always stores the FULL text
    (still reachable via moment_detail/_span_detail regardless of mode)."""
    long_text = ("we shipped it in three days flat and honestly nobody thought "
                "we could pull it off but the whole team just locked in")
    assert len(long_text) > 80
    cut = _cut("f:tr3", 1000, 4000, "short gist", channel="said", score=0.6,
               ladder=[_rung("balanced", 1000, 4000, "short gist", 0.6)])
    sentences = ({"speaker": "S0", "text": long_text, "src_in_ms": 1000, "src_out_ms": 4000},)
    with mock.patch.object(fm, "_sentences_for_file", return_value=sentences):
        tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    assert m["said_text"] == long_text, m["said_text"]
    compact_line = fm._moment_line(m, compact=True)
    assert "..." in compact_line, compact_line
    assert long_text not in compact_line, compact_line
    resident_line = fm._moment_line(m, compact=False)
    assert long_text in resident_line, resident_line
    print("ok  test_said_beat_transcript_truncates_in_compact_mode")


def test_said_beat_with_no_transcript_falls_back_to_visual_gist():
    """No dialogue_segments row for this file (older footage) -> graceful
    fallback to the visual gist as the primary quote, exactly like before
    this plan; no crash, no vis: tag (nothing to fold in separately)."""
    cut = _cut("f:tr4", 1000, 4000, "we shipped it", channel="said", score=0.6,
               ladder=[_rung("balanced", 1000, 4000, "we shipped it", 0.6)])
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    assert m["said_text"] == "", m["said_text"]
    line = fm._moment_line(m)
    assert '"we shipped it"' in line, line
    assert "vis:" not in line, line
    print("ok  test_said_beat_with_no_transcript_falls_back_to_visual_gist")


def test_span_detail_pads_and_filters_via_shared_sentences_cache():
    """_span_detail's refactor (beat_transcript.plan.md): it now reads
    through the shared _sentences_for_file cache instead of its own inline
    DB read, but keeps its OWN padded-window overlap filter (unlike
    build_clip_tree's exact-span said_text) -- no behavior change there."""
    sentences = (
        {"speaker": "S0", "text": "just inside the pad", "src_in_ms": 200, "src_out_ms": 900},
        {"speaker": "S0", "text": "far outside the span", "src_in_ms": 50000, "src_out_ms": 51000},
    )
    with mock.patch.object(fm, "_sentences_for_file", return_value=sentences):
        detail = fm._span_detail("ffffffff-1111", 1000, 4000)
    texts = [t["text"] for t in detail["transcript"]]
    assert texts == ["just inside the pad"], texts
    print("ok  test_span_detail_pads_and_filters_via_shared_sentences_cache")


def test_alt_pic_absent_without_co_occurrence():
    """A one-camera beat (no take-group link) carries no `alt-PIC` -- the fact
    simply isn't there to state."""
    cut = _cut("f:solo", 1000, 4000, "just one camera on this", channel="said",
               speaker="S0", score=0.5,
               ladder=[_rung("balanced", 1000, 4000, "just one camera on this", 0.5)])
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "alt-PIC" not in line, line
    print("ok  test_alt_pic_absent_without_co_occurrence")


def test_done_beat_pic_first_parity_and_snd_silence():
    """`done`/`shown` beats render the SAME PIC/SND shape as `said` -- equal
    citizens, not visually subordinate -- and a silent reaction reads SND:silence."""
    cut = _cut("f:d0", 1000, 3000, "nods, reacts", channel="done", subject="person",
               score=0.9, audio="silent", mute=False,
               ladder=[_rung("balanced", 1000, 3000, "nods, reacts", 0.9)],
               visible_persons=["G1"], framing="med")
    tree = fm.build_clip_tree("1e529bed-aaa", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "PIC:G1" in line, line
    assert "SND:silence" in line, line
    assert line.index("PIC:") < line.index("SND:"), line
    print("ok  test_done_beat_pic_first_parity_and_snd_silence")


def test_speaker_person_renders_on_cam_when_shown():
    """A speaking voice bound to a person who IS visible in this cut renders
    SND:<person> ON-CAM -- the straightforward, common case (voice_first_
    identity.plan.md Phase F/G: voice->person binding + visible_persons both
    resolve to the same id here)."""
    cut = _cut("48c93cef:m00", 1000, 4000, "we shipped it",
               score=0.7, ladder=[_rung("balanced", 1000, 4000, "we shipped it", 0.7)],
               voice_ids=["V0"], speaker_person="P1", visible_persons=["P1"], on_camera=True)
    tree = fm.build_clip_tree("48c93cef-aaa", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "SND:P1 ON-CAM speaking" in line, line
    assert "PIC:P1" in line, line
    print("ok  test_speaker_person_renders_on_cam_when_shown")


def test_annotate_dups_reads_take_group_id():
    """`_annotate_dups` reads dup_groups DIRECTLY off each moment's persisted
    `take_group_id`/`take_role` (see cuts_v3_to_brain.plan.md Phase 3) and
    folds Fact #2 onto each linked beat as `alt_pic` -- every OTHER member's
    raw facts, in place, no separate lookup table -- so `_moment_line` can
    render a beat's alternates on its own."""
    cut_a = _cut("48c93cef:mA", 1000, 4000, "that freedom he gave you", channel="said",
                speaker="S2", score=0.6,
                ladder=[_rung("balanced", 1000, 4000, "that freedom he gave you", 0.6)],
                take_group_id="tg1", take_role="winner")
    cut_b = _cut("1aedb093:mB", 1000, 4000, "that freedom he gave you", channel="said",
                speaker="S9", score=0.73,
                ladder=[_rung("balanced", 1000, 4000, "that freedom he gave you", 0.73)],
                take_group_id="tg1", take_role="take")
    tree_a = fm.build_clip_tree("48c93cef-aaa", {"name": "A", "duration_ms": 8000}, [cut_a])
    tree_b = fm.build_clip_tree("1aedb093-bbb", {"name": "B", "duration_ms": 8000}, [cut_b])

    summary = fm._annotate_dups([tree_a, tree_b])

    assert len(summary) == 1, summary
    g = summary[0]
    assert g["group_id"] == "tg1", g
    ma, mb = tree_a["moments"][0], tree_b["moments"][0]
    assert ma["dup_group"] == "tg1" and mb["dup_group"] == "tg1"
    assert ma["alt_pic"][0]["moment_id"] == mb["moment_id"], ma["alt_pic"]
    assert mb["alt_pic"][0]["moment_id"] == ma["moment_id"], mb["alt_pic"]
    assert {mf["take_role"] for mf in g["member_facts"]} == {"winner", "take"}, g["member_facts"]
    print("ok  test_annotate_dups_reads_take_group_id")


def test_junk_moment_renders_terse_line_not_the_rich_one():
    """A junk moment (kept in the map, labeled -- cuts_v3_continuity.plan.md)
    renders a terse one-liner: id, reason, continuity, span -- never the full
    PIC/SND/gist line, so keeping it visible doesn't bloat the index."""
    cont = {"clip": "ffffffff-1111", "cut_no": 2, "of": 3,
            "prev_contiguous": True, "next_contiguous": False,
            "seam_reason_prev": "continuous take",
            "seam_reason_next": "shot/scene boundary or transition inside the gap"}
    cut = _cut("f:junk", 1000, 1600, "and go", channel="shown", subject="object",
               speaker=None, score=0.1, ladder=[_rung("balanced", 1000, 1600, "and go", 0.1)],
               junk=True, junk_reason="camera cue", continuity=cont)
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    assert m["junk"] is True and m["junk_reason"] == "camera cue"
    line = fm._moment_line(m)
    assert line.strip().startswith("m00 [JUNK: camera cue]"), line
    assert "↔cut:2/3⋯" in line, line
    assert "PIC:" not in line and "SND:" not in line and "nrg:" not in line, line
    print("ok  test_junk_moment_renders_terse_line_not_the_rich_one")


def test_junk_moment_still_resolves_in_map_index():
    """Junk stays independently PLACEABLE -- its ref must still resolve through
    _MapIndex (skip-by-default in the prompt framing, never un-referenceable)."""
    cut = _cut("f:junk", 1000, 1600, "and go", channel="shown", speaker=None,
               ladder=[_rung("balanced", 1000, 1600, "and go", 0.1)], junk=True,
               junk_reason="camera cue")
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    idx = _MapIndex({"clips": [tree]})
    mid = tree["moments"][0]["moment_id"]
    assert idx.has(mid)
    from app.services.l3.arrange import Placement
    resolved = idx.resolve(Placement(ref=mid))
    assert resolved is not None and resolved.src_in_ms == 1000 and resolved.src_out_ms == 1600
    print("ok  test_junk_moment_still_resolves_in_map_index")


def test_non_junk_moment_shows_continuity_position_and_weld_marks():
    cont = {"clip": "ffffffff-1111", "cut_no": 4, "of": 9,
            "prev_contiguous": True, "next_contiguous": True,
            "seam_reason_prev": "continuous take", "seam_reason_next": "continuous take"}
    cut = _cut("f:c4", 1000, 4000, "the line", score=0.6,
               ladder=[_rung("balanced", 1000, 4000, "the line", 0.6)], continuity=cont)
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "↔cut:4/9↔" in line, line
    print("ok  test_non_junk_moment_shows_continuity_position_and_weld_marks")


def test_first_cut_has_no_prev_weld_mark():
    """No neighbor on that side (seam_reason unset) -> no fabricated mark."""
    cont = {"clip": "ffffffff-1111", "cut_no": 1, "of": 2,
            "prev_contiguous": False, "next_contiguous": False,
            "seam_reason_prev": None, "seam_reason_next": "speaker change across the seam"}
    cut = _cut("f:c1", 0, 1000, "first", score=0.6,
               ladder=[_rung("balanced", 0, 1000, "first", 0.6)], continuity=cont)
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "· cut:1/2⋯" in line, line   # no ↔/⋯ BEFORE cut: (no prev neighbor)
    print("ok  test_first_cut_has_no_prev_weld_mark")


def test_cut_with_no_continuity_block_has_no_continuity_tag():
    """A cut with no continuity block (e.g. a pre-migration run) renders
    exactly as before -- no 'cut:N/of' tag fabricated from nothing."""
    cut = _cut("f:legacy", 0, 3000, "held wide shot")
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 3000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "cut:" not in line, line
    print("ok  test_cut_with_no_continuity_block_has_no_continuity_tag")


def test_take_group_excludes_junk_member():
    """A junk cut is kept OUT of take-group linking even if it carries a
    take_group_id (defensive -- cuts_v3_continuity.plan.md keeps junk out of
    the recommended/take-group set)."""
    cut_a = _cut("a:mA", 1000, 4000, "the same line", channel="said", speaker="S0",
                score=0.6, ladder=[_rung("balanced", 1000, 4000, "the same line", 0.6)],
                take_group_id="tg1", take_role="winner")
    cut_b = _cut("b:mB", 1000, 4000, "the same line", channel="said", speaker="S1",
                score=0.5, ladder=[_rung("balanced", 1000, 4000, "the same line", 0.5)],
                take_group_id="tg1", take_role="take", junk=True, junk_reason="false start")
    tree_a = fm.build_clip_tree("aaaaaaaa-aaaa", {"name": "A", "duration_ms": 8000}, [cut_a])
    tree_b = fm.build_clip_tree("bbbbbbbb-bbbb", {"name": "B", "duration_ms": 8000}, [cut_b])
    summary = fm._annotate_dups([tree_a, tree_b])
    assert summary == [], summary   # only one non-junk member -- not a real choice
    print("ok  test_take_group_excludes_junk_member")


# --------------------------------------------------------------------------
# peak tag (interactive_ask_and_salience.plan.md WS2): `peak:+X.Xs`, the
# offset of post._salience's peak_ms into the cut -- rendered only when
# interior and backed by real signal.
# --------------------------------------------------------------------------

def _salience_cut(hero_id, in_ms, out_ms, salience):
    return _cut(hero_id, in_ms, out_ms, "watch this", salience=salience)


def test_peak_tag_renders_for_an_interior_peak():
    cut = _salience_cut("f:pk", 1000, 4000, {"peak_ms": 2500, "score": 0.9})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "peak:+1.5s" in line, line
    print("ok  test_peak_tag_renders_for_an_interior_peak")


def test_peak_tag_absent_when_salience_missing():
    cut = _salience_cut("f:pk", 1000, 4000, {})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "peak:" not in line, line
    print("ok  test_peak_tag_absent_when_salience_missing")


def test_peak_tag_absent_on_no_signal_fallback():
    # score == 0.0 means peak_ms just restates hero_ts_ms -- not a real peak.
    cut = _salience_cut("f:pk", 1000, 4000, {"peak_ms": 2500, "score": 0.0})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "peak:" not in line, line
    print("ok  test_peak_tag_absent_on_no_signal_fallback")


def test_peak_tag_absent_when_pinned_to_the_start():
    cut = _salience_cut("f:pk", 1000, 4000, {"peak_ms": 1050, "score": 0.7})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "peak:" not in line, line
    print("ok  test_peak_tag_absent_when_pinned_to_the_start")


def test_peak_tag_absent_when_pinned_to_the_end():
    cut = _salience_cut("f:pk", 1000, 4000, {"peak_ms": 3950, "score": 0.7})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "peak:" not in line, line
    print("ok  test_peak_tag_absent_when_pinned_to_the_end")


def test_cast_line_lists_majors_with_voices_and_others_by_id():
    persons = [
        {"person_id": "P0", "display": "bald man, beard", "is_major": True, "owned_voices": ["V0"]},
        {"person_id": "P1", "display": "woman, long hair", "is_major": True, "owned_voices": []},
        {"person_id": "P2", "display": "background extra", "is_major": False, "owned_voices": []},
    ]
    line = fm._cast_line(persons)
    assert line.startswith("CAST: "), line
    assert "P0 (bald man, beard) [voice:V0]" in line, line
    assert "P1 (woman, long hair)" in line and "[voice:" not in line.split("P1")[1].split(";")[0], line
    assert "other: P2" in line, line
    print("ok  test_cast_line_lists_majors_with_voices_and_others_by_id")


def test_cast_line_empty_with_no_persons():
    assert fm._cast_line([]) == ""
    print("ok  test_cast_line_empty_with_no_persons")


def test_default_energy_from_genre():
    """The tree opens the slider per genre: long-form calm, short-form punchy."""
    calm = fm.build_clip_tree("ffffffff-2222",
                              {"name": "Pod", "duration_ms": 8000, "content_type": "interview"},
                              [_thought_cut()])
    punchy = fm.build_clip_tree("ffffffff-3333",
                                {"name": "Ad", "duration_ms": 8000, "content_type": "product"},
                                [_thought_cut()])
    assert calm["default_energy"] < 0.5 < punchy["default_energy"], (
        calm["default_energy"], punchy["default_energy"])
    print("ok  test_default_energy_from_genre")


def main():
    test_thought_levels_become_variants()
    test_moment_line_flags_offcamera()
    test_moment_line_shows_channel_subject_v2()
    test_split_rung_becomes_keep_spans()
    test_no_ladder_uses_flat_span()
    test_facets_surface_on_moment()
    test_map_text_lists_variants_no_atoms()
    test_reconciled_shows_face_and_cam_override()
    test_said_beat_on_listener_camera_reads_pic_first_with_alt_pic()
    test_said_beat_with_transcript_quotes_verbatim_text_first()
    test_said_beat_shows_aud_tag_from_speech_quality()
    test_aud_tag_absent_without_speech_quality()
    test_action_beat_never_gets_said_text()
    test_said_beat_transcript_truncates_in_compact_mode()
    test_said_beat_with_no_transcript_falls_back_to_visual_gist()
    test_span_detail_pads_and_filters_via_shared_sentences_cache()
    test_alt_pic_absent_without_co_occurrence()
    test_done_beat_pic_first_parity_and_snd_silence()
    test_speaker_person_renders_on_cam_when_shown()
    test_annotate_dups_reads_take_group_id()
    test_junk_moment_renders_terse_line_not_the_rich_one()
    test_junk_moment_still_resolves_in_map_index()
    test_non_junk_moment_shows_continuity_position_and_weld_marks()
    test_first_cut_has_no_prev_weld_mark()
    test_cut_with_no_continuity_block_has_no_continuity_tag()
    test_take_group_excludes_junk_member()
    test_peak_tag_renders_for_an_interior_peak()
    test_peak_tag_absent_when_salience_missing()
    test_peak_tag_absent_on_no_signal_fallback()
    test_peak_tag_absent_when_pinned_to_the_start()
    test_peak_tag_absent_when_pinned_to_the_end()
    test_source_contiguous_beats_form_a_run_channel_agnostic()
    test_cast_line_lists_majors_with_voices_and_others_by_id()
    test_cast_line_empty_with_no_persons()
    test_default_energy_from_genre()
    print("\nall footage-map tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
