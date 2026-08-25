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
    assert m["cut_id"] == "ffffffff:c00", m["cut_id"]
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
    assert "levels:broad|calm|balanced|tight|sharp" in lines[1], lines[1]
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
    by_id = {m["cut_id"].split(":")[-1]: m for m in tree["moments"]}
    # m00..m03 are adjacent in source time (mixed channels) -> ONE run of 4.
    run0 = by_id["c00"].get("run_id")
    assert run0 is not None
    assert all(by_id[mid]["run_id"] == run0 for mid in ("c00", "c01", "c02", "c03")), by_id
    assert by_id["c00"]["run_len"] == 4, by_id["c00"]["run_len"]
    assert [by_id[mid]["run_pos"] for mid in ("c00", "c01", "c02", "c03")] == [0, 1, 2, 3]
    # The far-away beat (big gap) is its own lone run -> no tag.
    assert by_id["c04"].get("run_id") is None, "far-away beat starts no (multi) run"
    # Channel never entered it: a said and a shown sit inside the run too.
    assert by_id["c01"]["channel"] == "said" and by_id["c02"]["channel"] == "shown"
    # brain_material_truth.plan.md Part 4.1: run membership is no longer an
    # inline tag on the moment line -- it's structural now (_clip_block's
    # header + indentation). See test_clip_block_groups_run_members below.
    assert "run:" not in fm._moment_line(by_id["c01"]), fm._moment_line(by_id["c01"])
    print("ok  test_source_contiguous_beats_form_a_run_channel_agnostic")


def test_clip_block_groups_run_members_under_a_continuity_header():
    """brain_material_truth.plan.md Part 4.1: continuity is the SHAPE of the
    listing now, not a suffix tag. A run's members render as one block --
    a header (source-time span) followed by each member, indented, in
    source order -- and a lone moment (no run) renders exactly as before,
    at the top level, with no header at all."""
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
    block = fm._clip_block(tree)
    lines = block.splitlines()
    # Header, then the run's own continuity header, then 4 indented members,
    # then the lone moment with no header at all -- source order throughout.
    assert lines[0].startswith('CLIP ffffffff "Match"'), lines[0]
    assert "plays continuously" in lines[1] and "0:00" in lines[1] and "0:06" in lines[1], lines[1]
    member_lines = lines[2:6]
    for ln, cut_no in zip(member_lines, ("c00", "c01", "c02", "c03")):
        assert ln.startswith("    "), ln           # extra indent under the header
        assert cut_no in ln, ln
        assert "run:" not in ln, ln                # membership is structural now, not a tag
    assert "c04" in lines[6] and not lines[6].startswith("    "), lines[6]
    assert "plays continuously" not in lines[6], lines[6]   # lone moment -- no header
    # Members stay in source order inside the block.
    assert [ln.split()[0] for ln in member_lines] == ["c00", "c01", "c02", "c03"], member_lines
    print("ok  test_clip_block_groups_run_members_under_a_continuity_header")


def test_clip_block_run_indent_composes_with_takes_lines():
    """Hard requirement (Part 4.1): `_takes_lines`' own sub-block formatting
    is untouched -- a run member that also has take-group alternates still
    gets its `takes:` sub-block, uniformly shifted under the run's extra
    indent along with the rest of that member's line(s)."""
    cut_a = _cut("c:0", 0, 2000, "hello", channel="said", speaker="S0",
                ladder=[_rung("balanced", 0, 2000, "hello", 0.6)],
                take_group_id="tg1", take_role="winner")
    cut_b = _cut("c:1", 2100, 4000, "world", channel="said", speaker="S0",
                ladder=[_rung("balanced", 2100, 4000, "world", 0.6)])
    cut_other = _cut("o:0", 1000, 4000, "hello", channel="said", speaker="S9",
                     ladder=[_rung("balanced", 1000, 4000, "hello", 0.73)],
                     take_group_id="tg1", take_role="take")
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut_a, cut_b])
    tree_other = fm.build_clip_tree("99999999-1111", {"name": "O", "duration_ms": 8000}, [cut_other])
    fm._annotate_dups([tree, tree_other])

    block = fm._clip_block(tree)
    lines = block.splitlines()
    assert "plays continuously" in lines[1], lines[1]
    assert lines[2].startswith("    c00"), lines[2]
    assert lines[3].strip().startswith("takes:"), lines[3]
    assert lines[3].startswith("      "), lines[3]   # takes_lines' own 4sp + the block's 2sp
    assert "99999999:c00" in lines[4], lines[4]
    assert lines[4].startswith("      "), lines[4]
    assert lines[5].startswith("    c01"), lines[5]
    print("ok  test_clip_block_run_indent_composes_with_takes_lines")


def test_reconciled_shows_face_and_cam_override():
    """PIC reports WHOSE FACE is visible in the clip (per-cut-occurrence face
    clustering, voice_first_identity.plan.md Phase D) -- and SND names who's
    heard, tagged OFF-CAM when the speaking voice's bound person isn't the
    one shown here (the listener-camera case: P1 talks, but this clip shows
    P0). PIC leads the line; SND trails it."""
    cut = _cut("48c93cef:c03", 1000, 4000, "and then we shipped it", channel="said",
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
    cut = _cut("48c93cef:c24", 1000, 4000, "that freedom he gave you", channel="said",
               subject="person", score=0.6,
               ladder=[_rung("balanced", 1000, 4000, "that freedom he gave you", 0.6)],
               voice_ids=["V2"], speaker_person="P2", visible_persons=["P1"], on_camera=False,
               framing="med")
    tree = fm.build_clip_tree("48c93cef-aaa", {"name": "Cam A", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    # Folded on by `_annotate_dups` in the real path; set directly here to test
    # the render in isolation (mirrors how `_dups_block` fixtures worked before).
    m["alt_pic"] = [{"cut_id": "1aedb093:c13", "file": "1aedb093-bbb",
                     "visible_persons": ["P2"], "speaker_person": "P2",
                     "framing": "med", "score": 0.73, "restart": False}]
    line = fm._moment_line(m)
    assert "PIC:P1" in line, line
    assert "SND:P2 OFF-CAM speaking" in line, line
    assert line.index("PIC:") < line.index("SND:"), line
    assert "·alt-PIC:P2→1aedb093:c13" in line, line
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


def test_action_beat_surfaces_incidental_said_text_brain_mirror_readside():
    """brain_mirror_readside.plan.md section 3.2 (band-aid C retired): a
    done/shown beat's visual label still leads the beat line, but said_text
    is no longer gated on channel=="said" -- incidental spoken words under
    a picture cut (the slide-voiceover case) now surface too, instead of
    being silently hidden from the brain."""
    cut = _cut("f:tr2", 1000, 3000, "nods thoughtfully", channel="done", subject="person",
               score=0.6, ladder=[_rung("balanced", 1000, 3000, "nods thoughtfully", 0.6)])
    sentences = ({"speaker": "S0", "text": "narration that happens to overlap in time",
                 "src_in_ms": 1000, "src_out_ms": 3000},)
    with mock.patch.object(fm, "_sentences_for_file", return_value=sentences):
        tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    assert m["said_text"] == "narration that happens to overlap in time", m["said_text"]
    line = fm._moment_line(m)
    # The done beat's own visual gist still leads the primary quote (never
    # displaced by incidental words); the words surface as a SEPARATE tag.
    assert '"nods thoughtfully"' in line, line
    assert 'words:"narration that happens to overlap' in line, line
    # brain_material_truth.plan.md Part 4.4: "incidental:" claimed an
    # importance judgment this code has no basis to make -- renamed to the
    # neutral `words:`.
    assert "incidental:" not in line, line
    print("ok  test_action_beat_surfaces_incidental_said_text_brain_mirror_readside")


def test_action_beat_said_text_empty_when_nothing_overlaps():
    cut = _cut("f:tr2b", 1000, 3000, "nods thoughtfully", channel="done", subject="person",
               score=0.6, ladder=[_rung("balanced", 1000, 3000, "nods thoughtfully", 0.6)])
    with mock.patch.object(fm, "_sentences_for_file", return_value=()):
        tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    assert tree["moments"][0]["said_text"] == ""
    print("ok  test_action_beat_said_text_empty_when_nothing_overlaps")


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
    cut = _cut("48c93cef:c00", 1000, 4000, "we shipped it",
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
    assert ma["alt_pic"][0]["cut_id"] == mb["cut_id"], ma["alt_pic"]
    assert mb["alt_pic"][0]["cut_id"] == ma["cut_id"], mb["alt_pic"]
    assert {mf["take_role"] for mf in g["member_facts"]} == {"winner", "take"}, g["member_facts"]
    print("ok  test_annotate_dups_reads_take_group_id")


# --------------------------------------------------------------------------
# brain_cut_index_fidelity.plan.md 5 (A7): a `shown`/`done` cut and a `said`
# cut from the SAME FILE with overlapping timestamps are, deterministically,
# the same underlying audio -- placing both plays the identical words twice.
# --------------------------------------------------------------------------

def _said_and_shown_overlap(sentences=None):
    said_cut = _cut("f:s", 1000, 4000, "thanks for having me", channel="said",
                    speaker="S0", score=0.5,
                    ladder=[_rung("balanced", 1000, 4000, "thanks for having me", 0.5)])
    shown_cut = _cut("f:sh", 1500, 3500, "a wide shot of the room", channel="shown",
                     subject="object", speaker=None, score=0.4,
                     ladder=[_rung("balanced", 1500, 3500, "x", 0.4)])
    with mock.patch.object(fm, "_sentences_for_file",
                           return_value=sentences if sentences is not None else
                           ({"speaker": "S0", "text": "thanks for having me",
                             "src_in_ms": 1000, "src_out_ms": 4000},)):
        tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000},
                                  [said_cut, shown_cut])
    return tree


def test_annotate_shared_audio_links_overlapping_said_and_shown_same_file():
    tree = _said_and_shown_overlap()
    fm._annotate_shared_audio([tree])
    said_m = next(m for m in tree["moments"] if m["channel"] == "said")
    shown_m = next(m for m in tree["moments"] if m["channel"] == "shown")
    assert said_m["dup_audio_refs"] == [shown_m["cut_id"]], said_m
    assert shown_m["dup_audio_refs"] == [said_m["cut_id"]], shown_m
    print("ok  test_annotate_shared_audio_links_overlapping_said_and_shown_same_file")


def test_moment_line_renders_dup_audio_cross_reference_both_sides():
    tree = _said_and_shown_overlap()
    fm._annotate_shared_audio([tree])
    said_m = next(m for m in tree["moments"] if m["channel"] == "said")
    shown_m = next(m for m in tree["moments"] if m["channel"] == "shown")
    said_line = fm._moment_line(said_m)
    shown_line = fm._moment_line(shown_m)
    assert f"dup-audio:{shown_m['cut_id']}" in said_line, said_line
    assert f"dup-audio:{said_m['cut_id']}" in shown_line, shown_line
    print("ok  test_moment_line_renders_dup_audio_cross_reference_both_sides")


def test_annotate_shared_audio_no_link_without_temporal_overlap():
    said_cut = _cut("f:s2", 1000, 2000, "hello", channel="said", speaker="S0", score=0.5,
                    ladder=[_rung("balanced", 1000, 2000, "hello", 0.5)])
    shown_cut = _cut("f:sh2", 5000, 6000, "a wide shot", channel="shown", subject="object",
                     speaker=None, score=0.4, ladder=[_rung("balanced", 5000, 6000, "x", 0.4)])
    with mock.patch.object(fm, "_sentences_for_file",
                           return_value=({"speaker": "S0", "text": "hello",
                                         "src_in_ms": 1000, "src_out_ms": 2000},)):
        tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000},
                                  [said_cut, shown_cut])
    fm._annotate_shared_audio([tree])
    for m in tree["moments"]:
        assert "dup_audio_refs" not in m, m
    print("ok  test_annotate_shared_audio_no_link_without_temporal_overlap")


def test_annotate_shared_audio_no_link_when_shown_cut_carries_no_words():
    """Overlap alone isn't enough -- a shown cut with NO incidental words
    (silent b-roll) has no audio to duplicate, even if it overlaps a said
    cut in time."""
    said_cut = _cut("f:s3", 1000, 4000, "thanks for having me", channel="said",
                    speaker="S0", score=0.5,
                    ladder=[_rung("balanced", 1000, 4000, "thanks for having me", 0.5)])
    shown_cut = _cut("f:sh3", 1500, 3500, "a wide shot of the room", channel="shown",
                     subject="object", speaker=None, score=0.4,
                     ladder=[_rung("balanced", 1500, 3500, "x", 0.4)])
    with mock.patch.object(fm, "_sentences_for_file", return_value=()):
        tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000},
                                  [said_cut, shown_cut])
    fm._annotate_shared_audio([tree])
    for m in tree["moments"]:
        assert "dup_audio_refs" not in m, m
    print("ok  test_annotate_shared_audio_no_link_when_shown_cut_carries_no_words")


def test_annotate_shared_audio_no_link_across_different_files():
    said_cut = _cut("fa:s", 1000, 4000, "thanks for having me", channel="said",
                    speaker="S0", score=0.5,
                    ladder=[_rung("balanced", 1000, 4000, "thanks for having me", 0.5)])
    shown_cut = _cut("fb:sh", 1500, 3500, "a wide shot", channel="shown", subject="object",
                     speaker=None, score=0.4, ladder=[_rung("balanced", 1500, 3500, "x", 0.4)])
    with mock.patch.object(fm, "_sentences_for_file",
                           return_value=({"speaker": "S0", "text": "thanks for having me",
                                         "src_in_ms": 1000, "src_out_ms": 4000},)):
        tree_a = fm.build_clip_tree("ffffffff-1111", {"name": "A", "duration_ms": 8000}, [said_cut])
        tree_b = fm.build_clip_tree("22222222-2222", {"name": "B", "duration_ms": 8000}, [shown_cut])
    fm._annotate_shared_audio([tree_a, tree_b])
    for m in tree_a["moments"] + tree_b["moments"]:
        assert "dup_audio_refs" not in m, m
    print("ok  test_annotate_shared_audio_no_link_across_different_files")


def test_annotate_shared_audio_excludes_junk_on_either_side():
    said_cut = _cut("f:s4", 1000, 4000, "thanks for having me", channel="said",
                    speaker="S0", score=0.5,
                    ladder=[_rung("balanced", 1000, 4000, "thanks for having me", 0.5)],
                    junk=True, junk_reason="false start")
    shown_cut = _cut("f:sh4", 1500, 3500, "a wide shot of the room", channel="shown",
                     subject="object", speaker=None, score=0.4,
                     ladder=[_rung("balanced", 1500, 3500, "x", 0.4)])
    with mock.patch.object(fm, "_sentences_for_file",
                           return_value=({"speaker": "S0", "text": "thanks for having me",
                                         "src_in_ms": 1000, "src_out_ms": 4000},)):
        tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000},
                                  [said_cut, shown_cut])
    fm._annotate_shared_audio([tree])
    for m in tree["moments"]:
        assert "dup_audio_refs" not in m, m
    print("ok  test_annotate_shared_audio_excludes_junk_on_either_side")


def test_snd_state_says_muted_talk_is_recoverable():
    """brain_material_truth.plan.md Part 4.4: a cut still muted with real
    words underneath must say the words can be unmuted, not merely that
    they were suppressed -- the wording that let a founder's own slide
    narration read as unusable noise in the failing edit ("these are
    muted(talk) -- incidental founder speech")."""
    cut = _cut("f:mt", 1000, 2000, "slide up", channel="shown", subject="object",
               speaker=None, score=0.5, ladder=[_rung("balanced", 1000, 2000, "slide up", 0.5)],
               audio="speech", mute=True)
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "SND:muted(talk→unmute)" in line, line
    print("ok  test_snd_state_says_muted_talk_is_recoverable")


def test_snd_state_ambient_mute_stays_terse():
    """Only the talk case gets the recoverability wording -- an ambient/
    other mute is genuinely just suppressed sound with no words to recover,
    so it stays as terse as before."""
    cut = _cut("f:ma", 1000, 2000, "b-roll", channel="shown", subject="object",
               speaker=None, score=0.5, ladder=[_rung("balanced", 1000, 2000, "b-roll", 0.5)],
               audio="sound", mute=True)
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "SND:muted(ambient)" in line, line
    assert "unmute" not in line, line
    print("ok  test_snd_state_ambient_mute_stays_terse")


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
    assert line.strip().startswith("c00 [JUNK: camera cue]"), line
    assert "↔cut:2/3⋯" in line, line
    assert ("PIC:" not in line and "SND:" not in line
            and "energy:" not in line and "levels:" not in line), line
    print("ok  test_junk_moment_renders_terse_line_not_the_rich_one")


def test_junk_moment_renders_a_short_description():
    """brain_cut_index_fidelity.plan.md 5 (A6): a bare [JUNK: reason] code
    hides the whole clip behind it -- the brain can't judge whether the
    junk call was right without seeing what it actually is. Short, matching
    every other secondary tag's terse convention -- never the full line."""
    cut = _cut("f:junk2", 1000, 1600, "camera pans to catch the crew adjusting a light stand",
               channel="shown", subject="object", speaker=None, score=0.1,
               ladder=[_rung("balanced", 1000, 1600, "x", 0.1)],
               junk=True, junk_reason="camera cue")
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert '[JUNK: camera cue] "camera pans to catch the crew…"' in line, line
    print("ok  test_junk_moment_renders_a_short_description")


def test_junk_moment_renders_nothing_extra_when_no_description():
    cut = _cut("f:junk3", 1000, 1600, "", channel="shown", subject="object", speaker=None,
               score=0.1, ladder=[_rung("balanced", 1000, 1600, "x", 0.1)],
               junk=True, junk_reason="dead air")
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert line.strip() == f"{tree['moments'][0]['cut_id'].split(':')[-1]} [JUNK: dead air] [0:01-0:01 0.6s]", line
    print("ok  test_junk_moment_renders_nothing_extra_when_no_description")


def test_junk_moment_still_resolves_in_map_index():
    """Junk stays independently PLACEABLE -- its ref must still resolve through
    _MapIndex (skip-by-default in the prompt framing, never un-referenceable)."""
    cut = _cut("f:junk", 1000, 1600, "and go", channel="shown", speaker=None,
               ladder=[_rung("balanced", 1000, 1600, "and go", 0.1)], junk=True,
               junk_reason="camera cue")
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    idx = _MapIndex({"clips": [tree]})
    mid = tree["moments"][0]["cut_id"]
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


# --------------------------------------------------------------------------
# landmarks breadcrumb (brain_cut_index_fidelity.plan.md 4.1, A3): `sig:`
# renders each channel's own OFFSETS (act/adx/sil/shot), not a count -- a
# count said structure exists but never where, which is exactly what
# judging pacing from content needs.
# --------------------------------------------------------------------------

def _landmarks_cut(hero_id, in_ms, out_ms, landmarks):
    return _cut(hero_id, in_ms, out_ms, "watch this", landmarks=landmarks)


def test_landmarks_tag_fixed_channel_order_regardless_of_dict_order():
    m = {"landmarks": {
        "shot": {"n": 1, "cuts": [{"off": 2000, "hard": True}]},
        "act": {"n": 2, "hits": [500, 1500]},
        "sil": {"n": 1, "gaps": [{"off": 1000, "dur": 300}]},
    }}
    tag = fm._landmarks_tag(m)
    # act, then sil, then shot -- fixed order (_LANDMARK_TAG_ORDER),
    # independent of the dict's own insertion order above.
    assert tag == " sig:act+0.5s,+1.5s|sil+1.0s.0.3s|shot+2.0s!", tag
    print("ok  test_landmarks_tag_fixed_channel_order_regardless_of_dict_order")


def test_landmarks_tag_renders_offsets_via_moment_line():
    cut = _landmarks_cut("f:lm", 1000, 4000, {
        "shot": {"n": 1, "cuts": [{"off": 2000, "hard": True}]},
        "act": {"n": 3, "hits": [500, 1000, 1500]},
    })
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "sig:act+0.5s,+1.0s,+1.5s|shot+2.0s!" in line, line
    print("ok  test_landmarks_tag_renders_offsets_via_moment_line")


def test_landmarks_tag_adx_shows_direction():
    m = {"landmarks": {"adx": {"n": 2, "changes": [
        {"off": 800, "dir": "up"}, {"off": 2100, "dir": "down"}]}}}
    tag = fm._landmarks_tag(m)
    assert tag == " sig:adx+0.8s↑,+2.1s↓", tag
    print("ok  test_landmarks_tag_adx_shows_direction")


def test_landmarks_tag_sil_shows_offset_and_duration():
    m = {"landmarks": {"sil": {"n": 1, "gaps": [{"off": 1500, "dur": 300}]}}}
    tag = fm._landmarks_tag(m)
    assert tag == " sig:sil+1.5s.0.3s", tag
    print("ok  test_landmarks_tag_sil_shows_offset_and_duration")


def test_landmarks_tag_shot_marks_hard_cuts_not_soft_ones():
    m = {"landmarks": {"shot": {"n": 2, "cuts": [
        {"off": 2000, "hard": True}, {"off": 2500, "hard": False}]}}}
    tag = fm._landmarks_tag(m)
    assert tag == " sig:shot+2.0s!,+2.5s", tag
    print("ok  test_landmarks_tag_shot_marks_hard_cuts_not_soft_ones")


def test_landmarks_tag_caps_come_from_ingest_not_the_renderer():
    """The renderer does not additionally truncate -- l3/landmarks.py's own
    _LANDMARK_ACT_CAP (5) already capped this list at ingest, by strength,
    before it was ever stored; the renderer just formats whatever is there."""
    cut = _landmarks_cut("f:lm", 1000, 4000,
                         {"act": {"n": 5, "hits": [100, 200, 300, 400, 500]}})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "sig:act+0.1s,+0.2s,+0.3s,+0.4s,+0.5s" in line, line
    print("ok  test_landmarks_tag_caps_come_from_ingest_not_the_renderer")


def test_landmarks_tag_absent_when_landmarks_empty():
    cut = _landmarks_cut("f:lm", 1000, 4000, {})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "sig:" not in line, line
    print("ok  test_landmarks_tag_absent_when_landmarks_empty")


def test_landmarks_tag_skips_a_channel_with_no_stored_offsets():
    cut = _landmarks_cut("f:lm", 1000, 4000, {"act": {"n": 0, "hits": []}})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "sig:" not in line, line
    print("ok  test_landmarks_tag_skips_a_channel_with_no_stored_offsets")


# --------------------------------------------------------------------------
# scene specificity (cut_structure_and_scene_specificity.plan.md Part 3):
# `spec:"..."` -- ADDITIVE alongside the generic label/summary, never a
# replacement.
# --------------------------------------------------------------------------

def test_specific_tag_renders_when_scene_specifics_present():
    cut = _cut("f:sp", 1000, 4000, "a machine in a factory",
               scene_specifics={"specific": "CNC lathe turning a steel shaft", "label": "milling"})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert 'spec:"CNC lathe turning a steel shaft"' in line, line
    # Additive: the generic label still renders too (never replaced).
    assert "a machine in a factory" in line, line
    print("ok  test_specific_tag_renders_when_scene_specifics_present")


def test_specific_tag_absent_when_not_yet_enriched():
    cut = _cut("f:sp", 1000, 4000, "a machine in a factory")
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "spec:" not in line, line
    print("ok  test_specific_tag_absent_when_not_yet_enriched")


def test_specific_tag_absent_when_specifics_present_but_empty_string():
    cut = _cut("f:sp", 1000, 4000, "a machine in a factory", scene_specifics={"specific": "", "label": ""})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "spec:" not in line, line
    print("ok  test_specific_tag_absent_when_specifics_present_but_empty_string")


# --------------------------------------------------------------------------
# brain_cut_specifics_wiring.plan.md: the NEW flat, question-bank-keyed
# scene_specifics shape (vcut_pass2_video_specifics.plan.md) -- rendered via
# _specific_tag directly (minimal dicts) for precise coverage of the
# composition/dedupe/compact logic, plus one full-pipeline sanity check.
# --------------------------------------------------------------------------

def test_specific_tag_new_shape_absent_when_scene_specifics_missing_entirely():
    assert fm._specific_tag({}) == ""
    print("ok  test_specific_tag_new_shape_absent_when_scene_specifics_missing_entirely")


def test_specific_tag_new_shape_absent_when_scene_specifics_is_empty_dict():
    assert fm._specific_tag({"scene_specifics": {}}) == ""
    print("ok  test_specific_tag_new_shape_absent_when_scene_specifics_is_empty_dict")


def test_specific_tag_new_single_flag_renders_subject_action_shot_size_camera():
    m = {"scene_specifics": {
        "subject": "barista", "action": "pours latte", "shot_size": "medium", "camera_move": "push",
    }, "camera": "static"}
    line = fm._specific_tag(m)
    assert "barista pours latte" in line, line
    assert "medium" in line, line
    assert "push" in line, line
    print("ok  test_specific_tag_new_single_flag_renders_subject_action_shot_size_camera")


def test_specific_tag_new_single_flag_subject_only_no_dangling_space():
    m = {"scene_specifics": {"subject": "a dog"}}
    assert fm._specific_tag(m) == ' spec:"a dog"'
    print("ok  test_specific_tag_new_single_flag_subject_only_no_dangling_space")


def test_specific_tag_new_shape_omits_static_camera_move_and_none_motion_and_good_headroom():
    m = {"scene_specifics": {
        "subject": "x", "camera_move": "static", "motion_direction": "none", "headroom_lookroom": "good",
    }, "camera": ""}
    line = fm._specific_tag(m)
    assert "static" not in line, line
    assert "none" not in line, line
    assert "good" not in line, line
    print("ok  test_specific_tag_new_shape_omits_static_camera_move_and_none_motion_and_good_headroom")


def test_specific_tag_new_shape_renders_on_screen_text_gisted_and_quoted():
    m = {"scene_specifics": {"subject": "sign", "on_screen_text": "50% off everything this weekend only"},
        "screen_text": ""}
    line = fm._specific_tag(m)
    assert "text:'" in line, line
    print("ok  test_specific_tag_new_shape_renders_on_screen_text_gisted_and_quoted")


def test_specific_tag_new_shape_renders_usable_hook_and_capped_tags():
    m = {"scene_specifics": {
        "subject": "x", "usable": "weak", "hook_potential": "high",
        "tags": ["pour", "closeup", "coffee", "extra", "dropped"],
    }}
    line = fm._specific_tag(m)
    assert "usable:weak" in line, line
    assert "hook:high" in line, line
    assert "tags:pour,closeup,coffee" in line, line
    assert "extra" not in line and "dropped" not in line, line
    print("ok  test_specific_tag_new_shape_renders_usable_hook_and_capped_tags")


def test_specific_tag_dedupes_camera_move_against_code_derived_cam():
    m = {"scene_specifics": {"subject": "x", "camera_move": "pan"}, "camera": "pan left"}
    line = fm._specific_tag(m)
    assert "pan" not in line, line  # dropped -- the code cam: tag already says it
    print("ok  test_specific_tag_dedupes_camera_move_against_code_derived_cam")


def test_specific_tag_dedupes_on_screen_text_against_code_derived_screen_text():
    m = {"scene_specifics": {"subject": "x", "on_screen_text": "SALE 50% OFF"}, "screen_text": "SALE 50% OFF"}
    line = fm._specific_tag(m)
    assert "text:" not in line, line
    print("ok  test_specific_tag_dedupes_on_screen_text_against_code_derived_screen_text")


def test_specific_tag_camera_move_survives_when_it_does_not_match_code_cam():
    m = {"scene_specifics": {"subject": "x", "camera_move": "handheld"}, "camera": "static"}
    line = fm._specific_tag(m)
    assert "handheld" in line, line
    print("ok  test_specific_tag_camera_move_survives_when_it_does_not_match_code_cam")


def test_specific_tag_new_merged_renders_mini_shot_list_with_deltas():
    m = {"scene_specifics": {
        "subject": "barista", "action": "works counter", "shot_size": "medium",
        "moments": [
            {"t_ms": 1200, "summary": "barista grabs cup"},
            {"t_ms": 3400, "summary": "pours latte, close"},
            {"t_ms": 5100, "summary": "slides across counter"},
        ],
    }, "in_ms": 0}
    line = fm._specific_tag(m)
    assert 'spec:"barista works counter · medium"' in line, line
    assert "inside:[" in line, line
    assert "+1.2s barista grabs cup" in line, line
    assert "+3.4s pours latte, close" in line, line
    assert "+5.1s slides across counter" in line, line
    print("ok  test_specific_tag_new_merged_renders_mini_shot_list_with_deltas")


def test_specific_tag_new_merged_deltas_are_relative_to_the_cuts_own_in_ms():
    m = {"scene_specifics": {"subject": "x", "moments": [
        {"t_ms": 5200, "summary": "a"}, {"t_ms": 7200, "summary": "b"},
    ]}, "in_ms": 5000}
    line = fm._specific_tag(m)
    assert "+0.2s a" in line, line
    assert "+2.2s b" in line, line
    print("ok  test_specific_tag_new_merged_deltas_are_relative_to_the_cuts_own_in_ms")


def test_specific_tag_new_merged_caps_moments_list_with_more_suffix():
    moments = [{"t_ms": 1000 * i, "summary": f"moment {i}"} for i in range(8)]
    m = {"scene_specifics": {"subject": "x", "moments": moments}, "in_ms": 0}
    line = fm._specific_tag(m)
    assert line.count("moment ") == 5, line
    assert "…+3 more" in line, line
    print("ok  test_specific_tag_new_merged_caps_moments_list_with_more_suffix")


def test_specific_tag_new_merged_moment_falls_back_to_subject_action_with_no_summary():
    m = {"scene_specifics": {"subject": "x", "moments": [
        {"t_ms": 1000, "subject": "cup", "action": "fills"},
    ]}, "in_ms": 0}
    line = fm._specific_tag(m)
    assert "cup fills" in line, line
    print("ok  test_specific_tag_new_merged_moment_falls_back_to_subject_action_with_no_summary")


def test_specific_tag_compact_mode_drops_moments_list_and_secondary_tokens():
    m = {"scene_specifics": {
        "subject": "barista", "action": "pours latte", "shot_size": "medium",
        "usable": "strong", "tags": ["a", "b"],
        "moments": [{"t_ms": 1200, "summary": "grabs cup"}, {"t_ms": 3400, "summary": "pours"}],
    }, "in_ms": 0}
    line = fm._specific_tag(m, compact=True)
    assert "barista pours latte" in line and "medium" in line, line
    assert "usable" not in line, line
    assert "tags:" not in line, line
    assert "inside:" not in line, line
    print("ok  test_specific_tag_compact_mode_drops_moments_list_and_secondary_tokens")


def test_specific_tag_compact_mode_legacy_shape_unaffected():
    m = {"scene_specifics": {"specific": "CNC lathe turning a steel shaft", "label": "milling"}}
    assert fm._specific_tag(m, compact=True) == fm._specific_tag(m, compact=False)
    print("ok  test_specific_tag_compact_mode_legacy_shape_unaffected")


def test_specific_tag_full_pipeline_new_shape_renders_on_the_beat_line():
    """Full pipeline sanity (item 7): a real cut fixture carrying the new
    vcut scene_specifics shape renders spec:"..." on assemble_map's beat
    line, alongside the generic label -- additive, matching the legacy
    behavior already proven above."""
    cut = _cut("f:sp2", 1000, 4000, "a person at a counter",
              scene_specifics={"subject": "barista", "action": "pours latte", "shot_size": "medium"})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert 'spec:"barista pours latte · medium"' in line, line
    assert "a person at a counter" in line, line
    print("ok  test_specific_tag_full_pipeline_new_shape_renders_on_the_beat_line")


# --------------------------------------------------------------------------
# Fact dedup (brain_material_truth.plan.md Part 4.2): a slide cut's own text
# used to render three times -- primary quote, graphic:, spec:"...text:'...'"
# -- ~80 wasted chars per cut. Each duplicate is suppressed independently.
# --------------------------------------------------------------------------

def test_graphic_tag_suppressed_when_it_repeats_the_primary_quote():
    cut = _cut("f:dup1", 1000, 4000, "Slide: The Insight", channel="shown", subject="graphic",
               speaker=None, summary="The Insight")
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert '"Slide: The Insight"' in line, line
    assert "graphic:" not in line, line
    print("ok  test_graphic_tag_suppressed_when_it_repeats_the_primary_quote")


def test_graphic_tag_still_renders_when_it_says_something_new():
    cut = _cut("f:dup2", 1000, 4000, "Slide: The Insight", channel="shown", subject="graphic",
               speaker=None, summary="a bar chart trending upward")
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert '"Slide: The Insight"' in line, line
    assert 'graphic:"a bar chart trending upward"' in line, line
    print("ok  test_graphic_tag_still_renders_when_it_says_something_new")


# --------------------------------------------------------------------------
# brain_cut_index_fidelity.plan.md 3 (A2): `summary` is the rich stored
# description; `label` (which becomes `primary` for a shown/done cut) is a
# SEPARATE, already-6-word-capped field derived FROM summary at ingest
# (vcut/store.py._short_label), not an independent duplicate. The dedup
# check must be ASYMMETRIC: `primary` being a strict prefix of a LONGER
# `summary` is not redundancy, it's exactly the extra content 3/A2 exists
# to surface -- confirmed against real data (label is uniformly 6.0 words,
# summary median 11-15, max 93, on the same rows).
# --------------------------------------------------------------------------

def test_graphic_tag_renders_full_summary_when_primary_is_just_its_prefix():
    """The exact real-data shape: label (primary) is summary's own first 6
    words, truncated at ingest -- summary itself continues well past that
    and must render in FULL, not be suppressed as a duplicate of the prefix
    that was truncated FROM it."""
    cut = _cut("f:dup6", 1000, 4000, "A slide titled 'The Market' is",
               channel="shown", subject="graphic", speaker=None,
               summary="A slide titled 'The Market' is shown, discussing the "
                       "growth of the AI video editing market.")
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert ('graphic:"A slide titled \'The Market\' is shown, discussing the '
            'growth of the AI video editing market."') in line, line
    print("ok  test_graphic_tag_renders_full_summary_when_primary_is_just_its_prefix")


def test_graphic_tag_not_truncated_to_six_words():
    """Direct regression for the plan's own cited failure mode: a summary
    longer than 6 words must not get chopped to 'A chef uses tongs to
    stir' -- the discriminating tail is exactly what tells similar shots
    apart."""
    long_summary = ("A chef uses tongs to stir the vegetables sizzling in "
                    "the wok, glancing up at the camera briefly.")
    cut = _cut("f:dup7", 1000, 4000, "A chef uses tongs to",
               channel="shown", subject="person", speaker=None, summary=long_summary)
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert f'graphic:"{long_summary}"' in line, line
    assert "…" not in line, line
    print("ok  test_graphic_tag_not_truncated_to_six_words")


def test_graphic_tag_still_suppressed_when_summary_equals_primary_exactly():
    cut = _cut("f:dup8", 1000, 4000, "The speaker appears on screen.",
               channel="shown", subject="person", speaker=None,
               summary="The speaker appears on screen.")
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "graphic:" not in line, line
    print("ok  test_graphic_tag_still_suppressed_when_summary_equals_primary_exactly")


def test_spec_on_screen_text_suppressed_when_it_repeats_the_primary_quote():
    cut = _cut("f:dup3", 1000, 4000, "Slide: The Insight", channel="shown", subject="graphic",
               speaker=None,
               scene_specifics={"subject": "Slide", "on_screen_text": "The Insight"})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert '"Slide: The Insight"' in line, line
    assert 'spec:"Slide"' in line, line
    assert "text:'" not in line, line
    print("ok  test_spec_on_screen_text_suppressed_when_it_repeats_the_primary_quote")


def test_spec_on_screen_text_still_renders_when_it_says_something_new():
    cut = _cut("f:dup4", 1000, 4000, "Slide: The Insight", channel="shown", subject="graphic",
               speaker=None,
               scene_specifics={"subject": "Slide", "on_screen_text": "footnote: source, Q3 2026"})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert '"Slide: The Insight"' in line, line
    assert "text:'footnote: source, Q3 2026'" in line, line
    print("ok  test_spec_on_screen_text_still_renders_when_it_says_something_new")


def test_full_slide_scenario_says_the_fact_exactly_once():
    """The plan's own concrete example: a slide cut whose text used to
    render 3x (primary quote, graphic:, spec:"...text:'...'"). After the
    dedup, "The Insight" appears exactly once on the line."""
    cut = _cut("f:dup5", 1000, 4000, "Slide: The Insight", channel="shown", subject="graphic",
               speaker=None, summary="The Insight",
               scene_specifics={"subject": "Slide", "on_screen_text": "The Insight"})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert line.count("The Insight") == 1, line
    print("ok  test_full_slide_scenario_says_the_fact_exactly_once")


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


# --------------------------------------------------------------------------
# brain_cut_index_fidelity.plan.md 1.2 (A5): one quality scale. Video's
# total_quality is already clamped [0,1] at write time; speech's is not
# (fluency 0-1 + a weighted delivery fusion whose own ceiling is 5.0) --
# _qual must normalize speech onto the same [0,1] scale, and the rendered
# token is `score.` now, not `q.` (which read as "quality" regardless of
# what the prompt said, which was false for a video cut).
# --------------------------------------------------------------------------

def test_qual_renders_video_score_unchanged_on_its_native_0_1_scale():
    assert fm._qual(0.42, "video") == "score.42"
    assert fm._qual(1.0, "video") == "score.100"
    assert fm._qual(0.0, "video") == "score.00"
    print("ok  test_qual_renders_video_score_unchanged_on_its_native_0_1_scale")


def test_qual_normalizes_speech_score_by_its_own_ceiling():
    """Speech's total_quality (fluency 0-1 + weighted delivery, vcut/speech/
    params.py's own weight sum) has a code-defined ceiling of 5.0, not 1.0 --
    a raw 4.6 (near the real observed max) must NOT render as score.460."""
    assert fm._qual(5.0, "speech") == "score.100"
    assert fm._qual(0.0, "speech") == "score.00"
    lo = fm._qual(4.6, "speech")
    assert lo == "score.92", lo
    print("ok  test_qual_normalizes_speech_score_by_its_own_ceiling")


def test_qual_same_score_ranks_lower_for_speech_than_video():
    """The whole point of 1.2: a raw value that would be near-maximal on
    speech's 0-5 scale must not read as near-maximal when compared (in
    magnitude only) against a video score on its native 0-1 scale."""
    raw = 2.5
    video_tok = fm._qual(raw, "video")   # already clamped -- unchanged
    speech_tok = fm._qual(raw, "speech")  # normalized down
    assert video_tok == "score.100", video_tok   # clamped, not silently >100
    assert speech_tok == "score.50", speech_tok
    print("ok  test_qual_same_score_ranks_lower_for_speech_than_video")


def test_qual_no_kind_treats_score_as_already_0_1():
    """A legacy hero-cut moment with no `kind` on it (pre-migration data)
    falls back to treating the score as already on the 0-1 scale --
    matching cutrecord_map._legacy_score_for's own [0,1] contract."""
    assert fm._qual(0.75) == "score.75"
    print("ok  test_qual_no_kind_treats_score_as_already_0_1")


def test_qual_token_prefix_is_score_not_q():
    assert fm._qual(0.5, "video").startswith("score.")
    assert "q." not in fm._qual(0.5, "video")
    print("ok  test_qual_token_prefix_is_score_not_q")


def test_pic_segment_end_to_end_renders_score_prefix_for_a_video_cut():
    cut = _cut("f:vq", 1000, 3000, "a shot", channel="shown", subject="object",
               speaker=None, score=0.61, kind="video",
               ladder=[_rung("balanced", 1000, 3000, "a shot", 0.61)])
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "PIC:object (score.61)" in line, line
    assert "q.61" not in line, line
    print("ok  test_pic_segment_end_to_end_renders_score_prefix_for_a_video_cut")


def test_pic_segment_end_to_end_normalizes_a_speech_cut_score():
    cut = _cut("f:sq", 1000, 3000, "hello there", channel="said", subject="person",
               speaker="S0", score=4.6, kind="speech",
               ladder=[_rung("balanced", 1000, 3000, "hello there", 4.6)])
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "score.92" in line, line
    assert "score.460" not in line, line
    print("ok  test_pic_segment_end_to_end_normalizes_a_speech_cut_score")


def _flat_cut():
    """A ladder whose five levels all play the identical duration -- the
    common vcut shape (`_PACE_LEVELS = [1.0]*5`, vcut/store.py) where the
    zoom-ladder tag would be constant noise, not signal."""
    return _cut("f:flat", 1000, 2000, "steady beat", score=0.6,
                ladder=[_rung(level, 1000, 2000, "steady beat", 0.6)
                        for level in fm._LEVEL_NAMES])


def test_energy_tag_renders_real_grade_from_pace():
    """brain_material_truth.plan.md Part 3: energy_grade lives on the cut's
    pace envelope and used to reach the brain never -- render it as
    `energy:GRADE` whenever it's present."""
    cut = _flat_cut()
    cut["pace"] = {"energy_grade": "high"}
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "energy:high" in line, line
    print("ok  test_energy_tag_renders_real_grade_from_pace")


def test_energy_tag_absent_without_pace_envelope():
    """A legacy/pre-migration moment with no pace envelope at all renders no
    energy tag -- there is nothing real to report, so it stays silent rather
    than fabricate a grade."""
    cut = _flat_cut()
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "energy:" not in line, line
    print("ok  test_energy_tag_absent_without_pace_envelope")


def test_levels_tag_suppressed_when_ladder_is_uniform():
    """Part 2.2 route (b): a ladder where every level plays the identical
    duration (vcut's `[1.0]*5`) carries no real information -- the old
    `nrg:` tag rendered it unconditionally on nearly every video cut; the
    replacement `levels:` tag must stay silent instead."""
    cut = _flat_cut()
    cut["pace"] = {"energy_grade": "medium"}
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "levels:" not in line, line
    assert "energy:medium" in line, line
    print("ok  test_levels_tag_suppressed_when_ladder_is_uniform")


def test_levels_tag_renders_when_ladder_genuinely_varies():
    """A real multi-piece cluster ladder (durations differ across levels)
    still gets the honest `levels:` tag -- suppression is about uninformative
    ladders, not the tag itself."""
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000},
                              [_thought_cut()])
    line = fm._moment_line(tree["moments"][0])
    assert "levels:broad|calm|balanced|tight|sharp" in line, line
    print("ok  test_levels_tag_renders_when_ladder_genuinely_varies")


def test_no_dangling_separator_when_energy_and_levels_both_absent():
    """When neither tag has anything to say, the tag section must not leave a
    stray ` * ` bullet dangling off the quoted text (the old code had a
    static separator that assumed energy_tag was always non-empty)."""
    cut = _flat_cut()
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "energy:" not in line and "levels:" not in line, line
    assert "·  ·" not in line, line
    assert not line.rstrip().endswith("·"), line
    print("ok  test_no_dangling_separator_when_energy_and_levels_both_absent")


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


def _sent(a, b, text="x", spk="S0"):
    return {"speaker": spk, "text": text, "src_in_ms": a, "src_out_ms": b}


def test_snap_merges_a_midsentence_seam():
    """A dead-air jump-cut seam that falls INSIDE a sentence is merged away, so
    the sentence stays contiguous and can't be split/removed (the m02 class)."""
    sents = (_sent(0, 3000, "asked that question."),
             _sent(3000, 10000, "so what was the biggest challenge when you joined?"))
    with mock.patch.object(fm, "_sentences_for_file", return_value=sents):
        out = fm.snap_speech_spans_to_sentences("f", [(0, 5000), (5060, 10000)])
    assert out == [(0, 10000)], out   # 60ms mid-sentence gap swallowed
    print("ok  snap merges a mid-sentence seam")


def test_snap_keeps_a_between_sentence_seam():
    """A seam whose edges land on sentence boundaries is a legit jump-cut and is
    preserved (we only remove SEAMS that would sever a sentence)."""
    sents = (_sent(0, 3000, "first sentence."), _sent(4000, 8000, "second sentence."))
    with mock.patch.object(fm, "_sentences_for_file", return_value=sents):
        out = fm.snap_speech_spans_to_sentences("f", [(0, 3000), (4000, 8000)])
    assert out == [(0, 3000), (4000, 8000)], out
    print("ok  snap keeps a between-sentence seam")


def test_snap_drops_a_partial_tail_sliver_at_the_head():
    """A head that only catches the tail sliver of a prior sentence snaps forward
    to the next sentence start (the 'starts on think' class)."""
    sents = (_sent(0, 5000, "...I think"), _sent(5000, 9000, "we look forward to more."))
    with mock.patch.object(fm, "_sentences_for_file", return_value=sents):
        out = fm.snap_speech_spans_to_sentences("f", [(4800, 9000)])
    assert out == [(5000, 9000)], out
    print("ok  snap drops a partial tail sliver at the head")


def test_snap_keeps_a_mostly_present_head_sentence():
    """A head that already holds MOST of its sentence keeps it -- never amputate a
    near-whole opening line (the m01 non-regression)."""
    sents = (_sent(4000, 7000, "it's been an incredible journey,"),
             _sent(7000, 12000, "how has the journey been?"))
    with mock.patch.object(fm, "_sentences_for_file", return_value=sents):
        out = fm.snap_speech_spans_to_sentences("f", [(4200, 12000)])
    assert out == [(4200, 12000)], out   # 200ms into a 3s sentence -> kept
    print("ok  snap keeps a mostly-present head sentence")


def test_snap_drops_a_dangling_tail_sentence():
    """A tail that only catches the head sliver of a sentence snaps back to the
    previous sentence end (the 'ends on or,' class)."""
    sents = (_sent(0, 4000, "great projects."), _sent(4000, 9000, "whether it is X, or ..."))
    with mock.patch.object(fm, "_sentences_for_file", return_value=sents):
        out = fm.snap_speech_spans_to_sentences("f", [(0, 4600)])
    assert out == [(0, 4000)], out
    print("ok  snap drops a dangling tail sentence")


def test_snap_is_noop_without_transcript():
    with mock.patch.object(fm, "_sentences_for_file", return_value=()):
        spans = [(100, 200), (300, 400)]
        assert fm.snap_speech_spans_to_sentences("f", spans) == spans
    print("ok  snap is a no-op without transcript")


# --------------------------------------------------------------------------
# v4_cluster_read_act.plan.md Part B: multi-event cluster piece breakdown
# --------------------------------------------------------------------------

def _cluster_events():
    # Same fixture as test_cutrecord_map.py's _cluster_events: three widely
    # separated point events inside a 0-10000ms cluster, so resolve_cluster
    # at energy=1.0 keeps them as three SEPARATE pieces (no merge).
    return [
        {"peak_ms": 1000, "score": 0.6, "kind": "point", "onset_ms": 700, "settle_ms": 1500, "span_ms": None},
        {"peak_ms": 5000, "score": 1.0, "kind": "point", "onset_ms": 4600, "settle_ms": 5500, "span_ms": None},
        {"peak_ms": 9000, "score": 0.4, "kind": "point", "onset_ms": 8700, "settle_ms": 9400, "span_ms": None},
    ]


def _cluster_moment(hero_id="f:cl", in_ms=0, out_ms=10000, events=None, primary=1):
    events = _cluster_events() if events is None else events
    cut = _salience_cut(hero_id, in_ms, out_ms,
                        {"peak_ms": 5000, "score": 1.0, "kind": "point", "span_ms": None,
                         "events": events, "primary": primary, "density": 0.5, "shape": "center"})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": out_ms + 4000}, [cut])
    return tree["moments"][0]


def test_piece_breakdown_none_for_single_event_moment():
    cut = _salience_cut("f:pk", 1000, 4000, {"peak_ms": 2500, "score": 0.9})
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    assert fm.piece_breakdown(m) is None
    assert fm._piece_lines(m) == []
    assert fm.resolve_piece(m, 1) is None
    print("ok  test_piece_breakdown_none_for_single_event_moment")


def test_piece_breakdown_shape_for_three_separated_events():
    m = _cluster_moment()
    b = fm.piece_breakdown(m)
    assert b["broad_s"] == 10.0, b
    assert b["tight_count"] == 3, b
    assert b["tight_avg_s"] == 0.8, b
    assert [p["pos"] for p in b["pieces"]] == [1, 2, 3], b
    assert all(p["of"] == 3 for p in b["pieces"]), b
    assert [p["strength"] for p in b["pieces"]] == ["moderate", "strongest", "weak"], b
    assert [p["dur_s"] for p in b["pieces"]] == [0.8, 0.9, 0.7], b
    assert all(p["kind"] == "point" for p in b["pieces"]), b
    # The RENDERED filtering: at the sharpest band (energy 0.9) the salience
    # gate drops the two weak events (0.6, 0.4) and keeps only the strongest
    # (1.0) -- so punchy_count is 1 and only the strongest piece is `core`.
    assert b["punchy_count"] == 1, b
    assert b["punchy_avg_s"] > 0.0, b
    assert [p["core"] for p in b["pieces"]] == [False, True, False], b
    print("ok  test_piece_breakdown_shape_for_three_separated_events")


def test_piece_breakdown_all_strong_events_are_all_core():
    """When every event is genuinely comparable (none weak enough to gate out),
    all survive the sharp band -> every piece is core, punchy_count == of."""
    m = _cluster_moment(events=[
        {"peak_ms": 1500, "score": 0.95, "kind": "point", "onset_ms": 1100, "settle_ms": 2000, "span_ms": None},
        {"peak_ms": 5000, "score": 1.0, "kind": "point", "onset_ms": 4600, "settle_ms": 5500, "span_ms": None},
        {"peak_ms": 8500, "score": 0.9, "kind": "point", "onset_ms": 8100, "settle_ms": 9000, "span_ms": None},
    ], primary=1)
    b = fm.piece_breakdown(m)
    assert b["punchy_count"] == 3, b
    assert all(p["core"] for p in b["pieces"]), b
    print("ok  test_piece_breakdown_all_strong_events_are_all_core")


# --------------------------------------------------------------------------
# brain_cut_index_fidelity.plan.md 2.1 (A1): each piece carries its OWN
# event's summary, confirmed genuinely distinct on real multi-event cuts
# (2026-08-25 live-DB check: 280/280 real multi-event video cuts had zero
# duplicate/empty per-event summaries) -- never a broadcast copy, never
# fabricated when absent.
# --------------------------------------------------------------------------

def _cluster_events_with_summaries():
    events = _cluster_events()
    texts = ["the user grabs the mug off the counter",
             "the user pours water into the mug",
             "the user carries the mug out of frame"]
    for ev, t in zip(events, texts):
        ev["summary"] = t
    return events


def test_piece_breakdown_carries_each_events_own_distinct_summary():
    m = _cluster_moment(events=_cluster_events_with_summaries())
    b = fm.piece_breakdown(m)
    summaries = [p["summary"] for p in b["pieces"]]
    assert summaries == [
        "the user grabs the mug off the counter",
        "the user pours water into the mug",
        "the user carries the mug out of frame",
    ], summaries
    assert len(set(summaries)) == 3, summaries   # genuinely distinct, not a broadcast copy
    print("ok  test_piece_breakdown_carries_each_events_own_distinct_summary")


def test_piece_lines_quotes_each_events_own_text():
    """Piece text is kept short (_short_gist, matching the other secondary
    tags) -- Stage 3's full-description upgrade is for the cut's own PRIMARY
    quote, not the piece list."""
    m = _cluster_moment(events=_cluster_events_with_summaries())
    lines = fm._piece_lines(m)
    assert '"the user grabs the mug off…"' in lines[1], lines[1]
    assert '"the user pours water into the…"' in lines[2], lines[2]
    assert '"the user carries the mug out…"' in lines[3], lines[3]
    print("ok  test_piece_lines_quotes_each_events_own_text")


def test_piece_lines_render_nothing_extra_when_summary_absent():
    """A legacy/un-re-ingested cut whose events carry no descriptive payload
    (predates the salience bridge) must render the piece rows exactly as
    before -- no fabricated placeholder text."""
    m = _cluster_moment()   # _cluster_events(): no "summary" key at all
    lines = fm._piece_lines(m)
    assert '"' not in lines[1], lines[1]
    assert lines[1] == "    1/3 moderate 0.8s point · drops when tight", lines[1]
    print("ok  test_piece_lines_render_nothing_extra_when_summary_absent")


def test_moment_line_multi_event_cluster_is_multiline_and_generic():
    m = _cluster_moment()
    line = fm._moment_line(m)
    assert "\n" in line, line
    # The range line now makes the filtering explicit: plays whole, splits into
    # N addressable beats, or tightens down to the strongest few (punchy).
    assert "plays whole ~10.0s" in line, line
    assert "splits into its 3 pieces" in line, line
    assert "down to ~1" in line, line
    # Per-piece rows carry the core/drops marking so the brain sees which beats
    # survive tightening and which fall away first.
    assert "1/3 moderate 0.8s point \u00b7 drops when tight" in line, line
    assert "2/3 strongest 0.9s point \u00b7 core" in line, line
    assert "3/3 weak 0.7s point \u00b7 drops when tight" in line, line
    # Generic/no-domain-words requirement (plan Part B): no domain vocabulary
    # leaks into the piece rows.
    for banned in ("speaker", "action", "highlight", "rally", "tennis", "podcast"):
        assert banned not in line.lower(), line
    print("ok  test_moment_line_multi_event_cluster_is_multiline_and_generic")


def test_resolve_piece_addresses_each_separated_event():
    m = _cluster_moment()
    # UNGATED addressing (prune=False): every event stays individually
    # addressable at its own tightest window, even the weak one the rendered
    # dial would gate out.
    assert fm.resolve_piece(m, 1) == (700, 1500)
    assert fm.resolve_piece(m, 2) == (4700, 5500)
    assert fm.resolve_piece(m, 3) == (8700, 9500)
    print("ok  test_resolve_piece_addresses_each_separated_event")


def test_resolve_piece_out_of_range_is_none():
    m = _cluster_moment()
    assert fm.resolve_piece(m, 0) is None
    assert fm.resolve_piece(m, 4) is None
    print("ok  test_resolve_piece_out_of_range_is_none")


def test_resolve_piece_merged_events_share_one_span():
    # Two events only 300ms apart -- even at energy=1.0 their tight windows
    # (peak-300..peak+500) still overlap, so resolve_cluster fuses them into
    # ONE piece. Both piece positions must resolve to that SAME merged span,
    # not a naive (and wrong) per-event slice.
    events = [
        {"peak_ms": 5000, "score": 0.7, "kind": "point", "onset_ms": 4700, "settle_ms": 5300, "span_ms": None},
        {"peak_ms": 5300, "score": 0.9, "kind": "point", "onset_ms": 5000, "settle_ms": 5600, "span_ms": None},
    ]
    m = _cluster_moment(hero_id="f:mg", in_ms=4000, out_ms=6000, events=events, primary=1)
    merged = (4700, 5800)
    assert fm.resolve_piece(m, 1) == merged
    assert fm.resolve_piece(m, 2) == merged
    print("ok  test_resolve_piece_merged_events_share_one_span")


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
    test_action_beat_surfaces_incidental_said_text_brain_mirror_readside()
    test_action_beat_said_text_empty_when_nothing_overlaps()
    test_said_beat_transcript_truncates_in_compact_mode()
    test_said_beat_with_no_transcript_falls_back_to_visual_gist()
    test_span_detail_pads_and_filters_via_shared_sentences_cache()
    test_alt_pic_absent_without_co_occurrence()
    test_done_beat_pic_first_parity_and_snd_silence()
    test_speaker_person_renders_on_cam_when_shown()
    test_annotate_dups_reads_take_group_id()
    test_annotate_shared_audio_links_overlapping_said_and_shown_same_file()
    test_moment_line_renders_dup_audio_cross_reference_both_sides()
    test_annotate_shared_audio_no_link_without_temporal_overlap()
    test_annotate_shared_audio_no_link_when_shown_cut_carries_no_words()
    test_annotate_shared_audio_no_link_across_different_files()
    test_annotate_shared_audio_excludes_junk_on_either_side()
    test_snd_state_says_muted_talk_is_recoverable()
    test_snd_state_ambient_mute_stays_terse()
    test_junk_moment_renders_terse_line_not_the_rich_one()
    test_junk_moment_renders_a_short_description()
    test_junk_moment_renders_nothing_extra_when_no_description()
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
    test_landmarks_tag_fixed_channel_order_regardless_of_dict_order()
    test_landmarks_tag_renders_offsets_via_moment_line()
    test_landmarks_tag_adx_shows_direction()
    test_landmarks_tag_sil_shows_offset_and_duration()
    test_landmarks_tag_shot_marks_hard_cuts_not_soft_ones()
    test_landmarks_tag_caps_come_from_ingest_not_the_renderer()
    test_landmarks_tag_absent_when_landmarks_empty()
    test_landmarks_tag_skips_a_channel_with_no_stored_offsets()
    test_specific_tag_renders_when_scene_specifics_present()
    test_specific_tag_absent_when_not_yet_enriched()
    test_specific_tag_absent_when_specifics_present_but_empty_string()
    test_specific_tag_new_shape_absent_when_scene_specifics_missing_entirely()
    test_specific_tag_new_shape_absent_when_scene_specifics_is_empty_dict()
    test_specific_tag_new_single_flag_renders_subject_action_shot_size_camera()
    test_specific_tag_new_single_flag_subject_only_no_dangling_space()
    test_specific_tag_new_shape_omits_static_camera_move_and_none_motion_and_good_headroom()
    test_specific_tag_new_shape_renders_on_screen_text_gisted_and_quoted()
    test_specific_tag_new_shape_renders_usable_hook_and_capped_tags()
    test_specific_tag_dedupes_camera_move_against_code_derived_cam()
    test_specific_tag_dedupes_on_screen_text_against_code_derived_screen_text()
    test_specific_tag_camera_move_survives_when_it_does_not_match_code_cam()
    test_specific_tag_new_merged_renders_mini_shot_list_with_deltas()
    test_specific_tag_new_merged_deltas_are_relative_to_the_cuts_own_in_ms()
    test_specific_tag_new_merged_caps_moments_list_with_more_suffix()
    test_specific_tag_new_merged_moment_falls_back_to_subject_action_with_no_summary()
    test_specific_tag_compact_mode_drops_moments_list_and_secondary_tokens()
    test_specific_tag_compact_mode_legacy_shape_unaffected()
    test_specific_tag_full_pipeline_new_shape_renders_on_the_beat_line()
    test_graphic_tag_suppressed_when_it_repeats_the_primary_quote()
    test_graphic_tag_still_renders_when_it_says_something_new()
    test_graphic_tag_renders_full_summary_when_primary_is_just_its_prefix()
    test_graphic_tag_not_truncated_to_six_words()
    test_graphic_tag_still_suppressed_when_summary_equals_primary_exactly()
    test_spec_on_screen_text_suppressed_when_it_repeats_the_primary_quote()
    test_spec_on_screen_text_still_renders_when_it_says_something_new()
    test_full_slide_scenario_says_the_fact_exactly_once()
    test_source_contiguous_beats_form_a_run_channel_agnostic()
    test_clip_block_groups_run_members_under_a_continuity_header()
    test_clip_block_run_indent_composes_with_takes_lines()
    test_cast_line_lists_majors_with_voices_and_others_by_id()
    test_cast_line_empty_with_no_persons()
    test_qual_renders_video_score_unchanged_on_its_native_0_1_scale()
    test_qual_normalizes_speech_score_by_its_own_ceiling()
    test_qual_same_score_ranks_lower_for_speech_than_video()
    test_qual_no_kind_treats_score_as_already_0_1()
    test_qual_token_prefix_is_score_not_q()
    test_pic_segment_end_to_end_renders_score_prefix_for_a_video_cut()
    test_pic_segment_end_to_end_normalizes_a_speech_cut_score()
    test_energy_tag_renders_real_grade_from_pace()
    test_energy_tag_absent_without_pace_envelope()
    test_levels_tag_suppressed_when_ladder_is_uniform()
    test_levels_tag_renders_when_ladder_genuinely_varies()
    test_no_dangling_separator_when_energy_and_levels_both_absent()
    test_default_energy_from_genre()
    test_snap_merges_a_midsentence_seam()
    test_snap_keeps_a_between_sentence_seam()
    test_snap_drops_a_partial_tail_sliver_at_the_head()
    test_snap_keeps_a_mostly_present_head_sentence()
    test_snap_drops_a_dangling_tail_sentence()
    test_snap_is_noop_without_transcript()
    test_piece_breakdown_none_for_single_event_moment()
    test_piece_breakdown_shape_for_three_separated_events()
    test_piece_breakdown_all_strong_events_are_all_core()
    test_piece_breakdown_carries_each_events_own_distinct_summary()
    test_piece_lines_quotes_each_events_own_text()
    test_piece_lines_render_nothing_extra_when_summary_absent()
    test_moment_line_multi_event_cluster_is_multiline_and_generic()
    test_resolve_piece_addresses_each_separated_event()
    test_resolve_piece_out_of_range_is_none()
    test_resolve_piece_merged_events_share_one_span()
    print("\nall footage-map tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
