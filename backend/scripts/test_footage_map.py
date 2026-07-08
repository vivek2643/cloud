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

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import footage_map as fm  # noqa: E402


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
    (an off-screen interviewer / voiceover) renders an honest PIC:? -- never a
    fabricated face -- while SND still names who is heard."""
    cut = _cut("f:mo", 1000, 4000, "what made you start this", channel="said",
               subject="person", speaker="interviewer", score=0.7,
               ladder=[_rung("balanced", 1000, 4000, "what made you start this", 0.7)],
               flags=["offscreen"],
               people=[{"voice_speaker_id": "S9", "person_id": None, "on_camera": False}])
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "said.person" in line, line
    assert "PIC:?" in line, line
    assert "SND:interviewer speaking" in line, line
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
    """With the shoot cast (oncam map) supplied, PIC reports WHOSE FACE shows in
    the clip -- from the reconciled cast, right even when the per-moment flag
    disagrees (a clip whose A/V link was wrong) -- and SND names who's heard.
    PIC leads the line; SND trails it."""
    # Speaker G2 (voice S1) talks, but this clip SHOWS G1's face (off-cam speaker).
    cut = _cut("48c93cef:m03", 1000, 4000, "and then we shipped it", channel="said",
               subject="person", speaker="S1", score=0.6,
               ladder=[_rung("balanced", 1000, 4000, "and then we shipped it", 0.6)],
               people=[{"voice_speaker_id": "S1", "person_id": None, "on_camera": True}])
    tree = fm.build_clip_tree("48c93cef-aaa", {"name": "T", "duration_ms": 8000}, [cut])
    alias = {("48c93cef-aaa", "S1"): "G2"}
    oncam = {"48c93cef-aaa": "G1"}                       # this clip shows G1's face
    line = fm._moment_line(tree["moments"][0], alias=alias, oncam=oncam)
    assert "PIC:G1" in line, line              # reconciled shown-face overrides the flag
    assert "SND:G2 speaking" in line, line
    assert line.index("PIC:") < line.index("SND:"), line  # picture leads, never the speaker
    print("ok  test_reconciled_shows_face_and_cam_override")


def test_said_beat_on_listener_camera_reads_pic_first_with_alt_pic():
    """The exact podcast-bug shape: a said beat delivered on the LISTENER's
    camera renders PIC (who's shown) before SND (who's heard), so placing it can
    no longer be mistaken for showing the speaker -- and `alt-PIC` points at the
    camera that DOES show the speaker for the same words (Fact #2 folded onto
    the beat, not buried in a distant coverage block)."""
    cut = _cut("48c93cef:m24", 1000, 4000, "that freedom he gave you", channel="said",
               subject="person", speaker="S2", score=0.6,
               ladder=[_rung("balanced", 1000, 4000, "that freedom he gave you", 0.6)],
               people=[{"voice_speaker_id": "S2", "person_id": None, "on_camera": True}],
               framing="med")
    tree = fm.build_clip_tree("48c93cef-aaa", {"name": "Cam A", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    # Folded on by `_annotate_dups` in the real path; set directly here to test
    # the render in isolation (mirrors how `_dups_block` fixtures worked before).
    m["alt_pic"] = [{"moment_id": "1aedb093:m13", "file": "1aedb093-bbb", "voice": "S9",
                     "framing": "med", "score": 0.73, "restart": False}]
    alias = {("48c93cef-aaa", "S2"): "G2", ("1aedb093-bbb", "S9"): "G2"}
    oncam = {"48c93cef-aaa": "G1", "1aedb093-bbb": "G2"}   # cam A shows the LISTENER G1
    line = fm._moment_line(m, alias=alias, oncam=oncam)
    assert "PIC:G1" in line, line
    assert "SND:G2 speaking" in line, line
    assert line.index("PIC:") < line.index("SND:"), line
    assert "·alt-PIC:G2→1aedb093:m13" in line, line
    print("ok  test_said_beat_on_listener_camera_reads_pic_first_with_alt_pic")


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
               speaker=None, score=0.9, audio="silent", mute=False,
               ladder=[_rung("balanced", 1000, 3000, "nods, reacts", 0.9)],
               people=[{"person_id": "G1", "on_camera": True}], framing="med")
    tree = fm.build_clip_tree("1e529bed-aaa", {"name": "T", "duration_ms": 8000}, [cut])
    line = fm._moment_line(tree["moments"][0])
    assert "PIC:G1" in line, line
    assert "SND:silence" in line, line
    assert line.index("PIC:") < line.index("SND:"), line
    print("ok  test_done_beat_pic_first_parity_and_snd_silence")


def test_speaker_resolves_from_raw_handle_not_label():
    """The speaker id gap: a said cut's `speaker` is a human LABEL ('main subject')
    the registry can't match; the raw voice id lives in `people`. Resolution must
    key on the RAW handle so every line names its shoot-wide person -- else every
    speech line falls back to a role label and the brain can't match speaker to
    camera. Without a registry it still reads the label, never the bare id."""
    cut = _cut("48c93cef:m00", 1000, 4000, "we shipped it", speaker="main subject",
               score=0.7, ladder=[_rung("balanced", 1000, 4000, "we shipped it", 0.7)],
               people=[{"voice_speaker_id": "S0", "person_id": "p1", "on_camera": True}])
    tree = fm.build_clip_tree("48c93cef-aaa", {"name": "T", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    line = fm._moment_line(m, alias={("48c93cef-aaa", "S0"): "G2"})
    assert "SND:G2 speaking" in line, line        # resolved off the raw voice handle
    assert "main subject" not in line, line       # not the unresolvable label
    plain = fm._moment_line(m)                     # no registry -> readable label
    assert "SND:main subject speaking" in plain, plain
    print("ok  test_speaker_resolves_from_raw_handle_not_label")


class _FakeSettings:
    def __init__(self, footage_source: str):
        self.footage_source = footage_source


def test_annotate_dups_folds_alt_pic_onto_each_beat():
    """`_annotate_dups` folds Fact #2 onto each linked beat as `alt_pic` -- every
    OTHER member's raw facts, in place, no separate lookup table -- so
    `_moment_line` can render a beat's alternates on its own. Exercises the
    LEGACY hero path (token-overlap take grouping via `takes.build_take_groups`);
    the cut_records-mode path is covered by
    test_annotate_dups_cut_records_mode_reads_take_group_id below."""
    from app.services.l3 import takes

    cut_a = _cut("48c93cef:mA", 1000, 4000, "that freedom he gave you", channel="said",
                speaker="S2", score=0.6,
                ladder=[_rung("balanced", 1000, 4000, "that freedom he gave you", 0.6)])
    cut_b = _cut("1aedb093:mB", 1000, 4000, "that freedom he gave you", channel="said",
                speaker="S9", score=0.73,
                ladder=[_rung("balanced", 1000, 4000, "that freedom he gave you", 0.73)])
    tree_a = fm.build_clip_tree("48c93cef-aaa", {"name": "A", "duration_ms": 8000}, [cut_a])
    tree_b = fm.build_clip_tree("1aedb093-bbb", {"name": "B", "duration_ms": 8000}, [cut_b])

    group = takes.TakeGroup(group_id="tg1", content_key="k", attempts=[
        takes.Attempt("a1", "48c93cef-aaa", "u1", 1000, 4000, "said", "k",
                      "that freedom he gave you", speaker="S2"),
        takes.Attempt("a2", "1aedb093-bbb", "u1", 1000, 4000, "said", "k",
                      "that freedom he gave you", speaker="S9"),
    ])
    orig_build = takes.build_take_groups
    orig_settings = fm.get_settings
    takes.build_take_groups = lambda file_ids: [group]
    fm.get_settings = lambda: _FakeSettings("hero")
    try:
        summary = fm._annotate_dups([tree_a, tree_b])
    finally:
        takes.build_take_groups = orig_build
        fm.get_settings = orig_settings

    assert len(summary) == 1, summary
    ma, mb = tree_a["moments"][0], tree_b["moments"][0]
    assert ma["alt_pic"][0]["moment_id"] == mb["moment_id"], ma["alt_pic"]
    assert mb["alt_pic"][0]["moment_id"] == ma["moment_id"], mb["alt_pic"]
    print("ok  test_annotate_dups_folds_alt_pic_onto_each_beat")


def test_annotate_dups_cut_records_mode_reads_take_group_id():
    """cut_records mode (the default) reads dup_groups DIRECTLY off each
    moment's persisted `take_group_id`/`take_role` -- no token-overlap
    recompute, no IoU matching (see cuts_v3_to_brain.plan.md Phase 3)."""
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

    assert fm.get_settings().footage_source == "cut_records"   # exercise the real default
    summary = fm._annotate_dups([tree_a, tree_b])

    assert len(summary) == 1, summary
    g = summary[0]
    assert g["group_id"] == "tg1", g
    ma, mb = tree_a["moments"][0], tree_b["moments"][0]
    assert ma["dup_group"] == "tg1" and mb["dup_group"] == "tg1"
    assert ma["alt_pic"][0]["moment_id"] == mb["moment_id"], ma["alt_pic"]
    assert mb["alt_pic"][0]["moment_id"] == ma["moment_id"], mb["alt_pic"]
    assert {mf["take_role"] for mf in g["member_facts"]} == {"winner", "take"}, g["member_facts"]
    print("ok  test_annotate_dups_cut_records_mode_reads_take_group_id")


def test_moment_line_aliases_global_speaker():
    """A per-line speaker is shown as its global person id when the registry
    linked it; without the alias it falls back to the raw voice."""
    cut = _cut("f:g", 1000, 4000, "hello there", speaker="S0", score=0.6,
               ladder=[_rung("balanced", 1000, 4000, "hello there", 0.6)])
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [cut])
    m = tree["moments"][0]
    assert "SND:G3 speaking" in fm._moment_line(m, alias={("ffffffff-1111", "S0"): "G3"})
    assert "SND:S0 speaking" in fm._moment_line(m)         # no alias -> raw voice
    print("ok  test_moment_line_aliases_global_speaker")


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
    test_alt_pic_absent_without_co_occurrence()
    test_done_beat_pic_first_parity_and_snd_silence()
    test_speaker_resolves_from_raw_handle_not_label()
    test_annotate_dups_folds_alt_pic_onto_each_beat()
    test_annotate_dups_cut_records_mode_reads_take_group_id()
    test_moment_line_aliases_global_speaker()
    test_source_contiguous_beats_form_a_run_channel_agnostic()
    test_default_energy_from_genre()
    print("\nall footage-map tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
