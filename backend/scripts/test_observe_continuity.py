#!/usr/bin/env python3
"""Tests for brain_mirror_readside.plan.md's observe-side surfacing --
read_state's per-cut join/join_note, diagnose's jump-cut finding, review's
"continuity" category flag, and affordances' `runs` section. Pure, no DB/LLM
(same convention as test_observe_act.py).

Run:  PYTHONPATH=. python scripts/test_observe_continuity.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import cutrecord_map as cm, footage_map as fm, observe  # noqa: E402
from app.services.l3.arrange import _MapIndex  # noqa: E402

fm._sentences_for_file = lambda file_id: ()


def _cut(hero_id, in_ms, out_ms, label=""):
    return {"hero_id": hero_id, "file_id": "ffffffff-1111", "modality": "any",
            "channel": "shown", "label": label, "src_in_ms": in_ms, "src_out_ms": out_ms,
            "play_ms": out_ms - in_ms, "keep_spans": None, "score": 0.6,
            "speaker": None, "affordances": ["any"], "flags": [], "ladder": None}


def _struct(cuts):
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 20000}, cuts)
    return {"clips": [tree]}


def _ctx(struct, descriptors=None):
    return observe.EditContext(
        file_ids=["ffffffff-1111"], index=_MapIndex(struct), map_struct=struct,
        durations={"ffffffff-1111": 20000}, dup_groups=[],
        frame_descriptors=descriptors or {})


def _seg(seg_id, ref, in_ms, out_ms):
    return {"seg_id": seg_id, "ref": ref, "file_id": "ffffffff-1111",
           "in_ms": in_ms, "out_ms": out_ms, "content": "", "axis": "any"}


def _h(n):
    """A 64-bit int with exactly n low bits set -- hamming64(0, _h(n)) == n."""
    return (1 << n) - 1 if n > 0 else 0


# --------------------------------------------------------------------------
# read_state: per-cut join/join_note (section 3.2)
# --------------------------------------------------------------------------

def test_read_state_flags_a_jump_cut():
    struct = _struct([_cut("f:t0", 0, 1000), _cut("f:t1", 5000, 6000)])
    # Placed adjacent on the timeline in a source-non-contiguous order
    # (gap 4000ms > _JOIN_CONTIG_MS), same continuous framing (small
    # Hamming distance) -- a jump-cut.
    descriptors = {"ffffffff-1111": [(1000, _h(0)), (5000, _h(5))]}
    ctx = _ctx(struct, descriptors)
    doc = {"timeline": [_seg("s0", "ffffffff:m00", 0, 1000), _seg("s1", "ffffffff:m01", 5000, 6000)],
          "operations": []}
    st = observe.read_state(doc, ctx)
    assert "join" not in st["cuts"][0]                 # pos==1 has no seam INTO it
    assert st["cuts"][1]["join"] == "jump-cut", st["cuts"][1]
    assert st["cuts"][1]["join_pic_bits"] == 5, st["cuts"][1]
    assert "backward" not in st["cuts"][1]["join_note"] and "skips 4s" in st["cuts"][1]["join_note"], \
        st["cuts"][1]
    print("ok  read_state: a same-clip non-contiguous, pixel-similar seam is flagged join=jump-cut")


def test_read_state_flags_a_seamless_join():
    struct = _struct([_cut("f:t0", 0, 1000), _cut("f:t1", 1300, 2300)])
    ctx = _ctx(struct)
    # gap = 300ms -- within _JOIN_CONTIG_MS (500) -> seamless.
    doc = {"timeline": [_seg("s0", "ffffffff:m00", 0, 1000), _seg("s1", "ffffffff:m01", 1300, 2300)],
          "operations": []}
    st = observe.read_state(doc, ctx)
    assert st["cuts"][1]["join"] == "seamless", st["cuts"][1]
    assert "join_note" not in st["cuts"][1]
    print("ok  read_state: a same-clip contiguous seam is flagged join=seamless")


def test_read_state_no_join_field_for_a_normal_cross_clip_cut():
    c0 = _cut("f:t0", 0, 1000)
    struct = {"clips": [
        fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 4000}, [c0]),
        fm.build_clip_tree("22222222-2222", {"name": "T2", "duration_ms": 4000},
                           [{"hero_id": "f:t1", "file_id": "22222222-2222", "modality": "any",
                             "channel": "shown", "label": "", "src_in_ms": 0, "src_out_ms": 1000,
                             "play_ms": 1000, "keep_spans": None, "score": 0.6, "speaker": None,
                             "affordances": ["any"], "flags": [], "ladder": None}]),
    ]}
    ctx = observe.EditContext(
        file_ids=["ffffffff-1111", "22222222-2222"], index=_MapIndex(struct), map_struct=struct,
        durations={"ffffffff-1111": 4000, "22222222-2222": 4000}, dup_groups=[])
    doc = {"timeline": [
        {"seg_id": "s0", "ref": "ffffffff:m00", "file_id": "ffffffff-1111",
         "in_ms": 0, "out_ms": 1000, "content": "", "axis": "any"},
        {"seg_id": "s1", "ref": "22222222:m00", "file_id": "22222222-2222",
         "in_ms": 0, "out_ms": 1000, "content": "", "axis": "any"},
    ], "operations": []}
    st = observe.read_state(doc, ctx)
    assert "join" not in st["cuts"][1], st["cuts"][1]
    print("ok  read_state: a normal cross-clip cut carries no join field")


# --------------------------------------------------------------------------
# diagnose / review (section 3.2)
# --------------------------------------------------------------------------

def test_diagnose_emits_a_jump_cut_warning():
    struct = _struct([_cut("f:t0", 0, 1000), _cut("f:t1", 5000, 6000)])
    descriptors = {"ffffffff-1111": [(1000, _h(0)), (5000, _h(5))]}
    ctx = _ctx(struct, descriptors)
    doc = {"timeline": [_seg("s0", "ffffffff:m00", 0, 1000), _seg("s1", "ffffffff:m01", 5000, 6000)],
          "operations": []}
    findings = observe.diagnose(doc, ctx)
    jumps = [f for f in findings if "jump-cut" in f["message"]]
    assert len(jumps) == 1 and jumps[0]["severity"] == "warn" and jumps[0]["anchor"] == "cuts 1-2", jumps
    print("ok  diagnose: emits a jump-cut warning anchored at the seam")


def test_diagnose_does_not_flag_a_genuine_content_change_same_file():
    # Same file, non-contiguous, but the PICTURE genuinely changed (large
    # Hamming) -- must NOT read as a jump-cut (the false-positive flood
    # this plan retires).
    struct = _struct([_cut("f:t0", 0, 1000), _cut("f:t1", 5000, 6000)])
    descriptors = {"ffffffff-1111": [(1000, _h(0)), (5000, _h(40))]}
    ctx = _ctx(struct, descriptors)
    doc = {"timeline": [_seg("s0", "ffffffff:m00", 0, 1000), _seg("s1", "ffffffff:m01", 5000, 6000)],
          "operations": []}
    findings = observe.diagnose(doc, ctx)
    assert not any("jump-cut" in f["message"] for f in findings), findings
    print("ok  diagnose: a same-file seam with genuinely different pixels is never flagged")


def test_review_flags_the_jump_cut_under_the_continuity_category():
    struct = _struct([_cut("f:t0", 0, 1000), _cut("f:t1", 5000, 6000)])
    descriptors = {"ffffffff-1111": [(1000, _h(0)), (5000, _h(5))]}
    ctx = _ctx(struct, descriptors)
    doc = {"timeline": [_seg("s0", "ffffffff:m00", 0, 1000), _seg("s1", "ffffffff:m01", 5000, 6000)],
          "operations": []}
    out = observe.review(doc, ctx)
    continuity_flags = [f for f in out["flags"] if f.get("category") == "continuity"]
    assert len(continuity_flags) == 1 and "jump-cut" in continuity_flags[0]["message"], out["flags"]
    print("ok  review: jump-cut surfaces as a category=continuity flag")


def test_diagnose_and_review_silent_when_seamless():
    struct = _struct([_cut("f:t0", 0, 1000), _cut("f:t1", 1300, 2300)])
    ctx = _ctx(struct)   # no descriptors needed -- contiguous fast-path
    doc = {"timeline": [_seg("s0", "ffffffff:m00", 0, 1000), _seg("s1", "ffffffff:m01", 1300, 2300)],
          "operations": []}
    findings = observe.diagnose(doc, ctx)
    assert not any("jump-cut" in f["message"] for f in findings), findings
    out = observe.review(doc, ctx)
    assert not any(f.get("category") == "continuity" for f in out["flags"]), out["flags"]
    print("ok  diagnose/review: a seamless join never raises a continuity flag")


# --------------------------------------------------------------------------
# mirror / render_mirror / mirror_text: the always-present push (section 4.2)
# --------------------------------------------------------------------------

def test_mirror_empty_timeline_yields_no_segments_and_no_text():
    struct = _struct([_cut("f:t0", 0, 1000)])
    ctx = _ctx(struct)
    empty_doc = {"timeline": [], "operations": []}
    assert observe.mirror(empty_doc, ctx) == {"segments": []}
    assert observe.mirror_text(empty_doc, ctx) == ""
    print("ok  mirror: empty timeline yields no segments and no text")


def test_mirror_shows_jump_cut_join_and_flag():
    struct = _struct([_cut("f:t0", 0, 1000), _cut("f:t1", 5000, 6000)])
    descriptors = {"ffffffff-1111": [(1000, _h(0)), (5000, _h(5))]}
    ctx = _ctx(struct, descriptors)
    doc = {"timeline": [_seg("s0", "ffffffff:m00", 0, 1000), _seg("s1", "ffffffff:m01", 5000, 6000)],
          "operations": []}
    m = observe.mirror(doc, ctx)
    assert len(m["segments"]) == 2
    assert "join" not in m["segments"][0]              # pos==1 has no seam INTO it
    row = m["segments"][1]
    assert row["join"] == "jump-cut" and row["join_pic_bits"] == 5, row
    assert row.get("flags") == ["jump-cut"], row
    print("ok  mirror: a jump-cut seam surfaces join + flag")


def test_mirror_shows_seamless_join_with_no_flag():
    struct = _struct([_cut("f:t0", 0, 1000), _cut("f:t1", 1300, 2300)])
    ctx = _ctx(struct)   # contiguous fast-path -- no descriptors needed
    doc = {"timeline": [_seg("s0", "ffffffff:m00", 0, 1000), _seg("s1", "ffffffff:m01", 1300, 2300)],
          "operations": []}
    row = observe.mirror(doc, ctx)["segments"][1]
    assert row["join"] == "seamless" and "flags" not in row, row
    print("ok  mirror: a seamless join carries no flag")


def test_mirror_shows_incidental_speech_and_mute_state():
    struct = _struct([_cut("f:t0", 0, 1000)])
    ctx = _ctx(struct)
    doc = {"timeline": [{"seg_id": "s0", "ref": "ffffffff:m00", "file_id": "ffffffff-1111",
                        "in_ms": 0, "out_ms": 1000, "content": "", "axis": "any", "mute": False}],
          "operations": []}
    with mock.patch.object(cm, "speech_words_in_span", return_value=(5, True, False)), \
         mock.patch.object(cm, "speech_boundary_flags",
                           return_value={"clipped_head": False, "clipped_tail": False, "mid_sentence": False}), \
         mock.patch.object(fm, "_said_text_for_span", return_value="incidental narration here"):
        row = observe.mirror(doc, ctx)["segments"][0]
    assert row["speech"] == {"has_words": True, "muted": False, "gist": "incidental narration here"}, row
    assert row.get("flags") == ["incidental"], row
    print("ok  mirror: incidental speech under a non-said cut surfaces speech + flag")


def test_mirror_bounds_a_long_speech_gist():
    struct = _struct([_cut("f:t0", 0, 1000)])
    ctx = _ctx(struct)
    doc = {"timeline": [{"seg_id": "s0", "ref": "ffffffff:m00", "file_id": "ffffffff-1111",
                        "in_ms": 0, "out_ms": 1000, "content": "", "axis": "any"}],
          "operations": []}
    long_text = "A" * 60 + " middle filler that should get dropped " + "B" * 60
    with mock.patch.object(cm, "speech_words_in_span", return_value=(20, True, False)), \
         mock.patch.object(cm, "speech_boundary_flags",
                           return_value={"clipped_head": False, "clipped_tail": False, "mid_sentence": False}), \
         mock.patch.object(fm, "_said_text_for_span", return_value=long_text):
        gist = observe.mirror(doc, ctx)["segments"][0]["speech"]["gist"]
    assert len(gist) < len(long_text)
    assert gist.startswith("A" * 40) and gist.endswith("B" * 40) and "…" in gist, gist
    print("ok  mirror: a long speech gist is bounded to a head...tail")


def test_mirror_shows_broken_line_flag():
    struct = _struct([_cut("f:t0", 0, 1000)])
    ctx = _ctx(struct)
    doc = {"timeline": [{"seg_id": "s0", "ref": "ffffffff:m00", "file_id": "ffffffff-1111",
                        "in_ms": 0, "out_ms": 1000, "content": "", "axis": "any"}],
          "operations": []}
    with mock.patch.object(cm, "speech_words_in_span", return_value=(3, True, False)), \
         mock.patch.object(cm, "speech_boundary_flags",
                           return_value={"clipped_head": False, "clipped_tail": True, "mid_sentence": True}), \
         mock.patch.object(fm, "_said_text_for_span", return_value="and then"):
        row = observe.mirror(doc, ctx)["segments"][0]
    assert "broken-line" in row.get("flags", []), row
    assert "incidental" in row.get("flags", []), row   # both apply at once, additively
    print("ok  mirror: a clipped/mid-sentence boundary surfaces broken-line")


def test_mirror_is_idempotent_for_a_fixed_state():
    struct = _struct([_cut("f:t0", 0, 1000), _cut("f:t1", 1300, 2300)])
    ctx = _ctx(struct)
    doc = {"timeline": [_seg("s0", "ffffffff:m00", 0, 1000), _seg("s1", "ffffffff:m01", 1300, 2300)],
          "operations": []}
    assert observe.mirror(doc, ctx) == observe.mirror(doc, ctx)
    assert observe.mirror_text(doc, ctx) == observe.mirror_text(doc, ctx)
    print("ok  mirror: idempotent for a fixed document/ctx state")


def test_render_mirror_renders_a_readable_line_per_segment():
    m = {"segments": [
        {"pos": 1, "ref": "f:m00"},
        {"pos": 2, "ref": "f:m01", "join": "jump-cut", "join_pic_bits": 5,
         "speech": {"has_words": True, "muted": False, "gist": "hello"},
         "flags": ["jump-cut", "incidental"]},
    ]}
    text = observe.render_mirror(m)
    assert text.startswith("MIRROR"), text
    assert "cut 2" in text and "(f:m01)" in text and "join:jump-cut[5bits]" in text, text
    assert 'speech(audible):"hello"' in text, text
    assert "flags:jump-cut,incidental" in text, text
    print("ok  render_mirror: renders join/speech/flags per segment")


def test_render_mirror_empty_for_no_segments():
    assert observe.render_mirror({"segments": []}) == ""
    print("ok  render_mirror: empty for no segments")


# --------------------------------------------------------------------------
# read_transcript: the on-demand, unbounded full-text sense (section 3.3,
# replaces band-aid D -- no always-on full transcript in read_state).
# --------------------------------------------------------------------------

def test_read_transcript_joins_every_segments_words_in_program_order():
    struct = _struct([_cut("f:t0", 0, 1000), _cut("f:t1", 1300, 2300)])
    ctx = _ctx(struct)
    doc = {"timeline": [_seg("s0", "ffffffff:m00", 0, 1000), _seg("s1", "ffffffff:m01", 1300, 2300)],
          "operations": []}
    with mock.patch.object(
        fm, "_said_text_for_span",
        side_effect=lambda file_id, in_ms, out_ms: {(0, 1000): "hello there", (1300, 2300): "world"}
        .get((in_ms, out_ms), ""),
    ):
        out = observe.read_transcript(doc, ctx)
    assert out["text"] == "hello there world", out
    assert [s["text"] for s in out["segments"]] == ["hello there", "world"]
    assert out["segments"][0]["prog_start_ms"] == 0 and out["segments"][1]["prog_start_ms"] == 1000
    print("ok  read_transcript: joins every segment's words in program order")


def test_read_transcript_includes_incidental_words_under_a_non_said_segment():
    # Unlike review's played_text (axis=="speech" gated), read_transcript
    # surfaces words under ANY segment, matching build_clip_tree's own
    # ungated said_text (section 3.2) -- never hides incidental narration.
    struct = _struct([_cut("f:t0", 0, 1000)])
    ctx = _ctx(struct)
    doc = {"timeline": [{"seg_id": "s0", "ref": "ffffffff:m00", "file_id": "ffffffff-1111",
                        "in_ms": 0, "out_ms": 1000, "content": "", "axis": "any"}],
          "operations": []}
    with mock.patch.object(fm, "_said_text_for_span", return_value="incidental narration"):
        out = observe.read_transcript(doc, ctx)
    assert out["text"] == "incidental narration", out
    print("ok  read_transcript: surfaces incidental words under a non-speech-axis segment")


def test_read_transcript_empty_timeline_yields_empty_transcript():
    struct = _struct([_cut("f:t0", 0, 1000)])
    ctx = _ctx(struct)
    out = observe.read_transcript({"timeline": [], "operations": []}, ctx)
    assert out == {"text": "", "segments": []}, out
    print("ok  read_transcript: empty timeline yields an empty transcript")


def test_read_transcript_skips_segments_with_no_words():
    struct = _struct([_cut("f:t0", 0, 1000), _cut("f:t1", 1300, 2300)])
    ctx = _ctx(struct)
    doc = {"timeline": [_seg("s0", "ffffffff:m00", 0, 1000), _seg("s1", "ffffffff:m01", 1300, 2300)],
          "operations": []}
    with mock.patch.object(
        fm, "_said_text_for_span",
        side_effect=lambda file_id, in_ms, out_ms: "only the second has words" if in_ms == 1300 else "",
    ):
        out = observe.read_transcript(doc, ctx)
    assert len(out["segments"]) == 1 and out["segments"][0]["pos"] == 2, out
    print("ok  read_transcript: a genuinely silent segment contributes nothing")


# --------------------------------------------------------------------------
# affordances: the `runs` section (section 4.2b)
# --------------------------------------------------------------------------

def test_affordances_lists_a_run_with_members_in_source_order():
    # Source-contiguous (gap 200ms <= _RUN_GAP_MS) -> footage_map._assign_runs
    # groups these two into run "r0".
    struct = _struct([_cut("f:t0", 0, 1000), _cut("f:t1", 1200, 2200)])
    ctx = _ctx(struct)
    doc = {"timeline": [], "operations": []}
    aff = observe.affordances(doc, ctx)
    assert len(aff["runs"]) == 1, aff["runs"]
    run = aff["runs"][0]
    assert run["run_id"] == "r0"
    assert run["member_refs"] == ["ffffffff:m00", "ffffffff:m01"], run
    assert run["on_line"] is False   # nothing placed yet
    print("ok  affordances: a source-contiguous pair is grouped into one run, in order")


def test_affordances_run_on_line_true_once_every_member_is_placed():
    struct = _struct([_cut("f:t0", 0, 1000), _cut("f:t1", 1200, 2200)])
    ctx = _ctx(struct)
    doc = {"timeline": [_seg("s0", "ffffffff:m00", 0, 1000), _seg("s1", "ffffffff:m01", 1200, 2200)],
          "operations": []}
    aff = observe.affordances(doc, ctx)
    assert aff["runs"][0]["on_line"] is True, aff["runs"]
    print("ok  affordances: on_line is True once every run member is on the timeline")


def test_affordances_no_runs_section_entries_for_lone_moments():
    struct = _struct([_cut("f:t0", 0, 1000), _cut("f:t1", 5000, 6000)])   # too far apart -> no run
    ctx = _ctx(struct)
    aff = observe.affordances({"timeline": [], "operations": []}, ctx)
    assert aff["runs"] == [], aff["runs"]
    print("ok  affordances: lone (non-run) moments contribute no runs entries")


def main():
    test_read_state_flags_a_jump_cut()
    test_read_state_flags_a_seamless_join()
    test_read_state_no_join_field_for_a_normal_cross_clip_cut()
    test_diagnose_emits_a_jump_cut_warning()
    test_diagnose_does_not_flag_a_genuine_content_change_same_file()
    test_review_flags_the_jump_cut_under_the_continuity_category()
    test_diagnose_and_review_silent_when_seamless()
    test_mirror_empty_timeline_yields_no_segments_and_no_text()
    test_mirror_shows_jump_cut_join_and_flag()
    test_mirror_shows_seamless_join_with_no_flag()
    test_mirror_shows_incidental_speech_and_mute_state()
    test_mirror_bounds_a_long_speech_gist()
    test_mirror_shows_broken_line_flag()
    test_mirror_is_idempotent_for_a_fixed_state()
    test_render_mirror_renders_a_readable_line_per_segment()
    test_render_mirror_empty_for_no_segments()
    test_read_transcript_joins_every_segments_words_in_program_order()
    test_read_transcript_includes_incidental_words_under_a_non_said_segment()
    test_read_transcript_empty_timeline_yields_empty_transcript()
    test_read_transcript_skips_segments_with_no_words()
    test_affordances_lists_a_run_with_members_in_source_order()
    test_affordances_run_on_line_true_once_every_member_is_placed()
    test_affordances_no_runs_section_entries_for_lone_moments()
    print("\nall observe continuity tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
