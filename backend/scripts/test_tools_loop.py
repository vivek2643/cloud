#!/usr/bin/env python3
"""Tests for the agentic edit tool-loop (tools.run_edit_loop) with a scripted
fake LLM -- no network, no DB. Verifies observe/act tool calls thread through and
mutate the working document, and the loop ends on a prose turn.

Run:  PYTHONPATH=. python scripts/test_tools_loop.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import footage_map as fm, observe, tools  # noqa: E402
from app.services.l3.arrange import _MapIndex  # noqa: E402
from app.services.llm import LLMResponse, ToolCall  # noqa: E402

# This file is explicitly "no DB" (see module docstring): build_clip_tree
# calls _said_text_for_span -> _sentences_for_file for every "said" cut
# (beat_transcript.plan.md), which would otherwise hit a real Postgres
# connection. Stub it the same way test_footage_map.py does -- the same
# observable result as "no dialogue_segments row for this file."
fm._sentences_for_file = lambda file_id: ()


def _rung(level, in_ms, out_ms, text=""):
    return {"level": level, "spans": [{"in_ms": in_ms, "out_ms": out_ms}],
            "in_ms": in_ms, "out_ms": out_ms, "play_ms": out_ms - in_ms,
            "text": text, "score": 0.7}


def _cut(hero_id, in_ms, out_ms, label="", ladder=None):
    return {"hero_id": hero_id, "file_id": "ffffffff-1111", "modality": "speech",
            "channel": "said", "label": label, "src_in_ms": in_ms, "src_out_ms": out_ms,
            "play_ms": out_ms - in_ms, "keep_spans": None, "score": 0.75,
            "speaker": "S0", "affordances": ["speech"], "flags": [],
            "ladder": ladder or [_rung("balanced", in_ms, out_ms, label)]}


def _struct():
    c0 = _cut("f:t0", 0, 4000, "we almost shut down")
    c1 = _cut("f:t1", 4000, 8000, "one customer changed everything")
    tree = fm.build_clip_tree("ffffffff-1111", {"name": "T", "duration_ms": 8000}, [c0, c1])
    return {"clips": [tree]}


def _ctx(struct):
    return observe.EditContext(
        file_ids=["ffffffff-1111"], index=_MapIndex(struct), map_struct=struct,
        durations={"ffffffff-1111": 8000}, dup_groups=[])


def _seed_doc():
    return {"brief": {"aspect": "landscape"}, "format": {"aspect": "landscape"},
            "timeline": [], "operations": [], "open_questions": [], "notes": []}


class _ScriptedLLM:
    """Emits a pre-scripted sequence of tool-call rounds, then a final prose turn.
    Each script step is either a list[ToolCall] or a final str."""
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    @property
    def model(self):
        return "scripted"

    def run(self, *, system, messages, tools=None, max_tokens=2048, cache_system=False):
        step = self.script[self.calls] if self.calls < len(self.script) else "Done."
        self.calls += 1
        if isinstance(step, str):
            return LLMResponse(text=step, tool_calls=[], stop_reason="end_turn",
                               assistant_message={"role": "assistant", "content": step})
        # a tool-call round
        blocks = [{"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
                  for tc in step]
        return LLMResponse(text="", tool_calls=step, stop_reason="tool_use",
                           assistant_message={"role": "assistant", "content": blocks})


def test_loop_reads_then_places_then_replies():
    struct = _struct()
    ctx = _ctx(struct)
    script = [
        [ToolCall(id="t1", name="read_state", input={})],
        [ToolCall(id="t2", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t3", name="place", input={"ref": "ffffffff:m01", "level": "balanced"})],
        "I placed both beats on V1 in order -- it runs about 8 seconds.",
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "build it"}],
                              ctx=ctx, document=_seed_doc())
    assert res.changed is True, res
    assert res.steps == ["read_state", "place", "place"], res.steps
    # act appends raw segments (welding is a compile-time concern, as in the manual
    # edit path) -- so both beats land as distinct main-line cuts.
    assert len(res.document["timeline"]) == 2, res.document["timeline"]
    assert "8 seconds" in res.reply
    print("ok  loop: read_state -> place x2 -> prose reply; doc mutated")


def test_loop_pure_chat_no_change():
    struct = _struct()
    res = tools.run_edit_loop(
        _ScriptedLLM(["These two clips are a tight before/after -- want me to cut them?"]),
        system="sys", messages=[{"role": "user", "content": "what do i have?"}],
        ctx=_ctx(struct), document=_seed_doc())
    assert res.changed is False and not res.steps, res
    assert "before/after" in res.reply
    print("ok  loop: pure-chat turn leaves the document unchanged")


def test_loop_bad_tool_is_noop():
    struct = _struct()
    script = [
        [ToolCall(id="t1", name="remove", input={"target_id": "does-not-exist"})],
        "Nothing to remove there.",
    ]
    res = tools.run_edit_loop(_ScriptedLLM(script), system="sys",
                              messages=[{"role": "user", "content": "drop cut 9"}],
                              ctx=_ctx(struct), document=_seed_doc())
    assert res.changed is False, res            # no-op remove didn't change the doc
    assert res.steps == ["remove"], res.steps
    print("ok  loop: an unmatched act is a safe no-op")


def test_loop_ask_user_pauses_turn():
    struct = _struct()
    script = [
        [ToolCall(id="t1", name="ask_user", input={"questions": [
            {"prompt": "Split-screen these two, or play them in sequence?",
             "options": ["Split-screen (side by side)", "In sequence"]}]})],
        # a follow-up round should NOT run -- ask_user ends the turn
        "This should never be reached.",
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys",
                              messages=[{"role": "user", "content": "combine these"}],
                              ctx=_ctx(struct), document=_seed_doc())
    assert res.awaiting_user is True, res
    assert len(res.questions) == 1 and len(res.questions[0]["options"]) == 2, res.questions
    assert res.changed is False and res.steps == ["ask_user"], res
    assert llm.calls == 1, llm.calls          # loop stopped after the ask
    print("ok  loop: ask_user pauses the turn with pickable options")


# --------------------------------------------------------------------------
# _normalize_questions (interactive_ask_and_salience.plan.md WS1-A): the
# ask_user payload's recommended/why/preview enrichment.
# --------------------------------------------------------------------------

def test_normalize_questions_surfaces_a_valid_recommendation():
    out = tools._normalize_questions({"questions": [
        {"prompt": "Which take?", "options": ["Take 1", "Take 2"],
         "recommended": "Take 2", "why": "cleaner delivery", "preview": "I'll use Take 2."},
    ]})
    assert len(out) == 1, out
    q = out[0]
    assert q["recommended"] == "Take 2", q
    assert q["why"] == "cleaner delivery", q
    assert q["preview"] == "I'll use Take 2.", q
    print("ok  normalize_questions: valid recommended/why/preview surfaced")


def test_normalize_questions_drops_a_recommendation_not_in_options():
    out = tools._normalize_questions({"questions": [
        {"prompt": "Which take?", "options": ["Take 1", "Take 2"],
         "recommended": "Take 3", "why": "dangling default"},
    ]})
    assert len(out) == 1, out
    q = out[0]
    assert "recommended" not in q, q
    assert "why" not in q, q          # a reason with nothing to recommend is noise
    print("ok  normalize_questions: dangling recommended (not in options) is dropped")


def test_normalize_questions_still_drops_under_two_options():
    out = tools._normalize_questions({"questions": [
        {"prompt": "Only one option?", "options": ["Take 1"], "recommended": "Take 1"},
        {"prompt": "Real question", "options": ["A", "B"]},
    ]})
    assert len(out) == 1 and out[0]["prompt"] == "Real question", out
    print("ok  normalize_questions: <2 options still dropped (unchanged)")


def test_normalize_questions_no_recommendation_omits_the_keys():
    out = tools._normalize_questions({"questions": [
        {"prompt": "Plain ask", "options": ["A", "B"]},
    ]})
    assert len(out) == 1, out
    q = out[0]
    assert "recommended" not in q and "why" not in q and "preview" not in q, q
    print("ok  normalize_questions: no recommendation -> keys omitted entirely")


def test_loop_split_screen_after_answer():
    """After the user answered the split question, the brain builds V1 then lays a
    split_screen -> op + layout region land on the working doc."""
    struct = _struct()
    ctx = _ctx(struct)
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t2", name="split_screen", input={
            "ref": "ffffffff:m01", "template": "split_h", "from_ms": 500, "to_ms": 3000})],
        "Side-by-side over the first few seconds -- done.",
    ]
    res = tools.run_edit_loop(_ScriptedLLM(script), system="sys",
                              messages=[{"role": "user", "content": "split screen these two"}],
                              ctx=ctx, document=_seed_doc())
    assert res.changed is True and res.steps == ["place", "split_screen"], res.steps
    ops = [o for o in res.document["operations"] if o["type"] == "place_video"]
    assert len(ops) == 1, ops
    regs = res.document.get("layout_regions") or []
    assert len(regs) == 1 and regs[0]["template"] == "split_h", regs
    print("ok  loop: split_screen adds op + layout region")


# --------------------------------------------------------------------------
# edso_done_gate.plan.md: the done-gate (_verify_before_finish)
# --------------------------------------------------------------------------

def _gate_state():
    return {"struct_tries": 0, "length_surfaced": False, "reviewed": False}


def _clean_doc():
    """A document `observe.validate` finds structurally clean (real file_id,
    non-empty span within the file's known duration) -- so tests that stub
    `observe.diagnose` actually reach the diagnose-based checks instead of
    tripping the (unmocked, real) structural check first."""
    return {"timeline": [{"seg_id": "s", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 1000}],
           "operations": [], "brief": {}}


def test_gate_blocks_structural_error():
    ctx = _ctx(_struct())
    doc = {"timeline": [{"seg_id": "s0", "file_id": "ffffffff-1111",
                         "in_ms": 0, "out_ms": 0}],  # empty span -> validate flags it
           "operations": [], "brief": {}}
    st = _gate_state()
    fb = tools._verify_before_finish(doc, ctx, st, [])
    assert fb and "structural" in fb.lower(), fb
    assert st["struct_tries"] == 1, st
    print("ok  gate: structural error forces a fix before finishing")


def test_gate_clean_doc_finishes():
    ctx = _ctx(_struct())
    assert tools._verify_before_finish({"timeline": [], "operations": [], "brief": {}},
                                       ctx, _gate_state(), []) is None
    print("ok  gate: clean/empty edit finishes without nagging")


def test_gate_length_is_fix_or_justify_once():
    ctx = _ctx(_struct())
    orig = observe.diagnose
    observe.diagnose = lambda *a, **k: [
        {"severity": "warn", "anchor": "whole", "message": "over target: 8.0s vs 5.0s"}]
    try:
        st = _gate_state()
        fb1 = tools._verify_before_finish(_clean_doc(), ctx, st, [])
        assert fb1 and "length" in fb1.lower() and st["length_surfaced"], fb1
        fb2 = tools._verify_before_finish(_clean_doc(), ctx, st, [])
        assert fb2 is None, fb2          # surfaced once; do not nag again
    finally:
        observe.diagnose = orig
    print("ok  gate: over-target length is surfaced once (fix-or-justify)")


def test_gate_advisory_review_skipped_when_already_diagnosed():
    ctx = _ctx(_struct())
    orig = observe.diagnose
    observe.diagnose = lambda *a, **k: [
        {"severity": "warn", "anchor": "cuts 1-2", "message": "same speaker back-to-back"}]
    try:
        st = _gate_state()
        fb = tools._verify_before_finish(_clean_doc(), ctx, st, [])
        assert fb and "advisory" in fb.lower() and st["reviewed"], fb
        st2 = _gate_state()
        fb2 = tools._verify_before_finish(_clean_doc(), ctx, st2,
                                          ["place", "diagnose"])
        assert fb2 is None, fb2          # brain self-reviewed -> no forced nudge
    finally:
        observe.diagnose = orig
    print("ok  gate: advisory review fires once, skipped if brain already diagnosed")


def test_gate_forces_length_reconcile_end_to_end():
    ctx = _ctx(_struct())
    doc = _seed_doc()
    doc["brief"]["target_duration_s"] = 5      # the 8s edit will be over target (>1.2x)
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t2", name="place", input={"ref": "ffffffff:m01", "level": "balanced"})],
        "Placed both -- about 8 seconds.",                       # finish attempt #1
        "Both beats are essential, so I'm keeping it at ~8s.",   # justify -> finish
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys",
                              messages=[{"role": "user", "content": "cut a 5s teaser"}],
                              ctx=ctx, document=doc)
    assert "keeping it" in res.reply, res.reply
    assert llm.calls == 4, llm.calls           # the gate bought exactly one extra turn
    print("ok  gate: over-target forces one length reconcile, then finishes")


def test_gate_surfaces_review_flags_end_to_end():
    """edso_pacing_audit_timing.plan.md item 6: the done-gate's advisory step
    now folds in observe.review's program read-back flags (a foreign-speaker
    lead-in here), reusing the SAME reviewed guard as diagnose -- one extra
    turn, never a prescribed fix, then the loop still terminates."""
    ctx = _ctx(_struct())
    doc = _seed_doc()
    sentences = (
        {"speaker": "S9", "text": "um", "src_in_ms": 0, "src_out_ms": 300},
        {"speaker": "S0", "text": "we almost shut down", "src_in_ms": 400, "src_out_ms": 4000},
    )
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        "Placed the beat.",                               # finish attempt #1 -> gate flags the head
        "Kept it -- the lead-in is a natural warm-up.",    # acknowledge -> finish
    ]
    llm = _ScriptedLLM(script)
    orig = fm._sentences_for_file
    fm._sentences_for_file = lambda file_id: sentences
    try:
        res = tools.run_edit_loop(llm, system="sys",
                                  messages=[{"role": "user", "content": "cut this"}],
                                  ctx=ctx, document=doc)
    finally:
        fm._sentences_for_file = orig
    assert "Kept it" in res.reply, res.reply
    assert llm.calls == 3, llm.calls           # the gate bought exactly one extra turn
    print("ok  gate: review flags surface end-to-end and the loop still terminates")


# --------------------------------------------------------------------------
# edso_think_act_check.plan.md
# --------------------------------------------------------------------------

def test_latest_user_text_reads_the_newest_user_message():
    assert tools._latest_user_text([{"role": "user", "content": "cut a 5s teaser"}]) == "cut a 5s teaser"
    assert tools._latest_user_text([
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": [{"type": "text", "text": "second, split screen"}]},
    ]) == "second, split screen"
    assert tools._latest_user_text([]) == ""
    print("ok  _latest_user_text reads the newest user message, both content shapes")


def test_gate_flags_a_requested_feature_missing_end_to_end():
    """change 4: the done-gate's audit now also checks a NAMED feature (split
    screen, a music bed) is actually present -- the check that would have
    caught the motivating "never added the requested split screen" failure.
    One extra turn, never a prescribed fix, then the loop still terminates."""
    ctx = _ctx(_struct())
    doc = _seed_doc()
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        "Placed the beat.",                                          # finish #1 -> gate flags the missing split
        "Added a split screen instead of skipping it -- fixed now.",  # act -> finish
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys",
                              messages=[{"role": "user", "content": "make it a split screen"}],
                              ctx=ctx, document=doc)
    assert "Added a split screen" in res.reply, res.reply
    assert llm.calls == 3, llm.calls           # the gate bought exactly one extra turn
    print("ok  gate: a requested-but-missing feature is flagged, then the loop still terminates")


def test_gate_does_not_flag_a_feature_the_user_never_asked_for():
    ctx = _ctx(_struct())
    doc = _seed_doc()
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        "Placed the beat.",
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys",
                              messages=[{"role": "user", "content": "just cut it together"}],
                              ctx=ctx, document=doc)
    assert res.reply == "Placed the beat.", res.reply
    assert llm.calls == 2, llm.calls           # no gate nudge at all -- nothing to flag
    print("ok  gate: no feature nudge when the ask never named one")


def main():
    test_loop_reads_then_places_then_replies()
    test_loop_pure_chat_no_change()
    test_loop_bad_tool_is_noop()
    test_loop_ask_user_pauses_turn()
    test_normalize_questions_surfaces_a_valid_recommendation()
    test_normalize_questions_drops_a_recommendation_not_in_options()
    test_normalize_questions_still_drops_under_two_options()
    test_normalize_questions_no_recommendation_omits_the_keys()
    test_loop_split_screen_after_answer()
    test_gate_blocks_structural_error()
    test_gate_clean_doc_finishes()
    test_gate_length_is_fix_or_justify_once()
    test_gate_advisory_review_skipped_when_already_diagnosed()
    test_gate_forces_length_reconcile_end_to_end()
    test_gate_surfaces_review_flags_end_to_end()
    test_latest_user_text_reads_the_newest_user_message()
    test_gate_flags_a_requested_feature_missing_end_to_end()
    test_gate_does_not_flag_a_feature_the_user_never_asked_for()
    print("\nall tool-loop tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
