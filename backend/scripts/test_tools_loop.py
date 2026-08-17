#!/usr/bin/env python3
"""Tests for the agentic edit tool-loop (tools.run_edit_loop) with a scripted
fake LLM -- no network, no DB. Verifies observe/act tool calls thread through and
mutate the working document, and the loop ends on a prose turn.

Run:  PYTHONPATH=. python scripts/test_tools_loop.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.l3 import act, footage_map as fm, observe, tools  # noqa: E402
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
        # brain_mirror_readside.plan.md section 4.2: a snapshot (shallow copy
        # -- `messages` IS run_edit_loop's own mutating `convo` list, so a
        # bare reference would show every call as the FINAL state) of the
        # messages this call actually saw, for tests that need to inspect
        # what got pushed into the conversation between rounds (e.g. the
        # mirror).
        self.seen_messages = []

    @property
    def model(self):
        return "scripted"

    def run(self, *, system, messages, tools=None, max_tokens=2048, cache_system=False):
        self.seen_messages.append(list(messages))
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
        "I placed both beats on V1 in order.",             # finish attempt -> Stage 2 (craft) fires once
        # brain_accountability_architecture.plan.md PART 1: the ONE finalizer
        # now forces a wrap_up on ANY substantive finish with an empty surface,
        # not only a compromise-driven one -- a clean finish must also recap.
        [ToolCall(id="t4", name="wrap_up", input={"summary": "Reads clean -- it runs about 8 seconds."})],
        "Reads clean -- it runs about 8 seconds.",          # blind verdict -> finish
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "build it"}],
                              ctx=ctx, document=_seed_doc())
    assert res.changed is True, res
    assert res.steps == ["read_state", "place", "place", "wrap_up"], res.steps
    # act appends raw segments (welding is a compile-time concern, as in the manual
    # edit path) -- so both beats land as distinct main-line cuts.
    assert len(res.document["timeline"]) == 2, res.document["timeline"]
    assert "8 seconds" in res.reply
    assert llm.calls == 6, llm.calls           # Stage-2 craft + the mandatory wrap_up each bought a turn
    print("ok  loop: read_state -> place x2 -> prose reply; doc mutated")


def test_loop_pushes_the_mirror_after_a_tool_call():
    """brain_mirror_readside.plan.md section 4.2: the always-present mirror
    is pushed as a plain text block alongside each round's tool_result
    blocks -- the NEXT llm.run() call must already see it in its own
    messages, without the brain having to explicitly call a sense for it."""
    struct = _struct()
    ctx = _ctx(struct)
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        "Placed it.",
        "Reads fine.",
    ]
    llm = _ScriptedLLM(script)
    tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "place the first beat"}],
                        ctx=ctx, document=_seed_doc())
    # llm.seen_messages[1] is the round immediately AFTER the place() call --
    # its own messages must already carry the mirror text.
    last_msg = llm.seen_messages[1][-1]
    assert last_msg["role"] == "user"
    texts = [b.get("text", "") for b in last_msg["content"] if isinstance(b, dict) and b.get("type") == "text"]
    assert any("MIRROR" in t for t in texts), last_msg
    print("ok  loop: the mirror is pushed as a text block right after a tool call")


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
        "Side-by-side over the first few seconds.",   # finish attempt -> Stage 2 (craft) fires once
        # PART 1: the ONE finalizer now forces a wrap_up on any clean finish
        # with an empty surface.
        [ToolCall(id="t3", name="wrap_up", input={"summary": "Reads clean -- done."})],
        "Reads clean -- done.",                       # blind verdict -> finish
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys",
                              messages=[{"role": "user", "content": "split screen these two"}],
                              ctx=ctx, document=_seed_doc())
    assert res.changed is True and res.steps == ["place", "split_screen", "wrap_up"], res.steps
    assert llm.calls == 5, llm.calls           # Stage-2 craft + the mandatory wrap_up each bought a turn
    ops = [o for o in res.document["operations"] if o["type"] == "place_video"]
    assert len(ops) == 1, ops
    regs = res.document.get("layout_regions") or []
    assert len(regs) == 1 and regs[0]["template"] == "split_h", regs
    print("ok  loop: split_screen adds op + layout region")


# --------------------------------------------------------------------------
# audio_and_audit.plan.md Phase 3: beat-snap trim, shift-to-align move
# --------------------------------------------------------------------------

def _music_ctx():
    struct = _struct()
    return observe.EditContext(
        file_ids=["ffffffff-1111"], index=_MapIndex(struct), map_struct=struct,
        durations={"ffffffff-1111": 8000}, dup_groups=[],
        audio_features={"bed1": {"integrated_lufs": -14.0, "is_musical": True,
                                 "bpm": 120.0,
                                 "onsets_ms": [0, 500, 1000, 1500, 2000, 2500, 3000]}})


def _bed_doc():
    return {"format": {"aspect": "landscape"}, "timeline": [], "operations": [
        {"op_id": "pa1", "type": "place_audio", "role": "music",
         "source_file_id": "bed1", "src_in_ms": 0, "src_out_ms": 2000,
         "from_ms": 1000, "to_ms": 3000, "gain_db": 0.0, "duck_db": 0.0},
    ]}


def test_dispatch_inspect_cut_returns_windowed_payload_and_leaves_doc_unchanged():
    """brain_perception_upgrade.plan.md Change 1, Mechanism B: inspect_cut is
    an OBSERVE tool (read-only) -- dispatched, it must never mutate the
    working document, same as read_state/predict/validate."""
    struct = _struct()
    ctx = _ctx(struct)
    doc = _seed_doc()
    signals = {"motion": {}, "audio": {}, "scene": {}}
    with mock.patch.object(observe, "_fetch_signal_window", return_value=signals):
        obs, new, changed = tools._dispatch("inspect_cut", {"ref": "ffffffff:m00"}, ctx, doc)
    assert not changed, obs
    assert new is doc
    assert '"ref": "ffffffff:m00"' in obs, obs
    assert '"action"' in obs and '"audio"' in obs and '"shots"' in obs, obs
    print("ok  dispatch: inspect_cut returns the windowed payload, doc unchanged")


# --------------------------------------------------------------------------
# brain_plan_mechanism.plan.md: the durable plan artifact + PLAN mirror
# --------------------------------------------------------------------------

def test_dispatch_set_plan_applies_and_returns_the_rendered_plan():
    """§9.3/brain_plan_altitude.plan.md §7.3: a scripted set_plan call flows
    through _dispatch -- result JSON has "applied": true and the rendered
    plan; changed is True; the new document's plan.structure is the
    normalized BEAT list (each {beat, need}), in input order (not a
    read_state echo -- a tight confirmation, per §3.2)."""
    ctx = _ctx(_struct())
    doc = _seed_doc()
    obs, new, changed = tools._dispatch("set_plan", {
        "purpose": "teach a first-time viewer why the migration was worth it",
        "carries": ["founder VO leads throughout"],
        "structure": [
            {"beat": "cold-open on the outage line", "need": "required"},
            {"beat": "founder explains the stakes", "need": "required"},
            {"beat": "the demo", "need": "required"},
            {"beat": "close on the one-customer line", "need": "optional"},
        ],
        "watch": ["clip93 has a backward jump-cut risk"],
    }, ctx, doc)
    assert changed is True, obs
    assert new is not doc
    assert '"applied": true' in obs, obs
    assert new["plan"]["structure"] == [
        {"beat": "cold-open on the outage line", "need": "required"},
        {"beat": "founder explains the stakes", "need": "required"},
        {"beat": "the demo", "need": "required"},
        {"beat": "close on the one-customer line", "need": "optional"}], new["plan"]
    assert new["plan"]["rev"] == 1, new["plan"]
    assert '"read_state"' not in obs and '"state"' not in obs, obs
    print("ok  dispatch: set_plan applies and echoes the rendered beat plan, not read_state")


def test_loop_pushes_the_plan_mirror_after_a_set_plan_call():
    """§9.3/brain_plan_altitude.plan.md §7.3: after the set_plan round, the
    NEXT llm.run() call's own messages already carry a text_block with
    "PLAN (rev" and "structure (BEATS" -- the plan mirror is pushed the same
    way the edit mirror is (test_loop_pushes_the_mirror_after_a_tool_call)."""
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="set_plan", input={
            "purpose": "teach why the migration was worth it",
            "structure": [{"beat": "cold-open", "need": "required"},
                         {"beat": "the demo", "need": "required"},
                         {"beat": "close", "need": "required"}]})],
        "Wrote the plan.",
        "Reads fine.",
    ]
    llm = _ScriptedLLM(script)
    tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "plan this out"}],
                        ctx=ctx, document=_seed_doc())
    last_msg = llm.seen_messages[1][-1]
    assert last_msg["role"] == "user"
    texts = [b.get("text", "") for b in last_msg["content"] if isinstance(b, dict) and b.get("type") == "text"]
    assert any("PLAN (rev" in t for t in texts), last_msg
    assert any("structure (BEATS" in t for t in texts), last_msg
    print("ok  loop: the plan mirror is pushed as a text block right after set_plan, with beat markers")


def test_loop_pushed_plan_mirror_is_loud_when_the_edit_moved_without_a_plan():
    """§4.2: `building=changed` -- once the brain has placed a cut this turn
    but still has no plan, the pushed mirror uses the LOUD "building without
    a written plan" nudge, not the quiet turn-start one."""
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        "Placed it.",
        "Reads fine.",
    ]
    llm = _ScriptedLLM(script)
    tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "place the first beat"}],
                        ctx=ctx, document=_seed_doc())
    last_msg = llm.seen_messages[1][-1]
    texts = [b.get("text", "") for b in last_msg["content"] if isinstance(b, dict) and b.get("type") == "text"]
    assert any("building without a written plan" in t for t in texts), last_msg
    print("ok  loop: the plan mirror is loud once the edit moved with no plan written")


# --------------------------------------------------------------------------
# brain_plan_conformance.plan.md §6.1: act.wrap_up + dispatch
# --------------------------------------------------------------------------

def test_wrap_up_writes_surface_fields():
    doc = {"timeline": [], "operations": []}
    new = act.wrap_up(
        doc,
        summary="  tightened to 55s; the differentiator take was disfluent so "
                "I used the slide+VO route  ",
        open_questions=["want the raw take instead?", "   "])
    assert new is not doc
    assert new["summary"] == ("tightened to 55s; the differentiator take was disfluent "
                              "so I used the slide+VO route"), new
    assert new["open_questions"] == ["want the raw take instead?"], new   # blank dropped
    assert "notes" not in new                       # a field left None is untouched
    assert act.wrap_up(doc) is doc                  # all-None -> a clean no-op
    print("ok  wrap_up: writes summary/open_questions (stripped, blanks dropped); all-None no-op")


def test_dispatch_wrap_up_applies_and_echoes_surface():
    ctx = _ctx(_struct())
    doc = _seed_doc()
    obs, new, changed = tools._dispatch("wrap_up", {
        "summary": "tightened to 55s; used the slide+VO route",
    }, ctx, doc)
    assert changed is True, obs
    assert new is not doc
    assert '"applied": true' in obs, obs
    assert new["summary"] == "tightened to 55s; used the slide+VO route", new
    assert '"summary": "tightened to 55s; used the slide+VO route"' in obs, obs
    print("ok  dispatch: wrap_up applies and echoes the written surface, not read_state")


def test_trim_snap_beat_lands_an_ops_resulting_edge_on_the_grid():
    """trim only ever moves a placed op's program END (its program START,
    from_ms, never moves via trim) -- snap:'beat' lands THAT resulting edge
    on the nearest onset, then folds the correction back onto whichever
    source edge the brain named (here delta_in_ms)."""
    ctx = _music_ctx()
    doc = _bed_doc()
    obs, new, changed = tools._dispatch(
        "trim", {"target_id": "pa1", "delta_in_ms": 200, "snap": "beat"}, ctx, doc)
    assert changed, obs
    op = next(o for o in new["operations"] if o["op_id"] == "pa1")
    assert (op["src_in_ms"], op["src_out_ms"]) == (500, 2000), op
    assert op["to_ms"] == 2500, op          # snapped onto the nearest onset (program 2500ms)
    assert '"snap"' in obs, obs
    print("ok  trim snap:'beat' lands an op's resulting program edge on the grid")


def test_trim_snap_beat_is_a_noop_pass_through_without_music():
    ctx = _ctx(_struct())    # no audio_features -- no musical source in play
    doc = _bed_doc()
    obs, new, changed = tools._dispatch(
        "trim", {"target_id": "pa1", "delta_in_ms": 200, "snap": "beat"}, ctx, doc)
    assert changed, obs                      # trim itself still applies -- just unsnapped
    op = next(o for o in new["operations"] if o["op_id"] == "pa1")
    assert (op["src_in_ms"], op["src_out_ms"]) == (200, 2000), op
    assert '"snap"' not in obs, obs
    print("ok  trim snap:'beat' passes through unsnapped when no musical source is in play")


def test_move_shift_to_align_slides_an_op_by_the_computed_offset():
    """Shift-to-align: the brain names a CURRENT onset (in this op's own
    program time) and the program moment it should land on; the handle
    computes the offset and slides the op, keeping its own duration."""
    ctx = _music_ctx()
    doc = _bed_doc()
    obs, new, changed = tools._dispatch(
        "move", {"target_id": "pa1", "align_onset_ms": 1500, "align_to_ms": 5000}, ctx, doc)
    assert changed, obs
    op = next(o for o in new["operations"] if o["op_id"] == "pa1")
    assert op["from_ms"] == 4500 and op["to_ms"] == 6500, op   # +3500ms offset, duration kept
    print("ok  move align_onset_ms/align_to_ms shifts an op by the computed offset")


# --------------------------------------------------------------------------
# edso_done_gate.plan.md: the done-gate (_verify_before_finish)
# --------------------------------------------------------------------------

def _gate_state():
    return {"struct_tries": 0, "length_surfaced": False, "intent_surfaced": False,
           "craft_surfaced": False, "reviewed": False,
           "conformance_surfaced": False, "surface_blocked": 0}


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
        assert fb2 and "stage 2" in fb2.lower() and st["craft_surfaced"], fb2
        fb3 = tools._verify_before_finish(_clean_doc(), ctx, st, [])
        assert fb3 is None, fb3          # nothing left to flag; loop can finish
    finally:
        observe.diagnose = orig
    print("ok  gate: length fix-or-justify, then the Stage-2 craft check, then finish")


def test_gate_craft_check_fires_once_unconditionally():
    """Phase 5: Stage 2 (fit to craft) fires exactly once for ANY turn that
    changed a non-empty edit -- unlike Stage 3, it is NOT gated on whether
    diagnose/review found anything (a clean-per-the-checklist edit still
    gets the blind verdict), and NOT skippable via self-review (calling a
    deterministic sense is not the same as producing the blind verdict)."""
    ctx = _ctx(_struct())
    st = _gate_state()
    fb1 = tools._verify_before_finish(_clean_doc(), ctx, st, [])
    assert fb1 and "stage 2" in fb1.lower() and st["craft_surfaced"], fb1
    fb2 = tools._verify_before_finish(_clean_doc(), ctx, st, [])
    assert fb2 is None, fb2          # nothing else to flag for this clean doc

    st2 = _gate_state()
    fb3 = tools._verify_before_finish(_clean_doc(), ctx, st2, ["place", "diagnose", "review"])
    assert fb3 and "stage 2" in fb3.lower(), fb3   # fires even though the brain self-reviewed
    print("ok  gate: Stage 2 craft check is unconditional and not skippable via self-review")


def test_gate_advisory_review_skipped_when_already_diagnosed():
    ctx = _ctx(_struct())
    orig = observe.diagnose
    observe.diagnose = lambda *a, **k: [
        {"severity": "warn", "anchor": "cuts 1-2", "message": "same speaker back-to-back"}]
    try:
        st = _gate_state()
        fb1 = tools._verify_before_finish(_clean_doc(), ctx, st, [])
        assert fb1 and "stage 2" in fb1.lower(), fb1     # Stage 2 fires first
        fb2 = tools._verify_before_finish(_clean_doc(), ctx, st, [])
        assert fb2 and "advisory" in fb2.lower() and st["reviewed"], fb2

        st2 = _gate_state()
        fb3 = tools._verify_before_finish(_clean_doc(), ctx, st2, ["place", "diagnose"])
        assert fb3 and "stage 2" in fb3.lower(), fb3     # Stage 2 still fires (self-review doesn't skip it)
        fb4 = tools._verify_before_finish(_clean_doc(), ctx, st2, ["place", "diagnose"])
        assert fb4 is None, fb4          # brain self-reviewed -> Stage 3's redundant nudge skipped
    finally:
        observe.diagnose = orig
    print("ok  gate: Stage 3 advisory review fires once, skipped if brain already diagnosed")


def test_gate_jump_cut_finding_folds_into_stage_3_no_separate_continuity_stage():
    """brain_mirror_readside.plan.md section 4.2: the former Stage 2.5 (a
    one-shot, reviewed-gate-bypassing continuity reveal) is RETIRED -- the
    always-present mirror (pushed every tool call by run_edit_loop) now
    puts join/flag state in front of the brain continuously, so the gate no
    longer needs a special *revealing* role for jump-cuts. A jump-cut
    finding still reaches Stage 3's normal advisory path (folded into
    `rest`, same as any other flag) -- which self-review DOES skip, unlike
    the old Stage 2.5."""
    ctx = _ctx(_struct())
    orig = observe.diagnose
    observe.diagnose = lambda *a, **k: [
        {"severity": "warn", "anchor": "cuts 1-2", "message": "jump-cut: same shot, source non-contiguous"}]
    try:
        st = _gate_state()
        fb1 = tools._verify_before_finish(_clean_doc(), ctx, st, [])
        assert fb1 and "stage 2" in fb1.lower(), fb1   # Stage 2 (craft) still fires first

        fb2 = tools._verify_before_finish(_clean_doc(), ctx, st, [])
        assert fb2 and "jump-cut" in fb2.lower() and "stage 3" in fb2.lower(), fb2
        assert st["reviewed"], st

        fb3 = tools._verify_before_finish(_clean_doc(), ctx, st, [])
        assert fb3 is None, fb3   # nothing left to flag; loop can finish

        # Self-review skips it entirely now -- no special stage forces it
        # through; the mirror is the always-on reveal instead.
        st2 = _gate_state()
        fb4 = tools._verify_before_finish(_clean_doc(), ctx, st2, ["place", "diagnose"])
        assert fb4 and "stage 2" in fb4.lower(), fb4
        fb5 = tools._verify_before_finish(_clean_doc(), ctx, st2, ["place", "diagnose"])
        assert fb5 is None, fb5   # self-reviewed -> Stage 3's nudge (incl. the jump-cut) skipped
    finally:
        observe.diagnose = orig
    print("ok  gate: jump-cut finding folds into Stage 3 advisory; no separate continuity stage survives self-review")


def test_gate_blocks_a_mid_program_audio_gap_unless_surfaced():
    """audio_and_audit.plan.md Phase 5: a mid-program silence hole (Stage 3's
    audio-gap flag, from layers.audio_gaps -- a muted seg between two audible
    ones) can't finish silently; the brain must see it (the blind Stage-2
    check, then Stage 3's concrete fact) before the gate lets a turn finish."""
    ctx = _ctx(_struct())
    doc = {
        "timeline": [
            {"seg_id": "s0", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 2000},
            {"seg_id": "s1", "file_id": "ffffffff-1111", "in_ms": 2000, "out_ms": 4000, "mute": True},
            {"seg_id": "s2", "file_id": "ffffffff-1111", "in_ms": 4000, "out_ms": 6000},
        ],
        "operations": [], "brief": {},
    }
    st = _gate_state()
    fb1 = tools._verify_before_finish(doc, ctx, st, [])
    assert fb1 and "stage 2" in fb1.lower(), fb1     # the blind craft check fires first
    fb2 = tools._verify_before_finish(doc, ctx, st, [])
    assert fb2 and "no audio" in fb2.lower(), fb2    # Stage 3 surfaces the gap as a concrete fact
    fb3 = tools._verify_before_finish(doc, ctx, st, [])
    assert fb3 is None, fb3          # surfaced (twice); the gate itself never hard-blocks forever
    print("ok  gate: a mid-program audio gap is surfaced (blind check, then the concrete fact)")


def test_gate_forces_length_reconcile_end_to_end():
    ctx = _ctx(_struct())
    doc = _seed_doc()
    doc["brief"]["target_duration_s"] = 5      # the 8s edit will be over target (>1.2x)
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t2", name="place", input={"ref": "ffffffff:m01", "level": "balanced"})],
        "Placed both -- about 8 seconds.",                       # finish attempt #1 -> length gate fires
        "Both beats are essential, so I'm keeping it at ~8s.",   # justify -> Stage 2 (craft) fires
        # PART 1: the ONE finalizer now forces a wrap_up on any clean finish
        # with an empty surface.
        [ToolCall(id="t3", name="wrap_up", input={"summary": "Reads clean -- keeping it at ~8s."})],
        "Reads clean -- keeping it at ~8s.",                     # blind verdict -> finish
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys",
                              messages=[{"role": "user", "content": "cut a 5s teaser"}],
                              ctx=ctx, document=doc)
    assert "keeping it" in res.reply, res.reply
    assert llm.calls == 6, llm.calls           # length + Stage 2 + the mandatory wrap_up each bought a turn
    print("ok  gate: over-target forces one length reconcile + the craft check, then finishes")


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
        "Placed the beat.",                               # finish attempt #1 -> Stage 2 (craft) fires
        "Reads clean.",                                   # blind verdict -> Stage 3 flags the head
        # PART 1: the ONE finalizer now forces a wrap_up on any clean finish
        # with an empty surface.
        [ToolCall(id="t2", name="wrap_up", input={"summary": "Kept it -- the lead-in is a natural warm-up."})],
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
    assert llm.calls == 5, llm.calls           # Stage 2 + Stage 3 + the mandatory wrap_up each bought a turn
    print("ok  gate: review flags surface end-to-end (after the craft check) and the loop still terminates")


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
        "Placed the beat.",                                          # finish #1 -> Stage 1 flags the missing split
        "Added a split screen instead of skipping it -- fixed now.",  # claim fix -> Stage 2 (craft) fires
        # PART 1: the ONE finalizer now forces a wrap_up on any clean finish
        # with an empty surface.
        [ToolCall(id="t2", name="wrap_up", input={"summary": "Reads clean -- confirmed."})],
        "Reads clean -- confirmed.",                                  # blind verdict -> finish
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys",
                              messages=[{"role": "user", "content": "make it a split screen"}],
                              ctx=ctx, document=doc)
    assert res.reply == "Reads clean -- confirmed.", res.reply
    assert llm.calls == 5, llm.calls           # Stage 1 + Stage 2 + the mandatory wrap_up each bought a turn
    print("ok  gate: a requested-but-missing feature is flagged, then the loop still terminates")


def test_gate_does_not_flag_a_feature_the_user_never_asked_for():
    ctx = _ctx(_struct())
    doc = _seed_doc()
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        "Placed the beat.",   # finish attempt -> Stage 1 has nothing to flag; Stage 2 (craft) still fires
        # PART 1: the ONE finalizer now forces a wrap_up on any clean finish
        # with an empty surface.
        [ToolCall(id="t2", name="wrap_up", input={"summary": "Reads clean."})],
        "Reads clean.",       # blind verdict -> finish
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys",
                              messages=[{"role": "user", "content": "just cut it together"}],
                              ctx=ctx, document=doc)
    assert res.reply == "Reads clean.", res.reply
    assert llm.calls == 4, llm.calls           # no Stage-1 nudge -- Stage 2's craft check + wrap_up each ran once
    print("ok  gate: no Stage-1 feature nudge when the ask never named one (Stage 2 still runs)")


# --------------------------------------------------------------------------
# brain_plan_conformance.plan.md §6.3: Part B -- the conformance stage
# --------------------------------------------------------------------------

def test_gate_conformance_fires_once_with_required_beats_and_watch():
    ctx = _ctx(_struct())
    doc = _clean_doc()
    doc["plan"] = {
        "purpose": "teach why the migration was worth it",
        "structure": [{"beat": "differentiator -- shown through the demo", "need": "required"},
                     {"beat": "a customer quote if one lands", "need": "optional"}],
        "watch": ["land action cuts on the beat grid"],
        "carries": [], "rev": 1,
    }
    st = _gate_state()
    st["craft_surfaced"] = True                # drive past Stage 2; _clean_doc() has no diagnose/review findings
    fb1 = tools._verify_before_finish(doc, ctx, st, [])
    assert fb1 and "plan conformance" in fb1.lower(), fb1
    assert "differentiator -- shown through the demo" in fb1, fb1
    assert "a customer quote if one lands" not in fb1, fb1     # optional beat not in the required list
    assert "land action cuts on the beat grid" in fb1, fb1
    assert st["conformance_surfaced"], st
    fb2 = tools._verify_before_finish(doc, ctx, st, [])
    assert fb2 is None, fb2                     # fires once; no live hint here -> Part A doesn't fire either
    print("ok  gate: Part B fires once, lists required beats + declared carries/watch, then stays silent")


def test_gate_conformance_noops_without_a_plan():
    ctx = _ctx(_struct())
    doc = _clean_doc()
    st = _gate_state()
    st["craft_surfaced"] = True
    fb = tools._verify_before_finish(doc, ctx, st, [])
    assert fb is None, fb
    assert not st["conformance_surfaced"], st
    print("ok  gate: Part B no-ops (ladder finishes as today) when there's no plan of record")


def test_gate_conformance_old_shape_structure_of_strings():
    """Back-compat (brain_plan_altitude.plan.md §6): a pre-altitude
    structure:[str] plan is coerced via observe._beat_view and presented as
    required beats -- never raises."""
    ctx = _ctx(_struct())
    doc = _clean_doc()
    doc["plan"] = {"structure": ["cold open", "the demo"], "carries": [], "watch": [], "rev": 1}
    st = _gate_state()
    st["craft_surfaced"] = True
    fb = tools._verify_before_finish(doc, ctx, st, [])
    assert fb and "plan conformance" in fb.lower(), fb
    assert "cold open" in fb and "the demo" in fb, fb
    print("ok  gate: Part B coerces old-shape structure:[str] via _beat_view, never raises")


# --------------------------------------------------------------------------
# brain_plan_conformance.plan.md §6.2: Part A -- the surface-non-empty block
# --------------------------------------------------------------------------

def test_gate_blocks_finish_when_compromise_in_play_and_surface_empty():
    ctx = _ctx(_struct())
    doc = _clean_doc()
    st = _gate_state()
    st["craft_surfaced"] = True
    st["conformance_surfaced"] = True           # an account was already demanded this turn
    with mock.patch.object(tools, "_conformance_hints", return_value=["a live advisory hint"]):
        fb = tools._verify_before_finish(doc, ctx, st, [])
    assert fb and "surface" in fb.lower(), fb
    assert st["surface_blocked"] == 1, st
    print("ok  gate: Part A blocks finish when a compromise is in play and the surface is empty")


def test_gate_surface_block_clears_once_summary_written():
    ctx = _ctx(_struct())
    doc = _clean_doc()
    doc["summary"] = "tightened to 55s; used the slide+VO route"
    st = _gate_state()
    st["craft_surfaced"] = True
    st["conformance_surfaced"] = True
    with mock.patch.object(tools, "_conformance_hints", return_value=["a live advisory hint"]):
        fb = tools._verify_before_finish(doc, ctx, st, [])
    assert fb is None, fb
    assert st["surface_blocked"] == 0, st
    print("ok  gate: Part A does not fire once a real summary is written")


def test_gate_surface_block_is_bounded():
    ctx = _ctx(_struct())
    doc = _clean_doc()
    st = _gate_state()
    st["craft_surfaced"] = True
    st["conformance_surfaced"] = True
    with mock.patch.object(tools, "_conformance_hints", return_value=["a live advisory hint"]):
        fb1 = tools._verify_before_finish(doc, ctx, st, [])
        fb2 = tools._verify_before_finish(doc, ctx, st, [])
        fb3 = tools._verify_before_finish(doc, ctx, st, [])
    assert fb1 and "surface" in fb1.lower(), fb1
    assert fb2 and "surface" in fb2.lower(), fb2
    assert st["surface_blocked"] == tools._SURFACE_MAX_BLOCKS, st
    assert fb3 is None, fb3       # capped -> fail-open, finishes despite the still-empty surface
    print("ok  gate: Part A stops blocking after _SURFACE_MAX_BLOCKS (fail-open, terminates)")


def test_gate_no_surface_block_without_a_compromise():
    ctx = _ctx(_struct())
    doc = _clean_doc()
    st = _gate_state()
    st["craft_surfaced"] = True
    st["conformance_surfaced"] = True
    with mock.patch.object(tools, "_conformance_hints", return_value=[]):
        fb = tools._verify_before_finish(doc, ctx, st, [])
    assert fb is None, fb
    assert st["surface_blocked"] == 0, st
    print("ok  gate: no surface block when no live hint is in play (clean edits aren't nagged)")


# --------------------------------------------------------------------------
# brain_plan_conformance.plan.md §6.4: the deterministic hint helpers
# --------------------------------------------------------------------------

def test_hint_beatsync_fires_when_declared_and_no_op_on_grid():
    # onsets_ms: [0, 500, 1000, 1500, 2000, 2500, 3000] (_music_ctx). A grid
    # entry maps a source's onsets into PROGRAM time relative to the op's OWN
    # from_ms/src_in_ms -- so src_in_ms=100 (between the 0 and 500 onsets)
    # deliberately keeps the op's own from_ms off every mapped onset.
    ctx = _music_ctx()
    doc = {"timeline": [], "operations": [
        {"op_id": "pa1", "type": "place_audio", "source_file_id": "bed1",
         "from_ms": 1234, "to_ms": 3234, "src_in_ms": 100, "src_out_ms": 2100},
    ]}
    plan = {"watch": ["land cuts on the beat grid"]}
    hint = tools._hint_beatsync_declared_but_absent(doc, ctx, plan)
    assert hint and "beat-sync" in hint.lower(), hint
    print("ok  hint: beat-sync declared but no op edge lands on the grid -> fires")


def test_hint_beatsync_silent_when_an_op_lands_on_grid():
    ctx = _music_ctx()
    doc = _bed_doc()            # pa1's from_ms=1000, an actual onset on the grid
    plan = {"watch": ["land cuts on the beat grid"]}
    assert tools._hint_beatsync_declared_but_absent(doc, ctx, plan) is None
    print("ok  hint: beat-sync declared and an op DOES land on the grid -> silent")


def test_hint_beatsync_silent_without_music_or_ops():
    plan = {"watch": ["land cuts on the beat grid"]}
    assert tools._hint_beatsync_declared_but_absent(_bed_doc(), _ctx(_struct()), plan) is None
    assert tools._hint_beatsync_declared_but_absent(
        {"timeline": [], "operations": []}, _music_ctx(), plan) is None
    print("ok  hint: beat-sync silent with no musical source, or no ops at all (no false-fire)")


def _long_cuts_doc(pace_level=None):
    segs = []
    for i in range(3):
        seg = {"seg_id": f"s{i}", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 5000}
        if pace_level and i == 0:
            seg["pace_level"] = pace_level
        segs.append(seg)
    return {"timeline": segs, "operations": [], "brief": {}}


def test_hint_punchy_fires_when_declared_and_cuts_long_and_natural():
    ctx = _ctx(_struct())
    plan = {"purpose": "punchy, quick succession"}
    hint = tools._hint_punchy_declared_but_flat(_long_cuts_doc(), ctx, plan)
    assert hint and "punchy" in hint.lower(), hint
    print("ok  hint: punchy declared but cuts run long at natural pace -> fires")


def test_hint_punchy_silent_when_retimed_faster_or_mostly_tight():
    ctx = _ctx(_struct())
    plan = {"purpose": "punchy, quick succession"}
    assert tools._hint_punchy_declared_but_flat(_long_cuts_doc(pace_level="faster"), ctx, plan) is None
    doc_tight = {"timeline": [
        {"seg_id": "s0", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 1000},
        {"seg_id": "s1", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 1000},
        {"seg_id": "s2", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 5000},
    ], "operations": [], "brief": {}}
    assert tools._hint_punchy_declared_but_flat(doc_tight, ctx, plan) is None
    print("ok  hint: punchy silent when retimed faster, or when the main line is mostly tight")


def test_hint_punchy_silent_when_not_declared():
    ctx = _ctx(_struct())
    plan = {"purpose": "a calm documentary piece"}
    assert tools._hint_punchy_declared_but_flat(_long_cuts_doc(), ctx, plan) is None
    print("ok  hint: punchy hint silent when pacing was never declared")


def test_conformance_hints_fail_open():
    ctx = _ctx(_struct())
    plan = {"purpose": "punchy, quick succession"}
    with mock.patch.object(tools, "_hint_punchy_declared_but_flat", side_effect=RuntimeError("boom")):
        hints = tools._conformance_hints({"timeline": [], "operations": []}, ctx, plan)
    assert hints == [], hints           # the raising hint is swallowed, not propagated
    print("ok  _conformance_hints: a raising hint is swallowed (fail-open)")


# --------------------------------------------------------------------------
# brain_plan_conformance.plan.md §6.5: ladder order + the reply construction
# --------------------------------------------------------------------------

def test_gate_ladder_order_conformance_after_stage3_then_surface():
    """One gate invocation with everything else already surfaced yields Part
    B first (a plan is present); the NEXT invocation, with a live compromise
    and still no surface, yields Part A -- confirming Part B precedes Part A
    within the same finish attempt."""
    ctx = _ctx(_struct())
    doc = _clean_doc()
    doc["plan"] = {"structure": [{"beat": "hook", "need": "required"}],
                   "carries": [], "watch": [], "rev": 1}
    st = _gate_state()
    st["craft_surfaced"] = True
    with mock.patch.object(tools, "_conformance_hints", return_value=["a live advisory hint"]):
        fb1 = tools._verify_before_finish(doc, ctx, st, [])
        assert fb1 and "plan conformance" in fb1.lower(), fb1     # Part B first
        assert st["conformance_surfaced"] and st["surface_blocked"] == 0, st

        fb2 = tools._verify_before_finish(doc, ctx, st, [])
        assert fb2 and "surface" in fb2.lower(), fb2               # THEN Part A, same finish attempt
        assert st["surface_blocked"] == 1, st
    print("ok  gate: ladder order -- Part B (conformance) fires before Part A (surface)")


def test_loop_reply_prefers_wrap_up_summary():
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t2", name="wrap_up", input={
            "summary": "Cut a tight 8s teaser leading with the outage line."})],
        "This internal note should never surface as the reply.",
    ]
    res = tools.run_edit_loop(_ScriptedLLM(script), system="sys",
                              messages=[{"role": "user", "content": "cut a teaser"}],
                              ctx=ctx, document=_seed_doc())
    assert res.reply == "Cut a tight 8s teaser leading with the outage line.", res.reply

    # No wrap_up called at all -> brain_accountability_architecture.plan.md
    # PART 1's ONE finalizer now forces one (synthesized here, since the
    # script offers no more tool calls), so a clean finish NEVER surfaces
    # leftover internal prose, even without an explicit wrap_up call -- a
    # strictly better regression guard than the old "falls back to last_text".
    script2 = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        "Placed the first beat.",   # finish attempt -> Stage 2 (craft) fires
        "Placed the first beat.",   # blind verdict -> voluntary finish, surface still empty
    ]
    res2 = tools.run_edit_loop(_ScriptedLLM(script2), system="sys",
                               messages=[{"role": "user", "content": "cut a teaser"}],
                               ctx=ctx, document=_seed_doc())
    assert res2.reply != "Placed the first beat.", res2.reply
    assert res2.reply.startswith("Assembled 1 cut"), res2.reply

    # An awaiting_user (ask_user) turn still uses last_text, even with a summary.
    script3 = [
        [ToolCall(id="t1", name="wrap_up", input={"summary": "should never surface here"}),
         ToolCall(id="t2", name="ask_user", input={"questions": [
             {"prompt": "Split-screen these two?", "options": ["Yes", "No"]}]})],
    ]
    res3 = tools.run_edit_loop(_ScriptedLLM(script3), system="sys",
                               messages=[{"role": "user", "content": "combine these"}],
                               ctx=ctx, document=_seed_doc())
    assert res3.awaiting_user is True, res3
    assert res3.reply != "should never surface here", res3.reply
    print("ok  loop: the reply prefers a written wrap_up summary; falls back to last_text otherwise")


# --------------------------------------------------------------------------
# brain_loop_convergence.plan.md Part 1: budget/progress + churn detection
# --------------------------------------------------------------------------

def test_edit_fingerprint_ignores_random_ids():
    doc_a = {"timeline": [
        {"seg_id": "s1", "file_id": "f1", "in_ms": 0, "out_ms": 1000, "ref": "f1:m00", "level": "balanced"},
        {"seg_id": "s2", "file_id": "f1", "in_ms": 1000, "out_ms": 2000, "ref": "f1:m01", "level": "balanced"},
    ], "operations": []}
    doc_b = {"timeline": [
        {"seg_id": "zzz1", "file_id": "f1", "in_ms": 0, "out_ms": 1000, "ref": "f1:m00", "level": "balanced"},
        {"seg_id": "zzz2", "file_id": "f1", "in_ms": 1000, "out_ms": 2000, "ref": "f1:m01", "level": "balanced"},
    ], "operations": []}
    assert tools._edit_fingerprint(doc_a) == tools._edit_fingerprint(doc_b), \
        "same spine content, different seg_ids -> equal fingerprints"

    doc_reordered = {"timeline": list(reversed(doc_a["timeline"])), "operations": []}
    assert tools._edit_fingerprint(doc_a) != tools._edit_fingerprint(doc_reordered), \
        "spine ORDER matters -- a reorder is a real change"

    ops = [
        {"op_id": "o1", "type": "place_audio", "source_file_id": "bed1", "from_ms": 0, "to_ms": 1000,
         "src_in_ms": 0, "src_out_ms": 1000, "role": "music", "audio_kind": "bed",
         "gain_db": 0.0, "duck_db": 0.0},
        {"op_id": "o2", "type": "place_video", "source_file_id": "f2", "from_ms": 1000, "to_ms": 2000,
         "src_in_ms": 0, "src_out_ms": 1000},
    ]
    doc_ops = {"timeline": [], "operations": ops}
    doc_ops_reordered = {"timeline": [], "operations": list(reversed(ops))}
    assert tools._edit_fingerprint(doc_ops) == tools._edit_fingerprint(doc_ops_reordered), \
        "op ORDER does not matter -- resolve sorts them"
    print("ok  _edit_fingerprint: ignores seg_id/op_id randomness; spine order matters, op order doesn't")


def test_progress_note_is_convergence_not_countdown():
    note = tools._progress_note(0, 18)
    assert "CONVERGE" in note, note
    assert "reserve" in note.lower(), note
    assert "hurry" not in note.lower(), note
    near_end = tools._progress_note(15, 18)   # used=16, remaining=2 -> the firmer branch
    assert "lock the best version" in near_end, near_end
    assert "hurry" not in near_end.lower(), near_end
    assert "CONVERGE" in near_end, near_end
    print("ok  _progress_note: convergence framing throughout, never a countdown-panic")


def test_loop_pushes_progress_note_each_turn():
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        "Placed it.",
        "Reads fine.",
    ]
    llm = _ScriptedLLM(script)
    tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "place the first beat"}],
                        ctx=ctx, document=_seed_doc())
    last_msg = llm.seen_messages[1][-1]
    texts = [b.get("text", "") for b in last_msg["content"] if isinstance(b, dict) and b.get("type") == "text"]
    assert any("PROGRESS:" in t for t in texts), last_msg
    print("ok  loop: the progress note is pushed as a text block every turn")


def _toggled_doc():
    return {"timeline": [
        {"seg_id": "s0", "file_id": "ffffffff-1111", "in_ms": 0, "out_ms": 2000,
         "ref": "ffffffff:m00", "level": "balanced"},
    ], "operations": [], "brief": {}}


def test_loop_flags_churn_on_return_to_prior_state():
    """A generic, id-independent content-fingerprint churn: toggling a known
    segment's mute state off then back on returns the edit to a state it
    already held -- the churn hint must fire on the step that lands back
    there. Uses set_audio (a known target_id) rather than place/remove/place,
    since a scripted LLM can't reference an id place() mints mid-script --
    same underlying _edit_fingerprint mechanism either way."""
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="set_audio", input={"target_id": "s0", "mute": True})],
        [ToolCall(id="t2", name="set_audio", input={"target_id": "s0", "mute": False})],
        "Back to the original.",
        "Reads fine.",
    ]
    llm = _ScriptedLLM(script)
    tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "try muting it"}],
                        ctx=ctx, document=_toggled_doc())
    # seen_messages[2] is what the THIRD llm.run() call saw -- i.e. everything
    # pushed after the SECOND set_audio call, the one that lands back on the
    # original (pre-turn-0) state.
    last_msg = llm.seen_messages[2][-1]
    texts = [b.get("text", "") for b in last_msg["content"] if isinstance(b, dict) and b.get("type") == "text"]
    assert any("CHURN:" in t for t in texts), last_msg
    print("ok  loop: the churn hint fires when the edit returns to a state it already held")


def test_loop_no_churn_on_a_straight_line_build():
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t2", name="place", input={"ref": "ffffffff:m01", "level": "balanced"})],
        "Built both.",
        "Reads fine.",
    ]
    llm = _ScriptedLLM(script)
    tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "build it"}],
                        ctx=ctx, document=_seed_doc())
    for msgs in llm.seen_messages[1:]:
        last_msg = msgs[-1]
        if last_msg.get("role") != "user":
            continue
        texts = [b.get("text", "") for b in last_msg["content"] if isinstance(b, dict) and b.get("type") == "text"]
        assert not any("CHURN:" in t for t in texts), last_msg
    print("ok  loop: a straight-line build (no revert) never flags churn")


def test_churn_hint_dedups_per_state():
    """A 4-round True/False/True/False mute toggle: the FIRST return to the
    original state (round 2, landing back on mute=False) fires CHURN; the
    THIRD return to that SAME state (round 4) stays silent -- deduped per
    state, not per occurrence."""
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="set_audio", input={"target_id": "s0", "mute": True})],    # round 1
        [ToolCall(id="t2", name="set_audio", input={"target_id": "s0", "mute": False})],   # round 2 -> churn (back to original)
        [ToolCall(id="t3", name="set_audio", input={"target_id": "s0", "mute": True})],    # round 3 -> churn (back to round-1 state)
        [ToolCall(id="t4", name="set_audio", input={"target_id": "s0", "mute": False})],   # round 4 -> SAME state as round 2, deduped
        "Done toggling.",
        "Reads fine.",
    ]
    llm = _ScriptedLLM(script)
    tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "try toggling it"}],
                        ctx=ctx, document=_toggled_doc())
    # seen_messages[2] (after round 2) and [3] (after round 3) each carry a
    # fresh CHURN (two DIFFERENT states); seen_messages[4] (after round 4,
    # landing back on round 2's already-flagged state) must NOT.
    def _churn_present(idx):
        last_msg = llm.seen_messages[idx][-1]
        texts = [b.get("text", "") for b in last_msg["content"] if isinstance(b, dict) and b.get("type") == "text"]
        return any("CHURN:" in t for t in texts)

    assert _churn_present(2), llm.seen_messages[2]
    assert _churn_present(3), llm.seen_messages[3]
    assert not _churn_present(4), llm.seen_messages[4]
    print("ok  loop: churn hint dedups per STATE -- a re-recurring state only nags once")


# --------------------------------------------------------------------------
# brain_loop_convergence.plan.md Part 2: all termination paths finalize truthfully
# --------------------------------------------------------------------------

class _RaisingAfterNLLM:
    """Like _ScriptedLLM, but RAISES once the script is exhausted (instead of
    defaulting to "Done.") -- used to simulate the forced finalization step's
    own llm.run() call failing (brain_loop_convergence.plan.md Part 2's
    fail-open test)."""
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    @property
    def model(self):
        return "raising-after-n"

    def run(self, *, system, messages, tools=None, max_tokens=2048, cache_system=False):
        if self.calls >= len(self.script):
            raise RuntimeError("boom (simulated finalize failure)")
        step = self.script[self.calls]
        self.calls += 1
        if isinstance(step, str):
            return LLMResponse(text=step, tool_calls=[], stop_reason="end_turn",
                               assistant_message={"role": "assistant", "content": step})
        blocks = [{"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input} for tc in step]
        return LLMResponse(text="", tool_calls=step, stop_reason="tool_use",
                           assistant_message={"role": "assistant", "content": blocks})


def test_loop_cap_path_forces_a_wrap_up():
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t2", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t3", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        # the forced finalization step (call index 3, past the max_turns=3 cap):
        [ToolCall(id="t4", name="wrap_up", input={
            "summary": "Assembled a quick sequence; ran out of turns to polish further."})],
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "cut it"}],
                              ctx=ctx, document=_seed_doc(), max_turns=3)
    assert res.reply == "Assembled a quick sequence; ran out of turns to polish further.", res.reply
    assert res.document.get("summary") == res.reply, res.document
    assert res.changed is True, res
    assert llm.calls == 4, llm.calls   # 3 build turns (the cap) + 1 forced finalization step
    print("ok  loop: the max_turns cap path forces a wrap_up; the reply is that summary")


def test_loop_cap_path_synthesizes_summary_when_brain_declines():
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t2", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t3", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        "I couldn't quite finish everything.",   # finalization step declines to wrap_up
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "cut it"}],
                              ctx=ctx, document=_seed_doc(), max_turns=3)
    assert (res.document.get("summary") or "").startswith("Assembled 3 cut"), res.document.get("summary")
    assert res.reply == res.document["summary"], res.reply
    assert "I couldn't quite finish" not in res.reply, res.reply
    print("ok  loop: when the brain declines to wrap_up at the cap, a deterministic summary is synthesized")


def test_fallback_summary_names_live_compromise():
    ctx = _ctx(_struct())
    doc = _clean_doc()
    with mock.patch.object(tools, "_conformance_hints",
                           return_value=["some ceiling detail -- fix or accept it"]):
        summary = tools._fallback_summary(doc, ctx)
    assert "Ran out of editing room before fully resolving" in summary, summary
    assert "some ceiling detail" in summary, summary

    with mock.patch.object(tools, "_conformance_hints", return_value=[]):
        summary2 = tools._fallback_summary(doc, ctx)
    assert summary2.endswith("Ran out of editing room before a final self-review."), summary2
    print("ok  _fallback_summary: names a live compromise when one exists, else degrades to the plain size line")


def test_voluntary_clean_finish_skips_finalization():
    """brain_accountability_architecture.plan.md PART 1: _finalize_turn costs
    NO extra llm.run() call when there is genuinely nothing outstanding --
    which requires BOTH a clean mandatory ladder AND a non-empty surface (an
    empty surface is itself outstanding, per PART 1's design: every
    substantive finish must recap, not only a compromise-driven one)."""
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        "Placed it.",       # finish attempt -> Stage 2 (craft) fires
        [ToolCall(id="t2", name="wrap_up", input={"summary": "Reads clean."})],  # satisfies the mandatory surface
        "Reads clean.",     # blind verdict -> finish, voluntary + gated + surfaced
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "cut it"}],
                              ctx=ctx, document=_seed_doc())
    assert res.reply == "Reads clean.", res.reply
    assert llm.calls == 4, llm.calls   # Stage 2 + wrap_up each bought a turn; no extra forced-finalization call after
    print("ok  loop: a clean, surfaced voluntary finish never triggers the forced finalizer")


def test_finalization_is_fail_open():
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t2", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t3", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        # no 4th entry -> the finalization step's own llm.run() call raises
    ]
    llm = _RaisingAfterNLLM(script)
    with mock.patch.object(tools, "_fallback_summary", side_effect=RuntimeError("boom (simulated fallback failure)")):
        res = tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "cut it"}],
                                  ctx=ctx, document=_seed_doc(), max_turns=3)
    assert res.changed is True, res
    assert not (res.document.get("summary") or "").strip(), res.document
    assert res.reply == "Done.", res.reply   # both layers failed -> last_text("") -> the generic default
    print("ok  loop: finalization is fail-open -- a raising llm call AND a raising fallback still return cleanly")


# --------------------------------------------------------------------------
# brain_loop_convergence.plan.md Part 3: tools fail loud and immediately
# --------------------------------------------------------------------------

def test_trim_a2_collapse_rejects_loud():
    """Footgun (a): an A2 trim that collapses the span used to silently no-op."""
    ctx = _music_ctx()
    doc = _bed_doc()   # pa1: src_in_ms=0, src_out_ms=2000
    obs, new, changed = tools._dispatch("trim", {"target_id": "pa1", "out_ms": 0}, ctx, doc)
    assert changed is False, obs
    assert new is doc
    assert '"applied": false' in obs, obs
    assert "collapses" in obs and "delta" in obs, obs
    print("ok  dispatch: an A2 trim that collapses the span rejects loud with a delta/absolute hint")


def test_trim_unchanged_span_is_not_applied():
    ctx = _music_ctx()
    doc = _bed_doc()   # pa1: src_in_ms=0, src_out_ms=2000
    obs, new, changed = tools._dispatch(
        "trim", {"target_id": "pa1", "in_ms": 0, "out_ms": 2000}, ctx, doc)
    assert changed is False, obs
    assert new is doc
    assert '"applied": false' in obs, obs
    assert "unchanged" in obs, obs
    print("ok  dispatch: trim to an identical span is applied=false (regression guard for the old false true)")


def _duration_ctx(durations):
    struct = _struct()
    return observe.EditContext(
        file_ids=["ffffffff-1111"], index=_MapIndex(struct), map_struct=struct,
        durations=durations, dup_groups=[])


def test_trim_absolute_as_delta_rejects():
    """Footgun (b): an absolute source timestamp passed as delta_out_ms."""
    ctx = _duration_ctx({"bed1": 2000})   # bed1 is only 2000ms long
    doc = _bed_doc()   # pa1: source_file_id=bed1, src_out_ms=2000
    obs, new, changed = tools._dispatch(
        "trim", {"target_id": "pa1", "delta_out_ms": 50000}, ctx, doc)
    assert changed is False, obs
    assert new is doc
    assert '"applied": false' in obs, obs
    assert "past this source" in obs and "absolute" in obs.lower(), obs
    print("ok  dispatch: an absolute-looking delta_out_ms past the source length rejects loud")


def test_trim_domain_guard_fail_open():
    ctx = _ctx(_struct())   # no known duration for "bed1" at all
    doc = _bed_doc()
    assert tools._trim_domain_reason(doc, ctx, "pa1", None, None, None, 100) is None
    obs, new, changed = tools._dispatch("trim", {"target_id": "pa1", "delta_out_ms": 100}, ctx, doc)
    assert changed is True, obs   # the verb ran normally -- the guard deferred
    print("ok  _trim_domain_reason: unknown source duration -> None, the verb runs normally")


def test_edit_reject_surfaces_as_applied_false():
    ctx = _ctx(_struct())
    doc = _seed_doc()
    try:
        act.trim(doc, "nope", delta_in_ms=100)
        raise AssertionError("act.trim on an unknown id should raise EditReject")
    except act.EditReject as r:
        assert "no main-line cut" in r.reason, r.reason
    obs, new, changed = tools._dispatch("trim", {"target_id": "nope", "delta_in_ms": 100}, ctx, doc)
    assert changed is False, obs
    assert new is doc
    assert '"applied": false' in obs, obs
    assert "no main-line cut" in obs, obs
    print("ok  EditReject: a direct act.trim raises; through _dispatch it becomes applied=false + reason")


def test_unknown_id_verbs_reject_loud():
    ctx = _ctx(_struct())
    doc = _seed_doc()
    cases = [
        ("remove", {"target_id": "nope"}),
        ("move", {"target_id": "nope", "to_index": 0}),
        ("set_gain", {"target_id": "nope", "gain_db": -3.0}),
        ("duck", {"target_id": "nope", "amount_db": -6.0}),
        ("fade_audio", {"target_id": "nope", "in_ms": 200}),
    ]
    for name, args in cases:
        obs, new, changed = tools._dispatch(name, args, ctx, doc)
        assert changed is False, (name, obs)
        assert new is doc, (name, obs)
        assert '"applied": false' in obs, (name, obs)
        assert "nope" in obs, (name, obs)
    print("ok  dispatch: a representative sweep of verbs reject loud with an id-naming reason on a bogus id")


# --------------------------------------------------------------------------
# brain_accountability_architecture.plan.md PART 1: the done-gate as a
# declared, ordered ladder of stages (not nested early-returns)
# --------------------------------------------------------------------------

def test_gate_stage_registry_is_well_formed():
    keys = [st.key for st in tools._GATE_STAGES]
    assert len(keys) == len(set(keys)), keys
    labels = [st.label for st in tools._GATE_STAGES]
    assert len(labels) == len(set(labels)), labels
    for st in tools._GATE_STAGES:
        assert st.kind in (tools.MANDATORY, tools.ADVISORY), st
    mandatory_labels = {st.label for st in tools._GATE_STAGES if st.kind == tools.MANDATORY}
    assert mandatory_labels == {"structural", "length", "intent", "craft", "conformance", "surface"}, \
        mandatory_labels
    gate_state_keys = set(_gate_state().keys())
    for st in tools._GATE_STAGES:
        assert st.key in gate_state_keys, (st.key, gate_state_keys)
    print("ok  _GATE_STAGES: distinct keys/labels, valid kinds, the mandatory set is exactly what it should be")


def test_advisory_skip_skips_only_the_advisory_stage():
    """With steps=["place","review"] (self-reviewed) on a doc that would trip
    BOTH Stage 3 (flags) and conformance, the ladder must skip ONLY the
    advisory flags stage -- conformance (mandatory) still fires."""
    ctx = _ctx(_struct())
    orig = observe.diagnose
    observe.diagnose = lambda *a, **k: [
        {"severity": "warn", "anchor": "cuts 1-2", "message": "same speaker back-to-back"}]
    try:
        doc = _clean_doc()
        doc["plan"] = {"structure": [{"beat": "hook", "need": "required"}],
                       "carries": [], "watch": [], "rev": 1}
        st = _gate_state()
        st["craft_surfaced"] = True
        gc = tools._gate_ctx(doc, ctx, st, ["place", "review"])
        assert gc.advisory_skip is True, gc
        fb = tools._run_gate_stages(gc)
        assert fb and "plan conformance" in fb.lower(), fb
        flags_entry = next(e for e in gc.fired if e["stage"] == "flags")
        assert flags_entry == {"stage": "flags", "kind": "advisory", "fired": False,
                               "why": "advisory-skip: brain self-reviewed"}, flags_entry
        conformance_entry = next(e for e in gc.fired if e["stage"] == "conformance")
        assert conformance_entry["fired"] is True, conformance_entry
    finally:
        observe.diagnose = orig
    print("ok  gate: advisory-skip filters ONLY the advisory flags stage; conformance still fires")


def test_conformance_and_surface_run_after_self_review():
    """The direct regression for Bypass 2: previously `if reviewed or
    state["reviewed"]: return None` sat BEFORE conformance and surface, so any
    self-reviewed turn killed both. They now run to completion regardless."""
    ctx = _ctx(_struct())
    doc = _clean_doc()
    doc["plan"] = {"structure": [{"beat": "hook", "need": "required"}],
                   "carries": [], "watch": [], "rev": 1}
    st = _gate_state()
    st["craft_surfaced"] = True
    steps = ["place", "review"]        # self-reviewed -- used to kill both stages below
    with mock.patch.object(tools, "_conformance_hints", return_value=["a live advisory hint"]):
        fb1 = tools._verify_before_finish(doc, ctx, st, steps)
        assert fb1 and "plan conformance" in fb1.lower(), fb1
        assert st["conformance_surfaced"], st

        fb2 = tools._verify_before_finish(doc, ctx, st, steps)
        assert fb2 and "surface" in fb2.lower(), fb2
        assert st["surface_blocked"] == 1, st

        fb3 = tools._verify_before_finish(doc, ctx, st, steps)
        assert fb3 and "surface" in fb3.lower(), fb3
        assert st["surface_blocked"] == 2, st

        fb4 = tools._verify_before_finish(doc, ctx, st, steps)
        assert fb4 is None, fb4          # bounded -> the loop still terminates
    print("ok  gate: Bypass 2 regression -- conformance then surface run to completion after self-review")


def test_intent_stage_fires_after_self_review():
    """Bypass 2b direct regression: review_out used to be computed only when
    NOT self-reviewed, so Stage 1's ask-flags were always empty on a
    self-reviewed turn. _gate_ctx now computes review() unconditionally."""
    ctx = _ctx(_struct())
    doc = _clean_doc()
    orig = observe.review
    observe.review = lambda *a, **k: {"flags": [
        {"category": "ask", "message": "no split screen found, but the ask named one"}]}
    try:
        st = _gate_state()
        st["craft_surfaced"] = True
        fb = tools._verify_before_finish(doc, ctx, st, ["place", "diagnose"])
        assert fb and "stage 1" in fb.lower(), fb
        assert st["intent_surfaced"], st
    finally:
        observe.review = orig
    print("ok  gate: Bypass 2b regression -- Stage 1 sees ask-flags even on a self-reviewed turn")


def test_gate_stage_error_does_not_suppress_later_stages():
    ctx = _ctx(_struct())
    doc = _clean_doc()
    doc["plan"] = {"structure": [{"beat": "hook", "need": "required"}],
                   "carries": [], "watch": [], "rev": 1}
    orig_stages = tools._GATE_STAGES
    raising_craft = tools._Stage("craft_surfaced", tools.MANDATORY, "craft",
                                 lambda gc: (_ for _ in ()).throw(RuntimeError("boom")))
    tools._GATE_STAGES = tuple(raising_craft if st.label == "craft" else st for st in orig_stages)
    try:
        st = _gate_state()
        with mock.patch.object(tools, "_conformance_hints", return_value=["a live advisory hint"]):
            gc = tools._gate_ctx(doc, ctx, st, [])
            outstanding = tools._run_gate_stages(gc, kinds=(tools.MANDATORY,), collect=True)
        craft_entry = next(e for e in gc.fired if e["stage"] == "craft")
        assert craft_entry == {"stage": "craft", "kind": "mandatory", "fired": False, "why": "error"}, craft_entry
        assert any("plan conformance" in fb.lower() for fb in outstanding), outstanding
        assert st["conformance_surfaced"], st
    finally:
        tools._GATE_STAGES = orig_stages
    print("ok  gate: a raising stage fails open without suppressing the stages after it")


def test_collect_mode_returns_all_outstanding_mandatory():
    ctx = _ctx(_struct())
    doc = _clean_doc()
    doc["plan"] = {"structure": [{"beat": "hook", "need": "required"}],
                   "carries": [], "watch": [], "rev": 1}
    st = _gate_state()
    st["craft_surfaced"] = True
    orig = observe.diagnose
    observe.diagnose = lambda *a, **k: [
        {"severity": "warn", "anchor": "whole", "message": "over target: 8.0s vs 5.0s"}]
    try:
        gc = tools._gate_ctx(doc, ctx, st, [])
        outstanding = tools._run_gate_stages(gc, kinds=(tools.MANDATORY,), collect=True)
    finally:
        observe.diagnose = orig
    assert len(outstanding) >= 2, outstanding
    assert any("length" in fb.lower() for fb in outstanding), outstanding
    assert any("plan conformance" in fb.lower() for fb in outstanding), outstanding
    assert not any("stage 3" in fb.lower() for fb in outstanding), outstanding   # advisory never collected
    print("ok  _run_gate_stages: collect mode gathers every outstanding mandatory stage, never the advisory one")


def main():
    test_loop_reads_then_places_then_replies()
    test_loop_pushes_the_mirror_after_a_tool_call()
    test_loop_pure_chat_no_change()
    test_loop_bad_tool_is_noop()
    test_loop_ask_user_pauses_turn()
    test_normalize_questions_surfaces_a_valid_recommendation()
    test_normalize_questions_drops_a_recommendation_not_in_options()
    test_normalize_questions_still_drops_under_two_options()
    test_normalize_questions_no_recommendation_omits_the_keys()
    test_loop_split_screen_after_answer()
    test_dispatch_set_plan_applies_and_returns_the_rendered_plan()
    test_loop_pushes_the_plan_mirror_after_a_set_plan_call()
    test_loop_pushed_plan_mirror_is_loud_when_the_edit_moved_without_a_plan()
    test_wrap_up_writes_surface_fields()
    test_dispatch_wrap_up_applies_and_echoes_surface()
    test_dispatch_inspect_cut_returns_windowed_payload_and_leaves_doc_unchanged()
    test_trim_snap_beat_lands_an_ops_resulting_edge_on_the_grid()
    test_trim_snap_beat_is_a_noop_pass_through_without_music()
    test_move_shift_to_align_slides_an_op_by_the_computed_offset()
    test_gate_blocks_structural_error()
    test_gate_clean_doc_finishes()
    test_gate_length_is_fix_or_justify_once()
    test_gate_craft_check_fires_once_unconditionally()
    test_gate_advisory_review_skipped_when_already_diagnosed()
    test_gate_jump_cut_finding_folds_into_stage_3_no_separate_continuity_stage()
    test_gate_blocks_a_mid_program_audio_gap_unless_surfaced()
    test_gate_forces_length_reconcile_end_to_end()
    test_gate_surfaces_review_flags_end_to_end()
    test_latest_user_text_reads_the_newest_user_message()
    test_gate_flags_a_requested_feature_missing_end_to_end()
    test_gate_does_not_flag_a_feature_the_user_never_asked_for()
    test_gate_conformance_fires_once_with_required_beats_and_watch()
    test_gate_conformance_noops_without_a_plan()
    test_gate_conformance_old_shape_structure_of_strings()
    test_gate_blocks_finish_when_compromise_in_play_and_surface_empty()
    test_gate_surface_block_clears_once_summary_written()
    test_gate_surface_block_is_bounded()
    test_gate_no_surface_block_without_a_compromise()
    test_hint_beatsync_fires_when_declared_and_no_op_on_grid()
    test_hint_beatsync_silent_when_an_op_lands_on_grid()
    test_hint_beatsync_silent_without_music_or_ops()
    test_hint_punchy_fires_when_declared_and_cuts_long_and_natural()
    test_hint_punchy_silent_when_retimed_faster_or_mostly_tight()
    test_hint_punchy_silent_when_not_declared()
    test_conformance_hints_fail_open()
    test_gate_ladder_order_conformance_after_stage3_then_surface()
    test_loop_reply_prefers_wrap_up_summary()
    test_edit_fingerprint_ignores_random_ids()
    test_progress_note_is_convergence_not_countdown()
    test_loop_pushes_progress_note_each_turn()
    test_loop_flags_churn_on_return_to_prior_state()
    test_loop_no_churn_on_a_straight_line_build()
    test_churn_hint_dedups_per_state()
    test_loop_cap_path_forces_a_wrap_up()
    test_loop_cap_path_synthesizes_summary_when_brain_declines()
    test_fallback_summary_names_live_compromise()
    test_voluntary_clean_finish_skips_finalization()
    test_finalization_is_fail_open()
    test_trim_a2_collapse_rejects_loud()
    test_trim_unchanged_span_is_not_applied()
    test_trim_absolute_as_delta_rejects()
    test_trim_domain_guard_fail_open()
    test_edit_reject_surfaces_as_applied_false()
    test_unknown_id_verbs_reject_loud()
    test_gate_stage_registry_is_well_formed()
    test_advisory_skip_skips_only_the_advisory_stage()
    test_conformance_and_surface_run_after_self_review()
    test_intent_stage_fires_after_self_review()
    test_gate_stage_error_does_not_suppress_later_stages()
    test_collect_mode_returns_all_outstanding_mandatory()
    print("\nall tool-loop tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
