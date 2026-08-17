#!/usr/bin/env python3
"""SESSION-LEVEL tests for the edit tool-loop's done-gate (brain_accountability_
architecture.plan.md PART 1) -- no network, no DB.

Per the plan's own "Testing lesson": unit tests prove the gate WORKS WHEN
CALLED; they never proved it GETS CALLED in a realistic session, which is
exactly how two dead-code bugs (Bypass 2, Bypass 2b) shipped "verified". Every
test here drives the FULL `tools.run_edit_loop` with a scripted fake LLM and
asserts the mandatory invariants ACTUALLY FIRED, read off `res.trace`'s
`kind:"gate"` entry -- never inferred from a helper's return value. Covers the
three previously-bypassing shapes: a self-reviewing turn (Bypass 2 + 2b), a
cap-exhaustion run (Bypass 1), and a last-turn voluntary finish (Bypass 3).

Run:  PYTHONPATH=. python scripts/test_loop_sessions.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuses the shared fixture builders + the no-DB stub test_tools_loop already
# sets up (fm._sentences_for_file) -- same style, no duplication.
from test_tools_loop import _ctx, _ScriptedLLM, _seed_doc, _struct  # noqa: E402

from app.services.l3 import tools  # noqa: E402
from app.services.llm import ToolCall  # noqa: E402


def test_session_selfreview_turn_still_runs_conformance_and_surface():
    """The exact session shape that shipped 'verified' while both stages were
    dead code (Bypass 2): a turn that self-reviews (calls `review`) and has a
    plan with a required beat + a live compromise hint, reaching the done-gate
    via cap exhaustion so the trace's ONE kind:"gate" entry (recorded by
    _finalize_turn) reflects a single fresh pass over the WHOLE mandatory
    ladder in one shot."""
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="set_plan", input={
            "structure": [{"beat": "hook", "need": "required"}]})],
        [ToolCall(id="t2", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t3", name="review", input={})],
    ]
    llm = _ScriptedLLM(script)
    with mock.patch.object(tools, "_conformance_hints", return_value=["a live advisory hint"]):
        res = tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "cut it"}],
                                  ctx=ctx, document=_seed_doc(), max_turns=3)
    gate_entries = [e for e in res.trace if e.get("kind") == "gate"]
    assert len(gate_entries) == 1, res.trace
    fired = {s["stage"]: s["fired"] for s in gate_entries[0]["stages"]}
    assert fired.get("conformance") is True, fired
    assert fired.get("surface") is True, fired
    # the reply is a real wrap_up recap (synthesized here, since the brain
    # never called wrap_up itself), never the leftover "Done." default.
    assert res.reply != "Done.", res.reply
    assert res.reply == res.document.get("summary"), res.reply
    print("ok  session: a self-reviewing turn still runs conformance + surface (Bypass 2 regression)")


def test_session_cap_exhaustion_runs_mandatory_stages():
    """Bypass 1 direct regression: a run that exhausts max_turns while ALWAYS
    making tool calls used to leave via `for...else` and skip the entire
    ladder -- the done-gate was never reachable at all."""
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t2", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        [ToolCall(id="t3", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "cut it"}],
                              ctx=ctx, document=_seed_doc(), max_turns=3)
    gate_entries = [e for e in res.trace if e.get("kind") == "gate"]
    assert len(gate_entries) == 1, res.trace
    assert gate_entries[0]["exit_reason"] == "cap", gate_entries[0]
    stages_seen = {s["stage"] for s in gate_entries[0]["stages"]}
    mandatory_labels = {st.label for st in tools._GATE_STAGES if st.kind == tools.MANDATORY}
    assert mandatory_labels <= stages_seen, (mandatory_labels, stages_seen)
    print("ok  session: cap exhaustion still runs the full mandatory ladder (Bypass 1 regression)")


def test_session_last_turn_finish_runs_mandatory_stages():
    """Bypass 3 direct regression: a voluntary finish landing exactly on the
    LAST turn used to skip `_verify_before_finish` entirely (gated on
    `turn < max_turns - 1`), yet still set `finished_clean=True` -- which ALSO
    suppressed the post-loop finalizer. The gate now runs regardless of which
    turn it lands on, and the ONE finalizer still enforces the surface."""
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        "Done, reads clean.",     # the finish attempt lands exactly on turn == max_turns - 1
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "cut it"}],
                              ctx=ctx, document=_seed_doc(), max_turns=2)
    gate_entries = [e for e in res.trace if e.get("kind") == "gate"]
    assert len(gate_entries) == 1, res.trace
    assert gate_entries[0]["exit_reason"] == "voluntary", gate_entries[0]
    assert (res.document.get("summary") or "").strip(), res.document
    assert res.reply == res.document["summary"], res.reply
    print("ok  session: a last-turn voluntary finish still runs the mandatory ladder (Bypass 3 regression)")


def test_session_clean_finish_costs_no_extra_llm_call():
    """A voluntary finish with nothing outstanding AND a non-empty surface
    (the brain called wrap_up itself) must cost NO extra llm.run() call --
    proving the finalizer did not become an unconditional extra round trip."""
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"})],
        "Placed it.",                                                            # finish #1 -> Stage 2 fires
        [ToolCall(id="t2", name="wrap_up", input={"summary": "Reads clean."})],   # satisfies the mandatory surface
        "Reads clean.",                                                          # finish #2 -> clean + surfaced
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "cut it"}],
                              ctx=ctx, document=_seed_doc())
    assert res.reply == "Reads clean.", res.reply
    assert llm.calls == 4, llm.calls        # exactly the scripted rounds -- no extra finalize call
    gate_entries = [e for e in res.trace if e.get("kind") == "gate"]
    assert len(gate_entries) == 1, res.trace
    assert not any(s["fired"] for s in gate_entries[0]["stages"]), gate_entries[0]   # nothing outstanding
    print("ok  session: a clean, surfaced voluntary finish costs no extra llm call")


def test_session_ask_user_pause_defers_finalization():
    """A paused ask_user turn keeps its own question framing -- the ONE
    finalizer must not force a wrap_up on a turn that isn't over yet."""
    ctx = _ctx(_struct())
    script = [
        [ToolCall(id="t1", name="place", input={"ref": "ffffffff:m00", "level": "balanced"}),
         ToolCall(id="t2", name="ask_user", input={"questions": [
             {"prompt": "Split-screen these two?", "options": ["Yes", "No"]}]})],
    ]
    llm = _ScriptedLLM(script)
    res = tools.run_edit_loop(llm, system="sys", messages=[{"role": "user", "content": "combine these"}],
                              ctx=ctx, document=_seed_doc())
    assert res.awaiting_user is True, res
    assert res.reply == "Before I go further I need your call on a couple of things below.", res.reply
    assert (res.document.get("summary") or "") == "", res.document   # no forced wrap_up
    gate_entries = [e for e in res.trace if e.get("kind") == "gate"]
    assert len(gate_entries) == 1, res.trace
    assert gate_entries[0]["exit_reason"] == "asked", gate_entries[0]
    assert gate_entries[0].get("note", "").startswith("paused for ask_user"), gate_entries[0]
    print("ok  session: an ask_user pause defers finalization entirely (its own question framing kept)")


def main():
    test_session_selfreview_turn_still_runs_conformance_and_surface()
    test_session_cap_exhaustion_runs_mandatory_stages()
    test_session_last_turn_finish_runs_mandatory_stages()
    test_session_clean_finish_costs_no_extra_llm_call()
    test_session_ask_user_pause_defers_finalization()
    print("\nall loop-session tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
