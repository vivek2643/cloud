# EDSO: intent-contract + done-gate (anchor & verify the edit loop)

## Goal
Stop the editing loop from "finishing" just because the model went quiet. Give it a
lightweight **contract** (the ask, made explicit) and a **done-gate** that checks the
edit against that contract before the turn ends. Reuse what already exists
(`brief`, `observe.validate`, `observe.diagnose`). **No new tools. No new machinery.**

## Why (current failure mode)
- `tools.run_edit_loop` ends the moment the model replies with no tool call
  (`if not resp.tool_calls: break`). Nothing is checked against the ask.
- `observe.validate` (structural, render-breaking) and `observe.diagnose` (editorial:
  same-speaker runs, low-energy runs, redundant takes, over/under target length) already
  exist and are pure/cheap — but they're **opt-in**, so the model can skip them.
- `document["brief"]["target_duration_s"]` is seeded to `None` (`converse._seed_document`)
  and **never written**, so `diagnose`'s length check never fires today.

## Design decisions (already settled — do NOT re-litigate)
1. **Contract UX**: proceed and edit, but the model restates its read of the goal in prose;
   use `ask_user` only when the ask is materially ambiguous.
2. **Gate strictness**: structural errors AND over-target length are enforced
   (fix-or-justify); same-speaker/low-energy/redundant takes stay advisory.
3. **No target given**: do NOT invent one. `done = structurally clean + one diagnose review`.
   The model may `ask_user` for a rough length only if genuinely stuck.
4. **Scope**: implement all of it together (the pieces are interdependent).

## Non-goals
- No new tools; no `set_brief` tool. The length target is parsed deterministically from the
  user's own words (more robust than trusting the model to record it), and it can only be
  SET from an explicit number — never invented.
- No changes to cut boundaries, identity, pass1/pass2, or the beat index (the verbatim
  transcript / `vis:` / `aud:` surfacing is already landed in the prompt).

---

## Change 1 — `backend/app/services/l3/tools.py`: the done-gate

`observe` is already imported (`from app.services.l3 import act, observe`). `user_message`
is already imported and accepts a plain string.

### 1a. Add the gate helper immediately ABOVE `def run_edit_loop(...)`:

```python
_STRUCT_MAX_TRIES = 3


def _verify_before_finish(working: dict, ctx: EditContext,
                          state: Dict[str, Any], steps: List[str]) -> str | None:
    """The done-gate: when the brain tries to FINISH a turn that changed the edit,
    check the result against the contract. Returns feedback the brain MUST act on
    (the loop keeps going), or None to let it finish. HARD on structural legality
    (would break the render); the editorial findings are surfaced for a single
    review -- an over-target LENGTH is fix-or-justify, the rest is advisory -- so
    the loop always terminates. A brain that already called ``diagnose``/``validate``
    this turn has self-reviewed, so the advisory nudge is skipped."""
    issues = observe.validate(working, ctx)
    if issues and state["struct_tries"] < _STRUCT_MAX_TRIES:
        state["struct_tries"] += 1
        body = "; ".join(f"{i.get('kind')} {i.get('id')}: {i.get('message')}"
                         for i in issues[:8])
        return ("AUTOMATIC CHECK -- structural problems that would break the render. "
                "Fix these before finishing:\n" + body)

    findings = observe.diagnose(working, ctx)
    over = [f for f in findings if "over target" in (f.get("message") or "")]
    if over and not state["length_surfaced"]:
        state["length_surfaced"] = True
        return ("AUTOMATIC CHECK -- length: " + (over[0].get("message") or "") +
                ". Either trim to the target, or say in one line why this length is "
                "right, then finish.")

    rest = [f for f in findings if "target" not in (f.get("message") or "")]
    reviewed = "diagnose" in steps or "validate" in steps
    if rest and not state["reviewed"] and not reviewed:
        state["reviewed"] = True
        body = "\n".join(
            f"- [{f.get('severity') or 'info'}] "
            + (f"{f['anchor']}: " if f.get("anchor") else "") + (f.get("message") or "")
            for f in rest[:12])
        return ("AUTOMATIC CHECK -- review the edit against the ask before you finish. "
                "This is advisory: act on what serves the goal, ignore the rest, then "
                "finish:\n" + body)
    return None
```

### 1b. Wire it into the finish branch of `run_edit_loop`.

Find:
```python
    last_text = ""
    questions: List[dict] = []

    for turn in range(max_turns):
        resp = llm.run(system=system, messages=convo, tools=tools,
                       max_tokens=max_tokens, cache_system=True)
        last_text = (resp.text or "").strip() or last_text
        convo.append(resp.assistant_message)
        if not resp.tool_calls:
            break
```
Replace with:
```python
    last_text = ""
    questions: List[dict] = []
    verify = {"struct_tries": 0, "length_surfaced": False, "reviewed": False}

    for turn in range(max_turns):
        resp = llm.run(system=system, messages=convo, tools=tools,
                       max_tokens=max_tokens, cache_system=True)
        last_text = (resp.text or "").strip() or last_text
        convo.append(resp.assistant_message)
        if not resp.tool_calls:
            # Finish attempt: don't let a changed edit exit unchecked against the
            # contract (structural = hard, length = fix-or-justify, rest advisory).
            if changed and turn < max_turns - 1:
                feedback = _verify_before_finish(working, ctx, verify, steps)
                if feedback is not None:
                    convo.append(user_message(feedback))
                    continue
            break
```

**Termination guarantees:** structural forces at most `_STRUCT_MAX_TRIES` (3) extra turns;
length surfaces at most once (`length_surfaced`); advisory review at most once (`reviewed`).
Unchanged (chat-only) turns skip the gate entirely — no extra cost, no behavior change.

---

## Change 2 — `backend/app/services/l3/converse.py`: the anchor (write the target)

### 2a. Add `import re` to the top import block (after `import os`).

### 2b. Add these helpers (place them just below `_seed_document`):

```python
_DUR_RE = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>s(?:ec(?:onds?)?)?|m(?:in(?:ute)?s?)?)\b",
    re.IGNORECASE)
_WORD_MIN_RE = re.compile(r"\b(?:a|one)\s+minute\b", re.IGNORECASE)


def _extract_target_s(text: str) -> Optional[float]:
    """Best-effort target LENGTH (seconds) parsed from the user's OWN words -- e.g.
    '60s', '90 seconds', '2 min', 'a minute', '30-45s' (upper bound). Returns None
    when no explicit length is stated: we never INVENT a target (design choice B)."""
    if not text:
        return None
    best: Optional[float] = None
    for m in _DUR_RE.finditer(text):
        num = float(m.group("num"))
        secs = num * 60.0 if m.group("unit").lower().startswith("m") else num
        best = secs if best is None else max(best, secs)   # a range -> upper bound
    if best is None and _WORD_MIN_RE.search(text):
        best = 60.0
    return best


def _latest_user_text(messages: List[dict]) -> str:
    """The newest user message as plain text (content may be a bare string or a
    list of blocks -- see store.load_messages)."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(b.get("text", "") for b in c
                            if isinstance(b, dict) and b.get("type") == "text")
    return ""
```

### 2c. In `respond(...)`, after `working` is established and before the loop, set the target.

Find:
```python
    working = document if isinstance(document, dict) else _seed_document(file_ids)
    max_tokens = settings.autoedit_max_output_tokens
```
Replace with:
```python
    working = document if isinstance(document, dict) else _seed_document(file_ids)
    # Anchor the contract: capture an EXPLICIT target length from the user's latest
    # words into the brief so diagnose + the done-gate have something to check
    # against. Only ever SET from a stated number -- never cleared, never invented.
    _target_s = _extract_target_s(_latest_user_text(messages))
    if _target_s is not None:
        working.setdefault("brief", {})["target_duration_s"] = _target_s
    max_tokens = settings.autoedit_max_output_tokens
```

**False-positive note:** a phrase like "the first 30 seconds drag" would set target=30. This is
acceptable — the gate surfaces length only once as fix-or-justify, and the model (which sees
the real user message) can justify and finish. Keep it simple; do not build intent detection.

---

## Change 3 — `backend/app/services/l3/converse.py`: prompt norms (`_LOOP_SYSTEM`)

Two small additions to the `_LOOP_SYSTEM` string.

### 3a. CONTRACT + proportionality — append right after the sentence ending
`"...make the edit, then tell them what you did in a sentence or two. Use only ids that appear below.\n\n"`:

```python
    "THE ASK IS YOUR CONTRACT. On a turn that changes the edit, open by restating "
    "in ONE line how you read the goal -- the length if it was given, the must-haves, "
    "and the tone -- then edit to THAT. If the ask is materially ambiguous, or you'd "
    "need a rough length you cannot infer, use ask_user before editing rather than "
    "guessing big. Match ambition to the ask: make the SIMPLEST edit that satisfies "
    "the goal, and don't add layers, cutaways, effects, or pacing moves the ask "
    "didn't call for.\n\n"
```

### 3b. FINISHING — append at the very end of `_LOOP_SYSTEM`, after
`"...and edit verbs are described in the tools; call them as you need."` (before the closing `)`):

```python
    "\n\nFINISHING. When you stop editing you'll get one AUTOMATIC CHECK of the edit "
    "against the contract. Never finish with STRUCTURAL problems -- fix them. If "
    "you're over a target length, either trim to it or say in one line why the "
    "current length is right. The rest (speaker runs, low-energy stretches, "
    "redundant takes) is advisory -- act on what serves the goal, ignore the rest."
```

---

## Change 4 — `backend/scripts/test_tools_loop.py`: tests

Follow the file's existing style (plain asserts + `main()`, scripted `_ScriptedLLM`).
`_struct()` builds two same-speaker cuts but their moment nodes use `"speaker"` (not
`"speaker_person"`), so `diagnose` finds NO same-speaker run there — existing tests are
unaffected by the gate. `total_ms` comes straight from segment durations, so **over-target**
is a deterministic end-to-end signal.

### 4a. Unit tests for `_verify_before_finish` (stub `observe.validate`/`observe.diagnose`):

```python
def _gate_state():
    return {"struct_tries": 0, "length_surfaced": False, "reviewed": False}


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
        fb1 = tools._verify_before_finish({"timeline": [{"seg_id": "s"}]}, ctx, st, [])
        assert fb1 and "length" in fb1.lower() and st["length_surfaced"], fb1
        fb2 = tools._verify_before_finish({"timeline": [{"seg_id": "s"}]}, ctx, st, [])
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
        fb = tools._verify_before_finish({"timeline": [{"seg_id": "s"}]}, ctx, st, [])
        assert fb and "advisory" in fb.lower() and st["reviewed"], fb
        st2 = _gate_state()
        fb2 = tools._verify_before_finish({"timeline": [{"seg_id": "s"}]}, ctx, st2,
                                          ["place", "diagnose"])
        assert fb2 is None, fb2          # brain self-reviewed -> no forced nudge
    finally:
        observe.diagnose = orig
    print("ok  gate: advisory review fires once, skipped if brain already diagnosed")
```

### 4b. Integration test (real loop, deterministic over-target):

```python
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
```

### 4c. Register all five in `main()`:

```python
    test_gate_blocks_structural_error()
    test_gate_clean_doc_finishes()
    test_gate_length_is_fix_or_justify_once()
    test_gate_advisory_review_skipped_when_already_diagnosed()
    test_gate_forces_length_reconcile_end_to_end()
```

### 4d. Add a `_converse` target-parse unit test.
Either extend `test_tools_loop.py` (add `from app.services.l3 import converse`) or create
`backend/scripts/test_converse_target.py`:

```python
from app.services.l3 import converse

def test_extract_target_variants():
    f = converse._extract_target_s
    assert f("make it 60s") == 60.0
    assert f("about 90 seconds punchy") == 90.0
    assert f("2 min recap") == 120.0
    assert f("keep it to a minute") == 60.0
    assert f("30-45s teaser") == 45.0          # range -> upper bound
    assert f("just make an edit") is None      # no number -> never invented
    print("ok  converse: target length parsed from the user's words")
```

---

## Run
```bash
cd backend
PYTHONPATH=. python scripts/test_tools_loop.py
PYTHONPATH=. python scripts/test_converse_target.py   # if created separately
PYTHONPATH=. python scripts/test_observe_act.py        # regression: senses unchanged
```
All must print `ok ...` / `all tool-loop tests passed`.

## Acceptance
- Loop refuses to finish a *changed* edit with structural errors (up to 3 fix attempts).
- Over-target length (when a target exists) is surfaced once as fix-or-justify.
- Other findings are surfaced once as advisory, and skipped if the brain already called
  `diagnose`/`validate` this turn.
- Chat-only / unchanged turns are unaffected (no extra LLM turn).
- A stated length ("60s", "a minute", "30-45s") lands in `brief.target_duration_s`;
  no length is ever invented.
```
