# Brain GATE INTEGRITY — separate stage EVALUATION from stage SATISFACTION, and gate on COVERAGE

Status: proposal, READ-ONLY handoff. **Implement from another chat. Do NOT
implement from this doc; do NOT commit from this doc.**

**Scope.** Two architecture-level changes to the done-gate in
`backend/app/services/l3/tools.py`:

- **§1 — evaluate/commit split.** Today a gate stage marks itself satisfied
  *inside its own evaluation function*, before the caller has decided whether
  to deliver the message. On one real path the message is thrown away and the
  mark survives. §1 makes stages pure and moves the mark to the one place that
  knows the message was delivered. **This lands FIRST, before any new stage is
  registered anywhere** — including the stages that
  `brain_accountability_architecture.plan.md` PARTS 2 and 3 add.
  **§1.8 is part of the same change**: the split also makes it possible to say
  "this invariant could not be evaluated" without marking it satisfied, which at
  HEAD is impossible — an exception inside any of the six MANDATORY stages is
  recorded identically to that stage passing clean, and the edit finishes.
- **§2 — the coverage gate.** A new MANDATORY stage that answers the question
  nothing in the system currently asks: *you are about to finish having never
  looked at X, and X is in play.*

Plus two small items that belong here because they are the same shape:
**§3** wraps the one unguarded context injection in the loop, and **§4**
records what is explicitly NOT in this plan.

**Sequencing (recorded identically in all three plans).**

1. `junk_span_suppression.plan.md` — independent, does not touch L3.
2. **`brain_gate_integrity.plan.md` §1** (evaluate/commit split) — must precede
   any new stage, in this plan or any other.
3. `brain_perception_integrity.plan.md` — independent of the accountability
   parts, and unblocks them (PARTS 2–4 all consume perception).
4. **`brain_gate_integrity.plan.md` §2** (coverage gate).
5. `brain_accountability_architecture.plan.md` PART 2, then PART 3, then
   PARTS 4/5 (PART 5 is recommended EARLIER — see that plan's amendment §A.5).
6. Beat binding + the adversarial quote check, last
   (`brain_accountability_architecture.plan.md` amendment §A.6).

**Cross-references.** `brain_perception_integrity.plan.md` §4 introduces a
sense STATUS envelope; §1.6 below is the gate-side consumer of it and depends
on it. `brain_accountability_architecture.plan.md` PART 1 is the code this plan
modifies; its amendment §A.1–A.6 records the decisions that changed since it
was written.

---

## 0. Verified current state (checked against HEAD, commit `3337c31`)

### 0.1 The ladder as it stands

Seven stages, declared as data at `tools.py:1150-1158`:

```1150:1158:backend/app/services/l3/tools.py
_GATE_STAGES: Tuple[_Stage, ...] = (
    _Stage("struct_tries",         MANDATORY, "structural",  _stage_structural),
    _Stage("length_surfaced",      MANDATORY, "length",      _stage_length),
    _Stage("intent_surfaced",      MANDATORY, "intent",      _stage_intent),
    _Stage("craft_surfaced",       MANDATORY, "craft",       _stage_craft),
    _Stage("reviewed",             ADVISORY,  "flags",       _stage_flags),
    _Stage("conformance_surfaced", MANDATORY, "conformance", _stage_conformance),
    _Stage("surface_blocked",      MANDATORY, "surface",     _stage_surface),
)
```

`_Stage` is frozen with fields `key` (the `state` key the stage OWNS), `kind`,
`label`, `fn` (`tools.py:998-1003`). `_GateCtx` (`tools.py:1006-1026`) carries
`working`, `ctx`, `state`, `steps`, `user_ask`, `advisory_skip`, `findings`,
`review_out`, `fired`.

`_run_gate_stages` (`tools.py:1161-1200`) evaluates the ladder in declared
order in one of two modes — `collect=False` returns the FIRST stage's feedback
and stops (`tools.py:1197-1198`); `collect=True` evaluates every in-scope stage
and returns all outstanding feedback (`tools.py:1199`).

Two callers: `_verify_before_finish` (`tools.py:1203-1211`, interactive mode)
and `_finalize_turn` (`tools.py:1256`, enforcement mode restricted to
`kinds=(MANDATORY,)`).

### 0.2 The bug: satisfaction is written inside evaluation, and one caller discards the message

```1413:1427:backend/app/services/l3/tools.py
        if not resp.tool_calls:
            # Finish attempt. The gate now runs on EVERY finish attempt, the last
            # turn included (the old `turn < max_turns - 1` guard is what made a
            # last-turn finish skip the ladder entirely). On the last turn there
            # is no room to iterate, so feedback is not injected; the stages have
            # still marked `verify`, and _finalize_turn enforces whatever is left
            # outstanding.
            edit_moved = changed   # PART 4 will define edit_moved properly; until then an alias
            if edit_moved:
                feedback = _verify_before_finish(working, ctx, verify, steps, user_ask)
                if feedback is not None and turn < max_turns - 1:
                    convo.append(user_message(feedback))
                    continue
            exit_reason = "voluntary"
            break
```

On `turn == max_turns - 1`, `_verify_before_finish` runs, a stage fires, the
stage writes its own `state` key, and then `feedback` is **discarded** because
`turn < max_turns - 1` is False. The comment at `tools.py:1414-1419` claims
"`_finalize_turn` enforces whatever is left outstanding". **That claim is false
for four of the seven stages.** The state key is already set, so the
enforcement pass at `tools.py:1256` collects nothing from them.

The state write inside the evaluation function, stage by stage:

| stage | key | write site | shape | last-turn consequence |
|---|---|---|---|---|
| `structural` | `struct_tries` | `tools.py:1055` `gc.state["struct_tries"] += 1` | counter, bound `_STRUCT_MAX_TRIES = 3` (`tools.py:805`) | re-fires on the enforcement pass, but **one of three budgeted attempts is silently consumed by a message nobody read** |
| `length` | `length_surfaced` | `tools.py:1066` `gc.state["length_surfaced"] = True` | boolean once-fired | **message permanently lost** |
| `intent` | `intent_surfaced` | `tools.py:1079` | boolean once-fired | **message permanently lost** (this is the MANDATORY fit-to-ask invariant) |
| `craft` | `craft_surfaced` | `tools.py:1089` | boolean once-fired | **message permanently lost** |
| `flags` | `reviewed` | `tools.py:1113` | boolean once-fired | advisory; the enforcement pass runs `kinds=(MANDATORY,)` so it is recorded `why: "out-of-scope"` (`tools.py:1180-1181`) — not affected by this bug, but `reviewed` doubles as the `advisory_skip` input at `tools.py:1035`, so firing it also disables itself for the rest of the turn |
| `conformance` | `conformance_surfaced` | `tools.py:957` (inside `_plan_conformance`) | boolean once-fired | **message permanently lost** |
| `surface` | `surface_blocked` | `tools.py:1138` `+= 1` | counter, bound `_SURFACE_MAX_BLOCKS = 2` (`tools.py:810`) | re-fires once more, but **one of two budgeted blocks is consumed by a discarded message** |

So: four stages lose their message outright and two spend a budgeted attempt on
a message that was never delivered. `_record_gate` (`tools.py:1214-1229`) then
writes the *enforcement pass's* `gc.fired`, which reports `fired: false` for a
stage that DID fire moments earlier — and `_record_gate`'s own docstring
(`tools.py:1218-1221`) calls itself "what makes 'the mandatory invariant
ACTUALLY fired' assertable". **The one piece of evidence we built to detect a
dead gate is itself lying.**

> **Correction to the framing this plan was commissioned under.** The brief
> said "every stage mutates its `state` key … `_finalize_turn` … finds the
> state already marked, collects nothing" and "All 7 stages do it". The
> *coupling* is universal — all seven stages write `gc.state` inside their
> evaluation function, which is the class. The *outcome* is not uniform:
> exactly four stages lose the message permanently, two burn a budgeted
> attempt, and one is advisory and out of scope for the enforcement pass. The
> fix below is the same either way, but the verification in §1.7 must assert
> the per-stage outcomes as they actually are, or it will assert something
> false about `structural` and `surface` and be "fixed" by weakening it.

### 0.3 The same shape, second instance: `_finalize_after_loop`

```1319:1329:backend/app/services/l3/tools.py
        if outstanding:
            instruct += "\n\n" + "\n\n".join(outstanding)
        convo.append(user_message(instruct))
        resp = llm.run(system=system, messages=convo, tools=tools,
                       max_tokens=max_tokens, cache_system=True)
        for tc in (resp.tool_calls or []):
            if tc.name == "ask_user":       # finalization never pauses for a question
                continue
            _obs, working, _did = _dispatch(tc.name, tc.input or {}, ctx, working, user_ask)
    except Exception:
        logger.exception("tools: forced finalization step failed (continuing)")
```

`convo.append` happens BEFORE `llm.run`, so a raised `llm.run` leaves the
instruction in `convo` but discards the whole `outstanding` list — and the state
keys that produced it were already marked by the collection pass at
`tools.py:1256`. Same class: **the mark outlives the delivery.**

Note the ordering subtlety that makes this worse than it looks: the message IS
appended to `convo` before the raise. So the brain will see the text on some
future turn if the conversation is resumed — but the *outstanding* list is gone
from this turn's control flow, and nothing re-derives it. §1.5 fixes this by the
same construction, not with a second try/except.

### 0.4 The coverage gap

`steps` is built at `tools.py:1368` (`steps: List[str] = []`), appended at
`tools.py:1431` (`steps.append(tc.name)` — for EVERY tool call, senses and edit
verbs alike), threaded into `_verify_before_finish` (`tools.py:1422`) and
`_finalize_turn` (`tools.py:1256` via `_gate_ctx`), and stored on
`_GateCtx.steps` (`tools.py:1017`).

It is read at **exactly one place**:

```1033:1035:backend/app/services/l3/tools.py
    advisory_skip = bool(
        "diagnose" in steps or "validate" in steps or "review" in steps
        or state.get("reviewed"))
```

That is the only consumer, and it only ever RELAXES the gate. **Nothing in the
system asks what the brain never looked at.** The system prompt makes looking
discretionary:

```187:189:backend/app/services/l3/converse.py
    "extremely careful when you do. Your senses (read_state, predict, validate, "
    "diagnose, affordances, audio_state, review, read_transcript) and edit verbs "
    "are described in the tools; call them as you need.\n\n"
```

Observed consequences in edit thread `34bb9967-97aa-4818-9f21-01145156a9bb`:

- `audio_state` was never called. The brain dismissed a usable royalty-free
  music bed it had never seen, telling the user it "didn't fit". The asset was
  sitting in `ctx.audio_assets` (`observe.py:78`, populated at
  `observe.py:142-146`) and would have been listed by `audio_state`'s `assets`
  key (`observe.py:1009`) and counted in `affordances`' `audio_assets`
  (`observe.py:1866`).
- It was handed a duplicate-take group and inspected exactly one member before
  calling it "the strongest".
- `read_transcript` returned `{"text": "", "segments": []}` and it never noticed
  across 18 reasoning steps. (`brain_perception_integrity.plan.md` §4 is the
  other half of that one — an empty payload that could mean "no speech placed"
  or "the transcript load failed" is unactionable by any gate. §2.4 below
  depends on that plan for the `wording` subject.)

### 0.5 The one unguarded injection

Every other per-turn injection in the loop is either pure-by-construction or
wrapped. `mirror_text` is neither:

```1480:1489:backend/app/services/l3/tools.py
        mirror_text = observe.mirror_text(working, ctx)
        if mirror_text:
            results.append(text_block(mirror_text))
        # brain_plan_mechanism.plan.md §4.2: the PLAN mirror, pushed the same
        # way -- so a set_plan this round is reflected immediately, and building
        # WITHOUT a plan is nudged (building=changed makes the absence loud only
        # once the edit has actually moved this turn).
        plan_text = observe.plan_mirror_text(working, building=changed)
        if plan_text:
            results.append(text_block(plan_text))
```

`observe.mirror_text` → `observe.mirror` (`observe.py:1620`) calls
`feel.simulate`, `feel.join_continuity`, `feel._low_energy_runs`,
`cutrecord_map.speech_words_in_span`, `footage_map._said_text_for_span`, and
`cutrecord_map.speech_boundary_flags` — two of which hit the database. A raise
kills the whole turn's `convo.append(user_message(results))` at
`tools.py:1497`, losing every tool result for the step, not just the mirror.
The reasoning capture (`tools.py:1403-1412`), the churn detection
(`tools.py:1452-1472`) and every gate stage (`tools.py:1187-1193`) are all
wrapped; this is the outlier. `plan_mirror_text` is pure (document-only,
`observe.py:1734-1773`) and needs no wrap, but §3 wraps both for uniformity —
a rule with an exception is a rule nobody applies to stage #8.

---

## 1. §1 — Split stage EVALUATION from stage SATISFACTION

**This lands first, alone, and is independently landable and testable.**

### 1.1 The class, stated once

> **A stage's satisfaction record must be written by the code that DELIVERS the
> stage's message, never by the code that COMPUTES it.**

Every instance of the bug in §0.2 and §0.3 is one violation of that sentence.
Nothing else needs to be understood to review the change.

**The false-genericity trap.** The tempting fix is to change
`tools.py:1423` from `if feedback is not None and turn < max_turns - 1:` to `if
feedback is not None:` — deliver the message on the last turn too. That is
*exactly* the shape of fix the user has rejected: it makes the one observed
symptom go away, it is two tokens, and it leaves the coupling in place. It is
also wrong on its own terms, because on the last turn there is no room to
iterate, so injecting gate feedback the brain cannot act on just wastes the
final step. A second tempting patch is to reset the state keys after a discarded
message; that is a compensating write, needs one branch per stage, and the next
stage author will forget it. **Neither makes it structurally impossible for
stage #8 to reintroduce the bug. The split does.**

A third trap, subtler: adding an `if not committed: state[key] = ...` guard
*inside* each stage function. That keeps the write in the wrong place and just
adds a flag; the whole point is that the stage function must have no access to
the thing it would write.

### 1.2 Stages become pure

Change the contract on `_Stage.fn` from "(`_GateCtx`) -> `str | None`, may write
`gc.state`" to "(`_GateCtx`) -> `str | None`, **MUST NOT write `gc.state`**".
Rewrite each of the seven stage functions to read its own key and return, with
no assignment. Mechanical, one line removed per stage:

```python
def _stage_length(gc: _GateCtx) -> str | None:
    """PURE (gate_integrity §1.2): reads gc.state, NEVER writes it. Whether this
    stage counts as satisfied is decided by the caller that actually delivers
    the message -- see _commit_stages. A stage that writes its own `state` key
    marks itself satisfied on a path where the message is thrown away."""
    over = [f for f in gc.findings if "over target" in (f.get("message") or "")]
    if not over or gc.state["length_surfaced"]:
        return None
    return ("AUTOMATIC CHECK -- length: " + (over[0].get("message") or "") +
            ". Either trim to the target, or say in one line why this length is "
            "right, then finish.")
```

`_stage_structural` (`tools.py:1051-1059`), `_stage_intent`
(`tools.py:1072-1083`), `_stage_craft` (`tools.py:1086-1096`), `_stage_flags`
(`tools.py:1099-1120`) and `_stage_surface` (`tools.py:1127-1143`) get the
identical treatment: delete the one mutation line, keep every returned STRING
byte-for-byte identical (the 12 existing gate tests assert on those strings).

`_stage_conformance` (`tools.py:1123-1124`) is the awkward one: the mutation
lives inside `_plan_conformance` at `tools.py:957`
(`state["conformance_surfaced"] = True`), which also reads it at
`tools.py:943-944`. `_plan_conformance` is a public-ish helper used nowhere
else in `backend/app` (verify with a grep before editing; at HEAD its only
callers are `_stage_conformance` at `tools.py:1124` and the tests). Change its
signature to take `state` as a read-only mapping and delete line 957, keeping
the early-return at 943-944. Its docstring's "Fires at most once
(state-tracked)" sentence must be updated to say the once-only property is now
enforced by the committer, not by the function.

### 1.3 Satisfaction becomes a declared MUTATION, not an assignment

The `_Stage` dataclass gains one field describing HOW the stage is satisfied, so
the committer needs no per-stage branch:

```python
# tools.py, beside _Stage. Satisfaction is DATA: a stage declares how its own
# `state` key records "this fired", and the COMMITTER applies it -- so there is
# exactly one place in the file that writes gate state, and a stage function
# literally cannot mark itself.
COMMIT_ONCE, COMMIT_COUNT = "once", "count"


@dataclass(frozen=True)
class _Stage:
    key: str
    kind: str
    label: str
    fn: Any                      # (_GateCtx) -> str | None -- PURE, no state writes
    commit: str = COMMIT_ONCE    # COMMIT_ONCE -> state[key] = True
                                 # COMMIT_COUNT -> state[key] += 1
```

`_GATE_STAGES` becomes:

```python
_GATE_STAGES: Tuple[_Stage, ...] = (
    _Stage("struct_tries",         MANDATORY, "structural",  _stage_structural, COMMIT_COUNT),
    _Stage("length_surfaced",      MANDATORY, "length",      _stage_length),
    _Stage("intent_surfaced",      MANDATORY, "intent",      _stage_intent),
    _Stage("craft_surfaced",       MANDATORY, "craft",       _stage_craft),
    _Stage("reviewed",             ADVISORY,  "flags",       _stage_flags),
    _Stage("conformance_surfaced", MANDATORY, "conformance", _stage_conformance),
    _Stage("surface_blocked",      MANDATORY, "surface",     _stage_surface,    COMMIT_COUNT),
)
```

`COMMIT_ONCE` is the default precisely so that a new stage written without
thinking about it gets the safe shape (a boolean that only ever gets set once,
after delivery). The two counters are declared explicitly because they are the
exception.

### 1.4 `_run_gate_stages` returns `(feedback, stages_to_commit)`

```python
def _run_gate_stages(gc: _GateCtx, *, kinds=(MANDATORY, ADVISORY),
                     collect: bool = False) -> Tuple[Any, List[_Stage]]:
    """Evaluate the declared ladder in order and return
    (feedback, stages_to_commit) -- NEVER writing gc.state.

    `stages_to_commit` is the ordered list of stages whose feedback is part of
    the returned value. The caller commits them with _commit_stages AFTER the
    message has actually been appended to `convo`, and NOT AT ALL if it decides
    to drop it (gate_integrity §1). This is what makes "the mark outlived the
    delivery" structurally impossible rather than a thing each stage remembers.

    Semantics are otherwise unchanged from PART 1:
      * a stage whose `kind` is not in `kinds` is not evaluated (recorded as
        skipped-by-scope);
      * an ADVISORY stage is skipped when `gc.advisory_skip` -- the ONLY skip
        mechanism, and it can never reach a MANDATORY stage;
      * a stage that raises is logged and treated as "nothing to say"
        (fail-open, house rule) -- it can never suppress the stages after it;
      * collect=False (interactive): (first stage's feedback or None, [stage] or []);
      * collect=True (enforcement): (list of every outstanding feedback, every
        stage that produced one).
    Every outcome is appended to gc.fired for the trace."""
```

Body: identical to `tools.py:1177-1200` except that every `return`/`append`
carries the stage alongside its feedback. `gc.fired` bookkeeping
(`tools.py:1180-1194`) is untouched — including `fired: fb is not None`, which
now means "this stage had something to say", which is the honest reading. §1.6
adds one field to distinguish that from "…and it was delivered".

```python
def _commit_stages(state: Dict[str, Any], stages: List[_Stage]) -> None:
    """Record that these stages' feedback was DELIVERED. The ONLY writer of gate
    satisfaction state in this module. Fail-open: a commit error must never
    alter the edit or the reply, and an uncommitted stage simply fires again
    next evaluation -- which is the safe direction (a stage that nags twice is
    a nuisance; a stage that never fires is the bug class this closes)."""
    for st in stages:
        try:
            if st.commit == COMMIT_COUNT:
                state[st.key] = int(state.get(st.key) or 0) + 1
            else:
                state[st.key] = True
        except Exception:
            logger.exception("tools: gate commit failed for stage %s", st.label)
```

The "safe direction" note is load-bearing: if this plan's own change is
half-applied, the failure mode is a duplicated nudge, not a silent skip.

### 1.5 The two call sites, and the two discard paths

`_verify_before_finish` (`tools.py:1203-1211`) must stop being the place that
decides. Its signature is asserted unchanged by the existing tests
(`brain_accountability_architecture.plan.md` §"This is NOT a rebuild"), so keep
it as a thin wrapper that returns feedback only, and add a second entry point
the loop uses:

```python
def _verify_before_finish(working: dict, ctx: EditContext,
                          state: Dict[str, Any], steps: List[str],
                          user_ask: str = "") -> str | None:
    """The done-gate, interactive mode. Kept for signature compatibility with
    the existing gate tests: it evaluates AND commits in one call, which is
    correct ONLY when the caller unconditionally delivers the message. The loop
    itself uses _gate_once (below), which hands the commit back to the caller.
    This function MUST NOT grow an early return."""
    fb, to_commit = _gate_once(working, ctx, state, steps, user_ask)
    _commit_stages(state, to_commit)
    return fb


def _gate_once(working: dict, ctx: EditContext, state: Dict[str, Any],
               steps: List[str], user_ask: str = ""
               ) -> Tuple[str | None, List[_Stage]]:
    """Evaluate the ladder WITHOUT committing. The caller commits only if it
    actually delivers the feedback."""
    return _run_gate_stages(_gate_ctx(working, ctx, state, steps, user_ask))
```

The loop's finish branch becomes:

```python
            edit_moved = changed   # PART 4 will define edit_moved properly; until then an alias
            if edit_moved:
                feedback, to_commit = _gate_once(working, ctx, verify, steps, user_ask)
                if feedback is not None and turn < max_turns - 1:
                    convo.append(user_message(feedback))     # PART 5: _inject_message
                    _commit_stages(verify, to_commit)        # committed ONLY now
                    continue
                # Last turn: there is no room to iterate, so the feedback is NOT
                # injected -- and therefore NOT committed either. _finalize_turn's
                # enforcement pass re-derives it from unchanged state and presents
                # it through _finalize_after_loop, which is the only place that
                # can still act on it. (The old comment here claimed exactly this
                # while the stages had already marked themselves; that is what
                # made four MANDATORY stages silently no-op on the last turn.)
            exit_reason = "voluntary"
            break
```

`_finalize_turn` (`tools.py:1248-1266`) becomes:

```python
        gc = _gate_ctx(working, ctx, state, steps, user_ask)
        outstanding, to_commit = _run_gate_stages(gc, kinds=(MANDATORY,), collect=True)
        _record_gate(trace, exit_reason=exit_reason, gc=gc)
        if outstanding or _surface_is_empty(working):
            working = _finalize_after_loop(
                llm, system=system, convo=convo, ctx=ctx, working=working,
                tools=tools, verify=state, user_ask=user_ask, trace=trace,
                outstanding=outstanding, committer=lambda: _commit_stages(state, to_commit),
                max_tokens=max_tokens)
```

and `_finalize_after_loop` (`tools.py:1291-1337`) calls `committer()` on the
line immediately after `convo.append(user_message(instruct))` succeeds and
BEFORE `llm.run` — because at that point the text is genuinely in the brain's
context, which is the definition of delivered. If `llm.run` then raises, the
stages stay committed (correct: the brain was told) but the existing
`except Exception` at `tools.py:1328-1329` no longer silently loses an
uncommitted `outstanding` list, because there is no longer an uncommitted list
to lose. **This is why §0.3 needs no separate patch: it is the same violation
and the same fix.**

> Do NOT pass `to_commit` into `_finalize_after_loop` as a list and commit it
> there directly. That would put a second writer of gate state in a second
> function, which is the thing being removed. A zero-argument `committer`
> callable keeps `_commit_stages(state, …)` the single writer and makes the
> call site read as "deliver, then record delivery".

If `outstanding` is empty but `_surface_is_empty(working)` is True,
`_finalize_after_loop` still runs (existing behaviour, `tools.py:1258`) and
`committer()` commits an empty list — a no-op, which is correct.

### 1.6 The trace stops lying, and `unavailable` stops passing the gate

Two additions to `_GateCtx.fired`'s entries:

1. **`delivered`.** `fired` currently means "the stage had something to say".
   Add `"delivered": bool` set by `_commit_stages` writing back into the
   matching `gc.fired` entry (match on `stage` label; `_commit_stages` gains an
   optional `gc` parameter for this and stays fail-open). A last-turn
   evaluation then records `fired: true, delivered: false`, which is the truth
   and is exactly the assertion §1.7 V1 needs. **Without this field there is no
   way to write a test that distinguishes the fixed behaviour from the broken
   behaviour**, because in both cases *some* `_record_gate` entry says
   `fired: false`.

2. **`blocked_by_unavailable`.** This depends on
   `brain_perception_integrity.plan.md` §4 (the sense status envelope) and
   lands with it, not with §1. Today:

```1041:1048:backend/app/services/l3/tools.py
    try:
        review_out = observe.review(working, ctx, user_ask=user_ask)
    except Exception:
        logger.exception("_gate_ctx: review failed (continuing without it)")
        review_out = None
    return _GateCtx(working=working, ctx=ctx, state=state, steps=steps,
                    user_ask=user_ask, advisory_skip=advisory_skip,
                    findings=findings, review_out=review_out)
```

and the MANDATORY `_stage_intent` reads it as

```1075:1077:backend/app/services/l3/tools.py
    ask_flags = [f for f in (gc.review_out or {}).get("flags", [])
                 if f.get("category") == "ask"]
    if not ask_flags or gc.state["intent_surfaced"]:
        return None
```

so a `review` failure makes a MANDATORY invariant pass. That is the PART 1
bypass shape arriving through the DATA path instead of the control path, and
it must be closed in the same file that closes the control path. Once
`observe.review` returns a status (`observed` / `empty` / `unavailable`), the
rule is:

```python
# tools.py -- a MANDATORY stage may not be satisfied by a sense that did not
# run. "The invariant found nothing" and "the invariant could not be evaluated"
# are different answers and only one of them may pass the gate.
def _stage_intent(gc: _GateCtx) -> str | None:
    if _unavailable(gc.review_out):
        if gc.state["intent_surfaced"]:
            return None
        return ("AUTOMATIC CHECK -- Stage 1, fit to your ask: I could not read "
                "the assembled program back (" + _why(gc.review_out) + "), so "
                "whether the features you named actually landed is UNKNOWN, not "
                "confirmed. Verify by hand -- call read_state and check each "
                "feature you promised is present -- then finish, or say plainly "
                "that you could not verify it.")
    ...
```

Note what this deliberately does NOT do: it does not hard-block the loop
forever on a degraded sense. A `review` outage would then make every edit
unfinishable. It converts a silent pass into a stated uncertainty the brain must
answer, bounded by the same once-fired key. **`brain_perception_integrity.plan.md`
§4.5 owns the `_unavailable` / `_why` helpers and the exact status shape; this
section owns the policy that consumes them.** Ship them together or the status
field is the "field every caller ignores" trap that plan names.

### 1.7 Verification (§1), with revert signatures

Every item below must be observed **RED against HEAD** before the change lands.
A test that has never been seen failing against the old code proves nothing —
that is how four unreachable features shipped "verified".

**V1 — reachability of the bug itself, then of the fix.** In
`scripts/test_tools_loop.py`, a session test driving `run_edit_loop` with a
`_ScriptedLLM` whose script is: turn 0 `place` (so `changed` is True), then
`max_turns - 1` no-tool prose steps, so the finish attempt lands on
`turn == max_turns - 1`. Run with `max_turns` small (e.g. 3) and a document
whose `brief.target_duration_s` forces `diagnose`'s over-target finding
(`observe.py:1230-1232`) so `_stage_length` genuinely has something to say.
Assert, on the returned `trace`:

- exactly two `kind: "gate"` entries exist — the last-turn evaluation and the
  `_finalize_turn` enforcement pass;
- in the FIRST, the `length` stage has `fired: true, delivered: false`;
- in the SECOND, the `length` stage has `fired: true` — i.e. it is STILL
  outstanding, because nothing committed it;
- the finalize instruction the brain received (captured from the scripted LLM's
  recorded `messages`, or from the `kind:"injected"` `source:"finalize"` entry
  once PART 5 lands) CONTAINS the substring `"AUTOMATIC CHECK -- length:"`.

*Revert signature:* restore `gc.state["length_surfaced"] = True` inside
`_stage_length` and the second gate entry reports `fired: false` and the
finalize instruction no longer contains the length text. Both assertions fail.
This test is RED against HEAD today for exactly that reason.

**V2 — the interactive path is byte-identical.** A session test with
`max_turns = 6` where the brain attempts to finish on turn 1: assert the gate
feedback string it receives is unchanged from HEAD (capture HEAD's string into
the fixture first), that `length_surfaced` is True in the returned state
afterwards, and that the stage does NOT fire a second time on a later finish
attempt. *Revert signature:* if `_commit_stages` is never called on the
delivering path, the same stage fires on every finish attempt and the
"does not fire twice" assertion fails — which is the opposite failure and
catches a mis-wired committer.

**V3 — stages are pure, asserted structurally.** A test that, for each entry in
`_GATE_STAGES`, calls `st.fn(gc)` against a frozen copy of `gc.state` and
asserts `gc.state` is byte-identical afterwards (compare a `dict` snapshot).
Run it for all seven, table-driven off `_GATE_STAGES` so a new stage is covered
automatically. *Revert signature:* this is the assertion that makes stage #8
impossible to write wrong — any state write inside any stage function fails it,
including one added a year from now by someone who never read this document.
It is RED against HEAD for five of seven stages (`structural`, `length`,
`intent`, `craft`, `flags`) plus `conformance` via `_plan_conformance`, and
GREEN for none.

**V4 — the counters stop being spent on undelivered messages.** Drive the
last-turn path (V1's setup) with a document that has structural issues
(`observe.validate` non-empty — e.g. a `layout_regions` entry referencing a
missing layer, `observe.py:1188-1190`). Assert `verify["struct_tries"] == 0`
after the discarded last-turn evaluation and `== 1` only after
`_finalize_after_loop` has delivered. *Revert signature:* HEAD increments it on
the discarded evaluation, so `== 0` fails.

**V5 — `_finalize_after_loop`'s raise no longer loses the list.** Unit test with
an `llm` stub whose `run` raises: assert `committer()` was called (the stages
ARE marked, because the instruction reached `convo`), that `working` is
returned unmutated apart from the deterministic `_fallback_summary` path
(`tools.py:1332-1336`), and that the reply is non-empty. *Revert signature:*
move the `committer()` call after `llm.run` and the stages come back
uncommitted while the message is in `convo` — assert on the state and it fails.

**V6 — the MANDATORY set has not been silently demoted.** Extend the existing
test that asserts the MANDATORY set (PART 1's "silently demoting an invariant to
advisory fails CI") to also assert every stage's `commit` value, so
`COMMIT_ONCE` → `COMMIT_COUNT` (which would make a gating stage unbounded) or
the reverse (which would break a bound) cannot be changed silently.

### 1.8 The gate's OWN fail-opens — an errored MANDATORY stage currently PASSES

§1.6 closes the *data* path: a MANDATORY stage may not be satisfied by a sense
that did not run. There is a second, independent path with the same outcome, and
it is inside the ladder runner itself. It must be closed in the same change or
§1.6 is a half-fix that a single `raise` walks around.

#### 1.8.1 The site

```1187:1196:backend/app/services/l3/tools.py
        try:
            fb = st.fn(gc)
        except Exception:
            logger.exception("gate stage %s failed (fail-open)", st.label)
            gc.fired.append({"stage": st.label, "kind": st.kind, "fired": False,
                             "why": "error"})
            continue
        gc.fired.append({"stage": st.label, "kind": st.kind, "fired": fb is not None})
        if fb is None:
            continue
```

A stage function that RAISES is recorded exactly the way a stage that had
nothing to say is recorded — `fired: False` — and the loop `continue`s to the
next stage. For `structural`, `length`, `intent`, `craft`, `conformance` and
`surface`, all six MANDATORY, **an exception inside the invariant is
indistinguishable from the invariant holding**, and the edit finishes. The only
difference between "clean" and "crashed" is a `why` string in a trace field that
`_record_gate` writes and nothing asserts on, and a line in the log.

This is not a hypothetical. Each MANDATORY stage function calls into perception
on live documents — `_stage_length` reads `gc.findings`, `_stage_craft` and
`_stage_conformance` walk `plan` structures authored by the model, and
`_plan_conformance` reaches all the way into `observe.read_state` through
`_hint_punchy_declared_but_flat`. Any `TypeError` on a malformed
`plan.structure` entry, any `KeyError` from a document shape a new verb
introduces, silently converts the whole stage into a pass.

#### 1.8.2 Three MORE deliberate fail-opens inside `conformance`, stacked

The MANDATORY `conformance` stage has three nested swallows on the way in, each
individually documented as intentional:

```941:954:backend/app/services/l3/tools.py
    intentions (delivered / fixed / compromised-and-surfaced), mirroring the
    FIT-TO-CRAFT discipline. Fires at most once (state-tracked). Fail-open: any
    error -> None (the edit still finishes). Returns feedback or None."""
    if state.get("conformance_surfaced"):
        return None
    if not (plan and (plan.get("structure") or plan.get("carries") or plan.get("watch"))):
        return None                     # no plan of record -> nothing to conform to
    try:
        required = [b for b in (observe._beat_view(e) for e in plan.get("structure") or [])
                    if b[0] and b[1] != "optional"]           # (beat_text, need)
        declared = list(plan.get("carries") or []) + list(plan.get("watch") or [])
        hints = _conformance_hints(working, ctx, plan)        # advisory, few (§4.5)
    except Exception:
        logger.exception("_plan_conformance: presentation build failed (finishing)")
        return None
```

```904:917:backend/app/services/l3/tools.py
def _conformance_hints(working: dict, ctx: EditContext, plan: dict | None) -> List[str]:
    """The cheap, robust advisory signals fed to the conformance checkpoint
    (brain_plan_conformance.plan.md §4.5). Few by design; each fail-open."""
    if not plan:
        return []
    out: List[str] = []
    for fn in (_hint_beatsync_declared_but_absent, _hint_punchy_declared_but_flat):
        try:
            h = fn(working, ctx, plan)
            if h:
                out.append(h)
        except Exception:
            logger.exception("conformance hint %s failed (skipping)", getattr(fn, "__name__", "?"))
    return out
```

```887:890:backend/app/services/l3/tools.py
    try:
        cuts = observe.read_state(working, ctx).get("cuts") or []
    except Exception:
        return None
```

**The distinction this plan draws, and it is the whole of the fix:**

- Losing ONE ADVISORY HINT is legitimately fail-open. A hint is a nice-to-have
  observation; the checkpoint still presents the plan beside the timeline. Keep
  `_conformance_hints`'s per-hint swallow (`tools.py:915-916`) and
  `_hint_punchy`'s `read_state` swallow (`tools.py:889-890`) exactly as they
  are — but they must log, and `_hint_punchy`'s bare `except: return None`
  must gain the `logger.exception` its sibling already has, because a silent
  swallow with no log is not diagnosable after the fact.
- Losing THE CHECKPOINT ITSELF is not. `_plan_conformance`'s `except` at
  `tools.py:952-954` wraps the *required-beat and declared-intention*
  construction — the thing the MANDATORY stage exists to present. Returning
  `None` there means the brain is never asked to account for a single beat it
  wrote, and finishes clean.

The two are separable with no new machinery: move the `hints` call OUT of the
guarded block (it already fail-opens per hint internally and cannot raise), so
the remaining `try` covers only the beat/intention build — and then that `try`
becomes an ERROR outcome instead of a `None`.

#### 1.8.3 Target state — a three-valued stage outcome

§1.2 already makes stages pure functions returning `str | None`. Widen the
return to a small outcome so "could not evaluate" is expressible, which is the
same move §1.6 makes for senses and the same one
`brain_perception_integrity.plan.md` §4 makes for perception payloads. One
vocabulary across all three, or the codebase grows a third way to say
"unknown":

```python
# tools.py -- a stage's three possible answers. `None` (nothing to say) and
# ERROR (could not tell) were the SAME value at HEAD, which is why an exception
# inside a MANDATORY invariant finished the edit. A MANDATORY stage that cannot
# evaluate itself yields blocking feedback and is NOT committed, so the next
# turn re-evaluates it; an ADVISORY stage that cannot evaluate itself is
# dropped, exactly as today.
_STAGE_ERROR = object()          # sentinel: distinct from None and from str
```

and in the runner:

```python
        try:
            fb = st.fn(gc)
        except Exception:
            logger.exception("gate stage %s failed", st.label)
            gc.fired.append({"stage": st.label, "kind": st.kind, "fired": False,
                             "why": "error"})
            if st.kind == ADVISORY:
                continue
            fb = ("AUTOMATIC CHECK -- " + st.label + ": this check could not be "
                  "evaluated on the current edit (internal error). It is NOT "
                  "confirmed clean. State plainly in your summary that " +
                  st.label + " could not be verified, or verify it by hand, "
                  "then finish.")
            # NOT added to stages_to_commit -- an unevaluated invariant may not
            # record itself as satisfied.
```

Three properties to hold, each of which is a way this could go wrong:

1. **It does not hard-block forever.** The feedback is a once-per-turn ask, not
   a refusal; the brain can satisfy it by saying it could not be verified. A
   persistent bug in a stage must not make every edit in the system
   unfinishable — that is how a check gets deleted in a hotfix.
2. **It never commits.** The errored stage is absent from `stages_to_commit`
   (§1.4), so the next turn re-evaluates instead of inheriting a mark it never
   earned. This falls out of §1's split for free, which is the argument for
   landing §1.8 with §1 rather than after it.
3. **`_finalize_turn`'s collect mode** (`collect=True`, `tools.py:1197-1200`)
   appends the error feedback like any other, so an errored stage on the last
   turn arrives through the same delivery path as everything else.

#### 1.8.4 The false-genericity trap, named

The tempting version is "wrap each stage function in its own try/except and
return a message from inside it". That is seven copies of the same handler, it
puts the error text next to the stage that happens to have been observed
failing, and — critically — it lets a future stage author forget the wrapper,
which reintroduces exactly this bug for stage #8. The runner is the only place
that sees every stage, so it is the only place the rule can be enforced for
stages that do not exist yet. **Test the class, not the instance: §1.8.5 V9
registers a stage that raises unconditionally and asserts the gate blocks,
without touching any of the seven real stages.**

The second trap is treating this as "add error handling". The change is not
defensive coding; it is making a MANDATORY stage's *silence* mean one thing
instead of two. If the implementation ends up with `except Exception: pass` plus
a louder log, nothing has been fixed.

#### 1.8.5 Verification (§1.8), with revert signatures

**V7 — an errored MANDATORY stage blocks.** Monkeypatch `_stage_length`'s
function in the `_GATE_STAGES` tuple with one that raises `RuntimeError`, run
`_verify_before_finish` on a document that would otherwise pass every stage
clean, and assert the returned feedback is non-None and names the stage.
*Revert signature:* at HEAD this returns `None` (the edit finishes), so the
assertion fails. This is the reachability proof for the whole section.

**V8 — an errored MANDATORY stage is not committed.** Same setup, twice in a
row: assert the state key the stage would have set is still absent after the
first call, and that the second call blocks again. *Revert signature:* an
implementation that commits on error passes the first assertion but the second
call returns `None`.

**V9 — the rule covers stages that do not exist yet.** Append an eighth,
synthetic MANDATORY stage whose `fn` raises, with a `key` no other test
references. Assert it blocks with no change to `_GATE_STAGES`'s real entries
and no per-stage wrapper anywhere. *Revert signature:* an implementation that
wrapped the seven real stages individually leaves the eighth unwrapped and this
test finishes clean. **This is the test that distinguishes the class fix from
the instance fix.**

**V10 — an errored ADVISORY stage still drops.** Make `_stage_flags` raise;
assert the gate returns `None` on an otherwise-clean document. *Revert
signature:* an implementation that blocks on any error, MANDATORY or not,
fails this — which is the over-correction that would make a pHash outage
unfinishable.

**V11 — the conformance checkpoint survives a dead hint, and only a dead
hint.** Two cases, one test: (a) make `_hint_punchy_declared_but_flat` raise
and assert `_plan_conformance` still returns the beat/intention presentation
with the required beats in it; (b) make `observe._beat_view` raise and assert
the stage now BLOCKS instead of returning `None`. *Revert signature:* at HEAD
case (b) returns `None` and the edit finishes with the brain never asked about
a single beat.

**V12 — perception's `unavailable` reaches the MANDATORY stages that consume
it.** Not a duplicate of §1.7: this asserts the *composition* of §1.6 and
§1.8. With `brain_perception_integrity.plan.md` §4 in place, force
`ctx.durations = {}` by making the duration lookup raise
(`observe.py:131-135`) and assert `_stage_structural` does not report clean —
because `observe.validate`'s over-span check is skipped without durations
(`observe.py`'s dur guard), so an empty `durations` map *weakens a MANDATORY
invariant by making it silently check less*. This is the one fail-open in the
set that degrades a check without any exception being raised anywhere, and it
is the reason §4's envelope has to cover the CONTEXT loaders and not just the
sense payloads. *Revert signature:* at HEAD `validate` returns `[]` and
`structural` reports clean on a document with a cut that runs past the end of
its source file.

---

## 2. §2 — The COVERAGE gate

**Lands after §1 and after `brain_perception_integrity.plan.md`.** It is a new
stage, so §1 must be in place; and its `wording` subject reads a sense status,
so the perception plan must be in place.

### 2.1 The question, stated as policy

> **You are about to finish, X is in play in this document, and you never
> looked at X.**

"In play" must be a MATERIAL FACT read off the document and the context. Never a
parse of the brain's prose, never a keyword match on the user's ask, never a
turn threshold.

### 2.2 `steps` is not sufficient — the coverage gate needs ARGUMENTS

The brief's sketch has `requires=("affordances", "inspect_cut")` for the
`takes` subject. But the observed failure was *"inspected ONE member of a
duplicate group and called it the strongest"* — and `steps` is
`List[str]` of bare tool NAMES (`tools.py:1368`, `tools.py:1431`). `"inspect_cut"
in steps` is satisfied by inspecting one member, so the sketch as written cannot
express the incident it was designed to catch.

The arguments already exist in the trace:

```1444:1446:backend/app/services/l3/tools.py
            trace.append({"turn": turn, "kind": "tool", "name": tc.name,
                          "args": tc.input or {},
                          "applied": bool(did), "result": obs[:600]})
```

So add a second, richer projection beside `steps` rather than changing it
(`steps` is part of `LoopResult` at `tools.py:1526` and is consumed by callers
and tests):

```python
# tools.py -- what the brain actually LOOKED AT this turn: sense name -> the
# ordered args of every call to it. Built from the same source as `steps` so the
# two can never disagree; `steps` stays a bare name list for LoopResult/back-
# compat and for the advisory_skip test. A coverage rule that can only ask
# "was this verb called at all" cannot express "you inspected one of three
# takes", which is the failure it exists to catch.
_LOOKS_ARG_CAP = 40


def _looks_from_trace(trace: List[dict]) -> Dict[str, Tuple[dict, ...]]:
    """{tool_name: (args, ...)} over this turn's recorded tool calls. Pure,
    fail-open ({} on any error -- a coverage rule then sees "never looked",
    which is the SAFE direction: it nags rather than silently passing)."""
```

`_GateCtx` gains `looks: Dict[str, Tuple[dict, ...]] = field(default_factory=dict)`,
populated in `_gate_ctx` from a `trace` argument threaded in from both call
sites. Note the fail-open direction is deliberately opposite to the house
default here, and say so in the docstring: an empty `looks` makes the coverage
gate FIRE, not pass. A perception failure must not be able to satisfy a
coverage requirement.

### 2.3 The policy table

Declared as DATA, in the same style as `observe._FEATURE_GROUPS`
(`observe.py:1456-1465`) and the `_ESCALATE` table that
`brain_accountability_architecture.plan.md` §3.2 specifies:

```python
# tools.py -- the COVERAGE POLICY. One entry per SUBJECT the edit can have "in
# play". `applies` is a predicate over the DOCUMENT and the CONTEXT -- a
# material fact, never the brain's prose and never the user's words. `satisfied`
# is a predicate over what the brain actually looked at (gc.looks), so a rule
# can demand "more than one member inspected", not merely "the verb was called".
# `subject`/`ask` are the human strings the brain reads.
#
# ADDING A SUBJECT IS ONE TUPLE ENTRY. That is the bar this table is held to --
# see the test in §2.6 V4, which adds a FOURTH, unlisted subject and asserts it
# fires with no other change. Same bar PART 2's _AXES sets with its
# "proves it is not audio-specific" test (brain_accountability_architecture.
# plan.md line ~928).
@dataclass(frozen=True)
class _Coverage:
    subject: str                  # stable machine token, recorded in the trace
    applies: Any                  # (_GateCtx) -> bool   -- is it in play?
    satisfied: Any                # (_GateCtx) -> bool   -- did the brain look?
    ask: str                      # the sentence the brain reads when it didn't


_COVERAGE: Tuple[_Coverage, ...] = (
    _Coverage(
        subject="audio",
        applies=lambda gc: bool(gc.ctx.audio_assets) or _has_audio_ops(gc.working),
        satisfied=lambda gc: "audio_state" in gc.looks,
        ask=("this project has audio you have never looked at (uploaded assets, "
             "or beds already placed). Call audio_state before you decide "
             "anything about sound -- including deciding NOT to use a bed."),
    ),
    _Coverage(
        subject="takes",
        applies=lambda gc: bool(_dup_groups_on_main_line(gc.working, gc.ctx)),
        satisfied=lambda gc: _inspected_at_least_two_of_each_group(gc),
        ask=("you placed a cut that has alternate takes of the same beat, and "
             "you compared fewer than two of them. Call inspect_cut on the "
             "alternates before calling one the strongest."),
    ),
    _Coverage(
        subject="wording",
        applies=lambda gc: _main_line_carries_speech(gc.working),
        satisfied=lambda gc: _speech_text_was_observed(gc),
        ask=("this edit's spoken words are load-bearing and you never read them "
             "back. Call read_transcript (or review) and check the words that "
             "actually play."),
    ),
)
```

Each predicate helper, and why it is a material fact:

- **`_has_audio_ops(working)`** — `any(o.get("type") == "place_audio" for o in
  (working.get("operations") or []))`. Mirrors the existing check at
  `observe.py:1463-1464` and `observe.py:1854`. `ctx.audio_assets` is populated
  at `observe.py:142-146` from `_fetch_audio_assets` (`observe.py:215`).
  *Fires from the presence of assets or ops, not from the word "music".*
- **`_dup_groups_on_main_line(working, ctx)`** — the exact shape `diagnose`
  already uses at `observe.py:1238-1249`: build `group_of` from
  `ctx.dup_groups[*]["members"]`, then return the group ids that any
  `timeline[*]["ref"]` maps to **where the group has ≥2 members**. Note this is
  strictly broader than `diagnose`'s finding, which only fires when TWO members
  are both on the main line (`observe.py:1245`); coverage must fire when ONE is
  placed and alternates exist unexamined, which is the observed incident.
  *Fires from `dup_groups` + `timeline`.*
- **`_inspected_at_least_two_of_each_group(gc)`** — for each applicable group,
  count distinct `args["ref"]` values among `gc.looks.get("inspect_cut", ())`
  that are members of that group; require ≥2, OR ≥1 when the group itself has
  exactly 2 members and the other is the placed one (i.e. require that the
  brain looked at at least one member it did NOT place). Prefer the second
  formulation as the single rule: **at least one non-placed member of the group
  was inspected.** It is simpler, it is what "compare the alternatives" means,
  and it does not need a magic count. `inspect_cut`'s args are `ref` / `seg_id`
  (`tools.py:570-572`), so a `seg_id` call must be mapped back to its `ref` via
  the timeline.
- **`_main_line_carries_speech(working)`** — `any(s.get("axis") == "speech" for
  s in (working.get("timeline") or []))`. **This replaces the brief's
  `_required_beat_claims_speech(gc.working)`.** That predicate would have to
  read beat TEXT to decide whether a beat "claims speech", which is prose
  parsing — the exact thing the table forbids, in the table's own first
  version. Whether the program's words are load-bearing is answerable from the
  document: `axis == "speech"` is set at placement from the map channel
  (`act.py:96`, `axis="speech" if rc.channel == "said"`) and by `place_span`'s
  explicit `axis` argument (`act.py:170`). It is also the same field
  `observe.review` gates its transcript read on (`observe.py:1515`), so the
  requirement and the sense agree by construction.
- **`_speech_text_was_observed(gc)`** — `"read_transcript" in gc.looks or
  "review" in gc.looks`, **AND** the corresponding payload's status was
  `observed`, not `unavailable`. This is where the coverage gate meets
  `brain_perception_integrity.plan.md` §4: in thread `34bb9967` the brain DID
  effectively have a transcript path available and got `{"text": "",
  "segments": []}`; a coverage rule that only checks "was the verb called"
  would have been satisfied by that call. **A coverage rule satisfied by a
  degraded sense is not a coverage rule.** The cheapest correct form: re-derive
  the status inside the predicate by calling
  `observe.read_transcript(gc.working, gc.ctx)` once (it is already computed
  unconditionally for other stages' inputs — thread it onto `_GateCtx` beside
  `review_out` rather than calling it twice).

### 2.4 The stage

```python
_COVERAGE_MAX_BLOCKS = 2      # this stage GATES -> its own bound (PART 1 §1.5)


def _stage_coverage(gc: _GateCtx) -> str | None:
    """MANDATORY: you are about to finish having never looked at something that
    is in play. Reads _COVERAGE, a declared policy table of (applies, satisfied)
    predicates over the DOCUMENT and the CONTEXT -- never over the brain's prose
    and never over the user's words, so it cannot be talked out of firing and
    cannot false-fire on a phrasing.

    This is the first consumer of `steps`/`looks` that TIGHTENS the gate. Every
    other reader relaxes it (tools.py:1033-1035), which is why "it never called
    audio_state and told the user the bed didn't fit" was unaskable.

    PURE (gate_integrity §1.2): no state writes. Bounded by _COVERAGE_MAX_BLOCKS
    so a brain that refuses to look still terminates."""
    if gc.state.get("coverage_blocked", 0) >= _COVERAGE_MAX_BLOCKS:
        return None
    gaps: List[_Coverage] = []
    for cov in _COVERAGE:
        try:
            if cov.applies(gc) and not cov.satisfied(gc):
                gaps.append(cov)
        except Exception:
            logger.exception("coverage rule %s failed (skipping)", cov.subject)
    if not gaps:
        return None
    body = "\n".join(f"- {c.subject}: {c.ask}" for c in gaps)
    return ("AUTOMATIC CHECK -- coverage: you're about to finish without having "
            "looked at something this edit actually contains. These are facts "
            "about the material, not guesses about your intent:\n" + body +
            "\nLook, then decide. If you look and it genuinely doesn't serve the "
            "edit, say so in one line -- but a decision about material you never "
            "examined isn't a decision.")
```

Registered in `_GATE_STAGES` **before** `craft` and after `intent`. Rationale:
fit-to-craft asks the brain to judge the whole piece cold; it cannot do that
honestly about audio it has never heard about or words it has never read. So
coverage precedes it:

```python
    _Stage("intent_surfaced",      MANDATORY, "intent",      _stage_intent),
    _Stage("coverage_blocked",     MANDATORY, "coverage",    _stage_coverage, COMMIT_COUNT),
    _Stage("craft_surfaced",       MANDATORY, "craft",       _stage_craft),
```

`_gate_state()` / the `verify` dict at `tools.py:1373-1375` gains
`"coverage_blocked": 0`.

Prompt discipline — `converse.py:187-189` currently says the senses exist and
"call them as you need". Add one sentence in the same block, describing the
mechanism rather than issuing a command (house style, `_PROVENANCE`
`converse.py:262-261`'s "purely descriptive"): *before you finish, an automatic
check names anything this edit CONTAINS that you never looked at — audio in the
project, alternate takes of a cut you placed, the words your speech cuts play.
Looking is not optional for material that is actually in the edit.*

### 2.5 The false-genericity trap, named

**The trap: hardcoding exactly the three subjects that match the three observed
incidents.** Three rules for three incidents is a coincidence table wearing a
table's clothes. It would pass review — it *looks* like `_FEATURE_GROUPS`, it
*looks* like `_ESCALATE` — and it would be a point fix three times over.

The distinguishing test is not "is it a tuple". It is:

1. `applies` reads only the document and `ctx` (assert by inspection: no
   predicate may reference `gc.user_ask`, `gc.working["plan"]` prose,
   `gc.working["summary"]`, or any `re.` call);
2. `satisfied` reads only `gc.looks` / a sense status;
3. **a fourth subject can be added as one tuple entry with no other change** —
   V4 below.

A second, more specific trap for this stage: making `applies` depend on the
user's ask via `observe._FEATURE_GROUPS`-style keyword matching
(`observe.py:1456-1469`). That is legitimate for `_requested_feature_flags`,
whose whole job is "the user said X" — it is illegitimate here, because coverage
is about what the EDIT contains, not what was asked for. The `audio` rule must
fire on a project with an unexamined asset even if the user never said the word
"music"; that is precisely the thread-`34bb9967` case.

Third trap: writing `satisfied` as `cov.requires ⊆ set(steps)`. That is the
brief's sketch, it is one line, and it cannot express the `takes` incident
(§2.2). If a reviewer proposes collapsing `satisfied` back into a `requires`
tuple "for symmetry with `_ESCALATE`", the answer is V3 below, which fails.

### 2.6 Verification (§2), with revert signatures

**V0 — reachability against real production-shaped data.** Capture a fixture
from ingest run `24894c8a-9daa-4aa3-aea2-1f3b60c93478` / edit thread
`34bb9967-97aa-4818-9f21-01145156a9bb` (the same fixture
`junk_span_suppression.plan.md` §3.5 captures — reuse it, do not capture a
second one): the stored document at its final version, plus the `EditContext`
projection (`meta_by_ref`, `dup_groups`, `audio_assets`, `durations`,
`map_struct`). Then assert, with `looks = {}` (nothing examined):

- `_stage_coverage` returns a non-None string;
- the string names `audio` **and** `takes` (this run had an unused audio asset
  and placed `aad020e3:m06`, whose twin `7144c9fc:m07` is the same material);
- with `looks = {"audio_state": ({},), "inspect_cut": ({"ref": "7144c9fc:m07"},), "read_transcript": ({},)}`
  and an `observed` transcript status, it returns None.

*Revert signature:* remove `_stage_coverage` from `_GATE_STAGES` and the first
assertion gets None. Replace any `applies` predicate with a constant `False`
(the classic way a gate becomes unreachable) and the first assertion gets None.
Replace `satisfied` with `lambda gc: True` and the first assertion gets None.
**This is the assertion that would have caught `vcut/resolve.py:512`'s dead-air
gate and `vcut/store.py:135`'s hardcoded `junk=False`:** it asserts the stage
FIRES on real data, not that it computes correctly on synthetic data.

**V1 — the delivered effect, at session level.** `test_session_never_looked_at_audio_cannot_finish`
in `scripts/test_tools_loop.py`: a `_ScriptedLLM` that places one cut and then
attempts a prose finish, against a context with one `audio_assets` entry and no
`audio_state` call. Assert (a) the loop does NOT finish on that attempt, (b) the
brain receives a message containing `"AUTOMATIC CHECK -- coverage"` and the word
`audio`, (c) the `kind:"gate"` trace entry shows the `coverage` stage
`fired: true, delivered: true`, and (d) after a scripted `audio_state` call the
next finish attempt is not blocked by coverage. *Revert signature:* (a) and (c)
both fail without the stage; (d) fails if `satisfied` is wired to the wrong
token.

**V2 — no false fire.** Same session with `ctx.audio_assets = []`, no
`place_audio` ops, no dup groups, and a timeline of only `axis="any"` video
segments: assert the coverage stage returns None and the loop finishes exactly
as it does at HEAD. *Revert signature:* a rule whose `applies` is a constant
`True` (or which reads the user's ask) fails this and makes the gate un-shippable
noise. This is the assertion that keeps the stage from being disabled six weeks
later for crying wolf.

**V3 — the `takes` rule catches the actual incident, not a weaker one.** Against
the V0 fixture: place ONE member of a dup group, record
`looks = {"inspect_cut": ({"ref": <the placed member>},)}` — i.e. the brain
inspected exactly the cut it placed and none of the alternates. Assert coverage
FIRES on `takes`. *Revert signature:* implement `satisfied` as
`"inspect_cut" in gc.looks` (the brief's `requires`-tuple form) and this test
goes green-when-it-should-be-red — it PASSES the gate. So this test is the one
that forbids the simplification, and it must be observed failing against that
simpler implementation, not only against HEAD.

**V4 — a FOURTH subject is one tuple entry.** `test_coverage_admits_a_new_subject_with_no_other_change`:
inside the test, monkeypatch `tools._COVERAGE` to `_COVERAGE + (_Coverage(
subject="layout", applies=lambda gc: bool(gc.working.get("layout_regions")),
satisfied=lambda gc: "read_state" in gc.looks, ask="…"),)` and assert the stage
fires on `layout` for a document with a `layout_regions` entry and no
`read_state` call — with **no** change to `_stage_coverage`, `_GateCtx`,
`_run_gate_stages`, or `_GATE_STAGES`. *Revert signature:* if any subject's
logic has leaked into `_stage_coverage` (a branch on `cov.subject`, a special
case for `audio`), the new entry either does not fire or requires editing the
stage, and the test fails. This is the same bar
`brain_accountability_architecture.plan.md` line ~928's
`test_reversal_detected_for_take_swap_and_for_reorder_and_for_pace` sets for
`_AXES`, and it is the only assertion that distinguishes a policy table from
three hardcoded incidents.

**V5 — predicates are document-only, asserted structurally.**
`test_coverage_predicates_read_no_prose`: for each `_Coverage` entry, inspect
`cov.applies.__code__.co_names` / `co_consts` (and the same for any module-level
helper it calls) and assert none of `user_ask`, `summary`, `purpose`, `carries`,
`watch`, `search`, `match`, `findall` appears. Crude, and deliberately so: it is
a tripwire that makes "just grep the ask for 'music'" fail CI rather than fail
review.

**V6 — `looks` fails open in the SAFE direction.** Feed `_looks_from_trace` a
malformed trace (a non-dict entry, a missing `name`) and assert it returns `{}`
and that the coverage stage consequently FIRES rather than passing. *Revert
signature:* a `satisfied` predicate written as `gc.looks.get(x, ()) or True`, or
a `_looks_from_trace` that returns a sentinel meaning "assume looked", passes
this and inverts the whole stage. State in the docstring that this is the one
place in the module where fail-open means "nag", not "allow".

---

## 3. §3 — Wrap the unguarded `mirror_text` injection

```python
        # Every per-turn injection is wrapped: a perception error must never
        # cost the brain this step's TOOL RESULTS (results is appended as one
        # user message at the end of the loop body -- a raise here loses all of
        # it, not just the mirror). observe.mirror_text reads the DB via
        # cutrecord_map.speech_words_in_span / footage_map._said_text_for_span;
        # observe.plan_mirror_text is pure, and is wrapped anyway so the rule
        # has no exception for the next injection to be modelled on.
        for _src, _fn in (("mirror", lambda: observe.mirror_text(working, ctx)),
                          ("plan_mirror", lambda: observe.plan_mirror_text(working, building=changed))):
            try:
                _txt = _fn()
            except Exception:
                logger.exception("tools: %s injection failed (continuing)", _src)
                continue
            if _txt:
                results.append(text_block(_txt))
```

When `brain_accountability_architecture.plan.md` PART 5 lands, the
`results.append(text_block(...))` becomes `_inject(results, trace, turn=turn,
source=_src, text=_txt)` and the loop above is the natural place for it — which
is one more reason to move PART 5 earlier (see that plan's amendment §A.5).

**Interaction with `brain_perception_integrity.plan.md`.** Silently dropping the
mirror is itself the absence/emptiness collapse: the brain simply doesn't see a
mirror that turn and has no way to know one was owed. Once the status envelope
lands, the `except` branch should inject a one-line `MIRROR: unavailable (…)`
instead of `continue`. Spec'd there (§4.4); noted here so the two do not
diverge.

**Verification.** Unit: stub `observe.mirror_text` to raise; assert the turn's
`convo` message still contains every `tool_result` block and the
`plan_mirror`/progress blocks. *Revert signature:* unwrap it and the test raises
out of `run_edit_loop` entirely — the whole turn is lost, which is exactly the
failure this prevents. RED against HEAD.

---

## 4. §4 — Explicitly NOT in this plan

**Conformance-stage TIMING is not in this plan, and must not be added to it.**

`_stage_conformance` (`tools.py:1123-1124`) is reachable only from
`_verify_before_finish` (`tools.py:1422`) and `_finalize_turn`
(`tools.py:1256`), both of which run at a TERMINATION attempt. So the one
checkpoint that asks "you wrote X — did you do X?" never fires while build
budget remains, which is the worst possible time to learn a required beat is
missing: there is no room left to build it.

The obvious fix — run it at turn N — is a magic number, and a magic number is a
band-aid with a constant in it. The correct trigger is **"when a commitment's
fulfilment state changes"**, and that detector is
`brain_accountability_architecture.plan.md` **PART 2 §2.1–2.3** (`commit.ledger`
+ `commit.reversals` over declared `_AXES`, detected as a document diff). The
conformance-timing change is therefore a consumer of PART 2 and is sequenced
after it (step 5 of the sequencing above), specified in PART 2's own section,
not here.

Also not in this plan, and named so nobody folds them in:

- **Signal severity / the symmetric length contract.**
  `brain_accountability_architecture.plan.md` PART 3 §3.1–3.3. In particular
  `_stage_length`'s substring match at `tools.py:1063` (`"over target" in
  message`) is replaced there by a machine `kind`; §1.2 above keeps that line
  byte-identical on purpose, so the two changes do not collide. Do not
  "improve" it here.
- **`edit_moved`.** `tools.py:1420` and `tools.py:1509` both carry the comment
  `# PART 4 will define edit_moved properly; until then an alias`. §1.5 above
  preserves the alias verbatim. PART 4 owns it.
- **The injection funnel.** PART 5. §3 above is written so that rewiring it to
  `_inject` is a one-line change.
- **Beat binding + the adversarial quote check.** Sequenced last;
  `brain_accountability_architecture.plan.md` amendment §A.6 records the
  decision.
- **The perception-side fail-open sites themselves.** §1.8 changes what the GATE
  does when an input is missing or an invariant cannot evaluate. It does not
  attach the statuses: the 32 indistinguishable sites, their chains, and the 9
  that must deliberately stay fail-open are enumerated in
  `brain_perception_integrity.plan.md` §4.6, and Chain 5 there is exactly the
  set §1.6 and §1.8 consume. **Ship §1.8 with that plan's §4 or `_unavailable`
  has nothing to detect** — with one exception that stands alone: §1.8.3's
  errored-stage rule needs no perception change at all and lands with §1, since
  the exception it reacts to is raised inside the gate's own code.
