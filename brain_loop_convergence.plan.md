# Brain loop CONVERGENCE — budget/progress awareness, airtight finalization, and fail-loud verbs (Phase 4 of the loop hardening)

**Scope.** Three related-but-distinct fixes to the edit tool-loop
(`tools.run_edit_loop`, `backend/app/services/l3/tools.py`) so the brain finishes
on its OWN terms with a truthful recap instead of being guillotined mid-action
by the turn cap. Each part is independently implementable and testable:

1. **PART 1 — budget/progress awareness + convergence discipline + churn
   detection** (the deep root cause): give the brain per-turn visibility into how
   much room it has left, framed as *convergence*, and a generic no-progress
   (churn) signal that pushes it to accept-and-surface a ceiling EARLIER.
2. **PART 2 — all termination paths finalize truthfully** (the airtight FLOOR):
   the `max_turns` exit (and any non-voluntary exit) must run the SAME truthful
   finalization the voluntary finish runs — force a `wrap_up`/surface so the user
   gets a real recap, never leftover mid-action reasoning.
3. **PART 3 — tools fail loud and immediately** (generic ergonomics): an edit
   verb given invalid/out-of-domain/no-op input returns an IMMEDIATE `applied=false`
   + a plain reason the brain sees that turn, instead of silently doing nothing or
   silently misinterpreting an argument.

**READ-ONLY handoff — implement from another chat. Do NOT implement from this
doc; do NOT commit.**

**This is NOT a rebuild.** Everything shipped stays untouched: plan altitude
(`document["plan"]["structure"]` = `{beat, need}`, `brain_plan_altitude.plan.md`),
reasoning observability (`edit_turns.trace` `{"kind":"reasoning",...}`), and the
plan-conformance fix-or-surface machinery (`_verify_before_finish`, `_plan_conformance`,
`wrap_up`, the surface-non-empty block, the FINISHING "(4) FIT TO PLAN" prompt
step, `brain_plan_conformance.plan.md`). We ADD to the loop, the mirror push, the
verb-reject channel, and the prompt; we change no existing behavior on the
validated voluntary-finish path.

---

## 0. Motivation — the fix-or-surface ladder is FINISH-GATED (validated over 3 live edits)

The just-shipped fix-or-surface machinery (`_verify_before_finish` + `wrap_up` +
the surface-non-empty block) is **only reachable on a VOLUNTARY finish** — the
`if not resp.tool_calls:` branch of the loop
(`tools.py:1077-1085`). It is never reached when the loop runs out the clock.

Across **3 live edits**, **2 of 3 exhausted `_MAX_TURNS` (18)** via THRASH and hit
the `for turn in range(max_turns): … else:` "hit max_turns" exit
(`tools.py:1126-1127`), which just logs and falls through to the reply
construction — **bypassing the entire done-gate.** Consequences observed:

- The brain hit a real, self-diagnosed material ceiling, **routed around it**, and
  shipped with **EMPTY surface fields** (`summary`/`open_questions`/`notes`) —
  Part A of `brain_plan_conformance.plan.md` never got to fire.
- The stored user-facing reply was **leftover mid-action reasoning** ("Let me
  re-add the fades… and validate") because the cap path falls straight into the
  reply builder with an empty `summary`, so `reply = last_text` (`tools.py:1135-1138`).

The thrash pattern that burned the budget:
- **Net-zero churn:** swap take m13→m14→m13 over 4 turns to end where it started.
- **Remove/re-lay recovery cycles** forced by tool misuse: an illegal A2 `trim`
  that silently no-op'd; an absolute source timestamp passed where a relative
  delta was expected. Each silent failure cost a `review` round to discover, then
  a remove + re-place to recover.

So the silent-ceiling failure **recurred not because the ladder is wrong, but
because the ladder is finish-gated** — a genuinely hard edit thrashes past the
cap and never climbs it. This plan attacks all three layers: make the brain
converge earlier (Part 1), make EVERY exit finalize truthfully (Part 2), and stop
the tool-misuse recovery cycles that feed the thrash in the first place (Part 3).

### Scope discipline — what this plan explicitly does NOT do

- **Do NOT simply raise `_MAX_TURNS`.** That's the band-aid we're avoiding — every
  turn costs latency + tokens, and a higher cap just moves the guillotine, it
  doesn't make the brain converge or finalize truthfully.
- **Do NOT add an arbitrary "max N retries" counter.** The churn signal (Part 1)
  is a GENERIC no-progress detector (the edit returned to a prior state), reused
  as an advisory hint — NOT a hard cap that stops the loop after N tries.
- **Do NOT per-bug special-case in Part 3.** The two observed footguns are TEST
  CASES for one generic validation/feedback pattern, not two point fixes.
- **Budget awareness is a CONVERGENCE DISCIPLINE, NOT a countdown-panic.** (This
  caution is repeated at the design site in §Part 1.) Budget visibility exists to
  let the brain *reserve room to finish and know when to converge* — attempt a
  goal honestly, accept a genuine ceiling as best-available + surface it, reserve
  budget to finish cleanly. It must never read as "hurry up, N turns left".
- **Do NOT touch** the altitude reshape, the reasoning-capture observability, the
  plan mirror, `set_plan`, or the existing done-gate stages. Additive only.

---

## 1. Current state to build on (verified against the code)

### 1.1 The loop and its two exits — `tools.run_edit_loop` (`tools.py:1034-1140`)

```1056:1128:backend/app/services/l3/tools.py
    for turn in range(max_turns):
        resp = llm.run(system=system, messages=convo, tools=tools,
                       max_tokens=max_tokens, cache_system=True)
        last_text = (resp.text or "").strip() or last_text
        convo.append(resp.assistant_message)
        # ... reasoning capture into trace ...
        if not resp.tool_calls:
            # Finish attempt: don't let a changed edit exit unchecked against the
            # contract (structural = hard, length = fix-or-justify, rest advisory).
            if changed and turn < max_turns - 1:
                feedback = _verify_before_finish(working, ctx, verify, steps, user_ask)
                if feedback is not None:
                    convo.append(user_message(feedback))
                    continue
            break
        results = []
        asked = False
        for tc in resp.tool_calls:
            # ... dispatch each tool call, append tool_result_block + trace ...
        mirror_text = observe.mirror_text(working, ctx)
        if mirror_text:
            results.append(text_block(mirror_text))
        plan_text = observe.plan_mirror_text(working, building=changed)
        if plan_text:
            results.append(text_block(plan_text))
        convo.append(user_message(results))
        if asked and questions:
            break
    else:
        logger.info("tools: hit max_turns=%d; wrapping up", max_turns)
```

- **Voluntary finish** (`if not resp.tool_calls:`, `tools.py:1077`): a turn that
  makes no tool call. If the edit `changed` and it isn't the last turn, run
  `_verify_before_finish` (`tools.py:1080-1084`); a returned feedback string is
  appended as a `user_message` and the loop `continue`s; `None` → `break` (finish
  allowed). **This is the ONLY path that runs the done-gate.**
- **`max_turns` exit** (`for … else:`, `tools.py:1126-1127`): reached when the
  `for` completes WITHOUT a `break` — i.e. the brain used all 18 turns still
  making tool calls. It only logs. **The done-gate is never invoked.**
- **`ask_user` pause** (`tools.py:1124-1125`): breaks with `awaiting=True`.

### 1.2 `_MAX_TURNS` and the reply construction

```37:37:backend/app/services/l3/tools.py
_MAX_TURNS = 18
```
```1129:1140:backend/app/services/l3/tools.py
    awaiting = bool(questions)
    # ... Part A: prefer the brain's deliberate wrap-up over last internal prose ...
    summary = (working.get("summary") or "").strip()
    reply = (summary if summary and not awaiting else last_text) or (
        "Before I go further I need your call on a couple of things below."
        if awaiting else "Done.")
    return LoopResult(reply=reply, document=working, changed=changed, steps=steps,
                      trace=trace, questions=questions, awaiting_user=awaiting)
```
On the cap path `working["summary"]` is empty (the gate that would have demanded a
`wrap_up` never ran), so `reply = last_text` — the leftover mid-action prose.

### 1.3 The done-gate and its state dict

`_verify_before_finish(working, ctx, state, steps, user_ask)` (`tools.py:909-1031`)
is the fix-or-surface ladder: structural → length → Stage 1 intent → Stage 2 craft
→ Stage 3 flags → **Part B `_plan_conformance`** (`tools.py:1012-1015`) → **Part A
surface-non-empty block** (`tools.py:1017-1031`). Each stage fires at most once via
the `state` dict, initialized in the loop (`tools.py:1048-1050`):
```1048:1050:backend/app/services/l3/tools.py
    verify = {"struct_tries": 0, "length_surfaced": False, "intent_surfaced": False,
             "craft_surfaced": False, "reviewed": False,
             "conformance_surfaced": False, "surface_blocked": 0}
```
Reusable pieces Part 2 leans on: `_plan_conformance(working, ctx, plan, state)`
(`tools.py:864-906`), `_conformance_hints(working, ctx, plan)` (`tools.py:835-848`),
`_surface_is_empty(working)` (`tools.py:851-861`), and `act.wrap_up` (`act.py:665-683`)
+ its dispatch (`tools.py:653-663`).

### 1.4 The mirror push sites (where per-turn context is injected)

Two `text_block`s are appended to each turn's `results` (`tools.py:1112-1121`),
right before `convo.append(user_message(results))`:
- `observe.mirror_text(working, ctx)` — the edit mirror (`observe.py:1710`).
- `observe.plan_mirror_text(working, building=changed)` — the plan mirror
  (`observe.py:1734`), the model for advisory hints Part 1 reuses.

The static per-user-turn context (`converse._context_block`, `converse.py:539-601`)
is built ONCE before the loop and baked into the cached system prompt, so **it is
the wrong place for per-turn budget** — budget/progress + churn go at the mirror
push sites inside the loop (§Part 1).

### 1.5 Dispatch and how `applied`/`result` are built — `tools._dispatch` (`tools.py:508-685`)

Each ACT verb is called inside a `try:` (`tools.py:511`) whose broad
`except Exception` (`tools.py:683-685`) converts any crash to `{"error": …}`. The
shared tail builds the result:
```667:685:backend/app/services/l3/tools.py
        changed = new is not doc
        # Echo the resulting state so the model SEES the effect of its edit.
        result = {"applied": changed, "state": observe.read_state(new, ctx)}
        if not changed:
            result["note"] = "no-op (unknown id or illegal argument)"
        # ... snap info ...
        return _json(result), new, changed
    except Exception as e:  # a bad tool call must never crash the turn
        logger.exception("tools: %s failed", name)
        return _json({"error": f"{type(e).__name__}: {e}"}), doc, False
```
**Two silent-failure modes here** (Part 3 targets both):
1. A verb that rejects input `return document` (unchanged) → `changed=False` → the
   ONLY feedback is the generic `"no-op (unknown id or illegal argument)"` note.
   The brain can't tell an unknown id from a collapsed span from an
   absolute-vs-relative mix-up.
2. A verb that `_clone`s and reassigns fields to values IDENTICAL to the current
   ones (e.g. `trim` to the same span) returns `new is not doc` → `changed=True` →
   `applied=true` reported for a change that did nothing.

### 1.6 The verbs and their silent rejects — `act.py`

Verbs are pure `document -> document` and **total**: "an unknown id or illegal arg
is a no-op returning the doc unchanged (the caller diagnoses)" (`act.py:17-18`).
They are called from **exactly one production site** (`tools._dispatch`) plus the
test scripts — confirmed by grep: no other module calls `act.trim`/`place`/etc. So
the reject channel can be enriched without touching any other caller.

The two footgun sites, both in `act.trim` (`act.py:515-548`):
```526:548:backend/app/services/l3/act.py
    seg = next((s for s in doc["timeline"] if s.get("seg_id") == target_id), None)
    if seg is not None:
        cur_in, cur_out = int(seg["in_ms"]), int(seg["out_ms"])
        new_in = int(in_ms) if in_ms is not None else cur_in + int(delta_in_ms or 0)
        new_out = int(out_ms) if out_ms is not None else cur_out + int(delta_out_ms or 0)
        new_in = max(0, new_in)
        if new_out <= new_in:
            return document                      # <-- (a) silent no-op, generic note
        seg["in_ms"], seg["out_ms"] = new_in, new_out
        return doc                               # <-- reassign even if unchanged -> applied=true
    op = next((o for o in doc["operations"] if o.get("op_id") == target_id
               and o.get("type") in ("place_video", "place_audio")), None)
    if op is None:
        return document                          # <-- unknown id, generic note
    cur_in, cur_out = int(op["src_in_ms"]), int(op["src_out_ms"])
    new_in = int(in_ms) if in_ms is not None else cur_in + int(delta_in_ms or 0)
    new_out = int(out_ms) if out_ms is not None else cur_out + int(delta_out_ms or 0)
    new_in = max(0, new_in)
    if new_out <= new_in:
        return document                          # <-- (a) A2 trim collapses -> silent
    op["src_in_ms"], op["src_out_ms"] = new_in, new_out
    op["to_ms"] = int(op.get("from_ms", 0)) + (new_out - new_in)
    return doc
```
`_dispatch`'s trim branch (`tools.py:541-551`) already has `ctx` in scope (source
durations via `ctx.durations` / `ctx.audio_assets`), which is where the
absolute-vs-relative DOMAIN check (footgun b) belongs — the verb itself can't see
the source length.

### 1.7 House style to mirror

`brain_plan_conformance.plan.md`, `brain_plan_altitude.plan.md`: fail-open,
additive, no migration (document is jsonb), and the "prompt DISCIPLINE + robust
deterministic signals" split. Tests extend `scripts/test_tools_loop.py` (helpers
`_gate_state` `:459`, `_clean_doc` `:465`, `_ctx`/`_struct` `:42-52`), registered
in `main()` (`:979`).

---

## PART 1 — BUDGET / PROGRESS AWARENESS + CONVERGENCE DISCIPLINE + CHURN DETECTION

> **CRITICAL DESIGN CAUTION (repeat, non-negotiable): this is a CONVERGENCE
> discipline, not a countdown-panic.** Budget visibility exists to let the brain
> *reserve room to finish and know when to converge* — attempt each goal honestly;
> if it genuinely won't land after a real attempt, treat that as a CEILING to
> ACCEPT (take best-available) and SURFACE, not something to retry; and reserve
> budget to finish cleanly. Framing must be about progress & convergence, NEVER
> "hurry up, N turns left."

The brain thrashes because it has (a) no visibility into remaining budget and (b)
no generic signal that it's making no progress. Give it both, surfaced at the
existing mirror push sites, tied to the EXISTING accept-best-available + surface
contract (so the churn hint routes it onto the sanctioned "route around a ceiling"
path EARLIER instead of retrying to exhaustion).

### 1A. Budget/progress note — a per-turn convergence block

**File:** `backend/app/services/l3/tools.py`. New module-private helper beside the
other loop helpers (e.g. just above `run_edit_loop`, `tools.py:1034`):

```python
def _progress_note(turn: int, max_turns: int, *, churn: bool = False) -> str:
    """A per-turn CONVERGENCE note (brain_loop_convergence.plan.md Part 1),
    pushed beside the mirror. Deliberately framed around PROGRESS + converging,
    NOT a countdown to rush: budget visibility exists so the brain reserves room
    to finish and knows when to accept-and-surface a ceiling. Pure."""
    used, remaining = turn + 1, max_turns - (turn + 1)
    head = (f"PROGRESS: step {used} of {max_turns} this turn (~{remaining} left "
            "before it auto-finalizes).")
    body = (" This is room to CONVERGE, not a clock to beat: attempt each goal "
            "honestly, but if a goal genuinely won't land after a real attempt, "
            "that's a CEILING -- accept the best available version and SURFACE it "
            "with wrap_up rather than retrying. Reserve room to finish cleanly.")
    if remaining <= 4:
        body += (" You're near this turn's budget -- move to CONVERGE: lock the "
                 "best version you have, resolve or surface any open compromise, "
                 "and leave a wrap_up recap. Don't open new threads of work.")
    return head + body
```

**Insertion at the mirror push site** (`tools.py:1112-1121`), additive — the
progress note joins the existing two `text_block`s:
```python
        mirror_text = observe.mirror_text(working, ctx)
        if mirror_text:
            results.append(text_block(mirror_text))
        plan_text = observe.plan_mirror_text(working, building=changed)
        if plan_text:
            results.append(text_block(plan_text))
        # brain_loop_convergence.plan.md Part 1: per-turn budget/progress +
        # convergence discipline, and a churn hint when the edit is not making
        # progress (returned to a prior state). Advisory text blocks, same shape
        # as the mirrors; pure + fail-open (never mutate the edit or reply).
        results.append(text_block(_progress_note(turn, max_turns, churn=churn_now)))
        if churn_hint:
            results.append(text_block(churn_hint))
        convo.append(user_message(results))
```
(`churn_now`/`churn_hint` are computed in 1B, just above this block.)

### 1B. Churn detection — a GENERIC no-progress signal (not an arbitrary counter)

Detect when the brain is NOT making progress by comparing the edit's CONTENT state
turn-to-turn: if the edit has **returned to a state it already held at an earlier
turn** (a revert; a place→remove→place of the same ref; a swap A→B→A that lands
back on A), that's churn — surface it as an advisory ceiling hint reusing the
plan-mirror hint pattern. This is generic (state identity), NOT a "max N retries"
cap: the loop is never stopped by it; it only nudges the brain to accept + surface.

**Content fingerprint** — new module-private helper in `tools.py` (beside
`_beat_grid_ms`, `tools.py:429`):
```python
def _edit_fingerprint(doc: dict) -> tuple:
    """A CONTENT fingerprint of the edit's meaningful state -- the ordered spine
    plus the ops -- with the random per-place seg_id/op_id EXCLUDED, so a revert
    to an arrangement already held (place->remove->place the same ref; A->B->A)
    hashes IDENTICAL even though ids are freshly minted each place. Spine ORDER
    matters (a move is a real change); op order does not (resolve sorts them), so
    ops are sorted. Pure; used only for churn detection (Part 1)."""
    spine = tuple(
        (s.get("file_id"), int(s.get("in_ms") or 0), int(s.get("out_ms") or 0),
         s.get("ref"), s.get("level"), bool(s.get("mute")),
         s.get("audio_override") and tuple(sorted(s["audio_override"].items())))
        for s in (doc.get("timeline") or []))
    ops = tuple(sorted(
        (o.get("type"), o.get("source_file_id"), o.get("seam_seg_id"),
         int(o.get("from_ms") or 0), int(o.get("to_ms") or 0),
         int(o.get("src_in_ms") or 0), int(o.get("src_out_ms") or 0),
         o.get("role"), o.get("audio_kind"),
         float(o.get("gain_db") or 0.0), float(o.get("duck_db") or 0.0),
         int(o.get("audio_offset_ms") or 0), int(o.get("ms") or 0))
        for o in (doc.get("operations") or [])))
    return (spine, ops)
```

**Loop wiring** — track prior fingerprints and detect a return-to-prior-state.
Initialize before the loop (near `tools.py:1046-1050`):
```python
    # brain_loop_convergence.plan.md Part 1: fingerprints of every DISTINCT edit
    # state seen this turn (excluding random ids), and the states already flagged
    # as churn so the same revert isn't nagged every step. Generic no-progress
    # signal -- never a retry cap.
    seen_states: dict = {_edit_fingerprint(working): -1}   # fp -> turn first seen
    churned_states: set = set()
```
Compute the signal right after the per-turn dispatch loop, BEFORE the mirror push
(insert just before `mirror_text = observe.mirror_text(...)`, `tools.py:1112`):
```python
        # Churn: did this turn's edits land the program back on an arrangement it
        # already held at an EARLIER turn? (Revert / net-zero swap / place-remove-
        # place.) Generic, id-independent; advisory only.
        churn_now = churn_hint = None
        if changed:
            fp = _edit_fingerprint(working)
            prior_turn = seen_states.get(fp)
            # A match to a state first seen strictly before the last step means the
            # edit left that state and came back -- churn. (A match to the immediately
            # prior step is a same-turn no-op, handled by Part 3's fail-loud verbs.)
            if prior_turn is not None and prior_turn < turn - 1 and fp not in churned_states:
                churned_states.add(fp)
                churn_now = True
                churn_hint = (
                    "CHURN: this edit has returned to an arrangement you already "
                    "built earlier -- you're cycling (revert / swap-back / place-"
                    "remove-place) without net progress. Treat this as a CEILING: "
                    "pick the best version you have and MOVE ON (accept + surface "
                    "the trade-off with wrap_up), rather than trying the same "
                    "swap again.")
            seen_states.setdefault(fp, turn)
```
`churn_hint` reuses the advisory-hint pattern (a plain text block, exactly like the
plan-mirror nudges and `_conformance_hints`) — it INFORMS, never forces. It ties
directly to the required-beat/compromise contract: "accept + surface" is the
already-sanctioned route around a ceiling; the hint just makes the brain take it
earlier.

### Back-compat / additive / fail-open (Part 1)

- **Additive:** two extra advisory `text_block`s per turn; the mirror + plan mirror
  are unchanged. Nothing gates or blocks; the loop's control flow is untouched.
- **Fail-open:** `_progress_note` and `_edit_fingerprint` are pure and can't raise
  on well-formed docs; wrap the churn block in a `try/except` that logs and
  continues (an empty hint is fine) so a fingerprinting error never alters the
  edit or reply.
- **No countdown-panic:** the note's tone is convergence, and the firmer branch
  only triggers near the end and still says "converge/lock/surface", never "hurry".
- **Generic, not arbitrary:** churn is state-identity, deduped per state
  (`churned_states`); it never counts retries or caps the loop.

### Testing (Part 1)

**Unit** (`scripts/test_tools_loop.py`):
- `test_edit_fingerprint_ignores_random_ids`: two docs with the same
  file/in/out/ref spine but different `seg_id`s → equal fingerprints; a reordered
  spine → different; a reordered ops list (same ops) → equal.
- `test_progress_note_is_convergence_not_countdown`: `_progress_note(0, 18)`
  contains "CONVERGE"/"reserve", not "hurry"; the near-end branch
  (`_progress_note(15, 18)`) adds the "lock the best version" line but still no
  panic wording.
- `test_loop_pushes_progress_note_each_turn`: a scripted 2-round loop → each pushed
  `user_message` carries a `PROGRESS:` text block (inspect `_ScriptedLLM`'s message
  snapshots).
- `test_loop_flags_churn_on_return_to_prior_state`: script place(refA) →
  remove(seg) → place(refA) across 3 turns; the 3rd turn's push carries a `CHURN:`
  block; a straight-line build (place A, place B, place C, no revert) never does.
- `test_churn_hint_dedups_per_state`: a state that churns twice only emits the hint
  the first time it recurs.

**Manual e2e:** re-run the m13→m14→m13 swap scenario; confirm the CHURN hint
appears the turn the swap-back lands and the brain accepts+surfaces instead of
swapping again; confirm the PROGRESS note is present every turn and never reads as
a countdown.

---

## PART 2 — ALL TERMINATION PATHS FINALIZE TRUTHFULLY

Every way the loop can end must converge on the SAME truthful finalization. Today
only the voluntary finish runs `_verify_before_finish`; the `max_turns` exit (and
a voluntary finish on the very last turn, where the gate is skipped by
`turn < max_turns - 1`, `tools.py:1080`) returns `last_text` — leftover mid-action
reasoning with an empty surface. Fix: guarantee a `wrap_up`/surface on ANY
non-voluntary exit, and make the returned reply a genuine summary.

**Design decision (of the two the brief poses): a single FORCED FINALIZATION step
after the loop, plus a deterministic fallback.** Rationale: it is purely additive,
touches only the cap/gate-skipped paths, and leaves the validated voluntary-finish
path byte-for-byte unchanged. (The alternative — reserving the last build turn for
finalization — is noted in §2.4 but rejected: it would shrink usable build budget
for every edit and change the validated path.)

### 2.1 Track whether the loop finished cleanly on its own

Add a flag so we know an exit was VOLUNTARY-and-gated (no finalization needed) vs.
cap/gate-skipped (finalization needed). Initialize before the loop (`tools.py:1047`):
```python
    finished_clean = False   # a voluntary finish that PASSED the done-gate
```
Set it at the voluntary `break` (`tools.py:1085`) — the gate returned `None` (or the
edit didn't change / it's the last turn and we let it through):
```python
        if not resp.tool_calls:
            if changed and turn < max_turns - 1:
                feedback = _verify_before_finish(working, ctx, verify, steps, user_ask)
                if feedback is not None:
                    convo.append(user_message(feedback))
                    continue
            finished_clean = True          # <-- new: voluntary finish, gate satisfied
            break
```
(The last-turn voluntary finish sets `finished_clean=True` too, but its surface may
be empty because the gate was skipped — §2.3's fallback still covers it, since
finalization keys on `_surface_is_empty`, not on this flag alone.)

### 2.2 Forced finalization after the loop — `_finalize_after_loop`

New module-private helper (beside `_verify_before_finish`, `tools.py:1032`):
```python
def _finalize_after_loop(llm, *, system, convo, ctx, working, tools, verify,
                         user_ask, max_tokens) -> dict:
    """Part 2 (brain_loop_convergence.plan.md): make a NON-voluntary exit (the
    max_turns cap, or a last-turn finish that skipped the gate) finalize as
    truthfully as a voluntary finish would. Runs ONE bounded finalization step:
    it instructs the brain to STOP editing, ACCEPT the best-available version of
    anything that didn't land, and call wrap_up with a real recap + any surfaced
    compromise. Any wrap_up (or tiny lock-in edit) it makes is applied. Returns
    the (possibly updated) working doc. Fail-open: any error -> working unchanged."""
    try:
        # Present what still needs an account (required beats + live hints), so the
        # forced recap names real compromises rather than rubber-stamping. Reuses
        # the conformance presentation; state-tracked so it doesn't double-fire.
        account = _plan_conformance(working, ctx, working.get("plan"), verify)
        instruct = (
            "AUTOMATIC FINALIZE -- you've reached this turn's build budget. Stop "
            "opening new work. In ONE step: if a required beat or a declared "
            "intention didn't land, ACCEPT the best version you have (do NOT retry) "
            "and call wrap_up with a short, human `summary` of what you built and "
            "any compromise + why (open_questions if it's the user's to weigh). "
            "Make at most a single tiny edit only if it LOCKS the best-available "
            "version; otherwise just wrap_up.")
        if account:
            instruct += "\n\n" + account
        convo.append(user_message(instruct))
        resp = llm.run(system=system, messages=convo, tools=tools,
                       max_tokens=max_tokens, cache_system=True)
        for tc in (resp.tool_calls or []):
            if tc.name == "ask_user":       # finalization never pauses for a question
                continue
            _obs, working, _did = _dispatch(tc.name, tc.input or {}, ctx, working, user_ask)
    except Exception:
        logger.exception("tools: forced finalization step failed (continuing)")
    # Deterministic floor: if the brain still left the surface empty, synthesize a
    # truthful recap from the built state so the reply is NEVER leftover reasoning.
    try:
        if working.get("timeline") and _surface_is_empty(working):
            working = act.wrap_up(working, summary=_fallback_summary(working, ctx))
    except Exception:
        logger.exception("tools: fallback summary failed (continuing)")
    return working
```

Deterministic fallback recap (new helper, beside it):
```python
def _fallback_summary(working: dict, ctx: EditContext) -> str:
    """A plain, truthful one-liner built from the edit STATE (never leftover
    mid-action prose) for when the brain runs out of budget without a wrap_up.
    Names the size of what was built and any LIVE compromise the advisory hints
    already detected, so even the floor tells the user the truth. Pure/fail-open."""
    try:
        st = observe.read_state(working, ctx)      # returns cut_count + total_ms (observe.py:625-634)
        n = st.get("cut_count") or len(st.get("cuts") or [])
        total = st.get("total_ms")
    except Exception:
        n, total = len(working.get("timeline") or []), None
    dur = f", ~{int(total)//1000}s" if total else ""
    base = f"Assembled {n} cut{'s' if n != 1 else ''}{dur}."
    hints = _conformance_hints(working, ctx, working.get("plan"))
    if hints:
        base += (" Ran out of editing room before fully resolving: "
                 + "; ".join(h.split(" -- ")[0] for h in hints) + ".")
    else:
        base += " Ran out of editing room before a final self-review."
    return base
```
(`observe.read_state` returns `cut_count` + `total_ms` in its top-level state dict,
`observe.py:625-634`; the fallback degrades to the raw timeline count on any error.)

### 2.3 Call finalization after the loop, before building the reply

Insert between the `for … else:` block and the reply construction (`tools.py:1128`):
```python
    else:
        logger.info("tools: hit max_turns=%d; finalizing", max_turns)

    # Part 2: any NON-voluntary / gate-skipped exit that changed a non-empty edit
    # must still finalize truthfully (a real wrap_up recap, never leftover mid-
    # action prose). A clean voluntary finish already ran the done-gate, so skip
    # it there; a paused ask_user turn keeps its own question framing.
    if changed and not questions and not finished_clean and _surface_is_empty(working):
        working = _finalize_after_loop(
            llm, system=system, convo=convo, ctx=ctx, working=working,
            tools=tools, verify=verify, user_ask=user_ask, max_tokens=max_tokens)
```
The reply builder (`tools.py:1135-1138`) is unchanged: `working["summary"]` is now
populated (forced or fallback), so `reply = summary` — a genuine recap. Fail-open:
if `_finalize_after_loop` somehow left the surface empty, `reply` falls back to
`last_text` exactly as today (no regression).

### 2.4 Alternative considered (rejected)

Reserving the last build turn for finalization (run the build loop for
`max_turns - 1`, then always finalize) is simpler control-flow but (a) shrinks the
usable build budget for EVERY edit, including the ones that finish cleanly with
room to spare, and (b) changes the validated voluntary path. The forced-step-on-
exit design confines the change to the failing paths. Keep the single-forced-step
design.

### Back-compat / additive / fail-open (Part 2)

- **Voluntary-finish path unchanged.** `finished_clean=True` short-circuits
  finalization for a clean gated finish; the reply logic is untouched.
- **At most ONE extra `llm.run`**, only on a cap/gate-skipped exit that left the
  surface empty. Bounded — no new loop.
- **Fail-open at every step:** the LLM step, the dispatch of its calls, and the
  fallback are each try/excepted; a failure leaves `working` as-is and the reply
  falls back to `last_text`. `ask_user` during finalization is ignored (never
  re-pauses the turn).
- **Reuses shipped machinery:** `_plan_conformance`, `_conformance_hints`,
  `_surface_is_empty`, `act.wrap_up`, `_dispatch` — no new gate semantics.

### Testing (Part 2)

**Unit** (`scripts/test_tools_loop.py`):
- `test_loop_cap_path_forces_a_wrap_up`: a scripted `_ScriptedLLM` that ALWAYS
  returns tool calls (never a prose finish) with `max_turns` small (e.g. 3), whose
  post-cap finalization step scripts a `wrap_up(summary="…")` → `LoopResult.reply`
  is that summary, `working["summary"]` is set, `changed` True.
- `test_loop_cap_path_synthesizes_summary_when_brain_declines`: same but the
  finalization step returns prose with no `wrap_up` → the deterministic
  `_fallback_summary` is written; `reply` is the synthesized recap (contains
  "Assembled … cut"), NOT `last_text`.
- `test_fallback_summary_names_live_compromise`: stub `_conformance_hints` → one
  hint; `_fallback_summary` mentions it; with no hints it degrades to the plain
  size line.
- `test_voluntary_clean_finish_skips_finalization`: a scripted loop that finishes
  voluntarily with the gate satisfied and `finished_clean=True` → no extra
  `llm.run` (assert call count), reply unchanged from today.
- `test_finalization_is_fail_open`: make the forced `llm.run` raise → loop still
  returns; if surface remains empty, reply falls back to `last_text`.

**Manual e2e:** reproduce the 2/3 cap edits; confirm the cap exit now yields a
clean `summary` recap (stored reply is the recap, not "Let me re-add the fades…"),
`GET /threads/{id}` shows any `open_questions`, and a genuinely compromised edit
names the ceiling in the surface.

---

## PART 3 — TOOLS FAIL LOUD AND IMMEDIATELY

Generic rule: an edit verb given invalid, out-of-domain, or no-op-producing input
must return an IMMEDIATE `applied=false` + a plain reason the brain sees THAT turn,
instead of silently doing nothing (§1.5 mode 1) or silently misinterpreting an
argument (§1.6). One generic mechanism, applied across the verbs — not two point
fixes. The two observed footguns become the concrete test cases.

### 3.1 The generic feedback channel — a typed reject

Verbs already communicate a rejection by `return document`, but the REASON is lost.
Add a typed rejection so the reason travels with the rejection and the co-located
condition owns the wording. **Verbs are called from exactly one production site
(§1.6), so this is safe.**

**File:** `backend/app/services/l3/act.py`, near the top (below the imports):
```python
class EditReject(Exception):
    """A verb's STRUCTURED rejection of an inapplicable request -- an unknown id,
    an out-of-domain target, or an argument that produces no (or a degenerate)
    change. Carries a plain-language `reason` the loop surfaces to the brain the
    SAME turn (tools._dispatch catches it -> applied=false + reason), so the brain
    corrects in one turn instead of discovering the no-op later via review. It
    NEVER escapes the loop -- _dispatch converts it to feedback -- so a verb still
    never crashes a turn (the totality contract holds at the loop boundary)."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason
```
Update the module docstring line "never an exception that could crash a turn"
(`act.py:17-18`) to note `EditReject` is the sanctioned, loop-caught reject channel.

**Dispatch catch** — add BEFORE the broad `except Exception` (`tools.py:683`):
```python
    except act.EditReject as r:
        # Part 3 (brain_loop_convergence.plan.md): a verb rejected the request with
        # a specific reason -- surface it immediately as applied=false so the brain
        # corrects THIS turn, not later via review.
        return _json({"applied": False, "reason": r.reason}), doc, False
    except Exception as e:  # a bad tool call must never crash the turn
        logger.exception("tools: %s failed", name)
        return _json({"error": f"{type(e).__name__}: {e}"}), doc, False
```
The trace already records `applied=False` + the result string, so the reject is
audit-visible. The generic `"no-op (unknown id or illegal argument)"` note
(`tools.py:670-671`) stays as a fail-safe for any un-converted `return document`
path.

### 3.2 Convert the silent rejects to loud ones (co-located reasons)

Replace each verb's anonymous `return document` REJECTION with a specific
`raise EditReject(...)`, and add an explicit UNCHANGED-value reject where a verb
reassigns fields that may equal the current ones (§1.5 mode 2). The pattern, shown
on `act.trim` (`act.py:515-548`) — the footgun-rich verb:

```python
def trim(document, target_id, *, in_ms=None, out_ms=None,
         delta_in_ms=None, delta_out_ms=None):
    if in_ms is None and out_ms is None and delta_in_ms is None and delta_out_ms is None:
        raise EditReject("trim needs a source edit: in_ms/out_ms (ABSOLUTE source "
                         "ms) or delta_in_ms/delta_out_ms (RELATIVE nudge).")
    doc = _clone(document)
    seg = next((s for s in doc["timeline"] if s.get("seg_id") == target_id), None)
    if seg is not None:
        cur_in, cur_out = int(seg["in_ms"]), int(seg["out_ms"])
        new_in = int(in_ms) if in_ms is not None else cur_in + int(delta_in_ms or 0)
        new_out = int(out_ms) if out_ms is not None else cur_out + int(delta_out_ms or 0)
        new_in = max(0, new_in)
        if new_out <= new_in:
            raise EditReject(
                f"trim collapses the span (out {new_out}ms <= in {new_in}ms). "
                "in_ms/out_ms are ABSOLUTE source times; to nudge by a relative "
                "amount pass delta_in_ms/delta_out_ms instead.")
        if new_in == cur_in and new_out == cur_out:
            raise EditReject("trim leaves the span unchanged -- the in/out you gave "
                             "equal the current span.")
        seg["in_ms"], seg["out_ms"] = new_in, new_out
        return doc
    op = next((o for o in doc["operations"] if o.get("op_id") == target_id
               and o.get("type") in ("place_video", "place_audio")), None)
    if op is None:
        raise EditReject(f"no main-line cut or trimmable op with id '{target_id}'.")
    cur_in, cur_out = int(op["src_in_ms"]), int(op["src_out_ms"])
    new_in = int(in_ms) if in_ms is not None else cur_in + int(delta_in_ms or 0)
    new_out = int(out_ms) if out_ms is not None else cur_out + int(delta_out_ms or 0)
    new_in = max(0, new_in)
    if new_out <= new_in:
        raise EditReject(
            f"trim collapses this {op['type']} op's span (out {new_out}ms <= in "
            f"{new_in}ms). in_ms/out_ms are ABSOLUTE source times; use "
            "delta_in_ms/delta_out_ms for a relative nudge.")
    if new_in == cur_in and new_out == cur_out:
        raise EditReject("trim leaves this op's source span unchanged.")
    op["src_in_ms"], op["src_out_ms"] = new_in, new_out
    op["to_ms"] = int(op.get("from_ms", 0)) + (new_out - new_in)
    return doc
```

Apply the SAME pattern to the other verbs' reject sites — a mechanical, generic
sweep, not per-bug logic:
- **unknown-id rejects** (`remove` `act.py:474-475`; `move` `:493,:502`;
  `set_gain` `:333`; `duck` `:353`; `fade_audio` `:379-380`; `replace_audio`
  `:434-435`; `set_audio` `:599`; `set_arc_intent` `:619`; `crossfade` `:404-405`;
  `split_edit` `:570-571`; `place`/`split_screen` unresolved ref `:209,:757,:762`)
  → `raise EditReject(f"no … with id '{target_id}'")` (name the target kind the
  verb expected).
- **degenerate/invalid-arg rejects** (`place_audio` bad role/window/source
  `:274-299`; `place_span`/`split_screen`/`replace_audio` `out<=in`; `crossfade`
  `ms<0`; `retime`/`tighten` unknown level/pace; `set_grade` no-op dials
  `:706-707`) → `raise EditReject(<what was wrong + the valid domain>)`.
- **unchanged-value rejects** where a reassign could be a no-op (`set_gain` same
  gain; `move` to the same index/`to_ms`; `trim` above) → an explicit
  `EditReject("… unchanged — equals the current value")` so `applied=false` is
  honest.

> Keep each reason ONE plain sentence naming (1) what was wrong and (2) the valid
> alternative/domain. The point is a one-turn correction, not a stack trace.

### 3.3 The absolute-vs-relative DOMAIN guard (footgun b) — dispatch-level

`act.trim` cannot tell an absolute source timestamp passed as a delta from a
legitimate large delta, because it doesn't know the source's length. `_dispatch`
does (`ctx.durations` / `ctx.audio_assets`). Add a cheap, generic pre-check in the
trim dispatch branch (`tools.py:541-551`) — same `applied=false` + reason shape:
```python
        elif name == "trim":
            tr_in, tr_out = args.get("in_ms"), args.get("out_ms")
            tr_din, tr_dout = args.get("delta_in_ms"), args.get("delta_out_ms")
            # Part 3 domain guard: an ABSOLUTE source edge past the source's own
            # length almost always means an absolute value was passed where a
            # relative delta was meant. Reject loudly (the verb can't see the
            # source length; _dispatch can). Fail-open: no known duration -> skip.
            reason = _trim_domain_reason(doc, ctx, args["target_id"],
                                         tr_in, tr_out, tr_din, tr_dout)
            if reason:
                return _json({"applied": False, "reason": reason}), doc, False
            if args.get("snap") == "beat":
                ...  # (unchanged)
            new = act.trim(doc, args["target_id"], in_ms=tr_in, out_ms=tr_out,
                           delta_in_ms=tr_din, delta_out_ms=tr_dout)
```
with the helper (beside `_snap_trim_to_beat`, `tools.py:437`):
```python
def _trim_domain_reason(doc, ctx, target_id, in_ms, out_ms, delta_in_ms, delta_out_ms):
    """Part 3: reject a trim whose RESULTING absolute source edge lands well past
    the source's own length -- the fingerprint of an absolute source timestamp
    passed where a relative delta was meant (footgun b). Cheap + generic; returns
    a plain reason or None. Fail-open: unknown target/duration -> None (let the
    verb's own validation run)."""
    op = next((o for o in doc.get("operations") or []
               if o.get("op_id") == target_id
               and o.get("type") in ("place_video", "place_audio")), None)
    seg = None if op else next((s for s in doc.get("timeline") or []
                                if s.get("seg_id") == target_id), None)
    src = (op or {}).get("source_file_id") or (seg or {}).get("file_id")
    if not src:
        return None
    dur = ctx.durations.get(src)
    if dur is None:
        dur = next((a["dur_ms"] for a in getattr(ctx, "audio_assets", []) or []
                    if a["file_id"] == src), None)
    if not dur:
        return None
    cur_out = int((op or seg or {}).get("src_out_ms", (seg or {}).get("out_ms", 0)))
    new_out = int(out_ms) if out_ms is not None else cur_out + int(delta_out_ms or 0)
    if new_out > int(dur) + 1000:     # >1s past the source end -> almost certainly a mix-up
        return (f"trim's resulting source out ({new_out}ms) is past this source's "
                f"length ({int(dur)}ms). in_ms/out_ms/delta_* are SOURCE times -- "
                "did you pass an absolute timestamp where a relative delta_* was "
                "meant, or vice versa?")
    return None
```
(Read the exact seg source-out key when implementing: a main-line seg uses
`out_ms`; an op uses `src_out_ms` — the helper handles both.)

### Why this is generic, not per-bug

- One reject CHANNEL (`EditReject` → `_dispatch` catch), one presentation
  (`applied=false` + `reason`), applied uniformly across verbs.
- Footgun (a) — an A2 `trim` that produces no change — is caught by the co-located
  `EditReject` on the op-collapse/unchanged branches (§3.2), not special-cased.
- Footgun (b) — an absolute source timestamp passed as a delta — is caught by the
  generic source-domain guard (§3.3), which fires for ANY over-length source edge,
  not just this one input.

### Back-compat / additive / fail-open (Part 3)

- **Only production caller is `_dispatch`** (§1.6); the reject is caught there and
  converted to feedback, so no caller sees an exception. Test scripts that call
  `act.*` directly must expect `EditReject` on the reject paths (update assertions).
- **Loop totality preserved:** `_dispatch` never propagates `EditReject`; a turn
  still can't crash. The generic no-op note remains as a fail-safe.
- **No new no-ops:** every path that used to `return document` now either applies
  (unchanged) or rejects with a reason — the SET of applied edits is identical; only
  the FEEDBACK on non-applies improves (plus the unchanged-value case flips from a
  false `applied=true` to an honest `applied=false`).
- **Fail-open domain guard:** `_trim_domain_reason` returns `None` on any unknown
  target/duration, deferring to the verb's own validation.

### Testing (Part 3)

**Unit** (`scripts/test_tools_loop.py`, mirror the dispatch tests near `:301`):
- `test_trim_a2_collapse_rejects_loud` (footgun a): dispatch `"trim"` on a
  `place_audio` op with args that collapse the span → result `applied=false`,
  `reason` mentions "collapses" + "delta"; the working doc is unchanged.
- `test_trim_unchanged_span_is_not_applied`: dispatch `"trim"` with in/out equal to
  the current span → `applied=false`, reason "unchanged" (regression guard for the
  old false `applied=true`).
- `test_trim_absolute_as_delta_rejects` (footgun b): dispatch `"trim"` on an op
  with `delta_out_ms` = an absolute source timestamp far past the source length
  (via a stubbed `ctx.durations`) → `applied=false`, reason names the over-length
  source out + the absolute/relative confusion.
- `test_trim_domain_guard_fail_open`: unknown source duration → guard returns None,
  the verb runs normally.
- `test_edit_reject_surfaces_as_applied_false`: a direct `act.trim` on an unknown
  `target_id` raises `EditReject`; via `_dispatch` it becomes
  `{"applied": false, "reason": "no main-line cut or trimmable op …"}`.
- `test_unknown_id_verbs_reject_loud`: a representative sweep (`remove`, `move`,
  `set_gain`, `duck`, `fade_audio`) with a bogus id → each dispatches to
  `applied=false` with an id-naming reason.

**Manual e2e:** reproduce the two recovery cycles — an A2 trim that used to
silently no-op, and an absolute-as-delta trim — and confirm the brain corrects on
the NEXT turn from the reason (no `review` round, no remove/re-lay cycle).

---

## Exact change list per file

| File | Change |
|---|---|
| `backend/app/services/l3/tools.py` | **Part 1:** add `_edit_fingerprint`, `_progress_note`; in `run_edit_loop` init `seen_states`/`churned_states`, compute the churn signal before the mirror push, and push the progress + churn `text_block`s (`:1112-1121`). **Part 2:** add `_finalize_after_loop`, `_fallback_summary`; add `finished_clean` flag + set it at the voluntary break (`:1085`); call `_finalize_after_loop` after the `for/else` when `changed and not questions and not finished_clean and _surface_is_empty(working)` (`:1128`). **Part 3:** add `except act.EditReject` in `_dispatch` before the broad except (`:683`); add `_trim_domain_reason` + call it in the trim dispatch branch (`:541`). |
| `backend/app/services/l3/act.py` | **Part 3:** add `class EditReject`; convert each verb's silent `return document` reject to `raise EditReject(<reason>)` and add unchanged-value rejects (trim/set_gain/move); update the module docstring's totality note. |
| `backend/app/services/l3/converse.py` | Optional reinforcement: one line in `_LOOP_SYSTEM`'s FINISHING/PLAN block that budget is for CONVERGENCE (accept-and-surface a ceiling early, reserve room to finish), matching the Part 1 note — no behavior change. Skip if the existing "(4) FIT TO PLAN" wording already suffices. |
| `backend/scripts/test_tools_loop.py` | Add Part 1/2/3 unit tests above; register each in `main()` (`:979`). No `_gate_state` change needed (Part 2 reuses the existing `verify` dict; `_plan_conformance`'s `conformance_surfaced` key already exists). |
| migrations | **None** — document is jsonb; surface fields pre-seeded (`converse._seed_document`). |

---

## Summary of decisions

- **Loop structure found:** one voluntary-finish path (`if not resp.tool_calls:`,
  `tools.py:1077`) that runs `_verify_before_finish`, and a `for … else:`
  `max_turns` exit (`tools.py:1126`) that only logs and falls through to the reply
  builder — the done-gate is finish-gated, so 2/3 cap edits bypassed it entirely.
- **Part 2 makes the cap path finalize** via a single forced finalization step
  after the loop (`_finalize_after_loop`): it instructs the brain to accept
  best-available + `wrap_up`, applies that, and — if the surface is still empty —
  writes a deterministic `_fallback_summary` so the reply is a real recap, never
  leftover reasoning. The validated voluntary path is untouched (`finished_clean`).
- **Part 1 design:** a per-turn `PROGRESS` block (turns used/remaining) framed
  strictly as CONVERGENCE (attempt honestly → accept a real ceiling as
  best-available + surface → reserve room to finish), never a countdown; plus a
  GENERIC churn signal — an id-independent content fingerprint (`_edit_fingerprint`)
  flags when the edit returns to a prior arrangement (revert / A→B→A /
  place-remove-place), surfaced as an advisory "this is a ceiling — accept and move
  on" hint (reusing the plan-mirror hint pattern), deduped per state, never a retry
  cap.
- **Part 3 generic pattern:** a typed `act.EditReject(reason)` caught once in
  `_dispatch` → `applied=false` + a plain reason the brain sees that turn; every
  verb's silent `return document` becomes a co-located loud reject, plus unchanged-
  value rejects (fixing false `applied=true`) and a dispatch-level absolute-vs-
  relative source-domain guard. The two footguns (A2-trim no-op; absolute-as-delta)
  are TEST CASES, not special cases.
- **Out of scope (band-aids avoided):** raising `_MAX_TURNS`; any "max N retries"
  counter; per-bug fixes in Part 3. Altitude/observability untouched.
- **Back-compat / fail-open throughout:** additive text blocks, at most one extra
  `llm.run` on the failing paths only, every new path try/excepted, no migration.
