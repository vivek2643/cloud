# Brain plan mechanism — a durable plan artifact + a plan mirror (Phase 2 of the planning hardening)

**Scope.** Give the brain (EDSO, `backend/app/services/l3/`) a DURABLE, structured
plan artifact it writes with a tool and updates as it goes, and echo that plan
back into its context every turn — a "PLAN mirror" that sits right next to the
existing EDIT MIRROR. This converts the good thinking the locked *PLAN BEFORE YOU
BUILD* prompt block asks for into an adhered-to edit and kills drift. READ-ONLY
handoff — implement from another chat.

**Where this fits (three-phase hardening):**
1. **Thinking Framework — DONE.** `_LOOP_SYSTEM` in `converse.py:82-108` now carries
   the non-directive *PLAN BEFORE YOU BUILD* block. It tells the brain to reason to
   a plan, "write it down", build against it, and "update the plan rather than
   drifting" — but there is **no artifact** to write to, so "write it down" lives
   only in ephemeral reasoning. That is exactly why it drifts.
2. **Todo/plan mechanism — THIS PLAN.** The artifact + write verb + plan mirror.
3. **Prompt cleanup — LATER (out of scope).** Reconciling / trimming the prompt
   prose once the mechanism exists. Handoff seam noted in §8.

**The contract this artifact fulfills** (locked language from `_LOOP_SYSTEM`,
`converse.py:82-108`, do NOT edit that block):
> "…reason the rest of the way to a plan and **write it down**, then build against
> it… NAME WHAT IT'S FOR… DECIDE WHAT CARRIES IT, moment to moment… WORK OUT THE
> STRUCTURE THIS PIECE WANTS… **The ordered result of that reasoning is your
> plan.** …SEE WHERE IT WILL BE HARD… Carry those as **things to watch.** …When
> reading changes your mind, say so and **update the plan rather than drifting.**"

The artifact's fields are lifted verbatim from those four steps (§2); the write
verb fulfills "write it down" / "update the plan"; the plan mirror keeps the
written plan in front of the brain every turn so "build against it" is enforced by
visibility, exactly the way the EDIT MIRROR already enforces "read [the edit]
before your next move."

**Observed drift failures this targets** (from a real edit): out-of-order
selection, cutting mid-clause, putting silent b-roll on the main line, and
rubber-stamping over its own continuity signals. Each is a case of the brain
building something other than the ordered intent it reasoned to — with nowhere to
have recorded that intent and nothing echoing it back.

---

## 1. How the existing loop is assembled (what we mirror)

Read these before touching anything — the plan mechanism reuses each pattern
verbatim.

### 1.1 System prompt assembly (`converse.py`)
- `_LOOP_SYSTEM` (`:61-216`) — the lean identity + mechanics prompt. Contains a
  **THE MIRROR** paragraph (`:179-197`) that *describes the mechanism* (an
  always-present, auto-pushed edit mirror) in plain prose. **This is the exact
  paragraph style we copy for a new THE PLAN note (§6).**
- `_guidance_block()` (`:395-399`) and `_PROVENANCE` (`:231-363`) — static
  reference text folded into the cached prefix.
- `_context_block(file_ids, document, ctx)` (`:508-566`) — the per-turn dynamic
  context: PROJECT OVERVIEW → domain → voiceover → BEAT INDEX → **`CURRENT PROGRAM
  MAP`** (`:564-565`, the assembled edit AS IT STANDS AT TURN START). This is where
  the plan-of-record belongs at turn start (§4.1) — right next to the program map,
  so the brain sees **plan vs. timeline together** from message one.
- `respond()` (`:622-674`) assembles `system = _LOOP_SYSTEM + _guidance_block() +
  _PROVENANCE + "\n\n" + _context_block(file_ids, document, ctx)` (`:653-654`),
  loads the document via `store.latest_document(thread_id)` (`:638`), seeds a fresh
  one with `_seed_document(file_ids)` (`:569-582`, `:643`) when absent, and hands
  `working` (the loaded/seeded doc) into `tools.run_edit_loop(...)` (`:655-656`).

### 1.2 The tool loop (`tools.py`)
- `_specs()` (`:62-335`) — the neutral tool schema list (OBSERVE senses, ACT verbs,
  ask_user). Verbs are declared with `S(name, description, schema)`; **this is where
  `set_plan` is declared (§3.1).**
- `_dispatch(name, args, ctx, doc, user_ask)` (`:446-601`) — maps a tool call to
  `(observation_text, new_doc, changed)`. ACT verbs call `act.<verb>(doc, …)`,
  then `changed = new is not doc` (`:583`) and the result echoes
  `observe.read_state(new, ctx)` (`:585`). **This is where `set_plan` is dispatched
  (§3.2).**
- `run_edit_loop(...)` (`:726-800`) — the bounded perceive→act→re-perceive loop.
  After every round of tool calls it pushes the edit mirror:
  ```785:788:backend/app/services/l3/tools.py
          mirror_text = observe.mirror_text(working, ctx)
          if mirror_text:
              results.append(text_block(mirror_text))
          convo.append(user_message(results))
  ```
  **This is the exact injection point the plan mirror reuses (§4.2)** — a plain
  `text_block` alongside the same round's tool_result blocks.
- `_verify_before_finish(working, ctx, state, steps, user_ask)` (`:633-723`) — the
  FINISHING done-gate (structural → length → Stage 1 intent → Stage 2 craft →
  Stage 3 advisory). **This is the Phase 3 seam** for a plan-conformance gate
  (§8) — we only leave a named hook here, we do NOT build the gate.

### 1.3 The edit mirror renderer (`observe.py`)
- `mirror(document, ctx)` (`:1620-1677`) → dict; `render_mirror(m)` (`:1680-1707`)
  → compact prose ("MIRROR (the edit as it now stands…)"); `mirror_text(document,
  ctx)` (`:1710-1713`) is the one-call wrapper `run_edit_loop` pushes. **The plan
  mirror renderer lives beside these (§4.3), same file, same shape.**
- `affordances(document, ctx)` (`:1723-1816`) publishes `"verbs"` (`:1811-1813`)
  and `"senses"` (`:1814-1815`) lists — add `set_plan` to `"verbs"` (§3.3).

### 1.4 Persistence (`store.py`, `edit_threads.py`)
- The Edit Document is a versioned jsonb blob per thread: `save_document(thread_id,
  document, created_by)` (`store.py:171-185`), `latest_document(thread_id)`
  (`:188-201`). Table `edit_documents.document jsonb` (migration
  `014_l3_edit_threads.sql:37-48`).
- The router persists a new version **iff the turn changed the doc**:
  ```133:136:backend/app/routers/edit_threads.py
      version: Optional[int] = None
      if result.changed and result.document is not None:
          version = store.save_document(thread_id, result.document, created_by="auto")
  ```
- The human timeline-edit path `put_document` preserves unknown top-level keys via
  `new_doc = {**doc, "timeline": …, "operations": …}` (`edit_threads.py:222-231`),
  so a `plan` key rides through a manual edit untouched.
- `identity_map` on `ingest_runs` (`ingest_store.set_identity_map`/`get_identity_map`,
  `ingest_store.py:61-83`; migration `037_identity_map.sql`) is the *alternative*
  storage model (a dedicated column with explicit set/get). **We do NOT use it here
  — see the decision in §5.**

---

## 2. The artifact: `document["plan"]` — a small, structured, non-directive dict

The plan is a single top-level key on the Edit Document. Minimal and structured
(not free prose) so Phase 3 can later check the edit against it — but deliberately
**open/non-templated**: no fixed arc, no required sections, no narrative schema.

```jsonc
"plan": {
  "purpose":   "teach a first-time viewer why the migration was worth it",
  "carries":   ["founder VO leads throughout", "screen-capture demo carries the middle"],
  "structure": [
    "cold-open on the outage line (hook)",
    "founder explains the stakes over b-roll",
    "the demo, in the order the fix actually happened",
    "close on the one-customer line"
  ],
  "watch": [
    "clip93 has a backward jump-cut risk if I reorder the demo",
    "the outage line runs thin — may need coverage"
  ],
  "rev": 2,                 // bump on each set_plan; lets the mirror show churn
  "updated_note": "reordered the demo after reading the transcript"  // optional
}
```

### Field rationale (each mapped to a locked-prompt step)
| Field | Prompt step (`converse.py`) | Type | Why |
|---|---|---|---|
| `purpose` | "NAME WHAT IT'S FOR — teach, tease, pitch, document, move" (`:91-92`) | `str` | The single thing everything else answers to. One line. |
| `carries` | "DECIDE WHAT CARRIES IT, moment to moment… possibly flips between sections" (`:93-95`) | `list[str]` | What leads — a line, the picture, music, a mix — optionally per-section. A list, not one value, because the prompt explicitly allows it to flip between sections. |
| `structure` | "WORK OUT THE STRUCTURE THIS PIECE WANTS… **The ordered result of that reasoning is your plan.**" (`:96-100`) | `list[str]` (ORDERED) | The heart of the artifact: the ordered intent the brain settled on. **Free-form ordered list — NOT a fixed arc/template.** Each entry is one beat/section in the order it should play. This is what the edit's actual order gets checked against in Phase 3. |
| `watch` | "SEE WHERE IT WILL BE HARD… Carry those as **things to watch**" (`:101-103`) | `list[str]` | The hard spots the brain flagged (a seam that risks a jump, a thin stretch, a take choice). Phase 3's un-rubber-stampable checks read these. |
| `rev` | mechanism metadata | `int` | Increments each `set_plan`; the mirror shows it so a stale plan (rev unchanged while the edit moved) is visible. Not part of the contract. |
| `updated_note` | "When reading changes your mind, **say so** and update the plan" (`:107-108`) | `str?` | Optional one-liner the brain sets when a revision changes its mind — the durable form of "say so". |

**SURVEY** ("SURVEY WHAT YOU HAVE", `:87-90`) is intentionally NOT a field — it is
the perception act (reading the beat index/transcript) that precedes writing the
plan, not a value the plan records.

All fields are optional and free-form; nothing is validated for content. An empty
or partial plan is legal (fail-open, §7). We keep the shape flat and tiny so
`set_plan` is cheap to call and re-call.

---

## 3. Write path — one verb, `set_plan` (full replace)

**Decision: a single `set_plan` verb that fully replaces the plan.** Not set+update.
Rationale: the artifact is tiny (four short lists/strings), so a full re-emit is
cheap and dead-simple; "update the plan rather than drifting" is satisfied by
re-calling `set_plan` with the amended plan (bump `rev`, set `updated_note`). A
partial-merge `update_plan` would add API surface and merge-semantics ambiguity
(how do you delete a `watch` item?) for no real saving. One verb, full replace,
maps 1:1 onto both "write it down" and "update the plan".

### 3.1 Declare the verb in `tools._specs()` (`tools.py`, after ask_user `:334`, or grouped with ACT)
Mirror the `S(name, description, obj({...}))` pattern exactly:
```python
S("set_plan", "Write (or rewrite) your PLAN for this edit -- the ordered intent "
  "you reasoned to before building, per PLAN BEFORE YOU BUILD. It is echoed back "
  "to you every turn as the PLAN mirror and survives across turns, so build "
  "against it and re-call this to UPDATE it (rather than drifting) when reading "
  "changes your mind. Full replace each time: pass the whole plan. Keep it tight.",
  obj({"purpose": {"type": "string",
                   "description": "what the piece is FOR, in one line"},
       "carries": {"type": "array", "items": {"type": "string"},
                   "description": "what LEADS moment to moment (a line, the "
                   "picture, music, a mix) -- one entry, or one per section"},
       "structure": {"type": "array", "items": {"type": "string"},
                     "description": "the ORDERED intent: one entry per beat/"
                     "section in the order it should play. Free-form -- NOT a "
                     "template to fill."},
       "watch": {"type": "array", "items": {"type": "string"},
                 "description": "the hard spots to watch (a jump risk, a thin "
                 "stretch, a take choice)"},
       "note": {"type": "string",
                "description": "optional: one line on what changed your mind, "
                "when this call is an update"}})),
```

### 3.2 Add `act.set_plan(...)` and dispatch it (`act.py` + `tools._dispatch`)
The plan is document metadata, so mirror the metadata-verb pattern of
`act.set_audio`/`act.set_arc_intent` (`act.py:588-621`) — clone, mutate, return the
new doc; return the *original* doc unchanged on a no-op so `changed` stays False:
```python
# act.py -- beside set_arc_intent / set_grade
def set_plan(document: dict, *, purpose=None, carries=None,
             structure=None, watch=None, note=None) -> dict:
    """Write/replace document['plan'] -- the brain's durable, ordered plan
    (brain_plan_mechanism.plan.md). Full replace of the plan block; bumps
    `rev` and records an optional `updated_note`. Pure: returns a new doc."""
    doc = _clone(document)                       # act.py:44-53 (fresh copy)
    prev_rev = int((document.get("plan") or {}).get("rev") or 0)
    plan = {
        "purpose": (purpose or "").strip() or None,
        "carries": [str(c).strip() for c in (carries or []) if str(c).strip()],
        "structure": [str(s).strip() for s in (structure or []) if str(s).strip()],
        "watch": [str(w).strip() for w in (watch or []) if str(w).strip()],
        "rev": prev_rev + 1,
    }
    if note and note.strip():
        plan["updated_note"] = note.strip()
    doc["plan"] = plan
    return doc
```
Note: `act._clone` (`act.py:44-53`) does `dict(document)` then rebuilds
timeline/operations and pops `resolved`. Setting `doc["plan"]` on that fresh dict
guarantees `new is not doc`, so `changed` becomes True and the turn persists (§5).

Dispatch it in `tools._dispatch` (`tools.py:472-581`, in the ACT branch). Unlike a
timeline verb, echo the rendered plan (not `read_state`) so the tool result is a
tight confirmation:
```python
elif name == "set_plan":
    new = act.set_plan(doc, purpose=args.get("purpose"),
                       carries=args.get("carries"), structure=args.get("structure"),
                       watch=args.get("watch"), note=args.get("note"))
    changed = new is not doc
    return _json({"applied": changed,
                  "plan": (new.get("plan") if changed else None)}), new, changed
```
Place this branch BEFORE the shared `changed = new is not doc` / `read_state` echo
tail at `:583-598` (i.e. it returns directly, like the early returns already do),
so a plan write does not spuriously render a full `read_state`.

### 3.3 Discoverability
Add `"set_plan"` to the `"verbs"` list in `observe.affordances` (`observe.py:1811-1813`)
so the brain sees it in its menu of verbs.

---

## 4. Read path — the PLAN mirror (two injection points, both mirroring existing patterns)

The plan must be visible **both** from the first model call of a turn (the
plan-of-record written on a prior turn) **and** as it is written/updated *within*
the turn. That is exactly the split the edit state already uses:
`CURRENT PROGRAM MAP` in the cached system for the turn-start snapshot, and the
pushed `MIRROR` text_block for live within-turn state. We reuse both.

### 4.1 Turn-start snapshot — inside `converse._context_block` (`converse.py:508-566`)
Render the plan-of-record right next to `CURRENT PROGRAM MAP`. Add, just after the
program-map append (`converse.py:564-565`):
```python
    try:
        parts.append(observe.plan_mirror_text(document, building=False))
    except Exception:
        logger.exception("converse: plan mirror render failed (continuing without it)")
```
`_context_block` already receives `document` (the loaded, pre-turn doc — `respond`
passes `document`, `converse.py:654`), the same value `CURRENT PROGRAM MAP` reads,
so plan and timeline are the same snapshot. It sits in the cached system prefix and
is constant across the loop's internal iterations — correct, because within-turn
changes come through the pushed mirror (§4.2), exactly like the program map.
`plan_mirror_text` returns a short "no plan yet" line when absent (§4.3), so the
absence is visible from message one.

### 4.2 Within-turn push — inside `tools.run_edit_loop` (`tools.py:785-788`)
Push the plan mirror alongside the edit mirror after every round, reusing the
identical `text_block` mechanism:
```python
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
        convo.append(user_message(results))
```
`building=changed` gates the *loudness* of the absence nudge on real edit activity:
once the brain has changed the edit this turn but still has no `plan`, the mirror
says so in the imperative ("you're building without a written plan — call
set_plan"). This is the "make ABSENCE visible on an edit-changing turn" requirement.

### 4.3 The renderer — `observe.plan_mirror_text(document, *, building=False)` (`observe.py`, beside `mirror_text` `:1710-1713`)
Pure function of `document` only (no `ctx` needed — the plan is self-contained).
Idempotent (same plan → same text), so pushing it every step can never thrash the
loop, same discipline as `mirror()`'s docstring (`observe.py:1630-1632`).
```python
def plan_mirror_text(document: dict, *, building: bool = False) -> str:
    """The PLAN mirror (brain_plan_mechanism.plan.md §4): the brain's own
    durable plan echoed back compactly, right beside the EDIT MIRROR, so it
    builds against the ordered intent it committed to instead of drifting.
    Absent plan -> a visible nudge (loud when `building` and the edit has
    already moved this turn). Pure + idempotent."""
    plan = (document or {}).get("plan") or {}
    has = any(plan.get(k) for k in ("purpose", "carries", "structure", "watch"))
    if not has:
        return ("PLAN: none written yet -- you're building without a written plan. "
                "Reason to the ordered plan (purpose / what carries it / structure / "
                "what to watch) and call set_plan, then build against it."
                if building else
                "PLAN: none written yet -- write one with set_plan before you build.")
    lines = [f"PLAN (rev {plan.get('rev', 1)} -- build against this; "
             f"update with set_plan when reading changes your mind):"]
    if plan.get("purpose"):
        lines.append(f"  purpose: {plan['purpose']}")
    if plan.get("carries"):
        lines.append("  carries: " + "; ".join(plan["carries"]))
    if plan.get("structure"):
        lines.append("  structure:")
        for i, s in enumerate(plan["structure"], 1):
            lines.append(f"    {i}. {s}")
    if plan.get("watch"):
        lines.append("  watch: " + "; ".join(plan["watch"]))
    if plan.get("updated_note"):
        lines.append(f"  last change: {plan['updated_note']}")
    return "\n".join(lines)
```
Rendering the `structure` as an explicit numbered list is what makes "out-of-order
selection" self-evident: the brain reads the ordered plan directly above the
ordered EDIT MIRROR each turn.

---

## 5. Persistence & lifecycle — the plan lives IN the Edit Document (no migration)

**Decision: store the plan as `document["plan"]`, NOT as a new column.** It rides
the existing versioned document path end-to-end with zero new persistence code:

- **Created / written:** `act.set_plan` sets `document["plan"]` on the working copy
  during the loop.
- **Persisted:** `set_plan` makes `changed=True` (a new dict), so the router's
  existing `if result.changed: store.save_document(...)` (`edit_threads.py:134-135`)
  writes a new version — including on a **plan-only** turn (the brain planned but
  did not yet edit). No new save path.
- **Loaded / survives turns:** `converse.respond` loads it via
  `store.latest_document` (`converse.py:638`) and hands it in as `working`
  (`:643,:655`); `_context_block` renders it from `document`. So next turn's first
  model call already sees last turn's plan.
- **Seeded absent:** add `"plan": None` (or simply omit) to `_seed_document`
  (`converse.py:569-582`) for a brand-new thread — the mirror shows "none yet".
- **Cleared / superseded:** a fresh `set_plan` fully replaces it; on **undo**
  (restoring an older document version) the plan-of-record for that version comes
  back with the timeline it produced — a *bonus* of doc-versioning: the plan and
  the edit it explains travel together through history.
- **Human edits preserve it:** `put_document` spreads `{**doc, ...}`
  (`edit_threads.py:222`), so a manual timeline tweak keeps the plan.

**Why not a dedicated `edit_threads.plan` column (the `identity_map` model)?** That
pattern (`ingest_store.py:61-83`) fits run-scoped, write-once ingest metadata. The
plan is per-*document*, mutates every turn, and must move in lockstep with the
timeline through versioning/undo — which the document already gives for free. A
column would need a migration, its own get/set, threading through `respond` and the
router, and would *desync* from undo. The document is strictly simpler and more
correct here.

**No re-grade side effect:** `grade.job.compute_input_hash` (`grade/job.py:233-262`)
hashes only per-shot span/identity + `document["look"]` — it never reads `plan`, so
a plan-only change does NOT invalidate grades (`maybe_enqueue`,
`converse.py:668-671`, no-ops). Confirm this in the test plan (§7).

**Migration: NONE required.**

---

## 6. Prompt note — a minimal "THE PLAN" mechanism paragraph (`converse.py`)

The locked *PLAN BEFORE YOU BUILD* block (`:82-108`) must NOT be edited. But, just
as `_LOOP_SYSTEM` has a factual **THE MIRROR** paragraph (`:179-197`) telling the
brain the edit mirror mechanism exists, add a short factual **THE PLAN** paragraph
(a separate string segment in `_LOOP_SYSTEM`, e.g. right after THE MIRROR at `:197`)
so the brain knows *how* to write the plan down and that it is echoed back:

> "THE PLAN. Write the plan you reason to with set_plan -- purpose, what carries
> it, the ordered structure, and what to watch. It is DURABLE: echoed back to you
> every turn as the PLAN mirror (beside THE MIRROR) and carried across turns, so
> build against it and re-call set_plan to update it -- with one line on what
> changed your mind -- rather than drifting. If you start editing without one, the
> PLAN mirror will say so."

This is mechanism wiring (the brain can't use a verb it isn't told about), not the
prose "cleanup" that is Phase 3. Keep it to these few lines; broader reconciliation
of the planning prose is explicitly deferred (§8).

---

## 7. Additive & safe / fail-open

- Everything degrades to today's behavior when `plan` is absent: `plan_mirror_text`
  returns a short nudge string (never raises), and no sense/act requires a plan.
- Both injection sites are wrapped / guarded (`_context_block`'s `try/except`
  pattern, `:524` etc.; the loop push only appends non-empty text).
- **Always-on, no flag.** The mechanism is core to the drift fix and cheap (pure
  string render, a tiny doc key). Per the "prefer always-on" constraint, ship it on.
- `set_plan` on a bad/empty payload is a safe no-op-ish write (empty lists) — never
  an error.

---

## 8. Phase 3 handoff — the conformance seam (leave the hook, do NOT build it)

Phase 3 will add: (a) a **plan-conformance gate** that blocks FINISHING when the
built edit contradicts the written plan (e.g. timeline order vs `plan["structure"]`
order; a `plan["watch"]` item that a live MIRROR flag shows went unhandled), and
(b) **un-rubber-stampable signal checks** (a jump-cut / broken-line flag can't be
waved past without addressing the `watch` note that anticipated it). **Both are OUT
OF SCOPE here.**

Leave exactly one clearly-named seam so Phase 3 slots in, in
`tools._verify_before_finish` (`tools.py:633-723`) — the existing ordered done-gate.
Add, right after the structural check (`:660-666`, before the length check at
`:668`), a named no-op hook + comment:
```python
    # PHASE 3 SEAM (brain_plan_mechanism.plan.md §8): plan-vs-edit conformance.
    # Phase 3 implements _plan_conformance(working, ctx, working.get("plan"))
    # to return feedback when the built order/coverage contradicts the written
    # plan, or a watched risk went unaddressed -- surfaced once, like the other
    # stages (state-tracked). No-op in Phase 2: the plan is written + mirrored,
    # not yet enforced at the gate.
    #   feedback = _plan_conformance(working, ctx, working.get("plan"), state)
    #   if feedback is not None: return feedback
```
Do not add the `state` key or the function body in this phase — just the comment
seam naming the function and its insertion order, so Phase 3 is a localized add.

---

## 9. Test plan

Run everything with `backend/.venv/bin/python`, fully mocked (no network/DB), and
keep changed files pyflakes-clean. Extend the existing suites; mirror their fakes
(`scripts/test_tools_loop.py` uses `_ScriptedLLM` + a stubbed footage map;
`test_converse_context.py` exercises the context block).

### 9.1 Unit — `act.set_plan` (extend `scripts/test_observe_act.py`)
- `set_plan` on a seed doc returns a NEW doc (`new is not doc`) with normalized
  `plan` (empty inputs → empty lists, `rev=1`); whitespace-only entries dropped.
- Re-calling `set_plan` bumps `rev` to 2 and records `updated_note` from `note`.
- Original doc is not mutated (purity).

### 9.2 Unit — `observe.plan_mirror_text` (extend `scripts/test_observe_*`)
- Absent plan, `building=False` → the short "write one with set_plan" nudge.
- Absent plan, `building=True` → the loud "building without a written plan" nudge.
- Populated plan → contains "PLAN (rev N", a numbered `structure`, `carries`,
  `watch`, and (when set) "last change:". Idempotent: two calls, identical string.

### 9.3 Unit — dispatch + verb wiring (extend `scripts/test_tools_loop.py`)
- A scripted `set_plan` tool call flows through `_dispatch`: result JSON has
  `"applied": true` and the rendered `plan`; `res.changed is True`;
  `res.document["plan"]["structure"]` matches the input order.
- After the `set_plan` round, `llm.seen_messages[next]` (via the `_ScriptedLLM`
  `seen_messages` snapshot, `test_tools_loop.py:66-72`) contains a `text_block`
  with `"PLAN (rev"` — proving the plan mirror is pushed like the edit mirror
  (parallels `test_loop_pushes_the_mirror_after_a_tool_call`, `:115-136`).
- A `place` before any `set_plan` → the pushed plan mirror carries the loud
  `building` absence nudge (edit moved, no plan).
- `"set_plan"` appears in `observe.affordances(...)["verbs"]`.

### 9.4 Unit — turn-start injection (extend `scripts/test_converse_context.py`)
- `_context_block(file_ids, doc_with_plan, ctx)` output contains the rendered
  `PLAN` block next to `CURRENT PROGRAM MAP`.
- `_context_block` with a doc that has no `plan` contains the "none written yet"
  line (absence visible from message one).

### 9.5 Unit — persistence / no re-grade
- `grade.job.compute_input_hash(doc)` == `compute_input_hash({**doc, "plan": {...}})`
  — adding/altering a plan does NOT change the grade input hash (plan-only turns
  never re-grade).

### 9.6 Manual end-to-end (one real thread, staging LLM)
1. Open a thread on real footage; send an edit ask ("cut me a 60s explainer").
2. Confirm the brain calls `set_plan` early, and its reply says what it planned.
3. Inspect the persisted document version (`GET /api/edit/threads/{id}`): it carries
   `document.plan` with an ordered `structure`.
4. Send a follow-up that forces a rethink ("actually lead with the demo"). Confirm
   the brain re-calls `set_plan` (rev bumps, `updated_note` set) and reorders the
   timeline to match — i.e. it updates the plan instead of drifting.
5. Start a fresh thread and immediately ask for an edit; confirm the PLAN mirror's
   absence nudge appears in the trace before the first `set_plan`.

---

## 10. Exact change list per file

| File | Change |
|---|---|
| `backend/app/services/l3/act.py` | Add `set_plan(document, *, purpose, carries, structure, watch, note)` beside `set_arc_intent`/`set_grade` (`:604-621`), using `_clone` (`:44-53`); normalize + bump `rev`; return new doc (§3.2). |
| `backend/app/services/l3/tools.py` | Declare `set_plan` in `_specs()` (near `:334`) (§3.1); dispatch it in `_dispatch` ACT branch with a direct return (§3.2); push `observe.plan_mirror_text(working, building=changed)` beside the edit-mirror push in `run_edit_loop` (`:785-788`) (§4.2); add the Phase 3 conformance SEAM comment in `_verify_before_finish` after the structural check (`:666`) (§8). |
| `backend/app/services/l3/observe.py` | Add `plan_mirror_text(document, *, building=False)` beside `mirror_text` (`:1710-1713`) (§4.3); add `"set_plan"` to `affordances()["verbs"]` (`:1811-1813`) (§3.3). |
| `backend/app/services/l3/converse.py` | Render the plan-of-record in `_context_block` after `CURRENT PROGRAM MAP` (`:564-565`) via `observe.plan_mirror_text(document, building=False)`, guarded (§4.1); add `"plan": None` to `_seed_document` (`:569-582`) (§5); add the short **THE PLAN** mechanism paragraph to `_LOOP_SYSTEM` after THE MIRROR (`:197`) (§6). |
| tests | Extend `scripts/test_observe_act.py`, `scripts/test_tools_loop.py`, `scripts/test_converse_context.py` (+ an `observe` mirror test) per §9. |
| migrations | **None.** |

---

## 11. Summary of decisions
- **Artifact:** `document["plan"]` = `{purpose:str, carries:[str], structure:[str]
  (ordered, free-form), watch:[str], rev:int, updated_note?:str}` — fields lifted
  from the locked prompt's NAME/CARRIES/STRUCTURE/WATCH steps; `structure` is the
  ordered intent, deliberately un-templated.
- **Write verb:** a single `set_plan` (full replace); re-call to update.
- **Plan mirror injection:** turn-start in `converse._context_block` (beside
  `CURRENT PROGRAM MAP`); within-turn in `tools.run_edit_loop` (beside the
  `observe.mirror_text` push) — renderer `observe.plan_mirror_text`.
- **Storage:** the Edit Document jsonb (`edit_documents.document`), persisted by the
  existing changed→`save_document` path. **No migration.**
- **Fail-open, always-on, no flag.** Absence is made visible, loudly on an
  edit-changing turn.
- **Phase 3 seam:** a named, commented hook in `tools._verify_before_finish`
  (`_plan_conformance(...)`) — conformance gate + un-rubber-stampable checks are
  out of scope here.
