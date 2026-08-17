# Brain plan CONFORMANCE — a finish-time FIX-OR-SURFACE self-accountability check (Phase 3 of the planning hardening)

**Scope.** At the done-gate, hold the built edit against the plan the brain
itself wrote (its beats + `carries` + `watch`) AND against any material ceiling,
and force the brain to either **FIX** the gap or **SURFACE** it to the user as a
real, user-visible output — never finish silently with a known compromise. This
lands the long-reserved, currently-INERT `_plan_conformance(...)` seam in
`tools._verify_before_finish`. READ-ONLY handoff — implement from another chat.
Do NOT implement from this doc; do NOT commit.

**This is NOT a rebuild.** Everything shipped in the planning-hardening line
stays untouched: the `set_plan` verb + `{beat, need}` altitude
(`brain_plan_altitude.plan.md`), the `plan_mirror_text` renderer + its two
injection points, the per-step reasoning capture into `edit_turns.trace`
(observability). We touch a small, bounded set of symbols: `tools.
_verify_before_finish` (the seam + one new stage + one new structural block),
one new ACT verb (`wrap_up`) in `act.py` + its schema/dispatch in `tools.py`,
the FINISHING block in `converse._LOOP_SYSTEM`, and the loop's reply
construction in `tools.run_edit_loop`. Plus the existing done-gate test suite.

---

## 0. Motivation — the same disease found across BOTH validated live edits (n=2)

Two changes already shipped and were validated over two real edits — a pitch
([`33f100c4-…`](33f100c4)) and a padel reel ([`e46796f6-…`](e46796f6)):
- **Plan altitude** (`brain_plan_altitude.plan.md`): `document["plan"]
  ["structure"]` is ordered BEAT objects `{beat:str, need:"required"|"optional"}`
  at intent level, with **no clip ids**.
- **Reasoning observability**: per-step reasoning persisted in `edit_turns.
  trace` as `{"kind":"reasoning",...}` entries.

Watching those two edits closely surfaced **two systemic residual failures,
present in BOTH (n=2)** — this plan targets exactly them:

1. **Silent ceiling (Gap B).** The brain hits a real material ceiling — a
   required beat whose only takes are disfluent; the only fitting music asset is
   15.8s and silently dictates the whole reel's length — NAMES it plainly in its
   own reasoning, but leaves the user-facing surface fields (`open_questions` /
   `notes` / `summary`) EMPTY. The user is never told what was compromised or
   why. (In edit [`e46796f6-…`](e46796f6) the stored user-facing reply was an
   internal craft note, not a clean wrap-up — the surface leaked reasoning
   instead of carrying a real summary.)

2. **Declared-vs-delivered adherence.** The brain writes an intention into its
   plan/`watch` and then doesn't honor it — plan said "land action cuts on the
   music beat grid" but ZERO ops beat-snapped; purpose said "punchy / quick
   succession" but it shipped flat mid-length cuts at natural pace (its own
   reasoning admits one "drags"). Nothing checks the built edit against the plan
   it wrote.

Both are the **same disease: the brain isn't held accountable, at finish, to
the plan it wrote.** This plan installs that accountability as one new gate
stage (Part B) plus a structural guarantee that a known compromise cannot finish
silent (Part A).

**Design principle — mirror the existing FIT-TO-CRAFT discipline.** The
FINISHING block already forces the brain to "name exactly one of two reasons"
(the ask required it / the material can't support better) rather than
rubber-stamping (`converse.py:225-232`). The conformance stage copies that shape
exactly: present the facts, force an account per item, allow a pass ONLY with a
named reason — but the account here is per plan-beat and per declared craft
intention, and a named compromise now has a hard consequence (Part A: the
surface must be non-empty).

---

## 1. Current state to build on (verified against the code)

### 1.1 The done-gate ladder — `tools._verify_before_finish` (`backend/app/services/l3/tools.py:733-832`)

Called from the loop's finish path (`tools.py:877-885`): only when the turn
`changed` a non-empty edit and it's not the last turn. Returns a feedback string
(the loop appends it as a `user_message` and keeps going) or `None` (finish is
allowed). Each stage surfaces AT MOST ONCE via the `state` dict, so the loop
always terminates. The `state` dict is initialised at `tools.py:849-850`:
```849:850:backend/app/services/l3/tools.py
    verify = {"struct_tries": 0, "length_surfaced": False, "intent_surfaced": False,
             "craft_surfaced": False, "reviewed": False}
```

The ladder, IN ORDER, is:

| # | Stage | Kind | Blocks how | State key |
|---|---|---|---|---|
| 0 | **Structural legality** | deterministic (`observe.validate`) | HARD — returns until clean or `_STRUCT_MAX_TRIES=3` (`tools.py:760-766`) | `struct_tries` |
| — | **`_plan_conformance` seam** | *(inert comment today — `tools.py:768-775`)* | — | — |
| 1 | **Length** | deterministic (`observe.diagnose` "over target") | fix-or-justify, once (`tools.py:777-783`) | `length_surfaced` |
| 2 | **Stage 1 — FIT TO INTENT** | deterministic (`observe.review` `category=="ask"` flags) | once; a user-named feature actually present (`tools.py:785-799`) | `intent_surfaced` |
| 3 | **Stage 2 — FIT TO CRAFT** | LLM-driven checkpoint (prose verdict, gate never parses it) | fires unconditionally once for any changed non-empty edit (`tools.py:801-809`) | `craft_surfaced` |
| 4 | **Stage 3 — SPECIFIC FLAGS** | deterministic (`diagnose`/`review` rest) | advisory, once; skipped if the brain already self-reviewed (`tools.py:819-831`) | `reviewed` |

**The exact `_plan_conformance` seam — confirmed current location.** The prior
audit cited "around `tools.py:700-707`"; the seam has since moved (the reasoning-
capture helpers were inserted above the loop). It now sits at **`tools.py:768-775`**,
right AFTER the structural check and BEFORE the length check, as an inert comment
block:
```768:775:backend/app/services/l3/tools.py
    # PHASE 3 SEAM (brain_plan_mechanism.plan.md §8): plan-vs-edit conformance.
    # Phase 3 implements _plan_conformance(working, ctx, working.get("plan"))
    # to return feedback when the built order/coverage contradicts the written
    # plan, or a watched risk went unaddressed -- surfaced once, like the other
    # stages (state-tracked). No-op in Phase 2: the plan is written + mirrored,
    # not yet enforced at the gate.
    #   feedback = _plan_conformance(working, ctx, working.get("plan"), state)
    #   if feedback is not None: return feedback
```
`working` (the mutated doc, with `working.get("plan")` = the current plan of
record) and `ctx` are both in scope here; `state` is the per-turn dict. This is
where Part B slots in. **Ordering decision (§4.4): we place the conformance
stage LAST in the ladder, not here — see §4.4 for why; the seam comment moves
accordingly.**

**Stage 2's non-parsing discipline (the model we copy):**
```801:809:backend/app/services/l3/tools.py
    if working.get("timeline") and not state["craft_surfaced"]:
        state["craft_surfaced"] = True
        return ("AUTOMATIC CHECK -- Stage 2, fit to craft: forget the ask entirely -- "
                "judge this edit the way you'd judge any finished video handed to you "
                "cold. Does it look and sound like a real, high-quality piece of work? "
                "Fix what falls short. If you finish anyway with a known flaw, name in "
                "ONE line which of exactly two reasons applies: (a) the user's ask "
                "required it, or (b) the material can't support better. Any other "
                "reason ('looks fine anyway') isn't enough -- fix it instead.")
```
The gate NEVER parses the brain's reply. It forces a checkpoint by returning a
message once; the brain either acts (more tool calls → re-enters the gate) or
finishes with a named reason. Part B works identically.

### 1.2 The SURFACE fields — where they live and who writes them

`_seed_document` (`converse.py:594-608`) seeds three surface fields on every
Edit Document:
```603:605:backend/app/services/l3/converse.py
        "open_questions": [],
        "summary": "",
        "notes": [],
```
- **`open_questions`** is the ONLY one the API surfaces to the client:
  `GET /{thread_id}` returns `(document or {}).get("open_questions", [])`
  (`edit_threads.py:178`).
- **`summary`** / **`notes`** are written ONLY by the human `PUT /document` path
  (`edit_threads.py:223-226`), from client-supplied `body.summary`/`body.notes`.

**No verb or tool in the brain loop writes ANY of the three.** They are inert on
the auto/agentic path — which is precisely why Gap B ("names the ceiling in
reasoning, surface empty") happens: the brain has no write path to the surface,
and nothing requires one.

**The final user-facing reply** is produced in `tools.run_edit_loop`:
```930:932:backend/app/services/l3/tools.py
    reply = last_text or (
        "Before I go further I need your call on a couple of things below."
        if awaiting else "Done.")
```
`last_text` is the last non-empty assistant TEXT content of the loop
(`tools.py:859`). In a done-gate cycle that's the brain's LAST prose — i.e. its
internal craft justification, not a clean wrap-up. `respond()` returns it as
`ConverseResult.reply` (`converse.py:687,698`) and the router stores it verbatim
as the assistant turn (`edit_threads.py:131`). **This is why edit #2's stored
reply was an internal craft note** — the reply channel is "whatever the brain
last said", never a deliberate summary.

### 1.3 The plan at finish

`working.get("plan")` at the gate is the plan of record: `{purpose:str|None,
carries:[str], structure:[{beat,need}], watch:[str], rev:int, updated_note?}`
(post-altitude shape; §6 covers old shapes). Beats are **intent-level prose with
NO clip ids** — the central design constraint (§3). `carries` and `watch` are
free-form strings the brain wrote (`carries` = what leads moment to moment;
`watch` = the hard spots, e.g. "land action cuts on the beat grid", "keep it
punchy"). These are exactly the DECLARED CRAFT INTENTIONS the stage checks.

### 1.4 Op & cut representation for the deterministic hints

- **Cuts** (`observe.read_state`, `tools.py:552-585`): each main-line cut carries
  `pos`, `prog_start_ms`, `prog_end_ms`, `dur_ms`, `channel`, and — ONLY when the
  brain called `retime` — `pace_level` (+ `speed`) (`observe.py:581-585`). A cut
  with NO `pace_level` key is at natural pace (never retimed). This is the field
  the "punchy-but-flat" hint reads.
- **Ops** (`act.py`): `place_video` ops = `{op_id, type:"place_video", from_ms,
  src_in_ms, src_out_ms, ...}` (`act.py:166-174, 224-232`); `place_audio` ops =
  `{op_id, type:"place_audio", from_ms, to_ms, src_in_ms, src_out_ms,
  source_file_id, audio_kind, ...}` (`act.py:303-308`).
  **CRITICAL FINDING: there is NO `snap` flag recorded on an op.** Beat-snapping
  happens in `tools._dispatch` (`tools.py:531-536, 549-554, 585-589`), which
  computes the snapped coordinates and passes them to `act.*`; the resulting op
  stores only the final `from_ms`/`to_ms`, with no memory that a snap occurred.
  So the beat-sync hint canNOT read a flag — it must compare op program edges to
  the actual beat grid (§4.5.1).
- **Beat grid**: `observe._beat_grid(document, ctx)` (`observe.py:1043-1065`)
  returns one entry per musical source in play (an A2 bed or a main-line clip
  that `is_musical`), each with `onsets_ms` in program time; `tools._beat_grid_ms`
  (`tools.py:416-421`) flattens them. Empty when no musical source is placed.
  `observe.snap_to_beats(grid_ms, ms, max_move_ms=...)` (`observe.py:431`) is the
  nearest-onset helper the hint reuses to measure distance.

### 1.5 House style to mirror

`brain_plan_altitude.plan.md` and `brain_plan_mechanism.plan.md` §8 (the Phase-3
handoff that reserved this seam). Same conventions: fail-open, additive, no
migration (the surface fields already exist in `_seed_document`; jsonb), and the
"prompt DISCIPLINE + robust-but-few deterministic signals" split.

---

## 2. Design overview — the LLM-vs-deterministic split (read this first)

The single most important design fact:

> **Beats are intent-level PROSE with no clip ids. You CANNOT deterministically
> map a beat to a built cut.** ("differentiator — shown through the demo" does
> not name which of N placed cuts, if any, delivered it.)

Therefore conformance is **primarily an LLM SELF-ACCOUNTABILITY stage**, not a
deterministic beat→cut gate. The gate PRESENTS the plan (beats + `carries` +
`watch`) alongside the built timeline and a few advisory HINTS, and REQUIRES the
brain to account for each item as **delivered / fixed / compromised-and-surfaced**
— exactly the way Stage 2 forces a blind craft verdict. Deterministic code does
NOT decide conformance; it only contributes cheap, robust HINTS that make the LLM
check sharper.

**Two parts, two mechanisms:**

- **Part A — Surface as a required, real output (STRUCTURAL, deterministic).**
  The ONE thing we enforce structurally and robustly: when finishing an edit that
  changed, the surface channel must carry a genuine user-visible wrap-up, and it
  must be non-empty whenever a compromise is in play. Finishing with a known
  compromise and empty `summary`/`open_questions`/`notes` is BLOCKED. This is
  deterministic because "is the surface field non-empty?" is a fact, unlike
  "did beat 3 get delivered?".

- **Part B — Plan-conformance accountability (LLM-DRIVEN checkpoint).** A new
  once-fired gate stage that presents plan-vs-timeline + the hints and forces the
  brain to account per required beat + per declared craft intention, with the
  fix-or-surface discipline mirroring FIT-TO-CRAFT. The gate never parses the
  account; it forces the checkpoint, and Part A guarantees a named compromise
  can't finish silent.

**What deterministic code does NOT do (scope guard, §7):** no beat→cut mapping,
no "required beat N is missing" verdict, no hard-fail on a hint. Hints are
advisory inputs to the LLM stage; the LLM owns the conformance judgment.

---

## 3. The write path for "surface" — a new `wrap_up` ACT verb

The cleanest write path (none exists today, §1.2). A tiny ACT verb that writes
the three real, persisted, API-surfaced fields, so "surface" is a genuine output
the user sees, not leftover reasoning.

### 3.1 New helper/verb — `act.wrap_up` (`backend/app/services/l3/act.py`)

Add beside the other verbs (a natural spot: just below `set_plan`, since both
write top-level document keys). Pure, returns a new doc (so `changed` persists it
and the reply picks it up):
```python
def wrap_up(document: dict, *, summary=None, open_questions=None, notes=None) -> dict:
    """Write the user-facing WRAP-UP onto the document's surface fields
    (brain_plan_conformance.plan.md Part A): `summary` -- one short, clean
    user-facing recap of what you built and, when one exists, the compromise
    you made and why; `open_questions` / `notes` -- anything the user should
    decide or know. This is the ONLY brain write path to the surface the API
    shows the user (GET /threads/{id}: open_questions; the stored reply/summary).
    Full replace of whichever field is passed; a field left None is untouched.
    Pure: returns a new doc (so the turn persists and the reply carries it)."""
    if summary is None and open_questions is None and notes is None:
        return document
    doc = _clone(document)
    if summary is not None:
        doc["summary"] = str(summary).strip()
    if open_questions is not None:
        doc["open_questions"] = [str(q).strip() for q in open_questions if str(q).strip()]
    if notes is not None:
        doc["notes"] = [str(n).strip() for n in notes if str(n).strip()]
    return doc
```
`_clone` (`act.py:44-53`) guarantees `new is not doc` when any field is passed,
so `changed` flips True and the router persists the surface (§1.2). Note:
`wrap_up` alone flipping `changed` does NOT itself re-trigger the whole gate
ladder in a way that loops forever — the other stages are all `state`-tracked
(fire once) and Part A's own block clears the moment `summary` is non-empty (§4.3).

### 3.2 Schema + dispatch (`backend/app/services/l3/tools.py`)

Add to the ACT verbs in `_specs()` (near `set_plan`, `tools.py:344-383`):
```python
        S("wrap_up", "Write the user-facing WRAP-UP for this edit: `summary` -- "
          "ONE short, clean recap for the user of what you built and, when you "
          "made a compromise (a required beat you couldn't land cleanly, a ceiling "
          "the material or an asset forced), what it was and why. `open_questions` "
          "-- anything you need the user to decide; `notes` -- anything they should "
          "know. This is the ONLY channel the user actually SEES besides your reply, "
          "so when you finish with a known compromise it MUST be stated here -- "
          "never finish a compromised edit with an empty summary. Keep it tight and "
          "human; it is NOT a place to dump internal craft reasoning.",
          obj({"summary": {"type": "string"},
               "open_questions": {"type": "array", "items": {"type": "string"}},
               "notes": {"type": "array", "items": {"type": "string"}}})),
```
Dispatch in `_dispatch` (beside `set_plan`, `tools.py:629-639`) — a direct
return echoing the written surface (tight confirmation, not a `read_state`
dump):
```python
        elif name == "wrap_up":
            new = act.wrap_up(doc, summary=args.get("summary"),
                              open_questions=args.get("open_questions"),
                              notes=args.get("notes"))
            changed = new is not doc
            return _json({"applied": changed,
                          "surface": {"summary": new.get("summary"),
                                      "open_questions": new.get("open_questions"),
                                      "notes": new.get("notes")} if changed else None}), new, changed
```
Optionally add `"wrap_up"` to `observe.affordances(...)["verbs"]` for symmetry
with `set_plan` (mirrors `brain_plan_mechanism.plan.md §3.3`); not required for
function.

### 3.3 Make the stored reply a real wrap-up, not leftover reasoning

In `tools.run_edit_loop`, prefer the brain's deliberate `summary` over the last
internal prose when building the reply (§1.2 diagnosis). BEFORE (`tools.py:930-932`):
```python
    reply = last_text or (
        "Before I go further I need your call on a couple of things below."
        if awaiting else "Done.")
```
AFTER:
```python
    # brain_plan_conformance.plan.md Part A: the user-facing reply prefers the
    # brain's deliberate wrap-up (act.wrap_up -> document["summary"]) over the
    # last internal prose line, so a finished edit surfaces a clean recap (and
    # any compromise) instead of leftover craft reasoning. Falls back to
    # last_text (chat turns, no-op turns) so nothing regresses.
    summary = (working.get("summary") or "").strip()
    reply = (summary if summary and not awaiting else last_text) or (
        "Before I go further I need your call on a couple of things below."
        if awaiting else "Done.")
```
Fail-open: no `summary` written → identical to today's behavior. `awaiting`
(ask_user) still prefers `last_text` so a paused turn's question framing is
unchanged.

---

## 4. Part A + Part B in the gate — `tools._verify_before_finish`

### 4.1 New state keys (`tools.py:849-850`)

BEFORE:
```python
    verify = {"struct_tries": 0, "length_surfaced": False, "intent_surfaced": False,
             "craft_surfaced": False, "reviewed": False}
```
AFTER:
```python
    verify = {"struct_tries": 0, "length_surfaced": False, "intent_surfaced": False,
             "craft_surfaced": False, "reviewed": False,
             "conformance_surfaced": False, "surface_blocked": 0}
```
`conformance_surfaced` fires Part B once. `surface_blocked` is a small COUNTER
(not a bool) so Part A can block a bounded number of times (see §4.3 for why a
counter, not once) and never wedge the loop.

Mirror this in the test helper `_gate_state()` (`test_tools_loop.py:425-427`).

### 4.2 Part B — the conformance stage (the LLM checkpoint)

New module-private function above `_verify_before_finish` (beside the hint
helpers, §4.5). It builds a once-fired feedback string presenting plan +
timeline + hints and requiring an account; it does NOT parse anything:
```python
def _plan_conformance(working: dict, ctx: EditContext,
                      plan: dict | None, state: Dict[str, Any]) -> str | None:
    """Part B (brain_plan_conformance.plan.md): the finish-time plan-conformance
    accountability CHECKPOINT. Beats are intent-level prose with no clip ids, so
    this is an LLM self-accountability stage -- it PRESENTS the written plan
    (beats + carries + watch) beside the built timeline and a few advisory HINTS,
    and requires the brain to account for each required beat + each declared craft
    intention (delivered / fixed / compromised-and-surfaced), mirroring the
    FIT-TO-CRAFT discipline. Fires at most once (state-tracked). Fail-open: any
    error -> None (the edit still finishes). Returns feedback or None."""
    if state.get("conformance_surfaced"):
        return None
    if not (plan and (plan.get("structure") or plan.get("carries") or plan.get("watch"))):
        return None                     # no plan of record -> nothing to conform to
    try:
        required = [b for b in (_beat_view(e) for e in plan.get("structure") or [])
                    if b[0] and b[1] != "optional"]           # (beat_text, need)
        declared = list(plan.get("carries") or []) + list(plan.get("watch") or [])
        hints = _conformance_hints(working, ctx, plan)        # advisory, few (§4.5)
    except Exception:
        logger.exception("_plan_conformance: presentation build failed (finishing)")
        return None
    if not required and not declared and not hints:
        return None
    state["conformance_surfaced"] = True
    lines = ["AUTOMATIC CHECK -- plan conformance: hold the edit you BUILT against "
             "the plan you WROTE. For EACH item below, account in one line: "
             "DELIVERED (it's in the cut), FIXED (you're about to build/repair it), "
             "or COMPROMISED (the material genuinely can't land it) -- and a "
             "COMPROMISE MUST be surfaced to the user via wrap_up (summary/"
             "open_questions), never left silent. Same rule as fit-to-craft: a gap "
             "may pass ONLY for one of two reasons -- the ask required it, or the "
             "material can't support better -- named plainly."]
    if required:
        lines.append("  required beats (each must be delivered, or its ceiling surfaced):")
        lines += [f"    - {b}" for b, _ in required]
    if declared:
        lines.append("  declared craft intentions (carries / watch -- honor or surface):")
        lines += [f"    - {d}" for d in declared]
    if hints:
        lines.append("  advisory signals (cheap checks -- confirm or refute, don't trust blindly):")
        lines += [f"    - {h}" for h in hints]
    return "\n".join(lines)
```
`_beat_view` is the back-compat coercion from `brain_plan_altitude.plan.md §5.1`
(dict `{beat,need}` OR old bare string → `(text, need|None)`). It lives in
`observe.py`; import it (`from app.services.l3.observe import _beat_view` at the
top of `tools.py`, or reference `observe._beat_view`). If it is module-private
and awkward to import, inline the same two-line coercion in `tools.py` — the
altitude plan §6 requires any new reader use `_beat_view`-style coercion, never
`entry["beat"]`.

### 4.3 Part A — the surface-non-empty structural block

Deterministic. It fires when the edit changed and a COMPROMISE IS IN PLAY but the
surface is empty. "Compromise in play" is detected robustly from two cheap
signals available at the gate: (a) the conformance stage has fired (a compromise
account was demanded), and/or (b) any advisory hint is live. Because these are
the same conditions that make the conformance stage meaningful, Part A naturally
sequences AFTER Part B in the ladder.

New helper:
```python
def _surface_is_empty(working: dict) -> bool:
    """True when NONE of the user-facing surface fields carry content
    (brain_plan_conformance.plan.md Part A). summary is the primary channel;
    open_questions/notes also count as surfaced."""
    if (working.get("summary") or "").strip():
        return False
    if [q for q in (working.get("open_questions") or []) if str(q).strip()]:
        return False
    if [n for n in (working.get("notes") or []) if str(n).strip()]:
        return False
    return True
```
Block logic, inserted in the ladder (see §4.4 for exact position):
```python
    # PART A (brain_plan_conformance.plan.md): a KNOWN compromise must not finish
    # silent. "Compromise in play" = the conformance stage has run (an account was
    # demanded) AND/OR a live advisory hint. If so and the surface is empty, block
    # -- require a real wrap_up. Bounded by surface_blocked so a stubborn brain
    # still terminates (fail-open after _SURFACE_MAX_BLOCKS).
    compromise_in_play = state.get("conformance_surfaced") and bool(
        _conformance_hints(working, ctx, working.get("plan")))
    if (working.get("timeline") and compromise_in_play and _surface_is_empty(working)
            and state["surface_blocked"] < _SURFACE_MAX_BLOCKS):
        state["surface_blocked"] += 1
        return ("AUTOMATIC CHECK -- surface: this edit carries a known compromise "
                "but nothing is surfaced to the user. Call wrap_up with a short, "
                "human `summary` that states plainly what you built and what was "
                "compromised and why (and open_questions if the user should weigh "
                "in). Don't finish a compromised edit silent.")
```
with `_SURFACE_MAX_BLOCKS = 2` beside `_STRUCT_MAX_TRIES` (`tools.py:712`).

**Why a bounded counter, not once:** Part A must actually GATE (re-block until
the surface is filled), unlike the advisory stages that fire once and move on.
But it must never wedge the loop, so it caps like the structural check does.
After the cap the edit finishes (fail-open) — the compromise stays unsurfaced,
but that's a bounded, logged degradation, not a hang.

**Design choice — require a summary GENERALLY (cheap, always useful):** we scope
the HARD block to the compromise case (above) to avoid nagging on trivial clean
edits, but the FINISHING prose (§4.6) additionally ASKS for a one-line `wrap_up`
summary on every substantive finish. The prose is the always-on nudge (cheap,
frequently useful); the deterministic block is the hard floor for the compromise
case. This keeps the deterministic surface strictly for the fact we can verify
("known compromise + empty surface") and leaves "always summarize" as
discipline, matching how the ladder already splits Stage 2 (LLM) from Stage 3
(deterministic).

### 4.4 Ladder ORDER — where the two pieces slot in

The conformance account is a HOLISTIC, whole-edit judgment like Stage 2 — it
belongs after the mechanical checks (structural, length, intent) and after the
craft verdict, so the brain has already been pushed to fix obvious craft before
being asked to reconcile against its own plan. Part A must come AFTER Part B
(the account demand is what puts a compromise "in play"). So the new order,
appended after the existing Stage 3 (`tools.py:832`, the final `return None`):

```
0. structural   (unchanged, tools.py:760-766)
   [seam comment moves here -> now points to §4.4 order below]
1. length       (unchanged, tools.py:777-783)
2. Stage 1 intent (unchanged, tools.py:785-799)
3. Stage 2 craft  (unchanged, tools.py:801-809)
4. Stage 3 flags  (unchanged, tools.py:819-831)
5. Part B  plan conformance   (NEW -- _plan_conformance)
6. Part A  surface-non-empty  (NEW -- the compromise block)
   return None
```

Concretely: (1) UPDATE the inert seam comment at `tools.py:768-775` to note the
stage now lives at the ladder's END (leaving a one-line pointer, so a future
reader isn't misled), and (2) insert Parts B then A immediately before the final
`return None` at `tools.py:832`:
```python
    # (existing Stage 3 block ends at tools.py:831)
    # Part B (brain_plan_conformance.plan.md): plan-conformance accountability.
    feedback = _plan_conformance(working, ctx, working.get("plan"), state)
    if feedback is not None:
        return feedback
    # Part A: never finish a KNOWN compromise with an empty surface.
    compromise_in_play = state.get("conformance_surfaced") and bool(
        _conformance_hints(working, ctx, working.get("plan")))
    if (working.get("timeline") and compromise_in_play and _surface_is_empty(working)
            and state["surface_blocked"] < _SURFACE_MAX_BLOCKS):
        state["surface_blocked"] += 1
        return (...)  # the surface block message (§4.3)
    return None
```

> **Ordering rationale, explicit:** placing conformance LAST (not at the physical
> seam line) means every deterministic craft nudge the brain would act on anyway
> has already fired, so the conformance account is about what SURVIVED those
> fixes — the genuine residual gaps — not noise the brain was about to clean up.
> It also guarantees Part B runs before Part A within one gate invocation, so a
> compromise named in Part B is caught by Part A in the SAME finish attempt if
> the brain tries to end without surfacing.

### 4.5 Deterministic hint helpers (`_conformance_hints`) — robust, few, advisory

One dispatcher + two detectors. Each returns a short human string or nothing;
`_conformance_hints` collects the live ones. Advisory ONLY — they feed the LLM
stage's presentation and Part A's "compromise in play" test; they never hard-fail.

```python
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

#### 4.5.1 Beat-sync declared but no op lands on the grid

`plan` text mentions beat-sync intent AND a musical grid exists AND NO placed op
edge lands on it (within the snap cap). Because ops record no snap flag (§1.4),
we measure op edges against the real grid with the existing `snap_to_beats`
helper.
```python
_SYNC_WORDS_RE = re.compile(r"\b(beat|grid|snap|sync|on the beat|downbeat|onset)\b", re.IGNORECASE)


def _plan_text(plan: dict) -> str:
    """All free-form plan prose the brain wrote -- purpose + carries + watch +
    beat texts -- lowercased into one blob for keyword hints."""
    parts = [str(plan.get("purpose") or "")]
    parts += [str(x) for x in (plan.get("carries") or [])]
    parts += [str(x) for x in (plan.get("watch") or [])]
    parts += [_beat_view(e)[0] for e in (plan.get("structure") or [])]
    return " ".join(parts).lower()


def _hint_beatsync_declared_but_absent(working, ctx, plan) -> str | None:
    if not _SYNC_WORDS_RE.search(_plan_text(plan)):
        return None
    grid = _beat_grid_ms(working, ctx)          # tools.py:416
    if not grid:
        return None                              # no music grid -> nothing to sync TO
    ops = [o for o in working.get("operations") or []
           if o.get("type") in ("place_video", "place_audio")]
    if not ops:
        return None
    TOL = 60                                      # ms; well inside _SNAP_CAP_MS=400
    def _on_grid(ms):
        s = observe.snap_to_beats(grid, ms, max_move_ms=_SNAP_CAP_MS)
        return s.get("snapped") and abs(int(s.get("ms", ms)) - int(ms)) <= TOL
    aligned = any(_on_grid(o.get("from_ms")) or (o.get("to_ms") is not None and _on_grid(o["to_ms"]))
                  for o in ops)
    if aligned:
        return None
    return ("plan/watch calls for beat-sync but no placed overlay/bed edge lands "
            "on the music grid -- either snap the ones that should hit the beat, "
            "or surface that you chose not to.")
```
Robustness: fires only when the brain BOTH declared sync AND a grid exists AND
ops exist AND none align. No music, no ops, or any aligned edge → silent (no
false-fire). `TOL=60ms` is generous enough to count an intentionally-snapped
edge yet reject a coincidental natural landing.

#### 4.5.2 Punchy declared but cuts are flat/natural

`plan` text mentions punchy/quick pacing AND the main line is mostly long cuts
at natural pace (no `pace_level` toward faster, per §1.4).
```python
_PUNCHY_WORDS_RE = re.compile(r"\b(punch\w*|quick|fast|snappy|rapid|energetic|tight cut\w*|"
                              r"quick succession|fast[- ]?paced)\b", re.IGNORECASE)
_LONG_CUT_MS = 4000          # a "long" main-line cut for a piece that declared punch
_FAST_PACE = {"faster", "much_faster"}   # act._PACE_STEPS toward quick


def _hint_punchy_declared_but_flat(working, ctx, plan) -> str | None:
    if not _PUNCHY_WORDS_RE.search(_plan_text(plan)):
        return None
    try:
        cuts = observe.read_state(working, ctx).get("cuts") or []
    except Exception:
        return None
    main = [c for c in cuts if c.get("dur_ms") is not None]
    if len(main) < 3:
        return None                              # too short to judge pace
    long_cuts = [c for c in main if int(c.get("dur_ms") or 0) >= _LONG_CUT_MS]
    any_fast = any((c.get("pace_level") in _FAST_PACE) for c in main)
    if any_fast or len(long_cuts) < (len(main) + 1) // 2:   # <half are long -> fine
        return None
    return (f"plan calls for punchy/quick pacing but {len(long_cuts)} of "
            f"{len(main)} main-line cuts run long ({_LONG_CUT_MS}ms+) at natural "
            "pace -- tighten/retime the ones that drag, or surface that the "
            "material can't go faster.")
```
Robustness: fires only when punch was DECLARED AND a majority of a ≥3-cut main
line is long AND nothing was retimed faster. Any fast retime, a short piece, or a
mostly-tight main line → silent. Thresholds (`_LONG_CUT_MS`, majority) are
deliberately conservative to avoid nagging a legitimately varied edit.

> **These two hints are the whole deterministic contribution.** They map 1:1 to
> the two n=2 adherence failures (beat-snap declared/absent; punchy declared/
> flat). Keep the set SMALL and each fail-open — more hints = more false-fire
> risk and more thrash. Resist adding a third without a validated failure.

### 4.6 FINISHING prose additions (`converse._LOOP_SYSTEM`, `converse.py:219-236`)

Append one new numbered step to the FINISHING block, AFTER (3) SPECIFIC FLAGS,
in the file's concatenation style. This is where the plan-conformance stage's
discipline is stated to the brain (the gate only forces the checkpoint):
```python
    "(3) SPECIFIC FLAGS -- the rest (jump-cuts, speaker runs, low-energy stretches, "
    "redundant takes, rough heads/tails, overlay fit, audio gaps, loudness "
    "balance) is advisory -- you've likely already seen most of it in the mirror "
    "as you worked; act on what serves the goal, ignore the rest.\n"
    "(4) FIT TO PLAN -- hold what you BUILT against the plan you WROTE. For each "
    "REQUIRED beat and each declared craft intention (what carries it, what you "
    "said to watch), account in one line: it's DELIVERED, you're about to FIX it, "
    "or it's a COMPROMISE the material genuinely forced. Same two-reasons rule as "
    "fit to craft -- a gap passes only if the ask required it or the material "
    "can't support better. A COMPROMISE is never silent: state it to the user "
    "with wrap_up (a short human `summary`, plus open_questions if it's theirs to "
    "weigh). Whenever you finish a substantive edit, leave a clean one-line "
    "wrap_up summary of what you built -- that recap, not your last internal note, "
    "is what the user reads."
```
No other FINISHING line changes. Keep the two existing "name one of two reasons"
clauses (Stage 2) intact — Stage (4) deliberately echoes that exact discipline so
the brain reads it as the same rule applied to the plan.

---

## 5. Back-compat / additive / fail-open

- **No migration.** `open_questions`/`summary`/`notes` already seed on every doc
  (`converse.py:603-605`); `document` is jsonb. `wrap_up` writes existing keys.
- **Old documents / old plans.** `_plan_conformance` no-ops when there's no plan
  or an empty plan. Old-shape `structure:[str]` is handled by `_beat_view`
  coercion (`brain_plan_altitude.plan.md §5.1/§6`) — never `entry["beat"]`. A
  plan with no `watch`/`carries` simply contributes fewer presented items; the
  stage still fires on required beats alone, or no-ops if there's nothing.
- **Fail-open, every new path.** `_plan_conformance`, each hint, and
  `_conformance_hints` wrap their work in try/except → on ANY error return
  None/[] so the edit still finishes. Part A is bounded by `_SURFACE_MAX_BLOCKS`
  so a stubborn brain still terminates. `wrap_up` with no args is a no-op
  (`new is doc`, `changed` unchanged). The reply change (§3.3) falls back to
  `last_text` when no `summary` exists — identical to today on chat/no-op turns.
- **Loop can't wedge.** Part B is once-fired (`conformance_surfaced`); Part A is
  capped; every other stage is unchanged. Terminates exactly as before, plus at
  most (`_SURFACE_MAX_BLOCKS` + 1) extra gate passes.
- **Nothing touches** the altitude reshape, the observability trace capture, the
  plan mirror, or the `set_plan` verb. Additive only.

---

## 6. Testing

Run with `backend/.venv/bin/python`, fully mocked (no network/DB), changed files
pyflakes-clean. Extend the EXISTING done-gate suite in
`scripts/test_tools_loop.py` (helpers `_gate_state`, `_clean_doc`, `_ctx`,
`_struct` at `:425-436`) and the verb-dispatch tests (`:301+`). Add the new
`_gate_state()` keys (§4.1).

### 6.1 `act.wrap_up` + dispatch (`scripts/test_tools_loop.py`, mirror `:301`)
- `test_wrap_up_writes_surface_fields`: `act.wrap_up(doc, summary="tightened to
  55s; the differentiator take was disfluent so I used the slide+VO route",
  open_questions=["want the raw take instead?"])` returns a new doc with
  `summary` set (stripped) and `open_questions` normalized; a `None` field is
  untouched; all-None is a no-op (`new is doc`).
- `test_dispatch_wrap_up_applies_and_echoes_surface`: dispatch `"wrap_up"` →
  `applied True`, echo carries `surface.summary`, `working["summary"]` set.

### 6.2 Part A — surface block (`scripts/test_tools_loop.py`, mirror `:458`)
- `test_gate_blocks_finish_when_compromise_in_play_and_surface_empty`: with
  `conformance_surfaced=True` and a live hint (stub `_conformance_hints` →
  `["…"]`) and empty surface, the gate returns a `"surface"` message and bumps
  `surface_blocked`.
- `test_gate_surface_block_clears_once_summary_written`: same state but
  `working["summary"]="…"` → the surface block does NOT fire (returns None or
  falls through).
- `test_gate_surface_block_is_bounded`: repeated calls with empty surface stop
  blocking after `_SURFACE_MAX_BLOCKS` (fail-open terminates).
- `test_gate_no_surface_block_without_a_compromise`: `conformance_surfaced=True`
  but `_conformance_hints → []` and empty surface → no block (clean edits aren't
  nagged).

### 6.3 Part B — conformance stage (`scripts/test_tools_loop.py`)
- `test_gate_conformance_fires_once_with_required_beats_and_watch`: a doc whose
  `plan` has a required beat + a `watch` line; gate reaches Part B (drive the
  earlier stages to None via existing helpers) → a `"plan conformance"` message
  listing the required beat + the watch item; a second call returns None
  (`conformance_surfaced`).
- `test_gate_conformance_noops_without_a_plan`: `plan` absent/empty → Part B
  returns None; the ladder finishes as today.
- `test_gate_conformance_old_shape_structure_of_strings`: `structure:["cold
  open","the demo"]` (pre-altitude) is coerced via `_beat_view`, presented as
  required beats, does not raise (back-compat).

### 6.4 Hint helpers (`scripts/test_tools_loop.py`)
- `test_hint_beatsync_fires_when_declared_and_no_op_on_grid`: plan `watch`
  mentions "land cuts on the beat grid", a musical `place_audio` bed gives a grid
  (`_beat_grid` stub or a real `is_musical` audio feature in `ctx`), a
  `place_video` op whose `from_ms` is far off any onset → hint fires.
- `test_hint_beatsync_silent_when_an_op_lands_on_grid`: same but the op's
  `from_ms` == an onset (within `TOL`) → no hint.
- `test_hint_beatsync_silent_without_music_or_ops`: no musical source, or no ops
  → no hint (no false-fire).
- `test_hint_punchy_fires_when_declared_and_cuts_long_and_natural`: plan purpose
  "punchy, quick succession", ≥3 main-line cuts each ≥`_LONG_CUT_MS`, none
  retimed faster → hint fires.
- `test_hint_punchy_silent_when_retimed_faster_or_mostly_tight`: same declaration
  but one cut `pace_level="faster"` (or a majority short) → no hint.
- `test_hint_punchy_silent_when_not_declared`: long/natural cuts but no punchy
  words → no hint.
- `test_conformance_hints_fail_open`: a hint raising internally is swallowed;
  `_conformance_hints` returns the surviving hints (or []).

### 6.5 Ladder order + reply (`scripts/test_tools_loop.py`)
- `test_gate_ladder_order_conformance_after_stage3_then_surface`: with all
  earlier stages stubbed to have surfaced, one gate invocation yields Part B
  first, and — on the next attempt with a live compromise + empty surface — Part
  A; confirms Part B precedes Part A.
- `test_loop_reply_prefers_wrap_up_summary`: a scripted loop where the brain
  calls `wrap_up(summary="…")` then finishes → `LoopResult.reply == "…"` (not the
  last internal text); a loop with no `wrap_up` → reply falls back to `last_text`
  (regression guard for §3.3); an `awaiting_user` turn still uses `last_text`.

### 6.6 Imports / pyflakes + manual e2e
- `import act, tools, observe, converse` cleanly; no unused names; register the
  new test fns in the `__main__` runner (`test_tools_loop.py:716`).
- **Manual e2e:** re-run the two motivating scenarios. (1) A piece where a
  required beat's only takes are disfluent — confirm the brain, at finish, either
  routes/fixes OR calls `wrap_up` naming the ceiling, and CANNOT finish with an
  empty surface (Part A blocks); confirm the stored reply is the clean `summary`,
  not an internal note; confirm `GET /threads/{id}` shows the `open_questions`.
  (2) A reel where the brain's `watch` says "beat-sync" but it beat-snapped
  nothing, or purpose says "punchy" but cuts run long/natural — confirm the
  conformance stage presents the hint and the brain either tightens/snaps or
  surfaces the ceiling.

---

## 7. Scope discipline (what this plan explicitly does NOT do)

- **NO deterministic beat→cut mapping.** Beats are id-less intent prose (§2); the
  conformance judgment is the LLM's. Deterministic code contributes only the two
  advisory hints (§4.5). Do not build a "required beat N is missing" verdict.
- **Do NOT touch** the altitude reshape (`brain_plan_altitude.plan.md`), the
  reasoning-capture observability, the plan mirror, or `set_plan`. Additive only.
- **Budget misallocation is OUT OF SCOPE.** The brain sometimes thrashes on
  low-value corners (over-polishing a minor seam while a bigger gap waits) — a
  real observed inefficiency, but a DIFFERENT disease (effort allocation, not
  plan accountability). Note it as a WATCH ITEM for a future pass; do NOT build
  anything for it here. Adding effort-budgeting logic to this gate would blur the
  fix-or-surface contract this plan is about.
- **Keep hints few.** Two, each fail-open, each mapped to a validated n=2
  failure. Resist a third without evidence (§4.5).

---

## 8. Exact change list per file

| File | Change |
|---|---|
| `backend/app/services/l3/act.py` | Add `wrap_up(document, *, summary, open_questions, notes)` (pure, `_clone`, field-wise replace) beside `set_plan` (§3.1). |
| `backend/app/services/l3/tools.py` | (a) Add `wrap_up` to `_specs()` + dispatch in `_dispatch` (direct return, §3.2). (b) Add `_SURFACE_MAX_BLOCKS`, `_SYNC_WORDS_RE`, `_PUNCHY_WORDS_RE`, `_LONG_CUT_MS`, `_FAST_PACE`; add `_plan_text`, `_surface_is_empty`, `_hint_beatsync_declared_but_absent`, `_hint_punchy_declared_but_flat`, `_conformance_hints`, `_plan_conformance` (§4.2-4.5). (c) Add `conformance_surfaced`/`surface_blocked` to the `verify` state (`:849-850`, §4.1). (d) Insert Part B then Part A before the final `return None` (`:832`) and UPDATE the inert seam comment (`:768-775`) to point at the new tail location (§4.4). (e) Prefer `working["summary"]` in the reply (`:930-932`, §3.3). |
| `backend/app/services/l3/converse.py` | Append step "(4) FIT TO PLAN" + the always-leave-a-wrap_up-summary line to the FINISHING block (`:219-236`, §4.6). No other prompt change. |
| `backend/app/routers/edit_threads.py` | Optional: also surface `summary` in `GET /{thread_id}` alongside `open_questions` (`:174-180`) so the client can show the recap. Not required for the gate to function. |
| tests | Extend `scripts/test_tools_loop.py`: `wrap_up`/dispatch, Part A block (fires/clears/bounded/no-false-fire), Part B (fires-once/no-plan/old-shape), both hints (fire + silent + fail-open), ladder order, reply-prefers-summary. Update `_gate_state()` with the two new keys (§6). |
| migrations | **None** (surface fields pre-seeded; jsonb) (§5). |

---

## 9. Summary of decisions
- **Confirmed seam:** `_plan_conformance` is inert at `tools.py:768-775` (moved
  from the prior audit's `:700-707`), after the structural check, before length.
  We LAND it but relocate the live stage to the ladder's END (§4.4) so it judges
  the residual after craft fixes, and so Part B precedes Part A in one pass.
- **Part A (surface, structural):** a new `wrap_up` ACT verb is the brain's only
  write path to `summary`/`open_questions`/`notes`; the gate BLOCKS finishing
  when a known compromise is in play and the surface is empty (bounded by
  `_SURFACE_MAX_BLOCKS`); the loop reply now prefers the deliberate `summary`
  over the last internal prose line, so the user reads a real recap.
- **Part B (accountability, LLM):** a once-fired checkpoint presenting required
  beats + `carries`/`watch` + the advisory hints, demanding a per-item account
  (delivered / fixed / compromised-and-surfaced) with the FIT-TO-CRAFT two-reasons
  discipline; the gate never parses the account.
- **Deterministic hints:** exactly two, each fail-open — beat-sync declared but
  no op lands on the real grid (measured, since ops carry no snap flag), and
  punchy declared but a majority of a ≥3-cut main line runs long at natural pace.
- **The id-less-beats reality:** conformance is an LLM self-accountability stage,
  NOT a beat→cut gate; deterministic code only supplies advisory hints (§2, §7).
- **Back-compat / fail-open:** no migration, old plans coerced via `_beat_view`,
  every new path try/excepted to None/[], the loop still terminates.
- **Out of scope:** beat→cut mapping, the altitude/observability changes, and the
  budget-misallocation (thrash) issue — noted as a watch item, not built.
