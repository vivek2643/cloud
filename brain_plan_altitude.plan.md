# Brain plan ALTITUDE — pitch the plan at strategy, not tactics (Phase 2.5 of the planning hardening)

**Scope.** The durable plan artifact + plan mirror exist and are adhered to
(drift is fixed — `brain_plan_mechanism.plan.md`, Phase 2, shipped). This plan
fixes the *altitude* of that artifact: the plan currently commits to TACTICS
(specific clip ids) as if they were STRATEGY. We reshape `document["plan"]`'s
`structure` from a flat list of strings into an ordered list of **BEATS at intent
level**, each marked **required / optional**, and enforce the altitude
structurally-but-robustly (a plan-mirror NUDGE when a clip id leaks into a beat,
plus prompt guidance). READ-ONLY handoff — implement from another chat. Do NOT
implement from this doc; do NOT commit.

**This is NOT a rebuild.** Everything in `brain_plan_mechanism.plan.md` stays:
the `set_plan` verb, the `plan_mirror_text` renderer + its two injection points,
the doc-jsonb storage (no migration), the `_seed_document` behavior, the Phase-3
conformance seam. We touch exactly four production symbols (`act.set_plan`,
`tools._specs()`, `observe.plan_mirror_text`, and two prompt strings in
`converse._LOOP_SYSTEM`) plus the three existing test suites.

---

## 0. Motivation — the m14 goal-drop failure

The brain wrote a good plan and stuck to it — but the plan had named a *specific
take* (`m14`) for the "differentiator" beat. When `m14` wouldn't cut clean, the
brain deleted the whole BEAT along with the take: it abandoned the GOAL ("show
the differentiator") because the TACTIC ("use m14") failed. A plan pitched at the
right altitude would have held "differentiator — shown through the demo" as a
*requirement* that survives the death of any one clip, and forced a different
route (another take, slide + VO) or a surfaced ceiling.

**The design principle: two altitudes.**
- **Plan = strategy.** The commitments that survive contact with the material:
  purpose, the ordered BEATS the piece must hit, what carries each section,
  hazards to watch. True regardless of which specific clip you end up using.
  *The test for a plan line:* "would this still be true if a specific clip
  turned out unusable?" If no, it's a tactic, not a plan.
- **Execution = tactics.** Which exact clip, order among interchangeable pieces,
  whether a seam cuts clean — DISCOVERED by placing + sensing (predict/review) +
  adjusting. Being wrong here is normal and self-corrects in the loop.

**Concrete behavior we want:**
1. `structure` = ORDERED BEATS at intent level ("hook — founder's strongest
   intro", "what it is — one-line product definition", "differentiator — shown
   through the demo", "human close"), **NOT clip ids.** Clip selection moves
   entirely into execution.
2. Each beat carries **required vs. optional**. A required beat is non-negotiable:
   if its tactic fails, find ANOTHER ROUTE or **SURFACE the ceiling** — NEVER
   silently drop it.
3. The plan is **falsifiable only by purpose, never by material.** An unmet
   required beat is a surfaced problem, not a quiet deletion.
4. **Altitude enforced structurally but robustly:** clip ids should not appear
   in a beat. Primary mechanism = an active NUDGE in the PLAN mirror when an
   id-looking token leaks (robust, self-correcting), plus prompt guidance. A
   stricter alternative (hard schema rejection of id-looking tokens in
   `set_plan`) is noted as an option but NOT recommended — it's brittle (§9).

---

## 1. Current state to build on (verified against the code)

Read these; the change is surgical.

### 1.1 The artifact (today)
`document["plan"]` = `{purpose:str|None, carries:[str], structure:[str],
watch:[str], rev:int, updated_note?:str}`. `structure` is a flat ordered list of
free-form strings — **this is the field we reshape.** `purpose`/`carries`/
`watch`/`rev`/`updated_note` are already strategic and UNCHANGED.

### 1.2 Write verb — `act.set_plan` (`backend/app/services/l3/act.py:624-641`)
```624:641:backend/app/services/l3/act.py
def set_plan(document: dict, *, purpose=None, carries=None,
             structure=None, watch=None, note=None) -> dict:
    """Write/replace document['plan'] -- the brain's durable, ordered plan
    (brain_plan_mechanism.plan.md). Full replace of the plan block; bumps
    `rev` and records an optional `updated_note`. Pure: returns a new doc."""
    doc = _clone(document)
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
`_clone` (`act.py:44-53`) returns a fresh dict, so setting `doc["plan"]`
guarantees `new is not doc` (turn persists). Dispatched in `tools._dispatch`
(`tools.py:601-611`) with a direct return (echoes the rendered plan, not
`read_state`).

### 1.3 Verb schema — `tools._specs()` (`backend/app/services/l3/tools.py:336-355`)
```336:355:backend/app/services/l3/tools.py
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

### 1.4 Plan mirror renderer — `observe.plan_mirror_text` (`backend/app/services/l3/observe.py:1716-1744`)
```1716:1744:backend/app/services/l3/observe.py
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
Injected turn-start in `converse._context_block` (`converse.py:570-573`, right
after `CURRENT PROGRAM MAP`) and within-turn in `tools.run_edit_loop`
(`tools.py:833`, `building=changed`). Both call sites are UNCHANGED by this plan.
`re` is already imported at `observe.py:34`.

### 1.5 The clip-id / ref token format (for the nudge heuristic)
A moment id (= a `ref` the brain places by) is built in
`footage_map.py:294` as:
```294:294:backend/app/services/l3/footage_map.py
            "moment_id": f"{fid8}:m{idx:02d}",
```
where `fid8 = file_id[:8]` (`observe._fid8`, `observe.py:454-455`) — 8 lowercase
hex chars — and `idx` is zero-padded to ≥2 digits. So a full ref looks like
`93e94ec3:m14`. The BEAT INDEX also prints the SHORTHAND
`moment_id.split(':')[-1]` (`footage_map.py:1112,1188`), i.e. bare `m14`. The
brain places by either form (`tools._resolve_file`, `tools.py:359-370`). **Both
forms are what a leaked-tactic beat would contain**, so the nudge regex must
catch `<fid8>:m<NN>` AND bare `m<NN>` (§5).

### 1.6 The prompt language (two spots in `converse._LOOP_SYSTEM`)
- The **PLAN BEFORE YOU BUILD** block's "WORK OUT THE STRUCTURE" step + its
  "Then execute" tail (`converse.py:96-108`).
- The **THE PLAN** mechanism paragraph (`converse.py:196-201`).
`guidance_doc.md` has no plan-altitude line to change — leave it (§4.3).

---

## 2. The new artifact shape — `structure` becomes ordered BEAT objects

**Chosen shape (minimal, clean):**
```jsonc
"plan": {
  "purpose":   "teach a first-time viewer why the migration was worth it",  // str|null (UNCHANGED)
  "carries":   ["founder VO leads throughout", "screen-capture demo carries the middle"],  // [str] (UNCHANGED)
  "structure": [                                                            // NEW: [ {beat, need} ]
    {"beat": "hook -- the founder's strongest intro",            "need": "required"},
    {"beat": "what it is -- one-line product definition",        "need": "required"},
    {"beat": "the differentiator, shown through the demo",       "need": "required"},
    {"beat": "a customer quote if one lands",                    "need": "optional"},
    {"beat": "human close",                                      "need": "required"}
  ],
  "watch": ["the differentiator take may not cut clean -- have a slide+VO fallback"],  // [str] (UNCHANGED)
  "rev": 3,                                                                 // int (UNCHANGED)
  "updated_note": "raised the demo to required after re-reading the ask"    // str? (UNCHANGED)
}
```

### 2.1 Why this shape (and not the alternatives)
| Decision | Choice | Rationale (tied to the two-altitude principle) |
|---|---|---|
| Beat container | a `dict` `{beat, need}` per entry | The minimum needed to carry the required/optional distinction ALONGSIDE the beat text. A parallel-array shape (`structure:[str]` + `needs:[str]`) desyncs on reorder/edit; a rich object (id, status, chosen_clip…) would re-invite tactics into the plan. One object, two keys — the beat's JOB and whether it's negotiable. |
| `beat` (str) | the section's JOB at intent level | This is the strategy: "differentiator — shown through the demo", not "use m14". Passes the plan test — true no matter which clip lands it. |
| `need` (enum `"required"\|"optional"`) | the negotiability of the beat | Encodes principle #2/#3: a required beat is non-negotiable (find another route or surface the ceiling); an optional beat earns its place. Two values, not a free string, so the mirror + a future conformance gate can branch on it. |
| `need` default | **`"required"`** when omitted/invalid | A beat is a commitment by default. Making required the default means (a) an unmarked beat is treated as non-negotiable (fail-safe: the brain must *explicitly* downgrade to optional, never accidentally), and (b) the brain's old habit of passing bare strings still yields required beats — no silent optionality. |
| `purpose`/`carries`/`watch`/`rev`/`updated_note` | UNCHANGED | Already strategic and already survive material. No reshape. |

### 2.2 The normalized beat (what `set_plan` stores)
Every stored `structure` entry is `{"beat": <non-empty str>, "need":
"required"|"optional"}`. Empty-beat entries are dropped. Bare-string inputs
(the LLM passing a string) are coerced to `{beat: <str>, need: "required"}`.

---

## 3. Change A — `act.set_plan` normalization (`backend/app/services/l3/act.py`)

**Insertion point:** add a small module-private helper `_norm_beat` just ABOVE
`set_plan` (`act.py:624`), then swap the one `structure` line inside `set_plan`.
Keep the rest of `set_plan` byte-for-byte identical.

### 3.1 New helper — `_norm_beat` (above `set_plan`, `act.py:~623`)
```python
_BEAT_NEEDS = ("required", "optional")


def _norm_beat(entry) -> dict | None:
    """Normalize one `structure` entry to a beat {beat, need}. Accepts a
    {beat, need} dict OR a bare string (the LLM's old habit); an empty beat
    is dropped (returns None). `need` defaults to 'required' -- a beat is a
    commitment unless the brain explicitly marks it optional
    (brain_plan_altitude.plan.md SS2.1)."""
    if isinstance(entry, dict):
        beat = str(entry.get("beat") or "").strip()
        need = entry.get("need")
    else:
        beat, need = str(entry or "").strip(), None
    if not beat:
        return None
    return {"beat": beat, "need": need if need in _BEAT_NEEDS else "required"}
```

### 3.2 Edit inside `set_plan` — the `structure` line only
BEFORE (`act.py:634`):
```python
        "structure": [str(s).strip() for s in (structure or []) if str(s).strip()],
```
AFTER:
```python
        "structure": [b for b in (_norm_beat(s) for s in (structure or [])) if b],
```
Update the docstring's first line to note the reshape, e.g. append: `` structure
entries are normalized to {beat, need} beats (brain_plan_altitude.plan.md).`` No
other line of `set_plan` changes; `rev` bump, `updated_note`, purity, and the
`new is not doc` guarantee are preserved.

---

## 4. Change B — the verb schema + prompt

### 4.1 `tools._specs()` — the `set_plan` `structure` arg (`tools.py:346-349`)
BEFORE:
```python
               "structure": {"type": "array", "items": {"type": "string"},
                             "description": "the ORDERED intent: one entry per beat/"
                             "section in the order it should play. Free-form -- NOT a "
                             "template to fill."},
```
AFTER:
```python
               "structure": {"type": "array",
                             "description": "the ORDERED BEATS -- one per section, "
                             "at INTENT level (a JOB like 'hook -- founder's "
                             "strongest intro', 'differentiator -- shown through "
                             "the demo'), in play order. NEVER a clip id: which "
                             "clip carries a beat is a tactic you discover at the "
                             "timeline, not part of the plan.",
                             "items": {"type": "object",
                                       "properties": {
                                           "beat": {"type": "string",
                                                    "description": "the beat's job "
                                                    "-- what this section must "
                                                    "accomplish, at intent level"},
                                           "need": {"type": "string",
                                                    "enum": ["required", "optional"],
                                                    "description": "required = "
                                                    "non-negotiable (if its tactic "
                                                    "fails, find another route or "
                                                    "surface the ceiling -- never "
                                                    "silently drop it); optional = "
                                                    "include only if the material "
                                                    "supports it. Defaults to "
                                                    "required."}},
                                       "required": ["beat"]}},
```
The strict object schema advertises the intended shape; `act._norm_beat` is the
ROBUST layer that also swallows bare strings, so a model that ignores the schema
still produces valid beats (never an error). No other `set_plan` arg changes.
Optionally tighten the top-level `set_plan` description string (`tools.py:336`)
with one clause — "plan BEATS at intent level, never clip ids" — but this is
covered by the arg description; leave the verb blurb if you want a minimal diff.

### 4.2 Prompt — PLAN BEFORE YOU BUILD (`converse._LOOP_SYSTEM`, `converse.py:96-108`)
Two edits in this block, in the file's concatenation style (`--` dashes, CAPS
leads, string-per-line). Keep it non-directive and concise — NO narrative
template.

**Edit 1 — the "WORK OUT THE STRUCTURE" step.**
BEFORE (`converse.py:96-100`):
```python
    "-- WORK OUT THE STRUCTURE THIS PIECE WANTS. From the purpose and what carries "
    "it, reason about what the viewer should experience and in what order, and let "
    "THIS piece settle it -- never a formula. A build, a chronology, an "
    "association of images, a loop, a back-and-forth: the shape comes from the "
    "material in front of you. The ordered result of that reasoning is your plan.\n"
```
AFTER:
```python
    "-- WORK OUT THE STRUCTURE THIS PIECE WANTS as ordered BEATS at the level of "
    "INTENT. From the purpose and what carries it, reason about what the viewer "
    "should experience and in what order, and let THIS piece settle the shape -- "
    "a build, a chronology, an association of images, a loop, a back-and-forth -- "
    "never a formula. A beat names a JOB ('hook -- the founder's strongest intro', "
    "'differentiator -- shown through the demo', 'human close'), NOT a specific "
    "take or clip id -- which clip carries a beat is discovered later, at the "
    "timeline. Mark each beat REQUIRED or OPTIONAL: a required beat is the "
    "piece's job and non-negotiable; an optional beat earns its place only if the "
    "material supports it. The ordered, marked result is your plan.\n"
```

**Edit 2 — the "Then execute the plan directly" tail.**
BEFORE (`converse.py:104-108`):
```python
    "Then execute the plan directly: decide with predict/your senses before you "
    "place, place the selection you already settled on, and adjust from there -- "
    "don't discover the edit by placing, removing, and re-placing as you go; "
    "that's a sign you skipped the plan. When reading changes your mind, say so "
    "and update the plan rather than drifting.\n\n"
```
AFTER:
```python
    "Then execute the plan directly: decide with predict/your senses before you "
    "place, CHOOSE THE CLIP for each beat at the timeline, place the selection "
    "you settled on, and adjust from there -- don't discover the edit by placing, "
    "removing, and re-placing as you go; that's a sign you skipped the plan. A "
    "chosen take that won't cut clean falsifies the TACTIC, never the beat: for a "
    "REQUIRED beat, find another route (a different take, a slide + VO) or, if "
    "the material genuinely can't land it, SURFACE the ceiling in one line ('the "
    "material can't land the differentiator cleanly -- here are the options') -- "
    "NEVER silently drop a required beat. Only when reading changes your mind "
    "about the PLAN itself (its purpose, beats, or what carries it) do you say so "
    "and update the plan rather than drifting.\n\n"
```

### 4.3 Prompt — THE PLAN paragraph (`converse._LOOP_SYSTEM`, `converse.py:196-201`)
BEFORE:
```python
    "THE PLAN. Write the plan you reason to with set_plan -- purpose, what carries "
    "it, the ordered structure, and what to watch. It is DURABLE: echoed back to you "
    "every turn as the PLAN mirror (beside THE MIRROR) and carried across turns, so "
    "build against it and re-call set_plan to update it -- with one line on what "
    "changed your mind -- rather than drifting. If you start editing without one, the "
    "PLAN mirror will say so."
```
AFTER:
```python
    "THE PLAN. Write the plan you reason to with set_plan -- purpose, what carries "
    "it, the ordered BEATS (each marked required or optional), and what to watch. "
    "The plan is STRATEGY -- the commitments that survive contact with the "
    "material -- so it names beats at INTENT level, never specific clip ids (which "
    "clip carries a beat is a tactic you discover at the timeline). It is DURABLE: "
    "echoed back to you every turn as the PLAN mirror (beside THE MIRROR) and "
    "carried across turns, so build against it and re-call set_plan to update it "
    "-- with one line on what changed your mind -- rather than drifting. It is "
    "falsifiable only by PURPOSE, never by material: a required beat you cannot "
    "land cleanly is a problem to SURFACE, not a beat to delete. If you start "
    "editing without a plan, or a clip id leaks into a beat, the PLAN mirror will "
    "say so."
```

**`guidance_doc.md`:** no change. Its only plan-adjacent line
(`guidance_doc.md:14`, on binding planning) is altitude-neutral; adding altitude
prose there would duplicate the prompt. Leave it.

---

## 5. Change C — the plan mirror renders beats + NUDGES on a leaked clip id (`observe.py`)

**Insertion point:** `observe.plan_mirror_text` (`observe.py:1716-1744`). Add a
module-level regex + a tiny coercion helper above it, then swap the `structure`
rendering block. Everything else in the function (absent-plan nudges, purpose,
carries, watch, updated_note, purity/idempotence) is UNCHANGED.

### 5.1 The detection heuristic (robust, low false-positive)
Add near the other module regexes (or just above `plan_mirror_text`):
```python
# brain_plan_altitude.plan.md SS5: a clip-id-looking token leaking into a
# beat. A ref is `<fid8>:m<NN>` (footage_map.py:294) and the beat index also
# prints the bare `m<NN>` shorthand (footage_map.py:1112/1188). `m\d{2,}`
# (2+ digits, matching the `:02d` padding) keeps stray prose like "m1" from
# false-firing. Case-insensitive for the hex prefix.
_CLIP_ID_IN_BEAT_RE = re.compile(r"\b(?:[0-9a-f]{8}:)?m\d{2,}\b", re.IGNORECASE)


def _beat_view(entry):
    """Coerce a stored `structure` entry to (beat_text, need) for rendering.
    Accepts the new {beat, need} dict AND an OLD-shape bare string (a
    pre-altitude stored plan) -> (str, None). Fail-open, never raises
    (brain_plan_altitude.plan.md SS6)."""
    if isinstance(entry, dict):
        return str(entry.get("beat") or "").strip(), entry.get("need")
    return str(entry or "").strip(), None
```

### 5.2 The renderer swap — the `structure` block only
BEFORE (`observe.py:1736-1739`):
```python
    if plan.get("structure"):
        lines.append("  structure:")
        for i, s in enumerate(plan["structure"], 1):
            lines.append(f"    {i}. {s}")
```
AFTER:
```python
    if plan.get("structure"):
        lines.append("  structure (BEATS -- build against these, not clip ids):")
        leaked = False
        for i, entry in enumerate(plan["structure"], 1):
            beat, need = _beat_view(entry)
            mark = {"required": " [required]", "optional": " [optional]"}.get(need, "")
            lines.append(f"    {i}. {beat}{mark}")
            if _CLIP_ID_IN_BEAT_RE.search(beat):
                leaked = True
        if leaked:
            lines.append(
                "    ^^ a beat names a clip id -- the PLAN is STRATEGY (beats at "
                "intent level), not tactics. Which take carries a beat is chosen "
                "at the timeline; re-call set_plan with that beat rewritten as its "
                "JOB (what it must accomplish), not a specific take.")
```

**Notes.**
- Old-shape entries (bare strings) render with NO `[required]/[optional]` marker
  (`need is None` → empty `mark`) — faithful to a plan that never made the
  distinction, and never crashes (§6).
- The nudge is ADVISORY and self-correcting: it appears every turn the leak
  persists (idempotent — same plan → same text, preserving the function's
  push-every-step safety), and disappears the moment the brain re-writes the
  beat as a job. This is the PRIMARY altitude enforcement (robust), per
  requirement #4.

### 5.3 Optional secondary heuristic (note only, do NOT build unless wanted)
A beat could also embed the `CLIP <fid8>` form or a bare 8-hex file prefix.
Detecting a lone `\b[0-9a-f]{8}\b` risks false positives on hex-ish prose, so it
is deliberately EXCLUDED from the primary regex. If a future pass wants it, gate
it behind a word like `CLIP`/`clip` preceding the hex. Not needed for the m14
class of failure (that leaked `m14`, which `m\d{2,}` catches).

---

## 6. Back-compat — old-shape `structure:[str]` must render, never crash

Existing stored documents have `structure` as a flat list of strings. The
document is jsonb (`edit_documents.document`, migration `014_l3_edit_threads.sql`)
so there is **NO DB MIGRATION** — old plans sit unchanged until the brain next
calls `set_plan`, which upgrades them to beat objects on write.

Fail-open contract, enforced by the two coercion points:
- **Renderer:** `observe._beat_view` (§5.1) accepts BOTH a `{beat, need}` dict
  and a bare string; a string yields `(text, None)` → rendered with no marker.
  So `plan_mirror_text` on an old plan renders cleanly (numbered beats, no
  markers) and never raises.
- **Writer:** `act._norm_beat` (§3.1) accepts a bare string and upgrades it to
  `{beat, need: "required"}`, so re-emitting an old plan through `set_plan`
  migrates it in place.
- **Any other reader** (only the not-yet-built Phase-3 conformance, §8) MUST use
  `_beat_view`-style coercion, not `entry["beat"]`, so an old-shape plan is
  safe. State this in the Phase-3 handoff.

Nothing else reads `structure`. **No migration; no data backfill.**

---

## 7. Testing

Run with `backend/.venv/bin/python`, fully mocked (no network/DB), changed files
pyflakes-clean. Extend the EXISTING plan tests (they already cover the Phase-2
shape — update their assertions to the beat shape) and add the new cases.

### 7.1 `act.set_plan` / `_norm_beat` — extend `scripts/test_observe_act.py`
The existing tests at `test_observe_act.py:1245-1281` assert the OLD flat-string
shape and MUST be updated:
- `test_set_plan_on_a_seed_doc_returns_a_new_normalized_doc` (`:1245`): change
  the `structure` input to a mix — a `{beat, need}` dict, a bare string, an
  empty string — and assert the stored `structure` is
  `[{"beat": ..., "need": "optional"}, {"beat": ..., "need": "required"}]` with
  the empty dropped and the bare string defaulted to `"required"`.
- `test_set_plan_with_all_empty_inputs_is_still_a_legal_write` (`:1263`):
  unchanged expectation (`structure: []`), still passes.
- `test_set_plan_recall_bumps_rev_and_records_updated_note` (`:1271`): pass beat
  dicts; assert order + `need` survive the re-emit and `rev` bumps.
- NEW `test_set_plan_beat_need_defaults_to_required`: a bare-string beat and a
  dict beat with `need` omitted both store `need == "required"`; an explicit
  `"optional"` is preserved; a garbage `need` ("maybe") coerces to `"required"`.

### 7.2 `observe.plan_mirror_text` — extend `scripts/test_observe_act.py`
Update `test_plan_mirror_text_populated_plan_renders_all_fields` (`:1301`) to the
beat shape and add cases:
- Populated beat plan renders `structure (BEATS`, numbered beats, and the
  `[required]` / `[optional]` markers; idempotent (two calls equal).
- NEW `test_plan_mirror_text_nudges_on_a_leaked_clip_id`: a beat
  `{"beat": "differentiator -- use 93e94ec3:m14", "need": "required"}` renders
  the beat AND the `^^ a beat names a clip id` nudge; a beat with bare `m14`
  also fires; a clean beat ("show the differentiator through the demo") does NOT
  fire (no false positive); a beat mentioning "mp4" / "m1" does NOT fire.
- NEW `test_plan_mirror_text_renders_old_shape_structure_of_strings`: a plan
  with `structure: ["cold-open", "the demo"]` (old shape) renders numbered
  beats with NO markers and does not raise (back-compat, §6).
- Absent-plan quiet/loud nudges (`:1288`, `:1294`) — UNCHANGED, still pass.

### 7.3 dispatch + verb wiring — extend `scripts/test_tools_loop.py`
Update the two existing plan tests to the beat shape:
- `test_dispatch_set_plan_applies_and_returns_the_rendered_plan` (`:301`): pass
  `structure` as beat dicts; assert `new["plan"]["structure"]` is the normalized
  beat list in order (each `{beat, need}`), `rev == 1`, and the echo is the tight
  plan JSON (not `read_state`).
- `test_loop_pushes_the_plan_mirror_after_a_set_plan_call` (`:326`): pass beats;
  assert the pushed text_block still contains `"PLAN (rev"` and now
  `"structure (BEATS"`.
- `test_loop_pushed_plan_mirror_is_loud_when_the_edit_moved_without_a_plan`
  (`:349`) — UNCHANGED (absence path), still passes.
- Confirm `"set_plan" in observe.affordances(...)["verbs"]` still holds
  (`test_observe_act.py:358`) — no change, but re-run.

### 7.4 turn-start injection — `scripts/test_converse_context.py`
`test_context_block_renders_the_plan_of_record_beside_the_program_map` (`:143`)
and `test_context_block_shows_the_absence_nudge_when_no_plan` (`:158`): update
the seeded plan to the beat shape; assert `PLAN (rev N` still renders after
`CURRENT PROGRAM MAP` and now shows the beat markers. Absence line UNCHANGED.

### 7.5 Module imports / pyflakes
`import act, tools, observe, converse` cleanly; no unused names from the new
helpers/regex. Add the three new test fns to each file's `__main__` runner
block (mirror the existing `if __name__ == "__main__":` lists at
`test_observe_act.py:1391-1396`, `test_tools_loop.py:684-686`).

### 7.6 Manual e2e — eyeball altitude
1. Open a thread on real footage; ask for a piece with a clear differentiator
   ("cut a 60s product explainer that lands what makes us different").
2. Confirm the brain's `set_plan` `structure` is BEATS at intent level with
   required/optional markers, and contains NO clip ids (inspect the persisted
   `document.plan` via `GET /api/edit/threads/{id}`, and the PLAN mirror in the
   trace).
3. Force the m14 situation: pick footage where the obvious differentiator take
   won't cut clean. Confirm the brain does NOT delete the "differentiator" beat
   — it either routes to another take / slide+VO, OR surfaces the ceiling in its
   reply — and the PLAN mirror still lists the differentiator beat as
   `[required]`.
4. (Nudge check) If the brain ever writes a clip id into a beat, confirm the
   `^^ a beat names a clip id` nudge appears in the pushed PLAN mirror and that
   the brain rewrites the beat as a job on the next turn.

---

## 8. Phase-3 handoff — automated conformance is OUT OF SCOPE

This plan reshapes the ARTIFACT + PROMPT + MIRROR + WRITE VERB and adds the
surfacing DISCIPLINE via prompt (not automated enforcement). It does NOT check
whether each required beat actually got BUILT in the timeline.

That check — mapping `plan["structure"]` beats → placed cuts and blocking FINISH
when a `need=="required"` beat is unbuilt/uncovered — is Phase 3, and its seam
already exists, inert, in `tools._verify_before_finish`
(`tools.py:700-707`, the `_plan_conformance` comment hook, right after the
structural check). Phase 3 will:
- iterate `working.get("plan", {}).get("structure", [])` using a `_beat_view`-
  style coercion (§6) so old-shape plans are safe;
- for each `need=="required"` beat, decide "did the built timeline cover this?"
  (the hard, un-rubber-stampable part — beat→cut mapping);
- return once-surfaced feedback ("required beat 'differentiator' isn't in the
  cut — build it, route around the bad take, or surface the ceiling") exactly
  like the other gate stages (state-tracked, terminates).

That mapping is the genuinely hard, separate piece; keep it out of this plan.

---

## 9. Alternative considered — hard schema rejection of clip ids (NOT recommended)

Requirement #4 asks to note the stricter option: reject an id-looking token in
`set_plan` (raise / no-op the write) so a leaked clip id can never persist.
**Rejected as PRIMARY** because it is brittle:
- A false positive (a legitimate beat that happens to contain `m\d{2}`) would
  BLOCK a valid plan write mid-turn, a worse failure than an advisory nudge.
- Hard rejection gives the brain a wall, not a correction — the NUDGE both flags
  the leak AND tells the brain what to do instead (rewrite as a job), which is
  self-correcting and matches how every other signal in this loop works
  (advisory, surfaced, the brain acts).
- The nudge is idempotent and cheap; it degrades gracefully on old-shape plans.

If a future pass wants belt-and-suspenders, the least-brittle form is a SOFT
reject: `act.set_plan` strips a trailing "(m14)"-style parenthetical from a beat
and records it under `watch` instead — but that is extra surface for no proven
need. Ship the nudge; revisit only if leaks persist in practice.

---

## 10. Exact change list per file

| File | Change |
|---|---|
| `backend/app/services/l3/act.py` | Add `_norm_beat` + `_BEAT_NEEDS` above `set_plan` (`:624`); swap the one `structure` line inside `set_plan` (`:634`) to normalize entries to `{beat, need}` beats (§3). |
| `backend/app/services/l3/tools.py` | Rewrite the `set_plan` `structure` arg schema in `_specs()` (`:346-349`) to object items `{beat, need(enum), required:[beat]}` (§4.1). No dispatch change. |
| `backend/app/services/l3/observe.py` | Add `_CLIP_ID_IN_BEAT_RE` + `_beat_view` above `plan_mirror_text`; swap the `structure` render block (`:1736-1739`) to render beats with need markers + fire the leaked-clip-id nudge (§5). Back-compat via `_beat_view` (§6). |
| `backend/app/services/l3/converse.py` | Rewrite the "WORK OUT THE STRUCTURE" step + "Then execute" tail in PLAN BEFORE YOU BUILD (`:96-108`) and the THE PLAN paragraph (`:196-201`) for beats/required-optional/find-another-route-or-surface (§4.2, §4.3). |
| `backend/app/services/l3/guidance_doc.md` | **None** (§4.3). |
| tests | Update the existing plan tests + add beat/need + nudge + old-shape cases in `scripts/test_observe_act.py`, `scripts/test_tools_loop.py`, `scripts/test_converse_context.py` (§7). |
| migrations | **None** (jsonb, fail-open on old shape) (§6). |

---

## 11. Summary of decisions
- **New `structure`:** ordered `[{beat:str, need:"required"|"optional"}]`; `beat`
  is the section's JOB at intent level, `need` its negotiability, defaulting to
  **required**. `purpose`/`carries`/`watch`/`rev`/`updated_note` UNCHANGED.
- **Altitude enforced robustly:** clip ids barred from beats by (1) prompt
  guidance, (2) the verb arg schema, and — PRIMARY — (3) an advisory,
  self-correcting NUDGE in the PLAN mirror when `\b(?:[0-9a-f]{8}:)?m\d{2,}\b`
  matches a beat. Hard schema rejection considered and rejected as brittle (§9).
- **Required beats are non-negotiable:** the plan is falsifiable only by
  PURPOSE, never by material — a required beat that can't be landed is SURFACED
  (find another route / show the ceiling), never silently dropped. Enforced as
  prompt DISCIPLINE here; automated conformance is Phase 3 (§8).
- **Back-compat:** old-shape `structure:[str]` renders + upgrades fail-open via
  `_beat_view` / `_norm_beat`. **No migration** (jsonb).
- **Phase-3 seam untouched:** the inert `_plan_conformance` hook in
  `tools._verify_before_finish` is where beat→cut conformance lands later.
