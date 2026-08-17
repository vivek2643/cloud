# Brain ACCOUNTABILITY ARCHITECTURE — one finalization choke point, commitments/reversals, signal severity, real progress, complete trace (Phase 5 of the loop hardening)

---

# AMENDMENT (supersedes anything below it conflicts with)

**Status at this amendment: PART 1 is SHIPPED. PARTS 2, 3, 4 and 5 are
OUTSTANDING.** PART 1 landed as `_Stage` / `_GateCtx` / `_GATE_STAGES` /
`_run_gate_stages` / `_verify_before_finish` / `_record_gate` /
`_finalize_turn` / `_finalize_after_loop` in `backend/app/services/l3/tools.py`
(`tools.py:995-1337`), with the seven-stage ladder at `tools.py:1150-1158`.
Nothing in PARTS 2–5 has been implemented; the code sections quoted in those
parts still describe HEAD except where this amendment says otherwise.

Read this amendment before reading anything below it. It records decisions taken
after the plan was written, absorbs two separately-proposed fixes so they are
NOT built standalone, flags a known gap and two stale in-code promises, and
changes the recommended ordering.

## A.0 Sequencing of record (identical in all three plans)

1. `junk_span_suppression.plan.md` — independent, does not touch L3.
2. **`brain_gate_integrity.plan.md` §1** — split stage EVALUATION from stage
   SATISFACTION. **This must precede any new stage being registered in
   `_GATE_STAGES`, including PART 2's `reversals` stage and PART 3's
   `blocking_signals` stage.** See A.1.
3. `brain_perception_integrity.plan.md` — independent of the accountability
   parts, and it UNBLOCKS them: PARTS 2, 3 and 4 all consume perception, and
   two of them consume values this plan currently assumes are trustworthy.
4. `brain_gate_integrity.plan.md` §2 — the coverage gate.
5. **PART 2**, then **PART 3**, then **PARTS 4/5** — but see A.5: PART 5 should
   move EARLIER than last.
6. Beat binding + the adversarial quote check, last. See A.6.

## A.1 PART 1 shipped with a defect that must be fixed before PART 2 or PART 3 adds a stage

PART 1's stages each write their own `state` key *inside the evaluation
function* (`tools.py:1055`, `:1066`, `:1079`, `:1089`, `:1113`, `:957`,
`:1138`). On the last-turn finish path the evaluated feedback is DISCARDED
(`tools.py:1423`, `if feedback is not None and turn < max_turns - 1`) while the
marks survive, so `_finalize_turn`'s enforcement pass (`tools.py:1256`) finds
four MANDATORY stages already satisfied and collects nothing, and `_record_gate`
writes `fired: false` for stages that DID fire. **The comment at
`tools.py:1414-1419` asserting "`_finalize_turn` enforces whatever is left
outstanding" is false.**

The class is *stage satisfaction is coupled to stage evaluation*, and the fix is
specified in full in **`brain_gate_integrity.plan.md` §1** (stages become pure;
`_run_gate_stages` returns `(feedback, stages_to_commit)`; a single
`_commit_stages` writer runs only after the message is appended to `convo`).
Precise per-stage outcomes, which differ from a first reading: four stages
(`length`, `intent`, `craft`, `conformance`) lose the message permanently; two
(`structural`, `surface`) re-fire but burn one of their budgeted attempts on an
undelivered message; one (`flags`) is advisory and out of scope for the
enforcement pass.

**Consequence for this document:** PART 2 §2.3 registers `_stage_reversals` and
PART 3 §3.4 registers `_stage_blocking_signals`. Both are GATING stages with
their own bounds (`_REVERSAL_MAX_BLOCKS`, `_SIGNAL_MAX_BLOCKS`). Registered
against HEAD's ladder, both inherit the defect on the last-turn path, and both
would need their own compensating logic. Land the split first and they inherit
correctness instead. When writing them, follow
`brain_gate_integrity.plan.md` §1.3: declare `commit=COMMIT_COUNT` and write no
state inside the stage function.

Also amended: PART 5 §5.1's `_record_gate` docstring and this plan's several
"the mandatory invariant ACTUALLY fired is assertable" claims are only true once
`fired` is joined by `delivered` (`brain_gate_integrity.plan.md` §1.6). Until
then, a `kind:"gate"` entry cannot distinguish "did not fire" from "fired and
the message was thrown away", which is the exact ambiguity that let the defect
above ship inside a part written to end dead gates.

## A.2 ABSORBED: "force plan revision when a commitment changes" is PART 2 §2.1–2.3

A separately-proposed fix — *watch for the verb that changes a commitment and
force a `set_plan`* — is **absorbed into PART 2 §2.1–2.3 and must NOT be built
standalone.**

PART 2 is strictly better, and the reason is mechanical, not stylistic. PART 2
detects by **document diff on declared axes** (`commit._AXES`, §2.2), not by
watching a verb. A commitment on the `mute` axis can flip via `set_audio`
(`act.py:662-679`), via `place(audio="keep")` → `arrange._resolve_mute`
(`arrange.py:127-135`), via `tighten`/`retime` re-slicing, and via
`arrange.heal_adjacent_cuts`' unmute-on-merge. A commitment on the `presence`,
`take` or `order` axes flips via `place`, `remove` and `move`. **A verb watcher
needs one branch per verb and is wrong the moment a new verb lands or an
existing verb grows a side effect; the diff catches all of them with no
branches.** That is the difference between dissolving the class and patching the
instance.

Do not implement a `set_plan`-forcing check anywhere else. If one appears in a
diff, it is the band-aid this note exists to prevent.

## A.3 ABSORBED: "length undershoot + severity" is PART 3 §3.2–3.3

A separately-proposed fix — *notice when the edit lands far under the target and
raise its severity* — is **absorbed into PART 3 §3.2–3.3 and must NOT be built
standalone.**

The live defect, verified at HEAD: `observe.diagnose` computes both halves of
the length contract, and emits under-target at `severity: "info"`:

```1233:1235:backend/app/services/l3/observe.py
        elif cur < target_s * 0.7:
            findings.append({"severity": "info", "anchor": "whole",
                             "message": f"under target: {cur:.1f}s vs {target_s:.1f}s"})
```

and `_stage_length` filters it out by SUBSTRING:

```1062:1063:backend/app/services/l3/tools.py
def _stage_length(gc: _GateCtx) -> str | None:
    over = [f for f in gc.findings if "over target" in (f.get("message") or "")]
```

`"over target" in "under target: 34.0s vs 60.0s"` is False, so the finding is
dropped. **The brain has shipped far under target four consecutive runs with no
signal whatsoever.** PART 3 replaces the substring match with a machine `kind`
(`KIND_LENGTH_OVER` / `KIND_LENGTH_UNDER`), escalates both via `_ESCALATE`
(§3.2), and attaches the headroom fact — untouched files, unplaced moments —
to the under-target finding (§3.3).

The standalone version of this fix would be `over = [... if "target" in ...]`,
which is one character of change, would work, and is exactly the trap: it keeps
policy expressed as a substring match on prose meant for an LLM, which is how
the bug happened. **Any consumer that string-matches a `message` is the defect,
not the direction of the comparison.**

## A.4 KNOWN GAP: `act.set_plan` full-replaces the plan dict

PART 2 §2.3 (plan line ~897) requires `plan["reversals"]` to persist across
turns. `act.set_plan` builds a **fresh** plan dict and drops every key it does
not name:

```732:744:backend/app/services/l3/act.py
    doc = _clone(document)
    prev_rev = int((document.get("plan") or {}).get("rev") or 0)
    plan = {
        "purpose": (purpose or "").strip() or None,
        "carries": [str(c).strip() for c in (carries or []) if str(c).strip()],
        "structure": [b for b in (_norm_beat(s) for s in (structure or [])) if b],
        "watch": [str(w).strip() for w in (watch or []) if str(w).strip()],
        "rev": prev_rev + 1,
    }
    if note and note.strip():
        plan["updated_note"] = note.strip()
    doc["plan"] = plan
    return doc
```

So `plan["reversals"]` must be **explicitly carried over** — and note the trap
in PART 2's own acknowledgement rule: it requires `plan["rev"]` to have been
bumped after the reversal was detected (§2.3 (a)), and `set_plan` is the only
thing that bumps `rev`. **If `set_plan` drops `reversals`, then acknowledging a
reversal deletes the record of it, and the acknowledgement check reads an empty
ledger and passes vacuously.** That is a self-erasing gate, not a missing field.

The same applies to any future plan key. Preferred fix, stated once: `set_plan`
should carry forward every key it does not explicitly own, i.e. start from
`dict(document.get("plan") or {})` and overwrite only the named fields — so the
next plan key added does not have to remember. `test_set_plan_preserves_recorded_reversals`
(§2.3 testing) must be joined by a test that a plan key `set_plan` has never
heard of also survives, or the fix is one field, not the class.

## A.5 RECOMMENDATION: move PART 5 (the trace funnel) EARLIER

PART 5 is listed last. It should land immediately after PART 1's split
(A.0 step 2), before PART 2.

The reason is the defect in A.1. That bug was invisible for one specific
structural reason: the four per-turn injections at `tools.py:1480-1496` — the
edit mirror, the plan mirror, the progress note and the churn hint — plus the
gate feedback at `tools.py:1424` and the finalize instruction at
`tools.py:1321`, are **not recorded anywhere.** So "the brain saw this" was
never assertable from a stored thread, and `_record_gate`'s `stages` list was
the only available evidence about the gate — and `_record_gate` was reporting
`fired: false` for stages that had fired. **The single piece of evidence was
the wrong one, and there was no second source to contradict it.**

Every part after PART 1 adds another injection (PART 2's reversal notices,
PART 3's blocking-signal feedback, the coverage gate's message) and each one
lands unrecorded if PART 5 has not shipped. Landing PART 5 second means every
subsequent part's session test can assert on `kind:"injected"` entries instead
of reconstructing the state machine from source, and PART 5's own invariant test
(`test_no_raw_text_block_injection_in_the_loop`) is enforced from the moment the
next injection is written rather than retrofitted over five parts of them.

Cost of moving it: PART 5 §5.2 rewires call sites that PARTS 2/3/4 also touch,
so a later part must add its injection through `_inject` rather than editing a
raw `results.append`. That is a benefit, not a cost.

`brain_gate_integrity.plan.md` §3 (wrapping the unguarded
`observe.mirror_text` injection at `tools.py:1480`) is written so that its
`for _src, _fn in (...)` loop becomes the natural home for `_inject`, so the
two changes compose.

## A.6 DECISION RECORDED: beat DELIVERY is binding first, then an adversarial quote check — in that order

This supersedes the open question in `brain_plan_conformance.plan.md` (its Part B
presents required beats and accepts the brain's prose account) and closes the 3b
gap. **The decision: a required beat gains a separate, machine-readable BINDING
field naming the seg_ids currently carrying it; then, at finish, an adversarial
QUOTE check presents the beat text beside those cuts' actual played words and
requires a quote.** Binding first. Quote check second. Both, in that order.

### Why a binding field does not recreate the problem clip ids were banned for

Clip ids were banned from beats. The ban is real and enforced:

```209:211:backend/app/services/l3/converse.py
    "it, the ordered BEATS (each marked required or optional), and what to watch. "
    "The plan is STRATEGY -- the commitments that survive contact with the "
    "material -- so it names beats at INTENT level, never specific clip ids (which "
```

with a detector at `observe.py:1721` (`_CLIP_ID_IN_BEAT_RE`) surfaced in the
plan mirror at `observe.py:1761-1768`.

The ban was correct because **intent and tactic were FUSED INTO ONE STRING.**
A beat like `"open on 7144c9fc:m03"` has no separable intent; killing the tactic
killed the intent along with it, and the brain lost the ability to say what the
beat was FOR.

A separate field does not recreate that. The beat text stays pure strategic
intent — the `_CLIP_ID_IN_BEAT_RE` check on the text stays exactly as it is —
and gains a sibling field. The two have different lifetimes, and that difference
is the point:

- **Intent is stable.** A required beat is falsifiable only by PURPOSE, never by
  material (`converse.py:215-217`). It should not change often.
- **Binding is expected to change.** Which cut carries a beat is discovered at
  the timeline. A swap is normal.
- **And when the binding changes, that is exactly the signal PART 2 needs.** A
  bound beat gives `commit._AXES` (§2.2) a subject with a stable identity on a
  `beat` axis, so "the cut carrying required beat 3 was swapped" becomes a
  document-diff reversal like any other, detected without a verb watcher.

Today intent and binding are not separated — **they are DISCONNECTED.** There is
no link at all between a beat and the timeline, which is why

```948:950:backend/app/services/l3/tools.py
        required = [b for b in (observe._beat_view(e) for e in plan.get("structure") or [])
                    if b[0] and b[1] != "optional"]           # (beat_text, need)
        declared = list(plan.get("carries") or []) + list(plan.get("watch") or [])
```

can only PRESENT the required beats, and *"All required beats delivered"* is
assertable by the brain with the load-bearing sentence absent from the program.
Nothing can check it, because nothing knows where a beat is supposed to be.

### CONSTRAINT on the binding's identity — do NOT bind by `seg_id` alone

Verified at HEAD, and it invalidates the obvious implementation.
`observe.resolve_doc` runs once per user turn, after the loop
(`converse.py:700`), and it heals the main line with `reindex=True`:

```315:319:backend/app/services/l3/observe.py
    old_ids = [s.get("seg_id") for s in (document.get("timeline") or [])]
    old_segs = list(document.get("timeline") or [])
    document["timeline"], _merged = heal_adjacent_cuts(
        document.get("timeline") or [], gap_ms=get_settings().heal_gap_ms, reindex=True)
    _remap_seam_ops(document, old_ids, old_segs)
```

and `heal_adjacent_cuts` re-issues every surviving id:

```297:299:backend/app/services/l3/arrange.py
    if reindex:
        for i, s in enumerate(healed):
            s["seg_id"] = f"a{i:03d}"
```

`_remap_seam_ops` (`observe.py:341`) remaps `split_edit` / `crossfade` ops by
object identity. **Nothing remaps a plan binding.** So a beat bound to
`seg_id: "u3f9a1c22"` on turn 1 points at nothing on turn 2, and a beat bound to
`"a003"` may point at a DIFFERENT cut after the next heal renumbers. A binding
that silently evaporates between turns is worse than no binding: the "required
beat with no binding at finish is blockable" check (item 1 below) would fire on
every beat every turn after the first, and would be disabled within a week.

Therefore bind by the **map `ref`** (`seg["ref"]`, written at `act.py:105`,
carried through heal because heal keeps the predecessor dict), which is stable
across turns and across reindexing, and treat `seg_id` as a within-turn
convenience only. Two cases to spec explicitly:

- a `place_span` cut has `ref: None` (`act.py:179`), so a beat carried by a raw
  span cannot be bound by ref — bind by `(file_id, in_ms, out_ms)` for that case,
  or require the binding to name the seg and refresh it at `resolve_doc` time
  alongside `_remap_seam_ops`;
- heal MERGES segments, so two refs can collapse into one surviving segment
  (`arrange.py:283-294`). A binding must tolerate its ref having been absorbed —
  which is itself a legitimate reversal signal on the `presence` axis (PART 2
  §2.2), so route it there rather than inventing a second rule.

Whichever is chosen, the acceptance test is: **place a cut, bind a required beat
to it, complete a turn (so `resolve_doc` reindexes), and assert the binding still
resolves to the same cut.** That test is the one that separates a working binding
from one that looks right in a single-turn unit test.

### What binding buys — three checks, each structural

1. **A required beat with no binding at finish is BLOCKABLE.** "You committed to
   this beat and nothing in the timeline carries it" is a fact about the plan
   and the document, checkable with no prose parsing and no LLM.
2. **A swap of a bound cut is a detectable reversal**, which forces a stated
   plan revision through PART 2's existing acknowledgement rule (recorded via
   `set_plan`, surfaced via `wrap_up`) — see A.4 for the `reversals`
   preservation prerequisite.
3. **A required beat bound to a cut graded dead or junk is catchable
   structurally.** This connects directly to `junk_span_suppression.plan.md`:
   that plan makes `usability.grade` and the `junk` flag real producers on the
   video channel (§3.2–3.3), so "required beat → bound seg → its ref's moment
   grades `dead`" is a joinable chain. **This is the observed failure:** the
   *"clean human close"* beat in edit thread
   `34bb9967-97aa-4818-9f21-01145156a9bb` was delivered by `aad020e3:m06` —
   4.6 s of digital silence at −88.73 dBFS, the dead tail of a recording, the
   final 19% of the program (`junk_span_suppression.plan.md` §0, §1 Class 1).
   With binding plus a usability grade, the sentence "your required closing beat
   is carried by a span measured at its own file's noise floor" is emittable by
   code. Without binding, it is not, no matter how good the grade is.

### Then the adversarial quote check — the part that cannot be automated

Binding proves a beat is bound to SOMETHING. It cannot prove the bound cut
actually accomplishes the beat's intent. Nothing deterministic can: whether
"establish why this matters to a founder" is accomplished by a particular 4
seconds of speech is a judgment.

So the second half is an LLM checkpoint made adversarial by CONSTRUCTION: at
finish, present each required beat's text **beside the verbatim words the bound
cuts actually play** (`observe.review`'s `played_text` / `read_transcript` — see
`brain_perception_integrity.plan.md` §1, which is what makes those words
trustworthy in the first place) and require the brain to **quote the specific
words that deliver the beat, or declare it a compromise.** A quote is
falsifiable against the text it was quoted from; a claim ("delivered") is not.
This is the same fix-or-surface discipline `_plan_conformance` already uses
(`tools.py:958-975`), with the one addition that closes the hole: the brain must
point at the evidence, and the evidence is on the screen next to the question.

The dependency runs the other way from what it looks like: **the quote check is
worthless if `played_text` is wrong.** At HEAD `review.played_text` reports whole
sentences that merely OVERLAP the cut (`footage_map._said_text_for_span`'s
`_overlap_ms(...) > 0` filter, `footage_map.py:1517`), so a quote could be
"verified" against words the program never plays. That is
`brain_perception_integrity.plan.md` §1, and it is why this item is sequenced
last.

### EXPLICITLY RULED OUT: keyword-matching beat text against the transcript

Do not implement a deterministic check that tokenizes a beat's text and looks
for those words in the program's transcript.

Record the reason, because this idea is **dangerously tempting**: it would have
caught the one observed failure. A silent 4.6-second dead-tail cut contains zero
words, so any keyword overlap against any beat text is zero, and a naive
`overlap == 0 → block` rule fires perfectly on the example in hand.

It is still wrong, in both directions:

- **It fires on paraphrase.** A beat *"land the emotional close"* delivered by
  *"…and that's really what keeps me doing this"* has near-zero token overlap.
  A well-written beat names a JOB, not a phrase, so low overlap is the NORMAL
  case for a good beat, and the check would block correct edits constantly —
  after which it gets disabled, which is the worst outcome.
- **It misses anything worded differently.** A beat *"open with the product
  demo"* delivered by a cut that says the words "product demo" while showing the
  wrong thing passes cleanly.

It is a coincidence detector tuned to one example: high precision on the
observed instance, no relationship to the property being checked. The two
things that DO generalize are the two above — a structural fact (is it bound;
is what it is bound to usable) and a judgment made falsifiable (quote it) —
and they are separated on purpose, because they fail differently and are fixed
differently.

---

**Scope.** Five architecture-level changes to the edit tool-loop
(`tools.run_edit_loop` / `tools._verify_before_finish`,
`backend/app/services/l3/tools.py`) so that the accountability machinery we
already built **cannot be structurally bypassed**, so that a standing
commitment cannot be silently contradicted, so that a signal can actually
BLOCK, so that "progress" means the edit moved, and so that everything pushed
into the brain's context is auditable afterwards.

**READ-ONLY handoff — implement from another chat. Do NOT implement from this
doc; do NOT commit.**

**PART 1 lands FIRST, ALONE, and is independently landable/testable** (the user
must be able to re-run immediately after it). Parts 2–5 build on PART 1's
declared stage list and its injection funnel.

**This is NOT a rebuild.** Everything shipped stays: plan altitude
(`document["plan"]["structure"]` = `{beat, need}`), the plan mirror
(`observe.plan_mirror_text`), the edit mirror (`observe.mirror`), `set_plan` /
`wrap_up`, `act.EditReject`, per-step reasoning capture into `edit_turns.trace`,
the convergence/churn advisories. We RESTRUCTURE the done-gate's control flow
and ADD stages, a commitment model, a severity policy, an op-history progress
definition, and a trace invariant. Every existing public symbol keeps its
signature (`_verify_before_finish` included, so the 12 existing gate tests keep
passing unchanged).

---

## 0. Motivation — the machinery was right and never ran (3 bypasses, 1 real failure)

We built: a durable plan artifact (`set_plan`), a finish-time plan-conformance
checkpoint (`_plan_conformance`), a required user-facing surface (`wrap_up` +
the surface-non-empty block), and convergence/churn advisories. **Twice in a row
the machinery was correct but never executed**, and a third bypass is latent.

### Bypass 1 — the done-gate is FINISH-GATED (fixed once, reopened by construction)

`_verify_before_finish` is only reachable from the voluntary-finish branch
(`tools.py:1238-1247`). A loop that exhausts `_MAX_TURNS` leaves via `for … else`
(`tools.py:1321-1322`) and skips the entire ladder. `brain_loop_convergence.plan.md`
Part 2 patched the *symptom* with `_finalize_after_loop` guarded on a boolean —
which is how Bypass 3 exists.

### Bypass 2 — the `reviewed` early-return kills the mandatory stages (THE DAMAGING ONE)

```1067:1068:backend/app/services/l3/tools.py
    if reviewed or state["reviewed"]:
        return None
```
This `return None` sits BEFORE Part B plan-conformance (`tools.py:1081-1084`) and
BEFORE Part A surface-non-empty (`tools.py:1086-1099`). `reviewed` is
```1033:1033:backend/app/services/l3/tools.py
    reviewed = "diagnose" in steps or "validate" in steps or "review" in steps
```
so **any turn that calls `review`/`validate`/`diagnose` permanently disables
plan-conformance and the surface block.** The brain almost always self-reviews,
so **plan-conformance is effectively dead code in real sessions** (proven by
replay). The short-circuit was written to skip ONE redundant advisory nudge
(Stage 3); it silently orphaned every stage declared after it.

**Bypass 2b (same root cause, found while writing this plan).** `review_out` is
only computed when `not reviewed`:
```1034:1039:backend/app/services/l3/tools.py
    review_out = None
    if not reviewed:
        try:
            review_out = observe.review(working, ctx, user_ask=user_ask)
        except Exception:
            logger.exception("_verify_before_finish: review() failed (continuing with diagnose only)")
```
so `ask_flags = [f for f in (review_out or {}).get("flags", []) …]`
(`tools.py:1041`) is ALWAYS empty on a self-reviewed turn — **Stage 1, fit-to-intent,
is also dead whenever the brain self-reviews.** PART 1 closes this by the same
construction; it is NOT a separate patch.

### Bypass 3 — latent: a last-turn voluntary finish sets `finished_clean=True`

```1241:1247:backend/app/services/l3/tools.py
            if changed and turn < max_turns - 1:
                feedback = _verify_before_finish(working, ctx, verify, steps, user_ask)
                if feedback is not None:
                    convo.append(user_message(feedback))
                    continue
            finished_clean = True   # voluntary finish, gate satisfied (or skipped clean)
            break
```
On `turn == max_turns - 1` the gate is SKIPPED, yet `finished_clean = True` is set
anyway — and the post-loop finalizer is guarded on `not finished_clean`:
```1329:1332:backend/app/services/l3/tools.py
    if changed and not questions and not finished_clean and _surface_is_empty(working):
        working = _finalize_after_loop(
            llm, system=system, convo=convo, ctx=ctx, working=working,
            tools=tools, verify=verify, user_ask=user_ask, max_tokens=max_tokens)
```
So the one path that skipped the gate is also the path that suppresses the
finalizer. **PART 1 must close this by construction — do NOT spec a separate
patch for it.**

### The concrete failure this allowed — a silent, unchallenged reversal

The brain's own plan said *"slides carry muted incidental talk — keep muted,
they're b-roll"*. The footage map had already tagged those cuts `muted`
(`cutrecord_map._audio_mute_for`, `cutrecord_map.py:591-623` → moment `mute`/`flags`,
`footage_map.py:327-332` → `arrange._resolve_mute`, `arrange.py:127-135` →
`seg["mute"]`). Four steps later the brain called `set_audio(mute=False)` on them
anyway, shredding the voice track (5 mid-sentence jumps across 2 recordings).
Nothing recorded the reversal, nothing challenged it, and the one checkpoint
designed to ask *"you wrote X — did you do X?"* **could not run** (Bypass 2).

Compounding failures observed in the same run, each an instance of a general gap:
- **All signals advisory.** 4 of 7 cuts carried `broken-line` (`observe.mirror`,
  `observe.py:1666-1668`) and shipped. Nothing in the system can BLOCK.
- **Asymmetric length contract.** `diagnose` computes over- AND under-target
  (`observe.py:1226-1235`) but the gate filters only over:
  ```1026:1026:backend/app/services/l3/tools.py
      over = [f for f in findings if "over target" in (f.get("message") or "")]
  ```
  A 34s result against a 60s ask with an entire source file untouched passed
  silently.
- **`changed` conflates everything.** `set_plan`/`wrap_up` both return
  `changed=True` (`tools.py:714`, `tools.py:724`) and feed the same `changed` flag
  (`tools.py:1262`) that drives the gate, the churn detector, the plan mirror's
  `building` loudness and the finalizer → 2 false "you're cycling" advisories (one
  on a completely EMPTY timeline), and 1 MISSED genuine revert (the undo landed
  100ms off the natural boundary, so `_edit_fingerprint` hashes differed).
- **Untraceable injections.** The PROGRESS/CHURN advisories are `text_block`s
  (`tools.py:1314-1316`) that are never persisted; diagnosing the run required
  replaying the state machine from source.
- **A trim past the moment's natural boundary applied silently** — ballooned a cut
  by 30s and pushed the program over the user's stated 60s cap, costing 2 of the
  last 4 turns.

### The user's standing requirement

**NO BAND-AIDS. Architecture-level changes that generalize.** Every part below is
a general mechanism; the specific instance that exposed it is a TEST CASE, never
the fix.

### OUT OF SCOPE (band-aids explicitly rejected)

- **Raising `_MAX_TURNS`** (`tools.py:37`). A higher cap moves the guillotine.
- **Any arbitrary retry counter / "max N tries" cap** as a behavior driver. (The
  existing `_STRUCT_MAX_TRIES`/`_SURFACE_MAX_BLOCKS` bounds stay ONLY as
  loop-termination guarantees for interactive stages, never as the mechanism.)
- **Fingerprint tolerance / epsilon / ms-window matching** for churn. PART 4 is
  symbolic (op history), so no epsilon exists to tune.
- **Per-bug special-casing.** No `if ref == "slides"`, no mute-specific stage.
- **Audio-only fixes.** PART 2 must apply to mutes, takes, order, pacing,
  required beats, structure — audio is one axis in a table.
- **A separate patch for Bypass 3, or for Bypass 2b.** Both must fall out of
  PART 1's construction.
- **A DB migration.** `document` and `edit_turns.trace` are both jsonb
  (`store.append_turn`, `store.py:107-122`); every new field is additive inside
  them. If an implementer believes a migration is needed, STOP and state why.

---

## 1. Current state to build on (verified against the code)

### 1.1 The done-gate ladder as it stands — `tools._verify_before_finish` (`tools.py:978-1100`)

Signature: `_verify_before_finish(working, ctx, state, steps, user_ask="") -> str | None`.
Returns feedback the loop injects and keeps going, or `None` to allow finishing.
**Control flow is a chain of nested early-returns**, in source order:

| # | Stage | Line | Kind today | Once-fired key | Skippable by `reviewed`? |
|---|---|---|---|---|---|
| 0 | structural (`observe.validate`) | `1009-1015` | hard, ≤`_STRUCT_MAX_TRIES=3` | `struct_tries` | no |
| 1 | length (`diagnose`, `"over target"` substring) | `1025-1031` | fix-or-justify, once | `length_surfaced` | no |
| 2 | Stage 1 fit-to-intent (`review` `category=="ask"`) | `1041-1047` | once | `intent_surfaced` | **yes, silently (Bypass 2b)** |
| 3 | Stage 2 fit-to-craft (LLM verdict, never parsed) | `1049-1057` | unconditional once | `craft_surfaced` | no (by design) |
| — | **`reviewed` early-return** | **`1067-1068`** | — | — | — |
| 4 | Stage 3 specific flags | `1069-1079` | advisory once | `reviewed` | yes (intended) |
| 5 | Part B plan-conformance | `1081-1084` | once | `conformance_surfaced` | **yes, unintended (Bypass 2)** |
| 6 | Part A surface-non-empty | `1086-1099` | gates, ≤`_SURFACE_MAX_BLOCKS=2` | `surface_blocked` | **yes, unintended (Bypass 2)** |

The `state` dict is initialised in the loop:
```1201:1203:backend/app/services/l3/tools.py
    verify = {"struct_tries": 0, "length_surfaced": False, "intent_surfaced": False,
             "craft_surfaced": False, "reviewed": False,
             "conformance_surfaced": False, "surface_blocked": 0}
```
and mirrored in the tests' `_gate_state()` (`scripts/test_tools_loop.py:459-462`).

**The structural defect:** stage order is implicit in source order, "mandatory"
vs "advisory" is nowhere declared, and any stage may `return` on behalf of every
stage below it. That is exactly how stages 2/5/6 became dead code.

### 1.2 The loop's THREE exits — `tools.run_edit_loop` (`tools.py:1187-1345`)

- **voluntary finish** — `if not resp.tool_calls:` (`tools.py:1238-1247`), the ONLY
  gate caller; skips the gate on the last turn yet sets `finished_clean=True`.
- **cap exhaustion** — `for … else` (`tools.py:1321-1322`): logs only.
- **`ask_user` pause** — `if asked and questions: break` (`tools.py:1319-1320`).

Post-loop: the `finished_clean`-guarded `_finalize_after_loop` (`tools.py:1329-1332`)
then the reply builder (`tools.py:1334-1343`).

Reusable helpers: `_finalize_after_loop` (`tools.py:1125-1165`), `_fallback_summary`
(`tools.py:1103-1122`), `_surface_is_empty` (`tools.py:920-930`),
`_plan_conformance` (`tools.py:933-975`), `_conformance_hints` (`tools.py:904-917`),
`_progress_note` (`tools.py:1168-1184` — note its `churn` parameter is **accepted
and never used**; PART 4 removes it), `_edit_fingerprint` (`tools.py:437-458`).

### 1.3 Where per-turn context is INJECTED (all unrecorded today)

```1300:1317:backend/app/services/l3/tools.py
        mirror_text = observe.mirror_text(working, ctx)
        if mirror_text:
            results.append(text_block(mirror_text))
        plan_text = observe.plan_mirror_text(working, building=changed)
        if plan_text:
            results.append(text_block(plan_text))
        results.append(text_block(_progress_note(turn, max_turns, churn=churn_now)))
        if churn_hint:
            results.append(text_block(churn_hint))
        convo.append(user_message(results))
```
Plus two `user_message` injections: gate feedback (`tools.py:1244`) and the
finalize instruction (`tools.py:1149`). None reach `trace`.

### 1.4 The trace (the observability substrate PART 5 extends)

`LoopResult.trace` (`tools.py:44-64`) is an ordered list of `kind`-tagged entries
— `kind:"reasoning"` (`tools.py:1231-1235`) and `kind:"tool"` (`tools.py:1257-1259`,
`tools.py:1264-1266`) — persisted verbatim onto `edit_turns.trace` (jsonb) via
`store.append_turn(trace=result.trace)` (`store.py:107-122`, called from
`routers/edit_threads.py:131`). Readers filter by `kind`
(`scripts/test_reasoning_capture.py:67-68`), so new kinds are additive.

### 1.5 Where PERCEPTION DISPOSITIONS come from (PART 2's other commitment source)

The chain, verified end to end:

1. **`cutrecord_map._audio_mute_for(channel, row)`** (`cutrecord_map.py:591-623`)
   decides a video cut's default sound, from transcript truth
   (`speech_words_in_span`, `cutrecord_map.py:528-549`):
   ```618:623:backend/app/services/l3/cutrecord_map.py
           if is_musical:
               return "speech", True, ["speech", "incidental", "muted"]
           return "speech", False, ["speech", "incidental"]
       natural_sound = bool((row.get("pace") or {}).get("natural_sound"))
       if natural_sound:
           return "sound", False, []
       return "silent", True, ["muted"]
   ```
   written onto the cut record as `flags` / `audio` / `mute`
   (`cutrecord_map.py:661-663`).
2. **`footage_map`** carries them onto the map moment:
   ```327:332:backend/app/services/l3/footage_map.py
               "flags": cut.get("flags") or [],
               # Source-audio facet for video cuts: what's on the track ("speech"/
               # "ambient") and whether it's muted by default (stray speech under a
               # shot). Said cuts leave these unset (their audio is the point).
               "audio": cut.get("audio"),
               "mute": bool(cut.get("mute")),
   ```
   So `ctx.meta_by_ref[ref]` (`observe.py:105-108`) is the authoritative
   disposition record for any placed cut.
3. **`arrange._resolve_mute`** (`arrange.py:127-135`) folds the brain's per-place
   `audio:"keep"|"mute"` onto that default → `ResolvedCut.mute` (`arrange.py:91`,
   set at `arrange.py:170` and `:185`) → `act.place`'s segment
   (`act.py:104`: `"mute": True if rc.mute else None`).
4. **`act.set_audio`** (`act.py:662-679`) flips `seg["mute"]` afterwards with no
   reference to the disposition it is overriding.
5. **`observe.mirror`** re-derives the live per-cut flags each turn
   (`observe.py:1659-1671`): `incidental` (`:1665`), `broken-line` from
   `cutrecord_map.speech_boundary_flags` (`cutrecord_map.py:557-588`, used at
   `observe.py:1666-1668`), `dead-air`, `jump-cut`.

**Conclusion for PART 2:** the committed value and the current value of a
disposition are both readable, cheaply and generically —
`ctx.meta_by_ref[seg["ref"]]` vs the segment/op field. No new persistence needed
to DETECT a contradiction.

### 1.6 Signals and their (cosmetic) severity

`observe.diagnose` (`observe.py:1202-1250`) and `observe.review`
(`observe.py:1491-1566`) emit `{severity, anchor, message, category?}` dicts.
`severity` is a free string `"warn"|"info"`, used only for display
(`tools.py:1074`). `category` is `"ask"|"guidance"|"craft"|"continuity"`
(`observe._tag_category`, `observe.py:1443-1446`). **Nothing consumes `severity`
as a decision.** Consumers string-match prose (`tools.py:1026`, `tools.py:1069`) —
the fragility PART 3 replaces with a machine `kind`.

### 1.7 The verb validation layer (PART 3's fold-in)

`act.EditReject` (`act.py:38-48`) is raised co-located and caught once
(`tools.py:747-751`) → `{"applied": false, "reason": …}`. One dispatch-level
domain guard exists: `_trim_domain_reason` (`tools.py:497-525`), wired at
`tools.py:604-607`. It checks the SOURCE FILE length only. It does not know a
moment's **natural boundary** — which `act.retime` already reads off the same
map (`act.py:1020`: `base_in, base_out = int(m.get("in_ms", …)), int(m.get("out_ms", …))`)
and which `observe._seams_for_file` (`observe.py:377-392`) exposes as clean edit
points. Nor does any verb consult the **ask's stated constraints**
(`document["brief"]["target_duration_s"]`, set from the user's own words by
`converse._extract_target_s`, `converse.py:627-640`).

### 1.8 House style to mirror

`brain_loop_convergence.plan.md` / `brain_plan_conformance.plan.md`: additive,
fail-open, no migration, "prompt DISCIPLINE + robust deterministic signals",
before/after snippets against real symbols. Tests live in
`backend/scripts/test_tools_loop.py` (`_ScriptedLLM` `:60-89`, `_ctx`/`_struct`
`:42-52`, `_seed_doc` `:55-57`, `_gate_state` `:459-462`, `_clean_doc` `:465-471`),
registered in `main()`.

---

## PART 1 — ONE MANDATORY FINALIZATION CHOKE POINT *(land first, alone)*

**Goal.** (a) EVERY termination path converges on ONE finalizer. (b) The done-gate
becomes a **DECLARED, ORDERED LIST of stages**, each tagged **mandatory** or
**advisory**, evaluated by one explicit evaluator — **no nested early-returns**.
An advisory-skip may skip only advisory stages. Mandatory invariants
(plan-conformance account, surface-non-empty, fit-to-intent) become structurally
unskippable. A future stage cannot be silently orphaned by an earlier `return`.

### 1.1 New declarations — `tools.py`, immediately above `_verify_before_finish` (`tools.py:978`)

```python
# --------------------------------------------------------------------------
# brain_accountability_architecture.plan.md PART 1: the done-gate as a
# DECLARED ORDERED LIST of stages, not a chain of nested early-returns.
#
# Each stage is a pure (or fail-open) function of one _GateCtx and returns its
# OWN feedback string or None. A stage NEVER returns on behalf of a later
# stage -- the evaluator (_run_gate_stages) owns the ladder -- which is what
# makes a mandatory invariant structurally unskippable and makes it impossible
# to orphan a stage by adding a `return` above it.
#
#   kind="mandatory": an invariant. Runs on EVERY gate evaluation, on EVERY
#     termination path. Never filtered by the advisory-skip.
#   kind="advisory":  a redundant nudge. May be skipped when the brain already
#     self-reviewed this turn (the ONLY thing the old `reviewed` short-circuit
#     was ever meant to skip).
# --------------------------------------------------------------------------

MANDATORY, ADVISORY = "mandatory", "advisory"


@dataclass(frozen=True)
class _Stage:
    key: str        # the `state` key this stage OWNS (its once-fired bookkeeping)
    kind: str       # MANDATORY | ADVISORY
    label: str      # stable machine token, recorded in the trace (PART 5)
    fn: Any         # (_GateCtx) -> str | None


@dataclass
class _GateCtx:
    """Everything a stage may read, computed ONCE per gate evaluation.

    `findings` (observe.diagnose) and `review_out` (observe.review) are computed
    UNCONDITIONALLY -- the old code skipped review() whenever the brain had
    self-reviewed, which silently emptied Stage 1's ask-flags (Bypass 2b). A
    mandatory stage may never depend on an optimization for an advisory one."""
    working: dict
    ctx: EditContext
    state: Dict[str, Any]
    steps: List[str]
    user_ask: str
    advisory_skip: bool
    findings: List[dict] = field(default_factory=list)
    review_out: Dict[str, Any] | None = None
    # Per-evaluation AUDIT of the ladder: one entry per declared stage, in
    # declared order -- {stage, kind, fired, why?}. Recorded into the trace by
    # PART 5 so "did the mandatory invariant actually run?" is assertable from a
    # session test and diagnosable from a stored thread.
    fired: List[dict] = field(default_factory=list)


def _gate_ctx(working: dict, ctx: EditContext, state: Dict[str, Any],
              steps: List[str], user_ask: str = "") -> _GateCtx:
    """Build the per-evaluation context. Fail-open: a sense that raises yields
    an empty projection, never an aborted gate."""
    advisory_skip = bool(
        "diagnose" in steps or "validate" in steps or "review" in steps
        or state.get("reviewed"))
    try:
        findings = observe.diagnose(working, ctx)
    except Exception:
        logger.exception("_gate_ctx: diagnose failed (continuing with none)")
        findings = []
    try:
        review_out = observe.review(working, ctx, user_ask=user_ask)
    except Exception:
        logger.exception("_gate_ctx: review failed (continuing without it)")
        review_out = None
    return _GateCtx(working=working, ctx=ctx, state=state, steps=steps,
                    user_ask=user_ask, advisory_skip=advisory_skip,
                    findings=findings, review_out=review_out)
```

### 1.2 The stages — mechanical extraction, one function per rung, SAME strings

Each existing rung becomes a module-private `_stage_*` function whose body is the
CURRENT body verbatim (same wording, same `state` key, same once-fired logic),
with `return feedback` / `return None` now meaning *only* "this stage has / has
not something to say".

Before (nested, `tools.py:1009-1031`):
```1009:1031:backend/app/services/l3/tools.py
    issues = observe.validate(working, ctx)
    if issues and state["struct_tries"] < _STRUCT_MAX_TRIES:
        state["struct_tries"] += 1
        body = "; ".join(f"{i.get('kind')} {i.get('id')}: {i.get('message')}"
                         for i in issues[:8])
        return ("AUTOMATIC CHECK -- structural problems that would break the render. "
                "Fix these before finishing:\n" + body)

    # PHASE 3 SEAM (brain_plan_mechanism.plan.md §8) -- LANDED
    # (brain_plan_conformance.plan.md): plan-vs-edit conformance is now live,
    # but deliberately placed at the ladder's END (after Stage 3), not here --
    # so the conformance account judges what SURVIVED the earlier mechanical/
    # craft fixes, not noise the brain was about to clean up anyway. See Part
    # B (_plan_conformance) and Part A (the surface-non-empty block) right
    # before this function's final `return None`.

    findings = observe.diagnose(working, ctx)
    over = [f for f in findings if "over target" in (f.get("message") or "")]
    if over and not state["length_surfaced"]:
        state["length_surfaced"] = True
        return ("AUTOMATIC CHECK -- length: " + (over[0].get("message") or "") +
                ". Either trim to the target, or say in one line why this length is "
                "right, then finish.")
```

After (declared):
```python
def _stage_structural(gc: _GateCtx) -> str | None:
    issues = observe.validate(gc.working, gc.ctx)
    if not issues or gc.state["struct_tries"] >= _STRUCT_MAX_TRIES:
        return None
    gc.state["struct_tries"] += 1
    body = "; ".join(f"{i.get('kind')} {i.get('id')}: {i.get('message')}"
                     for i in issues[:8])
    return ("AUTOMATIC CHECK -- structural problems that would break the render. "
            "Fix these before finishing:\n" + body)


def _stage_length(gc: _GateCtx) -> str | None:
    # PART 3 replaces the "over target" substring match with a symmetric,
    # kind-keyed check. Until then this body is the current one, verbatim.
    over = [f for f in gc.findings if "over target" in (f.get("message") or "")]
    if not over or gc.state["length_surfaced"]:
        return None
    gc.state["length_surfaced"] = True
    return ("AUTOMATIC CHECK -- length: " + (over[0].get("message") or "") +
            ". Either trim to the target, or say in one line why this length is "
            "right, then finish.")


def _stage_intent(gc: _GateCtx) -> str | None:
    """Stage 1, fit to the ask. MANDATORY -- and it now actually sees flags on a
    self-reviewed turn (Bypass 2b), because _gate_ctx always computes review."""
    ask_flags = [f for f in (gc.review_out or {}).get("flags", [])
                 if f.get("category") == "ask"]
    if not ask_flags or gc.state["intent_surfaced"]:
        return None
    gc.state["intent_surfaced"] = True
    body = "\n".join(f"- {f.get('message')}" for f in ask_flags[:8])
    return ("AUTOMATIC CHECK -- Stage 1, fit to your ask: a feature you named "
            "isn't actually in the edit:\n" + body +
            "\nAdd it, or finish only if the material genuinely can't support it.")


def _stage_craft(gc: _GateCtx) -> str | None:      # body verbatim from tools.py:1049-1057
    ...


def _stage_flags(gc: _GateCtx) -> str | None:
    """Stage 3, specific flags. The ONE advisory stage -- the only thing the old
    `reviewed` short-circuit was ever meant to skip."""
    rest = [f for f in gc.findings if "target" not in (f.get("message") or "")]
    rest += [f for f in (gc.review_out or {}).get("flags", [])
             if f.get("category") != "ask"]
    if not rest or gc.state["reviewed"]:
        return None
    gc.state["reviewed"] = True
    ...                                            # body verbatim from tools.py:1073-1079


def _stage_conformance(gc: _GateCtx) -> str | None:
    return _plan_conformance(gc.working, gc.ctx, gc.working.get("plan"), gc.state)


def _stage_surface(gc: _GateCtx) -> str | None:    # body verbatim from tools.py:1086-1099
    ...
```

### 1.3 The declared ladder + the evaluator

```python
# THE LADDER. Order is data, not source order. A new stage is added HERE or it
# never runs (loud in review) -- it can no longer be orphaned by an earlier
# `return`. `_GATE_MANDATORY` is asserted in the tests, so silently demoting an
# invariant to advisory fails CI.
_GATE_STAGES: Tuple[_Stage, ...] = (
    _Stage("struct_tries",         MANDATORY, "structural",  _stage_structural),
    _Stage("length_surfaced",      MANDATORY, "length",      _stage_length),
    _Stage("intent_surfaced",      MANDATORY, "intent",      _stage_intent),
    _Stage("craft_surfaced",       MANDATORY, "craft",       _stage_craft),
    _Stage("reviewed",             ADVISORY,  "flags",       _stage_flags),
    _Stage("conformance_surfaced", MANDATORY, "conformance", _stage_conformance),
    _Stage("surface_blocked",      MANDATORY, "surface",     _stage_surface),
)


def _run_gate_stages(gc: _GateCtx, *, kinds=(MANDATORY, ADVISORY),
                     collect: bool = False) -> Any:
    """Evaluate the declared ladder in order.

    Semantics, explicit and total:
      * a stage whose `kind` is not in `kinds` is not evaluated (recorded as
        skipped-by-scope);
      * an ADVISORY stage is skipped when `gc.advisory_skip` -- this is the ONLY
        skip mechanism, and it can never reach a MANDATORY stage;
      * a stage that raises is logged and treated as "nothing to say" (fail-open,
        house rule) -- it can never suppress the stages after it;
      * `collect=False` (interactive): return the FIRST stage's feedback, or None
        -> the loop asks one thing at a time, exactly as today;
      * `collect=True` (enforcement, used by the ONE finalizer): evaluate EVERY
        in-scope stage and return the list of all outstanding feedback.
    Every outcome is appended to `gc.fired` for the trace (PART 5)."""
    out: List[str] = []
    for st in _GATE_STAGES:
        if st.kind not in kinds:
            gc.fired.append({"stage": st.label, "kind": st.kind, "fired": False,
                             "why": "out-of-scope"})
            continue
        if st.kind == ADVISORY and gc.advisory_skip:
            gc.fired.append({"stage": st.label, "kind": st.kind, "fired": False,
                             "why": "advisory-skip: brain self-reviewed"})
            continue
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
        if not collect:
            return fb
        out.append(fb)
    return out if collect else None
```

**`_verify_before_finish` keeps its exact signature** and becomes the interactive
entry point — so all 12 existing gate tests
(`test_tools_loop.py:474-…`) pass unchanged:
```python
def _verify_before_finish(working: dict, ctx: EditContext,
                          state: Dict[str, Any], steps: List[str],
                          user_ask: str = "") -> str | None:
    """The done-gate, interactive mode: evaluate the DECLARED ladder
    (_GATE_STAGES) and return the first stage's feedback, or None to allow
    finishing. The ladder's order, and which stages are mandatory vs advisory,
    are declared data -- see _GATE_STAGES. This function no longer contains any
    stage logic and MUST NOT grow an early return."""
    return _run_gate_stages(_gate_ctx(working, ctx, state, steps, user_ask))
```

### 1.4 The ONE finalizer — every termination path converges here

`finished_clean` is **DELETED**. Whether finalization work remains is derived
from the declared ladder + the `state` evidence, never from a boolean set at a
site that may have skipped the gate. That is Bypass 3, closed by construction.

New helper, replacing the `finished_clean`-guarded call site:
```python
def _finalize_turn(llm, *, system, convo, ctx, working, tools, state, steps,
                   user_ask, trace, edit_moved, questions, exit_reason,
                   max_tokens) -> dict:
    """THE ONE FINALIZATION CHOKE POINT (PART 1). Every termination path lands
    here exactly once -- voluntary finish, cap exhaustion, last-turn finish,
    ask_user pause -- and the MANDATORY stages are evaluated in ENFORCEMENT mode
    (collect=True) regardless of which path arrived. There is no flag that can
    say "already fine": whether anything is outstanding is READ OFF the declared
    ladder and the state dict.

    `exit_reason` is recorded, never consulted for control flow -- except for the
    one legitimate case: a paused ask_user turn keeps its own question framing
    (the turn isn't over, the user is being asked something).

    Fail-open: any error leaves `working` untouched; the reply builder's
    last_text fallback is unchanged."""
    if questions:
        _record_gate(trace, exit_reason=exit_reason, gc=None,
                     note="paused for ask_user -- finalization deferred")
        return working
    if not edit_moved:                    # PART 4: a plan-only turn is not an edit
        return working
    gc = _gate_ctx(working, ctx, state, steps, user_ask)
    outstanding = _run_gate_stages(gc, kinds=(MANDATORY,), collect=True)
    _record_gate(trace, exit_reason=exit_reason, gc=gc)      # PART 5
    if outstanding or _surface_is_empty(working):
        working = _finalize_after_loop(
            llm, system=system, convo=convo, ctx=ctx, working=working,
            tools=tools, verify=state, user_ask=user_ask, trace=trace,
            outstanding=outstanding, max_tokens=max_tokens)
    return working
```

`_finalize_after_loop` (`tools.py:1125-1165`) changes in two small ways:
- it takes `outstanding: List[str]` and appends those lines to `instruct`
  **instead of** calling `_plan_conformance` itself (`tools.py:1138`) — the
  conformance account is now just one of the collected mandatory lines, which
  removes the double-fire hazard;
- it takes `trace` and pushes its instruction through PART 5's `_inject_message`.

Its deterministic floor (`_fallback_summary` → `act.wrap_up`, `tools.py:1158-1164`)
is unchanged.

### 1.5 Loop restructure — `run_edit_loop`

Before:
```1238:1247:backend/app/services/l3/tools.py
        if not resp.tool_calls:
            # Finish attempt: don't let a changed edit exit unchecked against the
            # contract (structural = hard, length = fix-or-justify, rest advisory).
            if changed and turn < max_turns - 1:
                feedback = _verify_before_finish(working, ctx, verify, steps, user_ask)
                if feedback is not None:
                    convo.append(user_message(feedback))
                    continue
            finished_clean = True   # voluntary finish, gate satisfied (or skipped clean)
            break
```
After:
```python
        if not resp.tool_calls:
            # Finish attempt. The gate now runs on EVERY finish attempt, the last
            # turn included (the old `turn < max_turns - 1` guard is what made a
            # last-turn finish skip the ladder -- Bypass 3). On the last turn
            # there is no room to iterate, so feedback is not injected; the
            # stages have marked `verify`, and _finalize_turn enforces whatever
            # is still outstanding.
            if edit_moved:
                feedback = _verify_before_finish(working, ctx, verify, steps, user_ask)
                if feedback is not None and turn < max_turns - 1:
                    _inject_message(convo, trace, turn=turn, source="gate",
                                    text=feedback)          # PART 5
                    continue
            exit_reason = "voluntary"
            break
```
Before the loop: `exit_reason = "cap"` (replacing `finished_clean = False`,
`tools.py:1204`). At the `ask_user` break (`tools.py:1319-1320`):
`exit_reason = "asked"`.

Replacing the old guarded call (`tools.py:1329-1332`):
```python
    # PART 1: the ONE finalization choke point. Unconditional on the exit path --
    # no `finished_clean`, no `not questions and not …` guard chain deciding
    # whether the invariants get to run.
    working = _finalize_turn(
        llm, system=system, convo=convo, ctx=ctx, working=working, tools=tools,
        state=verify, steps=steps, user_ask=user_ask, trace=trace,
        edit_moved=edit_moved, questions=questions, exit_reason=exit_reason,
        max_tokens=max_tokens)
```
(Until PART 4 lands, `edit_moved` is simply `changed` — a one-line alias — so
PART 1 is landable alone.)

### 1.6 How this closes all three bypasses BY CONSTRUCTION

| Bypass | Closed by |
|---|---|
| **1** cap exit skips the gate | `_finalize_turn` sits OUTSIDE the loop and is called on every exit path; `exit_reason` is recorded, not consulted (except the ask_user pause). A cap exit gets the same mandatory `collect=True` pass as a voluntary finish. |
| **2** `reviewed` early-return kills conformance + surface | The short-circuit is no longer a `return`. It is `_GateCtx.advisory_skip`, a filter the EVALUATOR applies, and only to stages declared `ADVISORY`. `conformance`/`surface`/`intent` are `MANDATORY` → unreachable by the filter. Structurally, no stage can return on another's behalf: a stage fn returns only its own feedback and `_run_gate_stages` owns the ladder. |
| **2b** Stage 1 dead on a self-reviewed turn | `_gate_ctx` computes `review_out` unconditionally. A mandatory stage's inputs may never be gated on an advisory optimization. |
| **3** last-turn finish sets `finished_clean=True` | `finished_clean` is deleted. The gate runs on the last turn too; the finalizer derives outstanding work from `_GATE_STAGES` + `state`. There is no boolean left to lie. |
| **future orphaning** | A stage exists only if it is in `_GATE_STAGES`. Adding a stage function without registering it means it never runs (visible in review), and `test_gate_stage_registry_is_well_formed` asserts the registry's shape + the exact mandatory set. |

**Termination is still guaranteed:** interactive mode returns at most one
feedback per call, and every stage owns a once-fired key (`struct_tries` ≤
`_STRUCT_MAX_TRIES`, `surface_blocked` ≤ `_SURFACE_MAX_BLOCKS`, the rest boolean).
Every stage added by PARTS 2–3 must declare its own bound — stated at each site.

### Back-compat / additive / fail-open (PART 1)

- `_verify_before_finish`'s signature, the `state` dict's keys, and every
  feedback STRING are unchanged → the 12 existing gate tests and
  `_gate_state()` need no edits.
- Behavior changes are exactly the intended un-bypassings: conformance + surface
  + intent now also run on a self-reviewed turn, on a cap exit, and on a
  last-turn finish. Each is still once-fired/bounded, so no new nagging.
- One extra `observe.review` call per gate evaluation on self-reviewed turns
  (previously skipped). It is a pure projection; the brain had just called it.
- Fail-open at every level: `_gate_ctx` degrades senses to empty, each stage is
  try/excepted inside the evaluator, `_finalize_turn` returns `working`
  untouched on any error, and the reply builder's `last_text` fallback is
  unchanged.
- No migration.

### Testing (PART 1)

**Unit** (`scripts/test_tools_loop.py`):
- `test_gate_stage_registry_is_well_formed` — every `_Stage` has a distinct
  `key` present in `_gate_state()`, a `kind` in `{MANDATORY, ADVISORY}`, a
  distinct `label`; and `{s.label for s in _GATE_STAGES if s.kind == MANDATORY}
  == {"structural","length","intent","craft","conformance","surface"}` (so
  demoting an invariant fails CI).
- `test_advisory_skip_skips_only_the_advisory_stage` — with
  `steps=["place","review"]`, `_verify_before_finish` on a doc that would trip
  BOTH Stage 3 and conformance returns the CONFORMANCE feedback, not None; the
  `flags` stage records `fired=False, why="advisory-skip…"`.
- `test_conformance_and_surface_run_after_self_review` — the direct regression
  for Bypass 2: a doc with a plan + a live hint + empty surface, `steps` including
  `"review"` → successive calls yield the conformance account, then the surface
  block, then None.
- `test_intent_stage_fires_after_self_review` — Bypass 2b: stub `observe.review`
  to return one `category=="ask"` flag, `steps=["place","diagnose"]` → Stage 1
  fires.
- `test_gate_stage_error_does_not_suppress_later_stages` — monkeypatch
  `_stage_craft` to raise → the conformance/surface stages still fire.
- `test_collect_mode_returns_all_outstanding_mandatory` —
  `_run_gate_stages(gc, kinds=(MANDATORY,), collect=True)` returns >1 line for a
  doc failing length AND conformance, and never includes the advisory stage.

**Session-level (REQUIRED — see §Testing lesson).** Each drives
`tools.run_edit_loop` with `_ScriptedLLM` and asserts on `res.trace`'s
`kind:"gate"` entry that the mandatory stages ACTUALLY FIRED:
- `test_session_selfreview_turn_still_runs_conformance_and_surface` — script:
  `set_plan` → `place` → `review` → prose finish. Assert the trace's `gate`
  entry lists `conformance` and `surface` with `fired=True` (this is the exact
  session shape that shipped "verified" while both were dead), and that
  `res.reply` is a `wrap_up` summary, not leftover prose.
- `test_session_cap_exhaustion_runs_mandatory_stages` — a script that ALWAYS
  returns tool calls, `max_turns=3`. Assert a `kind:"gate"` entry with
  `exit_reason="cap"` exists and every mandatory stage was evaluated.
- `test_session_last_turn_finish_runs_mandatory_stages` — a script whose FINAL
  scripted step (at `turn == max_turns - 1`) is a prose finish after an applied
  `place`. Assert the gate ran (Bypass 3) and the surface is non-empty.
- `test_session_clean_finish_costs_no_extra_llm_call` — a voluntary finish with
  nothing outstanding and a non-empty surface → `llm.calls` count unchanged from
  today (proves we did not turn the finalizer into an unconditional extra round
  trip).
- `test_session_ask_user_pause_defers_finalization` — `ask_user` turn →
  `awaiting_user` True, reply is the question framing, no forced `wrap_up`.

**Manual e2e:** re-run the failing session. Confirm the stored
`edit_turns.trace` contains a `kind:"gate"` entry naming `conformance:fired` on
a turn that called `review`.

---

## PART 2 — COMMITMENTS AND EXPLICIT REVERSALS

**Problem.** There is no concept of a standing commitment or of an explicit
reversal. Plan items AND perception dispositions (§1.5) are commitments the
brain can silently contradict.

**Change.** A GENERAL commitment/reversal model: reversing a standing commitment
must be (1) explicit/acknowledged, (2) recorded onto the plan, (3) surfaced in
`wrap_up`.

### 2.1 What counts as a STANDING COMMITMENT

A commitment is a normalized record — **not** a verb, **not** a flag:

```python
@dataclass(frozen=True)
class Commitment:
    scope: str      # "plan" | "perception" | "ask"
    axis: str       # "mute" | "presence" | "take" | "order" | "pace" | "beat" | "length"
    subject: str    # a map ref, a take/dup group id, a beat text, or "whole"
    value: Any      # the committed value on that axis
    origin: str     # human provenance: "plan rev 3", "footage map default", "the ask"
```

Three sources, all already readable — **no new persistence to detect a
contradiction**:

| scope | axis | where the committed value lives |
|---|---|---|
| `plan` | `beat` | `document["plan"]["structure"]` entries with `need=="required"` (`act._norm_beat`, `act.py:709-722`; read via `observe._beat_view`, `observe.py:1724-1731`) |
| `plan` | `order` / `pace` / `mute` … | `plan["carries"]` + `plan["watch"]` prose, matched to an axis by the same keyword approach `_conformance_hints` already uses (`tools.py:837-841`) |
| `perception` | `mute` | `ctx.meta_by_ref[ref]["mute"]` (§1.5 step 2) — the footage map's default for that moment |
| `perception` | `presence`/`speech` | `ctx.meta_by_ref[ref]["flags"]` (`incidental`, `speech`, `muted`) and `["audio"]` |
| `perception` | `take` | `ctx.dup_groups` / `moment["take_group_id"]`+`["take_role"]` — the code-crowned winner is the standing take |
| `ask` | `length` | `document["brief"]["target_duration_s"]` |

New module `backend/app/services/l3/commit.py` (a new file, so PART 1's
restructure of `tools.py` is not re-touched):
```python
def ledger(document: dict, ctx: EditContext) -> List[Commitment]:
    """Every STANDING commitment the current edit is built on -- the plan the
    brain wrote plus the dispositions perception handed it. Pure, fail-open,
    derived: nothing here is persisted, because both sources are already durable
    (the plan lives in the document; the dispositions live in the map)."""
```

### 2.2 How a contradiction is detected GENERICALLY (state diff, not verb table)

The rule, stated once: **project the document onto a small set of AXES, and a
transition whose post-state differs from a standing commitment's value on that
axis is a REVERSAL.** Detection reads the DOCUMENT, never the verb's arguments —
which is exactly what makes it general: a mute can flip via `set_audio`
(`act.py:662-679`), via `place(audio="keep")` (`arrange._resolve_mute`), via
`tighten`/`retime` re-slicing (`act.py:918-943`, `act.py:984-1037`), or via
`arrange.heal_adjacent_cuts`' unmute-on-merge (`arrange.py:287-290`). All of them
show up in the diff; none of them needs its own branch.

```python
# commit.py -- the AXES. Each reads a value off the document; `subject` is a
# STABLE identity (a map ref, a group id), never a freshly-minted seg_id, which
# is why no tolerance/epsilon is needed anywhere in this file.
_AXES = (
    _Axis("mute",     per="seg",   read=lambda seg, doc, ctx: bool(seg.get("mute"))),
    _Axis("presence", per="ref",   read=lambda ref, doc, ctx: _on_main_line(ref, doc)),
    _Axis("take",     per="group", read=lambda gid, doc, ctx: _placed_member(gid, doc, ctx)),
    _Axis("order",    per="whole", read=lambda _, doc, ctx: tuple(
        s.get("ref") for s in (doc.get("timeline") or []))),
    _Axis("pace",     per="seg",   read=lambda seg, doc, ctx: seg.get("pace_level")),
)


def reversals(before: dict, after: dict, ctx: EditContext,
              led: List[Commitment]) -> List[dict]:
    """Standing commitments the transition before->after CONTRADICTS. One entry
    per reversal: {axis, subject, was, now, origin, commitment_scope}. Pure and
    fail-open (any axis that raises is skipped, never a blocked turn)."""
```

Wired in `run_edit_loop`, at the site where churn is computed today
(`tools.py:1267-1292`), using the pre-step document — cheap, since every verb
returns a NEW doc (`act._clone`, `act.py:63-72`) so holding the old reference is
free:
```python
        before_step = working
        for tc in resp.tool_calls:
            ...
            obs, working, did = _dispatch(tc.name, tc.input or {}, ctx, working, user_ask)
        ...
        # PART 2: did this step contradict a standing commitment? Generic:
        # measured as a DOCUMENT DIFF on declared axes, so it covers a mute, a
        # take swap, a reorder, a dropped required beat or a pace change alike.
        try:
            new_rev = commit.reversals(before_step, working, ctx,
                                       commit.ledger(working, ctx))
            for r in new_rev:
                verify.setdefault("reversals", []).append(
                    dict(r, turn=turn, plan_rev=(working.get("plan") or {}).get("rev", 0)))
                _inject(results, trace, turn=turn, source="reversal",
                        text=commit.render_reversal(r))        # PART 5
        except Exception:
            logger.exception("tools: reversal detection failed (continuing)")
```

The injected text (one shape for every axis — never audio-specific):
```
REVERSAL: you just changed <axis> on <subject> from <was> to <now>, contradicting
a standing commitment (<origin>). A reversal is legitimate -- material teaches you
things -- but it is never SILENT: either put it back, or (1) say plainly why, (2)
record it on the plan with set_plan(note="…") so the plan of record matches the
edit, and (3) state it to the user in wrap_up. Same two-reasons discipline as
fit-to-craft: the ask required it, or the material can't support better.
```

### 2.3 What is REQUIRED of the brain — a new MANDATORY gate stage

```python
_REVERSAL_MAX_BLOCKS = 2   # this stage GATES, so it needs its own bound (PART 1 §1.5)


def _stage_reversals(gc: _GateCtx) -> str | None:
    """PART 2: no unacknowledged reversal of a standing commitment may finish.

    ACKNOWLEDGED means BOTH, checked structurally (never by parsing prose):
      (a) RECORDED: the plan's `rev` was bumped after the reversal was detected
          (act.set_plan bumps rev + writes updated_note, act.py:732-743), i.e.
          plan["rev"] > the rev recorded at detection; and
      (b) SURFACED: the user-facing surface is non-empty (act.wrap_up ->
          document["summary"]/["open_questions"]/["notes"]; _surface_is_empty,
          tools.py:920-930).
    Reverting the change also clears it (the reversal is re-evaluated from the
    live document, so a put-back leaves nothing outstanding)."""
```
Registered in `_GATE_STAGES` between `craft` and `conformance` (a reversal is a
plan-of-record problem; the conformance account should read the UPDATED plan):
```python
    _Stage("craft_surfaced",       MANDATORY, "craft",       _stage_craft),
    _Stage("reversal_blocked",     MANDATORY, "reversals",   _stage_reversals),   # PART 2
    _Stage("reviewed",             ADVISORY,  "flags",       _stage_flags),
```
`_gate_state()` / `verify` gain `"reversals": []` and `"reversal_blocked": 0`.

Durability across turns (additive, jsonb, no migration): an acknowledged
reversal is appended to `document["plan"]["reversals"]` as
`{axis, subject, was, now, origin, note, rev}` by a tiny writer
`act.record_reversal(doc, item)`; and **`act.set_plan` must PRESERVE
`plan["reversals"]` across its full replace** (`act.py:734-743` builds a fresh
`plan` dict — carry the key over). `observe.plan_mirror_text` gains one rendered
line (`observe.py:1769-1772`, beside `last change:`) so the next turn sees the
reversals of record.

Prompt discipline — `converse._LOOP_SYSTEM`, "THE PLAN" block
(`converse.py:207-218`), one sentence: *the plan and what the footage map already
decided are STANDING COMMITMENTS; reversing one is allowed and sometimes right,
but never silent — say why, update the plan, and tell the user.* And `wrap_up`'s
schema description (`tools.py:385-396`) gains: *if you reversed a standing
commitment — a mute the map set, a take you chose, a beat you planned — say so
here.*

### Back-compat / fail-open (PART 2)

Additive module + one additive gate stage + additive jsonb keys. Detection is
pure and try/excepted; an empty ledger (no plan, no map metadata) yields no
reversals, so a thread with no plan behaves exactly as today. `_AXES` is
data — a new axis is one tuple entry, and an axis that raises is skipped.

### Testing (PART 2)

**Unit** (new `scripts/test_commitments.py`, plus gate tests in
`test_tools_loop.py`):
- `test_ledger_reads_plan_and_perception_commitments` — a plan with one required
  beat + a moment whose map `mute` is True → both appear, with distinct
  `scope`/`origin`.
- `test_reversal_detected_for_unmute_of_a_map_muted_cut` — the real failure, as a
  test case: place a `mute:True` moment, then `set_audio(mute=False)` → one
  reversal `{axis:"mute", was:True, now:False, origin:"footage map default"}`.
- `test_reversal_detected_for_take_swap_and_for_reorder_and_for_pace` — the SAME
  mechanism on three non-audio axes (proves it is not audio-specific).
- `test_reversal_detected_when_place_overrides_the_default_at_placement`
  (`place(..., audio="keep")` on a `mute:True` moment).
- `test_no_reversal_for_a_change_on_an_uncommitted_axis`.
- `test_stage_reversals_blocks_until_recorded_and_surfaced` — outstanding
  reversal → blocks; after `set_plan(note=…)` only → still blocks (not
  surfaced); after `wrap_up` too → passes; bounded by `_REVERSAL_MAX_BLOCKS`.
- `test_set_plan_preserves_recorded_reversals`.

**Session-level:** `test_session_silent_unmute_cannot_finish` — script
`set_plan(watch=["slides carry muted incidental talk -- keep muted"])` →
`place` → `set_audio(mute=False)` → `review` → prose finish. Assert (a) a
`kind:"injected"` trace entry with `source="reversal"` on the unmute turn,
(b) the `kind:"gate"` entry shows `reversals fired=True` **despite the `review`
call**, (c) the loop does not finish until the scripted `set_plan`+`wrap_up`
land, and (d) the final reply names the reversal.

---

## PART 3 — SEVERITY MODEL FOR SIGNALS

**Problem.** Every signal is advisory ("all warnings, no errors"), so nothing
forces action — 4 of 7 cuts carried `broken-line` and shipped. And the length
contract is checked only in one direction (§0).

**Change.** Signals carry SEVERITY; a blocking class (or a density threshold)
blocks finishing until fixed or explicitly justified with the existing
two-reasons discipline. Generic across all flags.

### 3.1 A machine `kind` on every signal (replacing prose matching)

`observe.py`: every finding/flag emitted by `diagnose` (`observe.py:1202-1250`),
`review` (`observe.py:1491-1566`) and the mirror flag set (`observe.py:1659-1671`)
gains a stable `kind` token beside its prose `message`. Additive — the existing
`severity`/`anchor`/`message`/`category` keys are untouched.

```python
# observe.py -- stable machine tokens for signals. Prose is for the brain;
# `kind` is for policy, so no consumer ever string-matches a message again
# (tools.py:1026 / :1069 did, and that is how the under-target half of the
# length contract got silently dropped).
KIND_SAME_SPEAKER   = "same-speaker-run"
KIND_JUMP_CUT       = "jump-cut"
KIND_LOW_ENERGY     = "low-energy-run"
KIND_LENGTH_OVER    = "length-over"
KIND_LENGTH_UNDER   = "length-under"
KIND_REDUNDANT_TAKE = "redundant-take"
KIND_ASK_MISSING    = "ask-missing"
KIND_BROKEN_LINE    = "broken-line"
KIND_INCIDENTAL     = "incidental"
KIND_DEAD_AIR       = "dead-air"
KIND_AUDIO_GAP      = "audio-gap"
KIND_LOUDNESS_OFF   = "loudness-off-median"
KIND_OVERLAY_FIT    = "overlay-fit"
KIND_ROUGH_EDGE     = "rough-head-tail"
```

New unified producer:
```python
def signals(document: dict, ctx: EditContext, *, user_ask: str = "") -> List[dict]:
    """The ONE signal set: diagnose findings + review flags + the mirror's
    per-cut flags, each normalized to {kind, severity, category, anchor,
    message, subject?, context?} -- `context` carrying the facts a policy needs
    (e.g. broken-line: {"audible": True} from the mirror's speech.muted, so a
    broken line on a MUTED b-roll cut is never escalated). Pure, fail-open."""
```

### 3.2 The declared severity POLICY (generic; density is a rule, not a special case)

```python
# observe.py -- SEVERITY is now a decision, not decoration.
SEV_INFO, SEV_WARN, SEV_BLOCKING = "info", "warn", "blocking"
_SEV_RANK = {SEV_INFO: 0, SEV_WARN: 1, SEV_BLOCKING: 2}

# The policy, as DATA. Two rule forms, both generic:
#   count=1  -> this kind is blocking on its own (a STATED contract was missed).
#   count=N  -> this kind is blocking at DENSITY N or more among signals that
#               satisfy `qualifies` (one is a craft call; several is a defect).
_ESCALATE = (
    # A constraint the USER stated, checked SYMMETRICALLY: missing the ask's
    # length in either direction is the same failure -- "check the result against
    # the contract" -- not two separate rules.
    _Escalation(KIND_LENGTH_OVER,  count=1),
    _Escalation(KIND_LENGTH_UNDER, count=1),
    _Escalation(KIND_ASK_MISSING,  count=1),
    # Density: several broken lines on AUDIBLE speech cuts is a shredded voice
    # track, whatever the material.
    _Escalation(KIND_BROKEN_LINE, count=2,
                qualifies=lambda s: bool((s.get("context") or {}).get("audible"))),
    _Escalation(KIND_JUMP_CUT, count=3),
)


def escalate(sigs: List[dict]) -> List[dict]:
    """Apply _ESCALATE: return `sigs` with severity raised to SEV_BLOCKING where
    the policy says so. Pure; unknown kinds pass through untouched (fail-open)."""
```

### 3.3 The symmetric length contract (folded in, not a special case)

`observe.diagnose` already computes both halves:
```1226:1235:backend/app/services/l3/observe.py
    # Target length (from the brief) vs current.
    target_s = (document.get("brief") or {}).get("target_duration_s")
    if target_s:
        cur = report.total_ms / 1000.0
        if cur > target_s * 1.15:
            findings.append({"severity": "warn", "anchor": "whole",
                             "message": f"over target: {cur:.1f}s vs {target_s:.1f}s"})
        elif cur < target_s * 0.7:
            findings.append({"severity": "info", "anchor": "whole",
                             "message": f"under target: {cur:.1f}s vs {target_s:.1f}s"})
```
Changes:
1. tag them `kind=KIND_LENGTH_OVER` / `KIND_LENGTH_UNDER`;
2. attach the **headroom fact** to the under-target finding, so it reads as a
   contract miss with an actionable cause rather than a taste note:
   ```python
           untouched = sorted(set(ctx.file_ids) - {s.get("file_id")
                                                   for s in timeline})
           unplaced = [r for r in ctx.meta_by_ref
                       if r not in {s.get("ref") for s in timeline}]
           findings.append({
               "severity": "warn", "anchor": "whole", "kind": KIND_LENGTH_UNDER,
               "context": {"untouched_files": len(untouched),
                           "unplaced_moments": len(unplaced)},
               "message": (f"under target: {cur:.1f}s vs {target_s:.1f}s"
                           + (f" -- {len(untouched)} source file(s) entirely "
                              f"untouched, {len(unplaced)} moment(s) unplaced"
                              if untouched or unplaced else ""))})
   ```
3. `_stage_length` (PART 1 §1.2) stops filtering `"over target"` and consumes
   both kinds via `kind`, with symmetric wording:
   ```python
   def _stage_length(gc: _GateCtx) -> str | None:
       """PART 3: the ask's LENGTH is a stated constraint, checked SYMMETRICALLY.
       Over and under are the same failure -- the result doesn't match the
       contract -- so both fix-or-justify here. (The old substring match on
       'over target' silently discarded the under-target half, which is how a
       34s result against a 60s ask shipped with a whole source file unused.)"""
       off = [s for s in gc.signals
              if s.get("kind") in (observe.KIND_LENGTH_OVER, observe.KIND_LENGTH_UNDER)]
       if not off or gc.state["length_surfaced"]:
           return None
       gc.state["length_surfaced"] = True
       return ("AUTOMATIC CHECK -- length: " + (off[0].get("message") or "") +
               ". Either bring it to the target (trim what drags, or use the "
               "material you haven't touched), or say in one line why THIS length "
               "is right for the ask, then finish.")
   ```
`_GateCtx` gains `signals: List[dict]` (`observe.escalate(observe.signals(...))`,
computed once in `_gate_ctx`); `findings`/`review_out` stay for the stages that
still read them.

### 3.4 The new MANDATORY blocking-signal stage

```python
_SIGNAL_MAX_BLOCKS = 2     # this stage GATES -> its own bound (PART 1 §1.5)


def _stage_blocking_signals(gc: _GateCtx) -> str | None:
    """PART 3: a BLOCKING-severity signal cannot ship unaddressed.

    "Or explicitly justified" is enforced structurally, never by parsing prose
    (house discipline, tools.py:996-1001): this stage re-blocks while blocking
    signals remain, bounded by _SIGNAL_MAX_BLOCKS -- and because the surface
    stage is MANDATORY and downstream, choosing to justify instead of fix
    necessarily lands the reason in the user-facing surface. Fixing clears it
    (signals are recomputed from the live document each evaluation)."""
    blocking = [s for s in gc.signals if s.get("severity") == observe.SEV_BLOCKING]
    if not blocking or gc.state["signals_blocked"] >= _SIGNAL_MAX_BLOCKS:
        return None
    gc.state["signals_blocked"] += 1
    body = "\n".join(f"- [{s['kind']}] " + (f"{s['anchor']}: " if s.get("anchor") else "")
                     + (s.get("message") or "") for s in blocking[:12])
    return ("AUTOMATIC CHECK -- BLOCKING signals: these are not taste calls, "
            "they are defects or a missed stated constraint:\n" + body +
            "\nFix them, or -- if one genuinely can't be fixed -- name in ONE "
            "line which of exactly two reasons applies (the ask required it, or "
            "the material can't support better) and surface it to the user with "
            "wrap_up. 'It reads fine' is not one of the two.")
```
Registered right after `craft` (before `reversals`), with
`"signals_blocked": 0` added to `verify` / `_gate_state()`.

### 3.5 Fold-in: verbs validate against their DOMAIN **and** against the ask's stated constraints

Two additions to the `act.EditReject` layer, framed as one rule.

**(a) Domain — a trim past the moment's NATURAL BOUNDARY is a reject.** A placed
cut's moment carries its own grounded span (`ctx.meta_by_ref[ref]["in_ms"/"out_ms"]`
— the same values `act.retime` reads at `act.py:1020`), and
`observe._seams_for_file` (`observe.py:377-392`) gives the clean edit points. A
trim whose resulting source edge lands past the moment's own boundary AND past
the nearest clean seam is out of domain — it silently ballooned a cut by 30s.
New guard beside `_trim_domain_reason` (`tools.py:497`), same shape, same
fail-open contract:
```python
def _trim_boundary_reason(doc, ctx, target_id, in_ms, out_ms, delta_in_ms, delta_out_ms):
    """PART 3 fold-in: reject a trim whose RESULTING source edge lands past the
    MOMENT's own natural boundary (ctx.meta_by_ref[ref]["in_ms"/"out_ms"] -- the
    grounded span the segmenter found, the same one act.retime re-derives from)
    and past the nearest clean cut boundary (observe._seams_for_file). Generic:
    it fires for any over-extension, on any clip -- a trim may TIGHTEN inside a
    moment freely, it may not invent footage outside it. Fail-open: no ref / no
    map metadata / no seam points -> None (the existing source-length guard and
    the verb's own validation still run)."""
```
wired directly after the existing guard (`tools.py:604-607`):
```python
            trim_reason = (_trim_domain_reason(doc, ctx, args["target_id"],
                                              tr_in, tr_out, tr_din, tr_dout)
                           or _trim_boundary_reason(doc, ctx, args["target_id"],
                                                    tr_in, tr_out, tr_din, tr_dout))
            if trim_reason:
                return _json({"applied": False, "reason": trim_reason}), doc, False
```

**(b) The ask's stated constraints — a same-turn NOTICE, not a reject.** In
`_dispatch`'s shared tail (`tools.py:731-746`), after the verb applied:
```python
        changed = new is not doc
        result = {"applied": changed, "state": observe.read_state(new, ctx)}
        ...
        # PART 3 fold-in: a verb is answerable to the ASK's stated constraints,
        # not only to its own domain. If this edit just pushed the program across
        # the user's stated length, say so THIS turn (the brain paid 2 of 4
        # remaining turns discovering it later). A NOTICE, not a reject: a
        # transient over-length mid-build is legitimate, and rejecting a legal
        # edit would change behavior and could deadlock a valid intermediate
        # state. The finish-time length stage (§3.3, now blocking + symmetric)
        # is what actually gates it.
        breach = _constraint_notice(doc, new, ctx)
        if breach:
            result["constraint"] = breach
```
`_constraint_notice(before, after, ctx)` returns
`{"kind": "length-over", "target_s": …, "was_ms": …, "now_ms": …}` only on a
crossing (within-cap → over-cap), so it never nags on an edit that was already
over. Generic in shape: one function per stated constraint in
`document["brief"]`; today `target_duration_s` is the only one populated
(`converse.py:683-685`).

### Back-compat / fail-open (PART 3)

`kind`/`context` are additive keys; `severity` values `"info"`/`"warn"` still
appear and Stage 3's renderer already prints whatever is there
(`tools.py:1074`). `escalate` passes unknown kinds through. `_ESCALATE` is data
— tuning a threshold is a one-line data edit with a test, not a code change.
Both verb-layer additions are fail-open (missing map metadata / missing brief →
no guard, no notice).

### Testing (PART 3)

**Unit:**
- `test_signals_carry_stable_kinds` — every emitted signal has a `kind` in the
  declared set; no consumer string-matches a message (grep-style assertion on
  `tools.py` for `"over target" in`).
- `test_escalate_blocks_on_stated_constraint_either_direction` — one
  `length-over` alone → blocking; one `length-under` alone → blocking.
- `test_escalate_density_rule_for_broken_lines` — 1 audible broken-line → warn;
  2 → blocking; 3 broken-lines on MUTED cuts → still warn (the `qualifies`
  predicate).
- `test_under_target_names_untouched_material` — a 34s edit against a 60s target
  with one of two source files unused → message names the untouched file and the
  unplaced moment count.
- `test_stage_blocking_signals_blocks_then_bounds` — blocks twice, then passes
  (`_SIGNAL_MAX_BLOCKS`), and passes immediately once the signals are fixed.
- `test_trim_past_moment_boundary_rejects` / `test_trim_inside_moment_is_allowed`
  / `test_trim_boundary_guard_fail_open` (no map metadata → None).
- `test_dispatch_attaches_constraint_notice_on_crossing_the_stated_length` and
  `…_not_when_already_over`.

**Session-level:**
- `test_session_broken_line_density_cannot_finish` — script places 4 audible
  cuts that trip `broken-line`, then `review`, then a prose finish → trace's
  `kind:"gate"` shows `blocking_signals fired=True`; the loop does not finish
  until the scripted fix or the scripted `wrap_up` justification lands.
- `test_session_under_target_forces_a_length_account` — a 34s build against a
  stated 60s ask → the length stage fires (the exact case that shipped silently).

---

## PART 4 — DEFINE "PROGRESS" PROPERLY

**Problem.** `changed` conflates ANY document write with the edit moving, and
churn compares a fuzzy content fingerprint. Result: 2 false "you're cycling"
advisories (once with a completely empty timeline) and 1 MISSED genuine revert
(the undo landed 100ms off the natural boundary, so `_edit_fingerprint` hashes
differed).

**Change.** Define progress from the brain's own OP HISTORY / intent, and make
the churn signal depend on the edit content actually moving. **NO arbitrary
tolerance/epsilon** — this subsumes the churn false-fire bug; it is not spec'd
separately.

### 4.1 Split `changed` into `doc_changed` and `edit_moved`

Classification is declarative by verb, not by diffing documents:
```python
# tools.py -- PART 4. The verbs that write the document SURFACE (the plan of
# record, the user-facing wrap-up) but touch no timeline content. An applied
# call to one of these is a DOCUMENT change (it must persist) and is NOT the
# edit moving (it must not drive the done-gate, the churn signal, the plan
# mirror's `building` loudness, or the finalizer).
_SURFACE_VERBS = frozenset({"set_plan", "wrap_up"})
```
Before (`tools.py:1262`):
```1261:1263:backend/app/services/l3/tools.py
            obs, working, did = _dispatch(tc.name, tc.input or {}, ctx, working, user_ask)
            changed = changed or did
            results.append(tool_result_block(tc.id, obs))
```
After:
```python
            obs, working, did = _dispatch(tc.name, tc.input or {}, ctx, working, user_ask)
            doc_changed = doc_changed or did
            if did and tc.name not in _SURFACE_VERBS:
                edit_moved = True
                ops.extend(_op_intents(tc.name, tc.input or {}, before_call, working, ctx))
            results.append(tool_result_block(tc.id, obs))
```
Consumers switch to `edit_moved`: the gate call (`tools.py:1241`),
`plan_mirror_text(building=…)` (`tools.py:1307`), the churn block
(`tools.py:1273`), `_finalize_turn` (PART 1 §1.4).
`LoopResult.changed` keeps meaning **`doc_changed`**, so
`converse.respond`'s persistence (`converse.py:698`) and
`routers/edit_threads.py:134` are untouched — a plan-only turn still saves a
version, it just no longer counts as an edit.

### 4.2 The OP HISTORY — progress and churn as SYMBOLIC facts (no epsilon)

```python
def _op_intents(name: str, args: dict, before: dict, after: dict,
                ctx: EditContext) -> List[tuple]:
    """This applied verb's INTENT IDENTITIES -- what it did, to what, on which
    axis -- resolved off the POST-state document so a freshly minted seg_id maps
    back to its STABLE map ref. That resolution is why no tolerance is needed
    anywhere here: the history is symbolic, so

        place(refA) -> remove(that seg) -> place(refA)

    reads exactly ("present", refA) / ("absent", refA) / ("present", refA)
    whether or not the re-place lands on the same millisecond. The 100ms-off
    revert that _edit_fingerprint missed is caught for free, and there is no
    epsilon to tune (an epsilon was explicitly rejected as a band-aid).

    Shapes (one per axis, mirroring commit._AXES so PART 2 and PART 4 agree):
        place / place_span   -> ("present", ref_or_file, level, piece, channel)
        remove               -> ("absent",  ref_or_id)
        move (seg)           -> ("order",   ref_or_id, to_index)
        set_audio            -> ("mute",    ref_or_id, bool)
        retime / tighten     -> ("pace",    ref_or_id, level_or_pace)
        trim                 -> ("span",    ref_or_id, new_in_ms, new_out_ms)
        <every other verb>   -> ("<verb>",  target_or_"whole")
    Pure; unknown/unresolvable -> [] (fail-open: no history entry, no signal)."""
```

**PROGRESS** = the turn's op history contains at least one intent not undone
later in the same turn. **CHURN** = for one `(axis, subject)`, the history reads
`A → ¬A → A` (or `A → B → A`) — an inverse pair followed by a re-assertion:
```python
def _churn_subjects(ops: List[tuple]) -> List[tuple]:
    """The (axis, subject) pairs whose intent sequence returned to a value it
    already held after leaving it -- the exact revert / swap-back / place-remove-
    place pattern, read off the OP HISTORY rather than off a content hash.
    Deduped per subject, so the same cycle is nagged once. Pure."""
```
The churn block (`tools.py:1271-1292`) becomes:
```python
        # PART 4: churn is read off the OP HISTORY, not a content fingerprint.
        # Two consequences that fix the observed misfires: a turn with no edit
        # ops (a set_plan/wrap_up turn, or an empty timeline) can produce no
        # cycle at all -- so the two false "you're cycling" advisories cannot
        # recur -- and a revert is matched on the STABLE subject, so an undo that
        # lands off the original millisecond is still a revert.
        churn_hint = None
        try:
            fresh = [s for s in _churn_subjects(ops) if s not in churned_subjects]
            if fresh:
                churned_subjects.update(fresh)
                churn_hint = _churn_note(fresh)     # names the axis + subject
        except Exception:
            logger.exception("tools: churn detection failed (continuing)")
```
Init before the loop replaces `seen_states`/`churned_states` (`tools.py:1210-1211`):
```python
    doc_changed = False
    edit_moved = False
    ops: List[tuple] = []            # this turn's applied EDIT intents, in order
    churned_subjects: set = set()
```
`_edit_fingerprint` (`tools.py:437-458`) is **retired as the churn source**. Keep
the function (it is pure and cheap) only if a corroborating "the whole program is
byte-identical to an earlier state" note is wanted; if kept, it must be labelled
as corroboration, never as the trigger. `_progress_note`'s unused `churn`
parameter (`tools.py:1168`) is removed at the same time.

### Back-compat / fail-open (PART 4)

`LoopResult.changed` semantics unchanged for every external consumer.
`_op_intents` is pure and returns `[]` on anything it can't resolve, so an
unrecognized verb simply contributes no history. The churn signal remains
ADVISORY (a text block), never a cap — the standing rule from
`brain_loop_convergence.plan.md`.

### Testing (PART 4)

**Unit:**
- `test_set_plan_only_turn_does_not_count_as_an_edit` — `edit_moved` False,
  `LoopResult.changed` True; the gate is not called; the plan mirror is not
  `building`-loud.
- `test_op_intents_resolve_a_fresh_seg_id_back_to_its_ref`.
- `test_churn_detected_when_a_revert_lands_off_the_original_ms` — place → remove
  → re-place with a 100ms-different span → churn fires (the MISSED case), with
  no tolerance parameter anywhere in the call.
- `test_no_churn_on_an_empty_timeline_or_a_surface_only_turn` — the two FALSE
  fires, as regression tests.
- `test_no_churn_on_a_straight_line_build` (place A/B/C).
- `test_churn_dedups_per_subject`.

**Session-level:**
- `test_session_plan_then_build_gate_runs_only_after_the_edit_moves` — script
  `set_plan` → prose finish (gate NOT called, no forced wrap_up), then a second
  scripted run with `set_plan` → `place` → prose finish (gate called).
- `test_session_revert_cycle_emits_one_churn_injection` — assert exactly one
  `kind:"injected"` `source="churn"` trace entry across the cycle.

---

## PART 5 — COMPLETE TRACE

**Problem.** The PROGRESS/CHURN advisories, the mirrors, the gate feedback and
the finalize instruction are all injected into the brain's context and none is
persisted; diagnosing the last run required replaying the state machine from
source.

**Change.** An observability INVARIANT: anything injected into the brain's
context is recorded in the trace. Enforced by making a funnel the only way to
inject.

### 5.1 The funnel — `tools.py`, beside the reasoning-capture helpers

```python
_INJECTED_CAP = 4000     # bounds one jsonb row; generous vs. a mirror/gate block


def _inject(blocks: list, trace: list, *, turn: int, source: str, text: str) -> None:
    """THE ONLY way text is injected into the brain's context as a block.
    Records a kind="injected" trace entry for every push -- the observability
    invariant: if the brain saw it, the trace has it. `source` is a stable token:
    "mirror" | "plan_mirror" | "progress" | "churn" | "reversal" | "constraint".
    Additive + fail-open: a recording error must never drop the injection."""
    if not text:
        return
    blocks.append(text_block(text))
    try:
        trace.append({"turn": turn, "kind": "injected", "source": source,
                      "text": text[:_INJECTED_CAP]})
    except Exception:
        logger.exception("tools: injected-trace record failed (continuing)")


def _inject_message(convo: list, trace: list, *, turn: int, source: str, text: str) -> None:
    """Same invariant for a standalone user-role injection -- the done-gate's
    feedback ("gate") and the forced-finalization instruction ("finalize"),
    which were the two least visible pushes in the loop."""
    convo.append(user_message(text))
    ...same recording...


def _record_gate(trace: list, *, exit_reason: str, gc, note: str = "") -> None:
    """The ladder's own outcome, once per gate evaluation:
    {kind:"gate", exit_reason, stages:[{stage, kind, fired, why?}], note?}.
    This is what makes "the mandatory invariant ACTUALLY fired" assertable in a
    session test and diagnosable from a stored thread -- the missing evidence
    that let two dead-code bugs ship 'verified'."""
```

### 5.2 Rewire every injection site

Before (`tools.py:1300-1316`, four unrecorded pushes):
```1300:1316:backend/app/services/l3/tools.py
        mirror_text = observe.mirror_text(working, ctx)
        if mirror_text:
            results.append(text_block(mirror_text))
        plan_text = observe.plan_mirror_text(working, building=changed)
        if plan_text:
            results.append(text_block(plan_text))
        ...
        results.append(text_block(_progress_note(turn, max_turns, churn=churn_now)))
        if churn_hint:
            results.append(text_block(churn_hint))
```
After:
```python
        _inject(results, trace, turn=turn, source="mirror",
                text=observe.mirror_text(working, ctx))
        _inject(results, trace, turn=turn, source="plan_mirror",
                text=observe.plan_mirror_text(working, building=edit_moved))
        _inject(results, trace, turn=turn, source="progress",
                text=_progress_note(turn, max_turns))
        _inject(results, trace, turn=turn, source="churn", text=churn_hint or "")
```
Plus: gate feedback via `_inject_message(..., source="gate")` (PART 1 §1.5), the
finalize instruction via `_inject_message(..., source="finalize")`
(`tools.py:1149`), reversal notices via `_inject(..., source="reversal")`
(PART 2 §2.2), and `_record_gate` at every `_run_gate_stages` call.

**The invariant is a test, not a convention:**
`test_no_raw_text_block_injection_in_the_loop` greps the `run_edit_loop` /
`_finalize_after_loop` source for `text_block(` and `convo.append(user_message(`
outside the funnel helpers and fails if any remain. That is what stops the next
injection from being invisible.

### Back-compat / fail-open (PART 5)

`edit_turns.trace` is jsonb and readers filter by `kind`
(`test_reasoning_capture.py:67-68`), so `kind:"injected"` / `kind:"gate"` are
purely additive — **no migration**. Recording is try/excepted, so a trace error
never alters the edit, the injection, or the reply. Row growth is bounded by
`_INJECTED_CAP` per entry.

### Testing (PART 5)

**Unit:**
- `test_inject_records_every_push_and_returns_the_block`.
- `test_inject_skips_empty_text_and_records_nothing`.
- `test_inject_is_fail_open_when_recording_raises` (edit + reply intact).
- `test_no_raw_text_block_injection_in_the_loop` (the invariant guard above).

**Session-level:**
- `test_session_trace_contains_every_injection_in_loop_order` — a scripted 3-turn
  run: the trace's ordered `kind` sequence interleaves
  `reasoning`/`tool`/`injected`/`gate` in true loop order, and the `injected`
  `source` set includes `mirror`, `plan_mirror`, `progress`.
- `test_session_gate_entry_proves_mandatory_stages_ran` — the assertion every
  other part's session test reuses.

---

## Testing lesson — MANDATORY for every part

Our previous unit tests proved the gate **works when called** and never that it
**gets called in a realistic session** — which is exactly how two dead-code bugs
shipped "verified". Therefore, per part:

1. **Normal unit tests** (pure helpers, one stage at a time) — as listed.
2. **At least one END-TO-END SESSION-LEVEL test** that drives
   `tools.run_edit_loop` with `_ScriptedLLM` and asserts the mandatory
   invariants **ACTUALLY FIRED**, read off `res.trace`'s `kind:"gate"` entry
   (PART 5) — never inferred from a helper's return value.
3. The session suite must cover all three previously-bypassing shapes:
   - a turn that calls `review`/`validate`/`diagnose` (Bypass 2 + 2b),
   - a cap-exhaustion run (`max_turns=3`, a script that always calls tools) —
     Bypass 1,
   - a last-turn voluntary finish (the scripted prose finish arrives exactly at
     `turn == max_turns - 1`) — Bypass 3.

New session tests live in a new file `backend/scripts/test_loop_sessions.py`
(the gate/dispatch unit tests stay in `test_tools_loop.py`), reusing
`_ScriptedLLM` / `_ctx` / `_struct` / `_seed_doc`. Register every test in each
file's `main()`.

---

## Exact change list per file

| File | Change |
|---|---|
| `backend/app/services/l3/tools.py` | **PART 1:** add `_Stage`/`_GateCtx`/`_gate_ctx`, the `_stage_*` extractions, `_GATE_STAGES`, `_run_gate_stages`; reduce `_verify_before_finish` (`:978-1100`) to a wrapper; add `_finalize_turn`; delete `finished_clean` (`:1204`, `:1246`, `:1329`) and add `exit_reason`; drop `turn < max_turns - 1` from the gate CALL (keep it on the `continue`); replace the guarded `_finalize_after_loop` call with the unconditional `_finalize_turn`; give `_finalize_after_loop` `outstanding`/`trace` params and drop its own `_plan_conformance` call (`:1138`). **PART 2:** `_stage_reversals` + registry entry + `verify` keys; reversal detection at the post-dispatch site (`:1267`). **PART 3:** `_stage_blocking_signals` + registry entry + `verify` key; `_stage_length` consumes `kind`; `_trim_boundary_reason` + wiring (`:604`); `_constraint_notice` in the `_dispatch` tail (`:731-746`). **PART 4:** `_SURFACE_VERBS`, `doc_changed`/`edit_moved`/`ops`, `_op_intents`, `_churn_subjects`, `_churn_note`; retire `_edit_fingerprint`/`seen_states` as the churn source; drop `_progress_note`'s unused `churn` param (`:1168`). **PART 5:** `_inject`/`_inject_message`/`_record_gate` + rewire all injection sites (`:1244`, `:1149`, `:1300-1316`). |
| `backend/app/services/l3/commit.py` | **NEW (PART 2):** `Commitment`, `_AXES`, `ledger`, `reversals`, `render_reversal`. |
| `backend/app/services/l3/observe.py` | **PART 3:** `KIND_*` tokens, `SEV_*`, `_ESCALATE`, `signals`, `escalate`; tag `diagnose` (`:1202-1250`) / `review` (`:1491-1566`) / mirror-derived flags with `kind`; symmetric length finding + `context` headroom facts (`:1226-1235`). **PART 2:** render `plan["reversals"]` in `plan_mirror_text` (`:1769-1772`). |
| `backend/app/services/l3/act.py` | **PART 2:** `record_reversal`; `set_plan` preserves `plan["reversals"]` across its full replace (`:725-744`). |
| `backend/app/services/l3/converse.py` | **PART 2:** one sentence in `_LOOP_SYSTEM`'s "THE PLAN" block (`:207-218`) on standing commitments + explicit reversal. **PART 3:** one clause in FINISHING (`:219-246`) that a BLOCKING signal and the ask's length (either direction) are fix-or-name-a-reason, not advisory. No behavior change. |
| `backend/scripts/test_tools_loop.py` | Unit tests per part; `_gate_state()` (`:459-462`) gains `reversals`/`reversal_blocked`/`signals_blocked`. |
| `backend/scripts/test_commitments.py` | **NEW (PART 2):** ledger/reversal unit tests. |
| `backend/scripts/test_loop_sessions.py` | **NEW:** the session-level suite (all parts). |
| migrations | **NONE.** `document` and `edit_turns.trace` are both jsonb; every new field is additive inside them. If an implementer concludes otherwise, STOP and state why before adding one. |

---

## Summary of decisions

- **PART 1 (lands first, alone).** The done-gate stops being a chain of nested
  early-returns and becomes `_GATE_STAGES`: a declared, ordered tuple of stages
  each tagged `MANDATORY` or `ADVISORY`, evaluated by `_run_gate_stages` with
  explicit semantics (advisory-skip filters advisory stages ONLY; a stage
  returns only its own feedback; a stage that raises fails open without
  suppressing later stages; interactive mode returns the first feedback,
  enforcement mode collects all). `_finalize_turn` becomes the single choke
  point every termination path reaches, and `finished_clean` is deleted. That
  closes Bypass 1 (finalizer outside the loop, exit-path-independent), Bypass 2
  and 2b (the `reviewed` short-circuit becomes a filter that cannot reach
  `intent`/`conformance`/`surface`; `review()` is computed unconditionally), and
  Bypass 3 (no boolean left to lie; the gate runs on the last turn too) — all by
  construction, plus a registry test that fails if a future stage is
  unregistered or an invariant is demoted.
- **PART 2.** A standing commitment is a `(scope, axis, subject, value, origin)`
  record drawn from the plan (required beats, carries, watch), from perception
  (`ctx.meta_by_ref`'s `mute`/`flags`/`audio`, take groups) and from the ask
  (stated length). A contradiction is detected as a DOCUMENT DIFF on declared
  axes — never from verb arguments — so mutes, take swaps, reorders, pace
  changes and dropped beats are all one mechanism. The brain is told that turn,
  and to finish it must record the reversal on the plan (`set_plan` bumps `rev`
  + `updated_note`; acknowledged reversals persist in `plan["reversals"]`) AND
  surface it in `wrap_up` — checked structurally, never by parsing prose.
- **PART 3.** Signals get a machine `kind` and a real `severity` driven by a
  declared policy `_ESCALATE` with two generic rule forms: blocking-on-its-own
  (a STATED constraint was missed) and blocking-at-density-N among qualifying
  signals (several audible `broken-line`s). A new mandatory
  `_stage_blocking_signals` gates on `SEV_BLOCKING`; "explicitly justified" is
  enforced by the downstream mandatory surface stage, not by parsing. The
  symmetric length contract folds in as the same "check the result against the
  contract" rule: `length-over` and `length-under` are one stage keyed on
  `kind`, and under-target carries the headroom fact (untouched source files,
  unplaced moments). The `EditReject` layer gains a moment-natural-boundary
  domain guard for `trim` (a reject) and a stated-constraint crossing notice in
  the dispatch tail (a notice, deliberately not a reject — rejecting a legal
  transient edit would change behavior).
- **PART 4.** `changed` splits into `doc_changed` (persistence, unchanged
  externally) and `edit_moved` (the timeline content moved), classified by a
  declared `_SURFACE_VERBS` set. Progress and churn are read off a symbolic OP
  HISTORY whose subjects are stable map refs resolved from the post-state
  document — so `place X → remove X → place X` is exact and **needs no epsilon**;
  the 100ms-off revert is caught, and a surface-only or empty-timeline turn can
  produce no cycle at all, which subsumes both false fires. `_edit_fingerprint`
  is retired as the churn trigger.
- **PART 5.** One funnel (`_inject`/`_inject_message`/`_record_gate`) is the only
  way anything enters the brain's context, each push recorded as
  `kind:"injected"` (with a stable `source`) or `kind:"gate"` (the ladder's own
  per-stage outcome) alongside the existing `kind:"reasoning"`/`kind:"tool"`
  entries — plus a source-grep test so the next injection cannot be invisible.
- **Testing.** Per part: normal unit tests PLUS at least one session-level test
  driving a realistic `run_edit_loop` and asserting from the trace that the
  mandatory invariants ACTUALLY FIRED — covering a self-reviewing turn, a
  cap-exhaustion run, and a last-turn finish, the three shapes that previously
  bypassed everything.
- **Additive, fail-open, back-compatible, no migration** throughout;
  `_verify_before_finish`'s signature, the `state` keys and every feedback
  string are preserved so the existing gate suite passes unchanged.
