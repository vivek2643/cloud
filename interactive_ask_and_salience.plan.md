# Interactive Ask/Suggest + Salience-on-Beat-Line — implementation plan

Status: ready to implement (from another chat). Branch base: `main`
(carries the perception upgrade + Flash-Lite Pass 2 + fused-reading prompt).

Two INDEPENDENT workstreams, orderable in either sequence:
- **WS1 — Interactive ask/suggest/get-it-done:** let the brain PROPOSE + SUGGEST
  (with a recommended option and a one-line why) when a choice is genuinely the
  user's, then act on the answer. An enrichment of the existing `ask_user`
  pause, NOT a new confirm-everything round-trip.
- **WS2 — Salience on the beat line:** surface the code-computed
  `salience.peak_ms` on each rendered beat so the brain can read where a cut
  peaks (emphasis / punch-in timing), grounding the guidance that mentions it.

Guiding principles (unchanged, enforce throughout):
- **Code owns numbers/structure; the LLM owns categories/text.** No model-
  emitted scores or ms values.
- **Fail OPEN.** Every path degrades to a plain reply / no change — a turn
  never hard-fails. WS1 must preserve this (a malformed proposal → plain chat).
- **Minimal prompt, specifics in data.** Keep `_LOOP_SYSTEM` lean; the
  interaction contract lives in the tool schema + guidance, not a prose essay.
- **Don't over-ask.** The default is guess-and-go (the fused-reading prompt).
  Asking/suggesting is the EXCEPTION, for genuinely user-owned or materially
  ambiguous choices only.

---

## WS1 — Interactive ask / suggest / get-it-done

### Current state (verified — anchor the change here)
The plumbing already pauses-and-asks; WS1 enriches it, it does not rebuild it.

- `tools.py`
  - `_specs()` declares the `ask_user` tool (~line 230): `questions[]`, each
    `{prompt, options[], allow_multiple}`. Calling it ENDS the turn.
  - `_normalize_questions()` (~line 278) coerces the payload (drops questions
    with <2 options).
  - `run_edit_loop()` (~line 440): on an `ask_user` call it collects questions,
    sets `asked=True`, appends a tool-result telling the model to end its turn,
    and **breaks** the loop. Returns `LoopResult.questions` + `awaiting_user`.
- `converse.py`: `ConverseResult` carries `questions` / `awaiting_user` / `trace`
  straight through.
- `edit_threads.py` (POST turn, ~line 110): returns `{reply, changed,
  document_version, awaiting_user, questions}` and sets thread status
  `awaiting_user`/`ready`. **The answer is just the user's next message** (no
  structured id round-trip).
- `frontend/src/lib/api.ts`: `ThreadQuestion { id, prompt, options: string[],
  allow_multiple? }`; `ThreadMessageResult { reply, changed, document_version,
  awaiting_user, questions }`.
- `frontend/src/components/ai-edit-panel.tsx`: `QuestionCard` (~line 718)
  renders `q.prompt` + each option as a button; picking one calls
  `handleSend(opt)` (sends the option TEXT as the next message). `allow_multiple`
  is NOT actually honored in the UI (each button sends immediately), and there's
  no recommended/why surfaced.

### Design decision (make explicit before coding)
Keep the elegant "answer = the user's next message" model — it's robust and
fails open. WS1 = make the ask RICHER (suggest, don't just ask) and make
multi-select real. Do NOT introduce a per-action confirm gate on ordinary edits
(edits still apply directly). "Confirm mid-edit" = for a big/ambiguous move the
brain proposes with a RECOMMENDED option; the user confirms or redirects; the
brain executes on the next turn (already works).

### WS1-A — Enrich the `ask_user` schema (backend)
File: `tools.py`
- Extend the `ask_user` spec: each question gains optional
  `recommended: string` (must be one of `options` — the brain's suggested
  pick) and `why: string` (one short line of rationale). Optionally
  `preview: string` per question — "what I'll do if you pick this" — to make it
  a real suggestion.
- `_normalize_questions()`: pass `recommended`/`why`/`preview` through when
  present and valid; drop `recommended` if it isn't among the kept `options`
  (never surface a dangling default). Keep the "≥2 options" rule.
- Nothing else in `run_edit_loop` changes — the enriched dicts already flow via
  `questions`.

### WS1-B — Router passthrough (backend, verify only)
File: `edit_threads.py`
- `result.questions` is returned verbatim (list of dicts) — the new fields flow
  automatically. **Confirm** no model/DTO strips unknown keys (it's a raw dict
  today; keep it that way or widen the response model).

### WS1-C — Frontend types + richer card
Files: `frontend/src/lib/api.ts`, `frontend/src/components/ai-edit-panel.tsx`
- `api.ts` `ThreadQuestion`: add `recommended?: string`, `why?: string`,
  `preview?: string`.
- `QuestionCard`:
  - Highlight the `recommended` option (e.g. accent border / "Recommended"
    pill) and render `why` as sub-text; render `preview` if present.
  - **Honor `allow_multiple`:** when true, options toggle a local selection set
    and a "Send" button submits the joined picks as one message; when false,
    keep click-to-send. Preserve the "…or type your own" affordance.
- No change to how the answer is sent (still `handleSend(text)`).

### WS1-D — Prompt guidance (minimal)
File: `converse.py` `_LOOP_SYSTEM`
- The `ask_user` sentence already exists ("If a choice is genuinely theirs and
  you can't reasonably settle it, use ask_user; otherwise proceed"). Add ONE
  clause: when you do ask, SUGGEST — include a `recommended` option and a
  one-line `why`; ask only for genuinely user-owned or materially ambiguous
  choices, never to offload a guess you could make. Keep it to a sentence — no
  format rules.

### WS1-E — Audit / persistence (verify)
File: `store.py`
- The turn `trace` (which includes the `ask_user` args) is already persisted on
  the assistant turn. Confirm the enriched payload lands in the trace so a
  proposal + the user's answer are replayable. No schema change expected.

### WS1-F — Verification
- Backend unit: `ask_user` with `recommended` not in `options` → dropped;
  valid `recommended`/`why`/`preview` → surfaced; `<2 options` → question
  dropped (unchanged).
- Frontend: `tsc` passes with the new optional fields; single-select still
  click-to-send; multi-select accumulates + submits once; recommended is
  visually distinct.
- End-to-end: a turn that calls `ask_user` pauses (`awaiting_user`), the picked
  option returns as the next message, and the following turn acts on it.

### WS1 — deferred (note, do NOT build now)
- Same-turn auto-resume (pick → execute without a manual round-trip).
- Structured answer mapping by option id (vs. the current text-as-message).

---

## WS2 — Salience on the beat line

### Current state (verified)
- `post._salience()` (~line 348) returns `{"peak_ms": <ABSOLUTE ms>, "score":
  0..1}`; falls back to `{peak_ms: hero_ts_ms, score: 0.0}` when no signal.
  Persisted to `cut_records.salience` (migration 038, already applied).
- `footage_map.py` `_moment` dict carries `"salience": cut.get("salience") or {}`
  (~line 337), but `_moment_line` (~line 747) does NOT render it.

### ⚠️ Scoring gotcha (read before designing the tag)
`score` is the peak's height normalized against the CUT'S OWN curve range, so
the argmax is ~always ≈ 1.0 for any cut with signal variation. **`score` is NOT
a cross-cut "how salient" magnitude** and must NOT be used as a render gate or
shown as a strength — it would read as "1.0" almost everywhere. The USEFUL
signal is `peak_ms` — WHERE in the cut the peak falls.

### WS2-A — Render the tag
File: `footage_map.py` `_moment_line`
- Compute the peak as an OFFSET into the cut: `peak_off = salience["peak_ms"] -
  m["in_ms"]`; render `peak:+X.Xs` next to `nrg`/`cam`.
- **Gate on POSITION, not score** (avoid per-line bloat + uninformative peaks):
  render only when the peak is meaningfully INTERIOR — i.e. more than ~1
  `hop_ms` (or a small fraction of the span) from BOTH the cut's start and end.
  A peak pinned to the first frame tells the brain nothing; skip it. Never
  render for `{}` (pre-migration) or `score == 0.0` (no-signal fallback ==
  `hero_ts_ms`, not a real peak).
- Deterministic, code-owned; reuse `_fmt_ts`-style formatting already in the
  file. Consider suppressing on `said`/speech beats if the peak there is noise
  (implementer's judgment during spot-check).

### WS2-B — Document the tag in the prompt
File: `converse.py` "READING A BEAT LINE"
- One clause: `peak:+Xs` = the cut's strongest INSTANT (offset into the cut),
  code-computed — use it for emphasis / punch-in / hold timing.

### WS2-C — Guidance (optional, one line)
File: `guidance_doc.md`
- At most one line (principle 1 picture-led paragraph or a new tiny note): when
  timing a punch-in or choosing where to hold, lean on the cut's `peak`.
  Keep minimal — skip if it reads as bloat.

### WS2-D — Verification
- Unit (`backend/scripts/test_footage_map*` or nearest): a cut with an interior
  `peak_ms` renders `peak:+X.Xs`; `{}` and `score==0.0` and edge-pinned peaks
  render nothing; offset math correct against `in_ms`.
- Smoke: re-render a beat index for an action/b-roll clip and confirm peaks read
  sensibly and don't bloat every line.

---

## File-touch summary
| Workstream | Files |
|---|---|
| WS1 | `tools.py` (schema + normalize), `edit_threads.py` (verify passthrough), `frontend/src/lib/api.ts`, `frontend/src/components/ai-edit-panel.tsx`, `converse.py` (1 prompt clause), `store.py` (verify trace) |
| WS2 | `footage_map.py` (`_moment_line`), `converse.py` (READING A BEAT LINE clause), `guidance_doc.md` (optional 1 line) |

No DB migration for either workstream (salience column already exists via 038;
WS1 adds no persisted columns). No re-ingest required for WS1; WS2 needs no
re-ingest either (it renders an already-persisted field).
