# EDSO: one think → act → check loop (work like the coding agent)

Make the editor behave like a single agent that **thinks through its whole
approach first, executes it, then checks the result against the ask** — the same
loop a good coding agent runs. This is NOT a rebuild: EDSO is already a single
agentic tool-loop (`tools.run_edit_loop`). It's missing two habits — *plan first*
and *audit before finishing* — plus a precedence rule that makes the guidance
doc a real ceiling. Add those, keep everything generic.

Motivating failures (both real, this week):
- A teaser hit the 12-turn cap mid-task and **never added the requested split
  screen** — because it discovered the edit by trial-and-error (`place×4 →
  remove×5 → place×2 → remove×6 → tighten`) instead of planning the approach up
  front. Classic "small steps, ends up nowhere."
- The **off-camera speaker** recurred even with §2 guidance, because guidance is
  currently a soft "lean toward," overridden under tactical pressure.

---

## Hard constraints (from the user — do not violate)
- **ONE loop, not two agents.** Do NOT build a separate planner LLM + executor
  LLM. A single model carries its own context through think → act → check. A
  handoff between two models just adds drift.
- **Prompt and guidance stay GENERIC.** NEVER write a procedure, step list, or
  format-specific recipe into the system prompt or the guidance doc (no "get the
  text → find beats → add music → balance levels", no per-format playbooks). The
  specific approach is **emergent per project** — the model generates it; we only
  tell it *to* plan, never *what* the plan is.
- **Precedence is a ceiling, enforced in TEXT at think-time, not by a
  deterministic check:** `user prompt > guidance doc > the model's own guessing`.
  The model's guesses may not go above the guidance ceiling unless the user
  overrides it. Accept "mostly works" — it's a floor-raiser, not a lock.
- **The audit PRESENTS and aligns, never DIRECTS.** It checks the finished edit
  against the ask (first) and the guidance (second) and surfaces divergences as
  facts; it never prescribes the specific fix.
- **Keep it simple.** No new agents, no new senses where existing ones serve.

---

## Change 1 — Think-first phase (the main gap)
**Where:** `_LOOP_SYSTEM` (`converse.py`).

Add a generic instruction that, on a turn that will change the edit, the model
**works out its whole approach before making moves** — reasoning from the user's
ask + the beat index about what the piece is for, roughly what to keep and in
what order, whose angle, where any requested features (split/PiP/music) go, and
the rough length — **then** executes. It should commit to that approach and act
on it, not discover the edit by placing-then-removing.

- Wording must be **capability + habit, never a recipe**: tell it *to* plan and
  *to* act from the plan; do NOT enumerate steps or name what to look for.
- The approach is the model's *reasoning* (first, before/with its opening
  senses) — a short plan, not an exhaustive script.
- Explicitly forbid, in our own heads (not in the prompt): no baked procedure,
  no format specifics.

**Acceptance:** on a changed-edit turn, the model states its approach up front in
reasoning; the prompt contains no step list or domain recipe.

---

## Change 2 — Precedence / guidance as a ceiling
**Where:** `guidance_doc.md` header + one line in `_LOOP_SYSTEM`.

Today the guidance header says everything is a *"lean toward", to be BLENDED or
OVERRIDDEN when the material or the user says otherwise* — i.e. soft. Raise it to
a binding default:

- **`guidance_doc.md` header:** reword from "lean toward / freely overridden" to
  *"these are binding defaults — follow them unless the USER's ask (or a clear
  material reality) calls for otherwise."*
- **`_LOOP_SYSTEM`:** one factual line stating the precedence —
  *"When you must guess, the order of authority is: the user's ask first, then
  the guidance defaults, then your own judgment. Don't let a guess override a
  guidance default unless the user asked for it."*

**Honest scope:** this is text, applied at think-time. It raises adherence
(the model reasons holistically with the ceiling salient) but does not guarantee
it. No deterministic enforcement (per user). If one specific rule keeps slipping,
revisit a targeted check later — not now.

**Acceptance:** guidance reads as binding-with-user-override; the prompt states
the precedence order; nothing domain-specific is added.

---

## Change 3 — Act phase: short, directed loop (kill the churn)
**Where:** `_LOOP_SYSTEM` (behavioral nudge; no new code required).

Once the approach is set, execute it directly — decide with `predict`/senses
*before* placing, place the intended selection, then adjust — rather than
over-placing and whittling down. This mostly falls out of Change 1; add at most a
one-line nudge against place-then-remove trial-and-error. No step recipe.

**Acceptance:** a normal edit uses noticeably fewer turns / less place-remove
churn than the current trace.

---

## Change 4 — Audit / check before finishing (the "check" habit)
**Where:** extend the LIVE done-gate `_verify_before_finish` (`tools.py:466`,
wired at `tools.py:530`) + the `review` sense (`observe.py:947`).

The audit is the **check phase**: before a changed edit finishes, verify the
assembled program against, in order:
1. **the user's ask / contract** — length, must-haves, and **requested features
   actually present** (e.g., user asked for a split screen but there are no
   `layout_regions`/split ops → flag "the requested split screen isn't in the
   edit"); then
2. **the guidance ceilings** — the existing `review` flags (off-camera-adjacent,
   rough heads/tails, overlay fit, etc.), framed as "diverges from the guidance
   default."

Surface all of it as **facts, never prescriptions**, and grant one repair pass
(reuse the existing one-shot `reviewed`/`struct_tries` guards so it can't loop).
The new piece is the **requested-feature presence check** — the thing that would
have caught the dropped split screen.

**Acceptance:** a changed edit that omits an explicitly requested feature (split
screen, music bed, target length) is flagged before finishing; flags never
prescribe a fix; the loop still terminates.

---

## Change 5 — Turn-budget headroom
**Where:** `_MAX_TURNS` (`tools.py:30`, currently `12`).

Modest bump (≈16–18) so compound asks ("30s **and** a split screen") have room
for both the spine and the added feature. **Note:** the real reduction comes from
Change 1 removing churn, not from the cap — don't over-raise it (latency/tokens
every edit).

**Acceptance:** a compound ask (length + split) can complete both within budget.

---

## Sequencing
1. **Changes 1 + 2** (think-first + precedence/ceiling) — prompt + guidance
   header only. Highest leverage, cheapest, no code risk.
2. **Change 4** (audit: requested-feature presence + alignment framing) — extends
   the existing gate.
3. **Changes 3 + 5** (churn nudge + turn bump) — small, do alongside.

## Non-goals (explicitly NOT doing)
- No two-agent planner/executor split — one loop only.
- No procedure, step list, or format-specifics in the prompt or guidance doc.
- No deterministic enforcement of the guidance ceiling (text-only, per user).
- No seg-anchored split-region rework — the split/PiP "program-pinned, add last,
  re-lay" tool note already shipped; the structural fix stays a separate future
  item if the note proves insufficient.

## Acceptance (whole)
The editor works like a single think → act → check agent: it plans its approach
from the ask + material first (emergent, never a baked recipe), executes it in a
short directed loop honoring `user > guidance > own judgment`, and checks the
finished edit against the ask (then the guidance) before finishing — flagging any
divergence (including a missing requested feature) for one repair pass. Nothing
domain-specific lives in the prompt or guidance.
