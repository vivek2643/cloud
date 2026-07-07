# Cuts v3 — Signal-Fed LLM Judgment (kill the band-aids)

Follow-on to `cuts_v3_boundaries_v2.plan.md`. Diagnosed on the Reel-trail
pickleball clip: boundaries-v2 handed *video keep/drop* to the LLM but then let
it silently drop real footage — including the single highest-energy beat in the
whole project (`act=0.99`, the ball-strike follow-through). Our reflex was more
deterministic patches (energy floors, wider `is_action` protection). Those are
band-aids. This plan removes complexity instead of adding it.

## Guiding philosophy — minimal guidance, maximal information

Strong models reason well when given full information, a clear goal, and the
output contract — and worse when boxed in by prescriptive imperatives. The old
prompt ("drop connective tissue", "keep every ACTION alone", "HARD CONSTRAINT:
don't cross a pause") is exactly what made it prune a `0.99` beat. So:

- **Prompt = goal + complete information + output contract + field meanings.**
  No editorial choreography, no rule lists. Let the brain ideate.
- **Mechanical invariants live in CODE, not the prompt.** No overlaps,
  boundaries on real edges, no cut swallowing an atom — all already repaired by
  `enforce_lattice_partition` + the post/pass2a validators. Strip them from the
  prompt entirely; the model reasons freely, code guarantees validity.
- **Keep exactly what informs, drop what restricts.** Field definitions and the
  fact that production cues/false-starts aren't part of the piece are *information*.
  "DIG AGGRESSIVELY, split it, watch the start/end" is *restriction* — cut it.

The key realization: pass 1 was **already** given per-atom motion numbers
(`act`, `cam`, `coh`, anchors) alongside the transcript. The LLM isn't blind to
motion — we under-informed it (mean energy only, coarse camera) and over-directed
it (prune). Fix both, then trust it.

## A. Enrich the per-atom signal (`lattice.py`)

Give the model the numbers it needs to judge motion, compactly. Add to `Atom`
and `render_atom_table`:

- `peak` — PEAK action energy over the span (not just the mean `act`). The mean
  hid the `0.99` inside a longer atom.
- `mot` — mean camera-motion magnitude 0..1 (so a real subject-tracking pan is
  distinguishable from a static hold; combined with the transcript the model
  can tell a meaningful move from a random reframe).

New line:
```
ATOM 7 [12300-15800] shot_cut->speech_edge act=0.70 peak=0.99 cam=pan mot=0.55 coh=0.90 anchors@13100
```

## B. Un-fragment the lattice — energy-defined action extent (`lattice.py`)

Root cause of the stubs: `_action_marks` padded each anchor cluster by a fixed
amount, so a swing became `[action][dropped 0.99 sliver][action]`. Replace the
fixed pad with an **energy-defined extent**: an action atom grows outward from
its anchor span while `action_energy` stays above a relative floor
(`ACTION_ENERGY_FLOOR_FRAC × the cluster's local peak`), bounded by the free
fragment and any grounded scene change (shot cut / transition). The whole motion
— wind-up, impact, follow-through — is ONE atom. No orphaned high-energy slivers.

`is_action` stays a LABEL (an atom containing an anchor cluster), useful signal
for the model and for the one safety net below.

## C. Rewrite the pass-1 mandate — free thinking (`pass1.py`)

Replace the imperative `_SYSTEM` with a lean brief: who it is (the editor's
judgment), what it's given (transcript + atom signals + hints, with field
meanings), what to return (speech_cuts, take_candidates, video groups,
junk_suspects, summaries — as word-indices / atom-ids, never ms), and the goal
(surface everything usable; cut only what isn't part of the piece; never
silently lose something real). Then stop talking. All the "how" comes out.

## D. Delete the band-aids (`pass1.py` enforce)

- Remove `_isolate_action_atoms` (forced action-alone grouping) — the model owns
  grouping now.
- Remove the `is_action`-only re-add guarantee.
- Keep the deterministic **contiguity split** (structural: a group must be one
  time-continuous run — an overlap-safety invariant, not editorial).
- **One safety net** (energy, not editorial): the single highest-`peak` atom per
  clip, if genuinely energetic, is guaranteed to surface even if the model drops
  it. Cheap insurance against the exact `0.99`-drop bug. Removable if you want it
  pure — say the word.

## E. Tests + re-ingest

Update lattice tests (energy extent, enriched fields), pass1 tests (drop
isolation asserts, add energy safety-net). Re-ingest Reel trail; verify the
pickleball swing is one intact beat and no high-energy footage is dropped.

## Non-goals / unchanged
Pass 2 (vision), the dial, the frontend, coverage-is-a-selection — all stay.
This is pass-1 signal + judgment only.
