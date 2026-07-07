# Cuts v3 — Deterministic Keep, Semantic Cull (kill every threshold)

Supersedes the threshold-based bits of `cuts_v3_boundaries_v2.plan.md` and
`cuts_v3_signal_judge.plan.md`. Written after the honest diagnosis that our
recurring failures come from TWO habits:

1. **We invent magic numbers** (`ENERGY_SAFETY_NET_MIN=0.6`,
   `ACTION_ENERGY_FLOOR_FRAC=0.5`, `CAMERA_*` cutoffs, junk-confidence tiers).
   Each is a band-aid tuned to one clip that won't travel.
2. **We ask the LLM for numbers** (junk_confidence high/low, and anywhere we'd
   trust a model-emitted score/threshold). A language model is not a reliable
   source of a number.

## The one rule

| Decision | Owner | Method | Never |
|---|---|---|---|
| **Quantitative / structural** — where boundaries fall, which beat is the peak, action extent, "is this span active", keep-vs-lose, pace, take similarity | **Code** | derived from *the clip's OWN signal distribution* (baseline, histogram split, ranking, gaps) | a hand-set constant |
| **Semantic / categorical** — delivered line vs production cue, do these atoms form one moment, same take?, what is it, is it junk, how important | **LLM** | *reads* the signals + full transcript, outputs categories / ids / text | a number, a score, a threshold |

The LLM judges **meaning**. Code does **all counting**, and counts relative to
the data itself — not against a constant a human picked.

## The load-bearing shift: KEEP is deterministic, CULL is semantic

The `0.99`-drop and the "lots of speech missing" both trace to the same thing:
the LLM's *omission* silently drops material, and our rescue was a threshold.

New guarantee, enforced in code:

- **Every word lands in some speech_cut. Every atom lands in some video group.**
  The LLM proposes the *boundaries and grouping* (meaning); code then fills any
  ungrouped word-run / atom into its own unit. **The candidate set loses
  nothing — a real moment cannot be silently dropped, ever.**
- **Removal happens only as an explicit, recoverable semantic JUNK label.** The
  visible feed = all candidates minus what the LLM calls junk *by meaning*.
  That is NOT full coverage (connective tissue is culled) — but nothing is lost
  to a number or to the model forgetting it; anything hidden is one toggle away.

This reconciles "nothing should be missed" with "don't manufacture arbitrary
cuts": the safe failure direction is an extra low-value tile (recoverable),
never a vanished action.

## Atomization without magic numbers

Non-speech remainder is cut at **grounded events** (shot cuts, transitions —
already deterministic) PLUS transitions between the clip's OWN quiet/active
energy regimes, where "quiet vs active" is found from **that clip's own energy
histogram** (a data-driven split — e.g. Otsu / 2-means — no global constant).
`action_points` become annotations (`anchors@`), not carvers. An "active"
regime is an action beat; a "quiet" one is a hold. Nothing is tuned per-footage.

## The LLM becomes categorical-only (`pass1`)

It reads the transcript + the signal table and returns ONLY:
- `speech_cuts` — word-index ranges (boundaries = meaning).
- `take_candidates` — same-line retakes (ids).
- `video_groups` — atom-ids that form one moment (merging = meaning).
- `junk` — binary, by meaning: production cues/counts (identified from the
  transcript itself, NOT diarization — diarization mislabels the off-camera
  operator), false starts the speaker re-delivers, dead air. Recoverable.
- optional `importance`/role as a categorical LABEL (hero / support / b-roll) —
  non-destructive; it orders and informs, it never drops.
- summaries.

No confidence, no score, no pace number, no threshold — ever.

## Deletions (every hardcoded number in the cuts-v3 keep/drop path)

- `ENERGY_SAFETY_NET_MIN` — gone (keep is deterministic, no rescue needed).
- `ACTION_ENERGY_FLOOR_FRAC` — gone (extent = the clip's own active regime).
- `ACTION_ANCHOR_MERGE_MS` — gone (regime segmentation replaces anchor clustering).
- `CAMERA_MOVE_FRAC_MIN / _HOLD_MOTION_MAX / _PAN_COHERENCE_MIN / _PAN_STABILITY_MIN`
  — gone; drop the derived `cam=` label and hand the LLM raw `mot`/`coh` so it
  categorizes camera behaviour itself.
- `junk_confidence` (LLM-emitted tier) — gone; junk is binary + recoverable.
- `is_action` re-add guarantee — gone (superseded by keep-everything).

**Kept, because they are physics/perception, not tuning** — exactly one each,
named and documented: `PERCEPTUAL_FLOOR_MS` (a shot shorter than this is a
flash), `LONG_PAUSE_MS` (human pause perception), `SNAP_MS` (analysis hop
resolution). These are not band-aids and don't scale with footage.

## Pace / dial (already deterministic — confirm, don't regress)

Pace envelope + dial view-math are computed in code from signal distributions,
not from the LLM. Keep them that way. If any model-emitted pace number exists,
remove it.

## Frontend dial (cosmetic, separate)

The energy slider shows `0` and is unstyled. Copy the main Cuts view's energy
control look/behaviour. Pure UI.

## Build order

1. `lattice.py` — regime-based atomizer (histogram split), drop `cam=` derived
   label + expose raw `mot`/`coh`, delete the action-carve constants.
2. `pass1.py` — total-coverage fill in `enforce_lattice_partition` (every word /
   atom covered); categorical-only prompt; cue detection framed as pure meaning.
3. Strip constants from `lattice_params.py`; remove `junk_confidence` end-to-end
   (`pass2a`, `pass2`, `post`, migration is additive so column can stay unused).
4. Tests: coverage guarantee, regime atomizer, no-threshold assertions.
5. Frontend: restyle the dial from the main Cuts energy control.
6. Re-ingest Reel trail; verify no action/speech lost, cues culled, no constants.

## Open question (flagged, not decided)
Pass 2 still emits numbers the model isn't reliable at (caption-zone coords,
crop rects, rotation_deg). Same principle applies there eventually — but that's
a separate pass; this plan is pass-1 keep/cull only.
