# Cuts v3 — Boundaries v2 + the Energy Dial

Follow-on to `cuts_v3.plan.md` and `cuts_v3_editorial.plan.md`. This is the
"boundaries are a selection, not a partition" rework, plus finally wiring the
energy dial. Grounded in a real diagnosis of the Reel-trail pickleball clip.

## The problem (diagnosed on real data)

The pickleball clip (D484) produced 7 video atoms, 4 of them junk slivers:

```
[   0- 880] hold        [5560-6100] hold ←sliver   [6100-8500] ACTION
[8500-8700] hold ←jitter [8700-8760] hold ←60ms    [9560-10100] ACTION   [10100-10320] hold
```

Two independent causes:

1. **Manufactured boundaries.** `_disturbance_marks` fires on handheld jitter and
   carves atoms (`8700`) where nothing happened. Deterministic, but not *grounded*
   in any real event.
2. **The full-coverage invariant.** `post._validate_file_coverage` raises on any
   gap, so every dead space between two grounded events (`5560→6100`) is *forced*
   to become its own tile. If every millisecond must be a cut, "cutting" is
   meaningless.

## North star — a three-way split

| Layer | Owns | Kind |
|---|---|---|
| **Lattice** | *Where* a cut may legally begin/end — grounded boundaries only | Deterministic |
| **LLM (pass 1/2)** | *Which* pieces become cuts — grouping, **merging, junk removal** | LLM |
| **Dial** | *How tight* a cut plays — **negative padding**, anchor-protected | View-time |

Grounded boundaries (the only legal video edges): clip edges, speech edges, hard
shot cuts, wipe/degenerate transitions, and action-anchor windows. **Everything
else (camera move/settle, disturbance jitter) is a LABEL, never a boundary.**

Coverage is NOT preserved. Ungrouped / junked space is simply a gap. Nothing
*important* is dropped (every detected event is surfaced); only non-events
(connective tissue, pre-roll, dead air, jitter) fall to the cutting-room floor.
The raw clip stays playable in Media view as the ultimate escape hatch.

## Constants policy — ratios, not magic ms

No scattered absolute-ms thresholds, and NOT clip-length-relative (a usable shot
length does not scale with the source file's duration). Instead: dimensionless
**ratios** applied to each clip's own measured timescales, clamped by one named
**perceptual floor**.

- **Action pad** = `ACTION_PAD_FRAC × median(inter-anchor gap of that cluster)`,
  floored at `PERCEPTUAL_FLOOR_MS`. A fast flurry gets a tight pad; a lone slow
  swing breathes more. Naturally bounded (a cluster merges only within
  `ACTION_ANCHOR_MERGE_MS`, so the gap can't run away).
- **Min viewable / dial floor** = `PERCEPTUAL_FLOOR_MS` (real physics of
  perception — a 150–200ms "shot" is never a shot; footage-independent).
- **Snap** = existing `SNAP_MS`, a rounding tolerance in analysis resolution.
- Merge/junk have NO deterministic threshold — that is the LLM's job.

## A. Lattice — grounded boundaries only (`lattice.py`)

1. Drop `_disturbance_marks` from `all_marks` in `build_atoms`. Jitter stops
   carving atoms. (`camera_desc="handheld"` already labels jitter, so the signal
   is not lost — it's a label now, not an edge.)
2. `_action_marks`: pad each cluster by the anchor-relative rule above instead of
   a flat `ACTION_PAD_MS`.
3. Keep shot cuts, transitions, action windows, speech/clip edges as edges.

Result on D484: the `8700` jitter split disappears; remaining atoms are speech
gaps carved only by real action windows.

## B. Coverage relaxed (`post.py`)

`_validate_file_coverage` → `_validate_no_overlap`: keep the overlap check, drop
the three gap checks. Relax "no cuts at all for file(s) …" (missing files) from a
raise to a `logger.warning` — a clip can legitimately be all dead air.

## C. LLM owns merge + junk (`pass1.py` prompt)

- Relax section-C's "keep every ACTION atom as its OWN group, never merged":
  an action group may INCLUDE its immediate lead-in / follow-through hold so the
  payoff breathes, but two distinct actions stay separate and an action is never
  buried by merging into unrelated footage.
- Explicitly: MERGE connective holds into the adjacent action/shot, and JUNK
  pre-roll / dead air / transitional holds that aren't usable. Coverage is not
  required — leave nothing important out, but don't manufacture filler cuts.

## D. The energy dial — negative padding, anchor-protected (frontend, view-time)

The per-cut envelope already exists in `cut_records` (`pace`: `min_ms`,
`natural_ms`, `max_ms`, `levels[]`; plus `hero_ts_ms`). No new ingest, no model
call — pure client-side view-math over stored fields.

- A single energy slider (0 → 1) in `cuts-v3-view.tsx`.
- Per cut, interpolate the PLAYED span between `max_ms`/`natural_ms` (low energy)
  and `min_ms` (high energy), centered so the trim is **negative padding toward
  `hero_ts_ms`** — the anchor/peak frame is never trimmed away (anchor-protected,
  guaranteed by `min_ms` already being floored at the anchor span in `post.py`).
- The dial never re-fetches or re-ingests; it only changes `src_in`/`src_out`
  used for playback + the tile's displayed length.
- Subdivision at the very top band (a grouped action → its atomic beats) needs
  per-atom spans exposed to the client; deferred to a follow-up (this pass ships
  tightness).

## Build order

A → B → C → tests → re-ingest Reel trail → D (dial) → verify. A/B/C change the
stored cuts (need a re-ingest); D is view-only (no re-ingest).
