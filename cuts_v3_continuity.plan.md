# Plan: Cut-centric brain + per-cut continuity (supersedes Phase 4)

## Goal
Stop feeding the brain the full raw-footage "continuous source" view. Instead, give it
**only cuts + rich per-cut data**, and add a deterministic **continuity** parameter to every
cut so the brain understands ordering and adjacency (which clip, which cut number, and whether
a neighbor is a weldable continuation vs a hard cut) — without ever reading raw footage.

This replaces the deferred **Phase 4** in `cuts_v3_to_brain.plan.md` (reconciling the
continuous-source digest with the v3 lattice). We are **removing** that view, not migrating it.

## Motivation (from the ideation)
- The brain can't meaningfully reason over raw footage; the whole-clip continuous digest is
  low-value token weight. It already gets each cut + that cut's rich data (framing, pace, look,
  channel, take group…).
- What it actually lacks is **continuity**: "cut 5 of clip A is physically followed by cut 6"
  vs "there's a break between them." That is small, structured, and deterministic.
- Same engine already exists: `seam.py` `classify_seam()` — its documented FUTURE hook is
  exactly "timeline weld: when the editor drops two cuts adjacent, the SAME rule decides weld
  vs hard cut." We wire that hook.

## Design principles
- **Code owns numbers.** Continuity is computed deterministically from each clip's own signals;
  the LLM never emits a number, order, or threshold here.
- **No magic tolerance.** Contiguity is NOT "spans within N ms." It is `seam.classify_seam()`:
  physical continuity (same clip, same speaker, no shot/scene/transition break in the seam,
  gap not longer than what it bridges). Clip-relative, zero tuned constants.
- **Keep junk visible to the brain (labeled), don't strip it.** Junk stays IN the ordered
  sequence marked `junk: true` + reason, so (a) numbering + contiguity are honest and the
  sequence truly tiles the clip, (b) the brain can skip-by-default yet recover a junk beat as a
  connective bridge, matching the recoverable-junk principle. The frontend still HIDES junk in
  its tray by default — display ≠ what the brain sees.

---

## Continuity, precisely
A `continuity` block on every cut (brain projection AND persisted on `cut_records`):

| field | meaning |
|-------|---------|
| `clip` | source clip (stable id + short alias/name) |
| `cut_no` | 1-based ordinal of this cut within its clip, in **source order over ALL cuts incl. junk** |
| `of` | total cuts in that clip (incl. junk) |
| `prev_contiguous` | is the previous cut in the same clip a weldable continuation of this one? (`seam.classify_seam`) |
| `next_contiguous` | is the next cut in the same clip a weldable continuation? |
| `seam_reason_prev` / `seam_reason_next` | the human reason from `SeamVerdict.reason` (e.g. "shot/scene boundary…", "continuous take") |

Numbering over ALL cuts (junk included) means the non-junk cuts the editor sees may read
1,2,4,5 — the skipped number *is* the signal that a junk beat sits between 2 and 4. Honest and free.

### Why compute it at INGEST, not read time
`classify_seam` needs the seam's categorical inputs — atom boundary reasons
(`BREAK_BOUNDARY_REASONS = {shot_cut, wipe, degenerate}` off the atoms' `state_in`/`state_out`),
speaker identity across the seam, pass-1 junk flags, and the ms gap. Those signals are all live
in `post.assemble_cut_records` (lattice + atoms + motion + silences already in hand) but are NOT
all persisted on `cut_records` afterward (`transition_in/out` aren't even written today). So
compute continuity ONCE at ingest where the signals are richest, and persist it. Read paths
(brain + UI) then just read the block — no re-derivation, no missing-signal guesswork.

---

## Implementation

### Phase A — compute + persist continuity at ingest
`backend/app/services/l3/post.py`:
- After per-file cuts are assembled and ordered by `src_in_ms` (all cuts incl. junk):
  - assign `cut_no` (1..N) and `of` per clip.
  - for each adjacent pair on the same clip, build a `seam.Seam` from the ingest signals
    (same_clip=True, `same_speaker` from the two cuts' speakers, `gap_ms = next.src_in - cur.src_out`,
    `bridged_speech_ms` from the two spans, `has_scene_or_transition` from the atom boundary
    reason at the shared edge, `has_flagged_break` from a junk suspect overlapping the seam) and
    call `classify_seam()`. Its verdict fills `next_contiguous` of cur and `prev_contiguous` of next.
  - first cut's `prev_contiguous=False`, last cut's `next_contiguous=False`.
- Add `continuity: dict` to the `CutRecord` dataclass + `to_dict()`.
- Migration `029_cut_continuity.sql`: add a `continuity jsonb not null default '{}'` column to
  `cut_records` (additive; existing rows get `{}` and are backfilled on re-ingest).
- `ingest_store.insert_cut_records`: include `continuity` in the INSERT.
- `cuts_v3_read.rows_for_run`: select `continuity`.

### Phase B — brain reads cuts + continuity, junk kept & labeled
`backend/app/services/l3/cutrecord_map.py`:
- **Stop dropping junk** in `cut_dicts_for_files` (and in `signatures_for`'s row count). Carry
  `junk` + `junk_reason` and the `continuity` block onto each cut dict.
- Mark junk cuts so downstream rendering can tag them and keep them OUT of the recommended/
  take-group set while leaving them placeable (a deliberate bridge).

`backend/app/services/l3/footage_map.py`:
- `build_clip_tree` / the moment-line renderer: surface continuity on each beat line —
  `clip · cut_no/of` and `↔` (weldable) / `⋯` (hard) to the neighbor, plus a `[junk: reason]`
  tag on junk beats. Prefer reading the persisted `continuity` over recomputing `run_id`
  (runs can remain as a secondary hint, or be retired in favor of continuity).
- Ensure `arrange._MapIndex` still resolves a junk moment's ref (so the brain CAN place one when
  it chooses to bridge). Junk is skip-by-default in the prompt framing, not un-referenceable.

### Phase C — drop the continuous-source view from the brain
`backend/app/services/l3/converse.py`:
- Remove the CONTINUOUS SOURCE block from `_assemble_source_context` (and thus `_context_block_v3`);
  the prompt becomes BEAT INDEX (cuts + continuity) + CURRENT TIMELINE only.
- Update `_LOOP_SYSTEM_V3` to describe the cut-centric world: place cuts by `ref`; use continuity
  (`next_contiguous`) to decide welding; there is no raw-footage scan.

`backend/app/services/l3/tools.py`:
- Remove `source_awareness`, `scan_source`, and `place_span` from `_specs()` and `_dispatch()`
  (the arbitrary-window + raw-scan verbs). Keep `place`, `trim`, `remove`, `move`, `set_audio`,
  `split_edit`, `tighten`, `split_screen`, `ask_user`.

Leave the underlying modules dormant/recoverable (do NOT hard-delete this pass):
`observe.source_awareness` / `observe.scan_source`, `clip_timeline_store`, `atoms` continuous
builders. They can be removed in a later cleanup once the cut-centric loop is proven. This keeps
the change reversible.

### Phase D — frontend continuity + timeline auto-join
`frontend/src/lib/api.ts`: add `continuity` to the `CutRecord` type.
`frontend/src/components/cuts-v3-view.tsx`:
- Show `clip · cut N/of` on each tile (subtle, grey — per the frontend design skill).
- Use `prev_contiguous`/`next_contiguous` to **auto-join adjacent picked cuts** into one timeline
  segment (the behavior asked for earlier: adjacent picks weld; a hard seam stays a cut).
- Junk stays hidden in the micro/discarded tray by default (unchanged).

---

## Files to touch
- `backend/app/services/l3/post.py` — compute continuity + `CutRecord.continuity`.
- `backend/migrations/029_cut_continuity.sql` — `continuity jsonb` column.
- `backend/app/services/l3/ingest_store.py` — INSERT `continuity`.
- `backend/app/services/l3/cuts_v3_read.py` — SELECT `continuity`.
- `backend/app/services/l3/cutrecord_map.py` — keep junk (labeled) + carry continuity.
- `backend/app/services/l3/footage_map.py` — render continuity + junk tag on the beat line.
- `backend/app/services/l3/converse.py` — drop CONTINUOUS SOURCE block; cut-centric system prompt.
- `backend/app/services/l3/tools.py` — remove `source_awareness`/`scan_source`/`place_span`.
- `frontend/src/lib/api.ts` + `frontend/src/components/cuts-v3-view.tsx` — continuity + auto-join.
- Reuse (no change): `backend/app/services/l3/seam.py` (`classify_seam`).

## Testing / verification
- Unit `test_seam.py` — already covers `classify_seam`; add cases for cut-to-cut seams (gap≈0
  shared lattice edge: shot_cut → hard; energy-regime edge → weldable; speaker change → hard).
- Unit `test_post.py` — continuity numbering over all cuts incl. junk; `prev/next_contiguous`
  matches the seam verdict; first/last edges are False.
- Unit `test_cutrecord_map.py` — junk is KEPT and labeled; `continuity` rides onto the cut dict;
  a junk moment ref still resolves in `_MapIndex`.
- Golden (Reel 4, run `3f2b8b41…`): assert non-junk cut numbers skip where a junk beat sits
  between them; a same-shot pair reads `next_contiguous=true`, a shot-change pair `false`.
- Loop smoke: `converse.respond` prompt no longer contains "CONTINUOUS SOURCE"; the brain places
  by ref and the timeline welds two `next_contiguous` picks into one segment.

## Open questions for the executor
- Junk token cost: keeping junk lengthens the index. If it bloats, render junk beats as a
  terse one-liner (`[junk: cue] clipA 3/12`) rather than the full rich line — still visible,
  cheap. (Recommended.)
- `run_id`/`run_pos` in `build_clip_tree`: keep as a secondary continuity hint or retire now that
  `continuity` is first-class? Recommend keeping this pass, retiring in the Phase-C cleanup.
- Hard-delete vs dormant for `place_span`/`scan_source`/`source_awareness`: this plan leaves them
  dormant for reversibility; schedule deletion once the cut-centric loop is validated.
