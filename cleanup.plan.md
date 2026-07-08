# Plan: Remove legacy versions & dead code (narrowed scope)

## Intent
Delete **only** the genuinely unnecessary things, in four buckets:
1. **Extra / unused UIs**
2. **Old Cuts *versions*** (the hero-cut substrate + the cuts-v2 partition stack + their old APIs)
3. **Not-needed analysis** (signals/side-effects nothing reads on the live path)
4. **Old brain *versions*** (the v1/v2 arranger prompt/context styles) + **band-aids / dead flags / stale copy**

## HARD PRESERVE — do NOT remove or weaken (explicit)
- **All brain TOOLS / capabilities** stay exactly as they are: `place`, `tighten`, `trim`, `remove`,
  `move`, `set_audio`, `split_edit`, **`split_screen`** (see snap note below), `ask_user`, and the
  senses `read_state`/`predict`/`validate`/`diagnose`/`affordances`. This cleanup removes prompt
  *versions*, never tools.
- **L2 and everything the new cuts/brain need:** `l2.perception`, `clip_perception`,
  `cast`, `relations`, `framing`, `diarize`, `energy.default_energy_for`,
  `video_segments._sharpest_ms`, `lattice.build_atoms` (code atoms), `footage_trees` cache, and
  all core L1 tables (`transcripts`, `motion_dynamics`, `scene_cuts`, `audio_features` core,
  `dialogue_segments`). Untouched.

## Confirmed decisions
- **Delete the old v1/v2 arranger versions;** v3 becomes simply "the prompt" (no version flag).
- **Keep `split_screen`;** re-point its seam-snap to `cut_records` so the old hero substrate can
  still be deleted (capability preserved, old code gone). See Bucket B.

---

## BUCKET A — Extra / unused UIs (frontend)
Delete unrouted components (zero importers): `frontend/src/components/hero-cuts-view.tsx`,
`frontend/src/components/dialogues-view.tsx`, `frontend/src/components/breadcrumb.tsx`.

`frontend/src/lib/api.ts` — remove the retired-feature API only:
- Hero Cuts block: `HeroChannel/HeroSubject/HeroTake/HeroCut/HeroCutsResponse`, `getHeroCuts`,
  `getHeroCutsFeed`.
- Dialogues block: `DialogueSegment`, `DialoguesResponse`, `getDialogues`.
- **Keep** folder/file ops, `getL1Index`, `listRenders`, `listEditThreads`, etc. (roadmap; backend exists).

`frontend/src/components/timeline-editor.tsx` — remove the `application/x-dialogue-segment` drop
handling (dead producer). Rename the `application/x-hero-cut` MIME to a neutral `application/x-cut`
and update the emitter in `cuts-v3-view.tsx`. Update copy "Drag clips here from Hero Cuts or
Dialogues" → "Drag cuts here".

Stale copy/comments (Bucket E too): `cuts-v3-view.tsx` (refs to deleted `cuts-view.tsx` /
"original Cuts tab" / "v2 endpoint untouched"), `ai-edit-panel.tsx` EmptyState (says edits need a
"go"; backend applies during the turn — align).

**Verify:** `rg -n "hero-cuts-view|dialogues-view|breadcrumb|getHeroCuts|getDialogues|x-dialogue-segment|Hero Cuts|Dialogues bin" frontend/src` empty; frontend typecheck/build clean.

---

## BUCKET B — Old Cuts versions (backend)

### B1. Re-point `split_screen` snap OFF the hero substrate (do FIRST)
Today `split_screen`'s raw-window path snaps via `observe._timeline` →
`clip_timeline_store.load_clip_timeline` → `hero_cuts._build_field_v2` (the old substrate).
- Add a tiny v3-native seam source: the clean cut boundaries for a file from its resolved
  `cut_records` run (cut `src_in_ms`/`src_out_ms` edges; `cutrecord_map`/`cuts_v3_read` already
  load these). A helper like `observe._seams_for_file(file_id, ctx)` returning boundary ms.
- Point `split_screen`'s snap at that helper. `split_screen` behavior is unchanged for the brain.
- This frees `clip_timeline` / `clip_timeline_store` for deletion (Bucket C).

### B2. Remove the hero-cut substrate
- Config: remove the `footage_source` flag; `footage_map._source_signatures` / `_build_trees` /
  `_annotate_dups` keep only the `cut_records` branch.
- Delete files: `hero_cuts.py`, `hero_store.py`, `combine.py`, `score_span.py`.
  - `energy.py`: keep `default_energy_for`; drop hero-only `energy_band`/`energy_to_params`/
    `EnergyParams` once no importer remains.
- `l2/perception.py`: remove the `hero_store.defer_precompute(...)` calls (both paths).
- `jobs.py`: remove the `l3_precompute_hero_cuts` task registration.
- `routers/files.py`: remove the `/hero-cuts` routes.
- DB (migration 030): `drop table if exists hero_cuts_cache;`

### B3. Remove the cuts-v2 partition stack
- **Extract first:** move `R_CLIP`, `R_SHOT`, `_shot_marks` from `base_cuts.py` into `lattice.py`
  (or a tiny `boundaries.py`) — `lattice` imports them, so this must precede deletion.
- Delete files: `partition.py`, `tightness.py`, `partition_params.py`, `base_cuts_params.py`,
  `speech_granularity_params.py`, `video_segment_params.py`, and `base_cuts.build_base_cuts`.
- `video_segments.py`: keep `_sharpest_ms`; delete `segment_video`.
- `routers/files.py`: remove the `GET/POST /cuts` (v2) routes.

**Verify:** `rg -n "footage_source|hero_cuts|hero_store|defer_precompute|combine\.|score_span|base_cuts|build_base_cuts|partition|tightness|segment_video" backend/app` shows only the extracted lattice constants + `_sharpest_ms`; ingest + `scripts/test_lattice.py`/`test_post.py`/`test_ingest.py`/`test_footage_map.py` green.

---

## BUCKET C — Not-needed analysis / dead infra
Now orphaned after A/B.

- Delete files: `clip_timeline.py`, `clip_timeline_store.py` (no callers after B1), `atoms.py`
  (the **VLM** atom builder — NOT `lattice.build_atoms`), `thought_segments.py`, `program_clock.py`.
- `l2/perception.py`: remove the `thought_segments.defer_thoughts(...)` call.
- Config: remove `enable_thought_segments`.
- `observe.py`: delete `source_awareness()`, `scan_source()`, the old `_timeline()` helper, and
  `EditContext.program_field` (never populated) + `tl_cache` if now unused. (These are NOT tools —
  they were removed from the tool set already; this just deletes the dormant functions.)
- `tools.py`: drop the dead `_snap_prog`/`program_clock` call inside `split_screen` (its snap now
  comes from B1). No tool is removed.
- `arrange.py`: (optional, safe) drop the `_MapIndex.atoms` branch — the v3 moment shape sets
  `atoms: []`, so it never resolves.
- L1 orphaned signals (LAST, most caution — only after the above; touches the L1 pipeline):
  `music_structure` (nothing reads it), `f0_hz` (partition-only), `dialogue_cut_*`/`beat_cut_*`
  grids (partition/hero/clip_timeline-only). Stop computing + drop columns/tables in migration 030.
  **Leave** `motion_dynamics`, `scene_cuts`, `transcripts`, `dialogue_segments`, and
  `audio_features.{rms_db,prosody_hop_ms,silence_intervals}`.

**Do NOT touch** the L2 schema fields (`presence_lane`/`activity_lane`/`take_quality_events`) — that
is L2 output; leaving it alone per the preserve rule.

**Verify:** `rg -n "clip_timeline|thought_segments|speech_thoughts|program_clock|source_awareness|scan_source|from app.services.l3 import atoms|music_structure|f0_hz|dialogue_cut|beat_cut" backend/app` — only `lattice.build_atoms` remains; imports clean; one full L1→L2→ingest on a test clip works.

---

## BUCKET D — Old brain versions + band-aids
`converse.py`
- Delete `_LOOP_SYSTEM` (v1), `_LOOP_SYSTEM_V2`, `_context_block` (v1), `_context_block_v2`,
  `_assemble_source_context`. Keep `_LOOP_SYSTEM_V3` / `_context_block_v3` (rename, dropping `_v3`).
- `respond()`: remove the version branch; always use the (single) v3 system + context.

`config.py` — remove `autoedit_arranger_version`.

Band-aids / stale flags & copy: remove the now-unreferenced config flags surfaced above
(`footage_source`, `autoedit_arranger_version`, `enable_thought_segments`); fix the frontend
stale strings (Bucket A) and the v1 `dup:tgN` vocabulary reference.

**Verify:** `rg -n "arranger_version|_LOOP_SYSTEM_V2|_context_block_v2|_assemble_source_context|_LOOP_SYSTEM\b" backend/app` empty; `converse` imports; a `converse.respond` dry-run prompt is the v3 BEAT INDEX only, and lists only tools that exist in `tools._specs()`.

---

## Migration
`backend/migrations/030_drop_legacy_cuts.sql` (idempotent, `if exists`):
- `drop table if exists hero_cuts_cache;`
- `drop table if exists speech_thoughts;`
- `drop table if exists music_structure;`
- `alter table audio_features drop column if exists f0_hz, drop column if exists dialogue_cut_cost, drop column if exists dialogue_cut_hop_ms, drop column if exists dialogue_cut_points, drop column if exists beat_cut_cost, drop column if exists beat_cut_hop_ms, drop column if exists beat_cut_points;`
(Apply like prior migrations — plain psycopg, idempotent. The L1-signal drops only after Bucket C.)

## Risks / guardrails
1. **`lattice` imports `base_cuts`** — extract `R_CLIP`/`R_SHOT`/`_shot_marks` BEFORE deleting
   `base_cuts` (B3), or all of Cuts v3 breaks.
2. **`split_screen` snap** must be re-pointed (B1) BEFORE deleting `clip_timeline` (C).
3. **Two atom builders** — delete `atoms.py` (VLM) but KEEP `lattice.build_atoms`.
4. **Keep `energy.default_energy_for` and `video_segments._sharpest_ms`** though their legacy
   neighbors are deleted.
5. **Do NOT drop L2 / `clip_perception` / cast / relations / framing** — they feed the live brain.
6. **`footage_trees` cache is LIVE** for v3 — not hero-specific; keep.
7. Prune tests for removed dormant functions in `scripts/test_observe_act.py` so the suite stays
   green; keep/extend `test_footage_map.py`, `test_post.py`, `test_cutrecord_map.py`,
   `test_lattice.py`, `test_seam.py`, `test_ingest.py`.

## Suggested slicing (dependency-safe order)
1. Bucket A (frontend) — independent.
2. B1 (re-point split_screen snap).
3. B2 (delete hero) → B3 (delete cuts-v2 partition).
4. Bucket C (dead infra; L1-signal drops last).
5. Bucket D (delete v1/v2 versions + flags).
One commit per step so each is reviewable and revertible; the migration lands with its bucket.

## Explicitly OUT of scope (per narrowed intent)
- No changes to brain tools/capabilities (only the split_screen *snap source* moves).
- No L2 schema trimming; no touching `clip_perception`/cast/relations/framing.
- Keep roadmap frontend API (folder/file ops, L1 index, renders, edit-thread list).

## Note on complete-cut-awareness fields
Adding the missing brain-awareness fields (`take_role`, `look`, pace bounds, seam reasons to the
BEAT INDEX) is a **separate additive task**, tracked in the previous review — intentionally kept
out of this deletion-only plan so cleanup and feature work stay in separate, revertible changes.
