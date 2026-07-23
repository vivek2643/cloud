# Cuts V4 only — retire the V3 algorithm, make V4 the sole cuts path

Status: **ready to execute**. Small, independently-shippable steps. Nothing here is
implemented yet. Verify each phase on **Reel 5** before starting the next.

## 0. Intent

Make the **V4 deterministic video segmenter** (`app/services/l3/v4_segment.py`) the
**only** cuts path. Remove the **V3 algorithm** (the LLM-emitted, atom-membership
video grouping that V4 replaces) entirely. Anything currently named "v3" that V4
still uses is **renamed** (drop the "v3"), not deleted. From now on, cuts are always
V4 — there is no flag, no alternate path.

### Decisions (state up front)

- **"Older cuts should exist" = existing `cut_records` data stays readable.** We do
  **not** run a destructive migration and we do **not** force re-ingest. This is
  free: every surface reads persisted `cut_records` through one shared layer using
  the stored `src_in_ms`/`src_out_ms`, and legacy rows (no `salience.kind`) already
  render via a **legacy ladder mode** (symmetric shrink) that is *row-rendering*,
  not *V3 algorithm code*. That path stays.
- **Atoms stay.** V4 does not use atoms for video, but atoms remain the **speech
  substrate** (pass1 speech grouping, seam/beat merge, junk, image-plan speech
  units, said_text). Only the *video* atom-membership paths are removed.
- **Only what V4 replaces is "the V3 algorithm."** Concretely: pass1's LLM video
  grouping + its enforcement, pass2's V3 video clause + atom-partition validators,
  and the atom-membership *video* span resolution in image_plan / post / identity.
  Speech grouping, junk, takes, continuity for speech pairs — all stay.

### Scope — do ONLY "A"

This plan does exactly one thing: **remove the V3 video-grouping algorithm so V4 owns
all video cuts** (pass 1 becomes pure speech grouping). Explicitly **out of scope**:

- **(B) Trimming atoms from pass 1's prompt** — atoms stay in the prompt as
  speech-bridge context. Do NOT remove the atom table from clip blocks.
- **(C) Removing atom generation (`lattice.build_atoms`)** — atoms are still the
  non-speech scaffolding several *speech* paths reference; generation stays. (An
  atom-free lattice is a separate future investigation, justified by simplicity not
  compute — the expensive work is optical flow / scene detect, which V4 needs anyway.)

### Guardrails

- V3 keeps working until the **final flip** — we make V4 the sole *runtime* path and
  verify BEFORE deleting any code.
- No renamed DB migrations, tables, or columns. No touched plan docs.
- Each phase is a separate commit, verifiable on Reel 5, revertible on its own.

---

## 1. Inventory (source of truth for the steps)

### DELETE (V3 algorithm — dead once V4 is sole path)

| File:line | What |
|-----------|------|
| `backend/app/config.py:132-138` | `Settings.cuts_segmenter` flag |
| `backend/app/services/l3/ingest.py:230-269` | `if settings.cuts_segmenter == "v4":` guard — make the V4 block unconditional |
| `backend/app/services/l3/ingest.py:291,324,378-382,391-397` | V3/V4 conditionals — collapse to the V4 arm |
| `backend/app/services/l3/pass1.py:100,149-150,161-164` | `Pass1Output.video_tentative_groups` as LLM output + prompt bullets |
| `backend/app/services/l3/pass1.py:251-283` | `_no_speech_cut_swallows_atoms` |
| `backend/app/services/l3/pass1.py:809-817` | `_contiguous_atom_runs` |
| `backend/app/services/l3/pass1.py:1014-1019,1209-1264` | `enforce_lattice_partition` steps 4–5 (video contiguity split + video coverage fill) |
| `backend/app/services/l3/pass1.py:1391-1397` | `render_pass1_output` "VIDEO TENTATIVE GROUPS" atom listing |
| `backend/app/services/l3/pass2.py:529-538` | `_V3_VIDEO_CLAUSE` |
| `backend/app/services/l3/pass2.py:637-651` | V3 default/branch in `system_prompt` / `gemini_system_prompt` — always V4 clause |
| `backend/app/services/l3/pass2.py:756-766` | `_video_group_is_v4` (invert to always-V4) |
| `backend/app/services/l3/pass2.py:769-822` | V3 atom backfill (`backfill_locators`) + `_locators_resolved` atom error |
| `backend/app/services/l3/pass2.py:825-891` | `_split_groups_partition_atoms`, `_no_duplicate_atoms` |
| `backend/app/services/l3/pass2.py:944-949,1026-1027` | V3 atom branch in `_resolve_cut_span_ms` + its semantic check |
| `backend/app/services/l3/image_plan.py:81-92,202-203` | `_atom_group_span` + V3 else branch |
| `backend/app/services/l3/post.py:845-852,936-942` | Atom-membership video span + V3 `_salience` recompute for video |
| `backend/app/services/l3/post.py:744-746` | V3 atom seam path for video-video pairs (keep speech-adjacent path) |
| `backend/app/services/l3/identity/faces.py:146-151` | Atom-membership branch in `_cut_span_ms` (video) |

### FIX (required for a correct V4-only path)

| File:line | Issue |
|-----------|-------|
| `backend/app/services/l3/pass2.py:429-442` | `apply_junk_suspects` matches video junk via `atom_ids ⊆ suspect.atom_ids`; V4 video cuts have no atoms → **switch to ms-span overlap matching** so pass1 video junk still applies. |

### RENAME (shared/canonical — drop "v3")

| Current | File | New |
|---------|------|-----|
| `cuts_v3_read.py` | `backend/app/services/l3/cuts_v3_read.py` | `cuts_read.py` |
| `load_cuts_v3` | `cuts_v3_read.py:81` | `load_cuts` |
| task `l3_cuts_v3_ingest` | `ingest.py:411-426` | `l3_cuts_ingest` (see risk R1) |
| route `GET /api/projects/{id}/cuts-v3` + `get_cuts_v3` | `routers/projects.py:62-63` | `/cuts` + `get_cuts` (see risk R2) |
| frontend `cuts-v3` stage / `CutsV3View` / `cuts-v3-view.tsx` / `CutsV3Response` / `getCutsV3` / `/cuts-v3` | see §4 | `cuts` / `CutsView` / `cuts-view.tsx` / `CutsResponse` / `getCuts` / `/cuts` |
| "Cuts v3" docstrings/comments | `ingest.py`, `pass1.py`, `pass2.py`, `post.py`, `image_plan.py`, `cutrecord_map.py`, `ingest_store.py`, `energy.py`, `run_workers.sh` | comment-only edits |

Import sites to update on the `cuts_v3_read` rename: `cutrecord_map.py:103,588`,
`routers/projects.py:21`, `footage_map.py:1071`, `observe.py:102,288`,
`auto_edit.py:78`, `captions/resolver.py:78`, `grade/scene_meta.py:80`,
`sync/audio_route.py:20,47`.

### LEAVE (do not touch)

- Atoms / `lattice.build_atoms` and all **speech** consumers.
- `ingest_runs`, `cut_records` tables; migrations `024_cuts_v3.sql` etc. (comments
  are historical).
- Legacy-row rendering: `cutrecord_map._symmetric_rung` / `_video_rung` `kind is None`
  branch (`cutrecord_map.py:120-132,343-344`) and frontend `cuts-v3-view.tsx:101-102`.
  This is **legacy ladder mode**, kept so old cuts still render.
- `cut_records.atom_ids` column (legacy rows keep values; new rows `NULL`/`[]`).
- `cuts_v3*.plan.md` docs.

---

## 2. Phase 0 — Baseline & safety (measure first)

1. On **Reel 5**, run ingest with `cuts_segmenter="v4"` (temporary env/config
   override) and capture the resulting `cut_records` as a golden snapshot
   (spans, salience, labels, junk).
2. Run ingest with `"v3"` on the same project for side-by-side, so we can confirm
   the later code-deletion doesn't change V4 output.
3. Confirm the frontend renders both V4 output and an **old V3-ingested project**
   (legacy rows) correctly — this proves "older cuts still exist" before we remove
   anything.

**Exit criteria:** V4 ingest on Reel 5 is sane and the golden snapshot is saved.

## 3. Phase 1 — Make V4 the only runtime path (no deletes yet)

1. `ingest.py:230-269`: run the V4 segmenter block **unconditionally** (remove the
   `if settings.cuts_segmenter == "v4"` guard); collapse the `291/324/378-382/391-397`
   conditionals to their V4 arms.
2. `pass2.py`: `system_prompt` / `gemini_system_prompt` always use the V4 clause.
3. **Apply the junk fix** (`pass2.py:429-442`, ms-span overlap) — required now that
   every video cut is a V4 cut with no atom_ids.
4. Leave `cuts_segmenter` defined but unused for one commit (easy revert).

**Verify on Reel 5:** ingest output matches the Phase-0 V4 golden snapshot; pass1
video junk suspects now actually apply to video cuts.

**Exit criteria:** V4 is the only path taken at runtime; output unchanged vs golden.

## 4. Phase 2 — Delete the V3 algorithm

Delete everything in the **DELETE** table (§1), including the `cuts_segmenter`
setting itself. Keep atoms and all speech paths. Keep legacy-row rendering.

**Verify on Reel 5:** ingest still reproduces the golden snapshot; an old
V3-ingested project still opens and edits in the brain + frontend.

**Exit criteria:** no reference to `cuts_segmenter` or the V3 video-grouping code
remains; `rg -n "cuts_segmenter|video_tentative_groups|_atom_group_span" backend`
is clean except intended speech uses.

## 5. Phase 3 — Rename "v3" → "cuts" (shared symbols)

Do the renames in the **RENAME** table (§1). Handle the two lockstep risks (below).
Update all import sites and the backend/frontend route + task name together.

**Verify:** backend tests `test_projects_router.py`, `test_pass2.py` updated and
green; frontend "Cuts" tab loads a project end-to-end.

**Exit criteria:** `rg -in "cuts.?v3|CutsV3" backend frontend` returns only
historical migration/plan-doc comments.

## 6. Phase 4 — Cleanup

Comment/docstring sweeps ("Cuts v3" → "Cuts"), delete now-unused test branches,
update `run_workers.sh` comments. Optional: a short note in a plan doc that cuts is
now single-path.

---

## 7. Risks & handling

- **R1 — Renaming the procrastinate task `l3_cuts_v3_ingest`.** In-flight/queued
  jobs enqueued under the old name will fail after the rename. Handling: do Phase 3
  when the ingest queue is drained, **or** register a temporary alias task
  `l3_cuts_v3_ingest` that forwards to `l3_cuts_ingest` for one release, then remove.
- **R2 — API route `/cuts-v3` → `/cuts`.** Backend and frontend must ship together,
  or add the new route while keeping `/cuts-v3` as an alias for one release. Since
  frontend + backend deploy from the same branch here, a lockstep change is fine.
- **R3 — Legacy rows.** Do **not** remove the `salience.kind is None` legacy ladder
  branches; they are what makes old cuts render. They are not "V3 algorithm."
- **R4 — Junk regression.** The `apply_junk_suspects` ms-fix (Phase 1) is the one
  behavior change with real bite; verify pass1 video junk still lands on Reel 5.

## 8. Verification checklist (run on Reel 5 each phase)

- [ ] V4 ingest reproduces the Phase-0 golden `cut_records` snapshot.
- [ ] Video junk suspects from pass1 apply to V4 video cuts.
- [ ] An **old V3-ingested** project still opens, edits, and renders (legacy rows).
- [ ] Brain footage map / captions / grade read cuts unchanged.
- [ ] `rg` sweeps for `cuts_segmenter`, `CutsV3`, `cuts-v3`, `video_tentative_groups`
      are clean except historical comments.
