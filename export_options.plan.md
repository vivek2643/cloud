# Export Options — Plan

## Goal

Give the user a single **Export** surface where they choose **what** to get and at **what
quality**, then we generate exactly those files and hand them a download. Scope is the
current, zero-failure feature set only:

- **Cuts** (order, in/out, J/L splits)
- **Framing / reframe** (per-clip scale + position + crop)
- **Static split-screen / PiP** (fixed layout regions — no animated cells)
- **Subtitles** (word-timed captions)

**Explicitly out of scope for this plan:** colour grading (LUT/CDL/grain/halation),
animated/moving split-screen cells, multi-track audio stems, AAF/OMF round-trip.

---

## The three deliverables

The user can pick any combination of these in the export dialog.

| Deliverable | Contents | Packaging | Reliability |
|---|---|---|---|
| **A. Finished video** | one baked MP4 (cuts + framing + split-screen + burned-in subtitles) | direct download link | ✅ reuses existing render stack |
| **B. Rough cut for NLE** | `.fcpxml` + `.srt` (+ optional original media + relink manifest) | **self-relinking ZIP** | ✅ deterministic; split-screen needs one golden-import validation |
| **C. Sidecar subtitles** | `.srt` only | direct download link | ✅ trivial |

**Quality is a user choice** (applies to A, and to media included in B):

| Preset | Long edge | Source | ffmpeg | Notes |
|---|---|---|---|---|
| `2160` (**4K**) | 3840 | original | CRF 18 | new preset |
| `1080` (**Full HD**) | 1920 | original | CRF 20 | ~existing `export` preset |
| `720` (**preview**) | 1280 | proxy | CRF 24 | existing `preview` preset |
| `source` | native | original | CRF 18 | passthrough resolution |

---

## Source of truth

Everything derives from the **resolved edit document**, not raw `cut_records`. The render
stack already resolves `timeline + operations + layout_regions + captions` into a
`layers.ResolvedTimeline` (video/audio layers, per-layer `transform`, `dest` sub-rects for
split-screen, resolved caption events). Both the MP4 and the FCPXML/SRT must read from the
**same resolved timeline** so the rough cut matches the finished video exactly.

- Resolver: `backend/app/services/l3/layers.py` → `resolve()`
- Document resolution for render: `backend/app/services/render/tasks.py` → `resolve_document()`
- Captions resolver: `backend/app/services/l3/captions/resolver.py` → `resolve_captions_for_document()`

---

## Phase 0 — Prerequisite: wire the render/export worker (blocker)

The `render_edit` task enqueues on queue `"render"` but **no worker listens to it** —
jobs sit `queued` forever. Export inherits this. Fix before anything else.

- **File:** `backend/run_workers.sh`
  - Add a worker (or fold into an existing CPU worker) that pulls the `render` queue, e.g.
    `WORKER_QUEUES=render,export python worker.py`.
  - Remove the stale comment block claiming "Renders were removed / cpu render queue is dead".
- **Verify:** enqueue a `render_edit` job and confirm a worker picks it up and produces
  `renders/{uuid}.mp4` in R2.

---

## Phase 1 — Backend: quality presets (4K + source)

- **File:** `backend/app/services/render/compositor.py`
  - Add presets to `PRESETS`:
    - `"2160"`: long edge 3840, `use_proxy: False`, CRF 18
    - `"source"`: passthrough native resolution, `use_proxy: False`, CRF 18
  - Keep existing `"export"` (1920) — alias it to `"1080"` or add a `"1080"` key.
  - Ensure `resolved_hash(resolved, preset)` includes the preset so 4K vs 1080 dedupe
    correctly (they must NOT collide in the `renders` dedup).
- **File:** `backend/app/routers/renders.py`
  - Accept the new preset values in the render request validation.

---

## Phase 2 — Backend: SRT exporter (sidecar C, and part of B)

- **New file:** `backend/app/services/export/srt.py`
  - `build_srt(resolved_captions) -> str`
  - Input: the resolved caption events (`resolve_captions_for_document(...)`), already
    timed against the final program timeline (post-cut, post-J/L).
  - Output: standards-compliant SRT (index, `HH:MM:SS,mmm --> HH:MM:SS,mmm`, text).
  - Group word-level events into readable caption lines (respect existing line-grouping
    from `captions/timing.py` if available; otherwise a simple max-chars/max-duration
    grouper). Keep line breaks; strip ASS-only styling.
- **Test:** `backend/scripts/test_export_srt.py` — golden SRT from a fixed resolved-caption
  fixture; assert monotonic, non-overlapping timecodes.

---

## Phase 3 — Backend: FCPXML exporter (rough cut B)

- **New file:** `backend/app/services/export/fcpxml.py`
  - `build_fcpxml(resolved, file_lookup, *, media_dir="media", srt_relpath=None) -> str`
  - Emit FCPXML v1.9+ with:
    - **`<resources>`**: one `<asset>` per source `file` (use `files.filename`; reference
      path `media/<filename>` — **relative**, so the ZIP self-relinks), plus `<format>`.
    - **`<sequence>`** on a single primary storyline: one `<asset-clip>` (or `<clip>`) per
      timeline segment, in **program order**, with `offset`, `start`, `duration` from the
      resolved timeline (frame-accurate; pick the project frame rate).
    - **Framing / reframe** → per-clip `<adjust-transform>` (scale + position) and
      `<adjust-crop>` from each layer's `transform` (`zoom`, `focus`, `fit`, `anchor`,
      crop rects). Map normalized coords → FCPXML transform space (see Phase 6).
    - **Static split-screen / PiP** → for each `layout_regions` entry, place the cell
      layers as **connected clips** above the base with `<adjust-transform>` scale+position
      derived from the cell's `dest` sub-rect `{x,y,w,h}`. No keyframes.
    - **J/L cuts** → connected audio / spine audio offsets from `split_edit` boundaries
      already baked into the resolved audio layers.
    - **Subtitles**: prefer attaching the `.srt` as a sidecar (Premiere/Resolve import
      captions from SRT). Do NOT try to emit styled caption titles in FCPXML — burned-in
      lives only in the MP4; editable lives as SRT.
  - **Frame rate:** derive one project fps; snap all offsets/durations to frames.
- **Test:** `backend/scripts/test_export_fcpxml.py` — build from a fixture resolved
  timeline; assert well-formed XML, correct clip count/order, transforms present for a
  split-screen region.

---

## Phase 4 — Backend: ZIP bundler + relink manifest (packaging for B)

- **New file:** `backend/app/services/export/bundle.py`
  - `build_rough_cut_bundle(resolved, file_lookup, *, include_media, quality) -> r2_key`
  - Assemble in a temp dir:
    ```
    <ProjectName>/
      <ProjectName>.fcpxml
      <ProjectName>.srt
      manifest.json          # relink manifest (see below)
      README.txt             # "open the .fcpxml; media relinks from ./media"
      media/                 # ONLY if include_media
        <filename1>
        <filename2>
    ```
  - **Media handling (user choice):**
    - `include_media = false` (**default**): project-only bundle (FCPXML + SRT + manifest).
      Tiny. The editor relinks to their own local originals. This is the common pro case.
    - `include_media = true`: download each source from R2 at the chosen quality into
      `media/`. For **very large** projects, instead of bloating the ZIP, write signed R2
      GET URLs into `manifest.json` and skip the `media/` copy (threshold configurable).
  - **manifest.json** shape:
    ```json
    {
      "project": "MyReel",
      "frame_rate": 23.976,
      "assets": [
        {"file_id": "...", "filename": "clipA.mov", "relpath": "media/clipA.mov",
         "download_url": null, "duration_ms": 12345}
      ]
    }
    ```
    (`download_url` populated with a signed R2 GET only when media is delivered by link.)
  - ZIP in **store mode** (no re-compression — video is already compressed).
  - Upload to R2 key `exports/{uuid}.zip` via `services/processing._upload_to_r2` /
    `services/r2.py`; return the key. Presign on read (`generate_presigned_get`, 24h).

---

## Phase 5 — Backend: export job + API

Reuse the render stack for A; add an export job for B/C bundling.

- **New table (migration):** `backend/migrations/0XX_exports.sql`
  - `exports(id, project_id, user_id, kind, quality, include_media, status,
    output_r2_key, error, created_at, updated_at)`
  - `kind ∈ {'mp4','rough_cut','srt'}`; `status ∈ {'queued','running','done','failed'}`.
  - Mirror the `renders` table (`016_renders.sql`) shape/CRUD.
- **New file:** `backend/app/services/export/store.py` — CRUD for `exports` (mirror
  `render/store.py`).
- **New file:** `backend/app/services/export/tasks.py`
  - `@app.task(queue="export") build_export(export_id)`:
    - Resolve the document (`resolve_document`) once.
    - `mp4` → delegate to existing `render_edit` path / `compositor.render_resolved` at the
      chosen quality preset; store output key.
    - `srt` → `srt.build_srt(...)` → upload → store key.
    - `rough_cut` → `bundle.build_rough_cut_bundle(...)` → store key.
  - Register in `backend/app/services/jobs.py` `register_tasks()`.
- **New router:** `backend/app/routers/exports.py`
  - `POST /api/projects/{id}/export` body: `{ kind, quality, include_media }` →
    create `exports` row, enqueue `build_export`, return `export_id`.
  - `GET /api/exports/{export_id}` → status + presigned `output_url` when `done`.
  - Auth via the projects ownership pattern (`_owned_project`).
  - Register in `backend/app/main.py`.

> **Note on document source:** the finished MP4 render currently scopes to an **edit
> thread**, not a project. Decide one:
> - (preferred) export operates on the project's **latest edit document** (look it up from
>   the project's thread), so cuts/framing/split-screen/captions all come from the resolved
>   doc; or
> - if a project has no edit doc yet, synthesize a transient timeline from `cut_records`
>   (per-file `src_in_ms` order + client `order`) — note cut order is **client-only** today,
>   so the export request must accept the `order: string[]` from the UI.
> Capture the chosen approach explicitly when implementing.

---

## Phase 6 — Split-screen golden-import validation (one-time, gates B)

Static split-screen is deterministic, but FCPXML transform coordinate space (origin, units,
scale semantics) differs from our normalized `{x,y,w,h}`. Validate the mapping once:

1. Build an FCPXML for a known `split_h` (and `pip`) fixture.
2. Import into **DaVinci Resolve** and **Premiere Pro**.
3. Confirm each cell lands in the correct rect (visually matches the MP4).
4. Lock the coordinate-mapping constants in `fcpxml.py` with a comment citing the verified
   convention. Add a regression test asserting the transform math for the fixture.

This is a verify-once task, not a fallback path.

---

## Phase 7 — Frontend: Export dialog + stage

- **Stage enum:** `frontend/src/stores/drive-store.ts` — add `"export"` to `ProjectStage`.
- **Sidebar:** `frontend/src/components/sidebar.tsx` — add an **Export** nav item
  (follow the frontend-design skill: black/white/grey, orange accent only).
- **Lens router:** `frontend/src/components/project-lenses.tsx` — map `"export"` →
  new `ExportView`.
- **New component:** `frontend/src/components/export-view.tsx`
  - Choices the user sees:
    - **What:** Finished video (MP4) · Rough cut (NLE) · Subtitles only (SRT)
    - **Quality:** 4K · 1080 · 720 · Source (applies to MP4 + included media)
    - **Include original media** (only when Rough cut) — default off (project-only,
      "relink to your own footage"); on = media in ZIP / signed links.
  - Flow (mirror `render-bar.tsx`): `createExport(...)` → poll `getExport(id)` →
    show download button when `done`.
- **API client:** `frontend/src/lib/api.ts` — add `createExport(projectId, {kind,
  quality, includeMedia}, token)` and `getExport(exportId, token)` + `ExportRecord` type.

---

## Phase 8 — Tests & verification

- Backend unit: `test_export_srt.py`, `test_export_fcpxml.py`, bundle assembly test
  (temp dir → ZIP contains expected entries; manifest well-formed).
- Router test: `backend/scripts/test_exports_router.py` (mirror
  `test_projects_router.py`; monkeypatch store + enqueue).
- Manual E2E on **Reel 5** (the project with a full edit doc / log):
  - Export MP4 at 1080 and 4K → both play, subtitles burned, split-screen correct.
  - Export rough cut (project-only) → open FCPXML in Resolve + Premiere; relink to local
    media; timeline order/framing/split-screen/SRT all correct.
  - Export rough cut (include media) → unzip → open FCPXML → media auto-relinks from
    `media/`.
  - Export SRT-only → opens in a text/subtitle editor with correct timings.
- Frontend: `tsc` / lint / vitest / build clean.

---

## Execution order (small steps)

1. **Phase 0** — worker wiring (unblocks everything).
2. **Phase 1** — quality presets.
3. **Phase 2** — SRT (smallest deliverable, immediately shippable as C).
4. **Phase 5 (partial)** — exports table + store + MP4 path + API + `ExportView` for
   Finished video (deliverable A end-to-end).
5. **Phase 3 + 6** — FCPXML + golden-import validation.
6. **Phase 4** — ZIP bundle + manifest (deliverable B end-to-end).
7. **Phase 7** — finish Export dialog (all three + quality + media toggle).
8. **Phase 8** — tests + Reel 5 E2E.

---

## Guardrails

- **No colour grade** in any export path (it's hidden in the UI and deferred). Do not emit
  CDL/LUT/grain/halation into FCPXML or bake grades into the MP4 for this scope.
- **No animated split-screen** — cells are static; emit fixed transforms only, never
  keyframes.
- **One source of truth** — MP4 and FCPXML/SRT must derive from the same resolved timeline
  so the rough cut matches the finished video.
- **Relative media paths** in FCPXML so the ZIP self-relinks; never embed absolute or R2
  paths in the FCPXML.
- **Don't re-compress in the ZIP** (store mode); video is already compressed.
- Reuse existing infra (`compositor`, `r2.py`, `processing._upload_to_r2`, the
  `renders`/`RenderBar` patterns) rather than new one-off machinery.
