# Plan — Cut-level A/V coupling with authoritative audio

## Goal

Make every cut a **self-contained, coupled A/V unit** whose audio is its
**authoritative audio**, decided and baked at **cut-assembly** time — instead of
re-deriving the audio source lazily at resolve/render time
(`resolve_audio_routes`). This is a **generic** change (no podcast / talking-head
assumptions): "authoritative audio" is already defined for every cut, and the
coupling rule is identical for solo clips, multicam groups, dual-system sound,
b-roll-over-interview, and music videos.

Explicitly **out of scope**: which *picture/angle* to show for a beat. That stays
the brain's generic decision. We only guarantee that whatever picture is chosen
comes welded to correct, in-sync authoritative audio.

## Problem / root cause (why the current design drifts)

A synced group is stored as **N independent files reconciled lazily**:

- Each angle is its own file with its own audio and per-angle cuts.
- Angles are glued only by **offsets applied late**: `layers.resolve()` calls
  `app/services/l3/sync/audio_route.py::resolve_audio_routes(timeline)`, which
  looks up `sync_groups`/`sync_group_members` and remaps each spine segment's
  dialogue audio to the group's authoritative source using a **single global
  per-file `offset_ms`**.

Consequences observed on the podcast (run `c8d47a66`, thread `fe089870`):

- **Lip-sync drift** at the start: when the picture is the guest angle
  (`48c93cef`) but audio is routed to the host camera (`1aedb093`) with an
  `offset_ms=800` solved at only **0.851 confidence**, the shown face drifts
  against the sound. The drift is structural: cross-file audio needs an offset,
  and one global offset per file is imprecise (and can clock-drift over a long
  take).
- The routing decision lives **downstream** (resolve time), so any fix there is
  a band-aid that never corrects the cut data itself.

## Principle (generic)

> **A cut is a coupled `(video, authoritative-audio)` unit. The coupling is a
> `(audio_file_id, audio_offset_ms)` pair baked onto the cut at assembly.
> `audio_offset_ms` is `0` when audio source == video source, and a per-cut
> measured value when they differ.**

Generic definition of "authoritative audio" per cut — no category assumptions:

- **Solo clip (no sync group):** authoritative = the clip's **own** audio →
  `audio_file_id == file_id`, `audio_offset_ms == 0`. Identity case; byte-identical
  to today for ~90% of footage.
- **Synced group:** authoritative = the group's `authoritative_audio_file_id` →
  `audio_file_id == auth_fid`, `audio_offset_ms == (per-cut refined delta)`.

Because the offset is stored as a **delta** (not an absolute span), it composes
with brain trimming: for any placed sub-span `[in_ms, out_ms]`, the coupled audio
span is `[in_ms + audio_offset_ms, out_ms + audio_offset_ms]`.

## Per-cut offset refinement (the precision fix)

For a cross-source cut (`audio_file_id != file_id`):

1. Start from the global delta (today's math, see `audio_route.py:71-73`):
   `delta = member[file_id].offset_ms - member[auth_fid].offset_ms`.
2. **Locally refine** it against this cut's own window:
   - Take the **video file's own** `rms_db` envelope over `[s, e]`.
   - Take the **authoritative file's** `rms_db` envelope over the globally-shifted
     window `[s+delta, e+delta]`.
   - Cross-correlate the two envelopes over a small search window (e.g. `±300 ms`,
     in `hop_ms` steps) to find the residual lag `r` that maximizes normalized
     correlation.
   - `audio_offset_ms = delta + r`.
3. **Guard:** if the correlation peak is weak/ambiguous (below a floor, or the
   video file has no audio envelope), keep `delta` unrefined (never refine on
   noise). Optionally store an alignment confidence for diagnostics.

Both envelopes are already available at assembly: `assemble_cut_records` receives
`audio_by_file` where `audio_by_file[fid]["rms_db"]` + `["hop_ms"]` is the
per-file envelope (see `post.py:750`, `post.py:811`).

Consequences, category-agnostic:

- **Same-source cuts** (all solo footage, and any angle coupled to its own audio):
  `offset == 0`, zero drift by definition.
- **Cross-source cuts**: aligned locally and exactly, so a loose global offset and
  clock drift can't accumulate into visible lip-sync error.

## Data model changes

Migration `backend/migrations/0XX_cut_av_coupling.sql`:

```sql
alter table cut_records
  add column if not exists audio_file_id text null,          -- coupled audio source (null = same as file_id, legacy rows)
  add column if not exists audio_offset_ms integer not null default 0,  -- add to video src ms to get audio src ms
  add column if not exists audio_align_confidence real null;  -- diagnostics for the per-cut refinement (null = not refined)
```

Semantics: `audio_file_id IS NULL` means "same-source" (use `file_id`,
`offset 0`) — so existing rows are correct without backfill.

## Component changes

### 1. New module — `backend/app/services/l3/sync/av_couple.py`
Pure helpers (mirrors the "fetch once, resolve pure" split `audio_route.py` uses):
- `authoritative_for(file_id, sync_info) -> (audio_file_id, global_delta_ms)`:
  returns `(file_id, 0)` when the file is in no group or is itself the auth;
  else `(auth_fid, member.offset - auth.offset)`.
- `refine_offset(video_rms, auth_rms, hop_ms, s, e, global_delta, *, search_ms=300)
   -> (offset_ms, confidence)`: the envelope cross-correlation above, with the
  weak-peak guard.

### 2. `backend/app/services/l3/post.py::assemble_cut_records`
- Add a param carrying resolved sync info: `sync_info_by_group` (or extend the
  existing `sync_group_by_file`) with `{group_id: {auth_fid, offsets: {file_id:
  offset_ms}}}`.
- Per cut, after `s, e` are resolved (around `post.py:846`), compute
  `(audio_file_id, audio_offset_ms, audio_align_confidence)` via `av_couple`
  (same-source → `(file_id, 0, None)`).
- Add the three fields to `CutRecord` (dataclass at `post.py:409`) + its
  `to_dict` (`post.py:481`).

### 3. `backend/app/services/l3/ingest_store.py::insert_cut_records`
- Add `audio_file_id, audio_offset_ms, audio_align_confidence` to the INSERT
  column list + values tuple (`ingest_store.py:117-163`).

### 4. `backend/app/services/l3/ingest.py`
- The run already pins outlook groups and knows members/offsets/authoritative
  (used for `authoritative_view` at `ingest.py:153-190`). Thread that group info
  (auth + per-member `offset_ms`) into `assemble_cut_records` so assembly can
  couple.

### 5. Read path — `backend/app/services/l3/cuts_v3_read.py` + `cutrecord_map.py`
- Surface `audio_file_id`, `audio_offset_ms` on the `ResolvedCut` the brain's
  map/`place` path uses, so a placed segment can carry the coupling.

### 6. `backend/app/services/l3/act.py`
- `_segments_from_cut` (`act.py:59`) and `place`/`place_span`: stamp
  `audio_file_id` + `audio_offset_ms` onto each spine segment dict it emits
  (default `file_id`/`0` when the resolved cut has none).
- `replace_audio` (`act.py:390`) still writes `audio_override` and continues to
  win over the baked coupling (escape hatch preserved).

### 7. `backend/app/services/l3/layers.py::resolve`
- Where the spine `AudioLayer` is built (`layers.py:615-630`): source the audio
  from the segment's baked coupling instead of `audio_routes`:
  - `audio_file_id = seg.get("audio_file_id") or seg["file_id"]`
  - `off = int(seg.get("audio_offset_ms", 0))`
  - `src_in = int(seg["in_ms"]) + off`, `src_out = int(seg["out_ms"]) + off`
  - `audio_override` still wins first (unchanged).
- **Legacy fallback:** if a segment has no `audio_file_id` (old edit built before
  this change), fall back to `audio_routes.get(seg_id)` then to coupled
  `file_id` — so existing edit documents render identically (no forced re-ingest
  to view old edits).

### 8. `backend/app/services/render/tasks.py`
- Keep `_resolve_audio_routes` (`tasks.py:46`) only as the legacy fallback feeding
  `layers.resolve` for old documents; new documents ignore it because segments
  carry their own coupling. Once all live edits are rebuilt on migrated runs,
  `audio_route.py` + this call can be deleted.

## Backward compatibility

- Old `cut_records` rows: `audio_file_id NULL` → treated as same-source → identical
  to today.
- Old edit documents (segments without coupling fields): `layers.resolve` falls
  back to `resolve_audio_routes` → byte-identical render.
- New re-ingested runs + new edits: fully cut-coupled; `resolve_audio_routes`
  never consulted.

## Testing

- `av_couple.refine_offset`: synthetic envelopes with a known lag recovered;
  weak/flat envelope falls back to the global delta; no-audio video file → global
  delta.
- `av_couple.authoritative_for`: solo → `(file_id, 0)`; grouped non-auth → auth +
  correct signed delta; the auth file itself → `(file_id, 0)`.
- `post.assemble_cut_records`: solo cut → `audio_file_id == file_id`, `offset 0`;
  synced cut → `audio_file_id == auth`, `offset ≈ global delta ± refinement`.
- `layers.resolve`: segment with baked coupling produces an `AudioLayer` from
  `audio_file_id` at `src + offset`; segment without coupling falls back to
  `audio_routes`; `audio_override` still wins over both.
- Regression: solo-clip edit resolves byte-identical to today.

## Rollout

1. Migration.
2. Land code behind the null-safe fallback (no behavior change until re-ingest).
3. Re-ingest projects (the podcast first) to populate the coupling fields.
4. Rebuild/regenerate the podcast edit and verify on the frontend:
   - Same-source talking-head segments show `audio_file_id == file_id`, `offset 0`
     (perfect sync).
   - Cross-source segments (guest picture under host audio, or the closing
     reaction) carry a refined `audio_offset_ms` and a sensible
     `audio_align_confidence`.
   - The resolved audio track is still continuous per block (no gaps at angle
     switches).

## Open questions / tuning knobs

- Cross-correlation `search_ms` window (±300 ms start) and the confidence floor
  for the refine-vs-fallback guard.
- Whether to also **loudness/tone-match per source** at the coupling boundary so a
  cross-source cut doesn't step in level vs its neighbors (a one-time per-angle
  dialogue-loudness normalization at conform; can be a follow-up, orthogonal to
  the sync fix — the existing `crossfade` op already smooths seams).
- Whether `audio_align_confidence` should ever *demote* a coupling back to
  same-source when the video angle has its own usable audio (only relevant if a
  future non-authoritative angle also carries good sound).
