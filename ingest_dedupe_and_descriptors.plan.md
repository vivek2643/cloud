# Ingest dedupe + duplicate-material awareness + frame-descriptor conformance

Investigation date: 2026-08-17. All numbers below are measured against the live
production database (`DATABASE_URL` in `.env`, schema `public`) and the current
working tree, not inferred. Every claim carries the evidence that proves it.

---

## PART 0 — FINDINGS (evidence first, design second)

### 0.1 The two file records

```
aad020e3-2d3f-4a19-9f30-8bdec1a13ac4  video1895834262.mp4  12131679 B  244.330667 s
  folder ce8247dc "demo trail"   created_at 2026-06-08 20:32:23.518798+00
  r2_key raw/00000000-…-0001/aad020e3-…/video1895834262.mp4
7144c9fc-86f8-4bb6-8c39-43cb86afe380  video1895834262.mp4  12131679 B  244.330667 s
  folder ce8247dc "demo trail"   created_at 2026-06-30 21:38:46.113696+00
  r2_key raw/00000000-…-0001/7144c9fc-…/video1895834262.mp4
```

Both `status='ready'`, `l1_status='ready'`, `1710x1108`, same user, **same folder**.

**Q1 — How did the same file get ingested twice?**
**A genuine second user upload, 22 days later. Not a retry, not an idempotency
failure, not a re-ingest path.** Evidence:

- The two `created_at` values are **22 days apart** (2026-06-08 → 2026-06-30). No
  retry, request replay, or transaction race spans 22 days.
- The June-30 upload was a **batch of two**: `7144c9fc` at `…46.113696` and
  `93e94ec3` (`video1864710564.mp4`) at `…46.118612` — 5 ms apart, one
  multi-file upload. The user re-dragged a folder that already contained
  `video1895834262.mp4`.
- The folder itself was created `2026-06-08 20:31:50`, 33 s before `aad020e3` —
  so June 8 was the first-ever upload into it, and June 30 added a second copy.
- Re-ingest does **not** create files. `cut_records` for these two files span 10
  distinct `ingest_run_id`s (2026-07-31 → 2026-08-11) and every run writes its
  own cut set — re-ingest duplicates *runs*, never *files*. The duplication is
  strictly at the `files` row level.
- Storage-key collision is structurally impossible: `r2_key =
  f"raw/{user_id}/{file_id}/{filename}"` (`backend/app/routers/upload.py:70`,
  `:197`) is keyed on a fresh `uuid.uuid4()`, so two uploads of identical bytes
  land at two different keys and can never collide into one.

**Q2 — Is there any dedupe at ingest at all?**
**No. Not one form of it.**

- `files` rows are created in exactly two places, both in
  `backend/app/routers/upload.py`: `presign_upload` (`:56-87`) and
  `multipart_create` (`:183-223`). Both mint `file_id = str(uuid.uuid4())` and
  insert unconditionally. The only validation is folder ownership.
- No content hash exists anywhere on the upload path. `PresignRequest` /
  `MultipartCreateRequest` (`backend/app/schemas.py:71-89`) carry
  `filename`, `content_type`, `file_size`, `folder_id` — no hash, no
  idempotency key.
- `file_size` is stored but never compared to an existing row. No
  same-folder-same-filename check. R2 ETags (`r2.py:121-151`) are used only to
  finalize the S3 multipart protocol and are never persisted.
- `backend/migrations/001_initial.sql:65-86` defines `files` with **PK on `id`
  only**, plus non-unique `idx_files_user_folder` and `idx_files_status`. No
  later migration adds a unique constraint (`003`, `020`, `032` only add/drop
  columns).
- The only "idempotency" on this path is `complete_upload`'s
  `if file_record["status"] != "uploading": raise 400`
  (`upload.py:335-336`) — that guards re-finalizing *one* file_id, not
  re-uploading the same bytes.

**Natural home for prevention:** `presign_upload` / `multipart_create` in
`upload.py`, backed by a DB-level unique constraint on `files` (application-level
checks alone would leave the second insert path as a bypass — there are two).

**Bonus defect found on the same pattern.** The two projects holding these files
were created 13 ms apart with identical `source_file_ids`:

```
8621c012-58c1-4bd5-8897-b8e8e4f24dca  2026-07-07 18:24:19.698155+00
5cd8f004-13c7-43f8-a1ed-d2f7e646fae7  2026-07-07 18:24:19.711533+00
both source_file_ids = {7144c9fc…, 93e94ec3…, aad020e3…}
```

Root cause: `find_or_create_project` (`backend/app/services/l3/projects.py:21-37`)
is a bare SELECT-then-INSERT with no transaction, no lock, no `ON CONFLICT`, and
`projects` has only a **GIN** index on `source_file_ids`
(`backend/migrations/006_edl.sql:22-35`), not a unique one. Both callers missed
the SELECT and both inserted. Those two projects have since accumulated 21 and
15 ingest runs respectively — the duplicate cost real compute.

**Q3 — Why does `dup_groups` catch speech pairs and miss every video pair?**
**Because video cut records have no comparator at all — not a scoped one, not a
mis-thresholded one. `take_group_id` is hardcoded `None` for every video cut, and
`dup_groups` reads nothing else.**

The consumer:

```1231:1239:backend/app/services/l3/footage_map.py
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for t in trees:
        for m in t.get("moments", []) or []:
            gid = m.get("take_group_id")
            # Junk is skip-by-default -- keep it OUT of the take-group /
            # coverage set even if pass 2 somehow tagged one (cuts_v3_continuity
            # .plan.md); it stays independently placeable via its own ref.
            if gid and not m.get("junk"):
                by_group.setdefault(gid, []).append(m)
```

`take_group_id` is the **sole** input. The producer for video:

```131:137:backend/app/services/vcut/store.py
        records.append(CutRecord(
            file_id=cut.file_id, src_in_ms=cut.in_ms, src_out_ms=cut.out_ms,
            kind="video", word_span=None, atom_ids=None,
            label=_short_label(cut.summary), summary=cut.summary,
            on_camera=None, junk=False, junk_reason="",
            framing=_framing_for(cut, file_seam), look={}, caption_zones=[],
            hero_ts_ms=cut.peak_ms, pace=pace, take_group_id=None, take_role=None,
```

Note line 135 (`junk=False`) is the *already-known* dead-junk-filter defect. Line
137 is the same line, one field over. The producer for speech:

```181:181:backend/app/services/vcut/speech/store.py
        pace=pace, take_group_id=rb.take_group_key, take_role=take_role,
```

fed by `take_group_key = f"speech[{sorted(b.id for b in group_beats)[0]}]"`
(`backend/app/services/vcut/speech/orchestrate.py:62`), where `group_beats` comes
from the LLM retake clusterer `segment_llm.build_take_groups` /
`run_segment_llm` (`vcut/speech/segment_llm.py:154-195`), which is explicitly
cross-file. There is **no video counterpart** to that module.

Whole-database proof of the mechanism:

| `cut_records.kind` | rows | rows with a non-null `take_group_id` |
|---|---|---|
| `speech` | 1939 | **1939** |
| `video`  | 2074 | **0** |

Zero of 2074. So the miss is not "threshold can't be met cross-file" and not
"scoped within-file" — the field the detector reads is universally `NULL` on the
video side. It is unreachable in the strict sense: no input value of any kind
makes `_annotate_dups` group two video moments.

Observed end-to-end on the real edit (run `24894c8a`, thread `34bb9967`), the
four `dup_groups` the agent actually saw were all speech:

```
928c7834  [93e94ec3:m02, 93e94ec3:m03]                        "Start with a demo"
bc657f37  [93e94ec3:m07, 93e94ec3:m10, 7144c9fc:m01, aad020e3:m00]  "Thanks for the opportunity"
d7310918  [93e94ec3:m13, 93e94ec3:m14]                        "Big players trying to automate…"
a0ab7de1  [7144c9fc:m02, aad020e3:m01]                        "Edso is an AI native video drive"
```

and every video moment on both duplicate files carried `dup_group=None`:

```
7144c9fc:m03 45100-48800   "A slide titled 'The Insight' is"   dup=None
aad020e3:m02 46100-48800   "The slide titled The Insight is"   dup=None
7144c9fc:m04 69100-73800   dup=None      aad020e3:m03 70200-74900   dup=None
7144c9fc:m05 147900-153100 dup=None      aad020e3:m04 149400-154000 dup=None
7144c9fc:m06 173200-178400 dup=None      aad020e3:m05 174200-179400 dup=None
7144c9fc:m07 235100-239600 dup=None      aad020e3:m06 236000-240600 dup=None   ← the closing shot
```

**Q4 — pHash has exactly one consumer, and it is seam-only. Confirmed.**

Every import of the module across the whole backend:

- `l1/pipeline.py:34,774,780` — the **writer** (`_stage_frame_descriptors`).
- `l1/snapshot.py:28` — a stage-name string in `_L1_STAGES`, for status display.
- `l3/observe.py:165-167` — the **only DB reader**
  (`load_descriptors(conn, file_ids)` into `EditContext.frame_descriptors`).
- `l3/feel.py:25` — the **only algorithmic consumer**: `hamming64` +
  `nearest_phash`, used at exactly two sites, `_project_scale` (`:276-280`,
  which computes the content-relative band by sampling the *same* seam pairs) and
  `join_continuity` (`:348-353`).

Both sites compare **the previous cut's `src_out_ms` against the current cut's
`src_in_ms`** — a seam between two already-placed cuts. Nothing else. There is no
call that hashes a span's own interior, no call that compares two candidate
moments, and no call anywhere in `vcut/` (verified: no `phash`/`hamming`/
`frame_descriptors` reference in the entire `vcut/` tree).

**Where a cross-file pHash duplicate check slots in:** not in `feel.py`. `feel`
is documented and built as **PURE — no DB, no LLM, no network** (`feel.py:11-14`),
operating on an already-placed timeline. Pool-level duplicate detection belongs
one layer earlier, at pool assembly:
`footage_map._annotate_dups` (`footage_map.py:1216`), called from
`assemble_map` (`footage_map.py:1358`) — the single funnel that every L3 read
(`observe.build_context:128`) passes through. See §3.

**Q5 — Blast radius.**

Candidate duplicate ingests by `(name, file_size, round(duration,3))`:

| name | size | dur (s) | copies | folders | dates |
|---|---|---|---|---|---|
| `IMG_4626.MOV` | 13181728 | 6.648 | 3 | 3 distinct | 07-23, 07-24, 07-26 |
| `IMG_4627.MOV` | 4559331 | 5.832 | 3 | 3 distinct | 07-23, 07-24, 07-26 |
| `IMG_4628.MOV` | 3792127 | 4.133 | 3 | 3 distinct | 07-23, 07-24, 07-26 |
| `IMG_4629.MOV` | 5645015 | 5.733 | 3 | 3 distinct | 07-23, 07-24, 07-26 |
| `video1864710564.mp4` | 53673656 | 683.392 | 2 | 2 distinct | 06-11, 06-30 |
| `video1895834262.mp4` | 12131679 | 244.331 | 2 | **same folder** | 06-08, 06-30 |

**6 duplicate groups · 16 files involved · 10 redundant files · out of 168 total
files (163 video) = 6.0 % of the library.** 2 projects and **14 edit threads**
have ≥2 members of the same duplicate group in scope.

**This is systemic, not a one-off.** The `IMG_462x` set is the same four clips
re-uploaded into a *new folder* on three separate days — the user's normal
workflow reproduces it.

pHash confirmation of all 15 candidate pairs (same-timecode Hamming over shared
sample points, 500 ms hop):

```
IMG_4626  0392bf97 vs e32e7e4e: n=13  mean=5.231 max=16 zero=0.077   ← NOT confirmed
IMG_4626  0392bf97 vs b4f33470: n=13  mean=5.077 max=16 zero=0.154   ← NOT confirmed
IMG_4626  e32e7e4e vs b4f33470: n=13  mean=0.462 max=4  zero=0.846   confirmed
IMG_4627  all 3 pairs:                mean=0.167-0.833 max=2         confirmed
IMG_4628  all 3 pairs:                mean=0.000-0.250 max=2         confirmed
IMG_4629  all 3 pairs:                mean=0.000-0.182 max=2         confirmed
video1864710564  6cc2afc6 vs 93e94ec3: n=1367 mean=0.152 max=2 zero=0.924  confirmed
video1895834262  aad020e3 vs 7144c9fc: n=489  mean=0.131 max=2 zero=0.935  confirmed
```

A **pHash-only sweep** (no name/size input, prefilter `|Δ sample_count| ≤ 2`,
accept `mean < 2.0 bits` over `≥10` shared points) over all
114 descriptor rows returns exactly **9 pairs — the 13 confirmed pairs' transitive
set, and no false positives at all.** Content alone is a sufficient and precise
detector here.

**Two calibration facts that must shape the design:**

1. `0392bf97` has the **same filename, same 13181728-byte size, same 6.648333 s
   duration** as its two siblings, yet its proxy's pHashes sit **mean 5.08–5.23
   bits / max 16** away from them, with only 8–15 % of samples identical. I
   slid the sequences ±6 hops looking for a time offset: the best alignment is
   offset 0 (mean 5.23); every shifted alignment is ~24 bits. So it is **not** a
   timing offset — it is the *same source re-encoded into a different proxy*.
   **A pHash-only detector with a tight threshold has a real false-negative rate
   on true byte-duplicates.** Only a raw-bytes hash is authoritative for
   "identical file".
2. `file_size` alone is not a safe key: size `1241` is shared by two files with
   **two different names**.

And a third fact that kills the obvious "fix":

3. Comparing the two duplicate spans **fraction-aligned** (sample both spans at
   the same normalised position, as any naive span-vs-span comparator would)
   gives:

```
aad020e3:m02 vs 7144c9fc:m03  fraction-aligned mean= 2.89 max=20 | same-timecode mean=0.44 max=2
aad020e3:m03 vs 7144c9fc:m04  fraction-aligned mean= 6.67 max=30 | same-timecode mean=0.00 max=0
aad020e3:m04 vs 7144c9fc:m05  fraction-aligned mean= 2.67 max=12 | same-timecode mean=0.00 max=0
aad020e3:m05 vs 7144c9fc:m06  fraction-aligned mean= 2.89 max=26 | same-timecode mean=0.00 max=0
aad020e3:m06 vs 7144c9fc:m07  fraction-aligned mean=10.22 max=32 | same-timecode mean=0.22 max=2
```

The two ingest runs' segmenters put the boundaries ~900 ms apart, so
fraction-alignment shears the comparison. **A span-level pHash comparator using
`feel.py`'s own `_PHASH_SEAMLESS_BITS = 4.0` band would classify the closing-shot
pair (mean 10.22) as different material.** Alignment must be established at the
**file** level and then applied to spans — never derived from the spans
themselves. This is the single most important design constraint in this plan.

### 0.2 Frame descriptors — actual state

**Q1 — Committed? Applied?**

- **Applied: yes.** `public.frame_descriptors` exists with the exact 5 columns
  from the migration, and `public.schema_migrations` records
  `055_frame_descriptors.sql`, checksum
  `20cf5b006373b9024419a47f2d7694b7464764f25b79faad2b1bae106d29a53a`, applied
  `2026-08-13 21:27:12.939725+00`. The on-disk file's `db_migrations.checksum()`
  is byte-identical to that — **no drift**.
- **Committed: yes, but not on this branch.** `055_frame_descriptors.sql` is in
  the index (`git ls-files --stage` → blob `731863b4`) but **not in `HEAD`**
  (`git cat-file -e HEAD:… ` → "exists on disk, but not in 'HEAD'"). It *is*
  committed on `origin/main` = `5428c5b feat(l1): persist per-frame
  perceptual-hash (pHash) descriptors`, and on local branch
  `l1-frame-descriptors`. The checked-out branch is `local-dev-isolation`
  (`3337c31`), which predates that commit — hence the "uncommitted" report. The
  original report was correct about this working tree and wrong about the repo.

  **The real, worse gap:** commit `5428c5b` contains only the WRITE side
  (`l1/frame_descriptors.py`, `frame_descriptors_params.py`, `l1/pipeline.py`,
  `l1/snapshot.py`, migration `055`, its test). The **READ side is committed
  nowhere.** `git show origin/main:backend/app/services/l3/feel.py | rg
  "join_continuity|phash|hamming|descriptors"` returns **nothing**, and
  `origin/main:backend/app/services/l3/observe.py` never mentions
  `frame_descriptors`. `join_continuity` and `EditContext.frame_descriptors`
  exist **only as uncommitted working-tree modifications**
  (`feel.py` +189 lines, `observe.py` +296 lines vs `HEAD`).

**Q2 — Populated?**

```
frame_descriptors rows: 114     all 114 have >0 samples
writes: 2026-08-14 05:47:28 → 05:54:13   (one backfill pass, ~7 min)
hop_ms=500, schema_version=1 for all 114;  samples per file: min 5, max 1411
```

Coverage is **complete for every eligible file**:

```
video files, status='ready', r2_proxy_key not null   : 114
of those, with a frame_descriptors row               : 114   (100 %)
video files total                                    : 163
video files WITHOUT a row                            :  49
  └─ 46 × status='uploading', l1_status='pending', r2_proxy_key IS NULL
  └─  3 × status='processing', l1_status='failed',  r2_proxy_key IS NULL
```

All 49 lack a proxy to hash. The backfill landed. The three files in question:

```
7144c9fc  hop_ms=500  489 samples  2026-08-14 05:51:45
aad020e3  hop_ms=500  489 samples  2026-08-14 05:47:28
93e94ec3  hop_ms=500 1367 samples  2026-08-14 05:52:25
```

(Side-proof of the duplication: `7144c9fc` and `aad020e3` both have
`t_ms=0 → 10207869954323698386`. Identical 64-bit hash, distance 0.)

**Q3 — Do descriptors actually reach `feel.join_continuity` at runtime?**
**Yes — proven twice, on real data, not by reading code.**

Direct re-execution of the production read path
(`observe.build_context(file_ids, run_id='24894c8a…')` → `feel.simulate` →
`feel.join_continuity`) against the *actual persisted timeline* of thread
`34bb9967` returned real Hamming values for every seam:

```
ctx.frame_descriptors: 7144c9fc n=489, 93e94ec3 n=1367, aad020e3 n=489
1→2 cut  same_clip=false pic_bits=36 pic_scale=0.944
2→3 cut  same_clip=true  src_gap=15454   pic_bits=26 pic_scale=0.944
3→4 cut  same_clip=true  src_gap=11875   pic_bits=34 pic_scale=0.944
4→5 cut  same_clip=true  src_gap=360784  pic_bits=36 pic_scale=0.944
5→6 cut  same_clip=false pic_bits=24 pic_scale=0.944
```

Every `pic_bits` is a real integer, so both `_seam_phash` lookups resolved for
all 5 seams — the `hA is None or hB is None` fail-open at `feel.py:350-351` fired
zero times. `pic_scale=0.944` further proves `_project_scale` had real distances
to take a median over (it returns exactly `1.0` when it has none).

Independent confirmation from the live agent's own persisted trace for that
thread: `edit_turns` contains

> `"feel": "5 cuts, 11.4s total, ~1.99 words/s when talking; jump-cut at 4→5 (same shot's framing, cut discontinuously within it)."`

That string is emitted only by `feel.narrate()` (`feel.py:122-126`) when
`join_continuity` returns `kind == "jump-cut"`, which is reachable **only**
through `feel.py:356` — i.e. only when both pHashes were present and the distance
fell inside the same-shot band. **The grader ran on real pixels on 2026-08-17.**

**Q4 — Is continuity grading live or silently blind?**

**Live in this working tree — provably. Non-existent in `origin/main`.** Split
plainly:

| | descriptor write | descriptor read / `join_continuity` |
|---|---|---|
| production DB | applied `08-13`, 114 files populated `08-14` | n/a |
| `origin/main` (`5428c5b`) | **present** | **absent — no `join_continuity` at all** |
| working tree (`local-dev-isolation` + edits) | present | **present and verified working** |

So the failure mode is not the one feared. It is *not* silently blind here; it is
**verified live here and undeployable from `origin/main`**. Anyone deploying
`origin/main` today gets descriptor writing with no continuity grading whatsoever
— not fail-open, just gone.

**But the fail-open observability hole is real and must still be closed.** For
the final 6-cut state of thread `34bb9967` the grader produced five genuine
verdicts (36/26/34/36/24 bits) and **not one of them appears anywhere in the
persisted record**:

```
edit_documents.document mentions of 'pic_bits'/'pic_scale'/'seamless'/'continuity': 0 / 0 / 0 / 0
edit_documents.diagnostics == {"engine": "agentic_loop"}
```

`narrate()` emits a continuity clause **only when a jump-cut is found**. A clean
edit and an ungraded edit produce byte-identical output. That is precisely the
condition the plan must eliminate.

---

## PART 1 — DESIGN PRINCIPLES

The user has rejected band-aids and asked for root-cause, architecture-level
fixes. Four rules bind everything below.

**P1 — Separate CONTENT from PLACEMENT.** Today `files` conflates "these bytes"
with "this thing in this folder with this name". Every duplicate in §0.1 Q5 is
the same content at a different placement. Once content has its own identity,
duplicate ingest stops being a thing you *detect* and becomes a thing that
*cannot be represented*.

**P2 — Invariants live in the database, not in a handler.** There are already two
`files` insert sites and one racing `find_or_create_project`; an application-level
check is a check one new code path can bypass. Every uniqueness claim in this plan
is a unique index, with the application code reading `ON CONFLICT` rather than
pre-checking.

**P3 — Two independent identity signals, never blended into one threshold.**
- **`content_sha256` over the raw bytes** → *authoritative* for "identical file".
  Exact, zero false positives, zero false negatives.
- **pHash sequence similarity over the proxy** → *heuristic* for "same material"
  (catches re-encodes, re-exports, trims that byte-hashing misses).

They must stay separately labelled and separately consumed.
`0392bf97` is the proof: byte-identical to its siblings (same size, same name,
same duration) yet **5.08–5.23 bits** away in pHash. Blend them into one score
with one cutoff and you lose that duplicate. Keep them separate and sha256
catches it while pHash abstains.

**P4 — Alignment comes from file identity, never from spans.** Established in
§0.1 Q5 fact 3: the closing-shot pair reads **0.22 bits** at matched timecodes and
**10.22 bits** fraction-aligned. Any comparator that samples two spans and
compares them is measuring segmenter boundary jitter, not content.

---

## PART 2 — (a) PREVENT DUPLICATE INGEST GOING FORWARD

### 2a.1 Content identity on `files`

New migration `056_file_content_identity.sql`:

```sql
alter table public.files
    add column if not exists content_sha256 text,
    add column if not exists content_sha256_source text;  -- 'client_claimed' | 'server_verified'

-- P2: the invariant is the index, not the handler. Partial so the 46 rows
-- still in status='uploading' (no bytes yet) and every pre-backfill row stay
-- legal, and so a failed/aborted upload never blocks a legitimate retry.
create unique index if not exists uq_files_user_content
    on public.files (user_id, content_sha256)
    where content_sha256 is not null and status = 'ready';

create index if not exists idx_files_content_sha256
    on public.files (content_sha256) where content_sha256 is not null;
```

Scope decision: `(user_id, content_sha256)`, not global. Two users uploading the
same stock clip are two independent assets — cross-user dedupe is a storage
optimisation with privacy and lifecycle consequences and is explicitly out of
scope.

### 2a.2 Client-claimed hash as a *lookup key*, server hash as *truth*

`PresignRequest` / `MultipartCreateRequest` gain optional
`content_sha256: str | None`. The browser already streams the whole file to
compute nothing; a `crypto.subtle.digest` pass over the same `ReadableStream`
costs one extra read.

`presign_upload` (`upload.py:56`) becomes:

1. If `content_sha256` is present, look for an existing row with
   `(user_id, content_sha256)` and `status='ready'`.
2. On hit, **insert no row and issue no upload URL**. Return
   `PresignResponse(file_id=<existing id>, upload_url=None, deduped=True,
   existing_placement={folder_id, name})`. The client skips the PUT entirely and
   shows "this file is already in your drive" with a link.
3. On miss, proceed exactly as today, storing `content_sha256` with
   `content_sha256_source='client_claimed'`.

`multipart_create` (`upload.py:183`) gets the identical branch. Both sites, one
shared helper — because two divergent implementations is how this bug class got
here.

**The client hash is never trusted as truth.** L1 already downloads the raw
object (`_download_from_r2`, used by `l1/pipeline.py`). Add a
`content_sha256` stage there that hashes the downloaded bytes and:

- writes `content_sha256_source='server_verified'` on match;
- on **mismatch**, overwrites `content_sha256` with the true digest, logs at
  `ERROR`, and lets the unique index arbitrate. If the true digest collides with
  an existing ready row, the row is marked and surfaced as a duplicate rather
  than silently kept.

This is what makes the client hash a cache key rather than a security boundary.

### 2a.3 Placement vs content (the deeper half of P1)

The dedupe response above is honest for the same-folder case (`video1895834262`)
but wrong for the `IMG_462x` case, where the user genuinely wants those clips in
a *new* folder. The correct model separates the two concerns:

```
file_contents   content_sha256 (PK) · r2_key · byte_size · duration_ms
                width · height · mime_type · first_seen_at
files           id (PK) · user_id · folder_id · name · content_sha256 (FK)
                status · created_at
```

Then "same clip in two folders" is **two placements over one content**, L1 runs
once per content, and every derived signal (`frame_descriptors`, `scene_cuts`,
`transcripts`, `motion_dynamics`, `color_stats`, `audio_features`, …) keys off
content. Duplicate *analysis* becomes unrepresentable, and the redundant-pool
problem in §0.1 disappears at the root rather than being patched at the read
side.

**Staging, honestly.** That migration re-keys ~10 L1 tables and is not a
one-sitting change. Sequence:

- **Phase A (this plan):** `files.content_sha256` + unique index + presign
  dedupe + backfill. Full prevention for the same-user case, no table re-keying.
- **Phase B (this plan):** read-side content grouping (§3 and §4). This is what
  makes (c) work, and it does **not** require Phase C.
- **Phase C (separate plan, explicitly out of scope here):** extract
  `file_contents`, re-key L1 signal tables, make placement a view over content.

Phase C is named so it isn't quietly forgotten, and deferred so this plan can
land.

### 2a.4 Fix the project race with the same mechanism

```sql
-- 056, same file
create unique index if not exists uq_projects_user_source_set
    on public.projects (user_id, source_file_ids);
```

`find_or_create_project` (`projects.py:21-37`) becomes
`insert … on conflict (user_id, source_file_ids) do nothing returning id`, then
re-select on empty. **Requires a one-time dedupe of existing rows first** — the
`8621c012` / `5cd8f004` pair, and any other collisions, must be merged (keep the
earliest, repoint `ingest_runs.project_id`) or the index creation fails. That
failure is a feature: it forces the existing damage to be dealt with.

**False-genericity trap for (a):** an implementation that adds
`content_sha256` and then *only ever reads it in `presign_upload`*. It looks
architectural — there's a hash column, there's an index — but if the two insert
sites don't share one helper, or if the unique index is omitted "because the
handler already checks", the invariant is advisory and the next insert path
re-opens the hole. **The reviewable test is: with the handler check deleted, a
direct `INSERT` of a second ready row with the same `(user_id, content_sha256)`
must still fail.** If it succeeds, the fix is application-level cosmetics.
A second, subtler variant: making the column `NOT NULL` and the index total
rather than partial — that breaks every one of the 46 `status='uploading'` rows
and legitimate re-upload-after-abort, and will get "fixed" by dropping the
constraint.

---

## PART 3 — (b) DETECT AND GROUP DUPLICATE MATERIAL THAT ALREADY EXISTS

### 3b.1 Backfill `content_sha256` for all 168 existing files

Script `backend/scripts/backfill_content_sha256.py`: for each file with
`r2_key is not null`, stream the object from R2 and hash it (no local temp file,
constant memory). ~168 objects; the largest is 53.7 MB. Idempotent, resumable,
writes `content_sha256_source='server_verified'`.

This is the authoritative pass. Run it **before** creating
`uq_files_user_content` — and expect the index creation to fail on the six known
groups. See §3b.3 for what to do then.

### 3b.2 Persist content groups (grouping is the deliverable, not a side effect)

```sql
-- 056
create table if not exists public.content_groups (
    id           uuid primary key default uuid_generate_v4(),
    user_id      uuid not null,
    canonical_file_id uuid not null references public.files(id) on delete restrict,
    basis        text not null,     -- 'identical_bytes' | 'same_material'
    confidence   text not null,     -- 'exact' | 'high'
    evidence     jsonb not null default '{}'::jsonb,
    created_at   timestamptz not null default now()
);

create table if not exists public.content_group_members (
    group_id     uuid not null references public.content_groups(id) on delete cascade,
    file_id      uuid not null references public.files(id) on delete cascade,
    offset_ms    int  not null default 0,   -- this file's time offset vs canonical
    primary key (group_id, file_id)
);
create index if not exists idx_cgm_file on public.content_group_members (file_id);
```

`offset_ms` is where P4 lives structurally: **the alignment is a property of the
file pair, stored once**, and every span-level question reads it. For the six
known groups it is 0 (proven by the ±6-hop slide in §0.1 Q5 fact 1). Storing it
rather than assuming zero is what stops this from being tuned to the observed
case — a genuine re-export with a 2 s handle would populate a non-zero offset
and everything downstream still works.

`canonical_file_id` election is deterministic and boring: **earliest
`created_at`, tie-break lowest `id::text`.** No quality heuristic, because
"which proxy is better" is exactly the kind of tunable that rots. For
`video1895834262` the canonical is `aad020e3` (06-08), the redundant is
`7144c9fc` (06-30).

### 3b.3 Two detectors, run in order, separately labelled

`backend/scripts/detect_content_groups.py` (also importable as a service so it
can run incrementally at L1 completion):

**Detector 1 — `identical_bytes` (authoritative, `confidence='exact'`).**
Group on `content_sha256`. `offset_ms=0` by definition. Expected result:
all 6 groups, all 16 files, 10 redundant. `evidence = {"sha256": "…"}`.

**Detector 2 — `same_material` (heuristic, `confidence='high'`).**
For files *not* already grouped by detector 1: compare pHash sequences with an
offset search (the ±N-hop slide already validated above), accept when the best
alignment clears both a distance and a coverage bar, record the winning offset.
This is the detector that would find a re-encode, a trimmed re-export, or a
different-container copy — none of which sha256 sees.

Both write to the same tables with different `basis`, so a consumer can require
`basis='identical_bytes'` for anything destructive and accept `same_material`
for advisory purposes.

**Note on the pHash detector's parameters.** I am deliberately *not* writing
"threshold = 2.0 bits" into this plan as a constant. The observed values
(0.00–0.83 mean for true duplicates; 5.08–5.23 for a re-proxied byte-duplicate;
~24 for a misaligned same-file comparison; 36 as `feel.py`'s own
"clearly different" anchor) say the separation is wide, but the bar must be
derived the way `feel._project_scale` already derives its bands —
content-relative — and it must be **stated as a calibration artifact with the
corpus it was measured on**, so the next person can re-derive it rather than
inherit a magic number. Detector 2's output is advisory-only precisely because
its bar is empirical.

### 3b.4 Remove, merge, or group? — **Group. Do not delete. Offer merge as an explicit, gated user action.**

**Decision: group only, in the default path. Never auto-delete, never auto-merge.**

Defence, in order of force:

1. **Existing edits reference the redundant files, and those references are not
   FK-protected.** 14 edit threads have ≥2 members of a duplicate group in
   `edit_threads.file_ids`, and `edit_documents.document->'timeline'` stores
   `{"ref": "aad020e3:m06", "file_id": "aad020e3-…"}` as **plain JSON**. The
   thread `34bb9967` timeline's closing shot is literally
   `aad020e3:m06`. Deleting `aad020e3` would cascade
   `frame_descriptors` (`055_frame_descriptors.sql:17`,
   `on delete cascade`), `cut_records`, and the rest of L1 — while leaving that
   JSON pointing at nothing. The edit would render with a hole and no error.
   There is no database mechanism that would catch it.
2. **Deletion is irreversible; grouping is not.** A wrong group is one `DELETE
   FROM content_group_members` away from undone. A wrong delete costs the R2
   object and every derived signal.
3. **Which copy is "the duplicate" is not always obvious.** `0392bf97`'s proxy
   differs measurably from its two byte-identical siblings' proxies. Any
   automated merge has to pick a survivor, and the honest answer is that all
   three are equally valid.
4. **Grouping already delivers the whole benefit for the stated problem.** The
   complaint is "the agent's pool is inflated and it can't tell two refs are the
   same shot." A group plus pool suppression (§4) fixes exactly that. Deletion
   adds only storage savings — 10 files out of 168 — which is not worth an
   irreversible operation on user data.

**The gated merge path** (opt-in, per group, user-initiated):
`POST /api/files/content-groups/{id}/merge` may proceed only when a
**precondition query returns zero rows**: no `edit_threads.file_ids` and no
`edit_documents.document` anywhere references a non-canonical member. If any
reference exists, the endpoint returns 409 listing them. It never rewrites an
edit document to repoint refs — silently editing a user's saved edit is worse
than leaving a duplicate.

**Storage stays.** The R2 objects for non-canonical members are not deleted even
on merge, for the same reason.

**False-genericity trap for (b):** a "generic duplicate detector" that takes
`(name, file_size, duration)` as its candidate key and calls pHash merely to
confirm. It reads as principled — content-verified! — but the *recall* is
entirely determined by the name/size join, which is the heuristic the user
explicitly rejected, and it fails the moment a duplicate is renamed or
re-encoded to a different size. My sweep proves the content-only path is
sufficient here: the pHash-only sweep found **all 9** true pairs and **zero**
false positives with no name or size input at all. Name/size may be used as a
*cheap prefilter for cost* (and `|Δ sample_count| ≤ 2` is a better, name-free one)
but must never be the thing that decides. **The reviewable test: rename one file
in a fixture group and change nothing else — the detector must still group it.**

---

## PART 4 — (c) MAKE THE EDITORIAL AGENT AWARE THAT TWO REFS ARE THE SAME SHOT

This is the highest-value fix and the one that survives even if no duplicate is
ever removed. It has two independent layers.

### 4c.1 Layer 1 — content-identity duplicate links (exact, no pixel comparison)

`footage_map._annotate_dups` (`footage_map.py:1216`) today reads one key,
`take_group_id`, which is `NULL` for 2074/2074 video cuts. Generalise it to
consume **a list of grouping providers**, each yielding
`(group_key, kind, members)`:

- `take` — today's `take_group_id` (unchanged, still LLM-clustered speech takes).
- `outlook` — today's `outlook_group_id` (unchanged).
- **`same_source` — new.** For every pair of moments whose `file_id`s belong to
  the same `content_groups` row, project both source spans into the canonical
  timeline using `content_group_members.offset_ms`, and link them when the
  projected spans overlap beyond a stated fraction.

For the closing shot: `aad020e3:m06` = 236000–240600 and `7144c9fc:m07` =
235100–239600, both `offset_ms=0`, overlap 235100..239600 ∩ 236000..240600 =
236000..239600 = **3600 ms of a 4600 ms span = 78 % IoU**. Linked, with **zero
pHash comparison and zero pixel threshold**. Same for the other four pairs
(m02↔m03, m03↔m04, m04↔m05, m05↔m06).

Why this is the right mechanism and not a threshold in disguise: the *identity*
claim ("these are the same bytes") is exact and comes from sha256. The only
tunable is the temporal-overlap fraction, which answers a genuinely different
question ("do these two spans of the same material cover the same moment?") and
is the same class of parameter as the existing `_RUN_GAP_MS = 500`
(`footage_map.py:131`) and `feel._JOIN_CONTIG_MS = 500`. It is a time-overlap
rule, not a similarity score.

Because it is keyed on content identity rather than on `take_group_id`, it works
for `video`, `speech`, and any future kind — and it works for the `IMG_462x`
triples in whatever project they land in.

### 4c.2 Layer 2 — a real video shot-group comparator (the missing symmetric half)

Layer 1 only fires when the two refs come from *the same material*. It says
nothing about a genuine second angle or a retake of a video-only beat. That gap
is `vcut/store.py:137`.

Add `backend/app/services/vcut/video/shot_groups.py` — the structural mirror of
`vcut/speech/segment_llm.py` + `vcut/speech/orchestrate._resolve_take_group`:

- Input: all resolved video cuts for the ingest run, across **all** files, plus
  `load_descriptors` for those files.
- Cluster by pHash distance between each cut's own interior samples (medoid per
  cut, so a moving shot is represented by its typical frame rather than its
  endpoints), across files.
- Stamp `take_group_id` / `take_role` at `vcut/store.py:137`, replacing the
  hardcoded `None`.

This is what makes `dup_groups`' video side *reachable at all*. It also means
`_annotate_dups` needs no video special-case: the field it already reads finally
has a producer.

**Critical caveat, and the reason Layer 2 must not be the primary fix for the
observed bug:** as proven in §0.1 Q5 fact 3, a span-level pHash comparator would
score the closing-shot pair at **mean 10.22 bits** — outside `feel.py`'s own
4-bit seamless band — because the segmenter boundaries differ by 900 ms. Layer 2
would very plausibly **miss** the exact case that prompted this investigation.
Layer 1 catches it deterministically. Ship Layer 1 first; Layer 2 is the general
capability, not the fix.

### 4c.3 What the agent actually sees

Three changes, all in the existing rendering path:

1. **A distinct inline marker.** `_moment_line` (`footage_map.py:1104-1191`)
   already renders `alt_pic` via `_alt_pic_segment` (`:1006`). Do **not** reuse
   it: `_alt_pic_segment`'s TAKE branch deliberately drops an alternate showing
   the same on-camera person (`:1015-1017`), which would silently drop an
   identical-material twin — the worst possible outcome. Add a separate
   `_same_source_segment` rendering e.g.
   `· same-source→7144c9fc:m07 (identical material, canonical)`, always shown,
   never collapsed.
2. **Pool suppression by default, addressability preserved.** Non-canonical
   members of a `basis='identical_bytes'` group are excluded from the default
   selectable pool (like `junk`, which "stays IN the map (labeled), never
   dropped" — `footage_map.py:1105-1109`). They remain placeable by explicit
   ref, so no existing edit breaks. This is what actually deflates the pool: had
   it been live, `aad020e3`'s five video moments would not have been offered and
   `aad020e3:m06` could not have been chosen by accident.
3. **A compile-time hard check.** At resolve/compile, two cuts from the same
   `identical_bytes` group with overlapping canonical-projected spans placed in
   one timeline is a **diagnostic on the document**, not a silent pass. Warning,
   not rejection — the user may deliberately want a repeated shot, and this plan
   does not get to overrule them. But it must be *recorded*.

**False-genericity trap for (c):** adding a `same_source` provider that computes
the group **inside `_annotate_dups` from the moments in hand** — comparing spans,
or names, or pHashes of the moments being rendered. That looks generic (it's a
provider! it's pluggable!) but it re-derives identity per call from the weakest
available evidence, and it is exactly the shape that produces the 10.22-bit miss.
The provider must be a **pure read of `content_groups` / `content_group_members`**
— identity established once, by sha256, stored, with an explicit `offset_ms`.
**The reviewable test: delete the `content_groups` rows and the marker must
disappear; add a row for two unrelated files and it must appear.** If the marker
survives with no group rows, the logic is inline and tuned.
A second variant: hardcoding `offset_ms = 0` "since duplicates are always
aligned". True for all six observed groups, false for the first re-export.

---

## PART 5 — FRAME DESCRIPTOR / MIGRATION REMEDIATION

Nothing is missing in the *data*. What is missing is **deployability** and
**observability**.

### 5.1 Land the read side (the actual gap)

`join_continuity`, `_project_scale`, `EditContext.frame_descriptors`, and
`observe`'s `load_descriptors` call exist only as uncommitted working-tree
changes and are absent from `origin/main`. Commit them together with `055`
(already on `origin/main`) so that no ref exists in which the reader ships
without the table or the table ships without the reader. Landing `l3/feel.py`'s
`from app.services.l1.frame_descriptors import hamming64, nearest_phash`
(`feel.py:25`) onto a branch lacking `l1/frame_descriptors.py` is an
`ImportError` at request time — mechanically prevented by keeping them in one
commit.

### 5.2 Migration conformance test (generic, not about 055)

`backend/scripts/test_migration_conformance.py`, run in CI:

1. Every `backend/migrations/*.sql` on disk is **tracked by git** (`git ls-files`),
   so a migration can never be applied from a working tree and then lost.
2. Every `*.sql` in the **committed tree** is present in `schema_migrations` for
   the target database.
3. **Every filename in `schema_migrations` has a corresponding file on disk.**
   This case is currently *silently ignored*: `db_migrations.pending()`
   (`db_migrations.py:127-137`) computes `not_applied` from on-disk files and
   `drifted` from the intersection — a row in the ledger with no file produces
   neither, so `assert_up_to_date` stays green while the schema contains DDL that
   no longer exists in the repo. That is the exact blind spot that let this
   situation persist.

This fixes the class, not the instance.

### 5.3 Make fail-open observable — absence must stop looking like cleanliness

Today `feel.py:350-351` returns `kind="cut"` when either pHash is missing, and
`narrate()` (`:122-126`) emits a continuity clause only when a jump-cut is found.
Proven consequence: the 6-cut final state of thread `34bb9967` was graded with
real values (36/26/34/36/24 bits) and the persisted document contains **zero**
mentions of `pic_bits`, `pic_scale`, `seamless`, or `continuity`. "Graded, clean"
and "never graded" are the same output.

Three changes:

1. **Every verdict states its provenance.** Add `pic_source` to each
   `join_continuity` dict: `"contiguous"` (step-1 fast path, no pixels needed),
   `"graded"` (both hashes resolved), `"no_descriptor_prev"` /
   `"no_descriptor_cur"` (the fail-open, naming *which* side was missing).
   `pic_bits=None` currently conflates the fast path with the failure; those are
   completely different facts.
2. **A per-turn coverage counter in `diagnostics`.** `observe` records
   `{"continuity": {"seams": N, "contiguous": a, "graded": b, "no_descriptor": c,
   "files_missing_descriptors": [...]}}` into the document's `diagnostics`
   (today: `{"engine": "agentic_loop"}`). A reader can then tell "5 seams, 5
   graded, 0 jump-cuts" from "5 seams, 0 graded" at a glance, forever.
3. **A runtime assertion for the impossible state.** When ≥1 non-contiguous seam
   fails open **while every file in `ctx.file_ids` has a `frame_descriptors`
   row**, that is not a data gap — it is a bug (wrong `file_id`, a `nearest_ms`
   too tight, an empty `phashes` array, a row-factory mishap of the kind
   `db.py:102-108` documents). Log `ERROR` and surface it in `diagnostics`.
   Distinguish it from the legitimate case (a file genuinely has no row, e.g. one
   of the 3 `l1_status='failed'` files) which stays an `INFO`-level, recorded
   fact.

### 5.4 Descriptor coverage for the 49 uncovered files

46 are `status='uploading'` with no proxy — nothing to do; they will get
descriptors when they complete L1 (`STAGES_DESCRIPTORS`,
`l1/pipeline.py:53`, `:931`). 3 are `l1_status='failed'`. Add a coverage query to
the same conformance test: **every `status='ready'` video with a
`r2_proxy_key` must have a `frame_descriptors` row with `>0` samples and
`schema_version = frame_descriptors.SCHEMA_VERSION`.** Currently 114/114 — so
this assertion is GREEN today and will catch the next silent write-guard
regression (`_stage_frame_descriptors` returns early and writes nothing on
`has_frames=False`, `l1/pipeline.py:776-780`).

---

## PART 6 — VERIFICATION

Every assertion below asserts a **delivered effect** on real or realistic data,
states its **revert signature**, and — where it guards a new code path — includes
a **reachability assertion**. No fixture counts as proof until it has been
**observed RED against HEAD** (`3337c31`, or `origin/main` `5428c5b` where the
read side is concerned). "Test passes" is not evidence; "test failed before the
change, for the stated reason" is.

### V1 — Duplicate ingest is prevented (fix a)

**Assert (effect):** two `presign` calls with the same `content_sha256` and the
same `user_id` produce **exactly one** `files` row; the second returns
`deduped=True, upload_url=None` and the *pre-existing* `file_id`.
**Assert (DB-level invariant, P2):** with the handler check bypassed entirely, a
direct `INSERT` of a second `status='ready'` row with the same
`(user_id, content_sha256)` raises `UniqueViolation`.
**Assert (no collateral):** 46 concurrent `status='uploading'` rows with
`content_sha256 IS NULL` all insert fine; an aborted upload followed by a
re-upload of the same bytes succeeds.
**RED at HEAD:** the column and index do not exist — the test errors on
`content_sha256` being unknown. That is the RED.
**Revert signature:** revert §2a.1's index → the direct-INSERT assertion passes
where it must fail (two ready rows coexist). Revert §2a.2's handler → the
presign assertion sees two `files` rows and two distinct `file_id`s.
**Reachability:** assert the dedupe branch executed, not merely that the
row count is 1 — e.g. `upload_url is None`. A branch that never runs and a
branch that runs and finds nothing both leave one row.

### V2 — Existing duplicates are grouped correctly (fix b)

**Assert (effect, real data):** after backfill + detection,
`content_groups` contains **6** `basis='identical_bytes'` groups with **16**
members total and **10** non-canonical members; the `video1895834262` group has
canonical `aad020e3` (earliest `created_at`) and member `7144c9fc` with
`offset_ms=0`; the `IMG_4626` group contains **all three** files including
`0392bf97` — the one pHash alone rejects at 5.08–5.23 bits.
**Assert (rename-invariance, anti-name-matching):** in a fixture, rename one
member and perturb nothing else — it must still group. Then change its bytes by
one byte — it must **not** group under `identical_bytes`.
**Assert (deletion is not performed):** the detection run leaves
`count(*) from files` unchanged at 168 and every `r2_key` intact.
**Assert (merge gate):** `merge` on the `video1895834262` group returns **409**
and names thread `34bb9967` — because its timeline references `aad020e3:m06`.
**RED at HEAD:** `content_groups` does not exist. Additionally, a probe asserting
"some mechanism groups `7144c9fc` with `aad020e3`" must be shown failing against
HEAD before the fix.
**Revert signature:** revert the detector → 0 groups, and the rename-invariance
test cannot even run. Revert only the sha256 detector and keep pHash → the
`IMG_4626` group loses `0392bf97` and drops to 2 members; that specific count is
the signature that the authoritative detector is gone.

### V3 — The agent is aware two refs are the same shot (fix c) — the load-bearing test

**Assert (effect, on the real run):** for
`footage_map.assemble_map(['93e94ec3…','7144c9fc…','aad020e3…'],
run_id='24894c8a…')`:
- `dup_groups` contains a group whose members include **both**
  `7144c9fc:m07` and `aad020e3:m06`, with `basis='identical_bytes'`;
- the same holds for the other four video pairs (m03↔m02, m04↔m03, m05↔m04,
  m06↔m05);
- the rendered `text` line for `aad020e3:m06` contains a `same-source` marker
  naming `7144c9fc:m07`;
- `aad020e3`'s non-canonical video moments are absent from the default
  selectable pool while still resolvable by explicit ref.

**Measured RED at HEAD (already observed, 2026-08-17):**
`assemble_map` returns exactly 4 `dup_groups`, all speech
(`928c7834`, `bc657f37`, `d7310918`, `a0ab7de1`); `7144c9fc:m07` and
`aad020e3:m06` both carry `dup_group=None`; the whole DB has
**0/2074** video cuts with a `take_group_id`. This RED is recorded, not
predicted.

**Revert signature:** revert the `same_source` provider → `dup_groups` returns to
exactly 4 groups, all speech, and the `same-source` marker vanishes from the
rendered line. Revert only §4c.3's pool suppression → the group and marker
survive but `aad020e3:m06` reappears as a default-selectable ref (this is the
assertion that catches "we told the agent but changed nothing").

**Reachability (mandatory — this is the class of bug that produced
`take_group_id=None`):**
- **Positive:** a fixture with two files in one `content_groups` row and
  overlapping projected spans must yield a link. If it does not, the provider is
  dead code.
- **Negative:** two files **not** in any content group, with identical-looking
  spans, must yield **no** link — proving the provider reads group rows and not
  span geometry.
- **Provenance:** delete the `content_group_members` rows and re-run — the link
  must disappear. If it survives, identity is being re-derived inline (§4c.3's
  trap) and the fix is not what it claims.
- **Offset:** a fixture with `offset_ms=2000` must link the correspondingly
  shifted spans and must **not** link the unshifted ones. This is the assertion
  that fails if someone hardcodes 0.

### V4 — Continuity grading is observable (fix, part 5)

**Assert (effect, real data):** running the real read path on thread
`34bb9967`'s persisted 6-cut timeline yields 5 verdicts, all with
`pic_source='graded'`, and `pic_bits` values `[36, 26, 34, 36, 24]`; the
document's `diagnostics.continuity` reads
`{"seams": 5, "contiguous": 0, "graded": 5, "no_descriptor": 0}`.
**Assert (the distinguishing property — the whole point):** with
`ctx.frame_descriptors` forced to `{}`, the same timeline yields
`{"seams": 5, "graded": 0, "no_descriptor": 5}` and every verdict carries
`pic_source='no_descriptor_prev'`/`'_cur'`. **The two runs must produce
different persisted output.** Today they produce identical output — that is
V4's RED and it is already measured: the real document contains **0** mentions
of `pic_bits`, `pic_scale`, `seamless`, or `continuity`, and
`diagnostics == {"engine": "agentic_loop"}`.
**Assert (the impossible-state alarm):** with all descriptor rows present but
`nearest_ms=0` forced, the ERROR-level "descriptors present but 5/5 seams failed
open" assertion fires. With a file genuinely lacking a row, it does **not** fire
(that stays INFO).
**Assert (coverage invariant, real DB):** every `status='ready'` video with an
`r2_proxy_key` has a `frame_descriptors` row with `>0` samples and current
`schema_version` — **114/114 today**, so this is GREEN-and-meaningful rather
than GREEN-and-vacuous, and it goes RED the moment `_stage_frame_descriptors`
starts silently no-op'ing.
**Revert signature:** revert `pic_source` → the forced-`{}` run and the real run
serialise identically and the distinguishing assertion fails. Revert
`diagnostics.continuity` → `diagnostics` collapses back to
`{"engine": "agentic_loop"}`.

### V5 — Migration conformance (fix, part 5)

**Assert:** every `migrations/*.sql` on disk is git-tracked; every committed
`*.sql` appears in `schema_migrations`; every `schema_migrations` filename has a
file on disk.
**RED at HEAD:** on the current branch `local-dev-isolation`,
`055_frame_descriptors.sql` is staged but **not in `HEAD`**
(`git cat-file -e HEAD:… ` → "exists on disk, but not in 'HEAD'") — assertion 1
fails. That is the measured RED, and it is exactly the reported symptom.
**Assert (the silent case):** insert a bogus `schema_migrations` row for a
non-existent file in a throwaway schema — assertion 3 must fail, while
`db_migrations.assert_up_to_date` stays green. This is what proves the new test
covers `pending()`'s blind spot rather than duplicating it.
**Revert signature:** delete the test → an applied-but-uncommitted migration
returns to being invisible to CI.

### V6 — Project duplication is prevented

**Assert:** 20 concurrent `find_or_create_project(user, same_ids)` calls yield
exactly one `projects` row.
**RED at HEAD:** reproducible today — `projects.py:21-37` is an unguarded
SELECT-then-INSERT with no unique index; the concurrency test produces >1 row.
The production evidence is `8621c012` / `5cd8f004`, 13 ms apart.
**Revert signature:** revert the unique index → the concurrency test produces
>1 row again.
**Precondition:** the pre-existing duplicate `projects` rows must be merged
before the index can be created. Index creation failing **is** the check that
this happened.

---

## PART 7 — SEQUENCING

1. `056_file_content_identity.sql`: `content_sha256` columns, `content_groups`,
   `content_group_members`. **No unique indexes yet.**
2. Backfill `content_sha256` for all 168 files from R2 (§3b.1). Server-verified.
3. Run detection (§3b.3). Expect 6 `identical_bytes` groups / 16 members / 10
   redundant. **V2.**
4. Merge duplicate `projects` rows; then `057` adds
   `uq_projects_user_source_set` + `ON CONFLICT` in `find_or_create_project`.
   **V6.**
5. `058` adds `uq_files_user_content` — after step 2 has proven the data is
   clean enough for a partial index scoped to `status='ready'`. **V1.**
6. Presign / multipart dedupe branch + client hash + L1 server-side verification.
   **V1.**
7. **Layer 1** `same_source` provider in `_annotate_dups`, `same-source`
   rendering, pool suppression. **V3** — highest value, ship before Layer 2.
8. `pic_source`, `diagnostics.continuity`, the impossible-state alarm. **V4.**
9. Commit the L1 write side and L3 read side together; add the conformance test.
   **V5.**
10. **Layer 2** `vcut/video/shot_groups.py`, replacing `vcut/store.py:137`'s
    hardcoded `take_group_id=None`. General capability; explicitly *not* the fix
    for the observed case.
11. Gated merge endpoint (§3b.4). Last, because it is the only step that can
    lose data.

---

## PART 8 — THE FAILURE PATTERN, RESTATED

Four pieces of unreachable-but-tested logic were already found in this codebase:
a dead-air gate (`vcut/resolve.py:512` / `:173`), a junk filter with no producer
(`vcut/store.py:135`), an unreachable silence renderer (`vcut/store.py:81`), and
a discarded done-gate ladder.

**`take_group_id=None` at `vcut/store.py:137` is the fifth — literally two
fields to the right of the third one, on the same `CutRecord(...)` call.**
`_annotate_dups` is correct, documented, and reads a field that is `NULL` in
2074 of 2074 rows.

The pattern is always the same shape: a consumer that reads a field, and no
producer that writes it. Which is why every new check in this plan carries a
reachability assertion, why every provider must be shown to go dark when its
input rows are deleted, and why every RED must be *observed* rather than
reasoned about.
