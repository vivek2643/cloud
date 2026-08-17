# Fix the brain's mirror — a truthful, qualities-aware `read_state`

**Goal.** Make the brain's self-view a truthful mirror of the edit it actually
built: not just the *arrangement* of placed clips, but *how it plays* — per-seam
visual continuity computed from real frame content, the assembled spoken
transcript, and the flags that follow. Every failure we diagnosed traces to the
mirror being **incomplete** (it shows arrangement, not qualities), built on a
**wrong signal** (`file_id` / `scene_cuts` / `frame_diff` — all rejected below),
or **not consulted**. This plan gives the brain a state signal it never had (a
per-frame perceptual descriptor), rebuilds seam continuity on top of it, folds in
transcript truth, and upgrades `read_state` into a self-view the brain can act on.

READ-ONLY handoff: someone codes directly from this. It cites real files/lines.

---

## 0. Problem statement (tied to concrete failures)

Three diagnoses converge on one root cause — **the mirror lies or is silent**:

1. **Fragmented edit `57a517fd`** (project `8621c012`, doc v1). The brain
   juxtaposed pieces of the *same* clip out of source order (clip93 @408s → 209s
   → 14s → 32s) and cross-recording intercut two takes of the same person. The
   only continuity data it had described each cut's *original source neighbors*,
   never the pairs it actually placed adjacent. It had no sense of the seams it
   was creating.

2. **Broken-transcript edit `720713d9`** (doc `9ffb16bf` v1, run `24894c8a`,
   2026-08-12). The brain built a spine of `channel="shown"` slide/demo cuts it
   believed were silent. The mirror showed `said=False`; the labels were wrong
   (the founder's voiceover *was* under every slide, in `dialogue_segments`), so
   the audio played as mid-sentence fragments in *picture* order. Then the
   continuity sense — keyed on `file_id` + source time — fired **~7 false
   same-shot-jump flags** (one screen-recording take holds logo + talking-head +
   slides + demo, so every content change tripped it). The brain spent its entire
   finishing turn rebutting the false visual flags and never audited the spoken
   narrative. The mirror actively distracted it from the real defect.

3. **The temporal-derivative-vs-state-signal insight** (from the `scene_cuts`
   investigation). Every visual signal we persist — `scene_cuts` (histogram
   drift), `motion_dynamics` (optical flow), `frame_diff` (frame-to-frame delta)
   — is a **temporal derivative over the original source timeline**. A seam asks
   a different question: *do segment A's out-frame and segment B's in-frame — two
   arbitrary, non-adjacent source instants — look like the same shot?* That needs
   a **per-frame STATE descriptor**, not a change-rate. Evidence: on the identical
   pitch file (`aad020e3` == `7144c9fc`), `scene_detect` found **1 shot cut in
   244s**; corpus-wide only **2 of 78** `scene_cuts` rows have any shot points;
   and `frame_diff` at the brain's seam instants is *below* the clip mean
   (0.005–0.015 vs mean 0.128). `scene_cuts` and `frame_diff` are the wrong signal
   and are **rejected**. Phase 1 exists to supply the missing state signal.

---

## Phase 1 — Per-frame visual descriptor (the missing state signal) · **requires re-ingest**

### 1.1 Rationale

To compare two arbitrary source instants we need a compact, comparable
**per-frame content fingerprint**. A **perceptual hash (pHash, 64-bit)** is the
right primitive: cheap to compute (one DCT per sampled frame), cheap to store
(8 bytes), and compared by Hamming distance (identical shot ≈ 0–8 bits apart;
different content ≫ 12). Unlike `scene_cuts`/`frame_diff`, a pHash is a *state at
an instant*, so `descriptor(fileA, tA)` vs `descriptor(fileB, tB)` answers "same
picture?" for any pair. This is also reusable later for shot-variety perception
(clustering repeated framings) — worth persisting rather than recomputing.

### 1.2 Current-state findings (file:line) — the pattern to mirror

- L1 detector pattern: `backend/app/services/l1/scene_cuts.py`
  - `SCHEMA_VERSION` (`scene_cuts.py:58`) — bump-to-recompute discipline.
  - `_decode_bgr_frames(video_path, w, h, fps)` (`scene_cuts.py:77-103`) — ffmpeg
    → raw `bgr24` → numpy frames, held inside `limits.ffmpeg_slot()`
    (`scene_cuts.py:89`). **Reuse this verbatim** for descriptor sampling.
  - `_hs_hist` (`scene_cuts.py:106-114`) — per-frame cv2 op; our per-frame op is a
    DCT pHash instead.
  - `compute_scene_cuts` (`scene_cuts.py:117-164`) — best-effort: any decode/opencv
    failure returns an empty `has_scenes=False` result, never fails L1.
  - Tunables live in `backend/app/services/l1/scene_cuts_params.py`.
- Stage wiring + persistence:
  - `_stage_scene_detect` (`pipeline.py:728-759`) — the template: compute → **write
    guard** (`if not sc.has_scenes: return`, `pipeline.py:737-738`) → `insert …
    on conflict (file_id) do update` (`pipeline.py:740-755`).
  - `_run_stage` (`pipeline.py:844-…`) — idempotency: `if _stage_status(...)
    == "done": return` (`pipeline.py:852`); records `processing_jobs`.
  - `_track_motion` (`pipeline.py:893-904`) runs `motion_dynamics → scene_detect →
    color_stats` on the shared video proxy; called by `_run_deep_stages_parallel`
    (`pipeline.py:907-931`) with `video_source` (the downloaded proxy path).
  - `cv2` + `numpy` are already L1 deps (used by `scene_cuts.py`).
- Migrations: additive per-file signal tables (e.g. `022_scene_cuts.sql`). Next
  number is **`055`** (highest existing is `054_vcut_speech_status.sql`).

### 1.3 Design

**New module `backend/app/services/l1/frame_descriptors.py`** (models `scene_cuts.py`):

```python
SCHEMA_VERSION = 1
SAMPLE_FPS = 2          # decision point 7.1 — tunable, parked in params

@dataclass
class FrameDescriptors:
    has_frames: bool = False
    hop_ms: int = 0
    phashes: List[Dict] = field(default_factory=list)  # [{"t_ms": int, "h": "<16 hex>"}]
    def to_dict(self) -> Dict: ...

def _phash64(frame_bgr) -> int:
    # cv2 -> gray -> resize 32x32 -> DCT -> take 8x8 low-freq block ->
    # bit = coeff > median(block excl. DC) -> 64-bit int. numpy+cv2 only.

def compute_frame_descriptors(video_path, duration_ms, *, fps=SAMPLE_FPS,
                              w=64, h=64) -> FrameDescriptors:
    # reuse _decode_bgr_frames(video_path, w, h, fps); one _phash64 per frame;
    # t_ms = round(frame_index * 1000 / fps). Best-effort: decode/opencv failure
    # -> FrameDescriptors(has_frames=False). Never raises.
```

- **Sampling rate: 2 fps** (one frame / 500ms). Justified: matches
  `feel._JOIN_CONTIG_MS`/`footage_map._RUN_GAP_MS` (500ms) so "nearest frame to a
  seam instant" is always within half a hop; resolves slide/demo/reframe changes
  (which persist ≥1s) with margin; storage is trivial (≈`2·duration_s` 8-byte
  hashes: ~1366 for the 683s file). Tunable (Decision Point 7.1) — parked in a new
  `frame_descriptors_params.py` mirroring `scene_cuts_params.py`.
- **pHash: 64-bit DCT hash** over a downscaled luma frame (decision point 7.2 for
  the match threshold, which lives in Phase 2). Store as 16-char hex (or `bigint`)
  — never a raw image.

**Storage — new migration `055_frame_descriptors.sql`** (per-file row, mirrors
`022_scene_cuts.sql` so the reader loads one row per file):

```sql
create table if not exists public.frame_descriptors (
    file_id        uuid primary key references public.files(id) on delete cascade,
    hop_ms         int   not null default 0,
    -- [{t_ms, h}] sampled perceptual hashes in ascending t_ms. 16-hex 64-bit pHash.
    phashes        jsonb not null default '[]'::jsonb,
    schema_version int   not null default 1,
    created_at     timestamptz not null default now()
);
```
(Per-file array, not a row-per-frame table: the classifier needs *all* hashes for
the handful of files in play, loaded once — one indexed PK lookup per file beats
thousands of point rows.)

**Stage wiring (`pipeline.py`):**
- Add `_stage_frame_descriptors(file_id, video_path, duration_s, conn)` next to
  `_stage_scene_detect` — compute, **write-guard on `has_frames`** (mirror
  `pipeline.py:737`), `insert … on conflict (file_id) do update`.
- Register in `_track_motion` (`pipeline.py:901-904`) after `scene_detect`
  (shares the same proxy already downloaded for that track):
  `_run_stage(conn, file_id, "frame_descriptors", _stage_frame_descriptors, …)`.
- Add `"frame_descriptors"` to the appropriate `STAGES_V2`-style tuple
  (near `pipeline.py:47`).

**Reader helper** (used by Phase 2, in `observe.build_context`):
```python
# app/services/l1/frame_descriptors.py (or a small l3 read helper)
def load_descriptors(conn, file_ids) -> Dict[str, List[Tuple[int, int]]]:
    # {file_id: [(t_ms, phash_int), ...]} sorted by t_ms; missing file -> absent.
```

### 1.4 Re-ingest + versioning (Phase 1)

- **Re-ingest: YES — one-time, all files** (the signal is genuinely new and must
  be persisted; nothing derives it today). Because `frame_descriptors` is its own
  stage guarded by `_run_stage`'s `processing_jobs` idempotency, a re-run only
  executes the new stage for files that lack it — a targeted backfill, not a full
  re-analysis. Provide a backfill script (mirror how other single stages are
  re-run) that iterates files and invokes `_stage_frame_descriptors`.
- **No `TREE_VERSION`/`CUTRECORD_MAP_VERSION` bump** — this signal isn't read
  through the footage-map or cutrecord caches.
- Bump `frame_descriptors.SCHEMA_VERSION` if the hash/sampling shape ever changes.

---

## Phase 2 — Seam continuity from frame comparison · uses Phase 1, read-side

### 2.1 Current-state findings (file:line)

- `feel.join_continuity(cuts)` (`feel.py:208-240`) — classifies each seam from
  `file_id` + `src_in_ms`/`src_out_ms` **only**:
  - different clip → `cut` (`feel.py:226-227`);
  - same clip, `|gap| ≤ _JOIN_CONTIG_MS` → `seamless` (`feel.py:230-231`);
  - same clip, gap beyond tol → `jump` (`feel.py:232-233`), `backward` set at
    `feel.py:238`. **This is the false-positive source** (Problem 2).
- `_JOIN_CONTIG_MS = 500` (`feel.py:35`). `CutFeel.src_in_ms/src_out_ms`
  (`feel.py:48-49`), populated in `feel.simulate` (`feel.py:133-145`) — the data
  path already carries source spans; it lacks only the *picture*.
- Consumers (all must keep working): `observe.read_state` (`observe.py:490`),
  `observe.diagnose` (`observe.py:1198-1201`), `observe.review`
  (`observe.py:1535-1536`), `feel.FeelReport.narrate` (`feel.py:90-93`),
  `tools._verify_before_finish` Stage 2.5 (`tools.py:697-704`, matches the
  verbatim string `"same-shot jump"`).

### 2.2 Design — rewrite `join_continuity` to consult frame content

> **This SUPERSEDES Fix 2 of `speech_label_and_shot_continuity_fix.plan.md`**
> (shot boundaries from `scene_cuts` / `frame_diff`). Those signals are rejected
> (see §0.3); the seam verdict is now decided by comparing the two frames at the
> seam via the Phase-1 pHash. Fix 1 of that plan is carried into Phase 3.

New signature (keep `feel` pure — descriptors passed in, no DB inside):
```python
def join_continuity(cuts: List[CutFeel],
                    descriptors: Optional[Dict[str, List[Tuple[int,int]]]] = None,
                    *, tol_bits: int = _PHASH_SAME_BITS,
                    nearest_ms: int = 600) -> List[dict]:
```

Per adjacent pair `(prev, cur)`:

1. **Different file →** `cut`. (Cross-clip is never a same-shot jump.)
2. **Same file, source-contiguous** (`|cur.src_in_ms − prev.src_out_ms| ≤
   _JOIN_CONTIG_MS`) → `seamless`. (Contiguous playback is definitionally one
   shot; no frame check needed.)
3. **Same file, non-contiguous / backward** → look up the picture at each side of
   the seam and compare:
   - `hA = descriptor(prev.file_id, prev.src_out_ms)`,
     `hB = descriptor(cur.file_id, cur.src_in_ms)` (nearest sampled `t_ms` within
     `nearest_ms`; else **missing**).
   - **Both present and `hamming(hA, hB) ≤ tol_bits`** → same picture → **`jump`**
     (the real same-shot jump — clip93 408→209 within one continuous framing).
   - **Both present and `hamming > tol_bits`** → picture genuinely changed
     (talking-head → slide inside one screen recording) → **`cut`**, NOT a jump.
   - **Either descriptor missing** → **fail-open to `cut`** (never a false jump;
     this is also the pre-Phase-1 behavior for un-ingested files).

Each verdict dict keeps its existing keys (`from_pos`, `to_pos`, `kind`,
`same_clip`, `src_gap_ms`, `backward`) and gains `pic_bits` (the Hamming distance
or `None` when unavailable) so the mirror can explain *why*.

- **Threshold `_PHASH_SAME_BITS` (decision point 7.2): recommend 8** (of 64).
  Justified: DCT-pHash literature and practice put "same image / same shot, minor
  motion" under ~8–10 bits and "different content" well above; 8 is conservative
  (biases toward calling things a `cut`, i.e. *away* from false jumps, matching
  our failure mode). Parked as a tunable in `feel.py` (or `frame_descriptors_params`).

### 2.3 Data path to descriptors at classify time

- `observe.build_context` loads `frame_descriptors` once for every file in play
  via `frame_descriptors.load_descriptors(conn, ctx_file_ids)` and stores it on
  `EditContext` (new field `frame_descriptors: Dict[str, List[Tuple[int,int]]]`).
- `feel.simulate(timeline, meta_by_ref, descriptors=None)` accepts and stashes
  `descriptors` on `FeelReport` (new field) so `FeelReport.narrate` (`feel.py:90`)
  can call `join_continuity(self.cuts, self.descriptors)`.
- Update every call site to pass `ctx.frame_descriptors`:
  `read_state` (`observe.py:490`), `diagnose` (`observe.py:1198`), `review`
  (`observe.py:1536`) — each already has `ctx` in scope.
- Nearest-`t_ms` lookup: `bisect` on the per-file sorted list; reject if the
  nearest sample is farther than `nearest_ms` → treated as missing → fail-open.

### 2.4 Re-ingest + versioning (Phase 2)

- **Read-side only** — no re-ingest beyond Phase 1's descriptor backfill.
- **No `TREE_VERSION`/`CUTRECORD_MAP_VERSION` bump** — `feel`/`observe` outputs
  are computed live per turn, not cached under those keys.

---

## Phase 3 — Transcript truth + assembled transcript in the mirror · read-side

### 3.1 Fix 1 (carried in from `speech_label_and_shot_continuity_fix.plan.md`) — speech presence from the transcript

**Current-state (file:line).**
- `cutrecord_map._audio_mute_for(channel, row)` (`cutrecord_map.py:476-488`):
  `channel=="said"` → `(None, False, [])` (unmuted); otherwise mute iff
  `pace.natural_sound` is false. **Blind to whether words actually play under the
  cut.**
- `cutrecord_map._to_cut_dict` (`cutrecord_map.py:491-494`) sets `audio/mute/flags`
  from that function; `channel` defaults from `kind` (`cutrecord_map.py:492`).
- `footage_map.build_clip_tree` gates the verbatim words on channel:
  `said_text = _said_text_for_span(...) if cut.get("channel") == "said" else ""`
  (`footage_map.py:281-282`) — so incidental speech under a `shown` cut is never
  surfaced.
- Whisper words live in `dialogue_segments` (L1; present for all three pitch
  files). `_said_text_for_span` already reads them (`footage_map.py`).

**Design (rule keyed on transcript truth, not cut type).**
- Add a read helper (in `cutrecord_map` or a small L1 read util):
  `speech_words_in_span(file_id, in_ms, out_ms) -> (word_count, has_speech)` by
  overlapping `dialogue_segments` word timings against the cut's `[src_in_ms,
  src_out_ms]`.
- In `_audio_mute_for`, before the `natural_sound` branch: if the span carries
  transcript words (`has_speech`), the cut is **speech-bearing**. For a
  *picture-chosen* cut (`channel != "said"`) with incidental speech:
  - **surface it** (Phase 3.2 / footage map) AND
  - **coverage-mute by default** — return `("speech", True, ["muted",
    "incidental-speech"])` so it doesn't drag in stray mid-sentence audio unless
    the brain deliberately unmutes. (A truly silent picture cut is unchanged.)
- In `footage_map.build_clip_tree` (`footage_map.py:281`): drop the
  `channel=="said"` gate — compute `said_text` whenever the span has transcript
  words, so the beat index shows the brain the words that *will* play.

**Re-ingest / versioning (Fix 1).** **Read-side, NO re-ingest** —
`dialogue_segments` is already persisted; these are recomputations off it. Bump
**`CUTRECORD_MAP_VERSION` 5 → 6** (`cutrecord_map.py:59`) and **`TREE_VERSION`
22 → 23** (`footage_map.py:121`) so cached ladders / beat-index rows recompute.

### 3.2 Assembled spoken transcript in `read_state` — the "ears"

**Design.** In `read_state`, after building `cuts`, assemble the words that play
in **program order**:
- For each placed segment, take its spoken text — `dialogue_segments` words
  overlapping `[in_ms, out_ms]` (verbatim, via the same overlap helper), falling
  back to `seg["content"]`.
- Emit `assembled_transcript`: an ordered list of `{pos, file, prog_start_ms,
  text, muted, boundary}` plus a joined `narrative` string, so the brain can read
  the script end-to-end and judge coherence.
- **Boundary/broken-line detection** (a flag, not a mutation): mark a segment
  `clipped_head`/`clipped_tail` when the first/last overlapping word straddles the
  cut edge (word `start_ms < in_ms` or `end_ms > out_ms`), i.e. the trim landed
  mid-word; mark `mid_sentence` when the boundary text lacks sentence-final
  punctuation. These become the transcript red flags used in Phase 4.

**Re-ingest / versioning.** Read-side, none — `read_state` is computed live.

---

## Phase 4 — The mirror upgrade + reliable surfacing · read-side

### 4.1 Current-state (file:line)

- `read_state` per-cut rows built at `observe.py:519-589`; the join is currently
  attached as the join **into** each cut and only when not a plain `cut`
  (`observe.py:580-587`).
- Push points: `diagnose` (`observe.py:1198-1201`) and `tools._verify_before_finish`
  Stage 2.5 (`tools.py:694-704`), which matches the verbatim `"same-shot jump"`
  string and surfaces at most once via `state["continuity_surfaced"]`
  (`tools.py:698-699`).

### 4.2 Design — a self-view: rows = clips, joins = verdicts

- Each `read_state` cut row shows its **join verdict to the NEXT clip**
  (`seamless` | `jump` | `cut`) with the `pic_bits` reason from Phase 2, plus
  per-row **flags**:
  - `jump` (real, frame-based same-shot jump);
  - `dead_air` (gap / low-energy run from existing `feel` derivations);
  - `broken_line` (`clipped_head`/`clipped_tail`/`mid_sentence` from Phase 3.2);
  - `incidental_speech` (un-muted words under a picture cut — the `incidental-
    speech` flag from Phase 3.1 when not muted).
- Keep the compact render consistent with the existing tag vocabulary
  (`observe.py:582-587`): `join: "seamless" | "jump"`, `join_note` explains
  (backward / picture-changed / skip Ns).
- Surface `assembled_transcript.narrative` at the top of `read_state` so the
  qualities view (how it *plays* and *sounds*) sits beside the arrangement.

### 4.3 Reliable surfacing — hybrid push/pull

- **Pull (full detail):** everything above is in `read_state` whenever the brain
  looks.
- **Push (red flags only, un-gated):** at decision time via `diagnose`
  (`observe.py:1198`) and at finish via Stage 2.5 (`tools.py:694-704`), push ONLY
  genuine red flags — now driven by the **corrected frame-based verdict** from
  Phase 2 (retiring the `file_id` false-positive flood) **plus** a broken-transcript
  flag (a `broken_line` on a speech-bearing seam). Keep the existing one-shot
  guards (`state["continuity_surfaced"]`, `tools.py:698`) so the loop terminates;
  add a parallel `state["transcript_surfaced"]` for the new transcript red flag.
- **Decision point 7.3 (push cadence):** recommend **pull-by-default + red-flag
  push at diagnose/finish** (above). The alternative — pushing the mirror after
  *every* action — is louder and risks re-introducing distraction; flagged as an
  explicit choice, not silently decided.

### 4.4 Re-ingest + versioning (Phase 4)

- Read-side, none. Render changes that flow through `footage_map` are already
  covered by the Phase 3 `TREE_VERSION` bump.

---

## 5. Exact change list per file

**Phase 1**
- `backend/migrations/055_frame_descriptors.sql` — **new** table (§1.3).
- `backend/app/services/l1/frame_descriptors.py` — **new**: `SCHEMA_VERSION`,
  `FrameDescriptors`, `_phash64`, `compute_frame_descriptors` (reusing
  `scene_cuts._decode_bgr_frames`), `load_descriptors`.
- `backend/app/services/l1/frame_descriptors_params.py` — **new**: `SAMPLE_FPS`,
  frame `w/h`.
- `backend/app/services/l1/pipeline.py` — add `_stage_frame_descriptors`
  (near `:728`); register in `_track_motion` (`:901-904`); add stage to the
  `STAGES_V2`-style tuple (`~:47`).
- Backfill script under `backend/scripts/` to run the one stage over all files.

**Phase 2**
- `backend/app/services/l3/feel.py` — rewrite `join_continuity` (`:208-240`) to
  the frame-comparison classifier; add `_PHASH_SAME_BITS`; add `hamming`/nearest
  helpers; `descriptors` param on `simulate` (`:112`) + `FeelReport` field;
  `narrate` (`:90`) passes descriptors.
- `backend/app/services/l3/observe.py` — `EditContext` gains `frame_descriptors`;
  `build_context` loads it; pass `ctx.frame_descriptors` at `:490`, `:1198`,
  `:1536`.

**Phase 3**
- `backend/app/services/l3/cutrecord_map.py` — transcript-overlap helper;
  `_audio_mute_for` (`:476`) speech-bearing branch; bump `CUTRECORD_MAP_VERSION`
  5→6 (`:59`).
- `backend/app/services/l3/footage_map.py` — drop `channel=="said"` gate at
  `:281`; bump `TREE_VERSION` 22→23 (`:121`).
- `backend/app/services/l3/observe.py` — `read_state` assembles + emits
  `assembled_transcript` (after `:589`); boundary/broken-line detection.

**Phase 4**
- `backend/app/services/l3/observe.py` — per-row join-to-next + flags
  (`:519-589`); red-flag list in `diagnose` (`:1198`).
- `backend/app/services/l3/tools.py` — Stage 2.5 (`:694-704`) reads corrected
  verdict; add `transcript_surfaced` one-shot for the broken-line push;
  `state` init for the new flag.
- `backend/app/services/l3/converse.py` — update the continuity wording
  (`:163`,`:183`) so the system prompt describes the frame-based verdict, not the
  file_id heuristic.

---

## 6. Re-ingest & version-bump summary (per phase)

| Phase | Re-ingest? | `CUTRECORD_MAP_VERSION` | `TREE_VERSION` | Other |
|---|---|---|---|---|
| 1 — descriptors | **YES, one-time, all files** (new persisted signal; targeted backfill via `processing_jobs` idempotency) | — | — | new `frame_descriptors.SCHEMA_VERSION=1` |
| 2 — seam continuity | No (uses Phase 1) | — | — | — |
| 3 — speech truth + transcript | No (reads persisted `dialogue_segments`) | **5 → 6** | **22 → 23** | — |
| 4 — mirror + surfacing | No | — | (covered by Phase 3) | — |

---

## 7. Decision points (call these out explicitly)

- **7.1 Sampling rate** — recommend **2 fps**; tunable. Lower (1 fps) halves
  storage but widens the nearest-frame tolerance; higher (4 fps) sharpens seam
  lookups on fast reframes at ~2× storage. Parked in `frame_descriptors_params`.
- **7.2 pHash match threshold** — recommend **Hamming ≤ 8 / 64** for "same
  picture." Biased conservative (toward `cut`) to kill the false-jump flood.
  Validate against the two known cases (below) and tune.
- **7.3 Surfacing cadence** — recommend **pull-by-default + red-flag push at
  diagnose/finish**. Alternative: push the mirror after every action (louder,
  distraction risk). Explicit choice, not silently defaulted.

---

## 8. Test plan (per phase)

**Phase 1 — descriptor compute/persist**
- `_phash64` is stable (same frame → same hash) and order-invariant to trivial
  re-encode; two visibly different frames differ by ≫ threshold bits.
- `compute_frame_descriptors` returns ascending `t_ms` at `hop = 1000/fps`;
  `has_frames=False` on a decode failure (monkeypatch `_decode_bgr_frames` to
  raise) and the stage **writes no row** (mirror the `has_scenes` guard test).
- `on conflict (file_id) do update` upserts; `load_descriptors` returns sorted
  `(t_ms, int)` and omits missing files.

**Phase 2 — seam classifier** (unit, synthetic `CutFeel` + a fake `descriptors`)
- same clip, contiguous → `seamless` (no descriptor needed).
- same clip, non-contiguous, **similar** frames (Hamming ≤ 8) → **`jump`**
  (the clip93 408→209 backward case still flagged).
- same clip, non-contiguous, **different** frames (Hamming > 8) → **`cut`**
  (the multi-content take: talking-head→slide inside one file → **no false jump**).
- cross-clip → `cut` regardless of similarity.
- descriptor missing on either side → **fail-open `cut`** (never a false jump).
- Regression fixture from the two real edits: `7144c9fc` multi-content seams →
  **zero** jumps; `57a517fd` clip93 backward seam → jump.

**Phase 3 — speech truth + assembled transcript**
- A `shown` cut whose span overlaps `dialogue_segments` words → classified
  speech-bearing: `said_text` populated in the beat index AND `_audio_mute_for`
  returns muted+`incidental-speech`.
- A genuinely silent `shown` cut → unchanged (`sound`/`silent` per `natural_sound`).
- `read_state.assembled_transcript` lays segments in program order; a
  mid-word boundary sets `clipped_tail`/`clipped_head`; a mid-sentence edge sets
  `mid_sentence`. Reconstruct `720713d9` read-only → the broken-line flags fire on
  the fragmented slide spine.

**Phase 4 — mirror render + gated push**
- `read_state` row shows join-to-next verdict + flags; `narrative` present.
- `diagnose` emits a red flag ONLY for a frame-based `jump` and for a
  `broken_line` speech seam; multi-content take produces none.
- `tools._verify_before_finish`: the corrected jump verdict survives the done-gate
  and surfaces once (`continuity_surfaced`); a broken-line flag surfaces once
  (`transcript_surfaced`); loop still terminates (no infinite re-push).

---

## 9. Relationship to `speech_label_and_shot_continuity_fix.plan.md`

- **Supersedes its Fix 2** (shot-aware continuity from `scene_cuts`/`frame_diff`
  shot boundaries). Those signals are temporal derivatives and were shown empty /
  unreliable for the pitch content; Phase 2 replaces them with a direct pHash
  comparison of the two seam frames (Phase 1).
- **Carries its Fix 1** (transcript-driven speech labeling) into **Phase 3.1**,
  unchanged in intent.
- Both earlier band-aids remain **rejected**: "mute slide-type cuts" (keys on cut
  type, not truth) and "gate/disarm Stage 2.5" (hides the symptom). This plan
  fixes the underlying truths — speech from the transcript, continuity from frame
  content — and makes the mirror show them.
