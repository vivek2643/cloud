# Brain Perception Upgrade — per-cut signals (hybrid) + provenance prompt — implementation plan

Status: ready to implement (from another chat). This is the executable spec;
accuracy has been ground against the code (file/line cites throughout).

## Branch + deploy (READ FIRST)

- **Implement on the `local-dev` branch (already checked out). Do NOT touch
  `main`.** When done, the implementer tests locally, then merges
  `local-dev` → `main` (fast-forward) to deploy.
- **RunPod propagation caveat (per-phase, always call it out):** L1 signal
  computation (the `motion_dynamics` / `audio_features` / `scene_cuts` arrays)
  runs in the **RunPod** GPU container (`backend/handler.py` only dispatches the
  `l1_*` tasks; `worker.py:86` — the dispatcher forwards `gpu`-queue jobs to
  RunPod when `gpu_execution == "runpod"`). So **IF any phase changed L1**, the
  RunPod image would need to rebuild (via the GitHub auto-build) after the merge
  to `main` before that change takes effect.
  - **This plan needs NO L1 change.** Every L1 signal it consumes already
    exists and is persisted (verified below), so **no RunPod rebuild is
    required for either change.** The work lands entirely on:
    - the **Render L3-ingest worker** (`app/services/l3/ingest.py` →
      `post.assemble_cut_records`), which is where cuts-v3 ingest runs (NOT
      RunPod — RunPod only runs L1), and
    - the **API / brain-runtime** (`converse.respond` → `footage_map` beat-line
      render + `observe` senses), which runs in the Render web/API process.
  - Each phase below is tagged **[RunPod L1]**, **[Render L3-ingest]**, or
    **[API/brain-runtime]**. Only a **[Render L3-ingest]** phase requires a
    re-ingest to take effect on existing projects; **[API/brain-runtime]** phases
    take effect on the next turn with no re-ingest and no rebuild.

## Guiding principles (unchanged, enforce throughout)

- **Code owns numbers/structure; the LLM owns categories/text.** The new
  landmarks/curves are all code-derived from L1 arrays — no model-emitted
  offsets, scores, or thresholds.
- **No fallback that masks a miss.** A cut with no signal renders no
  breadcrumb and inspection returns an explicit "no signal" for that channel —
  never a fabricated point.
- **Additive + backward-compatible.** Every new field is optional with a
  pre-migration default; old `cut_records` / `ingest_runs` keep working
  untouched (see "Backward compatibility").

## Out of scope (explicit)

- **Voiceover-as-spine wiring** (audio-only `dialogue_segments` + `cut_records`
  for pure-audio files, VO as the main line) is a **separate future effort** and
  is deliberately NOT folded into this plan. Do not touch the audio-file ingest
  or spine-selection paths here.

---

# Change 1 — Surface currently-withheld per-cut signals (HYBRID, ADOPTED)

## Why hybrid (the design decision, locked)

A single always-on rich dump per cut would blow the beat-index char budget and
drown the brain in numbers it rarely needs; a purely compact always-on landmark
set still **filters** the information (the user's pushback). A purely on-demand
sense keeps everything but the brain **may never think to ask**. So Change 1
ships BOTH, working together:

- **Mechanism A — beat-line breadcrumb (always-on, tiny):** a minimal per-cut
  marker that merely FLAGS that a cut has action structure / audio dynamics /
  internal shot cuts worth a closer look. Purpose: discoverability at ~near-zero
  char cost. It intentionally carries counts only, not the values.
- **Mechanism B — on-demand cut inspection (rich, windowed+downsampled):** a
  sense the brain calls for ONE cut to get the fuller windowed signal —
  action-energy hits/curve, audio change points + silence gaps, internal
  shot/composition cuts — computed from the L1 time-series. Windowed to the cut
  and downsampled to a fixed cap (never raw thousands of points).

The breadcrumb is the **pointer**; the sense is the **payload**. The breadcrumb
teaches the brain *when* to spend a tool call on Mechanism B.

Tradeoffs the implementer should keep in mind:
- Breadcrumb alone = lossy (counts only) but discoverable + free every turn.
- On-demand alone = full-ish but invisible; the brain may never invoke it.
- Hybrid = discoverable AND deep; the only added always-on cost is the
  breadcrumb's handful of chars.

## The signals (all currently WITHHELD from the brain — verified)

Audit of what reaches the brain today vs. what L1 computes and stores:

| Signal | L1 source (column) | In the beat line today? | In a sense today? |
|---|---|---|---|
| Action hits/peaks within a cut | `motion_dynamics.action_energy[]`, `action_points[]`, `hop_ms` (mig. 011/025/033) | No (only the single `peak:+Xs`, `post._salience`) | No (only `piece_breakdown` for multi-event V4 clusters) |
| Audio dynamics (RMS/prosody envelope) | `audio_features.rms_db[]`, `prosody_hop_ms` (mig. 009) | No | No |
| Silence gaps within a cut | `audio_features.silence_intervals[]` (mig. 003) | No | No |
| Internal shot / composition cuts | `scene_cuts.shot_points[]`, `composition_points[]` (mig. 022) | No | No |
| (optional) True peak | `audio_features.true_peak_db` (mig. 003) | No | No |
| (optional) Transcript language | `transcripts.language` (mig. 003) | No | No |
| (optional) Dialogue topic granularity | `dialogue_segments` | No | Partial |

Confirmations from code:
- All arrays are already read by `l1.snapshot.build_l1_snapshot`
  (`backend/app/services/l1/snapshot.py:139-216`): `audio_features` selects
  `rms_db`, `silence_intervals`, `prosody_hop_ms`, `true_peak_db`, `onsets_ms`;
  `motion_dynamics` selects `action_energy`, `action_points`, `hop_ms`;
  `scene_cuts` selects `shot_points`, `composition_points`, `hop_ms`. **Nothing
  new needs to be computed in L1.**
- `observe._fetch_audio_features` (`observe.py:153-174`) selects only
  `integrated_lufs, is_musical, bpm, onsets_ms, sections, drop_ms` — it **omits
  `rms_db` and `silence_intervals`**. That omission is exactly why those are
  invisible to the on-demand senses today; Mechanism B fixes it with a targeted
  windowed read (below), NOT by bloating `_fetch_audio_features` for every turn.
- `post.assemble_cut_records` (`post.py:780-986`) receives `motion_by_file`
  (has `action_energy`/`action_points`/`hop_ms`), `audio_by_file` (has
  `rms_db`/`hop_ms`/`onsets_ms`), and `silences_by_file` (has
  `silence_intervals`) — but is **NOT** passed `scene_by_file`
  (`shot_points`/`composition_points`). `ingest._load_signals`
  (`ingest.py:103-126`) already loads `scene_by_file` and passes it to
  `image_plan`/lattice but not to `post`. So Mechanism A's shot-cut count needs
  `scene_by_file` threaded into `assemble_cut_records`.

---

## Mechanism A — beat-line breadcrumb (always-on)

### A0. Storage decision — small precomputed structure at ingest (migration)

The breadcrumb needs, per cut, the COUNTS of interior action hits / audio
change points / silence gaps / internal shot cuts. Computing those at
render time would force the brain-runtime read path to pull every clip's full
L1 arrays on every turn — but the read path (`cuts_read.rows_for_run` →
`cutrecord_map._to_cut_dict` → `footage_map`) deliberately **never touches L1
tables today**. Recomputing there would be both a perf regression and a layering
break.

**DECISION (locked): precompute a compact per-cut `landmarks` structure at
ingest and store it on `cut_records`, mirroring exactly how `salience`,
`camera`, and `continuity` are already computed-once-at-ingest.** This is a
new small `jsonb` column and requires one additive migration + a re-ingest to
populate existing runs (old rows default to `{}` → no breadcrumb, see
Backward-compat).

The SAME stored structure powers both the breadcrumb (counts) AND is a cheap
warm cache the sense (Mechanism B) can consult, though the sense still does the
windowed downsample from the L1 arrays for the full payload.

### A1. `landmarks` shape (compact, code-owned)

Stored jsonb, all offsets are ms from the cut's own `src_in_ms`, all
edge-guarded (interior only, reusing `_peak_tag`'s guard idea — see A4):

```json
{
  "act":  {"n": 3, "hits": [820, 2100, 3950]},
  "adx":  {"n": 2, "changes": [{"off": 500, "dir": "up"}, {"off": 2300, "dir": "down"}]},
  "sil":  {"n": 1, "gaps": [{"off": 1000, "dur": 400}]},
  "shot": {"n": 2, "cuts": [{"off": 2000, "hard": true}, {"off": 4500, "hard": false}]}
}
```

- `act.hits` — interior local maxima of `action_energy` within the span AND/OR
  `action_points` (subject-motion impacts, `{ts_ms,kind,score}`) that fall
  inside the span; keep the top-K by score (K=5), store offsets sorted by time.
- `adx.changes` — significant rises/falls of the `rms_db` envelope over the
  span, using the SAME clip-relative normalization the quality scores use
  (`_series_lohi`/`_norm_in_clip`, `post.py`) so there are no absolute-dB
  constants; a change point is where the normalized envelope crosses a
  data-derived delta between consecutive downsampled bins. `dir` ∈ {up,down}.
  Cap K=5.
- `sil.gaps` — `silence_intervals` clipped to the span, each `{off, dur}`. Cap
  K=5. Ignore gaps shorter than one `prosody_hop_ms`.
- `shot.cuts` — `shot_points` (hard=true) + `composition_points` (hard=false)
  within `(src_in_ms, src_out_ms)`, exclusive of the exact edges (an edge
  "shot cut" is just this cut's own boundary). Cap K=8.
- `n` on each is the FULL interior count (may exceed the stored list length),
  so the breadcrumb can say "3 hits" even if only the top few offsets are kept.
- Any channel with zero interior events is **omitted** from the dict (so `{}` on
  a silent/static cut → no breadcrumb, no wasted chars/bytes).

Keep the stored offset lists small (they double as the sense's warm cache); the
authoritative full payload is still the windowed downsample in Mechanism B.

### A2. Compute site — `post.py` [Render L3-ingest]

- Add `_landmarks(...)` next to `_salience` (`post.py:350-407`). Signature
  takes: `action_energy, action_points, hop_ms, rms_db, rms_hop_ms, rms_lo,
  rms_hi, silence_intervals, shot_points, composition_points, s, e`. Pure,
  deterministic, reuses `_span_slice`/`_norm_in_clip`/`_series_lohi`.
- Add `landmarks: Dict[str, Any] = field(default_factory=dict)` to the
  `CutRecord` dataclass (`post.py:410-489`) and to `CutRecord.to_dict`
  (`post.py:491-512`).
- In `assemble_cut_records` (`post.py:884-984`), compute `landmarks` for every
  cut (speech AND video) alongside `salience`, from `motion`/`audio`/the new
  `scene` dict + `silences_by_file`. It is independent of the V4-vs-speech
  salience branch (compute it in both branches).
- **Thread `scene_by_file`** into `assemble_cut_records` as a new optional
  param `scene_by_file: Optional[Dict[str, dict]] = None` (default `{}` → no
  shot channel, back-compat for any other caller/test). Pass it from
  `ingest.py:456` (the `scene_by_file` local is already loaded at
  `ingest.py:246`).

### A3. Persist + read-path [Render L3-ingest + API/brain-runtime]

- Migration (see "Migration" section): `cut_records.landmarks jsonb not null
  default '{}'::jsonb`. **[Render L3-ingest]** (DDL; applied by the migration
  runner).
- `ingest_store.insert_cut_records` (`ingest_store.py:138-173`): add
  `landmarks` to the column list + `json.dumps(r.landmarks)` to the values
  tuple. **[Render L3-ingest]**
- `cuts_read.rows_for_run` (`cuts_read.py:57-66`): add `landmarks` to the
  SELECT list. **[API/brain-runtime]** (read path).
- `cutrecord_map._to_cut_dict` (`cutrecord_map.py:481-561`): add
  `"landmarks": row.get("landmarks") or {}` next to `"salience"`.
  **[API/brain-runtime]**
- `footage_map` moment builder (`footage_map.py:~356`, next to
  `"salience": cut.get("salience") or {}`): add
  `"landmarks": cut.get("landmarks") or {}`. Also carry the true cut span
  (`"src_in_ms"`, `"src_out_ms"`) on the moment so Mechanism B can window to
  the full cut, not the balanced-variant window (the moment's `in_ms/out_ms`
  are the anchor variant). **[API/brain-runtime]**

### A4. Beat-line marker format — `footage_map._moment_line` [API/brain-runtime]

Add a `_landmarks_tag(m)` helper (mirroring `_peak_tag`, `footage_map.py:680-707`)
and append it into the tag chain in `_moment_line` (`footage_map.py:933-936`,
after `peak_tag`, before `outlook_tag`).

Marker = a single terse token that names ONLY the present channels with their
counts, so it reads as "there's more here — inspect if relevant":

```
sig:act3,shot2,sil1
```

- Prefix `sig:` (short for "signals available"), then comma-joined
  `code<count>` for each present channel, in a fixed order `act,adx,sil,shot`.
- Codes: `act` = action hits, `adx` = audio change points, `sil` = silence
  gaps, `shot` = internal shot/composition cuts.
- Counts come straight from `landmarks[ch]["n"]`, displayed capped at `9+`.
- Rendered ONLY when `landmarks` is non-empty; a cut with no interior structure
  shows nothing (identical to today).
- Edge guard: reuse the `_peak_tag` interior rule (`> max(100ms, span/10)` from
  both edges) when COUNTING interior events at ingest, so a landmark pinned to
  the cut's own boundary never inflates the breadcrumb.

Example beat line (existing tags + new marker, video cut):

```
  m7 [cap] PIC:P2 (MS, q.71) SND:P2 ON-CAM [0:12-0:19 6.4s] "he yanks the cord and it finally catches" · nrg:calm|balanced|tight pace:0.8-1.4x cam:push-in peak:+2.1s sig:act3,shot1
```

Here `peak:+2.1s` is still the single strongest instant; `sig:act3,shot1` tells
the brain there are 3 action hits and 1 internal shot cut it can pull up with
the inspection sense.

### A5. Char-budget impact (Mechanism A) — against `_INDEX_CHAR_CAP = 110_000`

- Marker worst case (all four channels present, 2-digit counts):
  `sig:act9,adx9,sil9,shot9` = **25 chars**. Typical (1–2 channels): **~10
  chars**. Empty on structureless cuts: **0**.
- The breadcrumb NEVER lists offsets/values in the beat line — that is the whole
  point of the hybrid — so per-cut cost is bounded by that ~25-char ceiling
  regardless of how much structure a cut has.
- Scaling: for a large project of ~500 cuts with an average marker of ~12 chars,
  that is **~6 KB added** to a beat index that is capped at 110 KB. Even the
  degenerate all-cuts-worst-case (~500 × 25 = ~12.5 KB) stays comfortably under
  cap. The existing truncation guard (`converse.py:282-285`) remains the safety
  net. **Conclusion: Mechanism A cannot meaningfully threaten the cap.**

---

## Mechanism B — on-demand cut inspection (rich, windowed + downsampled)

### B0. Sense shape decision — new `inspect_cut(ref)` [API/brain-runtime]

`read_state(seg_id=...)` (`observe.py:394`, tool at `tools.py:68-78`, dispatch
`tools.py:432-433`) is keyed on a PLACED main-line `seg_id`. But the brain most
wants to inspect a cut it is DECIDING to place — i.e. a beat-index cut that is
not on the timeline yet, addressed by its map `ref`. Overloading `read_state`
would only reach placed cuts.

**DECISION (locked): add a dedicated sense `inspect_cut(ref)`** that resolves a
cut by its beat-index `ref` (via `ctx.index` / `ctx.meta_by_ref`, the same
resolution `place` uses) and returns the windowed signal payload. Accept an
optional `seg_id` alias that maps to the placed segment's `ref` so a placed cut
can be inspected too. This keeps `read_state` unchanged and gives the brain a
verb whose name matches the beat-index breadcrumb ("you saw `sig:...`, now
`inspect_cut`").
- Register it in `tools.py` `_tools()` (`tools.py:68` neighborhood) as
  `S("inspect_cut", "...", obj({"ref": {"type":"string"}, "seg_id":
  {"type":"string"}}))` (ref OR seg_id; ref preferred).
- Dispatch in `tools.py:432` neighborhood:
  `observe.inspect_cut(ctx, ref=args.get("ref"), seg_id=args.get("seg_id"), document=doc)`.

### B1. Compute site decision — compute-at-request in `observe.py` (NO migration)

Two options analyzed:

- **(chosen) Compute at request time in `observe.py` from the L1 arrays.** A
  single cut's inspection is rare and interactive; a targeted single-file read
  of `motion_dynamics`/`audio_features`/`scene_cuts` (or reuse
  `l1.snapshot.build_l1_snapshot(file_id)`), windowed to `[src_in_ms,
  src_out_ms]` and downsampled to a fixed cap, is cheap and needs **no schema
  change and no re-ingest**. It always reflects the current L1 data.
- **(rejected for B) Precompute the full curves at ingest.** Would add heavy
  per-cut jsonb (downsampled curves for every cut, most never inspected), a
  bigger migration, and a re-ingest — paying storage + write cost for data
  rarely read. The compact `landmarks` (Mechanism A) already gives ingest a
  cheap warm cache; the FULL curve does not need to be stored.

Justification: the read is on-demand and single-file; the arrays are already
persisted; downsampling is O(span/hop). Precompute would optimize the wrong
axis (write-once/read-rarely).

Implementation:
- Add `observe._fetch_signal_window(file_id, s, e) -> dict` (a targeted read):
  select `motion_dynamics(hop_ms, action_energy, action_points)`,
  `audio_features(prosody_hop_ms, rms_db, silence_intervals)`,
  `scene_cuts(hop_ms, shot_points, composition_points)` for the one file, then
  slice each array/point-list to `[s, e]`. (This is the one place the
  `_fetch_audio_features` omission of `rms_db`/`silence_intervals` is
  addressed — locally, for the inspected file only, NOT globally per turn.)
- Add `observe.inspect_cut(ctx, *, ref=None, seg_id=None, document=None) -> dict`
  that resolves `ref`→cut meta (file_id + true `src_in_ms`/`src_out_ms` carried
  on the moment per A3), calls `_fetch_signal_window`, and downsamples.

### B2. Payload shape + resolution cap

Fixed cap constant `observe._INSPECT_MAX_SAMPLES = 24`. Sampling step =
`max(hop_ms, ceil(span_ms / _INSPECT_MAX_SAMPLES))` so we **never upsample past
the native L1 hop** (~100 ms) and never exceed 24 curve points per channel.

```json
{
  "ref": "1a2b3c4d:m7",
  "file": "1a2b3c4d",
  "span_ms": 6400,
  "hop_ms": 100,
  "action": {
    "curve": [0.1, 0.2, 0.6, 0.9, 0.4, ...],        // <= 24 normalized samples
    "hits":  [{"off": 820, "score": 0.91}, ...]       // <= 8, offsets from cut start
  },
  "audio": {
    "curve": [0.3, 0.35, 0.8, ...],                   // <= 24 normalized rms samples
    "changes": [{"off": 500, "dir": "up"}, ...],      // <= 8
    "silence": [{"off": 1000, "dur": 400}, ...]       // <= 8
  },
  "shots": {
    "cuts": [{"off": 2000, "hard": true}, {"off": 4500, "hard": false}]  // <= 12
  }
}
```

- Everything is WINDOWED to `[src_in_ms, src_out_ms]` and offsets are ms from
  `src_in_ms`.
- Curves are the clip-relative-normalized (`_norm_in_clip`) downsample of the
  span (bin-mean), never raw sample dumps.
- A channel with no data for this file/span is returned as an explicit
  `{"curve": [], ...}` / omitted key — never fabricated.
- Point caps: action hits ≤ 8, audio changes ≤ 8, silence gaps ≤ 8, shot cuts
  ≤ 12 (keep the strongest / longest, ordered by time in the payload).

This sense does not enter the beat index, so it has **no `_INDEX_CHAR_CAP`
impact** — it is a tool result the brain reads only when it asks.

### B3. Where it runs

Entirely **[API/brain-runtime]** — new sense + targeted DB read in the API
process. No ingest change, no migration, no RunPod rebuild, and it works on
EXISTING runs immediately (it reads live L1 arrays, which pre-date this plan).

---

## Change 1 — optional extras (include only if clean; flag as optional)

- `audio_features.true_peak_db` — a single whole-file scalar; if surfaced, put
  it on the CLIP header line (`_clip_block`, `footage_map.py:941-954`), not
  per-cut. Optional, low value; skip unless trivial.
- `transcripts.language` — clip-level; add `lang:<code>` to the CLIP header when
  present and non-English. Optional.
- Dialogue `topic[]` granularity — optional; do not expand unless it is a free
  passthrough of an existing field. Not required for this plan.

---

# Change 2 — Provenance / self-model prompt section

## Goal

Add a distinct, detailed section to the brain's system prompt describing HOW
everything it reads was produced — segmentation → cuts → scoring → takes/outlooks
→ tags → identity → beat index / program map — **including the new breadcrumb +
on-demand senses from Change 1**.

**STRICT RULE (user directive):** describe HOW IT IS PRODUCED ONLY. Do NOT
include any "trust this / reliable / lossy / approximate / faithful" framing.
The brain judges reliability itself from context. Purely descriptive provenance.

## Where it goes (cached with the prefix) [API/brain-runtime]

- The system prompt is assembled in `converse.respond` (`converse.py:390-391`):
  `system = _LOOP_SYSTEM + _guidance_block() + "\n\n" + _context_block(...)`.
  `_LOOP_SYSTEM` (`converse.py:61-171`) and `_guidance_block()` (a cached file
  read, `converse.py:203-207`) form the STATIC cached prefix; `_context_block`
  is the per-turn dynamic tail.
- Add a new module constant `_PROVENANCE` (a triple-quoted string) and change
  the assembly to:
  `system = _LOOP_SYSTEM + _guidance_block() + _PROVENANCE + "\n\n" + _context_block(...)`.
  Because `_PROVENANCE` is a static constant, it stays inside the cached prefix
  (same cache behavior as `_LOOP_SYSTEM`/guidance). Place it AFTER guidance and
  BEFORE `_context_block` so the dynamic beat index remains the suffix.
- Also add the two new beat-line facts to the existing "READING A BEAT LINE"
  glossary in `_LOOP_SYSTEM` (`converse.py:110-122`, the "Then tags:" list): a
  one-clause description of `sig:` (the breadcrumb) and a pointer that
  `inspect_cut` returns the detail. Keep it descriptive, no reliability framing.

## `_PROVENANCE` section outline (headings + what each describes)

Header line: `HOW YOUR SENSES ARE PRODUCED. Everything below describes how the
text you read was made from the footage.` Then:

1. **From footage to cuts.** One GPU pass (L1) derives per-file signals from the
   proxy and audio: motion (action energy, camera motion, blur), audio (loudness
   envelope, silence, onsets), and scene/shot boundaries; a transcript with
   word timings and speaker diarization. A first text pass segments each clip
   into cuts along word and shot edges; a vision pass describes each cut from
   sampled frames; assembly snaps every boundary to word/atom edges.
2. **Cut spans and hero frame.** How `src_in_ms`/`src_out_ms` are set (word/atom
   edges), how `hero_ts_ms` is chosen (anchor > sharpest > midpoint).
3. **Scoring.** How `speech_quality` (delivery: loudness + crispness over the
   span) and `total_quality` (speech + visual) are computed clip-relative, and
   that `PIC`'s `q.XX` is the visual score.
4. **Takes & outlooks.** How same-words/same-setting cuts are grouped into a
   take (one winner crowned by score), and how same-words/different-camera cuts
   are grouped into an outlook that shares one authoritative audio track.
5. **Salience / peak.** How `peak:+Xs` is the argmax of a fused
   action+loudness+onset curve over the span, as an offset from the cut start.
6. **Camera / energy / pace.** How `cam:` is derived from the signed camera
   velocity model, how `nrg:` levels and `pace:`/`trim≤Xs` come from the pace
   envelope.
7. **Continuity.** How `cut:N/of` numbers a cut among ALL its clip's cuts
   (including junk), and how `↔`/`⋯` mark whether each neighbor welds.
8. **The new per-cut signal breadcrumb (`sig:`).** Describe that `sig:` reports
   COUNTS of interior action hits (`act`), audio change points (`adx`), silence
   gaps (`sil`), and internal shot/composition cuts (`shot`) that L1 detected
   within the cut — and that `inspect_cut` returns the windowed detail for one
   cut.
9. **On-demand cut inspection (`inspect_cut`).** Describe that it returns, for
   one cut, a downsampled action-energy curve + hit offsets, a downsampled
   loudness envelope + rise/fall change offsets + silence-gap offsets, and the
   internal shot/composition-cut offsets — all measured from the cut's start and
   sampled at the L1 hop.
10. **Identity / CAST.** How voices are diarized then clustered across clips, how
    a speaking voice is bound to a person, how on-screen persons come from face
    clustering, and how the CAST line and `Px` ids are produced.
11. **The BEAT INDEX and PROGRAM MAP.** How the beat index lists every cut in
    source order per clip with a placeable `ref`, and how the program map is
    rendered from the resolved layer stack on the shared program clock.

Each item is 1–3 descriptive sentences. No adjectives of reliability anywhere.

Where it runs: **[API/brain-runtime]** only — a prompt-string change; effective
next turn, no re-ingest, no rebuild.

---

# Phased breakdown (execution order)

| # | Phase | Where it runs | Migration? |
|---|---|---|---|
| 1 | Migration `050`: `cut_records.landmarks jsonb` | [Render L3-ingest] (DDL) | **Yes** |
| 2 | `post.py`: `_landmarks(...)`, `CutRecord.landmarks` + `to_dict`, compute in `assemble_cut_records`; thread `scene_by_file` | [Render L3-ingest] | — |
| 3 | `ingest.py`: pass `scene_by_file` to `assemble_cut_records` | [Render L3-ingest] | — |
| 4 | `ingest_store.insert_cut_records`: write `landmarks` | [Render L3-ingest] | — |
| 5 | `cuts_read.rows_for_run`: SELECT `landmarks` | [API/brain-runtime] | — |
| 6 | `cutrecord_map._to_cut_dict`: carry `landmarks` (+ `src_in_ms/src_out_ms` on the moment via `footage_map`) | [API/brain-runtime] | — |
| 7 | `footage_map`: moment carries `landmarks`/`src_*`; `_landmarks_tag` + `sig:` in `_moment_line` | [API/brain-runtime] | — |
| 8 | `observe.py`: `_fetch_signal_window` + `inspect_cut`; downsample + caps | [API/brain-runtime] | — |
| 9 | `tools.py`: register + dispatch `inspect_cut` | [API/brain-runtime] | — |
| 10 | `converse.py`: `_PROVENANCE` constant + assembly; `sig:`/`inspect_cut` glossary lines in `_LOOP_SYSTEM` | [API/brain-runtime] | — |
| 11 | `frontend/src/lib/api.ts`: optional `landmarks?` on `CutRecord` | frontend build | — |
| 12 | Re-ingest to populate `landmarks` on existing projects; verify | [Render L3-ingest] | — |

Phases 1–4 + 12 are the only ones needing a re-ingest to take effect on existing
projects. Phases 5–11 are read/runtime and take effect on the next turn.
**No phase touches L1 → no RunPod rebuild.**

Suggested landing: phases 1–4 (ingest write path) together, then 5–7 (breadcrumb
read/render), then 8–9 (sense), then 10 (prompt), then 11 (frontend), then 12
(re-ingest + verify). Each group is independently testable.

---

# Migration

One new file `backend/migrations/050_cut_landmarks.sql` (next number after
`049_exports_resolved_hash.sql`), idempotent/additive:

```sql
alter table public.cut_records
    add column if not exists landmarks jsonb not null default '{}'::jsonb;

comment on column public.cut_records.landmarks is
    'Compact per-cut signal landmarks distilled at ingest from L1 arrays '
    '(motion_dynamics.action_energy/action_points, audio_features.rms_db/'
    'silence_intervals, scene_cuts.shot_points/composition_points): interior '
    'action hits, audio change points, silence gaps, internal shot cuts -- '
    'counts + capped offset lists. Powers the beat-line sig: breadcrumb and '
    'warms inspect_cut. {} on a pre-migration / no-signal cut.';
```

- Mechanism B (on-demand `inspect_cut`) needs **NO migration** — it reads live
  L1 arrays at request time.
- So **Change 1 needs exactly ONE small additive migration** (for the
  breadcrumb). **Change 2 needs no migration.**

---

# Char-budget analysis (beat index, `_INDEX_CHAR_CAP = 110_000`)

- Only Mechanism A adds to the beat index. Per-cut marker ≤ 25 chars worst case,
  ~10 typical, 0 when a cut has no interior structure.
- ~500-cut project: ~6 KB typical / ~12.5 KB absolute-worst added to a 110 KB
  cap. No realistic project is pushed to truncation by the marker; the existing
  truncation guard (`converse.py:282-285`) stays as backstop.
- Mechanism B and the `_PROVENANCE` section do NOT enter the beat index (the
  sense is a tool result; the prompt section is in the cached system prefix), so
  neither counts against `_INDEX_CHAR_CAP`.

---

# Backward compatibility

- `cut_records.landmarks` defaults to `{}`; every pre-migration row (and any run
  not yet re-ingested) reads `{}` → no `sig:` marker, exactly today's beat line.
  `cutrecord_map`/`footage_map` already guard with `or {}`.
- `assemble_cut_records`'s new `scene_by_file` param defaults to `{}` (any other
  caller/test keeps working; the shot channel is simply absent).
- `inspect_cut` works on ANY run (it reads live L1), including runs ingested
  before this plan; a cut whose file has no `motion_dynamics`/`scene_cuts` row
  returns empty channels, never an error.
- `ingest_runs` unchanged. No read-path field is made required; no existing
  column is altered. Old threads (`edit_threads.ingest_run_id` pin) behave
  identically.

---

# Verification (do before calling done)

Static:
1. `pyflakes backend/app/services/l3/post.py backend/app/services/l3/ingest.py
   backend/app/services/l3/ingest_store.py backend/app/services/l3/cuts_read.py
   backend/app/services/l3/cutrecord_map.py backend/app/services/l3/footage_map.py
   backend/app/services/l3/observe.py backend/app/services/l3/tools.py
   backend/app/services/l3/converse.py` — no unused/undefined.
2. Backend tests: run the L3 suite (`cutrecord_map`, `footage_map`, `observe`,
   `post`, `tools` tests). Add:
   - a `_landmarks` unit test (interior counts, edge-guard exclusion, caps,
     empty on no-signal),
   - a `_landmarks_tag`/`_moment_line` test asserting `sig:` renders only
     present channels with capped counts and is absent on `{}`,
   - an `inspect_cut` test (windowing to span, downsample ≤ 24, point caps,
     empty channels on a file with no rows, resolves by `ref` and by `seg_id`),
   - a `gemini`/schema-agnostic assertion is N/A (no LLM field added).
3. Frontend: `tsc` passes with the optional `landmarks?` type.

Local end-to-end (`GPU_EXECUTION=local` worker OR RunPod for L1; L3 ingest on the
local worker):
4. Upload a VIDEO with clear action + internal shot changes and an AUDIO-bearing
   clip with pauses; run L1 (local `GPU_EXECUTION=local` or RunPod), then
   re-ingest the project on `local-dev`.
5. Inspect a cut's `cut_records.landmarks` in the DB — counts/offsets sane and
   interior; confirm silent/static cuts have `{}`.
6. Open an edit thread and read the BEAT INDEX (log `converse` `system` or dump
   `footage_map.assemble_map`): confirm `sig:` markers appear on cuts with
   structure and are absent otherwise; confirm the index stays within
   `_INDEX_CHAR_CAP`.
7. Drive `inspect_cut` on a marked cut (via a `tools`-loop smoke script or a
   direct `observe.inspect_cut` call): confirm windowed, downsampled payload
   with the documented caps; offsets are within the span.
8. Inspect the assembled brain prompt: confirm the `_PROVENANCE` section is
   present in the cached prefix (before the beat index), contains the 11
   headings, mentions `sig:` + `inspect_cut`, and contains NO
   trust/reliable/lossy/approximate wording.

---

# File-touch summary

- **New:** `backend/migrations/050_cut_landmarks.sql`.
- `backend/app/services/l3/post.py` — `_landmarks`, `CutRecord.landmarks` +
  `to_dict`, compute + `scene_by_file` param.
- `backend/app/services/l3/ingest.py` — pass `scene_by_file` to `post`.
- `backend/app/services/l3/ingest_store.py` — write `landmarks`.
- `backend/app/services/l3/cuts_read.py` — SELECT `landmarks`.
- `backend/app/services/l3/cutrecord_map.py` — carry `landmarks`.
- `backend/app/services/l3/footage_map.py` — moment carries `landmarks` +
  `src_in_ms/src_out_ms`; `_landmarks_tag` + `sig:` in `_moment_line`.
- `backend/app/services/l3/observe.py` — `_fetch_signal_window` + `inspect_cut`.
- `backend/app/services/l3/tools.py` — register + dispatch `inspect_cut`.
- `backend/app/services/l3/converse.py` — `_PROVENANCE` + assembly + glossary.
- `frontend/src/lib/api.ts` — optional `landmarks?` on `CutRecord`.
- Tests as listed under Verification.
