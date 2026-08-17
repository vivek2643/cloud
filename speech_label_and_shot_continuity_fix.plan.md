# Fix plan — transcript-driven speech labeling + shot-aware continuity

Two **generic root-cause fixes** for the bugs behind the broken-transcript edit
(thread `720713d9`, doc `9ffb16bf` v1, `by=auto`, run `24894c8a`, 2026-08-12
16:11 EDT — on the continuity-aware code):

- **Fix 1** — speech presence is derived from the **actual transcript** (Whisper words overlapping a cut's source span), not from cut type / a pace-envelope proxy. A cut carrying words is classified speech-bearing → its words are **surfaced** in the beat index AND the existing coverage-mute policy **fires correctly**.
- **Fix 2** — a "same-shot jump" is decided from **actual shot boundaries**, not from `file_id` + source time alone. A seam inside one file where the picture genuinely changed (a shot boundary between the two source instants) is a normal `cut`, not a `jump`.

> **These are the GENERIC fixes and they explicitly REPLACE the earlier band-aid
> ideas, which are rejected:** "mute slide-type cuts" (keys on cut type, not
> truth) and "gate/disarm Stage 2.5" (hides the symptom). We fix the underlying
> labels: (1) speech from the transcript, (2) continuity from shot boundaries.

---

## 1. Problem statements (with broken-edit evidence)

**Fix 1 — mislabeled incidental speech.** In `720713d9`, the brain built a spine
of `channel="shown"` slide/demo cuts it believed were silent. The beat index
showed `said=False`, and the render left them un-muted (`audio="sound"`,
`mute_default=False`), so the founder's continuous voiceover played as
mid-sentence fragments assembled in *picture* order. Confirmed against the run:
every slide span DOES contain transcript speech (read-only check on
`dialogue_segments` for `7144c9fc`):

| slide cut (src ms) | overlapping transcript |
|---|---|
| 45100–48800 | "So if we could actually analyze videos, could we actually even edit them?" |
| 69100–73800 | "and this is really powerful … So current[ly]…" (mid-sentence) |
| 147900–153100 | "AI integrate with a storage layer … the product is a combination of a cloud[ drive]…" |
| 173200–178400 | "So it's storage and created one place. So the market for this is huge." |

The words were always there — the label was wrong.

**Fix 2 — false same-shot-jump flood.** `feel.join_continuity` keyed "jump" on
`file_id` + source time only. The pitch recording `7144c9fc` holds logo +
talking-head + slides + demo in ONE take, so every content change tripped a
"same-file non-contiguous → jump". On this edit that produced ~7 flags; the
brain spent its **entire** finishing turn rebutting them (all correctly, since
the picture genuinely changes each time) and never audited the spoken narrative.
The flags were false positives because the classifier is blind to whether the
picture actually changed.

---

## 2. Current-state findings (file:line)

### 2.1 Where the speech/audio/mute labels originate

- **`channel`** is set at ingest and persisted on `cut_records`: video cuts get `channel="shown"` in `vcut/store.py:138` (`build_cut_records`, `kind="video"`); speech cuts get `channel="said"` from the speech channel. This is CORRECT and we do **not** change it — a slide's *picture* is a slide.
- **`audio` / `mute` / flags** are **derived at read time** (not persisted) in `cutrecord_map._audio_mute_for` (`cutrecord_map.py:476-488`):

```476:488:backend/app/services/l3/cutrecord_map.py
def _audio_mute_for(channel: str, row: Dict[str, Any]) -> Tuple[Optional[str], bool, List[str]]:
    ...
    if channel == "said":
        return None, False, []
    natural_sound = bool((row.get("pace") or {}).get("natural_sound"))
    if natural_sound:
        return "sound", False, []
    return "silent", True, ["muted"]
```

  The bug: it keys on `pace.natural_sound` (a "is source sound worth keeping" proxy), **never on whether that sound is SPEECH**. For the slides `natural_sound` was True → `audio="sound"`, `mute=False` → the voiceover plays.
- Consumed in `cutrecord_map._to_cut_dict:494` (`audio, mute, flags = _audio_mute_for(...)`) → cut dict fields `audio`/`mute`/`flags` (`:526-528`).
- **`said_text`** (the words shown in the beat index) is gated on `channel=="said"` in `footage_map.build_clip_tree`:

```281:282:backend/app/services/l3/footage_map.py
        said_text = (_said_text_for_span(file_id, anchor["in_ms"], anchor["out_ms"])
                    if cut.get("channel") == "said" else "")
```

  So a `shown` cut's overlapping words are computed away → the brain is blind to them. `_moment_line` renders `said_text` as the primary quote (`footage_map.py:1118-1119, 1161-1164`).
- **Mute flows to render**: moment `mute` (`build_clip_tree:323` `"mute": bool(cut.get("mute"))`) → `arrange._MapIndex.resolve` folds it into `ResolvedCut.mute` (`arrange.py:170,185` via `_resolve_mute`, `arrange.py:127-135`) → `act._seg_from_resolved` sets the spine seg's `"mute": True if rc.mute else None` (`act.py:85`) and `"axis": "speech" if rc.channel=="said" else "any"` (`act.py:77`). So fixing `mute` at the source (2.1) automatically mutes at render for future placements.

### 2.2 Where the transcript / word timings live, and the overlap test

- Per-file transcript is the L1 `dialogue_segments` table (immutable once L1 writes it). `footage_map._sentences_for_file(file_id)` reads it (`footage_map.py:1361-1389`, `select segments from dialogue_segments`, lru-cached), returning sentence rows.
- Sentence span: `footage_map._sentence_span(s)` → `(src_in_ms, src_out_ms)` (`footage_map.py:1392-1396`). Overlap test: `footage_map._overlap_ms(a0,a1,b0,b1) > 0` (`footage_map.py:155-156`).
- `footage_map._said_text_for_span(file_id, in_ms, out_ms)` already returns the verbatim words over ANY span (`footage_map.py:1487+`) — today only called for `said` cuts. **"Does this cut carry speech?"** = `any(_overlap_ms(*_sentence_span(s), in_ms, out_ms) > 0 for s in _sentences_for_file(file_id))`.
- Read-only confirmation: `dialogue_segments` for `7144c9fc` has 31 sentences that fully cover the slide spans (table in §1). So speech-presence is derivable at read time with **zero re-ingest**.

### 2.3 The join classifier and its inputs

```208:240:backend/app/services/l3/feel.py
def join_continuity(cuts: List[CutFeel]) -> List[dict]:
    ...
    for i in range(1, len(cuts)):
        prev, cur = cuts[i - 1], cuts[i]
        if not prev.file_id or not cur.file_id or prev.file_id != cur.file_id:
            kind = "cut"
        else:
            gap = cur.src_in_ms - prev.src_out_ms
            if -_JOIN_CONTIG_MS <= gap <= _JOIN_CONTIG_MS:
                kind = "seamless"
            else:
                kind = "jump"
        ...
        "backward": prev.file_id == cur.file_id and cur.src_in_ms < prev.src_out_ms,
```

- Inputs are only `CutFeel.file_id` / `src_in_ms` / `src_out_ms` (`feel.py:37,48-49`), populated in `simulate` from the seg's `file_id`/`in_ms`/`out_ms` (`feel.py:119,123,137`). Tolerance `_JOIN_CONTIG_MS = 500` (`feel.py:35`). No picture/shot input at all — that is the whole bug.
- Consumers: `read_state` (`observe.py:490,580-587`), `diagnose` (`observe.py:1198-1201`), `review` (`observe.py:1532-1537`, `category="continuity"`), `feel.narrate` (`feel.py:90-93`), and the done-gate Stage 2.5 which reuses `diagnose`'s findings (`tools.py:697-704`).

### 2.4 Shot-boundary availability (the hard part — investigated)

- **Canonical source: `scene_cuts`** (L1). Columns `hop_ms, shot_points, composition_points`; `shot_points`/`composition_points` are lists of dicts with `ts_ms` (absolute source ms) — see `landmarks._shot_cuts` reading `pt.get("ts_ms")` (`landmarks.py:213-215,221-223`). `observe._fetch_signal_window` already reads this table (`observe.py:274-277,283`).
- **BUT it is EMPTY for the broken edit's files** (read-only check): `scene_cuts` has **no row** for `7144c9fc` or `93e94ec3` (`aad020e3` has 1 `shot_point`, 0 `composition_points`). Table-wide, most files have `shot_points=0`. So the L1 discrete shot boundaries are **not currently reachable** for exactly these files, and the `cut_records.landmarks.shot` channel is `None` for their video cuts (it's dark whenever `scene_cuts` is empty — see `vcut/orchestrate.py:107-147`, `_add_landmark_signals_to_seam_cache`: "A file with no … scene_cuts row simply gets no … shot signal").
- **Reachable-today source (no re-ingest): the persisted seam cache `frame_diff`.** `vcut` persists per-file `{hop_ms, S, action_energy, frame_diff}` into `ingest_runs.seam_cache` (`vcut/store.py:182-184` `persist_seam_and_plan`; `vcut/store.py:194` `load_seam_and_plan`). Read-only check on run `24894c8a`: `frame_diff` is present and full-length for all three files (`7144c9fc`: len 2444 @ `hop_ms=100`; `93e94ec3`: len 6834). `frame_diff` is the normalized frame-difference curve — high where the picture changes (slide transitions, camera cuts) — so discrete shot boundaries can be peak-picked from it at read time. It is noisier than `scene_cuts` and dense on busy screen recordings (that's fine — see §4.3).

**Conclusion:** the classifier logic (Fix 2A) is easy; the real work is plumbing a
per-file shot-boundary source (Fix 2B). Primary source `scene_cuts` is correct
but empty for these files; the persisted `frame_diff` gives a no-re-ingest bridge
that makes run `24894c8a` work today; a L1 `scene_cuts` backfill is the clean
long-term source.

---

## 3. Fix 1 design — transcript-driven speech labeling

**Rule, keyed on transcript truth (not cut type):** a cut is *speech-bearing* iff
any transcript word overlaps its source span. Then:

- **Surface** the words in the beat index for every speech-bearing cut (so the brain is never blind to speech that will play).
- **Mute by default** speech that rides under a *picture-chosen* cut (`channel != "said"`) — this is incidental/coverage talk, exactly what the existing coverage-mute policy is for. The brain can still `audio:keep` it or place the real speech beat.

A `said` cut is unchanged (its audio is the point). A truly silent picture cut is
unchanged (no overlap → old `natural_sound` behavior).

### 3.1 Compute speech-presence once per file (cheap, cached)

Add to `cutrecord_map.py` a per-file transcript index (mirrors
`footage_map._sentences_for_file`; keep it local so `cutrecord_map` stays the
ingest-facing layer and doesn't import `footage_map`):

```python
from functools import lru_cache

@lru_cache(maxsize=512)
def _sentence_bounds(file_id: str) -> Tuple[Tuple[int, int], ...]:
    """Sorted (src_in_ms, src_out_ms) of this file's dialogue_segments sentences.
    () when the file has no transcript. One DB read per file (immutable L1)."""
    ...  # select segments from dialogue_segments where file_id = %s

def _has_speech(file_id: str, in_ms: int, out_ms: int) -> bool:
    return any(min(b, out_ms) - max(a, in_ms) > 0 for a, b in _sentence_bounds(file_id))
```

### 3.2 Correct `_audio_mute_for` (the render-side truth)

```python
def _audio_mute_for(channel: str, row: Dict[str, Any], has_speech: bool
                    ) -> Tuple[Optional[str], bool, List[str]]:
    if channel == "said":
        return None, False, []                 # its audio is the point — unchanged
    if has_speech:                             # incidental talk under a picture cut
        return "speech", True, ["muted(talk)"] # coverage-mute by default; brain can audio:keep
    natural_sound = bool((row.get("pace") or {}).get("natural_sound"))
    if natural_sound:
        return "sound", False, []              # real ambient worth keeping — unchanged
    return "silent", True, ["muted"]           # unchanged
```

Call it in `_to_cut_dict` (`cutrecord_map.py:494`) with
`has_speech=_has_speech(row["file_id"], int(row["src_in_ms"]), int(row["src_out_ms"]))`.
This makes `cut["mute"]=True` for speech-bearing picture cuts, which flows through
`build_clip_tree → _MapIndex.resolve → act` to mute at render (§2.1).

### 3.3 Surface the words in the beat index

In `footage_map.build_clip_tree` drop the `channel=="said"` gate so any
speech-bearing cut quotes its words (`footage_map.py:281-282`):

```python
        said_text = _said_text_for_span(file_id, anchor["in_ms"], anchor["out_ms"])
```

`_said_text_for_span` already returns `""` when nothing overlaps (`footage_map.py:1494`),
so truly-silent cuts are unaffected and no line is fabricated. `_moment_line`
already renders `said_text` as the primary quote and the `muted(talk)` flag rides
in `flags`, so the brain sees both the words and that they're muted-by-default.
(Optional polish: in `_moment_line`, when `channel!="said"` and `said_text` is
present, prefix it `talk(muted):"…"` so it reads as incidental rather than the
cut's own line — cosmetic, not required.)

### 3.4 Re-ingest? — NO. Read-side only + cache-version bumps.

- `channel` (persisted) is untouched; `audio`/`mute`/`said_text` are all derived at read time from the already-persisted `dialogue_segments` (L1) + `cut_records`. **No re-ingest.**
- Because footage trees are cached (`footage_trees`, keyed by signature + `TREE_VERSION`) and the signature embeds `CUTRECORD_MAP_VERSION`, bump both so caches rebuild with the new labels:
  - `CUTRECORD_MAP_VERSION` `5 → 6` (`cutrecord_map.py:59`) — audio/mute derivation changed.
  - `TREE_VERSION` `22 → 23` (`footage_map.py:121`) — `said_text` gate changed.
- **Verify on run `24894c8a`:** rebuild `7144c9fc`'s tree and assert cuts `m03/m04/m05/m06` now have non-empty `said_text`, `audio=="speech"`, `mute==True`, `flags==["muted(talk)"]`; assert a genuinely silent picture cut (no transcript overlap) is unchanged (`audio` in {"sound","silent"}, old behavior).

---

## 4. Fix 2 design — shot-aware continuity

### 4.1 Corrected classification

`join_continuity(cuts, shot_bounds)` where `shot_bounds: Dict[file_id, List[int]]`
is a sorted list of shot-boundary source-ms per file. For each adjacent pair:

- different file → `cut` (unchanged).
- same file, `|gap| ≤ _JOIN_CONTIG_MS` (contiguous) → `seamless` (unchanged).
- same file, non-contiguous/backward, **AND no shot boundary strictly between the two cut instants** → `jump` (the real same-shot jump).
- same file, non-contiguous, **but a shot boundary lies between them** (picture changed) → `cut`, NOT a jump.

The two "cut instants" that form the visible seam are `prev.src_out_ms` and
`cur.src_in_ms`:

```python
def _same_shot(shot_bounds, file_id, a_ms, b_ms) -> bool:
    lo, hi = (a_ms, b_ms) if a_ms <= b_ms else (b_ms, a_ms)
    bounds = shot_bounds.get(file_id) if shot_bounds else None
    if not bounds:                       # no shot evidence -> can't prove same shot
        return False                     # fail-open to `cut`: never flood false jumps
    return not any(lo < b < hi for b in bounds)
```

```python
def join_continuity(cuts, shot_bounds=None):
    ...
        else:  # same file
            gap = cur.src_in_ms - prev.src_out_ms
            if -_JOIN_CONTIG_MS <= gap <= _JOIN_CONTIG_MS:
                kind = "seamless"
            elif _same_shot(shot_bounds, cur.file_id, prev.src_out_ms, cur.src_in_ms):
                kind = "jump"            # non-contiguous within ONE continuous shot
            else:
                kind = "cut"             # picture changed across a shot boundary
```

**Fail-open policy (explicit tradeoff):** a file with no reachable shot data →
`_same_shot` returns False → seams there are `cut`, never `jump`. This kills the
false-positive flood on multi-content takes (the reported bug). The cost — a
missed genuine jump on a shot-data-less file — is recovered once shot boundaries
are available (§4.3). This is a principled default: without evidence that the
picture stayed the same, don't cry "same-shot jump".

### 4.2 Plumb `shot_bounds` into the read path

- Add `shot_bounds: Dict[str, List[int]] = field(default_factory=dict)` to `EditContext` (`observe.py` dataclass ~`:60-100`).
- Populate once per turn in `build_context` (`observe.py:103+`, alongside `durations`/`color_stats`) via a new `_fetch_shot_bounds(file_ids, run_id)` (§4.3).
- Thread `shot_bounds` through `feel`: give `FeelReport` a `shot_bounds` field, set it in `simulate(timeline, meta_by_ref, shot_bounds=None)`, and have `narrate()` (`feel.py:90-93`) call `join_continuity(self.cuts, self.shot_bounds)`. Update the module `join_continuity` signature (§4.1).
- Pass `ctx.shot_bounds` at every consumer, and pass it into the `feel.simulate(...)` calls those consumers already make so narration agrees:
  - `read_state`: `report = feel.simulate(timeline, ctx.meta_by_ref, ctx.shot_bounds)` (`observe.py:486`), `joins = {j["to_pos"]: j for j in feel.join_continuity(report.cuts, ctx.shot_bounds)}` (`observe.py:490`).
  - `diagnose` (`observe.py:1174,1198`), `review` (`observe.py:1532,1536`) — same two changes.
  - `tools._verify_before_finish` needs **no** change: it calls `observe.diagnose(working, ctx)` (`tools.py:660`), which now returns the corrected findings; Stage 2.5 (`tools.py:697-704`) consumes them. No separate gating.

### 4.3 The shot-boundary data source (`_fetch_shot_bounds`)

Build `shot_bounds` per file, preferring the authoritative source and falling
back so it works on run `24894c8a` today:

1. **Primary — `scene_cuts`:** `select hop_ms, shot_points, composition_points from scene_cuts where file_id = %s`; collect `int(pt["ts_ms"])` from `shot_points` (hard cuts) and `composition_points` (framing changes = picture change enough to defeat a "same-shot" claim). Sorted, deduped. (Same table `observe._fetch_signal_window` already reads.)
2. **Bridge (no re-ingest) — persisted `frame_diff`:** for a file with no/empty `scene_cuts` row, load `ingest_runs.seam_cache` via `vcut.store.load_seam_and_plan(run_id)` and peak-pick discrete boundaries from `frame_diff` (curve at `hop_ms`): a boundary at hop `i` where `fd[i]` is a local max above `mean + k*(max-mean)` with a min-separation (e.g. ≥ 500 ms); boundary ms = `i * hop_ms`. This makes shot boundaries reachable for exactly the vcut-processed files (confirmed present for all three files of this run). Note: on busy screen recordings `frame_diff` is dense, so most same-file seams correctly resolve to `cut` (the picture really is changing) — which is the desired outcome, not a defect.
3. **Long-term correct — L1 `scene_cuts` backfill (re-ingest):** the root reason boundaries are missing is that L1 scene detection never persisted `scene_cuts` for these files, which also leaves `cut_records.landmarks.shot` dark. Running the L1 scene stage for all files (a re-ingest or a targeted backfill job) makes `scene_cuts` the clean, thresholded, discrete source and lights up the landmark shot channel everywhere. This is the only part that needs a re-ingest, and it is **not** required for Fixes 1/2 to function — it upgrades the signal quality.

Recommended default: **primary `scene_cuts` + bridge `frame_diff`** (both read-time,
no re-ingest); schedule the L1 backfill separately.

### 4.4 Re-ingest? — NO for the classifier.

`feel`/`observe` changes are pure read-side; `scene_cuts` and
`ingest_runs.seam_cache.frame_diff` are already persisted. **No version bump**
(feel/observe aren't cached; `shot_bounds` is computed fresh per turn). Only the
optional §4.3(3) L1 backfill is a re-ingest, and it's independent.

### 4.5 Confirmation against the reported failure

- The `7144c9fc` seams (logo→intro→slide→slide→demo→…): with `scene_cuts` empty but `frame_diff`-derived boundaries present (or, worst case, fail-open), the picture-change seams resolve to `cut` → **the ~7 false jumps disappear**.
- A genuine static-shot backward jump (e.g. clip93 408→209 s with **no** shot boundary between 209 460 and 413 350) → `_same_shot` True → `jump` still fires. (Unit-tested with an explicit `shot_bounds` fixture, §6, so it's deterministic regardless of the real curve.)

---

## 5. Exact change list per file

| File | Change | Version bump | Re-ingest |
|---|---|---|---|
| `backend/app/services/l3/cutrecord_map.py` | Add `_sentence_bounds`/`_has_speech`; extend `_audio_mute_for(channel,row,has_speech)` (`:476`); pass `has_speech` in `_to_cut_dict` (`:494`). | `CUTRECORD_MAP_VERSION` 5→6 (`:59`) | no |
| `backend/app/services/l3/footage_map.py` | Drop the `channel=="said"` gate on `said_text` (`:281-282`); optional incidental-quote prefix in `_moment_line`. | `TREE_VERSION` 22→23 (`:121`) | no |
| `backend/app/services/l3/feel.py` | `join_continuity(cuts, shot_bounds=None)` + `_same_shot` (`:208-240`); add `FeelReport.shot_bounds`; `simulate(..., shot_bounds=None)` stores it (`:96-129`); `narrate` passes it (`:90-93`). | none | no |
| `backend/app/services/l3/observe.py` | `EditContext.shot_bounds`; `_fetch_shot_bounds(file_ids, run_id)` + call in `build_context`; pass `ctx.shot_bounds` into `feel.simulate`/`feel.join_continuity` in `read_state` (`:486,490`), `diagnose` (`:1174,1198`), `review` (`:1532,1536`). | none | no |
| `backend/app/services/l3/tools.py` | none (Stage 2.5 `:697-704` consumes the corrected `diagnose`). | none | no |
| L1 scene stage (optional, §4.3-3) | Backfill `scene_cuts` for files missing it (upgrades signal, lights up landmark shot channel). | n/a | **yes (backfill only)** |

No schema changes. `channel` on `cut_records` is never rewritten.

---

## 6. Test plan

**Fix 1 — transcript-driven labels** (`scripts/test_cutrecord_map.py` + `scripts/test_footage_map.py`):
1. `_has_speech` True when a sentence overlaps `[in,out]`, False otherwise (synthetic `dialogue_segments`, monkeypatch `_sentence_bounds`).
2. `_audio_mute_for("shown", row, has_speech=True)` → `("speech", True, ["muted(talk)"])`; `has_speech=False` + `natural_sound` True → `("sound", False, [])` (unchanged); `natural_sound` False → `("silent", True, ["muted"])` (unchanged); `channel=="said"` → `(None, False, [])` (unchanged).
3. `build_clip_tree`: a `shown` cut whose span overlaps transcript gets non-empty `said_text` and `mute==True`; a `shown` cut with no overlap gets `said_text==""` and old mute behavior. (Reuse `test_footage_map`'s `_sentences_for_file` monkeypatch pattern.)
4. End-to-end mute: place such a cut via `act.place` and assert the spine seg carries `mute==True` (label reaches render).

**Fix 2 — shot-aware continuity** (`scripts/test_feel.py` + `scripts/test_observe_continuity.py`):
5. same-file, non-contiguous, shot boundary **between** the instants → `cut` (not `jump`).
6. same-file, non-contiguous/backward, **no** boundary between → `jump` (e.g. `prev.src_out=413350`, `cur.src_in=209460`, `shot_bounds={file:[...none in (209460,413350)...]}`).
7. multi-content take fixture (logo/slide/slide/demo spans, boundaries between each) → **zero** jumps.
8. same-file contiguous (`|gap|≤500`) → `seamless` regardless of bounds; different file → `cut`.
9. fail-open: `shot_bounds` empty for the file → non-contiguous same-file seam is `cut`, never `jump`.
10. surfacing flows through with the boundary-aware verdict: `read_state.cuts[i]["join"]`, `diagnose` same-shot-jump finding, `review` `category=="continuity"` flag, and `narrate` all agree (extend `test_observe_continuity.py`, passing `ctx.shot_bounds`).

Run with `backend/.venv/bin/python scripts/<t>.py`; pyflakes-check every changed file.

---

## 7. Note — these replace the rejected band-aids

- ~~"Mute slide-type cuts"~~ keyed on cut type; Fix 1 mutes/surfaces based on **transcript truth** (`_has_speech`), so it's correct for any cut carrying incidental talk and never touches a truly-silent slide.
- ~~"Gate/disarm Stage 2.5"~~ hides the symptom; Fix 2 makes the **verdict itself correct** (shot-aware), so Stage 2.5 keeps firing on real same-shot jumps and simply stops flooding false ones. No gating added.
