# Heal adjacent same-source cuts at timeline-composition time

**Status:** implementation plan / handoff. No code has been changed. Investigation was read-only.

**Owner of the decision points:** see §5 — the threshold default and the "tighten vs heal" question both need a human sign-off before coding.

---

## 1. Problem statement + demo-trail evidence

The agentic editor (and the manual "snap-to-timeline" builder) places consecutive speech beats onto the spine as independent segments. When two **adjacent placed segments come from the SAME source file** and their source timecodes are **contiguous or nearly so** (a tiny sub-second gap), the composed program cuts from `src_out` of segment N straight to `src_in` of segment N+1 of the *same continuous take*. Because a few frames of one continuous shot were dropped, this reads as a **micro jump-cut / glitch**, not an edit.

This should be **HEALED** — the two segments merged into one continuous clip that plays the source straight through (bridging the dropped frames) — rather than left as a cut. Healing is **deterministic**: same file + adjacent timecode is a pure structural fact; the brain must not have to reason about micro-gaps. It should happen automatically when the timeline is composed.

### Concrete evidence (project `8621c012`, thread `76d88f5d`, ingest run `6f31d648`)

Two **post-weld** spine segments (ids assigned by the current weld pass — see §2):

| seg | file | src in→out | note |
|-----|------|------------|------|
| `a002` | same file | … → **60805ms** | ends here |
| `a003` | same file | **60935ms** → … | starts here |

Source gap = `60935 − 60805` = **130ms** of one continuous take dropped between two adjacent spine segments of the *same file*. That 130ms drop is the glitch.

**Why it isn't healed today:** a weld pass already exists (`arrange._weld_segments`) with tolerance `_WELD_TOL_MS = 120` (`backend/app/services/l3/arrange.py:40`). Its merge test is:

```194:225:backend/app/services/l3/arrange.py
def _weld_segments(segments: List[dict]) -> List[dict]:
    ...
    welded: List[dict] = []
    for s in segments:
        prev = welded[-1] if welded else None
        if (prev is not None
                and prev["file_id"] == s["file_id"]
                and prev["in_ms"] <= s["in_ms"] <= prev["out_ms"] + _WELD_TOL_MS):
            prev["out_ms"] = max(prev["out_ms"], s["out_ms"])
            ...
            continue
        welded.append(s)
    for i, s in enumerate(welded):
        s["seg_id"] = f"a{i:03d}"
    return welded
```

For `a002`/`a003` the condition evaluates `prev.in ≤ 60935 ≤ 60805 + 120 = 60925` → `60935 > 60925` → **False**. The gap (130ms) exceeds the weld tolerance by **10ms**, so the two segments are kept as a hard cut. That single off-by-10ms is the entire observed defect — **and it only ever runs on the brain/auto-assembly path; the manual/snap path never welds at all** (§2). The `a002`/`a003` ids themselves prove the weld ran (it re-issues ids as `a{i:03d}`) and still left the pair split.

The fix is therefore not "add a new subsystem" but "**generalize the existing weld into a configurable, shared HEAL pass and run it on BOTH composition paths**".

---

## 2. Current-state findings (how placement & composition work today)

### 2.1 The edit document shape

- Persisted as `edit_documents.document` jsonb, versioned (`backend/migrations/014_l3_edit_threads.sql:37`). The document carries `timeline` (the spine), `operations`, optional `layout_regions`, `format`, `look`, `captions`, and a cached `resolved` snapshot.
- The **spine** = `document["timeline"]`, an ordered list of coupled segments. Each segment: `{seg_id, file_id, in_ms, out_ms, axis, mute, ref, level, audio_file_id, audio_offset_ms, gain_db?, fade_in_ms?, fade_out_ms?, ...}` (shape emitted by `act._segments_from_cut`, `backend/app/services/l3/act.py:60`; and by the manual add path, `frontend/src/stores/edit-doc-store.ts:455`).
- Segments are laid **end-to-end on the program clock** by `layers.spine_spans` (`backend/app/services/l3/layers.py:388`). **Adjacent spine segments therefore always abut on the timeline by construction** — the "abut on the timeline" precondition of the heal rule is automatically satisfied for any adjacent pair. The only variable is whether their *source* spans are contiguous.

### 2.2 How a placed cut becomes spine layers

- `act.place` (`backend/app/services/l3/act.py:190`) resolves a map ref → `ResolvedCut` and calls `_segments_from_cut` (`act.py:60`), which emits **one segment per `keep_span`** (so a cut's own excised jump-cut becomes distinct, intentionally non-contiguous segments). Raw slices are appended to `timeline` unwelded — "welding is a compile-time concern" (`act.py:9`, `act.py:63`).
- Each segment bakes its **authoritative audio coupling**: `audio_file_id` (defaults to its own `file_id`) and `audio_offset_ms` (`act.py:91`).
- `layers.resolve` (`backend/app/services/l3/layers.py:562`) turns the spine into flat layers: **one `VideoLayer` + one dialogue `AudioLayer` per spine segment** (`layers.py:610-664`). The audio layer's source window is derived from the segment's `in_ms/out_ms` (+ coupling offset). **Audio is a pure per-segment derivation of the spine** — it has no independent geometry.

  **Consequence for heal:** if two spine segments merge into one, `layers.resolve` will emit ONE video layer and ONE audio layer spanning the merged `[in_ms, out_ms]`, and the audio window automatically bridges the same gap the video does. **Audio stays aligned by construction** — the heal only needs to operate on the spine, never on audio layers directly.

### 2.3 Where composition/weld runs today (two divergent paths)

**Path A — brain / auto-assembly** (`converse.respond` → tool loop → `observe.resolve_doc`):
```287:318:backend/app/services/l3/observe.py
def resolve_doc(document: dict, ctx: EditContext) -> dict:
    ...
    old_ids = [s.get("seg_id") for s in (document.get("timeline") or [])]
    old_segs = list(document.get("timeline") or [])
    document["timeline"] = _weld_segments(document.get("timeline") or [])
    _remap_split_edits(document, old_ids, old_segs)
    ...
    document["resolved"] = layers.resolve(...).to_dict()
    return document
```
Called from `converse.respond` after a changed turn (`backend/app/services/l3/converse.py:625`). **This path welds and persists the welded timeline.** After welding, `_remap_split_edits` (`observe.py:321`) fixes `split_edit` ops keyed on `seam_seg_id`. (Note: it remaps **only `split_edit`, not `crossfade`** — see §5 edge cases.)

**Path B — manual edit / SNAP** (`PUT /api/edit/threads/{id}/document` → `resolve_document`):
```183:219:backend/app/routers/edit_threads.py
@router.put("/{thread_id}/document")
def put_document(...):
    ...
    timeline = _sanitize_timeline(body.timeline, durations)
    operations = _sanitize_operations(body.operations)
    new_doc = {**doc, "timeline": timeline, "operations": operations}
    ...
    new_doc["resolved"] = resolve_document(new_doc, thread_id=thread_id)
    version = store.save_document(thread_id, new_doc, created_by="user")
```
`_sanitize_timeline` (`edit_threads.py:72`) only clamps/validates — **it never welds**. `resolve_document` (`backend/app/services/render/tasks.py:80`) calls `layers.resolve` directly (`tasks.py:115`) with **no weld step**. **Therefore the manual/snap path never heals anything today.** This is why the heal must be added to Path B, not just tuned in Path A.

### 2.4 The SNAP feature (build-by-clicking)

- Toggled by `snapAddEnabled` on the timeline toolbar (`frontend/src/stores/timeline-view.ts:41`, `frontend/src/components/timeline-editor.tsx:919`).
- In the Cuts view, when snap is on, clicking a cut card appends it to the spine (`frontend/src/components/cuts-view.tsx:1595` → `onSnapAdd` → `addToTimeline` at `cuts-view.tsx:413`), which calls the store's `addSegment`:
```455:470:frontend/src/stores/edit-doc-store.ts
addSegment: (seg, atIndex) =>
  set((st) => {
    const inMs = Math.max(0, Math.round(seg.in_ms));
    const outMs = Math.max(inMs + MIN_SEG_MS, Math.round(seg.out_ms));
    const newSeg: EditSegment = { seg_id: rid("se"), file_id: seg.file_id, in_ms: inMs, out_ms: outMs };
    const next = [...st.timeline];
    const idx = atIndex == null ? next.length : ...;
    next.splice(idx, 0, newSeg);
    return { timeline: next, selectedIds: [`seg:${newSeg.seg_id}`] };
  }),
```
- **Snap placement lands entirely client-side** (a new `se_…` segment appended to the in-memory spine). It becomes server-authoritative only when the doc is saved via `PUT …/document` (Path B). **The client-side resolver (`frontend/src/lib/resolve-timeline.ts`) also does NOT weld** (confirmed: no weld/merge logic), so the snap *preview* won't heal until the doc is saved and refetched.
- **Where to call the shared heal for snap:** server-side in `put_document` (Path B), on the sanitized timeline, before resolve. That heals every human save — including a snap that lands adjacent to a same-source contiguous segment — with the exact same function the brain path uses. (Optional client-side mirror for instant preview parity — see §4.5.)

### 2.5 There already is a documented hook for exactly this

`backend/app/services/l3/seam.py` is a pure deterministic weld-vs-hard classifier whose module docstring names this feature as its unwired future caller:

```8:16:backend/app/services/l3/seam.py
  * FUTURE (documented hook, not wired here) -- timeline weld: when the editor
    drops two cuts adjacent, the SAME rule decides weld vs hard cut. Cross-clip
    is always hard (different footage), which is why ``same_clip`` is an
    explicit input rather than assumed.
```
`classify_seam` (`seam.py:67`) already encodes: cross-clip → hard, speaker-change → hard, shot/scene/transition in gap → hard, flagged production break → hard, gap-longer-than-bridged-speech → hard, else weldable. We reuse its *spirit* (structural guards) but keep the compose-time pass cheap (see §3 threshold discussion — the atom-level signals `classify_seam` wants aren't cheaply available at compose time).

---

## 3. Design — the shared deterministic heal function

### 3.1 Contract

Add ONE pure function in `backend/app/services/l3/arrange.py` (co-located with `_weld_segments`, which it supersedes):

```python
def heal_adjacent_cuts(
    segments: List[dict],
    *,
    gap_ms: int,
    reindex: bool = True,
) -> Tuple[List[dict], Dict[str, str]]:
    """Merge adjacent SAME-SOURCE spine segments whose source spans are
    contiguous or separated by at most `gap_ms` of dropped source, so one
    continuous take plays as ONE segment (no micro jump-cut).

    Deterministic: same file + forward source-adjacency within `gap_ms`.
    A pair is HEALED when ALL of:
      * same file_id,
      * the next segment's in_ms is >= the previous in_ms (forward), and
      * next.in_ms <= prev.out_ms + gap_ms (contiguous / tiny gap), and
      * neither segment carries a `hard_seam` marker (see §5 tighten guard).
    Healing sets prev.out_ms = max(prev.out_ms, next.out_ms) -- the source
    plays straight through, bridging the dropped gap. Audio needs no handling
    here: layers.resolve derives the dialogue layer per merged segment, so it
    bridges the same gap automatically and stays aligned.

    Returns (healed_segments, merged_map) where merged_map maps every
    ABSORBED seg_id -> the surviving predecessor's seg_id, so ops keyed on a
    seam id (split_edit/crossfade) can be remapped and healed-away seams
    dropped.

    reindex=True re-issues ids as a000, a001, ... (the brain refers to cuts by
    these stable, human-readable ids). reindex=False keeps the predecessor's
    own id (the manual/snap path, so client selection/undo references survive).
    Idempotent: a second pass over an already-healed timeline is a no-op.
    """
```

- **Inputs:** the ordered spine segment list; a `gap_ms` threshold; an id policy.
- **Outputs:** the healed spine + a `merged_map` for op-seam remapping.
- **Purity:** no DB, no LLM, no I/O. Trivially unit-testable (mirrors `_weld_segments`).

### 3.2 How it coalesces video AND audio

- **Video:** two adjacent same-file segments become one segment `[prev.in_ms, max(prev.out_ms, next.out_ms)]`. `layers.resolve` emits a single `VideoLayer` for it — the picture plays straight through the previously-dropped frames. No cut, no jump.
- **Audio:** *no separate handling.* Because the dialogue `AudioLayer` is derived per spine segment inside `layers.resolve` (`layers.py:651-664`) from the same `in_ms/out_ms` (+ the segment's baked `audio_file_id`/`audio_offset_ms`), merging the video segment automatically produces a single continuous audio window that bridges the same gap. The merged segment keeps the **predecessor's** audio coupling; the heal must refuse to merge if the two sides have *different* `audio_file_id`/`audio_offset_ms` (a hand-`replace_audio`'d or split-routed seam) — see §5.
- **Provenance carry-over (from the existing weld, keep it):** merged segment keeps the first slice's `ref`/`level`; is marked `axis="speech"` if either side was; stays unmuted if either side wanted audio; concatenates `content` (`arrange.py:212-220`).

### 3.3 Where it runs — the two call sites (the shared invocation)

**Path A (brain / auto-assembly)** — replace the weld call inside `observe.resolve_doc`:
```python
# backend/app/services/l3/observe.py  (was: document["timeline"] = _weld_segments(...))
healed, merged_map = heal_adjacent_cuts(
    document.get("timeline") or [], gap_ms=get_settings().heal_gap_ms, reindex=True)
document["timeline"] = healed
_remap_seam_ops(document, old_ids, old_segs, merged_map)   # generalized remap, see §4.2
```

**Path B (manual / SNAP)** — add a heal step in `put_document`, before resolve:
```python
# backend/app/routers/edit_threads.py, inside put_document, after _sanitize_*
timeline = _sanitize_timeline(body.timeline, durations)
operations = _sanitize_operations(body.operations)
healed, merged_map = heal_adjacent_cuts(timeline, gap_ms=get_settings().heal_gap_ms, reindex=False)
operations = _remap_seam_ops_list(operations, merged_map)   # drop/rewrite seam ops
new_doc = {**doc, "timeline": healed, "operations": operations}
```

Both call sites hit the **same** `heal_adjacent_cuts`, differing only by `reindex`. A snap that lands adjacent to a same-source contiguous segment is healed identically to the brain's own placement.

### 3.4 Threshold — recommended default + config key

- Add to `backend/app/config.py` `Settings`:
  ```python
  # Two adjacent SAME-FILE spine segments whose source spans are within this
  # many ms are HEALED into one continuous clip at compose time (heal_adjacent_
  # cuts). Supersedes the old hard-coded _WELD_TOL_MS=120: 130ms real-world
  # micro-jumps (demo run 6f31d648) sat just above 120 and glitched. Kept below
  # the size of a DELIBERATE dead-air trim so healing never silently undoes a
  # tighten (see the plan's "tighten vs heal" decision).
  heal_gap_ms: int = 200
  ```
- `arrange.py` keeps a module fallback constant `HEAL_GAP_MS_DEFAULT = 200` for call sites/tests that don't thread settings, and `_WELD_TOL_MS` is removed (or aliased to the default) once all callers move to `heal_adjacent_cuts`.

**Recommended default: `200ms`.** Rationale:
  - Comfortably heals the observed 130ms glitch (and typical sub-frame/sub-quarter-second continuity gaps).
  - Stays **below** the magnitude of a *deliberate* pause/dead-air trim (speech tightening removes ≥ ~250–300ms of silence, per the retime path in `act.py:786`), so a modest threshold plus the `hard_seam` opt-out (§5) means healing does not undo an intentional tighten.
  - 150ms is the floor (barely clears the 130ms case, fragile); 250ms is the ceiling before it starts eating small intentional trims. 200ms is the middle.

**Alternative considered — derive from the silence/seam signal (NOT recommended as the default).** L1 persists `audio_features.silence_intervals` (surfaced in `EditContext.audio_features`, `observe.py:71`; read in `observe._edit_signals` around `observe.py:270`), and `seam.classify_seam` (`seam.py:67`) already exists to judge weldability from atom-level shot/speaker/junk signals. A signal-derived heal ("heal iff the dropped gap falls inside a detected silence and the seam is classified weldable") is more principled and directly reuses the documented hook in `seam.py`. **Why not default to it:** it needs the run's atoms/silence loaded at compose time (a DB read `layers.resolve`/`put_document` deliberately avoid — resolution is "pure, cheap, free of snapping/model logic", `layers.py:20`), and it makes an every-save composition path depend on ingest-run availability. Recommend shipping the **fixed-ms threshold first** (deterministic, zero new DB reads, works on the manual path even with no run pinned), and treating the seam-signal refinement as a fast-follow that can tighten the guard where signals are cheaply available (Path A already has `ctx`). **This is a decision point — see §5.**

---

## 4. Exact change list per file

### 4.1 `backend/app/services/l3/arrange.py`
- **Add** `heal_adjacent_cuts(segments, *, gap_ms, reindex=True) -> (segments, merged_map)` per §3.1. Implement by generalizing `_weld_segments` (`arrange.py:194`):
  - same merge test but with `gap_ms` instead of the constant `_WELD_TOL_MS`;
  - **refuse to merge** when the two segments carry different `audio_file_id`/`audio_offset_ms`, or when either carries a `hard_seam` truthy marker (§5);
  - build `merged_map` (absorbed `seg_id` → surviving `seg_id`) as it merges;
  - id pass: `reindex=True` → `a{i:03d}` (today's behavior); `reindex=False` → keep predecessor ids, only absorbed ids disappear.
- **Keep** `_weld_segments` as a thin wrapper `return heal_adjacent_cuts(segments, gap_ms=HEAL_GAP_MS_DEFAULT, reindex=True)[0]` during migration, or delete once callers move (prefer delete to avoid two entry points). Add `HEAL_GAP_MS_DEFAULT = 200`.
- Update the module docstring (`arrange.py:16-18`) to describe heal, not just weld.

### 4.2 `backend/app/services/l3/observe.py`
- In `resolve_doc` (`observe.py:287`): replace the `_weld_segments(...)` call with `heal_adjacent_cuts(..., gap_ms=ctx-or-settings, reindex=True)`; thread the settings value (or `ctx`) in. Prefer reading `get_settings().heal_gap_ms` once here.
- Generalize `_remap_split_edits` (`observe.py:321`) → `_remap_seam_ops` that, given `merged_map`, remaps **both `split_edit` AND `crossfade`** `seam_seg_id`s (both are keyed on a seam segment — `act.crossfade` at `act.py:388`, `layers._apply_crossfades` at `layers.py:456`), dropping an op whose seam healed away and rewriting one whose seam survived under a new id. (Today only `split_edit` is remapped — an existing latent gap for crossfades that becomes load-bearing once heal merges more seams.)
- Update `import` at `observe.py:39` (`_weld_segments` → `heal_adjacent_cuts`).

### 4.3 `backend/app/config.py`
- Add `heal_gap_ms: int = 200` to `Settings` (with the docstring in §3.4).

### 4.4 `backend/app/routers/edit_threads.py`
- Import `heal_adjacent_cuts` (+ a small `_remap_seam_ops_list` op-remap helper — factor the op-list rewrite out of observe's `_remap_seam_ops` so both routers/observe share it, e.g. put it in `arrange.py` next to the heal).
- In `put_document` (`edit_threads.py:183`): after `_sanitize_timeline`/`_sanitize_operations`, call `heal_adjacent_cuts(timeline, gap_ms=get_settings().heal_gap_ms, reindex=False)`, then remap `operations` via the returned `merged_map`, then build `new_doc` from the **healed** timeline. This makes the persisted human/snap spine healed and authoritative.
- (Optional consistency) also heal inside `render/tasks.resolve_document`'s recompute branch (`tasks.py:106-118`) so any document that reaches render/export without having gone through the healed `put_document` (e.g. an old stored version) still composes healed. Low-risk since it's idempotent; decide based on whether old versions must self-heal on render.

### 4.5 Frontend (snap preview parity — optional but recommended for UX)
- `frontend/src/lib/resolve-timeline.ts` (+ wherever the snap preview resolves): mirror the heal as a pure client function `healAdjacentCuts(timeline, gapMs)` so the snap *preview* shows the merged clip immediately, before the server save round-trips. Must use the SAME `gapMs` the backend uses (surface it via a small config/constant or the thread response) to avoid preview/render disagreement. If skipped, document that snap heals only after save+refetch (acceptable but slightly jarring).
- No change needed to `addSegment` itself — it stays a raw append; heal is a compose-time concern on both sides.

### 4.6 Tests — see §6.

---

## 5. Edge cases + explicit decision points

**DECISION POINT 1 — threshold value & source.** Recommend fixed `heal_gap_ms = 200` (§3.4). Sign off on the number and on **fixed-ms vs silence/seam-derived** (recommend fixed-ms now, seam-signal as fast-follow). *Needs user confirmation.*

**DECISION POINT 2 — "tighten vs heal".** When the brain deliberately *tightened out a pause* (`retime`/`tighten`, `act.py:755`/`act.py:786`) it creates same-file adjacent segments with a small removed gap. A naive heal within `heal_gap_ms` would **silently undo that tighten**. Recommended resolution (deterministic, in-scope):
  - Have `retime`/`tighten` stamp the seams they intentionally create with a `hard_seam=True` (or `heal="hard"`) marker on the following slice; `heal_adjacent_cuts` refuses to merge across a `hard_seam`.
  - Keep the default threshold (200ms) below typical deliberate trims as a second line of defense.
  This keeps branch 1 (accidental micro-gaps) healed while branch 2 (a real, intended discontinuity) is untouched. *Needs user confirmation on whether tightened pauses should ever heal — recommended answer: NO, guard with `hard_seam`.*

**Healing across a slide/coverage overlay boundary.** Heal operates only on the **spine** (`document["timeline"]`). V2 coverage / split-screen live in `operations`/`layout_regions` and are program-time-anchored (`act.split_screen` at `act.py:676`; `layers._apply_layout_regions` at `layers.py:530`). Merging two spine segments changes spine seg_ids but not program duration (source plays straight through, same total length), so program-anchored overlays keep their windows. **Guard:** confirm heal never merges a segment that a layout region's `cells` selects by `layer:"spine"` in a way that would move a slice boundary the region depends on — regions select the spine generically by program window, not by seg_id, so this is safe, but add a test.

**Erasing an intended cut.** A same-file hard cut the user/brain wants (e.g. a jump for effect) must not heal. Covered by: (a) `keep_spans` jump-cuts are non-contiguous and exceed `gap_ms` (`arrange.py` weld already relies on this — `test_arrange.test_intra_clip_jump_stays_separate`), and (b) the `hard_seam` marker (DECISION 2) for deliberate small trims.

**Document boundaries.** First segment has no predecessor; heal only ever merges a segment into the one before it. No wraparound. No-op on an empty/single-segment timeline.

**Idempotency.** After a heal pass, no adjacent same-file pair is within `gap_ms` (they've merged), so a second pass changes nothing. Must be covered by a test (re-running `heal_adjacent_cuts` on its own output returns an equal timeline and an empty `merged_map`).

**Crossfade / split_edit already on the healed seam.** If a `crossfade` or `split_edit` op sits on a seam that heals away, its seam ceases to exist — the op must be **dropped** (via `merged_map`); if the seam survives under a new id (reindex path), the op must be **rewritten**. This is the generalized `_remap_seam_ops` (§4.2). Today crossfades are *not* remapped after weld — fix that here.

**Different audio coupling on the two sides.** If the two candidate segments have different `audio_file_id`/`audio_offset_ms` (e.g. one was `replace_audio`'d, `act.py:419`), merging would silently drop one side's routing. **Refuse to heal** such a pair (fall back to leaving the cut) and add a test.

**Muted vs live.** Preserve the existing weld rule: the merged run stays live if *either* side wanted audio (`arrange.py:214-217`).

---

## 6. Test plan

New/updated `backend/scripts/test_arrange.py` (mirrors existing style, `test_arrange.py:38-109`), plus one loop-level test:

Unit tests for `heal_adjacent_cuts` (default `gap_ms=200`, `reindex=True` unless noted):
1. **contiguous → merged:** `[A:0–2000, A:2000–4000]` → 1 segment `[0,4000]`. (existing `test_contiguous_same_clip_welds`, retarget to new fn.)
2. **tiny-gap → merged (the demo case):** `[A:0–60805, A:60935–70000]` (130ms gap, < 200) → 1 segment `[0,70000]`; assert the bridged gap is played (out_ms==70000). This is the direct regression guard for `a002/a003`.
3. **big-gap → untouched:** `[A:0–2000, A:2500–4000]` (500ms gap > 200) → 2 segments.
4. **just-over-threshold → untouched:** gap `= gap_ms+1` stays split (boundary test).
5. **different-source → untouched:** `[A:0–2000, B:0–2000]` → 2 segments. (existing `test_different_clips_never_weld`.)
6. **audio stays aligned:** resolve a merged timeline via `layers.resolve` and assert exactly one dialogue `AudioLayer` spanning `[merged.in, merged.out]` with a source window that bridges the gap (no second layer, no gap in coverage).
7. **idempotent:** `heal_adjacent_cuts(heal_adjacent_cuts(x)[0])` equals the first result and returns an empty `merged_map`.
8. **keep_spans jump-cut survives:** (existing `test_keep_spans_jumpcut_survives`) — non-contiguous slices from one cut are NOT healed.
9. **hard_seam guard:** a pair within `gap_ms` but marked `hard_seam=True` on the second segment is NOT merged (tighten protection).
10. **different audio coupling → not merged:** two contiguous same-file segments with different `audio_file_id` stay split.
11. **id policy:** `reindex=True` yields `a000,a001,…`; `reindex=False` keeps predecessor ids and only drops absorbed ids; `merged_map` maps absorbed→survivor in both.
12. **seam-op remap:** a `split_edit` and a `crossfade` on a healed-away seam are dropped; on a surviving seam are rewritten to the new id (extend `test_observe_act.test_weld_remaps_split_edit_seams`, `backend/scripts/test_observe_act.py:442`).

Path-level tests:
13. **auto-assembly path** (`test_observe_act.py`): `observe.resolve_doc` on a doc with a 130ms same-file gap yields a single merged spine segment (was 2).
14. **snap path** (a `put_document`/router test, cf. `backend/scripts/test_exports_router.py` harness style): saving a timeline whose last two segments are same-file within `gap_ms` persists a healed timeline; `reindex=False` so surviving client ids are preserved.

Update existing tests that assumed `_WELD_TOL_MS=120` / the old `_weld_segments` name (`test_arrange.py`, `test_observe_act.py:66`) to the new fn + threshold.

Run: `PYTHONPATH=. .venv/bin/python scripts/test_arrange.py` (and the observe/router suites).

---

## 7. Rollout / verification (confirm on demo run `6f31d648`)

1. **Land + unit-verify:** implement §4, run the §6 suites. Test #2 is the direct proof the `a002/a003` 130ms gap now merges.
2. **Set the knob:** `heal_gap_ms=200` (default). No migration — it's a compose-time setting, applied on the next resolve/save.
3. **Reproduce read-only on the running local API** (do NOT re-ingest, do NOT restart the read-only `localhost:8000`):
   - Fetch the current document for thread `76d88f5d` (`GET /api/edit/threads/76d88f5d`) and locate the two spine segments around `src_out=60805ms` / `src_in=60935ms` (same `file_id`).
   - **Before:** confirm they are two separate segments with a 130ms source gap (the glitch), matching the prior diagnosis.
4. **Confirm heal on Path A:** trigger a no-op-ish agent turn (or re-resolve) so `observe.resolve_doc` runs the heal, then refetch: the two segments should now be a single continuous spine segment `[…, 60805]∪[60935, …]` played straight through (out_ms extended across the 130ms). Program length unchanged (source bridged, not trimmed).
5. **Confirm heal on Path B (snap):** with snap mode on, click a cut that lands adjacent to a same-source contiguous segment; save (`PUT …/document`); refetch and confirm the persisted spine merged them (and, if the client mirror in §4.5 is done, that the preview merged them pre-save).
6. **Confirm no over-heal:** verify a deliberately tightened pause (a `hard_seam` seam) and a genuine same-file jump (gap > 200ms) both remain separate cuts.
7. **Render parity:** render the healed thread; confirm the seam at ~60.8s no longer shows a jump-cut (continuous take), and that any crossfade/split_edit that was on a healed-away seam was correctly dropped (no ffmpeg graph error, `render/compositor.py`).

---

### Appendix — key file:line references
- Weld today: `backend/app/services/l3/arrange.py:40` (`_WELD_TOL_MS`), `:194` (`_weld_segments`).
- Compose/persist (brain): `backend/app/services/l3/observe.py:287` (`resolve_doc`), `:321` (`_remap_split_edits`); called at `backend/app/services/l3/converse.py:625`.
- Compose (manual/snap, no weld today): `backend/app/routers/edit_threads.py:183` (`put_document`), `:72` (`_sanitize_timeline`); `backend/app/services/render/tasks.py:80` (`resolve_document`), `:115` (`layers.resolve`).
- Spine → layers (audio derived per segment): `backend/app/services/l3/layers.py:388` (`spine_spans`), `:562` (`resolve`), `:610-664` (per-segment video+audio).
- Placement → segments: `backend/app/services/l3/act.py:60` (`_segments_from_cut`), `:190` (`place`).
- Snap: `frontend/src/components/cuts-view.tsx:413`/`:1595`, `frontend/src/stores/edit-doc-store.ts:455` (`addSegment`), `frontend/src/stores/timeline-view.ts:41` (`snapAddEnabled`).
- Documented hook: `backend/app/services/l3/seam.py:8-16`, `:67` (`classify_seam`).
- Config: `backend/app/config.py` (add `heal_gap_ms`).
- Doc shape: `backend/migrations/014_l3_edit_threads.sql:37`.
