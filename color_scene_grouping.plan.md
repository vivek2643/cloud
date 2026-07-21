# Color: Smarter Scene Grouping for Shot-to-Shot Matching (v1 grade pipeline)

Self-contained implementation plan. The implementer has **no other context** —
everything needed is here. Read this whole file before starting. You are
implementing changes to the **v1 grade pipeline only**; the `legacy` path stays
**byte-for-byte identical**.

> This builds directly on the just-shipped two-stage `group → balance → match`
> redesign (`color_shot_matching.plan.md`; new `balance.py`, `reference.py`;
> `grade_shot_match_v2=True`; `INPUT_HASH_SCHEMA_VERSION=3`). That redesign made
> matching/balance *converge within a group*. **This plan fixes the input to
> that machinery: the grouping itself.** Better groups directly amplify the
> matching and balance that already work.

---

## 1. Goal & non-goals

### Goal
Reliably group **same-scene** shots on a multi-file reel so that shot-to-shot
matching and balance act **across the reel**, not per-file. Today, grouping for
matching collapses to **same-`file_id`-only** because the metadata it keys on is
absent on every timeline shot — so on a multi-file reel every shot is a
singleton and matching/balance are a no-op (see §2). Concretely:

- Recover the real scene metadata (`speaker_person` / `on_camera` / `label` /
  `summary`, plus stronger structural signals) that **already exists upstream**
  (`cut_records`) but is not carried onto the timeline shot, by joining each
  grade shot's `(file_id, in_ms..out_ms)` to the `cut_record` it came from.
- Make grouping **degrade gracefully**: it must **never** return all-singletons
  when metadata is genuinely absent — an RGB/time-proximity base always groups
  adjacent same-look shots.
- Feed the resulting groups to **both** `solve_sequence_match` and the
  Balance/Match `GroupReference` (both already read `groups` in `run_grade_job`),
  so the fixes to matching/balance already shipped now act on many more shots.

### Non-goals
- **Do not touch the `legacy` path.** No behavior, no bytes. The legacy match
  path (`cluster_grade_groups` / `solve_match_deltas`, `MATCH_STRENGTH`) is not
  modified.
- **Do not re-litigate the matching/balance math.** `solve_sequence_match`,
  `solve_balance`, `compute_group_reference`, the composite guardrails
  (`COMPOSITE_SLOPE_MAX`, `COMPOSITE_MID_FLOOR`), and the match/balance strength
  constants are all **shipped and correct** — this plan only changes what
  `groups` contains.
- No document-schema change (do **not** persist metadata onto the timeline seg;
  see §4 Option B, rejected). No new measurement pass.
- No new LLM/model calls: every signal already exists deterministically in
  `cut_records`.

---

## 2. Root cause recap (verified against the code + DB)

`run_grade_job` (v1 only) builds `groups` for both Balance and Match. When
`grade_semantic=True` it first computes `semantic_groups` from
`scene_group.group_shots_semantically`, then (under `grade_shot_match_v2`) keeps
it only if it found real structure, else falls back to RGB adjacency.

### The metadata it groups on is empty on every shot
`run_grade_job` builds `ShotSceneMeta` from the **raw timeline seg / `place_video`
op** dict:

```393:402:backend/app/services/l3/grade/job.py
            scene_meta = [
                SceneMeta(key=s.key, file_id=s.file_id,
                         speaker_person=s.item.get("speaker_person"),
                         on_camera=s.item.get("on_camera"),
                         label=str(s.item.get("label") or ""), summary=str(s.item.get("summary") or ""))
                for s in shots
            ]
            semantic_groups = group_shots_semantically(scene_meta)
```

But `ordered_shots` only ever reads geometry off a seg/op — `seg_id`/`file_id`/
`in_ms`/`out_ms`/`hero_ts_ms` from a timeline seg, `op_id`/`source_file_id`/
`src_in_ms`/`src_out_ms` from a `place_video` op:

```164:185:backend/app/services/l3/grade/job.py
def ordered_shots(document: Dict[str, Any]) -> List[Shot]:
    ...
    for seg in document.get("timeline") or []:
        ...
        shots.append(Shot(
            key=str(seg["seg_id"]), file_id=str(seg["file_id"]),
            in_ms=int(seg.get("in_ms", 0)), out_ms=int(seg.get("out_ms", 0)),
            hero_ts_ms=seg.get("hero_ts_ms"), item=seg,
        ))
    for op in document.get("operations") or []:
        ...
```

Timeline segs simply **do not carry** `speaker_person` / `on_camera` / `label` /
`summary`. `s.item.get(...)` returns `None`/`""` for all four on every shot.

**DB evidence (verified live, Siri reel thread `947d7e91-4862-4e7f-afe9-dcd2ea28fef1`, 8 shots, 7 distinct files):**
every timeline seg has `speaker_person=None`, `on_camera=None`, `label=None`,
`summary=None`. (Segs *do* carry a `ref` = `moment_id`, e.g. `e3f85292:m00`,
on all 8 — a partial back-reference, discussed in §3.)

### The grouping logic (correct, but starved of signal)
`group_shots_semantically` chain-links shot `i` to the running group only if it
shares `file_id`, or equal non-null `speaker_person`, or (equal non-null
`on_camera` **and** label/summary word overlap) with the immediately preceding
shot:

```59:79:backend/app/services/l3/grade/scene_group.py
def group_shots_semantically(ordered_shots: List[ShotSceneMeta]) -> List[List[int]]:
    ...
    for i, shot in enumerate(ordered_shots):
        if groups:
            prev = ordered_shots[groups[-1][-1]]
            same_file = bool(shot.file_id) and shot.file_id == prev.file_id
            same_speaker = shot.speaker_person is not None and shot.speaker_person == prev.speaker_person
            weak_tiebreak = (
                shot.on_camera is not None and shot.on_camera == prev.on_camera
                and _label_overlap(shot, prev)
            )
            if same_file or same_speaker or weak_tiebreak:
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups
```

With all three signals empty this collapses to `same_file`. On the Siri reel (8
different files) that is **8 singletons**.

### What the shipped redesign already does, and the gap that remains
Under `grade_shot_match_v2` the job already refuses to let all-singletons kill
matching, degrading to RGB adjacency:

```419:421:backend/app/services/l3/grade/job.py
            if not _has_real_groups(semantic_groups):
                semantic_groups = None
            groups = semantic_groups if semantic_groups is not None else group_neighbors(ordered)
```

So today grouping is **RGB-adjacency only** on real multi-file reels (semantic is
always all-singletons → discarded). That is the whole gap this plan closes:

1. **Semantic grouping never actually fires** on real data (metadata empty), so
   we lose the meaning-based grouping the module was built for — same-speaker /
   same-take / same-outlook shots whose RGB happens to have drifted are **not**
   grouped, and unrelated shots that happen to share a palette **can** be.
2. **RGB adjacency can under-group.** On a montage reel where adjacent shots are
   the same production/lighting but genuinely different palettes, RGB adjacency
   leaves them as singletons → still no cross-shot convergence.

Fixing grouping is the highest-leverage remaining lever: the Balance/Match math
downstream is already correct and already reads `groups`.

**Line-ref drift note.** The task brief's line numbers predate the
`color_shot_matching` redesign. Verified-current refs: `job.py` `ShotSceneMeta`
build is **L393–402** (brief said ~L306–315); `ordered_shots` is **L164–185**
(brief said ~L88–109); `INPUT_HASH_SCHEMA_VERSION` is **`3`** at **L60** (brief
said "currently 3"). `scene_group.py` grouping fn **L59–79**, loop **L66–78** —
no drift.

---

## 3. Data-flow findings (verified live against the DB)

### Where the metadata lives: `cut_records`
`cut_records` (migration `024_cuts_v3.sql`, extended by later migrations) is the
per-cut source of truth written once at ingest. Verified columns relevant here:

| Column | Type | Meaning |
|---|---|---|
| `file_id` | uuid | source file |
| `src_in_ms`, `src_out_ms` | int | the cut's span in **source** time |
| `label` | text | short scene/action label |
| `summary` | text | longer gist |
| `speaker_person` | text | bound speaker person id |
| `on_camera` | boolean | speaker visible in this cut |
| `voice_ids` | jsonb | voice cluster id(s) heard |
| `take_group_id` | uuid | cross-clip retake/outlook group |
| `sync_group_id` | uuid | multicam outlook group (shared authoritative audio) |
| `continuity` | jsonb | `{cut_no, of, prev_contiguous, next_contiguous, ...}` |
| `hero_ts_ms` | int | best-still anchor |

**Population (verified, whole DB, 8701 rows):** `label` **8701/8701 (100%)**,
`summary` **100%**, `on_camera` **4832 (~55%)**, `speaker_person` **1121 (~13%)**,
`take_group_id` **2750 (~32%)**, `sync_group_id` **2189 (~25%)**, `voice_ids`
100% present (often an empty array), `continuity` 100% (with
`prev_contiguous=true` on 1444). **`label`/`summary` are the only universally
populated semantic fields**; the structural ids (`take_group_id`,
`sync_group_id`) are strong when present.

### The exact join key: `(file_id, span-overlap)` — verified reliable
A grade shot's span is a **tightened sub-span of exactly one `cut_record`**: the
seg's `in_ms/out_ms` come from a footage-map ladder rung that is derived from a
`cut_record`'s `src_in_ms/src_out_ms` (see `cutrecord_map.synth_ladder` →
`footage_map.build_clip_tree` → `arrange`/`act._segments_from_cut`). So the join
is:

> For a grade shot `(file_id, in_ms, out_ms)`, among `cut_records` with the same
> `file_id`, pick the row with **maximum time-overlap** of `[in_ms, out_ms]` with
> `[src_in_ms, src_out_ms]`. That row is the cut the shot came from.

**Verified on the Siri reel** (all 8 shots): every shot matched **exactly one**
`cut_record` with large overlap, and the spans line up cleanly. Example:
`a000 e3f85292 [0-3400] → cut [0-3400] overlap=3400 label='Siri Nature Valley Resort is a premium resort'`;
`a007 3ddf7db9 [4760-7380] → cut [4382-7380] overlap=2620 label='call to action'`.
So Option A is **feasible and precise**.

Resolve the covering ingest run with
`cuts_v3_read.latest_run_for_files(file_ids)` (the same resolver
`footage_map`/`cutrecord_map` use), then read rows with
`cuts_v3_read.rows_for_run(run_id, file_ids)` — that fetch already selects
`speaker_person, on_camera, label, summary, take_group_id, sync_group_id,
voice_ids, continuity, src_in_ms, src_out_ms, file_id` (see
`cuts_v3_read.rows_for_run`, L57–78).

### Do segs carry a cut back-reference? Partially — not enough to rely on
Segs minted from a map ref carry `ref` = `moment_id` (e.g. `e3f85292:m00`, set in
`act._segments_from_cut` L86, `"ref": rc.ref`). **But**: (a) `place_span` segs and
some welded segs have `ref=None` (`act.place_span` L158; `_weld_segments` re-issues
`seg_id` and only keeps the first slice's ref); (b) a `moment_id` does not map to a
`cut_record.id` directly — the tree would have to be rebuilt to resolve it
(`build_clip_tree` puts `hero_id = cut_record.id` only on the *variant*, not the
moment top-level). So `ref` is **not** a dependable join key. The `(file_id,
span-overlap)` join is direct, dependable, and needs no tree rebuild — **use it.**

### The decisive finding for the design
On the Siri reel the shots are a genuine **montage**: 8 different files, 8
different voices (`V0…V10`), **no shared `sync_group_id`**, and the one cross-file
`take_group_id` (`8381fe96`) links cuts that are **not** co-placed in the timeline.
`speaker_person`/`on_camera` are `None` even in these `cut_records`; only
`label`/`summary` are populated, and the labels are *different per shot*
(`'water park mention'` vs `'Employee discount offer'`). **Therefore:**

- **Option A (the cut_records join) reliably recovers `label`/`summary`** for
  every shot, plus `on_camera`/`speaker_person`/`take_group_id`/`sync_group_id`/
  `voice_ids` **where they exist** — this genuinely helps reels that have those
  signals (interviews, multicam, retakes).
- **But a montage reel like Siri has no strong same-scene metadata link**, so an
  **RGB/time-proximity base is the universal safety net** that actually groups a
  continuous reel. Grouping must fold this base in so it **never** returns
  all-singletons. (This is Option C, promoted from "fallback" to "always-on
  base" of the semantic grouper.)

---

## 4. Design

### Recommended approach
**Option A (primary) + Option C (always-on base), with Option B rejected.**

- **Option A — grade-time metadata lookup + join (primary).** Inside
  `run_grade_job`, join each shot's `(file_id, span)` to its covering
  `cut_record` and populate an **enriched** `ShotSceneMeta` from real values.
  No document-schema change; the metadata is already computed upstream, just not
  carried. Chosen because the join is exact and the data is 100%/55%/32%/25%
  populated depending on the field.
- **Option C — RGB/time-proximity base folded into the grouper (always on).**
  `group_shots_semantically` gains an optional per-shot `rgb_mean` (and an
  optional program gap) and OR-in the same adjacency test `group_neighbors`
  already uses, so semantic grouping **itself** degrades to RGB adjacency
  shot-by-shot and can never return all-singletons when RGB is available (it
  always is — from `measure_span`).
- **Option B — persist metadata onto the seg at creation (rejected).** Would
  require touching the arrange/compile write path and a document-schema/version
  change, re-ingest to backfill, and duplicates data that already lives in
  `cut_records`. More surface, more risk, no upside over the exact grade-time
  join.

### How it degrades (never regresses to all-singletons)
Trust hierarchy, evaluated against the **immediately preceding** shot only
(preserving the existing chain-link discipline — distant shots never group):

1. `same_file` — same `file_id` (a continuous take is one scene by construction).
2. `same_sync_group` — equal non-null `sync_group_id` (multicam outlook: the same
   moment from another camera — definitively same scene).
3. `same_take_group` — equal non-null `take_group_id` (retakes of the same
   content — same setup/lighting).
4. `same_speaker` — equal non-null `speaker_person`, **or** non-empty
   `voice_ids` intersection (same person talking → same setup).
5. `weak_tiebreak` — equal non-null `on_camera` **and** label/summary word
   overlap (unchanged; still only ever a corroborated tiebreak, never alone).
6. **`rgb_close` (base)** — `rgb_mean` within `SCENE_RGB_DIST_MAX` of the
   previous shot (the graceful universal base — same test `group_neighbors`
   uses). Guarantees a continuous same-look reel groups even with zero metadata.

Any one of 1–6 chains the shot into the running group; else it starts a new
group. This means: metadata **promotes** grouping where it exists (fixing
under-grouping of RGB-drifted same-scene shots), while the RGB base guarantees a
sensible floor (fixing all-singletons).

### How the groups feed downstream (already wired — do not rebuild)
`run_grade_job` already computes `groups` once and passes it to **both**
`compute_group_reference` (Balance's + Match's reference) and
`solve_sequence_match` / `solve_balance` (see `job.py` L404–462). This plan only
changes the *contents* of `semantic_groups`/`groups`. The
`_has_real_groups(semantic_groups) → group_neighbors` fallback (L419–421) stays as
a redundant belt-and-suspenders net (after this change semantic grouping will
essentially always find structure, so the fallback rarely triggers — that's
fine; keep it).

---

## 5. Phased implementation

Ordered by leverage: the metadata lookup+join first (it is what unlocks
everything), then the grouper base, then wiring, then flag/hash/tests/ops. Each
step is small enough to need no further design. All new behavior is gated to v1
under `grade_semantic` **and** a new `grade_scene_join` flag (§6); legacy is
never reached.

> Line numbers are current as of plan-writing (verified live). If they drift,
> locate by function / constant name.

### Phase 1 — Metadata lookup + `(file_id, span)` → `cut_record` join (HIGHEST LEVERAGE)

**New module — `backend/app/services/l3/grade/scene_meta.py`** (I/O isolated
here, so `scene_group.py` stays pure). It resolves each shot's covering
`cut_record` and returns enriched metadata keyed by shot key.

```python
"""color_scene_grouping.plan.md Phase 1: join each grade shot's
(file_id, in_ms..out_ms) to the cut_record it was cut from, so the real
scene metadata (already computed at ingest, but never carried onto the
timeline seg) is available for grouping. Pure lookup + max-overlap join;
best-effort and fail-open (no covering run / no overlapping cut -> that
shot simply gets empty metadata and falls back to the RGB base)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ShotCutMeta:
    """Real scene metadata for one shot, joined from its covering cut_record."""
    speaker_person: Optional[str] = None
    on_camera: Optional[bool] = None
    label: str = ""
    summary: str = ""
    voice_ids: List[str] = field(default_factory=list)
    take_group_id: Optional[str] = None
    sync_group_id: Optional[str] = None


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def lookup_shot_cut_meta(
    shots: List[Tuple[str, str, int, int]],   # (key, file_id, in_ms, out_ms)
) -> Dict[str, ShotCutMeta]:
    """key -> ShotCutMeta for every shot with a covering cut_record. Best-
    effort: any DB error, no covering run, or no overlapping cut yields an
    empty dict entry-free result for that shot (caller treats missing as
    'no metadata')."""
    out: Dict[str, ShotCutMeta] = {}
    file_ids = sorted({fid for _, fid, _, _ in shots})
    if not file_ids:
        return out
    try:
        from app.services.l3 import cuts_v3_read
        run_id = cuts_v3_read.latest_run_for_files(file_ids)
        if run_id is None:
            return out
        rows = cuts_v3_read.rows_for_run(run_id, file_ids)
    except Exception:
        return out   # fail-open: grouping falls back to the RGB base

    by_file: Dict[str, List[dict]] = {}
    for r in rows:
        by_file.setdefault(r["file_id"], []).append(r)

    for key, fid, in_ms, out_ms in shots:
        best, best_ov = None, 0
        for r in by_file.get(fid, []):
            ov = _overlap(in_ms, out_ms, int(r["src_in_ms"]), int(r["src_out_ms"]))
            if ov > best_ov:
                best, best_ov = r, ov
        if best is None or best_ov <= 0:
            continue
        out[key] = ShotCutMeta(
            speaker_person=best.get("speaker_person"),
            on_camera=best.get("on_camera"),
            label=str(best.get("label") or ""),
            summary=str(best.get("summary") or ""),
            voice_ids=list(best.get("voice_ids") or []),
            take_group_id=best.get("take_group_id"),
            sync_group_id=best.get("sync_group_id"),
        )
    return out
```

- **Why a new module, not inline in `job.py`:** keeps the one new DB read
  isolated and unit-testable (mockable exactly like the other `job.py` I/O in
  `test_grade.py`), and keeps `scene_group.py` pure/numeric.
- **`file_ids` scope:** `job.py` already has `file_ids = list({s.file_id for s in shots})` (L370) — pass the `(key, file_id, in_ms, out_ms)` tuples from `shots`.

### Phase 2 — Enrich `ShotSceneMeta` and give the grouper an RGB base

**`backend/app/services/l3/grade/scene_group.py`.**

**2a. Extend `ShotSceneMeta`** (L35–45) with the structural fields + an optional
`rgb_mean`:

```python
@dataclass
class ShotSceneMeta:
    key: str
    file_id: str
    speaker_person: Optional[str] = None
    on_camera: Optional[bool] = None
    label: str = ""
    summary: str = ""
    # color_scene_grouping.plan.md: stronger structural scene signals joined
    # from cut_records, and the span RGB used as the graceful grouping base.
    voice_ids: List[str] = field(default_factory=list)
    take_group_id: Optional[str] = None
    sync_group_id: Optional[str] = None
    rgb_mean: Optional[List[float]] = None
```

(Add `from dataclasses import field` and `List` to imports.)

**2b. Extend `group_shots_semantically`** (L59–79) with the trust hierarchy of
§4 plus the RGB base. Add a small `SCENE_RGB_DIST_MAX` constant (mirror
`match.SPAN_RGB_DIST_MAX`) and an `_rgb_dist` helper (or import
`match._rgb_dist`):

```python
# Same distance family as match.SPAN_RGB_DIST_MAX (per-SPAN rgb_mean); the
# graceful base so semantic grouping NEVER returns all-singletons when the
# metadata is genuinely absent (the common real-data case, verified: label/
# summary present but no speaker/on_camera/take/sync link).
SCENE_RGB_DIST_MAX = 0.12


def _rgb_dist(a, b) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def _voice_overlap(a: ShotSceneMeta, b: ShotSceneMeta) -> bool:
    return bool(set(a.voice_ids) & set(b.voice_ids))


def group_shots_semantically(ordered_shots: List[ShotSceneMeta]) -> List[List[int]]:
    groups: List[List[int]] = []
    for i, shot in enumerate(ordered_shots):
        if groups:
            prev = ordered_shots[groups[-1][-1]]
            same_file = bool(shot.file_id) and shot.file_id == prev.file_id
            same_sync = shot.sync_group_id is not None and shot.sync_group_id == prev.sync_group_id
            same_take = shot.take_group_id is not None and shot.take_group_id == prev.take_group_id
            same_speaker = (shot.speaker_person is not None and shot.speaker_person == prev.speaker_person) \
                or _voice_overlap(shot, prev)
            weak_tiebreak = (
                shot.on_camera is not None and shot.on_camera == prev.on_camera
                and _label_overlap(shot, prev)
            )
            rgb_close = (
                shot.rgb_mean is not None and prev.rgb_mean is not None
                and _rgb_dist(shot.rgb_mean, prev.rgb_mean) < SCENE_RGB_DIST_MAX
            )
            if same_file or same_sync or same_take or same_speaker or weak_tiebreak or rgb_close:
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups
```

- Keep the chain-link-to-previous-shot discipline exactly (never group distant
  shots). Update the module/function docstrings to note the new signals and that
  `rgb_mean` is the graceful base (mirroring `match.group_neighbors`).
- **Legacy safety:** `group_shots_semantically` is only ever called from
  `run_grade_job` (v1). No legacy caller exists (grep-confirm). New dataclass
  fields default to empty, so the existing `test_scene_group_*` fixtures (which
  pass no RGB, no structural ids) behave identically — the added `or` clauses are
  all `False` when their inputs are unset.

### Phase 3 — Wire the join into `run_grade_job`

**`backend/app/services/l3/grade/job.py`, `run_grade_job`, the `scene_meta`
build block (L393–402).** Replace the raw-`s.item` reads with the joined
metadata + span RGB:

```python
        semantic_groups = None
        if settings.grade_semantic:
            cut_meta = {}
            if settings.grade_scene_join:
                from app.services.l3.grade.scene_meta import lookup_shot_cut_meta
                cut_meta = lookup_shot_cut_meta(
                    [(s.key, s.file_id, s.in_ms, s.out_ms) for s in shots]
                )
            scene_meta = []
            for s in shots:
                cm = cut_meta.get(s.key)
                span_rgb = (shot_stats[s.key].stats or {}).get("rgb_mean")
                scene_meta.append(SceneMeta(
                    key=s.key, file_id=s.file_id,
                    speaker_person=(cm.speaker_person if cm else None),
                    on_camera=(cm.on_camera if cm else None),
                    label=(cm.label if cm else ""),
                    summary=(cm.summary if cm else ""),
                    voice_ids=(cm.voice_ids if cm else []),
                    take_group_id=(cm.take_group_id if cm else None),
                    sync_group_id=(cm.sync_group_id if cm else None),
                    rgb_mean=list(span_rgb) if span_rgb else None,
                ))
            semantic_groups = group_shots_semantically(scene_meta)
```

- `shot_stats` is already built above this block (L379–386), so each shot's span
  `rgb_mean` is available.
- Everything downstream (the `_has_real_groups` fallback L419–421, Balance,
  Match, references, Leveling) is unchanged — it consumes `semantic_groups` /
  `groups` exactly as today.
- **Gating:** the whole block is already inside `if settings.grade_semantic`; the
  DB join is additionally gated on `settings.grade_scene_join` (§6). Under
  `grade_shot_match_v2=False` (kill switch), the pre-redesign branch (L463–474)
  passes `semantic_groups` straight to `solve_sequence_match` with no RGB
  fallback — with the enriched grouper it will now also group by metadata + RGB
  base, which is a strict improvement and still crash-free; if you want the kill
  switch to reproduce the *old* semantic-or-nothing behavior byte-for-byte, also
  gate the enrichment (RGB base + structural signals) on `grade_shot_match_v2`
  (see §6, note).

### Phase 4 — Include the join in `compute_input_hash`

**`backend/app/services/l3/grade/job.py`, `compute_input_hash`, flags payload
(L214–219).** The join result can change grouping, so add the new flag so
toggling it invalidates cached grades:

```python
        "flags": {
            "grade_pipeline": settings.grade_pipeline,
            "grade_even_lighting": settings.grade_even_lighting,
            "grade_semantic": settings.grade_semantic,
            "grade_shot_match_v2": settings.grade_shot_match_v2,
            "grade_scene_join": settings.grade_scene_join,
        },
```

The per-shot `speaker_person`/`on_camera`/`label`/`summary` entries already in
the `shots` payload (L209–210) are always `None`/absent on the seg (that is the
bug) — leave them; they are harmless and removing them would itself change the
hash for no benefit. The `INPUT_HASH_SCHEMA_VERSION` bump (§8) covers the math
change.

---

## 6. New / changed constants & config

| Name | File | Old | New | Rationale |
|---|---|---|---|---|
| `SCENE_RGB_DIST_MAX` | `scene_group.py` (new) | — | `0.12` | Graceful RGB base so semantic grouping never returns all-singletons; same distance family as `match.SPAN_RGB_DIST_MAX`. |
| `grade_scene_join` | `config.py` (new) | — | `True` | Enables the cut_records lookup + enriched grouping; single clean kill switch for this plan (mirrors `grade_shot_match_v2`). |
| `INPUT_HASH_SCHEMA_VERSION` | `job.py:60` | `3` | `4` | Grouping (hence grade math) changed → invalidate cached grades. |

New `ShotSceneMeta` fields (`voice_ids`, `take_group_id`, `sync_group_id`,
`rgb_mean`) default to empty/`None` — no behavioral change for callers that omit
them.

**`config.py`** — add near the other grade flags (after `grade_shot_match_v2`,
~L161):

```python
    # color_scene_grouping.plan.md: join each grade shot's (file_id, span) to
    # its covering cut_record so real scene metadata (speaker/on_camera/label/
    # summary + take/sync/voice ids) drives grouping, with an RGB base so
    # grouping never degrades to all-singletons. Off = today's behavior
    # (empty metadata -> RGB-adjacency fallback only). v1-only.
    grade_scene_join: bool = True
```

Unchanged and to be **preserved exactly**: `match.SPAN_MATCH_STRENGTH=0.85`,
`CAST_MATCH_STRENGTH=0.6`, `MID_MATCH_CLAMP=1.5`, `SPAN_RGB_DIST_MAX=0.12`;
`balance.BALANCE_*`; `resolver.COMPOSITE_SLOPE_MAX=2.0`,
`COMPOSITE_MID_FLOOR=0.02`; `correct.*`; `leveling.*`.

---

## 7. Switchability / rollback

1. **Legacy untouched by construction.** `group_shots_semantically`,
   `scene_meta.py`, and the enriched `run_grade_job` block are all v1-only;
   `maybe_enqueue` returns early for `legacy`. `solve_correct_grade` /
   `resolve_clip_grade` / `solve_match_deltas` / `cluster_grade_groups` produce
   identical bytes.
2. **Single kill switch.** `grade_scene_join=False` restores today's behavior:
   the `scene_meta` block skips the DB join, so `ShotSceneMeta` is built with
   empty metadata **and no `rgb_mean`** — semantic grouping returns all-singletons
   → the existing `_has_real_groups` → `group_neighbors` RGB fallback engages,
   exactly as it does now.
   > To make `grade_scene_join=False` reproduce today byte-for-byte, ensure that
   > when the flag is off you also pass `rgb_mean=None` (i.e. don't populate the
   > RGB base) so the grouper's new `rgb_close` clause can't fire — the code in
   > Phase 3 already only enriches when `grade_scene_join` is on. Keep it that way.
3. **Clean revert.** All new logic is one new file (`scene_meta.py`) + additive
   dataclass fields + additive `or` clauses + one flag + one hash bump. A full
   revert = delete `scene_meta.py`, revert the `scene_group.py`/`job.py`/
   `config.py` diffs, drop the flag, and restore `INPUT_HASH_SCHEMA_VERSION=3`.
   Keep it in **one commit** for a clean `git revert`.

---

## 8. Testing

Tests are a **plain script** (not pytest):

```bash
cd backend && .venv/bin/python scripts/test_grade.py
```

Each test `print`s `ok  ...` and is called from `main()` (bottom of
`scripts/test_grade.py`). Add the new tests as functions and **append calls to
`main()`**. No DB/ffmpeg/R2 — use synthetic fixtures and `mock.patch` for
`scene_meta.lookup_shot_cut_meta` (mirror the existing `run_grade_job`
mock-patch tests, e.g. `test_run_grade_job_end_to_end_mocked`). Import
`grade.scene_group.group_shots_semantically`, `ShotSceneMeta`, and
`grade.scene_meta` helpers as needed. Add:

1. **Grouping never all-singletons when metadata absent (RGB base).**
   Build 8 `ShotSceneMeta` with **different `file_id`**, no speaker/on_camera/
   take/sync, but adjacent `rgb_mean` within `SCENE_RGB_DIST_MAX`. Assert
   `group_shots_semantically(meta)` yields **at least one group with ≥2
   members** (today's fixture without RGB would be all-singletons — assert that
   too by rebuilding with `rgb_mean=None` and checking `[[0],[1],...]`).

2. **Structural signals group across files.**
   Two shots, different `file_id`, same `sync_group_id` → `[[0,1]]`. Repeat for
   equal `take_group_id`, and for overlapping `voice_ids`. Each must group even
   with RGB far apart and no `on_camera`/`speaker_person`.

3. **The `(file_id, span)` join picks the max-overlap cut.**
   Unit-test `scene_meta.lookup_shot_cut_meta` with `cuts_v3_read.rows_for_run`
   and `latest_run_for_files` mocked to return synthetic rows: a file with two
   cut_records; a shot whose span overlaps the second more must map to the
   second's `label`. A shot with **no overlapping cut_record** must be **absent**
   from the result (no crash, no fabricated metadata).

4. **`run_grade_job` groups a synthetic multi-file reel via the join (mocked).**
   Mirror `test_run_grade_job_applies_leveling_and_semantic_grouping_when_flagged`:
   a 3-shot, 3-file doc; mock `measure_span` to return same-ish `rgb_mean`
   (so the RGB base chains them) **and** mock `lookup_shot_cut_meta` to return a
   shared `sync_group_id`. Assert every shot is graded and ≥2 shots receive a
   **non-identity** `cdl` (grouping fired → Balance/Match produced deltas). With
   the join mocked to return empty and `measure_span` giving RGB-far shots,
   assert grouping does **not** force a match (no over-grouping).

5. **Fail-open: join raises → no crash, RGB base still works.**
   Mock `lookup_shot_cut_meta` to raise (or `latest_run_for_files` to raise) and
   assert `run_grade_job` completes and grades all shots (mirrors
   `test_run_grade_job_records_error_never_crashes`'s spirit but expects
   **success**, since the join is fail-open).

6. **Legacy / kill-switch unchanged.**
   Assert `grade_scene_join=False` (mock `get_settings`) builds `ShotSceneMeta`
   with empty metadata and no `rgb_mean`, so `group_shots_semantically` returns
   all-singletons and the job still grades (RGB `group_neighbors` fallback path).
   Keep `test_run_grade_job_shot_match_v2_off_reproduces_pre_redesign_path`
   green.

Run the whole script; it must end with `all grade tests passed`.

---

## 9. Ops runbook (must do, in order)

After code + tests are green:

1. **Bump the cache-invalidation version.** In
   `backend/app/services/l3/grade/job.py:60`, change
   `INPUT_HASH_SCHEMA_VERSION = 3` → `= 4`. Grouping (hence grade math) changed,
   so every thread's `input_hash` changes and all cached `resolved_grades` are
   treated as stale and recomputed (the documented trigger — see the comment at
   `job.py:49–59`).
2. **Restart the grade worker** so it loads the new code (the Procrastinate
   worker on the `grade` queue — restart however this deployment runs it). A
   stale worker keeps applying old grouping.
3. **Re-grade all projects:**
   ```bash
   cd backend && .venv/bin/python scripts/_grade_all_projects.py
   ```
   It forces the grade flags on and runs `run_grade_job` for the latest
   gradeable thread of every project, printing per-project
   `state / rows / baked`.
4. **Verify** (§10). Spot-check the Siri reel
   (`947d7e91-4862-4e7f-afe9-dcd2ea28fef1`, 8 shots, 7 files) and another
   multi-file reel.

---

## 10. Acceptance criteria (quantified)

Measured across a reel's shots after re-grade, projecting each shot's measured
`mid_gray`/`black`/`white`/`rgb_mean` through its final resolved CDL via the v1
round-trip (same as the tests' `_roundtrip_v1`):

- **Grouping produces real scene groups.** On the Siri reel, grouping yields at
  least one **non-singleton** group (target: the reel chains into a small number
  of multi-shot groups rather than 8 singletons). Verify by logging `groups`
  from `run_grade_job` or asserting via a scripted re-run.
- **Matching/balance now act on far more shots.** The count of shots receiving a
  non-identity Balance and/or Match delta rises from **0/8 today** to **most of
  the reel** (target ≥ ~6/8 on the Siri reel).
- **Cross-shot exposure converges.** Cross-shot `mid_gray` standard deviation
  drops further toward **< 0.04** (from the ~0.09–0.11 baseline), on the Siri
  reel and one other multi-file reel.
- **Prior fixes preserved.** No shadow crush and no capped-dull regression: a
  display `0.5` mid lands in ~`0.46–0.65` and a display `0.15` shadow stays
  `> 0.02` for the dullest, most-lifted grouped shot; within-group convergence
  from `color_shot_matching` is intact.
- **No over-grouping.** Two genuinely dissimilar, non-adjacent shots are never
  grouped (grouping stays chain-linked to the previous shot only).

If grouping still under-fires, escalate in order: (i) add a program-time
adjacency relaxation (group consecutive shots whose program gap is `< N ms` even
when RGB differs — add a `PROGRAM_GAP_MS` constant and pass program positions
into the grouper); (ii) widen `SCENE_RGB_DIST_MAX` modestly (e.g. `0.15`);
(iii) fold `continuity.next_contiguous`/`prev_contiguous` into the trust
hierarchy for same-file runs.

---

## 11. Finish

When all tests pass (`all grade tests passed`) and the §9 runbook verification
meets §10, **commit and push** as a single revertable commit:

```bash
cd /Users/vivekgandhari/Documents/cloud
git add -A
git commit -m "color: smarter scene grouping via cut_records join for shot matching (v1)"
git push
```
