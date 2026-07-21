# Color: Shot-to-Shot Matching Redesign (v1 grade pipeline)

Self-contained implementation plan. The implementer has **no other context** —
everything needed is here. Read this whole file before starting.

> **You are implementing the v1 grade pipeline only.** The `legacy` path must
> stay **byte-for-byte identical**. Every change below is gated behind
> `pipeline == "v1"` / `working_space == WORKING_SPACE_V1` or a new setting.

---

## 1. Goal & non-goals

### Goal
Make footage on a multi-file reel grade **consistently shot-to-shot** in the v1
pipeline. Today, cross-shot exposure/white-balance/contrast spread is large and
often gets *worse* after grading. Concretely we want, after a re-grade of a
multi-file reel:

- cross-shot `mid_gray` standard deviation reduced from ~0.09–0.11 to **< 0.04**,
- contrast (white−black) and white-balance (R/G, B/G) spread reduced,
- **no shot left "capped-dull"** (a dull/log-flat shot stuck dark because a
  slope-only correction hit its ceiling),
- the already-shipped darkness fix preserved: a display mid-gray of 0.5 stays in
  ~0.46–0.65 and a 0.15 shadow is **not** crushed to zero.

### Non-goals
- **Do not touch the `legacy` path.** No behavior, no bytes.
- No new "specialty" creative features (reference-image transfer, presets, arc,
  soft-local/vignette, subject masking). Those layers already exist and stay as
  they are.
- No within-shot temporal stabilization or spatially-varying relight (documented
  non-goals in `leveling.py`).
- No new measurement passes beyond what `measure_span.py` already produces
  (measurement noise is *not* a primary driver; `SPAN_MAX_FRAMES = 4` is fine).

---

## 2. Root cause recap (verified against the code)

The v1 stack is: **Measure → Correct → Match → Leveling → Look → Arc →
Soft-local → bake**, composed in the v1 working space (`tone.to_working` =
inverse sRGB EOTF; `from_working` = filmic shoulder + sRGB OETF), baked to a
`.cube`, run as a background job (`job.run_grade_job`) and persisted to
`resolved_grades`. Flags live in `backend/app/config.py`:
`grade_pipeline` (`"v1"`, `backend/app/config.py:146`), `grade_even_lighting`
(`config.py:150`), `grade_semantic` (`config.py:153`) — **all currently `True`**.

### Primary cause — matching is a silent no-op on real data
With `grade_semantic = True`, `run_grade_job` builds groups from
`scene_group.group_shots_semantically`
(`backend/app/services/l3/grade/job.py:316-325`). That function
(`backend/app/services/l3/grade/scene_group.py:59-79`) only chain-links two
adjacent shots when one of these holds
(`scene_group.py:69-75`):

- `same_file` — same `file_id`, or
- `same_speaker` — equal non-null `speaker_person`, or
- `weak_tiebreak` — equal non-null `on_camera` **and** label/summary word overlap.

On the real DB, `speaker_person` / `on_camera` / `label` / `summary` are
populated on **zero** shots, so grouping degrades to **same-file-only**. On a
multi-file reel (e.g. the Siri reel, project `947d7e91…`, 8 files) every shot is
its own file → **8 singleton groups**. Then `run_grade_job` passes these groups
**unconditionally** to `solve_sequence_match`
(`job.py:326-329`), and `solve_sequence_match` uses `groups` **verbatim** when
it is not `None` (`backend/app/services/l3/grade/match.py:225`):

```225:227:backend/app/services/l3/grade/match.py
    for idxs in (groups if groups is not None else group_neighbors(ordered_shots)):
        if len(idxs) < 2:
            continue
```

Singleton groups are skipped (`len(idxs) < 2`), so **Match produces zero
deltas** — and because `groups` is non-`None`, the RGB-adjacency fallback
`group_neighbors` (`match.py:148-172`), which would otherwise chain ~5 of 8
shots, is **never called**. Net: on real reels, matching does nothing.

### Secondary causes
- **(a) Arbitrary anchor.** A group's anchor is
  `max(members, key=lambda s: (s.quality, s.key))`
  (`match.py:229`), but `ShotStats.quality` is **never set** — `job.py:302-309`
  constructs `ShotStats(key=..., file_id=..., stats=...)` with `quality`
  defaulting to `0.0` (`match.py:146`). So the tie always breaks on `key`
  (seg_id/op_id string), i.e. the anchor is an arbitrary shot, often an outlier.
- **(b) Weak match strength.** `SPAN_MATCH_STRENGTH = 0.4`,
  `CAST_MATCH_STRENGTH = 0.3` (`match.py:130-131`). Even when matching fires it
  only removes ~40%/30% of the difference; ~60% of the spread survives.
- **(c) No exposure convergence.** `solve_sequence_match`
  (`match.py:247-260`) matches black/white/mid **placement** via
  `_levels_delta_toward` (`match.py:175-198`) plus a per-channel cast nudge, but
  there is **no explicit mid-gray/exposure convergence term** across the group.
  The dominant inconsistency axis is exposure (dull shots mid ≈ 0.30 vs okay
  shots mid ≈ 0.55), and the current `_levels_delta_toward` mid nudge is
  clamped hard (`MID_MATCH_CLAMP = 1.2`, `match.py:133`) and only fires as a
  side effect of the black/white solve.
- **(d) Correct increases spread.** Per-shot `solve_correct_grade`
  (`backend/app/services/l3/grade/correct.py:178-223`) runs **independently**
  toward fixed targets (`TARGET_MID_GRAY = 0.42` `correct.py:67`,
  `TARGET_BLACK = 0.02` `correct.py:48`, `TARGET_WHITE = 0.97` `correct.py:49`).
  Dull/log-flat shots hit `LEVELS_SLOPE_MAX = 1.5` (`correct.py:63`) and later
  the composite `COMPOSITE_SLOPE_MAX = 2.0` (`resolver.py:69`), so they **cap
  and stay dull** while brighter shots move less → cross-shot spread widens
  (observed contrast std −74% worse on the Siri reel).
- **(e) Leveling fights Match.** `solve_leveling`
  (`backend/app/services/l3/grade/leveling.py:185-194`) targets a *local
  5-shot centered moving average* (`LEVELING_WINDOW = 5`, `leveling.py:42`) with
  a `±0.5`-stop exposure cap (`EXPOSURE_CAP_STOPS = 0.5`, `leveling.py:47`). This
  is a **different reference** than Match's group anchor, so Leveling re-diverges
  what Match aligned.

**Key guardrail interaction to respect.** `COMPOSITE_SLOPE_MAX = 2.0`
(`resolver.py:69`, applied by `_clamp_composite_v1`, `resolver.py:80-92`) means a
very dull/log-flat shot **cannot** be lifted to a bright reference by *slope
alone* — it caps and stays dull. Any new balance/exposure convergence must
therefore lift via a mechanism that is **not** pure unbounded slope (i.e. an
exposure gain *around a pivot* that adds offset, so midtones rise without the
slope ceiling clamping it), while never re-introducing the shadow crush the
guardrails prevent (`COMPOSITE_MID_FLOOR = 0.02`, `resolver.py:77`).

---

## 3. Design overview

Introduce an explicit **two-stage group → balance → match** model, all in the v1
working space, all computed **once per document** inside `run_grade_job` and
passed into `resolve_clip_grade` as pre-computed deltas (the same "resolve once,
compose as delta" pattern `match_delta` / `leveling_delta` already use).

```
                     ┌─────────────────────────────────────────────┐
                     │  run_grade_job  (v1 only, once per document)  │
                     └─────────────────────────────────────────────┘
   ordered shots ──► GROUP  (graceful: semantic → RGB-adjacency fallback)
                        │  groups = list[list[shot_index]]
                        ▼
                     per group: compute ROBUST REFERENCE (median member stats)
                        │
             ┌──────────┴───────────┐
             ▼                       ▼
        BALANCE delta            MATCH delta
   (pull each shot's         (existing black/white/mid
    exposure/WB/contrast      placement + cast, now
    toward group reference,   harder + explicit exposure
    via pivot gain+offset,    term, toward the SAME
    NOT slope-only)           robust reference)
             └──────────┬───────────┘
                        ▼
   resolve_clip_grade(pipeline="v1"):
     Correct → (Balance) → Match → Leveling(→ same reference) → Look → Arc → clamp
```

Per shot the composed stack becomes (v1):

```
Correct → Balance → Match → Leveling → Look → Arc → override → _clamp_composite_v1
```

Where **Balance** is the new primitive (§Phase 2). Match and Leveling both point
at the **same** robust group reference so they reinforce instead of fight.

The stages are ordered **by leverage**: turning grouping back on (Phase 1) is by
far the highest-impact change and is nearly self-contained; the rest build on it.

---

## 4. Phased implementation

Each phase lists exact file, function, approximate line, the change, new
constants (with defaults + one-line rationale), and how it is gated to v1.

> Line numbers are approximate and were verified against the code at plan-writing
> time. If they drift, locate by function/constant name.

---

### Phase 1 — Grouping degrades gracefully (turn matching back ON) — HIGHEST LEVERAGE

**Problem:** semantic grouping returns all-singletons on real data and *bypasses*
the RGB fallback (§2 primary cause).

**Two options considered:**

- **Option A** — inside `group_shots_semantically`, also accept each shot's span
  `rgb_mean` and OR-in the same `SPAN_RGB_DIST_MAX` proximity test
  `group_neighbors` already uses.
- **Option B (RECOMMENDED)** — in `job.py`, only *use* `semantic_groups` when it
  yields non-singleton structure; otherwise fall back to `group_neighbors`
  (i.e. pass `groups=None` so `solve_sequence_match` uses its own RGB grouping).

**Why B:** it keeps `scene_group.py` purely metadata-driven (semantic signals
stay a *separate concept* from RGB proximity), needs no signature change to
`group_shots_semantically`, and reuses the already-tested `group_neighbors`
verbatim. It is also the cleanest rollback point. Option A blends two distinct
ideas into one function and duplicates the RGB test.

**Change — `backend/app/services/l3/grade/job.py`, `run_grade_job`, ~lines 316-329.**

Current:

```316:329:backend/app/services/l3/grade/job.py
        semantic_groups = None
        if settings.grade_semantic:
            scene_meta = [
                SceneMeta(key=s.key, file_id=s.file_id,
                         speaker_person=s.item.get("speaker_person"),
                         on_camera=s.item.get("on_camera"),
                         label=str(s.item.get("label") or ""), summary=str(s.item.get("summary") or ""))
                for s in shots
            ]
            semantic_groups = group_shots_semantically(scene_meta)
        match_deltas: Dict[str, Grade] = solve_sequence_match(
            [shot_stats[s.key] for s in shots], groups=semantic_groups,
            working_space=WORKING_SPACE_V1,
        )
```

New behavior: build `semantic_groups` as today, but keep it **only if it
produced at least one non-singleton group**; otherwise set it to `None` so
`solve_sequence_match` runs its built-in `group_neighbors` RGB fallback. Add a
small helper (module-level in `job.py`):

```python
def _has_real_groups(groups: Optional[List[List[int]]]) -> bool:
    """True iff at least one group has 2+ members (i.e. grouping actually
    found structure to match on). All-singletons means the semantic signals
    were absent/unhelpful -- fall back to RGB adjacency instead."""
    return bool(groups) and any(len(g) >= 2 for g in groups)
```

Then, after computing `semantic_groups`:

```python
            semantic_groups = group_shots_semantically(scene_meta)
            if not _has_real_groups(semantic_groups):
                semantic_groups = None   # degrade to RGB adjacency (group_neighbors)
```

Leave the `solve_sequence_match(...)` call unchanged (it already interprets
`groups=None` as "use `group_neighbors`").

**Gating:** this block only runs under v1 (the whole `run_grade_job` is v1-only;
`maybe_enqueue` returns early for `legacy`, `job.py:238-239`). No legacy impact.

**Expected effect:** Siri reel 8 singleton groups → `group_neighbors` chains ~5
of 8 adjacent shots into non-singleton groups; matching acts on ~5/8 shots
instead of 0/8.

**One design note to leave in a comment:** we intentionally do *not* merge
semantic + RGB grouping (Option A). Semantic grouping remains authoritative when
it finds structure; RGB adjacency is the graceful fallback when it does not.

---

### Phase 2 — Robust group reference + Balance primitive (two-stage balance-then-match)

This phase does the real convergence work. It has two parts: a **robust
reference** (replaces the arbitrary anchor) and a **Balance delta** (the missing
"shot match" step that pulls exposure/WB/contrast toward that reference).

#### 2a. Robust group reference (replaces arbitrary anchor)

**Problem (§2b):** anchor = `max(members, (quality, key))` with `quality` always
0.0 → arbitrary outlier anchor.

**Two options:**
- populate `ShotStats.quality` from `cut_records.total_quality` at
  `job.py:302-309`, **or**
- **(RECOMMENDED)** compute a **median member** reference (median `mid_gray`,
  median `black_point`, median `white_point`, per-channel median `rgb_mean`)
  over the group and match/balance toward *that synthetic reference* rather than
  a single chosen shot.

**Why median:** no dependency on a maybe-empty `total_quality` field; a median is
robust to a single outlier shot in the group (exactly the failure mode of "pick
one shot"); and it gives balance/match a *stable* target. `cut_records` wiring is
extra plumbing for a field that may still be sparse.

**New module — `backend/app/services/l3/grade/reference.py`** (small, pure, no
I/O; keeps `match.py` focused). Define:

```python
"""Robust per-group color reference (color_shot_matching.plan.md Phase 2a):
the MEDIAN member stats of a scene-group, used as the single target both
Balance (Phase 2b) and Match (Phase 4) pull every group member toward. A
median (not a picked 'anchor' shot, not a mean) is robust to one outlier
shot in the group and needs no maybe-empty quality signal.

Inputs are WORKING-SPACE scalars (already projected by the caller in
job.py); this module stays pure numeric with no tone.py dependency, same
convention as leveling.py."""
from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional


@dataclass
class GroupReference:
    mid_gray: float
    black_point: float
    white_point: float
    rgb_mean: List[float]   # per-channel median [r, g, b]


def _median(values: List[float], default: float) -> float:
    vals = [v for v in values if v is not None]
    return float(median(vals)) if vals else default


def compute_group_reference(member_stats: List[Dict]) -> Optional[GroupReference]:
    """Median-member reference over a group's WORKING-SPACE stats dicts.
    None for a <2-member group (nothing to converge)."""
    if len(member_stats) < 2:
        return None
    mids   = [m.get("mid_gray")    for m in member_stats]
    blacks = [m.get("black_point") for m in member_stats]
    whites = [m.get("white_point") for m in member_stats]
    rgbs   = [m.get("rgb_mean") or [0.5, 0.5, 0.5] for m in member_stats]
    rgb_med = [_median([r[c] for r in rgbs], 0.5) for c in range(3)]
    return GroupReference(
        mid_gray=_median(mids, 0.5),
        black_point=_median(blacks, 0.0),
        white_point=_median(whites, 1.0),
        rgb_mean=rgb_med,
    )
```

> The reference is computed on **working-space-projected** stats (the caller in
> `job.py` already has `_to_working_scalar`; project `mid_gray`/`black_point`/
> `white_point` and each `rgb_mean` channel before calling
> `compute_group_reference`). This keeps Balance's math in the same space the CDL
> is applied (§the "everything too dark" bug this repo already fixed).

#### 2b. Balance delta (the missing shot-match step)

**New module — `backend/app/services/l3/grade/balance.py`** (pure numeric, no
I/O, mirrors `leveling.py`'s conventions). It pulls each shot toward its group's
`GroupReference` along three axes:

1. **Exposure (dominant):** move the shot's `mid_gray` toward the reference's via
   a **pivot gain + offset** (NOT slope-only), so a dull shot rises through
   midtones without relying on unbounded slope. In working (linear) space, an
   exposure move that preserves the black point is: pick `black` = shot black
   point (projected), solve `gain` so that `mid*gain' ...` lands on target where
   the op is `out = in*gain + offset`, `offset = black*(1 - gain)` (pivot at
   black). This adds a positive offset that lifts midtones — the composite slope
   ceiling clamps `slope` but the **offset** survives, so a capped-slope dull
   shot can still be lifted (respecting `COMPOSITE_MID_FLOOR`, which only *floors*
   negative offsets, never blocks a positive lift).
2. **White balance:** per-channel gain toward the reference's `rgb_mean` ratios
   (R/G and B/G aligned to the reference), damped and clamped.
3. **Contrast:** nudge the shot's (white−black) range toward the reference's
   range, damped, with the black point held (so we do not crush shadows).

All three are damped by strengths and hard-clamped so Balance is a *convergence
nudge*, not a full replacement (genuinely different shots that happen to be
adjacent must not be flattened into each other).

```python
"""Balance layer (color_shot_matching.plan.md Phase 2b): the missing
'shot match' step. Per scene-group, pull each member's exposure, white
balance, and contrast toward a ROBUST group reference (reference.py) so a
multi-file reel converges shot-to-shot. Runs once per document in
run_grade_job (v1 only), composed BEFORE Match in resolver.py.

Works entirely on WORKING-SPACE scalars (projected by the caller). The
exposure move is a PIVOT GAIN + OFFSET (not slope-only): the composite
slope ceiling (resolver.COMPOSITE_SLOPE_MAX) would otherwise cap a very
dull shot's lift; a pivot-at-black gain adds a positive offset that lifts
midtones and survives the slope clamp, without crushing shadows (the
positive offset never trips COMPOSITE_MID_FLOOR, which only floors negative
offsets)."""
from __future__ import annotations

from typing import Dict, List, Optional

from app.services.l3.grade.cdl import Grade, compose
from app.services.l3.grade.reference import GroupReference

# How hard Balance converges each axis toward the group reference. Higher =
# tighter shot-to-shot consistency, at the cost of erasing real intra-scene
# variation. Exposure is the dominant inconsistency axis, so it converges
# hardest; WB/contrast are gentler because they compose on top.
BALANCE_EXPOSURE_STRENGTH = 0.8
BALANCE_WB_STRENGTH = 0.6
BALANCE_CONTRAST_STRENGTH = 0.5

# Never-worse ceilings (a single shot can't be gained/contrasted past these
# multipliers toward the reference, so one wild member can't drag the math).
BALANCE_WB_CLAMP = 1.4
BALANCE_CONTRAST_CLAMP = 1.5


def _pivot_gain_offset(value: float, target: float, black: float,
                       strength: float) -> tuple[float, float]:
    """slope/offset (pivot at `black`) moving `value` toward `target` by
    `strength`. out = in*slope + offset, with offset = black*(1 - slope) so
    the black point is preserved and midtones lift via the offset term
    (survives the composite SLOPE ceiling)."""
    if value <= 1e-6:
        return 1.0, 0.0
    goal = value + (target - value) * strength
    slope = goal / value
    offset = black * (1.0 - slope)
    return slope, offset


def solve_balance(
    ordered_stats: List[Optional[Dict]],
    groups: List[List[int]],
    references: Dict[int, GroupReference],
    keys: List[str],
) -> Dict[str, Grade]:
    """shot_key -> a Balance delta toward its group's reference. `groups` is
    the SAME grouping match uses (list of index lists into ordered_stats);
    `references[gi]` is that group's GroupReference (None-groups skipped);
    `keys[i]` is shot i's key. Members of singleton/no-reference groups get
    no delta."""
    out: Dict[str, Grade] = {}
    for gi, idxs in enumerate(groups):
        ref = references.get(gi)
        if ref is None or len(idxs) < 2:
            continue
        for i in idxs:
            stats = ordered_stats[i] or {}
            mid = stats.get("mid_gray")
            black = float(stats.get("black_point") or 0.0)
            white = float(stats.get("white_point") or 1.0)
            rgb = stats.get("rgb_mean") or [0.5, 0.5, 0.5]

            # 1) exposure (pivot at black)
            if mid is not None:
                es, eo = _pivot_gain_offset(float(mid), ref.mid_gray, black,
                                            BALANCE_EXPOSURE_STRENGTH)
            else:
                es, eo = 1.0, 0.0
            exposure = Grade(slope=(es, es, es), offset=(eo, eo, eo))

            # 2) white balance (per-channel gain toward the reference cast)
            eps = 1e-6
            wb = []
            for c in range(3):
                full = ref.rgb_mean[c] / max(eps, rgb[c])
                g = 1.0 + (full - 1.0) * BALANCE_WB_STRENGTH
                wb.append(max(1.0 / BALANCE_WB_CLAMP, min(BALANCE_WB_CLAMP, g)))
            # normalize so WB doesn't change overall exposure (green channel = 1)
            wb = [w / wb[1] for w in wb]
            white_balance = Grade(slope=(wb[0], wb[1], wb[2]))

            # 3) contrast (range toward reference range, black held)
            shot_range = max(1e-4, white - black)
            ref_range = max(1e-4, ref.white_point - ref.black_point)
            full_cs = ref_range / shot_range
            cs = 1.0 + (full_cs - 1.0) * BALANCE_CONTRAST_STRENGTH
            cs = max(1.0 / BALANCE_CONTRAST_CLAMP, min(BALANCE_CONTRAST_CLAMP, cs))
            co = black * (1.0 - cs)
            contrast = Grade(slope=(cs, cs, cs), offset=(co, co, co))

            delta = compose(compose(exposure, white_balance, 1.0), contrast, 1.0)
            out[keys[i]] = delta
    return out
```

#### 2c. Wire Balance into the resolver and the job

**`backend/app/services/l3/grade/resolver.py`, `resolve_clip_grade`, ~lines 156-194.**
Add a `balance_delta` parameter and compose it **between Correct and Match**:

Current signature (`resolver.py:156-165`) and composition (`resolver.py:190-194`):

```190:194:backend/app/services/l3/grade/resolver.py
    stack = solve_correct_grade(color_stats, already_graded=already_graded, pipeline=pipeline)
    if match_delta is not None:
        stack = compose(stack, match_delta, 1.0)
    if leveling_delta is not None:
        stack = compose(stack, leveling_delta, 1.0)
```

Change to (add `balance_delta: Optional[Grade] = None` to the signature after
`match_delta`, and compose it before `match_delta`):

```python
    stack = solve_correct_grade(color_stats, already_graded=already_graded, pipeline=pipeline)
    if balance_delta is not None:
        stack = compose(stack, balance_delta, 1.0)
    if match_delta is not None:
        stack = compose(stack, match_delta, 1.0)
    if leveling_delta is not None:
        stack = compose(stack, leveling_delta, 1.0)
```

> **Tradeoff (Correct vs Balance) — recommendation.** Correct still runs first
> (per-shot, toward fixed targets), then Balance pulls the group together. We
> keep Correct because it also does WB from `wb_gray_world`/`wb_white_patch` and
> the log-flat pre-lift, which Balance does not replicate. But Correct is the
> layer that *increases* spread (§2d). To stop it fighting Balance without
> ripping it out, add a v1 gate: **when a shot is a member of a non-singleton
> group, skip Correct's mid-gray retarget** (the `_solve_levels_v1` extra nudge)
> and let Balance own exposure convergence; keep Correct's black/white anchoring
> and WB. This is optional polish — implement it only if, after Phases 1–4, the
> acceptance metrics are not met with Correct's mid retarget left on. If you do
> it, thread a `converge_exposure: bool` flag from `resolve_clip_grade` →
> `solve_correct_grade` → `_solve_levels_v1` that, when `True`, returns the
> base `_solve_levels` result (skips the mid nudge). Default `False` keeps
> today's behavior; legacy never sees it. **Do not** remove Correct.

**`backend/app/services/l3/grade/job.py`, `run_grade_job`, ~lines 311-366.**
After grouping is finalized (Phase 1) and before/around the `solve_sequence_match`
call, compute the grouping *explicitly* so Balance and Match share it, compute
per-group references, solve Balance, and pass `balance_delta` into
`resolve_clip_grade`.

Concretely, restructure the match block so the group index-lists are available
as a variable (today they are hidden inside `solve_sequence_match`). Add near the
match call:

```python
        from app.services.l3.grade.match import group_neighbors
        from app.services.l3.grade.balance import solve_balance
        from app.services.l3.grade.reference import compute_group_reference

        ordered = [shot_stats[s.key] for s in shots]
        # the SAME grouping match will use (Phase 1 already decided semantic vs None)
        groups = semantic_groups if semantic_groups is not None else group_neighbors(ordered)

        # working-space-projected stats per shot for reference/balance math
        def _ws_stats(st):
            st = st or {}
            return {
                "mid_gray": _to_working_scalar(st.get("mid_gray"), None) if st.get("mid_gray") is not None else None,
                "black_point": _to_working_scalar(st.get("black_point"), 0.0),
                "white_point": _to_working_scalar(st.get("white_point"), 1.0),
                "rgb_mean": [_to_working_scalar(c, 0.5) for c in (st.get("rgb_mean") or [0.5, 0.5, 0.5])],
            }
        ws = [_ws_stats(o.stats) for o in ordered]
        references = {}
        for gi, idxs in enumerate(groups):
            ref = compute_group_reference([ws[i] for i in idxs])
            if ref is not None:
                references[gi] = ref
        balance_deltas = solve_balance(ws, groups, references, [s.key for s in shots])

        match_deltas: Dict[str, Grade] = solve_sequence_match(
            ordered, groups=groups, working_space=WORKING_SPACE_V1,
        )
```

Then in the resolve loop (`job.py:352-358`) pass the new delta:

```python
            grade_json = resolve_clip_grade(
                s.item, color_stats=stats, sequence_look=sequence_look,
                balance_delta=balance_deltas.get(s.key),
                match_delta=match_deltas.get(s.key), leveling_delta=leveling_deltas.get(s.key),
                pipeline="v1",
            )
```

> Note: we now pass `groups=groups` (never `None`) to `solve_sequence_match`,
> since Phase 1's fallback is applied *before* this line (`groups` is
> `group_neighbors(ordered)` when semantic was singletons). This is behavior-
> identical to letting `solve_sequence_match` call `group_neighbors` itself, but
> guarantees Balance and Match group **identically**.

**Gating:** all of the above is inside `run_grade_job`, which is v1-only. Legacy
`resolve_clip_grade` callers never pass `balance_delta` (defaults `None` → no
compose). Byte-identical legacy preserved.

---

### Phase 3 — Robust reference in Match (remove the arbitrary anchor)

**`backend/app/services/l3/grade/match.py`, `solve_sequence_match`, ~lines 201-261.**

Today Match picks `anchor = max(members, key=(quality, key))` (`match.py:229`)
and matches every member toward that single shot. Replace the anchor's role with
the **same median `GroupReference`** Balance uses, so both stages converge on one
target.

Two implementation choices:

- **Minimal (RECOMMENDED):** add an optional `references` parameter to
  `solve_sequence_match` mapping group-index → a stats-like dict (or
  `GroupReference`). When provided, use the reference's `black_point`/
  `white_point`/`mid_gray`/`rgb_mean` as `a_*` (the "anchor" values) instead of a
  member shot. When absent (legacy default `None`), keep today's
  `max(members, ...)` anchor exactly. This preserves the existing signature's
  default behavior byte-for-byte and needs no `quality` field.
- Alternative: populate `ShotStats.quality` from `cut_records.total_quality` at
  `job.py:302-309`. **Not recommended** — depends on a sparse field and still
  matches toward a single shot, not a robust center.

Sketch of the change to `solve_sequence_match`:

```python
def solve_sequence_match(
    ordered_shots, groups=None, working_space="rec709", references=None,
):
    ...
    for gi, idxs in enumerate(groups if groups is not None else group_neighbors(ordered_shots)):
        if len(idxs) < 2:
            continue
        members = [ordered_shots[i] for i in idxs]
        ref = (references or {}).get(gi)
        if ref is not None:
            a_black = _proj(float(ref.black_point), working_space)
            a_white = _proj(float(ref.white_point), working_space)
            a_mid   = _proj(float(ref.mid_gray), working_space)
            a_rgb   = [_proj(float(c), working_space) for c in ref.rgb_mean]
            match_all = True   # no member is "the anchor" — every member matches the reference
        else:
            anchor = max(members, key=lambda s: (s.quality, s.key))
            a = anchor.stats or {}
            ... (today's a_black/a_white/a_mid/a_rgb from anchor) ...
            match_all = False
        for s in members:
            if not match_all and s.key == anchor.key:
                continue
            ... (unchanged per-member solve) ...
```

> **Important:** the `references` passed here must be in the **same space** as
> `_proj` expects. `_proj` (`match.py:36-46`) projects a *display* scalar into
> working space. Balance/`compute_group_reference` in `job.py` were fed
> **working-space** stats (via `_ws_stats`). To avoid double-projection, build a
> **second, display-space** reference for Match, or (cleaner) pass Match a
> reference computed from **display-space** member stats and let `_proj` handle
> the projection. **Recommendation:** compute two references per group in
> `job.py` — one from display-space stats for Match (so `_proj` works
> unchanged), one from working-space stats for Balance. Both use
> `compute_group_reference`; only the input space differs. Add a short comment
> in `job.py` making this explicit.

**Gating:** `references=None` default reproduces today's anchor behavior exactly,
so legacy and any other caller are unaffected. Only `run_grade_job` (v1) passes
`references`.

---

### Phase 4 — Match harder, add explicit exposure term, reconcile with Leveling

**4a. Explicit exposure/mid-gray match term.** With Balance now owning the bulk
of exposure convergence (Phase 2b), Match's job is the *placement + cast* it
already does. Raise its strengths so the residual spread closes:

**`backend/app/services/l3/grade/match.py`, ~lines 130-131.**

```130:131:backend/app/services/l3/grade/match.py
SPAN_MATCH_STRENGTH = 0.4     # mirrors MATCH_STRENGTH: a nudge, never a full match
CAST_MATCH_STRENGTH = 0.3     # gentler -- this composes ON TOP of the levels nudge
```

Change to:

```python
SPAN_MATCH_STRENGTH = 0.85    # converge placement hard toward the group reference
CAST_MATCH_STRENGTH = 0.6     # lift cast convergence for genuine same-scene groups
```

Also lift the mid nudge clamp so the exposure placement can actually reach the
reference (`match.py:133`):

```133:133:backend/app/services/l3/grade/match.py
MID_MATCH_CLAMP = 1.2         # never-worse ceiling on the extra mid-gray nudge
```

Change to:

```python
MID_MATCH_CLAMP = 1.5         # allow a real exposure convergence, still bounded
```

> These are safe because the **composite guardrails** (`COMPOSITE_SLOPE_MAX`,
> `COMPOSITE_MID_FLOOR`) still bound the final stacked CDL, and Balance's
> pivot-offset lift means Match rarely needs a large slope.

**4b. Reconcile Leveling with the same reference.** Leveling
(`leveling.py:185-194`) currently targets a local moving average, which re-
diverges Match/Balance. Two options:

- **(RECOMMENDED)** point Leveling at the **same per-group reference** for shots
  that belong to a non-singleton group, and keep the moving-average target only
  for ungrouped shots. Concretely, in `job.py`'s leveling block
  (`job.py:334-348`), after computing `references`, override each grouped shot's
  `ShotLevelInput` target-relevant values so the smooth target *equals* the group
  reference. Simplest low-risk version: **widen the window and raise the cap** so
  Leveling stops making per-shot corrections that undo Match:
  - `leveling.py:42` `LEVELING_WINDOW = 5` → `LEVELING_WINDOW = 9`
    (a wider low-pass flattens more, tracks only a slow arc — less fighting).
  - `leveling.py:47` `EXPOSURE_CAP_STOPS = 0.5` → keep at `0.5` (do **not** raise
    the cap; a wider window is the safer lever — raising the cap lets Leveling
    move shots *further*, which risks re-diverging).
- Alternative (more work, more precise): make `solve_leveling` accept an optional
  per-shot explicit target and, for grouped shots, feed the group reference's
  `mid_gray`/`black`/`white`. Only do this if the window-widening alone leaves
  Leveling fighting Match in testing.

**Recommendation:** start with the **window widen** (`LEVELING_WINDOW = 9`) — it
is one constant, easy to revert, and directly reduces Leveling's shot-to-shot
correction. Escalate to the explicit-target version only if acceptance metrics
miss.

**Gating:** `LEVELING_WINDOW` / `EXPOSURE_CAP_STOPS` are only *read* under v1
(Leveling runs only inside `run_grade_job`, gated on `grade_even_lighting`, and
`run_grade_job` is v1-only). Legacy never runs Leveling. Match constants are only
consumed by `solve_sequence_match`, which legacy calls with `working_space=
"rec709"` and no references — but **the constant values themselves changing would
also change legacy match output**. **Therefore:** legacy does **not** use
`solve_sequence_match` at all (it uses `solve_match_deltas` /
`cluster_grade_groups`, `match.py:84-111`, which reference `MATCH_STRENGTH`, a
*separate* constant at `match.py:29`). Confirm this before changing:
`SPAN_MATCH_STRENGTH`/`CAST_MATCH_STRENGTH`/`MID_MATCH_CLAMP` are used **only** by
`solve_sequence_match` (the v1 path). `MATCH_STRENGTH` (legacy) is untouched.

---

## 5. New / changed constants

| Constant | File | Old | New | Rationale |
|---|---|---|---|---|
| `BALANCE_EXPOSURE_STRENGTH` | `balance.py` (new) | — | `0.8` | Exposure is the dominant inconsistency axis; converge it hard. |
| `BALANCE_WB_STRENGTH` | `balance.py` (new) | — | `0.6` | Align cast toward group reference; gentler (composes on top). |
| `BALANCE_CONTRAST_STRENGTH` | `balance.py` (new) | — | `0.5` | Converge range without flattening genuine intra-scene contrast. |
| `BALANCE_WB_CLAMP` | `balance.py` (new) | — | `1.4` | One wild member can't drag WB past this. |
| `BALANCE_CONTRAST_CLAMP` | `balance.py` (new) | — | `1.5` | Bounded contrast convergence (never-worse). |
| `SPAN_MATCH_STRENGTH` | `match.py:130` | `0.4` | `0.85` | Close residual placement spread after Balance. |
| `CAST_MATCH_STRENGTH` | `match.py:131` | `0.3` | `0.6` | Stronger cast convergence for genuine same-scene groups. |
| `MID_MATCH_CLAMP` | `match.py:133` | `1.2` | `1.5` | Let the mid nudge actually reach the reference (still bounded). |
| `LEVELING_WINDOW` | `leveling.py:42` | `5` | `9` | Wider low-pass so Leveling stops fighting Match. |
| `INPUT_HASH_SCHEMA_VERSION` | `job.py:53` | `2` | `3` | Grade math changed → invalidate cached grades (see Ops). |

Unchanged guardrails to **preserve exactly**: `COMPOSITE_SLOPE_MAX = 2.0`,
`COMPOSITE_MID_FLOOR = 0.02` (`resolver.py:69,77`); `LEVELS_SLOPE_MAX = 1.5`,
`TARGET_MID_GRAY = 0.42`, `TARGET_BLACK = 0.02`, `TARGET_WHITE = 0.97`
(`correct.py:63,67,48,49`); `EXPOSURE_CAP_STOPS = 0.5` (`leveling.py:47`).

---

## 6. Switchability / rollback

The repo values easy rollback. Provide a **single kill switch** plus clean
revert points.

1. **Legacy is untouched by construction.** Every new delta is threaded as an
   optional parameter defaulting to `None`/off, gated behind `pipeline == "v1"`
   or a v1-only call site. Legacy `solve_correct_grade` / `resolve_clip_grade` /
   `solve_match_deltas` produce identical bytes. The legacy match path
   (`cluster_grade_groups` / `solve_match_deltas`, `MATCH_STRENGTH`) is not
   modified.

2. **New setting to disable the redesign without a revert.** Add to
   `backend/app/config.py` (near the other grade flags, ~line 153):

   ```python
   # color_shot_matching.plan.md: enable the two-stage group->balance->match
   # redesign (Phase 1-4). Off falls back to the pre-redesign v1 matching
   # (semantic-or-nothing grouping, no balance, weak match strengths).
   grade_shot_match_v2: bool = True
   ```

   Gate the new behavior in `run_grade_job` on `settings.grade_shot_match_v2`:
   - when **off**, restore the pre-redesign path: pass `semantic_groups` (no
     `_has_real_groups` fallback), no Balance delta, `references=None` to Match.
     Use the *old* constant values by leaving the constants but reading them
     conditionally — simplest is to branch the job logic, not the constants.

   > Simpler alternative if you prefer fewer branches: keep the new constants but
   > gate only the *structural* additions (Balance + `_has_real_groups` fallback +
   > references) on the flag. The constant bumps (`SPAN_MATCH_STRENGTH` etc.)
   > then apply whenever v1 matching fires. Decide based on how surgical a
   > rollback you want; the flag-gated structural additions are the important part.

   Also add `grade_shot_match_v2` to `compute_input_hash`'s `flags` payload
   (`job.py:138-142`) so toggling it invalidates cached grades.

3. **Clean revert point.** All new logic lives in two new files (`balance.py`,
   `reference.py`) plus additive parameters. A full revert = delete those two
   files, revert the constant bumps, revert the `job.py`/`resolver.py`/`match.py`
   diffs, and drop the flag. Keep the changes in one commit for a clean
   `git revert`.

---

## 7. Testing

Tests are a **plain script** (not pytest), run via:

```bash
cd backend && .venv/bin/python scripts/test_grade.py
```

Each test is a function that `print`s `ok  ...` and is called from `main()`
(`scripts/test_grade.py:763-813`). Add the new tests as functions and **append
calls to `main()`**. No DB/ffmpeg/R2 — use synthetic in-memory fixtures like the
existing `_cs(...)` helper (`scripts/test_grade.py:148-153`).

Add these:

1. **Grouping no longer all-singletons when metadata absent.**
   Build ordered `ShotStats` for a synthetic 8-shot multi-file reel where each
   shot has a *different* `file_id` but adjacent shots have RGB within
   `SPAN_RGB_DIST_MAX`. Assert `group_neighbors(shots)` yields at least one group
   with ≥2 members. Then assert the `job.py` fallback: with a
   `group_shots_semantically` result of all-singletons, `_has_real_groups(...)`
   is `False` (so the code would fall back). Import `_has_real_groups` from
   `grade.job`.

2. **Cross-shot spread SHRINKS after Balance+Match.**
   Synthetic 5-shot group: mids `[0.30, 0.55, 0.32, 0.50, 0.35]`, varied
   black/white and `rgb_mean` casts, all adjacent/same "scene". Compute
   pre-spread = stdev of mids. Run the new pipeline math (project to working
   space, `compute_group_reference`, `solve_balance`, `solve_sequence_match` with
   `references`), apply the composed per-shot delta to each shot's mid via the v1
   round-trip helper `_roundtrip_v1` (`scripts/test_grade.py:189-195`), and
   assert the **post-spread mid stdev is at least ~40% smaller** than pre-spread.
   Do the same assertion for contrast (white−black) and a WB ratio (R/G).

3. **No shadow-crush regression.**
   For the dullest shot in test (2), push a display 0.15 shadow through the full
   composed+`_clamp_composite_v1` grade via `_roundtrip_v1` and assert
   `shadow_out > 0.02` (mirrors `test_v1_grade_does_not_crush_midtones_or_shadows`,
   `scripts/test_grade.py:198-219`). Also assert a display 0.5 mid lands in
   `0.46–0.65` for the lifted dull shot (the "capped-dull is now lifted" check).

4. **Balance lifts a capped-dull shot via offset, not slope alone.**
   Give one shot a very low mid (e.g. 0.20) vs a reference mid ~0.50 and assert
   the resulting Balance `Grade` has a **positive offset** on all channels (the
   pivot-lift), and that after `_clamp_composite_v1` the mid still rises
   materially (display mid moves up by ≥ ~0.08). This is the explicit guardrail-
   interaction test.

5. **Legacy output unchanged.**
   Assert `solve_correct_grade(cs)` (no pipeline) is unchanged and that
   `resolve_clip_grade({}, color_stats=cs)` with no `balance_delta`/`pipeline`
   equals the pre-change result — the existing
   `test_correct_legacy_untouched_by_pipeline_param` and
   `test_layers_legacy_default_is_byte_identical_to_before` already cover the
   spirit; add one asserting `solve_sequence_match(shots)` (no `references`,
   default `working_space`) is **identical** to the anchor-based result for a
   fixed fixture (guards the `references=None` default path).

6. **Match `references` override uses the reference, not the anchor.**
   Two-member group; pass an explicit `references={0: GroupReference(...)}` and
   assert *both* members get a delta (no member is exempt as "the anchor"), and
   that with `references=None` exactly one member (the anchor) is exempt (today's
   behavior).

Keep every new test deterministic and DB-free. Run the whole script; it must end
with `all grade tests passed`.

---

## 8. Ops runbook (must do, in order)

After code + tests are green:

1. **Bump the cache-invalidation version.** In
   `backend/app/services/l3/grade/job.py:53`, change
   `INPUT_HASH_SCHEMA_VERSION = 2` → `= 3`. This changes every thread's
   `input_hash`, so all cached `resolved_grades` are treated as stale and
   re-computed on the next run. (The grade **math** changed, which is exactly the
   documented trigger for a bump — see the comment at `job.py:47-52`.)

2. **Restart the grade worker** so it loads the new code. (The Procrastinate
   worker on the `grade` queue — restart however this deployment runs it; a
   stale worker will keep applying old math.)

3. **Re-grade all projects.** Run:

   ```bash
   cd backend && .venv/bin/python scripts/_grade_all_projects.py
   ```

   It forces all grade flags on and runs `run_grade_job` for the latest gradeable
   thread of every project, printing a per-project `state / rows / baked` summary
   (`scripts/_grade_all_projects.py:83-86`).

4. **Verify** (next section). Spot-check the Siri reel (`947d7e91…`, 8 files) and
   `730bd940…`.

---

## 9. Acceptance criteria (quantified)

Measured across the shots of a reel, after re-grade, comparing corrected span
stats (project each shot's measured `mid_gray`/`black`/`white`/`rgb_mean` through
its final resolved CDL via the v1 round-trip, same as the tests):

- **Exposure:** cross-shot `mid_gray` standard deviation drops from ~0.09–0.11 to
  **< 0.04** on the Siri reel (`947d7e91…`) and `730bd940…`.
- **Contrast:** cross-shot (white−black) spread is **reduced** (target: ≥ ~40%
  smaller stdev than pre-redesign).
- **White balance:** cross-shot R/G and B/G ratio spread **reduced**.
- **No capped-dull shot:** no shot ends materially darker than the group
  reference because its correction hit `COMPOSITE_SLOPE_MAX`; every grouped shot's
  post-grade mid is within ~0.05 (working) of its group reference.
- **Darkness/shadow-crush fix preserved:** a display 0.5 mid lands in ~0.46–0.65;
  a display 0.15 shadow is **not** zeroed (`> 0.02`). This must remain true for
  the *dullest, most-lifted* shot in each group.
- **Matching is not a no-op:** on a multi-file reel with absent semantic
  metadata, ≥ ~5/8 shots receive a non-identity Balance and/or Match delta
  (vs 0/8 today).

If exposure stdev is still ≥ 0.04 after Phases 1–4, escalate the optional
levers in order: (i) skip Correct's mid retarget for grouped shots (§2c note),
(ii) explicit-target Leveling toward the group reference (§4b alternative),
(iii) raise `BALANCE_EXPOSURE_STRENGTH` toward `0.9`.

---

## 10. Finish

When all tests pass (`all grade tests passed`) and the Ops runbook verification
meets the acceptance criteria, **commit and push**:

```bash
cd /Users/vivekgandhari/Documents/cloud
git add -A
git commit -m "color: two-stage group->balance->match for shot-to-shot consistency (v1)"
git push
```

Keep the whole redesign in **one commit** so it can be `git revert`ed cleanly
(§6).
