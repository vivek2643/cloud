# Restore full brain parity for vcut cuts — the salience bridge

## 0. TL;DR

The old L3/v4 cuts pipeline handed the agentic brain a rich `salience` block
(`salience.events` + `kind`/`shape`/`peak_ms`) and `landmarks` on every cut
record. That block lights up a whole read-side machinery the brain depends on:
content-aware energy fragmentation (`cutrecord_map.synth_ladder` →
`_video_rung`/`_cluster_rung`/`_prune_events`/`resolve_cluster`), the per-piece
`range:` breakdown and per-piece addressing (`footage_map.piece_breakdown`/
`_piece_lines`/`resolve_piece`), the `peak:+Xs` key-frame tag (`_peak_tag`), the
`sig:` interior-structure tag (`_landmarks_tag`), and the asymmetric,
key-frame-anchored directional trim (`_before_rung`/`_after_rung`).

The new `vcut` pipeline (`app/services/vcut/`) is the default cuts producer now,
but `vcut/store.build_cut_records` (`backend/app/services/vcut/store.py:103`)
writes `hero_ts_ms=cut.peak_ms` and a composed `scene_specifics` (with a
`moments` mini shot-list) but **never populates `salience`/`landmarks`**. So on
every vcut cut all of the above is DARK: energy is a dumb symmetric shrink (not
content-aware), cuts never fragment into addressable pieces, and the directional
trim has no input. The brain sees LESS about a vcut cut than it saw about an old
cut — a straight regression.

This plan closes that gap with **one keystone**: synthesize the `salience` block
(and a scoped `landmarks` block) on each vcut cut record from its moment flags,
at STORE time, so the brain's EXISTING read machinery lights up with **zero new
brain code**. Re-ingest is acceptable (confirmed), so we populate authoritatively
in `build_cut_records`. `fit_to_window` is explicitly deferred.

---

## 1. Problem statement + the parity gap

### 1.1 What the brain consumes (the target contract)

`cutrecord_map._to_cut_dict` (`backend/app/services/l3/cutrecord_map.py:481`)
projects each persisted `cut_records` row into the cut dict the footage map
reads. Two blocks matter here and are passed straight through from the DB row:

```554:558:backend/app/services/l3/cutrecord_map.py
        "salience": row.get("salience") or {},
        # brain_perception_upgrade.plan.md Change 1: compact interior-
        # structure landmarks (post._landmarks), code-computed. {} on a
        # pre-migration row or a cut with no interior structure.
        "landmarks": row.get("landmarks") or {},
```

Everything downstream keys off `salience` / `salience.events` / `landmarks`:

- **Ladder synthesis** — `synth_ladder` → `_video_rung`
  (`cutrecord_map.py:304`) reads `salience`, `salience.events`, `salience.kind`,
  `salience.shape`, `salience.peak_ms`, `salience.span_ms`:
  - `len(events) > 1` → `_cluster_rung` (`cutrecord_map.py:281`) → multi-piece
    ("spans") rung → content-aware fragmentation via `resolve_cluster`
    (`cutrecord_map.py:213`) + the rising-salience gate `_prune_events`
    (`cutrecord_map.py:195`).
  - `len(events) <= 1`, `kind is None` → `_symmetric_rung` (legacy dumb shrink).
  - `kind == "span"` → `_span_rung`.
  - else (`kind` present, i.e. `"point"`/`"none"`) → `shape`-driven asymmetric
    trim: `"before"` → `_before_rung`, `"after"` → `_after_rung`, else symmetric
    (`cutrecord_map.py:342-349`).
- **Piece breakdown / addressing** — `footage_map.piece_breakdown`
  (`backend/app/services/l3/footage_map.py:578`), `_piece_lines`
  (`footage_map.py:640`), `resolve_piece` (`footage_map.py:663`) read
  `salience.events`, each event's `peak_ms`/`score`/`onset_ms`/`settle_ms`/`kind`,
  and `salience.primary`. This renders the brain-visible line
  `range: plays whole ~Xs, or splits into its N beats; tightening keeps only the
  strongest, down to ~K … dropping the weaker ones:` plus one `pos/of strength
  dur kind · core|drops when tight` line per piece.
- **Peak tag** — `_peak_tag` (`footage_map.py:695`) reads `salience.score` +
  `salience.peak_ms` → `peak:+X.Xs`.
- **Landmarks tag** — `_landmarks_tag` (`footage_map.py:732`) reads
  `landmarks[ch]["n"]` for `ch in ("act","adx","sil","shot")` → `sig:act3,shot1`.

### 1.2 The exact event dict shape (ground truth)

The old segmenter defines the contract in `l3/v4_segment.py`. A **point** event
(`_point_event`, `v4_segment.py:683-695`) and the top-level `salience`
(`_build_salience`, `v4_segment.py:991-1004`):

```694:695:backend/app/services/l3/v4_segment.py
    return {"peak_ms": peak_ms, "score": _score_at(curve, peak_i), "kind": "point",
            "onset_ms": onset_ms, "settle_ms": settle_ms, "span_ms": None}
```

```998:1004:backend/app/services/l3/v4_segment.py
    return {
        "peak_ms": prim["peak_ms"], "score": prim["score"], "kind": prim["kind"],
        "span_ms": prim["span_ms"],
        "events": [dict(ev) for ev in events],
        "primary": primary,
        "density": density,
    }
```

`post.finalize` then rides `shape` onto that block for a video cut:

```1162:1163:backend/app/services/l3/post.py
            salience = dict(v4_meta.get("salience") or {})
            salience["shape"] = cut.shape
```

So the **authoritative target `salience` shape** the bridge must emit is:

```
salience = {
  "peak_ms": int,                 # representative (primary) event peak, absolute ms
  "score":   float,               # primary event score, 0..1 (cut-relative)
  "kind":    "point",             # vcut flags are point moments (never "span"/None)
  "shape":   "before"|"after"|"center",   # from the representative flag (see §3.4)
  "span_ms": None,
  "primary": int,                 # index into events of the strongest
  "density": float,               # informational stat, 0..1
  "events": [
    { "peak_ms": int, "score": float, "kind": "point",
      "onset_ms": int, "settle_ms": int, "span_ms": None }, ...
  ],
}
```

Persistence: `salience` and `landmarks` are first-class `cut_records` columns
written straight from the `CutRecord` dataclass by `ingest_store.insert_cut_records`
(`backend/app/services/l3/ingest_store.py:178,204`); `CutRecord.salience` /
`CutRecord.landmarks` default to `{}` (`post.py:684,689`). **So the bridge only
needs to set `salience=`/`landmarks=` on the `CutRecord` that `build_cut_records`
already constructs** — no new column, no new migration, no new write path.

### 1.3 What vcut currently emits

`vcut/store.build_cut_records` (`store.py:103-135`) constructs each `CutRecord`
with `salience` / `landmarks` left at the dataclass default (`{}`). It writes:
`hero_ts_ms=cut.peak_ms`, `label`/`summary`, `framing`, `pace` (pinned 1x),
`total_quality`, `channel="shown"`, and — post-insert, separately — the composed
`scene_specifics` via `insert_video_cuts` → `update_cut_scene_specifics`
(`store.py:156-158`). Nothing sets `salience`/`landmarks`. Result: on every vcut
cut, `_video_rung` falls into the `kind is None` symmetric branch, `piece_breakdown`
returns `None` (single/zero events), `_peak_tag` renders nothing (no `salience.score`),
`_landmarks_tag` renders nothing (`landmarks == {}`).

### 1.4 The data is already there

The vcut moment flags carry exactly what's needed — see `MomentFlag`
(`backend/app/services/vcut/resolve.py:40-58`): `t_ms`, `shape`
(`"build"|"settle"|"both"`), `summary`, `specifics`. `resolve._resolve_file`
already refines each flag onto real content (`_refine_peak_ms`,
`resolve.py:179`), merges flags into groups, picks a representative
(`_representative_peak`, `resolve.py:329`), and composes a `moments` mini
shot-list (`_composed_specifics`, `resolve.py:363-382`). And `build_cut_records`
already has the per-file seam curves in hand (`seam[file_id]` = `{hop_ms, S,
action_energy, frame_diff, src_w, src_h}` — see `orchestrate.py:64-104`), which
is exactly what's needed to score events. The bridge is a small, deterministic
transform of data the pipeline already produces.

---

## 2. Current-state findings (grounded)

| Concern | Old L3/v4 | New vcut | Evidence |
|---|---|---|---|
| `salience` populated? | Yes — `post.finalize` sets it per cut | **No** — `{}` default | `post.py:1152-1163` vs `store.py:126-134` |
| `salience.events` | 1+ point/span events per cluster | absent | `v4_segment.py:991-1004` vs — |
| `kind`/`shape`/`peak_ms` | set (drives directional trim) | absent (`hero_ts_ms` only) | `cutrecord_map.py:336-349` |
| `landmarks` | 4 channels (act/adx/sil/shot) | `{}` | `post.py:1182-1188` |
| Energy behavior | content-aware fragment + prune | symmetric dumb shrink | `_video_rung` `kind is None` branch |
| Piece breakdown/addressing | `range:` line + `resolve_piece` | none (single event) | `footage_map.py:578,663` |
| Per-moment info recompose | events carry structure | frozen `scene_specifics.moments` | `store.py:156-158` |

The read side (`cutrecord_map`/`footage_map`) is **fully intact and generic** —
it already reads `row.get("salience")`/`row.get("landmarks")`. It is DARK only
because the vcut write side leaves those two fields empty. **This plan changes
the write side, not the read side.**

### 2.1 Where `resolve.py` drops per-moment structure today

`_composed_specifics` (`resolve.py:363-382`) emits a `moments` list only for a
**merged** cut, and each entry carries `t_ms` + `summary` + `**specifics}` but
**NOT `shape`** (line 378-381). A **single-flag** cut returns the flag's
specifics with no moments list at all (line 372-374). So today's persisted
structure is insufficient to synthesize one event per included flag with its own
direction. §3 and §5 fix this by having `resolve.py` hand `build_cut_records` a
first-class per-moment list (peak + shape + summary + specifics), independent of
the `scene_specifics` rendering artifact.

---

## 3. Design: the salience bridge

### 3.1 Where it lives (store-time, authoritative)

Populate `salience`/`landmarks` inside `vcut/store.build_cut_records`
(`store.py:103`), the single place both the initial ingest and the energy
re-resolve funnel through (`insert_video_cuts`, `store.py:138`; called by
`run_vcut_ingest` and by `routers/projects.py:_resolve_energy_preserving`,
`projects.py:113-134`). Because `build_cut_records` runs at EVERY resolve, the
`salience` block is **re-derived per energy** and is automatically
energy-invariant in the exact same way `framing`/`scene_specifics` already are
(see `store.py:110-113` and `reframe_vcut_geometry.plan.md` §3). Add a new pure
helper module `vcut/salience.py` (keep vcut's isolation contract — only import
`v4_segment_params` floors for constants, mirroring how `cutrecord_map.py:33-37`
imports them) so the transform is unit-testable without I/O.

### 3.2 `resolve.py` hands the composing moments to the bridge

`ResolvedCut` (`resolve.py:131-147`) must carry the per-flag data the bridge
needs. Add a first-class field (do NOT reuse the `scene_specifics.moments`
render artifact):

```python
# resolve.py — new field on ResolvedCut
moments: List[Dict[str, Any]] = field(default_factory=list)
# each: {"peak_ms": int (refined), "shape": str ("build"|"settle"|"both"),
#        "summary": str, "specifics": Dict[str, Any]}
```

Populate it in `_resolve_file` (`resolve.py:557-568`) from each group's
`group_peaks` (which already carry `(refined_peak, tag, summary, specifics, box)`
tuples — see `_GroupPeak`, `resolve.py:326`). A single-flag cut yields a
one-element `moments`; a merged cut yields one entry per absorbed flag, in time
order. This is a **superset** of what `_composed_specifics` already computes, so
it can be produced in the same loop with the same inputs. (§5 also persists
`shape` into the `scene_specifics.moments` render list for the resident line, but
the salience bridge reads THIS structured field, not that one.)

### 3.3 Flag → event mapping

For each entry in `cut.moments`, emit one event dict, matching the point-event
contract (`v4_segment.py:694-695`):

```python
RUN_UP_FLOOR_MS = 300         # v4_segment_params.RUN_UP_FLOOR_MS
FOLLOW_THROUGH_FLOOR_MS = 500 # v4_segment_params.FOLLOW_THROUGH_FLOOR_MS

def _event_for(moment, cut_in_ms, cut_out_ms, hop_ms, action_energy, ae_lo, ae_hi):
    peak = int(moment["peak_ms"])
    onset  = max(cut_in_ms,  peak - RUN_UP_FLOOR_MS)
    settle = min(cut_out_ms, peak + FOLLOW_THROUGH_FLOOR_MS)
    return {
        "peak_ms": peak,
        "score": _event_score(peak, hop_ms, action_energy, ae_lo, ae_hi),  # §3.5
        "kind": "point",
        "onset_ms": onset,
        "settle_ms": settle,
        "span_ms": None,
    }
```

Notes:
- `kind` is **always `"point"`** for a vcut event. This is deliberate and
  matches the ladder's arbitration rule (`cutrecord_map.py:336-349`,
  `_video_rung` docstring): `kind` is code-owned and decides point-vs-span-vs-none;
  a present, non-`"span"` `kind` routes to the `shape`-driven asymmetric branch.
  vcut has no camera-move "span" primitive at the flag level, so `"span"`/`"none"`
  are never emitted.
- `onset_ms`/`settle_ms` are display/duration inputs only for point events
  (`piece_breakdown` computes `dur_s = settle-onset`, `footage_map.py:625`);
  `resolve_cluster` recomputes geometry from `peak_ms` + cluster bounds + the
  per-event floors (`cutrecord_map.py:253-261`) and does NOT read them for the
  window. Using the same floors keeps the displayed durations consistent with
  the ladder's own math. Clamp both into `[cut.in_ms, cut.out_ms]`.

### 3.4 `kind`/`shape`/`peak_ms` derivation (representative)

The top-level `shape` (and the single-event directional trim) comes from the
**representative** flag — the one `resolve._representative_peak` already picked as
`cut.peak_ms`/`cut.tag` (`resolve.py:559`). Map vcut's shape vocabulary to the
ladder's:

| vcut `MomentFlag.shape` / `cut.tag` | ladder `salience.shape` | ladder rung | intent |
|---|---|---|---|
| `"build"` | `"before"` | `_before_rung` | keep the run-up, land on the climax |
| `"settle"` | `"after"` | `_after_rung` | cut into the peak, keep the settle |
| `"both"` (default) | `"center"` | `_symmetric_rung` | symmetric around the peak |

This mapping is the crux of directional-trim parity. Confirm against both ends:
vcut `TAG_SPLIT` (`params.py:58-62`: `build` = `(0.7,0.3)` = keep run-up) and the
ladder rung docstrings (`_before_rung`, `cutrecord_map.py:142-146`: "window ends
closer to the impact with a shrinking lead-in"). Both agree: `build`↔`before`,
`settle`↔`after`. Centralize the map as a constant in `vcut/salience.py`
(e.g. `_SHAPE_TO_LADDER = {"build": "before", "settle": "after", "both": "center"}`)
with `.get(shape, "center")` fallback.

Top-level fields:
- `peak_ms` = `cut.peak_ms` (already the representative refined peak).
- `kind` = `"point"`.
- `shape` = `_SHAPE_TO_LADDER.get(cut.tag, "center")`.
- `span_ms` = `None`.
- `primary` = index into `events` of the event whose `peak_ms == cut.peak_ms`
  (the representative). `_representative_peak` guarantees one event matches; if a
  degenerate tie makes it ambiguous, pick the highest-score event and hard-fail
  the assertion in tests rather than silently guessing (§9 — no silent fallbacks).
- `score` = that primary event's own `score`.

### 3.5 Score derivation

Mirror the old `_salience` philosophy (`post.py:356-407`): a score is the peak's
height normalized against the CUT's own curve range, 0..1 — a within-cut relative
magnitude, never a cross-cut absolute. `_prune_events` (`cutrecord_map.py:195`)
and `piece_breakdown` (`footage_map.py:621-633`) both use scores **relative to
the cluster's own max**, so absolute calibration is irrelevant — only the
ordering and the ratios among a cut's own events matter.

Concretely, for each event peak sample the file's clip-normalized `action_energy`
track (already persisted in `seam[file_id]["action_energy"]`, clip-normalized per
`orchestrate.py:64-66`) at the peak's hop bin, then min-max normalize across the
cut's own events so the strongest event scores ~1.0:

```python
raw = [action_energy[min(len(action_energy)-1, peak // hop_ms)] for peak in peaks]
lo, hi = min(raw), max(raw)
score_i = (raw[i] - lo) / (hi - lo) if hi > lo else 1.0
```

- Reuse `resolve._refine_peak_ms`'s bin math for consistency (`resolve.py:191-198`).
- No signal at all (empty/zero `action_energy`) → every event scores `1.0`
  (equal salience), so `_prune_events` keeps all and fragmentation still works
  off geometry — mirrors old `no_signal` behavior (`post.py:365`) without
  fabricating a hierarchy. This is the ONE benign degrade (equal scores), not a
  silent contract violation.

### 3.6 `density`

Informational only (no read-side consumer in `cutrecord_map`/`footage_map` gates
on it; the old pace path that used it is replaced by vcut's pinned `_pace_for`).
Set `density = mean(action_energy over [in,out])` (already computed as `mean_ae`
in `build_cut_records`, `store.py:121`), clamped 0..1, so the field is truthful
if a future reader wants it. Do not invent a novelty-rate.

### 3.7 Assembling the block

```python
def build_salience(cut, hop_ms, action_energy, mean_ae):
    events = [_event_for(mo, cut.in_ms, cut.out_ms, hop_ms, action_energy, ...)
              for mo in cut.moments]           # time-ordered (resolve emits in order)
    _score_events(events, hop_ms, action_energy)   # §3.5, in place
    primary = next(i for i, e in enumerate(events) if e["peak_ms"] == cut.peak_ms)
    prim = events[primary]
    return {
        "peak_ms": prim["peak_ms"], "score": prim["score"], "kind": "point",
        "shape": _SHAPE_TO_LADDER.get(cut.tag, "center"), "span_ms": None,
        "events": events, "primary": primary,
        "density": max(0.0, min(1.0, mean_ae)),
    }
```

Wire into `build_cut_records` (`store.py:126-134`):

```python
records.append(CutRecord(
    ...,
    hero_ts_ms=cut.peak_ms, pace=pace, ...,
    salience=build_salience(cut, hop_ms, action_energy, mean_ae),
    landmarks=build_landmarks(cut),        # §6.2 (act channel only)
))
```

### 3.8 Landmarks (scoped — partial parity)

Old `_landmarks` (`post.py:600-627`) fuses four channels from signals the vcut
seam cache does NOT persist: `adx` needs `rms_db`, `sil` needs
`silence_intervals`, `shot` needs `shot_points`/`composition_points`. The vcut
seam cache holds only `{hop_ms, S, action_energy, frame_diff, src_w, src_h}`
(`orchestrate.py:80-81`). Therefore:

- **`act` IS derivable** — the included moment flags ARE the interior action
  structure. Emit `landmarks["act"] = {"n": K, "cuts": [{"off": peak-in, ...}]}`
  where `K` = count of events whose peak is **interior** to the cut (reuse the
  same interior edge-guard as `_peak_tag`/`post._interior_edge_guard`: `max(100,
  span_ms // 10)`, `footage_map.py:719`, `post.py:432-437`). This drives the
  `sig:actN` breadcrumb (`_landmarks_tag`, `footage_map.py:732`). Match the old
  `act` sub-shape (`post.py:597`: `{"n": int, "cuts": [{"off": int, "hard": bool}]}`)
  — `hard=False` for vcut (no hard/soft distinction at the flag level).
- **`adx`/`sil`/`shot` are NOT derivable** from persisted vcut artifacts →
  **omitted** (channel absent = old-pipeline behavior for a cut with no interior
  structure on that channel, `post.py:614-626`). This is an honest scoped gap
  (§11 open question): a fully static/quiet vcut cut with a single flag lands
  `landmarks == {}`, identical to how the old pipeline treated a structureless
  cut. It does NOT regress anything the old pipeline surfaced for vcut-style
  content, because these channels require signals vcut never persisted.

---

## 4. Cut-info-from-moments + energy consistency

**Requirement:** a cut's rendered info must be the composition of exactly the
moments it currently contains, and must RECOMPOSE as energy regroups/shrinks.
Today `scene_specifics` (with its `moments` list) is frozen at the ingest energy
(written once in `insert_video_cuts`, `store.py:156-158`) and does not track
which events survive `_prune_events` at a given rung.

Because §3 populates `salience.events`, the read side ALREADY recomposes
geometry per rung (`_cluster_rung` → `resolve_cluster` → `_prune_events`,
`cutrecord_map.py:281-301`). To make the DESCRIPTIVE info track it too:

1. **Ride each moment's specifics/summary on its event.** Extend the event dict
   with the moment's own descriptive payload:
   `event["summary"] = moment["summary"]`, `event["specifics"] = moment["specifics"]`.
   These are additive keys the geometry readers ignore (`resolve_cluster`/
   `_prune_events` only read `peak_ms`/`score`), so they are harmless to existing
   consumers but available to rendering.
2. **Derive the rendered `moments` list from surviving events.** Today
   `_render_moments_list` (`footage_map.py:827-850`) reads
   `scene_specifics.moments` (all, frozen). Change `_specific_tag`/
   `_render_new_specifics` (`footage_map.py:853-899`) to prefer, when the moment
   carries `salience.events` with per-event `summary`/`specifics`, the events
   that SURVIVE the sharpest-band prune (`_prune_events(events,
   _BAND_ENERGIES[-1])`, the same survivor set `piece_breakdown` already computes,
   `footage_map.py:613-614`) — so `compose(current moments)` stays true at every
   energy. Fall back to `scene_specifics.moments` for legacy rows with no
   per-event descriptive payload (un-re-ingested runs).
3. **Resident stays compact; full detail is Tier-1.** The resident line keeps the
   compact `spec:"…"` + capped `moments:[…]` (`_render_new_specifics` compact
   mode, `footage_map.py:853-870`); FULL per-moment specifics remain reachable via
   Tier-1 `moment_detail`/`inspect_cut` (`footage_map.py:1494`, `observe.py`).
   No new resident bloat.

This keeps the write side authoritative (specifics still stored on
`scene_specifics` for legacy/Tier-1) while the resident rendering derives the
live moment set from the SAME events the ladder prunes — one source of truth per
energy.

> **Scope note / decision:** the minimal, lowest-risk cut of §4 is steps (1) +
> (2) only (ride specifics on events; render from survivors). If touching
> `_render_new_specifics` is deemed out of scope for the first landing, ship §3
> (geometry parity) alone and file §4 step 2 as an immediate follow-up — the
> geometry recompose is already correct from §3; only the descriptive text lags.
> Recommended: land §4 step 2 together, since it is a small, contained change and
> is the difference between "the pieces regroup" and "the words describing the
> pieces regroup with them."

---

## 5. Per-moment `shape` persistence (additive)

`_composed_specifics` moments (`resolve.py:378-381`) carry `t_ms` + `summary` +
`**specifics` but not `shape`. Add it:

```python
composed["moments"] = [
    {"t_ms": t_ms, "shape": tag, "summary": summary, **dict(specifics or {})}
    for t_ms, tag, summary, specifics, _box in sorted(group_peaks, key=lambda p: p[0])
]
```

(`tag` is already the second element of each `_GroupPeak` tuple, `resolve.py:326`.)
This is a purely additive field — existing readers ignore unknown keys. It
enables full per-moment directional trim (each event's own `shape`, not just the
representative's) as a future refinement, and makes the persisted
`scene_specifics.moments` self-describing.

**Backfill without VLM:** `shape` already lives on the persisted `MomentFlag`
(`FilePlan.to_dict`/`from_dict`, `resolve.py:66-102`) inside `loose_plan`, so the
energy re-resolve path (`projects.py:_resolve_energy_preserving`,
`projects.py:113-134`) rebuilds it deterministically from `MomentPlan.from_dict`
on every energy change — no re-ingest, no model call needed for the shape field
specifically. (A full re-ingest is still recommended to populate `salience` on
all cuts — see §9 — but this notes the field is not VLM-gated.)

---

## 6. Speech cuts: explicit treatment

**Decision: speech cuts need NO salience bridge.** Reasoning, grounded:

- Speech cuts (`vcut/speech/store.build_speech_cut_records`, `speech/store.py:193`)
  are built as `kind="speech"`, `channel="said"`, and leave `salience`/`landmarks`
  at `{}` — exactly like the OLD speech path (old `post.finalize` computed a
  clip-relative `_salience` for speech, `post.py:1164-1174`, but speech NEVER used
  the fragmenting cluster ladder).
- On the read side, `synth_ladder` dispatches `kind == "speech"` to `_speech_rung`
  (`cutrecord_map.py:413-414`), which ignores `salience` entirely and shaves
  interior dead-air via `pace.remove_spans` (`cutrecord_map.py:389-405`). Speech
  "energy" is dead-air/filler trim (`_chosen_remove_spans`,
  `cutrecord_map.py:354`), NOT action-peak fragmentation. `piece_breakdown` is
  video-only (multi-event clusters).
- Therefore the parity concern (dumb symmetric video shrink) does not exist for
  speech: speech was never fragmented by action salience in either pipeline. The
  speech pacing lever (`remove_spans`) is a separate, already-working mechanism.

**Action for speech:** none in this plan, other than a one-line assertion in the
bridge that it is only invoked for `kind="video"` (it already is — `build_cut_records`
is the video path; speech goes through `speech/store.py`). Note explicitly in the
plan doc so a future reader doesn't "fix" speech by adding a spurious salience block.

---

## 7. `guidance_doc.md` trim-policy paragraph

Add a binding paragraph to §3 "Fitting a clip to a target window"
(`backend/app/services/l3/guidance_doc.md:58-66`), immediately after the existing
paragraph (after line 66), matching the doc's second-person, imperative voice:

> When you're trimming a video shot rather than a speech line, aim the trim at
> the shot's key frame. Each shot tells you where its strongest instant is
> (`peak:+Xs`) and which way it leans: a shot that BUILDS wants its run-up kept
> and lands on the impact, so trim from the HEAD; one that SETTLES wants the
> landing kept, so trim from the TAIL; a balanced one tightens from BOTH sides
> toward the peak. Trim toward the key frame that way, then read the length back
> from `read_state`/`review` and adjust — never clip the peak itself, and don't
> compute the result blind.

(Placement: a new paragraph under `## 3`, before `## 4`. Keep the existing
"read the actual length back … that loop is exact, blind arithmetic isn't"
sentence intact — this paragraph specializes it for video shots.)

---

## 8. PARITY CHECKLIST (old brain inputs vs new)

Field-by-field, what the OLD pipeline surfaced to the brain (via
`cutrecord_map._to_cut_dict` + `footage_map._moment_line`) vs. vcut today vs.
after this plan. "✓ after" = closed by this plan.

| Brain-visible field / behavior | Old L3/v4 | vcut TODAY | vcut AFTER (this plan) | Where closed |
|---|---|---|---|---|
| `salience.peak_ms` (+ `peak:+Xs` tag) | ✓ | ✗ (dark) | ✓ | §3.4 / `_peak_tag` |
| `salience.score` | ✓ | ✗ | ✓ | §3.5 |
| `salience.kind` | ✓ (`point`/`span`/`none`) | ✗ | ✓ (`point`) | §3.3 |
| `salience.shape` (directional trim) | ✓ | ✗ (symmetric only) | ✓ (`before`/`after`/`center`) | §3.4 |
| `salience.events[]` (fragmentation) | ✓ | ✗ | ✓ (one per included flag) | §3.3 |
| event `onset_ms`/`settle_ms` | ✓ | ✗ | ✓ (floored windows) | §3.3 |
| `salience.primary` | ✓ | ✗ | ✓ | §3.4 |
| content-aware energy (`_prune_events`) | ✓ | ✗ (dumb shrink) | ✓ | §3 (events populate) |
| `piece_breakdown` / `range:` line | ✓ | ✗ (single event) | ✓ | §3 (multi-event) |
| per-piece addressing (`resolve_piece`) | ✓ | ✗ | ✓ | §3 (events) |
| asymmetric `_before/_after_rung` trim | ✓ | ✗ | ✓ | §3.4 shape map |
| `landmarks.act` (`sig:actN`) | ✓ | ✗ | ✓ | §6.2/§3.8 |
| `landmarks.adx/sil/shot` | ✓ | ✗ | **partial: omitted** (signals not persisted) | §3.8 / §11 |
| per-moment specifics/summaries | frozen | frozen | ✓ recompose per rung | §4 |
| per-moment `shape` persisted | n/a | ✗ | ✓ (additive) | §5 |
| `hero_ts_ms` (best still) | ✓ | ✓ (already) | ✓ | — |
| `scene_specifics` (resident spec tag) | ✓ | ✓ (already) | ✓ (+recompose) | §4 |
| `framing`/`subject_box` | ✓ | ✓ (reframe plan) | ✓ | — |

**Biggest gaps closed:** (1) `salience.events` → fragmentation + piece
addressing + content-aware prune (the single largest dark area); (2)
`shape`-driven asymmetric directional trim; (3) `peak:+Xs` key-frame tag; (4)
per-rung recompose of the described moment set. **Only remaining partial:**
`landmarks.adx/sil/shot` (out of reach without persisting extra L1 signals in the
seam cache — scoped, not silently dropped).

---

## 9. Rollout steps

No silent fallbacks — **hard-fail on malformed flags**, never degrade quietly.

1. **`vcut/salience.py`** (new, pure): `build_salience(cut, hop_ms, action_energy,
   mean_ae)` + `build_landmarks(cut)` + `_SHAPE_TO_LADDER`. Assert every
   `cut.moments` entry has `peak_ms`/`shape` (raise `ValueError` on a malformed
   flag — the ingest should fail loudly, not write a dark cut). Assert exactly one
   event matches `cut.peak_ms` for `primary`.
2. **`resolve.py`**: add `ResolvedCut.moments` (§3.2), populate in `_resolve_file`
   (`resolve.py:557-568`); add `shape` to `_composed_specifics` moments (§5).
3. **`store.build_cut_records`** (`store.py:126-134`): set `salience=`/`landmarks=`
   on each `CutRecord`. (Both flow to DB unchanged via `insert_cut_records`
   columns, `ingest_store.py:178,204`.)
4. **`footage_map`** (§4 step 2): ride `summary`/`specifics` on events; render the
   resident moments list from surviving events (`_render_new_specifics`).
5. **`guidance_doc.md`** (§7): insert the trim-policy paragraph.
6. **Cache/version bumps** (§11): bump `cutrecord_map.CUTRECORD_MAP_VERSION`
   (`cutrecord_map.py:49`, currently `3`) and `footage_map.TREE_VERSION`
   (`footage_map.py:114`, currently `20`) so the `footage_trees` cache and
   synthesized ladders rebuild for every file even where the underlying
   `cut_records` rows are re-ingested (the run-id/row-count signature already
   busts on re-ingest, but bumping the logic versions guarantees a rebuild even
   for any run whose row count is unchanged). There is no separate
   `PARAMS_VERSION`/`LADDER_VERSION` constant in this codebase — those two
   constants ARE the ladder-logic and tree-shape version levers.
7. **Re-ingest all projects** (confirmed acceptable). `salience` is populated only
   at store time, so existing cut records stay dark until re-run. Re-ingest
   authoritatively populates `salience`/`landmarks` on every video cut. (The
   energy re-resolve path also repopulates on any dial change, but a full
   re-ingest is the clean backfill.)
8. **Verification** (per §10 + live): after re-ingest of one project, eyeball the
   Tier-0 map text for a multi-flag cut — confirm the `range:` line renders, `sig:actN`
   appears, `peak:+Xs` appears, and dragging the energy dial regroups pieces AND
   the described moments.

### 9.1 Read-layer alternative (not recommended, documented)

Instead of store-time, `cutrecord_map._to_cut_dict` could synthesize `salience`
on the fly from the persisted `scene_specifics.moments` (no re-ingest). Rejected
because: (a) it duplicates scoring logic into the read path with no access to the
seam `action_energy` curves (scores would be geometry-only), (b) it couples the
brain read layer to the `scene_specifics` render shape, and (c) re-ingest is
already acceptable, so the authoritative store-time write is simpler and correct.
Keep as the documented fallback ONLY if a no-re-ingest constraint ever appears.

---

## 10. Test plan (all mocked; zero spend)

New `backend/tests/test_vcut_salience.py` (pure unit, no DB, no model):

1. **Bridge emits correct events from flags.** Given a `ResolvedCut` with 3
   moments at known peaks/shapes + a synthetic `action_energy` track, assert
   `salience.events` has 3 point events with correct `peak_ms`, monotone
   `onset_ms`/`settle_ms` (floored + clamped), and `score` normalized so the
   strongest is `1.0`. Assert `primary` indexes the representative.
2. **Shape mapping.** `build`→`before`, `settle`→`after`, `both`/unknown→`center`
   at top level; assert `_video_rung` (imported from `cutrecord_map`) then routes
   a single-event cut to `_before_rung`/`_after_rung`/`_symmetric_rung`
   accordingly (feed the synthesized `salience` straight into a fake row dict).
3. **Ladder fragments a multi-flag cut.** Build a row from a 4-flag cut's
   synthesized `salience`, call `cutrecord_map.synth_ladder`; assert the `sharp`
   rung has multiple `spans` (fragmented) and the `broad` rung is one whole span.
4. **Directional trim per shape.** For a single-event `before` cut, assert the
   tightened `sharp` rung's OUT edge stays near the peak (head-trimmed); for
   `after`, the IN edge stays near the peak (tail-trimmed) — assert against
   `_before_rung`/`_after_rung` outputs.
5. **Specifics recompose per rung.** Give events per-event `summary`; assert the
   resident moments list (via the updated `_render_new_specifics`) drops
   weak/pruned events at the sharpest band and keeps them at broad — i.e.
   `compose(survivors)` differs by rung.
6. **Parity of rendered beat line.** Golden-string test: feed a synthesized
   vcut `salience` and an equivalent old-pipeline `salience` (same peaks/scores)
   through `footage_map._moment_line` and assert the `range:`/`peak:`/`sig:`
   tags render identically.
7. **`landmarks.act`.** Assert `sig:actN` count = interior event count (edge-guard
   respected); a single-flag or edge-pinned cut → `landmarks == {}` → no `sig:` tag.
8. **Hard-fail.** A malformed moment (missing `peak_ms`) raises `ValueError`;
   assert the bridge does NOT emit a dark/partial cut.
9. **Energy-invariance / re-resolve.** Resolve the same `MomentPlan` at energy 0
   and 1; assert `salience` is regenerated each time (per-energy) and that a
   single moment's event peak is stable across energies.
10. **Speech untouched.** Assert `build_speech_cut_records` output still has
    `salience == {}` and `synth_ladder` routes it to `_speech_rung` (no regression).

All fixtures are in-memory dicts/dataclasses; the model is never called; the DB
is never touched (test the pure bridge + the pure `cutrecord_map`/`footage_map`
functions directly). Add a `pyflakes` clean check to the rollout.

---

## 11. Risks / open questions

- **Landmarks derivability (partial parity).** `adx`/`sil`/`shot` channels are
  NOT derivable from the persisted vcut seam cache (`{hop_ms,S,action_energy,
  frame_diff,src_w,src_h}`). Only `act` is emitted (§3.8). Open question: is
  `act`-only landmark parity acceptable, or should the seam cache be extended to
  persist `rms_db`/`silence_intervals`/`shot_points` so the other three channels
  can be filled? Recommendation: ship `act`-only now (it covers the
  action-fragmentation story the brain most needs); treat richer landmarks as a
  separate follow-up that widens `orchestrate._seam_cache` and re-ingests.
- **Single-flag cuts.** A cut with exactly one flag emits one event → `len(events)
  == 1` → `_video_rung` single-window path (`cutrecord_map.py:324`), i.e. NO
  `piece_breakdown` (correct — nothing to fragment) but STILL gets the
  `shape`-driven directional trim and `peak:+Xs`. Verify this is the desired
  behavior (it matches the old pipeline's cluster-of-one, `v4_segment.py:66-70`).
- **Score normalization.** Cut-relative min-max means a monotone/flat cut yields
  near-equal scores → `_prune_events` keeps most events (broad). This is
  intended (mirrors old `no_signal`/relative behavior) but means a genuinely
  low-energy cut won't over-prune. Acceptable; flagged so it isn't mistaken for a
  bug.
- **`primary` disambiguation.** If two events share `peak_ms == cut.peak_ms`
  (degenerate refine collision), pick the higher-score one and assert-fail in
  tests; never silently pick index 0.
- **Cache/signature bumps.** MUST bump `CUTRECORD_MAP_VERSION`
  (`cutrecord_map.py:49`) and `TREE_VERSION` (`footage_map.py:114`) so
  `footage_trees` and synthesized ladders rebuild. The content signature
  (`signatures_for`, `cutrecord_map.py:82-107`) already embeds
  `CUTRECORD_MAP_VERSION` and re-ingest changes the run id/row count, but the
  explicit bump guarantees no stale cached tree survives. (Prompt's
  `PARAMS_VERSION`/`LADDER_VERSION` correspond to these two constants; there is
  no third version constant.)
- **`_composed_specifics` vs `ResolvedCut.moments` duplication.** Two lists now
  carry per-moment data (the render artifact + the structured field). Keep them
  produced in the same loop from the same `group_peaks` so they can't drift; a
  test asserts they agree on peaks/shape.

---

## 12. Out of scope (explicit)

- **`fit_to_window` verb — DEFERRED/DROPPED for now.** The trim policy (§7) + the
  shape-aware ladder (§3.4) + the existing `trim`/`retime`/`review` verbs cover
  correctness. A dedicated one-shot "fit this clip to `[A,B]`" verb is only a
  round-trip optimization (saves a read-back-and-adjust loop), not a
  correctness requirement. Explicit future/optional, NOT in this plan's scope.
- **Brain-verb changes.** None required — the keystone is that the EXISTING read
  machinery lights up from populated `salience`/`landmarks`. No new brain tool,
  no arrange/act change.
- **Richer landmarks (`adx`/`sil`/`shot`).** Requires persisting extra L1 signals
  in the vcut seam cache; separate follow-up (§11).
- **Per-moment (non-representative) directional trim.** The `shape` field is
  persisted per moment (§5) so it's POSSIBLE, but the ladder's cluster path
  deliberately does not thread per-event shape (`resolve_cluster` docstring,
  `cutrecord_map.py:229-235`); wiring full per-event asymmetry is a future
  refinement, not parity.
- **Speech salience.** N/a by design (§6).
