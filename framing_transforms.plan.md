# Universal framing & transforms — phased plan

Per-layer geometry — **rotation, reframe (crop/anchor), zoom, and (later)
split-screen / PiP** — so the same cut delivers any aspect and frames content
correctly. Framing is **automatic**: Opus picks intent in words, perception (VLM +
L1) supplies facts, a deterministic **solver** computes the pixels. The **only**
thing we ask the user is the **zoom** option.

> This plan builds **Phases 1–3** (rotate / reframe / zoom). **Split-screen / PiP /
> multi-up is NOT being implemented** — Phase 4 only *reserves the `dest`
> rectangle* as a seam so it's possible later; the real work is deferred with
> multicam-stack.

## Decision model (who decides the numbers)

| Layer | Decides | How |
|---|---|---|
| **Opus (L3)** | intent in words (`shot:medium`, `follow`, rotate override) | tool args, **categorical** never pixels |
| **Perception (VLM + L1)** | facts: focus region, motion centroid, source dims, orientation | L2 coarse region, `motion_dynamics`, metadata |
| **Solver (code)** | the exact crop/rotate/scale/dest rectangle | geometry from intent + facts; clamped legal |
| **Renderers** | pixels | ffmpeg + CSS read the SAME final transform |
| **User** | **zoom only** | one `ask_user` fork (see table below) |

## The transform (one struct, two stages: frame source → place into dest)

| Field | Values | Resolved by |
|---|---|---|
| `rotate` | 0 / 90 / 180 / 270 | auto from metadata; Opus may override; applied **first** |
| `fit` | cover / contain / stretch | solver (default cover on aspect mismatch) |
| `crop` | rect from focus region | solver (focus-anchored, center fallback) |
| `zoom` | static value or a path | **user-chosen option** + solver |
| `dest` | full / {x,y,w,h} | **always `full` for now** — sub-rect is a reserved seam, not built |

Fixed op order (identical in ffmpeg + CSS preview): `rotate → crop/zoom/fit →
place into dest → composite by z`. Identity transform == today (back-compat).

## Phases

| Phase | Goal | Backend | New perception signal | Frontend | Tests |
|---|---|---|---|---|---|
| **1 — Substrate + solver** ✅ DONE | rotate-to-upright + centered cover-crop to any aspect, fully automatic (existing signals) | `transform` on `VideoLayer`; transform-aware `compositor` filter graph (`transpose`/`scale`/`crop`/`pad`); `layers.resolve` carries it; solver v1 uses `focus.py` target + source dims + orientation metadata | none new (reuse `focus.py`, metadata) | `transform` on `ResolvedVideoLayer`; CSS `rotate/scale/translate`+`object-fit`; resolve parity | resolve parity fixture; compositor smoke (rotated+cropped seg → canvas dims) |
| **2 — Perception-grounded** ✅ DONE (new uploads) | framing locks onto subject/action, not center | `l3/framing.py` annotates each segment with a `transform.focus` (speaking→event→motion→person) baked at author-time + persisted; compositor focus-centered crop; `read_framing` facts tool | **L2 `Region`** (normalized box) on speaking spans / events / persons + **`frame_orientation`** (schema v2); **L1 motion centroid** on action_points | preview `object-position` from focus | `test_framing.py` Phase-2 cases |
| **3 — Motion + zoom option** ✅ DONE | push-in / follow paths; **the one user question** | `transform.motion` from→to (scale+focus) eased over `dur_ms`; shared `sample_motion`; compositor `zoompan` over a cover base (works any aspect); framing builds motion per shot from `format.motion_style/_feel` with dwell/hysteresis; `set_brief` motion_style+motion_feel; orchestrator prompt asks the single zoom question | reuses Phase-2 focus (sampled at shot ends for follow) | `sampleMotion` mirror; preview shows the path **midpoint** (representative; render animates) | `test_framing.py` Phase-3 cases + zoompan pan smoke |
| **4 — Layout (DEFERRED, seam only)** | reserve the `dest` rect so split / PiP / multi-up is *possible* later — **not implemented now** (same primitive as deferred multicam-stack) | keep `dest` field in the transform spec (always `full`); no split/PiP verbs, no overlay-rect logic built | none new | none | none |

_Phase 2 note: the spatial signal (`Region` + `frame_orientation` + motion
centroid) is produced only by the NEW L2/L1 passes, so it lights up on
**newly-uploaded / re-processed clips**. Existing clips have no regions and stay
centered (Phase-1 behavior) until backfilled. The author-time `framing.annotate_document`
pass reads perception directly (not via `focus.py`), keeping `resolve` pure._

## Zoom — the only user question (Phase 3)

| Decision | `ask_user` options |
|---|---|
| Zoom style | static (locked) / punch-in (held tighter) / slow push-in / push + settle |
| Movement feel | snappy cut-to-tight / smooth glide |

Everything else (orientation, fit, anchor, when/where to crop) is solver-decided.

_Phase 3 note: motion is a from→to eased move (push_in / pull / follow), the
common case; richer multi-keyframe paths are a future extension (the data model
+ `zoompan` builder handle the two endpoints exactly). The preview shows a stable
**representative (midpoint) frame** for a moving shot rather than animating
frame-by-frame; the render evaluates the full eased path and is authoritative.
`follow` reuses the Phase-2 focus sampled at the shot's start/end and holds still
inside a dwell deadzone._

## Locked decisions (from the design brainstorm)

| # | Decision |
|---|---|
| 1 | Opus emits **categories/intent**, never pixels; code computes numbers. |
| 2 | Rotation is **orthogonal-only** (0/90/180/270), exact via `transpose`. |
| 3 | Orientation resolved by **detection** (metadata → one-per-clip flag), rotate-to-upright happens **first** — not inferred from focus. |
| 4 | Zoom = the **scale axis** of the crop; clamped between "tightest the source can still fill" and full frame (no upscaling past source res). |
| 5 | Framing anchor is the **general focus** (speaker/action/object), not "person". |
| 6 | **VLM-first** spatial grounding; **escalate to a tracker only** for tight-follow — never a standing per-clip cost. |
| 7 | **Framing is automatic** (perception + solver). The **only** `ask_user` is the **zoom** option. |
| 8 | Values are **sparse** (one transform / path per segment); continuity synthesized by the solver, never authored per-frame. |

## Parity contract (every phase)

`preview == render`. One spec: normalized 0..1 canvas coords; rotation degrees CW;
anchor = center; fixed op order above; animated zoom = piecewise-linear over
program time. A resolve fixture diffs backend `to_dict()` vs `resolve-timeline.ts`.

## Risks

| Risk | Mitigation |
|---|---|
| ffmpeg ↔ CSS parity (anchor, fill, rounding, paths) | single spec + parity fixture |
| Fast-path bypass | transforms force the filter graph; plain cuts stay on concat |
| Rotation × aspect (90° swaps w/h before fit) | explicit op order |
| Zoom-in quality bounded by source res | clamp, no upscale past source |
| Phase 2 spatial signal is net-new (L2 has none today) | load-bearing dependency; build region + centroid first |
