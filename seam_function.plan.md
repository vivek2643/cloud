# Seam Function — per-frame seam-quality curve `S(t)` for the NEW cuts pipeline

Status: FINAL PLAN — ready to execute (from another chat). No L1 changes. The old
cuts pipeline (`backend/app/services/l3/v4_segment.py` and friends) is left
completely untouched.

**v1 scope (DECIDED): all SIX signals ship in v1** — including the two that require a
proxy recompute: `frame_diff` and **non-speech onsets** (librosa onset detection on
proxy audio, speech spans excluded via `transcript.segments`). Onsets are **NOT
deferred**. `S(t)` ships complete on day one: gates (`g_sharp`, `g_gest`) + attractors
(`still`, and `audio` = musical beats **+** recomputed non-speech onsets). The only
"optional" item in v1 is the `logs/seam/*.json` cache (a convenience, not a signal).

---

## 1. Motivation, scope, non-goals

### Motivation
The NEW cuts pipeline separates *what to keep* (semantics, decided by a VLM) from
*where to physically land a cut* (frame cleanliness). This document specs the
**seam function**: for non-speech visual footage it emits a per-frame
**seam-quality curve** `S(t)` = "how clean would a cut be at this frame?".

`S(t)` is a **quality field only**. It never decides whether or where a cut exists.
Later, the VLM picks approximate moments and *snaps* each to a local maximum of
`S(t)`. That snap/selection step is explicitly out of scope here.

### In scope
- Define `S(t)` precisely, grounded in existing L1 fields.
- Determine, per input signal, how the new pipeline obtains it **after** L1 without
  changing L1 (persisted vs. recompute-from-proxy).
- Propose a new, fully separate module namespace + driver script + validation hooks.

### Explicit non-goals (the whole point — prior attempts failed by violating these)
- **The seam layer NEVER decides whether/where a cut exists.** No signal may create
  a cut. It only scores frame cleanliness. There is no thresholding, no peak
  *selection*, no segmentation in this layer.
- **No L1 change.** All normalizations and any missing signals are computed
  **post-hoc**, after L1, from L1 outputs or from the R2 proxy.
- **Do not touch the old pipeline** (`l3/v4_segment.py` etc.). It will be deleted
  later; not now.
- **No VLM here.** Semantics/keep-drop is a separate stage.
- **Speech reuses the OLD speech-cut logic.** `S(t)` is for non-speech visual
  footage; speech cutting is a separate channel, not built here.
- No cut locations are produced. No proximity-to-VLM-time term (that is applied at
  snap time, later).

### Signals that are DELIBERATELY EXCLUDED (and why)
These exist in L1/snapshot but must **not** feed `S(t)`, because each one *decides
or implies a cut* rather than scoring frame cleanliness:
- `scene_cuts.composition_points` (appearance/composition drift) — a *boundary*
  detector; asserts "a change happened here", i.e. a cut hint. Not a seam quality.
- `scene_cuts.shot_points` (hard-cut / shot splits) — shot-splitting is a separate
  **upstream** stage; a hard cut is not a seam signal.
- Any CPD / boundary model (`backend/scripts/cpd/**`, `cutviz` CPD overlay) — a
  boundary predictor = a cut decider. Excluded wholesale.
- `motion_dynamics.action_points` / `transition_points` (action impacts, occlusion
  wipes, degenerate spans) and any "action run/lull" logic — these are
  *cut-ON-this-instant* channels (hit channels), i.e. they propose cut locations.
- `action_cut_cost` / `camera_cut_cost` — the OLD derived cut-cost channels. Not
  reused; `S(t)` is rebuilt from raw-ish signals with the spec below.

`action_energy` **is** used, but only as a *gate* (mid-gesture attenuation), never
as an attractor — see §2.

---

## 2. The exact `S(t)` spec (do not redesign)

```
S(t) = g_sharp(t) · g_gest(t) · [ w_vis · still(t) + w_aud · audio(t) ]
```

All `_n` terms below are **clip-relative percentile-normalized to [0,1]**, computed
post-L1 (see §4). Every term names its source field, direction, and normalization.

### Gates (multiplicative; can shrink a frame toward 0)

- **`g_sharp(t) = 1 − blur_n(t)`** — STRONG gate. A motion-blurred landing frame is
  ~disqualified. `blur_n` direction: 1 = fully blurred/distorted, 0 = sharp.
  Source: `motion_dynamics.blur` (already 1=blurred, clip-normalized in L1 against
  the clip's own sharp-frame percentile).
- **`g_gest(t) = 1 − 0.6 · act_n(t)`** — SOFT gate. Mid-gesture attenuates toward a
  ~0.4 floor, never to 0. `act_n` = subject-motion energy (residual flow after the
  camera model is removed). Source: `motion_dynamics.action_energy`.
  Rationale for `0.6`: caps the worst-case gesture attenuation at `1 − 0.6·1 = 0.4`.

### Attractors (additive; pull toward a good frame)

- **`still(t) = 1 − max(cam_n(t), fd_n(t))`** — visual stillness. Requires the
  camera-vector magnitude **and** frame-diff to BOTH be quiet. They are
  complementary: `frame_diff` misses slow drift on low-texture/aerial footage;
  the fitted **camera vector** catches it. Taking `max` means either one being
  active kills stillness.
  - `cam_n` = normalized camera-vector magnitude (from `camera_dx/dy/zoom`, see §3).
  - `fd_n` = normalized `frame_diff` (model-free mean |gray delta| per hop).
  - **Deliberately EXCLUDES `action_energy`** so subject motion isn't
    double-counted — it is already the `g_gest` gate.
- **`audio(t) = Σ_k strength_k · exp(−Δt_k² / (2σ²))`**, `σ ≈ 60 ms`, summed over
  **musical beats + non-speech onsets**; `audio(t) = 0` when there are no audio
  events. `Δt_k = t − event_time_k`. `strength_k ∈ [0,1]` (beats → beat/pulse
  strength; onsets → normalized onset strength). This is a soft "landing on a
  transient sounds clean" attractor evaluated continuously on the grid.

### Weights (STARTING PRIORS — to be harness-tuned later; not final)

- `w_vis = 1.0`
- `w_aud = 1.2 · salience`, where
  `salience = (is_musical ? 1.0 : scaled_onset_strength)`.
  Audio only outranks stillness **when present and salient** → adaptive, with **no
  footage-type branch**. `scaled_onset_strength` = the clip's normalized onset
  strength near `t` (0 when silent), so quiet ambient audio can't dominate.

These weights and `σ`, the `0.6` gesture coefficient, and the salience scaling are
**tunable constants**, isolated in `params.py` (§5), and left for the harness.

### Priority summary (state explicitly)
1. **Sharpness gate** (strong; can disqualify a frame).
2. **Mid-gesture soft gate** (attenuates to a ~0.4 floor).
3. **Audio attractor when salient** (beats/onsets, adaptive via `salience`).
4. **Visual stillness attractor** (camera + frame-diff both quiet).
5. *(Proximity to a VLM-chosen time — applied LATER at snap time, NOT in `S(t)`.)*

The curve is standalone: no proximity term, no cut decision.

---

## 3. Signal-acquisition table (per signal → L1 source → persisted vs recompute → rate)

Persistence was verified against `build_l1_snapshot` in
`backend/app/services/l1/snapshot.py` (the `motion_dynamics` and `audio_features`
SELECTs) and the `motion_dynamics.to_dict()` dataclass.

| Signal (S(t) term) | L1 source field | Persisted & retrievable? | If not → acquisition | Sample rate / grid |
|---|---|---|---|---|
| `blur` (→ `g_sharp`) | `motion_dynamics.blur` | **YES** — selected by snapshot; direction 1=blurred, clip-normalized | — | motion hop `hop_ms` (≈100 ms, `MOTION_FPS`=10) |
| `action_energy` (→ `g_gest`) | `motion_dynamics.action_energy` | **YES** — selected by snapshot (already pctl-normalized) | — | motion hop |
| camera vector `dx,dy,zoom` (→ `cam_n` in `still`) | `motion_dynamics.camera_dx/dy/zoom` | **YES** — all three selected by snapshot (signed, un-normalized) | — (magnitude + renorm computed post-hoc, §4) | motion hop |
| `frame_diff` (→ `fd_n` in `still`) | `motion_dynamics.frame_diff` | **NO** — present in the dataclass/`to_dict` but **NOT in the DB schema / snapshot SELECT**; a recent field not yet deployed to the L1 schema/RunPod | **RECOMPUTE from proxy** via `compute_motion_dynamics(proxy)` — exactly as `routers/cutviz.py` already does ("Recompute motion on the proxy so the NEW signals (frame_diff, raw magnitudes) are present even before the L1 schema/RunPod update") | motion hop |
| musical `beats` (→ `audio`) | `audio_features.onsets_ms` | **YES** — selected by snapshot — BUT **only populated when `is_musical`** (the field actually stores beat-track times, not onsets) | if not musical → no beats (expected) | event timestamps (ms) |
| non-speech `onsets` (→ `audio`) | *(none)* | **NO** — there is **no** separate onset detection anywhere; `onsets_ms` = beats. Non-speech onsets are not computed or stored | **RECOMPUTE from proxy audio**: extract WAV from the video proxy (ffmpeg), run `librosa.onset.onset_detect` + `onset_strength`, then **exclude speech spans** using persisted `transcript.segments` (and/or `audio_features.silence_intervals`) | event timestamps (ms) + per-onset strength |

Notes on the two recompute cases:
- **`frame_diff`**: `compute_motion_dynamics` already returns `frame_diff` (and the
  raw magnitudes) — we simply re-run it on the downloaded proxy and read that field,
  mirroring `cutviz.signal_data`. This *also* gives us fresh `blur`, `action_energy`
  and `camera_dx/dy/zoom` on the same grid, so when we recompute for `frame_diff`
  we get the whole motion bundle consistently (avoids grid-mismatch between
  persisted vs. recomputed motion — see §6 policy).
- **non-speech onsets**: audio is embedded in the video proxy (`r2_proxy_key`);
  extract a mono 16 kHz WAV with ffmpeg (same pattern as
  `l1/audio_features._compute_prosody` / `active_speaker`). Speech exclusion uses
  `transcript.segments` (start/end ms) — drop any onset falling inside a speech
  segment. For non-`is_musical` clips with no transcript, keep all onsets.

### Common time grid
- **Motion signals** live on `hop_ms` (persisted per-file in
  `motion_dynamics.hop_ms`; ≈100 ms / 10 fps).
- **Audio events** are timestamps (ms), not a dense grid.
- `S(t)` is evaluated on the **motion grid** `t_i = i · hop_ms`,
  `i = 0..N−1` with `N = len(action_energy)` (or `duration_ms // hop_ms + 1`).
- `audio(t_i)` is the continuous kernel sum (§2) evaluated at each grid time — event
  timestamps are used directly inside the Gaussian, no resampling of events onto the
  grid is needed. Motion terms (`blur`, `action_energy`, `frame_diff`, camera
  magnitude) are already on the grid.

---

## 4. Clip-relative normalization spec (post-L1, no L1 change)

All `_n` use the **existing** `normalize_pctl` / `percentile` helpers in
`backend/app/services/l1/cut_grid_common.py` (import-only; no modification):

- `percentile(values, p)` = linear-interpolated p-th percentile.
- `normalize_pctl(values, p)` = divide by the p-th percentile, clamp to `[0,1]`
  (percentile, not max, so a few spikes don't flatten everything). Reuse
  `MOTION_NORM_PCTL` from `l1/cut_grid_params.py` as the default `p`.

Per term:
- **`blur_n`** = persisted `motion_dynamics.blur` **as-is** (already clip-normalized
  in L1 against the clip's own sharp-frame percentile, direction 1=blurred). No
  re-normalization needed; optionally re-`normalize_pctl` if recomputed from proxy.
- **`act_n`** = persisted `action_energy` as-is (already pctl-normalized in L1).
- **`cam_n`** = post-hoc: build per-hop camera-vector magnitude
  `mag_i = sqrt(dx_i² + dy_i² + zoom_i²)` from persisted signed
  `camera_dx/dy/zoom`, then `normalize_pctl(mag, MOTION_NORM_PCTL)`. (Rationale:
  gives a true clip-relative renorm of the *vector*; fallback = persisted
  `camera_motion`, which is already pctl-normalized fitted displacement.)
- **`fd_n`** = `normalize_pctl(frame_diff_raw_or_norm, MOTION_NORM_PCTL)` where
  `frame_diff` comes from the recompute (§3). If the recompute returns the already
  file-normalized `frame_diff`, use it directly; else normalize the raw.
- **audio `strength_k`**: beats → normalize pulse/beat strength to `[0,1]` across the
  clip; onsets → `normalize_pctl(onset_strengths)`. `scaled_onset_strength` (for
  `salience`) = the same normalized onset strength sampled near `t`.

"Clip-relative" = each curve is normalized against **its own clip's** distribution,
computed at seam-compute time (post-L1), never against a global/dataset statistic.

---

## 5. Module / file layout (NEW namespace, fully separate)

New directory: **`backend/app/services/seam/`** (parallel to `l1/`, `l3/`; the old
pipeline in `l3/v4_segment.py` is never imported or touched).

```
backend/app/services/seam/
  __init__.py            # exports build_seam_signals, compute_seam_curve, SeamCurve
  params.py              # tunable constants (STARTING PRIORS): W_VIS, W_AUD_BASE=1.2,
                         # GEST_COEFF=0.6, AUDIO_SIGMA_MS=60, salience scaling, pctl
  signals.py             # acquisition: persisted L1 (snapshot) + recompute fallbacks
  curve.py               # the S(t) math (gates × [w_vis·still + w_aud·audio])
```

### Data structures & function signatures

```python
# signals.py
@dataclass
class SeamSignals:
    file_id: str
    hop_ms: int                 # common motion grid step
    n: int                      # number of grid points
    blur_n: list[float]         # 1 = blurred (g_sharp source)
    act_n: list[float]          # subject motion (g_gest source)
    cam_n: list[float]          # camera-vector magnitude, pctl-normalized
    fd_n: list[float]           # frame_diff, pctl-normalized
    beats_ms: list[int]         # musical beats (empty if not musical)
    onsets_ms: list[int]        # non-speech onsets (recomputed, speech-excluded)
    onset_strength: list[float] # per-onset strength in [0,1] (aligns w/ onsets_ms)
    is_musical: bool
    meta: dict                  # provenance per signal: "persisted" | "recomputed"

def build_seam_signals(
    file_id: str,
    *,
    proxy_path: str | None = None,   # if given, skip download; else fetch from R2
    force_recompute_motion: bool = True,  # see §6 grid policy
) -> SeamSignals:
    """Assemble all six inputs on a common grid, reading persisted L1 where
    available (build_l1_snapshot) and recomputing frame_diff / non-speech onsets
    from the proxy where not (mirrors routers/cutviz.py). No L1 writes."""

# curve.py
@dataclass
class SeamCurve:
    hop_ms: int
    t_ms: list[int]              # grid timestamps
    S: list[float]               # the seam-quality curve, per grid point
    # per-term contributions (debug / validation overlay):
    g_sharp: list[float]
    g_gest: list[float]
    still: list[float]
    audio: list[float]
    w_aud: list[float]           # 1.2 · salience(t) (varies over time)
    meta: dict                   # weights used, sigma, provenance

def compute_seam_curve(signals: SeamSignals) -> SeamCurve:
    """Evaluate S(t) = g_sharp·g_gest·(w_vis·still + w_aud·audio) on the grid.
    Pure function of SeamSignals; no I/O, no cut decisions, no thresholding."""
```

`compute_seam_curve` is a **pure** function (numpy/list math only) so it is unit-
testable with synthetic `SeamSignals`, exactly like `_detect_structure` in
`audio_features.py`.

---

## 6. The "FOR ALL CLIPS" driver

New script: **`backend/scripts/seam/compute_all_seam.py`**.

Responsibilities:
- Enumerate every video file (`files.file_type = 'video'` with L1 done), reusing the
  same `_pg`/snapshot access patterns as `cutviz` and other `backend/scripts/*`.
- For each file: `build_seam_signals(file_id)` → `compute_seam_curve(signals)`.
- Read persisted L1 where available; recompute `frame_diff`/onsets from the proxy
  where needed (§3).
- **Parallelism**: a `ProcessPoolExecutor` over files (the work is
  ffmpeg/opencv/librosa-bound per clip). Bound concurrency with the existing
  `app.services.limits.ffmpeg_slot()` so proxy decodes don't oversubscribe, and
  cap pool size via an env var (default = CPU count). One clip per worker; failures
  are best-effort (log + skip), never fatal — mirrors motion_dynamics' non-fatal
  contract.

### Persist vs. compute-on-demand
- **v1: compute on demand + optional JSON cache**, NOT a DB table. This keeps the
  task free of any migration or schema change (a hard non-goal here). Optionally
  write `logs/seam/<file_id>.json` (mirroring `logs/l1/<file_id>.json`) as a cache
  for the validation UI.
- **Grid policy**: because `frame_diff` is not persisted, recomputing motion from
  the proxy is required anyway for at least one signal; to avoid mixing a persisted
  motion grid with a recomputed one, the driver **recomputes the full motion bundle
  from the proxy** (`force_recompute_motion=True`) so `blur/action/camera/frame_diff`
  are all on one consistent grid. Persisted motion is used only as a fast path once
  `frame_diff` lands in the L1 schema (future).
- A future option (explicitly deferred, needs its own plan + migration): a
  `seam_curves` table keyed by `file_id`, written by an additive L1 stage. Not part
  of this task.

---

## 7. Validation / visualization (extend `cutviz`, don't fork it)

Reuse the existing debug tool `backend/app/routers/cutviz.py`, which already
downloads the proxy, recomputes motion, builds the snapshot, and renders signal
tracks over the playable proxy.

Additive changes (no behavior change to existing overlays):
1. In `signal_data()` (`GET /api/debug/cutviz/data/{file_id}`), after motion is
   recomputed and the snapshot built, call `build_seam_signals` (passing the
   already-downloaded `local` proxy path to avoid a second download) and
   `compute_seam_curve`, then add a `"seam"` block to the JSON:
   `{ hop_ms, S, g_sharp, g_gest, still, audio, w_aud }` plus the recomputed
   `beats_ms` / `onsets_ms` used.
2. In the page JS, add a bold **SEAM `S(t)`** track and thin component tracks
   (`g_sharp`, `g_gest`, `still`, `audio`) with a legend, and event ticks for the
   audio beats/onsets that feed `audio(t)`.
3. **Eyeball test**: scrub the proxy and confirm `S(t)` **peaks land on genuinely
   clean frames** — sharp, camera settled, subject between gestures, ideally on a
   beat/onset — and that `S(t)` is **suppressed** during blur, whip-pans, and
   mid-gesture. The per-term overlays make it obvious *which* term drove a peak or a
   dip (e.g. a dip that is all `g_sharp` = blur veto).

This deliberately does **not** draw any "cut" from `S(t)` — no threshold shading —
because the seam layer makes no cut decisions. (The existing "Cut Score"/threshold
knobs and CPD/segmenter overlays stay as-is for the OLD pipeline; `S(t)` is an
independent overlay.)

---

## 8. Sequencing (small, testable steps)

1. **`params.py`** — drop in the starting-prior constants. (Trivial, reviewable.)
2. **`curve.py` + unit tests** — implement `compute_seam_curve` as a pure function;
   test with synthetic `SeamSignals` (e.g. a blurred hop → `S≈0`; a still + on-beat
   hop → high `S`; mid-gesture → attenuated but non-zero). No I/O.
3. **`signals.py` (persisted path)** — assemble `SeamSignals` from
   `build_l1_snapshot` for the fields that ARE persisted
   (`blur/action_energy/camera_*`, `onsets_ms` when musical); stub the recompute
   fields.
4. **`signals.py` (recompute path)** — add proxy recompute for `frame_diff` (via
   `compute_motion_dynamics`) and non-speech onsets (ffmpeg WAV + librosa +
   speech-span exclusion). Fill `meta` provenance.
5. **`cutviz` overlay** — wire the `"seam"` block + tracks; eyeball on real clips.
6. **Driver** `scripts/seam/compute_all_seam.py` — batch over all clips with the
   process pool; optional `logs/seam/*.json` cache.

**Where this later plugs into the VLM snap step:** once the VLM proposes an
approximate keep-moment time `t*`, a *separate* snap step searches a local window of
`S(t)` around `t*` (optionally multiplying in a proximity kernel there, not here) and
lands the physical cut on the local maximum — the seam curve is that step's input.

---

## 9. Honest risks / feasibility flags (found during investigation)

- **`frame_diff` is NOT persisted.** It exists in `MotionDynamics`/`to_dict()` but is
  absent from the `motion_dynamics` DB schema and the snapshot SELECT; `cutviz`
  explicitly recomputes it from the proxy ("before the L1 schema/RunPod update").
  → `S(t)` **cannot** be built for existing clips from the snapshot alone; a proxy
  recompute is mandatory today. Cost: one optical-flow pass per clip (bounded by
  `ffmpeg_slot()`), same as `cutviz`.
- **Non-speech onsets do NOT exist as a stored signal.** `audio_features.onsets_ms`
  stores **beat-track times**, and only when `is_musical`. There is no separate
  onset detector anywhere in L1. → non-speech onsets must be **recomputed** from the
  proxy audio (librosa `onset_detect`/`onset_strength`) with speech spans excluded
  via `transcript.segments`. Risks: (a) onset/speech overlap — onsets during speech
  are dropped, so busy dialogue reduces the audio attractor (acceptable: speech is a
  separate channel); (b) recompute adds an audio decode + librosa pass per clip.
- **Beats only exist for musical clips.** For non-musical footage `beats_ms` is
  empty and `audio(t)` relies solely on recomputed onsets; if the clip is also
  near-silent, `audio(t)=0` and `S(t)` reduces to the visual stillness term — which
  is the intended, correct behavior (no audio → stillness governs).
- **Camera magnitude choice.** `camera_motion` (persisted) is a *fitted-displacement*
  pctl-normalized scalar; the plan instead derives `cam_n` from signed
  `camera_dx/dy/zoom` for a truer vector magnitude + clip-relative renorm. Both are
  available; the persisted scalar is the fallback. Minor: `zoom` and translation are
  in different physical units — magnitude combines them heuristically (they're each
  frame-relative), acceptable for a normalized attractor.
- **Grid consistency.** Persisted motion (if `frame_diff` ever lands) vs. proxy
  recompute could differ in `hop_ms`/length; v1 sidesteps this by recomputing the
  whole motion bundle from the proxy (§6), at the cost of not using persisted motion.
- **No persistence in v1** means the validation UI and driver recompute every time;
  fine for a debug/tuning tool, but a `seam_curves` table (future, additive, its own
  migration) would be needed before this is a production dependency.
- **Weights are unvalidated priors.** `w_vis`, `w_aud`, `σ`, `0.6`, salience scaling
  are starting guesses to be tuned on the `cutviz` overlay / a later harness — stated
  as such, isolated in `params.py`.
