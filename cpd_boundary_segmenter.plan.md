# CPD Boundary Segmenter — build + train a generic event-boundary model

Status: PLAN ONLY (no code yet). Another chat will execute this.

## Goal
Train a **generic change-point / event-boundary model** that, given a NON-speech
video clip, outputs the timestamps where "one visual moment becomes another"
(e.g. a segment ends and a new one begins). We validate it against **human
boundary labels (Kinetics-GEBD)** so improvement is a measured number, not an
eyeball. The output is a reusable model + an inference function we can later
plug into the cuts pipeline.

## Scope (read this first)
- **CPD / boundary detection ONLY.** Where the boundaries are — nothing else.
- **Non-speech, visual footage only.** Speech is separated upstream (Pass-1) and
  is NOT part of this. The CPD never sees dialogue.
- **No phenomenon-specific features.** The detector sees only *generic*
  descriptors of visual dynamics and must *discover* structure (rhythm, subject
  entering, pans, reps…) as distribution shifts. If a feature can only be
  justified by naming an example, it does not go in.

## Explicit non-goals (do NOT build here)
- Junk trimming (camera/frame-diff/blur trimming of dead heads/tails).
- The energy/"usability" dial (short-form-peak vs long-form) — that is a
  downstream, purpose-selection layer, separate plan.
- Any production wiring of cuts / merge to main. This plan ends at a **validated,
  reusable boundary model + eval report** (+ optional debug overlay).

---

## The signal basis — LOCKED (generic channels only)
Per hop (10 fps / 100 ms), the multivariate series fed to the CPD:

| # | Channel | Source (today) | Generic meaning |
|---|---|---|---|
| 1 | `frame_diff` | motion_dynamics (added) | total temporal pixel change |
| 2 | `camera_dx`,`camera_dy`,`camera_zoom` | motion_dynamics | camera's raw motion vector |
| 3 | `action_energy` (+ `_raw`) | motion_dynamics | non-camera (residual) change |
| 4 | `camera_coherence` | motion_dynamics | rigid-body vs scattered motion |
| 5 | `blur` | motion_dynamics | image sharpness/usability state |
| 6 | `appearance_drift` | **NEW curve (see Phase C)** | rate the frame's *look* changes (HS/color histogram) |

Notes:
- Feed the **raw camera vector** (dx/dy/zoom), not derived "pan/scale-rate"
  features — a change in camera behaviour is already a distribution shift the CPD
  will find.
- `camera_stability` is intentionally EXCLUDED: it's derivable from the camera
  vector (redundant). `camera_coherence` is kept (independent info: rigidity).
- Everything is per-clip normalized the same way production does, so
  train == inference. CPD cares about *within-clip* shifts, so per-clip
  normalization is correct and generic.
- **Escalation option (only if the 6 channels underperform on GEBD):** replace
  hand channels with a generic per-frame embedding (small CNN/CLIP-style) and run
  CPD on the embedding sequence — the maximally assumption-free inputs. Decide
  via the benchmark, not intuition.

---

## Directory layout (new)
```
backend/scripts/cpd/
  fetch_gebd.py        # Phase A: get GEBD annotations + videos (subset)
  extract_features.py  # Phase C: video -> per-hop feature matrix (.npz)
  build_labels.py      # Phase D: GEBD timestamps -> per-hop boundary targets
  train_cpd.py         # Phase E: train boundary model
  eval_cpd.py          # Phase F: GEBD F1 + baselines
  infer_cpd.py         # Phase G: model -> boundaries, for reuse
backend/scripts/cpd/data/
  raw_videos/          # downloaded clips
  annotations/         # GEBD label files
  features/            # per-clip <id>.npz (features + labels + hop_ms)
backend/models/
  cpd_boundary.<ext>   # trained artifact
```

---

## Phase A — Data acquisition (Kinetics-GEBD)
1. **Annotations.** Pull the Kinetics-GEBD train/val boundary annotations from the
   official LOVEU release (per-video: list of boundary timestamps, multiple
   annotators). Store under `data/annotations/`.
2. **Videos.** GEBD is built on Kinetics (YouTube) clips. Download a **subset**
   (start ~500–1000 train + ~200 val) with `yt-dlp` at low res (256p is plenty —
   our features are computed on a tiny proxy anyway). Skip dead links; log
   coverage.
3. **Fallback if Kinetics download is too flaky:** use TAPOS (the other GEBD
   source, sports/Olympic instances) or any subset that actually downloads — the
   pipeline is source-agnostic once we have (video, boundary-timestamps).
4. Write a manifest `data/manifest.csv`: `clip_id, path, split, boundary_ms[...]`.

**Risk flagged:** Kinetics acquisition is the messiest step (link rot, rate
limits). Budget time here; a few hundred clips is enough for a first model.

## Phase B — Environment
- Deps already present: `numpy`, `opencv`, `ffmpeg`, `torch` (whisper/pyannote).
- Add if missing: `yt-dlp` (download), and ONE of `lightgbm`/`scikit-learn`
  (tree baseline) — torch covers the neural option.
- **All feature extraction and training run locally on CPU** (torch can use MPS
  on this Mac for the neural variant). No cloud, no RunPod.

## Phase C — Feature extraction (LOCAL, CPU — the "how")
This is the crux of "can we do it locally": **yes, by reusing the production
extractor.**
1. For each clip, call the existing
   `app.services.l1.motion_dynamics.compute_motion_dynamics(video_path, duration_ms)`
   → gives channels 1–5 as per-hop lists at 100 ms.
2. Add channel 6 `appearance_drift`: a small function computing the per-hop
   **HS (+ optionally value) histogram chi-square distance between consecutive
   sampled frames** — reuse the histogram logic already in
   `app/services/l1/scene_cuts.py` (which today only keeps thresholded points;
   here we want the *continuous curve*). Either expose the drift curve from
   `scene_cuts` or recompute it standalone in the extractor.
3. Assemble a `T × C` float matrix (T = hops, C ≈ 7 with dx/dy/zoom expanded),
   aligned on the 100 ms grid. Save `data/features/<clip_id>.npz` with:
   `features (T×C)`, `hop_ms`, `channel_names`, and (after Phase D) `labels (T,)`.
4. **Cost/feasibility:** the flow pass is a few seconds per short clip on CPU;
   parallelize across clips with `multiprocessing.Pool`. 500–2000 clips is an
   overnight/CPU-pool job, not a GPU job.
5. **Correctness win:** because this is the *same* `compute_motion_dynamics` used
   in production, the model trains on the exact features it will see at inference.

## Phase D — Label alignment
1. GEBD gives boundary *timestamps* with multiple annotators. Build a
   **consensus** boundary set per clip (e.g. keep a boundary where ≥ k annotators
   agree within a small window, or average nearby marks — follow the GEBD
   challenge's consensus rule).
2. Convert consensus timestamps → a per-hop binary target vector on the 100 ms
   grid: `label[t]=1` if t is within tolerance τ of a consensus boundary
   (τ from the GEBD Rel.Dis protocol, ~0.2 s for typical clip lengths).
3. Boundaries are RARE → note class imbalance for training (Phase E).

## Phase E — Model + training
Train two things so we know the features are even predictive:
1. **Unsupervised baseline (no training):** boundary score(t) = dissimilarity
   between the feature distribution in a window BEFORE t vs AFTER t
   (e.g. distance of mean feature vectors / a two-sample statistic). Peak-pick.
   This is honest classical CPD; if it already predicts GEBD boundaries, the
   features carry the signal.
2. **Supervised boundary model:** sliding window of features around t → P(boundary).
   Start simple, escalate only if needed:
   - (a) Gradient-boosted trees (LightGBM/sklearn) on the flattened window — CPU,
     fast, strong tabular baseline.
   - (b) If (a) plateaus: a small 1-D TCN / BiLSTM (torch, CPU/MPS).
   Handle imbalance (class weights / focal loss / positive up-sampling).
   Split train/val **by video** (no leakage).

## Phase F — Evaluation (the scoreboard)
1. Metric: **GEBD F1 with Relative-Distance threshold** (report F1@0.05 as the
   headline, plus the 0.05→0.5 curve).
2. Compare, on the held-out val split:
   - unsupervised baseline vs supervised model,
   - and, as a sanity check, our **current segmenter's** implied boundaries.
3. Emit `eval_cpd` report (F1 numbers + a few overlaid examples). This report is
   the deliverable that tells us whether to proceed to embeddings (escalation) or
   move on to the downstream trim/energy layers.

## Phase G — Inference artifact ("for us to use")
1. `infer_cpd.py`: `boundaries_ms = predict(video_path_or_features)` — runs the
   Phase C extractor + the trained model → smoothed probability → peak-pick
   (min gap between boundaries) → boundary timestamps.
2. **Optional (nice):** add a boundary overlay to the existing debug viz
   (`/api/debug/cutviz`) so we can *see* predicted boundaries on our own clips
   and hand-mark disagreements — turning the tool into the in-domain
   labeling/validation UI.
3. Save the trained model to `backend/models/cpd_boundary.<ext>` with a small
   inference wrapper so the cuts pipeline can import it later.

---

## Deliverables
- `backend/scripts/cpd/*` (fetch, extract, labels, train, eval, infer).
- `backend/models/cpd_boundary.<ext>` trained artifact.
- Eval report with GEBD F1 (supervised vs unsupervised vs current segmenter).
- (Optional) debug-viz boundary overlay.

## Open decisions (pick during execution)
- Dataset size / subset; Kinetics vs TAPOS fallback.
- Consensus rule + tolerance τ.
- Model: trees first vs straight to TCN.
- Whether to add the generic-embedding channel (only if 6 channels underperform).

## Honest risks
- **Domain gap:** GEBD is Kinetics/action footage, not drone/vlog/factory. GEBD
  proves "can we detect human boundaries generically"; a small **in-domain**
  hand-labeled set (mark boundaries on our own clips via the debug tool) is the
  real target and should be built alongside.
- **Event boundary ≠ editorial cut:** GEBD boundaries are perceptual, not an
  editor's cut points — treat GEBD as a general prior/benchmark, the in-domain
  set as ground truth.
- **Feature contamination on aerial** (parallax inflates `action_energy`): the
  multivariate basis has redundancy (frame_diff, appearance, camera vector) so
  the model can route around a lying channel — the benchmark will show if it does.
- **Download friction & class imbalance** (above) are the practical time sinks.
