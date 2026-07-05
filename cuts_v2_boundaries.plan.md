# Cuts v2 — boundary quality + the unified granularity/tightness dial

Follow-on to `cuts_v2.plan.md`. That plan built the deterministic non-overlapping
partition, the per-video row UI, and a tightness-only dial. This plan is about
the **quality of the cuts themselves** — *where* the boundaries land — and it
**re-scopes the dial** to do **granularity AND tightness together**, per an
explicit product decision (supersedes `cuts_v2.plan.md`'s "tightness only, no
granularity" North Star #5 for the video + speech channels).

Everything here stays **additive / versioned**, running in parallel with the
existing partition, same as `cuts_v2.plan.md`.

---

## Working assumptions (this plan)

1. **One clip = one continuous shot.** Scene/shot detection is **deferred** — we
   ignore `scene_cuts` as a boundary source for now and add it back later for
   real multi-shot footage. (The B1 module and table stay; just unused here.)
2. **done/shown *labeling* is deferred to the image pass** (Option A: one cheap
   Claude-vision call per clip on peak frames). This plan does NOT try to fix
   done-vs-shown classification from motion — it only fixes *segmentation*.
   Video cuts may keep a provisional/neutral tag until the image pass lands.
3. **`said` is the trusted channel**; we only make its boundaries smarter with
   prosody, we don't rebuild it.

---

## North star: ONE dial = granularity + tightness

Supersedes the tightness-only dial. Principle:

> **Detection finds ALL candidate boundaries. The dial decides how many to
> honor (granularity) and how tight to trim each (tightness).**

Low energy = few, long, full cuts (relaxed / contextual). High energy = many,
short, tight cuts (punchy / trailer). Dragging up **adds cuts AND tightens them
at the same time.** Speaker-change (and later shot-cut) always break, at every
level.

### The mapping

| Dial | Video — what to cut on | Video — length | Speech — unit | Speech — cleanup |
|------|------------------------|----------------|---------------|------------------|
| **Broad** | camera holds/moves only, merged into long segments | full span | whole turns (merge same speaker) | keep everything |
| **Calm** | + each camera move → settle | full | turn / setup | keep breaths |
| **Balanced** | + *major* subject-motion beats | natural | one thought | natural |
| **Tight** | + split holds on subject beats / sub-moves | inset toward peak | core sentence | trim long breaths |
| **Sharp** | over-split every distinct beat | tight (~1s) | punchline | aggressive breath removal |

Bias: **over-split, never under-split** — weld-on-select recovers an over-split;
an under-split is unrecoverable in the surface (no split tool). So err toward
more cuts, especially at the high end.

---

## Boundary sources (what actually creates a cut)

### Speech (`said`)
- **Spine:** transcript + diarization + `thought_segments` (unchanged).
- **Candidate boundaries graded by prosody**, not transcript alone:
  - `silence_intervals` + **gap length** (already computed),
  - `rms_db` intensity dip (already computed),
  - **pitch / f0** (NEW L1 signal): falling intonation + low energy + long gap =
    a real clause/thought end (cut here); sustained pitch across a gap = an
    *intentional* dramatic pause (keep together).
- **Dial** thresholds which graded candidates to take (Broad = only speaker
  change + strongest falls → whole turns; Sharp = every clause boundary →
  punchlines) and how much breath to excise (`keep_spans`).
- Speaker change is always a hard boundary.

### Video — moving footage (primary)
- **Camera-move state** is the primary boundary, from existing L1 motion:
  `camera_motion` + `camera_coherence` + `camera_stability` (stability is
  ABSOLUTE, not per-clip-normalized — the reliable one).
  - **hold** = stability high + low camera motion, sustained ≥ HOLD_MIN_MS.
  - **move** = a coherent run of camera motion between two holds (a pan / push).
  - **boundary sits at the *settle*** (start of a hold) — cut on the still, not
    mid-move.
  - **hysteresis** (must persist N hops to flip) so it doesn't flap.
- The **dial** decides whether distinct sub-moves within a run split out.

### Video — static / locked-off footage (fallback)
- No camera move → segment on **subject-motion beats** (`action_energy`
  rise→peak→fall) so an intentional action in a locked shot still becomes its own
  cut. Dial decides how finely (Broad = one whole cut; Sharp = every beat).

### Impacts — demoted
- `action_points` are **no longer a primary boundary**. They become:
  - the **peak / thumbnail** instant inside a cut, and
  - **candidate split points** only at high granularity.
- Kills the noisy ~1.5/sec impact-driven edges.

### Snapping (unchanged mechanism, extended attractors)
- Boundaries snap through the fused seam field: **mid-word veto** (never cut
  speech), **camera veto** (never cut mid-whip), attractors toward camera-settle
  / motion-valley / beat / pause.

---

## The one new L1 signal: pitch / f0

- New L1 stage `pitch` (or fold into `audio_features`): a coarse f0 track
  (librosa `pyin` or equivalent) at the existing prosody hop, best-effort
  (unavailable → empty, never fails L1), mirroring `audio_features` semantics.
- New column(s) on `audio_features` (or a small `prosody` table): `f0_hz` series
  + hop. Additive; easy to drop.
- **Needs a re-analyze** of existing clips to populate. Everything else in this
  plan (camera-move, RMS, gaps) needs no re-analyze.

---

## Tightness floor retune

- Lower the video peak-inset floor from 1000ms to a **~700–800ms safety net**;
  let the per-band fraction do the work so cuts *generally* land ~1s but can dip
  a little under for genuinely short beats. (`VIDEO_CORE_FLOOR_MS`, one place.)
- It is a FLOOR (min inset), never a ceiling — low energy keeps cuts full-length.

---

## Phases (build order)

### C1 — Camera-move segmentation + dial granularity for video (no re-analyze)
- New `l3/video_segments.py` (or extend `partition._done_candidates` /
  `_shown_candidates`): derive video candidate boundaries from camera-move state
  (hold/move/settle + hysteresis) instead of impact windows.
- Demote `action_points` to peak + high-granularity split points.
- Static-shot fallback: subject-motion beats.
- Wire the **dial's granularity** for video: Broad merges to long holds; Sharp
  over-splits. Keep the tightness inset (retune floor here).
- Centralize all thresholds in `partition_params` / a new
  `video_segment_params` (HOLD_MIN_MS, stability/coherence thresholds,
  hysteresis hops, subject-beat floor, per-band granularity, core floor/frac).
- Biggest visible win; no new signal, no re-analyze.

### C2 — Pitch + prosody pause-grading + dial granularity for speech
- New L1 `pitch` stage + storage (re-analyze required).
- New speech boundary grader: transcript candidates × (pitch fall + RMS dip +
  gap length) → keep/cut decision; wire the dial's speech granularity + breath
  cleanup onto it.
- Re-map the speech side of the dial (Broad turns → Sharp punchlines).

### C3 — Deferred (later)
- Re-enable **scene/shot detection** as the top-priority video boundary for
  multi-shot footage (drop the single-shot assumption).
- **Image pass** (Claude vision on peak frames) for correct done-vs-shown
  *labeling* + junk (Option A).

---

## What we reuse vs. what's new

- **Reuse:** `motion_dynamics` (camera_motion / coherence / stability / blur /
  action_energy / action_points), `audio_features` (rms_db, silence_intervals),
  `thought_segments`, diarization, `fused_seams` (snap + vetoes), the priority
  claim + non-overlap invariant, the per-video row UI, the energy slider.
- **New logic:** camera-move state segmentation, subject-beat static fallback,
  prosody pause-grading, the granularity axis of the dial for video + speech.
- **New signal:** pitch / f0 (C2).
- **Retune:** video core floor; centralized thresholds.

---

## Honest risks / tensions

1. **Detection is now energy-dependent** (granularity on the dial) — a departure
   from `cuts_v2.plan.md`'s "detect once" principle. Accepted: it's cheap
   (deterministic recompute per dial position) and more useful.
2. **Over-fragmentation on noisy handheld footage** at high granularity —
   mitigated by weld-on-select (over-split is recoverable, under-split isn't).
3. **One dial couples count↑ and length↓** — opinionated but matches editorial
   feel; can be split into two controls later if needed.
4. **Pitch depends on clean audio** and a re-analyze; degrade to RMS + gap-length
   when f0 is unavailable.
5. **Video tags stay provisional** until the image pass (C3) — segmentation
   improves now; done-vs-shown *meaning* still waits on vision.

---

## Validation

- Extend `scripts/viz_cuts.py` to overlay camera-move state (hold/move/settle),
  subject beats, and speech prosody boundaries so an editor can eyeball whether
  cuts land where they'd cut — the go/no-go, per clip, before wiring the full
  surface.
