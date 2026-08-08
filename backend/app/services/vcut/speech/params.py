"""
speech_cuts_pipeline.plan.md -- tunable constants for the speech-cuts
channel. Every knob boundaries.py/delivery.py/select.py reads lives here so
those pure modules never hard-code a number (mirrors app.services.vcut.
params' own convention). Model ids live in app.config.Settings.
"""
from __future__ import annotations

# --- Stage 4 -- boundaries.py (section 9) ----------------------------------

# Max extension of an edge into adjacent silence, snapped to the silence's
# own midpoint (never past a neighboring word) -- "the cut breathes."
BREATH_PAD_MS = 250

# --- Stage 5 -- delivery.py / select.py (section 10) ------------------------

# An intra-span gap longer than this counts as hesitation.
HESITATION_GAP_MS = 350

# Natural words/sec band -- pace scores 1.0 inside, falls off outside.
PACE_LO = 2.0
PACE_HI = 3.3

# Delivery term weights (group-normalized terms, except pace which is
# scored against the fixed band above -- section 10's own distinction).
W_ENERGY = 1.0
W_DYNAMICS = 0.6
W_PACE = 1.0
W_HESITATION = 1.0
# ASR confidence is intentionally NOT a v1 delivery term: it isn't captured on
# existing transcripts yet, and a neutral fallback for a not-yet-universal
# signal is a smell. Re-add it here (and in delivery.py) once L1 has backfilled
# per-word probability everywhere -- then it's always present, no fallback.
W_VISUAL = 0.4

# Top-level fusion: final = gate * (W_FLUENCY*fluency_llm + W_DELIVERY*delivery)
W_FLUENCY = 1.0
W_DELIVERY = 1.0

# --- Stage 6 -- frames.py (section 11) --------------------------------------

SPEECH_FRAME_LONG_MS = 6000   # a cut longer than this MAY get a 2nd frame
SPEECH_FRAME_MAX = 2

# Max on-camera cuts whose frames go into ONE complete_gemini call, so
# run_frame_analysis chunks its targets into bounded batches instead of one
# ever-growing call. Mirrors l3.pass2_params.MAX_CUTS_PER_PASS2_BATCH (=7):
# the frame call rides the same flash-lite structured-output path, which has
# the same output-budget reliability cliff -- a single call must emit one
# answer per cut, so at multicam fan-out scale (expand_outlook_cuts fans one
# winning beat into ~1 cut PER angle -- ~30 on a 30-angle shoot, all kept by
# select_cuts_needing_frames) the combined thinking+output tokens for ~30
# answers overrun max_output_tokens and the model returns zero JSON. Bounding
# the answers per call at the same 7 the video Pass-2 uses keeps every call
# comfortably inside the budget regardless of how wide the shoot's fan-out is.
SPEECH_FRAME_BATCH = 7

# --- Noise/hallucination gate (inputs.py + orchestrate.py) ------------------
# Whisper (run without a hard VAD) hallucinates confident-looking text over
# ambient/background noise; those spurious segments otherwise flow straight
# into speech cuts. Two independent gates drop them before they become cuts.

# (1) Per-SEGMENT Whisper self-confidence gate, applied at transcript load.
# A segment Whisper itself thinks is non-speech (no_speech_prob high) OR
# decoded with very low average token log-prob (garble) is dropped whole.
# Only fires when the signal is present -- older transcripts (captured before
# L1 stored these) have None here and are kept (fail-open, no false drops).
SEG_NO_SPEECH_MAX = 0.6        # P(no speech) at/above this => drop the segment
SEG_MIN_AVG_LOGPROB = -1.2     # avg token log-prob below this => drop the segment

# (2) Per-BEAT self-calibrating energy gate, applied after boundary snapping.
# A hallucinated beat sits near the file's own noise floor while real speech
# sits near its voiced level -- so calibrate against the file's rms_db
# distribution (no absolute dB constant, robust across mics/levels). A beat
# whose median energy is below floor + MIN_VOICED_FRAC*(voiced-floor) is not
# real speech and is dropped. Disabled (fail-open) when the file's loud/quiet
# spread is too small to discriminate, or there isn't enough rms data.
RMS_VOICED_PCTL = 75           # "typical voiced/speech" reference percentile
RMS_FLOOR_PCTL = 15            # "quiet/noise floor" reference percentile
MIN_VOICED_FRAC = 0.30         # beat median must clear this fraction up from floor
MIN_VOICED_SPREAD_DB = 8.0     # need at least this loud-vs-quiet range to gate

# speech_noise_gate.plan.md Improvement B: measure the true noise floor from
# detected silence frames rather than a blind percentile, when there's enough
# silence to be representative.
SILENCE_FLOOR_MIN_FRAMES = 10
