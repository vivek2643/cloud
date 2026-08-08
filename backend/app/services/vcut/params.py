"""
seam_cut_pipeline.plan.md -- tunable constants for the vcut pipeline.
Every knob resolve.py reads lives here so the algorithm itself never hard-
codes a number (plan section 15). Model ids live in app.config.Settings
(this repo's convention for anything env-overridable), not here.
"""
from __future__ import annotations

# --- Stage 1 -- non-speech spans (section 5) -------------------------------

# Gaps shorter than this between speech spans aren't worth cutting -- fold
# into the surrounding speech instead of emitting a sliver non-speech span.
MIN_NONSPEECH_SPAN_MS = 500

# --- Stage 2 -- sampling (section 6) ---------------------------------------
# No "density" knob exists anywhere in the old l3 pipeline to reuse (image_
# plan.py's FRAME_BUDGET_PER_CLIP is atom-aware and not applicable here) --
# these are net-new, vcut-only constants for "medium" density over non-
# speech spans only.

SAMPLE_INTERVAL_MS = 1500      # one frame roughly every 1.5s of non-speech footage
MAX_FRAMES_PER_FILE = 40       # hard cap regardless of span length (cost control)
SAMPLE_WIDTH_PX = 768          # matches pass2_params.STILL_WIDTH_PX elsewhere

# --- pass1_video_input.plan.md -- Pass 1 video input (subclip.py) ----------
# Behind config.vcut_pass1_input_mode="video" (default "frames" until
# validated live). The non-speech sub-clip ffmpeg cuts+concats per file.

VCUT_VIDEO_WIDTH_PX = 640      # sub-clip encode width (downsampled -- cost lever alongside fps)
SUBCLIP_MIN_MS = 1500          # below this total non-speech, not worth cutting/uploading -- use frames
SUBCLIP_MAX_MS = 20 * 60 * 1000  # safety ceiling against a pathological all-non-speech file
JOIN_GUARD_MS = 200            # a moment landing within this of a concat seam is a splice artifact, drop it

# --- vcut_pass2_video_specifics.plan.md section 4/9 -- qplan.py, the
# question planner. ---

QPLAN_CTX_MS = 4000            # local transcript window around each moment: [t-CTX, t+CTX]
QPLAN_MAX_CUSTOM = 2           # free-text custom probes per moment, on top of the closed bank

# --- Stage 3 -- resolve.py, "the algorithm" (section 7) --------------------

# Step 0: snap a VLM peak timestamp onto the nearest local max of
# action_energy (fallback frame_diff) within this radius.
PEAK_SNAP_MS = 250

# vcut_moment_energy.plan.md section 4.5/4.1: energy (0..1) -> per-flag
# keep-width. e=0 -> REACH_MAX_MS (the LOOSENESS knob -- how far apart two
# flags can be and still fuse into one long loose cut at energy 0), e=1 ->
# REACH_MIN_MS (the tight floor). Renamed from WIDE_MS/TIGHT_MS and RAISED
# (2500 -> 5000) so energy-0 cuts are genuinely long by default (plan
# section 0's "long, loose cuts by default" goal) -- this is the number to
# sweep on real footage (plan section 12.4).
REACH_MAX_MS = 5000
REACH_MIN_MS = 600

# Step 2: tag asymmetry -- fraction of the window kept before vs. after the
# refined peak. Each pair sums to 1.0.
TAG_SPLIT = {
    "both": (0.5, 0.5),
    "build": (0.7, 0.3),     # keep the run-up, land on/just past the climax
    "settle": (0.3, 0.7),    # cut into the peak, keep the settle/landing
}

# Step 4: seam-snap search radius around each provisional edge -- argmax
# S(t) within +/- SNAP_MS, never crossing past the window's own peak(s).
SNAP_MS = 300

# vcut_moment_energy.plan.md section 3: a seam between two adjacent flags is
# "strong" (a wall -- the two flags can NEVER fuse, at any energy) when its
# S(t) clears this fraction of the file's own max S. The other tunable to
# sweep on real footage (plan section 12.4) alongside REACH_MAX_MS.
STRONG_SEAM_FRAC = 0.6

# Step 5: junk / dead-air floor (energy-independent, section 7.6).
S_DEAD_FRAC = 0.35        # median-S-below-this-fraction-of-clip-median => dead air
MIN_CUT_MS = 500          # a snapped cut narrower than this is widened symmetrically
MIN_CUT_GAP_MS = 200      # minimum gap enforced between consecutive cuts in one file

# vcut_moment_energy.plan.md section 5: default global energy -- LOOSE (long
# cuts holding many moments) so a fresh ingest is immediately useful without
# ever dragging the dial; raising it live is what starts breaking cuts apart.
DEFAULT_ENERGY = 0.0
