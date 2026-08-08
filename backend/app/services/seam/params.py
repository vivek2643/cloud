"""
Tunable constants for the seam-quality curve S(t) (seam_function.plan.md).

STARTING PRIORS -- not final. Isolated here (never inlined in curve.py's
math) so a later harness can retune without touching the S(t) formula
itself. See the plan's §9 "Weights are unvalidated priors."
"""
from __future__ import annotations

from app.services.l1.cut_grid_params import MOTION_NORM_PCTL

# Percentile used for every clip-relative normalize_pctl() call in this
# module (§4) -- reuses L1's own default rather than inventing a second one.
NORM_PCTL = MOTION_NORM_PCTL

# --- Gates (§2) --------------------------------------------------------------
# g_gest(t) = 1 - GEST_COEFF * act_n(t) -- caps the worst-case mid-gesture
# attenuation at (1 - GEST_COEFF), a ~0.4 floor at GEST_COEFF=0.6, never to 0.
GEST_COEFF = 0.6

# --- Attractors (§2) -----------------------------------------------------
# audio(t) Gaussian-kernel half-width (ms): how far a beat/onset's influence
# reaches on the grid before decaying away.
AUDIO_SIGMA_MS = 60.0

# --- Weights (§2) --------------------------------------------------------
W_VIS = 1.0
# w_aud(t) = W_AUD_BASE * salience(t); salience = 1.0 when musical, else the
# clip's own scaled onset strength near t (0 when silent) -- see curve.py.
W_AUD_BASE = 1.2
