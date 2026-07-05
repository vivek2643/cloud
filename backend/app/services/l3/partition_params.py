"""
Tuning knobs for the cuts-v2 unified partition (``partition.py``), in one place.

Mirrors ``l1/cut_grid_params.py``'s convention: heuristic defaults, parked here
so they can be re-tuned in one place once the validation harness
(``scripts/viz_cuts.py``) has been run against real clips -- see
cuts_v2.plan.md's "Detection robustness" section.
"""
from __future__ import annotations

# --- Priority (North star #3: priority, deterministic) ----------------------
# Higher wins a contested span; the loser is demoted to a TAG on the winner
# (overlap >= OVERLAP_TAG_FRAC of the candidate) or trimmed to its free
# remainder. Values are the fused-field `energy`-independent CLAIM priority,
# not a display weight.
PRIORITY_SAID = 1.0
PRIORITY_DONE = 0.6
PRIORITY_SHOWN = 0.3

# A contested candidate is absorbed as a TAG (not its own cut) once this much
# of it is already claimed by a higher-priority cut; otherwise the free
# remainder becomes its own cut.
OVERLAP_TAG_FRAC = 0.60

# A free remainder shorter than this isn't a meaningful sub-unit -- it is
# silently absorbed (no cut, no tag: too small to be either) rather than
# emitted as a sliver cut. Keeps the bias toward over-split (many clean
# sub-units) without producing unusable micro-cuts.
MIN_SUBUNIT_MS = 300

# --- DONE candidates (action rise -> peak -> fall, bounded by calm) ---------
# An action window expands outward from its impact while motion energy (file-
# normalized 0..1, see motion_dynamics.normalize_pctl) stays above this floor;
# the floor IS the "calm" baseline the plan's B2 algorithm names.
ACTION_CALM_PCTL = 40.0
DONE_MIN_MS = 400

# --- Boundary snapping / merge -----------------------------------------------
# Two claimed cuts touching within this gap (after snapping) are the same
# continuous run, not two separate cuts with a hairline seam between them.
MERGE_GAP_MS = 60

# Fixed default tightness for B2's boundary SNAP only (candidate spans
# themselves are energy-independent by design -- see North Star #1/#5). B3
# layers the real tightness dial on top of the claimed cuts; this constant is
# just the field's attractor-weight lambda while detecting/claiming.
DEFAULT_SNAP_ENERGY = 0.5
