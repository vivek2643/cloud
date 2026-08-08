"""
seam_function.plan.md -- the seam-quality curve S(t) for the NEW cuts
pipeline. A quality field only: it never decides whether/where a cut
exists (see the plan's "Explicit non-goals"). Fully separate from the OLD
pipeline (l3/v4_segment.py and friends), which is untouched.
"""
from __future__ import annotations

from app.services.seam.curve import SeamCurve, compute_seam_curve
from app.services.seam.signals import SeamSignals, build_seam_signals

__all__ = ["SeamSignals", "build_seam_signals", "SeamCurve", "compute_seam_curve"]
