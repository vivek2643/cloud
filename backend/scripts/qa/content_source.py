"""color_phase1.plan.md Part 1c: the content-source classification seam.

Scoring a slide-deck screen recording ("demo trail") with photographic
metrics (saturation, skin, exposure) is meaningless -- there is no scene to
white-balance or expose, so it dominates `saturation_band`'s failures with
noise, not signal.

This is a SWAPPABLE, single-method interface, deliberately not a
classifier: today's only implementation is a hand-maintained map. A later
VLM-Pass-2 `content_type` field can replace the map body without changing
this function's signature or any caller -- do NOT build that path in
Phase 1 (explicitly deferred, see color_phase1.plan.md "Deferred / Later").
"""
from __future__ import annotations

# Keyed by project_id (stable across a project's re-graded threads) -- the
# demo-trail slide deck is the sole synthetic entry today, confirmed by eye
# against its contact sheet (branded title cards, UI mockups, a line chart;
# one real photographic talking-head shot is mixed into the same project,
# which this project-level map necessarily also marks synthetic -- a known,
# accepted imprecision of a project-level stopgap, not a per-shot one).
_SYNTHETIC_PROJECT_IDS = {
    "8621c012-58c1-4bd5-8897-b8e8e4f24dca",  # demo trail
    "5cd8f004-13c7-43f8-a1ed-d2f7e646fae7",  # demo trail (second project row, same thread)
}


_SYNTHETIC_PROJECT_LABELS = {"demo trail"}


def content_type_for(project_id: str, project_label: str) -> str:
    """Return "photographic" (default) or "synthetic" for a project. Today
    this is the hand-maintained map above; matched by `project_id` first
    (stable), falling back to an EXACT (case-insensitive, trimmed) label
    match so a re-created project with the same name is still caught
    without an immediate map update. Exact, not substring: "demo trail 2"
    is a different, unrelated project and must not match on "demo trail"."""
    if project_id in _SYNTHETIC_PROJECT_IDS:
        return "synthetic"
    if (project_label or "").strip().lower() in _SYNTHETIC_PROJECT_LABELS:
        return "synthetic"
    return "photographic"
