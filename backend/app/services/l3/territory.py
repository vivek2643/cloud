"""
Territory rules -- who "owns" a span for feed ranking (sync vs cutaway).

Overlay anchors that overlap the *same subject's* on-camera speech are demoted
(speaker nodding while talking is delivery, not a cutaway). Listener reactions
and low-speech b-roll spans stay strong. This is a rank multiplier, not deletion.
"""
from __future__ import annotations

from typing import List, Optional

from app.services.l3 import anchors as anc


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _speech_occupation(
    start_ms: int, end_ms: int, speaking: List[dict],
) -> float:
    """Fraction of [start_ms, end_ms] covered by any VLM speaking span."""
    span = max(1, end_ms - start_ms)
    covered = 0
    for s in speaking:
        a, b = int(s.get("start_ms", 0)), int(s.get("end_ms", 0))
        covered += _overlap(start_ms, end_ms, a, b)
    return min(1.0, covered / span)


def _subject_speaking_during(
    subject: Optional[str], start_ms: int, end_ms: int, speaking: List[dict],
) -> bool:
    """True if this person is visibly speaking for any overlap with the anchor."""
    if not subject:
        return False
    subj = subject.lower()
    for s in speaking:
        if (s.get("subject") or "").lower() != subj:
            continue
        a, b = int(s.get("start_ms", 0)), int(s.get("end_ms", 0))
        if _overlap(start_ms, end_ms, a, b) > 200:
            return True
    return False


def speech_occupation(start_ms: int, end_ms: int, speaking: List[dict]) -> float:
    """Fraction of a span covered by VLM speaking (public helper)."""
    return _speech_occupation(start_ms, end_ms, speaking)


def territory_multiplier(
    anchor: anc.Anchor,
    *,
    speaking: Optional[List[dict]] = None,
    strict: bool = False,
) -> float:
    """0..1 rank factor for an overlay anchor. Sync anchors always 1.0."""
    if anchor.affordance in (anc.AFF_SPEECH, anc.AFF_ACTION):
        return 1.0
    speaking = speaking or []

    if anchor.affordance in (anc.AFF_REACTION,) or anchor.kind == "gaze":
        if _subject_speaking_during(anchor.actor, anchor.start_ms, anchor.end_ms, speaking):
            return 0.15 if strict else 0.35

    if anchor.affordance == anc.AFF_BROLL:
        occ = _speech_occupation(anchor.start_ms, anchor.end_ms, speaking)
        if occ >= 0.55:
            return 0.25 if strict else 0.45
        if occ >= 0.30:
            return 0.55 if strict else 0.75

    return 1.0
