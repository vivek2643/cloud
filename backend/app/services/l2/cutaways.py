"""Post-parse thinning for the sparse L2 ``cutaways`` track."""
from __future__ import annotations

from typing import List, Optional, Tuple

from app.services.l2.schema import ClipPerception, CutawayAffordance, CutawayMoment

MERGE_GAP_MS = 2000
MIN_BROLL_MS = 1500
_SPEAKER_SELF_OVERLAP_MS = 200


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _speaker_self_during(
    cutaway: CutawayMoment,
    speaking: List[Tuple[int, int, str]],
) -> bool:
    """Drop overlay cutaways where the subject is visibly speaking (delivery, not cutaway)."""
    if not cutaway.subject or cutaway.affordance != CutawayAffordance.reaction:
        return False
    subj = cutaway.subject.lower()
    for start_ms, end_ms, subject in speaking:
        if (subject or "").lower() != subj:
            continue
        if _overlap(cutaway.start_ms, cutaway.end_ms, start_ms, end_ms) > _SPEAKER_SELF_OVERLAP_MS:
            return True
    return False


def _merge_key(c: CutawayMoment) -> tuple:
    return (c.affordance.value, c.kind.value, (c.subject or "").lower(), (c.label or "").strip().lower())


def _merge_cutaways(items: List[CutawayMoment]) -> List[CutawayMoment]:
    """Collapse adjacent similar cutaways into one span (keep strongest salience)."""
    if not items:
        return items
    ordered = sorted(items, key=lambda c: (c.start_ms, c.end_ms))
    out: List[CutawayMoment] = [ordered[0]]
    for c in ordered[1:]:
        prev = out[-1]
        if (
            _merge_key(prev) == _merge_key(c)
            and c.start_ms - prev.end_ms <= MERGE_GAP_MS
        ):
            sal_prev = prev.salience_hint or prev.intensity or 0.0
            sal_c = c.salience_hint or c.intensity or 0.0
            out[-1] = CutawayMoment(
                start_ms=prev.start_ms,
                end_ms=max(prev.end_ms, c.end_ms),
                kind=prev.kind if sal_prev >= sal_c else c.kind,
                affordance=prev.affordance,
                subject=prev.subject or c.subject,
                label=prev.label if sal_prev >= sal_c else c.label,
                trigger=prev.trigger or c.trigger,
                intensity=max(prev.intensity or 0.0, c.intensity or 0.0) or None,
                editorial_role=prev.editorial_role or c.editorial_role,
                salience_hint=max(sal_prev, sal_c) or None,
                peak_ms=(prev.peak_ms if sal_prev >= sal_c else c.peak_ms),
            )
            continue
        out.append(c)
    return out


def thin_cutaways(perception: ClipPerception) -> None:
    """In-place dedupe and drop obvious non-cutaway rows before persist."""
    if not perception.cutaways:
        return
    speaking = [(s.start_ms, s.end_ms, s.subject) for s in perception.speaking]
    kept: List[CutawayMoment] = []
    for c in _merge_cutaways(list(perception.cutaways)):
        if c.end_ms <= c.start_ms:
            continue
        if _speaker_self_during(c, speaking):
            continue
        if c.affordance == CutawayAffordance.broll and (c.end_ms - c.start_ms) < MIN_BROLL_MS:
            continue
        kept.append(c)
    perception.cutaways = kept
