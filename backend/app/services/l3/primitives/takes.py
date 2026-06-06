"""
Take / redundancy detection.

People repeat themselves: the same sentence delivered twice (a re-take), or
near-identical B-roll of the same subject. Keeping all of them makes an edit
feel padded. These helpers cluster near-duplicate units and keep the best take
per cluster (by unit quality), so recipes work from a de-duplicated pool.

Speech is clustered by normalized-text similarity. Visuals are clustered by
their L2 narrative_description similarity (a cheap proxy for SigLIP that needs
no DB round-trip); callers that already have embeddings can pass a custom
``similarity`` to do true cosine clustering.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Callable, List, Optional

from app.services.l3.primitives.units import EditUnit


def _norm(text: str) -> str:
    return " ".join("".join(c for c in text.lower() if c.isalnum() or c.isspace()).split())


def _text_sim(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def dedup_speech_units(units: List[EditUnit], threshold: float = 0.82) -> List[EditUnit]:
    """Keep the best take of each repeated utterance (text similarity)."""
    speech = [u for u in units if u.modality == "speech"]
    return _dedup(speech, threshold, lambda a, b: _text_sim(a.text, b.text))


def dedup_visual_units(
    units: List[EditUnit],
    threshold: float = 0.9,
    similarity: Optional[Callable[[EditUnit, EditUnit], float]] = None,
) -> List[EditUnit]:
    """Keep the best of each near-duplicate visual (narrative-text proxy, or a
    caller-supplied similarity like SigLIP cosine)."""
    visual = [u for u in units if u.modality == "visual"]
    sim = similarity or (lambda a, b: _text_sim(a.text, b.text))
    return _dedup(visual, threshold, sim)


def _dedup(
    units: List[EditUnit],
    threshold: float,
    sim: Callable[[EditUnit, EditUnit], float],
) -> List[EditUnit]:
    """
    Greedy clustering: process units best-quality first; a unit either joins an
    existing cluster (similarity >= threshold to its representative) or starts a
    new one. Only the representative (highest quality) survives.
    """
    kept: List[EditUnit] = []
    for u in sorted(units, key=lambda x: x.quality, reverse=True):
        if any(sim(u, rep) >= threshold for rep in kept):
            continue
        kept.append(u)
    # Restore chronological order for downstream assembly.
    kept.sort(key=lambda x: (x.in_ms, x.out_ms))
    return kept
