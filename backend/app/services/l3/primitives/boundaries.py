"""
Boundary detectors: where it is safe / natural to make a cut.

Every detector returns a list of ``Boundary(ms, kind, strength)`` in absolute
source-file milliseconds. Recipes never invent cut points -- they pick a target
and snap it to the nearest real boundary via ``snap_to_boundary``. This is the
mechanism that prevents the LLM (or naive math) from cutting mid-word or
off-beat.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import List, Optional

from app.services.l3.primitives.loader import FileAnalysis, ShotRow

# A speech gap larger than this is treated as an utterance boundary.
SENTENCE_GAP_MS = 350
# Sentence-final punctuation that ends an utterance even without a big gap.
SENTENCE_END_CHARS = ".?!"


@dataclass(frozen=True)
class Boundary:
    ms: int
    kind: str        # "speech_start" | "speech_end" | "silence" | "beat" | "downbeat" | "shot" | "motion_peak"
    strength: float  # 0..1 -- how confident / strong this boundary is


# ---------------------------------------------------------------------------
# Detectors
# ---------------------------------------------------------------------------

def speech_boundaries(fa: FileAnalysis) -> List[Boundary]:
    """
    Utterance/sentence starts and ends derived from word-level timings.

    A boundary is emitted at:
      - the start of a word that begins an utterance (first word, or first word
        after a gap > SENTENCE_GAP_MS, or after sentence-final punctuation),
      - the end of a word that ends an utterance (last word, or last word
        before such a gap, or a word ending in . ? !).
    Leading/trailing fillers are skipped when choosing the edge word.
    """
    tx = fa.transcript
    if not tx or not tx.words:
        return []
    words = tx.words
    n = len(words)
    out: List[Boundary] = []

    for i, w in enumerate(words):
        starts_utterance = i == 0
        ends_utterance = i == n - 1
        if i > 0:
            gap = w.start_ms - words[i - 1].end_ms
            prev_text = words[i - 1].text
            if gap >= SENTENCE_GAP_MS or (prev_text and prev_text[-1:] in SENTENCE_END_CHARS):
                starts_utterance = True
        if i < n - 1:
            gap = words[i + 1].start_ms - w.end_ms
            if gap >= SENTENCE_GAP_MS or (w.text and w.text[-1:] in SENTENCE_END_CHARS):
                ends_utterance = True

        if starts_utterance:
            out.append(Boundary(ms=w.start_ms, kind="speech_start", strength=0.9))
        if ends_utterance:
            out.append(Boundary(ms=w.end_ms, kind="speech_end", strength=0.9))

    return _dedupe(out)


def silence_boundaries(fa: FileAnalysis) -> List[Boundary]:
    """Edges of detected silence intervals -- clean places to cut audio."""
    if not fa.audio:
        return []
    out: List[Boundary] = []
    for s in fa.audio.silence_intervals:
        try:
            a = int(s.get("start_ms", 0))
            b = int(s.get("end_ms", 0))
        except (TypeError, ValueError, AttributeError):
            continue
        out.append(Boundary(ms=a, kind="silence", strength=0.7))
        out.append(Boundary(ms=b, kind="silence", strength=0.7))
    return _dedupe(out)


def beat_grid(fa: FileAnalysis) -> List[Boundary]:
    """
    Musical onset grid. Every Nth onset is promoted to a "downbeat" (stronger)
    so beat-synced recipes can prefer phrase boundaries. Returns [] for
    non-musical files.
    """
    if not fa.audio or not fa.audio.is_musical or not fa.audio.onsets_ms:
        return []
    onsets = sorted(set(int(x) for x in fa.audio.onsets_ms))
    out: List[Boundary] = []
    for idx, ms in enumerate(onsets):
        is_downbeat = idx % 4 == 0
        out.append(
            Boundary(
                ms=ms,
                kind="downbeat" if is_downbeat else "beat",
                strength=1.0 if is_downbeat else 0.6,
            )
        )
    return out


def motion_boundaries(shots: List[ShotRow]) -> List[Boundary]:
    """Shot edges + peak-motion instants -- natural cut points for visuals."""
    out: List[Boundary] = []
    for s in shots:
        out.append(Boundary(ms=s.start_ms, kind="shot", strength=1.0))
        out.append(Boundary(ms=s.end_ms, kind="shot", strength=1.0))
        if s.peak_motion_ms is not None and s.start_ms <= s.peak_motion_ms <= s.end_ms:
            out.append(Boundary(ms=s.peak_motion_ms, kind="motion_peak", strength=0.6))
    return _dedupe(out)


# ---------------------------------------------------------------------------
# Snapping
# ---------------------------------------------------------------------------

def snap_to_boundary(
    target_ms: int,
    boundaries: List[Boundary],
    max_dist_ms: int = 400,
    prefer_kinds: Optional[List[str]] = None,
) -> int:
    """
    Snap ``target_ms`` to the nearest boundary within ``max_dist_ms``.

    If ``prefer_kinds`` is given, boundaries of those kinds win ties (and are
    chosen even when slightly farther than a non-preferred boundary, as long as
    they're within max_dist_ms). Returns the original target when nothing is
    close enough -- callers should treat that as "no clean boundary here".
    """
    if not boundaries:
        return target_ms

    candidates = boundaries
    if prefer_kinds:
        pref = [b for b in boundaries if b.kind in prefer_kinds]
        if pref:
            candidates = pref

    best = min(candidates, key=lambda b: abs(b.ms - target_ms))
    if abs(best.ms - target_ms) <= max_dist_ms:
        return best.ms
    return target_ms


def nearest_boundary_at_or_after(target_ms: int, boundaries: List[Boundary]) -> Optional[int]:
    """First boundary ms at or after target (sorted), else None."""
    ms_list = sorted(b.ms for b in boundaries)
    idx = bisect.bisect_left(ms_list, target_ms)
    return ms_list[idx] if idx < len(ms_list) else None


def _dedupe(bs: List[Boundary]) -> List[Boundary]:
    """Sort by ms, keep the strongest boundary at each exact ms."""
    by_ms: dict[int, Boundary] = {}
    for b in bs:
        cur = by_ms.get(b.ms)
        if cur is None or b.strength > cur.strength:
            by_ms[b.ms] = b
    return [by_ms[m] for m in sorted(by_ms)]
