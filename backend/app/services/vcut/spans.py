"""
seam_cut_pipeline.plan.md section 5 -- non-speech span extraction.

The complement of a file's speech segments over [0, duration_ms]: sampling.py
/pass1.py only ever sample frames inside these spans, and resolve.py never
emits a cut outside them. A file with no transcript at all is one non-speech
span covering the whole clip (silent B-roll/drone footage) -- correct per
the plan's own note in section 5.
"""
from __future__ import annotations

from typing import List, Tuple

from app.services.vcut.params import MIN_NONSPEECH_SPAN_MS


def _pg_conn():
    from app.services import db
    return db.connection_dict_row()


def _speech_segments_ms(file_id: str) -> List[Tuple[int, int]]:
    """Raw (start_ms, end_ms) transcript segments for a file -- mirrors
    app.services.seam.signals._speech_spans' own direct SELECT (same table/
    shape), duplicated rather than imported so vcut stays fully separate
    from the seam package's own internals (both read transcripts.segments,
    a stable L1 shape, independently)."""
    with _pg_conn() as conn:
        row = conn.execute(
            "select segments from transcripts where file_id = %s", (file_id,)
        ).fetchone()
    if not row or not row["segments"]:
        return []
    return [
        (int(seg["start_ms"]), int(seg["end_ms"]))
        for seg in row["segments"]
        if isinstance(seg, dict) and "start_ms" in seg and "end_ms" in seg
    ]


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort + merge overlapping/touching spans so the complement below never
    sees a negative-length gap from two speech segments that overlap."""
    if not spans:
        return []
    ordered = sorted(spans)
    merged = [ordered[0]]
    for s, e in ordered[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def non_speech_spans_from_speech(
    speech_spans_ms: List[Tuple[int, int]], duration_ms: int,
) -> List[Tuple[int, int]]:
    """Pure core (section 5, steps 2-4): the non-speech complement of
    ``speech_spans_ms`` over [0, duration_ms], with any gap shorter than
    MIN_NONSPEECH_SPAN_MS dropped. No speech at all -> the whole clip is one
    non-speech span."""
    if duration_ms <= 0:
        return []
    speech = _merge_spans(speech_spans_ms)
    if not speech:
        return [(0, duration_ms)]

    spans: List[Tuple[int, int]] = []
    cursor = 0
    for s, e in speech:
        s = max(0, min(s, duration_ms))
        e = max(0, min(e, duration_ms))
        if s > cursor:
            spans.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < duration_ms:
        spans.append((cursor, duration_ms))

    return [(s, e) for s, e in spans if e - s >= MIN_NONSPEECH_SPAN_MS]


def non_speech_spans(file_id: str, duration_ms: int) -> List[Tuple[int, int]]:
    """Thin DB-touching wrapper: this file's transcript -> its non-speech
    spans. See non_speech_spans_from_speech for the actual (pure, unit-
    tested) logic."""
    return non_speech_spans_from_speech(_speech_segments_ms(file_id), duration_ms)
