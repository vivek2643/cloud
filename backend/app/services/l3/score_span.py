"""
Deterministic, windowed quality metrics for take selection.

The rule (see the take-selection design): quality is a FUNCTION OVER A TIME
RANGE, never a precomputed clip scalar. Everything here is pure math over data
L1/L2 already produced (Whisper words + fillers, silence intervals, loudness,
L2 gaze), so the same span can be scored whether it is a whole clip, one
sentence, or one of several attempts inside a single clip.

These metrics are comparable across clips *by construction* (same yardstick
every time), which is exactly why they -- not the VLM's per-clip judgement --
carry the ranking. The VLM's `take_quality_events` are layered on top by the
caller for the subjective dimensions only it can see.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.config import get_settings


# --------------------------------------------------------------------------
# Source bundle: everything needed to score any window of one clip, loaded once
# --------------------------------------------------------------------------

@dataclass
class SpanSource:
    file_id: str
    duration_ms: int
    words: List[dict] = field(default_factory=list)      # {start_ms,end_ms,text,is_filler}
    fillers: List[dict] = field(default_factory=list)    # {start_ms,end_ms,word}
    silences: List[dict] = field(default_factory=list)   # {start_ms,end_ms}
    gaze: List[dict] = field(default_factory=list)        # {start_ms,end_ms,direction}
    quality_events: List[dict] = field(default_factory=list)  # VLM take_quality_events
    integrated_lufs: Optional[float] = None
    true_peak_db: Optional[float] = None


def quality_events_in(source: "SpanSource", start_ms: int, end_ms: int) -> List[dict]:
    """The VLM's localized quality judgements overlapping a window (subjective
    dimensions only it can see; layered on top of the objective metrics)."""
    return [
        {
            "dimension": q.get("dimension"),
            "score": q.get("score"),
            "evidence": q.get("evidence"),
        }
        for q in source.quality_events
        if _overlap_ms(int(q.get("start_ms", 0)), int(q.get("end_ms", 0)), start_ms, end_ms) > 0
    ]


def _pg_conn():
    import psycopg
    settings = get_settings()
    return psycopg.connect(settings.database_url, autocommit=True)


def _as_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return []
    return []


def _as_doc(v: Any) -> Optional[dict]:
    if v is None:
        return None
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return None
    return None


def load_sources(file_ids: List[str]) -> Dict[str, SpanSource]:
    """Load span-scoring inputs for several clips in one query."""
    if not file_ids:
        return {}
    out: Dict[str, SpanSource] = {}
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select f.id::text,
                   coalesce(f.duration_seconds, 0),
                   t.segments, t.fillers,
                   af.silence_intervals, af.integrated_lufs, af.true_peak_db,
                   cp.perception
              from files f
              left join transcripts t     on t.file_id  = f.id
              left join audio_features af  on af.file_id = f.id
              left join clip_perception cp on cp.file_id = f.id
             where f.id = any(%s::uuid[])
            """,
            (file_ids,),
        ).fetchall()

    for fid, dur_s, segments, fillers, silences, lufs, peak, perception in rows:
        words: List[dict] = []
        for seg in _as_list(segments):
            for w in seg.get("words", []) or []:
                words.append(w)
        doc = _as_doc(perception) or {}
        out[fid] = SpanSource(
            file_id=fid,
            duration_ms=int(float(dur_s) * 1000),
            words=words,
            fillers=_as_list(fillers),
            silences=_as_list(silences),
            gaze=list(doc.get("gaze") or []),
            quality_events=list(doc.get("take_quality_events") or []),
            integrated_lufs=float(lufs) if lufs is not None else None,
            true_peak_db=float(peak) if peak is not None else None,
        )
    return out


def load_source(file_id: str) -> Optional[SpanSource]:
    return load_sources([file_id]).get(file_id)


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

def _overlap_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _in_window(item: dict, start_ms: int, end_ms: int) -> bool:
    s = int(item.get("start_ms", item.get("ms", 0)))
    e = int(item.get("end_ms", s))
    return _overlap_ms(s, e, start_ms, end_ms) > 0


def score_span(source: SpanSource, start_ms: int, end_ms: int) -> Dict[str, Any]:
    """Objective quality vector for one [start, end] window of a clip.

    All rates are normalized so windows of different lengths compare fairly.
    Fields are best-effort: a silent clip simply reports zero speech metrics.
    """
    start_ms = max(0, int(start_ms))
    end_ms = max(start_ms + 1, int(end_ms))
    dur = end_ms - start_ms
    dur_min = dur / 60000.0

    win_words = [w for w in source.words if _in_window(w, start_ms, end_ms)]
    real_words = [w for w in win_words if not w.get("is_filler")]
    word_count = len(real_words)
    filler_count = sum(1 for f in source.fillers if _in_window(f, start_ms, end_ms))

    # Pauses: how much of the window is silence, and the worst single gap.
    paused = 0
    longest_pause = 0
    for s in source.silences:
        ov = _overlap_ms(int(s.get("start_ms", 0)), int(s.get("end_ms", 0)), start_ms, end_ms)
        if ov > 0:
            paused += ov
            longest_pause = max(longest_pause, ov)
    pause_ratio = round(paused / dur, 4) if dur else 0.0
    speech_ratio = round(max(0.0, 1.0 - pause_ratio), 4)

    total_words = word_count + filler_count
    metrics: Dict[str, Any] = {
        "duration_ms": dur,
        "word_count": word_count,
        "wpm": round(word_count / dur_min, 1) if dur_min > 0 else 0.0,
        "filler_count": filler_count,
        "filler_per_min": round(filler_count / dur_min, 2) if dur_min > 0 else 0.0,
        "filler_ratio": round(filler_count / total_words, 4) if total_words else 0.0,
        "pause_ratio": pause_ratio,
        "longest_pause_ms": longest_pause,
        "speech_ratio": speech_ratio,
    }

    # Gaze-to-camera fraction (only meaningful when the VLM logged gaze).
    if source.gaze:
        cam = 0
        seen = 0
        for g in source.gaze:
            ov = _overlap_ms(int(g.get("start_ms", 0)), int(g.get("end_ms", 0)), start_ms, end_ms)
            if ov > 0:
                seen += ov
                if g.get("direction") == "to_camera":
                    cam += ov
        if seen > 0:
            metrics["gaze_to_camera_ratio"] = round(cam / seen, 4)

    if source.integrated_lufs is not None:
        metrics["integrated_lufs"] = round(source.integrated_lufs, 1)
    if source.true_peak_db is not None:
        metrics["true_peak_db"] = round(source.true_peak_db, 1)

    return metrics


# A single dead gap longer than this mid-line reads as a stumble / "stuck
# between words", not natural phrasing -- the thing that makes an otherwise
# fine sentence unusable. Penalty saturates here.
_DELIVERY_LONG_PAUSE_MS = 1500


def delivery_score(metrics: Dict[str, Any]) -> float:
    """Deterministic FLUENCY of a spoken span, 0..1 (higher = smoother delivery).

    The objective half of best-take selection: how cleanly the line is *said*,
    independent of WHAT is said. Penalizes fillers, a long mid-line dead gap (a
    stumble / the speaker getting stuck between words), too much overall silence,
    and a runaway/halting pace. Pure over the ``score_span`` metric vector, so it
    is comparable across any two spans by construction.
    """
    fillers = float(metrics.get("filler_per_min", 0.0))
    pause_ratio = float(metrics.get("pause_ratio", 0.0))
    longest_pause = float(metrics.get("longest_pause_ms", 0.0))
    wpm = float(metrics.get("wpm", 0.0))

    filler_pen = min(1.0, fillers / 12.0)                       # ~12/min -> full
    long_pause_pen = min(1.0, longest_pause / _DELIVERY_LONG_PAUSE_MS)
    pause_pen = min(1.0, pause_ratio / 0.5)                     # >50% silence -> full
    pace_pen = 0.0 if 90 <= wpm <= 190 else min(1.0, abs(wpm - 140) / 140.0)

    fluency = (
        1.0
        - 0.35 * filler_pen
        - 0.30 * long_pause_pen
        - 0.20 * pause_pen
        - 0.15 * pace_pen
    )
    return max(0.0, min(1.0, fluency))
