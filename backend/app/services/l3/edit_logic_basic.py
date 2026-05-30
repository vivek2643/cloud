"""
Phase 3a edit logic: minimal v1 that works on L1 fields only.

Inputs:
  - ranked candidate shots from the query executor
  - structured query dict (for duration target + rhythm_lock)
  - optional onset grids from audio_features (for beat snapping)

Output:
  - ordered list of TimelineClip ready for FCP7 XML compilation

Phase 3b will extend this into edit_logic.py with jump-cut elimination,
reaction layouts, and identity-aware character logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings
from app.services.l3.query_executor import CandidateShot

logger = logging.getLogger(__name__)

BEAT_SNAP_TOLERANCE_MS = 150

# --- Sub-clip trimming defaults (L1.C) -----------------------------------
# A shot is considered "internally chaotic" when its intra_shot_variance
# (cosine distance between anchor and peak-motion SigLIP vectors) exceeds
# this threshold. Calibrated on dev footage; tune per-corpus as needed.
SUBCLIP_VARIANCE_THRESHOLD = 0.25
# Don't bother trimming shots shorter than this -- there's no useful clean
# sub-window left after removing the chaotic moment.
SUBCLIP_MIN_SHOT_MS = 2000
# Margin around the peak-motion timestamp to drop. Symmetric.
SUBCLIP_MARGIN_MS = 250
# Refuse to emit a sub-clip shorter than this (otherwise just keep the full shot).
SUBCLIP_MIN_KEEP_MS = 800


@dataclass
class TimelineClip:
    """One clip on the output timeline."""
    file_id: str
    file_name: str
    file_r2_key: str
    file_r2_proxy_key: Optional[str]
    source_in_ms: int
    source_out_ms: int
    timeline_start_ms: int
    timeline_end_ms: int
    score: float
    # When non-None, this clip was trimmed to skip an internal chaotic moment
    # at `trimmed_around_ms`. Useful for the audit log and the UI.
    trimmed_around_ms: Optional[int] = None
    # Editor-level metadata, only populated by the smart/chat path. The chat
    # endpoint surfaces these so the frontend can store them in conversation
    # history and re-feed them to Claude on the next turn.
    shot_id: Optional[str] = None
    role_in_edit: Optional[str] = None
    why: Optional[str] = None


def _load_onsets(file_ids: List[str]) -> Dict[str, List[int]]:
    """Pull onsets_ms grids for the given files (only meaningful for musical content)."""
    if not file_ids:
        return {}
    with psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row) as conn:
        cur = conn.execute(
            "select file_id, onsets_ms, is_musical from audio_features where file_id = any(%s)",
            (file_ids,),
        )
        out: Dict[str, List[int]] = {}
        for row in cur.fetchall():
            if row["is_musical"] and row["onsets_ms"]:
                out[str(row["file_id"])] = list(row["onsets_ms"])
        return out


def _snap_to_nearest_onset(ts_ms: int, onsets: List[int]) -> int:
    if not onsets:
        return ts_ms
    # Binary-search the nearest onset within tolerance
    best = ts_ms
    best_d = BEAT_SNAP_TOLERANCE_MS + 1
    for o in onsets:
        d = abs(o - ts_ms)
        if d < best_d:
            best_d = d
            best = o
        elif o - ts_ms > BEAT_SNAP_TOLERANCE_MS:
            break
    return best if best_d <= BEAT_SNAP_TOLERANCE_MS else ts_ms


def _trim_chaotic_subwindow(
    c: CandidateShot,
) -> tuple[int, int, Optional[int]]:
    """
    If shot has high intra_shot_variance and a known peak_motion_ms, return
    a (start, end, trimmed_around) tuple that skips the chaotic moment by
    keeping the longer of [shot_start, peak_motion - margin] vs.
    [peak_motion + margin, shot_end]. Otherwise return the full shot.

    This is the default behaviour; the L3 prompt parser can opt out by
    setting `preserve_full_shots=true` (e.g. for raw / archival cuts).
    """
    if c.intra_shot_variance is None or c.peak_motion_ms is None:
        return c.start_ms, c.end_ms, None
    if c.intra_shot_variance < SUBCLIP_VARIANCE_THRESHOLD:
        return c.start_ms, c.end_ms, None
    if (c.end_ms - c.start_ms) < SUBCLIP_MIN_SHOT_MS:
        return c.start_ms, c.end_ms, None
    if not (c.start_ms < c.peak_motion_ms < c.end_ms):
        return c.start_ms, c.end_ms, None

    pre_len = c.peak_motion_ms - SUBCLIP_MARGIN_MS - c.start_ms
    post_len = c.end_ms - (c.peak_motion_ms + SUBCLIP_MARGIN_MS)

    # Prefer the longer side. If neither is usable, fall back to full shot.
    if pre_len >= post_len and pre_len >= SUBCLIP_MIN_KEEP_MS:
        return c.start_ms, c.peak_motion_ms - SUBCLIP_MARGIN_MS, c.peak_motion_ms
    if post_len >= SUBCLIP_MIN_KEEP_MS:
        return c.peak_motion_ms + SUBCLIP_MARGIN_MS, c.end_ms, c.peak_motion_ms
    return c.start_ms, c.end_ms, None


def build_timeline(
    candidates: List[CandidateShot],
    query: Dict[str, Any],
) -> List[TimelineClip]:
    """
    Pick clips off the candidate list until we hit duration_target_s (or fall
    back to top-20 if no target). Snap boundaries to beats when rhythm_lock=true.
    Order by (file_id, source-time) to keep neighbouring shots contiguous.

    Duration target is treated as a HARD cap with a soft minimum:
      - if total < target, we keep every selected shot (even if total < target)
      - if total > target, we trim the last shot's tail so the timeline lands
        on exactly target (unless that would leave the trimmed clip below
        MIN_TRIM_TAIL_MS, in which case we drop that last shot entirely).

    Sub-clip trimming (L1.C):
      - By default, when a candidate's `intra_shot_variance` exceeds
        `SUBCLIP_VARIANCE_THRESHOLD`, the timeline emits only the longest
        clean sub-window of that shot, skipping the chaotic moment around
        `peak_motion_ms`.
      - The user (via the prompt parser) can disable this by setting
        `preserve_full_shots=true` -- useful for raw / archival outputs.
    """
    duration_target_ms: Optional[int] = None
    if query.get("duration_target_s"):
        duration_target_ms = int(query["duration_target_s"]) * 1000

    rhythm_lock = bool(query.get("rhythm_lock", False))
    onsets_by_file = _load_onsets(list({c.file_id for c in candidates})) if rhythm_lock else {}

    preserve_full = bool(query.get("preserve_full_shots", False))

    # Resolve effective in/out for each candidate up front (after sub-clip
    # trimming) so the duration accounting below uses the trimmed lengths.
    resolved: List[tuple[CandidateShot, int, int, Optional[int]]] = []
    for c in candidates:
        if preserve_full:
            s_in, s_out, trimmed_at = c.start_ms, c.end_ms, None
        else:
            s_in, s_out, trimmed_at = _trim_chaotic_subwindow(c)
        if s_out > s_in:
            resolved.append((c, s_in, s_out, trimmed_at))

    selected: List[tuple[CandidateShot, int, int, Optional[int]]] = []
    total_ms = 0
    for entry in resolved:
        clip_len = entry[2] - entry[1]
        selected.append(entry)
        total_ms += clip_len
        if duration_target_ms is not None and total_ms >= duration_target_ms:
            break

    if duration_target_ms is None and len(selected) > 20:
        selected = selected[:20]

    # Order by source order so adjacent clips in the same file stay together
    selected.sort(key=lambda x: (x[0].file_id, x[1]))

    timeline: List[TimelineClip] = []
    cursor_ms = 0
    MIN_TRIM_TAIL_MS = 600  # don't leave a clip shorter than this when trimming

    for c, s_in, s_out, trimmed_at in selected:
        if rhythm_lock and c.file_id in onsets_by_file:
            grid = onsets_by_file[c.file_id]
            sn_in = _snap_to_nearest_onset(s_in, grid)
            sn_out = _snap_to_nearest_onset(s_out, grid)
            if sn_out > sn_in:
                s_in, s_out = sn_in, sn_out

        clip_len = s_out - s_in

        # Hard-cap on the last clip so total == duration_target_ms exactly.
        if duration_target_ms is not None:
            remaining = duration_target_ms - cursor_ms
            if remaining <= 0:
                break
            if clip_len > remaining:
                if remaining < MIN_TRIM_TAIL_MS and timeline:
                    break
                s_out = s_in + remaining
                clip_len = remaining

        timeline.append(TimelineClip(
            file_id=c.file_id,
            file_name=c.file_name,
            file_r2_key=c.file_r2_key,
            file_r2_proxy_key=c.file_r2_proxy_key,
            source_in_ms=s_in,
            source_out_ms=s_out,
            timeline_start_ms=cursor_ms,
            timeline_end_ms=cursor_ms + clip_len,
            score=c.score,
            trimmed_around_ms=trimmed_at,
        ))
        cursor_ms += clip_len

        if duration_target_ms is not None and cursor_ms >= duration_target_ms:
            break

    return timeline
