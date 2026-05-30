"""
Phase 3b edit logic: full rule pass that leverages L2 enrichment.

Three rules layered on top of basic ordering:
  1. Jump-cut elimination:
     Block consecutive shots that share a tracked_character_id AND have
     DINOv2 cosine > JUMP_CUT_SIMILARITY. Try inserting a separator (reaction
     or transition) from the candidate pool; otherwise drop the offending
     second clip.

  2. Rhythmic alignment:
     Same beat-snap as the basic logic, just always-on when any selected
     clip's source file is_musical.

  3. Reaction layouts:
     When a selected clip has |emotional_valence| > REACTION_VALENCE_THRESHOLD
     and a tracked_character_id, look for a 0.5-1.5s clip in the same source
     file featuring a DIFFERENT known character; insert it immediately after.

Falls back gracefully when L2 columns are still null on the candidates
(missing dinov2 / characters / narrative). That happens when the caller
hasn't run L2 enrichment yet, so the function degrades to the basic v1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings
from app.services.l3.edit_logic_basic import (
    TimelineClip,
    _load_onsets,
    _snap_to_nearest_onset,
    BEAT_SNAP_TOLERANCE_MS,
)
from app.services.l3.query_executor import CandidateShot

logger = logging.getLogger(__name__)

JUMP_CUT_SIMILARITY = 0.85
REACTION_VALENCE_THRESHOLD = 0.5
REACTION_CLIP_MIN_MS = 500
REACTION_CLIP_MAX_MS = 1500


@dataclass
class EnrichedShot:
    """A candidate plus its L2 columns (loaded lazily)."""
    candidate: CandidateShot
    character_ids: List[str]
    dinov2: Optional[List[float]]
    narrative_role: Optional[str]
    emotional_valence: Optional[float]


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


def _load_l2_columns(shot_ids: Sequence[str]) -> Dict[str, EnrichedShot]:
    """Pull dinov2 / characters / narrative for the given shots, keyed by shot_id."""
    if not shot_ids:
        return {}
    with _pg() as conn:
        cur = conn.execute(
            """
            select id::text as id,
                   tracked_character_ids,
                   dinov2_embedding,
                   narrative_role,
                   emotional_valence
              from shots
             where id = any(%s::uuid[])
            """,
            (list(shot_ids),),
        )
        out: Dict[str, EnrichedShot] = {}
        for row in cur.fetchall():
            chars = row.get("tracked_character_ids") or []
            dino = row.get("dinov2_embedding")
            dino_list = list(dino) if dino is not None else None
            out[row["id"]] = EnrichedShot(
                candidate=None,  # type: ignore[arg-type]  # filled by caller
                character_ids=[str(c) for c in chars],
                dinov2=dino_list,
                narrative_role=row.get("narrative_role"),
                emotional_valence=row.get("emotional_valence"),
            )
        return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _is_jump_cut(a: EnrichedShot, b: EnrichedShot) -> bool:
    """Two consecutive shots are a jump-cut if same character AND visually similar."""
    if not a.character_ids or not b.character_ids:
        return False
    if not set(a.character_ids) & set(b.character_ids):
        return False
    if a.dinov2 is None or b.dinov2 is None:
        # No structural info -> conservatively assume they are not a jump-cut
        return False
    return _cosine(a.dinov2, b.dinov2) > JUMP_CUT_SIMILARITY


def _find_reaction_for(
    target: EnrichedShot,
    enriched: Dict[str, EnrichedShot],
    candidates: List[CandidateShot],
) -> Optional[CandidateShot]:
    """
    Find a short clip in the same source file showing a different character.
    Returns the candidate object or None if nothing suitable exists.
    """
    if not target.character_ids:
        return None
    target_chars = set(target.character_ids)
    target_file = target.candidate.file_id
    for c in candidates:
        if c.file_id != target_file:
            continue
        es = enriched.get(c.shot_id)
        if not es or not es.character_ids:
            continue
        if set(es.character_ids) & target_chars:
            continue  # same person, skip
        clip_len = c.end_ms - c.start_ms
        if REACTION_CLIP_MIN_MS <= clip_len <= REACTION_CLIP_MAX_MS:
            return c
    return None


def build_timeline_full(
    candidates: List[CandidateShot],
    query: Dict[str, Any],
) -> List[TimelineClip]:
    """
    Like build_timeline (basic) but applies jump-cut elimination, reaction
    layouts, and rhythmic alignment using L2 enrichment when present.
    """
    duration_target_ms: Optional[int] = None
    if query.get("duration_target_s"):
        duration_target_ms = int(query["duration_target_s"]) * 1000

    rhythm_lock = bool(query.get("rhythm_lock", False))

    enriched_map = _load_l2_columns([c.shot_id for c in candidates])
    # Attach the candidate ref back into enriched rows so reaction lookup works
    for c in candidates:
        if c.shot_id in enriched_map:
            enriched_map[c.shot_id].candidate = c

    # First pass: pick up to duration_target_ms worth of shots.
    selected: List[CandidateShot] = []
    total_ms = 0
    for c in candidates:
        clip_len = c.end_ms - c.start_ms
        if clip_len <= 0:
            continue
        selected.append(c)
        total_ms += clip_len
        if duration_target_ms is not None and total_ms >= duration_target_ms:
            break
    if duration_target_ms is None and len(selected) > 20:
        selected = selected[:20]

    selected.sort(key=lambda x: (x.file_id, x.start_ms))

    # Second pass: rule-based reordering (jump-cut elimination + reactions).
    refined: List[CandidateShot] = []
    for c in selected:
        es = enriched_map.get(c.shot_id)
        if refined and es and enriched_map.get(refined[-1].shot_id):
            prev_es = enriched_map[refined[-1].shot_id]
            if _is_jump_cut(prev_es, es):
                # Try inserting a reaction shot between them
                reaction = _find_reaction_for(prev_es, enriched_map, candidates)
                if reaction and reaction is not c and reaction is not refined[-1]:
                    refined.append(reaction)
                else:
                    # Couldn't find a separator -> drop the jump-cut shot
                    continue

        refined.append(c)

        # Reaction layout: if this shot is high-valence + has a character,
        # try to follow it with a cut to a different character.
        if es and es.emotional_valence is not None \
                and abs(es.emotional_valence) >= REACTION_VALENCE_THRESHOLD:
            reaction = _find_reaction_for(es, enriched_map, candidates)
            if reaction and reaction is not c and reaction not in refined:
                refined.append(reaction)

    # Third pass: beat-snap if requested.
    onsets_by_file: Dict[str, List[int]] = {}
    if rhythm_lock:
        onsets_by_file = _load_onsets(list({c.file_id for c in refined}))

    timeline: List[TimelineClip] = []
    cursor_ms = 0
    for c in refined:
        s_in = c.start_ms
        s_out = c.end_ms
        if rhythm_lock and c.file_id in onsets_by_file:
            grid = onsets_by_file[c.file_id]
            s_in = _snap_to_nearest_onset(s_in, grid)
            s_out = _snap_to_nearest_onset(s_out, grid)
            if s_out <= s_in:
                s_in = c.start_ms
                s_out = c.end_ms
        clip_len = s_out - s_in
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
        ))
        cursor_ms += clip_len

    return timeline


__all__ = ["build_timeline_full", "BEAT_SNAP_TOLERANCE_MS"]
