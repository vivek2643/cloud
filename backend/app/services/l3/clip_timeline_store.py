"""
Clip Timeline store / loader.
=============================

Thin DB glue that assembles a :class:`clip_timeline.TimelineInputs` for a real
clip and fuses it into a :class:`clip_timeline.ClipTimeline`. It deliberately
**reuses the v1 loaders** (``hero_cuts._load_inputs`` for perception + motion +
audio grids, ``score_span.load_sources`` for words, ``atoms.build_atoms`` +
``hero_cuts._build_field_v2`` for the fused seam field) so the continuous
substrate reads exactly the same L1/L2 signals the ladder does -- no divergent
second copy of the loading logic.

The heavy lifting (change-point lanes, cut index) lives in ``clip_timeline``;
this module only maps rows into that pure builder. No new tables yet: timelines
are computed on demand (cheap -- pure Python over already-materialized signals)
and can be cached later behind ``CLIP_TIMELINE_VERSION`` if profiling asks for
it.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import psycopg

from app.config import get_settings
from app.services.l3 import atoms as atoms_mod
from app.services.l3 import hero_cuts, score_span
from app.services.l3.clip_timeline import (
    ClipTimeline, TimelineInputs, build_clip_timeline, render_awareness,
)

logger = logging.getLogger("l3.clip_timeline_store")

# Process-level cache: fusing a clip's continuous timeline (L1 inputs + the
# cuts-v2 seam field) costs ~seconds, and the v2 arranger reads it EVERY turn.
# Cache the built ClipTimeline keyed by (file_id, energy) with a cheap
# perception token (schema_version + updated marker); a lookup rebuilds only
# when the clip's perception actually changed (e.g. a re-analysis), so an edit
# session pays the cold build once and every later turn is instant. Not
# persisted across process restarts (a redeploy re-warms it on first use).
_TL_CACHE: Dict[Tuple[str, float], Tuple[str, ClipTimeline]] = {}


def _perception_token(file_id: str) -> Optional[str]:
    """A cheap validity token for a clip's perception -- schema_version +
    created_at. Changes iff the perception was (re)written, so it invalidates a
    stale cached timeline without re-fusing to check. None when unknown."""
    try:
        with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
            row = conn.execute(
                "select schema_version, created_at from clip_perception where file_id = %s",
                (file_id,),
            ).fetchone()
        if not row:
            return None
        return f"{row[0]}:{row[1].isoformat() if row[1] else ''}"
    except Exception:
        logger.exception("clip_timeline: perception token lookup failed for %s", file_id)
        return None


def load_timeline_inputs(file_id: str, *, energy: float = 0.5) -> Optional[TimelineInputs]:
    """Gather every fused signal for one clip into a TimelineInputs. None when
    the clip has no materialized L1/L2 yet."""
    clips = hero_cuts._load_inputs([file_id])
    clip = clips.get(file_id)
    if clip is None:
        return None

    src = score_span.load_source(file_id)
    words = list(src.words) if src else []

    # Reuse the exact cuts-v2 fused seam field (atom peaks protected) so the
    # continuous index snaps to the same clean boundaries the ladder uses.
    try:
        atom_list = atoms_mod.build_atoms(clip)
        field = hero_cuts._build_field_v2(clip, energy, atom_list)
    except Exception:
        logger.exception("clip_timeline: field build failed for %s", file_id)
        field = None

    perc = clip.perception or {}
    audio = clip.audio or {}
    motion = clip.motion or {}
    return TimelineInputs(
        file_id=file_id,
        duration_ms=clip.duration_ms,
        words=words,
        rms_db=list(audio.get("rms_db") or []),
        prosody_hop_ms=int(audio.get("prosody_hop_ms") or 0),
        persons=list(perc.get("persons") or []),
        speaking=list(perc.get("speaking") or []),
        gaze=list(perc.get("gaze") or []),
        camera_craft=list(perc.get("camera_craft") or []),
        atoms=list(perc.get("atoms") or []),
        quality_events=list(perc.get("take_quality_events") or []),
        action_points=list(motion.get("action_points") or []),
        presence_lane=list(perc.get("presence_lane") or []),
        activity_lane=list(perc.get("activity_lane") or []),
        field=field,
    )


def load_clip_timeline(file_id: str, *, energy: float = 0.5,
                       use_cache: bool = True) -> Optional[ClipTimeline]:
    """Build (or reuse) the continuous ClipTimeline for one clip. Cached in
    process behind a cheap perception token so repeated turns don't re-fuse."""
    key = (file_id, float(energy))
    token = _perception_token(file_id) if use_cache else None
    if use_cache and token is not None:
        hit = _TL_CACHE.get(key)
        if hit is not None and hit[0] == token:
            return hit[1]

    inputs = load_timeline_inputs(file_id, energy=energy)
    if inputs is None:
        return None
    tl = build_clip_timeline(inputs)
    if use_cache and token is not None:
        _TL_CACHE[key] = (token, tl)
    return tl


def load_clip_timelines(file_ids: List[str], *, energy: float = 0.5) -> List[ClipTimeline]:
    out: List[ClipTimeline] = []
    for fid in file_ids:
        tl = load_clip_timeline(fid, energy=energy)
        if tl is not None:
            out.append(tl)
    return out


def awareness_digest(file_ids: List[str], *, energy: float = 0.5) -> str:
    """The continuous-timeline awareness block for every clip in scope --
    the brain's read of the fully-addressable source."""
    blocks = [render_awareness(tl) for tl in load_clip_timelines(file_ids, energy=energy)]
    return "\n\n".join(blocks)
