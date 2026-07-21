"""
Caption endpoints (captions.plan.md SS13): mirrors `routers/grade.py`'s
split -- a global, un-scoped catalog listing, and a thread-scoped
generation endpoint that reads the SAME signals `captions.resolver` reads
(cut_records/audio_features/color_stats), so a suggestion is never out of
sync with what the resolver would actually produce for that edit.

  GET /api/captions/catalog                          fonts + Standards tiles
  GET /api/captions/suggestions?thread_id=&version=   5 Suggested bundles +
                                                       rationale + a shared
                                                       representative frame
                                                       + sample words (SS6/SS8)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from app.auth import get_current_user_id
from app.services.l3 import store
from app.services.l3.captions import styles as styles_mod
from app.services.l3.captions import suggest as suggest_mod
from app.services.l3.captions import timing as timing_mod
from app.services.l3.captions.colour import resolve_colour
from app.services.l3.captions.resolver import (
    fetch_audio_features,
    fetch_color_stats_for_captions,
    fetch_cut_records,
    fetch_transcripts,
)
from app.services.render.compositor import presigned_url_for

logger = logging.getLogger(__name__)
router = APIRouter(tags=["captions"])


@router.get("/api/captions/catalog")
def get_catalog(_user_id: str = Depends(get_current_user_id)) -> dict:
    """Tier 2 "Standards" listing (SS7) + the curated font set (SS5) -- no
    thread context needed, same un-scoped-catalog shape as
    `GET /api/grade/presets`."""
    return {"fonts": styles_mod.list_fonts(), "standards": styles_mod.list_standards()}


def _owned_thread(thread_id: str, user_id: str) -> dict:
    thread = store.get_thread(thread_id)
    if thread is None or thread["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="Thread not found")
    return thread


def _pick_representative_cut(cut_rows_all: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """SS16 "representative-frame selection": the cut with the biggest
    stable `caption_zone` among high-`total_quality`, on-camera, non-junk
    cuts. `hero_key` (already an R2 JPEG of this exact cut at its
    `hero_ts_ms`, extracted at ingest -- see `l3.frames.extract_still`) is
    reused as-is rather than re-decoding a frame (SS8's own suggested
    reuse-path, `l1.color_stats._decode_rgb_frame_at`, turns out to be a
    private single-purpose helper with no other callers; `hero_key` is the
    already-public, already-produced equivalent for exactly this need)."""
    candidates = [r for r in cut_rows_all if r.get("caption_zones") and not r.get("junk") and r.get("hero_key")]
    if not candidates:
        return None
    def biggest_zone_area(r: Dict[str, Any]) -> float:
        zones = r.get("caption_zones") or []
        return max((z[2] * z[3] for z in zones if len(z) == 4), default=0.0)
    on_camera_first = [r for r in candidates if r.get("on_camera")]
    pool = on_camera_first or candidates
    pool.sort(key=lambda r: (float(r.get("total_quality") or 0.0), biggest_zone_area(r)), reverse=True)
    return pool[0]


def _sample_words(cut: Dict[str, Any], transcripts_by_file: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The representative cut's own first punchy line (SS1.5 "real words
    from the edit's first punchy line") -- reuses `timing.build_events`
    against JUST this one cut's span so tiles get the exact same
    line-break/readability logic a real caption would, without depending on
    which style is being previewed (word list is style-independent; only
    line-WRAPPING varies by style, which the frontend tile does itself at
    render time against a generous default budget)."""
    transcript = transcripts_by_file.get(cut["file_id"])
    if not transcript:
        return []
    words = timing_mod.words_in_source_window(
        transcript.get("segments") or [], int(cut["src_in_ms"]), int(cut["src_out_ms"])
    )
    if not words:
        return []
    events = timing_mod.build_events(
        words, src_in_ms=int(cut["src_in_ms"]), prog_start_ms=0, layer_prog_end_ms=int(cut["src_out_ms"]) - int(cut["src_in_ms"]),
        max_chars_per_line=32, max_lines=1, case="as-is", emphasis_mode="loudness", beat_sync=False,
    )
    if not events:
        return []
    return events[0]["lines"][0]["words"]


@router.get("/api/captions/suggestions")
def get_suggestions(
    thread_id: str = Query(...),
    version: Optional[int] = Query(None),
    reshuffle_seed: int = Query(0),
    user_id: str = Depends(get_current_user_id),
) -> dict:
    _owned_thread(thread_id, user_id)
    from app.services.render.tasks import _load_document_version, resolve_document

    doc = _load_document_version(thread_id, version) if version is not None else None
    if doc is None:
        doc, _v = store.latest_document(thread_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="No edit document yet")

    resolved = resolve_document(doc, thread_id=thread_id)
    aspect = str(resolved.get("aspect") or "landscape")
    file_ids = list({
        v["source_file_id"] for v in (resolved.get("video_layers") or []) if v.get("kind") == "spine"
    })
    if not file_ids:
        return {"suggestions": [], "representative_frame": None, "sample_words": []}

    try:
        cut_records = fetch_cut_records(file_ids)
        audio_features = fetch_audio_features(file_ids)
        color_stats = fetch_color_stats_for_captions(file_ids)
        transcripts = fetch_transcripts(file_ids)
    except Exception:
        logger.exception("captions suggestions: signal fetch failed")
        raise HTTPException(status_code=500, detail="Could not load footage signals")

    grade_label = None
    look = doc.get("look") or {}
    if look.get("mode") == "preset" and look.get("preset_id"):
        grade_label = look["preset_id"].replace("_", " ")

    bundles = suggest_mod.generate_suggestions(
        resolved, cut_records_by_file=cut_records, audio_features_by_file=audio_features,
        color_stats_by_file=color_stats, aspect=aspect, grade_label=grade_label,
        reshuffle_seed=reshuffle_seed,
    )

    cut_rows_all = [r for rows in cut_records.values() for r in rows]
    rep_cut = _pick_representative_cut(cut_rows_all)
    representative_frame = None
    sample_words: List[Dict[str, Any]] = []
    if rep_cut is not None:
        zones = rep_cut.get("caption_zones") or []
        best_zone = max(zones, key=lambda z: z[2] * z[3]) if zones else None
        try:
            url = presigned_url_for(rep_cut["hero_key"])
        except Exception:
            logger.exception("captions suggestions: could not presign hero frame")
            url = None
        representative_frame = {
            "url": url,
            "hero_ts_ms": rep_cut.get("hero_ts_ms"),
            "caption_zone": list(best_zone) if best_zone else None,
            "subject_box": (rep_cut.get("framing") or {}).get("subject_box"),
        }
        sample_words = _sample_words(rep_cut, transcripts)
        # Contrast-resolve every suggestion's colours against the REP cut's
        # own footage (SS1.6 "Standards tiles ride the same representative
        # frame") so a tile's colours are already legible, not a generic
        # guess the frontend would have to re-derive.
        cs = color_stats.get(rep_cut["file_id"])
        for b in bundles:
            b["colour"] = {**b["colour"], **resolve_colour(b["colour"], color_stats=cs, grade=None)}

    return {
        "suggestions": bundles,
        "representative_frame": representative_frame,
        "sample_words": sample_words,
    }
