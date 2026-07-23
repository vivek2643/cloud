"""
The deterministic caption resolver (captions.plan.md SS3/SS4): turns a
document's `captions` selection (`{style_id, enabled, base_style,
overrides}`) plus whatever perception signals are available (transcripts,
audio_features, cut_records) into `resolved.captions` -- the ONE track both
the preview overlay and the ASS export read, same "measure once, resolve
pure" split as `grade.resolver`/`grade.measure`.

`resolve_captions_for_document` is the impure entry point (batch-fetches
signals, mirrors `grade.measure.fetch_color_stats`'s pattern); everything it
calls into (`resolve_captions`, timing/placement/colour) is pure over
already-fetched dicts, cheap to call once per document resolve.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import psycopg

from app.config import get_settings
from app.services.l3.captions import colour as colour_mod
from app.services.l3.captions import placement as placement_mod
from app.services.l3.captions import styles as styles_mod
from app.services.l3.captions import timing as timing_mod

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Fetch (impure) -- batch, once per document resolve.
# --------------------------------------------------------------------------

def _pg() -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, autocommit=True)


def fetch_transcripts(file_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """file_id -> {"segments": [...], "fillers": [...]}. Missing entries
    (no transcript yet) are simply absent -- that file's spine layers
    produce no captions, never an error (fail-open, same contract as
    `grade.measure.fetch_color_stats`)."""
    ids = list({f for f in file_ids if f})
    if not ids:
        return {}
    with _pg() as conn:
        rows = conn.execute(
            "select file_id::text, segments, fillers from transcripts where file_id = any(%s::uuid[])",
            (ids,),
        ).fetchall()
    return {r[0]: {"segments": r[1] or [], "fillers": r[2] or []} for r in rows}


def fetch_audio_features(file_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """file_id -> {onsets_ms, bpm, is_musical, rms_db, prosody_hop_ms,
    silence_intervals}."""
    ids = list({f for f in file_ids if f})
    if not ids:
        return {}
    with _pg() as conn:
        rows = conn.execute(
            "select file_id::text, onsets_ms, bpm, is_musical, rms_db, prosody_hop_ms, silence_intervals"
            " from audio_features where file_id = any(%s::uuid[])",
            (ids,),
        ).fetchall()
    cols = ("onsets_ms", "bpm", "is_musical", "rms_db", "prosody_hop_ms", "silence_intervals")
    return {r[0]: dict(zip(cols, r[1:])) for r in rows}


def fetch_cut_records(file_ids: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    """file_id -> [cut_records row, ...] (SS2 shape), off the latest ingest
    run covering these files. Empty/absent when a file was never ingested
    via Cuts v3 -- placement then falls back to a default anchor rect
    (never a crash)."""
    ids = list({f for f in file_ids if f})
    if not ids:
        return {}
    from app.services.l3 import cuts_read
    run_id = cuts_read.latest_run_for_files(ids)
    if run_id is None:
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in cuts_read.rows_for_run(run_id, ids):
        out.setdefault(row["file_id"], []).append(row)
    return out


def fetch_color_stats_for_captions(file_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Re-exposed under this module for callers that only need captions
    (avoids importing the grade package just for `fetch_color_stats`)."""
    from app.services.l3.grade.measure import fetch_color_stats
    return fetch_color_stats(list(file_ids))


# --------------------------------------------------------------------------
# Style selection (document.captions -> an effective CaptionStyle)
# --------------------------------------------------------------------------

def effective_style(captions_selection: Optional[Dict[str, Any]]) -> Optional[styles_mod.CaptionStyle]:
    """None when captions are off/unselected (SS1.3 "no auto-apply"). A
    `base_style` snapshot (SS3, always sent by the frontend on selection --
    see captions-view.tsx) is preferred over an id lookup so a Suggested
    pick stays resolvable even after suggest.py's cache moves on; a bare
    `style_id` with no snapshot falls back to the Standards catalog."""
    if not captions_selection or not captions_selection.get("enabled"):
        return None
    base = captions_selection.get("base_style")
    if isinstance(base, dict) and base.get("style_id"):
        base_style = styles_mod.CaptionStyle.from_dict(base)
    else:
        style_id = captions_selection.get("style_id")
        base_style = styles_mod.get_standard(style_id) if style_id else None
    if base_style is None:
        return None
    return styles_mod.apply_overrides(base_style, captions_selection.get("overrides"))


# --------------------------------------------------------------------------
# Weld-run grouping (SS9 stability: one placement per run of welded cuts)
# --------------------------------------------------------------------------

def _covering_rows(rows: List[Dict[str, Any]], src_in_ms: int, src_out_ms: int) -> List[Dict[str, Any]]:
    return [r for r in rows if int(r["src_in_ms"]) < src_out_ms and int(r["src_out_ms"]) > src_in_ms]


def _first_covering(rows: List[Dict[str, Any]], src_in_ms: int) -> Optional[Dict[str, Any]]:
    for r in rows:
        if int(r["src_in_ms"]) <= src_in_ms < int(r["src_out_ms"]):
            return r
    return None


def _group_weld_runs(
    spine_layers: List[Dict[str, Any]], cut_records_by_file: Dict[str, List[Dict[str, Any]]]
) -> List[List[int]]:
    """Spine-layer indices grouped into runs: a new run starts unless the
    layer's OWN covering cut_record says `continuity.prev_contiguous` AND
    the previous layer is the same source file (a weld literally means "this
    cut continues the previous one on the same clip")."""
    runs: List[List[int]] = []
    for i, layer in enumerate(spine_layers):
        rows = cut_records_by_file.get(layer["source_file_id"], [])
        cover = _first_covering(rows, int(layer["src_in_ms"]))
        contiguous = bool((cover or {}).get("continuity", {}).get("prev_contiguous"))
        same_file_as_prev = i > 0 and spine_layers[i - 1]["source_file_id"] == layer["source_file_id"]
        if runs and contiguous and same_file_as_prev:
            runs[-1].append(i)
        else:
            runs.append([i])
    return runs


def _suppress_non_said(words: List[Dict[str, Any]], cut_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """SS10 "no captions on channel != said beats": drop words whose
    midpoint falls inside a covering cut_record whose channel isn't "said".
    No covering row at all -> keep the word (fail-open: no signal to
    suppress on, never worse than showing it)."""
    if not cut_rows:
        return words
    out = []
    for w in words:
        mid = (int(w["start_ms"]) + int(w["end_ms"])) // 2
        cover = _first_covering(cut_rows, mid)
        if cover is not None and cover.get("channel") not in (None, "said"):
            continue
        out.append(w)
    return out


# --------------------------------------------------------------------------
# The resolver
# --------------------------------------------------------------------------

def resolve_captions(
    resolved_timeline: Dict[str, Any],
    *,
    style: Optional[styles_mod.CaptionStyle],
    transcripts_by_file: Optional[Dict[str, Dict[str, Any]]] = None,
    audio_features_by_file: Optional[Dict[str, Dict[str, Any]]] = None,
    cut_records_by_file: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    aspect: str = "landscape",
) -> List[Dict[str, Any]]:
    """Pure: every signal is already-fetched. `resolved_timeline` is a
    `layers.ResolvedTimeline.to_dict()`-shaped dict (or the equivalent
    already baked onto `document["resolved"]`) -- captions ride on the SAME
    spine spans grading/compositing already agreed on, so a retime/trim
    that moves `resolved.video_layers` moves captions with it for free
    (retime-aware: no separate retime math needed here).

    Colour is fully deterministic now (caption_style_mvp.plan.md #3) --
    no `color_stats`/`grade` signal is needed to resolve it, unlike the
    pre-MVP `match_grade`/`palette_accent`/`high_contrast` sources."""
    if style is None:
        return []
    transcripts_by_file = transcripts_by_file or {}
    audio_features_by_file = audio_features_by_file or {}
    cut_records_by_file = cut_records_by_file or {}
    style_dict = style.to_dict()
    colour_spec = style_dict["colour"]
    resolved_colour = colour_mod.resolve_colour(
        colour_spec["colour_id"],
        outline_enabled=colour_spec["outline_enabled"], shadow_enabled=colour_spec["shadow_enabled"],
    )

    spine_layers = [
        v for v in (resolved_timeline.get("video_layers") or []) if v.get("kind") == "spine"
    ]
    spine_layers.sort(key=lambda v: v["prog_start_ms"])
    runs = _group_weld_runs(spine_layers, cut_records_by_file)

    events: List[Dict[str, Any]] = []
    for run in runs:
        run_layers = [spine_layers[i] for i in run]
        run_rows: List[Dict[str, Any]] = []
        for layer in run_layers:
            rows = cut_records_by_file.get(layer["source_file_id"], [])
            run_rows.extend(_covering_rows(rows, int(layer["src_in_ms"]), int(layer["src_out_ms"])))
        placement_result = placement_mod.resolve_placement(
            run_rows, position=style_dict["position"], aspect=aspect, safe_area=True,
        )
        box = placement_result["box"]

        for li, layer in enumerate(run_layers):
            file_id = layer["source_file_id"]
            transcript = transcripts_by_file.get(file_id)
            if not transcript:
                continue
            words = timing_mod.words_in_source_window(
                transcript.get("segments") or [], int(layer["src_in_ms"]), int(layer["src_out_ms"])
            )
            if not words:
                continue
            rows = cut_records_by_file.get(file_id, [])
            cover_rows = _covering_rows(rows, int(layer["src_in_ms"]), int(layer["src_out_ms"]))
            words = _suppress_non_said(words, cover_rows)
            if not words:
                continue

            af = audio_features_by_file.get(file_id) or {}
            # Successor for the readability-hold clamp: the next layer WITHIN
            # this run if any, else the next run's first layer, else None
            # (last caption of the whole timeline -- no clamp needed).
            if li + 1 < len(run_layers):
                next_start: Optional[int] = run_layers[li + 1]["prog_start_ms"]
            else:
                next_start = None
                cur_idx = run[-1]
                if cur_idx + 1 < len(spine_layers):
                    next_start = spine_layers[cur_idx + 1]["prog_start_ms"]

            evs = timing_mod.build_events(
                words,
                src_in_ms=int(layer["src_in_ms"]), prog_start_ms=int(layer["prog_start_ms"]),
                layer_prog_end_ms=int(layer["prog_end_ms"]),
                max_chars_per_line=style_dict["max_chars_per_line"], max_lines=style_dict["max_lines"],
                case=style_dict["case"],
                # Emphasis/beat-sync are internal-only now (caption_style_mvp.
                # plan.md: "do not expose ... beat sync"): "loudness" still
                # decides which word Pop/Bounce bounces; beat_sync is always
                # off (removed from the MVP suggestion path entirely).
                emphasis_mode="loudness", beat_sync=False,
                rms_db=af.get("rms_db"), rms_hop_ms=int(af.get("prosody_hop_ms") or 0),
                onsets_ms=af.get("onsets_ms"), is_musical=bool(af.get("is_musical")),
                next_event_start_ms=next_start,
            )
            for ev in evs:
                events.append({
                    **ev,
                    "box": box,
                    "style_ref": style_dict["style_id"],
                    "style": {**style_dict, "colour": {**colour_spec, **resolved_colour}},
                    "anim": style_dict["animation"],
                })
    events.sort(key=lambda e: e["prog_start_ms"])
    return events


def resolve_captions_for_document(
    document: Dict[str, Any], resolved_timeline: Dict[str, Any], aspect: str = "landscape"
) -> List[Dict[str, Any]]:
    """Impure entry point: reads `document["captions"]`, batch-fetches every
    signal the resolve needs, and calls the pure `resolve_captions`. Called
    from `render/tasks.resolve_document` -- same call site + "cheap, no
    ffmpeg" guarantee as the grade resolve (SS4)."""
    style = effective_style(document.get("captions"))
    if style is None:
        return []
    file_ids = list({
        v["source_file_id"] for v in (resolved_timeline.get("video_layers") or []) if v.get("kind") == "spine"
    })
    if not file_ids:
        return []
    try:
        transcripts = fetch_transcripts(file_ids)
        audio_features = fetch_audio_features(file_ids)
        cut_records = fetch_cut_records(file_ids)
    except Exception:
        logger.exception("resolve_captions_for_document: signal fetch failed (continuing without captions)")
        return []
    return resolve_captions(
        resolved_timeline, style=style,
        transcripts_by_file=transcripts, audio_features_by_file=audio_features,
        cut_records_by_file=cut_records, aspect=aspect,
    )
