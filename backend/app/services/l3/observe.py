"""
Observe: the brain's deterministic SENSES over an edit -- no VLM, no LLM.

Where ``act`` mutates the document, ``observe`` reads it and reports, so the
agentic loop can perceive -> act -> re-perceive cheaply (a coding agent reading
the repo before/after an edit). Every function is a pure projection of the
document + a per-turn ``EditContext`` (footage map, source durations) assembled
ONCE by ``build_context``. Most of the DB access happens there; a couple of
senses (``review``, and ``read_state`` when asked for a ``seg_id`` detail) do
their own SMALL, on-demand transcript fetch (edso_pacing_audit_timing.plan.md
item 3/6) rather than loading every file's words into ``EditContext`` up
front. The senses:

  * read_state  -- what the edit IS now (cuts, channels, duration, feel, the
                   z-stack); optionally one cut's words resolved to program time.
  * predict     -- what a proposed change WOULD do (e.g. length after tightening)
                   without applying it.
  * validate    -- structural legality (spans in range, refs real, no bad ops).
  * diagnose    -- editorial problems worth fixing (sags, jump-cuts, over length,
                   redundant takes) -- read off feel + the map.
  * affordances -- what CAN be done, and to what (tighter takes, alternates,
                   mute toggles, channels) -- the brain's menu.
  * audio_state -- the audio digest (beds, outlook-authoritative runs, unplaced assets).
  * review      -- the ASSEMBLED program read back per cut (played text, word
                   offsets) + presented (never prescribed) rough-head/tail and
                   overlay-fit flags.

Feel is delegated to ``feel.simulate`` (also pure). Nothing here renders.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3 import feel, footage_map, framing, layers
from app.services.l3.arrange import _MapIndex, _weld_segments
from app.services.l3.captions import resolver as captions_resolver
from app.services.l3.captions import timing as captions_timing
from app.services.l3.grade.steer import explain_grade

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Per-turn context (the ONLY DB access)
# --------------------------------------------------------------------------

@dataclass
class EditContext:
    file_ids: List[str]
    index: _MapIndex
    map_struct: dict
    durations: Dict[str, int]          # file_id -> ms
    color_stats: Dict[str, dict] = field(default_factory=dict)  # file_id -> L1 color_stats row
    dup_groups: List[dict] = field(default_factory=list)
    # L1 audio_features (audio_brain.plan.md) for THIS turn's file_ids -- loudness,
    # musicality, bpm, onsets. file_id -> {integrated_lufs, is_musical, bpm,
    # onsets_ms, silence_intervals} (missing entry -> that file has no L1 audio
    # analysis yet; callers treat that as "no fact", never an error).
    audio_features: Dict[str, dict] = field(default_factory=dict)
    # This user's uploaded audio-type files available to place as a bed --
    # NOT limited to file_ids (an uploaded music/SFX file may never have been
    # used as a video source). {file_id, name, dur_ms, is_musical, bpm}. Whether
    # one is already placed in THIS document is a per-document fact the digest
    # (observe.audio_state) computes, not something build_context can know.
    audio_assets: List[dict] = field(default_factory=list)
    # The Cuts v3 ingest run this turn is resolved against -- the thread's pinned
    # run (migration 028) or the live "latest covering run", resolved ONCE here so
    # every projection in the turn (the map struct + the re-assembled BEAT INDEX
    # text) reads the SAME snapshot. None when nothing is ingested yet.
    run_id: Optional[str] = None

    @property
    def meta_by_ref(self) -> Dict[str, dict]:
        """moment_id -> moment node (speaker / channel / variants / flags)."""
        return self.index.moments


def build_context(file_ids: List[str], run_id: Optional[str] = None) -> EditContext:
    """Assemble the per-turn context once (footage map + durations). All DB reads
    live here; the sense functions stay pure over the result.

    ``run_id`` is the thread's pinned Cuts v3 ingest run (migration 028). We
    resolve the EFFECTIVE run once here -- pinned when given, else the latest
    covering run -- and thread that concrete id through every projection so the
    map struct and the re-assembled BEAT INDEX text agree within the turn (and a
    re-ingest mid-turn can't swap the snapshot between the two reads)."""
    eff_run = run_id
    if eff_run is None and file_ids:
        try:
            from app.services.l3 import cuts_v3_read
            eff_run = cuts_v3_read.latest_run_for_files(file_ids)
        except Exception:
            logger.exception("observe: run resolve failed (continuing live)")
    fmap = footage_map.assemble_map(file_ids, run_id=eff_run) if file_ids else {}
    map_struct = fmap.get("struct") or {"clips": []}
    durations: Dict[str, int] = {}
    try:
        from app.services.render.tasks import _durations
        durations = _durations(file_ids)
    except Exception:
        logger.exception("observe: duration lookup failed (continuing)")
    color_stats: Dict[str, dict] = {}
    try:
        from app.services.l3.grade.measure import fetch_color_stats
        color_stats = fetch_color_stats(file_ids)
    except Exception:
        logger.exception("observe: color_stats lookup failed (continuing)")
    audio_assets: List[dict] = []
    try:
        audio_assets = _fetch_audio_assets(file_ids)
    except Exception:
        logger.exception("observe: audio_assets lookup failed (continuing)")
    audio_features: Dict[str, dict] = {}
    try:
        # Cover both this turn's video/audio file_ids AND every uploaded audio
        # asset in scope (a music/SFX file may never have been a file_id) --
        # onsets/bpm feed the beat grid regardless of which the brain places.
        all_audio_ids = list(file_ids) + [a["file_id"] for a in audio_assets]
        audio_features = _fetch_audio_features(all_audio_ids)
    except Exception:
        logger.exception("observe: audio_features lookup failed (continuing)")
    return EditContext(
        file_ids=list(file_ids),
        index=_MapIndex(map_struct),
        map_struct=map_struct,
        durations=durations,
        color_stats=color_stats,
        dup_groups=fmap.get("dup_groups") or [],
        audio_features=audio_features,
        audio_assets=audio_assets,
        run_id=eff_run,
    )


def _pg_conn():
    import psycopg
    from app.config import get_settings
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _fetch_audio_features(file_ids: List[str]) -> Dict[str, dict]:
    """L1 audio_features rows for this turn's file_ids, keyed by file_id --
    feeds per-cut `loudness_rel` (a fact, never auto-normalized) and lets a
    VIDEO source double as a musical bed's `is_musical`/`bpm` when it's used
    for one (audio_brain.plan.md SS1c/2b). A file with no row simply has no
    entry -- observe treats that as "no fact", not zero/silence."""
    if not file_ids:
        return {}
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select file_id::text, integrated_lufs, is_musical, bpm, onsets_ms
              from audio_features where file_id = any(%s::uuid[])
            """,
            (file_ids,),
        ).fetchall()
    return {
        r[0]: {"integrated_lufs": r[1], "is_musical": r[2], "bpm": r[3], "onsets_ms": r[4]}
        for r in rows
    }


def _fetch_audio_assets(file_ids: List[str]) -> List[dict]:
    """This user's ready, audio-type files -- candidates `place_audio` can use
    that aren't necessarily among this turn's file_ids (a music/SFX file is
    uploaded for scoring, never as a video source). Scoped to the SAME user as
    the current file_ids (whoever owns this edit); an unknown/empty file_ids
    or no matching user -> no assets, never a fabricated list."""
    if not file_ids:
        return []
    with _pg_conn() as conn:
        owner = conn.execute(
            "select user_id from files where id = any(%s::uuid[]) limit 1",
            (file_ids,),
        ).fetchone()
        if not owner:
            return []
        rows = conn.execute(
            """
            select f.id::text, f.filename,
                   coalesce(f.duration_seconds, 0),
                   af.is_musical, af.bpm
              from files f
              left join audio_features af on af.file_id = f.id
             where f.user_id = %s and f.file_type = 'audio' and f.status = 'ready'
             order by f.created_at desc
            """,
            (owner[0],),
        ).fetchall()
    return [
        {"file_id": r[0], "name": r[1], "dur_ms": int(float(r[2]) * 1000),
         "is_musical": r[3], "bpm": r[4]}
        for r in rows
    ]


def resolve_doc(document: dict, ctx: EditContext) -> dict:
    """Finalize the working document (in place) after a batch of acts, so preview
    == render and the timeline reads cleanly:
      1. WELD the main line -- merge adjacent same-clip source-contiguous cuts
         into one segment (act appends slices raw; welding removes the redundant
         hard cuts, matching the old compile path). Jump-cuts / distant slices
         stay separate by construction.
      2. bake the reframe transform, then resolve the flat layer set.
    Reuses the same resolve the manual-edit path uses."""
    old_ids = [s.get("seg_id") for s in (document.get("timeline") or [])]
    old_segs = list(document.get("timeline") or [])
    document["timeline"] = _weld_segments(document.get("timeline") or [])
    _remap_split_edits(document, old_ids, old_segs)
    try:
        framing.annotate_document(document)
    except Exception:
        logger.exception("observe: framing annotation failed (continuing)")
    document["resolved"] = layers.resolve(document, ctx.durations, ctx.color_stats).to_dict()
    return document


def _remap_split_edits(document: dict, old_ids: List[Optional[str]],
                       old_segs: List[dict]) -> None:
    """Welding re-issues seg_ids, which would orphan split_edit (J/L cut) ops
    keyed on ``seam_seg_id``. Weld keeps the SAME dicts for surviving segments
    (mutating ids in place), so identity tells us where each seam went: a
    surviving segment gets its op remapped to the new id; a segment that merged
    into its predecessor lost its seam -- the split there is meaningless, drop it."""
    ops = document.get("operations") or []
    if not any(o.get("type") == "split_edit" for o in ops):
        return
    survivors = {id(s): s.get("seg_id") for s in (document.get("timeline") or [])}
    new_by_old = {old: survivors.get(id(seg))
                  for old, seg in zip(old_ids, old_segs) if old}
    kept: List[dict] = []
    for o in ops:
        if o.get("type") != "split_edit":
            kept.append(o)
            continue
        new_id = new_by_old.get(o.get("seam_seg_id"))
        if new_id is None:
            continue                      # seam welded away -> split is moot
        o["seam_seg_id"] = new_id
        kept.append(o)
    document["operations"] = kept


# --------------------------------------------------------------------------
# Seam snapping (v3-native): clean cut-boundary points from cut_records, for
# split_screen's raw-window cell. Not the old hero/cuts-v2 fused seam field --
# just the edges the ingest pipeline already snapped to a word/atom edge (see
# cleanup.plan.md B1).
# --------------------------------------------------------------------------

def _seams_for_file(ctx: EditContext, file_id: str) -> List[int]:
    """Clean cut-boundary ms points for one clip, read straight off its
    resolved cut_records run -- every cut's src_in_ms/src_out_ms (already
    word/atom-snapped by the ingest pipeline, so each is a genuine clean edit
    point). Empty when the file has no cut_records in scope (fail-open -- the
    caller's snap then degrades to a no-op)."""
    if ctx.run_id is None:
        return []
    try:
        from app.services.l3 import cuts_v3_read
        rows = cuts_v3_read.rows_for_run(ctx.run_id, [file_id])
    except Exception:
        logger.exception("observe: seam points lookup failed for %s", file_id)
        return []
    pts = {int(r["src_in_ms"]) for r in rows} | {int(r["src_out_ms"]) for r in rows}
    return sorted(pts)


def snap_span_to_seams(points: List[int], in_ms: Any, out_ms: Any, *,
                       max_move_ms: Optional[int] = None) -> Dict[str, Any]:
    """Snap ``[in_ms, out_ms]`` to the nearest boundary in ``points`` on each
    edge. Mirrors the sovereignty-cap contract the old fused-seam-field
    snapper used: with ``max_move_ms`` set, an edge whose nearest boundary
    sits FURTHER than the cap stays where the caller put it, reported as
    ``in_suggested_ms``/``out_suggested_ms`` instead. Every point in
    ``points`` is already a verified cut edge (word/atom-snapped at ingest),
    so a snap always reports quality 1.0 -- there is no gradation to make.
    Degrades to an unchanged no-op (``snapped=False``) when there are no
    boundaries."""
    try:
        a, b = int(in_ms), int(out_ms)
    except (TypeError, ValueError):
        return {"in_ms": in_ms, "out_ms": out_ms, "snapped": False}
    if not points or b <= a:
        return {"in_ms": a, "out_ms": b, "snapped": False}
    si = min(points, key=lambda p: abs(p - a))
    so = min(points, key=lambda p: abs(p - b))
    out: Dict[str, Any] = {"snapped": True}
    if max_move_ms is not None and abs(si - a) > int(max_move_ms):
        out["in_suggested_ms"] = si
        si = a
    if max_move_ms is not None and abs(so - b) > int(max_move_ms):
        out["out_suggested_ms"] = so
        so = b
    if so <= si:                       # capping one edge must not invert
        si, so = a, b
    out.update({
        "in_ms": si, "out_ms": so,
        "in_delta_ms": si - a, "out_delta_ms": so - b,
        "in_q": 1.0, "out_q": 1.0,
    })
    return out


def snap_to_beats(grid_ms: List[int], ms: Any, *, max_move_ms: Optional[int] = None) -> Dict[str, Any]:
    """Snap a single program-time edge `ms` to the nearest beat/onset in
    `grid_ms` (flattened `onsets_ms` from `audio_state`'s `beat_grid`,
    audio_brain.plan.md 2b). Same sovereignty-cap contract as
    `snap_span_to_seams`: beyond `max_move_ms` the caller's edge is kept,
    reported as `suggested_ms` instead. No grid -> unchanged, `snapped=False`
    -- there's nothing to snap to where there's no music."""
    try:
        m = int(ms)
    except (TypeError, ValueError):
        return {"ms": ms, "snapped": False}
    if not grid_ms:
        return {"ms": m, "snapped": False}
    nearest = min(grid_ms, key=lambda p: abs(p - m))
    if max_move_ms is not None and abs(nearest - m) > int(max_move_ms):
        return {"ms": m, "snapped": True, "suggested_ms": nearest, "delta_ms": 0}
    return {"ms": nearest, "snapped": True, "delta_ms": nearest - m}


# --------------------------------------------------------------------------
# 1. read_state
# --------------------------------------------------------------------------

def _fid8(s: str) -> str:
    return (s or "")[:8]


def _word_offsets_for_seg(seg: dict, prog_start_ms: int) -> List[Dict[str, Any]]:
    """Word-level program-time offsets for ONE timeline segment (edso_pacing_
    audit_timing.plan.md item 3) -- reuses captions/timing.py's linear
    source->program mapping. Correct and complete per segment even though
    the mapping itself is linear/single-span: a cut's own keep_spans jump-
    cuts are already flattened into SEPARATE timeline segments at placement
    time (`act._segments_from_cut`), so every segment IS one contiguous
    source window by construction -- there is no multi-span case left to
    handle here. [] when the file has no transcript yet (never fabricated)."""
    file_id = seg.get("file_id")
    if not file_id:
        return []
    transcripts = captions_resolver.fetch_transcripts([file_id])
    words_src = (transcripts.get(file_id) or {}).get("segments") or []
    in_ms, out_ms = int(seg.get("in_ms", 0)), int(seg.get("out_ms", 0))
    words = captions_timing.words_in_source_window(words_src, in_ms, out_ms)
    out: List[Dict[str, Any]] = []
    for w in words:
        t_in, t_out = captions_timing._to_program(w, in_ms, prog_start_ms)
        out.append({"text": w.get("text", ""), "prog_start_ms": t_in, "prog_end_ms": t_out})
    return out


def read_state(document: dict, ctx: EditContext, *, seg_id: Optional[str] = None) -> dict:
    """The current edit at a glance: ordered cuts, channels in use, duration, and
    the feel narration. This is the brain's primary 'look at the timeline'.
    `seg_id` (optional): also resolve that ONE main-line segment's spoken
    words down to program-time offsets (item 3) -- e.g. to land an overlay
    precisely on a line. On-demand only; omitted by default so a plain look
    never pays a transcript fetch."""
    timeline = document.get("timeline") or []
    ops = document.get("operations") or []
    report = feel.simulate(timeline, ctx.meta_by_ref)

    # Grade summaries (color_grading.plan.md SS10 "explain the grade") read
    # from the LAST resolve's snapshot -- same staleness as everything else
    # read_state reports, and avoids a full re-resolve just to describe it.
    grade_by_layer_id = {
        v.get("layer_id"): v.get("grade")
        for v in ((document.get("resolved") or {}).get("video_layers") or [])
    }

    # Audio facts (audio_brain.plan.md 1c). `replace_windows`: program spans a
    # place_audio op tagged audio_kind="replace" covers -- a brain-authored fact
    # about ITS OWN past decision, not a code-enforced mute of the cut beneath.
    replace_windows = [
        (int(o.get("from_ms", 0)), int(o.get("to_ms", 0)))
        for o in ops if o.get("type") == "place_audio" and o.get("audio_kind") == "replace"
    ]
    lufs_by_file = {
        fid: af["integrated_lufs"] for fid, af in ctx.audio_features.items()
        if af.get("integrated_lufs") is not None
    }
    _timeline_lufs = sorted(
        lufs_by_file[seg["file_id"]] for seg in timeline
        if seg.get("file_id") in lufs_by_file
    )
    lufs_median = (
        _timeline_lufs[len(_timeline_lufs) // 2] if _timeline_lufs else None
    )

    cuts = []
    prog = 0
    for i, seg in enumerate(timeline):
        dur = int(seg.get("out_ms", 0)) - int(seg.get("in_ms", 0))
        meta = ctx.meta_by_ref.get(seg.get("ref") or "") or {}
        snippet = (seg.get("content") or "").replace("\n", " ").strip()
        if len(snippet) > 60:
            snippet = snippet[:57] + "..."
        prog_end = prog + dur
        if seg.get("mute"):
            audio_source = "muted"
        elif any(f < prog_end and prog < t for f, t in replace_windows):
            audio_source = "replaced"
        elif meta.get("outlook_group_id"):
            audio_source = "group-authoritative"
        else:
            audio_source = "own"
        cut = {
            "pos": i + 1,
            "seg_id": seg.get("seg_id"),
            "ref": seg.get("ref"),
            "file": _fid8(seg.get("file_id") or ""),
            # Program window (ms) so the brain can target split/PiP by time.
            "prog_start_ms": prog,
            "prog_end_ms": prog_end,
            "dur_ms": dur,
            "channel": meta.get("channel") or seg.get("axis"),
            "speaker": meta.get("speaker"),
            "muted": bool(seg.get("mute")),
            "audio_source": audio_source,
            "text": snippet,
        }
        natural_sound = (meta.get("pace") or {}).get("natural_sound")
        if natural_sound is not None:
            cut["natural_sound"] = bool(natural_sound)
        file_lufs = lufs_by_file.get(seg.get("file_id"))
        if file_lufs is not None and lufs_median is not None:
            cut["loudness_rel"] = round(file_lufs - lufs_median, 1)
        grade = grade_by_layer_id.get(f"v_{seg.get('seg_id')}")
        if grade and grade.get("cdl"):
            summary = explain_grade(grade["cdl"])
            if "No change" not in summary:
                cut["grade"] = summary
        # Pacing state (from `retime`): the chosen step, and for a VIDEO cut the
        # recorded playback speed -- flagged, since the render doesn't bake speed
        # yet (the on-screen dur_ms above is still the 1x length).
        if seg.get("pace_level"):
            cut["pace_level"] = seg.get("pace_level")
            if seg.get("speed") is not None:
                cut["speed"] = seg.get("speed")
                cut["speed_note"] = "recorded; not yet applied to the export length"
        cuts.append(cut)
        prog += dur

    channels = ["V1", "A1"] if timeline else []
    if any(o.get("type") == "place_video" for o in ops):
        channels.append("V2")
    if any(o.get("type") == "place_audio" for o in ops):
        channels.append("A2")

    layouts = [
        {"region_id": r.get("region_id"), "template": r.get("template"),
         "from_ms": r.get("from_ms"), "to_ms": r.get("to_ms"),
         "cells": {c: (s or {}).get("layer") for c, s in (r.get("cells") or {}).items()}}
        for r in (document.get("layout_regions") or [])
    ]

    state = {
        "cut_count": len(timeline),
        "op_count": len(ops),
        "total_ms": report.total_ms,
        "channels": channels,
        "layouts": layouts,
        "feel": report.narrate(),
        "feel_detail": report.to_dict(),
        "cuts": cuts,
    }
    splits = [{"op_id": o.get("op_id"), "seam_seg_id": o.get("seam_seg_id"),
               "audio_offset_ms": o.get("audio_offset_ms")}
              for o in ops if o.get("type") == "split_edit"]
    if splits:
        state["split_edits"] = splits

    # edso_pacing_audit_timing.plan.md item 2: report the z-stack (coverage
    # layers + layout), not just the spine `cuts` above, so this on-demand
    # look matches what the Program Map already shows -- resolved FRESH here
    # (not read off `document["resolved"]`, which is only as fresh as the
    # LAST full turn's resolve_doc, stale for a tool call mid-loop).
    try:
        resolved = layers.resolve(document, ctx.durations)
        coverage = [
            {"layer_id": v.layer_id, "op_id": v.op_id, "z": v.z, "layout": v.layout,
             "prog_start_ms": v.prog_start_ms, "prog_end_ms": v.prog_end_ms,
             "source_file_id": _fid8(v.source_file_id)}
            for v in resolved.video_layers if v.kind != "spine"
        ]
        if coverage:
            state["video_stack"] = coverage
        extra_audio = [
            {"layer_id": a.layer_id, "op_id": a.op_id, "role": a.role, "kind": a.kind,
             "prog_start_ms": a.prog_start_ms, "prog_end_ms": a.prog_end_ms,
             "gain_db": a.gain_db, "duck_db": a.duck_db}
            for a in resolved.audio_layers if a.kind != "spine"
        ]
        if extra_audio:
            state["audio_layers"] = extra_audio
    except Exception:
        logger.exception("read_state: layer resolve failed (continuing without the z-stack)")

    if seg_id:
        seg = next((s for s in timeline if s.get("seg_id") == seg_id), None)
        match = next((c for c in cuts if c["seg_id"] == seg_id), None)
        if seg is not None and match is not None:
            try:
                state["word_offsets"] = {
                    "seg_id": seg_id,
                    "words": _word_offsets_for_seg(seg, match["prog_start_ms"]),
                }
            except Exception:
                logger.exception("read_state: word-offset detail failed for seg %s", seg_id)
    return state


# --------------------------------------------------------------------------
# 1b. audio_state (audio_brain.plan.md 1c -- beds, continuous runs, assets)
# --------------------------------------------------------------------------

def audio_state(document: dict, ctx: EditContext) -> dict:
    """The audio digest: placed beds (with the asset-vs-window length so a
    shortfall is a visible fact, never auto-looped), outlook-authoritative runs
    (a fact, not a lock -- `split_edit` still works inside one), and this
    user's unused uploaded audio files. Facts only -- no "use this for X"."""
    timeline = document.get("timeline") or []
    ops = document.get("operations") or []
    name_by_file = {a["file_id"]: a["name"] for a in ctx.audio_assets}
    dur_by_file = {a["file_id"]: a["dur_ms"] for a in ctx.audio_assets}

    beds = []
    used_source_ids = set()
    for o in ops:
        if o.get("type") != "place_audio":
            continue
        src = o.get("source_file_id")
        used_source_ids.add(src)
        window_ms = int(o.get("to_ms", 0)) - int(o.get("from_ms", 0))
        # `ctx.durations` only covers this turn's file_ids (video sources); an
        # uploaded audio-only asset's length lives in `ctx.audio_assets`.
        asset_dur_ms = ctx.durations.get(src)
        if asset_dur_ms is None:
            asset_dur_ms = dur_by_file.get(src)
        bed = {
            "op_id": o.get("op_id"), "role": o.get("role"),
            "kind": o.get("audio_kind", "bed"),
            "from_ms": o.get("from_ms"), "to_ms": o.get("to_ms"),
            "gain_db": o.get("gain_db", 0.0), "duck_db": o.get("duck_db", 0.0),
            "asset": name_by_file.get(src) or _fid8(src or ""),
            "window_ms": window_ms,
        }
        if asset_dur_ms is not None:
            bed["asset_dur_ms"] = asset_dur_ms
            src_span_ms = int(o.get("src_out_ms", 0)) - int(o.get("src_in_ms", 0))
            if src_span_ms < window_ms:
                bed["shortfall_ms"] = window_ms - src_span_ms
        beds.append(bed)

    # continuous_beds: consecutive main-line cuts sharing one outlook sync
    # group -- their coupled audio is already one continuous authoritative
    # source across the angle switches (audio_sync.plan.md), surfaced here as
    # a fact so the brain doesn't need to re-derive it from per-cut meta. A
    # run of exactly one cut isn't a cross-cut fact worth surfacing.
    continuous_beds = []
    run_group, run_start_ms, run_end_ms, run_len = None, 0, 0, 0
    prog = 0

    def _flush():
        if run_group and run_len > 1:
            continuous_beds.append({
                "from_ms": run_start_ms, "to_ms": run_end_ms, "group_id": run_group,
                "note": "shares one authoritative audio source across these angle switches",
            })

    for seg in timeline:
        dur = int(seg.get("out_ms", 0)) - int(seg.get("in_ms", 0))
        meta = ctx.meta_by_ref.get(seg.get("ref") or "") or {}
        group = meta.get("outlook_group_id")
        if group and group == run_group:
            run_end_ms = prog + dur
            run_len += 1
        else:
            _flush()
            run_group, run_start_ms, run_end_ms, run_len = group, prog, prog + dur, 1
        prog += dur
    _flush()

    channels = ["V1", "A1"] if timeline else []
    if any(o.get("type") == "place_video" for o in ops):
        channels.append("V2")
    if beds:
        channels.append("A2")

    out = {
        "beds": beds,
        "continuous_beds": continuous_beds,
        "assets": [a for a in ctx.audio_assets if a["file_id"] not in used_source_ids],
        "channels": channels,
    }
    grid = _beat_grid(document, ctx)
    if grid:
        out["beat_grid"] = grid
    return out


def _beat_grid(document: dict, ctx: EditContext) -> List[dict]:
    """BPM + onset positions mapped to PROGRAM time (audio_brain.plan.md 2b),
    one entry per musical source currently in play -- an A2 bed or a main-line
    video clip doubling as one. Surfaced ONLY when a musical source exists;
    the brain doesn't try to snap where there's no music."""
    grid: List[dict] = []
    for o in document.get("operations") or []:
        if o.get("type") != "place_audio":
            continue
        af = ctx.audio_features.get(o.get("source_file_id") or "")
        if not af or not af.get("is_musical"):
            continue
        src_in = int(o.get("src_in_ms", 0))
        f, t = int(o.get("from_ms", 0)), int(o.get("to_ms", 0))
        onsets = [f + (int(ms) - src_in) for ms in (af.get("onsets_ms") or [])
                 if src_in <= int(ms) < src_in + (t - f)]
        grid.append({"source": o.get("op_id"), "bpm": af.get("bpm"),
                     "from_ms": f, "to_ms": t, "onsets_ms": onsets})

    prog = 0
    for seg in document.get("timeline") or []:
        dur = int(seg.get("out_ms", 0)) - int(seg.get("in_ms", 0))
        af = ctx.audio_features.get(seg.get("file_id") or "")
        if af and af.get("is_musical"):
            src_in = int(seg.get("in_ms", 0))
            onsets = [prog + (int(ms) - src_in) for ms in (af.get("onsets_ms") or [])
                     if src_in <= int(ms) < src_in + dur]
            grid.append({"source": seg.get("seg_id"), "bpm": af.get("bpm"),
                        "from_ms": prog, "to_ms": prog + dur, "onsets_ms": onsets})
        prog += dur
    return grid


# --------------------------------------------------------------------------
# 2. predict (what a change would do -- without applying it)
# --------------------------------------------------------------------------

def _seg_ms(seg: dict) -> int:
    return max(0, int(seg.get("out_ms", 0)) - int(seg.get("in_ms", 0)))


def _variant_ms(ctx: EditContext, ref: Optional[str], level: str) -> Optional[int]:
    """Play length of a moment ref at a level (post dead-air), or None."""
    m = ctx.index.moments.get(ref or "")
    if not m:
        return None
    v = (m.get("variants") or {}).get(level)
    if not v:
        return None
    return int(v.get("play_ms", int(v.get("out_ms", 0)) - int(v.get("in_ms", 0))))


def predict(document: dict, ctx: EditContext, *,
            set_level: Optional[str] = None,
            drop: Optional[List[str]] = None,
            add: Optional[List[dict]] = None) -> dict:
    """Project the program LENGTH under a proposed change, without mutating:
      * set_level: re-take every main-line cut at this level (pacing dial);
      * drop:      seg_ids to remove;
      * add:       [{ref, level}] cuts to append.
    Returns {current_ms, projected_ms, delta_ms, note}. Approximate (uses map
    play lengths; ignores weld overlaps) -- a planning aid, not the final cut."""
    timeline = document.get("timeline") or []
    drop_set = set(drop or [])
    current = sum(_seg_ms(s) for s in timeline)
    projected = 0
    for seg in timeline:
        if seg.get("seg_id") in drop_set:
            continue
        if set_level and seg.get("ref"):
            vm = _variant_ms(ctx, seg.get("ref"), set_level)
            projected += vm if vm is not None else _seg_ms(seg)
        else:
            projected += _seg_ms(seg)
    for a in (add or []):
        vm = _variant_ms(ctx, a.get("ref"), a.get("level", "balanced"))
        projected += vm if vm is not None else 0

    delta = projected - current
    note = f"{current / 1000:.1f}s -> {projected / 1000:.1f}s ({'+' if delta >= 0 else ''}{delta / 1000:.1f}s)"
    return {"current_ms": current, "projected_ms": projected, "delta_ms": delta, "note": note}


# --------------------------------------------------------------------------
# 3. validate (structural legality)
# --------------------------------------------------------------------------

def validate(document: dict, ctx: EditContext) -> List[dict]:
    """Structural problems that would break resolve/render. Empty list == clean.
    Each issue: {kind, id, message}."""
    issues: List[dict] = []
    for seg in document.get("timeline") or []:
        sid = seg.get("seg_id")
        fid = str(seg.get("file_id") or "")
        in_ms, out_ms = int(seg.get("in_ms", 0)), int(seg.get("out_ms", 0))
        if not fid:
            issues.append({"kind": "segment", "id": sid, "message": "missing file_id"})
        if out_ms <= in_ms:
            issues.append({"kind": "segment", "id": sid, "message": f"empty span {in_ms}-{out_ms}"})
        dur = ctx.durations.get(fid)
        if dur and out_ms > dur:
            issues.append({"kind": "segment", "id": sid,
                           "message": f"out {out_ms}ms exceeds source {dur}ms"})
        if in_ms < 0:
            issues.append({"kind": "segment", "id": sid, "message": "negative in_ms"})

    program_end = _seg_span_total(document)
    for op in document.get("operations") or []:
        oid = op.get("op_id")
        if op.get("type") == "place_video":
            frm, to = int(op.get("from_ms", 0)), int(op.get("to_ms", 0))
            if to <= frm:
                issues.append({"kind": "op", "id": oid, "message": f"empty V2 cutaway window {frm}-{to}"})
            if program_end and frm > program_end:
                issues.append({"kind": "op", "id": oid,
                               "message": f"V2 cutaway starts {frm}ms past program end {program_end}ms"})
            if int(op.get("src_out_ms", 0)) <= int(op.get("src_in_ms", 0)):
                issues.append({"kind": "op", "id": oid, "message": "empty source span"})
        elif op.get("type") == "split_edit":
            seam = op.get("seam_seg_id")
            seg_ids = [s.get("seg_id") for s in (document.get("timeline") or [])]
            if seam not in seg_ids:
                issues.append({"kind": "op", "id": oid,
                               "message": f"split_edit seam {seam!r} is not a main-line seg_id"})
            elif seg_ids and seam == seg_ids[0]:
                issues.append({"kind": "op", "id": oid,
                               "message": "split_edit on the first cut (no seam before it)"})

    # A split/PiP cell feed whose region is gone is an ORPHAN: it would paste
    # full-frame over the main line as a silent take-over (the teardown should
    # have retired it with its region). Precise -- keyed on the split_cell marker
    # so ordinary V2 cutaways (deliberately region-less) are never flagged.
    referenced_layers = {
        (sel or {}).get("layer")
        for r in (document.get("layout_regions") or [])
        for sel in (r.get("cells") or {}).values()
    }
    for op in document.get("operations") or []:
        if op.get("type") == "place_video" and op.get("purpose") == "split_cell" \
                and op.get("op_id") not in referenced_layers:
            issues.append({"kind": "op", "id": op.get("op_id"),
                           "message": "orphaned split cell (no layout region references it -- "
                                      "remove it or re-add its split_screen region)"})

    op_ids = {o.get("op_id") for o in (document.get("operations") or [])}
    for r in document.get("layout_regions") or []:
        rid = r.get("region_id")
        if not layers.LAYOUT_TEMPLATES.get(str(r.get("template") or "")):
            issues.append({"kind": "layout", "id": rid, "message": f"unknown template {r.get('template')!r}"})
        if int(r.get("to_ms", 0)) <= int(r.get("from_ms", 0)):
            issues.append({"kind": "layout", "id": rid, "message": "empty layout window"})
        for cell, sel in (r.get("cells") or {}).items():
            layer = (sel or {}).get("layer")
            if layer and layer != "spine" and layer not in op_ids:
                issues.append({"kind": "layout", "id": rid,
                               "message": f"cell {cell} references missing layer {layer}"})
    return issues


def _seg_span_total(document: dict) -> int:
    return sum(_seg_ms(s) for s in (document.get("timeline") or []))


# --------------------------------------------------------------------------
# 4. diagnose (editorial problems worth fixing)
# --------------------------------------------------------------------------

def diagnose(document: dict, ctx: EditContext) -> List[dict]:
    """Editorial findings computed from the edit. Each: {severity, message,
    anchor?}. Derived from feel + the map -- OBSERVATIONS only, no suggested
    fix (the brain decides what, if anything, to do)."""
    timeline = document.get("timeline") or []
    findings: List[dict] = []
    if not timeline:
        return findings
    report = feel.simulate(timeline, ctx.meta_by_ref)

    for lo, hi in feel._same_speaker_runs(report.cuts):
        findings.append({"severity": "warn", "anchor": f"cuts {lo}-{hi}",
                         "message": "same speaker back-to-back (jump-cut risk)"})
    for lo, hi in feel._low_energy_runs(report.cuts):
        findings.append({"severity": "info", "anchor": f"cuts {lo}-{hi}",
                         "message": "low-energy run"})

    # Target length (from the brief) vs current.
    target_s = (document.get("brief") or {}).get("target_duration_s")
    if target_s:
        cur = report.total_ms / 1000.0
        if cur > target_s * 1.15:
            findings.append({"severity": "warn", "anchor": "whole",
                             "message": f"over target: {cur:.1f}s vs {target_s:.1f}s"})
        elif cur < target_s * 0.7:
            findings.append({"severity": "info", "anchor": "whole",
                             "message": f"under target: {cur:.1f}s vs {target_s:.1f}s"})

    # Redundant takes: two main-line cuts from the same dup group.
    group_of: Dict[str, str] = {}
    for g in ctx.dup_groups:
        for mid in g.get("members") or []:
            group_of[mid] = g.get("group_id")
    seen_groups: Dict[str, int] = {}
    for i, seg in enumerate(timeline):
        gid = group_of.get(seg.get("ref") or "")
        if gid and gid in seen_groups:
            findings.append({"severity": "warn", "anchor": f"cuts {seen_groups[gid]+1} & {i+1}",
                             "message": "same-beat takes both on the main line"})
        elif gid:
            seen_groups[gid] = i
    return findings


# --------------------------------------------------------------------------
# 4b. review (the assembled program, read back -- edso_pacing_audit_timing.
# plan.md item 6)
# --------------------------------------------------------------------------

# A sentence made up ENTIRELY of these reads as pure filler/backchannel
# ("um", "yeah", "okay, right") rather than real content -- generic spoken-
# language patterns, not a domain assumption (the same words show up in any
# genre's dialogue).
_FILLER_WORDS = frozenset("""
um uh uhh umm er erm ah hm hmm mhm mm huh
yeah yep yup okay ok so like right well
""".split())

# A gap this long between a cut's own boundary and the first/last word still
# inside it reads as a real silent lead-in/tail worth flagging.
_DEAD_AIR_HEAD_TAIL_MS = 400

# An overlay covering less than this fraction of the beat it sits over
# clearly underfills it; bleeding this many ms past the beat's own edge
# clearly overruns into a neighboring one.
_OVERLAY_UNDERFILL_RATIO = 0.5
_OVERLAY_OVERRUN_MS = 500


def _is_filler_sentence(text: Optional[str]) -> bool:
    words = re.sub(r"[^a-zA-Z' ]", "", (text or "")).lower().split()
    return bool(words) and all(w in _FILLER_WORDS for w in words)


def _head_tail_flags(seg: dict, anchor: str) -> List[dict]:
    """Facts about a SPEECH cut's played head/tail -- a speaker mismatch
    against the cut's own majority speaker, a filler/backchannel line, or
    silent dead air still inside the cut's own boundary. Presented, never a
    prescribed fix -- the brain decides whether/how to trim."""
    file_id = seg.get("file_id") or ""
    in_ms, out_ms = int(seg.get("in_ms", 0)), int(seg.get("out_ms", 0))
    sentences = sorted(
        (s for s in footage_map._sentences_for_file(file_id)
         if footage_map._overlap_ms(*footage_map._sentence_span(s), in_ms, out_ms) > 0),
        key=lambda s: footage_map._sentence_span(s)[0],
    )
    if not sentences:
        return []
    findings: List[dict] = []

    speakers = [s.get("speaker") for s in sentences if s.get("speaker")]
    if len(set(speakers)) > 1:
        # Dominance by TOTAL SPOKEN DURATION, not sentence count -- a single
        # 300ms filler and a single 4s thought are both "one sentence", but
        # only one of them is actually carrying the cut.
        durations: Dict[str, int] = {}
        for s in sentences:
            spk = s.get("speaker")
            if not spk:
                continue
            s0, s1 = footage_map._sentence_span(s)
            durations[spk] = durations.get(spk, 0) + max(0, s1 - s0)
        dominant = max(durations.items(), key=lambda kv: kv[1])[0]
        head_spk, tail_spk = sentences[0].get("speaker"), sentences[-1].get("speaker")
        if head_spk and head_spk != dominant:
            findings.append({"severity": "info", "anchor": anchor,
                             "message": f"lead-in spoken by {head_spk}, not this cut's "
                                        f"dominant speaker ({dominant})"})
        if len(sentences) > 1 and tail_spk and tail_spk != dominant:
            findings.append({"severity": "info", "anchor": anchor,
                             "message": f"tail spoken by {tail_spk}, not this cut's "
                                        f"dominant speaker ({dominant})"})

    if _is_filler_sentence(sentences[0].get("text")):
        findings.append({"severity": "info", "anchor": anchor,
                         "message": f"lead-in reads as filler/backchannel: "
                                    f"\"{sentences[0].get('text')}\""})
    if len(sentences) > 1 and _is_filler_sentence(sentences[-1].get("text")):
        findings.append({"severity": "info", "anchor": anchor,
                         "message": f"tail reads as filler/backchannel: "
                                    f"\"{sentences[-1].get('text')}\""})

    first_s, _first_e = footage_map._sentence_span(sentences[0])
    _last_s, last_e = footage_map._sentence_span(sentences[-1])
    if first_s - in_ms >= _DEAD_AIR_HEAD_TAIL_MS:
        findings.append({"severity": "info", "anchor": anchor,
                         "message": f"~{(first_s - in_ms) / 1000:.1f}s of dead air "
                                    f"before the first word"})
    if out_ms - last_e >= _DEAD_AIR_HEAD_TAIL_MS:
        findings.append({"severity": "info", "anchor": anchor,
                         "message": f"~{(out_ms - last_e) / 1000:.1f}s of dead air "
                                    f"after the last word"})
    return findings


def _overlay_fit_flags(resolved: "layers.ResolvedTimeline") -> List[dict]:
    """Flag a V2 coverage layer whose program window clearly overruns or
    underfills the spine beat it sits over -- a fact, never a prescribed
    fix (the brain decides whether/how to retime it)."""
    findings: List[dict] = []
    spine = [v for v in resolved.video_layers if v.kind == "spine"]
    for cov in resolved.video_layers:
        if cov.kind == "spine":
            continue
        host = max(
            spine, key=lambda s: _overlap_ms(cov.prog_start_ms, cov.prog_end_ms,
                                             s.prog_start_ms, s.prog_end_ms),
            default=None,
        )
        if host is None or _overlap_ms(cov.prog_start_ms, cov.prog_end_ms,
                                       host.prog_start_ms, host.prog_end_ms) <= 0:
            continue
        host_dur = host.prog_end_ms - host.prog_start_ms
        cov_dur = cov.prog_end_ms - cov.prog_start_ms
        anchor = f"overlay {cov.op_id or cov.layer_id}"
        if host_dur > 0 and cov_dur < host_dur * _OVERLAY_UNDERFILL_RATIO:
            findings.append({"severity": "info", "anchor": anchor,
                             "message": f"covers {cov_dur}ms of a {host_dur}ms beat -- "
                                        f"underfills the line it sits over"})
        overrun_before = host.prog_start_ms - cov.prog_start_ms
        overrun_after = cov.prog_end_ms - host.prog_end_ms
        if overrun_before >= _OVERLAY_OVERRUN_MS or overrun_after >= _OVERLAY_OVERRUN_MS:
            findings.append({"severity": "info", "anchor": anchor,
                             "message": "extends past the beat it sits over into a "
                                        "neighboring one"})
    return findings


def _overlap_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _tag_category(findings: List[dict], category: str) -> List[dict]:
    for f in findings:
        f["category"] = category
    return findings


# edso_think_act_check.plan.md change 4: the ask/contract half of the audit --
# a GENERIC keyword -> structural-presence check, not a per-genre feature
# list. Each group is (display name, phrases that name the feature in plain
# English, a check against the assembled document). Deterministic, signal-
# driven (post.py/cutrecord_map.py convention): a plain word match against
# the user's own words, checked against a structural fact -- never an LLM
# judgment call on whether the feature was "really" wanted.
_FEATURE_GROUPS: Tuple[Tuple[str, Tuple[str, ...], Any], ...] = (
    ("split screen",
     ("split screen", "split-screen", "side by side", "side-by-side",
      "picture in picture", "picture-in-picture"),
     lambda doc: bool(doc.get("layout_regions"))),
    ("a music bed",
     ("music", "soundtrack", "background song", "backing track"),
     lambda doc: any(o.get("type") == "place_audio" and o.get("role") == "music"
                     for o in (doc.get("operations") or []))),
)


def _mentions(text: str, phrase: str) -> bool:
    return re.search(r"\b" + re.escape(phrase) + r"\b", text) is not None


def _requested_feature_flags(document: dict, user_ask: str) -> List[dict]:
    """Flag a feature the user's own words named that isn't actually in the
    assembled edit (e.g. "make it a split screen" but no layout_regions
    exist) -- the check that would have caught a dropped split screen. A
    fact ("the ask mentions X but the edit doesn't have it"), never a
    prescribed fix; '' ask -> no check (never a false positive from a
    missing ask). `category` marks these as ASK-level (vs. the guidance-
    level flags below), so a caller can prioritize/label them."""
    if not user_ask:
        return []
    text = user_ask.lower()
    findings: List[dict] = []
    for name, phrases, present in _FEATURE_GROUPS:
        if any(_mentions(text, p) for p in phrases) and not present(document):
            findings.append({"severity": "warn", "anchor": "whole", "category": "ask",
                             "message": f"the ask mentions {name} but the edit doesn't have it"})
    return findings


def review(document: dict, ctx: EditContext, *, user_ask: str = "") -> dict:
    """The ASSEMBLED program read back, per timeline seg in SOURCE order --
    the brain's own "play back what I actually built" check, distinct from
    read_state (structure) and diagnose (aggregate editorial findings). Each
    item: {idx, id, ref, played_ms, played_text, words} -- played_text is
    the VERBATIM words actually spoken over the seg's PLAYED span (reuse
    footage_map._said_text_for_span; a non-speech seg gets ""), words are
    the same word->program offsets read_state's seg_id detail returns (item
    3). Plus {total_ms, target_ms, cut_count} and `flags`: presented facts,
    checked in order against (1) the ask/contract -- a `user_ask`-named
    feature (split screen, a music bed) missing from the edit, `category`
    "ask" -- then (2) the guidance ceilings -- rough heads/tails, overlay
    fit, `category` "guidance". NEVER a prescribed fix ("trim to +Xms"); the
    brain decides what, if anything, to do. `user_ask` defaults to '' (no
    ask-level check) so a direct `review` tool call still works without it."""
    timeline = document.get("timeline") or []
    items: List[Dict[str, Any]] = []
    flags: List[dict] = list(_requested_feature_flags(document, user_ask))
    prog = 0
    for i, seg in enumerate(timeline):
        dur = int(seg.get("out_ms", 0)) - int(seg.get("in_ms", 0))
        is_speech = seg.get("axis") == "speech"
        played_text = footage_map._said_text_for_span(
            seg.get("file_id") or "", int(seg.get("in_ms", 0)), int(seg.get("out_ms", 0))
        ) if is_speech else ""
        try:
            words = _word_offsets_for_seg(seg, prog) if is_speech else []
        except Exception:
            words = []
        items.append({
            "idx": i + 1, "id": seg.get("seg_id"), "ref": seg.get("ref"),
            "played_ms": dur, "played_text": played_text, "words": words,
        })
        if is_speech:
            try:
                flags.extend(_tag_category(
                    _head_tail_flags(seg, f"cut {i + 1} ({seg.get('seg_id')})"), "guidance"))
            except Exception:
                logger.exception("review: head/tail flags failed for seg %s", seg.get("seg_id"))
        prog += dur

    try:
        resolved = layers.resolve(document, ctx.durations)
        flags.extend(_tag_category(_overlay_fit_flags(resolved), "guidance"))
    except Exception:
        logger.exception("review: overlay-fit flags failed (continuing without them)")

    target_s = (document.get("brief") or {}).get("target_duration_s")
    return {
        "items": items,
        "total_ms": prog,
        "target_ms": int(target_s * 1000) if target_s else None,
        "cut_count": len(timeline),
        "flags": flags,
    }


# --------------------------------------------------------------------------
# 5. affordances (what can be done, to what)
# --------------------------------------------------------------------------

_LEVELS = ("broad", "calm", "balanced", "tight", "sharp")


def affordances(document: dict, ctx: EditContext) -> dict:
    """The brain's menu: per-cut options (retake levels, alternates, mute) + the
    global channel state. Purely what's POSSIBLE -- the brain decides what to do."""
    timeline = document.get("timeline") or []
    group_members: Dict[str, List[str]] = {}
    for g in ctx.dup_groups:
        for mid in g.get("members") or []:
            group_members[mid] = [x for x in (g.get("members") or []) if x != mid]

    per_cut = []
    prog = 0
    for i, seg in enumerate(timeline):
        ref = seg.get("ref")
        meta = ctx.meta_by_ref.get(ref or "") or {}
        variants = list((meta.get("variants") or {}).keys())
        cur = seg.get("level")
        tighter = [L for L in _LEVELS if L in variants and _idx(L) > _idx(cur)]
        wider = [L for L in _LEVELS if L in variants and _idx(L) < _idx(cur)]
        is_video = (meta.get("channel") in ("done", "shown")) or seg.get("axis") == "any"
        dur = int(seg.get("out_ms", 0)) - int(seg.get("in_ms", 0))
        entry = {
            "pos": i + 1,
            "seg_id": seg.get("seg_id"),
            "ref": ref,
            "can_tighten_to": tighter,
            "can_widen_to": wider,
            "alternate_takes": group_members.get(ref or "", []),
            "can_toggle_audio": bool(is_video),
            # `place_audio`'s from_ms/to_ms for a bed spanning exactly this cut.
            "can_place_bed_here": {"from_ms": prog, "to_ms": prog + dur},
        }
        entry.update(_pace_affordance(meta, is_video))
        per_cut.append(entry)
        prog += dur

    # Video moments in the library not already on the main line. Junk is left
    # out of this pool (still placeable by ref if the brain chooses to).
    on_line = {s.get("ref") for s in timeline}
    cutaway_pool = [
        m["moment_id"] for clip in ctx.map_struct.get("clips", [])
        for m in clip.get("moments", []) or []
        if m.get("channel") in ("done", "shown") and m["moment_id"] not in on_line
        and not m.get("junk")
    ]
    channels = ["V1", "A1"] if timeline else []
    if any(o.get("type") == "place_video" for o in (document.get("operations") or [])):
        channels.append("V2")
    if any(o.get("type") == "place_audio" for o in (document.get("operations") or [])):
        channels.append("A2")

    return {
        "cuts": per_cut,
        "channels_in_use": channels,
        "can_add_channel": ["V2", "A2"],
        "cutaway_pool": cutaway_pool[:50],
        "layout_templates": ["split_h", "split_v", "pip"],
        # This user's unused-audio-file count (audio_brain.plan.md 1d) -- the
        # full list with names/duration/bpm is `audio_state`'s `assets`.
        "audio_assets": len(ctx.audio_assets),
        # cuts_v3_continuity.plan.md: the cut-centric loop places by ref only --
        # no raw-footage scan (source_awareness/scan_source/place_span retired
        # from the tool loop; the beat index + its continuity block is the only
        # source of awareness now).
        "verbs": ["place", "trim", "remove", "move", "set_audio", "place_audio",
                  "set_gain", "duck", "fade_audio", "crossfade", "replace_audio",
                  "tighten", "retime", "split_screen"],
        "senses": ["read_state", "predict", "validate", "diagnose", "affordances",
                   "audio_state", "review"],
    }


# Pacing steps the brain can pass to `retime` (mirrors act._PACE_STEPS).
_PACE_STEPS = ("much_slower", "slower", "natural", "faster", "much_faster")


def _pace_affordance(meta: dict, is_video: bool) -> dict:
    """What `retime` can do to this cut, read off its pace envelope:
      * VIDEO -> the reachable playback SPEED per step (levels[0..4]); empty when
        the cut is pinned to one speed (no room).
      * SPEECH -> the removable dead-air/filler budget in ms ('faster' shaves it;
        speed/pitch never change).
    Always lists the ``retime`` steps so the brain knows the verb applies here."""
    pace = meta.get("pace") or {}
    if is_video:
        levels = pace.get("levels") or []
        try:
            speeds = [round(float(x), 2) for x in levels]
        except (TypeError, ValueError):
            speeds = []
        has_room = len(set(speeds)) > 1
        return {"retime_kind": "video_speed", "retime_steps": list(_PACE_STEPS),
                "speed_by_step": (dict(zip(_PACE_STEPS, speeds)) if has_room and len(speeds) == 5 else {})}
    budget = 0
    for sp in pace.get("remove_spans") or []:
        try:
            budget += int(sp[1]) - int(sp[0])
        except (IndexError, TypeError, ValueError):
            continue
    return {"retime_kind": "speech_trim", "retime_steps": list(_PACE_STEPS),
            "trim_budget_ms": budget}


def _idx(level: Optional[str]) -> int:
    try:
        return _LEVELS.index(level)
    except (ValueError, TypeError):
        return _LEVELS.index("balanced")
