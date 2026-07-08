"""
Observe: the brain's deterministic SENSES over an edit -- no VLM, no LLM.

Where ``act`` mutates the document, ``observe`` reads it and reports, so the
agentic loop can perceive -> act -> re-perceive cheaply (a coding agent reading
the repo before/after an edit). Every function is a pure projection of the
document + a per-turn ``EditContext`` (footage map, source durations, clip
valence) assembled ONCE by ``build_context`` -- the only place that touches the
DB. The five senses:

  * read_state  -- what the edit IS now (cuts, channels, duration, feel).
  * predict     -- what a proposed change WOULD do (e.g. length after tightening)
                   without applying it.
  * validate    -- structural legality (spans in range, refs real, no bad ops).
  * diagnose    -- editorial problems worth fixing (sags, jump-cuts, over length,
                   redundant takes) -- read off feel + the map.
  * affordances -- what CAN be done, and to what (tighter takes, alternates,
                   mute toggles, channels) -- the brain's menu.

Feel is delegated to ``feel.simulate`` (also pure). Nothing here renders.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psycopg

from app.config import get_settings
from app.services.l3 import feel, footage_map, framing, layers
from app.services.l3.arrange import _MapIndex, _weld_segments

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
    valence_by_file: Dict[str, str]    # file_id -> clip valence
    dup_groups: List[dict] = field(default_factory=list)
    # Cross-clip relations derived from existing perception (same-event time
    # offsets between clips + one person identity across files).
    relations: dict = field(default_factory=dict)
    # The Cuts v3 ingest run this turn is resolved against -- the thread's pinned
    # run (migration 028) or the live "latest covering run", resolved ONCE here so
    # every projection in the turn (the map struct + the re-assembled BEAT INDEX
    # text) reads the SAME snapshot. None when nothing is ingested yet.
    run_id: Optional[str] = None

    @property
    def meta_by_ref(self) -> Dict[str, dict]:
        """moment_id -> moment node (speaker / channel / variants / flags)."""
        return self.index.moments


def _valence_by_file(file_ids: List[str]) -> Dict[str, str]:
    if not file_ids:
        return {}
    try:
        with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
            rows = conn.execute(
                "select file_id::text, perception->>'valence' "
                "from clip_perception where file_id = any(%s::uuid[])",
                (file_ids,),
            ).fetchall()
        return {fid: v for fid, v in rows if v}
    except Exception:
        logger.exception("observe: valence lookup failed (continuing without)")
        return {}


def build_context(file_ids: List[str], run_id: Optional[str] = None) -> EditContext:
    """Assemble the per-turn context once (footage map + durations + valence +
    cross-clip relations). All DB reads live here; the sense functions stay pure
    over the result.

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
    # Identities first: the footage map aliases per-clip voices to global people.
    rels: dict = {}
    try:
        from app.services.l3 import relations as relations_mod
        rels = relations_mod.build_relations(list(file_ids))
        # Safety net: catch any structural corruption in the reconciled cast and
        # surface it honestly (as a warning the brain will read) rather than let
        # it silently mislead the edit.
        violations = relations_mod.validate(rels)
        if violations:
            logger.warning("observe: identity invariant violations: %s", violations)
            rels.setdefault("warnings", []).extend(violations)
    except Exception:
        logger.exception("observe: relations build failed (continuing without)")
    fmap = footage_map.assemble_map(file_ids, relations=rels, run_id=eff_run) if file_ids else {}
    map_struct = fmap.get("struct") or {"clips": []}
    durations: Dict[str, int] = {}
    try:
        from app.services.render.tasks import _durations
        durations = _durations(file_ids)
    except Exception:
        logger.exception("observe: duration lookup failed (continuing)")
    return EditContext(
        file_ids=list(file_ids),
        index=_MapIndex(map_struct),
        map_struct=map_struct,
        durations=durations,
        valence_by_file=_valence_by_file(file_ids),
        dup_groups=fmap.get("dup_groups") or [],
        relations=rels,
        run_id=eff_run,
    )


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
    document["resolved"] = layers.resolve(document, ctx.durations).to_dict()
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


# --------------------------------------------------------------------------
# 1. read_state
# --------------------------------------------------------------------------

def _fid8(s: str) -> str:
    return (s or "")[:8]


def read_state(document: dict, ctx: EditContext) -> dict:
    """The current edit at a glance: ordered cuts, channels in use, duration, and
    the feel narration. This is the brain's primary 'look at the timeline'."""
    timeline = document.get("timeline") or []
    ops = document.get("operations") or []
    report = feel.simulate(timeline, ctx.meta_by_ref, ctx.valence_by_file)

    cuts = []
    prog = 0
    for i, seg in enumerate(timeline):
        dur = int(seg.get("out_ms", 0)) - int(seg.get("in_ms", 0))
        meta = ctx.meta_by_ref.get(seg.get("ref") or "") or {}
        snippet = (seg.get("content") or "").replace("\n", " ").strip()
        if len(snippet) > 60:
            snippet = snippet[:57] + "..."
        cuts.append({
            "pos": i + 1,
            "seg_id": seg.get("seg_id"),
            "ref": seg.get("ref"),
            "file": _fid8(seg.get("file_id") or ""),
            # Program window (ms) so the brain can target split/PiP by time.
            "prog_start_ms": prog,
            "prog_end_ms": prog + dur,
            "dur_ms": dur,
            "channel": meta.get("channel") or seg.get("axis"),
            "speaker": meta.get("speaker"),
            "muted": bool(seg.get("mute")),
            "text": snippet,
        })
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
    return state


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
    """Editorial findings the brain may want to act on. Each: {severity, message,
    anchor?}. Derived from feel + the map -- opinion-free about the fix."""
    timeline = document.get("timeline") or []
    findings: List[dict] = []
    if not timeline:
        return findings
    report = feel.simulate(timeline, ctx.meta_by_ref, ctx.valence_by_file)

    for lo, hi in feel._same_speaker_runs(report.cuts):
        findings.append({"severity": "warn", "anchor": f"cuts {lo}-{hi}",
                         "message": "same speaker back-to-back (jump-cut risk); consider a cutaway or reorder"})
    for lo, hi in feel._low_energy_runs(report.cuts):
        findings.append({"severity": "info", "anchor": f"cuts {lo}-{hi}",
                         "message": "energy sags; consider tightening or trimming"})

    # Target length (from the brief) vs current.
    target_s = (document.get("brief") or {}).get("target_duration_s")
    if target_s:
        cur = report.total_ms / 1000.0
        if cur > target_s * 1.15:
            findings.append({"severity": "warn", "anchor": "whole",
                             "message": f"over target: {cur:.1f}s vs {target_s:.1f}s (tighten or drop)"})
        elif cur < target_s * 0.7:
            findings.append({"severity": "info", "anchor": "whole",
                             "message": f"under target: {cur:.1f}s vs {target_s:.1f}s (widen or add)"})

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
                             "message": "same-beat takes both on the main line (redundant); keep the stronger"})
        elif gid:
            seen_groups[gid] = i
    return findings


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
    for i, seg in enumerate(timeline):
        ref = seg.get("ref")
        meta = ctx.meta_by_ref.get(ref or "") or {}
        variants = list((meta.get("variants") or {}).keys())
        cur = seg.get("level")
        tighter = [L for L in _LEVELS if L in variants and _idx(L) > _idx(cur)]
        wider = [L for L in _LEVELS if L in variants and _idx(L) < _idx(cur)]
        is_video = (meta.get("channel") in ("done", "shown")) or seg.get("axis") == "any"
        per_cut.append({
            "pos": i + 1,
            "seg_id": seg.get("seg_id"),
            "ref": ref,
            "can_tighten_to": tighter,
            "can_widen_to": wider,
            "alternate_takes": group_members.get(ref or "", []),
            "can_toggle_audio": bool(is_video),
        })

    # Video moments in the library not already on the main line -> cutaway pool.
    # Junk is skip-by-default (cuts_v3_continuity.plan.md) -- kept out of the
    # RECOMMENDED pool, though still placeable by ref if the brain chooses to.
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
        # cuts_v3_continuity.plan.md: the cut-centric loop places by ref only --
        # no raw-footage scan (source_awareness/scan_source/place_span retired
        # from the tool loop; the beat index + its continuity block is the only
        # source of awareness now).
        "verbs": ["place", "trim", "remove", "move", "set_audio",
                  "tighten", "split_screen"],
        "senses": ["read_state", "predict", "validate", "diagnose", "affordances"],
    }


def _idx(level: Optional[str]) -> int:
    try:
        return _LEVELS.index(level)
    except (ValueError, TypeError):
        return _LEVELS.index("balanced")
