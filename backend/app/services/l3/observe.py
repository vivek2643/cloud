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
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    # offsets between clips + one person identity across files). Rendered into
    # source_awareness; empty when clips share nothing.
    relations: dict = field(default_factory=dict)
    # The PROGRAM clock's cut field (program_clock.build_program_field) -- None
    # until a program-side source (e.g. a music bed's beat grid) populates it.
    # Consumers treat None as "no opinion".
    program_field: Optional[object] = None
    # Lazily-built continuous Clip Timelines (source-only, edit-independent), so
    # source_awareness/scan_source don't re-fuse a clip within the same turn.
    tl_cache: Dict[str, object] = field(default_factory=dict)

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


def build_context(file_ids: List[str]) -> EditContext:
    """Assemble the per-turn context once (footage map + durations + valence +
    cross-clip relations). All DB reads live here; the sense functions stay pure
    over the result."""
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
    fmap = footage_map.assemble_map(file_ids, relations=rels) if file_ids else {}
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
    on_line = {s.get("ref") for s in timeline}
    cutaway_pool = [
        m["moment_id"] for clip in ctx.map_struct.get("clips", [])
        for m in clip.get("moments", []) or []
        if m.get("channel") in ("done", "shown") and m["moment_id"] not in on_line
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
        "verbs": ["place", "place_span", "trim", "remove", "move", "set_audio",
                  "tighten", "split_screen"],
        "senses": ["read_state", "predict", "validate", "diagnose", "affordances",
                   "source_awareness", "scan_source"],
    }


def _idx(level: Optional[str]) -> int:
    try:
        return _LEVELS.index(level)
    except (ValueError, TypeError):
        return _LEVELS.index("balanced")


# --------------------------------------------------------------------------
# 6. source_awareness (the continuous, fully-addressable source)
# --------------------------------------------------------------------------

def _timeline(ctx: EditContext, file_id: str):
    """Lazily build + cache the continuous Clip Timeline for one clip (None if its
    L1/L2 inputs aren't materialized). Cached on ctx for the turn."""
    if file_id in ctx.tl_cache:
        return ctx.tl_cache[file_id]
    tl = None
    try:
        from app.services.l3 import clip_timeline_store as cts
        tl = cts.load_clip_timeline(file_id)
    except Exception:
        logger.exception("observe: clip timeline build failed for %s", file_id)
    ctx.tl_cache[file_id] = tl
    return tl


# Disclosure tiers: how much per-clip detail the shoot digest carries. The
# brain must NEVER lose awareness silently -- as the clip count grows the
# digests get shorter, the header SAYS so, and scan_source/source_awareness
# recover any elided detail on demand.
_FULL_DETAIL_MAX_CLIPS = 6      # <= this many clips: full digest each
_COMPACT_DETAIL_MAX_CLIPS = 14  # <= this many: compact digest each; above: summary lines


def source_awareness(ctx: EditContext) -> str:
    """The CONTINUOUS clip timeline for every clip in scope: change-point lanes
    (who is present / who is speaking on camera / gaze / shot / action over the
    whole clock), the cleanest seams, the impact/reveal peaks, and a scored cut
    INDEX. Unlike ``affordances`` (which lists the pre-baked cuts), this exposes
    the clip as a fully-addressable source: any span the brain can describe can
    be placed with ``place_span``, seam-snapped to a clean boundary. Read-only.

    Scales by PROGRESSIVE DISCLOSURE, never silent truncation: few clips get
    full digests, many clips get compact ones, very many get one-line summaries
    -- and the header states the tier so the brain knows to drill down with
    scan_source. The shoot's PEOPLE (one global person across clips) lead the
    digest, since the per-clip blocks after them reference those ids.

    Degrades to a short notice (never raises) when the continuous store or its
    L1/L2 inputs are unavailable, so the loop is unaffected."""
    try:
        from app.services.l3.clip_timeline import render_awareness, render_summary
        n = len(ctx.file_ids)
        detail = ("full" if n <= _FULL_DETAIL_MAX_CLIPS
                  else "compact" if n <= _COMPACT_DETAIL_MAX_CLIPS
                  else "summary")
        blocks: List[str] = []
        if n > 1:
            note = {"full": "full detail per clip",
                    "compact": ("compact detail per clip -- every section is present "
                                "but shortened; scan_source recovers anything elided"),
                    "summary": ("ONE LINE per clip -- drill into any clip with "
                                "scan_source (lanes/facets) before cutting from it"),
                    }[detail]
            blocks.append(f"SHOOT: {n} clips. Disclosure tier: {note}.")
        try:
            from app.services.l3.relations import render_relations
            rel = render_relations(ctx.relations)
            if rel:
                blocks.append(rel)
        except Exception:
            logger.exception("observe: relations render failed (continuing)")
        for fid in ctx.file_ids:
            tl = _timeline(ctx, fid)
            if tl is None:
                continue
            blocks.append(render_summary(tl) if detail == "summary"
                          else render_awareness(tl, detail=detail))
        body = "\n\n".join(blocks)
        return body if any(b.startswith("CLIP") for b in blocks) else \
            "(no continuous timeline available for these clips yet)"
    except Exception:
        logger.exception("observe: source_awareness failed (continuing)")
        return "(continuous timeline unavailable)"


_GLOBAL_RE = re.compile(r"^G\d+$")


def _parse_window(within_ms: object) -> Optional[Tuple[int, int]]:
    """Normalize a caller-supplied time window to (lo, hi) ms, or None. Accepts
    [a, b], (a, b), or {in_ms, out_ms}."""
    if not within_ms:
        return None
    try:
        if isinstance(within_ms, dict):
            a, b = int(within_ms.get("in_ms")), int(within_ms.get("out_ms"))
        else:
            a, b = int(within_ms[0]), int(within_ms[1])
        return (min(a, b), max(a, b)) if b > a else (a, a)
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def _resolve_global(ctx: EditContext, fid: str, token: object) -> Tuple[object, bool]:
    """Map a shoot-wide person id (G1, G2, ...) to its LOCAL handle in file
    ``fid`` (person id, else voice). Returns (resolved, ok): a non-global token
    passes through unchanged (ok=True); a global id not present in this file
    yields (None, False) so the caller can skip the clip without erroring."""
    if token is None or not _GLOBAL_RE.match(str(token)):
        return token, True
    try:
        from app.services.l3 import relations as relations_mod
        m = relations_mod.local_of(ctx.relations, fid, str(token))
    except Exception:
        logger.exception("scan_source: global id resolution failed")
        return None, False
    if not m:
        return None, False
    local = m.get("person") or m.get("voice")
    return (local, True) if local else (None, False)


def _scan_one(ctx: EditContext, fid: str, lane: str,
              match: Optional[Dict[str, object]],
              win: Optional[Tuple[int, int]], cap: int) -> Tuple[List[dict], dict]:
    """Scan one clip, resolving any global person id in the lane suffix or match
    values to that clip's local handle first. Returns (hits, meta) where meta
    reports the resolved lane, which match keys were APPLIED vs. IGNORED (a
    guessed facet name the lane doesn't carry), and the lane's real query
    vocabulary -- so a slightly-wrong query self-corrects instead of dead-ending.
    ([], {}) when the clip has no timeline or the referenced person isn't in it."""
    tl = _timeline(ctx, fid)
    if tl is None:
        return [], {}
    prefix, sep, suf = lane.partition(":")
    lane_name = lane
    if sep:
        rt, ok = _resolve_global(ctx, fid, suf)
        if not ok:
            return [], {}
        lane_name = f"{prefix}:{rt}"
    rmatch: Dict[str, object] = {}
    for k, v in (match or {}).items():
        rt, ok = _resolve_global(ctx, fid, v)
        if not ok:
            return [], {}
        rmatch[k] = rt
    known = tl.lane_value_keys(lane_name)
    meta: dict = {}
    if tl.lane(lane_name) is None:
        meta["missing_lane"] = lane_name
    elif rmatch:
        ignored = sorted(k for k in rmatch if k not in known)
        if ignored:                                  # guessed facet name(s)
            meta["ignored_match"] = ignored
            meta["lane_vocab"] = tl.lane_vocab(lane_name)
    hits: List[dict] = []
    for it in tl.scan(lane_name, **rmatch):
        if win and not (it.start_ms < win[1] and it.end_ms > win[0]):
            continue
        mid = (it.start_ms + it.end_ms) // 2
        hits.append({"file": fid[:8], "in_ms": it.start_ms, "out_ms": it.end_ms,
                     **it.value, "facets": tl.facet_at(mid)})
        if len(hits) >= cap:
            break
    return hits, meta


def scan_source(ctx: EditContext, file_ref: str, lane: str,
                match: Optional[Dict[str, object]] = None,
                within_ms: object = None) -> dict:
    """Facet QUERY over the continuous timeline -- ONE clip or the WHOLE shoot.

    ``lane`` (e.g. 'presence:p2', 'speaking', 'shot', 'action') + optional
    ``match`` (e.g. {state:'on'} or {subject:'p1'}). Each hit carries the full
    ``facets`` at its midpoint, so a compound question ("p2 present AND silent")
    is answered by scanning presence:p2 and reading each hit's facets.

    CROSS-CLIP: ``file_ref`` of '*'/'all' (or empty) scans every clip in scope;
    a full id or 'CLIP <file8>' prefix scans just that one. A global PERSON id
    (G1, G2, ...) in the lane suffix or a match value is resolved to each clip's
    local handle automatically, so "where is G2 on screen anywhere" is one call.

    ``within_ms`` ([a,b] or {in_ms,out_ms}) constrains hits to a time window --
    e.g. take a coverage-group member's own window and scan a co-delivered clip
    near it for a plausible reaction, without any shoot-wide clock.

    Returns {file, lane, hits:[{file,in_ms,out_ms,<value>,facets}], files}.
    Read-only; empty hits when the lane/clip/person is unknown."""
    ref = str(file_ref or "").strip()
    if ref in ("", "*", "all", "any", "CLIP"):
        targets = list(ctx.file_ids)
    else:
        targets = [f for f in ctx.file_ids
                   if f == ref or f.startswith(ref) or ref.startswith(f[:8])][:1] \
            or [f for f in ctx.file_ids if f[:8] == ref[:8]]
    if not targets:
        return {"file": file_ref, "lane": lane, "hits": [],
                "note": "no such clip in scope", "files": []}
    win = _parse_window(within_ms)
    multi = len(targets) > 1
    cap = 40 if not multi else max(4, 40 // len(targets))
    hits: List[dict] = []
    scanned: List[str] = []
    meta: dict = {}
    for fid in targets:
        fh, fmeta = _scan_one(ctx, fid, lane, match, win, cap)
        if _timeline(ctx, fid) is not None:
            scanned.append(fid[:8])
        hits.extend(fh)
        if fmeta and not meta:                       # first informative clip wins
            meta = fmeta
    out: dict = {"file": (targets[0][:8] if not multi else "*"),
                 "lane": lane, "hits": hits[:60], "files": scanned}
    # Always advertise the lanes so a mistaken lane name (the common cause of an
    # empty scan) self-corrects; ditto the applied-vs-ignored match keys + the
    # lane's real value vocabulary when a guessed facet name was dropped.
    lanes_here: List[str] = []
    for fid in (targets if multi else targets[:1]):
        tl = _timeline(ctx, fid)
        if tl is not None:
            lanes_here = sorted({*lanes_here, *(ln.name for ln in tl.lanes)})
    if lanes_here:
        out["lanes_available"] = lanes_here
    elif not multi:
        out["note"] = "no continuous timeline for this clip"
    if meta.get("missing_lane"):
        out["note"] = (f"lane '{meta['missing_lane']}' not on this clip -- see "
                       f"lanes_available")
    if meta.get("ignored_match"):
        out["ignored_match"] = meta["ignored_match"]
        out["lane_vocab"] = meta["lane_vocab"]
        out["note"] = ("some match keys aren't facets of this lane (ignored); "
                       "query the keys in lane_vocab, or read each hit's facets")
    return out
