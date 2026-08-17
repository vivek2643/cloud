"""
Act: the brain's deterministic EDIT VERBS -- immutable ``document -> document``
transforms.

These mirror EXACTLY what the manual timeline editor does (edit ``timeline`` +
``operations`` in place, then re-resolve layers) -- one edit model for both human
and brain, no parallel engine. A verb never renders or calls an LLM; it returns
a new document with a stale ``resolved`` dropped, and the loop re-resolves once
via ``observe.resolve_doc``.

Channel model (see the V1/V2/A1/A2 vocabulary): the main line is V1 video + A1
audio (the ``timeline`` spine); a silent video cutaway over the ongoing audio is
a V2 ``place_video`` op; audio beds are A2 ``place_audio`` ops. ``place`` writes
onto V1 or V2; the span-level verbs (``trim``/``set_audio``) act on whichever
clip owns the id.

Every verb is total: an unknown id, out-of-domain target, or a request that
produces no (or a degenerate) change is a REJECT -- `raise EditReject(reason)`,
the sanctioned, loop-caught reject channel (brain_loop_convergence.plan.md Part
3): `tools._dispatch` catches it and surfaces `applied=false` + the reason to
the brain the SAME turn. It never escapes the loop and never crashes a turn --
the totality contract holds at the loop boundary, just with a legible reason
attached instead of a silent no-op.
"""
from __future__ import annotations

import dataclasses
import uuid
from typing import List, Optional

from app.services.l3 import cutrecord_map, footage_map, layers
from app.services.l3.grade.arc import ARC_INTENTS
from app.services.l3.grade.cdl import Grade, compose
from app.services.l3.grade.steer import solve_steer_grade
from app.services.l3.arrange import Placement, ResolvedCut, _MapIndex


class EditReject(Exception):
    """A verb's STRUCTURED rejection of an inapplicable request -- an unknown id,
    an out-of-domain target, or an argument that produces no (or a degenerate)
    change. Carries a plain-language `reason` the loop surfaces to the brain the
    SAME turn (tools._dispatch catches it -> applied=false + reason), so the brain
    corrects in one turn instead of discovering the no-op later via review. It
    NEVER escapes the loop -- _dispatch converts it to feedback -- so a verb still
    never crashes a turn (the totality contract holds at the loop boundary)."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


# Pacing scale for `retime`, broad..sharp (index 0..4). Maps onto a video cut's
# cross-clip-normalized ``pace.levels`` (idx 2 = the cut's ~natural 1x); for a
# speech cut, where playback speed is NEVER touched, the same ordinal instead
# sets how aggressively removable dead-air/filler is shaved (only the "faster"
# end trims -- speech can be tightened, never slowed). See ``retime``.
_PACE_STEPS = ("much_slower", "slower", "natural", "faster", "much_faster")
# Fraction of a speech cut's removable dead-air/filler budget to shave per step
# (mirrors the dial: 0.85 == cutrecord_map._SPEECH_TRIM_MAX at the sharpest).
_SPEECH_TRIM_FRAC = {"much_slower": 0.0, "slower": 0.0, "natural": 0.0,
                     "faster": 0.5, "much_faster": 0.85}


def _clone(document: dict) -> dict:
    """A working copy with fresh timeline/operations lists and no stale resolve.
    Deep-copies the mutable edit surface only; the rest is shared (read-only)."""
    doc = dict(document)
    doc["timeline"] = [dict(s) for s in (document.get("timeline") or [])]
    doc["operations"] = [dict(o) for o in (document.get("operations") or [])]
    if document.get("layout_regions"):
        doc["layout_regions"] = [dict(r) for r in document["layout_regions"]]
    doc.pop("resolved", None)
    return doc


def _new_seg_id() -> str:
    return f"u{uuid.uuid4().hex[:8]}"


def _segments_from_cut(rc: ResolvedCut) -> List[dict]:
    """A resolved cut -> one or more main-line segments (one per keep_span so a
    breath-excised jump-cut survives). The segment shape matches what the timeline
    / render read; adjacent contiguous slices are later merged by
    ``arrange.heal_adjacent_cuts`` in ``observe.resolve_doc``. ``rc.keep_spans`` is the
    canonical ``[(in_ms, out_ms), ...]`` (normalized in ``_MapIndex.resolve``)."""
    spans = rc.keep_spans or [(rc.src_in_ms, rc.src_out_ms)]
    out: List[dict] = []
    for a, b in spans:
        in_ms, out_ms = int(a), int(b)
        if out_ms <= in_ms:
            continue
        out.append({
            "seg_id": _new_seg_id(),
            "file_id": rc.file_id,
            "in_ms": in_ms,
            "out_ms": out_ms,
            "axis": "speech" if rc.channel == "said" else "any",
            "beat_id": None,
            "content": rc.label,
            "rationale": rc.reason or None,
            "priority": 3,
            "cut_in_cost": 0.0,
            "cut_out_cost": 0.0,
            "warnings": [],
            "mute": True if rc.mute else None,
            "ref": rc.ref or None,
            "level": rc.level,
            # av_coupling_authoritative.plan.md: this cut's baked authoritative
            # audio coupling -- default file_id/0 (same-source) when the
            # resolved cut carries none.
            "audio_file_id": rc.audio_file_id or rc.file_id,
            "audio_offset_ms": rc.audio_offset_ms,
        })
    return out


def _snap_cut_to_sentences(rc: ResolvedCut) -> ResolvedCut:
    """Sentence-align a SPEECH cut's kept spans so it never opens/closes
    mid-thought and no dead-air jump-cut seam severs a sentence (which is how a
    placed line gets a filler head or loses its payload to a later remove). A
    no-op for non-speech cuts or footage with no transcript. Reuses
    ``footage_map.snap_speech_spans_to_sentences`` -- the only place the brain's
    placements meet the sentence grid."""
    if rc.channel != "said":
        return rc
    spans = rc.keep_spans or [(rc.src_in_ms, rc.src_out_ms)]
    snapped = footage_map.snap_speech_spans_to_sentences(rc.file_id, spans)
    if snapped == spans:
        return rc
    return dataclasses.replace(
        rc, src_in_ms=snapped[0][0], src_out_ms=snapped[-1][1],
        keep_spans=(None if len(snapped) == 1 else snapped))


def _program_end(document: dict) -> int:
    _, total = layers.spine_spans(document.get("timeline") or [])
    return total


def place_span(document: dict, file_id: str, *, in_ms: int, out_ms: int,
               channel: str = "V1", at: Optional[int] = None,
               from_ms: Optional[int] = None, audio: Optional[str] = None,
               axis: str = "any", content: str = "", reason: str = "") -> dict:
    """Place an ARBITRARY source span ``[in_ms, out_ms]`` of ``file_id`` -- the
    continuous-editing verb. Unlike ``place`` (which can only place a pre-baked
    map ref), this addresses the clip as a continuous source, so the brain can
    lift a person's *silent* reaction, a held beat, or any window the awareness
    lanes revealed -- not just a minted cut.

    channel="V1": insert on the main line at index ``at`` (default append).
    channel="V2": lay a video cutaway over the program at ``from_ms``.
    ``axis="speech"`` marks the span as audio-load-bearing (protects it in the
    weld/coverage). Boundaries are expected to be seam-snapped by the caller."""
    try:
        a, b = int(in_ms), int(out_ms)
    except (TypeError, ValueError):
        raise EditReject(f"in_ms/out_ms must be integers (got in_ms={in_ms!r}, "
                         f"out_ms={out_ms!r}).")
    if not file_id:
        raise EditReject("place_span needs a file_id.")
    if b <= a:
        raise EditReject(f"place_span's span collapses (out {b}ms <= in {a}ms).")

    doc = _clone(document)
    if channel.upper() == "V1":
        seg = {
            "seg_id": _new_seg_id(),
            "file_id": file_id,
            "in_ms": a,
            "out_ms": b,
            "axis": "speech" if axis == "speech" else "any",
            "beat_id": None,
            "content": content,
            "rationale": reason or None,
            "priority": 3,
            "cut_in_cost": 0.0,
            "cut_out_cost": 0.0,
            "warnings": [],
            "mute": True if audio == "mute" else None,
            "ref": None,
            "level": "span",
        }
        tl = doc["timeline"]
        idx = len(tl) if at is None else max(0, min(int(at), len(tl)))
        doc["timeline"] = tl[:idx] + [seg] + tl[idx:]
        return doc

    anchor = _program_end(doc) if from_ms is None else max(0, int(from_ms))
    span = b - a
    doc["operations"].append({
        "op_id": f"sp_{uuid.uuid4().hex[:6]}",
        "type": "place_video",
        "source_file_id": file_id,
        "src_in_ms": a,
        "src_out_ms": b,
        "from_ms": anchor,
        "to_ms": anchor + span,
        "layout": layers.DEFAULT_LAYOUT,
        "z": layers.Z_COVERAGE,
        "opacity": 1.0,
        "rationale": reason or None,
        "warnings": [],
        "mute": False if audio == "keep" else True,
    })
    return doc


# --------------------------------------------------------------------------
# Verbs
# --------------------------------------------------------------------------

def place(document: dict, index: _MapIndex, ref: str, *,
          level: str = "balanced", channel: str = "V1",
          at: Optional[int] = None, from_ms: Optional[int] = None,
          audio: Optional[str] = None, reason: str = "",
          piece: Optional[int] = None) -> dict:
    """Add a cut from a map ``ref``.

    channel="V1": insert on the main line at index ``at`` (default append).
    channel="V2": lay a SILENT (unless audio="keep") video cutaway over the
    program at ``from_ms`` (default the current program end -> effectively a tail
    cutaway; pass from_ms to place it precisely).
    ``piece`` (v4_cluster_read_act.plan.md Part C): place just ONE beat of a
    multi-beat cluster (the 1-based position shown in the Beat Index /
    read_state) instead of the whole moment; ``level`` is ignored when set.
    """
    p = Placement(ref=ref, level=level, track=(0 if channel.upper() == "V1" else 1),
                  from_ms=from_ms, reason=reason, audio=audio, piece=piece)
    rc = index.resolve(p)
    if rc is None:
        raise EditReject(f"no placeable moment matching ref '{ref}' at level "
                         f"'{level}'" + (f" piece {piece}" if piece is not None else "")
                         + " -- unknown ref, or piece out of range.")
    rc = _snap_cut_to_sentences(rc)
    doc = _clone(document)

    if channel.upper() == "V1":
        segs = _segments_from_cut(rc)
        if not segs:
            raise EditReject(f"ref '{ref}' resolved to no placeable span (its keep_spans "
                             "excise the whole cut).")
        tl = doc["timeline"]
        idx = len(tl) if at is None else max(0, min(int(at), len(tl)))
        doc["timeline"] = tl[:idx] + segs + tl[idx:]
        return doc

    # V2 (and up): a program-anchored video cutaway operation.
    anchor = _program_end(doc) if from_ms is None else max(0, int(from_ms))
    span = rc.src_out_ms - rc.src_in_ms
    doc["operations"].append({
        "op_id": f"ov_{uuid.uuid4().hex[:6]}",
        "type": "place_video",
        "source_file_id": rc.file_id,
        "src_in_ms": int(rc.src_in_ms),
        "src_out_ms": int(rc.src_in_ms + span),
        "from_ms": anchor,
        "to_ms": anchor + span,
        "layout": layers.DEFAULT_LAYOUT,
        "z": layers.Z_COVERAGE,
        "opacity": 1.0,
        "rationale": reason or None,
        "warnings": [],
        "mute": False if audio == "keep" else True,
    })
    return doc


_AUDIO_ROLES = ("music", "voiceover", "sfx")
_AUDIO_KINDS = ("bed", "replace", "sfx")


def place_audio(
    document: dict, *, source_file_id: str, role: str, from_ms: int, to_ms: int,
    src_in_ms: int = 0, src_out_ms: Optional[int] = None,
    gain_db: float = 0.0, duck_db: float = 0.0, audio_kind: str = "bed",
    asset_dur_ms: Optional[int] = None, reason: str = "",
) -> dict:
    """Place a music/voiceover/SFX bed on program window [from_ms, to_ms]
    (audio_brain.plan.md Phase 1a) -- the op ``layers.resolve``'s
    ``place_audio`` branch already reads; this is what actually CREATES it.

    ``role="voiceover"`` is stored as-is (so observe/the brain can see its
    own intent) -- ``layers.resolve`` is what maps it onto the plain
    ``ROLE_DIALOGUE`` bed at render time (no auto-duck PRIORITY; VO is
    balanced by the brain like any other bed, never privileged by code).

    Never auto-loops or auto-stretches: if the requested window is longer
    than the source has (``asset_dur_ms``), ``src_out_ms`` clamps to the
    asset's actual end while ``to_ms`` keeps the brain's full requested
    window -- the shortfall renders as silence for the remainder (no
    special-casing needed downstream) and is exactly what observe's digest
    surfaces as a fact (``asset_dur_ms < window_ms``) for the brain to act
    on, never silently patched over here.

    `duck_db` defaults to 0 (no duck) -- ``layers._apply_levels`` only ever
    ducks when the caller (the brain, via this op or the `duck` verb)
    explicitly sets it negative."""
    if role not in _AUDIO_ROLES:
        raise EditReject(f"role must be one of {sorted(_AUDIO_ROLES)} (got {role!r}).")
    if not source_file_id:
        raise EditReject("place_audio needs a source file id.")
    try:
        f, t = int(from_ms), int(to_ms)
        s_in = max(0, int(src_in_ms))
    except (TypeError, ValueError):
        raise EditReject(f"from_ms/to_ms/src_in_ms must be integers (got "
                         f"from_ms={from_ms!r}, to_ms={to_ms!r}, src_in_ms={src_in_ms!r}).")
    if t <= f:
        raise EditReject(f"place_audio's program window collapses (to_ms {t}ms <= "
                         f"from_ms {f}ms).")
    window = t - f

    if src_out_ms is not None:
        try:
            s_out = int(src_out_ms)
        except (TypeError, ValueError):
            raise EditReject(f"src_out_ms must be an integer (got {src_out_ms!r}).")
    elif asset_dur_ms is not None:
        s_out = min(int(asset_dur_ms), s_in + window)
    else:
        # Duration unknown (a lookup miss, not a missing asset) -- assume the
        # requested window is available rather than guessing it's short.
        s_out = s_in + window
    if asset_dur_ms is not None:
        s_out = min(s_out, int(asset_dur_ms))
    if s_out <= s_in:
        raise EditReject(f"place_audio's source span collapses (src_out {s_out}ms <= "
                         f"src_in {s_in}ms) -- the asset may be shorter than src_in_ms.")

    doc = _clone(document)
    doc["operations"].append({
        "op_id": f"pa_{uuid.uuid4().hex[:6]}",
        "type": "place_audio",
        "role": role,
        "source_file_id": source_file_id,
        "src_in_ms": s_in, "src_out_ms": s_out,
        "from_ms": f, "to_ms": t,
        "gain_db": float(gain_db), "duck_db": float(duck_db),
        "audio_kind": audio_kind if audio_kind in _AUDIO_KINDS else "bed",
        "rationale": reason or None,
    })
    return doc


def set_gain(document: dict, target_id: str, *, gain_db: float) -> dict:
    """Set a layer's OWN level (dB) -- a main-line seg's coupled audio, or an
    A2 `place_audio` bed. Separate from `duck` (a side-chain reduction applied
    only where a bed overlaps dialogue); this is the layer's base level."""
    try:
        g = float(gain_db)
    except (TypeError, ValueError):
        raise EditReject(f"gain_db must be a number (got {gain_db!r}).")
    doc = _clone(document)
    seg = next((s for s in doc["timeline"] if s.get("seg_id") == target_id), None)
    if seg is not None:
        if seg.get("gain_db") == g:
            raise EditReject(f"gain_db is already {g} for '{target_id}' -- unchanged.")
        seg["gain_db"] = g
        return doc
    op = next((o for o in doc["operations"]
               if o.get("op_id") == target_id and o.get("type") == "place_audio"), None)
    if op is None:
        raise EditReject(f"no main-line cut or place_audio bed with id '{target_id}'.")
    if op.get("gain_db") == g:
        raise EditReject(f"gain_db is already {g} for '{target_id}' -- unchanged.")
    op["gain_db"] = g
    return doc


def duck(document: dict, target_id: str, *, amount_db: float) -> dict:
    """Set an A2 bed's explicit duck (dB, typically <=0) -- applied ONLY where
    it overlaps live spine dialogue (`layers._apply_levels`); 0 clears it.
    There is no auto-duck: a bed ducks only by what this (or `place_audio`)
    sets. Only targets a `place_audio` op -- the live spine track is never a
    valid target (nothing ducks it; it's what other beds duck under)."""
    try:
        d = float(amount_db)
    except (TypeError, ValueError):
        raise EditReject(f"amount_db must be a number (got {amount_db!r}).")
    doc = _clone(document)
    op = next((o for o in doc["operations"]
               if o.get("op_id") == target_id and o.get("type") == "place_audio"), None)
    if op is None:
        raise EditReject(f"no place_audio bed with id '{target_id}'.")
    if op.get("duck_db") == d:
        raise EditReject(f"duck_db is already {d} for '{target_id}' -- unchanged.")
    op["duck_db"] = d
    return doc


def fade_audio(document: dict, target_id: str, *,
               in_ms: Optional[int] = None, out_ms: Optional[int] = None) -> dict:
    """Set a fade envelope on a layer's own edges -- a main-line seg's coupled
    audio, or an A2 bed. `in_ms`/`out_ms` are fade DURATIONS in ms (0 clears
    that edge); an omitted edge is left as-is. Hard start/stop by default --
    nothing fades unless this sets it."""
    if in_ms is None and out_ms is None:
        raise EditReject("fade_audio needs at least one of in_ms/out_ms.")
    try:
        f_in = int(in_ms) if in_ms is not None else None
        f_out = int(out_ms) if out_ms is not None else None
    except (TypeError, ValueError):
        raise EditReject(f"in_ms/out_ms must be integers (got in_ms={in_ms!r}, "
                         f"out_ms={out_ms!r}).")
    if (f_in is not None and f_in < 0) or (f_out is not None and f_out < 0):
        raise EditReject(f"fade durations must be >= 0 (got in_ms={f_in}, out_ms={f_out}).")
    doc = _clone(document)
    target = next((s for s in doc["timeline"] if s.get("seg_id") == target_id), None)
    if target is None:
        target = next((o for o in doc["operations"]
                      if o.get("op_id") == target_id and o.get("type") == "place_audio"), None)
    if target is None:
        raise EditReject(f"no main-line cut or place_audio bed with id '{target_id}'.")
    provided = [(k, v) for k, v in (("fade_in_ms", f_in), ("fade_out_ms", f_out)) if v is not None]
    if provided and all(target.get(k) == v for k, v in provided):
        raise EditReject("fade_audio: the given edge(s) already match the current fade -- unchanged.")
    if f_in is not None:
        target["fade_in_ms"] = f_in
    if f_out is not None:
        target["fade_out_ms"] = f_out
    return doc


def crossfade(document: dict, seam_seg_id: str, *, ms: int) -> dict:
    """Cross-dissolve the spine AUDIO across the seam just before main-line cut
    `seam_seg_id`: the previous cut's audio and the next cut's audio overlap by
    `ms` (split evenly, each extending toward the other) and fade across that
    overlap (`layers._apply_crossfades`). One crossfade per seam (re-issuing
    replaces); `ms<=0` clears. The first segment has no seam before it, and a
    negative `ms` makes no sense here (unlike `split_edit`'s signed J/L offset)."""
    try:
        m = int(ms)
    except (TypeError, ValueError):
        raise EditReject(f"ms must be an integer (got {ms!r}).")
    if m < 0:
        raise EditReject(f"ms must be >= 0 (got {m}); crossfade has no signed J/L sense "
                         "like split_edit's audio_offset_ms.")
    timeline = document.get("timeline") or []
    i = next((idx for idx, s in enumerate(timeline) if s.get("seg_id") == seam_seg_id), None)
    if i is None or i == 0:
        raise EditReject(f"no seam before '{seam_seg_id}' -- unknown id, or it's the "
                         "first main-line cut (no seam precedes it).")
    existing = next((o for o in document.get("operations") or []
                     if o.get("type") == "crossfade" and o.get("seam_seg_id") == seam_seg_id), None)
    if m == 0 and existing is None:
        raise EditReject(f"no crossfade at the seam before '{seam_seg_id}' to clear.")
    if existing is not None and m > 0 and existing.get("ms") == m:
        raise EditReject(f"crossfade at the seam before '{seam_seg_id}' is already {m}ms -- unchanged.")
    doc = _clone(document)
    doc["operations"] = [
        o for o in doc["operations"]
        if not (o.get("type") == "crossfade" and o.get("seam_seg_id") == seam_seg_id)
    ]
    if m > 0:
        doc["operations"].append({
            "op_id": f"xf_{uuid.uuid4().hex[:6]}", "type": "crossfade",
            "seam_seg_id": seam_seg_id, "ms": m,
        })
    return doc


def replace_audio(document: dict, target_id: str, *,
                  source_file_id: str, src_in_ms: int, src_out_ms: int) -> dict:
    """Override a main-line cut's coupled audio source with an explicit pick --
    the escape hatch for outlook authoritative routing (audio_sync.plan.md):
    this wins over the auto-computed route (`layers.resolve` checks the
    override first). Also the general tool for swapping a cut's sound to any
    other file's span regardless of routing."""
    try:
        s_in, s_out = int(src_in_ms), int(src_out_ms)
    except (TypeError, ValueError):
        raise EditReject(f"src_in_ms/src_out_ms must be integers (got "
                         f"src_in_ms={src_in_ms!r}, src_out_ms={src_out_ms!r}).")
    if not source_file_id:
        raise EditReject("replace_audio needs a source file id.")
    if s_out <= s_in:
        raise EditReject(f"replace_audio's span collapses (src_out {s_out}ms <= "
                         f"src_in {s_in}ms).")
    doc = _clone(document)
    seg = next((s for s in doc["timeline"] if s.get("seg_id") == target_id), None)
    if seg is None:
        raise EditReject(f"no main-line cut with id '{target_id}'.")
    new_override = {"source_file_id": source_file_id, "src_in_ms": s_in, "src_out_ms": s_out}
    if seg.get("audio_override") == new_override:
        raise EditReject(f"'{target_id}' already has this exact audio override -- unchanged.")
    seg["audio_override"] = new_override
    return doc


def remove(document: dict, target_id: str) -> dict:
    """Drop a main-line segment (by seg_id), an operation (by op_id), or a layout
    region (by region_id) -- and tear the split/PiP down SYMMETRICALLY:
      * removing an op that a region references also drops that now-dangling
        region, and
      * removing a region also drops the coverage op(s) it fed (so no orphaned
        full-frame silent paste-over is left behind).
    A cell pointing at 'spine' is the main line, never dropped."""
    doc = _clone(document)
    regions = doc.get("layout_regions") or []
    before = len(doc["timeline"]) + len(doc["operations"]) + len(regions)

    ops_to_drop = {target_id}     # target may itself be an op id
    kept_regions: List[dict] = []
    for r in regions:
        cell_layers = {(sel or {}).get("layer")
                       for sel in (r.get("cells") or {}).values()}
        if r.get("region_id") == target_id or target_id in cell_layers:
            # This region goes (removed directly, or its op was the target): also
            # retire the coverage op(s) it fed so nothing dangles.
            ops_to_drop |= {ly for ly in cell_layers if ly and ly != "spine"}
        else:
            kept_regions.append(r)

    doc["timeline"] = [s for s in doc["timeline"] if s.get("seg_id") != target_id]
    doc["operations"] = [o for o in doc["operations"]
                         if o.get("op_id") not in ops_to_drop]
    if kept_regions:
        doc["layout_regions"] = kept_regions
    else:
        doc.pop("layout_regions", None)
    after = len(doc["timeline"]) + len(doc["operations"]) + len(kept_regions)
    if after == before:
        raise EditReject(f"no main-line cut, operation, or layout region with id "
                         f"'{target_id}' -- nothing removed.")
    return doc


def move(document: dict, target_id: str,
        to_index: Optional[int] = None, to_ms: Optional[int] = None) -> dict:
    """Reorder a main-line segment to ``to_index`` (0-based), OR reposition a
    placed op (a V2 ``place_video`` cutaway or an A2 ``place_audio`` bed) to
    start at program ``to_ms``, keeping its own duration (audio_brain.plan.md:
    extends this one verb to beds instead of a separate op-reposition verb).
    Exactly one of ``to_index``/``to_ms`` applies, matching whichever kind of
    id ``target_id`` resolves to."""
    doc = _clone(document)
    if to_index is not None:
        tl = doc["timeline"]
        src = next((i for i, s in enumerate(tl) if s.get("seg_id") == target_id), None)
        if src is None:
            raise EditReject(f"no main-line cut with id '{target_id}'.")
        dst = max(0, min(int(to_index), len(tl) - 1))
        if dst == src:
            raise EditReject(f"'{target_id}' is already at index {src} -- unchanged.")
        seg = tl.pop(src)
        tl.insert(dst, seg)
        return doc
    if to_ms is not None:
        op = next((o for o in doc["operations"] if o.get("op_id") == target_id
                   and o.get("type") in ("place_video", "place_audio")), None)
        if op is None:
            raise EditReject(f"no V2/A2 operation with id '{target_id}'.")
        try:
            new_from = max(0, int(to_ms))
        except (TypeError, ValueError):
            raise EditReject(f"to_ms must be an integer (got {to_ms!r}).")
        dur = int(op.get("to_ms", 0)) - int(op.get("from_ms", 0))
        if dur <= 0:
            raise EditReject(f"'{target_id}' has a degenerate duration ({dur}ms) -- cannot move.")
        if new_from == int(op.get("from_ms", 0)):
            raise EditReject(f"'{target_id}' is already at from_ms={new_from} -- unchanged.")
        op["from_ms"], op["to_ms"] = new_from, new_from + dur
        return doc
    raise EditReject("move needs exactly one of to_index (for a main-line cut) or "
                     "to_ms (for a V2/A2 operation).")


def trim(document: dict, target_id: str, *,
         in_ms: Optional[int] = None, out_ms: Optional[int] = None,
         delta_in_ms: Optional[int] = None, delta_out_ms: Optional[int] = None) -> dict:
    """Adjust the SOURCE span of a main-line segment, or of a placed op (a V2
    ``place_video`` cutaway or an A2 ``place_audio`` bed -- audio_brain.plan.md
    extends this verb to beds instead of adding a separate one).

    Absolute (``in_ms``/``out_ms``) or relative (``delta_in_ms``/``delta_out_ms``,
    e.g. delta_in_ms=+200 nudges the in-point 200ms later). The result is clamped
    so out stays > in."""
    if in_ms is None and out_ms is None and delta_in_ms is None and delta_out_ms is None:
        raise EditReject("trim needs a source edit: in_ms/out_ms (ABSOLUTE source "
                         "ms) or delta_in_ms/delta_out_ms (RELATIVE nudge).")
    doc = _clone(document)
    seg = next((s for s in doc["timeline"] if s.get("seg_id") == target_id), None)
    if seg is not None:
        cur_in, cur_out = int(seg["in_ms"]), int(seg["out_ms"])
        new_in = int(in_ms) if in_ms is not None else cur_in + int(delta_in_ms or 0)
        new_out = int(out_ms) if out_ms is not None else cur_out + int(delta_out_ms or 0)
        new_in = max(0, new_in)
        if new_out <= new_in:
            raise EditReject(
                f"trim collapses the span (out {new_out}ms <= in {new_in}ms). "
                "in_ms/out_ms are ABSOLUTE source times; to nudge by a relative "
                "amount pass delta_in_ms/delta_out_ms instead.")
        if new_in == cur_in and new_out == cur_out:
            raise EditReject("trim leaves the span unchanged -- the in/out you gave "
                             "equal the current span.")
        seg["in_ms"], seg["out_ms"] = new_in, new_out
        return doc
    op = next((o for o in doc["operations"] if o.get("op_id") == target_id
               and o.get("type") in ("place_video", "place_audio")), None)
    if op is None:
        raise EditReject(f"no main-line cut or trimmable op with id '{target_id}'.")
    cur_in, cur_out = int(op["src_in_ms"]), int(op["src_out_ms"])
    new_in = int(in_ms) if in_ms is not None else cur_in + int(delta_in_ms or 0)
    new_out = int(out_ms) if out_ms is not None else cur_out + int(delta_out_ms or 0)
    new_in = max(0, new_in)
    if new_out <= new_in:
        raise EditReject(
            f"trim collapses this {op['type']} op's span (out {new_out}ms <= in "
            f"{new_in}ms). in_ms/out_ms are ABSOLUTE source times; use "
            "delta_in_ms/delta_out_ms for a relative nudge.")
    if new_in == cur_in and new_out == cur_out:
        raise EditReject("trim leaves this op's source span unchanged.")
    op["src_in_ms"], op["src_out_ms"] = new_in, new_out
    op["to_ms"] = int(op.get("from_ms", 0)) + (new_out - new_in)
    return doc


def split_edit(document: dict, seam_seg_id: str, *, audio_offset_ms: int) -> dict:
    """J/L cut: decouple the AUDIO boundary from the VIDEO boundary at the seam
    just BEFORE main-line segment ``seam_seg_id``.

    offset > 0 (L-cut): the previous cut's audio lingers over the incoming
    picture. offset < 0 (J-cut): the incoming cut's audio leads under the
    previous picture. offset 0 clears any existing split at that seam.

    One split per seam: re-issuing replaces the previous offset. The first
    segment has no seam before it. The offset itself is applied at resolve time
    (``layers._apply_split_edits``), so it survives welds that keep the seam and
    simply no-ops if the seam disappears."""
    try:
        offset = int(audio_offset_ms)
    except (TypeError, ValueError):
        raise EditReject(f"audio_offset_ms must be an integer (got {audio_offset_ms!r}).")
    timeline = document.get("timeline") or []
    idx = next((i for i, s in enumerate(timeline)
                if s.get("seg_id") == seam_seg_id), None)
    if idx is None or idx == 0:
        raise EditReject(f"no seam before '{seam_seg_id}' -- unknown id, or it's the "
                         "first main-line cut (no seam precedes it).")
    existing = next((o for o in document.get("operations") or []
                     if o.get("type") == "split_edit" and o.get("seam_seg_id") == seam_seg_id), None)
    if offset == 0 and existing is None:
        raise EditReject(f"no split_edit at the seam before '{seam_seg_id}' to clear.")
    if existing is not None and offset != 0 and existing.get("audio_offset_ms") == offset:
        raise EditReject(f"split_edit at the seam before '{seam_seg_id}' is already "
                         f"audio_offset_ms={offset} -- unchanged.")

    doc = _clone(document)
    ops = [o for o in doc["operations"]
           if not (o.get("type") == "split_edit"
                   and o.get("seam_seg_id") == seam_seg_id)]
    if offset != 0:
        ops.append({
            "op_id": f"se_{uuid.uuid4().hex[:6]}",
            "type": "split_edit",
            "seam_seg_id": seam_seg_id,
            "audio_offset_ms": offset,
        })
    doc["operations"] = ops
    return doc


def set_audio(document: dict, target_id: str, *, mute: bool) -> dict:
    """Mute/unmute the SOURCE audio of a main-line segment (A1) or a V2 cutaway,
    keeping its picture. mute=True silences; mute=False plays the source sound."""
    doc = _clone(document)
    seg = next((s for s in doc["timeline"] if s.get("seg_id") == target_id), None)
    if seg is not None:
        if bool(seg.get("mute")) == bool(mute):
            raise EditReject(f"'{target_id}' is already {'muted' if mute else 'unmuted'} -- unchanged.")
        seg["mute"] = True if mute else None
        return doc
    op = next((o for o in doc["operations"]
               if o.get("op_id") == target_id and o.get("type") == "place_video"), None)
    if op is None:
        raise EditReject(f"no main-line cut or V2 cutaway with id '{target_id}'.")
    if bool(op.get("mute")) == bool(mute):
        raise EditReject(f"'{target_id}' is already {'muted' if mute else 'unmuted'} -- unchanged.")
    op["mute"] = bool(mute)
    return doc


def set_arc_intent(document: dict, target_id: str, *, intent: str) -> dict:
    """Tag a main-line segment (A1) or a V2 cutaway with its position in the
    emotional arc (one of ARC_INTENTS): calm/build/peak/resolve. Invisible
    by default -- has no effect until the user's arc intensity dial is
    raised above 0."""
    if intent not in ARC_INTENTS:
        raise EditReject(f"intent must be one of {sorted(ARC_INTENTS)} (got {intent!r}).")
    doc = _clone(document)
    seg = next((s for s in doc["timeline"] if s.get("seg_id") == target_id), None)
    if seg is not None:
        if seg.get("arc_intent") == intent:
            raise EditReject(f"'{target_id}' is already tagged arc_intent={intent!r} -- unchanged.")
        seg["arc_intent"] = intent
        return doc
    op = next((o for o in doc["operations"]
               if o.get("op_id") == target_id and o.get("type") == "place_video"), None)
    if op is None:
        raise EditReject(f"no main-line cut or V2 cutaway with id '{target_id}'.")
    if op.get("arc_intent") == intent:
        raise EditReject(f"'{target_id}' is already tagged arc_intent={intent!r} -- unchanged.")
    op["arc_intent"] = intent
    return doc


_BEAT_NEEDS = ("required", "optional")


def _norm_beat(entry) -> dict | None:
    """Normalize one `structure` entry to a beat {beat, need}. Accepts a
    {beat, need} dict OR a bare string (the LLM's old habit); an empty beat
    is dropped (returns None). `need` defaults to 'required' -- a beat is a
    commitment unless the brain explicitly marks it optional
    (brain_plan_altitude.plan.md SS2.1)."""
    if isinstance(entry, dict):
        beat = str(entry.get("beat") or "").strip()
        need = entry.get("need")
    else:
        beat, need = str(entry or "").strip(), None
    if not beat:
        return None
    return {"beat": beat, "need": need if need in _BEAT_NEEDS else "required"}


def set_plan(document: dict, *, purpose=None, carries=None,
             structure=None, watch=None, note=None) -> dict:
    """Write/replace document['plan'] -- the brain's durable, ordered plan
    (brain_plan_mechanism.plan.md). Full replace of the plan block; bumps
    `rev` and records an optional `updated_note`. `structure` entries are
    normalized to {beat, need} beats (brain_plan_altitude.plan.md). Pure:
    returns a new doc."""
    doc = _clone(document)
    prev_rev = int((document.get("plan") or {}).get("rev") or 0)
    plan = {
        "purpose": (purpose or "").strip() or None,
        "carries": [str(c).strip() for c in (carries or []) if str(c).strip()],
        "structure": [b for b in (_norm_beat(s) for s in (structure or [])) if b],
        "watch": [str(w).strip() for w in (watch or []) if str(w).strip()],
        "rev": prev_rev + 1,
    }
    if note and note.strip():
        plan["updated_note"] = note.strip()
    doc["plan"] = plan
    return doc


def wrap_up(document: dict, *, summary=None, open_questions=None, notes=None) -> dict:
    """Write the user-facing WRAP-UP onto the document's surface fields
    (brain_plan_conformance.plan.md Part A): `summary` -- one short, clean
    user-facing recap of what you built and, when one exists, the compromise
    you made and why; `open_questions` / `notes` -- anything the user should
    decide or know. This is the ONLY brain write path to the surface the API
    shows the user (GET /threads/{id}: open_questions; the stored reply/summary).
    Full replace of whichever field is passed; a field left None is untouched.
    Pure: returns a new doc (so the turn persists and the reply carries it)."""
    if summary is None and open_questions is None and notes is None:
        return document
    doc = _clone(document)
    if summary is not None:
        doc["summary"] = str(summary).strip()
    if open_questions is not None:
        doc["open_questions"] = [str(q).strip() for q in open_questions if str(q).strip()]
    if notes is not None:
        doc["notes"] = [str(n).strip() for n in notes if str(n).strip()]
    return doc


def set_grade(
    document: dict,
    target_id: Optional[str],
    *,
    warmth: Optional[float] = None,
    tint: Optional[float] = None,
    brightness: Optional[float] = None,
    contrast: Optional[float] = None,
    saturation: Optional[float] = None,
) -> dict:
    """NL steering (color_grading.plan.md SS10): "warmer", "less teal", "more
    contrast" -> named, bounded dials (-1..1, unset = 0) -> a deterministic
    CDL delta (grade.steer.solve_steer_grade), composed onto whatever grade
    override the target already has (stacks with earlier steers instead of
    replacing them). With target_id -> that cut; without -> every main-line
    cut (a whole-document steer)."""
    delta = solve_steer_grade(
        warmth=warmth, tint=tint, brightness=brightness, contrast=contrast, saturation=saturation,
    )
    if delta == Grade():
        raise EditReject("set_grade's dials are all 0/omitted -- nothing to nudge.")

    doc = _clone(document)
    if target_id:
        seg = next((s for s in doc["timeline"] if s.get("seg_id") == target_id), None)
        targets: List[dict] = [seg] if seg is not None else []
        if not targets:
            op = next((o for o in doc["operations"]
                       if o.get("op_id") == target_id and o.get("type") == "place_video"), None)
            targets = [op] if op is not None else []
        if not targets:
            raise EditReject(f"no main-line cut or V2 cutaway with id '{target_id}'.")
    else:
        targets = list(doc["timeline"]) + [o for o in doc["operations"] if o.get("type") == "place_video"]
    if not targets:
        raise EditReject("no cuts to grade -- the timeline is empty.")

    for t in targets:
        existing = Grade.from_dict(t.get("grade"))
        t["grade"] = compose(existing, delta, 1.0).to_dict()
    return doc


# Default cell assignment per template: (spine cell, added-ref cell).
_SPLIT_CELLS = {
    "split_h": ("left", "right"),
    "split_v": ("top", "bottom"),
    "pip": ("base", "inset"),
}


def split_screen(document: dict, index: _MapIndex, ref: Optional[str] = None, *,
                 file: Optional[str] = None,
                 in_ms: Optional[int] = None, out_ms: Optional[int] = None,
                 template: str = "split_h", from_ms: int, to_ms: int,
                 level: str = "balanced", audio: Optional[str] = None,
                 reason: str = "") -> dict:
    """Show the ongoing main line (V1) AND a second source side-by-side (or PiP)
    over the program window [from_ms, to_ms].

    The added cell's source is EITHER a pre-baked map ``ref`` OR any raw source
    window ``file`` + ``in_ms`` + ``out_ms`` (the continuous-source path, mirroring
    ``place_span`` -- so a silent listener or any window the awareness lanes reveal
    can fill a cell, not just a minted cut). Adds a V2 place_video op and a LAYOUT
    REGION assigning the spine to one cell and the op to the other per the template
    (split_h/split_v/pip). The added cell is silent by default (audio="keep" plays
    its sound). This is a user-owned look -- the brain should ``ask_user`` before
    calling it."""
    cells = _SPLIT_CELLS.get(template)
    if cells is None:
        raise EditReject(f"template must be one of {sorted(_SPLIT_CELLS)} (got {template!r}).")
    # Resolve the added source: a map ref, else a raw (file, in, out) window.
    if ref:
        rc = index.resolve(Placement(ref=ref, level=level))
        if rc is None:
            raise EditReject(f"no placeable moment matching ref '{ref}' at level '{level}'.")
        src_file, src_in, src_out = rc.file_id, int(rc.src_in_ms), int(rc.src_out_ms)
    elif file and in_ms is not None and out_ms is not None:
        try:
            src_in, src_out = int(in_ms), int(out_ms)
        except (TypeError, ValueError):
            raise EditReject(f"in_ms/out_ms must be integers (got in_ms={in_ms!r}, "
                             f"out_ms={out_ms!r}).")
        if src_out <= src_in:
            raise EditReject(f"the second cell's source window collapses (out "
                             f"{src_out}ms <= in {src_in}ms).")
        src_file = file
    else:
        raise EditReject("split_screen needs a second-cell source: either `ref` "
                         "(a map moment) or `file`+`in_ms`+`out_ms` (a raw window).")
    try:
        f, t = int(from_ms), int(to_ms)
    except (TypeError, ValueError):
        raise EditReject(f"from_ms/to_ms must be integers (got from_ms={from_ms!r}, "
                         f"to_ms={to_ms!r}).")
    if t <= f:
        raise EditReject(f"the program window collapses (to_ms {t}ms <= from_ms {f}ms).")
    span = min(t - f, src_out - src_in)
    if span < 200:
        raise EditReject(f"the resulting split-screen span ({span}ms, the shorter of "
                         "the program window and the source window) is under the "
                         "200ms floor.")

    doc = _clone(document)
    doc.setdefault("layout_regions", list(document.get("layout_regions") or []))
    doc["layout_regions"] = [dict(r) for r in doc["layout_regions"]]
    spine_cell, ref_cell = cells
    op_id = f"sp_{uuid.uuid4().hex[:6]}"
    doc["operations"].append({
        "op_id": op_id,
        "type": "place_video",
        # Marks this op as a split/PiP cell feed (not a standalone cutaway), so a
        # region teardown can retire it and validate can spot it if orphaned.
        "purpose": "split_cell",
        "source_file_id": src_file,
        "src_in_ms": src_in,
        "src_out_ms": src_in + span,
        "from_ms": f,
        "to_ms": f + span,
        "layout": layers.DEFAULT_LAYOUT,
        "z": layers.Z_COVERAGE,
        "opacity": 1.0,
        "rationale": reason or None,
        "warnings": [],
        "mute": False if audio == "keep" else True,
    })
    doc["layout_regions"].append({
        "region_id": f"lr_{uuid.uuid4().hex[:6]}",
        "from_ms": f,
        "to_ms": f + span,
        "template": template,
        "cells": {spine_cell: {"layer": "spine"}, ref_cell: {"layer": op_id}},
    })
    return doc


def tighten(document: dict, index: _MapIndex, *,
            seg_id: Optional[str] = None, level: str = "tight") -> dict:
    """Re-take main-line cut(s) at a different energy ``level`` (pacing).

    With ``seg_id`` -> just that cut; without -> every main-line cut that has a
    map ref and a variant at ``level``. A cut whose ref lacks that level is left
    as-is (partial tighten is fine). Re-resolves each cut's span from the map at
    the new level, preserving order + provenance."""
    if seg_id is not None and not any(
            s.get("seg_id") == seg_id for s in document.get("timeline") or []):
        raise EditReject(f"no main-line cut with id '{seg_id}'.")
    doc = _clone(document)
    changed = False
    new_tl: List[dict] = []
    for seg in doc["timeline"]:
        ref = seg.get("ref")
        want = seg_id is None or seg.get("seg_id") == seg_id
        if want and ref and index.level_ok(ref, level):
            rc = index.resolve(Placement(ref=ref, level=level))
            if rc is not None:
                segs = _segments_from_cut(rc)
                if segs:
                    # keep the first slice's id so selections/refs stay stable
                    segs[0]["seg_id"] = seg.get("seg_id") or segs[0]["seg_id"]
                    # heal_adjacent_cuts.plan.md §5: the intra-cut seams this
                    # tighter energy level opens are DELIBERATE dead-air trims --
                    # mark them hard so the compose-time heal never stitches the
                    # tightened take back together (undoing the tighten).
                    for extra in segs[1:]:
                        extra["hard_seam"] = True
                    new_tl.extend(segs)
                    changed = True
                    continue
        new_tl.append(seg)
    if not changed:
        if seg_id is not None:
            raise EditReject(f"'{seg_id}' has no '{level}' variant to tighten to.")
        raise EditReject(f"no main-line cut has a '{level}' variant to tighten to.")
    doc["timeline"] = new_tl
    return doc


def retime(document: dict, index: _MapIndex, *,
           seg_id: Optional[str] = None, pace: str = "natural") -> dict:
    """Set the PLAYBACK PACE of main-line cut(s) -- a DIFFERENT axis from
    ``tighten`` (which chooses how much of the beat to keep). What it does
    depends on the cut's kind, and the effect on the content is explicit:

      * VIDEO cut -> plays at that SPEED (``pace.levels`` cross-clip-normalized
        so the same step reads smoothly against its neighbors; 'natural' ~= 1x,
        'faster' compresses time / speeds motion, 'slower' stretches it). This
        changes how the shot MOVES and how long it runs. NOTE: the render engine
        does not apply speed yet, so the chosen ``speed`` is recorded on the cut
        (and surfaced in read_state) but the exported length is unchanged until
        the retime render pass lands -- the brain should know it's queued, not
        yet baked.
      * SPEECH cut -> pitch and speed are NEVER touched (sped-up speech reads
        amateur). The pacing lever instead shaves removable DEAD-AIR + FILLERS
        inside the cut: 'faster'/'much_faster' tighten the delivery (fewer/shorter
        pauses, 'um's dropped) by re-slicing into a jump-cut keep-list;
        'natural'/'slower' keep every pause. This applies TODAY (it's just tighter
        source spans).

    With ``seg_id`` -> just that cut; without -> every main-line cut. A cut whose
    ref/pace envelope is unknown is left as-is."""
    if pace not in _PACE_STEPS:
        raise EditReject(f"pace must be one of {_PACE_STEPS} (got {pace!r}).")
    idx = _PACE_STEPS.index(pace)
    tl = document.get("timeline") or []
    focus = None
    if seg_id is not None:
        focus = next((s for s in tl if s.get("seg_id") == seg_id), None)
        if focus is None:
            raise EditReject(f"no main-line cut with id '{seg_id}'.")
    focus_ref = focus.get("ref") if focus is not None else None

    doc = _clone(document)
    changed = False
    rebuilt_speech: set = set()          # speech refs already re-derived this pass
    new_tl: List[dict] = []
    for seg in doc["timeline"]:
        ref = seg.get("ref")
        m = index.moments.get(ref or "")
        # A seg is in scope when retiming the whole line (m known) or when it
        # shares the focus cut's ref (so a multi-slice speech cut retimes whole).
        in_scope = m is not None and (seg_id is None or ref == focus_ref)
        if not in_scope:
            new_tl.append(seg)
            continue
        env = m.get("pace") or {}
        is_speech = (m.get("kind") == "speech" or m.get("channel") == "said"
                     or seg.get("axis") == "speech")
        if not is_speech:
            # VIDEO: stamp the recorded playback speed on THIS seg (per-seg). The
            # render honors it in the retime render pass -- program geometry
            # stays 1x until then, so preview == export.
            if seg_id is not None and seg.get("seg_id") != seg_id:
                new_tl.append(seg)
                continue
            levels = env.get("levels") or []
            speed = float(levels[idx]) if idx < len(levels) and levels[idx] else 1.0
            nseg = dict(seg)
            nseg["pace_level"], nseg["speed"] = pace, speed
            new_tl.append(nseg)
            changed = True
            continue

        # SPEECH: never sped -- re-derive the dead-air/filler trim from the cut's
        # FULL grounded span every time (idempotent, exactly like the dial: a
        # lower step widens back, a higher step tightens), so the whole cut is
        # replaced by its jump-cut keep-list at this pace. Later slices of an
        # already-rebuilt cut are dropped (folded into the rebuild).
        if ref in rebuilt_speech:
            changed = True
            continue
        rebuilt_speech.add(ref)
        base_in, base_out = int(m.get("in_ms", seg["in_ms"])), int(m.get("out_ms", seg["out_ms"]))
        frac = _SPEECH_TRIM_FRAC.get(pace, 0.0)
        rem = [(int(a), int(b)) for a, b in (env.get("remove_spans") or []) if int(b) > int(a)]
        chosen = cutrecord_map._chosen_remove_spans(rem, frac) if (frac > 0 and rem) else []
        kept = cutrecord_map._kept_segments(base_in, base_out, chosen)
        for j, (a, b) in enumerate(kept):
            slc = {k: v for k, v in seg.items() if k not in ("speed", "pace_level")}
            slc["in_ms"], slc["out_ms"] = int(a), int(b)
            if frac > 0:
                slc["pace_level"] = pace
            if j > 0:
                slc["seg_id"] = _new_seg_id()
                # heal_adjacent_cuts.plan.md §5: this seam is a DELIBERATELY
                # removed pause/filler -- mark it hard so the compose-time heal
                # never merges the tightened dead air back in (undoing the trim).
                slc["hard_seam"] = True
            new_tl.append(slc)
        changed = True
    if not changed:
        if seg_id is not None:
            raise EditReject(f"'{seg_id}' has no known ref/pace envelope to retime.")
        raise EditReject("no main-line cut has a known ref/pace envelope to retime.")
    doc["timeline"] = new_tl
    return doc
