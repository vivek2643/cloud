"""
L3 layered timeline: the general A/V substrate the verbs compile onto.

The authoritative edit is still the SPINE (`document["timeline"]`, an ordered
list of coupled segments that owns the program clock) PLUS a list of typed
OPERATIONS (`document["operations"]`). This module resolves those two into a
flat, deterministic set of LAYERS that say exactly what is shown and heard at
every program instant -- the form diagnostics, export, and (later) preview read.

Design intent: GENERAL, not B-roll-specific.
  * Video is a z-ordered stack of LAYERS. Each layer carries a `layout`
    (default "full_frame"), `z`, and `opacity`. The compositor paints
    bottom->top. A full-frame, opaque, top layer is the degenerate "replace the
    picture" case (a full-frame V2 cutaway / alternate angle). Non-full-frame
    layouts (split-screen, PiP) are the SAME primitive with a different layout
    -- the model already admits them.
  * Audio is a set of ROLE layers (dialogue / music / sfx) that the mixer SUMS,
    applying per-layer gain and side-chain ducking. Audio has no spatial axis.

Operations (verbs) are compiled + validated at author time (in tools.py, using
the L1 cut-cost grids); this module only LAYS OUT already-compiled operations,
so resolution is pure, cheap, and free of snapping/model logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.services.l3.grade.match import solve_match_deltas
from app.services.l3.grade.resolver import resolve_clip_grade

# Z bands so layer kinds stack predictably regardless of insertion order.
Z_SPINE_VIDEO = 0
Z_COVERAGE = 10        # a V2 cutaway painted above the V1 spine (full-frame or a cell)
DEFAULT_LAYOUT = "full_frame"

# Audio roles (the mixer knows how to duck dialogue under music, etc.).
ROLE_DIALOGUE = "dialogue"
ROLE_MUSIC = "music"
ROLE_SFX = "sfx"
AUDIO_ROLES = (ROLE_DIALOGUE, ROLE_MUSIC, ROLE_SFX)

# Output FRAME shape (delivery format). The spine/operations are shape-agnostic;
# aspect only changes the canvas the resolved layers are painted onto, so the
# same edit can deliver landscape, vertical (reel/short), or square.
ASPECT_LANDSCAPE = "landscape"   # 16:9
ASPECT_PORTRAIT = "portrait"     # 9:16 (reels / shorts / tiktok)
ASPECT_SQUARE = "square"         # 1:1
ASPECTS = (ASPECT_LANDSCAPE, ASPECT_PORTRAIT, ASPECT_SQUARE)
DEFAULT_ASPECT = ASPECT_LANDSCAPE


def aspect_of(document: dict) -> str:
    """The delivery aspect declared on the document (default landscape)."""
    fmt = document.get("format") or {}
    a = (fmt.get("aspect") if isinstance(fmt, dict) else None) or DEFAULT_ASPECT
    return a if a in ASPECTS else DEFAULT_ASPECT


# --------------------------------------------------------------------------
# Per-layer geometric transform (framing) -- Phase 1
# --------------------------------------------------------------------------
#
# Every video layer carries a TRANSFORM that says how its source pixels are
# framed onto the canvas, in one fixed order of operations (identical in ffmpeg
# and the CSS preview, so what you preview is what you render):
#
#     rotate (orthogonal) -> fit into the canvas (cover/contain) -> zoom-crop
#     -> place into `dest` (always the full canvas for now) -> composite by z.
#
# The numbers are produced by a deterministic SOLVER (`solve_transform`), never
# by the model: the model/document declares INTENT (delivery aspect, an explicit
# fit, or -- later -- a focus/zoom), and the solver turns that into the rectangle.
# Phase 1 uses only the delivery aspect: vertical/square deliveries FILL the
# frame (cover, centered) instead of letterboxing; landscape stays contain so
# existing edits are byte-identical. The focus-anchored crop and animated zoom
# arrive in Phases 2-3 and only enrich this same struct.

ROTATIONS = (0, 90, 180, 270)
FIT_COVER = "cover"        # fill the canvas, crop the overflow
FIT_CONTAIN = "contain"    # fit inside the canvas, letterbox the remainder
FITS = (FIT_COVER, FIT_CONTAIN)
ANCHORS = ("center", "left", "right", "top", "bottom")
DEST_FULL = "full"         # whole canvas; a sub-rect dict {x,y,w,h} = split/PiP cell

DEFAULT_ROTATE = 0
DEFAULT_ANCHOR = "center"
DEFAULT_ZOOM = 1.0

# Spatial layout TEMPLATES (Phase 5): a template + cell name -> a normalized dest
# rect (x, y, w, h in 0..1 canvas coords). Split-screen and PiP are the SAME
# layered primitive as coverage -- a video layer painted into a sub-rect of the
# canvas instead of the full frame. A time-scoped LAYOUT REGION on the document
# assigns cells to layers (spine / a place_video op) over a program window, and
# the resolver slices + stamps the dest rect onto the layers in that window.
LAYOUT_TEMPLATES: Dict[str, Dict[str, Tuple[float, float, float, float]]] = {
    "split_h": {"left": (0.0, 0.0, 0.5, 1.0), "right": (0.5, 0.0, 0.5, 1.0)},
    "split_v": {"top": (0.0, 0.0, 1.0, 0.5), "bottom": (0.0, 0.5, 1.0, 0.5)},
    "pip": {"base": (0.0, 0.0, 1.0, 1.0), "inset": (0.66, 0.66, 0.32, 0.32)},
}


def _rect(t: Tuple[float, float, float, float]) -> dict:
    x, y, w, h = t
    return {"x": round(x, 4), "y": round(y, 4), "w": round(w, 4), "h": round(h, 4)}


def solve_layout(template: str) -> Dict[str, dict]:
    """A layout template -> {cell_name: dest_rect}. Empty for an unknown template
    (the region is then a no-op -- never an error that could break a render)."""
    cells = LAYOUT_TEMPLATES.get(template)
    return {name: _rect(t) for name, t in cells.items()} if cells else {}


def is_rect(dest) -> bool:
    """True when a transform `dest` is a real sub-rect (split/PiP cell) rather
    than the full-canvas default."""
    return isinstance(dest, dict) and all(k in dest for k in ("x", "y", "w", "h"))

# Phase 3 -- animated motion. A `motion` path makes scale + focus VARY over the
# layer's program span (push-in, pull-out, follow). It is a from->to move (two
# endpoints) eased over `dur_ms`; both renderers evaluate the SAME closed form
# `sample_motion`, so the preview and the render animate identically. A static
# zoom/focus (Phase 1-2) is the degenerate from==to case and needs no motion.
EASE_LINEAR = "linear"
EASE_SMOOTH = "smooth"     # smoothstep 3p^2-2p^3 ("glide")
EASES = (EASE_LINEAR, EASE_SMOOTH)


def _clamp01f(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _norm_motion_point(p: dict) -> Optional[dict]:
    try:
        return {
            "scale": max(1.0, float(p["scale"])),
            "cx": _clamp01f(float(p["cx"])),
            "cy": _clamp01f(float(p["cy"])),
        }
    except (KeyError, TypeError, ValueError):
        return None


def normalize_motion(motion: Optional[dict]) -> Optional[dict]:
    """Validate + clamp a motion path, or None if malformed/degenerate.
    A motion that doesn't actually move (from == to) is dropped -- it collapses
    to the static zoom/focus the rest of the transform already carries."""
    if not isinstance(motion, dict):
        return None
    fr = _norm_motion_point(motion.get("from") or {})
    to = _norm_motion_point(motion.get("to") or {})
    if fr is None or to is None:
        return None
    ease = motion.get("ease") if motion.get("ease") in EASES else EASE_LINEAR
    try:
        dur_ms = max(1, int(motion.get("dur_ms") or 0))
    except (TypeError, ValueError):
        return None
    moves = (abs(fr["scale"] - to["scale"]) > 1e-4
             or abs(fr["cx"] - to["cx"]) > 1e-4
             or abs(fr["cy"] - to["cy"]) > 1e-4)
    if not moves:
        return None
    return {"from": fr, "to": to, "ease": ease, "dur_ms": dur_ms}


def sample_motion(motion: dict, rel_ms: float) -> dict:
    """The {scale, cx, cy} of a motion path at `rel_ms` into the layer (the
    single source of truth both renderers evaluate). Linear or smoothstep ease."""
    dur = max(1, int(motion.get("dur_ms") or 1))
    p = _clamp01f(rel_ms / dur)
    if motion.get("ease") == EASE_SMOOTH:
        p = 3.0 * p * p - 2.0 * p * p * p
    fr, to = motion["from"], motion["to"]
    return {
        "scale": fr["scale"] + (to["scale"] - fr["scale"]) * p,
        "cx": fr["cx"] + (to["cx"] - fr["cx"]) * p,
        "cy": fr["cy"] + (to["cy"] - fr["cy"]) * p,
    }


def default_fit(aspect: str) -> str:
    """The automatic fit for a delivery aspect: vertical/square FILL the frame,
    landscape keeps the safe letterbox (no regression on existing edits)."""
    return FIT_COVER if aspect in (ASPECT_PORTRAIT, ASPECT_SQUARE) else FIT_CONTAIN


def identity_transform(aspect: str = DEFAULT_ASPECT) -> dict:
    return {
        "rotate": DEFAULT_ROTATE,
        "fit": default_fit(aspect),
        "anchor": DEFAULT_ANCHOR,
        "zoom": DEFAULT_ZOOM,
        "dest": DEST_FULL,
    }


def _clamp_rotate(v) -> int:
    try:
        r = int(v) % 360
    except (TypeError, ValueError):
        return DEFAULT_ROTATE
    return r if r in ROTATIONS else DEFAULT_ROTATE


def solve_transform(document: dict, override: Optional[dict] = None) -> dict:
    """Deterministic framing solver (Phase 1).

    Resolves the layer's transform from the delivery aspect plus an optional
    explicit `override` (a partial transform an operation/segment may carry, e.g.
    a creative rotate or a static zoom). No perception or model logic -- a pure
    function so the frontend resolver can mirror it exactly.
    """
    aspect = aspect_of(document)
    fmt = document.get("format") or {}
    fit = fmt.get("fit") if isinstance(fmt, dict) else None
    if fit not in FITS:
        fit = default_fit(aspect)
    t = {
        "rotate": DEFAULT_ROTATE,
        "fit": fit,
        "anchor": DEFAULT_ANCHOR,
        "zoom": DEFAULT_ZOOM,
        "dest": DEST_FULL,
    }
    if override:
        if "rotate" in override:
            t["rotate"] = _clamp_rotate(override.get("rotate"))
        if override.get("fit") in FITS:
            t["fit"] = override["fit"]
        if override.get("anchor") in ANCHORS:
            t["anchor"] = override["anchor"]
        if override.get("zoom") is not None:
            try:
                t["zoom"] = max(1.0, float(override["zoom"]))
            except (TypeError, ValueError):
                pass
        focus = override.get("focus")
        if isinstance(focus, dict):
            try:
                cx = min(1.0, max(0.0, float(focus["cx"])))
                cy = min(1.0, max(0.0, float(focus["cy"])))
                t["focus"] = {"cx": round(cx, 4), "cy": round(cy, 4)}
            except (KeyError, TypeError, ValueError):
                pass
        motion = normalize_motion(override.get("motion"))
        if motion is not None:
            t["motion"] = motion
    return t


# --------------------------------------------------------------------------
# Resolved layer records (the output of resolution)
# --------------------------------------------------------------------------

@dataclass
class VideoLayer:
    layer_id: str
    source_file_id: str
    src_in_ms: int
    src_out_ms: int
    prog_start_ms: int
    prog_end_ms: int
    z: int = Z_SPINE_VIDEO
    layout: str = DEFAULT_LAYOUT
    opacity: float = 1.0
    kind: str = "spine"          # spine | coverage
    op_id: Optional[str] = None  # operation that produced it (None = spine)
    # Geometric framing (rotate/fit/anchor/zoom/dest); see solve_transform.
    transform: dict = field(default_factory=dict)
    # Resolved color grade ({cdl, creative_lut_ref, working_space, grade_hash});
    # see grade.resolver.resolve_clip_grade. Applied identically in preview
    # (WebGL LUT sample) and export (ffmpeg lut3d) -- see color_grading.plan.md SS4.
    grade: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "layer_id": self.layer_id,
            "source_file_id": self.source_file_id,
            "src_in_ms": self.src_in_ms,
            "src_out_ms": self.src_out_ms,
            "prog_start_ms": self.prog_start_ms,
            "prog_end_ms": self.prog_end_ms,
            "z": self.z,
            "layout": self.layout,
            "opacity": self.opacity,
            "kind": self.kind,
            "op_id": self.op_id,
            "transform": self.transform or {},
            "grade": self.grade or {},
        }


@dataclass
class AudioLayer:
    layer_id: str
    role: str
    source_file_id: str
    src_in_ms: int
    src_out_ms: int
    prog_start_ms: int
    prog_end_ms: int
    gain_db: float = 0.0
    # Side-chain duck applied to THIS layer when a higher-priority role is
    # active over it (dialogue ducks music, etc.). Stored resolved (<=0 dB).
    duck_db: float = 0.0
    kind: str = "spine"          # spine | bed | replace | sfx
    op_id: Optional[str] = None
    # Fade envelope on this layer's own edges (audio_brain.plan.md `fade_audio`
    # + `crossfade`) -- ms from prog_start_ms / into prog_end_ms. 0 = hard
    # start/stop (the default: beds never fade unless the brain sets one).
    fade_in_ms: int = 0
    fade_out_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "layer_id": self.layer_id,
            "role": self.role,
            "source_file_id": self.source_file_id,
            "src_in_ms": self.src_in_ms,
            "src_out_ms": self.src_out_ms,
            "prog_start_ms": self.prog_start_ms,
            "prog_end_ms": self.prog_end_ms,
            "gain_db": self.gain_db,
            "duck_db": self.duck_db,
            "kind": self.kind,
            "op_id": self.op_id,
            "fade_in_ms": self.fade_in_ms,
            "fade_out_ms": self.fade_out_ms,
        }


@dataclass
class ResolvedTimeline:
    duration_ms: int
    video_layers: List[VideoLayer] = field(default_factory=list)
    audio_layers: List[AudioLayer] = field(default_factory=list)
    aspect: str = DEFAULT_ASPECT   # delivery frame shape; render + preview read it

    def video_at(self, ms: int) -> Optional[VideoLayer]:
        """The single top picture shown at `ms` (highest-z covering layer). For
        split/PiP use ``video_stack_at`` -- multiple cells show at once."""
        shown: Optional[VideoLayer] = None
        for v in self.video_layers:
            if v.prog_start_ms <= ms < v.prog_end_ms:
                if shown is None or v.z > shown.z:
                    shown = v
        return shown

    def video_stack_at(self, ms: int) -> List[VideoLayer]:
        """Every video layer covering `ms`, bottom->top by z -- the compositing
        stack. With split/PiP, several cells (each with its own dest rect) are
        live at once; without, this is just the spine (+ any coverage) layer."""
        return sorted(
            [v for v in self.video_layers if v.prog_start_ms <= ms < v.prog_end_ms],
            key=lambda v: v.z,
        )

    def audio_at(self, ms: int) -> List[AudioLayer]:
        """All audio layers sounding at `ms` (they sum)."""
        return [a for a in self.audio_layers if a.prog_start_ms <= ms < a.prog_end_ms]

    def to_dict(self) -> dict:
        return {
            "duration_ms": self.duration_ms,
            "video_layers": [v.to_dict() for v in self.video_layers],
            "audio_layers": [a.to_dict() for a in self.audio_layers],
            "aspect": self.aspect,
        }


# --------------------------------------------------------------------------
# Program-time geometry of the spine
# --------------------------------------------------------------------------

@dataclass
class SpineSpan:
    seg: dict
    prog_start_ms: int
    prog_end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.prog_end_ms - self.prog_start_ms


def spine_spans(timeline: List[dict]) -> Tuple[List[SpineSpan], int]:
    """Lay the spine segments end-to-end on the program clock."""
    spans: List[SpineSpan] = []
    t = 0
    for seg in timeline:
        dur = max(0, int(seg["out_ms"]) - int(seg["in_ms"]))
        spans.append(SpineSpan(seg=seg, prog_start_ms=t, prog_end_ms=t + dur))
        t += dur
    return spans, t


def prog_to_source(spans: List[SpineSpan], prog_ms: int) -> Optional[Tuple[dict, int, SpineSpan]]:
    """Map a program-time instant to (spine segment, source ms, span)."""
    if not spans:
        return None
    prog_ms = max(0, prog_ms)
    for s in spans:
        if s.prog_start_ms <= prog_ms <= s.prog_end_ms:
            seg = s.seg
            src = int(seg["in_ms"]) + (prog_ms - s.prog_start_ms)
            return seg, src, s
    last = spans[-1]
    return last.seg, int(last.seg["out_ms"]), last


# --------------------------------------------------------------------------
# Resolution: spine + operations -> layers
# --------------------------------------------------------------------------

def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else (hi if v > hi else v)


def _apply_split_edits(
    spans: List[SpineSpan],
    audio_layers: List[AudioLayer],
    operations: List[dict],
    durations: Dict[str, int],
) -> None:
    """J/L cuts: offset the AUDIO boundary at a seam from the video boundary.

    offset > 0 (L-cut): previous clip's audio lingers over the next clip's
    picture -- prev audio extends, next audio starts later.
    offset < 0 (J-cut): next clip's audio leads under the previous picture.
    `audio_layers` are the per-span spine (dialogue) layers, index-aligned with
    `spans`; we shift the shared boundary between layer i-1 and i.
    """
    by_seg = {s.seg["seg_id"]: i for i, s in enumerate(spans)}
    for op in operations:
        if op.get("type") != "split_edit":
            continue
        i = by_seg.get(op.get("seam_seg_id"))
        if i is None or i == 0 or i >= len(audio_layers):
            continue
        offset = int(op.get("audio_offset_ms", 0))
        if offset == 0:
            continue
        prev_a, cur_a = audio_layers[i - 1], audio_layers[i]
        boundary = cur_a.prog_start_ms + offset
        # Previous audio extends/retracts to the new boundary.
        prev_dur_room = durations.get(prev_a.source_file_id, prev_a.src_out_ms)
        prev_a.prog_end_ms = max(prev_a.prog_start_ms, boundary)
        prev_a.src_out_ms = _clamp(prev_a.src_out_ms + offset, prev_a.src_in_ms, prev_dur_room)
        # Next audio starts at the new boundary.
        cur_a.prog_start_ms = max(0, boundary)
        cur_a.src_in_ms = _clamp(cur_a.src_in_ms + offset, 0, cur_a.src_out_ms)


def _apply_crossfades(
    spans: List[SpineSpan],
    audio_layers: List[AudioLayer],
    operations: List[dict],
    durations: Dict[str, int],
) -> None:
    """`crossfade`: overlap the two spine (dialogue) layers either side of a
    seam by `ms` total (split evenly), each extending toward the other --
    previous audio plays on into the next picture, next audio starts under the
    previous picture -- then fades the overlapping edges (`fade_out_ms` on the
    outgoing layer, `fade_in_ms` on the incoming one) so they cross-dissolve
    instead of doubling up at full volume. Audibly equivalent to ffmpeg's
    `acrossfade` for a linear/triangular curve, built on the SAME per-layer
    delay-then-`amix` mix compositor already uses (no 2-input topology
    needed). One crossfade per seam (re-issuing replaces, `ms<=0` clears)."""
    by_seg = {s.seg["seg_id"]: i for i, s in enumerate(spans)}
    for op in operations:
        if op.get("type") != "crossfade":
            continue
        i = by_seg.get(op.get("seam_seg_id"))
        if i is None or i == 0 or i >= len(audio_layers):
            continue
        ms = int(op.get("ms", 0))
        if ms <= 0:
            continue
        prev_a, cur_a = audio_layers[i - 1], audio_layers[i]
        fwd = ms // 2
        back = ms - fwd
        prev_dur_room = durations.get(prev_a.source_file_id, prev_a.src_out_ms)
        prev_a.prog_end_ms += fwd
        prev_a.src_out_ms = _clamp(prev_a.src_out_ms + fwd, prev_a.src_in_ms, prev_dur_room)
        cur_a.prog_start_ms = max(0, cur_a.prog_start_ms - back)
        cur_a.src_in_ms = _clamp(cur_a.src_in_ms - back, 0, cur_a.src_out_ms)
        prev_a.fade_out_ms = max(prev_a.fade_out_ms, fwd + back)
        cur_a.fade_in_ms = max(cur_a.fade_in_ms, fwd + back)


def _slice_video(v: VideoLayer, ps: int, pe: int, dest: Optional[dict]) -> VideoLayer:
    """A sub-span [ps, pe) of a video layer, source-mapped, with an optional dest
    rect stamped on (and fit forced to cover so a split/PiP cell fills, not
    letterboxes). Used to carve the spine where a layout region overlaps it."""
    src_in = v.src_in_ms + (ps - v.prog_start_ms)
    src_out = v.src_in_ms + (pe - v.prog_start_ms)
    tf = dict(v.transform or {})
    if dest is not None:
        tf["dest"] = dest
        tf["fit"] = FIT_COVER
    suffix = "" if (ps == v.prog_start_ms and pe == v.prog_end_ms) else f"__{ps}"
    return VideoLayer(
        layer_id=f"{v.layer_id}{suffix}", source_file_id=v.source_file_id,
        src_in_ms=src_in, src_out_ms=src_out, prog_start_ms=ps, prog_end_ms=pe,
        z=v.z, layout=v.layout, opacity=v.opacity, kind=v.kind, op_id=v.op_id,
        transform=tf,
    )


def _dest_spine_window(video: List[VideoLayer], f: int, t: int, dest: dict) -> List[VideoLayer]:
    """Stamp `dest` onto the SPINE picture across [f, t), slicing spine layers that
    straddle the window (the parts outside keep the full frame). Coverage/op layers
    are untouched here (they're addressed by op_id)."""
    out: List[VideoLayer] = []
    for v in video:
        if v.kind != "spine" or v.prog_end_ms <= f or v.prog_start_ms >= t:
            out.append(v)
            continue
        os_, oe = max(v.prog_start_ms, f), min(v.prog_end_ms, t)
        if v.prog_start_ms < os_:
            out.append(_slice_video(v, v.prog_start_ms, os_, None))
        out.append(_slice_video(v, os_, oe, dest))
        if oe < v.prog_end_ms:
            out.append(_slice_video(v, oe, v.prog_end_ms, None))
    return out


def _apply_layout_regions(video: List[VideoLayer], regions: List[dict]) -> List[VideoLayer]:
    """Turn each layout region into dest rects on the layers it names.

    A region is {from_ms, to_ms, template, cells:{cell_name:{layer:"spine"|op_id}}}.
    For each cell we look up the template's rect and stamp it: the "spine" cell
    slices the picture across the window; an op cell stamps its place_video layer.
    Unknown templates/cells are skipped (never fatal)."""
    for r in regions:
        template = str(r.get("template") or "")
        rects = solve_layout(template)
        if not rects:
            continue
        try:
            f, t = int(r.get("from_ms")), int(r.get("to_ms"))
        except (TypeError, ValueError):
            continue
        if t <= f:
            continue
        for cell_name, sel in (r.get("cells") or {}).items():
            rect = rects.get(cell_name)
            if rect is None:
                continue
            layer_sel = (sel or {}).get("layer") if isinstance(sel, dict) else None
            if layer_sel == "spine":
                video = _dest_spine_window(video, f, t, rect)
            elif layer_sel:
                for v in video:
                    if v.op_id == layer_sel or v.layer_id == layer_sel:
                        v.transform = {**(v.transform or {}), "dest": rect, "fit": FIT_COVER}
    return video


def resolve(
    document: dict,
    durations: Optional[Dict[str, int]] = None,
    color_stats: Optional[Dict[str, dict]] = None,
    audio_routes: Optional[Dict[str, dict]] = None,
) -> ResolvedTimeline:
    """Compile the spine + operations into the flat resolved layer set.

    Pure layout: operations are assumed already compiled+validated (exact times,
    snapped) by the executor. `durations` (file_id -> ms) is used only to clamp
    split-edit audio extensions to real footage; missing entries skip the clamp.
    `color_stats` (file_id -> L1 color_stats row, see grade.measure) feeds the
    correct layer (SS5); a missing/absent entry just means that clip's grade
    stays identity (never-worse: no measurement, no basis to change anything).
    `audio_routes` (audio_sync.plan.md SS8, see `sync.audio_route.resolve_audio_routes`):
    seg_id -> {source_file_id, src_in_ms, src_out_ms} for a spine segment
    whose cut belongs to a synced group -- the picture stays the shown
    angle, but the coupled AudioLayer is re-routed to the group's
    authoritative source instead. Missing entry -> today's coupled audio,
    unchanged (the no-multicam-regression guarantee).
    """
    durations = durations or {}
    color_stats = color_stats or {}
    audio_routes = audio_routes or {}
    timeline = document.get("timeline") or []
    operations = document.get("operations") or []
    spans, total = spine_spans(timeline)
    sequence_look = document.get("look")
    # SS6 match layer: grade-groups are a whole-document clustering decision,
    # so resolve every file's delta ONCE here rather than per-clip.
    match_deltas = solve_match_deltas(color_stats) if color_stats else {}

    video: List[VideoLayer] = []
    audio: List[AudioLayer] = []

    # --- base spine layers: coupled video + dialogue, one per span ---
    for idx, s in enumerate(spans):
        seg = s.seg
        video.append(VideoLayer(
            layer_id=f"v_{seg['seg_id']}",
            source_file_id=seg["file_id"],
            src_in_ms=int(seg["in_ms"]), src_out_ms=int(seg["out_ms"]),
            prog_start_ms=s.prog_start_ms, prog_end_ms=s.prog_end_ms,
            z=Z_SPINE_VIDEO, kind="spine",
            transform=solve_transform(document, seg.get("transform")),
            grade=resolve_clip_grade(
                seg, color_stats=color_stats.get(seg["file_id"]), sequence_look=sequence_look,
                match_delta=match_deltas.get(seg["file_id"]),
            ),
        ))
        # `replace_audio`'s override (audio_brain.plan.md) wins over the
        # auto-computed outlook route -- the escape hatch that makes the one
        # structural default (authoritative routing) fully overridable.
        route = seg.get("audio_override") or audio_routes.get(seg["seg_id"])
        audio.append(AudioLayer(
            layer_id=f"a_{seg['seg_id']}",
            role=ROLE_DIALOGUE,
            source_file_id=route["source_file_id"] if route else seg["file_id"],
            src_in_ms=int(route["src_in_ms"]) if route else int(seg["in_ms"]),
            src_out_ms=int(route["src_out_ms"]) if route else int(seg["out_ms"]),
            prog_start_ms=s.prog_start_ms, prog_end_ms=s.prog_end_ms,
            kind="spine",
            # A muted segment (stray speech under a video cut) keeps its picture
            # but drops its source audio -- rendered as volume=0 downstream.
            # Otherwise `set_gain` may have set an explicit level (0 = untouched).
            gain_db=-120.0 if seg.get("mute") else float(seg.get("gain_db", 0.0) or 0.0),
            fade_in_ms=int(seg.get("fade_in_ms", 0) or 0),
            fade_out_ms=int(seg.get("fade_out_ms", 0) or 0),
        ))

    # --- J/L split edits reshape the spine audio boundaries ---
    _apply_split_edits(spans, audio, operations, durations)
    # --- crossfades overlap two adjacent spine layers at a seam ---
    _apply_crossfades(spans, audio, operations, durations)

    # --- operations that add/override layers ---
    for op in operations:
        t = op.get("type")
        if t == "place_video":
            video.append(VideoLayer(
                layer_id=op["op_id"],
                source_file_id=op["source_file_id"],
                src_in_ms=int(op["src_in_ms"]), src_out_ms=int(op["src_out_ms"]),
                prog_start_ms=int(op["from_ms"]), prog_end_ms=int(op["to_ms"]),
                z=int(op.get("z", Z_COVERAGE)),
                layout=op.get("layout", DEFAULT_LAYOUT),
                opacity=float(op.get("opacity", 1.0)),
                kind="coverage", op_id=op["op_id"],
                transform=solve_transform(document, op.get("transform")),
                grade=resolve_clip_grade(
                    op, color_stats=color_stats.get(op["source_file_id"]), sequence_look=sequence_look,
                    match_delta=match_deltas.get(op["source_file_id"]),
                ),
            ))
        elif t == "place_audio":
            # "voiceover" is a plain ROLE_DIALOGUE bed at render time (a fact
            # the brain can still see via the op's own literal role string) --
            # audio_brain.plan.md: no auto-duck PRIORITY, no privilege. It stays
            # duck-able like any other bed because `_apply_levels` below only
            # exempts spine (kind=="spine") layers from ducking, not every
            # dialogue-role one.
            role = ROLE_DIALOGUE if op.get("role") == "voiceover" else op.get("role", ROLE_MUSIC)
            audio.append(AudioLayer(
                layer_id=op["op_id"],
                role=role,
                source_file_id=op["source_file_id"],
                src_in_ms=int(op["src_in_ms"]), src_out_ms=int(op["src_out_ms"]),
                prog_start_ms=int(op["from_ms"]), prog_end_ms=int(op["to_ms"]),
                gain_db=float(op.get("gain_db", 0.0)),
                kind=op.get("audio_kind", "bed"), op_id=op["op_id"],
                fade_in_ms=int(op.get("fade_in_ms", 0) or 0),
                fade_out_ms=int(op.get("fade_out_ms", 0) or 0),
            ))

    # --- spatial layout regions (split-screen / PiP): stamp dest sub-rects ---
    regions = document.get("layout_regions") or []
    if regions:
        video = _apply_layout_regions(video, regions)

    # --- ducking + level automation: dialogue presence ducks beds ---
    _apply_levels(audio, operations)

    return ResolvedTimeline(
        duration_ms=total, video_layers=video, audio_layers=audio,
        aspect=aspect_of(document),
    )


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and a_end > b_start


def _apply_levels(audio_layers: List[AudioLayer], operations: List[dict]) -> None:
    """Resolve duck + explicit level automation.

    1. Duck (opt-in, audio_brain.plan.md): a bed/sfx layer whose `place_audio`
       op set a negative `duck_db`, and that overlaps live spine dialogue,
       gets that duck applied. `duck_db=0` (the default) never ducks -- there
       is no auto-duck. The live spine track itself is never ducked (it's
       what things duck FOR), but a `voiceover`-role bed IS duck-able like any
       other bed -- it's a plain dialogue-role layer, not privileged.
    2. `level` ops apply an explicit gain (or mute) to a role over a range.
    """
    dialogue_spans = [
        (a.prog_start_ms, a.prog_end_ms)
        for a in audio_layers if a.role == ROLE_DIALOGUE and a.kind == "spine"
    ]
    # Map place_audio op_id -> requested duck so we can apply it on its layer.
    duck_by_op = {
        op["op_id"]: float(op.get("duck_db", 0.0))
        for op in operations if op.get("type") == "place_audio" and op.get("op_id")
    }
    for a in audio_layers:
        if a.kind == "spine":
            continue
        duck = duck_by_op.get(a.op_id, 0.0)
        if duck < 0 and any(_overlaps(a.prog_start_ms, a.prog_end_ms, ds, de)
                            for ds, de in dialogue_spans):
            a.duck_db = duck

    for op in operations:
        if op.get("type") != "level":
            continue
        role = op.get("role")
        fr, to = int(op.get("from_ms", 0)), int(op.get("to_ms", 0))
        for a in audio_layers:
            if role and a.role != role:
                continue
            if not _overlaps(a.prog_start_ms, a.prog_end_ms, fr, to):
                continue
            if op.get("mute"):
                a.gain_db = -120.0
            elif op.get("gain_db") is not None:
                a.gain_db = float(op["gain_db"])


