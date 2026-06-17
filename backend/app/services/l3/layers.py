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
    picture" case (coverage / cutaway / multicam angle). Non-full-frame layouts
    (split-screen, PiP, overlay) are the SAME primitive with a different layout
    -- deferred, but the model already admits them.
  * Audio is a set of ROLE layers (dialogue / music / sfx) that the mixer SUMS,
    applying per-layer gain and side-chain ducking. Audio has no spatial axis.

Operations (verbs) are compiled + validated at author time (in tools.py, using
the L1 cut-cost grids); this module only LAYS OUT already-compiled operations,
so resolution is pure, cheap, and free of snapping/model logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Z bands so layer kinds stack predictably regardless of insertion order.
Z_SPINE_VIDEO = 0
Z_ANGLE = 5            # a synced multicam angle re-pointing the spine picture
Z_COVERAGE = 10        # video laid over the spine (coverage / cutaway / overlay)
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
DEST_FULL = "full"         # whole canvas; sub-rects (split/PiP) are deferred

DEFAULT_ROTATE = 0
DEFAULT_ANCHOR = "center"
DEFAULT_ZOOM = 1.0

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
        }


@dataclass
class ResolvedTimeline:
    duration_ms: int
    video_layers: List[VideoLayer] = field(default_factory=list)
    audio_layers: List[AudioLayer] = field(default_factory=list)
    aspect: str = DEFAULT_ASPECT   # delivery frame shape; render + preview read it

    def video_at(self, ms: int) -> Optional[VideoLayer]:
        """The picture shown at `ms`: the highest-z full-frame opaque layer
        covering it. (Spatial compositing of partial layouts is deferred.)"""
        shown: Optional[VideoLayer] = None
        for v in self.video_layers:
            if v.prog_start_ms <= ms < v.prog_end_ms:
                if shown is None or v.z > shown.z:
                    shown = v
        return shown

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


def covering_segments(spans: List[SpineSpan], from_ms: int, to_ms: int) -> List[SpineSpan]:
    """Spine spans overlapping a program range."""
    return [s for s in spans if s.prog_start_ms < to_ms and s.prog_end_ms > from_ms]


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


def resolve(document: dict, durations: Optional[Dict[str, int]] = None) -> ResolvedTimeline:
    """Compile the spine + operations into the flat resolved layer set.

    Pure layout: operations are assumed already compiled+validated (exact times,
    snapped) by the executor. `durations` (file_id -> ms) is used only to clamp
    split-edit audio extensions to real footage; missing entries skip the clamp.
    """
    durations = durations or {}
    timeline = document.get("timeline") or []
    operations = document.get("operations") or []
    spans, total = spine_spans(timeline)

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
        ))
        audio.append(AudioLayer(
            layer_id=f"a_{seg['seg_id']}",
            role=ROLE_DIALOGUE,
            source_file_id=seg["file_id"],
            src_in_ms=int(seg["in_ms"]), src_out_ms=int(seg["out_ms"]),
            prog_start_ms=s.prog_start_ms, prog_end_ms=s.prog_end_ms,
            kind="spine",
        ))

    # --- J/L split edits reshape the spine audio boundaries ---
    _apply_split_edits(spans, audio, operations, durations)

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
            ))
        elif t == "pick_angle":
            # A synced multicam angle re-points the SPINE picture (a normal cut),
            # so it resolves as a spine-band video layer just above the base
            # spine -- not as coverage. Audio is untouched (stays the spine).
            video.append(VideoLayer(
                layer_id=op["op_id"],
                source_file_id=op["source_file_id"],
                src_in_ms=int(op["src_in_ms"]), src_out_ms=int(op["src_out_ms"]),
                prog_start_ms=int(op["from_ms"]), prog_end_ms=int(op["to_ms"]),
                z=int(op.get("z", Z_ANGLE)),
                layout=op.get("layout", DEFAULT_LAYOUT),
                opacity=1.0, kind="angle", op_id=op["op_id"],
                transform=solve_transform(document, op.get("transform")),
            ))
        elif t == "place_audio":
            audio.append(AudioLayer(
                layer_id=op["op_id"],
                role=op.get("role", ROLE_MUSIC),
                source_file_id=op["source_file_id"],
                src_in_ms=int(op["src_in_ms"]), src_out_ms=int(op["src_out_ms"]),
                prog_start_ms=int(op["from_ms"]), prog_end_ms=int(op["to_ms"]),
                gain_db=float(op.get("gain_db", 0.0)),
                kind=op.get("audio_kind", "bed"), op_id=op["op_id"],
            ))

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

    1. Auto-duck: any bed/sfx layer overlapping live dialogue gets the op's
       requested duck (or a default) so speech stays intelligible.
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
        if a.role == ROLE_DIALOGUE:
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


# --------------------------------------------------------------------------
# Spine-lock validation (operations are legal-by-construction)
# --------------------------------------------------------------------------
#
# The spine declares, per region, which channels are LOCKED (irreplaceable).
# Until regions carry explicit program-time bounds, a single region governs the
# whole timeline; with several, we combine conservatively (a lock anywhere
# locks globally). The safe default -- no spine, or a `sync` region -- keeps
# A/V coupled, so decoupling operations are refused unless positively allowed.

def _regions(document: dict) -> List[dict]:
    spine = document.get("spine") or {}
    return spine.get("regions") or []


def video_is_free(document: dict) -> bool:
    """True only if the declared spine positively frees the video channel
    (dialogue/music spine, or visual with video NOT locked). No spine or any
    `sync`/video-locked region => not free (coupling is the safe default)."""
    regions = _regions(document)
    if not regions:
        return False
    for r in regions:
        if r.get("kind") == "sync":
            return False
        if "video" in (r.get("locked_channels") or []):
            return False
    return True


def av_is_coupled(document: dict) -> bool:
    """True when A and V must stay together (no spine, or a `sync` region) --
    split edits and coverage are then refused."""
    regions = _regions(document)
    if not regions:
        return True
    return any(r.get("kind") == "sync" for r in regions)


def protected_windows_for(document: dict, file_id: str) -> List[Tuple[int, int, str]]:
    """Do-not-cover spans declared on the spine for a given source clip."""
    out: List[Tuple[int, int, str]] = []
    for r in _regions(document):
        for w in r.get("protected_windows") or []:
            if w.get("file_id") == file_id:
                out.append((int(w.get("start_ms", 0)), int(w.get("end_ms", 0)),
                            w.get("reason", "")))
    return out


def coverage_conflicts(
    document: dict, spans: List[SpineSpan], from_ms: int, to_ms: int
) -> Optional[str]:
    """Reason a video-coverage over [from_ms,to_ms] would be illegal, else None.

    Checks: (1) video must be freed by the spine, (2) the covered program range
    must not overlap a protected window of the spine clip(s) underneath it.
    """
    if av_is_coupled(document):
        return "spine keeps A/V coupled (sync / no spine); cannot cover the picture"
    if not video_is_free(document):
        return "spine locks the video channel; the picture is the content and must show"
    for s in covering_segments(spans, from_ms, to_ms):
        seg = s.seg
        ov_start = max(from_ms, s.prog_start_ms)
        ov_end = min(to_ms, s.prog_end_ms)
        src_a = int(seg["in_ms"]) + (ov_start - s.prog_start_ms)
        src_b = int(seg["in_ms"]) + (ov_end - s.prog_start_ms)
        for (ws, we, reason) in protected_windows_for(document, seg["file_id"]):
            if _overlaps(src_a, src_b, ws, we):
                return (f"covers a protected window in {seg['file_id']} "
                        f"[{ws}-{we}ms]{f': {reason}' if reason else ''}")
    return None


def angle_conflicts(
    document: dict, spans: List[SpineSpan], from_ms: int, to_ms: int
) -> Optional[str]:
    """Reason a multicam ANGLE switch over [from_ms,to_ms] would be illegal, else
    None.

    Unlike coverage, an angle switch is legal on ANY spine kind (a synced angle
    is the same moment, so A/V stay coherent) -- the sync-group membership
    requirement is enforced by the executor. The only document-level block is a
    protected window of the spine clip underneath (switching away would hide
    the very thing the window protects)."""
    for s in covering_segments(spans, from_ms, to_ms):
        seg = s.seg
        ov_start = max(from_ms, s.prog_start_ms)
        ov_end = min(to_ms, s.prog_end_ms)
        src_a = int(seg["in_ms"]) + (ov_start - s.prog_start_ms)
        src_b = int(seg["in_ms"]) + (ov_end - s.prog_start_ms)
        for (ws, we, reason) in protected_windows_for(document, seg["file_id"]):
            if _overlaps(src_a, src_b, ws, we):
                return (f"covers a protected window in {seg['file_id']} "
                        f"[{ws}-{we}ms]{f': {reason}' if reason else ''}")
    return None
