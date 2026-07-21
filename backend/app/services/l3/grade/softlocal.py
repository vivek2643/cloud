"""
Soft-local layer (color_grading.plan.md SS9, Fork B): feathered, subject-
anchored spatial adjustments. Deliberately NOT baked into the CDL/LUT -- a
3D color LUT is a pointwise value->value map with no notion of pixel
position, so the plan doc's "bakes into the per-clip LUT" is a shorthand,
not literal (see `lut_bake.py`'s module docstring for the full reasoning).
This module instead produces a small, JSON-safe SPATIAL descriptor, applied
as an ADDITIONAL deterministic pass alongside (not inside) the color LUT.

Scope: only the attention vignette is implemented -- a soft, subject-
anchored radial darkening toward the frame edges, the most universal and
well-defined of the three effects SS9 lists. Sky-gradient (horizon_y-aware)
and directional side-lift are the same MECHANISM (a feathered spatial
multiplier) but need signals (horizon detection) this pass doesn't produce;
left as a documented follow-up, not silently dropped.

Honest parity note (unlike the CDL/LUT engine, which is byte-identical both
sides by construction): the export side applies this via ffmpeg's own
`vignette` filter (a well-tested standard primitive) while preview applies
an independently-written WebGL radial falloff. Both anchor on the same
subject point and scale with the same strength, so they read as the same
soft effect, but the exact falloff CURVE is not guaranteed pixel-identical
the way the CDL engine is. That's an intentional, bounded trade for a
"soft, feathered, never a hard mask" cosmetic effect, not a corner cut on
the parity contract the plan calls make-or-break for actual color values --
see SS16's own framing of soft-local as approximate by nature.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

DEFAULT_STRENGTH = 0.25  # gentle by default -- "all feathered, no hard masks"
MAX_ANGLE_RAD = 1.0472   # PI/3 -- ffmpeg vignette's angle at strength=1.0

# halation_grain.plan.md: film-texture finishing (SS9 follow-up). Fixed
# sub-params kept as CONSTANTS (not LookSpec fields) to keep the look
# vocabulary small -- a look only ever dials `halation`/`grain` strength.
HALATION_THRESHOLD = 0.75    # 0..1 luma; only pixels brighter than this glow
HALATION_SIGMA = 8.0         # gblur sigma at HALATION_SIGMA_REF_H, scaled by frame height
HALATION_SIGMA_REF_H = 1080  # reference height the sigma constant above is tuned for
HALATION_TINT = (1.0, 0.35, 0.1)   # red-orange per-channel scale (colorchannelmixer rr:gg:bb)
GRAIN_MAX_ALLS = 40           # ffmpeg noise filter's `alls` at grain strength=1.0


def solve_vignette(
    subject_box: Optional[Tuple[float, float, float, float]] = None,
    *,
    strength: float = DEFAULT_STRENGTH,
) -> Dict[str, Any]:
    """A center (cx, cy, normalized 0..1) + strength descriptor for a soft
    radial darkening, anchored on the subject when known (else frame
    center). `subject_box` is the normalized (x, y, w, h) box from
    `cut_records.framing.subject_box` (SS1's re-use note: reframing/subject
    detection already exists, this only reuses it for metering)."""
    if subject_box:
        x, y, w, h = subject_box
        cx = max(0.0, min(1.0, x + w / 2.0))
        cy = max(0.0, min(1.0, y + h / 2.0))
    else:
        cx, cy = 0.5, 0.5
    return {"cx": cx, "cy": cy, "strength": max(0.0, min(1.0, strength))}


def vignette_ffmpeg_filter(vignette: Optional[Dict[str, Any]]) -> Optional[str]:
    """`vignette` descriptor -> an ffmpeg `vignette` filter clause, or None
    for a no-op (absent/zero strength). x0/y0 are pixel EXPRESSIONS (ffmpeg
    evaluates `w`/`h` at filter time to the frame size)."""
    if not vignette:
        return None
    strength = float(vignette.get("strength") or 0.0)
    if strength <= 0.001:
        return None
    cx, cy = float(vignette.get("cx", 0.5)), float(vignette.get("cy", 0.5))
    angle = strength * MAX_ANGLE_RAD
    return f"vignette=angle={angle:.4f}:x0=w*{cx:.4f}:y0=h*{cy:.4f}"


def grain_ffmpeg_filter(grain: Optional[Dict[str, Any]]) -> Optional[str]:
    """`grain` descriptor -> an ffmpeg `noise` filter clause, or None for a
    no-op (absent/zero strength). Temporal + uniform (`allf=t+u`) so it reads
    as film-emulsion texture, not a static overlay -- a NEW random field every
    frame, uniformly distributed rather than Gaussian (cheap, visually close
    enough for the "subtle texture" bar this pass sets, per the plan's own
    "simplest, temporal" framing)."""
    if not grain:
        return None
    strength = float(grain.get("strength") or 0.0)
    if strength <= 0.001:
        return None
    alls = max(1, min(GRAIN_MAX_ALLS, round(strength * GRAIN_MAX_ALLS)))
    return f"noise=alls={alls}:allf=t+u"


def halation_ffmpeg_subgraph(halation: Optional[Dict[str, Any]], *, frame_height: int = HALATION_SIGMA_REF_H) -> Optional[str]:
    """`halation` descriptor -> an ffmpeg filtergraph FRAGMENT (not a single
    clause like the vignette/noise above): isolate highlights, red-orange
    tint them, blur into a glow, screen-blend back over the untouched frame.
    None for a no-op (absent/zero strength).

    Unlike a single filter clause, this uses named pads (`split`/labeled
    sub-chains/`blend`) -- valid spliced into one comma-joined `-vf` string
    (ffmpeg's filtergraph mini-language allows `;`-separated labeled chains
    inside a single `-vf`), as long as the fragment's OWN start/end still
    resolve to one implicit (unlabeled) input/output, which this does: it
    reads its input implicitly from whatever precedes it in the chain (the
    `split`'s input) and writes its output implicitly from the final
    `blend` (no trailing label), so the caller can just treat it as one
    more comma-joined chain element, exactly like `vignette_ffmpeg_filter`'s
    return value.

    Runs on an 8-bit YUV frame (matches where this is spliced in
    `_transform_vf`, right after the LUT's `format=yuv420p`), but converts
    to `gbrp` ONCE up front and keeps BOTH `blend` inputs (the untouched
    base and the isolated/tinted/blurred glow) in that same RGB space the
    whole way through, converting back to `yuv420p` only ONCE, after the
    blend. Verified live (real footage, real ffmpeg): converting `[hbase]`
    and `[hglow]` to `yuv420p` INDEPENDENTLY before blending -- two separate
    gbrp->yuv420p conversions of nominally "the same format" -- produced a
    visible magenta cast across the WHOLE frame (even pure-black letterbox
    bars, which the highlight threshold guarantees get zero glow
    contribution, so the cast could only be a range/space mismatch in
    `blend`'s raw per-sample arithmetic, not a real color choice). Doing
    the blend itself in `gbrp` and converting once afterward removed it.

    `frame_height` scales `HALATION_SIGMA` (tuned at `HALATION_SIGMA_REF_H`)
    so the glow's blur radius reads the same relative size on a 480p proxy
    and a 4K export, not a fixed pixel radius that looks tiny/huge depending
    on render resolution."""
    if not halation:
        return None
    strength = float(halation.get("strength") or 0.0)
    if strength <= 0.001:
        return None
    threshold_8bit = round(max(0.0, min(1.0, HALATION_THRESHOLD)) * 255.0)
    sigma = max(1.0, HALATION_SIGMA * (max(1, frame_height) / float(HALATION_SIGMA_REF_H)))
    tint_r, tint_g, tint_b = HALATION_TINT
    keep_bright = f"if(gt(val\\,{threshold_8bit})\\,val\\,0)"
    return (
        "format=gbrp,split=2[hbase][htmp];"
        f"[htmp]lutrgb=r='{keep_bright}':g='{keep_bright}':b='{keep_bright}',"
        f"colorchannelmixer=rr={tint_r:.3f}:gg={tint_g:.3f}:bb={tint_b:.3f},"
        f"gblur=sigma={sigma:.3f}[hglow];"
        f"[hbase][hglow]blend=all_mode=screen:all_opacity={max(0.0, min(1.0, strength)):.3f},"
        "format=yuv420p"
    )
