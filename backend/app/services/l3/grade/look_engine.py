"""
Parametric color-response engine (color_response_engine.plan.md): the Look
layer's fourth mode (`mode == "engine"`), turning a small `LookSpec` (a
colorist-style parameter set) into a real 3D LUT grid -- split-toning by
tonal zone, per-hue rotation, per-hue saturation, all things a CDL
slope/offset/power/sat delta fundamentally cannot express (a CDL can only
push a WHOLE channel, never "rotate greens toward teal without dragging
skin"). The grid rides the exact same `creative_lut_grid` seam an uploaded
`.cube` already flows through (`lut_bake.bake_cube_text`), so preview (WebGL)
and export (ffmpeg `lut3d`) inherit it identically -- no new bake path.

Pure, deterministic, no I/O: `build_look_grid(spec)` is a function of its
`LookSpec` alone, which is why the whole look is captured by the content
hash (`cdl.grade_hash`'s `look_engine` payload field) without hashing pixels.

Operates on the SAME `(size,size,size,3)` grid `lut_bake._identity_grid`
produces, in DISPLAY-encoded RGB (0..1) -- the space `bake_cube_text` samples
`creative_lut_grid` in, AFTER `from_working`'s tone curve (see that module's
docstring). This is exactly the space a colorist's creative LUT lives in.

This plan ships the ENGINE only, not the look library: two catalog entries
(`engine_identity`, an exact-identity parity anchor; `engine_punchy`, one
real look to prove per-hue moves land) -- the full library (YouTube + film +
ad styles) is deliberately deferred to a follow-up plan (see module docstring
of that plan for why: taste-validation needs real footage + iteration, not
more engine plumbing).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.grade.cdl import LUMA_B, LUMA_G, LUMA_R

# A hue band's Gaussian falloff needs a nonzero width; a caller-supplied 0
# would divide-by-zero -- floor it instead of raising (fail-open, matches
# every other bounded solver in this pipeline).
_MIN_BAND_WIDTH_DEG = 1e-3


@dataclass(frozen=True)
class LookSpec:
    """A colorist-style parameter set -- the vocabulary a CDL delta can't
    express. Every field defaults to a NO-OP, so `LookSpec()` is an EXACT
    identity grid (parity anchor + what "no look selected" resolves to)."""
    # Extra per-look contrast on top of the global tone curve (reuses
    # tone._contrast_pivot). 0.0 = none; positive = more contrast; NEGATIVE
    # = soften (color_look_library.plan.md -- "Bright & Airy"/"Vintage
    # Faded" need a softer, not just flatter-by-omission, midtone slope).
    # `g = 1.0 + contrast`, floored at 0.2 so a caller can't invert the
    # curve; `_contrast_pivot` is monotonic for any g>0.
    contrast: float = 0.0
    # 3-zone split-tone (lift/gamma/gain color balance). Each is an RGB tint
    # added, luma-weighted into that zone; magnitudes are small (~+/-0.1).
    shadow_tint: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    mid_tint: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    highlight_tint: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Hue-vs-hue: rotate a hue band. Each = (center_deg, width_deg, rotate_deg).
    hue_rotate: Tuple[Tuple[float, float, float], ...] = ()
    # Hue-vs-sat: scale saturation in a hue band. Each = (center_deg,
    # width_deg, sat_mult). e.g. (30,40,1.3) pops orange; (140,50,0.7) calms green.
    hue_sat: Tuple[Tuple[float, float, float], ...] = ()
    # Global saturation multiplier (1.0 = unchanged), applied last.
    sat: float = 1.0
    # halation_grain.plan.md: spatial finishing (SS9 follow-up) -- NOT baked
    # into the grid (build_look_grid ignores these two fields entirely; a
    # 3D color LUT has no notion of neighboring pixels or randomness, see
    # that module's docstring). Routed into the soft_local spatial
    # descriptor by resolve_clip_grade instead, alongside the vignette.
    halation: float = 0.0   # 0 = off; ~0.15-0.4 tasteful. Glow strength.
    grain: float = 0.0      # 0 = off; ~0.02-0.08 tasteful. Noise amplitude.
    # color_look_library.plan.md: faded/milky blacks -- raise the shadow
    # floor without touching white. 0 = off; ~0.03-0.10 tasteful.
    # `out = black_lift + (1-black_lift)*out` -> black lifts to
    # `black_lift`, white stays exactly 1 (a classic film "fade"; reduces
    # contrast in the toe specifically, distinct from -contrast's global
    # softening). Applied LAST (after contrast) so the fade isn't
    # re-crushed by the contrast op.
    black_lift: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Canonical (lists, not tuples; every field present) so the same
        spec always hashes identically -- `cdl.grade_hash`'s `look_engine`
        payload depends on this, not on pixels."""
        return {
            "contrast": self.contrast,
            "shadow_tint": list(self.shadow_tint),
            "mid_tint": list(self.mid_tint),
            "highlight_tint": list(self.highlight_tint),
            "hue_rotate": [list(b) for b in self.hue_rotate],
            "hue_sat": [list(b) for b in self.hue_sat],
            "sat": self.sat,
            "halation": self.halation,
            "grain": self.grain,
            "black_lift": self.black_lift,
        }

    @staticmethod
    def from_dict(d: Optional[Dict[str, Any]]) -> "LookSpec":
        """Missing keys -> no-op defaults (fail-open: a malformed/partial
        spec degrades toward identity, never crashes or produces a wilder
        look than intended)."""
        if not d:
            return LookSpec()

        def _tuple3(v, default=(0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
            if not v or len(v) != 3:
                return default
            try:
                return (float(v[0]), float(v[1]), float(v[2]))
            except (TypeError, ValueError):
                return default

        def _bands(v) -> Tuple[Tuple[float, float, float], ...]:
            if not v:
                return ()
            out = []
            for band in v:
                if not band or len(band) != 3:
                    continue
                try:
                    out.append((float(band[0]), float(band[1]), float(band[2])))
                except (TypeError, ValueError):
                    continue
            return tuple(out)

        return LookSpec(
            contrast=float(d.get("contrast") or 0.0),
            shadow_tint=_tuple3(d.get("shadow_tint")),
            mid_tint=_tuple3(d.get("mid_tint")),
            highlight_tint=_tuple3(d.get("highlight_tint")),
            hue_rotate=_bands(d.get("hue_rotate")),
            hue_sat=_bands(d.get("hue_sat")),
            sat=float(d.get("sat") if d.get("sat") is not None else 1.0),
            halation=float(d.get("halation") or 0.0),
            grain=float(d.get("grain") or 0.0),
            black_lift=float(d.get("black_lift") or 0.0),
        )

    def is_identity(self, eps: float = 1e-9) -> bool:
        return (
            abs(self.contrast) < eps
            and all(abs(v) < eps for v in self.shadow_tint)
            and all(abs(v) < eps for v in self.mid_tint)
            and all(abs(v) < eps for v in self.highlight_tint)
            and abs(self.black_lift) < eps
            and not self.hue_rotate
            and not self.hue_sat
            and abs(self.sat - 1.0) < eps
            and abs(self.halation) < eps
            and abs(self.grain) < eps
        )


# --------------------------------------------------------------------------
# Small, dependency-free RGB<->HSV pair (vectorized over the whole grid --
# 33**3 = ~36k points, arrays throughout, never a per-pixel Python loop).
# Verified against Python's stdlib `colorsys.rgb_to_hsv`/`hsv_to_rgb` (exact
# agreement across primaries, secondaries, grays, and arbitrary colors).
# Hue in DEGREES (0..360), matching LookSpec's `center_deg`/`width_deg`.
# --------------------------------------------------------------------------

def _rgb_to_hsv(rgb):
    import numpy as np

    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    v = maxc
    delta = maxc - minc
    delta_safe = np.where(delta > 1e-8, delta, 1.0)
    s = np.where(maxc > 1e-8, delta / np.where(maxc > 1e-8, maxc, 1.0), 0.0)

    # Mutually exclusive priority masks (r, then g, then b) -- standard
    # if/elif/elif semantics; a naive chained np.where without the
    # exclusion masks would let a LATER branch silently overwrite an
    # earlier tie (e.g. pure yellow, r==g==max) with the wrong formula.
    is_r = maxc == r
    is_g = (maxc == g) & ~is_r
    is_b = (maxc == b) & ~is_r & ~is_g

    h60 = np.zeros_like(r)
    h60 = np.where(is_r, ((g - b) / delta_safe) % 6.0, h60)
    h60 = np.where(is_g, (b - r) / delta_safe + 2.0, h60)
    h60 = np.where(is_b, (r - g) / delta_safe + 4.0, h60)
    h = (h60 * 60.0) % 360.0
    h = np.where(delta > 1e-8, h, 0.0)   # achromatic: hue undefined -> 0
    return h, s, v


def _hsv_to_rgb(h, s, v):
    import numpy as np

    hh = np.mod(h, 360.0) / 60.0
    i = np.floor(hh).astype(np.int32) % 6
    f = hh - np.floor(hh)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    conditions = [i == k for k in range(6)]
    r = np.select(conditions, [v, q, p, p, t, v], default=v)
    g = np.select(conditions, [t, v, v, q, p, p], default=v)
    b = np.select(conditions, [p, p, t, v, v, q], default=v)
    return np.stack([r, g, b], axis=-1)


def _hue_distance_deg(h, center: float):
    """Circular hue distance in degrees, always in [0, 180]."""
    import numpy as np

    d = np.abs(h - center) % 360.0
    return np.minimum(d, 360.0 - d)


def _band_weight(h, s, center: float, width: float):
    """Gaussian falloff over hue distance, ADDITIONALLY weighted by
    saturation `s` -- the achromatic guard: a near-gray pixel's hue is
    numerically ~0 (arbitrary/undefined) but its weight collapses to ~0
    regardless of `center`/`width`, so hue ops never inject a cast the
    Correct/WB layer just removed, and never tint a true neutral."""
    import numpy as np

    sigma = max(width / 2.0, _MIN_BAND_WIDTH_DEG)
    d = _hue_distance_deg(h, center)
    return np.exp(-0.5 * (d / sigma) ** 2) * s


# --------------------------------------------------------------------------
# The ops, in the fixed order build_look_grid applies them. All no-op at
# their default LookSpec values (verified by test_look_identity_spec_is_
# identity_grid). No intermediate clamp -- only the final grid clamp in
# build_look_grid -- so a small overshoot from one op doesn't get clipped
# before the next op sees it (matches the plan's minimal pseudocode).
# --------------------------------------------------------------------------

def _apply_split_tone(rgb, spec: LookSpec):
    if not (any(spec.shadow_tint) or any(spec.mid_tint) or any(spec.highlight_tint)):
        return rgb
    import numpy as np

    y = rgb[..., 0] * LUMA_R + rgb[..., 1] * LUMA_G + rgb[..., 2] * LUMA_B
    w_shadow = ((1.0 - y) ** 2)[..., None]
    w_high = (y ** 2)[..., None]
    w_mid = 1.0 - w_shadow - w_high
    shadow = np.array(spec.shadow_tint, dtype=np.float32)
    mid = np.array(spec.mid_tint, dtype=np.float32)
    high = np.array(spec.highlight_tint, dtype=np.float32)
    return rgb + w_shadow * shadow + w_mid * mid + w_high * high


def _apply_hue_rotate(rgb, spec: LookSpec):
    if not spec.hue_rotate:
        return rgb
    h, s, v = _rgb_to_hsv(rgb)
    for center, width, rotate_deg in spec.hue_rotate:
        w = _band_weight(h, s, center, width)
        h = (h + rotate_deg * w) % 360.0
    return _hsv_to_rgb(h, s, v)


def _apply_hue_sat(rgb, spec: LookSpec):
    if not spec.hue_sat:
        return rgb
    import numpy as np

    h, s, v = _rgb_to_hsv(rgb)
    for center, width, mult in spec.hue_sat:
        w = _band_weight(h, s, center, width)
        s = np.clip(s * (1.0 + (mult - 1.0) * w), 0.0, 1.0)
    return _hsv_to_rgb(h, s, v)


def _apply_global_sat(rgb, spec: LookSpec):
    if abs(spec.sat - 1.0) < 1e-9:
        return rgb
    luma = (rgb[..., 0] * LUMA_R + rgb[..., 1] * LUMA_G + rgb[..., 2] * LUMA_B)[..., None]
    return luma + spec.sat * (rgb - luma)


def _apply_black_lift(rgb, spec: LookSpec):
    """`out = black_lift + (1-black_lift)*out` -- an affine remap: black
    (0) lifts to `black_lift`, white (1) stays EXACTLY 1 (slope
    `1-black_lift`, always positive for the intended `[0, ~0.1]` range, so
    always monotonic). Applied last, after contrast, so the fade isn't
    re-crushed by it (color_look_library.plan.md's "Bright & Airy"/
    "Vintage Faded" need lifted/milky blacks, distinct from -contrast's
    global softening)."""
    if abs(spec.black_lift) < 1e-9:
        return rgb
    return spec.black_lift + (1.0 - spec.black_lift) * rgb


def build_look_grid(spec: LookSpec, size: int = 33):
    """Evaluate the color-response ops over an identity grid and return
    `(grid, size)` -- the SAME shape `parse_cube_text` returns, so it drops
    straight into `bake_cube_text(creative_lut_grid=...)`. Pure/deterministic:
    the whole look is a function of `spec` alone. Fixed op order (split-tone
    -> hue-rotate -> hue-sat -> global-sat -> contrast -> black_lift) --
    documented, not incidental; the validation looks and contact-sheet
    tuning assume it.

    Deliberately does NOT read `spec.halation`/`spec.grain`
    (halation_grain.plan.md) -- those are spatial/stochastic (neighbor +
    randomness), which a pointwise 3D LUT grid cannot express by
    construction (see `lut_bake.py`'s module docstring). `resolve_clip_grade`
    routes them into the `soft_local` descriptor instead, alongside the
    vignette."""
    import numpy as np

    from app.services.l3.grade.lut_bake import _identity_grid
    from app.services.l3.grade.tone import _contrast_pivot

    grid = _identity_grid(size)
    out = grid
    out = _apply_split_tone(out, spec)
    out = _apply_hue_rotate(out, spec)
    out = _apply_hue_sat(out, spec)
    out = _apply_global_sat(out, spec)
    if spec.contrast != 0.0:
        # color_look_library.plan.md: NEGATIVE contrast (soften) is now
        # allowed too -- g<1 gently lifts shadows/rolls off highlights
        # instead of steepening the midtone slope. Floored at 0.2 so a
        # caller can't invert the curve; _contrast_pivot is monotonic for
        # any g>0 (verified: g=0.2..2.5 all stay monotonic/endpoint-pinned).
        #
        # _contrast_pivot uses FRACTIONAL powers (x/p)**g -- on the RAW
        # base identity grid this is always safe (x in [0,1] exactly), but
        # split-tone/hue-sat can push a channel slightly outside [0,1]
        # (small tints/boosts near the grid's 0/1 corners), and a fractional
        # power of a negative or >1-relative base is NaN, not a large
        # number. Clip immediately before this specific op -- the only one
        # in this chain that requires a bounded domain (HSV round-trips and
        # the additive tint are well-defined for any real RGB).
        g = max(0.2, 1.0 + spec.contrast)
        out = _contrast_pivot(np.clip(out, 0.0, 1.0), g)
    out = _apply_black_lift(out, spec)
    return np.clip(out, 0.0, 1.0).astype(np.float32), size


# --------------------------------------------------------------------------
# Look catalog (color_look_library.plan.md): a real, YouTube-centric library
# -- 6 creator looks (priority: most users), 6 film looks (the minority
# family, carries halation+grain), 4 ad/commercial looks. `engine_identity`
# stays as the parity anchor. The two engine-validation-only entries the
# prior plan shipped (`engine_punchy`/`engine_film`) are subsumed by
# `punchy_vibrant`/`kodak_2383` below and dropped -- this catalog is the
# real product, not instrument validation anymore.
#
# "Fit to film-stock data" here means INFORMED AUTHORING, not spectral
# simulation: `LookSpec` is a creative vocabulary (contrast, split-tone,
# hue-rotate, hue-sat, sat, halation, grain, black_lift), not a physical
# film model (density curves, dye couplers, spectral crosstalk). A "Kodak
# 2383" entry means dialing our knobs to match that stock's PUBLISHED
# color-science character (contrasty, teal-orange separation, warm
# highlights, fine grain) -- never their .cube files (licensing), never a
# numerical spectral fit. Values below are STARTING points tuned on real
# footage via `scripts/_diag_look_contact_sheet.py` (see that script for
# the actual contact-sheet renders this catalog was checked against).
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EngineLook:
    look_id: str
    label: str
    description: str
    spec: LookSpec = field(default_factory=LookSpec)
    # "creator" (YouTube/vlog, the priority family -- most users), "film"
    # (carries halation+grain, needs grade_film_texture too), or "ad"
    # (commercial/product). Purely a frontend-filtering hook -- "as many as
    # we want for now, filter later," so tag, don't prune.
    family: str = "creator"


LOOKS: List[EngineLook] = [
    EngineLook(
        "engine_identity", "Engine Identity",
        "Exact identity -- the engine's parity anchor, no stylization.",
        LookSpec(), family="creator",
    ),

    # ---- Creator family (priority -- most users) --------------------------
    EngineLook(
        "clean_natural", "Clean & Natural",
        "Safe, true-to-life default -- a light contrast lift and a touch of "
        "saturation on top of the fully-corrected image, nothing more.",
        LookSpec(contrast=0.05, sat=1.03),
        family="creator",
    ),
    EngineLook(
        "bright_airy", "Bright & Airy",
        "Lifted, soft, warm -- the vlog/lifestyle look: softened contrast, "
        "milky-lifted blacks, a gentle warm cast top to bottom.",
        LookSpec(
            contrast=-0.10, black_lift=0.05,
            shadow_tint=(0.02, 0.015, 0.0), highlight_tint=(0.03, 0.02, -0.01),
            sat=0.98,
        ),
        family="creator",
    ),
    EngineLook(
        "punchy_vibrant", "Punchy Vibrant",
        "Saturated, contrasty pop for tech / gaming / product footage -- "
        "popped orange/skin, calmed green, real per-hue moves a CDL preset "
        "can't make.",
        LookSpec(contrast=0.20, sat=1.15, hue_sat=((30.0, 40.0, 1.2), (150.0, 50.0, 0.9))),
        family="creator",
    ),
    EngineLook(
        "warm_cozy", "Warm Cozy",
        "Gentle orange warmth for sit-down/talking-head content -- warm "
        "shadows and highlights, a mild orange-band saturation lift.",
        LookSpec(
            contrast=0.08, shadow_tint=(0.02, 0.005, -0.01), highlight_tint=(0.05, 0.02, -0.03),
            hue_sat=((30.0, 45.0, 1.15),), sat=1.05,
        ),
        family="creator",
    ),
    EngineLook(
        "cool_clean", "Cool Clean",
        "Slightly cool, crisp -- a reviewer/tech-desk look: cool shadow/"
        "highlight tint, a firmer contrast lift, restrained saturation.",
        LookSpec(
            contrast=0.12, shadow_tint=(-0.02, 0.0, 0.03), highlight_tint=(-0.01, 0.0, 0.02),
            sat=1.05,
        ),
        family="creator",
    ),
    EngineLook(
        "moody_cinematic", "Moody Cinematic",
        "Desaturated, teal shadows, a gentle crush -- an editorial/somber "
        "mood for narrative or B-roll-heavy cuts.",
        LookSpec(
            contrast=0.18, shadow_tint=(-0.04, 0.0, 0.05), highlight_tint=(0.03, 0.015, -0.02),
            hue_sat=((150.0, 50.0, 0.7),), sat=0.82,
        ),
        family="creator",
    ),

    # ---- Film family (informed by stock character; carries halation+grain,
    # needs grade_film_texture on too -- their COLOR still applies under
    # grade_look_engine alone, but the glow/grain texture stays off without
    # the second flag; not a bug, see config.py's grade_film_texture note) --
    EngineLook(
        "kodak_2383", "Kodak 2383",
        "Print-film character: contrasty, teal-orange separation, warm "
        "highlights, fine grain. Informed by Kodak 2383's published "
        "color-science description, not its .cube or a spectral fit. "
        "Needs grade_film_texture for the halation/grain half.",
        LookSpec(
            contrast=0.22, shadow_tint=(-0.04, 0.0, 0.05), highlight_tint=(0.05, 0.02, -0.03),
            hue_sat=((30.0, 40.0, 1.2), (150.0, 50.0, 0.85)), sat=1.05,
            halation=0.25, grain=0.04,
        ),
        family="film",
    ),
    EngineLook(
        "fuji_eterna", "Fuji Eterna",
        "Soft, low-saturation, green-leaning -- Fuji Eterna's motion-picture "
        "character (gentle contrast, desaturated, a cool-green midtone "
        "lean). Needs grade_film_texture for the halation/grain half.",
        LookSpec(contrast=0.04, mid_tint=(-0.01, 0.02, -0.01), sat=0.88, halation=0.15, grain=0.05),
        family="film",
    ),
    EngineLook(
        "vision3_250d", "Vision3 250D",
        "Natural negative stock: gentle contrast, a slight warm lean, "
        "faint lifted blacks. Needs grade_film_texture for the "
        "halation/grain half.",
        LookSpec(
            contrast=0.08, black_lift=0.02, mid_tint=(0.02, 0.01, -0.01), sat=0.98,
            halation=0.12, grain=0.05,
        ),
        family="film",
    ),
    EngineLook(
        "portra_400", "Portra 400",
        "Warm, skin-flattering portrait stock: soft, low-contrast, an "
        "orange-band lift for warm skin. Needs grade_film_texture for the "
        "halation/grain half.",
        LookSpec(
            contrast=0.0, black_lift=0.03, highlight_tint=(0.04, 0.02, -0.02),
            hue_sat=((30.0, 45.0, 1.1),), sat=0.95, halation=0.10, grain=0.04,
        ),
        family="film",
    ),
    EngineLook(
        "vintage_faded", "Vintage Faded",
        "Lifted milky blacks, warm, desaturated -- an aged-print fade. "
        "Needs grade_film_texture for the halation/grain half.",
        LookSpec(
            contrast=-0.05, black_lift=0.08, mid_tint=(0.03, 0.015, -0.02), sat=0.80,
            halation=0.20, grain=0.06,
        ),
        family="film",
    ),
    EngineLook(
        "bw_film", "B&W Film",
        "Near-monochrome, contrasty, grainy -- a graphic black & white "
        "film look (a creative near-desaturation, not a true single-channel "
        "conversion). Needs grade_film_texture for the halation/grain half.",
        LookSpec(contrast=0.20, sat=0.05, halation=0.10, grain=0.06),
        family="film",
    ),

    # ---- Ad / commercial family --------------------------------------------
    EngineLook(
        "clean_commercial", "Clean Commercial",
        "Crisp, neutral, punchy -- a no-grain commercial base for product/"
        "corporate cuts.",
        LookSpec(contrast=0.15, sat=1.08),
        family="ad",
    ),
    EngineLook(
        "high_key_beauty", "High-Key Beauty",
        "Bright, airy, soft skin, warm -- a beauty/lifestyle high-key look: "
        "softened contrast, lifted blacks, a warm highlight + orange-band lift.",
        LookSpec(
            contrast=-0.08, black_lift=0.05, highlight_tint=(0.03, 0.02, -0.01),
            hue_sat=((30.0, 45.0, 1.1),), sat=1.0,
        ),
        family="ad",
    ),
    EngineLook(
        "tech_sleek", "Tech Sleek",
        "Cool, high-contrast, restrained saturation -- a product/tech-launch "
        "look.",
        LookSpec(
            contrast=0.20, shadow_tint=(-0.03, 0.0, 0.04), highlight_tint=(-0.01, 0.0, 0.02),
            sat=0.95,
        ),
        family="ad",
    ),
    EngineLook(
        "food_vibrant", "Food Vibrant",
        "Warm, saturated, appetizing -- pops orange/red and yellow for food "
        "and product close-ups.",
        LookSpec(
            contrast=0.12, highlight_tint=(0.04, 0.02, -0.02),
            hue_sat=((30.0, 45.0, 1.25), (60.0, 40.0, 1.15)), sat=1.18,
        ),
        family="ad",
    ),
]

_BY_ID: Dict[str, EngineLook] = {look.look_id: look for look in LOOKS}


def get_engine_look(look_id: str) -> Optional[EngineLook]:
    return _BY_ID.get(look_id)


def list_engine_looks() -> List[Dict[str, str]]:
    """Gallery listing, same shape as `presets.list_presets` plus a `mode`
    tag (both catalogs share one gallery endpoint -- see routers/grade.py)
    and a `family` tag (color_look_library.plan.md's frontend-filtering
    hook: "creator" / "film" / "ad")."""
    return [
        {
            "look_id": look.look_id, "label": look.label, "description": look.description,
            "mode": "engine", "family": look.family,
        }
        for look in LOOKS
    ]


def resolve_look_spec(sequence_look: Dict[str, Any]) -> Optional[LookSpec]:
    """`sequence_look.look_id` names a catalog look (preferred, e.g. a user's
    gallery pick) -> its spec. Else `sequence_look.look_params` builds an
    inline spec (a preview/one-off, not yet catalogued). `None` when neither
    resolves -- the caller then applies no engine look at all (fail-open,
    same discipline as `presets.get_preset` returning `None` on a bad id)."""
    look_id = sequence_look.get("look_id")
    if look_id:
        found = get_engine_look(str(look_id))
        if found is not None:
            return found.spec
    params = sequence_look.get("look_params")
    if params:
        return LookSpec.from_dict(params)
    return None
