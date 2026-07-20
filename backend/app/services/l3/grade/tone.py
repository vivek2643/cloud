"""
Working-space + tone-mapping seam (color_grading_upgrade.plan.md Step 1.1):
what turns the CDL's flat "slope/offset filter" into something that reads as
GRADED, without touching either side of the parity contract separately --
both `to_working`/`from_working` get baked straight into the same `.cube`
`grade/lut_bake.py` already produces, so preview (WebGL) and export (ffmpeg
`lut3d`) inherit the change automatically.

Two pure functions over `(...,3)` float32 arrays in 0..1, plus their
implicit inverse relationship (`from_working` undoes `to_working`'s encoding,
not its tone curve -- that's one-directional by design):

  * `to_working(rgb_display, working_space)`: linearizes DISPLAY-encoded
    input into a scene-referred WORKING space. For `v1` this is a single,
    well-documented transfer -- the inverse sRGB/Rec.709 EOTF. This is
    deliberately the "slot": a fuller ACES input transform (IDT) can replace
    it later without any caller change (see the plan's "Libraries deferred").
  * `from_working(rgb_working, working_space)`: a FILMIC highlight shoulder
    (Reinhard-style, identity below `_SHOULDER_START` so shadows/midtones are
    literally untouched, asymptotic toward 1.0 above it so a bright/contrasty
    clip's highlights roll off instead of hard-clipping) -> re-encode to
    display (the sRGB/Rec.709 OETF).

`working_space != WORKING_SPACE_V1` (including the `legacy` default and any
older/unrecognized value) is IDENTITY on both functions -- this is exactly
what makes the `legacy` flag reproduce today's bytes: `apply_cdl` runs
directly on display values, no working-space transform, no tone map.

Why a shoulder starting well below 1.0 rather than a classic HDR filmic curve
(Hable/Uncharted2 with an 11.2-stop white point): that calibration assumes
input can run several stops over display white, so its overall gain is
LOWER than 1 across the whole 0..1 range -- applied to already-SDR footage
it visibly darkens shadows and midtones, which fails "never-worse"/"pleasing"
outright. A shoulder that's exact identity below `_SHOULDER_START` and only
compresses the top of the range keeps everything below it byte-for-byte
predictable while still preventing a hard highlight clip -- the actual
"parity-safe, no clipped tone" ask.
"""
from __future__ import annotations

WORKING_SPACE_V1 = "rec709_v1"

_SRGB_A = 0.055
_SRGB_LINEAR_THRESH = 0.0031308   # linear value where the sRGB OETF's two branches meet
_SRGB_DISPLAY_THRESH = 0.04045    # display value where the sRGB EOTF's two branches meet

# Below this (linear, post to_working), from_working is EXACT identity --
# shadows/midtones never move. Above it, highlights compress asymptotically
# toward 1.0 instead of hard-clipping.
_SHOULDER_START = 0.8


def to_working(rgb_display, working_space: str):
    """Display-encoded (gamma) RGB, 0..1 -> scene-referred linear RGB.
    Identity unless `working_space == WORKING_SPACE_V1`."""
    import numpy as np

    arr = np.asarray(rgb_display, dtype=np.float32)
    if working_space != WORKING_SPACE_V1:
        return arr
    arr = np.clip(arr, 0.0, 1.0)
    lo = arr / 12.92
    hi = np.power((arr + _SRGB_A) / (1.0 + _SRGB_A), 2.4)
    return np.where(arr <= _SRGB_DISPLAY_THRESH, lo, hi).astype(np.float32)


def _tonemap_shoulder(x):
    """Reinhard-style highlight shoulder: identity below `_SHOULDER_START`
    (C1-continuous at the boundary -- both sides approach slope 1 there, so
    there's no visible kink), then compresses everything above it
    asymptotically toward 1.0. Never exceeds 1.0 regardless of input."""
    import numpy as np

    headroom = 1.0 - _SHOULDER_START
    over = np.clip(x - _SHOULDER_START, 0.0, None)
    compressed = headroom * over / (headroom + over)
    return np.where(x <= _SHOULDER_START, x, _SHOULDER_START + compressed)


def from_working(rgb_working, working_space: str):
    """Scene-referred linear RGB -> filmic-tone-mapped, display-encoded RGB.
    Identity unless `working_space == WORKING_SPACE_V1`."""
    import numpy as np

    arr = np.asarray(rgb_working, dtype=np.float32)
    if working_space != WORKING_SPACE_V1:
        return arr
    arr = np.clip(arr, 0.0, None)
    toned = np.clip(_tonemap_shoulder(arr), 0.0, 1.0)
    lo = toned * 12.92
    hi = (1.0 + _SRGB_A) * np.power(toned, 1.0 / 2.4) - _SRGB_A
    return np.where(toned <= _SRGB_LINEAR_THRESH, lo, hi).astype(np.float32)
