"""
Bake a resolved Grade (CDL + optional creative .cube) into ONE 3D LUT, in the
standard Iridas/Resolve `.cube` text format that both sides of the parity
contract consume: ffmpeg's `lut3d` filter (export, render/compositor.py) and
a WebGL 3D-texture sampler (preview, frontend use-program-player.ts). Same
file, same math both places -- see color_grading.plan.md SS4 ("Fork A").

Grid convention (matches ffmpeg's `lut3d` reader and every mainstream tool):
R is the fastest-varying index, then G, then B -- i.e. row order is
`for b: for g: for r: yield (r, g, b)`.

Soft-local (SS9) is deliberately NOT part of this module: a 3D color LUT is a
pointwise value->value map with no notion of pixel position, so spatial
effects (a vignette, a sky gradient) cannot actually be "baked into" one --
despite the plan doc's shorthand. Those stay a small, separate set of
deterministic spatial parameters applied by an additional filter/shader pass
on both sides (still parity-safe: same numbers, same math), not squeezed
into this grid. See the soft-local layer for that piece.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from app.services.l3.grade.cdl import Grade, apply_cdl
from app.services.l3.grade.tone import from_working, to_working

DEFAULT_LUT_SIZE = 33


def _identity_grid(size: int):
    """(size, size, size, 3) float32 grid of RGB in 0..1, R fastest-varying."""
    import numpy as np

    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    # meshgrid with indexing="ij" on (b, g, r) so axis 0 = B, axis 1 = G, axis 2 = R
    # -- then stack as (..., [r,g,b]) so flattening axis order 0,1,2 = b,g,r
    # naturally yields the required "r fastest" row order when reshaped.
    bb, gg, rr = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([rr, gg, bb], axis=-1)  # (size,size,size,3), [b,g,r] index order


def _sample_lut_trilinear(lut_grid, coords):
    """Trilinearly sample `lut_grid` (M,M,M,3, indexed [b,g,r]) at `coords`
    (...,3) RGB values in 0..1. Returns an array shaped like `coords`."""
    import numpy as np

    m = lut_grid.shape[0]
    scaled = np.clip(coords, 0.0, 1.0) * (m - 1)
    r = scaled[..., 0]
    g = scaled[..., 1]
    b = scaled[..., 2]

    r0 = np.floor(r).astype(np.int32)
    g0 = np.floor(g).astype(np.int32)
    b0 = np.floor(b).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, m - 1)
    g1 = np.clip(g0 + 1, 0, m - 1)
    b1 = np.clip(b0 + 1, 0, m - 1)
    r0 = np.clip(r0, 0, m - 1)
    g0 = np.clip(g0, 0, m - 1)
    b0 = np.clip(b0, 0, m - 1)

    fr = (r - r0)[..., None]
    fg = (g - g0)[..., None]
    fb = (b - b0)[..., None]

    def at(bi, gi, ri):
        return lut_grid[bi, gi, ri]

    c000 = at(b0, g0, r0)
    c100 = at(b0, g0, r1)
    c010 = at(b0, g1, r0)
    c110 = at(b0, g1, r1)
    c001 = at(b1, g0, r0)
    c101 = at(b1, g0, r1)
    c011 = at(b1, g1, r0)
    c111 = at(b1, g1, r1)

    c00 = c000 * (1 - fr) + c100 * fr
    c10 = c010 * (1 - fr) + c110 * fr
    c01 = c001 * (1 - fr) + c101 * fr
    c11 = c011 * (1 - fr) + c111 * fr

    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg

    return c0 * (1 - fb) + c1 * fb


def bake_cube_text(
    grade: Grade,
    *,
    size: int = DEFAULT_LUT_SIZE,
    creative_lut_grid: Optional[Tuple] = None,
    title: str = "edso_grade",
    working_space: str = "rec709",
) -> str:
    """Bake `grade`'s CDL (and, if given, compose a parsed creative LUT on
    top) into `.cube` text. `creative_lut_grid` is the `(grid, size)` tuple
    `parse_cube_text` returns -- pass it straight through.

    `working_space` (color_grading_upgrade.plan.md Step 1.1): any value other
    than `tone.WORKING_SPACE_V1` (including the `legacy` default) makes
    `to_working`/`from_working` pure identity, so the pipeline collapses back
    to exactly `apply_cdl(grid, grade)` -- byte-identical to before this
    step existed. Under `v1`, the CDL runs INSIDE the working-space wrapper
    (linearize -> CDL -> filmic shoulder -> re-encode), which is what turns
    a flat slope/offset filter into something that reads as graded."""
    import numpy as np

    grid = _identity_grid(size)                       # (size,size,size,3) [b,g,r] index order
    working = to_working(grid, working_space)
    graded = apply_cdl(working, grade)                 # CDL (SS3 stack: Correct/Match/Arc live in cdl)
    out = from_working(graded, working_space)

    if creative_lut_grid is not None:
        lut_grid, _lut_size = creative_lut_grid
        out = _sample_lut_trilinear(lut_grid, out)

    out = np.clip(out, 0.0, 1.0)

    lines = [
        f'TITLE "{title}"',
        f"LUT_3D_SIZE {size}",
        "DOMAIN_MIN 0.0 0.0 0.0",
        "DOMAIN_MAX 1.0 1.0 1.0",
    ]
    # Flatten with axis order (b, g, r) -> row order r fastest, matching the
    # grid's own [b,g,r] index layout, i.e. no reordering needed here.
    flat = out.reshape(-1, 3)
    lines.extend(f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in flat.tolist())
    return "\n".join(lines) + "\n"


def parse_cube_text(text: str):
    """Parse `.cube` text into `(grid, size)` where `grid` is a
    `(size,size,size,3)` float32 array indexed `[b,g,r]` (matching
    `_identity_grid`'s layout, so it plugs straight into
    `_sample_lut_trilinear`/`bake_cube_text`'s `creative_lut_grid` arg)."""
    import numpy as np

    size: Optional[int] = None
    values: List[Tuple[float, float, float]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith("LUT_3D_SIZE"):
            size = int(line.split()[-1])
            continue
        if upper.startswith(("TITLE", "DOMAIN_MIN", "DOMAIN_MAX", "LUT_1D_SIZE")):
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            values.append((float(parts[0]), float(parts[1]), float(parts[2])))
        except ValueError:
            continue

    if not values:
        raise ValueError(".cube parse error: no data rows found")
    if size is None:
        # Infer from row count if the header was missing/malformed.
        size = round(len(values) ** (1.0 / 3.0))
    if size < 2:
        raise ValueError(f".cube parse error: LUT_3D_SIZE must be >= 2, got {size}")
    expected = size ** 3
    if len(values) != expected:
        raise ValueError(
            f".cube parse error: expected {expected} rows for LUT_3D_SIZE {size}, got {len(values)}"
        )

    arr = np.array(values, dtype=np.float32).reshape(size, size, size, 3)  # row order b,g,r -> [b,g,r] index
    return arr, size
