"""
Render the RESOLVED layer set to an MP4.

Two paths, picked automatically:

  * Pure spine (no operations -> every layer is `kind="spine"`): normalize each
    spine segment with ffmpeg (cached per file+range+preset) and concat with the
    demuxer. Warm renders are near-free. This is the cut-only fast path.

  * Layered (coverage / beds / split edits / ducking): one `filter_complex`
    graph. Video layers are composited bottom->top by z over a black canvas
    (each shifted to its program start; opacity via alpha); audio layers are
    delayed to their program start, gained (gain+duck dB), and summed with amix.

Both consume `layers.ResolvedTimeline.to_dict()` so the render and the preview
can never disagree about what plays.

Hard cuts only -- no transitions/speed/text yet (those are later phases and
slot in as extra filter nodes without changing this contract).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.services.l3.captions.ass_export import captions_to_ass
from app.services.l3.grade.cache import ensure_cube_file
from app.services.l3.grade.cdl import is_identity, Grade
from app.services.l3.grade.softlocal import grain_ffmpeg_filter, halation_ffmpeg_subgraph, vignette_ffmpeg_filter
from app.services.processing import _download_from_r2, _upload_to_r2
from app.services.r2 import generate_presigned_get

logger = logging.getLogger(__name__)

CACHE_ROOT = os.environ.get("EDSO_RENDER_CACHE", "/tmp/edso_render_cache")
RENDER_PREFIX = "renders"
# Self-hosted caption font binaries for the ASS burn's `fontsdir=` (captions.
# plan.md SS12/SS13). Defaults to the fonts bundled in this repo (the same 6
# families the frontend @font-face-registers from public/fonts, so the burn
# matches the preview typographically); an env override still wins for a
# custom deploy. An empty/missing dir is NOT an error -- libass/fontconfig
# just falls back to whatever system font matches the family name.
_BUNDLED_CAPTION_FONTS = os.path.join(os.path.dirname(__file__), "caption_fonts")
CAPTION_FONTS_DIR = os.environ.get("EDSO_CAPTION_FONTS_DIR", _BUNDLED_CAPTION_FONTS)

# `long_edge` is the quality tier; the actual W x H is derived from the edit's
# delivery aspect so the SAME preset renders landscape, portrait, or square.
PRESETS: Dict[str, Dict[str, Any]] = {
    "preview": {
        "long_edge": 1280, "fps": 30,
        "video_codec": "libx264", "video_preset": "veryfast", "video_crf": 24,
        "audio_codec": "aac", "audio_bitrate": "128k", "use_proxy": True,
    },
    "export": {
        "long_edge": 1920, "fps": 30,
        "video_codec": "libx264", "video_preset": "medium", "video_crf": 20,
        "audio_codec": "aac", "audio_bitrate": "192k", "use_proxy": False,
    },
}

# Delivery aspect -> (w_ratio, h_ratio). The long edge maps to the larger ratio
# component, so the frame is exactly the requested shape at the preset quality.
ASPECT_RATIOS: Dict[str, Tuple[int, int]] = {
    "landscape": (16, 9),
    "portrait": (9, 16),
    "square": (1, 1),
}


def _even(n: int) -> int:
    """yuv420p needs even dimensions."""
    n = int(round(n))
    return n - (n % 2)


def canvas_dims(long_edge: int, aspect: str) -> Tuple[int, int]:
    """(width, height) for `aspect` at the given long edge (default landscape)."""
    rw, rh = ASPECT_RATIOS.get(aspect, ASPECT_RATIOS["landscape"])
    if rw >= rh:  # landscape / square: width is the long edge
        return _even(long_edge), _even(long_edge * rh / rw)
    return _even(long_edge * rw / rh), _even(long_edge)  # portrait: height is long


def _cfg_for(preset: str, aspect: str) -> Dict[str, Any]:
    """Concrete render config: the preset's quality tier sized to `aspect`."""
    base = PRESETS[preset]
    w, h = canvas_dims(int(base["long_edge"]), aspect)
    return {**base, "width": w, "height": h}

ProgressCb = Optional[Callable[[int, str], None]]


@dataclass
class FileEntry:
    file_id: str
    r2_key: str
    r2_proxy_key: Optional[str]
    has_video: bool = True


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def resolved_hash(resolved: Dict[str, Any], preset: str) -> str:
    """Stable fingerprint of (timeline, preset) for render de-dup / caching."""
    payload = json.dumps(resolved, sort_keys=True) + "|" + preset
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def render_resolved(
    resolved: Dict[str, Any],
    file_lookup: Dict[str, FileEntry],
    preset: str = "preview",
    progress_cb: ProgressCb = None,
) -> Tuple[str, int]:
    """Render `resolved` to MP4, upload to R2, return (output_r2_key, duration_ms)."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset {preset!r}; known: {list(PRESETS)}")
    aspect = str(resolved.get("aspect") or "landscape")
    cfg = _cfg_for(preset, aspect)
    os.makedirs(CACHE_ROOT, exist_ok=True)

    video = sorted(resolved.get("video_layers") or [], key=lambda v: (v["prog_start_ms"], v["z"]))
    audio = sorted(resolved.get("audio_layers") or [], key=lambda a: a["prog_start_ms"])
    total_ms = int(resolved.get("duration_ms") or 0)
    if not video and not audio:
        raise ValueError("Nothing to render: resolved timeline is empty.")
    if total_ms <= 0:
        total_ms = max([v["prog_end_ms"] for v in video] + [a["prog_end_ms"] for a in audio] + [0])
    if total_ms <= 0:
        raise ValueError("Resolved timeline has zero duration.")

    def report(pct: int, label: str) -> None:
        logger.info("render: %d%% %s", pct, label)
        if progress_cb:
            try:
                progress_cb(pct, label)
            except Exception:
                logger.exception("render progress_cb raised; ignoring")

    # The concat fast path encodes video+audio from ONE shared source window per
    # segment, so it is only valid while every audio layer is still coupled to a
    # video layer (same source span, same program window). A J/L split edit
    # decouples them -- those must render through the layered graph.
    v_windows = {(v.get("source_file_id"), v.get("src_in_ms"), v.get("src_out_ms"),
                  v.get("prog_start_ms"), v.get("prog_end_ms")) for v in video}
    pure_spine = (
        all(v.get("kind") == "spine" for v in video)
        and all(a.get("kind") == "spine" for a in audio)
        and len(audio) <= len(video)  # spine: one dialogue layer per picture
        # A muted/gained spine layer needs the filter graph (the concat fast path
        # can't apply per-segment volume) -- fall through to the layered renderer.
        and all(abs(float(a.get("gain_db", 0.0))) < 0.01 for a in audio)
        and all((a.get("source_file_id"), a.get("src_in_ms"), a.get("src_out_ms"),
                 a.get("prog_start_ms"), a.get("prog_end_ms")) in v_windows for a in audio)
    )

    with tempfile.TemporaryDirectory(prefix="edso_render_") as tmp:
        ass_path = _write_ass_file(resolved.get("captions") or [], cfg, tmp)
        if pure_spine and video:
            out_key = _render_spine_concat(video, cfg, file_lookup, tmp, report, ass_path=ass_path)
        else:
            out_key = _render_layers(video, audio, total_ms, cfg, file_lookup, tmp, report, ass_path=ass_path)
    report(100, "done")
    return out_key, total_ms


def presigned_url_for(out_key: str, expires_in: int = 86400) -> str:
    return generate_presigned_get(out_key, expires_in=expires_in)


# --------------------------------------------------------------------------
# Source media
# --------------------------------------------------------------------------

def _source_path(
    entry: FileEntry, cfg: Dict[str, Any], tmp: str, cache: Dict[str, str]
) -> str:
    key = (entry.r2_proxy_key if cfg["use_proxy"] and entry.r2_proxy_key else entry.r2_key)
    if key not in cache:
        ext = os.path.splitext(key)[1] or ".mp4"
        local = os.path.join(tmp, f"src_{len(cache):04d}{ext}")
        _download_from_r2(key, local)
        cache[key] = local
    return cache[key]


# --------------------------------------------------------------------------
# Pure-spine fast path: per-segment normalized cache + concat demuxer
# --------------------------------------------------------------------------

def _render_spine_concat(
    video: List[dict],
    cfg: Dict[str, Any],
    file_lookup: Dict[str, FileEntry],
    tmp: str,
    report: Callable[[int, str], None],
    ass_path: Optional[str] = None,
) -> str:
    report(4, "starting (spine)")
    src_cache: Dict[str, str] = {}
    segments: List[str] = []
    n = len(video)
    cube_dir = os.path.join(tmp, "cubes")
    for i, v in enumerate(video):
        entry = file_lookup.get(v["source_file_id"])
        if entry is None:
            raise RuntimeError(f"no FileEntry for {v['source_file_id']}")
        in_ms, out_ms = int(v["src_in_ms"]), int(v["src_out_ms"])
        transform = v.get("transform") or {}
        grade = v.get("grade") or {}
        cache_path = _segment_cache_path(v["source_file_id"], in_ms, out_ms, cfg, transform, grade)
        if not (os.path.exists(cache_path) and os.path.getsize(cache_path) > 0):
            src = _source_path(entry, cfg, tmp, src_cache)
            cube_path = ensure_cube_file(grade, cube_dir)
            soft_local = grade.get("soft_local") or {}
            vignette_filter = vignette_ffmpeg_filter(soft_local.get("vignette"))
            halation_filter = halation_ffmpeg_subgraph(soft_local.get("halation"), frame_height=cfg["height"])
            grain_filter = grain_ffmpeg_filter(soft_local.get("grain"))
            report(_pct(i, n, 8, 78), f"cut {i + 1}/{n}")
            _produce_segment(src=src, dst=cache_path, in_ms=in_ms, out_ms=out_ms,
                             cfg=cfg, transform=transform, cube_path=cube_path,
                             vignette_filter=vignette_filter,
                             halation_filter=halation_filter, grain_filter=grain_filter)
        else:
            report(_pct(i, n, 8, 78), f"cut {i + 1}/{n} (cache)")
        segments.append(cache_path)

    report(82, "concatenating")
    out_local = os.path.join(tmp, "out.mp4")
    _concat_segments(segments, out_local, cfg)
    if ass_path:
        report(88, "burning captions")
        out_local = _burn_captions(out_local, ass_path, cfg)
    report(92, "uploading")
    out_key = f"{RENDER_PREFIX}/{uuid.uuid4().hex}.mp4"
    _upload_to_r2(out_local, out_key, "video/mp4")
    return out_key


def _transform_key(transform: Optional[Dict[str, Any]]) -> str:
    """Stable cache token for a layer transform; identity collapses to '' so
    untouched (letterbox) segments keep their existing cache entries."""
    t = transform or {}
    rotate = int(t.get("rotate") or 0)
    fit = t.get("fit") or "contain"
    anchor = t.get("anchor") or "center"
    try:
        zoom = round(max(1.0, float(t.get("zoom") or 1.0)), 4)
    except (TypeError, ValueError):
        zoom = 1.0
    focus = t.get("focus") if isinstance(t.get("focus"), dict) else None
    motion = t.get("motion") if isinstance(t.get("motion"), dict) else None
    if (rotate == 0 and fit == "contain" and anchor == "center"
            and zoom == 1.0 and not focus and not motion):
        return ""
    fkey = ""
    if focus:
        try:
            fkey = f"|f{float(focus['cx']):.4f},{float(focus['cy']):.4f}"
        except (KeyError, TypeError, ValueError):
            fkey = ""
    mkey = ""
    if motion:
        try:
            fr, to = motion["from"], motion["to"]
            mkey = (f"|m{fr['scale']:.3f},{fr['cx']:.3f},{fr['cy']:.3f}"
                    f">{to['scale']:.3f},{to['cx']:.3f},{to['cy']:.3f}"
                    f":{motion.get('ease', 'linear')}:{int(motion.get('dur_ms') or 0)}")
        except (KeyError, TypeError, ValueError):
            mkey = ""
    return f"r{rotate}|{fit}|{anchor}|z{zoom}{fkey}{mkey}"


def _grade_key(grade: Optional[Dict[str, Any]]) -> str:
    """Stable cache token for a resolved grade; identity (no CDL, no creative
    LUT, no soft-local, no tone_contrast, no look_engine) collapses to '' so
    ungraded segments keep their existing cache entries, mirroring
    `_transform_key`'s identity-collapse contract.

    color_tone_contrast.plan.md: `tone_contrast` (baked into `from_working`
    independent of the CDL) MUST be part of this identity check too -- an
    identity CDL with `tone_contrast>0` is NOT a no-op bake, and without this
    check it would collapse to the SAME '' token as a truly-identity segment
    rendered before the flag existed, silently reusing a stale cached
    segment clip that never got the contrast curve.

    color_response_engine.plan.md: `look_engine` is the same class of gap --
    an identity CDL with a non-identity LookSpec baked into the creative LUT
    grid is NOT a no-op bake either."""
    if not grade:
        return ""
    cdl = Grade.from_dict(grade.get("cdl"))
    lut_ref = grade.get("creative_lut_ref")
    soft_local = grade.get("soft_local")
    tone_contrast = float(grade.get("tone_contrast") or 0.0)
    look_engine = grade.get("look_engine")
    if (
        is_identity(cdl) and not lut_ref and not soft_local
        and tone_contrast <= 0.0 and not look_engine
    ):
        return ""
    h = grade.get("grade_hash")
    return f"g{h}" if h else ""


def _segment_cache_path(
    file_id: str, in_ms: int, out_ms: int, cfg: Dict[str, Any],
    transform: Optional[Dict[str, Any]] = None,
    grade: Optional[Dict[str, Any]] = None,
) -> str:
    key = f"{file_id}|{in_ms}|{out_ms}|{cfg['width']}x{cfg['height']}|{cfg['fps']}|{cfg['video_crf']}"
    tk = _transform_key(transform)
    if tk:
        key += f"|{tk}"
    gk = _grade_key(grade)
    if gk:
        key += f"|{gk}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return os.path.join(CACHE_ROOT, f"{digest}.mp4")


def _transpose_chain(rotate: int) -> List[str]:
    """Orthogonal rotation via ffmpeg `transpose` (1 = 90 CW, 2 = 90 CCW)."""
    if rotate == 90:
        return ["transpose=1"]
    if rotate == 270:
        return ["transpose=2"]
    if rotate == 180:
        return ["transpose=1", "transpose=1"]
    return []


def _crop_to(w: int, h: int, anchor: str, focus: Optional[Dict[str, Any]] = None) -> str:
    """Crop a (>= w x h) frame down to w x h. When `focus` (normalized cx,cy) is
    given, center the crop window on it (clamped into the frame); otherwise bias
    to `anchor` (else center). The focus offset is an ffmpeg expression over the
    post-scale dims (iw/ih), with commas escaped + single-quoted so the
    filtergraph parser keeps it as one argument."""
    if focus and "cx" in focus and "cy" in focus:
        cx = max(0.0, min(1.0, float(focus["cx"])))
        cy = max(0.0, min(1.0, float(focus["cy"])))
        x = f"'min(max(0\\,{cx:.4f}*iw-{w // 2})\\,iw-{w})'"
        y = f"'min(max(0\\,{cy:.4f}*ih-{h // 2})\\,ih-{h})'"
        return f"crop={w}:{h}:{x}:{y}"
    if anchor == "left":
        x = "0"
    elif anchor == "right":
        x = f"iw-{w}"
    else:
        x = f"(iw-{w})/2"
    if anchor == "top":
        y = "0"
    elif anchor == "bottom":
        y = f"ih-{h}"
    else:
        y = f"(ih-{h})/2"
    return f"crop={w}:{h}:{x}:{y}"


def _ease_expr(motion: Dict[str, Any], fps: int) -> str:
    """Eased progress 0..1 over the layer span, as an ffmpeg expr in `on`
    (output frame index). Matches layers.sample_motion."""
    dur_s = max(0.05, float(motion.get("dur_ms") or 50) / 1000.0)
    p = f"min(1\\,on/({fps}*{dur_s:.3f}))"
    if motion.get("ease") == "smooth":
        return f"(3*({p})*({p})-2*({p})*({p})*({p}))"
    return f"({p})"


def _zoompan(motion: Dict[str, Any], w: int, h: int, fps: int) -> str:
    """Animate scale + focus over a cover-filled w x h base via `zoompan`,
    evaluating the SAME from->to eased path as layers.sample_motion. The zoom
    window is centered on the (animated) focus and clamped into the frame."""
    fr, to = motion["from"], motion["to"]
    pe = _ease_expr(motion, fps)
    z = f"({fr['scale']:.4f}+({to['scale'] - fr['scale']:.4f})*{pe})"
    cx = f"({fr['cx']:.4f}+({to['cx'] - fr['cx']:.4f})*{pe})"
    cy = f"({fr['cy']:.4f}+({to['cy'] - fr['cy']:.4f})*{pe})"
    zmax = max(fr["scale"], to["scale"])
    zexpr = f"max(1\\,min({zmax:.4f}\\,{z}))"
    xexpr = f"max(0\\,min(iw-iw/zoom\\,({cx})*iw-(iw/zoom)/2))"
    yexpr = f"max(0\\,min(ih-ih/zoom\\,({cy})*ih-(ih/zoom)/2))"
    return f"zoompan=z='{zexpr}':x='{xexpr}':y='{yexpr}':d=1:s={w}x{h}:fps={fps}"


def _lut3d_arg(cube_path: str) -> str:
    """Escape a local cube path for ffmpeg's filter-argument syntax (colons
    and backslashes are filtergraph metacharacters)."""
    escaped = cube_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return f"lut3d=file='{escaped}':interp=trilinear"


def _filter_path_arg(path: str) -> str:
    """Same escaping `_lut3d_arg` uses, generalized -- colons/backslashes are
    filtergraph metacharacters regardless of which filter's `file=`/
    `fontsdir=` argument they're quoted into."""
    return path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")


def _write_ass_file(captions: List[Dict[str, Any]], cfg: Dict[str, Any], tmp: str) -> Optional[str]:
    """`resolved.captions` -> a local `.ass` file sized to this render's
    canvas, or None when there's nothing to burn (SS1.3 "no auto-apply" --
    an empty/absent captions track must add zero cost to a render, not even
    an extra ffmpeg pass)."""
    if not captions:
        return None
    ass_text = captions_to_ass(captions, canvas_w=cfg["width"], canvas_h=cfg["height"])
    path = os.path.join(tmp, "captions.ass")
    with open(path, "w", encoding="utf-8") as f:
        f.write(ass_text)
    return path


def _burn_captions(video_path: str, ass_path: str, cfg: Dict[str, Any]) -> str:
    """Burn `ass_path` onto `video_path` via libass, returning the new local
    path. A separate, isolated final pass (not folded into the segment-cache
    or filter_complex graph above) -- captions are a SEQUENCE-level overlay
    spanning cut boundaries, not a per-segment/per-layer effect, so they
    can't share those caches; keeping the burn as its own pass also means
    the grading/compositing paths above are untouched when captions are off
    (the common case pre-selection, SS1.3)."""
    out_path = video_path.replace(".mp4", ".cap.mp4")
    vf = f"ass='{_filter_path_arg(ass_path)}'"
    if os.path.isdir(CAPTION_FONTS_DIR) and os.listdir(CAPTION_FONTS_DIR):
        vf += f":fontsdir='{_filter_path_arg(CAPTION_FONTS_DIR)}'"
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-vf", vf,
        "-c:v", cfg["video_codec"], "-preset", cfg["video_preset"], "-crf", str(cfg["video_crf"]),
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", out_path,
    ]
    _run_ffmpeg(cmd, out_path, "caption burn")
    return out_path


def _transform_vf(
    cfg: Dict[str, Any],
    transform: Optional[Dict[str, Any]] = None,
    cube_path: Optional[str] = None,
    vignette_filter: Optional[str] = None,
    halation_filter: Optional[str] = None,
    grain_filter: Optional[str] = None,
) -> str:
    """Filter chain that frames a source onto the canvas for ONE video layer,
    in the canonical order rotate -> fit -> zoom-crop -> grade -> soft-local.
    An empty/None transform is the identity (contain, no rotate, no zoom) --
    byte-identical to the old normalize so untouched edits keep their warm
    segment cache; likewise `cube_path=None` (no grade) emits no color
    filter and `vignette_filter`/`halation_filter`/`grain_filter=None` (no
    soft-local, SS9) emits no spatial filter at all.

    The LUT is applied in 8-bit RGB (`format=gbrp`), matching the WebGL
    preview shader's own 8-bit texture sampling (browsers hand back 8-bit
    frames from `<video>`/canvas) -- both sides are already precision-capped
    by that 8-bit source, so matching bit depth here is what parity actually
    requires, not a corner cut. See color_grading.plan.md SS4/SS16.

    `vignette_filter` (from `grade.softlocal.vignette_ffmpeg_filter`) is
    applied AFTER the LUT (a vignette is a final polish over the graded
    picture) and is an approximate-parity effect by design -- see
    `softlocal.py`'s module docstring. `halation_filter`/`grain_filter`
    (halation_grain.plan.md, `grade.softlocal.halation_ffmpeg_subgraph`/
    `grain_ffmpeg_filter`) apply AFTER the vignette, in that order -- halation
    is a glow over the graded+vignetted picture, grain is the final texture
    on top of everything (real film grain is in the emulsion, i.e. last).
    Both pre-built (like `vignette_filter`) by the caller, which knows the
    frame height `halation_ffmpeg_subgraph` needs to scale its blur radius."""
    W, H, FPS = cfg["width"], cfg["height"], cfg["fps"]
    t = transform or {}
    rotate = int(t.get("rotate") or 0)
    fit = t.get("fit") or "contain"
    anchor = t.get("anchor") or "center"
    focus = t.get("focus") if isinstance(t.get("focus"), dict) else None
    motion = t.get("motion") if isinstance(t.get("motion"), dict) else None
    try:
        zoom = max(1.0, float(t.get("zoom") or 1.0))
    except (TypeError, ValueError):
        zoom = 1.0

    parts: List[str] = _transpose_chain(rotate)
    if motion:
        # Animated push-in / follow: fill a centered cover base (room to zoom +
        # pan), then let zoompan drive scale + focus. The motion path subsumes
        # any static zoom/focus, so they're not applied here.
        parts.append(f"scale=w={W}:h={H}:force_original_aspect_ratio=increase")
        parts.append(_crop_to(W, H, "center", None))
        parts.append(_zoompan(motion, W, H, FPS))
    else:
        if fit == "cover":
            parts.append(f"scale=w={W}:h={H}:force_original_aspect_ratio=increase")
            parts.append(_crop_to(W, H, anchor, focus))
        else:  # contain (letterbox) -- the historical default
            parts.append(f"scale=w={W}:h={H}:force_original_aspect_ratio=decrease")
            parts.append(f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=black")
        if zoom > 1.0001:
            parts.append(f"scale=w={_even(W * zoom)}:h={_even(H * zoom)}")
            parts.append(_crop_to(W, H, anchor, focus))
    if cube_path:
        parts.append("format=gbrp")
        parts.append(_lut3d_arg(cube_path))
        parts.append("format=yuv420p")
    if vignette_filter:
        parts.append(vignette_filter)
    if halation_filter:
        parts.append(halation_filter)
    if grain_filter:
        parts.append(grain_filter)
    parts.append("setsar=1")
    parts.append(f"fps={FPS}")
    return ",".join(parts)


def _vf_normalize(cfg: Dict[str, Any]) -> str:
    return _transform_vf(cfg, None)


def _dest_px(transform: Optional[Dict[str, Any]], W: int, H: int) -> Optional[Tuple[int, int, int, int]]:
    """A split/PiP cell's pixel rect (x, y, w, h) on the W x H canvas, or None for
    the full-frame default. Mirrors layers.is_rect + the normalized dest rect."""
    d = (transform or {}).get("dest")
    if not isinstance(d, dict):
        return None
    try:
        x, y = _even(W * float(d["x"])), _even(H * float(d["y"]))
        w, h = _even(W * float(d["w"])), _even(H * float(d["h"]))
    except (KeyError, TypeError, ValueError):
        return None
    return (x, y, w, h) if (w > 0 and h > 0) else None


def _produce_segment(
    *, src: str, dst: str, in_ms: int, out_ms: int, cfg: Dict[str, Any],
    transform: Optional[Dict[str, Any]] = None,
    cube_path: Optional[str] = None,
    vignette_filter: Optional[str] = None,
    halation_filter: Optional[str] = None,
    grain_filter: Optional[str] = None,
) -> None:
    in_s = max(in_ms / 1000.0, 0.0)
    dur_s = max((out_ms - in_ms) / 1000.0, 0.05)
    tmp_path = dst.replace(".mp4", ".part.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{in_s:.3f}", "-i", src, "-t", f"{dur_s:.3f}",
        "-vf", _transform_vf(cfg, transform, cube_path=cube_path, vignette_filter=vignette_filter,
                             halation_filter=halation_filter, grain_filter=grain_filter),
        "-c:v", cfg["video_codec"], "-preset", cfg["video_preset"], "-crf", str(cfg["video_crf"]),
        "-pix_fmt", "yuv420p",
        "-c:a", cfg["audio_codec"], "-b:a", cfg["audio_bitrate"], "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", "-avoid_negative_ts", "make_zero", "-fflags", "+genpts",
        tmp_path,
    ]
    _run_ffmpeg(cmd, tmp_path, f"cut {os.path.basename(src)} {in_ms}-{out_ms}ms")
    shutil.move(tmp_path, dst)


def _concat_segments(segments: List[str], dst: str, cfg: Dict[str, Any]) -> None:
    list_path = dst + ".list"
    with open(list_path, "w") as f:
        for s in segments:
            f.write(f"file '{s}'\n")
    try:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
               "-c", "copy", "-movflags", "+faststart", dst]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or not _ok(dst):
            logger.warning("concat copy failed; re-encoding merge.")
            cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                   "-c:v", cfg["video_codec"], "-preset", cfg["video_preset"], "-crf", str(cfg["video_crf"]),
                   "-pix_fmt", "yuv420p", "-c:a", cfg["audio_codec"], "-b:a", cfg["audio_bitrate"],
                   "-ar", "48000", "-ac", "2", "-movflags", "+faststart", dst]
            _run_ffmpeg(cmd, dst, "concat re-encode")
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


# --------------------------------------------------------------------------
# Layered path: one filter_complex graph (video layer stack + audio mix)
# --------------------------------------------------------------------------

def _render_layers(
    video: List[dict],
    audio: List[dict],
    total_ms: int,
    cfg: Dict[str, Any],
    file_lookup: Dict[str, FileEntry],
    tmp: str,
    report: Callable[[int, str], None],
    ass_path: Optional[str] = None,
) -> str:
    report(6, "preparing sources")
    src_cache: Dict[str, str] = {}
    total_s = total_ms / 1000.0
    W, H, FPS = cfg["width"], cfg["height"], cfg["fps"]

    inputs: List[str] = []          # ffmpeg -i paths, one per layer
    filt: List[str] = []
    cube_dir = os.path.join(tmp, "cubes")
    # Black base canvas for the full program duration.
    filt.append(f"color=c=black:s={W}x{H}:r={FPS}:d={total_s:.3f},format=yuv420p[base]")

    # --- video layers: trim -> normalize -> grade -> alpha -> shift to program start ---
    cur = "[base]"
    for i, v in enumerate(video):
        entry = file_lookup.get(v["source_file_id"])
        if entry is None:
            raise RuntimeError(f"no FileEntry for {v['source_file_id']}")
        idx = len(inputs)
        inputs.append(_source_path(entry, cfg, tmp, src_cache))
        in_s = int(v["src_in_ms"]) / 1000.0
        out_s = int(v["src_out_ms"]) / 1000.0
        ps = int(v["prog_start_ms"]) / 1000.0
        opacity = float(v.get("opacity", 1.0))
        # A split/PiP cell frames its source to the CELL size and composites at
        # the cell origin; a full-frame layer frames to the whole canvas at (0,0).
        # (ffmpeg's `overlay` filter below is just the compositing primitive.)
        cell = _dest_px(v.get("transform"), W, H)
        vf_cfg = {**cfg, "width": cell[2], "height": cell[3]} if cell else cfg
        ov_xy = f"x={cell[0]}:y={cell[1]}:" if cell else ""
        v_grade = v.get("grade") or {}
        cube_path = ensure_cube_file(v_grade, cube_dir)
        v_soft_local = v_grade.get("soft_local") or {}
        vignette_filter = vignette_ffmpeg_filter(v_soft_local.get("vignette"))
        halation_filter = halation_ffmpeg_subgraph(v_soft_local.get("halation"), frame_height=vf_cfg["height"])
        grain_filter = grain_ffmpeg_filter(v_soft_local.get("grain"))
        layer_vf = _transform_vf(
            vf_cfg, v.get("transform"), cube_path=cube_path, vignette_filter=vignette_filter,
            halation_filter=halation_filter, grain_filter=grain_filter,
        )
        chain = (
            f"[{idx}:v]trim=start={in_s:.3f}:end={out_s:.3f},setpts=PTS-STARTPTS,"
            f"{layer_vf},format=yuva420p"
        )
        if opacity < 0.999:
            chain += f",colorchannelmixer=aa={max(0.0, min(1.0, opacity)):.3f}"
        # Shift the layer's timeline so it composites at its program start.
        chain += f",setpts=PTS+{ps:.3f}/TB[v{i}]"
        filt.append(chain)
        out = f"[vt{i}]"
        filt.append(f"{cur}[v{i}]overlay={ov_xy}eof_action=pass:format=auto{out}")
        cur = out
    video_out = cur

    # --- audio layers: trim -> delay to program start -> gain -> mix ---
    audio_labels: List[str] = []
    for j, a in enumerate(audio):
        entry = file_lookup.get(a["source_file_id"])
        if entry is None:
            raise RuntimeError(f"no FileEntry for {a['source_file_id']}")
        idx = len(inputs)
        inputs.append(_source_path(entry, cfg, tmp, src_cache))
        in_s = int(a["src_in_ms"]) / 1000.0
        out_s = int(a["src_out_ms"]) / 1000.0
        delay_ms = max(0, int(a["prog_start_ms"]))
        gain_db = float(a.get("gain_db", 0.0)) + float(a.get("duck_db", 0.0))
        chain = (
            f"[{idx}:a]atrim=start={in_s:.3f}:end={out_s:.3f},asetpts=PTS-STARTPTS,"
            f"aresample=48000,aformat=channel_layouts=stereo,"
            f"adelay={delay_ms}|{delay_ms}"
        )
        if gain_db <= -119:
            chain += ",volume=0"
        elif abs(gain_db) > 0.01:
            chain += f",volume={gain_db:.2f}dB"
        # Fade envelope (audio_brain.plan.md `fade_audio`/`crossfade`): hard
        # start/stop unless the brain set an edge. `adelay` above already put
        # this layer's PTS on the ABSOLUTE program timeline, so `afade`'s `st`
        # is just this layer's own prog_start/prog_end in seconds -- no extra
        # bookkeeping needed to line it up with the rest of the mix.
        fade_in_ms = int(a.get("fade_in_ms", 0) or 0)
        fade_out_ms = int(a.get("fade_out_ms", 0) or 0)
        if fade_in_ms > 0:
            chain += f",afade=t=in:st={delay_ms / 1000.0:.3f}:d={fade_in_ms / 1000.0:.3f}"
        if fade_out_ms > 0:
            prog_end_ms = int(a.get("prog_end_ms", 0))
            st_out = max(0, prog_end_ms - fade_out_ms) / 1000.0
            chain += f",afade=t=out:st={st_out:.3f}:d={fade_out_ms / 1000.0:.3f}"
        chain += f"[a{j}]"
        filt.append(chain)
        audio_labels.append(f"[a{j}]")

    has_audio = bool(audio_labels)
    if has_audio:
        if len(audio_labels) == 1:
            # Single bus: still clamp to the program duration.
            filt.append(f"{audio_labels[0]}apad,atrim=0:{total_s:.3f}[aout]")
        else:
            filt.append(
                "".join(audio_labels)
                + f"amix=inputs={len(audio_labels)}:normalize=0:dropout_transition=0,"
                + f"atrim=0:{total_s:.3f}[aout]"
            )

    # Trim the composited video to the exact program duration.
    filt.append(f"{video_out}trim=0:{total_s:.3f},setpts=PTS-STARTPTS[vout]")

    report(20, "compositing")
    out_local = os.path.join(tmp, "out.mp4")
    cmd: List[str] = ["ffmpeg", "-y"]
    for p in inputs:
        cmd += ["-i", p]
    cmd += ["-filter_complex", ";".join(filt), "-map", "[vout]"]
    if has_audio:
        cmd += ["-map", "[aout]"]
    cmd += [
        "-c:v", cfg["video_codec"], "-preset", cfg["video_preset"], "-crf", str(cfg["video_crf"]),
        "-pix_fmt", "yuv420p",
    ]
    if has_audio:
        cmd += ["-c:a", cfg["audio_codec"], "-b:a", cfg["audio_bitrate"], "-ar", "48000", "-ac", "2"]
    cmd += ["-movflags", "+faststart", "-t", f"{total_s:.3f}", out_local]

    _run_ffmpeg(cmd, out_local, "filter_complex composite")
    if ass_path:
        report(88, "burning captions")
        out_local = _burn_captions(out_local, ass_path, cfg)
    report(92, "uploading")
    out_key = f"{RENDER_PREFIX}/{uuid.uuid4().hex}.mp4"
    _upload_to_r2(out_local, out_key, "video/mp4")
    return out_key


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _ok(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _run_ffmpeg(cmd: List[str], out_path: str, what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not _ok(out_path):
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except OSError:
            pass
        tail = proc.stderr.decode("utf-8", errors="ignore")[-1200:]
        raise RuntimeError(f"ffmpeg failed ({what}):\n{tail}")


def _pct(i: int, n: int, lo: int, hi: int) -> int:
    if n <= 0:
        return lo
    return int(lo + (hi - lo) * (i / n))
