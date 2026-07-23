"""
export_options.plan.md Phase 3 (+ Phase 6 coordinate mapping): FCPXML v1.9
rough-cut exporter.

Builds an FCPXML document from the SAME resolved timeline the MP4 render and
SRT sidecar read (render.tasks.resolve_document's output): cuts (program
order), framing/reframe (per-clip <adjust-transform>/<adjust-crop>), and
static split-screen/PiP (connected clips positioned into their `dest` cell).
Subtitles are NOT emitted here -- burned-in captions live only in the MP4;
editable captions are the SRT sidecar the caller attaches separately (see
srt.py, bundle.py). No animated split-screen/motion is emitted -- only
static transforms, per export_options.plan.md's own guardrails ("No
animated split-screen -- cells are static; emit fixed transforms only,
never keyframes"). A layer with an animated `motion` path (Ken-Burns
push/pull, itself out of this plan's stated scope) is represented by its
`motion.from` endpoint's static zoom/focus -- a deliberate simplification,
not a bug.

COORDINATE MAPPING (Phase 6): the position/scale math below (`_cell_transform`)
is a carefully-reasoned, INTERNALLY CONSISTENT convention -- see its own
docstring -- but has NOT been round-tripped through a real NLE (DaVinci
Resolve / Premiere Pro) as of this writing. Phase 6 calls for exactly that
"golden-import validation" before this deliverable is trustworthy for
split-screen/PiP; a human with those apps still needs to do it. The
regression test (test_export_fcpxml.py) covers the MATH's own internal
consistency, not real-NLE agreement -- treat this exporter's split-screen/
PiP output as unverified until that manual pass happens.

J/L cuts: represented via each clip's `audioStart`/`audioDuration` (FCPXML's
native attribute pair for an audio in-point/duration independent of the
video's `start`/`duration`). A shift big enough that the audio's PROGRAM-TIME
window extends past its own clip's boundary into a neighbor's (the "lingers
over the next clip's picture" case `layers._apply_split_edits` describes) is
a known simplification this exporter does not fully represent -- also a
Phase 6 follow-up, not a silent correctness bug: it degrades to the audio's
in/out point shifting, without the cross-clip program-time bleed.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from xml.dom import minidom

from app.services.render.compositor import canvas_dims

# Fixed project frame rate: every compositor.PRESETS entry declares fps=30
# and there is no per-project/per-file frame rate stored anywhere yet (see
# export_options.plan.md Phase 3's own "derive one project fps" note) -- 30
# is the one fps this codebase already commits to, so every offset/duration
# snaps to a 1/30s frame grid.
FPS = 30

# The FCPXML sequence's own nominal canvas resolution -- independent of
# whatever render quality preset the user picks for included media (FCPXML
# references ORIGINAL source files, not a baked frame, so this is just the
# project's declared working resolution). 1920 matches the "1080"/"export"
# render tier, the most common delivery size.
_CANONICAL_LONG_EDGE = 1920


def _frames(ms: int) -> int:
    """ms snapped to the nearest 1/FPS-second frame boundary, as an integer
    frame count (never negative)."""
    return max(0, round(ms * FPS / 1000.0))


def _time(ms: int) -> str:
    """FCPXML rational time: an integer frame count over FPS, e.g. "90/30s"
    -- always frame-exact by construction (see `_frames`)."""
    return f"{_frames(ms)}/{FPS}s"


def _frame_duration_str() -> str:
    return f"1/{FPS}s"


@dataclass
class FcpxmlAsset:
    """One source file, as the FCPXML exporter needs to know it -- decoupled
    from `render.compositor.FileEntry` (which is R2-key-shaped) since FCPXML
    needs the human filename for the ZIP's relative `media/<filename>` path,
    never an R2 key or a presigned URL (export_options.plan.md's "relative
    media paths" guardrail)."""
    file_id: str
    filename: str
    duration_ms: int = 0
    width: Optional[int] = None
    height: Optional[int] = None


# --------------------------------------------------------------------------
# Phase 6: coordinate mapping (framing + split-screen cells -> FCPXML
# <adjust-transform>/<adjust-crop>)
# --------------------------------------------------------------------------

def _cell_transform(
    transform: Optional[Dict[str, Any]],
    canvas_w: int, canvas_h: int,
    native_w: Optional[int], native_h: Optional[int],
) -> Tuple[float, float, float, float, Optional[Tuple[float, float, float, float]]]:
    """(pos_x, pos_y, scale_x, scale_y, crop_fractions) for one layer's
    `transform` on a `canvas_w` x `canvas_h` sequence.

    CONVENTION (locked here per export_options.plan.md Phase 6; UNVALIDATED
    against a real NLE import -- see module docstring):

      * FCP places a clip on an EMPTY <adjust-transform> already auto-fit
        (letterboxed) to the FULL sequence frame, aspect preserved -- that
        auto-fit IS what `scale="1 1" position="0 0"` means. Our own
        `zoom`/`dest`/`fit` describe an INTENSE beyond that baseline, so the
        math below computes the ADDITIONAL scale needed on top of it, never
        an absolute pixel scale.
      * `position` is in CANVAS PIXELS, origin at the frame CENTER, +X right,
        +Y UP (screen-space Y is inverted from our own top-left-origin,
        Y-down `dest` rects -- flipped explicitly below).
      * `scale` is applied UNIFORMLY (same value both axes) to preserve the
        source's own aspect ratio, matching how a real camera clip is
        reframed in practice; `fit="cover"` scales to FILL the dest cell
        (crop the overflow via `<adjust-crop>`), `fit="contain"` scales to
        FIT inside it (letterbox, no crop needed).
      * When the source's native pixel size is unknown (`native_w`/`h` is
        None -- e.g. a pre-migration file row with no stored dimensions),
        this falls back to assuming the source's aspect ratio already
        matches the canvas, which is exact for a full-frame layer and an
        approximation for a cropped one -- the only degradation is losing
        crop precision, never a wrong POSITION.

    Returns `crop_fractions` as `(left, right, top, bottom)`, each a 0..1
    fraction of the SCALED clip's own dimension to trim from that edge (None
    when nothing needs trimming, i.e. `fit="contain"` or an exact/undersized
    fit -- see `_apply_layout_regions`/`_slice_video` for why a real
    split-screen cell always forces `fit="cover"`)."""
    t = transform or {}
    dest = t.get("dest")
    if isinstance(dest, dict) and all(k in dest for k in ("x", "y", "w", "h")):
        dx, dy, dw, dh = float(dest["x"]), float(dest["y"]), float(dest["w"]), float(dest["h"])
    else:
        dx, dy, dw, dh = 0.0, 0.0, 1.0, 1.0
    fit = t.get("fit") or "contain"
    try:
        zoom = max(1.0, float(t.get("zoom") or 1.0))
    except (TypeError, ValueError):
        zoom = 1.0

    cell_w_px = dw * canvas_w
    cell_h_px = dh * canvas_h

    nw = float(native_w) if native_w else float(canvas_w)
    nh = float(native_h) if native_h else float(canvas_h)
    # FCP's own baseline auto-fit: the source letterboxed into the FULL
    # sequence frame (this is what scale=1 already gives us for free).
    baseline_scale = min(canvas_w / nw, canvas_h / nh)

    # What we actually want displayed: the source fit/cover-scaled into the
    # DEST CELL (not the full canvas), then the extra `zoom` beyond that.
    if fit == "cover":
        cell_scale = max(cell_w_px / nw, cell_h_px / nh)
    else:
        cell_scale = min(cell_w_px / nw, cell_h_px / nh)
    cell_scale *= zoom

    scale = cell_scale / baseline_scale if baseline_scale > 0 else 1.0

    # Cell center in canvas pixels (top-left-origin, Y-down) -> offset from
    # canvas center, Y-flipped to FCP's center-origin, Y-up convention.
    cx_px = (dx + dw / 2.0) * canvas_w
    cy_px = (dy + dh / 2.0) * canvas_h
    pos_x = cx_px - canvas_w / 2.0
    pos_y = -(cy_px - canvas_h / 2.0)

    crop: Optional[Tuple[float, float, float, float]] = None
    if fit == "cover":
        displayed_w = nw * cell_scale
        displayed_h = nh * cell_scale
        excess_w_frac = max(0.0, (displayed_w - cell_w_px) / displayed_w / 2.0) if displayed_w > 0 else 0.0
        excess_h_frac = max(0.0, (displayed_h - cell_h_px) / displayed_h / 2.0) if displayed_h > 0 else 0.0
        if excess_w_frac > 1e-6 or excess_h_frac > 1e-6:
            anchor = t.get("anchor") or "center"
            left = right = excess_w_frac
            top = bottom = excess_h_frac
            if anchor == "left":
                left, right = 0.0, excess_w_frac * 2.0
            elif anchor == "right":
                left, right = excess_w_frac * 2.0, 0.0
            elif anchor == "top":
                top, bottom = 0.0, excess_h_frac * 2.0
            elif anchor == "bottom":
                top, bottom = excess_h_frac * 2.0, 0.0
            crop = (left, right, top, bottom)

    return pos_x, pos_y, scale, scale, crop


def _add_transform_and_crop(
    parent: ET.Element, transform: Optional[Dict[str, Any]],
    canvas_w: int, canvas_h: int, native_w: Optional[int], native_h: Optional[int],
) -> None:
    """Append <adjust-transform>/<adjust-crop> children to `parent` (an
    <asset-clip>/<clip> element) for one layer's `transform`. Identity
    (full-frame, no zoom, no rotate) emits NOTHING -- an untouched clip
    should round-trip with no adjust elements at all, matching how FCP
    itself omits them for a plain cut."""
    t = transform or {}
    rotate = int(t.get("rotate") or 0)
    pos_x, pos_y, scale_x, scale_y, crop = _cell_transform(t, canvas_w, canvas_h, native_w, native_h)
    is_identity = (
        rotate == 0 and abs(pos_x) < 1e-6 and abs(pos_y) < 1e-6
        and abs(scale_x - 1.0) < 1e-6 and abs(scale_y - 1.0) < 1e-6
    )
    if not is_identity:
        adjust = ET.SubElement(parent, "adjust-transform")
        adjust.set("position", f"{pos_x:.3f} {pos_y:.3f}")
        adjust.set("scale", f"{scale_x:.4f} {scale_y:.4f}")
        if rotate:
            # FCP rotation is degrees, clockwise-positive by convention;
            # our own rotate is already CW degrees (see layers.ROTATIONS /
            # compositor._transpose_chain) -- direct passthrough.
            adjust.set("rotation", str(rotate))
    if crop is not None:
        left, right, top, bottom = crop
        adjust_crop = ET.SubElement(parent, "adjust-crop")
        adjust_crop.set("mode", "trim")
        trim_rect = ET.SubElement(adjust_crop, "trim-rect")
        trim_rect.set("left", f"{left:.4f}")
        trim_rect.set("right", f"{right:.4f}")
        trim_rect.set("top", f"{top:.4f}")
        trim_rect.set("bottom", f"{bottom:.4f}")


# --------------------------------------------------------------------------
# Document assembly
# --------------------------------------------------------------------------

def _collect_assets(resolved: Dict[str, Any]) -> List[str]:
    """Distinct source_file_ids referenced by the resolved timeline, in
    first-use (program) order -- stable, deterministic resource ids."""
    seen: List[str] = []
    seen_set = set()
    layers = sorted(
        (resolved.get("video_layers") or []) + (resolved.get("audio_layers") or []),
        key=lambda x: x.get("prog_start_ms", 0),
    )
    for layer in layers:
        fid = layer.get("source_file_id")
        if fid and fid not in seen_set:
            seen_set.add(fid)
            seen.append(fid)
    return seen


def build_fcpxml(
    resolved: Dict[str, Any],
    file_lookup: Dict[str, FcpxmlAsset],
    *,
    project_name: str = "Untitled",
    media_dir: str = "media",
) -> str:
    """`resolved` (render.tasks.resolve_document's output) -> an FCPXML v1.9
    document string. `file_lookup` maps `source_file_id` -> `FcpxmlAsset`
    (filename + optional native dimensions); a referenced file missing from
    `file_lookup` is skipped (its layers are dropped from the sequence
    rather than failing the whole export -- a rough cut with one missing
    source is still useful, an empty download is not)."""
    aspect = str(resolved.get("aspect") or "landscape")
    canvas_w, canvas_h = canvas_dims(_CANONICAL_LONG_EDGE, aspect)
    total_ms = int(resolved.get("duration_ms") or 0)

    video = sorted(
        (v for v in (resolved.get("video_layers") or []) if v.get("source_file_id") in file_lookup),
        key=lambda v: (int(v["prog_start_ms"]), int(v.get("z", 0))),
    )
    audio = sorted(
        (a for a in (resolved.get("audio_layers") or []) if a.get("source_file_id") in file_lookup),
        key=lambda a: int(a["prog_start_ms"]),
    )
    if total_ms <= 0:
        total_ms = max([v["prog_end_ms"] for v in video] + [a["prog_end_ms"] for a in audio] + [0])

    asset_ids = _collect_assets(resolved)
    asset_ids = [fid for fid in asset_ids if fid in file_lookup]
    resource_id = {fid: f"r{i + 2}" for i, fid in enumerate(asset_ids)}  # r1 is the format

    fcpxml = ET.Element("fcpxml", version="1.9")
    resources = ET.SubElement(fcpxml, "resources")
    fmt = ET.SubElement(resources, "format")
    fmt.set("id", "r1")
    fmt.set("name", f"FFVideoFormat{canvas_h}p{FPS}")
    fmt.set("frameDuration", _frame_duration_str())
    fmt.set("width", str(canvas_w))
    fmt.set("height", str(canvas_h))

    for fid in asset_ids:
        entry = file_lookup[fid]
        asset = ET.SubElement(resources, "asset")
        asset.set("id", resource_id[fid])
        asset.set("name", entry.filename)
        asset.set("src", f"{media_dir}/{entry.filename}")
        asset.set("start", "0s")
        # Asset duration is the SOURCE file's own full length so any src_in/
        # src_out window we reference is always inside bounds; fall back to
        # a generous ceiling when unknown rather than under-declaring it
        # (an under-declared asset duration would make a valid trim look
        # out-of-range to the importing NLE).
        asset.set("duration", _time(max(entry.duration_ms, total_ms, 1)))
        asset.set("hasVideo", "1")
        asset.set("hasAudio", "1")
        asset.set("format", "r1")

    library = ET.SubElement(fcpxml, "library")
    event = ET.SubElement(library, "event")
    event.set("name", project_name)
    project = ET.SubElement(event, "project")
    project.set("name", project_name)
    sequence = ET.SubElement(project, "sequence")
    sequence.set("format", "r1")
    sequence.set("duration", _time(total_ms))
    sequence.set("tcStart", "0s")
    spine = ET.SubElement(sequence, "spine")

    # --- primary storyline: spine video layers, program order ---
    spine_clips: List[Tuple[dict, ET.Element]] = []
    for v in video:
        if v.get("kind") != "spine":
            continue
        entry = file_lookup[v["source_file_id"]]
        clip = ET.SubElement(spine, "asset-clip")
        clip.set("ref", resource_id[v["source_file_id"]])
        clip.set("name", entry.filename)
        clip.set("offset", _time(int(v["prog_start_ms"])))
        clip.set("start", _time(int(v["src_in_ms"])))
        clip.set("duration", _time(int(v["src_out_ms"]) - int(v["src_in_ms"])))
        _add_transform_and_crop(clip, v.get("transform"), canvas_w, canvas_h, entry.width, entry.height)
        spine_clips.append((v, clip))

    def _anchor_for(prog_start_ms: int) -> ET.Element:
        """The spine clip whose program window contains (or is closest
        before) `prog_start_ms` -- connected clips must nest inside SOME
        primary-storyline element; this picks the natural one."""
        best: Optional[ET.Element] = None
        for v, el in spine_clips:
            if int(v["prog_start_ms"]) <= prog_start_ms:
                best = el
            else:
                break
        return best if best is not None else (spine_clips[0][1] if spine_clips else spine)

    # --- static split-screen / PiP cells: non-spine video layers, as
    # connected clips (own lane, above the base) -----------------------
    lane = 1
    for v in video:
        if v.get("kind") == "spine":
            continue
        entry = file_lookup[v["source_file_id"]]
        anchor = _anchor_for(int(v["prog_start_ms"]))
        clip = ET.SubElement(anchor, "asset-clip")
        clip.set("ref", resource_id[v["source_file_id"]])
        clip.set("name", entry.filename)
        clip.set("lane", str(lane))
        clip.set("offset", _time(int(v["prog_start_ms"])))
        clip.set("start", _time(int(v["src_in_ms"])))
        clip.set("duration", _time(int(v["src_out_ms"]) - int(v["src_in_ms"])))
        _add_transform_and_crop(clip, v.get("transform"), canvas_w, canvas_h, entry.width, entry.height)
        lane += 1

    # --- audio layers: independent connected clips (own negative lane),
    # each on its OWN resolved program timing -- this is what makes a J/L
    # cut (audio's prog_start/prog_end diverging from any picture clip's)
    # come through correctly, without relying on a single clip's
    # audioStart/audioDuration to span a neighboring clip's window (see
    # module docstring's J/L caveat). ------------------------------------
    audio_lane = -1
    for a in audio:
        entry = file_lookup[a["source_file_id"]]
        anchor = _anchor_for(int(a["prog_start_ms"]))
        clip = ET.SubElement(anchor, "asset-clip")
        clip.set("ref", resource_id[a["source_file_id"]])
        clip.set("name", f"{entry.filename} (audio)")
        clip.set("lane", str(audio_lane))
        clip.set("offset", _time(int(a["prog_start_ms"])))
        clip.set("start", _time(int(a["src_in_ms"])))
        clip.set("duration", _time(int(a["src_out_ms"]) - int(a["src_in_ms"])))
        gain_db = float(a.get("gain_db", 0.0)) + float(a.get("duck_db", 0.0))
        if abs(gain_db) > 0.01:
            adjust_volume = ET.SubElement(clip, "adjust-volume")
            adjust_volume.set("amount", f"{gain_db:.2f}dB")
        audio_lane -= 1

    xml_bytes = ET.tostring(fcpxml, encoding="utf-8")
    pretty = minidom.parseString(xml_bytes).toprettyxml(indent="    ", encoding="utf-8")
    header = b'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE fcpxml>\n\n'
    body = pretty.split(b"\n", 1)[1]  # drop minidom's own xml decl line, we write our own above
    return (header + body).decode("utf-8")
