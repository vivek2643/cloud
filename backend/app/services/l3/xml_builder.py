"""
Compile a timeline into a Final Cut Pro 7 / Apple XML (.xml) blueprint.

Output is import-compatible with Adobe Premiere Pro and DaVinci Resolve.
Source media is referenced by pathurl. For our cloud-hosted media, we pass
either presigned R2 GET URLs (works for DaVinci which can fetch HTTPS) or
a `file://` URL pointing at a local download path (preferred for Premiere).
"""
from __future__ import annotations

from typing import Callable, List, Optional
from xml.dom import minidom
import xml.etree.ElementTree as ET

from app.services.l3.edit_logic_basic import TimelineClip

DEFAULT_FPS = 24


def _add_rate(parent: ET.Element, fps: int) -> None:
    rate = ET.SubElement(parent, "rate")
    ET.SubElement(rate, "timebase").text = str(fps)
    ET.SubElement(rate, "ntsc").text = "FALSE"


def build_fcp7_xml(
    sequence_name: str,
    clips: List[TimelineClip],
    pathurl_for: Optional[Callable[[TimelineClip], str]] = None,
    fps: int = DEFAULT_FPS,
) -> str:
    """
    Compile a list of TimelineClip into a pretty-printed FCP7 XML string.

    pathurl_for(clip) -> str returns the pathurl to embed for that clip's
    source media. If omitted, we use file://{file_r2_key}.
    """
    root = ET.Element("xmeml", version="4")
    seq = ET.SubElement(root, "sequence", id=sequence_name)
    ET.SubElement(seq, "name").text = sequence_name
    _add_rate(seq, fps)

    # Sequence duration in frames
    total_frames = 0
    if clips:
        total_frames = max(
            int((c.timeline_end_ms / 1000.0) * fps) for c in clips
        )
    ET.SubElement(seq, "duration").text = str(total_frames)
    ET.SubElement(seq, "in").text = "-1"
    ET.SubElement(seq, "out").text = "-1"
    ET.SubElement(seq, "timecode").append(_timecode(fps))

    media = ET.SubElement(seq, "media")
    video = ET.SubElement(media, "video")
    v_format = ET.SubElement(video, "format")
    v_format.append(_sample_characteristics(1920, 1080, fps))
    v_track = ET.SubElement(video, "track")
    audio = ET.SubElement(media, "audio")
    a_track = ET.SubElement(audio, "track")

    seen_file_ids: dict[str, str] = {}

    for idx, clip in enumerate(clips):
        in_frame = int((clip.source_in_ms / 1000.0) * fps)
        out_frame = int((clip.source_out_ms / 1000.0) * fps)
        start_frame = int((clip.timeline_start_ms / 1000.0) * fps)
        end_frame = int((clip.timeline_end_ms / 1000.0) * fps)
        clip_id = f"clip_{idx}"
        file_id_xml = f"file_{clip.file_id}"

        # Video clipitem
        vci = ET.SubElement(v_track, "clipitem", id=clip_id)
        ET.SubElement(vci, "name").text = clip.file_name
        _add_rate(vci, fps)
        ET.SubElement(vci, "in").text = str(in_frame)
        ET.SubElement(vci, "out").text = str(out_frame)
        ET.SubElement(vci, "start").text = str(start_frame)
        ET.SubElement(vci, "end").text = str(end_frame)

        is_first_ref = clip.file_id not in seen_file_ids
        seen_file_ids[clip.file_id] = file_id_xml
        if is_first_ref:
            file_node = ET.SubElement(vci, "file", id=file_id_xml)
            ET.SubElement(file_node, "name").text = clip.file_name
            ET.SubElement(file_node, "pathurl").text = (
                pathurl_for(clip) if pathurl_for else f"file://{clip.file_r2_key}"
            )
            _add_rate(file_node, fps)
            media_node = ET.SubElement(file_node, "media")
            ET.SubElement(ET.SubElement(media_node, "video"), "duration").text = str(
                int((clip.source_out_ms / 1000.0) * fps)
            )
        else:
            ET.SubElement(vci, "file", id=file_id_xml)

        # Audio clipitem on its own track, referencing the same file
        aci = ET.SubElement(a_track, "clipitem", id=f"audio_{idx}")
        ET.SubElement(aci, "name").text = clip.file_name
        _add_rate(aci, fps)
        ET.SubElement(aci, "in").text = str(in_frame)
        ET.SubElement(aci, "out").text = str(out_frame)
        ET.SubElement(aci, "start").text = str(start_frame)
        ET.SubElement(aci, "end").text = str(end_frame)
        ET.SubElement(aci, "file", id=file_id_xml)

    return _pretty(root)


def _timecode(fps: int) -> ET.Element:
    tc = ET.Element("timecode")
    _add_rate(tc, fps)
    ET.SubElement(tc, "string").text = "00:00:00:00"
    ET.SubElement(tc, "frame").text = "0"
    ET.SubElement(tc, "displayformat").text = "NDF"
    return tc


def _sample_characteristics(width: int, height: int, fps: int) -> ET.Element:
    sc = ET.Element("samplecharacteristics")
    _add_rate(sc, fps)
    ET.SubElement(sc, "width").text = str(width)
    ET.SubElement(sc, "height").text = str(height)
    ET.SubElement(sc, "pixelaspectratio").text = "square"
    return sc


def _pretty(root: ET.Element) -> str:
    raw = ET.tostring(root, encoding="utf-8")
    return minidom.parseString(raw).toprettyxml(indent="  ", encoding="UTF-8").decode("utf-8")
