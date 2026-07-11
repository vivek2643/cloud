"""
Export bundle (color_grading.plan.md SS11): the professional round-trip
path out of EDSO's grade. Three related artifacts, all keyed off the SAME
resolved `Grade` values this whole system already computes -- nothing here
invents new color math, it only reformats existing numbers:

  - `.cdl` (single grade) / `.ccc` (a named collection, one per clip) --
    the ASC CDL XML interchange format every color pipeline (Resolve, Avid,
    Premiere, Baselight) reads natively.
  - CMX3600 EDL `*ASC_SOP`/`*ASC_SAT` comment lines -- the standard
    convention for carrying CDL values inline in an EDL, so a colorist
    re-linking footage in their own NLE gets the grade for free.
  - `.cube`: already built (`lut_bake.bake_cube_text`); this module doesn't
    duplicate it, just documents it as the third artifact in the bundle.

Working-space is stamped on every artifact's `<Description>` (the plan's
own "#1 CDL round-trip bug" to avoid -- an implicit color-space assumption
silently breaking a round-trip between tools).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, List

from app.services.l3.grade.cdl import Grade

_CDL_NS = "urn:ASC:CDL:v1.2"


def _sop_sat_elements(parent: ET.Element, grade: Grade) -> None:
    sop = ET.SubElement(parent, "SOPNode")
    ET.SubElement(sop, "Slope").text = " ".join(f"{v:.6f}" for v in grade.slope)
    ET.SubElement(sop, "Offset").text = " ".join(f"{v:.6f}" for v in grade.offset)
    ET.SubElement(sop, "Power").text = " ".join(f"{v:.6f}" for v in grade.power)
    sat = ET.SubElement(parent, "SatNode")
    ET.SubElement(sat, "Saturation").text = f"{grade.sat:.6f}"


def cdl_xml(grade: Grade, *, working_space: str, cc_id: str = "cc001") -> str:
    """Single-grade ASC CDL XML (`.cdl`)."""
    root = ET.Element("ColorDecisionList", {"xmlns": _CDL_NS})
    cc = ET.SubElement(root, "ColorCorrection", id=cc_id)
    _sop_sat_elements(cc, grade)
    ET.SubElement(cc, "Description").text = f"working_space={working_space}"
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def ccc_xml(entries: List[Dict[str, Any]], *, working_space: str) -> str:
    """Multi-grade Color Correction Collection (`.ccc`) -- one
    `ColorCorrection` per clip, id-addressable so a colorist can re-link by
    id in their NLE. `entries`: `[{"id": str, "cdl": {...}}, ...]`."""
    root = ET.Element("ColorCorrectionCollection", {"xmlns": _CDL_NS})
    ET.SubElement(root, "Description").text = f"working_space={working_space}"
    for e in entries:
        grade = Grade.from_dict(e.get("cdl"))
        cc = ET.SubElement(root, "ColorCorrection", id=str(e["id"]))
        _sop_sat_elements(cc, grade)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")


def edl_asc_comment_lines(grade: Grade) -> List[str]:
    """`*ASC_SOP`/`*ASC_SAT` comment lines -- the standard CMX3600 EDL
    convention for carrying CDL values inline (Resolve/Avid/Premiere all
    read these on import)."""
    slope = " ".join(f"{v:.6f}" for v in grade.slope)
    offset = " ".join(f"{v:.6f}" for v in grade.offset)
    power = " ".join(f"{v:.6f}" for v in grade.power)
    return [
        f"*ASC_SOP ({slope})({offset})({power})",
        f"*ASC_SAT {grade.sat:.6f}",
    ]


def edl_bundle(
    entries: List[Dict[str, Any]], *, title: str = "EDSO_GRADE_EXPORT"
) -> str:
    """A minimal CMX3600 EDL whose sole purpose is carrying one ASC_SOP/
    ASC_SAT comment pair per clip -- NOT a full source-timecode EDL (this
    system's source-of-truth is the JSON edit document, not EDL timecode
    math). Framed as a grade-only reference list a colorist imports
    alongside their own conform, one event per graded clip in document
    order. `entries`: `[{"id": str, "cdl": {...}}, ...]`."""
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    for i, e in enumerate(entries, start=1):
        grade = Grade.from_dict(e.get("cdl"))
        event = f"{i:03d}"
        clip_id = str(e["id"])
        lines.append(f"{event}  {clip_id[:8]:<8} V     C        "
                     f"00:00:00:00 00:00:00:00 00:00:00:00 00:00:00:00")
        lines.append(f"* FROM CLIP NAME: {clip_id}")
        lines.extend(edl_asc_comment_lines(grade))
        lines.append("")
    return "\n".join(lines)
