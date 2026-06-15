"""
Angle menu for synced multicam.

For a detected SYNC GROUP, this assembles -- deterministically, with NO VLM and
NO cross-clip identity -- a per-window table of: who holds the floor (the spine
clip's diarization, mapped to program time) and, for each angle, what that angle
OFFERS at that moment (is the speaker visibly on camera, their shot size, any
active reaction, gaze to camera).

The "is the speaker on camera" test needs no identity matching: each angle's own
VLM `speaking` spans, mapped through the group's sync offset, tell us whether
someone is visibly speaking on that angle while the floor is held -- which, by
time alignment, is the floor speaker. Identity matching (the roster) is only for
Opus's higher-level reasoning, not for this menu.

Pure functions over already-loaded dicts; the only side effect is the optional
`build_menu` convenience that loads perceptions + turns from the DB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.services.l3.layers import SpineSpan
from app.services.l3.sync import SyncGroup

Turn = Tuple[int, int, str]


@dataclass
class AngleOffer:
    file_id: str
    speaker_in_frame: bool          # someone is visibly speaking on this angle now
    shot_size: Optional[str]        # nearest camera_craft shot_size
    reaction: Optional[Tuple[str, float]]  # (type, intensity) of the strongest active reaction
    gaze_to_camera: bool

    def to_dict(self) -> dict:
        return {
            "file_id": self.file_id,
            "speaker_in_frame": self.speaker_in_frame,
            "shot_size": self.shot_size,
            "reaction": {"type": self.reaction[0], "intensity": self.reaction[1]} if self.reaction else None,
            "gaze_to_camera": self.gaze_to_camera,
        }


@dataclass
class AngleWindow:
    prog_start_ms: int
    prog_end_ms: int
    floor_speaker: Optional[str]    # clip-local diarization label of the spine clip
    offers: List[AngleOffer] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "prog_start_ms": self.prog_start_ms,
            "prog_end_ms": self.prog_end_ms,
            "floor_speaker": self.floor_speaker,
            "offers": [o.to_dict() for o in self.offers],
        }


# --------------------------------------------------------------------------
# Track lookups (all in SOURCE-time ms of a single clip)
# --------------------------------------------------------------------------

def _covers(spans: Optional[list], t: int) -> bool:
    for s in spans or []:
        if int(s.get("start_ms", 0)) <= t < int(s.get("end_ms", 0)):
            return True
    return False


def _shot_at(spans: Optional[list], t: int) -> Optional[str]:
    for s in spans or []:
        if int(s.get("start_ms", 0)) <= t < int(s.get("end_ms", 0)):
            return s.get("shot_size")
    return None


def _reaction_at(spans: Optional[list], t: int) -> Optional[Tuple[str, float]]:
    best: Optional[Tuple[str, float]] = None
    for s in spans or []:
        if int(s.get("start_ms", 0)) <= t < int(s.get("end_ms", 0)):
            inten = float(s.get("intensity") or 0.0)
            if best is None or inten > best[1]:
                best = (s.get("type") or "reaction", inten)
    return best


def _gaze_cam(spans: Optional[list], t: int) -> bool:
    for s in spans or []:
        if int(s.get("start_ms", 0)) <= t < int(s.get("end_ms", 0)) and s.get("direction") == "to_camera":
            return True
    return False


def _floor_windows(
    span: SpineSpan, ov_start: int, ov_end: int, turns: List[Turn]
) -> List[Tuple[int, int, Optional[str]]]:
    """Split a program overlap into windows by the spine clip's speaker turns.

    Turns are in the spine clip's source time; program = source + base. Gaps
    between turns become floor_speaker=None windows so the menu still covers
    silence."""
    base = span.prog_start_ms - int(span.seg["in_ms"])  # prog = src + base
    items: List[Tuple[int, int, Optional[str]]] = []
    for ts, te, spk in turns:
        ps = max(ov_start, ts + base)
        pe = min(ov_end, te + base)
        if pe > ps:
            items.append((ps, pe, spk))
    items.sort(key=lambda x: x[0])

    out: List[Tuple[int, int, Optional[str]]] = []
    cursor = ov_start
    for ps, pe, spk in items:
        if ps > cursor:
            out.append((cursor, ps, None))
        out.append((max(ps, cursor), pe, spk))
        cursor = max(cursor, pe)
    if cursor < ov_end:
        out.append((cursor, ov_end, None))
    return out or [(ov_start, ov_end, None)]


# --------------------------------------------------------------------------
# Menu construction
# --------------------------------------------------------------------------

def build_angle_menu(
    spans: List[SpineSpan],
    group: SyncGroup,
    from_ms: int,
    to_ms: int,
    perceptions: Dict[str, dict],
    turns_by_file: Dict[str, List[Turn]],
) -> List[AngleWindow]:
    """Per-window floor speaker + per-angle offers across [from_ms, to_ms].

    Offers are sampled at each window's midpoint. A window only carries offers
    when the spine clip beneath it is itself a member of the group (so program
    time maps to each angle's source time via the sync offsets)."""
    windows: List[AngleWindow] = []
    for span in spans:
        ov_start = max(from_ms, span.prog_start_ms)
        ov_end = min(to_ms, span.prog_end_ms)
        if ov_end <= ov_start:
            continue
        spine_file = span.seg["file_id"]
        spine_off = group.offset_of(spine_file)
        turns = turns_by_file.get(spine_file, [])

        for ps, pe, spk in _floor_windows(span, ov_start, ov_end, turns):
            offers: List[AngleOffer] = []
            if spine_off is not None:
                mid = (ps + pe) // 2
                spine_src = int(span.seg["in_ms"]) + (mid - span.prog_start_ms)
                for m in group.members:
                    a_off = group.offset_of(m.file_id)
                    if a_off is None:
                        continue
                    a_src = spine_src + (spine_off - a_off)
                    p = perceptions.get(m.file_id) or {}
                    offers.append(AngleOffer(
                        file_id=m.file_id,
                        speaker_in_frame=_covers(p.get("speaking"), a_src),
                        shot_size=_shot_at(p.get("camera_craft"), a_src),
                        reaction=_reaction_at(p.get("reactions"), a_src),
                        gaze_to_camera=_gaze_cam(p.get("gaze"), a_src),
                    ))
            windows.append(AngleWindow(ps, pe, spk, offers))
    return windows


def build_menu(
    group: SyncGroup,
    spans: List[SpineSpan],
    from_ms: int,
    to_ms: int,
) -> List[AngleWindow]:
    """Convenience: load perceptions (member angles) + diarization turns (member
    angles + any spine clip) from the DB, then build the menu."""
    from app.services.l3.catalog import load_perceptions
    from app.services.l3.diarize import load_turns

    member_ids = [m.file_id for m in group.members]
    spine_ids = {span.seg["file_id"] for span in spans}
    perceptions = load_perceptions(member_ids)
    turns_by_file: Dict[str, List[Turn]] = {}
    for fid in set(member_ids) | spine_ids:
        turns_by_file[fid] = load_turns(fid)[2]
    return build_angle_menu(spans, group, from_ms, to_ms, perceptions, turns_by_file)


def render_angle_menu_text(windows: List[AngleWindow], limit: int = 40) -> str:
    """Compact rendering for a tool result Opus can read when overriding."""
    if not windows:
        return "(no angle windows in range)"
    lines: List[str] = []
    for w in windows[:limit]:
        sec = f"{w.prog_start_ms/1000:.1f}-{w.prog_end_ms/1000:.1f}s"
        floor = w.floor_speaker or "-"
        bits = []
        for o in w.offers:
            tag = []
            if o.speaker_in_frame:
                tag.append("speaker")
            if o.reaction:
                tag.append(f"react:{o.reaction[0]}({o.reaction[1]:.1f})")
            if o.shot_size:
                tag.append(o.shot_size)
            if o.gaze_to_camera:
                tag.append("to-cam")
            bits.append(f"{o.file_id[:6]}[{','.join(tag) or '-'}]")
        lines.append(f"  {sec} floor={floor}: " + " ".join(bits))
    if len(windows) > limit:
        lines.append(f"  ... (+{len(windows) - limit} more windows)")
    return "\n".join(lines)
