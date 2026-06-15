"""
Angle menu: the per-instant "what does each synced camera show?" table.

This is the anti-collapse FACT layer for multicam. Given a spine clip, its
VERIFIED synced angles (each with an exact offset from align_clips), and a
focus timeline (l3/focus.py), it answers -- for every focus interval -- what
each camera is actually showing relative to the current focus:

  * is the visible subject the FOCUS (the speaker / in the action), or a
    LISTENER / bystander?  (the speaker-following signal)
  * shot size (close-up reads as reaction; wide reads as establishing)
  * the strongest reaction in that window (a genuine laugh/nod is a reason to
    cut to the listener even mid-answer)
  * gaze direction

Opus reads this and decides pick_angle -- follow the speaker, cut to a reaction,
vary shots -- biased by its principles. The menu never picks; it only informs.

No cross-clip identity is needed for the common case: each camera's OWN speaking
spans say whether the person IT frames is the one currently talking, so we never
have to match person p1 on camera A to a person on camera B. All facts come from
the L2 perception already in the DB; the only cross-clip input is the verified
sync offset, which re-bases each angle's clock onto the spine's.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.services.l3.focus import FocusInterval


def _mmss(ms: int) -> str:
    s = max(0, ms) // 1000
    return f"{s // 60:d}:{s % 60:02d}"


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


# --------------------------------------------------------------------------
# Per-camera shot facts over one window
# --------------------------------------------------------------------------

@dataclass
class AngleShot:
    file_id: str
    is_spine: bool
    role: str                       # speaker | listener | action | hold | view | off
    shot_size: Optional[str] = None
    reaction: Optional[str] = None  # "laugh 0.8"
    gaze: Optional[str] = None


@dataclass
class MenuRow:
    focus: FocusInterval
    shots: List[AngleShot] = field(default_factory=list)


def _present(perception: dict, win_s: int, win_e: int) -> bool:
    """Is anyone on screen during the window? (enters/exits null = whole clip)."""
    persons = perception.get("persons") or []
    if not persons:
        return False
    for p in persons:
        ent = p.get("enters_ms")
        ext = p.get("exits_ms")
        ent = int(ent) if ent is not None else None
        ext = int(ext) if ext is not None else None
        if (ent is None or ent <= win_e) and (ext is None or ext >= win_s):
            return True
    return False


def _dominant_shot_size(perception: dict, win_s: int, win_e: int) -> Optional[str]:
    best, best_ov = None, 0
    for c in perception.get("camera_craft") or []:
        ov = _overlap(int(c.get("start_ms", 0)), int(c.get("end_ms", 0)), win_s, win_e)
        if ov > best_ov and c.get("shot_size"):
            best, best_ov = c.get("shot_size"), ov
    return best


def _strongest_reaction(perception: dict, win_s: int, win_e: int) -> Optional[str]:
    best, best_int = None, -1.0
    for r in perception.get("reactions") or []:
        if _overlap(int(r.get("start_ms", 0)), int(r.get("end_ms", 0)), win_s, win_e) <= 0:
            continue
        inten = float(r.get("intensity") or 0.0)
        if inten > best_int:
            best_int = inten
            kind = r.get("type") or "reaction"
            best = f"{kind} {inten:.1f}" if inten > 0 else str(kind)
    return best


def _dominant_gaze(perception: dict, win_s: int, win_e: int) -> Optional[str]:
    tally: Dict[str, int] = {}
    for g in perception.get("gaze") or []:
        ov = _overlap(int(g.get("start_ms", 0)), int(g.get("end_ms", 0)), win_s, win_e)
        d = g.get("direction")
        if ov > 0 and d:
            tally[d] = tally.get(d, 0) + ov
    return max(tally, key=tally.get) if tally else None


def _is_speaking(perception: dict, win_s: int, win_e: int) -> bool:
    for s in perception.get("speaking") or []:
        if _overlap(int(s.get("start_ms", 0)), int(s.get("end_ms", 0)), win_s, win_e) > 0:
            return True
    return False


def _is_acting(perception: dict, win_s: int, win_e: int) -> bool:
    for e in perception.get("events") or []:
        if e.get("change") not in ("action_starts", "action_peak", "action_ends"):
            continue
        if _overlap(int(e.get("start_ms", 0)), int(e.get("end_ms", 0)), win_s, win_e) > 0:
            return True
    return False


def _shot_facts(perception: Optional[dict], fid: str, is_spine: bool,
                win_s: int, win_e: int, focus_kind: str) -> AngleShot:
    if not perception or win_e <= win_s:
        return AngleShot(fid, is_spine, "off")

    if focus_kind == "speaker":
        if _is_speaking(perception, win_s, win_e):
            role = "speaker"
        elif _present(perception, win_s, win_e):
            role = "listener"
        else:
            role = "off"
    elif focus_kind == "action":
        if _is_acting(perception, win_s, win_e):
            role = "action"
        elif _present(perception, win_s, win_e):
            role = "hold"
        else:
            role = "view"
    else:
        role = "view" if _present(perception, win_s, win_e) else "off"

    return AngleShot(
        file_id=fid,
        is_spine=is_spine,
        role=role,
        shot_size=_dominant_shot_size(perception, win_s, win_e),
        reaction=_strongest_reaction(perception, win_s, win_e),
        gaze=_dominant_gaze(perception, win_s, win_e),
    )


# --------------------------------------------------------------------------
# The menu
# --------------------------------------------------------------------------

# Keep a single on-demand menu bounded so a long region can't blow the budget.
_MAX_ROWS = 40


def build_angle_menu(
    spine_fid: str,
    angle_offsets: Dict[str, int],
    focus_intervals: List[FocusInterval],
    perceptions: Dict[str, dict],
) -> List[MenuRow]:
    """One row per focus interval. `angle_offsets[fid]` = that angle's start in
    the spine's clock (from align_clips), so the same instant in the angle is
    (spine_time - offset). Times in the rows stay in the SPINE clip's clock."""
    rows: List[MenuRow] = []
    intervals = focus_intervals[:_MAX_ROWS]
    for f in intervals:
        shots = [_shot_facts(perceptions.get(spine_fid), spine_fid, True,
                             f.start_ms, f.end_ms, f.kind)]
        for fid, off in angle_offsets.items():
            shots.append(_shot_facts(perceptions.get(fid), fid, False,
                                     f.start_ms - off, f.end_ms - off, f.kind))
        rows.append(MenuRow(focus=f, shots=shots))
    return rows


def _shot_str(s: AngleShot) -> str:
    bits = [s.role]
    if s.shot_size:
        bits.append(s.shot_size)
    if s.reaction:
        bits.append(f"reaction:{s.reaction}")
    if s.gaze:
        bits.append(f"gaze:{s.gaze}")
    tag = "spine" if s.is_spine else s.file_id[:8]
    return f"{tag}[{', '.join(bits)}]"


def render_synced_angles_text(verified: list) -> str:
    """The compact, always-affordable prompt block surfaced when the scope has
    >=1 verified synced angle. It does NOT dump the per-moment menu (that can be
    huge for a long interview); it tells Opus the pairs exist and to pull the
    menu on demand with read_angles. `verified` is a list of sync.VerifiedAngle."""
    if not verified:
        return ""
    lines = [
        "SYNCED ANGLES (verified same-moment cameras -- these are REAL second "
        "angles, not B-roll; cut the picture between them with pick_angle while "
        "one audio track plays under). For any region, call read_angles to see, "
        "moment by moment, which camera shows the speaker vs. the listener (and "
        "any reaction) so you follow the focus instead of riding one camera:"
    ]
    for v in verified:
        r = v.result
        lines.append(
            f"  {v.file_a[:8]} <-> {v.file_b[:8]}: offset {r.offset_ms:+d}ms, "
            f"confidence {r.confidence:.2f} (overlap {r.overlap_ms // 1000}s)"
        )
    return "\n".join(lines)


def render_angle_menu_text(spine_fid: str, rows: List[MenuRow]) -> str:
    """On-demand read_angles output: the per-moment menu over the requested
    region (times in the spine clip's clock)."""
    if not rows:
        return ("ANGLE MENU: no focus intervals in this region (the spine clip "
                "has no speaker turns / action beats / sections here).")
    head = (f"ANGLE MENU for spine {spine_fid[:8]} (times in this clip's clock; "
            "each row: the focus, then what each camera shows -- follow the "
            "focus, cut to a listener on a genuine reaction or a long hold):")
    out = [head]
    for r in rows:
        f = r.focus
        label = f"{f.kind}={f.label}"
        out.append(f"  [{_mmss(f.start_ms)}-{_mmss(f.end_ms)}] {label}: "
                   + "  ".join(_shot_str(s) for s in r.shots))
    return "\n".join(out)
