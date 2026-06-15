"""
Focus timeline: WHO/WHAT the edit should be on at each instant of a spine
region -- the genre-general driver behind angle selection.

The angle menu (l3/angle_menu.py) answers "what does each synced camera show
right now?"; to turn that into "which camera should we be on?" we first need a
notion of the current FOCUS. That notion is different per spine kind, but the
SHAPE is identical -- a list of (span -> focus descriptor) -- so the rest of the
pipeline never branches on genre:

  * dialogue / sync -> the ACTIVE SPEAKER (diarized turns on the spine clip).
  * action          -> the ACTION BEAT (subject-motion impacts on the spine clip).
  * music           -> the SECTION / phrase (musical structure of the spine bed).
  * visual          -> empty (the picture is locked; there is nothing to switch).

Each extractor is a pure function over PRE-LOADED signals, so a new spine kind
adds one function to the registry and is unit-testable without a database. The
signals are loaded once by `load_focus_signals`. All times are in the SPINE
CLIP's own source clock (ms); the caller re-bases other cameras by the verified
sync offset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class FocusInterval:
    start_ms: int          # spine-clip source clock
    end_ms: int
    kind: str              # "speaker" | "action" | "section"
    label: str             # who/what is the focus (e.g. a speaker id, a section letter)
    detail: str = ""       # short human note

    def to_dict(self) -> dict:
        d = {"start_ms": self.start_ms, "end_ms": self.end_ms,
             "kind": self.kind, "label": self.label}
        if self.detail:
            d["detail"] = self.detail
        return d


# --------------------------------------------------------------------------
# Signal loading (one DB read; extractors stay pure over the result)
# --------------------------------------------------------------------------

@dataclass
class FocusSignals:
    file_id: str
    # dialogue: merged diarized turns (start_ms, end_ms, speaker) in clip clock.
    turns: List[tuple]
    # action: subject-motion impacts [{ts_ms, ...}] in clip clock.
    action_points: List[dict]
    # music: sections [{start_ms, end_ms, label}] in clip clock.
    sections: List[dict]


def load_focus_signals(spine_fid: str) -> FocusSignals:
    """Pull every focus signal for the spine clip in one place. Best-effort:
    any missing channel is simply an empty list (so a silent action clip still
    yields an action timeline, a music bed yields sections, etc.)."""
    from app.services.l3.diarize import load_turns

    turns: List[tuple] = []
    try:
        _txt, _spk, turns = load_turns(spine_fid)
    except Exception:
        logger.debug("focus: no turns for %s", spine_fid, exc_info=True)

    action_points: List[dict] = []
    sections: List[dict] = []
    try:
        from app.services.l3.engine import _pg_conn

        with _pg_conn() as conn:
            mrow = conn.execute(
                "select action_points from motion_dynamics where file_id = %s",
                (spine_fid,),
            ).fetchone()
            if mrow and mrow[0]:
                action_points = mrow[0] if isinstance(mrow[0], list) else []
            srow = conn.execute(
                "select sections from music_structure where file_id = %s",
                (spine_fid,),
            ).fetchone()
            if srow and srow[0]:
                sections = srow[0] if isinstance(srow[0], list) else []
    except Exception:
        logger.debug("focus: motion/music load failed for %s", spine_fid, exc_info=True)

    return FocusSignals(spine_fid, turns or [], action_points or [], sections or [])


# --------------------------------------------------------------------------
# Extractors (pure: signals -> intervals). Registered by spine kind.
# --------------------------------------------------------------------------

Extractor = Callable[[FocusSignals, int, int], List[FocusInterval]]
_REGISTRY: Dict[str, Extractor] = {}


def register(kind: str) -> Callable[[Extractor], Extractor]:
    def deco(fn: Extractor) -> Extractor:
        _REGISTRY[kind] = fn
        return fn
    return deco


def _clip(s: int, e: int, lo: int, hi: int) -> Optional[tuple]:
    a, b = max(s, lo), min(e, hi)
    return (a, b) if b > a else None


@register("speaker")
def _speaker_focus(sig: FocusSignals, start_ms: int, end_ms: int) -> List[FocusInterval]:
    """Dialogue/sync: the focus is whoever is speaking. One interval per diarized
    turn -- the alternation IS the cut rhythm a 2-camera conversation follows."""
    out: List[FocusInterval] = []
    for t in sig.turns:
        s, e, spk = int(t[0]), int(t[1]), str(t[2])
        c = _clip(s, e, start_ms, end_ms)
        if c:
            out.append(FocusInterval(c[0], c[1], "speaker", spk,
                                     detail=f"{(c[1] - c[0]) / 1000:.1f}s turn"))
    return out


@register("action")
def _action_focus(sig: FocusSignals, start_ms: int, end_ms: int) -> List[FocusInterval]:
    """Action: the focus is the current action beat, delimited by subject-motion
    impacts. Each interval runs from one impact to the next (cut ON the impact)."""
    pts = sorted(int(p.get("ts_ms", 0)) for p in sig.action_points
                 if start_ms <= int(p.get("ts_ms", 0)) <= end_ms)
    bounds = [start_ms] + pts + [end_ms]
    out: List[FocusInterval] = []
    for i in range(len(bounds) - 1):
        c = _clip(bounds[i], bounds[i + 1], start_ms, end_ms)
        if c:
            out.append(FocusInterval(c[0], c[1], "action", f"beat{i + 1}",
                                     detail="post-impact" if i > 0 else "opening"))
    return out


@register("section")
def _section_focus(sig: FocusSignals, start_ms: int, end_ms: int) -> List[FocusInterval]:
    """Music: the focus is the current section/phrase of the bed."""
    out: List[FocusInterval] = []
    for sec in sig.sections:
        s, e = int(sec.get("start_ms", 0)), int(sec.get("end_ms", 0))
        c = _clip(s, e, start_ms, end_ms)
        if c:
            out.append(FocusInterval(c[0], c[1], "section",
                                     str(sec.get("label") or "?"), detail="section"))
    return out


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

# Spine kinds (set_spine vocabulary) -> the focus signal that drives angle choice.
_KIND_TO_FOCUS = {
    "dialogue": "speaker",
    "sync": "speaker",
    "music": "section",
    "visual": None,        # picture locked; no angle switching
    "action": "action",    # not a spine kind today, but kept general
}


def infer_focus_kind(sig: FocusSignals) -> Optional[str]:
    """When the spine kind is unknown (e.g. before set_spine), pick the focus
    from whatever signal the spine clip actually has."""
    if sig.turns:
        return "speaker"
    if sig.action_points:
        return "action"
    if sig.sections:
        return "section"
    return None


def focus_for_spine_kind(spine_kind: Optional[str], sig: FocusSignals) -> Optional[str]:
    """Resolve a set_spine kind to a focus kind, falling back to inference."""
    if spine_kind in _KIND_TO_FOCUS:
        return _KIND_TO_FOCUS[spine_kind]
    return infer_focus_kind(sig)


def focus_timeline(focus_kind: Optional[str], sig: FocusSignals,
                   start_ms: int, end_ms: int) -> List[FocusInterval]:
    """The (span -> focus) timeline for a region of the spine clip. Empty when
    the focus kind is None/unknown or the channel has no signal."""
    if not focus_kind or focus_kind not in _REGISTRY or end_ms <= start_ms:
        return []
    return _REGISTRY[focus_kind](sig, int(start_ms), int(end_ms))
