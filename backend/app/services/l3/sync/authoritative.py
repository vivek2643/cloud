"""
Authoritative audio selection, Level 1 (audio_sync.plan.md SS5.5): within a
synced group, pick ONE source every downstream consumer (speech lattice,
seam welds, the resolved audio bed) treats as ground truth. Code-picked,
user-overridable -- never an LLM call (SS3.4).

Preference order: a dedicated external audio file (role="audio") if the
group has one -- it's there specifically because it's the clean feed; else
the best-sounding camera by a simple, explainable loudness/clipping/silence
heuristic. Deferred: Level 2 per-speaker routing (SS1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class GroupMember:
    file_id: str
    role: str  # "video_angle" | "audio"
    # audio_features row for this file, or None if not yet analyzed.
    audio_features: Optional[Dict[str, Any]] = None


def _camera_score(af: Dict[str, Any]) -> float:
    """Higher is better. Rewards loud-but-not-clipping, non-silent audio --
    a simple, explainable stand-in for "sounds clean" (no ML classifier)."""
    lufs = af.get("integrated_lufs")
    true_peak = af.get("true_peak_db")
    silences = af.get("silence_intervals") or []
    score = 0.0
    if lufs is not None:
        # Louder is generally better UP TO clipping risk; -14 LUFS is a
        # reasonable "well-recorded speech" reference point.
        score += max(-40.0, float(lufs)) + 40.0
    if true_peak is not None and float(true_peak) > -1.0:
        score -= 15.0  # clipping risk -- penalize, don't disqualify outright
    silent_ms = sum(max(0, int(b) - int(a)) for a, b in silences if isinstance(a, (int, float)))
    score -= silent_ms / 10000.0  # a lot of dead air suggests a mic that wasn't close to the action
    return score


def pick_authoritative(members: List[GroupMember]) -> Optional[str]:
    """The file_id to treat as this group's authoritative audio, or None
    when the group is empty (never called on an empty group in practice,
    but fail-open rather than raise)."""
    if not members:
        return None
    external = [m for m in members if m.role == "audio"]
    pool = external if external else members
    scored = [
        (m.file_id, _camera_score(m.audio_features) if m.audio_features else float("-inf"))
        for m in pool
    ]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[0][0]
