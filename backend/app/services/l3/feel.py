"""
Feel simulator: a PURE, VLM-free read of how the current edit FEELS.

The agentic brain (Phase 3) shouldn't need an expensive pass to know whether its
cut drags, races, or jump-cuts. Everything about "feel" is a free projection of
the edit's OWN structure -- shot lengths, speaking pace, who is on screen. This
module computes those dimensions deterministically from the timeline + a little
map context, and narrates them back in anchored prose the brain can act on
("energy sags cuts 4-6; two same-speaker cuts back-to-back at cut 7").

Design:
  * PURE. Inputs are the timeline segments + a read-only lookup (moment meta by
    ref). No DB, no LLM, no network -- safe to call every loop turn. The context
    builder (``observe.build_context``) does the DB reads once.
  * Honest, not prescriptive. It reports what IS (pace, rhythm, tone), it does
    not decide what SHOULD change -- the brain does. Numbers are anchored to cut
    positions so the brain can point its edits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional

# A cut under this plays as "fast"; over the long threshold it "lingers".
_FAST_MS = 2000
_LINGER_MS = 6000
# A run of at least this many fast cuts back-to-back reads as a burst.
_FAST_RUN = 3


@dataclass
class CutFeel:
    """Feel of one timeline cut, in play order (1-based ``pos`` for narration)."""
    pos: int
    ref: Optional[str]
    file_id: str
    dur_ms: int
    words: int
    pace_wps: float          # spoken words per second (0 for silent/video cuts)
    is_speech: bool
    speaker: Optional[str]
    channel: Optional[str]   # said | done | shown | None
    energy: float            # 0..1 proxy (short + fast -> high; long + slow -> low)


@dataclass
class FeelReport:
    cuts: List[CutFeel] = field(default_factory=list)
    total_ms: int = 0

    # -- aggregate reads (all derived, all cheap) --
    @property
    def avg_pace(self) -> float:
        spoken = [c.pace_wps for c in self.cuts if c.is_speech and c.pace_wps > 0]
        return round(sum(spoken) / len(spoken), 2) if spoken else 0.0

    def narrate(self) -> str:
        """Anchored prose: what the edit feels like, pointed at cut positions."""
        if not self.cuts:
            return "The timeline is empty -- nothing to feel yet."
        parts: List[str] = []
        n = len(self.cuts)
        secs = round(self.total_ms / 1000.0, 1)
        parts.append(f"{n} cut{'s' if n != 1 else ''}, {secs}s total"
                     + (f", ~{self.avg_pace} words/s when talking" if self.avg_pace else ""))

        for lo, hi in _fast_runs(self.cuts):
            parts.append(f"cuts {lo}-{hi} race (a burst of sub-{_FAST_MS // 1000}s cuts)"
                         if hi > lo else f"cut {lo} is a quick sub-{_FAST_MS // 1000}s beat")
        for lo, hi in _low_energy_runs(self.cuts):
            parts.append(f"energy sags cuts {lo}-{hi}" if hi > lo else f"cut {lo} drags")
        for pos in _lingers(self.cuts):
            parts.append(f"cut {pos} lingers ({self.cuts[pos - 1].dur_ms // 1000}s)")
        for lo, hi in _same_speaker_runs(self.cuts):
            spk = self.cuts[lo - 1].speaker or "one speaker"
            parts.append(f"cuts {lo}-{hi} stay on {spk} back-to-back (jump-cut risk)")

        return "; ".join(parts) + "."

    def to_dict(self) -> dict:
        return {
            "total_ms": self.total_ms,
            "cut_count": len(self.cuts),
            "avg_pace_wps": self.avg_pace,
            "narration": self.narrate(),
            "cuts": [
                {"pos": c.pos, "ref": c.ref, "dur_ms": c.dur_ms, "pace_wps": c.pace_wps,
                 "channel": c.channel, "speaker": c.speaker,
                 "energy": round(c.energy, 2)}
                for c in self.cuts
            ],
        }


def simulate(
    timeline: List[dict],
    meta_by_ref: Optional[Dict[str, dict]] = None,
) -> FeelReport:
    """Compute the feel of a timeline. ``meta_by_ref`` maps a segment's map ref
    to its moment node (for speaker/channel). Optional -- feel degrades
    gracefully to what the timeline alone reveals (pace + rhythm)."""
    meta_by_ref = meta_by_ref or {}
    cuts: List[CutFeel] = []
    total = 0
    for i, seg in enumerate(timeline):
        dur = max(0, int(seg.get("out_ms", 0)) - int(seg.get("in_ms", 0)))
        if dur <= 0:
            continue
        total += dur
        content = (seg.get("content") or "").strip()
        words = len(content.split()) if content else 0
        is_speech = seg.get("axis") == "speech"
        pace = round(words / (dur / 1000.0), 2) if (is_speech and words and dur) else 0.0
        meta = meta_by_ref.get(seg.get("ref") or "") or {}
        cuts.append(CutFeel(
            pos=len(cuts) + 1,
            ref=seg.get("ref"),
            file_id=str(seg.get("file_id") or ""),
            dur_ms=dur,
            words=words,
            pace_wps=pace,
            is_speech=is_speech,
            speaker=meta.get("speaker"),
            channel=meta.get("channel") or ("said" if is_speech else None),
            energy=0.0,  # filled below (needs the whole set for normalisation)
        ))
    _score_energy(cuts)
    return FeelReport(cuts=cuts, total_ms=total)


# --------------------------------------------------------------------------
# Derivations
# --------------------------------------------------------------------------

def _score_energy(cuts: List[CutFeel]) -> None:
    """Energy proxy per cut: short shots and faster speech read as higher energy;
    long, slow ones as lower. Normalised against this edit's own median cut so it
    is relative to the piece, not an absolute scale."""
    if not cuts:
        return
    med = median([c.dur_ms for c in cuts]) or 1
    paces = [c.pace_wps for c in cuts if c.pace_wps > 0]
    pace_med = median(paces) if paces else 0.0
    for c in cuts:
        # Shortness: 1.0 at half the median length, 0.0 at double it.
        short = _clamp01(1.0 - (c.dur_ms / (med * 2.0)))
        if c.pace_wps > 0 and pace_med > 0:
            fast = _clamp01(c.pace_wps / (pace_med * 1.5))
            c.energy = round(0.5 * short + 0.5 * fast, 3)
        else:
            c.energy = round(short, 3)


def _fast_runs(cuts: List[CutFeel]) -> List[tuple[int, int]]:
    return _runs([c.pos for c in cuts if c.dur_ms < _FAST_MS], min_len=_FAST_RUN)


def _lingers(cuts: List[CutFeel]) -> List[int]:
    return [c.pos for c in cuts if c.dur_ms >= _LINGER_MS]


def _low_energy_runs(cuts: List[CutFeel]) -> List[tuple[int, int]]:
    if len(cuts) < 3:
        return []
    med = median([c.energy for c in cuts])
    low = [c.pos for c in cuts if c.energy < med * 0.6]
    return _runs(low, min_len=3)


def _same_speaker_runs(cuts: List[CutFeel]) -> List[tuple[int, int]]:
    """Positions of >=2 adjacent speech cuts sharing one (known) speaker."""
    out: List[tuple[int, int]] = []
    i = 0
    while i < len(cuts):
        c = cuts[i]
        if c.is_speech and c.speaker:
            j = i
            while (j + 1 < len(cuts) and cuts[j + 1].is_speech
                   and cuts[j + 1].speaker == c.speaker):
                j += 1
            if j > i:
                out.append((cuts[i].pos, cuts[j].pos))
            i = j + 1
        else:
            i += 1
    return out


def _runs(positions: List[int], min_len: int) -> List[tuple[int, int]]:
    """Collapse a sorted list of positions into contiguous [lo, hi] runs of at
    least ``min_len`` consecutive positions."""
    if not positions:
        return []
    positions = sorted(set(positions))
    runs: List[tuple[int, int]] = []
    lo = prev = positions[0]
    for p in positions[1:]:
        if p == prev + 1:
            prev = p
            continue
        if prev - lo + 1 >= min_len:
            runs.append((lo, prev))
        lo = prev = p
    if prev - lo + 1 >= min_len:
        runs.append((lo, prev))
    return runs


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
