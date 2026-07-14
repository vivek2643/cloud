"""
Map index + shared placement primitives for the agentic editor.

The LLM brain lives in ``converse`` + ``tools``: it sees the whole footage map
and edits the document DIRECTLY with tools (``observe``/``act``). Those tools
refer to content by this module's stable map ids and resolve them through
``_MapIndex``:

  * a MOMENT id (e.g. ``ab12cd34:m07``) taken at one of its available energy
    LEVELS (broad/calm/balanced/tight/sharp), or
  * an ATOM id (a moment's finest sub-cut) when it wants just a piece.

What lives here now (the compile/arrange pipeline is gone -- ``act`` mutates the
document and ``observe.resolve_doc`` resolves it):
  * ``Placement`` / ``ResolvedCut`` -- the neutral pick + its resolved span.
  * ``_MapIndex`` -- validate a ref + resolve (ref, level) -> a source span.
  * ``_weld_segments`` -- merge adjacent same-clip contiguous main-line cuts
    (used by ``observe.resolve_doc`` to keep the agentic timeline clean).
  * ``render_timeline`` -- render the current timeline for a chat turn.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Channel model: track 0 is the V1 main line; track >= 1 is a V2+ video cutaway
# lane, anchored at a program time (never a "layer over" -- a full channel).
_MAIN_TRACK = 0

# Two adjacent main-line segments from the SAME clip whose source spans touch (or
# overlap) within this tolerance are welded into one continuous segment -- no
# redundant hard cut. ~3 frames @ 25fps; small enough that a real intra-clip jump
# (distant slices) and a cut's own keep_spans jump-cuts stay separate.
_WELD_TOL_MS = 120


# --------------------------------------------------------------------------
# Result shape
# --------------------------------------------------------------------------

@dataclass
class Placement:
    """One arranger choice: a map id taken at a level, placed on a track."""
    ref: str                       # moment_id (validated against map)
    level: str = "balanced"        # energy level
    track: int = _MAIN_TRACK       # 0 = V1 main line; >=1 = V2+ cutaway lane
    from_ms: Optional[int] = None  # V2+ cutaway anchor on the program clock (track>=1)
    reason: str = ""
    # Per-pick audio override on a video shot's DEFAULT mute policy: "keep" plays
    # its source sound (the shot's own audio is the point -- an action, a laugh,
    # applause, music), "mute" silences it, None = use the cut's default.
    audio: Optional[str] = None


@dataclass
class ResolvedCut:
    """A placement resolved against the map to a concrete source span.

    ``keep_spans`` is the CANONICAL jump-cut list -- a list of ``(in_ms, out_ms)``
    pairs, or None when the span plays whole. ``_MapIndex.resolve`` normalizes
    whatever the map carries into this one shape (see ``_norm_keep_spans``), so no
    downstream verb ever has to guess the encoding."""
    file_id: str
    src_in_ms: int
    src_out_ms: int
    keep_spans: Optional[List[Tuple[int, int]]]
    channel: Optional[str]        # said | done | shown
    label: str
    track: int
    from_ms: Optional[int]
    reason: str
    ref: str = ""               # the map id (carried onto segments for refinement)
    level: str = "balanced"
    mute: bool = False          # final source-audio mute (video default folded with the brain's audio:keep/mute)
    # av_coupling_authoritative.plan.md: this cut's baked authoritative audio
    # coupling (identity coupling -- file_id/0 -- for the ~90% solo-clip
    # case). Carried onto segments so `layers.resolve` never has to re-derive
    # audio routing lazily at render time.
    audio_file_id: str = ""
    audio_offset_ms: int = 0


# --------------------------------------------------------------------------
# Map index (validation + resolution)
# --------------------------------------------------------------------------

def _norm_keep_spans(raw: Any) -> Optional[List[Tuple[int, int]]]:
    """Coerce a map variant/atom keep-list into the canonical ``[(in, out), ...]``.

    The footage map stores jump-cut spans as ``[in_ms, out_ms]`` PAIRS
    (``footage_map._variant_from_rung``); some serialized forms use
    ``{"in_ms","out_ms"}`` dicts (``HeroCut.to_dict``). Accept both, drop anything
    malformed or empty. None/[] -> None (the span plays whole)."""
    if not raw:
        return None
    out: List[Tuple[int, int]] = []
    for sp in raw:
        try:
            if isinstance(sp, dict):
                a, b = int(sp["in_ms"]), int(sp["out_ms"])
            else:
                a, b = int(sp[0]), int(sp[1])
        except (KeyError, IndexError, TypeError, ValueError):
            continue
        if b > a:
            out.append((a, b))
    return out or None


def _resolve_mute(default_mute: bool, audio_override: Optional[str]) -> bool:
    """Fold the arranger's per-pick audio choice onto a cut's DEFAULT mute.
    'keep' plays the source sound, 'mute' silences it, anything else defers to
    the deterministic default the combiner set on the moment."""
    if audio_override == "keep":
        return False
    if audio_override == "mute":
        return True
    return default_mute


class _MapIndex:
    """Fast lookup over an ``assemble_map`` struct: moment_id -> moment. Owns
    the resolution of a (ref, level) to a span."""

    def __init__(self, map_struct: Dict[str, Any]) -> None:
        self.moments: Dict[str, dict] = {}
        for clip in (map_struct or {}).get("clips", []) or []:
            for m in clip.get("moments", []) or []:
                self.moments[m["moment_id"]] = m

    def has(self, ref: str) -> bool:
        return ref in self.moments

    def resolve(self, p: Placement) -> Optional[ResolvedCut]:
        m = self.moments.get(p.ref)
        if m is None:
            return None
        variants = m.get("variants") or {}
        level = p.level if p.level in variants else "balanced"
        v = variants.get(level) or variants.get("balanced") or next(iter(variants.values()), None)
        if v is None:
            return None
        return ResolvedCut(
            file_id=m["file_id"], src_in_ms=int(v["in_ms"]),
            src_out_ms=int(v["out_ms"]), keep_spans=_norm_keep_spans(v.get("keep_spans")),
            channel=m.get("channel"), label=m.get("gist") or "",
            track=p.track, from_ms=p.from_ms, reason=p.reason,
            ref=p.ref, level=v.get("level", level),
            mute=_resolve_mute(bool(m.get("mute")), p.audio),
            audio_file_id=m.get("audio_file_id") or m["file_id"],
            audio_offset_ms=int(m.get("audio_offset_ms") or 0),
        )

    def level_ok(self, ref: str, level: str) -> bool:
        m = self.moments.get(ref)
        return bool(m and level in (m.get("variants") or {}))


# --------------------------------------------------------------------------
# Welding (used by observe.resolve_doc to keep the agentic main line clean)
# --------------------------------------------------------------------------

def _weld_segments(segments: List[dict]) -> List[dict]:
    """Merge consecutive main-line segments from the SAME clip whose source spans
    are contiguous/overlapping (the next starts within ``_WELD_TOL_MS`` of where
    the previous ended), so two adjacent slices of one continuous shot play as ONE
    segment -- no redundant hard cut, no stutter.

    Safe by construction: a cut's own ``keep_spans`` jump-cuts and any intentional
    intra-clip jump (distant slices) are NON-contiguous, so they fail the test and
    stay separate. The merged segment keeps the first slice's level/ref/provenance
    and is marked ``speech`` if either side carried audio. Seg ids are re-issued
    (they are opaque everywhere downstream)."""
    welded: List[dict] = []
    for s in segments:
        prev = welded[-1] if welded else None
        if (prev is not None
                and prev["file_id"] == s["file_id"]
                and prev["in_ms"] <= s["in_ms"] <= prev["out_ms"] + _WELD_TOL_MS):
            prev["out_ms"] = max(prev["out_ms"], s["out_ms"])
            if s.get("axis") == "speech":
                prev["axis"] = "speech"
            # Keep audio if EITHER side wants it -- only a fully-stray merged span
            # stays muted (never silence real speech that welded onto a video cut).
            if not s.get("mute"):
                prev["mute"] = None
            if s.get("content") and s["content"] != prev.get("content"):
                prev["content"] = f"{(prev.get('content') or '').strip()} "\
                                  f"{s['content'].strip()}".strip()
            continue
        welded.append(s)
    for i, s in enumerate(welded):
        s["seg_id"] = f"a{i:03d}"
    return welded


def render_timeline(document: Optional[dict]) -> str:
    """Render the current timeline for a refinement turn.

    Each main-line (V1) segment is shown with its map id when known (so the model
    can keep/move/replace it in the SAME vocabulary as the map) or as a raw source
    span when it was hand-edited and no longer maps to a moment. V2 cutaways are
    listed after. Returns "" when there is nothing to refine."""
    if not document:
        return ""
    segs = document.get("timeline") or []
    ops = [o for o in (document.get("operations") or [])
           if isinstance(o, dict) and o.get("type") == "place_video"]
    if not segs and not ops:
        return ""
    lines: List[str] = []
    t = 0
    for s in segs:
        dur = int(s.get("out_ms", 0)) - int(s.get("in_ms", 0))
        if dur <= 0:
            continue
        ref = s.get("ref")
        if ref:
            tag = f"{ref}@{s.get('level', 'balanced')}"
        else:
            tag = (f"raw {str(s.get('file_id', '?'))[:8]} "
                   f"{int(s.get('in_ms', 0))}-{int(s.get('out_ms', 0))}ms")
        gist = (s.get("content") or "").replace("\n", " ").strip()
        if len(gist) > 60:
            gist = gist[:57] + "..."
        lines.append(f"  {_ms(t)} V1 {tag} \"{gist}\"")
        t += dur
    for o in ops:
        f = int(o.get("from_ms", 0))
        lines.append(f"  {_ms(f)} V2 cutaway raw {str(o.get('source_file_id', '?'))[:8]}")
    return "\n".join(lines)


def _ms(ms: int) -> str:
    s = max(0, int(ms)) // 1000
    return f"{s // 60}:{s % 60:02d}"
