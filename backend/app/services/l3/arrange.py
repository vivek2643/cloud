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
  * ``render_program_map`` -- render the ASSEMBLED edit (the fully-resolved
    layer stack) as two small tables for a chat turn.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3 import layers

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


def _label(text: Optional[str]) -> str:
    gist = (text or "").replace("\n", " ").strip()
    return gist[:57] + "..." if len(gist) > 60 else gist


def _source_tag(seg: Optional[dict]) -> str:
    """The same map-id vocabulary a `place`/`trim`/`retime` call uses: a moment
    ref@level when the row traces back to one, else a raw source window --
    matches how the BEAT INDEX and the old flat render both cite sources."""
    if not seg:
        return "?"
    ref = seg.get("ref")
    return f"{ref}@{seg.get('level', 'balanced')}" if ref \
        else f"raw {str(seg.get('file_id', '?'))[:8]}"


def render_program_map(document: Optional[dict], durations: Optional[Dict[str, int]] = None) -> str:
    """Render the ASSEMBLED edit as two small, time-aligned tables (VIDEO,
    AUDIO) built from the fully-resolved layer stack (``layers.resolve``) --
    edso_pacing_audit_timing.plan.md item 2, replacing the old flat V1-then-V2
    line-per-line render. Every row carries a STABLE id (a seg_id or op_id)
    the brain can act on directly, plus its program window, layout, source
    ref, and a neutral label -- so stacking/overlap (a V2 cutaway over two V1
    spine cuts, a music bed under everything) is visible from the shared
    clock + z alone, with no prose needed. Generic: no speaker/role-of-person
    column, just the compositing structure. Returns "" when there is nothing
    to show yet."""
    if not document:
        return ""
    timeline = document.get("timeline") or []
    operations = document.get("operations") or []
    if not timeline and not operations:
        return ""
    resolved = layers.resolve(document, durations=durations)
    if not resolved.video_layers and not resolved.audio_layers:
        return ""

    seg_by_id = {s["seg_id"]: s for s in timeline if s.get("seg_id")}
    op_by_id = {o["op_id"]: o for o in operations if o.get("op_id")}

    lines: List[str] = [
        f"PROGRAM MAP  {_ms(0)}-{_ms(resolved.duration_ms)}  {resolved.aspect}"
    ]

    if resolved.video_layers:
        lines.append("VIDEO")
        lines.append("  lane id  z  prog(ms)  dur  layout  source  label")
        for v in sorted(resolved.video_layers, key=lambda x: (x.kind != "spine", x.prog_start_ms)):
            dur = v.prog_end_ms - v.prog_start_ms
            if v.kind == "spine":
                vid = v.layer_id[len("v_"):] if v.layer_id.startswith("v_") else v.layer_id
                seg = seg_by_id.get(vid)
                lane, source, label = "V1", _source_tag(seg), _label((seg or {}).get("content"))
            else:
                op = op_by_id.get(v.op_id or "")
                lane = "V2"
                source = f"raw {str(v.source_file_id)[:8]}"
                vid = v.op_id or v.layer_id
                label = _label((op or {}).get("rationale") or (op or {}).get("purpose"))
            lines.append(f"  {lane} {vid}  z{v.z}  {v.prog_start_ms}-{v.prog_end_ms}ms "
                         f"({dur}ms)  {v.layout}  {source}  \"{label}\"")

    if resolved.audio_layers:
        lines.append("AUDIO")
        lines.append("  lane id  role  prog(ms)  source  gain/duck/fade")
        for a in sorted(resolved.audio_layers, key=lambda x: (x.kind != "spine", x.prog_start_ms)):
            if a.kind == "spine":
                aid = a.layer_id[len("a_"):] if a.layer_id.startswith("a_") else a.layer_id
                lane, source = "A1", "(main line)"
            else:
                aid = a.op_id or a.layer_id
                lane, source = "A2", f"raw {str(a.source_file_id)[:8]}"
            tags = [f"gain:{a.gain_db:.0f}"]
            if a.duck_db:
                tags.append(f"duck:{a.duck_db:.0f}")
            if a.fade_in_ms:
                tags.append(f"fade-in:{a.fade_in_ms}ms")
            if a.fade_out_ms:
                tags.append(f"fade-out:{a.fade_out_ms}ms")
            lines.append(f"  {lane} {aid}  {a.role}  {a.prog_start_ms}-{a.prog_end_ms}ms  "
                         f"{source}  {' '.join(tags)}")

    return "\n".join(lines)


def _ms(ms: int) -> str:
    s = max(0, int(ms)) // 1000
    return f"{s // 60}:{s % 60:02d}"
