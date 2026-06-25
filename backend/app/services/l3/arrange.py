"""
The arranger: one free-canvas LLM call that takes the WHOLE footage map and
returns a timeline.

This replaces the prescriptive Director -> Editor -> Coverage chain (spine /
overlay / angle) with a single editorial call modelled on how Cursor hands a
model a codebase: give it total awareness of every usable moment (the Tier-0
``footage_map``), let it pick + order freely, and resolve its choices
deterministically.

The model's only job -- exactly as specified -- is to identify the right
moments, at the right energy, and place them in order. It refers to content by
the map's stable ids:

  * a MOMENT id (e.g. ``ab12cd34:m07``) taken at one of its available energy
    LEVELS (broad/calm/balanced/tight/sharp), or
  * an ATOM id (a moment's finest sub-cut) when it wants just a piece.

Everything it places on track 0 is the main line and plays back-to-back (the
simple "no gaps" critic is satisfied by construction -- segments are laid
contiguously by the resolver). Tracks >= 1 are overlays anchored at a program
time. Hallucinated ids and illegal levels are dropped/normalised, never trusted.

``compile_placements`` turns the validated placements into the SAME Edit
Document the preview / timeline / render already read, so nothing downstream
changes.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.services.llm import user_message

logger = logging.getLogger(__name__)

# Overlay z (matches layers.Z_COVERAGE); imported lazily in the compiler to keep
# this module import-light. Track 0 is the main line.
_MAIN_TRACK = 0


# --------------------------------------------------------------------------
# Result shape
# --------------------------------------------------------------------------

@dataclass
class Placement:
    """One arranger choice: a map id taken at a level, placed on a track."""
    ref: str                       # moment_id or atom_id (validated against map)
    level: str = "balanced"        # energy level (moments only; atoms ignore it)
    track: int = _MAIN_TRACK       # 0 = main line; >=1 = overlay
    from_ms: Optional[int] = None  # overlay anchor on the program clock (track>=1)
    reason: str = ""


@dataclass
class ResolvedCut:
    """A placement resolved against the map to a concrete source span."""
    file_id: str
    src_in_ms: int
    src_out_ms: int
    keep_spans: Optional[List[Dict[str, int]]]
    modality: Optional[str]
    label: str
    track: int
    from_ms: Optional[int]
    reason: str
    ref: str = ""               # the map id (carried onto segments for refinement)
    level: str = "balanced"


# --------------------------------------------------------------------------
# Map index (validation + resolution)
# --------------------------------------------------------------------------

class _MapIndex:
    """Fast lookup over an ``assemble_map`` struct: moment_id -> moment, and
    atom_id -> (moment, atom). Owns the resolution of a (ref, level) to a span."""

    def __init__(self, map_struct: Dict[str, Any]) -> None:
        self.moments: Dict[str, dict] = {}
        self.atoms: Dict[str, Tuple[dict, dict]] = {}
        for clip in (map_struct or {}).get("clips", []) or []:
            for m in clip.get("moments", []) or []:
                self.moments[m["moment_id"]] = m
                for a in m.get("atoms", []) or []:
                    if a.get("atom_id"):
                        self.atoms[a["atom_id"]] = (m, a)

    def has(self, ref: str) -> bool:
        return ref in self.moments or ref in self.atoms

    def resolve(self, p: Placement) -> Optional[ResolvedCut]:
        if p.ref in self.atoms:
            m, a = self.atoms[p.ref]
            return ResolvedCut(
                file_id=m["file_id"], src_in_ms=int(a["in_ms"]),
                src_out_ms=int(a["out_ms"]), keep_spans=a.get("keep_spans"),
                modality=m.get("modality"), label=a.get("gist") or m.get("gist") or "",
                track=p.track, from_ms=p.from_ms, reason=p.reason,
                ref=p.ref, level="atom",
            )
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
            src_out_ms=int(v["out_ms"]), keep_spans=v.get("keep_spans"),
            modality=m.get("modality"), label=m.get("gist") or "",
            track=p.track, from_ms=p.from_ms, reason=p.reason,
            ref=p.ref, level=v.get("level", level),
        )

    def level_ok(self, ref: str, level: str) -> bool:
        m = self.moments.get(ref)
        return bool(m and level in (m.get("variants") or {}))


# --------------------------------------------------------------------------
# The arranger call
# --------------------------------------------------------------------------

_ARRANGER_SYSTEM = (
    "You are the EDITOR of a video. You are given a BRIEF and a complete MAP of "
    "all available footage: every clip, and within each clip every usable MOMENT "
    "with the FULL line of what is said, the energy LEVELS it can be taken at, and "
    "the finer ATOMS it can split into. Nothing is hidden -- this is the whole "
    "library, and you can read every line in full. Edit like a real editor who has "
    "watched all the footage: read for meaning, not keywords.\n\n"
    "Map notation (one moment per line):\n"
    "  m07 speech S1 .82 [0:14-0:21] \"the full line text\" · nrg:calm|balanced (+3 atoms) · dup:tg4*\n"
    "  - the id is <clip8>:<m##>; refer to a moment by its FULL id (e.g. ab12cd34:m07).\n"
    "  - the quoted text is the COMPLETE line -- judge delivery and relevance from it.\n"
    "  - nrg lists the levels you may take it at: broad (whole answer) .. balanced "
    "(one thought) .. sharp (tightest). Pick the level that fits the pacing.\n"
    "  - '(+N atoms)' means the moment splits into N finer sub-cuts; to use just a "
    "piece, refer to an atom id.\n"
    "  - 'dup:tgN' means this moment is the SAME content as others in group tgN "
    "(another take or camera angle). The '*' marks the engine's best take. Place "
    "exactly ONE moment per dup group -- prefer the '*' unless another reads or "
    "looks better. A DUPLICATE GROUPS section lists each group's members.\n\n"
    "What makes a good edit (your taste):\n"
    "- Open strong: a hook/cold-open that earns attention, then build, then land.\n"
    "- Tell ONE coherent story arc; cut anything off-brief, slates, mic-checks, "
    "banter, dead tangents. The map is raw footage -- not all of it belongs.\n"
    "- Never say the same thing twice (respect dup groups AND near-repeats).\n"
    "- Fewer, stronger moments beat a long flabby list. Honor speaker roles.\n"
    "- Choose energy LEVEL per cut for pacing; tighten for momentum, widen to let "
    "a key beat breathe.\n"
    "- EVERY moment is an equal candidate for the MAIN LINE no matter its "
    "category -- speech, reaction, insert, b-roll, action, moment, or anything "
    "else. NEVER prefer or rank one category over another; the label describes "
    "the shot, it does not decide its place. Pick purely by what the edit needs "
    "-- any category can open, carry, or close the cut. Everything on track 0 is "
    "the main line and plays back-to-back with no gaps. A higher track is an "
    "OPTIONAL silent video cutaway laid over the track-0 audio -- never where any "
    "category must go, and rarely needed.\n"
    "- Respect the target length if one is given.\n\n"
    "How to work, in two steps:\n"
    "1) DRAFT: first write a one-paragraph THESIS (the arc and why), then the draft "
    "timeline JSON.\n"
    "2) You will then be asked to CRITIQUE your own draft and return the FINAL JSON.\n\n"
    "JSON shape (the timeline is what matters):\n"
    '{"thesis": "<the arc>", "timeline": [{"ref": "<id>", "level": "balanced", '
    '"track": 0, "reason": "<short why>"}], "notes": "<one line>"}'
)

_CRITIQUE_PROMPT = (
    "Now critique your own draft as a tough editor, then return the FINAL timeline.\n"
    "Check, in order:\n"
    "- Redundancy: any two cuts that say the same thing? Any dup-group placed more "
    "than once? Drop the weaker one.\n"
    "- Opening: does cut 1 actually hook? If not, replace it.\n"
    "- Arc + order: does it build and land, or wander? Reorder for story.\n"
    "- Delivery/angle: for each dup group, is the chosen take the strongest read?\n"
    "- Length: within the target? Cut the flabbiest beats to fit.\n"
    "- Junk: remove any slate/mic-check/banter/off-brief moment that slipped in.\n"
    "Return ONLY the final JSON (no prose), same shape as before."
)


def _arranger_prompt(brief: str, plan, map_text: str,
                     current_timeline: Optional[str]) -> str:
    lines = [f"BRIEF: {brief.strip() or '(none -- infer a sensible edit)'}"]
    if getattr(plan, "intent", ""):
        lines.append(f"INTENT: {plan.intent}")
    bits = [f"energy~{getattr(plan, 'energy', 0.5):.2f}",
            f"aspect:{getattr(plan, 'aspect', 'landscape')}"]
    if getattr(plan, "target_duration_ms", None):
        bits.append(f"target:{round(plan.target_duration_ms / 1000.0, 1)}s")
    lines.append("PLAN: " + " · ".join(bits))
    if current_timeline:
        lines += ["", "CURRENT TIMELINE (refine this; keep what works, change what "
                  "the brief asks):", current_timeline]
    lines += ["", "FOOTAGE MAP:", map_text]
    return "\n".join(lines)


def arrange(brief: str, map_struct: Dict[str, Any], plan, *,
            llm, map_text: Optional[str] = None,
            current_timeline: Optional[str] = None) -> List[Placement]:
    """Run the arranger: brief + footage map -> ordered, validated placements.

    The brain reads the whole map in one resident context and reasons in a
    draft -> self-critique cycle (the faithful "edit like a human who watched
    everything" path). When the map is too large to hold resident it pages: a
    compact index + on-demand inspect tools (see ``orchestrator.run_paged``).

    ``plan`` carries energy/aspect/target defaults (the Director's read).
    ``current_timeline`` is the optional refinement overlay. Returns [] on any
    failure so the caller can fall back deterministically."""
    index = _MapIndex(map_struct)
    map_text = map_text or ""
    settings = get_settings()
    budget = int(getattr(settings, "arranger_resident_char_budget", 180_000))

    if map_text and len(map_text) > budget:
        try:
            from app.services.l3 import orchestrator   # lazy: paged path only
            return orchestrator.run_paged(
                brief, map_struct, plan, llm=llm, current_timeline=current_timeline)
        except Exception:
            logger.exception("arrange: paged path failed; deterministic fallback")
            return []

    return _resident_arrange(brief, plan, map_text, current_timeline, index, llm)


def _resident_arrange(brief: str, plan, map_text: str,
                      current_timeline: Optional[str], index: "_MapIndex",
                      llm) -> List[Placement]:
    """The whole map in one resident, cached context, reasoned over in a
    draft -> critique+revise cycle. System + map are a stable prefix reused
    across passes (``cache_system`` lets the provider reuse it), so iterating is
    cheap. Falls back to the draft if the critique pass yields nothing usable."""
    from app.services.l3.auto_edit import _parse_json   # lazy: avoid import cycle

    settings = get_settings()
    passes = int(getattr(settings, "arranger_passes", 2))
    max_tokens = settings.autoedit_max_output_tokens
    final_effort = (getattr(settings, "autoedit_effort", None) or "high")
    draft_effort = (getattr(settings, "arranger_draft_effort", None)
                    or (final_effort if passes <= 1 else "medium"))

    messages = [user_message(_arranger_prompt(brief, plan, map_text, current_timeline))]

    # Pass 1: thesis + draft.
    r1 = llm.run(system=_ARRANGER_SYSTEM, messages=messages, max_tokens=max_tokens,
                 effort=(final_effort if passes <= 1 else draft_effort),
                 cache_system=True)
    draft = _coerce_placements(_parse_json(r1.text), index)
    if passes <= 1:
        return draft

    # Pass 2: critique own draft + revise. Reuses the identical cached prefix.
    messages.append(r1.assistant_message)
    messages.append(user_message(_CRITIQUE_PROMPT))
    r2 = llm.run(system=_ARRANGER_SYSTEM, messages=messages, max_tokens=max_tokens,
                 effort=final_effort, cache_system=True)
    final = _coerce_placements(_parse_json(r2.text), index)
    return final or draft


def _coerce_placements(doc: Optional[dict], index: _MapIndex) -> List[Placement]:
    """Validate the model's timeline: keep only real ids, normalise illegal
    levels to the moment's balanced take, de-dupe, preserve order."""
    out: List[Placement] = []
    seen: set = set()
    for item in (doc or {}).get("timeline", []) or []:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("ref") or "").strip()
        if not ref or ref in seen or not index.has(ref):
            continue
        seen.add(ref)
        level = str(item.get("level") or "balanced").strip().lower()
        if ref in index.moments and not index.level_ok(ref, level):
            level = "balanced"
        try:
            track = int(item.get("track", _MAIN_TRACK))
        except (TypeError, ValueError):
            track = _MAIN_TRACK
        from_ms = item.get("from_ms")
        try:
            from_ms = int(from_ms) if from_ms is not None else None
        except (TypeError, ValueError):
            from_ms = None
        out.append(Placement(ref=ref, level=level, track=max(0, track),
                              from_ms=from_ms, reason=str(item.get("reason") or "").strip()))
    return out


# --------------------------------------------------------------------------
# Compile placements -> Edit Document (Phase 2, + the no-gap critic)
# --------------------------------------------------------------------------

def resolve_placements(placements: List[Placement],
                       map_struct: Dict[str, Any]) -> List[ResolvedCut]:
    index = _MapIndex(map_struct)
    out: List[ResolvedCut] = []
    for p in placements:
        rc = index.resolve(p)
        if rc is not None and rc.src_out_ms > rc.src_in_ms:
            out.append(rc)
    return out


def _segments_from_main(cuts: List[ResolvedCut]) -> List[dict]:
    """Main-line cuts -> contiguous timeline segments (the no-gap critic: track-0
    segments are laid back-to-back by the resolver, so order alone removes gaps).
    A breath-removal edit-list (``keep_spans``) expands into one segment per kept
    span so the jump-cuts survive."""
    segments: List[dict] = []
    i = 0
    for rc in cuts:
        if rc.track != _MAIN_TRACK:
            continue
        spans = rc.keep_spans or [{"in_ms": rc.src_in_ms, "out_ms": rc.src_out_ms}]
        for j, sp in enumerate(spans):
            in_ms, out_ms = int(sp["in_ms"]), int(sp["out_ms"])
            if out_ms <= in_ms:
                continue
            segments.append({
                "seg_id": f"a{i:03d}_{j}",
                "file_id": rc.file_id,
                "in_ms": in_ms,
                "out_ms": out_ms,
                "axis": "speech" if rc.modality == "speech" else "any",
                "beat_id": None,
                "content": rc.label,
                "rationale": rc.reason or None,
                "priority": 3,
                "cut_in_cost": 0.0,
                "cut_out_cost": 0.0,
                "warnings": [],
                # Map provenance: lets a refinement turn speak in the same ids.
                "ref": rc.ref or None,
                "level": rc.level,
            })
        i += 1
    return segments


def _operations_from_overlays(cuts: List[ResolvedCut], total_ms: int) -> List[dict]:
    """Track>=1 cuts -> place_video overlay operations anchored on the program
    clock. Skipped when no anchor / out of range -- overlays never break the
    main line."""
    from app.services.l3 import layers

    ops: List[dict] = []
    for rc in cuts:
        if rc.track == _MAIN_TRACK or rc.from_ms is None:
            continue
        from_ms = max(0, min(int(rc.from_ms), total_ms))
        cut_len = rc.src_out_ms - rc.src_in_ms
        span = min(cut_len, max(0, total_ms - from_ms)) if total_ms else cut_len
        if span < 200:
            continue
        ops.append({
            "op_id": f"ov_{uuid.uuid4().hex[:6]}",
            "type": "place_video",
            "source_file_id": rc.file_id,
            "src_in_ms": int(rc.src_in_ms),
            "src_out_ms": int(rc.src_in_ms + span),
            "from_ms": from_ms,
            "to_ms": from_ms + span,
            "layout": layers.DEFAULT_LAYOUT,
            "z": layers.Z_COVERAGE + max(0, rc.track - 1),
            "opacity": 1.0,
            "rationale": rc.reason or None,
            "warnings": [],
        })
    return ops


def fallback_placements(map_struct: Dict[str, Any], plan) -> List[Placement]:
    """Deterministic draft when the arranger fails: top moments by score, taken
    at balanced, ordered chronologically, capped to the target (or 60s). Mirrors
    the old ``_fallback_picks`` over the map."""
    moments = [m for clip in (map_struct or {}).get("clips", []) or []
               for m in clip.get("moments", []) or []]
    if not moments:
        return []
    target = getattr(plan, "target_duration_ms", None) or 60000
    ranked = sorted(moments, key=lambda m: float(m.get("score", 0)), reverse=True)
    chosen: List[dict] = []
    total = 0
    for m in ranked:
        chosen.append(m)
        bal = (m.get("variants") or {}).get("balanced") or {}
        total += int(bal.get("play_ms", m.get("play_ms", 0)))
        if total >= target:
            break
    chosen.sort(key=lambda m: (m["file_id"], m.get("in_ms", 0)))
    return [Placement(ref=m["moment_id"], level="balanced", track=_MAIN_TRACK,
                      reason="auto (fallback: top score)") for m in chosen]


def timeline_overlay(document: Optional[dict]) -> str:
    """Render the current timeline as a map-overlay for a refinement turn.

    Each main-line segment is shown with its map id when known (so the model can
    keep/move/replace it in the SAME vocabulary as the map) or as a raw source
    span when it was hand-edited and no longer maps to a moment. Returns "" when
    there is nothing to refine."""
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
        lines.append(f"  {_ms(t)} track0 {tag} \"{gist}\"")
        t += dur
    for o in ops:
        f = int(o.get("from_ms", 0))
        lines.append(f"  {_ms(f)} overlay raw {str(o.get('source_file_id', '?'))[:8]} "
                     f"(z{o.get('z', 1)})")
    return "\n".join(lines)


def _ms(ms: int) -> str:
    s = max(0, int(ms)) // 1000
    return f"{s // 60}:{s % 60:02d}"


def compile_placements(brief: str, plan, placements: List[Placement],
                       map_struct: Dict[str, Any], file_ids: List[str]) -> dict:
    """Validated placements -> the resolved Edit Document (same shape the rest of
    the system reads). Track 0 becomes the contiguous spine; tracks >= 1 become
    overlay operations."""
    from app.services.l3.auto_edit import _build_document
    from app.services.l3 import layers

    cuts = resolve_placements(placements, map_struct)
    segments = _segments_from_main(cuts)
    _, total_ms = layers.spine_spans(segments)
    operations = _operations_from_overlays(cuts, total_ms)
    summary = getattr(plan, "intent", "") or "Auto-assembled edit."
    return _build_document(brief, plan, segments, operations, file_ids, summary, [])
