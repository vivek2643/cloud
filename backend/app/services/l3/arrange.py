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
  * ``heal_adjacent_cuts`` -- the shared deterministic HEAL pass: merge adjacent
    SAME-SOURCE spine segments whose source spans are contiguous (or within a
    tiny gap) into one continuous clip, so one take plays through with no micro
    jump-cut. Wired into BOTH composition paths -- the auto-assembly/brain path
    (``observe.resolve_doc``) and the manual/SNAP path (``put_document`` ->
    ``resolve_document``). Generalizes the old ``_weld_segments``.
  * ``render_program_map`` -- render the ASSEMBLED edit (the fully-resolved
    layer stack) as two small tables for a chat turn.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3 import footage_map, layers

logger = logging.getLogger(__name__)

# Channel model: track 0 is the V1 main line; track >= 1 is a V2+ video cutaway
# lane, anchored at a program time (never a "layer over" -- a full channel).
_MAIN_TRACK = 0

# heal_adjacent_cuts.plan.md: module fallback for call sites / tests that don't
# thread `get_settings().heal_gap_ms`. Two adjacent SAME-FILE spine segments
# whose source spans touch (or sit within this many ms) are HEALED into one
# continuous segment -- no redundant hard cut, no micro jump-cut. Supersedes the
# old hard-coded _WELD_TOL_MS=120 (a real 130ms micro-jump sat just above it).
HEAL_GAP_MS_DEFAULT = 200


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
    # v4_cluster_read_act.plan.md Part C: place ONE piece of a multi-event
    # cluster instead of the whole moment -- the 1-based position the brain
    # was shown in the Beat Index / read_state (footage_map.piece_breakdown's
    # "pos"). None (the common case) places the whole moment at `level`, same
    # as before this field existed.
    piece: Optional[int] = None
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
        if p.piece is not None:
            # v4_cluster_read_act.plan.md Part C: one piece of a multi-event
            # cluster, not the whole moment. footage_map.resolve_piece already
            # fails closed (None) for a single-event moment, an out-of-range
            # piece, or the shouldn't-happen no-containing-piece case -- so a
            # bad `piece` just falls through to this verb's normal unknown-ref
            # no-op, same as any other illegal Placement.
            span = footage_map.resolve_piece(m, p.piece)
            if span is None:
                return None
            return ResolvedCut(
                file_id=m["file_id"], src_in_ms=span[0], src_out_ms=span[1],
                keep_spans=None, channel=m.get("channel"), label=m.get("gist") or "",
                track=p.track, from_ms=p.from_ms, reason=p.reason,
                ref=p.ref, level="sharp",
                mute=_resolve_mute(bool(m.get("mute")), p.audio),
                audio_file_id=m.get("audio_file_id") or m["file_id"],
                audio_offset_ms=int(m.get("audio_offset_ms") or 0),
            )
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
# Heal (the shared deterministic compose-time pass; wired into BOTH the
# auto-assembly path -- observe.resolve_doc -- and the manual/SNAP path --
# put_document). heal_adjacent_cuts.plan.md.
# --------------------------------------------------------------------------

# Op types keyed on a SEAM segment id (the "next" cut a transition sits before);
# both must be remapped/dropped when the heal merges the seam away.
_SEAM_OP_TYPES = ("split_edit", "crossfade")


def _audio_coupling(seg: dict) -> Tuple[str, int, Any]:
    """A segment's authoritative audio identity (file/offset/override). Two
    contiguous slices are only heal-mergeable when these MATCH -- merging would
    otherwise silently drop one side's routing (e.g. a replace_audio'd seam).
    Defaults (missing keys) mirror _segments_from_cut: file_id / 0 / no override."""
    override = seg.get("audio_override")
    return (
        str(seg.get("audio_file_id") or seg.get("file_id") or ""),
        int(seg.get("audio_offset_ms") or 0),
        None if override is None else _freeze(override),
    )


def _freeze(val: Any) -> Any:
    if isinstance(val, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in val.items()))
    if isinstance(val, (list, tuple)):
        return tuple(_freeze(v) for v in val)
    return val


def _can_heal(prev: dict, s: dict, gap_ms: int) -> bool:
    """Deterministic weld-vs-hard decision for one adjacent pair. HEAL when ALL:
      * same file_id,
      * the next segment is forward + contiguous within ``gap_ms``
        (prev.in_ms <= s.in_ms <= prev.out_ms + gap_ms),
      * ``s`` carries no ``hard_seam`` marker (a deliberately tightened pause,
        stamped by retime/tighten, must NEVER heal -- §5 tighten guard), and
      * both sides share the same authoritative audio coupling."""
    if prev.get("file_id") != s.get("file_id"):
        return False
    if s.get("hard_seam"):
        return False
    try:
        p_in, p_out, s_in = int(prev["in_ms"]), int(prev["out_ms"]), int(s["in_ms"])
    except (KeyError, TypeError, ValueError):
        return False
    if not (p_in <= s_in <= p_out + gap_ms):
        return False
    return _audio_coupling(prev) == _audio_coupling(s)


def heal_adjacent_cuts(
    segments: List[dict],
    *,
    gap_ms: int,
    reindex: bool = True,
) -> Tuple[List[dict], Dict[str, str]]:
    """Merge adjacent SAME-SOURCE spine segments whose source spans are
    contiguous or separated by at most ``gap_ms`` of dropped source, so one
    continuous take plays as ONE segment (no micro jump-cut).

    Deterministic: same file + forward source-adjacency within ``gap_ms`` (and a
    matching audio coupling, no ``hard_seam`` guard -- see ``_can_heal``).
    Healing sets ``prev.out_ms = max(prev.out_ms, next.out_ms)`` -- the source
    plays straight through, bridging the dropped gap. Audio needs no handling
    here: ``layers.resolve`` derives the dialogue layer per merged segment, so it
    bridges the same gap automatically and stays aligned.

    Returns ``(healed_segments, merged_map)`` where ``merged_map`` maps every
    ABSORBED seg_id -> the surviving predecessor's (final) seg_id, so ops keyed
    on a seam id (split_edit/crossfade) can be dropped/remapped.

    ``reindex=True`` re-issues ids as ``a000, a001, ...`` (the brain refers to
    cuts by these stable, human-readable ids). ``reindex=False`` keeps the
    predecessor's own id (the manual/snap path, so client selection/undo
    references survive). Idempotent: a second pass over an already-healed
    timeline is a no-op and returns an empty ``merged_map``.

    The merged segment keeps the first slice's level/ref/provenance, is marked
    ``speech`` if either side carried audio, and concatenates content (matching
    the old weld). Surviving segment DICTS are kept (mutated in place), so a
    caller can still track a seam by object identity across the pass."""
    healed: List[dict] = []
    absorbed: List[Tuple[Optional[str], dict]] = []  # (absorbed seg_id, survivor dict)
    for s in segments:
        prev = healed[-1] if healed else None
        if prev is not None and _can_heal(prev, s, gap_ms):
            prev["out_ms"] = max(prev["out_ms"], s["out_ms"])
            if s.get("axis") == "speech":
                prev["axis"] = "speech"
            # Keep audio if EITHER side wants it -- only a fully-stray merged span
            # stays muted (never silence real speech that healed onto a video cut).
            if not s.get("mute"):
                prev["mute"] = None
            if s.get("content") and s["content"] != prev.get("content"):
                prev["content"] = f"{(prev.get('content') or '').strip()} "\
                                  f"{s['content'].strip()}".strip()
            absorbed.append((s.get("seg_id"), prev))
            continue
        healed.append(s)
    if reindex:
        for i, s in enumerate(healed):
            s["seg_id"] = f"a{i:03d}"
    merged_map = {aid: surv.get("seg_id") for aid, surv in absorbed
                  if aid and surv.get("seg_id")}
    return healed, merged_map


def _remap_seam_ops_list(operations: List[dict], merged_map: Dict[str, str]) -> List[dict]:
    """Drop the split_edit/crossfade ops whose seam segment was healed AWAY
    (its id is a key in ``merged_map`` -- it merged into its predecessor, so the
    seam no longer exists). Ops on surviving seams pass through unchanged -- the
    id-stable (``reindex=False``) call site's shared op-remap helper. The
    reindex=True path (observe) remaps surviving ids too, by object identity."""
    if not merged_map or not any(o.get("type") in _SEAM_OP_TYPES for o in operations):
        return operations
    kept: List[dict] = []
    for o in operations:
        if o.get("type") in _SEAM_OP_TYPES and o.get("seam_seg_id") in merged_map:
            continue                      # seam healed away -> transition is moot
        kept.append(o)
    return kept


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


def render_program_map(document: Optional[dict], durations: Optional[Dict[str, int]] = None,
                       audio_features: Optional[Dict[str, dict]] = None) -> str:
    """Render the ASSEMBLED edit as two small, time-aligned tables (VIDEO,
    AUDIO) built from the fully-resolved layer stack (``layers.resolve``) --
    edso_pacing_audit_timing.plan.md item 2, replacing the old flat V1-then-V2
    line-per-line render. Every row carries a STABLE id (a seg_id or op_id)
    the brain can act on directly, plus its program window, layout, source
    ref, and a neutral label -- so stacking/overlap (a V2 cutaway over two V1
    spine cuts, a music bed under everything) is visible from the shared
    clock + z alone, with no prose needed. Generic: no speaker/role-of-person
    column, just the compositing structure. ``audio_features`` (file_id ->
    {integrated_lufs,...}, audio_and_audit.plan.md Phase 2) adds each audio
    row's own source loudness and a trailing GAPS line for any stretch with
    no audible layer at all -- omitted facts when not passed, never guessed.
    Returns "" when there is nothing to show yet."""
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
            lufs = ((audio_features or {}).get(a.source_file_id) or {}).get("integrated_lufs")
            if lufs is not None:
                tags.append(f"lufs:{lufs:.1f}")
            lines.append(f"  {lane} {aid}  {a.role}  {a.prog_start_ms}-{a.prog_end_ms}ms  "
                         f"{source}  {' '.join(tags)}")
        gaps = layers.audio_gaps(resolved)
        if gaps:
            lines.append("  GAPS (no audio): " +
                        ", ".join(f"{_ms(a)}-{_ms(b)}" for a, b in gaps))

    return "\n".join(lines)


def _ms(ms: int) -> str:
    s = max(0, int(ms)) // 1000
    return f"{s // 60}:{s % 60:02d}"
