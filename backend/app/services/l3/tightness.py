"""
Cuts v2, Phase B3: TIGHTNESS ONLY.

Applies just the tightness axis of the energy dial on top of an already-
partitioned (detected, not chosen) set of cuts -- it never re-scopes a cut's
boundaries or which channel claimed it (that's ``partition.py``'s job, done
once, energy-independent; see the plan's North Star #5). Granularity params
(``speech_unit``, ``fuse_gap_ms``) are ignored here on purpose -- v2 has no
granularity slider, only tightness.

  * ``said`` primary       -> progressive breath/dead-air excision into
    ``keep_spans`` (a jump-cut edit-list: same content, dead air removed).
  * ``done``/``shown`` primary -> peak inset toward ``peak_ms`` (negative
    padding, proportional to the cut's own span).

The breath-excision and peak-inset math mirror ``hero_cuts.py``/``combine.py``
(v1) exactly, but are RE-DERIVED here rather than imported: this module must
stay usable after Phase R guts the v1 speech/video overlap builders, so it
takes no dependency on hero_cuts internals.

v1 ships with a FIXED default tightness (Balanced, energy=0.5) applied once at
partition time (see ``partition_params.DEFAULT_SNAP_ENERGY``); a slider wiring
this module per-cut is optional future work.
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Optional, Tuple

from app.services.l3 import vocab
from app.services.l3.energy import energy_to_params
from app.services.l3.partition import Cut

# Dead-air floor: even when the energy band's own progressive breath-removal
# is off (Broad..Tight), a said cut never plays a hole longer than this --
# mirrors hero_cuts._DEAD_AIR_FLOOR_MS.
DEAD_AIR_FLOOR_MS = 1500


def _breath_keep_spans(words: List[dict], lo: int, hi: int,
                       gap_ms: int) -> Optional[List[Tuple[int, int]]]:
    """Progressive breath removal: the spoken runs to KEEP inside [lo, hi],
    excising every internal silent gap >= ``gap_ms``. None when there's
    nothing to excise (the cut just plays contiguously)."""
    if gap_ms <= 0:
        return None
    span = sorted(
        (w for w in words
         if int(w.get("start_ms", 0)) < hi and int(w.get("end_ms", 0)) > lo),
        key=lambda w: int(w.get("start_ms", 0)))
    if len(span) < 2:
        return None
    spans: List[Tuple[int, int]] = []
    seg_start = lo
    for prev, nxt in zip(span, span[1:]):
        if int(nxt.get("start_ms", 0)) - int(prev.get("end_ms", 0)) >= gap_ms:
            seg_end = min(hi, int(prev.get("end_ms", 0)))
            if seg_end > seg_start:
                spans.append((seg_start, seg_end))
            seg_start = max(lo, int(nxt.get("start_ms", 0)))
    if hi > seg_start:
        spans.append((seg_start, hi))
    return spans if len(spans) > 1 else None


def _core_ms_for(primary: str, done_core_frac: Optional[float],
                 shown_core_frac: Optional[float], core_floor_ms: int,
                 span_ms: int) -> Optional[int]:
    frac = done_core_frac if primary == vocab.CHANNEL_DONE else shown_core_frac
    if frac is None:
        return None
    return max(core_floor_ms, int(round(frac * span_ms)))


def _core_inset(core_in: int, core_out: int, peak: int,
                target: Optional[int], *, lead_frac: float = 0.5) -> Tuple[int, int]:
    """Inset [core_in, core_out] toward ``peak`` to ``target`` ms length,
    keeping ``lead_frac`` of the window before the peak. Only ever shrinks;
    unchanged when ``target`` is None or the span is already shorter."""
    if not target or core_out - core_in <= target:
        return core_in, core_out
    peak = max(core_in, min(peak, core_out))
    lead = int(round(target * lead_frac))
    ci = max(core_in, peak - lead)
    co = min(core_out, ci + target)
    ci = max(core_in, co - target)   # rebalance if clipped at the tail
    return ci, co


def apply_tightness(cut: Cut, energy: float, *, words: Optional[List[dict]] = None) -> Cut:
    """A NEW Cut with tightness applied at ``energy`` (``cut`` is unchanged --
    pure). ``words`` (the clip's flat word list) is needed only to excise
    breaths on a ``said``-primary cut; omit it for done/shown cuts."""
    params = energy_to_params(energy)

    if cut.primary == vocab.CHANNEL_SAID:
        gap = params.speech_breath_gap_ms or DEAD_AIR_FLOOR_MS
        keep = _breath_keep_spans(words, cut.src_in_ms, cut.src_out_ms, gap) if words else None
        return dataclasses.replace(cut, keep_spans=keep)

    target = _core_ms_for(cut.primary, params.done_core_frac, params.shown_core_frac,
                          params.core_floor_ms, cut.src_out_ms - cut.src_in_ms)
    new_in, new_out = _core_inset(cut.src_in_ms, cut.src_out_ms, cut.peak_ms, target)
    if (new_in, new_out) == (cut.src_in_ms, cut.src_out_ms):
        return dataclasses.replace(cut)
    return dataclasses.replace(cut, src_in_ms=new_in, src_out_ms=new_out, keep_spans=None)


def apply_tightness_all(cuts: List[Cut], energy: float, *,
                        words_by_file: Optional[Dict[str, List[dict]]] = None) -> List[Cut]:
    """Apply tightness to every cut in a partition (across one or many
    files). ``words_by_file`` maps file_id -> flat word list, needed only
    when said-primary cuts are present."""
    words_by_file = words_by_file or {}
    return [apply_tightness(c, energy, words=words_by_file.get(c.file_id)) for c in cuts]
