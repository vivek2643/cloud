"""
Cuts v2: SAID tightness (breath removal).

Originally Phase B3 of ``cuts_v2.plan.md`` handled tightness for every
channel here, applied once on top of an already-partitioned, energy-
independent set of cuts. ``cuts_v2_boundaries.plan.md`` (Phase C1) re-scoped
the dial to do GRANULARITY too for video, which means video boundaries
themselves now depend on energy -- so video tightness moved INTO
``partition.py`` (``_tighten_video``), where the granularity split already
has to happen, rather than as a separate post-hoc pass. This module now
covers only what's left: ``said``-primary breath/dead-air excision into
``keep_spans`` (a jump-cut edit-list: same content, dead air removed) --
which never changes a cut's own claimed boundaries, so it's still a clean,
independent, post-hoc pass. A done/shown cut is returned UNCHANGED (already
tightened by ``partition_clip``); applying this twice would double-inset it.

The breath-excision math mirrors ``hero_cuts.py`` (v1) exactly, but is
RE-DERIVED here rather than imported: this module must stay usable after
Phase R guts the v1 speech/video overlap builders, so it takes no dependency
on hero_cuts internals.
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


def apply_tightness(cut: Cut, energy: float, *, words: Optional[List[dict]] = None) -> Cut:
    """A NEW Cut with SAID breath-tightness applied at ``energy`` (``cut`` is
    unchanged -- pure). A done/shown cut is returned as-is: ``partition_clip``
    already tightened it (granularity and tightness are decided together
    there now). ``words`` (the clip's flat word list) is needed only to
    excise breaths; omit it and the cut is returned unchanged too."""
    if cut.primary != vocab.CHANNEL_SAID:
        return cut
    params = energy_to_params(energy)
    gap = params.speech_breath_gap_ms or DEAD_AIR_FLOOR_MS
    keep = _breath_keep_spans(words, cut.src_in_ms, cut.src_out_ms, gap) if words else None
    return dataclasses.replace(cut, keep_spans=keep)


def apply_tightness_all(cuts: List[Cut], energy: float, *,
                        words_by_file: Optional[Dict[str, List[dict]]] = None) -> List[Cut]:
    """Apply said-breath tightness to every said-primary cut in a partition
    (across one or many files); done/shown cuts pass through unchanged.
    ``words_by_file`` maps file_id -> flat word list."""
    words_by_file = words_by_file or {}
    return [apply_tightness(c, energy, words=words_by_file.get(c.file_id)) for c in cuts]
