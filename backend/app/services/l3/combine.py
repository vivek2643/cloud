"""
The uniform energy combiner -- the single deterministic step that turns gated,
folded ATOMS (l3.atoms) into hero cuts and capture-moments.

One algorithm for every video channel (no bespoke per-affordance engines):

  * Within-channel FUSE: same-channel atoms within `params.fuse_gap_ms` are one
    continuous beat (a demo run, a rally) -> one cut. The gap is the energy dial:
    wide at Broad (a whole run is one moment), 0 at Tight/Sharp (atomize -> each
    beat is its own punchy cut). "action+action fuses when continuous, not when
    it's just successive beats."
  * PEAK zoom: at Tight/Sharp the cut insets toward the atom's peak (the impact /
    reveal) to the band's handle length, and a Done beat may SPLIT at its impact
    (excise the field-safe lull -> windup|payoff jump-cut).
  * Every boundary flows through the fused seam field (`hero_cuts._snap_segment`),
    so a cut can never land inside a spoken word / camera move / impact.

SAID is not combined here -- it keeps the richer linguistic thought-ladder in
`hero_cuts._speech_candidates`; the caller passes those cuts straight through.

`derive_moments` then groups cuts ACROSS channels into capture-moments
(proximity + a shared actor / region / non-person subject). Narrative grouping
(a reaction answering a line three cuts away) is left to the brain.

Pure given the clip artifacts; `hero_cuts` is imported lazily to avoid an import
cycle (it imports this module).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.services.l3 import vocab
from app.services.l3.atoms import Atom
from app.services.l3.energy import EnergyParams

# Peak lead: a Done beat pays off at its impact (keep mostly the run-in), a held
# Shown subject is uniform (center is fine).
_DONE_LEAD = 0.0
_SHOWN_LEAD = 0.5

# Below this composite salience a video cut isn't worth surfacing.
_MIN_VIDEO_SCORE = 0.18

# Done split (Sharp): excise an interior motion lull only if both kept pieces and
# the lull itself clear these floors -- otherwise the beat plays contiguously.
_LULL_MIN_MS = 500
_SPLIT_MIN_KEEP_MS = 300


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _gap(a: Atom, b: Atom) -> int:
    """Silent gap between two time-sorted atoms (negative when they overlap)."""
    return b.start_ms - a.end_ms


def _cluster_by_gap(atoms: List[Atom], gap_ms: int) -> List[List[Atom]]:
    """Fuse a time-sorted, single-channel atom list into runs separated by more
    than `gap_ms`. gap_ms <= 0 -> only overlapping atoms group (atomize)."""
    out: List[List[Atom]] = []
    for a in sorted(atoms, key=lambda x: (x.start_ms, x.peak_ms)):
        if out and _gap(out[-1][-1], a) <= gap_ms:
            out[-1].append(a)
        else:
            out.append([a])
    return out


def _core_ms_for(channel: str, params: EnergyParams) -> Optional[int]:
    """The band's handle length for a channel (None below Tight = keep full)."""
    return params.action_core_ms if channel == vocab.CHANNEL_DONE else params.broll_core_ms


def _excise_lull(motion: Optional[dict], in_ms: int, out_ms: int,
                 peak_ms: int) -> Optional[List[Tuple[int, int]]]:
    """A Done beat's windup|payoff jump-cut: excise the longest interior run of
    LOW motion energy (a lull) that does not contain the peak, keeping the windup
    and the payoff. Field-safe by construction -- only quiet interior is removed,
    never the impact. None when there's no lull worth cutting."""
    if not motion:
        return None
    energy = motion.get("action_energy") or []
    hop = int(motion.get("hop_ms", 0))
    if not energy or hop <= 0:
        return None
    i0, i1 = max(0, in_ms // hop), min(len(energy), out_ms // hop)
    if i1 - i0 < 3:
        return None
    window = energy[i0:i1]
    thr = 0.4 * (sum(window) / len(window))   # quiet = well below the beat's mean
    pk_i = peak_ms // hop
    best: Optional[Tuple[int, int]] = None
    run_start = None
    for i in range(i0, i1):
        quiet = energy[i] <= thr and abs(i - pk_i) > 1
        if quiet and run_start is None:
            run_start = i
        elif not quiet and run_start is not None:
            if best is None or (i - run_start) > (best[1] - best[0]):
                best = (run_start, i)
            run_start = None
    if run_start is not None and (i1 - run_start) > (0 if best is None else best[1] - best[0]):
        best = (run_start, i1)
    if best is None:
        return None
    lull_a, lull_b = best[0] * hop, best[1] * hop
    if lull_b - lull_a < _LULL_MIN_MS:
        return None
    if lull_a - in_ms < _SPLIT_MIN_KEEP_MS or out_ms - lull_b < _SPLIT_MIN_KEEP_MS:
        return None
    return [(in_ms, lull_a), (lull_b, out_ms)]


def _video_score(channel: str, members: List[Atom], clip, in_ms: int, out_ms: int):
    """Composite 0..1 rank for a video cut. Confidence (the VLM's keep signal)
    carries it; Done blends in measured motion energy, Shown the VLM quality."""
    from app.services.l3 import hero_cuts as hc
    conf = sum(m.confidence for m in members) / max(1, len(members))
    quality_events = (clip.perception or {}).get("take_quality_events") or []
    vlm = hc._vlm_quality_score(quality_events, in_ms, out_ms)
    if channel == vocab.CHANNEL_DONE and clip.motion:
        base = 0.5 * conf + 0.5 * hc._action_score(clip.motion, in_ms, out_ms, vlm)
    elif vlm is not None:
        base = 0.6 * conf + 0.4 * vlm
    else:
        base = conf
    return max(0.0, min(1.0, base))


def _people_facet(actor: Optional[str], region: Optional[dict]) -> List[dict]:
    if not actor:
        return []
    p: dict = {"person_id": actor, "on_camera": True}
    if region:
        p["region"] = region
    return [p]


def combine_video(atoms: List[Atom], params: EnergyParams, field, clip,
                  source=None) -> List["object"]:
    """Build Done/Shown hero cuts from gated+folded atoms via the uniform fuse +
    peak-zoom + fused-field snap. Heard is suppressed (not surfaced); Said is
    handled by the caller's thought-ladder."""
    from app.services.l3 import hero_cuts as hc
    from app.services.l3 import anchors as anc

    out: List[object] = []
    for channel in (vocab.CHANNEL_DONE, vocab.CHANNEL_SHOWN):
        group = [a for a in atoms if a.channel == channel]
        if not group:
            continue
        aff = anc.AFF_ACTION if channel == vocab.CHANNEL_DONE else anc.AFF_BROLL
        lead = _DONE_LEAD if channel == vocab.CHANNEL_DONE else _SHOWN_LEAD
        for ci, members in enumerate(_cluster_by_gap(group, params.fuse_gap_ms)):
            core_in = min(m.start_ms for m in members)
            core_out = max(m.end_ms for m in members)
            best = max(members, key=lambda m: m.confidence)

            # Peak SPLIT (Done, Sharp): windup|payoff around the impact -- excise
            # the field-safe interior lull of the FULL beat (before any inset, so
            # there's a lull left to find). Otherwise peak ZOOM: inset toward the
            # impact to the band's handle length.
            keep_spans = None
            if channel == vocab.CHANNEL_DONE and params.action_split_at_impact and len(members) == 1:
                split = _excise_lull(clip.motion, core_in, core_out, best.peak_ms)
                if split:
                    keep_spans = [hc._snap_segment(field, a, b, params, clip, aff) for a, b in split]
                    in_ms, out_ms = keep_spans[0][0], keep_spans[-1][1]
            if keep_spans is None:
                cin, cout = hc._core_inset(core_in, core_out, best.peak_ms,
                                           _core_ms_for(channel, params), lead_frac=lead)
                in_ms, out_ms = hc._snap_segment(field, cin, cout, params, clip, aff)
            if out_ms <= in_ms:
                continue
            score = _video_score(channel, members, clip, in_ms, out_ms)
            if score < _MIN_VIDEO_SCORE:
                continue
            label = (best.label or vocab.channel_label(channel)).strip()[:200]
            ladder = [hc.Rung(_ladder_level(params.band),
                              keep_spans or [(in_ms, out_ms)], text=label, score=score)]
            out.append(hc.HeroCut(
                hero_id=f"{clip.file_id[:8]}:{channel[:2]}{ci}",
                file_id=clip.file_id,
                modality=channel,
                label=label,
                src_in_ms=in_ms,
                src_out_ms=out_ms,
                score=score,
                speaker=best.actor,
                flags=list(best.flags or []),
                affordances=[channel],
                keep_spans=keep_spans,
                ladder=ladder,
                people=_people_facet(best.actor, best.region),
                framing=hc._framing_facet(clip, in_ms, out_ms, best.region),
                summary=best.summary,
                primitive=best.subject,
                channel=channel,
                subject=best.subject,
            ))
    return out


def _ladder_level(band: int) -> str:
    levels = ("broad", "calm", "balanced", "tight", "sharp")
    return levels[max(0, min(band, len(levels) - 1))]


# --------------------------------------------------------------------------
# Capture-moments: cross-channel fusion (proximity + shared subject/actor/region)
# --------------------------------------------------------------------------

def _region_overlap(ra: Optional[dict], rb: Optional[dict]) -> bool:
    if not ra or not rb:
        return False
    ax0, ay0 = float(ra.get("x", 0)), float(ra.get("y", 0))
    ax1, ay1 = ax0 + float(ra.get("w", 0)), ay0 + float(ra.get("h", 0))
    bx0, by0 = float(rb.get("x", 0)), float(rb.get("y", 0))
    bx1, by1 = bx0 + float(rb.get("w", 0)), by0 + float(rb.get("h", 0))
    return max(0.0, min(ax1, bx1) - max(ax0, bx0)) > 0 and \
        max(0.0, min(ay1, by1) - max(ay0, by0)) > 0


def _cut_region(h) -> Optional[dict]:
    return (h.framing or {}).get("region") if h.framing else None


def _actor_id(h) -> Optional[str]:
    """The person local_id this cut is about, from its people facet (works for a
    speech cut and a video cut alike -- the raw `speaker` field uses different
    namespaces, the cast-resolved person_id is the common key)."""
    for p in (h.people or []):
        pid = p.get("person_id")
        if pid:
            return pid
    return None


def _shares_capture(a, b) -> bool:
    """Two cuts belong to the SAME captured moment when they share the human
    performing it (actor), the patch of frame they happen in (region), or -- for
    non-person subjects -- the same thing on screen. Person<->person across
    different actors is left to the brain (a podcast stays moment-free)."""
    ai, bi = _actor_id(a), _actor_id(b)
    if ai and ai == bi:
        return True
    if _region_overlap(_cut_region(a), _cut_region(b)):
        return True
    sa, sb = getattr(a, "subject", None), getattr(b, "subject", None)
    if sa and sa == sb and sa != vocab.SUBJECT_PERSON:
        return True
    return False


def _time_gap(a, b) -> int:
    return max(b.src_in_ms, a.src_in_ms) - min(b.src_out_ms, a.src_out_ms)


def derive_moments(cuts: List["object"], params: EnergyParams) -> None:
    """Stamp `moment_id` on cuts that form a cross-channel capture-moment: cuts
    on DIFFERENT channels, proximate within `params.fuse_gap_ms`, that share a
    capture (actor / region / non-person subject). A cluster needs >= 2 distinct
    channels to be a moment, so adjacent same-channel beats never masquerade as
    one (a smash + a line = a moment; two lines are not). Mutates in place."""
    parent: Dict[int, int] = {i: i for i in range(len(cuts))}

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        parent[find(i)] = find(j)

    reach = max(0, params.fuse_gap_ms)
    for i in range(len(cuts)):
        for j in range(i + 1, len(cuts)):
            a, b = cuts[i], cuts[j]
            if getattr(a, "channel", None) == getattr(b, "channel", None):
                continue
            if _time_gap(a, b) > reach:
                continue
            if _shares_capture(a, b):
                union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(len(cuts)):
        clusters.setdefault(find(i), []).append(i)

    m = 0
    for members in clusters.values():
        if len(members) < 2:
            continue
        channels = {getattr(cuts[i], "channel", None) for i in members}
        if len(channels) < 2:
            continue
        file_id = cuts[members[0]].file_id
        mid = f"{file_id[:8]}:m{m}"
        m += 1
        for i in members:
            cuts[i].moment_id = mid
