"""
The uniform energy combiner -- the single deterministic step that turns gated,
folded ATOMS (l3.atoms) into hero cuts.

One algorithm for every video channel (no bespoke per-affordance engines):

  * Within-channel FUSE: same-channel atoms within `params.fuse_gap_ms` are one
    continuous beat (a demo run, a rally) -> one cut. The gap is the energy dial:
    wide at Broad (a whole run is one moment), 0 at Tight/Sharp (atomize -> each
    beat is its own punchy cut). "action+action fuses when continuous, not when
    it's just successive beats."
  * PEAK zoom: at Tight/Sharp the cut insets toward the atom's peak (the impact /
    reveal) to the band's handle length. At Tight a beat may instead SPLIT at its
    impact (excise the field-safe lull -> windup|payoff jump-cut) when that lands
    tighter; Sharp never splits -- it is the pure, tightest banger.
  * Every boundary flows through the fused seam field (`hero_cuts._snap_segment`),
    so a cut can never land inside a spoken word / camera move / impact.

SAID is not combined here -- it keeps the richer linguistic thought-ladder in
`hero_cuts._speech_candidates`; the caller passes those cuts straight through.

Cross-channel grouping (a line + its reaction + illustrating b-roll) is left to
the brain, which reads same-clip overlapping timestamps directly off the map.

Pure given the clip artifacts; `hero_cuts` is imported lazily to avoid an import
cycle (it imports this module).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.services.l3 import energy as en
from app.services.l3 import vocab
from app.services.l3.atoms import Atom
from app.services.l3.energy import EnergyParams

# The five ladder bands a video cut is zoomed at (broad .. sharp).
_LADDER_BANDS = (0, 1, 2, 3, 4)
_ANCHOR_LADDER_BAND = 2          # balanced -- the flat span the cut plays at

# Peak lead: a Done beat pays off at its impact (keep mostly the run-in), a held
# Shown subject is uniform (center is fine).
_DONE_LEAD = 0.0
_SHOWN_LEAD = 0.5

# Below this composite salience a video cut isn't worth surfacing.
_MIN_VIDEO_SCORE = 0.18

# Windup|payoff split (Tight): excise an interior motion lull only if both kept
# pieces and the lull itself clear these floors -- otherwise the beat is contiguous.
_LULL_MIN_MS = 500
_SPLIT_MIN_KEEP_MS = 300


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


def _core_ms_for(channel: str, params: EnergyParams, span_ms: int) -> Optional[int]:
    """The band's negative-padding handle for a video beat, as a span-proportional
    length (None below Tight = keep full). The handle scales with the beat's own
    duration -- a long shot keeps proportionally more than a short one -- floored
    so a tiny beat never insets below a usable handle."""
    frac = params.done_core_frac if channel == vocab.CHANNEL_DONE else params.shown_core_frac
    if frac is None:
        return None
    return max(params.core_floor_ms, int(round(frac * span_ms)))


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


def _video_span_at(level_params: EnergyParams, channel: str, core_in: int,
                   core_out: int, peak_ms: int, lead: float, field, clip,
                   *, can_split: bool) -> Tuple[int, int, Optional[List[Tuple[int, int]]]]:
    """Zoom ONE beat at ONE band -> (in_ms, out_ms, keep_spans|None).

    The beat insets toward its peak to the band's span-proportional handle (None
    below Tight = keep the full beat). At TIGHT (split_at_peak) a single-member
    beat may instead play a windup|payoff jump-cut -- excise the field-safe dead
    interior lull from the full beat, keeping both ends -- but only when that
    lands TIGHTER than the plain inset (a genuinely big dead middle). So a rung is
    never looser than its own inset, and Sharp (which never splits) stays the pure
    banger: the ladder is monotonic broad..sharp. Every edge flows through the
    fused seam field. This is the per-rung primitive the ladder loop calls."""
    from app.services.l3 import hero_cuts as hc
    cin, cout = hc._core_inset(core_in, core_out, peak_ms,
                               _core_ms_for(channel, level_params, core_out - core_in),
                               lead_frac=lead)
    in_ms, out_ms = hc._snap_segment(field, cin, cout, level_params, clip, channel)
    if level_params.split_at_peak and can_split:
        split = _excise_lull(clip.motion, core_in, core_out, peak_ms)
        if split:
            ks = [hc._snap_segment(field, a, b, level_params, clip, channel) for a, b in split]
            if sum(b - a for a, b in ks) < (out_ms - in_ms):   # jump-cut only if tighter
                return ks[0][0], ks[-1][1], ks
    return in_ms, out_ms, None


def combine_video(atoms: List[Atom], params: EnergyParams, field, clip,
                  source=None) -> List["object"]:
    """Build Done/Shown hero cuts from gated+folded atoms via the uniform fuse +
    peak-zoom + fused-field snap. Heard is suppressed (not surfaced); Said is
    handled by the caller's thought-ladder."""
    from app.services.l3 import hero_cuts as hc

    out: List[object] = []
    for channel in (vocab.CHANNEL_DONE, vocab.CHANNEL_SHOWN):
        group = [a for a in atoms if a.channel == channel]
        if not group:
            continue
        lead = _DONE_LEAD if channel == vocab.CHANNEL_DONE else _SHOWN_LEAD
        for ci, members in enumerate(_cluster_by_gap(group, params.fuse_gap_ms)):
            core_in = min(m.start_ms for m in members)
            core_out = max(m.end_ms for m in members)
            best = max(members, key=lambda m: m.confidence)
            can_split = len(members) == 1   # a windup|payoff split needs one beat
            label = (best.label or vocab.channel_label(channel)).strip()[:200]

            # Build the OWNED broad..sharp ladder by zooming this SAME beat at each
            # band (mirrors the speech thought-ladder): Broad/Calm/Balanced keep
            # the full beat, Tight/Sharp inset toward the peak, Tight may jump-cut
            # (windup|payoff). The cut's identity (the fused beat) is fixed; only
            # the zoom varies.
            ladder = []
            flat: Optional[Tuple[int, int, Optional[List[Tuple[int, int]]], float]] = None
            for band in _LADDER_BANDS:
                lp = en.params_for_band(band)
                i, o, ks = _video_span_at(lp, channel, core_in, core_out,
                                          best.peak_ms, lead, field, clip, can_split=can_split)
                if o <= i:
                    continue
                rscore = _video_score(channel, members, clip, i, o)
                ladder.append(hc.Rung(_ladder_level(band), ks or [(i, o)],
                                      text=label, score=rscore))
                if band == params.band:        # flat span follows the called band
                    flat = (i, o, ks, rscore)

            # The cut plays at the called band's rung (like a speech cut plays its
            # selected level); fall back to Balanced, then the widest rung, if that
            # band's zoom collapsed.
            if flat is None:
                pick = next((r for r in ladder if r.level == _ladder_level(_ANCHOR_LADDER_BAND)),
                            ladder[0] if ladder else None)
                if pick is None:
                    continue
                flat = (pick.in_ms(), pick.out_ms(), pick.keep_spans(), pick.score)
            in_ms, out_ms, keep_spans, score = flat
            if out_ms <= in_ms or score < _MIN_VIDEO_SCORE:
                continue

            out.append(hc.HeroCut(
                hero_id=f"{clip.file_id[:8]}:{channel[:2]}{ci}",
                file_id=clip.file_id,
                channel=channel,
                label=label,
                src_in_ms=in_ms,
                src_out_ms=out_ms,
                score=score,
                speaker=best.actor,
                flags=list(best.flags or []),
                keep_spans=keep_spans,
                ladder=ladder,
                people=_people_facet(best.actor, best.region),
                framing=hc._framing_facet(clip, in_ms, out_ms, best.region),
                summary=best.summary,
                subject=best.subject,
            ))
    return out


def _ladder_level(band: int) -> str:
    levels = ("broad", "calm", "balanced", "tight", "sharp")
    return levels[max(0, min(band, len(levels) - 1))]
