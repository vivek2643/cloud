"""
Voice <-> camera binding (identity_map.plan.md Phase 1): which diarized voice
a camera's picture actually shows, derived from motion -- the person talking
moves more (gestures, head turns, mouth) during their own turns than a
listener does. Deterministic, physical, aggregated over the whole clip; the
LLM is never involved in this decision.

Two entry points:
  - `bind_file`: a lone camera (not in an outlook group) -- simple argmax
    over its own turns/energy, gated by a confidence margin.
  - `bind_outlook_group`: 2+ cameras sharing the SAME (re-based) turns
    (`sync.lattice_merge.authoritative_view` already put them on one clock)
    -- a bipartite max-weight assignment so two cameras in the same group
    can never bind to the same voice, which independent per-file argmax
    cannot guarantee (camera-motion leakage/noise could make two cameras
    both correlate best with the same voice).

Fail-open throughout: no turns, no energy, or a margin below `BIND_MARGIN`
all resolve to `voice=None` (unknown) rather than a forced guess -- the
caller then keeps the existing per-still `on_camera` flag untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from app.services.l3.diarize import Turn

# Below this normalized margin between the top and second-best voice, the
# binding is too close to call -- stay unknown rather than guess (identity_
# map.plan.md Phase 1's explicit tunable).
BIND_MARGIN = 0.15


@dataclass
class Binding:
    voice: Optional[str]
    confidence: float  # the winning margin; 0.0 when unbound


def mean_energy_by_voice(
    turns: List[Turn], action_energy: List[float], hop_ms: int,
) -> Dict[str, float]:
    """`mean_energy_during(file, voice)` for every voice appearing in
    `turns`, against this file's OWN `action_energy` envelope. A turn spans
    every hop whose `[h*hop_ms, (h+1)*hop_ms)` window overlaps it."""
    if not turns or not action_energy or hop_ms <= 0:
        return {}
    n_hops = len(action_energy)
    sums: Dict[str, float] = {}
    counts: Dict[str, int] = {}
    for start_ms, end_ms, voice in turns:
        if not voice:
            continue
        s, e = int(start_ms), int(end_ms)
        lo = max(0, s // hop_ms)
        hi = min(n_hops - 1, (e - 1) // hop_ms) if e > s else lo
        if hi < lo or lo >= n_hops:
            continue
        for h in range(lo, hi + 1):
            sums[voice] = sums.get(voice, 0.0) + action_energy[h]
            counts[voice] = counts.get(voice, 0) + 1
    return {v: sums[v] / counts[v] for v in sums if counts.get(v)}


def _margin(top_e: float, second_e: float) -> float:
    return (top_e - second_e) / (top_e + 1e-9) if top_e > 0 else 0.0


def bind_file(turns: List[Turn], action_energy: List[float], hop_ms: int) -> Binding:
    """A lone camera's bound voice: argmax mean-energy-during-turn, gated by
    `BIND_MARGIN`. A single-speaker file binds trivially (no ambiguity to
    measure a margin against)."""
    means = mean_energy_by_voice(turns, action_energy, hop_ms)
    if not means:
        return Binding(voice=None, confidence=0.0)
    if len(means) == 1:
        (only_voice,) = means.keys()
        return Binding(voice=only_voice, confidence=1.0)
    ranked = sorted(means.items(), key=lambda kv: kv[1], reverse=True)
    top_voice, top_e = ranked[0]
    margin = _margin(top_e, ranked[1][1])
    return Binding(voice=top_voice if margin >= BIND_MARGIN else None, confidence=margin)


def bind_outlook_group(
    member_ids: List[str],
    turns_by_file: Dict[str, List[Turn]],
    action_energy_by_file: Dict[str, List[float]],
    hop_ms_by_file: Dict[str, int],
) -> Dict[str, Binding]:
    """Bipartite refine (identity_map.plan.md Phase 1): builds the
    cameras x voices mean-energy matrix for one outlook group and solves a
    max-weight assignment so no two cameras in the group bind to the same
    voice. Falls back to independent `bind_file` per member when there's
    nothing to disambiguate (< 2 members, or no voices found at all)."""
    means_by_file: Dict[str, Dict[str, float]] = {}
    all_voices: set = set()
    for fid in member_ids:
        means = mean_energy_by_voice(
            turns_by_file.get(fid) or [], action_energy_by_file.get(fid) or [], hop_ms_by_file.get(fid, 0),
        )
        means_by_file[fid] = means
        all_voices.update(means.keys())
    voices = sorted(all_voices)

    if len(member_ids) < 2 or not voices:
        return {
            fid: bind_file(turns_by_file.get(fid) or [], action_energy_by_file.get(fid) or [], hop_ms_by_file.get(fid, 0))
            for fid in member_ids
        }

    import numpy as np
    from scipy.optimize import linear_sum_assignment

    matrix = np.array([[means_by_file[fid].get(v, 0.0) for v in voices] for fid in member_ids])
    # linear_sum_assignment MINIMIZES total cost; negate to maximize energy.
    row_idx, col_idx = linear_sum_assignment(-matrix)
    assigned_voice = {member_ids[r]: voices[c] for r, c in zip(row_idx, col_idx)}

    out: Dict[str, Binding] = {}
    for fid in member_ids:
        voice = assigned_voice.get(fid)
        row_means = means_by_file[fid]
        if voice is None or not row_means:
            out[fid] = Binding(voice=None, confidence=0.0)
            continue
        top_e = row_means.get(voice, 0.0)
        others = [e for v, e in row_means.items() if v != voice]
        margin = _margin(top_e, max(others) if others else 0.0)
        out[fid] = Binding(voice=voice if margin >= BIND_MARGIN else None, confidence=margin)
    return out
