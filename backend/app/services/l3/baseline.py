"""
Deterministic multicam baseline.

Given a sync group's angle menu (`angles.build_angle_menu`) and the active
principle weights, seed a competent speaker/reaction-aware angle cut. This is the
ANTI-COLLAPSE mechanism: instead of staring at one backbone camera, the picture
follows whoever holds the floor and cuts to a strong reaction when one beats the
talker -- all weighted by principles, never hard-coded.

The output is ADVISORY: a list of rough (from_ms, to_ms, file_id, rationale)
picks that the caller turns into `pick_angle` ops through the normal executor
(so they're snapped + validated identically). Opus then keeps, tweaks, or wipes
them. Picks where the chosen angle is already the spine's own picture are
dropped (nothing to switch).
"""
from __future__ import annotations

from typing import Dict, List, Optional

from app.services.l3 import layers, principles
from app.services.l3.angles import AngleOffer, AngleWindow
from app.services.l3.sync import SyncGroup

# Shot sizes that read as a good "talking" framing.
_GOOD_TALK_SHOTS = {"close_up", "medium_close_up", "medium"}

_SCORE_KNOBS = ("favor_speaker", "reward_reaction", "prefer_well_framed", "shot_variety")


def _framing_quality(offer: AngleOffer) -> float:
    fr = 0.0
    if offer.shot_size in _GOOD_TALK_SHOTS:
        fr += 0.6
    elif offer.shot_size:
        fr += 0.3
    if offer.gaze_to_camera:
        fr += 0.4
    return min(1.0, fr)


def _score(offer: AngleOffer, prev_shot: Optional[str], w: Dict[str, float]) -> float:
    s = 0.0
    if offer.speaker_in_frame:
        s += w["favor_speaker"]
    if offer.reaction:
        s += w["reward_reaction"] * offer.reaction[1]
    s += w["prefer_well_framed"] * _framing_quality(offer)
    if prev_shot is not None and offer.shot_size and offer.shot_size != prev_shot:
        s += w["shot_variety"] * 0.5
    return s


def _reason(offer: AngleOffer) -> str:
    if offer.reaction:
        return f"reaction {offer.reaction[0]} ({offer.reaction[1]:.1f})"
    if offer.speaker_in_frame:
        return "speaker on camera"
    if offer.shot_size:
        return f"framing ({offer.shot_size})"
    return "angle variety"


def seed_multicam(
    spans: List["layers.SpineSpan"],
    group: SyncGroup,
    menu: List[AngleWindow],
    document: dict,
) -> List[dict]:
    """Advisory list of rough pick_angle dicts {from_ms,to_ms,file_id,rationale}."""
    w = {k: principles.weight_of(document, k) for k in _SCORE_KNOBS}
    # pace: higher => faster cuts => shorter minimum hold (1.0s .. 2.5s).
    pace = principles.weight_of(document, "pace")
    min_hold_ms = int(2500 - 1500 * max(0.0, min(1.0, pace)))

    # 1. best angle per window (windows with offers only)
    seq: List[list] = []  # [start, end, file_id, shot, reason]
    prev_shot: Optional[str] = None
    for win in menu:
        if not win.offers:
            continue
        best = max(win.offers, key=lambda o: _score(o, prev_shot, w))
        seq.append([win.prog_start_ms, win.prog_end_ms, best.file_id, best.shot_size, _reason(best)])
        prev_shot = best.shot_size

    # 2. coalesce adjacent same-file runs
    coalesced: List[list] = []
    for item in seq:
        if coalesced and coalesced[-1][2] == item[2] and item[0] <= coalesced[-1][1] + 1:
            coalesced[-1][1] = item[1]
        else:
            coalesced.append(item[:])

    # 3. anti-flicker: fold too-short runs into the previous pick
    merged: List[list] = []
    for item in coalesced:
        if merged and (item[1] - item[0]) < min_hold_ms:
            merged[-1][1] = item[1]
        else:
            merged.append(item[:])

    # 4. emit picks only where the chosen angle differs from the spine's own
    #    picture at the midpoint (otherwise the spine already shows it)
    picks: List[dict] = []
    for st, en, fid, _shot, reason in merged:
        m = layers.prog_to_source(spans, (st + en) // 2)
        spine_file = m[0]["file_id"] if m else None
        if fid == spine_file:
            continue
        picks.append({"from_ms": st, "to_ms": en, "file_id": fid, "rationale": f"auto-multicam: {reason}"})
    return picks
