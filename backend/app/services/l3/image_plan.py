"""
Cuts v3: the deterministic IMAGE PLAN. No model call -- this falls straight
out of pass 1's output plus signals L1 already computed (blur, action energy,
composition drift). Decides exactly which frames pass 2 needs to see, and
what each one is FOR, so pass 2 never has to guess what a numbered image
means. See cuts_v3.plan.md section 4/5.

Per-clip frame budget, priority order when a clip's plan is over budget
(drop the lowest tier first -- never drop a higher tier to make room for a
lower one):

    take members  >  speech cuts  >  anchored video groups
                  >  unanchored video groups  >  composition-drift extras

Take members are never dropped: eye-to-eye comparison across a take is the
one thing pass 2 cannot do without seeing every candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from app.services.l3.lattice import Lattice, resolve_speech_span_ms
from app.services.l3.pass1 import Pass1Output
from app.services.l3.video_segments import _sharpest_ms

FRAME_BUDGET_PER_CLIP = 24

REASON_TAKE_MEMBER = "take_member"
REASON_SPEECH_CUT = "speech_cut"
REASON_VIDEO_GROUP_ANCHOR = "video_group_anchor"
REASON_VIDEO_GROUP_CALM = "video_group_calm"
REASON_COMPOSITION_DRIFT = "composition_drift"

_TIER_INDEX = {
    REASON_TAKE_MEMBER: 0,
    REASON_SPEECH_CUT: 1,
    REASON_VIDEO_GROUP_ANCHOR: 2,
    REASON_VIDEO_GROUP_CALM: 3,
    REASON_COMPOSITION_DRIFT: 4,
}
_NUM_TIERS = 5


@dataclass
class PlannedFrame:
    file_id: str
    ts_ms: int
    reason: str   # one of the REASON_* constants above
    ref: str      # human-readable label of what this frame is FOR, e.g. "speech_cut[2]"

    def to_dict(self) -> Dict[str, Any]:
        return {"file_id": self.file_id, "ts_ms": self.ts_ms, "reason": self.reason, "ref": self.ref}


def _word_span_ms(lattices: Dict[str, Lattice], silences_by_file: Dict[str, List[dict]],
                   file_id: str, word_span: Tuple[int, int]) -> Tuple[int, int]:
    lattice = lattices[file_id]
    silences = silences_by_file.get(file_id, [])
    return resolve_speech_span_ms(lattice.words, lattice.atoms, word_span, silences)


def _atom_group_span(lattices: Dict[str, Lattice], file_id: str,
                      atom_ids: List[int]) -> Tuple[int, int, List[int]]:
    """(start_ms, end_ms, sorted anchor_ms union) over the given atom ids.
    (0, 0, []) when none of the ids resolve (stale/malformed model output)."""
    atoms_by_id = {a.atom_id: a for a in lattices[file_id].atoms}
    members = [atoms_by_id[i] for i in atom_ids if i in atoms_by_id]
    if not members:
        return 0, 0, []
    s = min(a.start_ms for a in members)
    e = max(a.end_ms for a in members)
    anchors = sorted({t for a in members for t in a.anchor_ms})
    return s, e, anchors


def _calm_and_sharp_ms(motion: Dict[str, Any], s: int, e: int, default_ms: int) -> int:
    """Fallback still for an unanchored video group: the instant in [s, e)
    that minimizes action_energy + blur together (a calm, in-focus moment) --
    there's no impact/audio onset to pin the still to, so pick the steadiest
    one instead of an arbitrary midpoint."""
    hop = int(motion.get("hop_ms") or 0)
    if hop <= 0:
        return default_ms
    action = motion.get("action_energy") or []
    blur = motion.get("blur") or []
    lo, hi = max(0, s // hop), max(s // hop, (e - 1) // hop)
    n = max(len(action), len(blur))
    hi = min(hi, n - 1)
    if hi < lo:
        return default_ms
    best_i = min(
        range(lo, hi + 1),
        key=lambda i: (action[i] if i < len(action) else 0.0) + (blur[i] if i < len(blur) else 0.0),
    )
    return best_i * hop


def build_image_plan(
    pass1: Pass1Output,
    lattices: Dict[str, Lattice],
    motion_by_file: Dict[str, Dict[str, Any]],
    scene_by_file: Dict[str, Dict[str, Any]],
    silences_by_file: Dict[str, List[dict]],
) -> List[PlannedFrame]:
    """Turn pass 1's output into a concrete, budgeted list of frames to pull
    and hand to pass 2. Deterministic: same inputs always produce the same
    plan. Files absent from ``lattices`` are silently skipped (not yet
    ingest-ready)."""
    tiers_by_file: Dict[str, List[List[PlannedFrame]]] = {}

    def tiers_for(file_id: str) -> List[List[PlannedFrame]]:
        return tiers_by_file.setdefault(file_id, [[] for _ in range(_NUM_TIERS)])

    # Take members: always kept, highest priority.
    for tc in pass1.take_candidates:
        for m in tc.members:
            if m.file_id not in lattices:
                continue
            s, e = _word_span_ms(lattices, silences_by_file, m.file_id, m.word_span)
            motion = motion_by_file.get(m.file_id, {})
            ts = _sharpest_ms(motion.get("blur") or [], int(motion.get("hop_ms") or 0), s, e, (s + e) // 2)
            frame = PlannedFrame(m.file_id, ts, REASON_TAKE_MEMBER, f"take[{tc.group_id}]")
            tiers_for(m.file_id)[_TIER_INDEX[REASON_TAKE_MEMBER]].append(frame)

    # Speech cuts (+ any composition-drift point that falls inside one).
    for i, sc in enumerate(pass1.speech_cuts):
        if sc.file_id not in lattices:
            continue
        s, e = _word_span_ms(lattices, silences_by_file, sc.file_id, sc.word_span)
        motion = motion_by_file.get(sc.file_id, {})
        ts = _sharpest_ms(motion.get("blur") or [], int(motion.get("hop_ms") or 0), s, e, (s + e) // 2)
        ref = f"speech_cut[{i}]"
        tiers_for(sc.file_id)[_TIER_INDEX[REASON_SPEECH_CUT]].append(
            PlannedFrame(sc.file_id, ts, REASON_SPEECH_CUT, ref)
        )
        drift_points = (scene_by_file.get(sc.file_id, {}) or {}).get("composition_points") or []
        for p in drift_points:
            pts = int(p.get("ts_ms", -1))
            if s < pts < e:
                tiers_for(sc.file_id)[_TIER_INDEX[REASON_COMPOSITION_DRIFT]].append(
                    PlannedFrame(sc.file_id, pts, REASON_COMPOSITION_DRIFT, ref)
                )

    # Video tentative groups: one frame per anchor if any, else one calm+sharp fallback.
    for gi, vg in enumerate(pass1.video_tentative_groups):
        if vg.file_id not in lattices:
            continue
        s, e, anchors = _atom_group_span(lattices, vg.file_id, vg.atom_ids)
        if e <= s:
            continue
        ref = f"video_group[{gi}]"
        if anchors:
            for a_ts in anchors:
                tiers_for(vg.file_id)[_TIER_INDEX[REASON_VIDEO_GROUP_ANCHOR]].append(
                    PlannedFrame(vg.file_id, a_ts, REASON_VIDEO_GROUP_ANCHOR, ref)
                )
        else:
            motion = motion_by_file.get(vg.file_id, {})
            ts = _calm_and_sharp_ms(motion, s, e, (s + e) // 2)
            tiers_for(vg.file_id)[_TIER_INDEX[REASON_VIDEO_GROUP_CALM]].append(
                PlannedFrame(vg.file_id, ts, REASON_VIDEO_GROUP_CALM, ref)
            )

    # Apply the per-clip budget, tier by tier, highest priority first.
    out: List[PlannedFrame] = []
    for file_id, tiers in tiers_by_file.items():
        budget = FRAME_BUDGET_PER_CLIP
        for tier in tiers:
            if budget <= 0:
                break
            take = tier[:budget]
            out.extend(take)
            budget -= len(take)
    return out
