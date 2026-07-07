"""
Cuts v3: the deterministic IMAGE PLAN. No model call -- this falls straight
out of pass 1's output plus signals L1 already computed (blur, action energy,
composition drift). Decides exactly which frames pass 2 needs to see, and
what each one is FOR, so pass 2 never has to guess what a numbered image
means. See cuts_v3.plan.md section 4/5.

Every pass-1 unit (speech cut, video group, take member) gets AT LEAST one
frame, unconditionally: pass 2b's merge requires a visual judgment for every
cut, and a cut the model never saw pixels for can't be judged at all --
observed against a real 11-minute clip, the old per-clip budget truncated
whole tiers and the run died in pass 2b with "no images resolved". The
budget only ever trims EXTRAS beyond that floor, priority order when over
budget (drop the lowest tier first):

    extra anchor frames (beyond a group's first)  >  composition-drift extras
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
    # (mandatory frames, extra-anchor frames, drift frames) per clip.
    # Mandatory = one frame per pass-1 unit, NEVER dropped (see module
    # docstring); the budget only trims the two extras tiers.
    mandatory_by_file: Dict[str, List[PlannedFrame]] = {}
    extra_anchors_by_file: Dict[str, List[PlannedFrame]] = {}
    drift_by_file: Dict[str, List[PlannedFrame]] = {}

    # Take members: one frame each, mandatory.
    for tc in pass1.take_candidates:
        for m in tc.members:
            if m.file_id not in lattices:
                continue
            s, e = _word_span_ms(lattices, silences_by_file, m.file_id, m.word_span)
            motion = motion_by_file.get(m.file_id, {})
            ts = _sharpest_ms(motion.get("blur") or [], int(motion.get("hop_ms") or 0), s, e, (s + e) // 2)
            mandatory_by_file.setdefault(m.file_id, []).append(
                PlannedFrame(m.file_id, ts, REASON_TAKE_MEMBER, f"take[{tc.group_id}]"))

    # Speech cuts: one frame each, mandatory (+ drift extras inside the span).
    for i, sc in enumerate(pass1.speech_cuts):
        if sc.file_id not in lattices:
            continue
        s, e = _word_span_ms(lattices, silences_by_file, sc.file_id, sc.word_span)
        motion = motion_by_file.get(sc.file_id, {})
        ts = _sharpest_ms(motion.get("blur") or [], int(motion.get("hop_ms") or 0), s, e, (s + e) // 2)
        ref = f"speech_cut[{i}]"
        mandatory_by_file.setdefault(sc.file_id, []).append(
            PlannedFrame(sc.file_id, ts, REASON_SPEECH_CUT, ref))
        drift_points = (scene_by_file.get(sc.file_id, {}) or {}).get("composition_points") or []
        for p in drift_points:
            pts = int(p.get("ts_ms", -1))
            if s < pts < e:
                drift_by_file.setdefault(sc.file_id, []).append(
                    PlannedFrame(sc.file_id, pts, REASON_COMPOSITION_DRIFT, ref))

    # Video tentative groups: first frame mandatory (first anchor, else the
    # calm+sharp instant); anchors beyond the first are budgeted extras.
    for gi, vg in enumerate(pass1.video_tentative_groups):
        if vg.file_id not in lattices:
            continue
        s, e, anchors = _atom_group_span(lattices, vg.file_id, vg.atom_ids)
        if e <= s:
            continue
        ref = f"video_group[{gi}]"
        if anchors:
            mandatory_by_file.setdefault(vg.file_id, []).append(
                PlannedFrame(vg.file_id, anchors[0], REASON_VIDEO_GROUP_ANCHOR, ref))
            for a_ts in anchors[1:]:
                extra_anchors_by_file.setdefault(vg.file_id, []).append(
                    PlannedFrame(vg.file_id, a_ts, REASON_VIDEO_GROUP_ANCHOR, ref))
        else:
            motion = motion_by_file.get(vg.file_id, {})
            ts = _calm_and_sharp_ms(motion, s, e, (s + e) // 2)
            mandatory_by_file.setdefault(vg.file_id, []).append(
                PlannedFrame(vg.file_id, ts, REASON_VIDEO_GROUP_CALM, ref))

    out: List[PlannedFrame] = []
    all_files = set(mandatory_by_file) | set(extra_anchors_by_file) | set(drift_by_file)
    for file_id in all_files:
        mandatory = mandatory_by_file.get(file_id, [])
        out.extend(mandatory)
        budget = max(0, FRAME_BUDGET_PER_CLIP - len(mandatory))
        for tier in (extra_anchors_by_file.get(file_id, []), drift_by_file.get(file_id, [])):
            take = tier[:budget]
            out.extend(take)
            budget -= len(take)
    return out
