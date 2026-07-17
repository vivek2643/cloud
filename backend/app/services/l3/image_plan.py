"""
Cuts v3: the deterministic IMAGE PLAN. No model call -- this falls straight
out of pass 1's output plus signals L1 already computed (blur, action energy,
composition drift). Decides exactly which frames pass 2 needs to see, and
what each one is FOR, so pass 2 never has to guess what a numbered image
means. See cuts_v3.plan.md section 4/5.

Every pass-1 unit (speech cut, video group, take member) gets AT LEAST one
frame, unconditionally: pass 2's merge requires a visual judgment for every
cut, and a cut the model never saw pixels for can't be judged at all --
observed against a real 11-minute clip, the old per-clip budget truncated
whole tiers and the run died in pass 2 with "no images resolved". The
budget only ever trims EXTRAS beyond that floor, priority order when over
budget (drop the lowest tier first):

    2nd (early/late) moment  >  extra anchor frames (beyond a group's first)  >  composition-drift extras

perception_upgrade.plan.md Part B: a unit whose span is long enough (by that
CLIP's OWN unit-length distribution, never a hardcoded ms) gets a SECOND
frame -- the sharpest instant in the second half of its span, alongside the
first half's -- so pass 2 can perceive change over time, not just one still.
A short/runt unit (or one whose two candidate instants would be
near-duplicates) stays single-frame; this is the "runt guard", entirely
code-owned and deterministic.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.lattice import Lattice, resolve_speech_span_ms
from app.services.l3.pass1 import Pass1Output
from app.services.l3.v4_segment_params import MAX_CLUSTER_EVENT_FRAMES
from app.services.l3.video_segments import _sharpest_ms

FRAME_BUDGET_PER_CLIP = 40

REASON_TAKE_MEMBER = "take_member"
REASON_SPEECH_CUT = "speech_cut"
REASON_VIDEO_GROUP_ANCHOR = "video_group_anchor"
REASON_VIDEO_GROUP_CALM = "video_group_calm"
REASON_COMPOSITION_DRIFT = "composition_drift"
# cuts_v4_segmentation.plan.md section 7: for a V4 video cut with a "point"
# salience, the two frames straddle the peak (one shortly before, one shortly
# after) instead of the generic early/late halves -- so pass 2 can actually
# tell before/after/both apart, since it never sees a timestamp.
REASON_SHAPE_STRADDLE = "shape_straddle"
_STRADDLE_OFFSET_MS = 400
# v4_cluster_tree_cuts.plan.md section 8: a multi-event cluster gets one
# frame per (evenly-sampled) event peak instead of the single-peak straddle,
# so the model sees every sub-moment even though it labels the cluster once.
REASON_CLUSTER_EVENT = "cluster_event"
# The 2nd (early/late pairing) frame for a unit that clears the runt guard --
# its own extras tier, ranked above extra_anchors/drift but below the 1st
# (mandatory, never-dropped) frame per unit (module docstring).
REASON_SECOND_MOMENT = "second_moment"


@dataclass
class PlannedFrame:
    file_id: str
    ts_ms: int
    reason: str   # one of the REASON_* constants above
    ref: str      # human-readable label of what this frame is FOR, e.g. "speech_cut[2]"
    # "only" (this ref's one frame) | "early" | "late" -- perception_upgrade.
    # plan.md Part B. A unit that clears the runt guard gets an "early"
    # mandatory frame + a "late" REASON_SECOND_MOMENT candidate; if that
    # candidate is later dropped by budget pressure, the survivor is
    # relabeled "only" (see _relabel_orphaned_phases) so pass2's caption
    # never claims a partner that isn't there.
    phase: str = "only"

    def to_dict(self) -> Dict[str, Any]:
        return {"file_id": self.file_id, "ts_ms": self.ts_ms, "reason": self.reason,
                "ref": self.ref, "phase": self.phase}


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


def _is_runt_span(span_ms: int, median_span_ms: float, hop_ms: int) -> bool:
    """Deterministic, CLIP-RELATIVE: too short (by THIS clip's own
    unit-length distribution) to be worth a second frame -- no hardcoded
    absolute ms, the floor derives from the clip's own spans/hop."""
    return span_ms < median_span_ms and span_ms < 2 * max(hop_ms, 1)


def _too_close(a_ms: int, b_ms: int, hop_ms: int) -> bool:
    """Two candidate instants within one hop of each other would be
    near-duplicate stills -- not worth sending both."""
    return abs(a_ms - b_ms) <= max(hop_ms, 1)


def _evenly_sample_events(events: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """At most ``n`` of a cluster's events, evenly spread across its own
    TIME ORDER (always including the first and last) so the sampled frames
    give visual coverage of the whole moment, not just its biggest few hits
    clustered together. n<=1 keeps only the single strongest event."""
    ordered = sorted(events, key=lambda ev: ev.get("peak_ms", 0))
    if len(ordered) <= max(n, 1):
        return ordered if n >= 1 else []
    if n <= 1:
        return [max(ordered, key=lambda ev: ev.get("score", 0.0))]
    idxs = sorted({round(i * (len(ordered) - 1) / (n - 1)) for i in range(n)})
    return [ordered[i] for i in idxs]


def _early_late_ms(blur: List[float], hop_ms: int, s: int, e: int) -> Tuple[int, int]:
    """(early_ts, late_ts) for a unit's span [s, e): the sharpest instant in
    each half, so the pair reads as two genuinely different moments rather
    than a coin-flip between two near-identical stills near the middle."""
    mid = s + (e - s) // 2
    quarter = max(1, (e - s) // 4)
    early = _sharpest_ms(blur, hop_ms, s, mid, s + quarter)
    late = _sharpest_ms(blur, hop_ms, mid, e, e - quarter)
    return early, late


def _relabel_orphaned_phases(frames: List[PlannedFrame]) -> None:
    """After budget trimming, a ref that ended up with exactly one frame
    (its REASON_SECOND_MOMENT partner got cut) should read "only", not a
    stale "early"/"late" -- pass2's caption otherwise implies a partner
    frame that was never actually sent. In place."""
    by_ref: Dict[str, List[PlannedFrame]] = {}
    for f in frames:
        by_ref.setdefault(f.ref, []).append(f)
    for group in by_ref.values():
        if len(group) == 1 and group[0].phase in ("early", "late"):
            group[0].phase = "only"


def build_image_plan(
    pass1: Pass1Output,
    lattices: Dict[str, Lattice],
    motion_by_file: Dict[str, Dict[str, Any]],
    scene_by_file: Dict[str, Dict[str, Any]],
    silences_by_file: Dict[str, List[dict]],
    v4_meta_by_ref: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[PlannedFrame]:
    """Turn pass 1's output into a concrete, budgeted list of frames to pull
    and hand to pass 2. Deterministic: same inputs always produce the same
    plan. Files absent from ``lattices`` are silently skipped (not yet
    ingest-ready). ``v4_meta_by_ref`` (cuts_v4_segmentation.plan.md): {ref:
    {"salience": {...}, ...}} for a V4 ingest's video cuts -- a "point"
    salience straddles its peak instead of the generic early/late split (see
    REASON_SHAPE_STRADDLE). None/empty -> identical to today (V3)."""
    # --- Pass 0: resolve every mandatory unit's (file_id, s, e, ...) up
    # front -- needed BEFORE deciding 1-vs-2 frames, since the runt guard
    # compares a unit's span against its OWN clip's median unit span.
    take_units: List[Tuple[str, int, int, str]] = []
    for tc in pass1.take_candidates:
        for m in tc.members:
            if m.file_id not in lattices:
                continue
            s, e = _word_span_ms(lattices, silences_by_file, m.file_id, m.word_span)
            take_units.append((m.file_id, s, e, f"take[{tc.group_id}]"))

    speech_units: List[Tuple[str, int, int, str]] = []
    for i, sc in enumerate(pass1.speech_cuts):
        if sc.file_id not in lattices:
            continue
        s, e = _word_span_ms(lattices, silences_by_file, sc.file_id, sc.word_span)
        speech_units.append((sc.file_id, s, e, f"speech_cut[{i}]"))

    video_units: List[Tuple[str, int, int, List[int], str]] = []
    for gi, vg in enumerate(pass1.video_tentative_groups):
        if vg.file_id not in lattices:
            continue
        ref0 = f"video_group[{gi}]"
        v4_meta = (v4_meta_by_ref or {}).get(ref0)
        if v4_meta is not None:
            # The segmenter's own tight span, NOT the bounding box of the
            # (coarser, informational-only) atoms it happens to overlap --
            # using the atom span here would let mandatory/extra frames land
            # outside what the V4 cut actually plays. No atom-derived
            # anchors either: a "point" salience gets its own straddle
            # frames below; "span"/"none" fall through to the calm+sharp
            # instant within the segmenter's real bounds.
            s, e, anchors = int(v4_meta["src_in_ms"]), int(v4_meta["src_out_ms"]), []
        else:
            s, e, anchors = _atom_group_span(lattices, vg.file_id, vg.atom_ids)
        if e <= s:
            continue
        video_units.append((vg.file_id, s, e, anchors, f"video_group[{gi}]"))

    spans_by_file: Dict[str, List[int]] = {}
    for fid, s, e, _ref in take_units:
        spans_by_file.setdefault(fid, []).append(e - s)
    for fid, s, e, _ref in speech_units:
        spans_by_file.setdefault(fid, []).append(e - s)
    for fid, s, e, _anchors, _ref in video_units:
        spans_by_file.setdefault(fid, []).append(e - s)
    median_by_file: Dict[str, float] = {
        fid: statistics.median(spans) for fid, spans in spans_by_file.items()
    }

    # (mandatory frames, 2nd-moment extras, extra-anchor frames, drift
    # frames) per clip. Mandatory = one frame per pass-1 unit, NEVER dropped
    # (see module docstring); the budget only trims the three extras tiers.
    mandatory_by_file: Dict[str, List[PlannedFrame]] = {}
    second_moment_by_file: Dict[str, List[PlannedFrame]] = {}
    extra_anchors_by_file: Dict[str, List[PlannedFrame]] = {}
    drift_by_file: Dict[str, List[PlannedFrame]] = {}

    def _two_frame_ok(file_id: str, s: int, e: int, hop_ms: int) -> bool:
        return not _is_runt_span(e - s, median_by_file.get(file_id, 0.0), hop_ms)

    # Take members: one frame each, mandatory (+ 2nd moment if not a runt).
    for file_id, s, e, ref in take_units:
        motion = motion_by_file.get(file_id, {})
        blur = motion.get("blur") or []
        hop_ms = int(motion.get("hop_ms") or 0)
        if _two_frame_ok(file_id, s, e, hop_ms):
            early, late = _early_late_ms(blur, hop_ms, s, e)
            if not _too_close(early, late, hop_ms):
                mandatory_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, early, REASON_TAKE_MEMBER, ref, "early"))
                second_moment_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, late, REASON_TAKE_MEMBER, ref, "late"))
                continue
        ts = _sharpest_ms(blur, hop_ms, s, e, (s + e) // 2)
        mandatory_by_file.setdefault(file_id, []).append(
            PlannedFrame(file_id, ts, REASON_TAKE_MEMBER, ref, "only"))

    # Speech cuts: one frame each, mandatory (+ 2nd moment; + drift extras).
    for i, (file_id, s, e, ref) in enumerate(speech_units):
        motion = motion_by_file.get(file_id, {})
        blur = motion.get("blur") or []
        hop_ms = int(motion.get("hop_ms") or 0)
        if _two_frame_ok(file_id, s, e, hop_ms):
            early, late = _early_late_ms(blur, hop_ms, s, e)
            if not _too_close(early, late, hop_ms):
                mandatory_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, early, REASON_SPEECH_CUT, ref, "early"))
                second_moment_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, late, REASON_SPEECH_CUT, ref, "late"))
            else:
                ts = _sharpest_ms(blur, hop_ms, s, e, (s + e) // 2)
                mandatory_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, ts, REASON_SPEECH_CUT, ref, "only"))
        else:
            ts = _sharpest_ms(blur, hop_ms, s, e, (s + e) // 2)
            mandatory_by_file.setdefault(file_id, []).append(
                PlannedFrame(file_id, ts, REASON_SPEECH_CUT, ref, "only"))
        drift_points = (scene_by_file.get(file_id, {}) or {}).get("composition_points") or []
        for p in drift_points:
            pts = int(p.get("ts_ms", -1))
            if s < pts < e:
                drift_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, pts, REASON_COMPOSITION_DRIFT, ref, "only"))

    # Video tentative groups: first frame mandatory (first anchor, else the
    # calm+sharp instant); anchors beyond the first/last stay budgeted
    # extras. A 2nd moment is the LAST anchor (if far enough from the
    # first) or the calm+sharp instant over the LATE half otherwise.
    for file_id, s, e, anchors, ref in video_units:
        motion = motion_by_file.get(file_id, {})
        hop_ms = int(motion.get("hop_ms") or 0)
        mid = s + (e - s) // 2
        quarter = max(1, (e - s) // 4)

        v4_sal = ((v4_meta_by_ref or {}).get(ref) or {}).get("salience") or {}
        v4_events = v4_sal.get("events") or []
        if len(v4_events) > 1:
            # A genuine multi-event cluster: one frame per (evenly-sampled)
            # event peak instead of a single straddle pair, so pass 2 sees
            # every sub-moment even though it labels the cluster once.
            sampled = _evenly_sample_events(v4_events, MAX_CLUSTER_EVENT_FRAMES)
            frames = []
            for ev in sampled:
                ts = max(s, min(e - 1, int(ev.get("peak_ms", mid))))
                frames.append(ts)
            # Dedup near-identical timestamps (a tight cluster's evenly-sampled
            # peaks can land within one hop of each other).
            deduped: List[int] = []
            for ts in frames:
                if not deduped or not _too_close(ts, deduped[-1], hop_ms):
                    deduped.append(ts)
            if len(deduped) == 2:
                mandatory_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, deduped[0], REASON_CLUSTER_EVENT, ref, "early"))
                second_moment_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, deduped[1], REASON_CLUSTER_EVENT, ref, "late"))
            else:
                mand, extra = deduped[0], deduped[1:]
                mandatory_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, mand, REASON_CLUSTER_EVENT, ref, "only"))
                for ts in extra:
                    second_moment_by_file.setdefault(file_id, []).append(
                        PlannedFrame(file_id, ts, REASON_CLUSTER_EVENT, ref, "only"))
            continue

        if v4_sal.get("kind") == "point":
            peak_ms = int(v4_sal.get("peak_ms", mid))
            early = max(s, peak_ms - _STRADDLE_OFFSET_MS)
            late = min(e - 1, peak_ms + _STRADDLE_OFFSET_MS)
            if late > early and not _too_close(early, late, hop_ms):
                mandatory_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, early, REASON_SHAPE_STRADDLE, ref, "early"))
                second_moment_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, late, REASON_SHAPE_STRADDLE, ref, "late"))
            else:
                mandatory_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, peak_ms, REASON_SHAPE_STRADDLE, ref, "only"))
            continue

        if anchors:
            early_ts = anchors[0]
            two_frame = _two_frame_ok(file_id, s, e, hop_ms)
            late_ts: Optional[int] = None
            leftover_anchors = anchors[1:]
            if two_frame:
                if len(anchors) > 1 and not _too_close(anchors[0], anchors[-1], hop_ms):
                    late_ts = anchors[-1]
                    leftover_anchors = anchors[1:-1]
                else:
                    late_ts = _calm_and_sharp_ms(motion, mid, e, e - quarter)
                    leftover_anchors = anchors[1:]
                if _too_close(early_ts, late_ts, hop_ms):
                    late_ts = None
                    leftover_anchors = anchors[1:]
            if late_ts is not None:
                mandatory_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, early_ts, REASON_VIDEO_GROUP_ANCHOR, ref, "early"))
                second_moment_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, late_ts, REASON_VIDEO_GROUP_ANCHOR, ref, "late"))
            else:
                mandatory_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, early_ts, REASON_VIDEO_GROUP_ANCHOR, ref, "only"))
            for a_ts in leftover_anchors:
                extra_anchors_by_file.setdefault(file_id, []).append(
                    PlannedFrame(file_id, a_ts, REASON_VIDEO_GROUP_ANCHOR, ref, "only"))
        else:
            if _two_frame_ok(file_id, s, e, hop_ms):
                blur = motion.get("blur") or []
                early, late = _early_late_ms(blur, hop_ms, s, e)
                if not _too_close(early, late, hop_ms):
                    mandatory_by_file.setdefault(file_id, []).append(
                        PlannedFrame(file_id, early, REASON_VIDEO_GROUP_CALM, ref, "early"))
                    second_moment_by_file.setdefault(file_id, []).append(
                        PlannedFrame(file_id, late, REASON_VIDEO_GROUP_CALM, ref, "late"))
                    continue
            ts = _calm_and_sharp_ms(motion, s, e, (s + e) // 2)
            mandatory_by_file.setdefault(file_id, []).append(
                PlannedFrame(file_id, ts, REASON_VIDEO_GROUP_CALM, ref, "only"))

    out: List[PlannedFrame] = []
    all_files = (set(mandatory_by_file) | set(second_moment_by_file)
                | set(extra_anchors_by_file) | set(drift_by_file))
    for file_id in all_files:
        mandatory = mandatory_by_file.get(file_id, [])
        out.extend(mandatory)
        budget = max(0, FRAME_BUDGET_PER_CLIP - len(mandatory))
        for tier in (second_moment_by_file.get(file_id, []),
                    extra_anchors_by_file.get(file_id, []),
                    drift_by_file.get(file_id, [])):
            take = tier[:budget]
            out.extend(take)
            budget -= len(take)
    _relabel_orphaned_phases(out)
    return out
