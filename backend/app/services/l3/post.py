"""
Cuts v3 post-compute: deterministic assembly of the final ``cut_records``
from pass 2's judged output. No model call here, and no fallback -- the
remaining invariants (zero overlap, boundary-on-edge) are enforced in code; a
violation fails the ingest run loudly for re-run rather than being silently
patched over. Full coverage is NO LONGER an invariant: cuts are a selection,
not a partition, so gaps (dropped connective tissue) are legal. See
cuts_v3.plan.md section 6 and cuts_v3_boundaries_v2.plan.md.

Note on "framing motion" (plan sec. 6, the subject-centroid-follows-crop
bullet): that machinery already exists in ``app.services.l3.framing``
(``focus_for_range``), reading straight off ``motion_dynamics``/
``clip_perception`` for an arbitrary ``(file_id, src_in_ms, src_out_ms)``
span at ARRANGE time. A cut_record's own ``file_id``/``src_in_ms``/
``src_out_ms`` are already everything that machinery needs, so there is
nothing new to build or store here for that bullet.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.lattice import Lattice, _anchors_in, resolve_speech_span_ms
from app.services.l3.pass2 import Pass2Cut, Pass2Output
from app.services.l3.post_params import (
    ANCHOR_PAD_MS, ENERGY_GRADE_BANDS, FLATLINE_BAND, PACE_LEVEL_TARGETS, SPEED_CEIL, SPEED_FLOOR,
)
from app.services.l3.video_segments import _sharpest_ms

logger = logging.getLogger(__name__)


@dataclass
class PaceEnvelope:
    min_ms: int
    natural_ms: int
    max_ms: int
    levels: List[float]
    energy_grade: str
    natural_sound: bool

    def to_dict(self) -> Dict[str, Any]:
        return {"min_ms": self.min_ms, "natural_ms": self.natural_ms, "max_ms": self.max_ms,
                "levels": list(self.levels), "energy_grade": self.energy_grade,
                "natural_sound": self.natural_sound}


@dataclass
class CutRecord:
    file_id: str
    src_in_ms: int
    src_out_ms: int
    kind: str                              # "speech" | "video"
    word_span: Optional[Tuple[int, int]]
    atom_ids: Optional[List[int]]
    label: str
    summary: str
    speaker: Optional[str]
    on_camera: Optional[bool]
    junk: bool
    junk_reason: str
    junk_confidence: str                   # "high" -> hidden by default; "low"/doubtful -> shown
    framing: Dict[str, Any]
    look: Dict[str, Any]
    caption_zones: List[Tuple[float, float, float, float]]
    hero_ts_ms: int
    pace: PaceEnvelope
    take_group_id: Optional[str]
    take_role: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id, "src_in_ms": self.src_in_ms, "src_out_ms": self.src_out_ms,
            "kind": self.kind,
            "word_span": list(self.word_span) if self.word_span else None,
            "atom_ids": self.atom_ids, "label": self.label, "summary": self.summary,
            "speaker": self.speaker, "on_camera": self.on_camera,
            "junk": self.junk, "junk_reason": self.junk_reason,
            "junk_confidence": self.junk_confidence,
            "framing": self.framing, "look": self.look,
            "caption_zones": [list(z) for z in self.caption_zones],
            "hero_ts_ms": self.hero_ts_ms, "pace": self.pace.to_dict(),
            "take_group_id": self.take_group_id, "take_role": self.take_role,
        }


# --------------------------------------------------------------------------
# hero_ts_ms: anchor > subject-sharp > midpoint
# --------------------------------------------------------------------------

def pick_hero_ts_ms(anchors: List[int], blur: List[float], hop_ms: int, s: int, e: int) -> int:
    if anchors:
        return anchors[0]
    return _sharpest_ms(blur, hop_ms, s, e, (s + e) // 2)


# --------------------------------------------------------------------------
# Pace envelope
# --------------------------------------------------------------------------

def _mean_in_span(arr: List[float], hop_ms: int, s: int, e: int) -> float:
    if not arr or hop_ms <= 0:
        return 0.0
    lo, hi = max(0, s // hop_ms), min(len(arr) - 1, max(s // hop_ms, (e - 1) // hop_ms))
    if hi < lo:
        return 0.0
    seg = arr[lo:hi + 1]
    return sum(seg) / len(seg) if seg else 0.0


def _flatline_bound_ms(action_energy: List[float], hop_ms: int, from_ms: int, ceiling_ms: int) -> int:
    """How far past ``from_ms`` action_energy stays within FLATLINE_BAND of
    its own value at ``from_ms``, capped at ``ceiling_ms`` (the next cut's
    start, or the file's end) -- past that point there's nothing new
    happening to justify a longer hold."""
    if not action_energy or hop_ms <= 0 or ceiling_ms <= from_ms:
        return from_ms
    i0 = min(len(action_energy) - 1, from_ms // hop_ms)
    baseline = action_energy[i0]
    ceiling_i = ceiling_ms // hop_ms
    i = i0
    while i + 1 < len(action_energy) and i < ceiling_i:
        i += 1
        if abs(action_energy[i] - baseline) > FLATLINE_BAND:
            return min(i * hop_ms, ceiling_ms)
    return ceiling_ms


def _move_completion_ms(atom_spans: List[Tuple[int, int, str]]) -> int:
    """The longest deliberate-pan atom's own duration inside this cut -- a
    cut can't be tightened shorter than the camera move it contains without
    the move reading as truncated."""
    pans = [e - s for s, e, desc in atom_spans if desc == "pan"]
    return max(pans) if pans else 0


def _anchor_span_ms(anchors: List[int]) -> int:
    if not anchors:
        return 0
    return (max(anchors) - min(anchors)) + 2 * ANCHOR_PAD_MS


def _energy_grade(mean_action_energy: float) -> str:
    for grade, upper in ENERGY_GRADE_BANDS:
        if mean_action_energy < upper:
            return grade
    return "high"


def _pace_levels(intrinsic_velocity: float, min_speed: float, max_speed: float) -> List[float]:
    lo, hi = max(min_speed, SPEED_FLOOR), min(max_speed, SPEED_CEIL)
    if lo > hi:
        lo, hi = hi, lo   # an inverted taste fence -- clamp to a single reachable point rather than crash
    if intrinsic_velocity <= 0:
        return [hi] * len(PACE_LEVEL_TARGETS)
    return [min(max(target / intrinsic_velocity, lo), hi) for target in PACE_LEVEL_TARGETS]


def compute_pace_envelope(
    *, kind: str, s: int, e: int, readability_ms: int, anchors: List[int],
    atom_spans: List[Tuple[int, int, str]], action_energy: List[float], hop_ms: int,
    next_cut_start_ms: int, max_tasteful_speed: float, min_tasteful_speed: float,
    natural_sound: bool,
) -> PaceEnvelope:
    natural_ms = max(1, e - s)
    mean_ae = _mean_in_span(action_energy, hop_ms, s, e)
    grade = _energy_grade(mean_ae)

    if kind == "speech":
        # Speech always plays at native speed -- see cuts_v3.plan.md sec. 6.
        return PaceEnvelope(min_ms=max(readability_ms, natural_ms), natural_ms=natural_ms,
                            max_ms=natural_ms, levels=[1.0] * len(PACE_LEVEL_TARGETS),
                            energy_grade=grade, natural_sound=natural_sound)

    min_ms = max(readability_ms, _anchor_span_ms(anchors), _move_completion_ms(atom_spans))
    flatline_end_ms = _flatline_bound_ms(action_energy, hop_ms, e, next_cut_start_ms)
    max_ms = max(natural_ms, flatline_end_ms - s)
    levels = _pace_levels(mean_ae, min_tasteful_speed, max_tasteful_speed)
    return PaceEnvelope(min_ms=min_ms, natural_ms=natural_ms, max_ms=max_ms, levels=levels,
                        energy_grade=grade, natural_sound=natural_sound)


# --------------------------------------------------------------------------
# Invariant enforcement: zero overlap, per file (coverage gaps are legal).
# --------------------------------------------------------------------------

def _validate_no_overlap(file_id: str, spans: List[Tuple[int, int]], duration_ms: int) -> None:
    """Boundaries-v2: cuts are a SELECTION, not a partition -- GAPS ARE LEGAL
    (connective tissue / pre-roll / dead air is dropped, not tiled). The only
    invariant left is zero overlap: two cuts must never claim the same instant,
    or the timeline is ambiguous. Coverage gaps used to raise here; that was the
    full-coverage invariant that forced every dead sliver to become a junk tile
    (see cuts_v3_boundaries_v2.plan.md)."""
    spans = sorted(spans)
    for (s0, e0), (s1, e1) in zip(spans, spans[1:]):
        if s1 < e0:
            raise ValueError(f"{file_id}: overlap between [{s0}-{e0}] and [{s1}-{e1}]")


# --------------------------------------------------------------------------
# Action protection: a genuine motion payoff can never be silently discarded.
#
# Observed against real Reel-trail data: the model labeled high-motion,
# multi-anchor END-OF-CLIP spans as "trailing junk" ("Trailing frames after
# carbs line", "Pointing gesture at clip end") -- and the frontend hides junk,
# so the action "never comes up". This is exactly the user's rule inverted:
# "if doubtful, SHOW." A span carrying impact anchors or strong subject motion
# is a payoff, not trailing dead air, so its junk flag is overridden here in
# code -- the model does not get the final say on discarding a real action.
#
# Video-only (visual action); speech junk -- cue words, false starts -- is
# untouched. The bar is deliberately high enough that a truly still trailing
# frame (no anchors, low energy) still stays junk.
# --------------------------------------------------------------------------

ACTION_PROTECT_MIN_ANCHORS = 2       # >= this many impact anchors => real payoff
ACTION_PROTECT_ENERGY = 0.45         # or mean subject-motion energy over the span


def _mean_action_energy(motion: Dict[str, Any], s: int, e: int) -> float:
    hop = int(motion.get("hop_ms") or 0)
    ae = motion.get("action_energy") or []
    if hop <= 0 or not ae:
        return 0.0
    lo, hi = max(0, s // hop), min(len(ae), max(s // hop + 1, e // hop))
    seg = ae[lo:hi]
    return sum(seg) / len(seg) if seg else 0.0


def _is_protected_action(kind: str, anchors: List[int], mean_energy: float) -> bool:
    return kind == "video" and (
        len(anchors) >= ACTION_PROTECT_MIN_ANCHORS or mean_energy >= ACTION_PROTECT_ENERGY
    )


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def assemble_cut_records(
    pass2_output: Pass2Output,
    lattices: Dict[str, Lattice],
    motion_by_file: Dict[str, Dict[str, Any]],
    silences_by_file: Dict[str, List[dict]],
) -> List[CutRecord]:
    """Resolve every judged cut to its final ms span (word/atom edges only,
    by construction), enforce zero-overlap per file (gaps are legal -- cuts are
    a selection, not a partition), then compute hero_ts_ms + the pace envelope
    for each. Raises ``ValueError`` (stage ``post``, per the plan's "no
    fallback" rule) on any invariant violation -- the caller marks the ingest
    run ``failed`` for re-run."""
    resolved: List[Tuple[Pass2Cut, int, int, List[int], List[Tuple[int, int, str]]]] = []
    for cut in pass2_output.cuts:
        lattice = lattices.get(cut.file_id)
        if lattice is None:
            raise ValueError(f"assemble_cut_records: unknown file_id {cut.file_id!r} ({cut.source_ref})")

        atom_spans: List[Tuple[int, int, str]] = []
        if cut.kind == "speech":
            silences = silences_by_file.get(cut.file_id, [])
            s, e = resolve_speech_span_ms(lattice.words, lattice.atoms, cut.word_span, silences)
        else:
            atoms_by_id = {a.atom_id: a for a in lattice.atoms}
            members = [atoms_by_id[i] for i in (cut.atom_ids or []) if i in atoms_by_id]
            if not members:
                raise ValueError(f"assemble_cut_records: no resolvable atoms for {cut.source_ref} "
                                 f"in {cut.file_id}")
            s = min(a.start_ms for a in members)
            e = max(a.end_ms for a in members)
            atom_spans = [(a.start_ms, a.end_ms, a.camera_desc) for a in members]

        motion = motion_by_file.get(cut.file_id, {})
        anchors = _anchors_in(motion, s, e)
        resolved.append((cut, s, e, anchors, atom_spans))

    by_file: Dict[str, List[int]] = {}
    for idx, (cut, *_rest) in enumerate(resolved):
        by_file.setdefault(cut.file_id, []).append(idx)

    # A file with zero assigned cuts is now LEGAL (boundaries-v2): a clip that's
    # all dead air / all junk contributes nothing, and that's a valid outcome,
    # not a failure. Still worth a warning -- it's ALSO what a silently truncated
    # pass-2 response looks like (see llm.client._truncated), so a surprise empty
    # file is something to eyeball, just not something to abort the run over.
    missing = set(lattices.keys()) - set(by_file.keys())
    if missing:
        logger.warning("no cuts assigned for file(s) %s -- all-junk clip, or a "
                        "pass-2 omission worth checking", sorted(missing))

    next_start: Dict[int, int] = {}
    for file_id, idxs in by_file.items():
        idxs.sort(key=lambda i: resolved[i][1])
        duration_ms = lattices[file_id].duration_ms
        spans = [(resolved[i][1], resolved[i][2]) for i in idxs]
        _validate_no_overlap(file_id, spans, duration_ms)
        for pos, i in enumerate(idxs):
            next_start[i] = resolved[idxs[pos + 1]][1] if pos + 1 < len(idxs) else duration_ms

    out: List[CutRecord] = []
    for idx, (cut, s, e, anchors, atom_spans) in enumerate(resolved):
        motion = motion_by_file.get(cut.file_id, {})
        blur = motion.get("blur") or []
        hop_ms = int(motion.get("hop_ms") or 0)
        action_energy = motion.get("action_energy") or []
        hero_ts = pick_hero_ts_ms(anchors, blur, hop_ms, s, e)
        pace = compute_pace_envelope(
            kind=cut.kind, s=s, e=e, readability_ms=cut.readability_ms, anchors=anchors,
            atom_spans=atom_spans, action_energy=action_energy, hop_ms=hop_ms,
            next_cut_start_ms=next_start[idx],
            max_tasteful_speed=cut.taste_fences.max_tasteful_speed,
            min_tasteful_speed=cut.taste_fences.min_tasteful_speed,
            natural_sound=cut.natural_sound,
        )

        junk, junk_reason, junk_conf = cut.junk, cut.junk_reason, cut.junk_confidence
        if junk and _is_protected_action(cut.kind, anchors, _mean_action_energy(motion, s, e)):
            logger.info("post: overriding junk on %s [%d-%d] -- protected action "
                        "(%d anchors, mean_energy=%.2f), model said %r",
                        cut.source_ref, s, e, len(anchors),
                        _mean_action_energy(motion, s, e), cut.junk_reason)
            junk, junk_reason, junk_conf = False, "", "low"

        out.append(CutRecord(
            file_id=cut.file_id, src_in_ms=s, src_out_ms=e, kind=cut.kind,
            word_span=cut.word_span, atom_ids=cut.atom_ids, label=cut.label, summary=cut.summary,
            speaker=cut.speaker, on_camera=cut.on_camera, junk=junk, junk_reason=junk_reason,
            junk_confidence=junk_conf,
            framing=cut.framing.model_dump(), look=cut.look.model_dump(),
            caption_zones=list(cut.caption_zones), hero_ts_ms=hero_ts, pace=pace,
            take_group_id=cut.take_group_id, take_role=cut.take_role,
        ))
    return out
