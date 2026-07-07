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
import re
import statistics
from dataclasses import dataclass, field
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
    # Removable dead-air + filler spans across a SPEECH cut (absolute ms, inside
    # [src_in, src_out]) that the dial MAY shave -- edge silence/fillers, interior
    # disfluencies, and pause-excess. Code owns these numbers; the dial (view-math)
    # owns how much of them to apply. Empty for video or a clean spoken beat.
    remove_spans: List[Tuple[int, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"min_ms": self.min_ms, "natural_ms": self.natural_ms, "max_ms": self.max_ms,
                "levels": list(self.levels), "energy_grade": self.energy_grade,
                "natural_sound": self.natural_sound,
                "remove_spans": [list(sp) for sp in self.remove_spans]}


# Fillers safe to shave from the EDGES of a spoken beat (a leading "so"/"you
# know" or trailing "right" is throat-clearing). Interior removal uses the
# tighter _INTERIOR_FILLER_TOKENS below -- a mid-line "so"/"like"/"right" is
# usually real content, so we never touch it. Both sets + how the dial scales
# the budget are the whole tuning knob.
_FILLER_EDGE_TOKENS = {
    "um", "uh", "umm", "uhm", "erm", "er", "ah", "ahh", "hmm", "mm", "mmm", "uhh",
    "so", "well", "okay", "ok", "like", "right", "yeah", "anyway", "basically",
    "actually", "literally", "you", "know", "i", "mean",
}
_INTERIOR_FILLER_TOKENS = {
    "um", "uh", "umm", "uhm", "erm", "er", "ah", "ahh", "hmm", "mm", "mmm", "uhh",
}
_FILLER_WORD_RE = re.compile(r"[a-z']+")


def _pure_filler(text: Optional[str], vocab: set) -> bool:
    """True only if EVERY alphabetic token in the word is in ``vocab`` (so 'um,'
    counts, but 'important' never does)."""
    toks = _FILLER_WORD_RE.findall((text or "").lower())
    return bool(toks) and all(t in vocab for t in toks)


def _merge_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Sort + merge overlapping/adjacent [start, end] spans (drops empties)."""
    ordered = sorted((int(a), int(b)) for a, b in spans if int(b) > int(a))
    out: List[Tuple[int, int]] = []
    for a, b in ordered:
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def compute_speech_remove_spans(
    words: List[dict], word_span: Optional[Tuple[int, int]], s: int, e: int
) -> List[Tuple[int, int]]:
    """Deterministic removable dead-air + filler spans across a speech cut, for
    the dial to shave (edge AND interior). Everything is derived from the cut's
    OWN word timings -- no absolute constants -- so nothing "real" is ever
    proposed for removal:
      * EDGE silence + edge fillers: the run of pure-filler words at each end
        plus the dead air out to the cut boundary.
      * INTERIOR fillers: clear disfluencies ('um'/'uh'/...) mid-line only.
      * INTERIOR pauses: silence between kept words BEYOND the speaker's own
        median inter-word gap (their natural rhythm); only the EXCESS is
        removable, taken from the middle so a natural beat of silence remains.
    Returns a sorted, merged list of [start_ms, end_ms] inside [s, e]. The dial
    (view-math) decides how much of this budget actually gets cut."""
    if not words or word_span is None:
        return []
    a, b = int(word_span[0]), int(word_span[1])
    a, b = max(0, a), min(len(words) - 1, b)
    if a > b:
        return []
    content = [i for i in range(a, b + 1) if not _pure_filler(words[i].get("text"), _FILLER_EDGE_TOKENS)]
    if not content:
        return []  # an all-filler span is left whole (likely junk, handled elsewhere)
    first, last = content[0], content[-1]
    spans: List[Tuple[int, int]] = []

    # Edges: dead air + filler run out to each boundary.
    cs = int(words[first].get("start_ms", s))
    ce = int(words[last].get("end_ms", e))
    if cs > s:
        spans.append((s, cs))
    if e > ce:
        spans.append((ce, e))

    # Interior: fillers -> remove the word; everything else is "kept" and sets
    # the natural-rhythm baseline for pause trimming.
    kept: List[int] = []
    for i in range(first, last + 1):
        if i not in (first, last) and _pure_filler(words[i].get("text"), _INTERIOR_FILLER_TOKENS):
            ws, we = int(words[i].get("start_ms")), int(words[i].get("end_ms"))
            if we > ws:
                spans.append((ws, we))
        else:
            kept.append(i)

    # Interior pauses: excess over the speaker's median gap between kept words.
    gaps = [(int(words[kept[j]].get("end_ms")), int(words[kept[j + 1]].get("start_ms")))
            for j in range(len(kept) - 1)]
    positive = [g1 - g0 for g0, g1 in gaps if g1 - g0 > 0]
    if positive:
        baseline = statistics.median(positive)
        for g0, g1 in gaps:
            excess = (g1 - g0) - baseline
            if excess > 0:
                keep = baseline / 2.0  # leave a natural beat on each side
                rs, re_ = int(round(g0 + keep)), int(round(g1 - keep))
                if re_ > rs:
                    spans.append((rs, re_))

    return _merge_spans(spans)


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
    framing: Dict[str, Any]
    look: Dict[str, Any]
    caption_zones: List[Tuple[float, float, float, float]]
    hero_ts_ms: int
    pace: PaceEnvelope
    take_group_id: Optional[str]
    take_role: Optional[str]
    channel: str                           # "said" | "done" | "shown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id, "src_in_ms": self.src_in_ms, "src_out_ms": self.src_out_ms,
            "kind": self.kind,
            "word_span": list(self.word_span) if self.word_span else None,
            "atom_ids": self.atom_ids, "label": self.label, "summary": self.summary,
            "speaker": self.speaker, "on_camera": self.on_camera,
            "junk": self.junk, "junk_reason": self.junk_reason,
            "framing": self.framing, "look": self.look,
            "caption_zones": [list(z) for z in self.caption_zones],
            "hero_ts_ms": self.hero_ts_ms, "pace": self.pace.to_dict(),
            "take_group_id": self.take_group_id, "take_role": self.take_role,
            "channel": self.channel,
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
    action_energy: List[float], hop_ms: int,
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

    # min_ms floors tightening at readability + the anchor envelope (impacts
    # stay in frame). No camera-move floor: the derived pan label is gone
    # (deterministic-keep), and pace stays purely signal-driven.
    min_ms = max(readability_ms, _anchor_span_ms(anchors))
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
# Assembly
#
# No action-protection override here any more (deterministic-keep): it used
# hardcoded thresholds (>=2 anchors, mean_energy>=0.45) to un-junk a span the
# model discarded -- exactly the band-aid the plan removes. Keeping a real
# action is now guaranteed upstream instead: pass 1's total-coverage fill
# means an action is never silently dropped, and junk is a recoverable label
# (shown in the Discarded tray), never a deletion. Code no longer second-
# guesses the model's semantic junk call with a number.
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
    resolved: List[Tuple[Pass2Cut, int, int, List[int]]] = []
    for cut in pass2_output.cuts:
        lattice = lattices.get(cut.file_id)
        if lattice is None:
            raise ValueError(f"assemble_cut_records: unknown file_id {cut.file_id!r} ({cut.source_ref})")

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

        motion = motion_by_file.get(cut.file_id, {})
        anchors = _anchors_in(motion, s, e)
        resolved.append((cut, s, e, anchors))

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
    for idx, (cut, s, e, anchors) in enumerate(resolved):
        motion = motion_by_file.get(cut.file_id, {})
        blur = motion.get("blur") or []
        hop_ms = int(motion.get("hop_ms") or 0)
        action_energy = motion.get("action_energy") or []
        hero_ts = pick_hero_ts_ms(anchors, blur, hop_ms, s, e)
        pace = compute_pace_envelope(
            kind=cut.kind, s=s, e=e, readability_ms=cut.readability_ms, anchors=anchors,
            action_energy=action_energy, hop_ms=hop_ms,
            next_cut_start_ms=next_start[idx],
            max_tasteful_speed=cut.taste_fences.max_tasteful_speed,
            min_tasteful_speed=cut.taste_fences.min_tasteful_speed,
            natural_sound=cut.natural_sound,
        )
        if cut.kind == "speech":
            pace.remove_spans = compute_speech_remove_spans(
                lattices[cut.file_id].words, cut.word_span, s, e)

        out.append(CutRecord(
            file_id=cut.file_id, src_in_ms=s, src_out_ms=e, kind=cut.kind,
            word_span=cut.word_span, atom_ids=cut.atom_ids, label=cut.label, summary=cut.summary,
            speaker=cut.speaker, on_camera=cut.on_camera,
            junk=cut.junk, junk_reason=cut.junk_reason,
            framing=cut.framing.model_dump(), look=cut.look.model_dump(),
            caption_zones=list(cut.caption_zones), hero_ts_ms=hero_ts, pace=pace,
            take_group_id=cut.take_group_id, take_role=cut.take_role,
            # Channel is a SEMANTIC category the model owns: speech is always
            # "said" (code owns that fact); a video cut is "done" (an action is
            # performed/demonstrated) or "shown" (b-roll/display). Missing/unknown
            # on a video cut resolves to the conservative "shown".
            channel=("said" if cut.kind == "speech"
                     else (cut.channel if cut.channel in ("done", "shown") else "shown")),
        ))
    _enforce_one_winner_per_take_group(out)
    return out


def _enforce_one_winner_per_take_group(records: List[CutRecord]) -> None:
    """Exactly one winner per take group (in place). Pass 2 resolves takes and
    can crown two winners in a group (each member looked like the keeper), which
    the same-line guard in pass 1 mostly prevents but can't fully guarantee. This
    is the deterministic backstop: within each group keep the longest cut as the
    single winner (most complete take) and demote the other winners to plain
    'take'; a lone winner or a winner-less group is left as is. Outlooks
    (different-angle members) are never touched."""
    from collections import defaultdict
    groups: Dict[str, List[CutRecord]] = defaultdict(list)
    for r in records:
        if r.take_group_id:
            groups[r.take_group_id].append(r)
    for gid, members in groups.items():
        winners = [m for m in members if m.take_role == "winner"]
        if len(winners) <= 1:
            continue
        winners.sort(key=lambda m: m.src_out_ms - m.src_in_ms, reverse=True)
        for m in winners[1:]:
            m.take_role = "take"
        logger.info("post: take group %s had %d winners -> kept the longest, demoted %d to 'take'",
                    gid, len(winners), len(winners) - 1)
