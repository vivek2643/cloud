"""
Cuts v2: the unified priority partition.

Replaces the overlapping, energy-laddered, channel-tabbed hero-cuts feed with a
DETERMINISTIC, NON-OVERLAPPING partition of each video into tag-bearing cuts.
See ``cuts_v2.plan.md`` for the original design (this module is that plan's
Phase B2, the core claim algorithm) and ``cuts_v2_boundaries.plan.md`` for the
follow-on that re-scopes the dial and video boundary detection (Phase C1).

North star (see the plans for the full rationale):
  1. One unified partition pass owns the timeline -- never independent per-
     channel passes merged after the fact. Overlap is impossible by
     construction: every cut is claimed against the SAME running "already
     claimed" ledger, in priority order.
  2. Simultaneity is TAGS, not parallel cuts -- talking-while-gesturing is one
     cut ``[said, done]``, never a said cut overlapping a done cut.
  3. Priority is deterministic: said (1.0) > done (0.6) > shown (0.3). A lower-
     priority candidate mostly inside a higher one is demoted to a TAG on it;
     otherwise it is trimmed to its free remainder and kept as its own cut.
  4. "Never cut said" -- word-level veto via the L1 dialogue cut-cost grid
     (cost 1.0 inside a word), composed into the fused seam field every
     boundary snaps through.
  5. Boundaries are DETECTED, not chosen by a slider for ``said`` (still
     energy-independent). SUPERSEDED for video by ``cuts_v2_boundaries.plan``:
     the dial now also drives GRANULARITY there -- see ``video_segments.py``
     -- on top of tightness (still applied here, in ``_tighten_video``).
     A deliberate, accepted departure from "detect once" for video only.
  6. Over-split, never under-split: a free remainder too small to be a
     meaningful sub-unit is silently absorbed (no cut, no tag) rather than
     forced into one.

Pure core: ``partition_clip(clip: ClipArtifacts, energy) -> List[Cut]`` takes
already-loaded clip artifacts and does no DB/model call (mirrors
``atoms.build_atoms``'s purity), so it is trivially testable -- see
``scripts/test_partition.py``. ``build_partition(file_id, energy)`` is the
convenience wrapper that loads + partitions for real callers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.l1 import fused_seams as fseams
from app.services.l3 import vocab
from app.services.l3.energy import energy_band, energy_to_params
from app.services.l3.partition_params import (
    ANCHOR_PAD_MS,
    AUDIO_ANCHOR_MIN_GAP_MS,
    AUDIO_ANCHOR_RISE_DB,
    DEFAULT_SNAP_ENERGY,
    MERGE_GAP_MS,
    MIN_SUBUNIT_MS,
    OVERLAP_TAG_FRAC,
    PRIORITY_DONE,
    PRIORITY_SAID,
    PRIORITY_SHOWN,
    VIDEO_CORE_FLOOR_MS,
)
from app.services.l3.speech_granularity_params import (
    ENERGY_DROP_DB,
    MAX_BRIDGE_GAP_MS,
    PITCH_FALL_HZ,
    PROSODY_TAIL_MS,
)
from app.services.l3.thought_segments import Thought
from app.services.l3.video_segments import segment_video

logger = logging.getLogger(__name__)

# Bump when the partition algorithm's shape changes so a v2 precompute cache
# (Phase B4) recomputes even if the underlying L1/L3 artifacts did not.
CUTS_VERSION = 1

_PRIORITY: Dict[str, float] = {
    vocab.CHANNEL_SAID: PRIORITY_SAID,
    vocab.CHANNEL_DONE: PRIORITY_DONE,
    vocab.CHANNEL_SHOWN: PRIORITY_SHOWN,
}
# Claim order: highest priority first (said claims before done claims before shown).
_CLAIM_ORDER: Tuple[str, ...] = (vocab.CHANNEL_SAID, vocab.CHANNEL_DONE, vocab.CHANNEL_SHOWN)


# --------------------------------------------------------------------------
# The Cut shape
# --------------------------------------------------------------------------

@dataclass
class Cut:
    """One disjoint interval of a file's timeline, tagged with every channel it
    serves. INVARIANT: no two cuts from the same partition overlap in a file."""
    file_id: str
    src_in_ms: int
    src_out_ms: int
    tags: List[str]                                  # subset of {said,done,shown}, >=1, priority-ordered
    primary: str                                      # the highest-priority tag
    label: str                                        # transcript text (said) | type placeholder (video)
    speaker: Optional[str] = None
    peak_ms: int = 0                                  # representative frame instant (thumbnail)
    keep_spans: Optional[List[Tuple[int, int]]] = None   # set by tightness (Phase B3); None = contiguous
    # Deferred/empty until the image pass lands (see plan's "out of scope").
    people: List[dict] = field(default_factory=list)
    framing: Optional[dict] = None
    subject: Optional[str] = None
    summary: Optional[str] = None
    quality: Optional[dict] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "src_in_ms": self.src_in_ms,
            "src_out_ms": self.src_out_ms,
            "tags": list(self.tags),
            "primary": self.primary,
            "label": self.label,
            "speaker": self.speaker,
            "peak_ms": self.peak_ms,
            "keep_spans": [list(s) for s in self.keep_spans] if self.keep_spans else None,
            "people": self.people,
            "framing": self.framing,
            "subject": self.subject,
            "summary": self.summary,
            "quality": self.quality,
        }


@dataclass
class ClipArtifacts:
    """Pre-loaded L1/L3 artifacts for one clip -- everything ``partition_clip``
    needs, with no DB access of its own."""
    file_id: str
    duration_ms: int
    thoughts: List[Thought] = field(default_factory=list)
    motion: Optional[dict] = None    # {hop_ms, action_energy, action_points, camera_cut_cost, camera_coherence, camera_stability, camera_motion, blur}
    scene: Optional[dict] = None     # {hop_ms, shot_points, composition_points}
    audio: Optional[dict] = None     # {dialogue_cut_cost, dialogue_cut_hop_ms, dialogue_cut_points, beat_cut_cost, beat_cut_hop_ms, beat_cut_points, rms_db, prosody_hop_ms, f0_hz}


# --------------------------------------------------------------------------
# Candidate spans per channel (energy-independent -- North Star #1/#5)
# --------------------------------------------------------------------------

@dataclass
class _Candidate:
    channel: str
    start_ms: int
    end_ms: int
    peak_ms: int
    label: str
    speaker: Optional[str] = None


def _thought_start_ms(t: Thought) -> int:
    """The earliest ms of a thought INCLUDING its setup run-up."""
    return int(t.setup.raw_in_ms if t.setup else t.thought.raw_in_ms)


def _thought_text_with_setup(t: Thought) -> str:
    if t.setup and (t.setup.text or "").strip():
        return (t.setup.text + " " + t.thought.text).strip()
    return (t.thought.text or "").strip()


def _band_thought_span(t: Thought, band: int) -> Tuple[int, int, str]:
    """(in_ms, out_ms, text) for one thought at a non-Broad granularity band."""
    if band == 1:      # Calm: the thought + the speaker's own run-up
        in_ms = t.setup.raw_in_ms if t.setup else t.thought.raw_in_ms
        return int(in_ms), int(t.thought.raw_out_ms), _thought_text_with_setup(t)
    if band == 2:      # Balanced: the complete thought proper
        return int(t.thought.raw_in_ms), int(t.thought.raw_out_ms), (t.thought.text or "")
    if band == 3:      # Tight: the core sentence
        return int(t.core.raw_in_ms), int(t.core.raw_out_ms), (t.core.text or "")
    return int(t.punch.raw_in_ms), int(t.punch.raw_out_ms), (t.punch.text or "")   # Sharp: punchline


def _trend(values: List[float], hop_ms: int, start_ms: int, end_ms: int) -> Optional[float]:
    """(last - first) of the non-zero (voiced/non-silent) samples in
    [start_ms, end_ms) -- a simple trailing trend. None when there aren't at
    least two usable samples to compare."""
    if not values or hop_ms <= 0:
        return None
    lo, hi = max(0, start_ms // hop_ms), min(len(values) - 1, end_ms // hop_ms)
    usable = [v for v in values[lo:hi + 1] if v]
    if len(usable) < 2:
        return None
    return usable[-1] - usable[0]


def _prosody_bridges_gap(f0_hz: List[float], rms_db: List[float], hop_ms: int,
                         thought_end_ms: int) -> Optional[bool]:
    """True = an intentional pause (bridge -- merge into one turn); False = a
    real thought-end (break); None = no pitch signal, fall back to gap-length
    alone. Reads the TRAILING pitch + energy trend just before the gap (the
    gap itself is silence, no pitch to read): falling pitch + dropping energy
    is the classic declarative-statement-ending shape (a real break);
    anything else means the speaker likely isn't finished (bridge it)."""
    if not f0_hz or hop_ms <= 0:
        return None
    tail_start = max(0, thought_end_ms - PROSODY_TAIL_MS)
    pitch_trend = _trend(f0_hz, hop_ms, tail_start, thought_end_ms)
    if pitch_trend is None:
        return None
    energy_trend = _trend(rms_db, hop_ms, tail_start, thought_end_ms)
    falling = pitch_trend <= -PITCH_FALL_HZ
    dropping = energy_trend is not None and energy_trend <= -ENERGY_DROP_DB
    return not (falling and dropping)


def _merge_thoughts_by_prosody(thoughts: List[Thought], gap_ms: int, f0_hz: List[float],
                               rms_db: List[float], hop_ms: int) -> List[List[Thought]]:
    """Group consecutive SAME-SPEAKER thoughts into one turn (the Broad zoom-
    out) -- graded by PROSODY, not gap-length alone: falling pitch + dropping
    energy across the gap is a real break (never merge); sustained/rising
    pitch is an intentional dramatic pause (bridge it), even past the base
    ``gap_ms`` threshold, up to the absolute MAX_BRIDGE_GAP_MS ceiling.
    Degrades to the old gap-length-only rule when pitch is unavailable (no
    re-analyze yet). A speaker change always breaks a group."""
    items = sorted(thoughts, key=_thought_start_ms)
    groups: List[List[Thought]] = []
    for t in items:
        if groups:
            prev = groups[-1][-1]
            gap = _thought_start_ms(t) - int(prev.thought.raw_out_ms)
            if t.speaker == prev.speaker and 0 <= gap <= MAX_BRIDGE_GAP_MS:
                verdict = _prosody_bridges_gap(f0_hz, rms_db, hop_ms, int(prev.thought.raw_out_ms))
                bridge = verdict if verdict is not None else (gap < gap_ms)
                if bridge:
                    groups[-1].append(t)
                    continue
        groups.append([t])
    return groups


def _said_candidates(thoughts: List[Thought], audio: Optional[dict], energy: float) -> List[_Candidate]:
    """Speech candidates at ``energy``'s GRANULARITY (Phase C2 of
    ``cuts_v2_boundaries.plan.md``) -- read off each Thought's OWN nested
    hierarchy (setup+thought / thought / core / punch), already the clean,
    speaker-pure, silence-ceilinged units ``thought_segments`` exists to
    produce. Broad additionally merges consecutive same-speaker thoughts into
    whole TURNS, graded by prosody (see ``_merge_thoughts_by_prosody``) rather
    than a bare gap threshold. `said` boundaries are still DETECTED, never
    invented outright -- but which of a thought's nested levels the dial reads
    is now energy-dependent, the same exception ``cuts_v2_boundaries.plan.md``
    already makes for video (Phase C1)."""
    if not thoughts:
        return []
    band = energy_band(energy)
    audio = audio or {}
    f0_hz = audio.get("f0_hz") or []
    rms_db = audio.get("rms_db") or []
    hop_ms = int(audio.get("prosody_hop_ms") or 0)

    out: List[_Candidate] = []
    if band == 0:
        merge_gap_ms = energy_to_params(energy).speech_merge_gap_ms
        for group in _merge_thoughts_by_prosody(thoughts, merge_gap_ms, f0_hz, rms_db, hop_ms):
            s = min(_thought_start_ms(t) for t in group)
            e = max(int(t.thought.raw_out_ms) for t in group)
            if e <= s:
                continue
            text = " ".join(_thought_text_with_setup(t) for t in group).strip()
            out.append(_Candidate(vocab.CHANNEL_SAID, s, e, (s + e) // 2, text, group[0].speaker))
    else:
        for t in thoughts:
            s, e, text = _band_thought_span(t, band)
            if e <= s:
                continue
            peak = (s + e) // 2
            if band >= 3 and t.punch and t.punch.raw_out_ms > t.punch.raw_in_ms:
                peak = (t.punch.raw_in_ms + t.punch.raw_out_ms) // 2
            out.append(_Candidate(vocab.CHANNEL_SAID, s, e, peak, text, t.speaker))
    out.sort(key=lambda c: c.start_ms)
    return out


def _video_candidates(motion: Optional[dict], scene: Optional[dict],
                      duration_ms: int, energy: float) -> List[_Candidate]:
    """Video candidates from the camera-move-state segmentation
    (``video_segments.segment_video``, Phases C1 + C3a of
    ``cuts_v2_boundaries.plan.md``) -- replaces the old impact-WINDOW `done`
    detector and the scene/shot-bounded `shown` detector (both
    "known-imperfect" per that plan). ONE segmentation already covers the
    whole clip and tags each piece done/shown (provisional -- real
    classification waits on the image pass), so this is a thin adapter, not a
    second detector: `energy` drives GRANULARITY here (how many camera
    settles'/subject-beat sub-splits the dial admits) -- a deliberate exception
    to "detection is energy-independent" for video only, see the plan's
    "Honest risks" #1 -- while a hard shot cut in `scene` is a TOP-PRIORITY
    boundary at every granularity (Phase C3a re-enables scene detection as a
    video boundary source for real multi-shot footage)."""
    segs = segment_video(motion, duration_ms, energy, scene=scene)
    return [_Candidate(s.tag, s.start_ms, s.end_ms, s.peak_ms, s.tag) for s in segs]


# --------------------------------------------------------------------------
# Fused cost field (boundary snapping only -- fixed default tightness)
# --------------------------------------------------------------------------

def _build_field(clip: ClipArtifacts) -> Optional[fseams.FusedField]:
    """The v2 fused field: composes the L1 dialogue veto (word-level -- "never
    cut said"), camera veto, and action/beat attractors -- exactly the v1
    channel grids (no VLM atoms in v2; see the plan's out-of-scope list).
    Fixed DEFAULT_SNAP_ENERGY -- detection/claiming is energy-independent by
    design; the real tightness dial (Phase B3) is layered on afterward."""
    a, m = clip.audio or {}, clip.motion or {}
    if not a and not m:
        return None
    return fseams.compute_fused_field(
        duration_ms=clip.duration_ms, energy=DEFAULT_SNAP_ENERGY,
        dialogue_cost=a.get("dialogue_cut_cost"),
        dialogue_hop=a.get("dialogue_cut_hop_ms", 100),
        dialogue_points=a.get("dialogue_cut_points"),
        camera_cost=m.get("camera_cut_cost"),
        action_cost=m.get("action_cut_cost"),
        action_points=m.get("action_points"),
        motion_hop=m.get("hop_ms", 100),
        beat_cost=a.get("beat_cut_cost"),
        beat_points=a.get("beat_cut_points"),
        beat_hop=a.get("beat_cut_hop_ms", 100),
    )


# --------------------------------------------------------------------------
# The priority claim (the core algorithm -- North Star #1/#2/#3)
# --------------------------------------------------------------------------

@dataclass
class _Placed:
    start_ms: int
    end_ms: int
    tags: List[str]
    primary: str
    label: str
    speaker: Optional[str]
    peak_ms: int


def _overlap_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _subtract(span: Tuple[int, int], claimed: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """``span`` minus the union of ``claimed`` intervals -> free fragments, in
    time order. Zero, one, or several fragments (a candidate straddling two
    already-claimed cuts splits into the gaps around/between them)."""
    free = [span]
    for cs, ce in claimed:
        nxt: List[Tuple[int, int]] = []
        for fs, fe in free:
            if ce <= fs or cs >= fe:
                nxt.append((fs, fe))
                continue
            if cs > fs:
                nxt.append((fs, cs))
            if ce < fe:
                nxt.append((ce, fe))
        free = nxt
        if not free:
            break
    return free


def _claim(candidates: List[_Candidate], field: Optional[fseams.FusedField],
           duration_ms: int) -> List[_Placed]:
    """Claim every candidate onto ONE timeline, highest priority first.

    A candidate mostly covered by ONE already-claimed cut is demoted to a TAG
    on it; otherwise its free remainder(s) become their own cut(s), snapped to
    the nearest clean seam and hard-clamped to never invade an already-claimed
    neighbor (so overlap stays impossible by construction regardless of how
    far the seam search reaches).

    The overlap-fraction check only applies when a candidate leaves AT MOST
    ONE contiguous free remainder. Two or more disjoint remainders (e.g. a
    catch-all `shown` candidate spanning a whole clip with a `said` cut
    carved out of its MIDDLE) prove the candidate isn't "the same moment" as
    whatever claimed the middle -- a genuine simultaneity (talking while
    gesturing) can only ever leave a SINGLE remainder, at one edge or none at
    all, never split the candidate in two. So a multi-remainder candidate
    always keeps every qualifying fragment as its own cut instead.
    """
    placed: List[_Placed] = []
    for ch in _CLAIM_ORDER:
        chan_candidates = sorted((c for c in candidates if c.channel == ch),
                                 key=lambda c: c.start_ms)
        for c in chan_candidates:
            claimed_ranges = [(p.start_ms, p.end_ms) for p in placed]
            free = _subtract((c.start_ms, c.end_ms), claimed_ranges)

            if len(free) <= 1:
                free_total = sum(e - s for s, e in free)
                cand_len = max(1, c.end_ms - c.start_ms)
                covered_frac = 1.0 - (free_total / cand_len)
                if covered_frac >= OVERLAP_TAG_FRAC:
                    best = max(
                        placed,
                        key=lambda p: _overlap_ms(p.start_ms, p.end_ms, c.start_ms, c.end_ms),
                        default=None,
                    )
                    if best is not None and ch not in best.tags:
                        best.tags.append(ch)
                        best.tags.sort(key=lambda t: -_PRIORITY[t])
                    continue

            for fs, fe in free:
                if fe - fs < MIN_SUBUNIT_MS:
                    continue   # too small a sliver to be its own cut or a tag
                lo_bound = max((p.end_ms for p in placed if p.end_ms <= fs), default=0)
                hi_bound = min((p.start_ms for p in placed if p.start_ms >= fe), default=duration_ms)
                if field is not None:
                    sm_s, sm_e = fseams.snap_bounds(
                        field, fs, fe, energy=DEFAULT_SNAP_ENERGY, duration_ms=duration_ms)
                    sm_s = max(sm_s, lo_bound)
                    sm_e = min(sm_e, hi_bound)
                    if sm_e <= sm_s:
                        sm_s, sm_e = fs, fe
                else:
                    sm_s, sm_e = fs, fe
                peak = c.peak_ms if fs <= c.peak_ms <= fe else (fs + fe) // 2
                placed.append(_Placed(sm_s, sm_e, [ch], ch, c.label, c.speaker, peak))
        placed.sort(key=lambda p: p.start_ms)
    return placed


def _merge_continuous(placed: List[_Placed], scene: Optional[dict]) -> List[_Placed]:
    """Merge adjacent same-primary, same-speaker cuts that are TRULY
    continuous (touching within MERGE_GAP_MS) -- never across a hard shot cut
    or a speaker change, which always break a run."""
    shot_bounds = [int(p.get("ts_ms", -1)) for p in ((scene or {}).get("shot_points") or [])]
    out: List[_Placed] = []
    for p in placed:
        if (out and out[-1].primary == p.primary and out[-1].speaker == p.speaker
                and 0 <= p.start_ms - out[-1].end_ms <= MERGE_GAP_MS
                and not any(out[-1].end_ms <= b <= p.start_ms for b in shot_bounds)):
            prev = out[-1]
            prev.end_ms = p.end_ms
            for t in p.tags:
                if t not in prev.tags:
                    prev.tags.append(t)
            prev.tags.sort(key=lambda t: -_PRIORITY[t])
            if p.label and p.label not in (prev.label or ""):
                prev.label = f"{prev.label} {p.label}".strip()
        else:
            out.append(p)
    return out


def _assert_non_overlap(cuts: List[Cut]) -> None:
    """The core invariant (North Star #1): overlap is impossible BY
    CONSTRUCTION, so a violation is a real bug, not a best-effort concern --
    fail loud rather than silently serve corrupt cuts."""
    ordered = sorted(cuts, key=lambda c: c.src_in_ms)
    prev_end: Optional[int] = None
    for c in ordered:
        if prev_end is not None and c.src_in_ms < prev_end:
            raise AssertionError(
                f"partition overlap: cut at {c.src_in_ms} starts before prior ends at {prev_end}"
            )
        prev_end = c.src_out_ms


def _core_inset(core_in: int, core_out: int, peak: int, target: Optional[int],
                *, lead_frac: float = 0.5) -> Tuple[int, int]:
    """Inset [core_in, core_out] toward ``peak`` to ``target`` ms, keeping
    ``lead_frac`` of the window before the peak. Only ever shrinks. The
    anchor-less fallback: used when a cut has no detected payoff instant to
    protect, so the plain peak is the best (only) thing to center on."""
    if not target or core_out - core_in <= target:
        return core_in, core_out
    peak = max(core_in, min(peak, core_out))
    lead = int(round(target * lead_frac))
    ci = max(core_in, peak - lead)
    co = min(core_out, ci + target)
    ci = max(core_in, co - target)
    return ci, co


def _audio_onset_anchors(rms_db: List[float], hop_ms: int, s: int, e: int) -> List[int]:
    """Instants in [s, e) where the rms_db envelope jumps by at least
    AUDIO_ANCHOR_RISE_DB over one hop -- a percussive transient (impact,
    ball-crack, clap) that optical flow tends to miss because flow peaks on the
    swing, not the contact. De-duplicated to one anchor per
    AUDIO_ANCHOR_MIN_GAP_MS so a single loud event isn't counted many times."""
    if not rms_db or hop_ms <= 0:
        return []
    lo = max(1, s // hop_ms)
    hi = min(len(rms_db) - 1, (e - 1) // hop_ms)
    out: List[int] = []
    last = -AUDIO_ANCHOR_MIN_GAP_MS
    for i in range(lo, hi + 1):
        if rms_db[i] - rms_db[i - 1] >= AUDIO_ANCHOR_RISE_DB:
            t = i * hop_ms
            if s <= t < e and t - last >= AUDIO_ANCHOR_MIN_GAP_MS:
                out.append(t)
                last = t
    return out


def _video_anchors(motion: Optional[dict], audio: Optional[dict], s: int, e: int) -> List[int]:
    """The important instants inside a video cut that tightness must NOT trim
    off: L1 subject-motion impacts (``action_points``) + sharp audio onsets.
    Deliberately SPARSE -- only genuine payoff moments, not every motion
    wiggle -- so a normal cut still insets tight, while a cut that contains a
    real impact keeps that impact. Sorted, clamped to (s, e)."""
    m, a = motion or {}, audio or {}
    anchors = {int(p["ts_ms"]) for p in (m.get("action_points") or [])
               if isinstance(p, dict) and "ts_ms" in p and s <= int(p["ts_ms"]) < e}
    anchors.update(_audio_onset_anchors(
        a.get("rms_db") or [], int(a.get("prosody_hop_ms") or 0), s, e))
    return sorted(anchors)


def _anchored_inset(s: int, e: int, peak: int, target: int, anchors: List[int]) -> Tuple[int, int]:
    """Shrink [s, e] toward its important core to ~``target`` ms, but GUARANTEE
    the kept core contains every anchor (with ANCHOR_PAD_MS breathing room).
    Only ever shrinks. With no anchors, falls back to the plain peak inset.

    When the anchor envelope is already wider than ``target``, the envelope
    WINS -- the core stays as wide as needed to hold every payoff instant
    (this is exactly the "clip is all action -> keep (almost) the whole thing"
    case: many spread-out anchors => the core can't shrink without dropping
    one, so it doesn't)."""
    if e - s <= target:
        return s, e
    if not anchors:
        return _core_inset(s, e, peak, target)
    a0, a1 = anchors[0], anchors[-1]
    lo, hi = a0 - ANCHOR_PAD_MS, a1 + ANCHOR_PAD_MS
    if hi - lo < target:                      # pad out to the desired tightness
        extra = target - (hi - lo)
        lo -= extra // 2
        hi = lo + target
    # Guarantee containment of every anchor, then clamp inside the cut. Clamping
    # can only pull the bounds further OUTWARD toward an anchor, never past it,
    # so no anchor is ever excluded -- the core-contains-anchors invariant.
    lo = max(s, min(lo, a0))
    hi = min(e, max(hi, a1))
    return lo, hi


def _tighten_video(placed: List[_Placed], energy: float,
                   motion: Optional[dict], audio: Optional[dict]) -> List[_Placed]:
    """The energy dial's TIGHTNESS effect on VIDEO cuts (done/shown), layered
    on top of the already-claimed partition (said is left untouched here):
    ANCHOR-AWARE peak-inset. Each cut shrinks toward its important core, but the
    core is guaranteed to contain every payoff instant inside the cut (L1
    impacts + audio onsets) -- tightness trims dead time, never a moment that
    matters. At Balanced and below, ``done_core_frac``/``shown_core_frac`` are
    None, so nothing changes -- the default feed stays a contiguous filmstrip.
    GRANULARITY (how many video cuts there are) is decided upstream, in
    ``video_segments.segment_video`` -- this is tightness only, so it never
    splits and never extends (extension would break the non-overlap invariant);
    only ever shrinks within the claimed span."""
    params = energy_to_params(energy)
    for p in placed:
        if p.primary not in (vocab.CHANNEL_DONE, vocab.CHANNEL_SHOWN):
            continue
        frac = params.done_core_frac if p.primary == vocab.CHANNEL_DONE else params.shown_core_frac
        if frac is None:
            continue
        target = max(VIDEO_CORE_FLOOR_MS, int(round(frac * (p.end_ms - p.start_ms))))
        anchors = _video_anchors(motion, audio, p.start_ms, p.end_ms)
        new_in, new_out = _anchored_inset(p.start_ms, p.end_ms, p.peak_ms, target, anchors)
        # The invariant this whole change exists to enforce: tightness may drop
        # dead time, but must never trim an important instant out of its cut.
        assert all(new_in <= t <= new_out for t in anchors), (
            f"anchored tightness dropped an anchor: cut [{new_in},{new_out}] "
            f"anchors={anchors}")
        p.start_ms, p.end_ms = new_in, new_out
    return placed


def _to_cut(file_id: str, p: _Placed) -> Cut:
    return Cut(
        file_id=file_id, src_in_ms=p.start_ms, src_out_ms=p.end_ms,
        tags=list(p.tags), primary=p.primary, label=p.label or "",
        speaker=p.speaker, peak_ms=p.peak_ms,
    )


# --------------------------------------------------------------------------
# Public entry point (pure)
# --------------------------------------------------------------------------

def partition_clip(clip: ClipArtifacts, energy: float = DEFAULT_SNAP_ENERGY) -> List[Cut]:
    """One clip's artifacts -> its non-overlapping, tag-bearing partition at a
    given energy. Pure: no DB/model call. Deterministic given (artifacts,
    energy).

    The energy dial acts here as:
      * GRANULARITY for said (which of a thought's nested levels the dial
        reads, and whether consecutive same-speaker thoughts merge into a
        turn) -- see ``_said_candidates``.
      * GRANULARITY for video (how many camera-settle/subject-beat boundaries
        ``video_segments.segment_video`` admits) -- see ``_video_candidates``.
      * TIGHTNESS for video (anchor-aware peak-inset -- shrinks each claimed
        cut toward its core but never trims off a payoff instant) --
        ``_tighten_video``.
      * TIGHTNESS for said (breath excision) is layered on separately, in
        ``tightness.py`` -- it never changes a said cut's own boundaries."""
    candidates = (
        _said_candidates(clip.thoughts, clip.audio, energy)
        + _video_candidates(clip.motion, clip.scene, clip.duration_ms, energy)
    )
    field = _build_field(clip)
    placed = _claim(candidates, field, clip.duration_ms)
    placed = _merge_continuous(placed, clip.scene)
    placed = _tighten_video(placed, energy, clip.motion, clip.audio)
    placed.sort(key=lambda p: p.start_ms)
    cuts = [_to_cut(clip.file_id, p) for p in placed]
    _assert_non_overlap(cuts)
    return cuts


# --------------------------------------------------------------------------
# DB loader + convenience wrapper (not pure -- real callers only)
# --------------------------------------------------------------------------

def _pg_conn():
    import psycopg
    from app.config import get_settings
    return psycopg.connect(get_settings().database_url, autocommit=True)


def load_clip_artifacts(file_id: str) -> Optional[ClipArtifacts]:
    """Load one file's L1/L2 artifacts into a ``ClipArtifacts``. None when the
    file doesn't exist / has no duration yet. Best-effort per artifact -- a
    missing table row simply leaves that facet empty."""
    from app.services.l3 import thought_segments as thoughts_mod

    with _pg_conn() as conn:
        row = conn.execute(
            "select duration_seconds from files where id = %s", (file_id,)
        ).fetchone()
        if not row or not row[0]:
            return None
        duration_ms = int(float(row[0]) * 1000)

        m_row = conn.execute(
            """
            select hop_ms, action_energy, action_points, camera_cut_cost,
                   camera_coherence, camera_stability, blur, camera_motion
              from motion_dynamics where file_id = %s
            """,
            (file_id,),
        ).fetchone()
        motion = None
        if m_row:
            motion = {
                "hop_ms": m_row[0], "action_energy": m_row[1] or [],
                "action_points": m_row[2] or [], "camera_cut_cost": m_row[3] or [],
                "camera_coherence": m_row[4] or [], "camera_stability": m_row[5] or [],
                "blur": m_row[6] or [], "camera_motion": m_row[7] or [],
            }

        s_row = conn.execute(
            "select hop_ms, shot_points, composition_points from scene_cuts where file_id = %s",
            (file_id,),
        ).fetchone()
        scene = None
        if s_row:
            scene = {"hop_ms": s_row[0], "shot_points": s_row[1] or [],
                     "composition_points": s_row[2] or []}

        a_row = conn.execute(
            """
            select dialogue_cut_cost, dialogue_cut_hop_ms, dialogue_cut_points,
                   beat_cut_cost, beat_cut_hop_ms, beat_cut_points,
                   rms_db, prosody_hop_ms, f0_hz
              from audio_features where file_id = %s
            """,
            (file_id,),
        ).fetchone()
        audio = None
        if a_row:
            audio = {
                "dialogue_cut_cost": a_row[0] or [], "dialogue_cut_hop_ms": a_row[1] or 100,
                "dialogue_cut_points": a_row[2] or [], "beat_cut_cost": a_row[3] or [],
                "beat_cut_hop_ms": a_row[4] or 100, "beat_cut_points": a_row[5] or [],
                "rms_db": a_row[6] or [], "prosody_hop_ms": a_row[7] or 100,
                "f0_hz": a_row[8] or [],
            }

    thoughts = thoughts_mod.get_thoughts(file_id)
    return ClipArtifacts(file_id=file_id, duration_ms=duration_ms, thoughts=thoughts,
                         motion=motion, scene=scene, audio=audio)


def build_partition(file_id: str, energy: float = DEFAULT_SNAP_ENERGY) -> List[Cut]:
    """Convenience: load + partition one file at ``energy``. Empty when the
    file has no duration yet (upload still in flight)."""
    clip = load_clip_artifacts(file_id)
    if clip is None:
        return []
    return partition_clip(clip, energy)
