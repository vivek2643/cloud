"""
Cuts v2: the unified priority partition.

Replaces the overlapping, energy-laddered, channel-tabbed hero-cuts feed with a
DETERMINISTIC, NON-OVERLAPPING partition of each video into tag-bearing cuts.
See ``cuts_v2.plan.md`` for the full design; this module is Phase B2, the core.

North star (see the plan for the full rationale):
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
  5. Boundaries are DETECTED, not chosen by a granularity slider (candidate
     spans below are ENERGY-INDEPENDENT; the only knob is tightness, Phase B3,
     applied on top of the claimed cuts, never here).
  6. Over-split, never under-split: a free remainder too small to be a
     meaningful sub-unit is silently absorbed (no cut, no tag) rather than
     forced into one.

Pure core: ``partition_clip(clip: ClipArtifacts) -> List[Cut]`` takes already-
loaded clip artifacts and does no DB/model call (mirrors ``atoms.build_atoms``'s
purity), so it is trivially testable -- see ``scripts/test_partition.py``.
``build_partition(file_id)`` is the convenience wrapper that loads + partitions
for real callers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.l1 import fused_seams as fseams
from app.services.l1.cut_grid_common import percentile
from app.services.l3 import vocab
from app.services.l3.partition_params import (
    ACTION_CALM_PCTL,
    DEFAULT_SNAP_ENERGY,
    DONE_MIN_MS,
    MERGE_GAP_MS,
    MIN_SUBUNIT_MS,
    OVERLAP_TAG_FRAC,
    PRIORITY_DONE,
    PRIORITY_SAID,
    PRIORITY_SHOWN,
)
from app.services.l3.thought_segments import Thought

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
    motion: Optional[dict] = None    # {hop_ms, action_energy, action_points, camera_cut_cost, camera_coherence, camera_stability, blur}
    scene: Optional[dict] = None     # {hop_ms, shot_points, composition_points}
    audio: Optional[dict] = None     # {dialogue_cut_cost, dialogue_cut_hop_ms, dialogue_cut_points, beat_cut_cost, beat_cut_hop_ms, beat_cut_points}


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


def _said_candidates(thoughts: List[Thought]) -> List[_Candidate]:
    """One candidate per THOUGHT (the complete idea) -- already the clean,
    speaker-pure, silence-ceilinged unit ``thought_segments`` exists to produce
    (LLM-derived when available, else its own deterministic L1
    dialogue_segments fallback -- both of the plan's "what we reuse" bullets
    for said live behind this one call)."""
    out: List[_Candidate] = []
    for t in thoughts:
        s, e = t.thought.raw_in_ms, t.thought.raw_out_ms
        if e <= s:
            continue
        peak = (s + e) // 2
        if t.punch and t.punch.raw_out_ms > t.punch.raw_in_ms:
            peak = (t.punch.raw_in_ms + t.punch.raw_out_ms) // 2
        out.append(_Candidate(vocab.CHANNEL_SAID, s, e, peak, t.thought.text, t.speaker))
    out.sort(key=lambda c: c.start_ms)
    return out


def _at(arr: List[float], hop: int, ts_ms: int) -> float:
    if not arr or hop <= 0:
        return 0.0
    i = max(0, min(len(arr) - 1, ts_ms // hop))
    return arr[i]


def _done_candidates(motion: Optional[dict]) -> List[_Candidate]:
    """Action beats: each impact's energy rise -> peak -> fall window, expanded
    outward from ``action_points`` while ``action_energy`` stays above the
    calm floor. Overlapping/touching windows from separate impacts merge into
    one (keeping the stronger peak)."""
    motion = motion or {}
    energy = motion.get("action_energy") or []
    points = motion.get("action_points") or []
    hop = int(motion.get("hop_ms") or 0)
    if not energy or not points or hop <= 0:
        return []

    floor = percentile(energy, ACTION_CALM_PCTL)
    windows: List[Tuple[int, int, int]] = []   # (start_ms, end_ms, peak_ms)
    for p in points:
        peak_ms = int(p.get("ts_ms", 0))
        pi = max(0, min(len(energy) - 1, peak_ms // hop))
        lo, hi = pi, pi
        while lo > 0 and energy[lo - 1] > floor:
            lo -= 1
        while hi < len(energy) - 1 and energy[hi + 1] > floor:
            hi += 1
        windows.append((lo * hop, (hi + 1) * hop, peak_ms))
    windows.sort(key=lambda w: w[0])

    merged: List[Tuple[int, int, int]] = []
    for s, e, pk in windows:
        if merged and s <= merged[-1][1]:
            ps, pe, ppk = merged[-1]
            better_pk = ppk if _at(energy, hop, ppk) >= _at(energy, hop, pk) else pk
            merged[-1] = (ps, max(pe, e), better_pk)
        else:
            merged.append((s, e, pk))

    return [
        _Candidate(vocab.CHANNEL_DONE, s, e, pk, "action")
        for s, e, pk in merged if e - s >= DONE_MIN_MS
    ]


def _sharpest_ms(motion: Optional[dict], start_ms: int, end_ms: int, default_ms: int) -> int:
    """The least-blurred instant in [start,end] -- the thumbnail-worthy frame
    for a held (shown) span. Falls back to ``default_ms`` when blur isn't
    available for this window."""
    motion = motion or {}
    blur = motion.get("blur") or []
    hop = int(motion.get("hop_ms") or 0)
    if not blur or hop <= 0:
        return default_ms
    lo, hi = max(0, start_ms // hop), min(len(blur) - 1, end_ms // hop)
    if hi < lo:
        return default_ms
    best_i = min(range(lo, hi + 1), key=lambda i: blur[i])
    return best_i * hop


def _shown_candidates(scene: Optional[dict], motion: Optional[dict], duration_ms: int) -> List[_Candidate]:
    """Held/stable stretches, bounded by scene/composition change (Phase B1).

    Candidates cover the WHOLE clip by construction (shot + composition
    boundaries as internal split points, or one candidate spanning the whole
    clip when scene detection found nothing) -- shown is the lowest-priority,
    catch-all channel, so full coverage here is what guarantees the partition
    is gap-free (North Star: "no junk removal yet" = a contiguous filmstrip).
    Whatever's actually action-heavy within a shown candidate will already
    have been claimed by `done` before `shown` gets to it (claim order), so
    the "low action energy" character the plan names falls out of the claim
    order rather than needing a separate pre-filter here."""
    scene = scene or {}
    bounds = {0, duration_ms}
    for p in (scene.get("shot_points") or []) + (scene.get("composition_points") or []):
        ts = int(p.get("ts_ms", -1))
        if 0 < ts < duration_ms:
            bounds.add(ts)
    ordered = sorted(bounds)
    out: List[_Candidate] = []
    for s, e in zip(ordered, ordered[1:]):
        if e <= s:
            continue
        peak = _sharpest_ms(motion, s, e, (s + e) // 2)
        out.append(_Candidate(vocab.CHANNEL_SHOWN, s, e, peak, "shown"))
    return out


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


def _to_cut(file_id: str, p: _Placed) -> Cut:
    return Cut(
        file_id=file_id, src_in_ms=p.start_ms, src_out_ms=p.end_ms,
        tags=list(p.tags), primary=p.primary, label=p.label or "",
        speaker=p.speaker, peak_ms=p.peak_ms,
    )


# --------------------------------------------------------------------------
# Public entry point (pure)
# --------------------------------------------------------------------------

def partition_clip(clip: ClipArtifacts) -> List[Cut]:
    """One clip's artifacts -> its non-overlapping, tag-bearing partition.

    Pure: no DB/model call. Deterministic given the same artifacts."""
    candidates = (
        _said_candidates(clip.thoughts)
        + _done_candidates(clip.motion)
        + _shown_candidates(clip.scene, clip.motion, clip.duration_ms)
    )
    field = _build_field(clip)
    placed = _claim(candidates, field, clip.duration_ms)
    placed = _merge_continuous(placed, clip.scene)
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
                   camera_coherence, camera_stability, blur
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
                "blur": m_row[6] or [],
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
                   beat_cut_cost, beat_cut_hop_ms, beat_cut_points
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
            }

    thoughts = thoughts_mod.get_thoughts(file_id)
    return ClipArtifacts(file_id=file_id, duration_ms=duration_ms, thoughts=thoughts,
                         motion=motion, scene=scene, audio=audio)


def build_partition(file_id: str) -> List[Cut]:
    """Convenience: load + partition one file. Empty when the file has no
    duration yet (upload still in flight)."""
    clip = load_clip_artifacts(file_id)
    if clip is None:
        return []
    return partition_clip(clip)
