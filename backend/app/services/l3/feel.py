"""
Feel simulator: a PURE, VLM-free read of how the current edit FEELS.

The agentic brain (Phase 3) shouldn't need an expensive pass to know whether its
cut drags, races, or jump-cuts. Everything about "feel" is a free projection of
the edit's OWN structure -- shot lengths, speaking pace, who is on screen. This
module computes those dimensions deterministically from the timeline + a little
map context, and narrates them back in anchored prose the brain can act on
("energy sags cuts 4-6; two same-speaker cuts back-to-back at cut 7").

Design:
  * PURE. Inputs are the timeline segments + a read-only lookup (moment meta by
    ref). No DB, no LLM, no network -- safe to call every loop turn. The context
    builder (``observe.build_context``) does the DB reads once.
  * Honest, not prescriptive. It reports what IS (pace, rhythm, tone), it does
    not decide what SHOULD change -- the brain does. Numbers are anchored to cut
    positions so the brain can point its edits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional, Tuple

from app.services.l1.frame_descriptors import hamming64, nearest_phash

# A cut under this plays as "fast"; over the long threshold it "lingers".
_FAST_MS = 2000
_LINGER_MS = 6000
# A run of at least this many fast cuts back-to-back reads as a burst.
_FAST_RUN = 3

# brain_continuity_awareness.plan.md section 3.1 Step B: a same-clip forward/
# backward gap within this tolerance reads as one continuous shot -- matches
# footage_map._RUN_GAP_MS (the source-run grouping threshold), so "these two
# beats are one source run" and "this join is seamless" agree.
_JOIN_CONTIG_MS = 500

# brain_mirror_readside.plan.md section 2.2: pHash lookup tolerance for a seam
# instant -- matches frame_descriptors.NEAREST_MS_DEFAULT (just over one 500ms
# sampling hop), so the nearest sample always resolves when one exists.
_JOIN_NEAREST_MS = 600

# Content-relative verdict bands (band-aid A retired: no single fixed cutoff).
# Calibrated against the L1 empirical anchors (test_frame_descriptors.py):
# identical ~=0 bits, visually similar ~=6 bits, clearly different ~=36 bits.
_PHASH_SEAMLESS_BITS = 4.0    # <= this: same framing, negligible change (a match cut).
_PHASH_SAMESHOT_BITS = 12.0   # <= this (and same file): still one continuous
                              # framing but the content moved -- a jump-cut,
                              # not a new shot. Beyond it, the picture itself
                              # changed (a motivated cut, or a genuinely
                              # different piece of the same file, e.g.
                              # talking-head -> slide).
# The "clearly different" empirical anchor -- the reference point the
# project's own scale is measured against (see _project_scale).
_PHASH_DIFFERENT_ANCHOR = 36.0
# Never let a project's own scale shrink/grow the bands beyond this range --
# a degenerate scale (too little data, or a pathological project) still
# yields sane, boring verdicts rather than a collapsed or runaway band.
_PHASH_SCALE_MIN = 0.5
_PHASH_SCALE_MAX = 2.0


@dataclass
class CutFeel:
    """Feel of one timeline cut, in play order (1-based ``pos`` for narration)."""
    pos: int
    ref: Optional[str]
    file_id: str
    # brain_continuity_awareness.plan.md section 3.1 Step A: the SOURCE span
    # (seg["in_ms"]/["out_ms"]) -- distinct from dur_ms (the PLAYED length) --
    # so join_continuity can compare where each cut sits in its own source
    # clip, not just how long it plays.
    src_in_ms: int
    src_out_ms: int
    dur_ms: int
    words: int
    pace_wps: float          # spoken words per second (0 for silent/video cuts)
    is_speech: bool
    speaker: Optional[str]
    channel: Optional[str]   # said | done | shown | None
    energy: float            # 0..1 proxy (short + fast -> high; long + slow -> low)


@dataclass
class FeelReport:
    cuts: List[CutFeel] = field(default_factory=list)
    total_ms: int = 0
    # brain_mirror_readside.plan.md section 2.4: stashed so narrate() can
    # call join_continuity(self.cuts, self.descriptors) without the caller
    # having to thread it through separately. {} (not None) when the caller
    # passed nothing -- join_continuity then fails open to "cut" for every
    # non-contiguous seam (no false "seamless"/"jump-cut" without pixels).
    descriptors: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)

    # -- aggregate reads (all derived, all cheap) --
    @property
    def avg_pace(self) -> float:
        spoken = [c.pace_wps for c in self.cuts if c.is_speech and c.pace_wps > 0]
        return round(sum(spoken) / len(spoken), 2) if spoken else 0.0

    def narrate(self) -> str:
        """Anchored prose: what the edit feels like, pointed at cut positions."""
        if not self.cuts:
            return "The timeline is empty -- nothing to feel yet."
        parts: List[str] = []
        n = len(self.cuts)
        secs = round(self.total_ms / 1000.0, 1)
        parts.append(f"{n} cut{'s' if n != 1 else ''}, {secs}s total"
                     + (f", ~{self.avg_pace} words/s when talking" if self.avg_pace else ""))

        for lo, hi in _fast_runs(self.cuts):
            parts.append(f"cuts {lo}-{hi} race (a burst of sub-{_FAST_MS // 1000}s cuts)"
                         if hi > lo else f"cut {lo} is a quick sub-{_FAST_MS // 1000}s beat")
        for lo, hi in _low_energy_runs(self.cuts):
            parts.append(f"energy sags cuts {lo}-{hi}" if hi > lo else f"cut {lo} drags")
        for pos in _lingers(self.cuts):
            parts.append(f"cut {pos} lingers ({self.cuts[pos - 1].dur_ms // 1000}s)")
        for lo, hi in _same_speaker_runs(self.cuts):
            spk = self.cuts[lo - 1].speaker or "one speaker"
            parts.append(f"cuts {lo}-{hi} stay on {spk} back-to-back (jump-cut risk)")
        jumps = [j for j in join_continuity(self.cuts, self.descriptors) if j["kind"] == "jump-cut"]
        if jumps:
            spans = ", ".join(f"{j['from_pos']}→{j['to_pos']}" for j in jumps)
            parts.append(f"jump-cut{'s' if len(jumps) != 1 else ''} at {spans} "
                         "(same shot's framing, cut discontinuously within it)")

        return "; ".join(parts) + "."

    def to_dict(self) -> dict:
        return {
            "total_ms": self.total_ms,
            "cut_count": len(self.cuts),
            "avg_pace_wps": self.avg_pace,
            "narration": self.narrate(),
            "cuts": [
                {"pos": c.pos, "ref": c.ref, "dur_ms": c.dur_ms, "pace_wps": c.pace_wps,
                 "channel": c.channel, "speaker": c.speaker,
                 "energy": round(c.energy, 2)}
                for c in self.cuts
            ],
        }


def simulate(
    timeline: List[dict],
    meta_by_ref: Optional[Dict[str, dict]] = None,
    descriptors: Optional[Dict[str, List[Tuple[int, int]]]] = None,
) -> FeelReport:
    """Compute the feel of a timeline. ``meta_by_ref`` maps a segment's map ref
    to its moment node (for speaker/channel). ``descriptors`` is the per-file
    pHash samples (``l1.frame_descriptors.load_descriptors``' own shape) this
    turn's seam-continuity grading reads -- optional, stashed on the returned
    ``FeelReport`` so ``narrate()``/callers can re-grade without re-fetching.
    Feel degrades gracefully with either omitted (pace/rhythm still compute;
    seam grading fails open to "cut" without pixels)."""
    meta_by_ref = meta_by_ref or {}
    cuts: List[CutFeel] = []
    total = 0
    for i, seg in enumerate(timeline):
        src_in_ms, src_out_ms = int(seg.get("in_ms", 0)), int(seg.get("out_ms", 0))
        dur = max(0, src_out_ms - src_in_ms)
        if dur <= 0:
            continue
        total += dur
        content = (seg.get("content") or "").strip()
        words = len(content.split()) if content else 0
        is_speech = seg.get("axis") == "speech"
        pace = round(words / (dur / 1000.0), 2) if (is_speech and words and dur) else 0.0
        meta = meta_by_ref.get(seg.get("ref") or "") or {}
        cuts.append(CutFeel(
            pos=len(cuts) + 1,
            ref=seg.get("ref"),
            file_id=str(seg.get("file_id") or ""),
            src_in_ms=src_in_ms, src_out_ms=src_out_ms,
            dur_ms=dur,
            words=words,
            pace_wps=pace,
            is_speech=is_speech,
            speaker=meta.get("speaker_person"),
            channel=meta.get("channel") or ("said" if is_speech else None),
            energy=0.0,  # filled below (needs the whole set for normalisation)
        ))
    _score_energy(cuts)
    return FeelReport(cuts=cuts, total_ms=total, descriptors=descriptors or {})


# --------------------------------------------------------------------------
# Derivations
# --------------------------------------------------------------------------

def _score_energy(cuts: List[CutFeel]) -> None:
    """Energy proxy per cut: short shots and faster speech read as higher energy;
    long, slow ones as lower. Normalised against this edit's own median cut so it
    is relative to the piece, not an absolute scale."""
    if not cuts:
        return
    med = median([c.dur_ms for c in cuts]) or 1
    paces = [c.pace_wps for c in cuts if c.pace_wps > 0]
    pace_med = median(paces) if paces else 0.0
    for c in cuts:
        # Shortness: 1.0 at half the median length, 0.0 at double it.
        short = _clamp01(1.0 - (c.dur_ms / (med * 2.0)))
        if c.pace_wps > 0 and pace_med > 0:
            fast = _clamp01(c.pace_wps / (pace_med * 1.5))
            c.energy = round(0.5 * short + 0.5 * fast, 3)
        else:
            c.energy = round(short, 3)


def _fast_runs(cuts: List[CutFeel]) -> List[tuple[int, int]]:
    return _runs([c.pos for c in cuts if c.dur_ms < _FAST_MS], min_len=_FAST_RUN)


def _lingers(cuts: List[CutFeel]) -> List[int]:
    return [c.pos for c in cuts if c.dur_ms >= _LINGER_MS]


def _low_energy_runs(cuts: List[CutFeel]) -> List[tuple[int, int]]:
    if len(cuts) < 3:
        return []
    med = median([c.energy for c in cuts])
    low = [c.pos for c in cuts if c.energy < med * 0.6]
    return _runs(low, min_len=3)


def _same_speaker_runs(cuts: List[CutFeel]) -> List[tuple[int, int]]:
    """Positions of >=2 adjacent speech cuts sharing one (known) speaker."""
    out: List[tuple[int, int]] = []
    i = 0
    while i < len(cuts):
        c = cuts[i]
        if c.is_speech and c.speaker:
            j = i
            while (j + 1 < len(cuts) and cuts[j + 1].is_speech
                   and cuts[j + 1].speaker == c.speaker):
                j += 1
            if j > i:
                out.append((cuts[i].pos, cuts[j].pos))
            i = j + 1
        else:
            i += 1
    return out


def _is_source_contiguous(prev: CutFeel, cur: CutFeel) -> bool:
    return bool(prev.file_id) and prev.file_id == cur.file_id \
        and -_JOIN_CONTIG_MS <= (cur.src_in_ms - prev.src_out_ms) <= _JOIN_CONTIG_MS


def _seam_phash(
    descriptors: Dict[str, List[Tuple[int, int]]], file_id: str, t_ms: int, nearest_ms: int,
) -> Optional[int]:
    return nearest_phash(descriptors.get(file_id, []), t_ms, nearest_ms=nearest_ms)


def _project_scale(
    cuts: List[CutFeel], descriptors: Dict[str, List[Tuple[int, int]]], nearest_ms: int,
) -> float:
    """brain_mirror_readside.plan.md section 2.2: how far apart is "genuinely
    different" content FOR THIS PROJECT, as a multiplier on the (absolute,
    empirically-anchored) verdict bands -- a noisy/busy shoot widens the
    bands, a clean one narrows them, so the same raw bit-count doesn't mean
    the same thing everywhere. Computed once per classify call from the
    median Hamming distance across this timeline's own same-file, source-
    NON-contiguous seams (the population the mid/cut boundary actually
    has to separate) -- relative to the L1 empirical "clearly different"
    anchor (~36 bits), clamped to [_PHASH_SCALE_MIN, _PHASH_SCALE_MAX].
    1.0 (no adjustment) when there isn't enough same-file data to judge --
    the common case for a project with few multi-cut same-source takes."""
    dists: List[int] = []
    for i in range(1, len(cuts)):
        prev, cur = cuts[i - 1], cuts[i]
        if not prev.file_id or prev.file_id != cur.file_id or _is_source_contiguous(prev, cur):
            continue
        hA = _seam_phash(descriptors, prev.file_id, prev.src_out_ms, nearest_ms)
        hB = _seam_phash(descriptors, cur.file_id, cur.src_in_ms, nearest_ms)
        if hA is None or hB is None:
            continue
        dists.append(hamming64(hA, hB))
    if not dists:
        return 1.0
    scale = median(dists) / _PHASH_DIFFERENT_ANCHOR
    return max(_PHASH_SCALE_MIN, min(_PHASH_SCALE_MAX, scale))


def join_continuity(
    cuts: List[CutFeel],
    descriptors: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    *,
    nearest_ms: int = _JOIN_NEAREST_MS,
) -> List[dict]:
    """brain_mirror_readside.plan.md section 2.2: classify the JOIN from each
    cut to the one before it (pos>=2) -- distinct from every other
    continuity signal in this codebase, which describes a cut's SOURCE
    neighbors (what it welded to in the raw clip); this describes the pair
    of cuts the brain actually placed adjacent ON THE TIMELINE. Pure --
    ``descriptors`` (``l1.frame_descriptors.load_descriptors``'s shape) is
    passed in, never fetched here.

    Retires two band-aids:
      - **A (fixed threshold):** the seamless/jump-cut/cut boundary is a
        content-relative BAND (``_project_scale``), not one magic cutoff.
      - **B (file_id gate):** file identity is no longer an arbiter of
        continuity -- same-source contiguity is only a cheap FAST PATH
        (step 1); the pixel grading in step 2 runs for ANY pair, same-file
        or not, so a cross-file match-cut can read "seamless" and a
        same-file content change (talking-head -> slide) correctly reads
        "cut", not a false jump.

    Step 1 (fast path, no pixels): same file AND source-contiguous
    (|gap| <= _JOIN_CONTIG_MS) -> "seamless". Continuous same-source
    playback is one shot by construction.

    Step 2 (otherwise, pixel-graded): look up the nearest sampled frame at
    the previous segment's OUT instant and the current segment's IN
    instant. Either missing -> fail open to "cut" (never a false jump from
    absent data). Both present -> grade the Hamming distance through the
    project-relative bands:
      - d <= seamless band -> "seamless" (same framing, negligible change).
      - seamless < d <= same-shot band, AND same file -> "jump-cut" (one
        continuous framing, source played out of order -- e.g. clip93
        408s -> 209s). The identical band, cross-file, is a "cut" instead
        (a deliberate match-adjacent join, not a jump).
      - beyond the same-shot band -> "cut" (the picture genuinely changed).

    Each verdict keeps the original keys (``from_pos``, ``to_pos``, ``kind``,
    ``same_clip``, ``src_gap_ms``, ``backward``) and gains ``pic_bits`` (the
    Hamming distance, ``None`` when not looked up/available) and
    ``pic_scale`` (this call's project-relative multiplier) so the mirror
    can explain why a seam reads the way it does."""
    descriptors = descriptors or {}
    scale = _project_scale(cuts, descriptors, nearest_ms)
    seamless_bits = _PHASH_SEAMLESS_BITS * scale
    sameshot_bits = _PHASH_SAMESHOT_BITS * scale

    out: List[dict] = []
    for i in range(1, len(cuts)):
        prev, cur = cuts[i - 1], cuts[i]
        same_clip = bool(prev.file_id) and prev.file_id == cur.file_id
        src_gap_ms = (cur.src_in_ms - prev.src_out_ms) if same_clip else None
        backward = same_clip and cur.src_in_ms < prev.src_out_ms
        pic_bits: Optional[int] = None

        if _is_source_contiguous(prev, cur):
            kind = "seamless"
        else:
            hA = _seam_phash(descriptors, prev.file_id, prev.src_out_ms, nearest_ms)
            hB = _seam_phash(descriptors, cur.file_id, cur.src_in_ms, nearest_ms)
            if hA is None or hB is None:
                kind = "cut"
            else:
                pic_bits = hamming64(hA, hB)
                if pic_bits <= seamless_bits:
                    kind = "seamless"
                elif pic_bits <= sameshot_bits and same_clip:
                    kind = "jump-cut"
                else:
                    kind = "cut"

        out.append({
            "from_pos": prev.pos, "to_pos": cur.pos, "kind": kind,
            "same_clip": same_clip, "src_gap_ms": src_gap_ms, "backward": backward,
            "pic_bits": pic_bits, "pic_scale": round(scale, 3),
        })
    return out


def _runs(positions: List[int], min_len: int) -> List[tuple[int, int]]:
    """Collapse a sorted list of positions into contiguous [lo, hi] runs of at
    least ``min_len`` consecutive positions."""
    if not positions:
        return []
    positions = sorted(set(positions))
    runs: List[tuple[int, int]] = []
    lo = prev = positions[0]
    for p in positions[1:]:
        if p == prev + 1:
            prev = p
            continue
        if prev - lo + 1 >= min_len:
            runs.append((lo, prev))
        lo = prev = p
    if prev - lo + 1 >= min_len:
        runs.append((lo, prev))
    return runs


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
