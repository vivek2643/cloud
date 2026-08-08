"""
vcut_moment_energy.plan.md -- THE ALGORITHM, flag-based and file-wide. Turns
a per-file pool of "moment flags" (Pass 1 plants keep-frame timestamps only
-- no boundaries, no grouping, no selection) into final cut boundaries using
the seam-quality curve S(t) and a single global energy. Pure function, no
I/O, no model call, no footage-type branch.

    S(t) decides the FRAME, never the cut. Energy only ever SHRINKS a
    flag's reach; it never invents footage or merges across a STRONG SEAM
    (section 3) -- cut count is a side effect of shrinking, never a
    selection (nothing is ever ranked or dropped here).

Supersedes seam_cut_pipeline.plan.md's per-loose-cut model: today's
(pre-this-plan) resolver clamped each peak's window to its OWN VLM-drawn
loose-cut span and the midpoint to its neighbor PEAK. This version pools
ALL of a file's flags together, clamps to the FILE extent and either a
STRONG-SEAM WALL (a genuinely different piece of content) or the neighbor
midpoint (section 4.1) -- so Pass 1 no longer draws any geometry at all.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.vcut.params import (
    MIN_CUT_GAP_MS, MIN_CUT_MS, PEAK_SNAP_MS, REACH_MAX_MS, REACH_MIN_MS, S_DEAD_FRAC,
    SNAP_MS, STRONG_SEAM_FRAC, TAG_SPLIT,
)

DEFAULT_TAG = "both"


# --------------------------------------------------------------------------
# Input types (section 2.4) -- a flat pool of moment flags per file. A
# back-compat from_dict (section 7.3) lets the energy endpoint keep re-
# resolving OLD runs (persisted under the previous loose_cuts/peaks shape)
# without a re-ingest.
# --------------------------------------------------------------------------

@dataclass
class MomentFlag:
    t_ms: int
    shape: str = DEFAULT_TAG   # "build" | "settle" | "both"
    summary: str = ""
    # vcut_pass2_video_specifics.plan.md section 7.1: planner output
    # (qplan.py) and Pass 2's answers, living on the FLAG rather than any
    # ephemeral cut id -- resolve.py composes these onto whatever cut ends
    # up containing the flag, at ANY energy, so they can never be wiped by
    # an energy re-resolve.
    question_ids: List[str] = field(default_factory=list)
    custom_questions: List[Dict[str, str]] = field(default_factory=list)
    specifics: Dict[str, Any] = field(default_factory=dict)
    # reframe_vcut_geometry.plan.md section 3: the subject anchor, normalized
    # [x, y, w, h] in the (uprighted) source frame -- Pass 2 emits it for
    # EVERY moment (not gated by question_ids/the bank, unlike specifics),
    # and it composes onto a resolved cut the same representative-flag way,
    # so the energy dial can't wipe it either.
    subject_box: Optional[Tuple[float, float, float, float]] = None


@dataclass
class FilePlan:
    file_id: str
    flags: List[MomentFlag] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {"flags": [
            {"t_ms": f.t_ms, "shape": f.shape, "summary": f.summary,
             "question_ids": f.question_ids, "custom_questions": f.custom_questions,
             "specifics": f.specifics,
             "subject_box": list(f.subject_box) if f.subject_box else None}
            for f in self.flags
        ]}

    @staticmethod
    def from_dict(file_id: str, data: Dict[str, Any]) -> "FilePlan":
        if "flags" in data:
            flags = [
                MomentFlag(
                    t_ms=int(f["t_ms"]), shape=f.get("shape") or DEFAULT_TAG, summary=f.get("summary") or "",
                    question_ids=list(f.get("question_ids") or []),
                    custom_questions=list(f.get("custom_questions") or []),
                    specifics=dict(f.get("specifics") or {}),
                    subject_box=tuple(f["subject_box"]) if f.get("subject_box") else None,
                )
                for f in (data.get("flags") or [])
            ]
            return FilePlan(file_id=file_id, flags=flags)
        # Legacy shape (seam_cut_pipeline.plan.md section 4):
        # {meaning, question_ids, loose_cuts: [{span_ms, peaks: [{t_ms, tag}]}]}
        # -- flatten every loose_cut's peaks into flags, carrying the
        # clip-level meaning down onto each flattened flag's summary so the
        # dial keeps working on a not-yet-re-ingested run. question_ids/
        # custom_questions/specifics stay at their dataclass defaults (empty)
        # -- a pre-existing run simply has no planner/Pass-2 output yet.
        meaning = data.get("meaning") or ""
        flags = [
            MomentFlag(t_ms=int(p["t_ms"]), shape=p.get("tag") or DEFAULT_TAG, summary=meaning)
            for lc in (data.get("loose_cuts") or [])
            for p in (lc.get("peaks") or [])
        ]
        return FilePlan(file_id=file_id, flags=flags)


@dataclass
class MomentPlan:
    files: List[FilePlan] = field(default_factory=list)
    # vcut_pass2_video_specifics.plan.md section 4.4: the planner's one
    # run-level judgment (e.g. "performance"/"tutorial"/"vlog") -- carried
    # alongside the per-file flags under a reserved "__meta__" key so old
    # plans (no key at all) round-trip to genre="" without any migration.
    genre: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {f.file_id: f.to_dict() for f in self.files}
        if self.genre:
            out["__meta__"] = {"genre": self.genre}
        return out

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MomentPlan":
        meta = data.get("__meta__") or {}
        files = [FilePlan.from_dict(file_id, d) for file_id, d in data.items() if file_id != "__meta__"]
        return MomentPlan(files=files, genre=meta.get("genre") or "")


# --------------------------------------------------------------------------
# Output type
# --------------------------------------------------------------------------

@dataclass
class ResolvedCut:
    file_id: str
    in_ms: int
    out_ms: int
    peak_ms: int
    tag: str
    summary: str
    # section 7.2: composed from the group of flags this cut absorbed --
    # a single-flag cut's specifics pass through unchanged; a merged cut
    # carries the representative flag's fields plus a "moments" mini shot-
    # list (see _composed_specifics). Always JSON-serializable.
    specifics: Dict[str, Any] = field(default_factory=dict)
    # reframe_vcut_geometry.plan.md section 3: the representative flag's own
    # subject_box (no merge -- a box doesn't compose across moments the way
    # specifics do); None when no absorbed flag had one.
    subject_box: Optional[Tuple[float, float, float, float]] = None


# --------------------------------------------------------------------------
# Internal candidate -- a resolved window still carrying its own (wall-
# aware) extent bounds, needed by the floors below (dropped before
# returning).
# --------------------------------------------------------------------------

@dataclass
class _Candidate:
    file_id: str
    a: float
    b: float
    peak_ms: int
    tag: str
    summary: str
    span_lo: float
    span_hi: float
    has_peak: bool = True
    specifics: Dict[str, Any] = field(default_factory=dict)
    subject_box: Optional[Tuple[float, float, float, float]] = None


# --------------------------------------------------------------------------
# Step 0 -- refine a flag onto real content (unchanged)
# --------------------------------------------------------------------------

def _has_signal(track: List[float]) -> bool:
    return bool(track) and any(v > 0 for v in track)


def _refine_peak_ms(
    t_ms: int, hop_ms: int, action_energy: List[float], frame_diff: List[float],
    radius_ms: int = PEAK_SNAP_MS,
) -> int:
    """Snap ``t_ms`` to the index of highest action_energy within
    +/- radius_ms (falls back to frame_diff when action_energy carries no
    signal at all). Using "highest value in the window" rather than a
    strict interior local-maximum test is deliberate: a strict test can find
    NOTHING inside a small window when the track is monotonic across it,
    which would leave the VLM's raw (approximate) timestamp ungrounded --
    the whole point of this step."""
    track = action_energy if _has_signal(action_energy) else frame_diff
    if not track or hop_ms <= 0:
        return t_ms
    lo_i = max(0, (t_ms - radius_ms) // hop_ms)
    hi_i = min(len(track) - 1, (t_ms + radius_ms) // hop_ms)
    if lo_i > hi_i:
        return t_ms
    best_i = max(range(int(lo_i), int(hi_i) + 1), key=lambda i: track[i])
    return best_i * hop_ms


# --------------------------------------------------------------------------
# Steps 1-2 -- energy -> keep-width -> per-flag window (unchanged)
# --------------------------------------------------------------------------

def _window_width_ms(energy: float) -> float:
    e = min(1.0, max(0.0, energy))
    return REACH_MAX_MS + (REACH_MIN_MS - REACH_MAX_MS) * e


def _tag_window(peak_ms: int, tag: str, width_ms: float) -> Tuple[float, float]:
    before_frac, after_frac = TAG_SPLIT.get(tag, TAG_SPLIT[DEFAULT_TAG])
    return (peak_ms - width_ms * before_frac, peak_ms + width_ms * after_frac)


# --------------------------------------------------------------------------
# Section 3 -- strong-seam wall
# --------------------------------------------------------------------------

def _strong_seam_between(
    peak_i: int, peak_j: int, hop_ms: int, S: List[float], threshold: float,
) -> Optional[int]:
    """The ms position of the strongest seam strictly BETWEEN two adjacent
    refined peaks, if it clears ``threshold`` -- else None (falls back to
    the plain midpoint). Searches (peak_i, peak_j) exclusive, so a returned
    wall always sits strictly between the two peaks, never on top of
    either."""
    if not S or hop_ms <= 0 or peak_j <= peak_i:
        return None
    lo_i = max(0, int(peak_i // hop_ms) + 1)
    hi_i = min(len(S) - 1, int(peak_j // hop_ms) - 1)
    if lo_i > hi_i:
        return None
    best_i = max(range(lo_i, hi_i + 1), key=lambda i: S[i])
    if S[best_i] < threshold:
        return None
    return best_i * hop_ms


def _wall_positions(peaks_ms: List[int], hop_ms: int, S: List[float]) -> List[Optional[int]]:
    """One entry per ADJACENT PAIR of sorted peaks (len == len(peaks_ms)-1):
    the strong-seam wall between them, or None (falls back to the
    midpoint, section 4.1)."""
    if len(peaks_ms) < 2:
        return []
    threshold = STRONG_SEAM_FRAC * max(S) if S else 0.0
    return [
        _strong_seam_between(peaks_ms[i], peaks_ms[i + 1], hop_ms, S, threshold)
        for i in range(len(peaks_ms) - 1)
    ]


# --------------------------------------------------------------------------
# Step 3 -- clamp to file extent + (wall else midpoint); division emergent
# (section 4.2 step 5)
# --------------------------------------------------------------------------

def _clamp_windows(
    peaks_ms: List[int], raw_windows: List[Tuple[float, float]], extent: Tuple[int, int],
    walls: List[Optional[int]],
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """Clamp each flag's window to the FILE extent and to (a strong-seam
    wall if one clears the threshold between it and its neighbor, else the
    plain midpoint) -- never past that bound. This is what makes division
    emergent (a weak/no seam lets adjacent windows reach and touch at low
    energy, fusing them) AND what makes a strong seam an absolute wall (its
    two sides can never touch, at ANY energy -- the wall side gets a strict
    +1ms gap from the midpoint side so touching-clamped windows can never
    accidentally merge across it; see _merge_windows' ``a <= cur["b"]``
    equality check).

    Returns (clamped_windows, per_flag_bounds) -- the second is threaded
    through _merge_windows (unchanged merge algorithm, section 4.2 step 6)
    so a MERGED group's own outer extent (needed by the snap step, section
    3, so it never searches past a wall either) reflects whichever bound
    constrained its outermost flags."""
    n = len(peaks_ms)
    windows: List[Tuple[float, float]] = []
    bounds: List[Tuple[float, float]] = []
    for i, (a, b) in enumerate(raw_windows):
        lo, hi = extent
        if i > 0:
            wall = walls[i - 1]
            lo = max(lo, wall + 1 if wall is not None else (peaks_ms[i - 1] + peaks_ms[i]) / 2.0)
        if i < n - 1:
            wall = walls[i]
            hi = min(hi, wall if wall is not None else (peaks_ms[i] + peaks_ms[i + 1]) / 2.0)
        windows.append((max(a, lo), min(b, hi)))
        bounds.append((lo, hi))
    return windows, bounds


# --------------------------------------------------------------------------
# Step 4 -- merge overlapping/touching windows (algorithm unchanged; each
# group additionally carries its own wall-aware extent now)
# --------------------------------------------------------------------------

def _merge_windows(
    peaks_ms: List[int], tags: List[str], summaries: List[str], specifics_list: List[Dict[str, Any]],
    subject_boxes: List[Optional[Tuple[float, float, float, float]]],
    windows: List[Tuple[float, float]], bounds: List[Tuple[float, float]],
) -> List[Dict[str, Any]]:
    """Merge overlapping/touching clamped windows: peaks_ms must already be
    sorted ascending, and windows/bounds aligned/ordered the same way
    (guaranteed by _clamp_windows, which preserves input order)."""
    groups: List[Dict[str, Any]] = [{
        "peaks": [(peaks_ms[0], tags[0], summaries[0], specifics_list[0], subject_boxes[0])],
        "a": windows[0][0], "b": windows[0][1],
        "extent_lo": bounds[0][0], "extent_hi": bounds[0][1],
    }]
    for i in range(1, len(peaks_ms)):
        a, b = windows[i]
        cur = groups[-1]
        if a <= cur["b"]:
            cur["b"] = max(cur["b"], b)
            cur["peaks"].append((peaks_ms[i], tags[i], summaries[i], specifics_list[i], subject_boxes[i]))
            cur["extent_hi"] = bounds[i][1]
        else:
            groups.append({
                "peaks": [(peaks_ms[i], tags[i], summaries[i], specifics_list[i], subject_boxes[i])],
                "a": a, "b": b,
                "extent_lo": bounds[i][0], "extent_hi": bounds[i][1],
            })
    return groups


_GroupPeak = Tuple[int, str, str, Dict[str, Any], Optional[Tuple[float, float, float, float]]]


def _representative_peak(
    group_peaks: List[_GroupPeak], hop_ms: int, action_energy: List[float],
) -> Tuple[int, str]:
    """A merged window may carry several flags (e.g. 8 push-up tops at low
    energy) -- pick the strongest one (by refined action_energy) as the
    single representative peak_ms/tag a CutRecord needs (hero_ts_ms)."""
    if len(group_peaks) == 1:
        t_ms, tag, _summary, _specifics, _box = group_peaks[0]
        return t_ms, tag

    def _strength(t_ms: int) -> float:
        if not action_energy or hop_ms <= 0:
            return 0.0
        i = max(0, min(len(action_energy) - 1, int(round(t_ms / hop_ms))))
        return action_energy[i]

    t_ms, tag, _summary, _specifics, _box = max(group_peaks, key=lambda pt: _strength(pt[0]))
    return t_ms, tag


def _joined_summary(group_peaks: List[_GroupPeak]) -> str:
    """A merged cut's summary (section 4.4): the joined summaries of every
    flag it absorbed, order-preserving and de-duplicated (a repeated action
    -- "does a push-up" x8 -- collapses to one mention, not eight) -- more
    informative for a loose, multi-moment cut than a single representative
    flag's summary alone."""
    ordered = sorted(group_peaks, key=lambda pt: pt[0])
    seen: List[str] = []
    for _t_ms, _tag, summary, _specifics, _box in ordered:
        if summary and summary not in seen:
            seen.append(summary)
    return "; ".join(seen)


def _composed_specifics(group_peaks: List[_GroupPeak], representative_t_ms: int) -> Dict[str, Any]:
    """vcut_pass2_video_specifics.plan.md section 7.2: a resolved cut's
    scene_specifics, composed from the flags it absorbed. A single-flag cut
    passes that flag's own specifics through unchanged. A merged multi-flag
    cut (loose energy) carries the REPRESENTATIVE flag's fields at the top
    level (so a single-question UI still shows something sensible) plus a
    "moments" mini shot-list -- one entry per absorbed flag, each with its
    own t_ms/summary/specifics -- so nothing a non-representative flag
    answered is lost. Always JSON-serializable (plain dicts/lists only)."""
    if len(group_peaks) == 1:
        _t_ms, _tag, _summary, specifics, _box = group_peaks[0]
        return dict(specifics or {})

    rep = next((p for p in group_peaks if p[0] == representative_t_ms), group_peaks[0])
    composed = dict(rep[3] or {})
    composed["moments"] = [
        {"t_ms": t_ms, "summary": summary, **dict(specifics or {})}
        for t_ms, _tag, summary, specifics, _box in sorted(group_peaks, key=lambda p: p[0])
    ]
    return composed


def _representative_subject_box(
    group_peaks: List[_GroupPeak], representative_t_ms: int,
) -> Optional[Tuple[float, float, float, float]]:
    """reframe_vcut_geometry.plan.md section 3: a merged cut carries the
    REPRESENTATIVE flag's own subject_box, unlike specifics -- a box
    doesn't compose sensibly across multiple moments (there's no sensible
    "joined" rectangle), so no moments-list here. None when that flag had
    none. Energy-invariant the same way _composed_specifics is: the SAME
    representative_t_ms _representative_peak already picked."""
    rep = next((p for p in group_peaks if p[0] == representative_t_ms), group_peaks[0])
    return rep[4]


# --------------------------------------------------------------------------
# Step 5 -- seam-snap the edges (unchanged code; now called with each
# merged group's own wall-aware extent instead of a VLM loose-cut span)
# --------------------------------------------------------------------------

def _argmax_s_ms(
    center_ms: float, radius_ms: int, hop_ms: int, S: List[float], lo_bound: float, hi_bound: float,
) -> int:
    """argmax S(t) in [center-radius, center+radius], intersected with
    [lo_bound, hi_bound] (the "never cross the peak" constraint). Falls
    back to ``center_ms`` clamped into bounds when S carries no signal at
    all (e.g. this file's seam computation failed) -- an unsnapped edge is
    strictly better than crashing on a debug/best-effort curve."""
    fallback = int(min(max(center_ms, lo_bound), hi_bound))
    if not S or hop_ms <= 0 or lo_bound > hi_bound:
        return fallback
    lo_ms = max(lo_bound, center_ms - radius_ms)
    hi_ms = min(hi_bound, center_ms + radius_ms)
    if lo_ms > hi_ms:
        return fallback
    lo_i = max(0, int(lo_ms // hop_ms))
    hi_i = min(len(S) - 1, int(hi_ms // hop_ms))
    if lo_i > hi_i:
        return fallback
    best_i = max(range(lo_i, hi_i + 1), key=lambda i: S[i])
    return best_i * hop_ms


def _snap_group_edges(
    a: float, b: float, group_peaks: List[_GroupPeak], hop_ms: int, S: List[float],
    extent: Tuple[float, float],
) -> Tuple[int, int]:
    earliest_peak = min(p[0] for p in group_peaks)
    latest_peak = max(p[0] for p in group_peaks)
    extent_lo, extent_hi = extent

    a_star = _argmax_s_ms(a, SNAP_MS, hop_ms, S, lo_bound=extent_lo, hi_bound=earliest_peak)
    b_star = _argmax_s_ms(b, SNAP_MS, hop_ms, S, lo_bound=latest_peak, hi_bound=extent_hi)

    if a_star >= b_star:
        # Degenerate (very tight span/window): fall back to the unsnapped,
        # already-clamped provisional edges rather than emit an inverted cut.
        a_star, b_star = int(a), int(b)
    # Safety clamp: the peak(s) must always lie inside the emitted window.
    a_star = min(a_star, earliest_peak)
    b_star = max(b_star, latest_peak)
    return a_star, b_star


# --------------------------------------------------------------------------
# Step 6 -- junk / dead-air floor, min-cut, min-gap (unchanged)
# --------------------------------------------------------------------------

def _median(values: List[float]) -> float:
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return 0.0
    n = len(xs)
    mid = n // 2
    if n % 2:
        return xs[mid]
    return (xs[mid - 1] + xs[mid]) / 2.0


def _median_s_in_range(a_ms: float, b_ms: float, hop_ms: int, S: List[float]) -> float:
    if not S or hop_ms <= 0:
        return 0.0
    lo_i = max(0, int(a_ms // hop_ms))
    hi_i = min(len(S) - 1, int(b_ms // hop_ms))
    if lo_i > hi_i:
        return 0.0
    return _median(S[lo_i:hi_i + 1])


def _drop_dead_air(cands: List[_Candidate], hop_ms: int, S: List[float]) -> List[_Candidate]:
    """Drop any candidate whose median S over its extent is below
    S_DEAD_FRAC * (clip median S) AND which carries no peak. Every
    candidate this module builds carries >=1 flag by construction (nothing
    is ever dropped/selected, section 1's own invariant) -- this floor is a
    defensive invariant (see test_vcut_resolve.py, which exercises it
    directly against a synthetic peak-less candidate) rather than a path
    real Pass-1 output can trigger."""
    if not S:
        return cands
    clip_median = _median(S)
    if clip_median <= 0:
        return cands
    floor = S_DEAD_FRAC * clip_median
    return [
        c for c in cands
        if c.has_peak or _median_s_in_range(c.a, c.b, hop_ms, S) >= floor
    ]


def _widen_to_min_cut(cands: List[_Candidate]) -> List[_Candidate]:
    out = []
    for c in cands:
        if c.b - c.a >= MIN_CUT_MS:
            out.append(c)
            continue
        mid = (c.a + c.b) / 2.0
        a = max(c.span_lo, mid - MIN_CUT_MS / 2.0)
        b = min(c.span_hi, mid + MIN_CUT_MS / 2.0)
        out.append(_Candidate(c.file_id, a, b, c.peak_ms, c.tag, c.summary,
                              c.span_lo, c.span_hi, c.has_peak, c.specifics, c.subject_box))
    return out


def _enforce_min_gap(cuts: List[ResolvedCut]) -> List[ResolvedCut]:
    """Widen any gap narrower than MIN_CUT_GAP_MS by pulling the two edges
    apart symmetrically, never past either cut's own peak_ms. A usability
    guarantee, not a hard invariant: if one side is peak-blocked the other
    side is not asked to make up the difference (see params.py's docstring
    -- this floor sits alongside the dead-air/min-cut floors, all "best
    effort," not mathematically enforced contracts)."""
    cuts = sorted(cuts, key=lambda c: c.in_ms)
    for i in range(1, len(cuts)):
        prev, cur = cuts[i - 1], cuts[i]
        if prev.file_id != cur.file_id:
            continue
        gap = cur.in_ms - prev.out_ms
        if gap >= MIN_CUT_GAP_MS:
            continue
        shrink = (MIN_CUT_GAP_MS - gap) / 2.0
        prev.out_ms = int(max(prev.peak_ms, prev.out_ms - shrink))
        cur.in_ms = int(min(cur.peak_ms, cur.in_ms + shrink))
    return cuts


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def _resolve_file(
    file_id: str, flags: List[MomentFlag], energy: float,
    hop_ms: int, S: List[float], action_energy: List[float], frame_diff: List[float],
) -> List[_Candidate]:
    if not flags:
        return []
    extent = (0.0, float(len(S) * hop_ms))

    refined = sorted(
        ((_refine_peak_ms(f.t_ms, hop_ms, action_energy, frame_diff), f.shape or DEFAULT_TAG, f.summary,
          f.specifics, f.subject_box)
         for f in flags),
        key=lambda x: x[0],
    )
    peaks_ms = [r[0] for r in refined]
    tags = [r[1] for r in refined]
    summaries = [r[2] for r in refined]
    specifics_list = [r[3] for r in refined]
    subject_boxes = [r[4] for r in refined]

    width_ms = _window_width_ms(energy)
    raw_windows = [_tag_window(pm, tag, width_ms) for pm, tag in zip(peaks_ms, tags)]
    walls = _wall_positions(peaks_ms, hop_ms, S)
    clamped, bounds = _clamp_windows(peaks_ms, raw_windows, extent, walls)
    groups = _merge_windows(peaks_ms, tags, summaries, specifics_list, subject_boxes, clamped, bounds)

    out: List[_Candidate] = []
    for g in groups:
        peak_ms, tag = _representative_peak(g["peaks"], hop_ms, action_energy)
        summary = _joined_summary(g["peaks"])
        specifics = _composed_specifics(g["peaks"], peak_ms)
        subject_box = _representative_subject_box(g["peaks"], peak_ms)
        group_extent = (g["extent_lo"], g["extent_hi"])
        a_star, b_star = _snap_group_edges(g["a"], g["b"], g["peaks"], hop_ms, S, group_extent)
        out.append(_Candidate(
            file_id=file_id, a=a_star, b=b_star, peak_ms=peak_ms, tag=tag, summary=summary,
            span_lo=g["extent_lo"], span_hi=g["extent_hi"], specifics=specifics, subject_box=subject_box,
        ))
    return out


def resolve_cuts(plan: MomentPlan, seam: Dict[str, dict], energy: float) -> List[ResolvedCut]:
    """THE ALGORITHM (section 4): a file-wide pool of moment flags + per-file
    S(t) + one global energy -> final cut boundaries. Pure -- no I/O, no
    model call, no footage-type branch. A file missing from ``seam`` (its
    seam computation failed, or it simply has no non-speech content)
    contributes no cuts rather than emitting unsnapped guesses."""
    all_candidates: List[_Candidate] = []
    for file_plan in plan.files:
        file_seam = seam.get(file_plan.file_id) or {}
        hop_ms = int(file_seam.get("hop_ms") or 0)
        S = list(file_seam.get("S") or [])
        action_energy = list(file_seam.get("action_energy") or [])
        frame_diff = list(file_seam.get("frame_diff") or [])
        if not S or hop_ms <= 0:
            continue
        all_candidates.extend(_resolve_file(
            file_plan.file_id, file_plan.flags, energy, hop_ms, S, action_energy, frame_diff,
        ))

    by_file: Dict[str, List[_Candidate]] = {}
    for c in all_candidates:
        by_file.setdefault(c.file_id, []).append(c)

    resolved: List[ResolvedCut] = []
    for file_id, cands in by_file.items():
        file_seam = seam.get(file_id) or {}
        hop_ms = int(file_seam.get("hop_ms") or 0)
        S = list(file_seam.get("S") or [])
        cands = _drop_dead_air(cands, hop_ms, S)
        cands = _widen_to_min_cut(cands)
        resolved.extend(
            ResolvedCut(file_id=c.file_id, in_ms=int(round(c.a)), out_ms=int(round(c.b)),
                       peak_ms=c.peak_ms, tag=c.tag, summary=c.summary, specifics=c.specifics,
                       subject_box=c.subject_box)
            for c in cands
        )

    return _enforce_min_gap(resolved)
