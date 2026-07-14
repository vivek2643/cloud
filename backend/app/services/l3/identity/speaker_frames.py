"""
Deterministic close-burst frame planning for voice->face binding
(voice_first_identity.plan.md Phase E, section 4 -- the plan's own
"centerpiece"). No model call: for each GLOBAL voice, find the handful of
instants that most reliably reveal WHOSE MOUTH MOVES while that voice
speaks. The small Gemini pass downstream (identity/speaker_pass.py) only
ever judges the pixels this module hands it -- "model perceives, code
decides" applies here too: the WINDOW/TIMESTAMP selection is entirely code.

Pipeline (mirrors the plan's five steps):
  1. `voice_turns` -- V's turns across every file, via the global voice map.
  2. `clean_windows` -- the maximal sub-span of each turn where no OTHER
     voice's turn comes within GUARD_MS (removes overlapping-speech
     contamination).
  3. `_covering_candidates` -- the Pass-2 cut(s) covering a window and their
     resolved visible person(s) (identity/reconcile.py Phase D output) --
     one candidate (a strong recurrent-subject prior) or several (a shared
     camera / outlook group).
  4. `_window_score` -- loudness peak height + sharpness + isolation margin
     + face prominence, then a diversity-aware greedy top-K pick.
  5. `_burst_ts` -- N timestamps centered on the loudness-peak instant,
     each snapped to the nearest sharp frame.

A voice with no clean window carrying ANY visible-person candidate at all
is flagged off-camera here (step 6's first half); the second half --
"across chosen windows no mouth moves" -- can only be judged from the
actual pixels, so identity/apply.py folds that in after speaker_pass runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.lattice import resolve_speech_span_ms
from app.services.l3.video_segments import _sharpest_ms

# Chosen defaults (voice_first_identity.plan.md section 7) -- tunable,
# clip-relative where possible.
GUARD_MS = 150
MIN_WIN_MS = 400
K = 4
N = 3
D_MS = 90


@dataclass
class Burst:
    """One candidate-person's micro-burst within one clean window of one
    voice's speech -- the unit speaker_pass.py judges. A multi-candidate
    window produces one `Burst` per candidate, sharing the same `ts_ms`
    (the pass frames EACH candidate at the same instant so it can compare
    whose mouth actually moves)."""
    voice: str
    window_id: str
    candidate_person: Optional[str]   # Phase D's recurrent-subject hint, or None
    file_id: str
    ts_ms: List[int] = field(default_factory=list)


def voice_turns(
    turns_by_file: Dict[str, List[Tuple[int, int, str]]],
    voice_of: Dict[Tuple[str, str], str],
) -> Dict[str, List[Tuple[str, int, int]]]:
    """voice -> [(file_id, start_ms, end_ms), ...] -- every turn across
    every file whose local speaker maps to that global voice (identity/
    voices.assign_voices). A (file_id, local_speaker) with no voice mapping
    contributes nothing -- defensive; the full-roster `assign_voices` call
    should never actually leave one uncovered."""
    out: Dict[str, List[Tuple[str, int, int]]] = {}
    for fid, turns in turns_by_file.items():
        for start_ms, end_ms, local in turns:
            if not local:
                continue
            voice = voice_of.get((fid, local))
            if voice is None:
                continue
            out.setdefault(voice, []).append((fid, int(start_ms), int(end_ms)))
    return out


def _other_voice_turns_by_file(
    turns_by_file: Dict[str, List[Tuple[int, int, str]]],
    voice_of: Dict[Tuple[str, str], str],
    exclude_voice: str,
) -> Dict[str, List[Tuple[int, int]]]:
    out: Dict[str, List[Tuple[int, int]]] = {}
    for fid, turns in turns_by_file.items():
        for s, e, local in turns:
            v = voice_of.get((fid, local))
            if v is not None and v != exclude_voice:
                out.setdefault(fid, []).append((int(s), int(e)))
    return out


def _merge_intervals(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for s, e in ordered[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _largest_clean_subspan(s: int, e: int, forbidden: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """The largest sub-interval of [s, e) that avoids every forbidden
    interval -- None when [s, e) is fully covered by forbidden zones."""
    merged = _merge_intervals(forbidden)
    pieces: List[Tuple[int, int]] = []
    cur = s
    for fs, fe in merged:
        if fe <= cur or fs >= e:
            continue
        if fs > cur:
            pieces.append((cur, min(fs, e)))
        cur = max(cur, fe)
        if cur >= e:
            break
    if cur < e:
        pieces.append((cur, e))
    if not pieces:
        return None
    return max(pieces, key=lambda p: p[1] - p[0])


def clean_windows(
    turns: List[Tuple[str, int, int]],
    other_turns_by_file: Dict[str, List[Tuple[int, int]]],
    guard_ms: int = GUARD_MS, min_win_ms: int = MIN_WIN_MS,
) -> List[Tuple[str, int, int]]:
    """For each of V's turns, the maximal sub-span with no OTHER voice's
    turn within `guard_ms` on either side, kept only when >= `min_win_ms`
    (overlapping/nearby speech contaminates "whose mouth moves"). One
    window per turn (the plan's "carve the maximal sub-span", singular) --
    [(file_id, start_ms, end_ms), ...]."""
    out: List[Tuple[str, int, int]] = []
    for fid, s, e in turns:
        forbidden = [(os_ - guard_ms, oe + guard_ms) for os_, oe in other_turns_by_file.get(fid, [])]
        span = _largest_clean_subspan(s, e, forbidden)
        if span is not None and span[1] - span[0] >= min_win_ms:
            out.append((fid, span[0], span[1]))
    return out


def _cut_span_ms(cut: Any, lattice: Any) -> Optional[Tuple[int, int]]:
    """Best-effort (s, e) ms for one Pass2Cut-shaped object (duck-typed --
    file_id/kind/word_span/atom_ids -- this module stays decoupled from the
    pass2 schema, same pattern the rest of identity/ already uses)."""
    if lattice is None:
        return None
    if cut.kind == "speech" and cut.word_span:
        return resolve_speech_span_ms(lattice.words, lattice.atoms, tuple(cut.word_span), [])
    if cut.kind == "video" and cut.atom_ids:
        atoms_by_id = {a.atom_id: a for a in lattice.atoms}
        members = [atoms_by_id[i] for i in cut.atom_ids if i in atoms_by_id]
        if not members:
            return None
        return min(a.start_ms for a in members), max(a.end_ms for a in members)
    return None


def _covering_candidates(
    fid: str, s: int, e: int, cuts_by_file: Dict[str, List[Any]], lattice: Any,
    visible_persons: Dict[Tuple[str, str], List[str]],
) -> List[Tuple[str, str]]:
    """[(person_id, source_ref), ...] for every DISTINCT visible person
    across every cut in `fid` whose resolved span overlaps [s, e) --
    "single candidate" (one entry) or "multi candidate" (several, e.g. a
    shared camera or outlook group). Empty when no covering cut resolves
    any visible person at all (b-roll, an unclustered/crowd cut)."""
    persons: Dict[str, str] = {}   # person_id -> the covering cut's source_ref
    for cut in cuts_by_file.get(fid, []):
        span = _cut_span_ms(cut, lattice)
        if span is None or span[1] <= s or span[0] >= e:
            continue
        for pid in visible_persons.get((fid, cut.source_ref), []):
            persons.setdefault(pid, cut.source_ref)
    return sorted(persons.items())


def _series_lohi(arr: List[float]) -> Tuple[Optional[float], Optional[float]]:
    vals = [float(v) for v in (arr or []) if v is not None]
    return (min(vals), max(vals)) if vals else (None, None)


def _norm(value: Optional[float], lo: Optional[float], hi: Optional[float]) -> float:
    if value is None or lo is None or hi is None or hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _value_at(arr: List[float], hop_ms: int, ts_ms: int) -> Optional[float]:
    if not arr or hop_ms <= 0:
        return None
    i = ts_ms // hop_ms
    return arr[i] if 0 <= i < len(arr) else None


def _loudness_peak_ms(rms_db: List[float], hop_ms: int, s: int, e: int) -> int:
    """argmax rms_db over [s, e) -- the loudest voiced instant (t* in the
    plan): louder = mouth most open. Midpoint fallback with no usable
    audio signal."""
    mid = (s + e) // 2
    if not rms_db or hop_ms <= 0:
        return mid
    lo, hi = max(0, s // hop_ms), min(len(rms_db) - 1, (e - 1) // hop_ms)
    if hi < lo:
        return mid
    return max(range(lo, hi + 1), key=lambda i: rms_db[i]) * hop_ms


def _window_score(
    t_star: int, isolation_margin_ms: int,
    rms_db: List[float], rms_hop_ms: int, rms_lo: Optional[float], rms_hi: Optional[float],
    blur: List[float], blur_hop_ms: int, blur_lo: Optional[float], blur_hi: Optional[float],
    subject_box_area: Optional[float], has_position: bool,
) -> float:
    """loudness peak height + sharpness + isolation margin + face
    prominence -- the plan's four reliability signals, clip-relative where
    a normalizer is available. Deterministic, no absolute constants beyond
    the guard/window defaults already governing which windows exist."""
    loud = _norm(_value_at(rms_db, rms_hop_ms, t_star), rms_lo, rms_hi)
    blur_v = _value_at(blur, blur_hop_ms, t_star)
    sharp = 1.0 - _norm(blur_v, blur_lo, blur_hi) if blur_v is not None else 0.0
    isolation = max(0.0, min(1.0, isolation_margin_ms / (2.0 * GUARD_MS)))
    prominence = (subject_box_area or 0.0) + (0.2 if has_position else 0.0)
    return loud + sharp + isolation + prominence


def _burst_offsets(win_len_ms: int, d_ms: int = D_MS, n: int = N) -> List[int]:
    """N=3 offsets [-d, 0, +d] when the window comfortably fits them
    (articulation is ~3-7 Hz, so frames need ~d apart to show mouth CHANGE
    rather than collapsing to duplicate stills); drops to 2, then 1, for a
    short window rather than reaching outside it."""
    if win_len_ms >= 2 * d_ms:
        return [-d_ms, 0, d_ms][: n if n < 3 else 3]
    if win_len_ms >= d_ms:
        half = d_ms // 2
        return [-half, half]
    return [0]


def _burst_ts(t_star: int, s: int, e: int, blur: List[float], hop_ms: int) -> List[int]:
    """N timestamps centered on `t_star`, each snapped to the nearest sharp
    instant within +/-1 hop (`_sharpest_ms`), clamped inside [s, e)."""
    out: List[int] = []
    for off in _burst_offsets(e - s):
        raw = max(s, min(e - 1, t_star + off))
        if blur and hop_ms > 0:
            seg_s, seg_e = max(s, raw - hop_ms), min(e, raw + hop_ms + 1)
            out.append(_sharpest_ms(blur, hop_ms, seg_s, seg_e, raw))
        else:
            out.append(raw)
    return out


def plan_bursts(
    turns_by_file: Dict[str, List[Tuple[int, int, str]]],
    voice_of: Dict[Tuple[str, str], str],
    cuts_by_file: Dict[str, List[Any]],
    lattices: Dict[str, Any],
    visible_persons: Dict[Tuple[str, str], List[str]],
    audio_by_file: Dict[str, dict],
    motion_by_file: Dict[str, dict],
    *, k: int = K,
) -> Tuple[List[Burst], set]:
    """The whole Phase E pipeline. Returns `(bursts, off_camera_voices)`:
    `bursts` is bounded by `K * candidates * n_voices` per the plan (a
    handful of voices -> a tiny plan); `off_camera_voices` are voices with
    NO clean window carrying any visible-person candidate at all -- the
    expected narration/podcast-listener case, never forced to a face."""
    all_bursts: List[Burst] = []
    off_camera: set = set()

    for voice, turns in sorted(voice_turns(turns_by_file, voice_of).items()):
        other_turns = _other_voice_turns_by_file(turns_by_file, voice_of, voice)
        windows = clean_windows(turns, other_turns)

        scored: List[Tuple[float, str, int, int, List[Tuple[str, str]]]] = []
        for fid, s, e in windows:
            candidates = _covering_candidates(fid, s, e, cuts_by_file, lattices.get(fid), visible_persons)
            if not candidates:
                continue
            audio = audio_by_file.get(fid) or {}
            motion = motion_by_file.get(fid) or {}
            rms_db, rms_hop = audio.get("rms_db") or [], int(audio.get("hop_ms") or 0)
            blur, blur_hop = motion.get("blur") or [], int(motion.get("hop_ms") or 0)
            rms_lo, rms_hi = _series_lohi(rms_db)
            blur_lo, blur_hi = _series_lohi(blur)
            t_star = _loudness_peak_ms(rms_db, rms_hop, s, e)
            # Isolation margin: how far this window's actual span sits from
            # the guard-band edges its own turn started with (>= 0; capped
            # by the guard band itself, since that's what bounded the carve).
            isolation_ms = min(e - s, 2 * GUARD_MS)

            cut_by_ref = {c.source_ref: c for c in cuts_by_file.get(fid, [])}
            for pid, ref in candidates:
                cut = cut_by_ref.get(ref)
                area = None
                has_pos = False
                if cut is not None:
                    box = (getattr(cut.framing, "subject_box", None)
                          if hasattr(cut, "framing") else None)
                    if box:
                        area = float(box[2]) * float(box[3])
                    for p in (cut.people or []):
                        if p.get("position"):
                            has_pos = True
                            break
                score = _window_score(t_star, isolation_ms, rms_db, rms_hop, rms_lo, rms_hi,
                                      blur, blur_hop, blur_lo, blur_hi, area, has_pos)
                scored.append((score, fid, s, e, [(pid, ref)]))

        if not scored:
            off_camera.add(voice)
            continue

        # Diversity-aware greedy top-K: prefer covering a new (file, person)
        # combo before repeating one, in score order; fill any remaining
        # slots from what's left once every combo has one window.
        scored.sort(key=lambda row: row[0], reverse=True)
        picked: List[Tuple[float, str, int, int, List[Tuple[str, str]]]] = []
        seen_combo: set = set()
        for row in scored:
            if len(picked) >= k:
                break
            _, fid, s, e, cands = row
            combo = (fid, cands[0][0])
            if combo in seen_combo:
                continue
            seen_combo.add(combo)
            picked.append(row)
        if len(picked) < k:
            for row in scored:
                if len(picked) >= k:
                    break
                if row not in picked:
                    picked.append(row)

        for i, (_score, fid, s, e, _cands) in enumerate(picked):
            window_id = f"{voice}:w{i}"
            motion = motion_by_file.get(fid) or {}
            blur, blur_hop = motion.get("blur") or [], int(motion.get("hop_ms") or 0)
            audio = audio_by_file.get(fid) or {}
            t_star = _loudness_peak_ms(audio.get("rms_db") or [], int(audio.get("hop_ms") or 0), s, e)
            ts = _burst_ts(t_star, s, e, blur, blur_hop)
            candidates = _covering_candidates(fid, s, e, cuts_by_file, lattices.get(fid), visible_persons)
            for pid, _ref in candidates:
                all_bursts.append(Burst(voice=voice, window_id=window_id, candidate_person=pid,
                                        file_id=fid, ts_ms=list(ts)))

    return all_bursts, off_camera
