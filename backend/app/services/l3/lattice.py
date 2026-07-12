"""
Cuts v3 -- the LATTICE: the deterministic substrate the LLM ingest reasons
over. Evolved from the cuts-v2 ``base_cuts.py`` (see cuts_v3.plan.md, section
2), which has since been retired (cleanup.plan.md B3).

Two sides, two different shapes, because they need different things from the
model:

  * SPEECH has no atoms. The lattice IS the word list -- Whisper word timings
    are already millisecond-precise, so there is nothing to detect. Turns,
    long pauses and speaker changes are exported as PROMPT HINTS (informational
    text), never boundaries: pass 1 picks the actual ``word_span`` grouping
    itself. This is a deliberate reversal from ``base_cuts.py``, which turned
    speech turns into fixed cuts -- v3 wants the LLM's judgment there, not a
    deterministic turn-merge.
  * VIDEO gets ATOMS: carved at boundaries we trust as genuine scene changes
    -- hard shot cut, ``transition_points`` (wipe/degenerate) -- PLUS the
    edges where the clip crosses between its OWN quiet and active energy
    regimes (an Otsu split on this clip's action_energy; see ``_regime_marks``),
    over the NON-SPEECH remainder of the clip. Camera move/settle are
    DELIBERATELY NOT boundaries: motion enters the model as RAW numbers
    (``mot``/``coh``) on each atom, and the LLM categorizes camera behaviour
    itself -- there is no code-side "pan"/"handheld" label any more, and
    nothing per-footage is tuned. A video atom overlapping speech never
    happens by construction: atoms are built inside the gaps between speech
    turns, not against the whole timeline (unchanged rule -- never cut under
    speech).

Atoms are deliberately fine (over-segmenting bias is safe -- pass 1/2 merge
them back into ``video_tentative_groups`` / final cuts; under-splitting, which
would hide a real boundary forever, is the only fatal error).

Pure core: ``build_atoms(...)`` / ``speech_hints(...)`` / the pure
``_snap_word_edge`` take already-loaded artifacts and do no DB call, so they
are trivially testable -- see ``scripts/test_lattice.py``. ``load_lattice``
and ``snap_word_edge`` are the DB-loading convenience wrappers for real callers.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.diarize import Turn
from app.services.l3.lattice_params import LONG_PAUSE_MS, MIN_ATOM_MS, SNAP_MS

logger = logging.getLogger(__name__)

# Boundary reason tags this module owns outright now (extracted from the
# retired base_cuts.py -- cuts_v3.plan.md section 2 / cleanup.plan.md B3).
R_CLIP = "clip_edge"
R_SHOT = "shot_cut"
R_DISTURB = "disturbance"

# cuts-v3 transition-point reasons (base_cuts.py predates transition_points).
# Ranked alongside the reasons above for the same "several signals, one
# instant -> report the strongest" collapsing.
R_WIPE = "wipe"
R_DEGENERATE = "degenerate"
R_ACTION = "action"             # this edge is a quiet<->active energy-regime crossing (Otsu)
R_SPEECH_EDGE = "speech_edge"   # this atom's own edge borders a speech turn

# Only GROUNDED events are atom boundaries now (boundaries-v2): hard shot cut,
# wipe/degenerate transition, ACTION window edge, and the speech/clip edges
# bounding the remainder. Camera move/settle and disturbance jitter are
# LABELS, not edges -- R_DISTURB is retained in the rank for completeness
# only; this module never emits a disturbance mark itself.
_REASON_RANK = {
    R_SHOT: 0, R_DISTURB: 1, R_WIPE: 1, R_DEGENERATE: 1,
    R_ACTION: 2, R_SPEECH_EDGE: 3, R_CLIP: 4,
}


# --------------------------------------------------------------------------
# The Atom shape
# --------------------------------------------------------------------------

@dataclass
class Atom:
    atom_id: int
    file_id: str
    start_ms: int
    end_ms: int
    state_in: str          # why this atom's LEFT edge exists
    state_out: str         # why this atom's RIGHT edge exists
    action_energy: float   # mean subject-motion energy over the span, 0..1
    coherence: float       # mean camera_coherence over the span, 0..1
    anchor_ms: List[int] = field(default_factory=list)   # action_points inside this atom
    is_action: bool = False   # LABEL: this atom sits in the clip's active energy regime
    peak_action_energy: float = 0.0  # PEAK subject-motion energy over the span, 0..1
    camera_motion: float = 0.0       # mean camera-motion magnitude over the span, 0..1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atom_id": self.atom_id, "file_id": self.file_id,
            "start_ms": self.start_ms, "end_ms": self.end_ms,
            "state_in": self.state_in, "state_out": self.state_out,
            "action_energy": self.action_energy,
            "coherence": self.coherence, "anchor_ms": list(self.anchor_ms),
            "is_action": self.is_action,
            "peak_action_energy": self.peak_action_energy,
            "camera_motion": self.camera_motion,
        }


@dataclass
class Lattice:
    """Everything one clip's ingest reasons over: the raw word list + prompt
    hints (speech side) and the video atoms (non-speech side)."""
    file_id: str
    duration_ms: int
    words: List[dict] = field(default_factory=list)
    turns: List[Turn] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)
    atoms: List[Atom] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id, "duration_ms": self.duration_ms,
            "words": self.words, "turns": [list(t) for t in self.turns],
            "hints": self.hints, "atoms": [a.to_dict() for a in self.atoms],
        }


# --------------------------------------------------------------------------
# Video side: atoms over the non-speech remainder
# --------------------------------------------------------------------------

def _shot_marks(scene: Optional[dict]) -> List[Tuple[int, str]]:
    """Hard SHOT CUT marks from scene detection (extracted from the retired
    base_cuts.py -- cleanup.plan.md B3)."""
    return [(int(p["ts_ms"]), R_SHOT)
            for p in ((scene or {}).get("shot_points") or [])
            if isinstance(p, dict) and "ts_ms" in p]


def _transition_marks(motion: Optional[dict]) -> List[Tuple[int, str]]:
    """``transition_points`` (Phase B) as atom-edge marks -- a wipe or a
    degenerate span onset is exactly as trusted a boundary as a shot cut or a
    camera settle."""
    out: List[Tuple[int, str]] = []
    for p in ((motion or {}).get("transition_points") or []):
        kind = p.get("kind")
        ts = p.get("ts_ms")
        if ts is None or kind not in ("wipe", "degenerate"):
            continue
        out.append((int(ts), R_WIPE if kind == "wipe" else R_DEGENERATE))
    return out


def _otsu(values: List[float]) -> Optional[float]:
    """Otsu's method: the energy level that best splits THIS clip's own
    action-energy distribution into a quiet class and an active class, by
    maximizing between-class variance. Returns None when there is no
    meaningful split (near-constant energy -> the clip is one regime).

    This is the load-bearing "no magic numbers" primitive: where the motion
    genuinely rises and falls is read off the clip's OWN histogram, never a
    hand-set floor like 0.5 or 0.6. A calm talking-head clip yields None
    (one regime, coarse atoms); a rally/swing clip yields a clean split so
    the burst becomes its own active atom."""
    xs = [float(v) for v in values if v is not None]
    if len(xs) < 2:
        return None
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-6:
        return None
    bins = 64
    hist = [0] * bins
    for v in xs:
        b = int((v - lo) / (hi - lo) * (bins - 1))
        hist[b] += 1
    total = len(xs)
    sum_all = sum((i + 0.5) * hist[i] for i in range(bins))
    w_b = 0.0
    sum_b = 0.0
    best_var = -1.0
    best_bin = 0
    for i in range(bins):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += (i + 0.5) * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_all - sum_b) / w_f
        var = w_b * w_f * (m_b - m_f) ** 2
        if var > best_var:
            best_var, best_bin = var, i
    return lo + (best_bin + 1) / bins * (hi - lo)


def _regime_marks(motion: Optional[dict]) -> List[Tuple[int, str]]:
    """Atom edges where the clip crosses between its OWN quiet and active
    energy regimes (the Otsu split on this clip's action_energy). This
    REPLACES the old action-anchor carve (fixed pads + a 0.5-of-peak energy
    floor): instead of hand-tuned windows around detected impacts, the clip
    itself tells us where motion rises and falls, so a swing / pan / rally
    becomes one coherent active atom bordered by quiet ones -- with zero
    hardcoded energy numbers. Runs shorter than MIN_ATOM_MS are folded into
    their neighbor (a flicker isn't a regime); over-splitting is harmless
    anyway -- pass 1 merges coarser and total-coverage never drops a span."""
    m = motion or {}
    hop = int(m.get("hop_ms") or 0)
    energy = m.get("action_energy") or []
    if not hop or len(energy) < 2:
        return []
    thr = _otsu(energy)
    if thr is None:
        return []
    regime = [1 if (e or 0.0) >= thr else 0 for e in energy]
    min_hops = max(1, MIN_ATOM_MS // hop)

    runs: List[List[int]] = []  # [state, start_i, end_i)
    i = 0
    while i < len(regime):
        j = i
        while j < len(regime) and regime[j] == regime[i]:
            j += 1
        runs.append([regime[i], i, j])
        i = j
    merged: List[List[int]] = []
    for st, a, b in runs:
        if merged and (b - a) < min_hops:
            merged[-1][2] = b   # swallow a sub-MIN_ATOM_MS flicker into the prior regime
        else:
            merged.append([st, a, b])

    return [(a * hop, R_ACTION) for _st, a, _b in merged[1:]]


def _subtract(span: Tuple[int, int], claimed: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """``span`` minus the union of ``claimed`` intervals -> free fragments, in
    time order (the non-speech remainder of the clip)."""
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


def _snap_collect(marks: List[Tuple[int, str]], lo: int, hi: int) -> Dict[int, str]:
    """{ts: reason} for marks strictly inside (lo, hi), snapping any two
    within SNAP_MS onto the earlier one and keeping the strongest reason."""
    at: Dict[int, str] = {}
    for ts, reason in sorted((t, r) for t, r in marks if lo < t < hi):
        near = next((k for k in at if abs(k - ts) <= SNAP_MS), None)
        key = near if near is not None else ts
        if key not in at or _REASON_RANK.get(reason, 99) < _REASON_RANK.get(at[key], 99):
            at[key] = reason
    return at


def _mean(xs: List[float], lo_i: int, hi_i: int) -> float:
    seg = xs[max(0, lo_i):max(lo_i + 1, hi_i)]
    return sum(seg) / len(seg) if seg else 0.0


def _peak(xs: List[float], lo_i: int, hi_i: int) -> float:
    seg = xs[max(0, lo_i):max(lo_i + 1, hi_i)]
    return max(seg) if seg else 0.0


def _anchors_in(motion: Optional[dict], s_ms: int, e_ms: int) -> List[int]:
    pts = (motion or {}).get("action_points") or []
    return sorted(int(p["ts_ms"]) for p in pts if s_ms <= int(p.get("ts_ms", -1)) < e_ms)


def _word_claim_spans(words: List[dict], merge_gap_ms: int = LONG_PAUSE_MS) -> List[Tuple[int, int]]:
    """Time spans claimed by the words themselves, adjacent words merged
    when their gap is below ``merge_gap_ms`` (a sub-long-pause gap belongs
    to the speech around it, and per-word confetti spans would shred the
    remainder into useless atom slivers). Needed because diarization TURNS
    and Whisper word timings are two different oracles: observed against
    real data (a long stutter section), a run of transcript words fell
    outside every diarized turn -- atoms then got built OVER those words,
    and any speech cut using them overlapped atom territory by
    construction. Words are ground truth for where speech is; turns alone
    are not sufficient."""
    spans: List[Tuple[int, int]] = []
    for w in sorted(words, key=lambda w: int(w.get("start_ms", 0))):
        s, e = int(w.get("start_ms", 0)), int(w.get("end_ms", 0))
        if e <= s:
            continue
        if spans and s - spans[-1][1] < merge_gap_ms:
            spans[-1] = (spans[-1][0], max(spans[-1][1], e))
        else:
            spans.append((s, e))
    return spans


def build_atoms(file_id: str, duration_ms: int, motion: Optional[dict],
                scene: Optional[dict], turns: List[Turn],
                words: Optional[List[dict]] = None) -> List[Atom]:
    """Pure core: video ATOMS over the non-speech remainder of
    [0, duration_ms]. Speech gets no atoms at all -- atoms are only ever
    built inside the gaps between speech turns AND word-claimed spans (see
    ``_word_claim_spans`` for why turns alone aren't enough), so a video
    atom overlapping speech cannot happen by construction (not a filter, a
    structural guarantee)."""
    if duration_ms <= 0:
        return []
    claimed = [(s, e) for (s, e, _spk) in turns if e > s]
    claimed += _word_claim_spans(words or [])
    speech_spans = sorted(claimed)
    non_speech = _subtract((0, duration_ms), speech_spans)
    # Deterministic-keep (cuts_v3_deterministic_keep.plan.md): atom edges come
    # from GROUNDED events plus the clip's OWN energy regimes -- never from
    # hand-set pads. Hard shot cut + wipe/degenerate transition are grounded
    # scene changes; _regime_marks adds edges where THIS clip crosses between
    # its quiet and active energy (Otsu split), so a burst of motion becomes
    # its own atom with no magic energy floor. Camera move/settle and handheld
    # jitter remain LABELS (mot/coh on each atom), never edges.
    all_marks = _shot_marks(scene) + _transition_marks(motion) + _regime_marks(motion)

    m = motion or {}
    hop = int(m.get("hop_ms") or 0)
    action = m.get("action_energy") or []
    coherence = m.get("camera_coherence") or []
    cam_motion = m.get("camera_motion") or []
    # One Otsu split for the whole clip -> the boundary between its quiet and
    # active regimes. is_action is then a pure data-derived LABEL (this atom
    # sits in the active regime), not a keep/drop gate: nothing is dropped for
    # lacking it. None == the clip never leaves one regime (calm throughout).
    active_thr = _otsu(action)

    atoms: List[Atom] = []
    for seg_s, seg_e in non_speech:
        # NOT skipped when shorter than MIN_ATOM_MS: this whole segment is
        # everything between two speech turns (or the file's edge) with
        # nowhere else for its span to go -- dropping it would leave a real
        # coverage gap. MIN_ATOM_MS only governs merging INTERNAL boundary
        # marks below (over-splitting one long segment), never whether the
        # segment gets an atom at all. Confirmed via a real ingest run: a
        # 20ms trailing sliver was silently vanishing before this fix.
        at = _snap_collect(all_marks, seg_s, seg_e)
        bounds = sorted({seg_s, seg_e} | set(at))

        # Merge sub-MIN_ATOM_MS slivers forward so the segment's own coverage
        # stays total (over-split is safe; a too-short shard just isn't worth
        # its own atom -- its reason folds into the next real boundary).
        kept = [bounds[0]]
        for b in bounds[1:-1]:
            if b - kept[-1] >= MIN_ATOM_MS:
                kept.append(b)
        kept.append(bounds[-1])

        for a, b in zip(kept, kept[1:]):
            if b <= a:
                continue
            reason_in = at.get(a, R_CLIP if a == 0 else R_SPEECH_EDGE)
            reason_out = at.get(b, R_CLIP if b == duration_ms else R_SPEECH_EDGE)
            lo_i, hi_i = (a // hop, b // hop) if hop else (0, 0)
            anchors = _anchors_in(motion, a, b)
            mean_ae = round(_mean(action, lo_i, hi_i), 3) if hop else 0.0
            atoms.append(Atom(
                atom_id=len(atoms), file_id=file_id, start_ms=a, end_ms=b,
                state_in=reason_in, state_out=reason_out,
                action_energy=mean_ae,
                coherence=round(_mean(coherence, lo_i, hi_i), 3) if hop else 0.0,
                anchor_ms=anchors,
                # Pure LABEL: this atom sits in the clip's ACTIVE regime (its
                # mean energy clears the clip's own Otsu split) or carries a
                # detected impact. Never a keep/drop gate -- see build_atoms.
                is_action=bool(anchors) or (active_thr is not None and mean_ae >= active_thr),
                peak_action_energy=round(_peak(action, lo_i, hi_i), 3) if hop else 0.0,
                camera_motion=round(_mean(cam_motion, lo_i, hi_i), 3) if hop else 0.0,
            ))
    return atoms


def render_atom_table(atoms: List[Atom]) -> str:
    """The compact numbered text block for prompts -- motion enters as RAW
    NUMBERS, pixels as stills (pass 2). One line per atom:
    ``ATOM 7 [12300-15800] shot_cut->speech_edge act=0.70 peak=0.99 mot=0.55 coh=0.90 anchors@13100``
    where act/peak = mean/peak subject-motion energy (0..1), mot = camera-motion
    magnitude (0..1), coh = camera coherence (0..1), anchors = detected impacts.

    No derived ``cam=`` label: the model reads the raw mot/coh signals and
    categorizes camera behavior itself (deterministic-keep -- code hands over
    signals, the LLM does the semantics)."""
    lines = []
    for a in atoms:
        anchors = f" anchors@{','.join(str(x) for x in a.anchor_ms)}" if a.anchor_ms else ""
        lines.append(
            f"ATOM {a.atom_id} [{a.start_ms}-{a.end_ms}] {a.state_in}->{a.state_out} "
            f"act={a.action_energy:.2f} peak={a.peak_action_energy:.2f} "
            f"mot={a.camera_motion:.2f} coh={a.coherence:.2f}{anchors}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Speech side: the word list itself + prompt hints (never boundaries)
# --------------------------------------------------------------------------

def speech_hints(words: List[dict], turn_gap_ms: int = LONG_PAUSE_MS) -> List[str]:
    """Human-readable PROMPT HINTS for pass 1 -- where turns/speaker changes/
    long pauses fall in the word stream. Informational only: pass 1 decides
    the actual ``word_span`` grouping, these never constrain it."""
    hints: List[str] = []
    for i in range(1, len(words)):
        prev, cur = words[i - 1], words[i]
        prev_spk = prev.get("speaker") or "S?"
        cur_spk = cur.get("speaker") or "S?"
        gap = int(cur.get("start_ms", 0)) - int(prev.get("end_ms", 0))
        if cur_spk != prev_spk:
            hints.append(f"speaker change after word {i - 1} ({prev_spk} -> {cur_spk})")
        elif gap >= turn_gap_ms:
            hints.append(f"long pause after word {i - 1} ({gap / 1000:.1f}s)")
    return hints


def _snap_word_edge(words: List[dict], word_idx: int, silences: List[dict]) -> int:
    """The precise ms boundary immediately BEFORE ``words[word_idx]`` --
    snapped into the inter-word silence so a cut edge never lands mid-word.
    ``word_idx`` may equal ``len(words)`` (the edge after the last word)."""
    if not words:
        return 0
    if word_idx <= 0:
        return int(words[0].get("start_ms", 0))
    if word_idx >= len(words):
        return int(words[-1].get("end_ms", 0))
    prev_end = int(words[word_idx - 1].get("end_ms", 0))
    cur_start = int(words[word_idx].get("start_ms", 0))
    if cur_start <= prev_end:
        return cur_start   # words touch/overlap -- no gap to snap into
    for s in silences:
        s0, s1 = int(s.get("start_ms", 0)), int(s.get("end_ms", 0))
        if s0 < cur_start and s1 > prev_end:
            lo, hi = max(s0, prev_end), min(s1, cur_start)
            if hi > lo:
                return (lo + hi) // 2
    return (prev_end + cur_start) // 2


def resolve_speech_span_ms(
    words: List[dict], atoms: List[Atom], word_span: Tuple[int, int], silences: List[dict],
) -> Tuple[int, int]:
    """A speech cut's (start_ms, end_ms), snapped into inter-word silence
    like ``_snap_word_edge`` but then CLAMPED so the cushioning padding
    never intrudes into a neighboring atom's already-claimed territory.

    Needed because an inter-word gap and an atom-filled gap aren't always
    the same boundary: atoms are carved from the non-speech remainder using
    diarization TURNS, while a speech cut's own word_span/silence snap can
    independently reach partway into that same gap for a natural cut point.
    Observed against real data: a 2.8s pause between two words, almost
    entirely consumed by 3 video atoms, still let the silence-midpoint snap
    land ~1000ms inside the atoms' own span -- a real, reproducible
    coverage overlap, not model noise. Atoms and speech together must
    partition the timeline; this is the one adjustment enforcing that at
    the boundary the two disagree about."""
    s = _snap_word_edge(words, word_span[0], silences)
    e = _snap_word_edge(words, word_span[1] + 1, silences)
    if not words or not atoms:
        return s, e
    idx0, idx1 = word_span[0], word_span[1]
    first_word_start = int(words[idx0].get("start_ms", s)) if 0 <= idx0 < len(words) else s
    last_word_end = int(words[idx1].get("end_ms", e)) if 0 <= idx1 < len(words) else e
    following = [a.start_ms for a in atoms if a.start_ms >= last_word_end]
    if following:
        e = min(e, min(following))
    preceding = [a.end_ms for a in atoms if a.end_ms <= first_word_start]
    if preceding:
        s = max(s, max(preceding))
    return s, e


# --------------------------------------------------------------------------
# DB loaders + convenience wrappers (not pure -- real callers only)
# --------------------------------------------------------------------------

def _pg_conn():
    import psycopg
    from app.config import get_settings
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _load_words(file_id: str) -> List[dict]:
    """Flat, filler-free, time-ordered words for one file. A small, separate
    flattening from ``diarize.load_turns`` (which discards the flat list and
    is shared by other L2/L3 consumers -- not ours to change)."""
    with _pg_conn() as conn:
        row = conn.execute(
            "select segments from transcripts where file_id = %s", (file_id,)
        ).fetchone()
    if not row or not row[0]:
        return []
    segments = row[0] if isinstance(row[0], list) else json.loads(row[0])
    words = [w for seg in segments for w in (seg.get("words") or []) if not w.get("is_filler")]
    words.sort(key=lambda w: w.get("start_ms", 0))
    return words


def _load_silences(file_id: str) -> List[dict]:
    with _pg_conn() as conn:
        row = conn.execute(
            "select silence_intervals from audio_features where file_id = %s", (file_id,)
        ).fetchone()
    return (row[0] or []) if row else []


def snap_word_edge(file_id: str, word_idx: int) -> int:
    """The precise ms boundary immediately before word ``word_idx`` in
    ``file_id`` -- snapped into inter-word silence. See ``_snap_word_edge``."""
    words = _load_words(file_id)
    silences = _load_silences(file_id)
    return _snap_word_edge(words, word_idx, silences)


def load_motion_scene(file_id: str) -> Tuple[Optional[dict], Optional[dict]]:
    """The ``motion_dynamics`` + ``scene_cuts`` signals for one file, in the
    exact shape ``build_atoms`` expects. Extracted from ``load_lattice`` so the
    sync speech-swap can REBUILD a synced angle's atoms against the
    authoritative (re-based) words while still using THIS angle's own motion/
    scene -- see ``sync/lattice_merge.authoritative_view``."""
    with _pg_conn() as conn:
        m = conn.execute(
            """select hop_ms, camera_stability, camera_coherence, camera_motion,
                      blur, camera_cut_cost, action_energy, action_points,
                      transition_points
                 from motion_dynamics where file_id = %s""",
            (file_id,),
        ).fetchone()
        motion = None
        if m:
            motion = {
                "hop_ms": m[0], "camera_stability": m[1] or [],
                "camera_coherence": m[2] or [], "camera_motion": m[3] or [],
                "blur": m[4] or [], "camera_cut_cost": m[5] or [],
                "action_energy": m[6] or [], "action_points": m[7] or [],
                "transition_points": m[8] or [],
            }
        s = conn.execute(
            "select shot_points from scene_cuts where file_id = %s", (file_id,)
        ).fetchone()
        scene = {"shot_points": s[0] or []} if s else None
    return motion, scene


def load_lattice(file_id: str) -> Optional[Lattice]:
    """Load one file's full lattice (words + hints + video atoms). None when
    the file has no duration yet."""
    from app.services.l3.diarize import load_turns

    with _pg_conn() as conn:
        row = conn.execute(
            "select duration_seconds from files where id = %s", (file_id,)
        ).fetchone()
        if not row or not row[0]:
            return None
        duration_ms = int(float(row[0]) * 1000)

    motion, scene = load_motion_scene(file_id)
    words = _load_words(file_id)
    _text, _speakers, turns = load_turns(file_id, turn_gap_ms=LONG_PAUSE_MS)
    atoms = build_atoms(file_id, duration_ms, motion, scene, turns, words=words)
    hints = speech_hints(words)
    return Lattice(file_id=file_id, duration_ms=duration_ms, words=words,
                   turns=turns, hints=hints, atoms=atoms)
