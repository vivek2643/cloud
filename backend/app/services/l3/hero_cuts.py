"""
Hero-cuts assembly: the single ranked feed of "usable moments" for a project.

This is the V1 product surface. It is a deterministic, post-VLM assembly over
artifacts L1/L2/L3 already produced -- no new model call:

  * SPEECH heroes  come from the L1 ``dialogue_segments`` lens (topic/sentence),
    which are already snapped to silence troughs and carry off-camera /
    backchannel flags. Crisp boundaries, the strongest path.
  * ACTION / VISUAL heroes come from the VLM's ``content_units`` (kind = action
    / visual), with boundaries snapped to the deterministic motion grid
    (``camera_cut_cost`` troughs = the calmest, most stable frame to cut on).
    Fuzzier boundaries, as expected without a transcript to lean on.

The split of responsibility (see the design discussion):
  * The VLM PREDICTS what is usable + groups takes (``content_units`` /
    ``content_key`` / ``take_quality_events``). Fuzzy, run once, persisted.
  * Deterministic models CUT the frames (silence troughs, motion troughs) and
    RANK (``score_span``). Pure arithmetic over stored data -> reproducible.

Repeats of the same content are collapsed into ONE hero with its takes stacked
behind it (best in front), reusing the near-duplicate clustering from
``l3.takes``. The ``energy`` knob (0..1) is deterministic: it chooses the speech
granularity (topic vs sentence) and how tightly action spans are trimmed.

Best-effort throughout: a missing artifact for a file simply yields fewer (or
no) heroes for it; nothing here raises.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from difflib import SequenceMatcher

from app.config import get_settings
from app.services.l3 import score_span as ss
from app.services.l3.takes import build_take_groups, normalize_key

# Cross-file consolidation: collapse heroes whose *displayed line* is the same
# scripted line shot as separate takes. Strict on purpose -- high ratio + a
# length guard + a token floor -- so distinct sentences never merge (this is the
# pass that must not regress into the old fuzzy over-merge).
_MERGE_RATIO = 0.90
_MERGE_MIN_TOKENS = 4
_MERGE_LEN_RATIO = 0.7

logger = logging.getLogger(__name__)


# --- Energy knob ----------------------------------------------------------
# Deterministic mapping from the single energy slider (0 = calm/broad,
# 1 = punchy/sharp) to concrete, signal-level choices. Phase 2 refines this;
# for now it controls speech granularity + action trim tightness.
ENERGY_SENTENCE_THRESHOLD = 0.5   # >= this -> sentence-level speech (shorter)
ACTION_SNAP_WINDOW_MS = 1_500     # how far to search for a calm motion seam
ACTION_TIGHT_WINDOW_MS = 600      # the search window shrinks as energy rises

# A speech hero needs at least this much real content to be worth surfacing.
MIN_SPEECH_WORDS = 3
# Hidden-by-default flags inherited from the dialogue lens.
_OFFCAMERA_FLAGS = ("offscreen", "production_cue")


@dataclass
class HeroTake:
    """One delivery of a hero's content (the best is the hero itself)."""
    file_id: str
    src_in_ms: int
    src_out_ms: int
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_id": self.file_id,
            "src_in_ms": self.src_in_ms,
            "src_out_ms": self.src_out_ms,
            "score": round(self.score, 3),
        }


@dataclass
class HeroCut:
    hero_id: str
    file_id: str
    modality: str                 # speech | action | visual
    label: str                    # human-facing text / description
    src_in_ms: int
    src_out_ms: int
    score: float                  # 0..1 rank key
    speaker: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    take_count: int = 1           # how many comparable takes exist (incl. this)
    alt_takes: List[HeroTake] = field(default_factory=list)  # the losers, best-first

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hero_id": self.hero_id,
            "file_id": self.file_id,
            "modality": self.modality,
            "label": self.label,
            "src_in_ms": self.src_in_ms,
            "src_out_ms": self.src_out_ms,
            "duration_ms": self.src_out_ms - self.src_in_ms,
            "score": round(self.score, 3),
            "speaker": self.speaker,
            "flags": self.flags,
            "take_count": self.take_count,
            "alt_takes": [t.to_dict() for t in self.alt_takes],
        }


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def _pg_conn():
    import psycopg
    settings = get_settings()
    return psycopg.connect(settings.database_url, autocommit=True)


def _as_list(v: Any) -> list:
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return []
    return []


def _as_doc(v: Any) -> Optional[dict]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return None
    return None


@dataclass
class _ClipInputs:
    file_id: str
    duration_ms: int
    dialogue: Dict[str, list]                # {"sentence": [...], "topic": [...]}
    perception: Optional[dict]
    motion: Optional[dict]                    # motion_dynamics row as a dict


def _load_inputs(file_ids: List[str]) -> Dict[str, _ClipInputs]:
    """One query for everything the assembly needs across the clips in scope."""
    if not file_ids:
        return {}
    out: Dict[str, _ClipInputs] = {}
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select f.id::text,
                   coalesce(f.duration_seconds, 0),
                   ds.segments,
                   cp.perception,
                   md.hop_ms, md.action_energy, md.action_cut_cost,
                   md.camera_cut_cost, md.action_points
              from files f
              left join dialogue_segments ds on ds.file_id = f.id
              left join clip_perception   cp on cp.file_id = f.id
              left join motion_dynamics   md on md.file_id = f.id
             where f.id = any(%s::uuid[])
            """,
            (file_ids,),
        ).fetchall()

    for (fid, dur_s, segments, perception, hop_ms, action_energy,
         action_cost, camera_cost, action_points) in rows:
        seg_doc = _as_doc(segments) or {}
        motion = None
        if hop_ms:
            motion = {
                "hop_ms": int(hop_ms),
                "action_energy": _as_list(action_energy),
                "action_cut_cost": _as_list(action_cost),
                "camera_cut_cost": _as_list(camera_cost),
                "action_points": _as_list(action_points),
            }
        out[fid] = _ClipInputs(
            file_id=fid,
            duration_ms=int(float(dur_s) * 1000),
            dialogue={
                "sentence": seg_doc.get("sentence", []) or [],
                "topic": seg_doc.get("topic", []) or [],
            },
            perception=_as_doc(perception),
            motion=motion,
        )
    return out


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------

def _vlm_quality_score(events: List[dict], start_ms: int, end_ms: int) -> Optional[float]:
    """Mean of the VLM's 1..5 take-quality scores overlapping a window,
    normalized to 0..1. None when the VLM logged no quality there."""
    scores = [
        int(q.get("score", 0))
        for q in events
        if ss._overlap_ms(int(q.get("start_ms", 0)), int(q.get("end_ms", 0)), start_ms, end_ms) > 0
        and q.get("score") is not None
    ]
    if not scores:
        return None
    return (sum(scores) / len(scores) - 1.0) / 4.0  # 1..5 -> 0..1


def _speech_score(metrics: Dict[str, Any], vlm: Optional[float]) -> float:
    """Composite 0..1 rank for a speech span: reward clean, on-camera, content-
    bearing delivery; penalize fillers, dead air, runaway pace. Objective
    metrics carry it; the VLM's subjective score nudges when present."""
    speech_ratio = float(metrics.get("speech_ratio", 0.0))
    fillers = float(metrics.get("filler_per_min", 0.0))
    wpm = float(metrics.get("wpm", 0.0))
    words = int(metrics.get("word_count", 0))
    gaze = metrics.get("gaze_to_camera_ratio")

    filler_pen = min(1.0, fillers / 12.0)            # ~12 fillers/min -> full penalty
    # Comfortable speaking pace ~110-170 wpm; penalize outside that band.
    pace_pen = 0.0 if 90 <= wpm <= 190 else min(1.0, abs(wpm - 140) / 140.0)
    content = min(1.0, words / 12.0)                 # a few words -> enough to use

    base = (
        0.40 * speech_ratio
        + 0.25 * content
        + 0.20 * (1.0 - filler_pen)
        + 0.15 * (1.0 - pace_pen)
    )
    if gaze is not None:
        base = 0.85 * base + 0.15 * float(gaze)
    if vlm is not None:
        base = 0.7 * base + 0.3 * vlm
    return max(0.0, min(1.0, base))


def _mean(xs: List[float], lo: int, hi: int) -> float:
    seg = xs[max(0, lo):max(lo + 1, hi)]
    return sum(seg) / len(seg) if seg else 0.0


def _action_score(motion: dict, raw_in: int, raw_out: int, vlm: Optional[float]) -> float:
    """Salience of an action/visual span: how much subject motion it carries
    (normalized 0..1 action energy), nudged by the VLM's quality score."""
    hop = max(1, int(motion.get("hop_ms", 0)))
    energy = motion.get("action_energy") or []
    base = _mean(energy, raw_in // hop, raw_out // hop) if energy else 0.3
    if vlm is not None:
        base = 0.6 * base + 0.4 * vlm
    return max(0.0, min(1.0, base))


# --------------------------------------------------------------------------
# Boundary snapping (action/visual) -- speech is already snapped by L1
# --------------------------------------------------------------------------

def _snap_calm(cost: List[float], hop_ms: int, raw_ms: int,
               lo_ms: int, hi_ms: int) -> int:
    """Snap a boundary to the calmest (lowest camera_cut_cost) frame in a
    window -- the most stable, cheapest place to cut. Falls back to raw_ms."""
    if not cost or hop_ms <= 0 or hi_ms <= lo_ms:
        return raw_ms
    i0 = max(0, lo_ms // hop_ms)
    i1 = min(len(cost) - 1, hi_ms // hop_ms)
    if i1 < i0:
        return raw_ms
    best_i, best_v = i0, cost[i0]
    for i in range(i0 + 1, i1 + 1):
        if cost[i] < best_v:
            best_v, best_i = cost[i], i
    return best_i * hop_ms


def _snap_action_bounds(motion: dict, raw_in: int, raw_out: int,
                        duration_ms: int, energy: float) -> Tuple[int, int]:
    """Pull an action span's in/out to the nearest calm motion seam. The search
    window tightens as energy rises (punchier cuts hug the action)."""
    cost = motion.get("camera_cut_cost") or []
    hop = int(motion.get("hop_ms", 0))
    win = int(ACTION_SNAP_WINDOW_MS - (ACTION_SNAP_WINDOW_MS - ACTION_TIGHT_WINDOW_MS) * energy)
    in_ms = _snap_calm(cost, hop, raw_in, max(0, raw_in - win), raw_in)
    out_ms = _snap_calm(cost, hop, raw_out, raw_out, min(duration_ms or raw_out + win, raw_out + win))
    if out_ms <= in_ms:
        out_ms = max(raw_out, in_ms + 1)
    return in_ms, out_ms


# --------------------------------------------------------------------------
# Per-modality candidate builders
# --------------------------------------------------------------------------

def _speech_candidates(
    clip: _ClipInputs, source: Optional[ss.SpanSource], energy: float
) -> List[HeroCut]:
    """Speech heroes from the dialogue lens at the energy-selected granularity.
    Off-camera / backchannel selects are dropped (not usable as heroes)."""
    level = "sentence" if energy >= ENERGY_SENTENCE_THRESHOLD else "topic"
    segs = clip.dialogue.get(level) or clip.dialogue.get("topic") or []
    quality_events = (clip.perception or {}).get("take_quality_events") or []

    out: List[HeroCut] = []
    for seg in segs:
        flags = list(seg.get("flags") or [])
        if any(f in flags for f in _OFFCAMERA_FLAGS) or "backchannel" in flags:
            continue
        text = (seg.get("text") or "").strip()
        in_ms, out_ms = int(seg.get("src_in_ms", 0)), int(seg.get("src_out_ms", 0))
        if out_ms <= in_ms:
            continue

        if source is not None:
            metrics = ss.score_span(source, in_ms, out_ms)
        else:
            metrics = {"speech_ratio": 1.0, "word_count": len(text.split())}
        if int(metrics.get("word_count", 0)) < MIN_SPEECH_WORDS:
            continue
        vlm = _vlm_quality_score(quality_events, seg.get("raw_in_ms", in_ms),
                                 seg.get("raw_out_ms", out_ms))
        out.append(HeroCut(
            hero_id=f"{clip.file_id[:8]}:{seg.get('seg_id', 'sp')}",
            file_id=clip.file_id,
            modality="speech",
            label=text,
            src_in_ms=in_ms,
            src_out_ms=out_ms,
            score=_speech_score(metrics, vlm),
            speaker=seg.get("speaker"),
            flags=[f for f in flags if f in ("noisy", "overlap")],
        ))
    return out


def _action_candidates(clip: _ClipInputs, energy: float) -> List[HeroCut]:
    """Action / visual heroes from the VLM's content_units, snapped to the
    deterministic motion grid. Only fires when both the VLM units and the motion
    grid exist (no transcript anchor for these)."""
    if not clip.perception or not clip.motion:
        return []
    units = clip.perception.get("content_units") or []
    quality_events = clip.perception.get("take_quality_events") or []

    out: List[HeroCut] = []
    for u in units:
        kind = (u.get("kind") or "").lower()
        if kind not in ("action", "visual"):
            continue
        raw_in, raw_out = int(u.get("start_ms", 0)), int(u.get("end_ms", 0))
        if raw_out <= raw_in:
            continue
        in_ms, out_ms = _snap_action_bounds(clip.motion, raw_in, raw_out,
                                            clip.duration_ms, energy)
        vlm = _vlm_quality_score(quality_events, raw_in, raw_out)
        label = (u.get("label") or u.get("content_key") or kind).strip()
        out.append(HeroCut(
            hero_id=f"{clip.file_id[:8]}:{u.get('unit_id', 'act')}",
            file_id=clip.file_id,
            modality=kind,
            label=label,
            src_in_ms=in_ms,
            src_out_ms=out_ms,
            score=_action_score(clip.motion, raw_in, raw_out, vlm),
        ))
    return out


# --------------------------------------------------------------------------
# Take stacking: collapse repeats into one hero, best in front
# --------------------------------------------------------------------------

def _overlap_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _stack_takes(heroes: List[HeroCut], file_ids: List[str]) -> List[HeroCut]:
    """Collapse re-delivered content into one hero carrying its alternates.

    The grouping is *not* re-derived here -- it comes from `l3.takes`, the same
    authoritative source the rest of the system uses. `build_take_groups` finds
    deliveries of the same content via the VLM's content_keys and restart
    markers (and a transcript-sentence fallback), and only returns groups with
    >= 2 real attempts. We map each attempt onto the speech hero it overlaps and
    fold the losers into the winner's `alt_takes`. This is conservative on
    purpose: it avoids the fuzzy-text over-merging that previously hid dozens of
    distinct sentences behind a single card.

    Action/visual heroes are not take-comparable, so they pass through.
    """
    speech = [h for h in heroes if h.modality == "speech"]
    others = [h for h in heroes if h.modality != "speech"]
    if not speech:
        return heroes

    by_file: Dict[str, List[HeroCut]] = {}
    for h in speech:
        by_file.setdefault(h.file_id, []).append(h)

    def hero_for(att) -> Optional[HeroCut]:
        best, best_ov = None, 0
        for h in by_file.get(att.file_id, []):
            ov = _overlap_ms(h.src_in_ms, h.src_out_ms, att.start_ms, att.end_ms)
            if ov > best_ov:
                best, best_ov = h, ov
        return best

    consumed: set[str] = set()
    stacked: List[HeroCut] = []
    for group in build_take_groups(file_ids):
        # Pair every attempt with the hero it overlaps (or None -> raw span).
        members = [(hero_for(a), a) for a in group.attempts]
        # The front of the stack is the best-scored hero among the deliveries.
        candidates = [h for h, _ in members if h is not None and h.hero_id not in consumed]
        if not candidates:
            continue
        front = max(candidates, key=lambda h: h.score)
        front_att_id = next(a.attempt_id for h, a in members if h is front)

        alts: List[HeroTake] = []
        for h, a in members:
            if a.attempt_id == front_att_id:
                continue
            if h is not None and h is not front:
                alts.append(HeroTake(h.file_id, h.src_in_ms, h.src_out_ms, h.score))
                consumed.add(h.hero_id)
            else:
                # A delivery with no distinct hero (e.g. a restart sub-span of
                # the front line): expose it as a raw alternate span.
                alts.append(HeroTake(a.file_id, a.start_ms, a.end_ms, 0.0))
        if not alts:
            continue
        front.take_count = 1 + len(alts)
        front.alt_takes = sorted(alts, key=lambda t: t.score, reverse=True)
        consumed.add(front.hero_id)
        stacked.append(front)

    solo = [h for h in speech if h.hero_id not in consumed]
    # Second pass: collapse remaining heroes that show the *same line* (scripted
    # repeats across takes/files) which content-key grouping missed.
    speech_out = _consolidate_speech(stacked + solo)
    return speech_out + others


def _strict_same_line(a: str, b: str) -> bool:
    """True when two displayed lines are the same scripted line (take of each
    other): high text ratio, similar length, and long enough to be meaningful."""
    ta, tb = a.split(), b.split()
    if len(ta) < _MERGE_MIN_TOKENS or len(tb) < _MERGE_MIN_TOKENS:
        return False
    if min(len(ta), len(tb)) / max(len(ta), len(tb)) < _MERGE_LEN_RATIO:
        return False
    return SequenceMatcher(None, a, b).ratio() >= _MERGE_RATIO


def _consolidate_speech(heroes: List[HeroCut]) -> List[HeroCut]:
    """Greedy, strict merge of heroes that present the same line. Folds losers
    (and their existing alternates) into the highest-scoring hero's stack."""
    groups: List[List[HeroCut]] = []
    keys: List[str] = []
    for h in heroes:
        k = normalize_key(h.label)
        placed = False
        for gi, gk in enumerate(keys):
            if _strict_same_line(k, gk):
                groups[gi].append(h)
                placed = True
                break
        if not placed:
            groups.append([h])
            keys.append(k)

    out: List[HeroCut] = []
    for g in groups:
        if len(g) == 1:
            out.append(g[0])
            continue
        front = max(g, key=lambda h: h.score)
        alts: List[HeroTake] = list(front.alt_takes)
        for m in g:
            if m is front:
                continue
            alts.append(HeroTake(m.file_id, m.src_in_ms, m.src_out_ms, m.score))
            alts.extend(m.alt_takes)
        front.alt_takes = sorted(alts, key=lambda t: t.score, reverse=True)
        front.take_count = 1 + len(front.alt_takes)
        out.append(front)
    return out


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def build_hero_cuts(file_ids: List[str], energy: float = 0.5) -> List[Dict[str, Any]]:
    """The ranked hero feed for a set of clips.

    `energy` (0..1) is deterministic: it selects speech granularity (topic vs
    sentence) and tightens action trims. Returns hero dicts sorted best-first.
    """
    if not file_ids:
        return []
    energy = max(0.0, min(1.0, float(energy)))

    inputs = _load_inputs(file_ids)
    sources = ss.load_sources(file_ids)

    heroes: List[HeroCut] = []
    for fid, clip in inputs.items():
        heroes.extend(_speech_candidates(clip, sources.get(fid), energy))
        heroes.extend(_action_candidates(clip, energy))

    heroes = _stack_takes(heroes, file_ids)
    heroes.sort(key=lambda h: h.score, reverse=True)
    return [h.to_dict() for h in heroes]
