"""
Hero-cuts assembly: the single ranked feed of "usable moments" for a project.

This is the V1 product surface. It is a deterministic, post-VLM assembly over
artifacts L1/L2/L3 already produced -- no new model call:

  * SPEECH heroes  come from the L1 ``dialogue_segments`` lens (topic/sentence),
    which are already snapped to silence troughs and carry off-camera /
    backchannel flags. Crisp boundaries, the strongest path.
  * ACTION / VISUAL heroes come from the VLM's ``content_units`` (kind = action
    / visual), with boundaries snapped through the FUSED SEAM FIELD -- one grid
    that fuses dialogue/camera vetoes with action/beat attractors, so an action
    cut can never land inside a spoken word (the old camera-only snapper bled
    background dialogue) while still hugging the motion impact.

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
from app.services.l1 import fused_seams as fseams
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
# Only at high energy do we sub-split a multi-sentence VLM unit into its
# sentences; below this the *delivered line* (the VLM unit) stays whole, so we
# never chop one continuous delivery into fragments.
ENERGY_SUBSPLIT_THRESHOLD = 0.75
ACTION_SNAP_WINDOW_MS = 1_500     # how far to search for a calm motion seam
ACTION_TIGHT_WINDOW_MS = 600      # the search window shrinks as energy rises

# A speech hero needs at least this much real content to be worth surfacing.
MIN_SPEECH_WORDS = 3
# A spoken span this poorly covered by the VLM's visible-speaking spans is
# almost certainly off-camera audio (a crew cue like "go", a voice off-frame).
SPEAKING_COVERAGE_MIN = 0.25
# Speech and action units within this gap are one moment (the dialogue+action cut).
COMBINE_GAP_MS = 1_500
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
    audio: Optional[dict] = None             # audio_features cut grids (dialogue/beat)


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
                   md.camera_cut_cost, md.action_points,
                   af.dialogue_cut_cost, af.dialogue_cut_hop_ms, af.dialogue_cut_points,
                   af.beat_cut_cost, af.beat_cut_hop_ms, af.beat_cut_points
              from files f
              left join dialogue_segments ds on ds.file_id = f.id
              left join clip_perception   cp on cp.file_id = f.id
              left join motion_dynamics   md on md.file_id = f.id
              left join audio_features    af on af.file_id = f.id
             where f.id = any(%s::uuid[])
            """,
            (file_ids,),
        ).fetchall()

    for (fid, dur_s, segments, perception, hop_ms, action_energy,
         action_cost, camera_cost, action_points,
         dlg_cost, dlg_hop, dlg_points, beat_cost, beat_hop, beat_points) in rows:
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
        audio = None
        if dlg_cost or beat_cost:
            audio = {
                "dialogue_cut_cost": _as_list(dlg_cost),
                "dialogue_cut_hop_ms": int(dlg_hop) if dlg_hop else 100,
                "dialogue_cut_points": _as_list(dlg_points),
                "beat_cut_cost": _as_list(beat_cost),
                "beat_cut_hop_ms": int(beat_hop) if beat_hop else 100,
                "beat_cut_points": _as_list(beat_points),
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
            audio=audio,
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

def _build_field(clip: _ClipInputs, energy: float) -> Optional[fseams.FusedField]:
    """The fused seam field for a clip: one cut-cost grid composing the dialogue
    and camera vetoes with the action and beat attractors. None when the clip has
    neither audio cut grids nor a motion grid (nothing to fuse)."""
    a, m = clip.audio or {}, clip.motion or {}
    if not a and not m:
        return None
    return fseams.compute_fused_field(
        duration_ms=clip.duration_ms, energy=energy,
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

def _speaking_coverage(speaking: List[dict], start_ms: int, end_ms: int) -> float:
    """Fraction of [start,end] covered by the union of the VLM's visible-speaking
    spans. ~1.0 means someone is clearly speaking on camera the whole time; ~0
    means the audio has no visible speaker (off-frame voice / crew cue)."""
    dur = max(1, end_ms - start_ms)
    ivals = sorted(
        (max(start_ms, int(s.get("start_ms", 0))), min(end_ms, int(s.get("end_ms", 0))))
        for s in speaking
        if ss._overlap_ms(int(s.get("start_ms", 0)), int(s.get("end_ms", 0)), start_ms, end_ms) > 0
    )
    covered, cur_e = 0, start_ms
    for a, b in ivals:
        a = max(a, cur_e)
        if b > a:
            covered += b - a
            cur_e = b
    return covered / dur


def _overlapping_sentences(sentences: List[dict], start_ms: int, end_ms: int) -> List[dict]:
    """Dialogue sentences (already silence-snapped by L1) overlapping a VLM
    unit's rough span -- the bridge that turns fuzzy VLM ms into clean cuts."""
    out = []
    for s in sentences:
        a = int(s.get("raw_in_ms", s.get("src_in_ms", 0)))
        b = int(s.get("raw_out_ms", s.get("src_out_ms", 0)))
        if ss._overlap_ms(a, b, start_ms, end_ms) > 0:
            out.append(s)
    return out


def _make_speech_hero(
    clip: _ClipInputs, source: Optional[ss.SpanSource], quality_events: List[dict],
    uid: str, in_ms: int, out_ms: int, raw_in: int, raw_out: int,
    text: str, speaker: Optional[str], extra_flags: List[str],
) -> Optional[HeroCut]:
    if out_ms <= in_ms:
        return None
    if source is not None:
        metrics = ss.score_span(source, in_ms, out_ms)
    else:
        metrics = {"speech_ratio": 1.0, "word_count": len(text.split())}
    if int(metrics.get("word_count", 0)) < MIN_SPEECH_WORDS:
        return None
    vlm = _vlm_quality_score(quality_events, raw_in, raw_out)
    return HeroCut(
        hero_id=f"{clip.file_id[:8]}:{uid}",
        file_id=clip.file_id,
        modality="speech",
        label=text.strip(),
        src_in_ms=in_ms,
        src_out_ms=out_ms,
        score=_speech_score(metrics, vlm),
        speaker=speaker,
        flags=extra_flags,
    )


def _speech_candidates(
    clip: _ClipInputs, source: Optional[ss.SpanSource], energy: float
) -> List[HeroCut]:
    """Speech heroes anchored on the VLM's understanding of the clip.

    The VLM's speech ``content_units`` define *what one delivered line is* (often
    spanning several transcript sentences); we snap each unit's boundaries to the
    L1 silence-snapped dialogue sentences it overlaps, so the cut is clean AND
    the delivery stays whole instead of being chopped at every sentence period.
    Energy only sub-splits a multi-sentence unit at high settings.

    Off-camera audio is dropped two ways: the L1 production-cue lexicon, and --
    the authoritative signal -- the VLM's visible-speaking spans (audio with no
    on-camera speaker is a crew cue / off-frame voice). Flub/restart attempts are
    suppressed. When the VLM never ran, we fall back to the L1 dialogue lens.
    """
    perception = clip.perception or {}
    quality_events = perception.get("take_quality_events") or []
    speaking = perception.get("speaking") or []
    units = [u for u in (perception.get("content_units") or [])
             if (u.get("kind") or "").lower() == "speech"]
    sentences = clip.dialogue.get("sentence") or []

    # Fallback: no VLM speech units -> the old dialogue-lens path (best effort).
    if not units:
        return _speech_candidates_from_lens(clip, source, energy, quality_events)

    out: List[HeroCut] = []
    for u in units:
        uid = str(u.get("unit_id", "sp"))
        raw_in, raw_out = int(u.get("start_ms", 0)), int(u.get("end_ms", 0))
        if raw_out <= raw_in:
            continue

        members = _overlapping_sentences(sentences, raw_in, raw_out)
        # Off-camera gate. When the VLM logged visible-speaking spans they are the
        # authority (more reliable than the L1 lexicon, which over-flags): a unit
        # the speaker is not visibly delivering is off-frame audio / a crew cue.
        # Only when the VLM gave us nothing do we fall back to the L1 lexicon.
        if speaking:
            if _speaking_coverage(speaking, raw_in, raw_out) < SPEAKING_COVERAGE_MIN:
                continue
        elif members and all(
            any(f in (s.get("flags") or []) for f in _OFFCAMERA_FLAGS)
            or "backchannel" in (s.get("flags") or [])
            for s in members
        ):
            continue

        # Snap to L1 sentence boundaries (clean silence cuts); fall back to raw.
        if members:
            in_ms = min(int(s.get("src_in_ms", raw_in)) for s in members)
            out_ms = max(int(s.get("src_out_ms", raw_out)) for s in members)
            speaker = members[0].get("speaker")
            extra = [f for s in members for f in (s.get("flags") or []) if f in ("noisy", "overlap")]
        else:
            in_ms, out_ms, speaker, extra = raw_in, raw_out, None, []

        # Default keeps the delivered line whole; only high energy fragments it.
        sub = members if (energy >= ENERGY_SUBSPLIT_THRESHOLD and len(members) > 1) else None
        if sub:
            for s in sub:
                h = _make_speech_hero(
                    clip, source, quality_events,
                    f"{uid}:{s.get('seg_id', 's')}",
                    int(s.get("src_in_ms", 0)), int(s.get("src_out_ms", 0)),
                    int(s.get("raw_in_ms", 0)), int(s.get("raw_out_ms", 0)),
                    s.get("text") or "", s.get("speaker"),
                    [f for f in (s.get("flags") or []) if f in ("noisy", "overlap")],
                )
                if h:
                    out.append(h)
        else:
            text = " ".join((s.get("text") or "").strip() for s in members).strip() \
                if members else (u.get("label") or u.get("content_key") or "")
            h = _make_speech_hero(clip, source, quality_events, uid,
                                  in_ms, out_ms, raw_in, raw_out, text,
                                  speaker, list(dict.fromkeys(extra)))
            if h:
                out.append(h)
    return out


def _speech_candidates_from_lens(
    clip: _ClipInputs, source: Optional[ss.SpanSource], energy: float,
    quality_events: List[dict],
) -> List[HeroCut]:
    """Fallback speech path for clips with no VLM perception: the L1 dialogue
    lens at the energy-selected granularity, off-camera/backchannel dropped."""
    level = "sentence" if energy >= ENERGY_SENTENCE_THRESHOLD else "topic"
    segs = clip.dialogue.get(level) or clip.dialogue.get("topic") or []
    out: List[HeroCut] = []
    for seg in segs:
        flags = list(seg.get("flags") or [])
        if any(f in flags for f in _OFFCAMERA_FLAGS) or "backchannel" in flags:
            continue
        h = _make_speech_hero(
            clip, source, quality_events, seg.get("seg_id", "sp"),
            int(seg.get("src_in_ms", 0)), int(seg.get("src_out_ms", 0)),
            int(seg.get("raw_in_ms", seg.get("src_in_ms", 0))),
            int(seg.get("raw_out_ms", seg.get("src_out_ms", 0))),
            seg.get("text") or "", seg.get("speaker"),
            [f for f in flags if f in ("noisy", "overlap")],
        )
        if h:
            out.append(h)
    return out


def _combined_candidates(
    clip: _ClipInputs, speech: List[HeroCut], action: List[HeroCut]
) -> List[HeroCut]:
    """The dialogue+action cut: when a spoken line sits right next to an action
    beat, surface the whole moment as one hero spanning both -- cut on the speech
    (silence) boundary at one end and the action (motion) boundary at the other.
    Emitted *in addition* to the speech-only and action-only heroes, so the
    editor sees every usable framing of the moment (just the line, just the
    action, or both together)."""
    out: List[HeroCut] = []
    for a in action:
        # Pair the action with its single nearest adjacent spoken line.
        best, best_gap = None, COMBINE_GAP_MS + 1
        for s in speech:
            (e_in, e_out), (l_in, l_out) = sorted(
                [(s.src_in_ms, s.src_out_ms), (a.src_in_ms, a.src_out_ms)]
            )
            gap = l_in - e_out  # <=0 when they overlap
            if gap < best_gap:
                best, best_gap = s, gap
        if best is None or best_gap > COMBINE_GAP_MS:
            continue
        s = best
        in_ms, out_ms = min(s.src_in_ms, a.src_in_ms), max(s.src_out_ms, a.src_out_ms)
        label = (f"{a.label} \u2192 {s.label}" if a.src_in_ms < s.src_in_ms
                 else f"{s.label} \u2192 {a.label}")
        out.append(HeroCut(
            hero_id=f"{clip.file_id[:8]}:moment:{a.hero_id.rsplit(':', 1)[-1]}",
            file_id=clip.file_id,
            modality="moment",
            label=label[:200],
            src_in_ms=in_ms,
            src_out_ms=out_ms,
            score=max(s.score, a.score),
            speaker=s.speaker,
        ))
    return out


def _action_candidates(clip: _ClipInputs, energy: float,
                       field: Optional[fseams.FusedField]) -> List[HeroCut]:
    """Action / visual heroes from the VLM's content_units, snapped through the
    FUSED SEAM FIELD (dialogue/camera vetoes + action/beat attractors) so a cut
    never lands inside speech. Falls back to the camera-only snapper if the fused
    field is unavailable. Only fires when the VLM units and motion grid exist."""
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
        if field is not None:
            in_ms, out_ms = fseams.snap_bounds(field, raw_in, raw_out,
                                               energy=energy, duration_ms=clip.duration_ms)
        else:
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
        field = _build_field(clip, energy)
        sp = _speech_candidates(clip, sources.get(fid), energy)
        ac = _action_candidates(clip, energy, field)
        heroes.extend(sp)
        heroes.extend(ac)
        heroes.extend(_combined_candidates(clip, sp, ac))

    heroes = _stack_takes(heroes, file_ids)
    heroes.sort(key=lambda h: h.score, reverse=True)
    return [h.to_dict() for h in heroes]
