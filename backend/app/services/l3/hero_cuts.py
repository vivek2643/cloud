"""
Hero-cuts assembly: the single ranked feed of "usable moments" for a project.

This is the V1 product surface. It is a deterministic, post-VLM assembly over
artifacts L1/L2/L3 already produced -- no new model call:

  * SPEECH heroes  come from the L1 ``dialogue_segments`` SENTENCES (already
    snapped to silence troughs, speaker-tagged, off-camera-flagged, and present
    on every clip). The energy dial sets granularity by CLUSTERING them: whole
    answers at low energy, sentences in the middle, clauses at the top.
  * ACTION / VISUAL heroes come from the VLM's ``content_units`` (kind = action
    / visual).

Every hero's boundaries flow through the FUSED SEAM FIELD -- one grid that
fuses dialogue/camera vetoes with action/beat attractors -- so a cut can never
land inside a spoken word or a camera move, and the energy dial sets the
tightness (snap-window width + veto-bounded breathing room) identically for
speech, action and moments.

The split of responsibility (see the design discussion):
  * The VLM PREDICTS what is usable + groups takes (``content_units`` /
    ``content_key`` / ``take_quality_events`` / visible-``speaking`` spans).
    Fuzzy, run once, persisted.
  * Deterministic models CUT the frames (the fused seam field) and RANK
    (``score_span``). Pure arithmetic over stored data -> reproducible.

Repeats of the same content are collapsed into ONE hero with its takes stacked
behind it (best in front), reusing the near-duplicate clustering from
``l3.takes``. The ``energy`` knob (0..1) is the single deterministic dial:
granularity (answers -> sentences -> clauses), tightness (loose -> frame-tight),
and whether action+dialogue is fused into one moment -- see ``l3.energy``.

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
from app.services.l3.energy import EnergyParams, energy_to_params
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
# The single energy dial -> concrete, deterministic cut parameters lives in
# ``l3.energy`` (``energy_to_params``). It controls BOTH axes monotonically:
#   * granularity -- low energy clusters sentences into whole answers; high
#     energy keeps sentences; the top sub-splits sentences into clauses.
#   * tightness   -- low energy cuts loose (wide fused search + pre/post-roll);
#     high energy cuts close (narrow search, zero padding).
ACTION_SNAP_WINDOW_MS = 1_500     # fallback (no fused field) calm-seam search
ACTION_TIGHT_WINDOW_MS = 600      # ... shrinks as energy rises

# A speech hero needs at least this much real content to be worth surfacing.
MIN_SPEECH_WORDS = 3
# A spoken span this poorly covered by the VLM's visible-speaking spans is
# almost certainly off-camera audio (a crew cue like "go", a voice off-frame).
SPEAKING_COVERAGE_MIN = 0.25
# Tolerance when deciding if an edge word is "on camera": the VLM's speaking
# spans start a touch late / end a touch early, so give a word's midpoint this
# much slack before calling it off-camera.
EDGE_TRIM_TOL_MS = 150
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


# Cost above this in the fused field counts as "no longer safe" -- padding stops
# here so loose breathing room never bleeds into speech / a camera move / impact.
_PAD_VETO_COST = 0.5


def _pad_safe(field: Optional[fseams.FusedField], ts: int, delta_ms: int,
              direction: int, duration_ms: int) -> int:
    """Extend a (already-snapped) boundary by up to ``delta_ms`` in ``direction``
    (+1 = outward past the out-point, -1 = before the in-point) for loose
    breathing room -- but ONLY through frames the fused field still calls safe.
    Stops the instant cost rises into a veto (speech, camera move, impact), so
    loose padding can only ever fill clean dead air, never pull in bad footage."""
    if field is None or delta_ms <= 0 or not field.cost:
        return ts
    hop = field.hop_ms or 100
    cur = ts
    for _ in range(max(0, delta_ms // hop)):
        nxt = cur + direction * hop
        if nxt < 0 or (duration_ms and nxt > duration_ms):
            break
        j = max(0, min(len(field.cost) - 1, int(round(nxt / hop))))
        if field.cost[j] > _PAD_VETO_COST:     # would enter a veto -> stop here
            break
        cur = nxt
    return cur


def _snap_fused(field: Optional[fseams.FusedField], raw_in: int, raw_out: int,
                params: EnergyParams, duration_ms: int) -> Tuple[int, int]:
    """Snap a rough [in,out] to the safest fused seams (search width = the
    energy-scaled snap window), then add energy-scaled, veto-bounded breathing
    room. The single path every hero's boundaries flow through, so the dial
    controls tightness identically for speech, action and moments."""
    if field is None:
        return raw_in, raw_out
    in_ms, out_ms = fseams.snap_bounds(
        field, raw_in, raw_out, energy=params.energy, duration_ms=duration_ms,
        base_win_ms=params.snap_window_ms, tight_win_ms=params.snap_window_ms)
    in_ms = _pad_safe(field, in_ms, params.pad_in_ms, -1, duration_ms)
    out_ms = _pad_safe(field, out_ms, params.pad_out_ms, +1, duration_ms)
    if out_ms <= in_ms:
        out_ms = in_ms + 1
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


def _on_camera_word(w: dict, speaking: List[dict]) -> bool:
    """True if a word's midpoint falls within (a bit of slack around) any of the
    VLM's visible-speaking spans -- i.e. someone is on camera delivering it."""
    mid = (int(w.get("start_ms", 0)) + int(w.get("end_ms", 0))) // 2
    return any(
        int(s.get("start_ms", 0)) - EDGE_TRIM_TOL_MS <= mid <= int(s.get("end_ms", 0)) + EDGE_TRIM_TOL_MS
        for s in speaking
    )


def _trim_to_speaking(
    words: List[dict], speaking: List[dict], win_in: int, win_out: int
) -> Optional[Tuple[int, int, List[dict]]]:
    """Trim off-camera words bleeding into the EDGES of a speech span (the crew
    "go"/"action" before a take, an off-frame voice after). Walks in from both
    ends dropping words whose midpoint isn't covered by a visible-speaking span,
    stopping at the first on-camera word. Interior words are never touched.

    Returns (new_in, new_out, kept_words) when an edge was trimmed, landing the
    cut in the silence gap just outside the kept words; None when nothing needed
    trimming (so the caller keeps the clean sentence boundary + text)."""
    span = [w for w in words
            if int(w.get("start_ms", 0)) < win_out and int(w.get("end_ms", 0)) > win_in]
    if not span:
        return None
    i, j = 0, len(span)
    while i < j and not _on_camera_word(span[i], speaking):
        i += 1
    while j > i and not _on_camera_word(span[j - 1], speaking):
        j -= 1
    if i == 0 and j == len(span):
        return None                       # no off-camera edge words
    kept = span[i:j]
    if len(kept) < MIN_SPEECH_WORDS:
        return None                       # nothing usable survives the trim
    new_in = win_in
    if i > 0:                             # land in the gap after the last crew word
        new_in = max(win_in, (int(span[i - 1]["end_ms"]) + int(kept[0]["start_ms"])) // 2)
    new_out = win_out
    if j < len(span):
        new_out = min(win_out, (int(kept[-1]["end_ms"]) + int(span[j]["start_ms"])) // 2)
    return new_in, new_out, kept


def _lexicon_offcamera(seg: dict) -> bool:
    """The L1 lexicon's verdict that a sentence is off-camera audio: a production
    cue / off-screen line / pure backchannel. Used as the pre-filter before
    clustering (the VLM speaking-span gate refines the edges afterwards)."""
    flags = seg.get("flags") or []
    return any(f in flags for f in _OFFCAMERA_FLAGS) or "backchannel" in flags


def _cluster_sentences(sentences: List[dict], cluster_gap_ms: int) -> List[List[dict]]:
    """Granularity = clustering. Merge adjacent sentences from the same speaker
    whose inter-sentence silence is shorter than ``cluster_gap_ms`` into one
    block. Low energy -> big gap -> whole answers; ``cluster_gap_ms == 0`` ->
    never merge -> each sentence stands alone. Speaker changes always break a
    cluster (we never fuse two people into one 'answer')."""
    sents = sorted(sentences, key=lambda s: int(s.get("src_in_ms", 0)))
    clusters: List[List[dict]] = []
    for s in sents:
        if clusters:
            prev = clusters[-1][-1]
            gap = int(s.get("src_in_ms", 0)) - int(prev.get("src_out_ms", 0))
            if s.get("speaker") == prev.get("speaker") and gap < cluster_gap_ms:
                clusters[-1].append(s)
                continue
        clusters.append([s])
    return clusters


def _cluster_span(members: List[dict]) -> Tuple[int, int, int, int, str, Optional[str], List[str]]:
    """Collapse a cluster of sentences into one rough span: silence-snapped
    in/out, the raw (un-snapped) envelope for VLM quality lookup, joined text,
    the speaker, and any quality flags carried up from the members."""
    in_ms = min(int(s.get("src_in_ms", 0)) for s in members)
    out_ms = max(int(s.get("src_out_ms", 0)) for s in members)
    raw_in = min(int(s.get("raw_in_ms", s.get("src_in_ms", 0))) for s in members)
    raw_out = max(int(s.get("raw_out_ms", s.get("src_out_ms", 0))) for s in members)
    text = " ".join((s.get("text") or "").strip() for s in members).strip()
    speaker = members[0].get("speaker")
    flags = list(dict.fromkeys(
        f for s in members for f in (s.get("flags") or []) if f in ("noisy", "overlap")))
    return in_ms, out_ms, raw_in, raw_out, text, speaker, flags


def _split_by_pauses(words: List[dict], lo: int, hi: int,
                     gap_ms: int) -> List[Tuple[int, int, str]]:
    """Top-of-dial clause split: break a span's words wherever the inter-word
    pause reaches ``gap_ms``. Each chunk -> (start, end, text). Returns [] when
    no words fall in the span (caller keeps the whole cluster)."""
    span = [w for w in words
            if int(w.get("start_ms", 0)) < hi and int(w.get("end_ms", 0)) > lo]
    if not span:
        return []
    chunks: List[List[dict]] = [[span[0]]]
    for w in span[1:]:
        if int(w.get("start_ms", 0)) - int(chunks[-1][-1].get("end_ms", 0)) >= gap_ms:
            chunks.append([w])
        else:
            chunks[-1].append(w)
    return [(int(c[0]["start_ms"]), int(c[-1]["end_ms"]),
             " ".join((x.get("text") or "").strip() for x in c).strip())
            for c in chunks]


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
    clip: _ClipInputs, source: Optional[ss.SpanSource],
    field: Optional[fseams.FusedField], params: EnergyParams,
) -> List[HeroCut]:
    """Speech heroes from the L1 dialogue sentences, clustered by energy.

    Sentences are the universal base atom -- already silence-snapped, speaker-
    tagged and off-camera-flagged, and present on every clip (unlike the VLM's
    content_units, which are often empty on long-form footage). The energy dial
    sets the granularity by *clustering* them:

      * low energy  -> merge adjacent same-speaker sentences into whole answers,
      * mid energy   -> sentences stand alone,
      * top energy   -> sub-split each sentence into clauses at its pauses.

    Each block's boundaries then flow through the fused seam field for the
    energy-scaled tightness (snap window + veto-bounded breathing room). Off-
    camera audio is dropped by the L1 production-cue lexicon and, where the VLM
    logged them, refined by its visible-speaking spans (edge-trim crew cues,
    drop wholly off-frame blocks).
    """
    perception = clip.perception or {}
    quality_events = perception.get("take_quality_events") or []
    speaking = perception.get("speaking") or []
    sentences = clip.dialogue.get("sentence") or clip.dialogue.get("topic") or []
    if not sentences:
        return []

    usable = [s for s in sentences if not _lexicon_offcamera(s)]
    if not usable:
        return []

    words = source.words if source is not None else []
    out: List[HeroCut] = []
    for ci, members in enumerate(_cluster_sentences(usable, params.cluster_gap_ms)):
        c_in, c_out, raw_in, raw_out, c_text, speaker, flags = _cluster_span(members)

        # Top-of-dial: fragment the block into clauses at its internal pauses.
        if params.clause_gap_ms > 0 and words:
            pieces = _split_by_pauses(words, c_in, c_out, params.clause_gap_ms) \
                or [(c_in, c_out, c_text)]
        else:
            pieces = [(c_in, c_out, c_text)]

        for pi, (p_in, p_out, p_text) in enumerate(pieces):
            in_ms, out_ms, text = p_in, p_out, (p_text or c_text)

            # Trim off-camera crew cues bleeding into the edges (VLM authority).
            if speaking and source is not None:
                tr = _trim_to_speaking(source.words, speaking, in_ms, out_ms)
                if tr:
                    in_ms, out_ms, kw = tr
                    text = " ".join((w.get("text") or "").strip() for w in kw).strip()

            # Drop a block whose audio has no visible speaker (off-frame voice).
            if speaking and _speaking_coverage(speaking, in_ms, out_ms) < SPEAKING_COVERAGE_MIN:
                continue

            # Tightness: snap to fused seams + energy-scaled breathing room.
            in_ms, out_ms = _snap_fused(field, in_ms, out_ms, params, clip.duration_ms)

            uid = f"sp{ci}" if len(pieces) == 1 else f"sp{ci}:{pi}"
            h = _make_speech_hero(clip, source, quality_events, uid,
                                  in_ms, out_ms, raw_in, raw_out, text, speaker, flags)
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


def _action_candidates(clip: _ClipInputs, params: EnergyParams,
                       field: Optional[fseams.FusedField]) -> List[HeroCut]:
    """Action / visual heroes from the VLM's content_units, snapped through the
    FUSED SEAM FIELD (dialogue/camera vetoes + action/beat attractors) so a cut
    never lands inside speech and gets the same energy-scaled tightness as
    speech. Falls back to the camera-only snapper if the fused field is
    unavailable. Only fires when the VLM units and motion grid exist."""
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
            in_ms, out_ms = _snap_fused(field, raw_in, raw_out, params, clip.duration_ms)
        else:
            in_ms, out_ms = _snap_action_bounds(clip.motion, raw_in, raw_out,
                                                clip.duration_ms, params.energy)
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

    `energy` (0..1) is the single deterministic dial: it sets speech granularity
    (clustered answers -> sentences -> clauses), cut tightness (loose breathing
    room -> frame-tight) and whether action+dialogue is fused into one moment.
    Returns hero dicts sorted best-first.
    """
    if not file_ids:
        return []
    params = energy_to_params(energy)

    inputs = _load_inputs(file_ids)
    sources = ss.load_sources(file_ids)

    heroes: List[HeroCut] = []
    for fid, clip in inputs.items():
        field = _build_field(clip, params.energy)
        sp = _speech_candidates(clip, sources.get(fid), field, params)
        ac = _action_candidates(clip, params, field)
        heroes.extend(sp)
        heroes.extend(ac)
        if params.fuse_moments:
            heroes.extend(_combined_candidates(clip, sp, ac))

    heroes = _stack_takes(heroes, file_ids)
    heroes.sort(key=lambda h: h.score, reverse=True)
    return [h.to_dict() for h in heroes]
