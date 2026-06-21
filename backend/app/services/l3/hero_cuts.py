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
from app.services.l3 import anchors as anc
from app.services.l3 import recommend
from app.services.l3 import score_span as ss
from app.services.l3 import territory as terr
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

# Primary content (speech/action) ranks above silent cutaways by default, so the
# top of the feed is the substance; cutaways stay one filter-click away. A pure
# rank nudge -- it never hides anything.
_AFFORDANCE_WEIGHT = {
    anc.AFF_SPEECH: 1.00,
    anc.AFF_ACTION: 1.00,
    anc.AFF_REACTION: 0.82,
    anc.AFF_BROLL: 0.72,
    anc.AFF_INSERT: 0.70,
}
# A beat segment needs at least this salience to be worth surfacing as a card
# (everything is still reachable in raw via the timeline; this just declutters).
_MIN_BEAT_SALIENCE = 0.15


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
    modality: str                 # the DOMINANT affordance (speech|action|reaction|broll|insert)
    label: str                    # human-facing text / description
    src_in_ms: int
    src_out_ms: int
    score: float                  # 0..1 rank key
    speaker: Optional[str] = None
    flags: List[str] = field(default_factory=list)
    affordances: List[str] = field(default_factory=list)  # all editorial uses (filter keys)
    take_count: int = 1           # how many comparable takes exist (incl. this)
    alt_takes: List[HeroTake] = field(default_factory=list)  # the losers, best-first
    # Source dialogue sentence ids this cut spans (speech/moment cuts only).
    # Internal: drives the LLM-filtration "contains a keeper" recommendation rule.
    member_seg_ids: List[str] = field(default_factory=list)
    recommended: bool = True            # in the curated "Recommended" pool?
    recommend_reason: Optional[str] = None  # why dropped (when not recommended)

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
            "affordances": self.affordances or [self.modality],
            "take_count": self.take_count,
            "alt_takes": [t.to_dict() for t in self.alt_takes],
            "recommended": self.recommended,
            "recommend_reason": self.recommend_reason,
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
                   af.beat_cut_cost, af.beat_cut_hop_ms, af.beat_cut_points,
                   af.rms_db, af.prosody_hop_ms
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
         dlg_cost, dlg_hop, dlg_points, beat_cost, beat_hop, beat_points,
         rms_db, prosody_hop) in rows:
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
        if dlg_cost or beat_cost or rms_db:
            audio = {
                "dialogue_cut_cost": _as_list(dlg_cost),
                "dialogue_cut_hop_ms": int(dlg_hop) if dlg_hop else 100,
                "dialogue_cut_points": _as_list(dlg_points),
                "beat_cut_cost": _as_list(beat_cost),
                "beat_cut_hop_ms": int(beat_hop) if beat_hop else 100,
                "beat_cut_points": _as_list(beat_points),
                "rms_db": _as_list(rms_db),
                "prosody_hop_ms": int(prosody_hop) if prosody_hop else 0,
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
    member_seg_ids: Optional[List[str]] = None,
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
        modality=anc.AFF_SPEECH,
        label=text.strip(),
        src_in_ms=in_ms,
        src_out_ms=out_ms,
        score=_speech_score(metrics, vlm),
        speaker=speaker,
        flags=extra_flags,
        affordances=[anc.AFF_SPEECH],
        member_seg_ids=list(member_seg_ids or []),
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
        # All clause pieces of this block inherit the block's source sentence ids
        # (the LLM verdict lives at the sentence level; clauses roll up to it).
        # Namespaced by file: per-file seg_ids collide across clips.
        member_seg_ids = [
            recommend.member_key(clip.file_id, s["seg_id"])
            for s in members if s.get("seg_id")
        ]

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
                                  in_ms, out_ms, raw_in, raw_out, text, speaker, flags,
                                  member_seg_ids=member_seg_ids)
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
            affordances=[anc.AFF_SPEECH, anc.AFF_ACTION],
            member_seg_ids=list(s.member_seg_ids),
        ))
    return out


def _merge_gap_for_aff(params: EnergyParams, aff: str) -> int:
    if aff == anc.AFF_ACTION:
        return params.action_merge_gap_ms
    if aff == anc.AFF_REACTION:
        return params.reaction_merge_gap_ms
    if aff == anc.AFF_BROLL:
        return params.broll_merge_gap_ms
    if aff == anc.AFF_INSERT:
        return 0
    return params.cluster_gap_ms


def _motion_onset_ms(motion: Optional[dict], unit_in: int, unit_out: int) -> int:
    """First hop inside the unit where action energy rises above a local baseline."""
    if not motion:
        return unit_in
    hop = max(1, int(motion.get("hop_ms", 100)))
    energy = motion.get("action_energy") or []
    if not energy:
        return unit_in
    i0 = max(0, unit_in // hop)
    i1 = min(len(energy) - 1, unit_out // hop)
    if i0 > i1:
        return unit_in
    seg = [energy[i] for i in range(i0, i1 + 1)]
    lo = sorted(seg)[max(0, len(seg) // 4)]
    hi = max(seg)
    if hi - lo < 0.05:
        return unit_in
    thresh = lo + 0.4 * (hi - lo)
    for i in range(i0, i1 + 1):
        if energy[i] >= thresh:
            return max(unit_in, i * hop)
    return unit_in


def _action_core(anchor: anc.Anchor, motion: Optional[dict], mode: str) -> Tuple[int, int]:
    """Editorial core for an action/performance beat before snapping."""
    unit_in, unit_out = anchor.start_ms, anchor.end_ms
    impact = anchor.ts_ms
    if mode == "unit":
        return unit_in, unit_out
    onset = _motion_onset_ms(motion, unit_in, unit_out)
    if mode == "onset":
        return onset, unit_out
    # impact: mandatory region is contact through settle (preamble is droppable)
    return min(max(onset, impact), unit_out), unit_out


def _snap_segment(
    field: Optional[fseams.FusedField], core_in: int, core_out: int,
    params: EnergyParams, clip: _ClipInputs, aff: str,
) -> Tuple[int, int]:
    if field is not None:
        in_ms, out_ms = fseams.snap_around_core(
            field, core_in, core_out,
            win_ms=params.snap_window_ms, duration_ms=clip.duration_ms)
        if aff != anc.AFF_ACTION:
            in_ms = _pad_safe(field, in_ms, params.pad_in_ms, -1, clip.duration_ms)
            out_ms = _pad_safe(field, out_ms, params.pad_out_ms, +1, clip.duration_ms)
        return in_ms, out_ms
    if clip.motion and aff == anc.AFF_ACTION:
        return _snap_action_bounds(clip.motion, core_in, core_out,
                                   clip.duration_ms, params.energy)
    return core_in, core_out


def _collapse_insert_anchors(items: List[anc.Anchor], gap_ms: int) -> List[anc.Anchor]:
    """Merge repeated graphic/on-screen labels into one anchor (Broad/Calm)."""
    if not items:
        return items
    out: List[anc.Anchor] = []
    for a in sorted(items, key=lambda x: x.start_ms):
        key = (a.text or a.kind or "").strip().lower()
        if out and key and key == (out[-1].text or out[-1].kind or "").strip().lower():
            prev = out[-1]
            if a.start_ms - prev.end_ms <= gap_ms:
                out[-1] = anc.Anchor(
                    ts_ms=prev.ts_ms, start_ms=prev.start_ms, end_ms=max(prev.end_ms, a.end_ms),
                    kind=prev.kind, affordance=prev.affordance,
                    salience=max(prev.salience, a.salience), actor=prev.actor or a.actor,
                    text=prev.text, source_id=prev.source_id,
                )
                continue
        out.append(a)
    return out


def _prep_overlay_group(
    group: List[anc.Anchor], aff: str, params: EnergyParams,
    speaking: List[dict],
) -> List[anc.Anchor]:
    """Filter + collapse overlay anchors for this energy band."""
    kept: List[anc.Anchor] = []
    for a in group:
        if a.kind == "audio_event":
            if a.salience < params.audio_min_salience:
                continue
        elif aff == anc.AFF_REACTION:
            dur = a.end_ms - a.start_ms
            if dur < params.reaction_min_duration_ms:
                continue
            if a.salience < params.reaction_min_warrant:
                continue
        elif aff == anc.AFF_BROLL:
            if a.salience < params.broll_min_salience:
                continue
            if params.broll_prefer_low_speech:
                occ = terr.speech_occupation(a.start_ms, a.end_ms, speaking)
                if occ >= 0.5 and a.salience < 0.65:
                    continue
        elif aff == anc.AFF_INSERT:
            if a.salience < params.insert_min_salience:
                continue
        kept.append(a)
    if aff == anc.AFF_INSERT and params.insert_collapse_graphics:
        kept = _collapse_insert_anchors(kept, gap_ms=30_000)
    return kept


# Sharp-band split: editorial hinge at impact; fused field snaps outer edges only.
_ACTION_SPLIT_MIN_WINDUP_MS = 150
_ACTION_SPLIT_MAX_WINDUP_MS = 400


def _action_min_windup_ms(unit_len_ms: int) -> int:
    """Scale windup floor with beat length (short actions need less runway)."""
    if unit_len_ms <= 0:
        return _ACTION_SPLIT_MIN_WINDUP_MS
    scaled = unit_len_ms // 5
    return max(_ACTION_SPLIT_MIN_WINDUP_MS, min(_ACTION_SPLIT_MAX_WINDUP_MS, scaled))


def _action_pieces(
    anchor: anc.Anchor, motion: Optional[dict], params: EnergyParams,
    field: Optional[fseams.FusedField], clip: _ClipInputs,
) -> List[Tuple[int, int, str]]:
    """One or two (core_in, core_out, label_suffix) per action anchor."""
    mode = params.action_anchor_mode
    if params.action_split_at_impact:
        impact = anchor.ts_ms
        onset = _motion_onset_ms(motion, anchor.start_ms, anchor.end_ms)
        min_windup = _action_min_windup_ms(anchor.end_ms - anchor.start_ms)
        if impact >= onset + min_windup and onset < impact < anchor.end_ms:
            return [
                (onset, impact, " · windup"),
                (impact, anchor.end_ms, " · payoff"),
            ]
    cin, cout = _action_core(anchor, motion, mode)
    return [(cin, cout, "")]


def _action_segments(
    group: List[anc.Anchor], clip: _ClipInputs, field: Optional[fseams.FusedField],
    params: EnergyParams, quality_events: List[dict],
) -> List[HeroCut]:
    out: List[HeroCut] = []
    motion = clip.motion
    for ci, members in enumerate(_cluster_anchors(group, params.action_merge_gap_ms)):
        if params.action_merge_gap_ms > 0 and len(members) > 1:
            # Broad/Calm: one card for merged beats
            anchor = max(members, key=lambda m: m.salience)
            cin, cout = min(m.start_ms for m in members), max(m.end_ms for m in members)
            pieces = [(cin, cout, "")]
            label_base = (anchor.text or "action").strip()
        else:
            anchor = members[0]
            pieces = _action_pieces(anchor, motion, params, field, clip)
            label_base = (anchor.text or "action").strip()
        for pi, (cin, cout, suffix) in enumerate(pieces):
            if cout <= cin:
                continue
            in_ms, out_ms = _snap_segment(field, cin, cout, params, clip, anc.AFF_ACTION)
            vlm = _vlm_quality_score(quality_events, cin, cout)
            score = _beat_score([anchor], vlm, territory_mult=1.0)
            if score < _MIN_BEAT_SALIENCE:
                continue
            out.append(HeroCut(
                hero_id=f"{clip.file_id[:8]}:act{ci}{pi}",
                file_id=clip.file_id,
                modality=anc.AFF_ACTION,
                label=(label_base + suffix)[:200],
                src_in_ms=in_ms,
                src_out_ms=out_ms,
                speaker=anchor.actor,
                affordances=[anc.AFF_ACTION],
                score=score,
            ))
    return out


def _cluster_anchors(anchors: List[anc.Anchor], gap_ms: int) -> List[List[anc.Anchor]]:
    """Cluster same-affordance, same-actor anchors whose inter-gap is shorter
    than ``gap_ms`` -- the SAME granularity machine as speech, generalized to any
    anchor. Low energy merges adjacent beats into one moment; ``gap_ms == 0``
    keeps every beat separate."""
    items = sorted(anchors, key=lambda a: a.start_ms)
    clusters: List[List[anc.Anchor]] = []
    for a in items:
        if clusters:
            prev = clusters[-1][-1]
            gap = a.start_ms - prev.end_ms
            if a.actor == prev.actor and gap < gap_ms:
                clusters[-1].append(a)
                continue
        clusters.append([a])
    return clusters


def _beat_score(
    members: List[anc.Anchor], vlm: Optional[float], *, territory_mult: float = 1.0,
) -> float:
    """Rank for a non-speech segment, with optional territory demotion."""
    base = max(m.salience for m in members)
    if vlm is not None:
        base = 0.7 * base + 0.3 * vlm
    w = _AFFORDANCE_WEIGHT.get(members[0].affordance, 0.7)
    return max(0.0, min(1.0, base * w * territory_mult))


def _core_inset(core_in: int, core_out: int, peak: int,
                target: Optional[int]) -> Tuple[int, int]:
    """Inset an overlay span (b-roll / reaction) toward its peak to ``target`` ms
    (the energy band's handle length = negative padding). ``target`` None / span
    already shorter -> keep the full shot. Only ever shrinks; the fused-seam snap
    then cleans the inset edges."""
    if not target or core_out - core_in <= target:
        return core_in, core_out
    half = target // 2
    center = max(core_in, min(peak, core_out))
    ci = max(core_in, center - half)
    co = min(core_out, ci + target)
    ci = max(core_in, co - target)  # rebalance if clipped at the tail
    return ci, co


def _beat_segments(clip: _ClipInputs, field: Optional[fseams.FusedField],
                   params: EnergyParams, anchors: List[anc.Anchor]) -> List[HeroCut]:
    """NON-speech anchors as segments -- per-affordance energy semantics."""
    if field is None and not clip.motion:
        return []
    perception = clip.perception or {}
    speaking = perception.get("speaking") or []
    quality_events = perception.get("take_quality_events") or []
    out: List[HeroCut] = []
    by_aff: Dict[str, List[anc.Anchor]] = {}
    for a in anchors:
        if a.affordance == anc.AFF_SPEECH:
            continue
        by_aff.setdefault(a.affordance, []).append(a)

    for aff, raw in by_aff.items():
        if aff == anc.AFF_ACTION:
            group = raw
            out.extend(_action_segments(group, clip, field, params, quality_events))
            continue
        group = _prep_overlay_group(raw, aff, params, speaking)
        merge_gap = _merge_gap_for_aff(params, aff)
        if aff == anc.AFF_INSERT:
            merge_gap = 2000 if params.insert_collapse_graphics else 0
        elif aff == anc.AFF_REACTION and any(m.kind == "audio_event" for m in group):
            merge_gap = params.audio_merge_gap_ms
        for ci, members in enumerate(_cluster_anchors(group, merge_gap)):
            core_in = min(m.start_ms for m in members)
            core_out = max(m.end_ms for m in members)
            best = max(members, key=lambda m: m.salience)
            if aff == anc.AFF_BROLL:
                # Energy-aware handle: the VLM hands us the full end-to-end shot;
                # inset toward the peak to the band's core target (Broad = full).
                core_in, core_out = _core_inset(
                    core_in, core_out, best.ts_ms, params.broll_core_ms)
            elif aff == anc.AFF_REACTION:
                # Same negative-padding mechanism: trim the expression toward its
                # peak as energy rises (Broad = full span).
                core_in, core_out = _core_inset(
                    core_in, core_out, best.ts_ms, params.reaction_core_ms)
            elif aff == anc.AFF_INSERT:
                # Onset-anchored, so the peak is the start -> this trims the TAIL
                # from the reveal as energy rises (Broad = full onset handle).
                core_in, core_out = _core_inset(
                    core_in, core_out, best.ts_ms, params.insert_core_ms)
            in_ms, out_ms = _snap_segment(field, core_in, core_out, params, clip, aff)
            t_mult = terr.territory_multiplier(
                best, speaking=speaking, strict=params.territory_strict)
            vlm = _vlm_quality_score(quality_events, core_in, core_out)
            score = _beat_score(members, vlm, territory_mult=t_mult)
            if score < _MIN_BEAT_SALIENCE:
                continue
            label = (best.text or aff).strip()
            out.append(HeroCut(
                hero_id=f"{clip.file_id[:8]}:{aff[:3]}{ci}",
                file_id=clip.file_id,
                modality=aff,
                label=label[:200],
                src_in_ms=in_ms,
                src_out_ms=out_ms,
                speaker=best.actor,
                affordances=[aff],
                score=score,
            ))
    return out


def _action_candidates(clip: _ClipInputs, params: EnergyParams,
                       field: Optional[fseams.FusedField]) -> List[HeroCut]:
    """DEPRECATED shim: action/visual now flow through ``_beat_segments`` over the
    anchor layer. Kept as a thin wrapper so older callers/tests still work."""
    anchors = anc.gather_anchors(
        duration_ms=clip.duration_ms,
        perception=clip.perception, motion=clip.motion)
    action = [a for a in anchors if a.affordance in (anc.AFF_ACTION, anc.AFF_BROLL, anc.AFF_INSERT)]
    return _beat_segments(clip, field, params, action)


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

def build_hero_cuts(
    file_ids: List[str], energy: float = 0.5,
    affordances: Optional[List[str]] = None,
    recommendation_verdict: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """The ranked, universal segment feed for a set of clips.

    Every editable moment -- speech, action, reaction, b-roll hold, insert -- is
    one ``segment`` built over the ANCHOR layer (``l3.anchors``): cluster anchors
    by the energy dial, then snap boundaries through the fused seam field *around
    the core* so the words / impact / expression are never clipped.

    `energy` (0..1) is the single deterministic dial (granularity + tightness).
    `affordances` is an optional FILTER over the one feed -- the way every edit
    style (action reel, podcast A-roll, B-roll cutaways) is served without a
    separate pipeline. Returns hero dicts sorted best-first.
    """
    if not file_ids:
        return []
    params = energy_to_params(energy)

    inputs = _load_inputs(file_ids)
    sources = ss.load_sources(file_ids)

    heroes: List[HeroCut] = []
    for fid, clip in inputs.items():
        field = _build_field(clip, params.energy)
        anchors = anc.gather_anchors(
            duration_ms=clip.duration_ms, dialogue=clip.dialogue,
            perception=clip.perception, motion=clip.motion, audio=clip.audio)
        sp = _speech_candidates(clip, sources.get(fid), field, params)
        beats = _beat_segments(clip, field, params, anchors)
        heroes.extend(sp)
        heroes.extend(beats)
        if params.fuse_moments:
            action = [b for b in beats if b.modality == anc.AFF_ACTION]
            heroes.extend(_combined_candidates(clip, sp, action))

    heroes = _stack_takes(heroes, file_ids)
    _apply_recommendations(heroes, file_ids, recommendation_verdict)
    if affordances:
        want = set(affordances)
        heroes = [h for h in heroes if want & set(h.affordances or [h.modality])]
    heroes.sort(key=lambda h: h.score, reverse=True)
    return [h.to_dict() for h in heroes]


def _apply_recommendations(
    heroes: List[HeroCut], file_ids: List[str],
    verdict: Optional[Dict[str, Dict[str, Any]]] = None,
) -> None:
    """Tag each cut `recommended` from the energy-independent LLM sentence
    verdict, using the "contains a keeper" rule: a speech/moment cut is
    recommended if it spans at least one kept sentence (an unseen sentence
    defaults to kept). Non-speech cuts are recommended by default (they're
    already sparse-curated by L2 + the energy salience floors).

    The verdict can be passed in (the API fetches it once, with a readiness
    flag); otherwise it's fetched here (non-blocking). Fail-open: an empty
    verdict leaves everything recommended, so the pool is never hidden.
    """
    if verdict is None:
        verdict = recommend.get_recommendation_map(file_ids)
    if not verdict:
        return

    for h in heroes:
        if not h.member_seg_ids:
            continue  # non-speech cut -> stays recommended by default
        kept = any(verdict.get(sid, {}).get("keep", True) for sid in h.member_seg_ids)
        h.recommended = kept
        if not kept:
            # Surface the first available drop reason for the full-feed view.
            for sid in h.member_seg_ids:
                r = verdict.get(sid, {}).get("reason")
                if r:
                    h.recommend_reason = r
                    break
