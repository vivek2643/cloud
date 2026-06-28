"""
Hero-cuts assembly: the single ranked feed of "usable moments" for a project.

This is the V1 product surface. It is a deterministic, post-VLM assembly over
artifacts L1/L2/L3 already produced -- no new model call:

  * SPEECH heroes  come from the THOUGHT hierarchy (``l3.thought_segments``):
    one speaker's self-contained idea, with a zoom ladder (turn -> setup ->
    thought -> core -> punch). The energy dial picks the LEVEL: a whole turn at
    the bottom, the complete thought in the middle, the punchline clause at the
    top. (When a clip has no thoughts we fall back to the L1 sentence/topic
    units.)
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
from app.services.l3 import cast as cst
from app.services.l3 import vocab
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

# Bump whenever the per-file cut logic changes (energy params, candidate
# builders, scoring) so the precompute cache invalidates and recomputes.
# v2: speech bands now cut from the THOUGHT hierarchy (l3.thought_segments)
# instead of the L1 sentence/topic tiers.
# v3: cuts are facet records -- each carries its owned zoom ladder + people
# (cast-resolved speaker / on-camera / region) + delivery-quality facet; the
# off-frame interviewer is flagged, not dropped.
# v4: moments form ONLY from VLM interaction windows (the blind 1.5s proximity
# union is gone) and are take-grouped by the speech they involve.
# v5: new CAPTURE layer (behavior cuts from the VLM event timeline, continuity-
# grouped; held listening shots synthesized from the speaking-turn inverse) and
# new REPRESENTATION (the synthetic "moment" entity is retired -- a cut carries
# every affordance it serves and links alternate framings as `coverage`).
# v6: closed 5-affordance vocabulary (behavior->action, listening->reaction; see
# vocab.py). FLAT model -- the spine/fold/coverage hierarchy is gone; every cut
# is first-class. Connections come from the VLM's TYPED relation graph mapped
# onto cuts (`relations`); a "moment" is a connected cluster (`moment_id`).
# v7: SYMMETRIC padding -- positive breathing room below the Balanced pivot
# (now for every affordance, incl. action), the negative core inset only above
# it; reactions/b-roll/inserts keep their full natural span through Balanced.
PARAMS_VERSION = 7

# The five canonical product energy LEVELS = the band centers (Broad .. Sharp).
# Hero cuts are precomputed at exactly these after L2; any requested energy
# snaps to the nearest band, so a level is always served from cache.
BAND_ENERGIES: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)


def band_energy(band: int) -> float:
    """Representative energy for a band index (0..4)."""
    return BAND_ENERGIES[max(0, min(band, len(BAND_ENERGIES) - 1))]


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
    anc.AFF_REACTION: 0.88,   # reactions are first-class story beats, not filler
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

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HeroTake":
        return cls(
            file_id=d["file_id"], src_in_ms=int(d["src_in_ms"]),
            src_out_ms=int(d["src_out_ms"]), score=float(d.get("score", 0.0)),
        )


@dataclass
class Rung:
    """One zoom level on a cut's intrinsic ladder.

    A rung is "one-or-more spans": a single span is a plain zoom; a single
    WIDER span is a coarsen/merge (it covers grouped neighbors); SEVERAL spans
    are a split / jump-cut (the gaps between them are excised, exactly like
    ``HeroCut.keep_spans``). The cut OWNS its ladder -- the levels come from its
    own anchor/unit, never reconstructed by matching across separate passes.
    """
    level: str                              # broad|calm|balanced|tight|sharp (or modality-native)
    spans: List[Tuple[int, int]]            # >=1 (in_ms, out_ms); gaps = jump-cuts
    text: str = ""
    score: float = 0.0

    def in_ms(self) -> int:
        return min(a for a, _ in self.spans) if self.spans else 0

    def out_ms(self) -> int:
        return max(b for _, b in self.spans) if self.spans else 0

    def play_ms(self) -> int:
        """On-screen duration once excised gaps are removed (kept spans only)."""
        return sum(max(0, b - a) for a, b in self.spans)

    def keep_spans(self) -> Optional[List[Tuple[int, int]]]:
        """The jump-cut edit-list when this rung is a split (>1 span), else None
        so a single-span rung plays contiguously."""
        return list(self.spans) if len(self.spans) > 1 else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "level": self.level,
            "spans": [{"in_ms": a, "out_ms": b} for a, b in self.spans],
            "in_ms": self.in_ms(),
            "out_ms": self.out_ms(),
            "play_ms": self.play_ms(),
            "text": self.text,
            "score": round(self.score, 3),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Rung":
        spans = [(int(s["in_ms"]), int(s["out_ms"])) for s in (d.get("spans") or [])]
        if not spans:
            spans = [(int(d.get("in_ms", 0)), int(d.get("out_ms", 0)))]
        return cls(level=str(d.get("level") or ""), spans=spans,
                   text=str(d.get("text") or ""), score=float(d.get("score", 0.0)))


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
    # Internal jump-cut edit-list: the spoken runs to KEEP after progressive
    # breath removal (Sharp band). None = play the span contiguously. Each pair
    # is (in_ms, out_ms) inside [src_in_ms, src_out_ms]; the gaps between them
    # are the excised breaths.
    keep_spans: Optional[List[Tuple[int, int]]] = None
    # --- Facets (the rich cut record; brain-facing, additive -- the UI ignores
    # what it doesn't read). Populated by the per-modality builders; empty here.
    # The cut's intrinsic zoom ladder (broad..sharp of ITS OWN content). The
    # flat src_in/out_ms above is the rung selected for the requested energy.
    ladder: List[Rung] = field(default_factory=list)
    # Who is in this cut: [{voice_speaker_id, person_id, role, on_camera, region,
    # av_link_confidence}] from the cast map. Never empty for resolved speech.
    people: List[dict] = field(default_factory=list)
    # Coarse framing for reframing/quality: {shot_size, angle, movement, region}.
    framing: Optional[dict] = None
    # Quality facet: {delivery, vlm, ...} -- deterministic + subjective scores.
    quality: Optional[dict] = None
    # Typed relation edges that touch this cut, mapped from the VLM relation
    # graph onto the built cuts. Each is {type, dir: 'out'|'in', other: hero_id,
    # note} where `type` is a vocab relation. The cut stays a first-class card on
    # its own tab; this is just how it CONNECTS to other cuts (a reaction
    # responds_to a line, b-roll illustrates a topic). Flat model -- nothing is
    # folded or dropped.
    relations: List[dict] = field(default_factory=list)
    # The connected-cluster this cut belongs to, when it is linked to other cuts
    # by a moment-forming relation. None for a standalone cut. The "Moments" view
    # is just the set of clusters -- a moment is a BUNDLE of first-class cuts, not
    # a separate entity. A podcast (mostly independent lines) yields few moments;
    # a reel (reaction <- line <- b-roll chains) yields rich ones.
    moment_id: Optional[str] = None
    # Narrative intent of this cut (vocab role: hook/answer/cta/establishing/
    # climax/listener), read from the VLM's per-beat role or synthesized for a
    # held listening shot. None for ordinary middle content. The brain uses it to
    # build structure (open on the hook, land on the answer), not just adjacency.
    role: Optional[str] = None

    def is_moment(self) -> bool:
        """True when this cut is part of a multi-cut moment cluster (it has a
        moment_id), i.e. the VLM stated a real relationship between it and other
        cuts. A lone talking-head line with no relations is not a moment."""
        return self.moment_id is not None

    def play_ms(self) -> int:
        """On-screen duration once breaths are excised (kept spans only)."""
        if not self.keep_spans:
            return self.src_out_ms - self.src_in_ms
        return sum(max(0, b - a) for a, b in self.keep_spans)

    def primitives(self) -> List[str]:
        """The capture primitive(s) this cut delivers (person/action/place/
        object/graphic/speech), derived from its affordance(s). The intrinsic
        'what was captured' substrate beneath the editor-facing affordance."""
        return vocab.primitives_for(self.affordances or [self.modality])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hero_id": self.hero_id,
            "file_id": self.file_id,
            "modality": self.modality,
            "label": self.label,
            "src_in_ms": self.src_in_ms,
            "src_out_ms": self.src_out_ms,
            "duration_ms": self.src_out_ms - self.src_in_ms,
            "play_ms": self.play_ms(),
            "keep_spans": ([{"in_ms": a, "out_ms": b} for a, b in self.keep_spans]
                           if self.keep_spans else None),
            "score": round(self.score, 3),
            "speaker": self.speaker,
            "flags": self.flags,
            "affordances": self.affordances or [self.modality],
            # Intrinsic capture substrate beneath the editor-facing affordances.
            "primitives": self.primitives(),
            "take_count": self.take_count,
            "alt_takes": [t.to_dict() for t in self.alt_takes],
            # Facets (additive; UI-safe). Only emitted when populated so the
            # feed/cache stay compact for cuts that don't carry them yet.
            "ladder": [r.to_dict() for r in self.ladder] if self.ladder else None,
            "people": self.people or None,
            "framing": self.framing,
            "quality": self.quality,
            "relations": self.relations or None,
            "moment_id": self.moment_id,
            "role": self.role,
            "is_moment": self.is_moment(),
        }

    def to_cache(self) -> Dict[str, Any]:
        """Serialize for the per-file precompute cache. Cached cuts are
        PRE-stacking, so alt_takes is empty."""
        return self.to_dict()

    @classmethod
    def from_cache(cls, d: Dict[str, Any]) -> "HeroCut":
        ks = d.get("keep_spans")
        keep = [(int(s["in_ms"]), int(s["out_ms"])) for s in ks] if ks else None
        return cls(
            hero_id=d["hero_id"], file_id=d["file_id"], modality=d["modality"],
            label=d.get("label", ""), src_in_ms=int(d["src_in_ms"]),
            src_out_ms=int(d["src_out_ms"]), score=float(d.get("score", 0.0)),
            speaker=d.get("speaker"), flags=list(d.get("flags") or []),
            affordances=list(d.get("affordances") or []),
            take_count=int(d.get("take_count", 1)),
            alt_takes=[HeroTake.from_dict(t) for t in (d.get("alt_takes") or [])],
            keep_spans=keep,
            ladder=[Rung.from_dict(r) for r in (d.get("ladder") or [])],
            people=list(d.get("people") or []),
            framing=d.get("framing"),
            quality=d.get("quality"),
            relations=list(d.get("relations") or []),
            moment_id=d.get("moment_id"),
            role=d.get("role"),
        )


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
    thoughts: list = field(default_factory=list)  # l3.thought_segments.Thought (speech source)
    cast: Optional["cst.ClipCast"] = None    # voice<->person map (built in _file_heroes)


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

    # Attach the thought primitives (the speech source). Cached per file and
    # lazily computed on a miss; imported here so the module import never pulls
    # in the LLM adapter. The legacy sentence/topic units stay as the fallback
    # when a clip has no thoughts (LLM unavailable + no dialogue segments).
    from app.services.l3 import thought_segments as _tseg
    for fid, clip in out.items():
        try:
            clip.thoughts = _tseg.get_thoughts(fid)
        except Exception:
            logger.exception("hero: thought load failed for %s", fid)
            clip.thoughts = []
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


def _usable_seg(seg: dict) -> bool:
    """A speech segment worth a card: on-camera and not a standalone backchannel
    ('mhm'/'yeah' only survive bridged inside a topic, never as their own clip)."""
    if _lexicon_offcamera(seg):
        return False
    if "backchannel" in (seg.get("flags") or []):
        return False
    return True


def _unit_from_seg(clip: _ClipInputs, seg: dict) -> dict:
    """One L1 segment (sentence or topic) -> a speech unit."""
    return {
        "in": int(seg.get("src_in_ms", 0)),
        "out": int(seg.get("src_out_ms", 0)),
        "raw_in": int(seg.get("raw_in_ms", seg.get("src_in_ms", 0))),
        "raw_out": int(seg.get("raw_out_ms", seg.get("src_out_ms", 0))),
        "text": (seg.get("text") or "").strip(),
        "speaker": seg.get("speaker"),
        "flags": [f for f in (seg.get("flags") or []) if f in ("noisy", "overlap")],
    }


def _merge_topics(topics: List[dict], gap_ms: int) -> List[List[dict]]:
    """Merge consecutive SAME-SPEAKER topics whose silence gap is shorter than
    ``gap_ms`` into one block (the coarse zoom-out levels). Speaker change always
    breaks a block -- we never fuse two people into one clip."""
    tops = sorted(topics, key=lambda s: int(s.get("src_in_ms", 0)))
    blocks: List[List[dict]] = []
    for t in tops:
        if blocks and gap_ms > 0:
            prev = blocks[-1][-1]
            gap = int(t.get("src_in_ms", 0)) - int(prev.get("src_out_ms", 0))
            if t.get("speaker") == prev.get("speaker") and gap < gap_ms:
                blocks[-1].append(t)
                continue
        blocks.append([t])
    return blocks


def _block_unit(clip: _ClipInputs, group: List[dict]) -> dict:
    """A merged run of topics -> one speech unit (block)."""
    text = " ".join((g.get("text") or "").strip() for g in group).strip()
    return {
        "in": min(int(g.get("src_in_ms", 0)) for g in group),
        "out": max(int(g.get("src_out_ms", 0)) for g in group),
        "raw_in": min(int(g.get("raw_in_ms", g.get("src_in_ms", 0))) for g in group),
        "raw_out": max(int(g.get("raw_out_ms", g.get("src_out_ms", 0))) for g in group),
        "text": text,
        "speaker": group[0].get("speaker"),
        "flags": list(dict.fromkeys(
            f for g in group for f in (g.get("flags") or []) if f in ("noisy", "overlap"))),
    }


def _legacy_speech_units(clip: _ClipInputs, params: EnergyParams,
                         sentences: List[dict], topics: List[dict]) -> List[dict]:
    """Fallback speech units from the L1 sentence/topic hierarchy, used only when
    a clip has no thoughts. The energy level maps onto the old tiers: a thought
    is roughly a topic, so turn->block(merged topics), setup/thought->topic,
    core/punch->sentence. Each level has clean, language-bounded edges."""
    tier = params.speech_unit
    if tier in ("core", "punch"):
        segs = [s for s in sentences if _usable_seg(s)] or [t for t in topics if _usable_seg(t)]
        return [_unit_from_seg(clip, s) for s in segs]
    # Thought/setup map onto the topic tier; turn merges topics into a block.
    tops = [t for t in topics if _usable_seg(t)] or [s for s in sentences if _usable_seg(s)]
    if tier in ("setup", "thought"):
        return [_unit_from_seg(clip, t) for t in tops]
    return [_block_unit(clip, g) for g in _merge_topics(tops, params.speech_merge_gap_ms)]


# --- Thought-based speech units (the primary path) ------------------------

def _thought_start_ms(t) -> int:
    """The earliest ms of a thought INCLUDING its setup run-up."""
    return int(t.setup.raw_in_ms if t.setup else t.thought.raw_in_ms)


def _thought_text_with_setup(t) -> str:
    if t.setup and (t.setup.text or "").strip():
        return (t.setup.text + " " + t.thought.text).strip()
    return (t.thought.text or "").strip()


def _merge_thoughts(thoughts: list, gap_ms: int) -> List[list]:
    """Group consecutive SAME-SPEAKER thoughts whose gap is shorter than
    ``gap_ms`` into one turn (the Broad zoom-out). A speaker change always breaks
    a group -- we never fuse two people into one clip."""
    items = sorted(thoughts, key=_thought_start_ms)
    groups: List[list] = []
    for t in items:
        if groups and gap_ms > 0:
            prev = groups[-1][-1]
            if t.speaker == prev.speaker and _thought_start_ms(t) - int(prev.thought.raw_out_ms) < gap_ms:
                groups[-1].append(t)
                continue
        groups.append([t])
    return groups


def _speech_unit(in_ms: int, out_ms: int, text: str, speaker) -> Optional[dict]:
    text = (text or "").strip()
    if out_ms <= in_ms or not text:
        return None
    return {"in": int(in_ms), "out": int(out_ms), "raw_in": int(in_ms),
            "raw_out": int(out_ms), "text": text, "speaker": speaker, "flags": []}


def _band_thought_span(t, band: int) -> Tuple[int, int, str]:
    """(in_ms, out_ms, text) for one thought at a non-Broad band."""
    if band <= 1:        # Calm: the thought + the speaker's own run-up
        in_ms = t.setup.raw_in_ms if t.setup else t.thought.raw_in_ms
        return int(in_ms), int(t.thought.raw_out_ms), _thought_text_with_setup(t)
    if band == 2:        # Balanced: the complete thought proper
        return int(t.thought.raw_in_ms), int(t.thought.raw_out_ms), (t.thought.text or "")
    if band == 3:        # Tight: the core sentence
        return int(t.core.raw_in_ms), int(t.core.raw_out_ms), (t.core.text or "")
    return int(t.punch.raw_in_ms), int(t.punch.raw_out_ms), (t.punch.text or "")  # Sharp: punchline


# Level names for a speech cut's owned ladder (broad..sharp), aligned with the
# UI / footage-map band names. The cut OWNS these rungs -- the zoom levels of
# ITS content -- so the energy dial just selects one (see Phase 4 read path).
_SPEECH_LEVELS = ("broad", "calm", "balanced", "tight", "sharp")
# The turn (broad) rung merges consecutive same-speaker thoughts; defined at the
# Broad merge gap regardless of the band currently being emitted.
_TURN_MERGE_GAP_MS = energy_to_params(0.0).speech_merge_gap_ms


def _thought_ladder(t, turn_span: Tuple[int, int], turn_text: str) -> List[Rung]:
    """The intrinsic zoom ladder a single thought OWNS: broad(turn) -> calm
    (setup+thought) -> balanced(thought) -> tight(core) -> sharp(punch). Raw,
    language-bounded spans; the fused-seam snap + breath excision are applied to
    the rung the energy dial selects (the flat cut), not stored per rung yet."""
    calm_in = int(t.setup.raw_in_ms if t.setup else t.thought.raw_in_ms)
    s = float(getattr(t, "strength", 0.5))
    return [
        Rung("broad", [(int(turn_span[0]), int(turn_span[1]))], text=turn_text, score=s),
        Rung("calm", [(calm_in, int(t.thought.raw_out_ms))],
             text=_thought_text_with_setup(t), score=s),
        Rung("balanced", [(int(t.thought.raw_in_ms), int(t.thought.raw_out_ms))],
             text=(t.thought.text or ""), score=s),
        Rung("tight", [(int(t.core.raw_in_ms), int(t.core.raw_out_ms))],
             text=(t.core.text or ""), score=s),
        Rung("sharp", [(int(t.punch.raw_in_ms), int(t.punch.raw_out_ms))],
             text=(t.punch.text or ""), score=s),
    ]


def _turn_of(thoughts: list) -> Dict[int, Tuple[Tuple[int, int], str]]:
    """Map each thought (by identity) to its turn's (span, text) -- the merged
    run of consecutive same-speaker thoughts it belongs to (the broad rung)."""
    out: Dict[int, Tuple[Tuple[int, int], str]] = {}
    for group in _merge_thoughts(thoughts, _TURN_MERGE_GAP_MS):
        tin = min(_thought_start_ms(t) for t in group)
        tout = max(int(t.thought.raw_out_ms) for t in group)
        ttext = " ".join(_thought_text_with_setup(t) for t in group).strip()
        for t in group:
            out[id(t)] = ((tin, tout), ttext)
    return out


def _thought_speech_units(thoughts: list, params: EnergyParams) -> List[dict]:
    """Speech units for this energy band, built from the THOUGHT hierarchy --
    one speaker's self-contained idea zoomed to the band level (turn / setup /
    thought / core / punch). Edges are language-bounded by construction. Each
    unit carries the thought's full owned ``ladder`` so the cut keeps every zoom
    (the energy dial selected this band's rung for the flat span)."""
    band = params.band
    turn = _turn_of(thoughts)
    raw: List[Optional[dict]] = []
    if band == 0:        # Broad: merge consecutive same-speaker thoughts -> turn
        for group in _merge_thoughts(thoughts, params.speech_merge_gap_ms):
            in_ms = min(_thought_start_ms(t) for t in group)
            out_ms = max(int(t.thought.raw_out_ms) for t in group)
            text = " ".join(_thought_text_with_setup(t) for t in group).strip()
            u = _speech_unit(in_ms, out_ms, text, group[0].speaker)
            if u:
                u["ladder"] = [Rung("broad", [(in_ms, out_ms)], text=text,
                                    score=max(float(getattr(t, "strength", 0.5)) for t in group))]
                raw.append(u)
    else:
        for t in thoughts:
            in_ms, out_ms, text = _band_thought_span(t, band)
            u = _speech_unit(in_ms, out_ms, text, t.speaker)
            if u:
                turn_span, turn_text = turn.get(id(t), ((in_ms, out_ms), text))
                u["ladder"] = _thought_ladder(t, turn_span, turn_text)
                raw.append(u)
    return [u for u in raw if u]


def _breath_keep_spans(words: List[dict], lo: int, hi: int,
                       gap_ms: int) -> Optional[List[Tuple[int, int]]]:
    """Progressive breath removal: return the spoken runs to KEEP inside
    [lo, hi], excising every internal silent gap >= ``gap_ms`` (the breaths
    between words). The result is a jump-cut edit-list -- the same content,
    dead air deleted. Edges stay on the snapped boundaries (lo/hi); only
    *internal* breaths are cut. Returns None when there's nothing to excise
    (no qualifying gap), so the cut just plays contiguously."""
    if gap_ms <= 0:
        return None
    span = sorted(
        (w for w in words
         if int(w.get("start_ms", 0)) < hi and int(w.get("end_ms", 0)) > lo),
        key=lambda w: int(w.get("start_ms", 0)))
    if len(span) < 2:
        return None
    spans: List[Tuple[int, int]] = []
    seg_start = lo
    for prev, nxt in zip(span, span[1:]):
        if int(nxt.get("start_ms", 0)) - int(prev.get("end_ms", 0)) >= gap_ms:
            seg_end = min(hi, int(prev.get("end_ms", 0)))
            if seg_end > seg_start:
                spans.append((seg_start, seg_end))
            seg_start = max(lo, int(nxt.get("start_ms", 0)))
    if hi > seg_start:
        spans.append((seg_start, hi))
    return spans if len(spans) > 1 else None


def _speech_quality(metrics: Dict[str, Any], vlm: Optional[float],
                    on_camera: Optional[float]) -> Optional[dict]:
    """The cut's quality facet: deterministic delivery fluency + the VLM's
    subjective score + how on-camera the speaker is. None when nothing to say."""
    q: Dict[str, Any] = {}
    if metrics:
        q["delivery"] = round(ss.delivery_score(metrics), 3)
    if vlm is not None:
        q["vlm"] = round(vlm, 3)
    if on_camera is not None:
        q["on_camera"] = round(on_camera, 3)
    return q or None


def _make_speech_hero(
    clip: _ClipInputs, source: Optional[ss.SpanSource], quality_events: List[dict],
    uid: str, in_ms: int, out_ms: int, raw_in: int, raw_out: int,
    text: str, speaker: Optional[str], extra_flags: List[str],
    keep_spans: Optional[List[Tuple[int, int]]] = None,
    ladder: Optional[List[Rung]] = None,
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

    # Resolve WHO is speaking via the cast map: the right on-screen person (not
    # the raw diarization id), whether they're on camera, and their frame box.
    member = clip.cast.resolve(speaker) if clip.cast else None
    display_speaker = member.label() if member is not None else speaker
    people = [member.to_dict()] if member is not None else []
    on_cam = member.on_camera_ratio(in_ms, out_ms) if member is not None else None
    region = member.region if member is not None else None

    return HeroCut(
        hero_id=f"{clip.file_id[:8]}:{uid}",
        file_id=clip.file_id,
        modality=anc.AFF_SPEECH,
        label=text.strip(),
        src_in_ms=in_ms,
        src_out_ms=out_ms,
        score=_speech_score(metrics, vlm),
        speaker=display_speaker,
        flags=extra_flags,
        affordances=[anc.AFF_SPEECH],
        keep_spans=keep_spans,
        ladder=ladder or [],
        people=people,
        framing=_framing_facet(clip, in_ms, out_ms, region),
        quality=_speech_quality(metrics, vlm, on_cam),
    )


def _speech_candidates(
    clip: _ClipInputs, source: Optional[ss.SpanSource],
    field: Optional[fseams.FusedField], params: EnergyParams,
) -> List[HeroCut]:
    """Speech heroes selected from the THOUGHT hierarchy by energy band.

    The dial picks a LEVEL of the thought (not a silence threshold), so every
    band has clean, language-bounded edges by construction:

      * Broad     -> turn: merge consecutive same-speaker thoughts,
      * Calm      -> setup + thought: the idea plus the speaker's own run-up,
      * Balanced  -> thought: one complete idea (the pivot),
      * Tight     -> core: the one sentence that carries it,
      * Sharp     -> punch: the tightest landing clause (loses its breath later).

    When a clip has no thoughts (LLM unavailable AND no dialogue segments) we
    fall back to the legacy L1 sentence/topic units. Each unit's boundaries then
    flow through the fused seam field for the energy-scaled tightness (snap
    window + veto-bounded breathing room). Off-camera audio is dropped/edge-
    trimmed using the VLM's visible-speaking spans where logged.
    """
    perception = clip.perception or {}
    quality_events = perception.get("take_quality_events") or []
    speaking = perception.get("speaking") or []
    if clip.thoughts:
        units = _thought_speech_units(clip.thoughts, params)
    else:
        sentences = clip.dialogue.get("sentence") or []
        topics = clip.dialogue.get("topic") or []
        units = _legacy_speech_units(clip, params, sentences, topics)
    if not units:
        return []

    out: List[HeroCut] = []
    for ci, u in enumerate(units):
        in_ms, out_ms, text = u["in"], u["out"], u["text"]
        raw_in, raw_out, speaker, flags = u["raw_in"], u["raw_out"], u["speaker"], u["flags"]
        flags = list(flags)

        # Trim off-camera crew cues bleeding into the edges (VLM authority).
        if speaking and source is not None:
            tr = _trim_to_speaking(source.words, speaking, in_ms, out_ms)
            if tr:
                in_ms, out_ms, kw = tr
                text = " ".join((w.get("text") or "").strip() for w in kw).strip()

        # NEVER DISCARD: audio with no visible speaker is off-frame voice (an
        # off-camera interviewer's question, a voiceover) -- still usable, so we
        # FLAG it rather than drop it and let the cast/ranker decide. The crew
        # "go"/"action" cues are already edge-trimmed above; what survives here
        # is real off-screen speech the brain may well want to keep.
        if speaking and _speaking_coverage(speaking, in_ms, out_ms) < SPEAKING_COVERAGE_MIN:
            if "offscreen" not in flags:
                flags.append("offscreen")

        # Tightness: snap to fused seams + energy-scaled breathing room.
        in_ms, out_ms = _snap_fused(field, in_ms, out_ms, params, clip.duration_ms)

        # Sharp band: excise internal breaths -> jump-cut edit-list.
        keep_spans = None
        if params.speech_breath_gap_ms > 0 and source is not None:
            keep_spans = _breath_keep_spans(
                source.words, in_ms, out_ms, params.speech_breath_gap_ms)

        # Reconcile the rung the energy dial selected with the snapped/excised
        # flat output, so the cut's owned ladder agrees with what actually plays.
        ladder = list(u.get("ladder") or [])
        _reconcile_selected_rung(ladder, params.band, in_ms, out_ms, keep_spans, text)

        h = _make_speech_hero(clip, source, quality_events, f"sp{ci}",
                              in_ms, out_ms, raw_in, raw_out, text, speaker, flags,
                              keep_spans=keep_spans, ladder=ladder)
        if h:
            out.append(h)
    return out


def _reconcile_selected_rung(
    ladder: List[Rung], band: int, in_ms: int, out_ms: int,
    keep_spans: Optional[List[Tuple[int, int]]], text: str,
) -> None:
    """Update the rung the dial selected (this band) to the snapped + breath-
    excised spans that actually play, so the owned ladder stays truthful for the
    level being served. Other rungs keep their raw, language-bounded spans."""
    if not ladder or not (0 <= band < len(_SPEECH_LEVELS)):
        return
    level = _SPEECH_LEVELS[band]
    spans = list(keep_spans) if keep_spans else [(int(in_ms), int(out_ms))]
    for r in ladder:
        if r.level == level:
            r.spans = spans
            if text:
                r.text = text
            return


def _l2_id_spans(perception: Optional[dict]) -> Dict[str, Tuple[int, int]]:
    """Index every addressable L2 beat by its local id -> (start_ms, end_ms), so
    the VLM relation graph (which references those ids) can be mapped onto the
    built cuts purely by time. Covers events, content_units, cutaways, reactions
    -- the four id-bearing tracks a relation endpoint can name."""
    spans: Dict[str, Tuple[int, int]] = {}

    def put(_id, a, b):
        if _id and int(b) > int(a):
            spans[str(_id)] = (int(a), int(b))

    p = perception or {}
    for e in p.get("events") or []:
        put(e.get("id"), e.get("start_ms", 0), e.get("end_ms", 0))
    for u in p.get("content_units") or []:
        put(u.get("unit_id"), u.get("start_ms", 0), u.get("end_ms", 0))
    for c in p.get("cutaways") or []:
        put(c.get("id"), c.get("start_ms", 0), c.get("end_ms", 0))
    for r in p.get("reactions") or []:
        put(r.get("id"), r.get("start_ms", 0), r.get("end_ms", 0))
    return spans


def _role_spans(perception: Optional[dict]) -> List[Tuple[int, int, str]]:
    """Every (start, end, role) the VLM stated, across the tracks that carry a
    role. Only vocab roles are kept, so a stray free-text value is ignored."""
    out: List[Tuple[int, int, str]] = []
    p = perception or {}
    for track in ("events", "content_units", "cutaways", "reactions"):
        for x in p.get(track) or []:
            role = (x.get("role") or "").lower()
            if role in vocab.ROLE_SET:
                a, b = int(x.get("start_ms", 0)), int(x.get("end_ms", 0))
                if b > a:
                    out.append((a, b, role))
    return out


def _assign_roles(clip: _ClipInputs, cuts: List[HeroCut]) -> None:
    """Stamp each cut with its narrative role from the VLM's per-beat role,
    mapped by best time-overlap. A synthesized held-listening shot has no L2 row,
    so it gets the 'listener' role from its flag. Leaves role None for ordinary
    middle content (most lines) -- the brain only needs the structural beats."""
    spans = _role_spans(clip.perception)
    for h in cuts:
        best, best_ov = None, 0
        for a, b, role in spans:
            ov = _overlap_ms(h.src_in_ms, h.src_out_ms, a, b)
            if ov > best_ov:
                best, best_ov = role, ov
        if best is not None:
            h.role = best
        elif "listening" in (h.flags or []):
            h.role = vocab.ROLE_LISTENER


def _cut_for_span(cuts: List[HeroCut], a: int, b: int) -> Optional[HeroCut]:
    """The built cut that best CORRESPONDS to the L2 span [a, b], by intersection
    over union -- so a tight b-roll cut wins its own cutaway over a long speech
    cut that merely contains it (raw overlap would tie and pick by order)."""
    best, best_iou = None, 0.0
    for h in cuts:
        inter = _overlap_ms(h.src_in_ms, h.src_out_ms, a, b)
        if inter <= 0:
            continue
        union = max(h.src_out_ms, b) - min(h.src_in_ms, a)
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best, best_iou = h, iou
    return best


def _annotate_moments(clip: _ClipInputs, cuts: List[HeroCut]) -> List[HeroCut]:
    """Wire the VLM's typed relation graph onto the built cuts -- the flat model
    that replaces the old spine/fold/coverage hierarchy.

    Every cut stays a FIRST-CLASS card: nothing is dropped or folded. We resolve
    each L2 relation's endpoints (event/unit/cutaway/reaction ids) to the cut
    that best covers them in time, then:
      * record the typed edge on BOTH cuts (directional: 'out' on the source,
        'in' on the target) so the brain reads real connections, not time guesses;
      * union cuts joined by a MOMENT-forming relation (responds_to / illustrates
        / leads_into / same_instant / answers) into a connected cluster and stamp
        each member with a shared ``moment_id``.
    A 'moment' is therefore just a connected bundle of independent cuts (a line +
    its reaction + the b-roll that illustrates it). ``take_of`` edges are recorded
    but do NOT form a moment -- alternates are one slot, handled by take-stacking.
    Degrades to a no-op when the perception predates the relation schema."""
    rels = (clip.perception or {}).get("relations") or []
    if not rels or len(cuts) < 2:
        return cuts
    spans = _l2_id_spans(clip.perception)
    by_id = {c.hero_id: c for c in cuts}

    # Union-find over moment-forming edges.
    parent = {c.hero_id: c.hero_id for c in cuts}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in rels:
        t = (r.get("type") or "").lower()
        if t not in vocab.RELATION_SET:
            continue
        fa = spans.get(str(r.get("from_id")))
        ta = spans.get(str(r.get("to_id")))
        if not fa or not ta:
            continue
        fc = _cut_for_span(cuts, *fa)
        tc = _cut_for_span(cuts, *ta)
        if fc is None or tc is None or fc.hero_id == tc.hero_id:
            continue
        note = (r.get("note") or "").strip() or None
        fc.relations.append({"type": t, "dir": "out", "other": tc.hero_id, "note": note})
        tc.relations.append({"type": t, "dir": "in", "other": fc.hero_id, "note": note})
        if t in vocab.MOMENT_RELATIONS:
            union(fc.hero_id, tc.hero_id)

    # Stamp a moment_id on every cut that ended up in a cluster of >1.
    members: Dict[str, List[str]] = {}
    for hid in parent:
        members.setdefault(find(hid), []).append(hid)
    fid8 = clip.file_id[:8]
    for i, (root, group) in enumerate(sorted(members.items())):
        if len(group) < 2:
            continue
        mid = f"{fid8}:mo{i:02d}"
        for hid in group:
            by_id[hid].moment_id = mid
    return cuts


def _merge_gap_for_aff(params: EnergyParams, aff: str) -> int:
    if aff == anc.AFF_ACTION:
        return params.action_merge_gap_ms
    if aff == anc.AFF_REACTION:
        return params.reaction_merge_gap_ms
    if aff == anc.AFF_BROLL:
        return params.broll_merge_gap_ms
    if aff == anc.AFF_INSERT:
        return 0
    return 0


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
        # Symmetric breathing room for EVERY affordance below the Balanced pivot
        # (pad is 0 at Balanced+, so this only adds air at Broad/Calm). For action
        # the pad is veto-bounded, so it can only fill calm footage around the
        # beat -- it stops dead at the next motion impact / camera move.
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
    # A performance (song/dance/bit) keeps its full duration -- never trimmed.
    if anchor.kind == "performance":
        return [(anchor.start_ms, anchor.end_ms, "")]
    impact = anchor.ts_ms
    core = params.action_core_ms
    if params.action_split_at_impact:
        onset = _motion_onset_ms(motion, anchor.start_ms, anchor.end_ms)
        min_windup = _action_min_windup_ms(anchor.end_ms - anchor.start_ms)
        if impact >= onset + min_windup and onset < impact < anchor.end_ms:
            # Payoff is impact-forward and core-capped; windup stays the run-up.
            pin, pout = _core_inset(impact, anchor.end_ms, impact, core, lead_frac=_ACTION_LEAD)
            return [
                (onset, impact, " · windup"),
                (pin, pout, " · payoff"),
            ]
    cin, cout = _action_core(anchor, motion, params.action_anchor_mode)
    # Negative padding: impact-forward core cap (Broad/Calm core None = full).
    cin, cout = _core_inset(cin, cout, impact, core, lead_frac=_ACTION_LEAD)
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
            ladder_base = [Rung("broad", [(cin, cout)], text=label_base,
                                score=float(anchor.salience))]
        else:
            anchor = members[0]
            pieces = _action_pieces(anchor, motion, params, field, clip)
            label_base = (anchor.text or "action").strip()
            ladder_base = _beat_ladder(anchor, motion, anc.AFF_ACTION)
        for pi, (cin, cout, suffix) in enumerate(pieces):
            if cout <= cin:
                continue
            in_ms, out_ms = _snap_segment(field, cin, cout, params, clip, anc.AFF_ACTION)
            vlm = _vlm_quality_score(quality_events, cin, cout)
            score = _beat_score([anchor], vlm, territory_mult=1.0)
            if score < _MIN_BEAT_SALIENCE:
                continue
            label = (label_base + suffix)[:200]
            ladder = _copy_ladder(ladder_base)
            _reconcile_selected_rung(ladder, params.band, in_ms, out_ms, None, label)
            out.append(HeroCut(
                hero_id=f"{clip.file_id[:8]}:act{ci}{pi}",
                file_id=clip.file_id,
                modality=anc.AFF_ACTION,
                label=label,
                src_in_ms=in_ms,
                src_out_ms=out_ms,
                speaker=anchor.actor,
                affordances=[anc.AFF_ACTION],
                score=score,
                ladder=ladder,
                people=_beat_people(anchor),
                framing=_framing_facet(clip, in_ms, out_ms, anchor.region),
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


# How much of the negative-padding window sits BEFORE the peak (the rest is the
# payoff / settle after it). The peak is rarely centered, so each affordance
# anchors its core differently: an onset insert keeps only the tail, an action
# stays impact-forward, a reaction keeps the apex + settle, a b-roll move keeps
# the run-in to its arrival.
_INSERT_LEAD = 0.0        # peak = onset -> keep the reveal + tail
_ACTION_LEAD = 0.0        # peak = impact -> impact-forward, drop the windup
_REACTION_LEAD = 0.3      # peak = apex -> keep mostly the apex + settle
_BROLL_HOLD_LEAD = 0.5    # uniform hold -> symmetric is fine
_BROLL_MOVE_LEAD = 0.8    # peak = arrival -> keep the run-in to it


def _core_inset(core_in: int, core_out: int, peak: int,
                target: Optional[int], *, lead_frac: float = 0.5) -> Tuple[int, int]:
    """Inset a span toward ``peak`` to ``target`` ms (the energy band's handle
    length = negative padding). Keep ``lead_frac`` of the window BEFORE the peak
    and the remainder after (the payoff / settle), clamped to [core_in, core_out].
    ``target`` None / span already shorter -> unchanged. Only ever shrinks; the
    fused-seam snap then cleans the inset edges. ``lead_frac=0.5`` is centered."""
    if not target or core_out - core_in <= target:
        return core_in, core_out
    peak = max(core_in, min(peak, core_out))
    lead = int(round(target * lead_frac))
    ci = max(core_in, peak - lead)
    co = min(core_out, ci + target)
    ci = max(core_in, co - target)  # rebalance if clipped at the tail
    return ci, co


# --- Facets for non-speech cuts (framing / people / owned ladder) ------------

def _camera_framing(perception: Optional[dict], in_ms: int, out_ms: int) -> dict:
    """The dominant camera framing over a window from the VLM's camera_craft
    timeline -- shot size / angle / movement / what the framing favors."""
    best, best_ov = None, 0
    for c in (perception or {}).get("camera_craft") or []:
        ov = _overlap_ms(int(c.get("start_ms", 0)), int(c.get("end_ms", 0)), in_ms, out_ms)
        if ov > best_ov:
            best, best_ov = c, ov
    if best is None:
        return {}
    out: dict = {}
    for k in ("shot_size", "angle", "movement", "subject_focus"):
        v = best.get(k)
        if v is not None:
            out[k] = v
    return out


def _framing_facet(clip: _ClipInputs, in_ms: int, out_ms: int,
                   region: Optional[dict] = None) -> Optional[dict]:
    """Framing facet for any cut: camera craft + the subject's frame box (for
    reframing). None when there's nothing to record."""
    f = _camera_framing(clip.perception, in_ms, out_ms)
    if region:
        f = dict(f)
        f["region"] = region
    return f or None


def _beat_people(anchor: anc.Anchor) -> List[dict]:
    """The actor facet for a non-speech cut (the person performing the beat)."""
    if not anchor.actor:
        return []
    p: dict = {"person_id": anchor.actor, "on_camera": True}
    if anchor.region:
        p["region"] = anchor.region
    return [p]


def _overlay_core(aff: str, params: EnergyParams,
                  kind: Optional[str] = None) -> Tuple[Optional[int], float]:
    """(target handle length, lead fraction) for an overlay affordance at a band."""
    if aff == anc.AFF_BROLL:
        return params.broll_core_ms, (_BROLL_MOVE_LEAD if kind == "move" else _BROLL_HOLD_LEAD)
    if aff == anc.AFF_REACTION:
        return params.reaction_core_ms, _REACTION_LEAD
    if aff == anc.AFF_INSERT:
        return params.insert_core_ms, _INSERT_LEAD
    return None, 0.5


def _beat_ladder(anchor: anc.Anchor, motion: Optional[dict], aff: str) -> List[Rung]:
    """The intrinsic zoom ladder a non-speech beat OWNS: broad/calm keep the full
    beat, balanced..sharp inset toward the peak (impact / apex / arrival / onset)
    to the per-band handle length -- the same negative-padding the flat cut uses,
    computed once per rung. Raw (pre-snap); the selected rung is reconciled to
    the snapped flat span at emit time."""
    rungs: List[Rung] = []
    for band in range(len(BAND_ENERGIES)):
        p = energy_to_params(band_energy(band))
        if aff == anc.AFF_ACTION:
            cin, cout = _action_core(anchor, motion, p.action_anchor_mode)
            cin, cout = _core_inset(cin, cout, anchor.ts_ms, p.action_core_ms, lead_frac=_ACTION_LEAD)
        else:
            core_ms, lead = _overlay_core(aff, p, anchor.kind)
            cin, cout = _core_inset(anchor.start_ms, anchor.end_ms, anchor.ts_ms,
                                    core_ms, lead_frac=lead)
        if cout <= cin:
            cin, cout = anchor.start_ms, anchor.end_ms
        rungs.append(Rung(_SPEECH_LEVELS[band], [(int(cin), int(cout))],
                          text=(anchor.text or aff), score=float(anchor.salience)))
    return rungs


def _copy_ladder(rungs: List[Rung]) -> List[Rung]:
    return [Rung(r.level, list(r.spans), r.text, r.score) for r in rungs]


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
                # Energy-aware handle: the VLM hands us the full end-to-end shot.
                # A held composition is uniform (center is fine); a MOVE pays off
                # at its arrival, so keep the run-in to the peak instead.
                lead = _BROLL_MOVE_LEAD if best.kind == "move" else _BROLL_HOLD_LEAD
                core_in, core_out = _core_inset(
                    core_in, core_out, best.ts_ms, params.broll_core_ms, lead_frac=lead)
            elif aff == anc.AFF_REACTION:
                # Trim toward the apex (peak_ms when the VLM gives one, else the
                # midpoint biased late) so we keep the apex + settle, not build-up.
                core_in, core_out = _core_inset(
                    core_in, core_out, best.ts_ms, params.reaction_core_ms,
                    lead_frac=_REACTION_LEAD)
            elif aff == anc.AFF_INSERT:
                # Onset-anchored (peak = start) -> trims the TAIL from the reveal.
                core_in, core_out = _core_inset(
                    core_in, core_out, best.ts_ms, params.insert_core_ms,
                    lead_frac=_INSERT_LEAD)
            in_ms, out_ms = _snap_segment(field, core_in, core_out, params, clip, aff)
            t_mult = terr.territory_multiplier(
                best, speaking=speaking, strict=params.territory_strict)
            vlm = _vlm_quality_score(quality_events, core_in, core_out)
            score = _beat_score(members, vlm, territory_mult=t_mult)
            if score < _MIN_BEAT_SALIENCE:
                continue
            label = (best.text or aff).strip()[:200]
            if len(members) > 1:
                ladder = [Rung("broad", [(in_ms, out_ms)], text=label, score=float(best.salience))]
            else:
                ladder = _beat_ladder(best, clip.motion, aff)
                _reconcile_selected_rung(ladder, params.band, in_ms, out_ms, None, label)
            out.append(HeroCut(
                hero_id=f"{clip.file_id[:8]}:{aff[:3]}{ci}",
                file_id=clip.file_id,
                modality=aff,
                label=label,
                src_in_ms=in_ms,
                src_out_ms=out_ms,
                speaker=best.actor,
                flags=list(best.flags or []),
                affordances=[aff],
                score=score,
                ladder=ladder,
                people=_beat_people(best),
                framing=_framing_facet(clip, in_ms, out_ms, best.region),
            ))
    return out


# --------------------------------------------------------------------------
# Take stacking: collapse repeats into one hero, best in front
# --------------------------------------------------------------------------

def _overlap_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _take_rank(h: HeroCut) -> Tuple[float, float, float]:
    """Deterministic best-take order for the front of a stack: prefer the take
    where the speaker is ON CAMERA, then the cleaner DELIVERY (fluency: fewer
    fillers / no stumble), then the base score. Unknown on-camera/delivery sit at
    a neutral 0.5 so they neither win nor lose by default. This is the objective
    half of take selection -- exactly what the cast map + delivery score were
    built to feed."""
    q = h.quality or {}
    on_cam = q.get("on_camera")
    deliv = q.get("delivery")
    return (
        float(on_cam) if on_cam is not None else 0.5,
        float(deliv) if deliv is not None else 0.5,
        float(h.score),
    )


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

    Take-stacking is orthogonal to the relation graph: a speech cut that is part
    of a moment cluster still take-groups on its SPEECH here, exactly like a plain
    line. Pure action/visual/reaction heroes carry no groupable text, so they pass
    through untouched (their `take_of` relations, if any, stay as edges).
    """
    speech = [h for h in heroes if h.modality == "speech"]
    others = [h for h in heroes if h.modality != "speech"]

    groups = build_take_groups(file_ids)
    if not speech:
        return others

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
    for group in groups:
        # Pair every attempt with the hero it overlaps (or None -> raw span).
        members = [(hero_for(a), a) for a in group.attempts]
        # The front of the stack is the best-scored hero among the deliveries.
        candidates = [h for h, _ in members if h is not None and h.hero_id not in consumed]
        if not candidates:
            continue
        front = max(candidates, key=_take_rank)
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
    other): high text ratio, similar length, and long enough to be meaningful.

    ``quick_ratio()`` is a guaranteed upper bound on ``ratio()`` and far cheaper,
    so it prunes the bulk of non-matching pairs before the full edit-distance --
    same result, but it keeps the O(n^2) consolidation pass fast on big feeds."""
    ta, tb = a.split(), b.split()
    if len(ta) < _MERGE_MIN_TOKENS or len(tb) < _MERGE_MIN_TOKENS:
        return False
    if min(len(ta), len(tb)) / max(len(ta), len(tb)) < _MERGE_LEN_RATIO:
        return False
    sm = SequenceMatcher(None, a, b)
    if sm.quick_ratio() < _MERGE_RATIO:
        return False
    return sm.ratio() >= _MERGE_RATIO


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
        front = max(g, key=_take_rank)
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
        heroes.extend(_file_heroes(clip, sources.get(fid), params))

    return _assemble(heroes, file_ids, affordances)


def _file_heroes(
    clip: _ClipInputs, source: Optional[ss.SpanSource], params: EnergyParams,
) -> List[HeroCut]:
    """All hero cuts for ONE clip at one energy -- the heavy, per-file work
    (fused field, anchors, speech/beat/combined candidates). Pure given the
    clip's stored artifacts, so its output is what the precompute cache stores
    (one entry per file per energy band). No cross-file stacking here."""
    field = _build_field(clip, params.energy)
    # Resolve the cast (voice<->person, on/off camera, frame box) once per clip,
    # so every speech cut carries the right speaker + framing facets.
    if clip.cast is None:
        clip.cast = cst.build_cast(clip.perception, source.words if source else [])
    anchors = anc.gather_anchors(
        duration_ms=clip.duration_ms, dialogue=clip.dialogue,
        perception=clip.perception, motion=clip.motion, audio=clip.audio)
    sp = _speech_candidates(clip, source, field, params)
    beats = _beat_segments(clip, field, params, anchors)
    heroes: List[HeroCut] = list(sp) + list(beats)
    # Flat model: every cut is first-class. We only ANNOTATE -- stamp each cut's
    # narrative role, then wire the VLM's typed relation graph onto the cuts and
    # mark connected bundles with a shared moment_id. Energy-independent metadata.
    _assign_roles(clip, heroes)
    heroes = _annotate_moments(clip, heroes)
    return heroes


def _assemble(
    heroes: List[HeroCut], file_ids: List[str],
    affordances: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Cross-file finishing pass over per-file heroes: stack repeated takes,
    optional affordance filter, rank best-first. Cheap relative to
    ``_file_heroes`` -- this is the only work done on a cache hit."""
    heroes = _stack_takes(heroes, file_ids)
    if affordances:
        want = set(affordances)
        heroes = [h for h in heroes if want & set(h.affordances or [h.modality])]
    heroes.sort(key=lambda h: h.score, reverse=True)
    return [h.to_dict() for h in heroes]


def compute_file_cache(file_id: str, energy: float) -> List[Dict[str, Any]]:
    """The per-file, pre-stacking hero cuts at one energy, serialized for the
    precompute cache. Empty if the file has no usable artifacts yet."""
    inputs = _load_inputs([file_id])
    clip = inputs.get(file_id)
    if clip is None:
        return []
    params = energy_to_params(energy)
    source = ss.load_sources([file_id]).get(file_id)
    return [h.to_cache() for h in _file_heroes(clip, source, params)]


def assemble_cached(
    cached_by_file: Dict[str, List[Dict[str, Any]]], file_ids: List[str],
    affordances: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Rehydrate per-file cached cuts and run the cross-file finishing pass --
    the read path that avoids recomputing the per-file work."""
    heroes: List[HeroCut] = []
    for fid in file_ids:
        for d in cached_by_file.get(fid, []) or []:
            heroes.append(HeroCut.from_cache(d))
    return _assemble(heroes, file_ids, affordances)
