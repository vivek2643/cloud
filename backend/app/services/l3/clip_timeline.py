"""
Clip Timeline (continuous-editing substrate, v2).
================================================

The v1 cut pipeline (``hero_cuts`` + ``footage_map``) treats a clip as a small
set of pre-baked HERO CUTS: the brain can only place a cut the builder already
decided to mint. That is great for scale but caps creativity -- there is no way
to place an arbitrary span (e.g. a person's *silent* reaction under a
split-screen), because no atom was minted there.

The Clip Timeline inverts that. It models the clip as a set of **dense,
full-coverage lanes over the clip's own clock** -- at *any* millisecond you can
ask "who is present? who is speaking (audio)? who is visibly speaking? where is
the gaze? what shot size? what action?". Cuts become a *fresh scored index*
(bookmarks) over this continuous source, NOT the source itself. A verb can then
place any span the brain can describe, and the deterministic solvers (seam-snap,
peaks, room) keep it frame-accurate.

Design rules
------------
* **Change-point intervals, not fixed hops.** A lane is a list of intervals
  bounded by *real* changes (a speaker turn ends, the shot size changes, a
  person exits). Adjacent intervals with an identical value are merged, so the
  representation is as sparse as the content allows yet still answers
  ``value_at(t)`` for every ``t`` in ``[0, duration]``.
* **Full coverage where a state is always defined** (speech/silence, per-person
  presence, visible-speaking, gaze, shot). These tile ``[0, duration]`` with an
  explicit default fill, so a gap means "we asserted this default", never
  "unknown by omission". Event lanes (action atoms, peaks) stay sparse.
* **Reuse, don't reinvent.** Seams come straight from the L1 fused seam field
  (``fused_seams``); peaks reuse atom ``peak_ms`` + motion impacts; energy is
  the stored prosody RMS. This module only *fuses and indexes* -- it computes no
  new signal.
* **Additive / versioned.** This lives alongside the v1 ladder; nothing here
  deletes or mutates the old path. ``CLIP_TIMELINE_VERSION`` gates the store.

Pure Python, no DB and no network: ``build_clip_timeline`` takes a plain
``TimelineInputs`` (populated by a loader elsewhere) so the whole fusion is unit
-testable with synthetic fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from app.services.l1.fused_seams import FusedField, snap_bounds, snap_point

# Bump when the lane set, value payloads, or cut-index shape change. The store
# regenerates any clip whose stored version is lower.
CLIP_TIMELINE_VERSION = 1

# Speaker-turn merge gap: consecutive words by the same speaker closer than this
# are one turn (mirrors l3.diarize.merge_words_to_turns' default).
_TURN_GAP_MS = 800

# Canonical lane names. Per-person presence lanes are suffixed with the local id
# (``presence:p1``); everything else is a single lane.
LANE_SPEECH = "speech"       # audio: speaker turns vs silence (full coverage)
LANE_PRESENCE = "presence"   # per-person on/off screen (full coverage, one/id)
LANE_SPEAKING = "speaking"   # who is VISIBLY speaking on camera (full coverage)
LANE_GAZE = "gaze"           # gaze direction/target (full coverage)
LANE_SHOT = "shot"           # camera shot size/angle/movement (full coverage)
LANE_ACTION = "action"       # done/shown capture atoms (sparse events)

_FULL_COVERAGE = {LANE_SPEECH, LANE_SPEAKING, LANE_GAZE, LANE_SHOT}


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class Interval:
    """One change-point interval on a lane. ``value`` is a small lane-specific
    facet dict; two adjacent intervals with an equal ``value`` are merged."""
    start_ms: int
    end_ms: int
    value: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"a": self.start_ms, "b": self.end_ms, **self.value}


@dataclass
class Lane:
    name: str
    intervals: List[Interval] = field(default_factory=list)
    full_coverage: bool = False

    def value_at(self, ms: int) -> Optional[Dict[str, Any]]:
        for it in self.intervals:
            if it.start_ms <= ms < it.end_ms:
                return it.value
        return None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "coverage": "full" if self.full_coverage else "events",
            "intervals": [it.to_dict() for it in self.intervals],
        }


@dataclass
class Peak:
    ts_ms: int
    kind: str                      # done | shown | action_impact
    strength: float = 1.0
    subject: Optional[str] = None  # person local id when known

    def to_dict(self) -> dict:
        return {"ts_ms": self.ts_ms, "kind": self.kind,
                "strength": round(self.strength, 3), "subject": self.subject}


@dataclass
class PersonCard:
    """The durable identikit surfaced to the brain for within/cross-clip
    awareness. A description precise enough to *pick this person out* is the key
    that lets the brain (and later a matcher) reason about who is on screen."""
    local_id: str
    role: Optional[str] = None
    description: Optional[str] = None
    durable: Optional[dict] = None
    region: Optional[dict] = None
    enters_ms: Optional[int] = None
    exits_ms: Optional[int] = None
    voice_speaker_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.local_id, "role": self.role, "description": self.description,
            "durable": self.durable, "region": self.region,
            "enters_ms": self.enters_ms, "exits_ms": self.exits_ms,
            "voice_speaker_id": self.voice_speaker_id,
        }


@dataclass
class IndexCut:
    """A fresh scored bookmark over the continuous source -- a *suggestion*, not
    the substrate. ``in_ms/out_ms`` are seam-snapped when a field is present."""
    cut_id: str
    kind: str                       # said | done | shown
    in_ms: int
    out_ms: int
    score: float
    peak_ms: Optional[int] = None
    speaker: Optional[str] = None   # audio speaker id (said)
    subject: Optional[str] = None   # person/place/object/graphic (done/shown)
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "cut_id": self.cut_id, "kind": self.kind,
            "in_ms": self.in_ms, "out_ms": self.out_ms,
            "peak_ms": self.peak_ms, "score": round(self.score, 3),
            "speaker": self.speaker, "subject": self.subject, "label": self.label,
        }


@dataclass
class ClipTimeline:
    file_id: str
    duration_ms: int
    lanes: List[Lane] = field(default_factory=list)
    seams: List[dict] = field(default_factory=list)         # [{ts_ms,q,kind,sources}]
    peaks: List[Peak] = field(default_factory=list)
    energy_hop_ms: int = 0
    energy: List[float] = field(default_factory=list)        # normalized 0..1
    persons: List[PersonCard] = field(default_factory=list)
    cuts: List[IndexCut] = field(default_factory=list)
    version: int = CLIP_TIMELINE_VERSION
    # In-memory only (not serialized in to_dict): the dense fused seam field, kept
    # so an arbitrary span placed by the brain snaps to the SAME clean boundaries
    # the pre-scored cut index already uses. None when the clip has no L1 grids.
    seam_field: Optional[FusedField] = None

    # -- queries (the brain's "senses" over the continuous clock) -----------

    def lane(self, name: str) -> Optional[Lane]:
        for ln in self.lanes:
            if ln.name == name:
                return ln
        # Forgiving fallback: exact match failed, try case-insensitive so a
        # guessed-case lane name ("Speaking") still resolves. No fuzzy aliasing
        # beyond that -- the available-lane list (surfaced by scan_source) is how
        # the caller learns the real names, not magic.
        nl = (name or "").strip().lower()
        for ln in self.lanes:
            if ln.name.lower() == nl:
                return ln
        return None

    def lane_value_keys(self, name: str) -> set:
        """The value KEYS a lane's intervals actually carry -- the honest set of
        facets one can match on. Empty when the lane is unknown."""
        ln = self.lane(name)
        if ln is None:
            return set()
        keys: set = set()
        for it in ln.intervals:
            keys.update(k for k, v in it.value.items() if v is not None)
        return keys

    def lane_vocab(self, name: str, *, cap: int = 8) -> Dict[str, List[Any]]:
        """Per-key observed VALUE set for a lane (capped) -- so a caller reads the
        real query vocabulary ('shot_size' takes wide/medium/close ...) instead of
        guessing tokens that silently miss."""
        ln = self.lane(name)
        if ln is None:
            return {}
        vocab: Dict[str, set] = {}
        for it in ln.intervals:
            for k, v in it.value.items():
                if v is None or isinstance(v, (list, dict)):
                    continue
                vs = vocab.setdefault(k, set())
                if len(vs) < cap:
                    vs.add(v)
        return {k: sorted(vs, key=str) for k, vs in vocab.items()}

    def facet_at(self, ms: int) -> Dict[str, Any]:
        """Sample every full-coverage lane at ``ms`` -- "what is true right now".
        Per-person presence collapses to the list of on-screen ids."""
        out: Dict[str, Any] = {"ms": ms}
        present: List[str] = []
        for ln in self.lanes:
            if ln.name.startswith(LANE_PRESENCE + ":"):
                v = ln.value_at(ms) or {}
                if v.get("state") == "on":
                    present.append(ln.name.split(":", 1)[1])
                continue
            if ln.full_coverage:
                out[ln.name] = ln.value_at(ms)
        out[LANE_PRESENCE] = present
        return out

    def scan(self, lane_name: str, **match: Any) -> List[Interval]:
        """Return the intervals of ``lane_name`` whose value matches every
        ``key=value`` in ``match`` (a facet query over the continuous clock).

        FORGIVING: only match keys the lane actually carries are applied; a key
        the lane doesn't have (a guessed facet name) is IGNORED rather than
        silently zeroing every hit -- so a slightly-wrong query still returns the
        lane's spans (each with full facets) to read, and ``scan_source`` reports
        which keys applied vs. the lane's real vocabulary so the query self-
        corrects. Reactions/other angles come from the coverage groups now; this
        stays the open-ended escape hatch it was meant to be."""
        ln = self.lane(lane_name)
        if ln is None:
            return []
        known = self.lane_value_keys(lane_name)
        applied = {k: v for k, v in match.items() if k in known}
        return [it for it in ln.intervals
                if all(it.value.get(k) == v for k, v in applied.items())]

    def handles(self, in_ms: int, out_ms: int) -> Dict[str, int]:
        """Unused source room around ``[in,out]`` -- how far a span can be
        extended (lead/tail) before it runs off the clip. The facet of that room
        is available via ``facet_at`` at the extended instant."""
        return {"lead_ms": max(0, int(in_ms)), "tail_ms": max(0, self.duration_ms - int(out_ms))}

    def snap_span(self, in_ms: int, out_ms: int, *,
                  max_move_ms: Optional[int] = None) -> Dict[str, Any]:
        """Snap a raw ``[in_ms, out_ms]`` to the clip's fused seam field -- the
        same deterministic snapper the pre-scored cut index uses (word gaps /
        silence / motion impacts, with the mid-word + camera-shake vetoes). This
        is what lets the brain nominate an APPROXIMATE window (a silent reaction,
        a beat before a line) and still land on a clean cut.

        SOVEREIGNTY CAP. The snapper is a service, never a veto: with
        ``max_move_ms`` set, an edge the field wants to move FURTHER than the cap
        stays where the brain put it, and the seam the field preferred is
        reported as ``in_suggested_ms`` / ``out_suggested_ms`` instead -- the
        brain can take the suggestion or keep its deliberate (e.g. match-cut)
        edge.

        Returns ``{in_ms, out_ms, snapped, [in_delta_ms, out_delta_ms, in_q,
        out_q, in_suggested_ms?, out_suggested_ms?]}``. Degrades to an unchanged
        no-op (``snapped=False``) when the clip has no seam field."""
        try:
            a, b = int(in_ms), int(out_ms)
        except (TypeError, ValueError):
            return {"in_ms": in_ms, "out_ms": out_ms, "snapped": False}
        fld = self.seam_field
        if fld is None or not fld.cost or b <= a:
            return {"in_ms": a, "out_ms": b, "snapped": False}
        si, so = snap_bounds(fld, a, b, duration_ms=self.duration_ms)
        out: Dict[str, Any] = {"snapped": True}
        if max_move_ms is not None and abs(si - a) > int(max_move_ms):
            out["in_suggested_ms"] = si
            si = a
        if max_move_ms is not None and abs(so - b) > int(max_move_ms):
            out["out_suggested_ms"] = so
            so = b
        if so <= si:                       # capping one edge must not invert
            si, so = a, b
        out.update({
            "in_ms": si, "out_ms": so,
            "in_delta_ms": si - a, "out_delta_ms": so - b,
            "in_q": round(fld.q_at(si), 3), "out_q": round(fld.q_at(so), 3),
        })
        return out

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "file_id": self.file_id,
            "duration_ms": self.duration_ms,
            "persons": [p.to_dict() for p in self.persons],
            "lanes": [ln.to_dict() for ln in self.lanes],
            "seams": self.seams,
            "peaks": [p.to_dict() for p in self.peaks],
            "energy": {"hop_ms": self.energy_hop_ms, "curve": self.energy},
            "cuts": [c.to_dict() for c in self.cuts],
        }


# --------------------------------------------------------------------------
# Builder input contract (populated by a DB loader elsewhere; kept plain so the
# fusion is unit-testable without a database or network).
# --------------------------------------------------------------------------

@dataclass
class TimelineInputs:
    file_id: str
    duration_ms: int
    # L1 audio
    words: List[dict] = field(default_factory=list)          # {start_ms,end_ms,text,speaker,is_filler}
    rms_db: List[float] = field(default_factory=list)
    prosody_hop_ms: int = 0
    # L2 perception (ClipPerception dict fields)
    persons: List[dict] = field(default_factory=list)
    speaking: List[dict] = field(default_factory=list)
    gaze: List[dict] = field(default_factory=list)
    camera_craft: List[dict] = field(default_factory=list)
    atoms: List[dict] = field(default_factory=list)
    quality_events: List[dict] = field(default_factory=list)
    # v8 dense change-point lanes (preferred over enters/exits + atoms when present)
    presence_lane: List[dict] = field(default_factory=list)   # {start_ms,end_ms,present:[...],primary}
    activity_lane: List[dict] = field(default_factory=list)   # {start_ms,end_ms,mode,subject,actor,label,peak_ms}
    # L1 motion
    action_points: List[dict] = field(default_factory=list)  # {ts_ms,kind,score,...}
    # Reused fused seam field (built by the loader via fused_seams.compute_fused_field)
    field: Optional[FusedField] = None


# --------------------------------------------------------------------------
# Change-point helpers
# --------------------------------------------------------------------------

def _merge_adjacent(intervals: List[Interval]) -> List[Interval]:
    """Collapse touching intervals that carry an identical value -- the whole
    point of a change-point lane: a boundary only where something actually
    changed."""
    out: List[Interval] = []
    for it in intervals:
        if it.end_ms <= it.start_ms:
            continue
        if out and out[-1].end_ms == it.start_ms and out[-1].value == it.value:
            out[-1] = Interval(out[-1].start_ms, it.end_ms, out[-1].value)
        else:
            out.append(Interval(it.start_ms, it.end_ms, dict(it.value)))
    return out


def _full_coverage(spans: Sequence[Tuple[int, int, Dict[str, Any]]],
                   duration_ms: int, default: Dict[str, Any]) -> List[Interval]:
    """Tile ``[0, duration]`` from possibly-overlapping/sparse ``spans``
    (``(start,end,value)``); gaps take ``default``; on overlap the *later*
    span (input order = priority) wins. Result is change-point merged."""
    if duration_ms <= 0:
        return []
    clean: List[Tuple[int, int, Dict[str, Any]]] = []
    bounds = {0, duration_ms}
    for a, b, v in spans:
        a, b = max(0, int(a)), min(duration_ms, int(b))
        if b > a:
            clean.append((a, b, v))
            bounds.add(a)
            bounds.add(b)
    pts = sorted(bounds)
    out: List[Interval] = []
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        mid = (a + b) / 2.0
        val = default
        for sa, sb, sv in clean:            # later span wins the overlap
            if sa <= mid < sb:
                val = sv
        out.append(Interval(a, b, dict(val)))
    return _merge_adjacent(out)


def _merge_turns(words: Sequence[dict], gap_ms: int = _TURN_GAP_MS) -> List[dict]:
    """Group flat words into speaker turns (same speaker, gap <= gap_ms)."""
    ws = sorted((w for w in words if w.get("start_ms") is not None),
                key=lambda w: int(w["start_ms"]))
    turns: List[dict] = []
    for w in ws:
        spk = w.get("speaker")
        a, b = int(w["start_ms"]), int(w.get("end_ms", w["start_ms"]))
        txt = (w.get("text") or "").strip()
        if turns and turns[-1]["speaker"] == spk and a - turns[-1]["end_ms"] <= gap_ms:
            turns[-1]["end_ms"] = max(turns[-1]["end_ms"], b)
            turns[-1]["words"] += 1
            if txt:
                turns[-1]["text"] = (turns[-1]["text"] + " " + txt).strip()
        else:
            turns.append({"start_ms": a, "end_ms": b, "speaker": spk,
                          "words": 1, "text": txt})
    return turns


# --------------------------------------------------------------------------
# Lane builders
# --------------------------------------------------------------------------

def _speech_lane(turns: List[dict], duration_ms: int) -> Lane:
    spans = [(t["start_ms"], t["end_ms"],
              {"state": "speech", "speaker": t.get("speaker"),
               "words": t["words"], "text": t.get("text", "")})
             for t in turns]
    ivs = _full_coverage(spans, duration_ms, {"state": "silence"})
    return Lane(LANE_SPEECH, ivs, full_coverage=True)


def _presence_lanes(persons: List[dict], duration_ms: int) -> List[Lane]:
    """Per-person on/off from coarse enters/exits (v7 fallback)."""
    lanes: List[Lane] = []
    for p in persons:
        pid = p.get("local_id")
        if not pid:
            continue
        enters = p.get("enters_ms")
        exits = p.get("exits_ms")
        a = 0 if enters is None else max(0, int(enters))
        b = duration_ms if exits is None else min(duration_ms, int(exits))
        spans = [(a, b, {"state": "on"})] if b > a else []
        ivs = _full_coverage(spans, duration_ms, {"state": "off"})
        lanes.append(Lane(f"{LANE_PRESENCE}:{pid}", ivs, full_coverage=True))
    return lanes


def _presence_lanes_from_lane(presence_lane: List[dict], persons: List[dict],
                              duration_ms: int) -> List[Lane]:
    """Per-person on/off from the DENSE v8 presence_lane (change-point coverage
    of who is on screen) -- far richer than enters/exits (handles re-entry)."""
    ids = [p.get("local_id") for p in persons if p.get("local_id")]
    for sp in presence_lane:                     # include ids only seen in the lane
        for pid in sp.get("present") or []:
            if pid and pid not in ids:
                ids.append(pid)
    lanes: List[Lane] = []
    for pid in ids:
        spans = [(sp.get("start_ms", 0), sp.get("end_ms", 0), {"state": "on"})
                 for sp in presence_lane if pid in (sp.get("present") or [])]
        ivs = _full_coverage(spans, duration_ms, {"state": "off"})
        lanes.append(Lane(f"{LANE_PRESENCE}:{pid}", ivs, full_coverage=True))
    return lanes


def _speaking_lane(speaking: List[dict], duration_ms: int) -> Lane:
    spans = [(s.get("start_ms", 0), s.get("end_ms", 0), {"subject": s.get("subject")})
             for s in speaking if s.get("subject")]
    ivs = _full_coverage(spans, duration_ms, {"subject": None})
    return Lane(LANE_SPEAKING, ivs, full_coverage=True)


def _gaze_lane(gaze: List[dict], duration_ms: int) -> Lane:
    spans = [(g.get("start_ms", 0), g.get("end_ms", 0),
              {"subject": g.get("subject"), "direction": g.get("direction"),
               "target": g.get("target")})
             for g in gaze if g.get("start_ms") is not None]
    ivs = _full_coverage(spans, duration_ms, {"direction": "unknown"})
    return Lane(LANE_GAZE, ivs, full_coverage=True)


def _shot_lane(camera: List[dict], duration_ms: int) -> Lane:
    spans = [(c.get("start_ms", 0), c.get("end_ms", 0),
              {"shot_size": c.get("shot_size"), "angle": c.get("angle"),
               "movement": c.get("movement"), "focus": c.get("subject_focus")})
             for c in camera if c.get("start_ms") is not None]
    ivs = _full_coverage(spans, duration_ms, {"shot_size": "unsure"})
    return Lane(LANE_SHOT, ivs, full_coverage=True)


def _action_lane(atoms: List[dict]) -> Lane:
    ivs: List[Interval] = []
    for a in atoms:
        ch = a.get("channel")
        if ch not in ("done", "shown"):
            continue
        s, e = a.get("start_ms"), a.get("end_ms")
        if s is None or e is None or int(e) <= int(s):
            continue
        ivs.append(Interval(int(s), int(e), {
            "channel": ch, "subject": a.get("subject"), "actor": a.get("actor"),
            "label": a.get("label") or "", "summary": a.get("summary"),
            "peak_ms": a.get("peak_ms"), "confidence": a.get("confidence"),
        }))
    ivs.sort(key=lambda it: it.start_ms)
    return Lane(LANE_ACTION, ivs, full_coverage=False)


_MODE_CHANNEL = {"action": "done", "held": "shown"}


def _action_lane_from_activity(activity_lane: List[dict]) -> Lane:
    """The DENSE v8 activity_lane -> action-lane events. Skips 'idle' lulls (the
    speech/presence lanes already cover those); keeps action + held so a silent
    held reaction is a visible, addressable interval."""
    ivs: List[Interval] = []
    for a in activity_lane:
        mode = a.get("mode")
        ch = _MODE_CHANNEL.get(mode)
        if ch is None:                            # idle / unknown -> not an event
            continue
        s, e = a.get("start_ms"), a.get("end_ms")
        if s is None or e is None or int(e) <= int(s):
            continue
        ivs.append(Interval(int(s), int(e), {
            "channel": ch, "mode": mode, "subject": a.get("subject"),
            "actor": a.get("actor"), "label": a.get("label") or "",
            "peak_ms": a.get("peak_ms"), "confidence": a.get("confidence"),
        }))
    ivs.sort(key=lambda it: it.start_ms)
    return Lane(LANE_ACTION, ivs, full_coverage=False)


# --------------------------------------------------------------------------
# Peaks / energy / person cards
# --------------------------------------------------------------------------

def _build_peaks(atoms: List[dict], action_points: List[dict],
                 activity_lane: Optional[List[dict]] = None) -> List[Peak]:
    peaks: List[Peak] = []
    # Prefer the dense activity_lane peaks when present; else the sparse atoms.
    if activity_lane:
        for a in activity_lane:
            ch = _MODE_CHANNEL.get(a.get("mode"))
            pk = a.get("peak_ms")
            if ch is None or pk is None:
                continue
            peaks.append(Peak(int(pk), ch, float(a.get("confidence") or 0.5), a.get("actor")))
    else:
        for a in atoms:
            ch = a.get("channel")
            if ch not in ("done", "shown"):
                continue
            pk = a.get("peak_ms")
            if pk is None:
                s, e = a.get("start_ms"), a.get("end_ms")
                if s is None or e is None:
                    continue
                pk = (int(s) + int(e)) // 2
            peaks.append(Peak(int(pk), ch, float(a.get("confidence") or 0.5), a.get("actor")))
    for p in action_points or []:
        ts = p.get("ts_ms")
        if ts is not None:
            peaks.append(Peak(int(ts), "action_impact", float(p.get("score") or 1.0)))
    peaks.sort(key=lambda p: p.ts_ms)
    return peaks


def _normalize_energy(rms_db: List[float]) -> List[float]:
    """Map the stored prosody RMS (dB) to a 0..1 loudness curve; flat/empty ->
    empty. Purely for the brain's awareness read-out (louder = higher)."""
    vals = [float(x) for x in rms_db if x is not None]
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-6:
        return [0.0] * len(rms_db)
    return [round((float(x) - lo) / (hi - lo), 3) if x is not None else 0.0 for x in rms_db]


def _person_cards(persons: List[dict]) -> List[PersonCard]:
    cards: List[PersonCard] = []
    for p in persons:
        pid = p.get("local_id")
        if not pid:
            continue
        cards.append(PersonCard(
            local_id=pid, role=p.get("role"),
            description=p.get("canonical_description"),
            durable=p.get("durable"), region=p.get("frame_region"),
            enters_ms=p.get("enters_ms"), exits_ms=p.get("exits_ms"),
            voice_speaker_id=p.get("voice_speaker_id"),
        ))
    return cards


# --------------------------------------------------------------------------
# Fresh scored cut index (bookmarks over the continuous source)
# --------------------------------------------------------------------------

def _fluency_at(quality_events: List[dict], a: int, b: int) -> float:
    """Mean fluency/energy quality (0..1) over [a,b] from L2 quality events;
    neutral 0.6 when none overlap."""
    scores: List[float] = []
    for q in quality_events or []:
        qs, qe = q.get("start_ms"), q.get("end_ms")
        if qs is None or qe is None:
            continue
        if int(qe) > a and int(qs) < b and q.get("dimension") in ("fluency", "energy", "naturalness"):
            scores.append(int(q.get("score", 3)) / 5.0)
    return sum(scores) / len(scores) if scores else 0.6


def _speech_crest(rms: List[float], hop: int, a: int, b: int) -> Optional[int]:
    if not rms or hop <= 0 or b <= a:
        return None
    i0, i1 = max(0, a // hop), min(len(rms) - 1, b // hop)
    if i1 < i0:
        return None
    best = max(range(i0, i1 + 1), key=lambda i: rms[i])
    return best * hop


def _derive_cut_index(inputs: TimelineInputs, turns: List[dict],
                      norm_energy: List[float]) -> List[IndexCut]:
    cuts: List[IndexCut] = []
    fld = inputs.field
    n = 0
    # Said bookmarks: one per speaker turn, seam-snapped.
    for t in turns:
        a, b = int(t["start_ms"]), int(t["end_ms"])
        if b - a < 200:
            continue
        if fld is not None:
            a, b = snap_bounds(fld, a, b, duration_ms=inputs.duration_ms)
        dur_s = (b - a) / 1000.0
        length_score = min(1.0, dur_s / 6.0)      # favor substantive turns
        score = 0.5 * length_score + 0.5 * _fluency_at(inputs.quality_events, a, b)
        peak = _speech_crest(inputs.rms_db, inputs.prosody_hop_ms, a, b)
        n += 1
        cuts.append(IndexCut(f"c{n}", "said", a, b, score, peak_ms=peak,
                             speaker=t.get("speaker"), label=(t.get("text") or "")[:80]))
    # Video bookmarks: prefer the DENSE activity_lane (so a silent held reaction
    # becomes a placeable bookmark), else the sparse atoms highlights.
    if inputs.activity_lane:
        video_beats = [
            {"channel": _MODE_CHANNEL.get(a.get("mode")), "start_ms": a.get("start_ms"),
             "end_ms": a.get("end_ms"), "peak_ms": a.get("peak_ms"),
             "subject": a.get("subject"), "label": a.get("label"),
             "confidence": a.get("confidence")}
            for a in inputs.activity_lane if _MODE_CHANNEL.get(a.get("mode"))
        ]
    else:
        video_beats = inputs.atoms
    for atom in video_beats:
        ch = atom.get("channel")
        if ch not in ("done", "shown"):
            continue
        a, b = atom.get("start_ms"), atom.get("end_ms")
        if a is None or b is None or int(b) <= int(a):
            continue
        a, b = int(a), int(b)
        pk = atom.get("peak_ms")
        pk = int(pk) if pk is not None else (a + b) // 2
        if fld is not None:
            a = snap_point(fld, a, max(0, a - 600), pk)
            b = snap_point(fld, b, pk, min(inputs.duration_ms, b + 600))
            if b <= a:
                b = a + 1
        score = float(atom.get("confidence") or 0.5)
        n += 1
        cuts.append(IndexCut(f"c{n}", ch, a, b, score, peak_ms=pk,
                             subject=atom.get("subject"),
                             label=atom.get("label") or ""))
    cuts.sort(key=lambda c: -c.score)
    return cuts


# --------------------------------------------------------------------------
# Public builder
# --------------------------------------------------------------------------

def build_clip_timeline(inputs: TimelineInputs) -> ClipTimeline:
    """Fuse L1 + L2 signals into the continuous Clip Timeline: change-point
    lanes, reused seams, peaks, energy, person cards, and a fresh scored cut
    index. Deterministic and pure -- no DB, no VLM."""
    dur = max(0, int(inputs.duration_ms))
    turns = _merge_turns(inputs.words)

    lanes: List[Lane] = [_speech_lane(turns, dur)]
    if inputs.presence_lane:                       # v8 dense coverage preferred
        lanes.extend(_presence_lanes_from_lane(inputs.presence_lane, inputs.persons, dur))
    else:
        lanes.extend(_presence_lanes(inputs.persons, dur))
    lanes.append(_speaking_lane(inputs.speaking, dur))
    lanes.append(_gaze_lane(inputs.gaze, dur))
    lanes.append(_shot_lane(inputs.camera_craft, dur))
    if inputs.activity_lane:                        # v8 dense coverage preferred
        lanes.append(_action_lane_from_activity(inputs.activity_lane))
    else:
        lanes.append(_action_lane(inputs.atoms))

    seams = [s.to_dict() for s in inputs.field.seams] if inputs.field else []
    peaks = _build_peaks(inputs.atoms, inputs.action_points, inputs.activity_lane)
    norm_energy = _normalize_energy(inputs.rms_db)
    persons = _person_cards(inputs.persons)
    cuts = _derive_cut_index(inputs, turns, norm_energy)

    return ClipTimeline(
        file_id=inputs.file_id, duration_ms=dur, lanes=lanes, seams=seams,
        peaks=peaks, energy_hop_ms=inputs.prosody_hop_ms, energy=norm_energy,
        persons=persons, cuts=cuts, seam_field=inputs.field,
    )


# --------------------------------------------------------------------------
# Brain awareness surface
# --------------------------------------------------------------------------
# The whole point of the continuous model is that the brain can see the clip as
# a fully-addressable source, not a fixed menu. ``render_awareness`` turns a
# ClipTimeline into a compact, human/LLM-readable digest: WHO is in it, WHAT is
# true across the clock (change-point lanes), WHERE the clean cut points and
# impacts are, and a scored INDEX of ready bookmarks. Every span is addressable,
# so a verb (place_span / roll) can target any window the brain describes.

def _secs(ms: Optional[int]) -> str:
    if ms is None:
        return "?"
    return f"{ms / 1000.0:.1f}s"


def _lane_line(lane: Lane, *, key: str, max_ivs: int = 10) -> str:
    """One-line change-point read-out of a lane: ``a-b VALUE | a-b VALUE ...``."""
    parts: List[str] = []
    for it in lane.intervals[:max_ivs]:
        v = it.value.get(key)
        if v is None:
            v = "-"
        parts.append(f"{_secs(it.start_ms)}-{_secs(it.end_ms)} {v}")
    if len(lane.intervals) > max_ivs:
        parts.append("…")
    return " | ".join(parts)


def render_summary(tl: ClipTimeline, *, fid8: Optional[str] = None) -> str:
    """One line per clip -- the SUMMARY disclosure tier for large shoots. Enough
    to know what the clip IS (duration, who, how much speech/action, index size)
    so the brain can drill in with source_awareness/scan_source on the clips
    that matter, instead of every clip's full digest crowding the context."""
    fid = fid8 or tl.file_id[:8]
    who = ", ".join(f"{p.local_id}{f'({p.role})' if p.role else ''}"
                    for p in tl.persons) or "no tracked people"
    speech = tl.lane(LANE_SPEECH)
    talk_ms = sum(it.end_ms - it.start_ms for it in (speech.intervals if speech else [])
                  if it.value.get("state") == "speech")
    action = tl.lane(LANE_ACTION)
    n_act = len(action.intervals) if action else 0
    return (f"CLIP {fid}  {_secs(tl.duration_ms)}  people: {who}  "
            f"speech {_secs(talk_ms)}  action events {n_act}  "
            f"cut index {len(tl.cuts)}")


def render_awareness(tl: ClipTimeline, *, fid8: Optional[str] = None,
                     max_cuts: int = 12, detail: str = "full") -> str:
    """A compact digest of the continuous clip for the brain -- complete
    awareness in one screen: people, change-point lanes, seams, peaks, and the
    scored cut index (bookmarks it can place with place_span).

    ``detail`` is the disclosure tier: 'full' (default) or 'compact' (roughly
    half the per-section budget, for shoots with many clips -- every section
    still appears, just shorter; scan_source recovers anything elided)."""
    compact = detail == "compact"
    max_ivs = 4 if compact else 10
    n_action, n_seams, n_peaks = (4, 4, 4) if compact else (8, 8, 10)
    if compact:
        max_cuts = min(max_cuts, 6)
    fid = fid8 or tl.file_id[:8]
    lines: List[str] = [f"CLIP {fid}  duration {_secs(tl.duration_ms)}"]

    if tl.persons:
        lines.append("PEOPLE:")
        for p in tl.persons:
            win = f"{_secs(p.enters_ms)}–{_secs(p.exits_ms)}" if (p.enters_ms is not None or p.exits_ms is not None) else "whole clip"
            voice = f"  voice={p.voice_speaker_id}" if p.voice_speaker_id else ""
            desc = f' — "{p.description}"' if p.description else ""
            lines.append(f"  {p.local_id} {p.role or ''}{desc}  on {win}{voice}".rstrip())

    lines.append("LANES (change-point intervals over the clip clock):")
    speech = tl.lane(LANE_SPEECH)
    if speech:
        lines.append(f"  speech:   {_lane_line(speech, key='state', max_ivs=max_ivs)}")
    for ln in tl.lanes:
        if ln.name.startswith(LANE_PRESENCE + ":"):
            on = [it for it in ln.intervals if it.value.get("state") == "on"]
            span = f"{_secs(on[0].start_ms)}–{_secs(on[-1].end_ms)}" if on else "never"
            lines.append(f"  present {ln.name.split(':', 1)[1]}: on {span}")
    speaking = tl.lane(LANE_SPEAKING)
    if speaking and any(it.value.get("subject") for it in speaking.intervals):
        lines.append(f"  speaking: {_lane_line(speaking, key='subject', max_ivs=max_ivs)}")
    gaze = tl.lane(LANE_GAZE)
    if gaze and any(it.value.get("direction") not in (None, 'unknown') for it in gaze.intervals):
        lines.append(f"  gaze:     {_lane_line(gaze, key='direction', max_ivs=max_ivs)}")
    shot = tl.lane(LANE_SHOT)
    if shot and any(it.value.get("shot_size") not in (None, 'unsure') for it in shot.intervals):
        lines.append(f"  shot:     {_lane_line(shot, key='shot_size', max_ivs=max_ivs)}")

    action = tl.lane(LANE_ACTION)
    if action and action.intervals:
        lines.append("ACTION (capture events):")
        for it in action.intervals[:n_action]:
            v = it.value
            lines.append(f"  [{v.get('channel')}] {_secs(it.start_ms)}-{_secs(it.end_ms)} "
                         f"peak {_secs(v.get('peak_ms'))} {v.get('subject') or ''} "
                         f"\"{v.get('label')}\" c{v.get('confidence')}".rstrip())
        if len(action.intervals) > n_action:
            lines.append(f"  … {len(action.intervals) - n_action} more (scan_source lane 'action')")

    if tl.seams:
        top = sorted(tl.seams, key=lambda s: -s.get("q", 0))[:n_seams]
        lines.append("SEAMS (cleanest cut points): " +
                     ", ".join(f"{_secs(s['ts_ms'])}({s.get('kind')} {s.get('q')})" for s in top))
    if tl.peaks:
        lines.append("PEAKS (impacts/reveals): " +
                     ", ".join(f"{_secs(p.ts_ms)}({p.kind})" for p in tl.peaks[:n_peaks]))

    if tl.cuts:
        lines.append("CUT INDEX (scored bookmarks — place any with place_span, or any custom span):")
        for c in tl.cuts[:max_cuts]:
            who = c.speaker or c.subject or ""
            lbl = f' "{c.label}"' if c.label else ""
            lines.append(f"  {c.cut_id} [{c.kind}] {_secs(c.in_ms)}-{_secs(c.out_ms)} "
                         f"peak {_secs(c.peak_ms)} {who}{lbl} score {c.score:.2f}".rstrip())
        if len(tl.cuts) > max_cuts:
            lines.append(f"  … {len(tl.cuts) - max_cuts} more bookmarks")
    return "\n".join(lines)
