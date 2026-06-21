"""
The anchor layer -- the foundation of the universal "segment" feed.

An ANCHOR is one timestamped *reason to keep a piece of footage*, read from the
L1/L2 artifacts we ALREADY store (no new model call). Every editable moment --
a spoken sentence, a motion impact, a facial reaction, a held composition, a
reveal/entrance -- becomes an anchor on one shared timeline. Downstream, the
segment engine clusters anchors by the energy dial and snaps safe boundaries
*around* them, so the rule becomes simple and universal:

    coverage = anchors are dense + seams tile the clip
             => every usable instant lives inside some candidate segment
             => "never touch raw" is structural, not best-effort.

Each anchor carries:
  * a CORE extent [start_ms, end_ms] and a representative ``ts_ms`` (the instant
    the moment is "about" -- the impact frame, the expression peak) -- this is
    what boundaries must never clip.
  * a ``kind`` (the source signal) and an ``affordance`` (the editorial bucket:
    what you'd reach for it for) -- affordance is a *filter*, never a pipeline.
  * a ``salience`` 0..1 derived from existing signals, so marginal moments stay
    reachable but rank low (never hidden).

This module is pure (plain dicts in, dataclasses out) and has no DB or VLM
dependency, so it is trivially testable and reusable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Affordance buckets (what an editor reaches for). Modality in the feed is just
# the dominant affordance of a segment -- these are filters, not pipelines.
AFF_SPEECH = "speech"      # a spoken line / answer (sync)
AFF_ACTION = "action"      # a physical action beat / impact (sync)
AFF_REACTION = "reaction"  # a facial reaction / expression (overlay cutaway)
AFF_BROLL = "broll"        # a held, stable composition (overlay)
AFF_INSERT = "insert"      # a beat worth an insert: reveal/entrance/graphic (overlay)

# VLM event `change` values that mark a discrete visual insert beat. (Action is
# NOT read from events -- ``action_*`` is overloaded with speech onsets; action
# comes only from the VLM's content_units. See ``_action_anchors``.)
_INSERT_CHANGES = {"reveal", "enters_frame", "exits_frame", "setup", "holds"}

# Hidden-by-default dialogue lexicon flags (off-camera audio).
_OFFCAMERA_FLAGS = ("offscreen", "production_cue")

# A b-roll / hold cutaway is a short handle the editor lays over A-roll, not the
# whole take. A static span can run for minutes (a locked-off interview); we emit
# a representative, bounded slice from its middle so it stays a usable card.
MAX_HOLD_MS = 5000
MIN_HOLD_MS = 1500
# B-roll keeps its FULL extent into the cut engine (energy then insets it to a
# per-band core); this is only a sanity guard against a multi-minute locked span.
BROLL_MAX_HOLD_MS = 15000

# A gaze is a cutaway only when it DEPARTS from the subject's dominant eyeline
# (their baseline composition). In an interview the baseline is off-camera (the
# subject faces the interviewer) so off-camera is not 69 cutaways; in a vlog the
# baseline is to-camera so a glance off-screen IS the event. Per-subject, no
# content_type. `unsure` is noise; a look must be held (not a micro-dart).
MIN_GAZE_MS = 700
# An interaction kind that is just people talking is already covered by speech.
_INTERACTION_SKIP = {"conversation"}


def _hold_core(a: int, b: int, cap: int = MAX_HOLD_MS) -> tuple:
    """Bound a long held span to a representative middle slice of <= ``cap``."""
    if b - a <= cap:
        return a, b
    mid = (a + b) // 2
    half = cap // 2
    return mid - half, mid + half


@dataclass
class Anchor:
    ts_ms: int                       # representative instant (impact / peak / start)
    start_ms: int                    # core extent start (never clipped by a cut)
    end_ms: int                      # core extent end
    kind: str                        # speech|impact|action_beat|expression|hold|reveal|...
    affordance: str                  # AFF_* bucket
    salience: float = 0.5            # 0..1, higher = more worth surfacing
    actor: Optional[str] = None      # person local_id when known
    region: Optional[dict] = None    # coarse frame box for reframing
    text: Optional[str] = None       # label / spoken text / description
    speaker: Optional[str] = None    # diarized speaker (speech)
    flags: List[str] = field(default_factory=list)
    source_id: Optional[str] = None  # originating artifact id (seg_id/unit_id/event id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_ms": self.ts_ms, "start_ms": self.start_ms, "end_ms": self.end_ms,
            "kind": self.kind, "affordance": self.affordance,
            "salience": round(self.salience, 3), "actor": self.actor,
            "text": self.text, "speaker": self.speaker, "flags": self.flags,
            "source_id": self.source_id,
        }


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _mean(xs: List[float], lo: int, hi: int) -> float:
    seg = xs[max(0, lo):max(lo + 1, hi)]
    return sum(seg) / len(seg) if seg else 0.0


def _is_offcamera(seg: dict) -> bool:
    flags = seg.get("flags") or []
    return any(f in flags for f in _OFFCAMERA_FLAGS) or "backchannel" in flags


# --------------------------------------------------------------------------
# Per-track anchor builders
# --------------------------------------------------------------------------

def _speech_anchors(sentences: List[dict], quality: List[dict]) -> List[Anchor]:
    """One anchor per on-camera sentence (the proven speech atom). The core is
    the whole sentence -- a cut can never land inside the words."""
    out: List[Anchor] = []
    for s in sentences:
        if _is_offcamera(s):
            continue
        a = int(s.get("src_in_ms", 0))
        b = int(s.get("src_out_ms", a))
        if b <= a:
            continue
        text = (s.get("text") or "").strip()
        words = len(text.split())
        sal = _clamp01(words / 12.0)
        q = _vlm_quality(quality, a, b)
        if q is not None:
            sal = _clamp01(0.6 * sal + 0.4 * q)
        out.append(Anchor(
            ts_ms=a, start_ms=a, end_ms=b, kind="speech", affordance=AFF_SPEECH,
            salience=sal, text=text, speaker=s.get("speaker"),
            flags=[f for f in (s.get("flags") or []) if f in ("noisy", "overlap")],
            source_id=str(s.get("seg_id", "")),
        ))
    return out


# An action_* VLM event only counts as a PHYSICAL action beat if a real motion
# impact this strong lands inside it -- otherwise it's just narrated speech
# ("p1 speaks ...") that the VLM happened to tag action_starts.
_IMPACT_MIN_SCORE = 0.5


def _action_anchors(perception: dict, motion: dict, quality: List[dict]) -> List[Anchor]:
    """Action beats and held b-roll from the VLM's content segmentation -- ONE
    source of truth, no category heuristics:

      * ``kind=action`` content_units -> action beats (sync). This is the VLM's
        considered "this is physical action" judgment; it generalizes across
        sports, cooking, dance, etc., because it's semantic, not categorical.
      * ``kind=visual`` content_units -> b-roll holds (overlay).

    We deliberately do NOT mint action from ``action_*`` events: the VLM overloads
    that change to mean "speech begins", and motion can't disambiguate (a
    talking-head fires impacts wall-to-wall -- hundreds, all maxed -- while a real
    swing fires a handful). Recovering action when the VLM gives no units is a
    separate, motion-native source (isolated-transient detection), not a guess
    layered on an ambiguous label.

    The motion ``action_points`` only sharpen the representative impact instant
    and supply salience for spans the VLM already called action.
    """
    hop = max(1, int((motion or {}).get("hop_ms", 100)))
    energy = (motion or {}).get("action_energy") or []
    impacts = [(int(p.get("ts_ms", 0)), float(p.get("score", 0.0)))
               for p in ((motion or {}).get("action_points") or [])
               if isinstance(p, dict) and p.get("ts_ms") is not None]

    def _impact_in(a: int, b: int):
        return [(ts, sc) for ts, sc in impacts if a <= ts <= b]

    out: List[Anchor] = []
    # (start, end, text, region, source_id, is_performance, actor)
    action_spans: List[tuple] = []
    for u in (perception.get("content_units") or []):
        kind = (u.get("kind") or "").lower()
        a, b = int(u.get("start_ms", 0)), int(u.get("end_ms", 0))
        if b <= a:
            continue
        text = (u.get("label") or u.get("content_key") or kind)
        if kind in ("action", "performance"):
            # A performance (song/dance/bit) is a sustained delivery: same sync
            # bucket as an action beat, but the *kind* is preserved so cut-time
            # tightening can keep its full duration (never trim to a core).
            parts = u.get("participants") or []
            actor = u.get("subject") or u.get("actor") or (parts[0] if parts else None)
            action_spans.append((a, b, text, u.get("region"),
                                 str(u.get("unit_id", "")), kind == "performance", actor))
        elif kind == "visual":
            ha, hb = _hold_core(a, b)
            out.append(Anchor(
                ts_ms=(ha + hb) // 2, start_ms=ha, end_ms=hb, kind="hold",
                affordance=AFF_BROLL, salience=_clamp01((b - a) / 6000.0),
                text=text[:200], source_id=str(u.get("unit_id", "")),
            ))

    for (a, b, text, region, sid, is_perf, actor) in action_spans:
        inside = _impact_in(a, b)
        ts = max(inside, key=lambda t: t[1])[0] if inside else (a + b) // 2
        sal = _mean(energy, a // hop, b // hop) if energy else 0.4
        if inside:
            sal = max(sal, max(sc for _, sc in inside))
        q = _vlm_quality(quality, a, b)
        if q is not None:
            sal = _clamp01(0.7 * sal + 0.3 * q)
        out.append(Anchor(
            ts_ms=ts, start_ms=a, end_ms=b,
            kind="performance" if is_perf else "action_beat", affordance=AFF_ACTION,
            salience=_clamp01(sal), actor=actor, region=region, text=text[:200], source_id=sid,
        ))
    return out


def _reaction_anchors(perception: dict) -> List[Anchor]:
    """Facial reactions -> reaction anchors. Salience starts at intensity; the
    warrant pass (``_apply_reaction_warrant``) later upgrades it from context."""
    out: List[Anchor] = []
    for r in (perception.get("reactions") or []):
        a, b = int(r.get("start_ms", 0)), int(r.get("end_ms", 0))
        if b <= a:
            continue
        typ = (r.get("type") or "reaction")
        trig = r.get("trigger")
        label = f"{typ}" + (f" \u00b7 {trig}" if trig else "")
        out.append(Anchor(
            ts_ms=(a + b) // 2, start_ms=a, end_ms=b, kind="expression",
            affordance=AFF_REACTION, salience=_clamp01(float(r.get("intensity") or 0.5)),
            actor=r.get("subject"), text=label,
        ))
    return out


# --- Reaction warrant: a reaction earns a card only when it is reacting TO
# something -- a physical action beat, a strong expression, or another person's
# long preceding turn. We fold these into the reaction's salience so the flat
# energy floor (energy.py) keeps only the justified ones, regardless of band.
_REACTION_ACTION_LOOKBACK_MS = 1500   # an action this close before counts as the trigger
_REACTION_TURN_GAP_MS = 800           # merge same-speaker spans into one turn
_REACTION_LISTEN_FULL_MS = 8000       # preceding turn length that earns full warrant
_REACTION_ACTION_WARRANT = 0.85       # warrant granted by an action trigger


def _speech_turns(speaking: List[dict]) -> List[tuple]:
    """Merge consecutive same-subject speaking spans into turns -> (subject, a, b)."""
    spans = sorted(
        ((s.get("subject"), int(s.get("start_ms", 0)), int(s.get("end_ms", 0)))
         for s in speaking if int(s.get("end_ms", 0)) > int(s.get("start_ms", 0))),
        key=lambda t: t[1],
    )
    turns: List[list] = []
    for subj, a, b in spans:
        if turns and turns[-1][0] == subj and a - turns[-1][2] <= _REACTION_TURN_GAP_MS:
            turns[-1][2] = max(turns[-1][2], b)
        else:
            turns.append([subj, a, b])
    return [tuple(t) for t in turns]


def _action_spans(perception: dict) -> List[tuple]:
    """Physical action / performance beats from content_units -> (start, end)."""
    out: List[tuple] = []
    for u in (perception.get("content_units") or []):
        if (u.get("kind") or "").lower() in ("action", "performance"):
            a, b = int(u.get("start_ms", 0)), int(u.get("end_ms", 0))
            if b > a:
                out.append((a, b))
    return out


def _reaction_warrant(a: int, b: int, actor, intensity: float,
                      turns: List[tuple], actions: List[tuple]) -> float:
    """Best available reason this reaction earns a spot (0..1)."""
    w = float(intensity)
    # Reacting to an action beat that overlaps or just precedes the onset.
    lo = a - _REACTION_ACTION_LOOKBACK_MS
    for (sa, sb) in actions:
        if sb >= lo and sa <= b:
            w = max(w, _REACTION_ACTION_WARRANT)
            break
    # Reacting after ANOTHER person's turn -- longer point => more warranted.
    for (subj, ta, tb) in turns:
        if subj == actor:
            continue
        if ta <= a <= tb or 0 <= a - tb <= _REACTION_ACTION_LOOKBACK_MS:
            w = max(w, min(1.0, (tb - ta) / _REACTION_LISTEN_FULL_MS))
    return _clamp01(w)


def _apply_reaction_warrant(anchors: List[Anchor], perception: dict) -> None:
    """Rewrite each reaction anchor's salience to its context warrant (in place)."""
    reacts = [a for a in anchors if a.affordance == AFF_REACTION]
    if not reacts:
        return
    turns = _speech_turns(perception.get("speaking") or [])
    actions = _action_spans(perception)
    for a in reacts:
        a.salience = _reaction_warrant(a.start_ms, a.end_ms, a.actor, a.salience, turns, actions)


# Intentional moves are more dynamic than a locked-off shot, so rank them a touch
# higher at equal length.
_MOVE_SALIENCE_BONUS = 0.15


def _broll_anchors(perception: dict) -> List[Anchor]:
    """Usable shots -> overlay b-roll anchors. A shot is usable footage iff it is
    either held (``static``) OR an INTENTIONAL move -- and intentionality is read
    from the VLM's universal ``is_deliberate`` flag, NOT a movement-name allowlist.
    So a deliberate handheld push survives and an incidental wobble drops, with no
    per-movement special-casing. A move is bad to cut *through* (the fused field
    vetoes mid-move seams) but premium to cut *to*; boundaries snap later so you
    get the whole move as one clean handle."""
    out: List[Anchor] = []
    seen: List[tuple] = []
    primary = _primary_subjects(perception)
    for c in (perception.get("camera_craft") or []):
        mv = (c.get("movement") or "static")
        moving = mv != "static"
        # Universal property: keep held shots and deliberate moves; drop incidental
        # motion. (When the VLM didn't judge deliberateness, trust a named move.)
        if moving and c.get("is_deliberate") is False:
            continue
        # Self-quieting: a STATIC hold framed on the dominant on-screen subject is
        # the A-roll, not a cutaway (mirrors the cutaway-track gate).
        if not moving and _focus_names_primary(c.get("subject_focus"), primary):
            continue
        a, b = int(c.get("start_ms", 0)), int(c.get("end_ms", 0))
        if b - a < MIN_HOLD_MS:           # too short to be a usable shot
            continue
        seen.append((a, b))
        ha, hb = _hold_core(a, b, cap=BROLL_MAX_HOLD_MS)
        sal = _clamp01((b - a) / 6000.0 + (_MOVE_SALIENCE_BONUS if moving else 0.0))
        label = (c.get("subject_focus") or ("held shot" if not moving else mv.replace("_", " ")))
        out.append(Anchor(
            ts_ms=(ha + hb) // 2, start_ms=ha, end_ms=hb,
            kind=("move" if moving else "hold"), affordance=AFF_BROLL,
            salience=sal, region=c.get("region"),
            text=(f"{mv.replace('_', ' ')} \u00b7 {label}" if moving else label),
        ))
    for e in (perception.get("events") or []):
        if e.get("change") != "holds":
            continue
        if e.get("actor") in primary:        # held framing on the dominant subject
            continue
        a, b = int(e.get("start_ms", 0)), int(e.get("end_ms", 0))
        if b - a < MIN_HOLD_MS or any(_overlap(a, b, x0, x1) > 0 for x0, x1 in seen):
            continue
        ha, hb = _hold_core(a, b, cap=BROLL_MAX_HOLD_MS)
        out.append(Anchor(
            ts_ms=(ha + hb) // 2, start_ms=ha, end_ms=hb, kind="hold", affordance=AFF_BROLL,
            salience=_clamp01((b - a) / 6000.0), actor=e.get("actor"),
            region=e.get("region"), text=(e.get("description") or "held shot")[:200],
        ))
    return out


def _insert_anchors(perception: dict) -> List[Anchor]:
    """Discrete beats worth an insert: reveals / entrances / exits / on-screen
    graphics / environment changes -> overlay anchors. An insert marks an ONSET
    (the reveal, the graphic appearing), so the handle is start-anchored and
    bounded -- a persistent watermark must not become a clip-length card."""
    def _ins(a: int) -> int:
        return a + MAX_HOLD_MS

    out: List[Anchor] = []
    for e in (perception.get("events") or []):
        ch = e.get("change")
        if ch not in _INSERT_CHANGES or ch == "holds":
            continue
        a, b = int(e.get("start_ms", 0)), int(e.get("end_ms", 0))
        if b <= a:
            continue
        out.append(Anchor(
            ts_ms=a, start_ms=a, end_ms=min(b, _ins(a)), kind=str(ch), affordance=AFF_INSERT,
            salience=0.5, actor=e.get("actor"), region=e.get("region"),
            text=(e.get("description") or ch)[:200], source_id=str(e.get("id", "")),
        ))
    for g in (perception.get("graphic_text_events") or []):
        a, b = int(g.get("start_ms", 0)), int(g.get("end_ms", 0))
        if b <= a:
            continue
        txt = (g.get("text") or g.get("kind") or "graphic")
        out.append(Anchor(
            ts_ms=a, start_ms=a, end_ms=min(b, _ins(a)), kind="graphic", affordance=AFF_INSERT,
            salience=0.55 if g.get("text") else 0.4, text=str(txt)[:200],
        ))
    for ev in (perception.get("environment_events") or []):
        a, b = int(ev.get("start_ms", 0)), int(ev.get("end_ms", 0))
        if b <= a:
            continue
        out.append(Anchor(
            ts_ms=a, start_ms=a, end_ms=min(b, _ins(a)), kind="environment", affordance=AFF_INSERT,
            salience=0.45, text=(ev.get("description") or "environment")[:200],
        ))
    return out


def _primary_subjects(perception: dict) -> set:
    """The on-camera people who ARE the A-roll: anyone who speaks on camera, plus
    anyone the VLM tags as the main subject/host. A static b-roll 'hold' OF one of
    these is just the talking-head framing, not a cutaway."""
    ids = {s.get("subject") for s in (perception.get("speaking") or []) if s.get("subject")}
    for p in (perception.get("persons") or []):
        role = (p.get("role") or "").lower()
        if any(k in role for k in ("main", "subject", "host", "primary")):
            if p.get("local_id"):
                ids.add(p.get("local_id"))
    return ids


# B-roll kinds that are merely the dominant subject held in frame (vs a deliberate
# move, which can be a legit push/pan worth cutting to).
_BROLL_STATIC_KINDS = {"broll_hold", "hold", ""}


def _focus_names_primary(text: Optional[str], primary: set) -> bool:
    """True when a free-text subject_focus (e.g. 'p1 looking at laptop') names one
    of the primary on-screen people."""
    if not text or not primary:
        return False
    toks = set(re.split(r"[^a-z0-9]+", text.lower()))
    return any((pid or "").lower() in toks for pid in primary)


def _filter_redundant_broll(cutaways: List[dict], perception: dict) -> List[dict]:
    """Self-quieting baseline: drop b-roll cutaways that are a STATIC hold of the
    dominant on-screen subject -- they show nothing beyond the A-roll framing
    (e.g. a podcast's endless 'p1 looking at laptop'). Non-dominant/None subjects
    (true cutaways to objects, environment, a listener) and deliberate MOVES
    survive. Intent-free: a clip with no on-camera speaker has no dominant subject,
    so nothing is dropped (a visual reel keeps everything)."""
    primary = _primary_subjects(perception)
    if not primary:
        return cutaways
    out: List[dict] = []
    for c in cutaways:
        if (
            (c.get("affordance") or "").lower() == "broll"
            and c.get("subject") in primary
            and (c.get("kind") or "").lower() in _BROLL_STATIC_KINDS
        ):
            continue
        out.append(c)
    return out


def _cutaway_anchors(cutaways: List[dict]) -> List[Anchor]:
    """Sparse L2 cutaways -> overlay anchors (reactions, b-roll, inserts)."""
    _KIND_MAP = {
        "reaction": "expression",
        "gaze": "gaze",
        "broll_hold": "hold",
        "broll_move": "move",
        "reveal": "reveal",
        "graphic": "graphic",
        "environment": "environment",
        "interaction": "interaction",
    }
    _AFF_MAP = {
        "reaction": AFF_REACTION,
        "broll": AFF_BROLL,
        "insert": AFF_INSERT,
    }
    out: List[Anchor] = []
    for c in cutaways:
        aff_key = (c.get("affordance") or "").lower()
        affordance = _AFF_MAP.get(aff_key)
        if not affordance:
            continue
        a, b = int(c.get("start_ms", 0)), int(c.get("end_ms", 0))
        if b <= a:
            continue
        kind = (c.get("kind") or aff_key).lower()
        anchor_kind = _KIND_MAP.get(kind, kind)
        if affordance == AFF_BROLL:
            ha, hb = _hold_core(a, b, cap=BROLL_MAX_HOLD_MS)
        elif affordance == AFF_INSERT and kind != "interaction":
            ha, hb = a, min(b, a + MAX_HOLD_MS)
        else:
            ha, hb = a, b
        peak = c.get("peak_ms")
        ts = int(peak) if peak is not None else (ha + hb) // 2
        ts = max(ha, min(ts, hb))
        sal = c.get("salience_hint")
        if sal is None:
            sal = c.get("intensity")
        if sal is None:
            sal = 0.55 if affordance == AFF_INSERT else 0.5
        label = (c.get("label") or kind).strip()
        out.append(Anchor(
            ts_ms=ts, start_ms=ha, end_ms=hb, kind=anchor_kind, affordance=affordance,
            salience=_clamp01(float(sal)), actor=c.get("subject"), text=label[:200],
        ))
    return out


def _overlay_anchors_legacy(perception: dict) -> List[Anchor]:
    """Pre-cutaways L2 artifacts -> overlay anchors (backward compatible)."""
    out: List[Anchor] = []
    out += _reaction_anchors(perception)
    out += _gaze_anchors(perception)
    out += _broll_anchors(perception)
    out += _insert_anchors(perception)
    out += _interaction_anchors(perception)
    return out


def _gaze_anchors(perception: dict) -> List[Anchor]:
    """Held looks that DEPART from the subject's dominant eyeline -> overlay
    reaction anchors. The cutaway value of a glance is in the *shift* (attention
    leaves the baseline), so we first learn each subject's modal direction (by
    total time) and surface only the held departures. This self-quiets on
    interviews (off-camera baseline) and lights up on vlogs (to-camera baseline),
    with no content-type logic."""
    spans = [g for g in (perception.get("gaze") or [])
             if (g.get("direction") or "") and int(g.get("end_ms", 0)) > int(g.get("start_ms", 0))]
    # Per-subject baseline eyeline = the direction holding the most total time.
    by_subj: Dict[str, Dict[str, int]] = {}
    for g in spans:
        d = (g.get("direction") or "").lower()
        dur = int(g.get("end_ms", 0)) - int(g.get("start_ms", 0))
        by_subj.setdefault(g.get("subject"), {})[d] = by_subj.get(g.get("subject"), {}).get(d, 0) + dur
    baseline = {s: max(dd, key=dd.get) for s, dd in by_subj.items()}

    out: List[Anchor] = []
    for g in spans:
        d = (g.get("direction") or "").lower()
        if d == "unsure" or d == baseline.get(g.get("subject")):
            continue                                   # noise, or the baseline eyeline
        a, b = int(g.get("start_ms", 0)), int(g.get("end_ms", 0))
        if b - a < MIN_GAZE_MS:
            continue
        ha, hb = _hold_core(a, b)
        tgt = g.get("target")
        label = "looks " + d.replace("_", " ") + (f" \u00b7 {tgt}" if tgt else "")
        out.append(Anchor(
            ts_ms=(ha + hb) // 2, start_ms=ha, end_ms=hb, kind="gaze",
            affordance=AFF_REACTION, salience=0.5 if tgt else 0.4,
            actor=g.get("subject"), text=label,
        ))
    return out


def _interaction_anchors(perception: dict) -> List[Anchor]:
    """Relational beats (handshake / hug / hand-off / ...) -> overlay insert
    anchors: a discrete people-moment worth cutting to. Plain 'conversation' is
    skipped -- that content already lives in speech."""
    out: List[Anchor] = []
    for it in (perception.get("interactions") or []):
        kind = (it.get("kind") or "").lower()
        if kind in _INTERACTION_SKIP:
            continue
        a, b = int(it.get("start_ms", 0)), int(it.get("end_ms", 0))
        if b <= a:
            continue
        ha, hb = _hold_core(a, b)
        parts = it.get("participants") or []
        label = (it.get("description") or kind or "interaction")
        out.append(Anchor(
            ts_ms=(ha + hb) // 2, start_ms=ha, end_ms=hb, kind="interaction",
            affordance=AFF_INSERT, salience=0.55,
            actor=(parts[0] if parts else None),
            text=(f"{kind} \u00b7 {label}" if kind else label)[:200],
            source_id=str(it.get("id", "")),
        ))
    return out


# --- Non-speech audio events (the generic detection-gap closer) ----------------
# A moment is a non-verbal audio event when sound is PRESENT but speech is ABSENT
# -- a universal physical property (no "is it a laugh?" classifier). Laughter,
# applause, a gasp, a door slam, a music sting all qualify; the VLM may name it
# later, but the anchor exists on physics alone.
MIN_AUDIO_EVENT_MS = 500       # shorter bursts are clicks/noise, not a usable beat
_AUDIO_EVENT_MERGE_MS = 300    # bridge tiny dips within one event
_SPEECH_PAD_MS = 200           # ignore energy hugging speech edges (on/offset transients)


def _audio_event_anchors(audio: dict, sentences: List[dict], duration_ms: int) -> List[Anchor]:
    """Audible NON-SPEECH moments from the RMS envelope minus the speech mask.
    Pure signal: a contiguous run that is loud (relative to the clip's own speech
    level) yet covered by no spoken words (the sound is the point). Generic -- no
    laugh/applause labels, just 'audible, not speech'."""
    rms = audio.get("rms_db") or []
    hop = int(audio.get("prosody_hop_ms") or 0)
    if not rms or hop <= 0:
        return []
    n = len(rms)

    # Speech mask (padded) over the envelope grid.
    speech = [False] * n
    for s in sentences:
        a = int(s.get("src_in_ms", s.get("raw_in_ms", 0))) - _SPEECH_PAD_MS
        b = int(s.get("src_out_ms", s.get("raw_out_ms", 0))) + _SPEECH_PAD_MS
        for i in range(max(0, a // hop), min(n, b // hop + 1)):
            speech[i] = True

    srt = sorted(rms)
    floor = srt[int(0.20 * (n - 1))]                       # quiet/noise level
    sp_vals = [rms[i] for i in range(n) if speech[i]]
    speech_level = (sorted(sp_vals)[len(sp_vals) // 2] if sp_vals
                    else srt[int(0.90 * (n - 1))])          # typical speech loudness
    if speech_level <= floor:
        return []
    thr = floor + 0.5 * (speech_level - floor)             # clearly audible, self-calibrated

    # Contiguous loud + non-speech runs, with tiny gaps bridged.
    runs: List[List[int]] = []
    i = 0
    while i < n:
        if rms[i] >= thr and not speech[i]:
            j = i
            gap = 0
            k = i
            while k < n:
                if rms[k] >= thr and not speech[k]:
                    j = k
                    gap = 0
                elif speech[k]:
                    break
                else:
                    gap += hop
                    if gap > _AUDIO_EVENT_MERGE_MS:
                        break
                k += 1
            runs.append([i, j])
            i = k
        else:
            i += 1

    out: List[Anchor] = []
    span = max(1.0, speech_level - floor)
    for a_i, b_i in runs:
        a, b = a_i * hop, (b_i + 1) * hop
        if b - a < MIN_AUDIO_EVENT_MS:
            continue
        ha, hb = _hold_core(a, b)
        loud = _mean(rms, a_i, b_i + 1)
        sal = _clamp01(0.35 + 0.5 * ((loud - floor) / span))   # louder vs speech -> higher
        out.append(Anchor(
            ts_ms=(ha + hb) // 2, start_ms=ha, end_ms=hb, kind="audio_event",
            affordance=AFF_REACTION, salience=sal,
            text="audible (non-speech)",
        ))
    return out


def _vlm_quality(events: List[dict], start_ms: int, end_ms: int) -> Optional[float]:
    scores = [int(q.get("score", 0)) for q in events
              if q.get("score") is not None
              and _overlap(int(q.get("start_ms", 0)), int(q.get("end_ms", 0)), start_ms, end_ms) > 0]
    if not scores:
        return None
    return (sum(scores) / len(scores) - 1.0) / 4.0


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def gather_anchors(
    *, duration_ms: int, dialogue: Optional[dict] = None,
    perception: Optional[dict] = None, motion: Optional[dict] = None,
    audio: Optional[dict] = None,
) -> List[Anchor]:
    """Every editable moment in a clip as one time-sorted anchor list, drawn
    from all stored tracks. Best-effort: missing tracks simply contribute no
    anchors. The basis for coverage (and therefore 'never touch raw')."""
    dialogue = dialogue or {}
    perception = perception or {}
    motion = motion or {}
    audio = audio or {}
    quality = perception.get("take_quality_events") or []
    sentences = dialogue.get("sentence") or dialogue.get("topic") or []

    anchors: List[Anchor] = []
    anchors += _speech_anchors(sentences, quality)
    anchors += _action_anchors(perception, motion, quality)
    cutaways = perception.get("cutaways") or []
    if cutaways:
        cutaways = _filter_redundant_broll(cutaways, perception)
        anchors += _cutaway_anchors(cutaways)
    else:
        anchors += _overlay_anchors_legacy(perception)
    anchors += _audio_event_anchors(audio, sentences, duration_ms)

    # Reactions earn their salience from context (action / intensity / listening),
    # uniformly across the legacy and cutaways paths.
    _apply_reaction_warrant(anchors, perception)

    # Clamp to the clip and drop degenerate spans.
    for a in anchors:
        a.start_ms = max(0, a.start_ms)
        if duration_ms:
            a.end_ms = min(duration_ms, a.end_ms)
            a.ts_ms = max(a.start_ms, min(a.ts_ms, a.end_ms))
    anchors = [a for a in anchors if a.end_ms > a.start_ms]
    anchors.sort(key=lambda a: (a.start_ms, a.ts_ms))
    return anchors
