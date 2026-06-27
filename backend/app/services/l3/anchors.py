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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.l3 import vocab

# Affordance buckets (what an editor reaches for). Modality in the feed is just
# the dominant affordance of a segment -- these are filters, not pipelines.
# The closed set lives in vocab.py (single source of truth); we alias the names
# here so the rest of this module reads naturally. There is NO separate
# "behavior" or "listening" affordance: incidental physical business IS action,
# held attention IS a reaction (distinguished by a `kind`/flag, not a bucket).
AFF_SPEECH = vocab.AFF_SPEECH      # a spoken line / answer (sync)
AFF_ACTION = vocab.AFF_ACTION      # see something done/happen (incl. incidental business)
AFF_REACTION = vocab.AFF_REACTION  # see someone respond/attend (incl. listening)
AFF_BROLL = vocab.AFF_BROLL        # see a place/thing/texture (held composition)
AFF_INSERT = vocab.AFF_INSERT      # register a reveal/entrance/graphic

# Hidden-by-default dialogue lexicon flags (off-camera audio).
_OFFCAMERA_FLAGS = ("offscreen", "production_cue")

# A b-roll / hold cutaway is a short handle the editor lays over A-roll, not the
# whole take. A static span can run for minutes (a locked-off interview); we emit
# a representative, bounded slice from its middle so it stays a usable card.
MAX_HOLD_MS = 5000
# B-roll keeps its FULL extent into the cut engine (energy then insets it to a
# per-band core); this is only a sanity guard against a multi-minute locked span.
BROLL_MAX_HOLD_MS = 15000


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


# --- Behavior: incidental physical business from the VLM event timeline -------
# The VLM logs a sequential beat timeline ("p1 gestures", "p1 drinks coffee",
# "p1 leans in"). These are NOT speech (no line) and NOT the sustained
# performance the action content_units capture -- they're the on-camera subject's
# physical business. They were previously DROPPED (events never became cuts). We
# promote them into the ACTION affordance (kept distinct only by kind="behavior")
# so a coffee sip / a lean-in / an emphatic gesture becomes a cuttable action
# shot -- not a separate bucket. Because the VLM emits events sequentially (they
# never overlap), consecutive same-actor beats are stitched into ONE continuous
# behavior (walk -> open -> step out) -- the deterministic continuity grouping
# that replaces a fragile per-event before/after flag.
EVENT_CONTINUITY_GAP_MS = 1200   # same-actor events within this gap = one behavior
BEHAVIOR_MIN_MS = 600            # shorter than this isn't a usable held shot


def _merge_events(events: List[dict]) -> List[dict]:
    """Stitch consecutive SAME-ACTOR events whose inter-gap is short (or that share
    a VLM interaction_id) into one continuous behavior span."""
    usable = [e for e in events
              if int(e.get("end_ms", 0)) > int(e.get("start_ms", 0)) and e.get("actor")]
    usable.sort(key=lambda e: int(e.get("start_ms", 0)))
    groups: List[List[dict]] = []
    for e in usable:
        if groups:
            prev = groups[-1][-1]
            gap = int(e.get("start_ms", 0)) - int(prev.get("end_ms", 0))
            same_actor = e.get("actor") == prev.get("actor")
            same_iid = bool(e.get("interaction_id")) and \
                e.get("interaction_id") == prev.get("interaction_id")
            if same_actor and (gap <= EVENT_CONTINUITY_GAP_MS or same_iid):
                groups[-1].append(e)
                continue
        groups.append([e])
    merged: List[dict] = []
    for g in groups:
        a = min(int(e.get("start_ms", 0)) for e in g)
        b = max(int(e.get("end_ms", 0)) for e in g)
        desc = " \u2192 ".join((e.get("description") or "").strip()
                               for e in g if (e.get("description") or "").strip())
        merged.append({
            "start_ms": a, "end_ms": b, "actor": g[0].get("actor"),
            "description": (desc or g[0].get("description") or "behavior")[:200],
            "region": g[0].get("region"),
            "id": str(g[0].get("id", "")),
        })
    return merged


def _behavior_anchors(perception: dict, motion: dict) -> List[Anchor]:
    """On-camera physical BEHAVIOR from the VLM event timeline -- the business the
    subject does (gesture, sip, lean, handle an object) that is neither a spoken
    line nor a sustained action/performance. Continuity-grouped per actor; salience
    rises with motion under the span so a still pose ranks below an active beat.
    The duration floor + the on-camera actor requirement are the usability gate
    that keeps this coverage, not a flood of micro-twitches."""
    hop = max(1, int((motion or {}).get("hop_ms", 100)))
    energy = (motion or {}).get("action_energy") or []
    out: List[Anchor] = []
    for ev in _merge_events(perception.get("events") or []):
        a, b = int(ev["start_ms"]), int(ev["end_ms"])
        if b - a < BEHAVIOR_MIN_MS:
            continue
        mo = _mean(energy, a // hop, b // hop) if energy else 0.0
        sal = _clamp01(0.45 + 0.55 * mo)
        out.append(Anchor(
            ts_ms=(a + b) // 2, start_ms=a, end_ms=b, kind="behavior",
            affordance=AFF_ACTION, salience=sal, actor=ev.get("actor"),
            region=ev.get("region"), text=ev.get("description") or "behavior",
            source_id=str(ev.get("id", "")),
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


# --- Listening: the held attention shot, derived from the speaking-turn inverse.
# While one person delivers a sustained turn, the OTHER on-camera person(s) are
# listening -- a long, deliberate reaction the VLM rarely logs as its own cutaway
# (it is the absence of an event). We synthesize it from the speaking track so a
# deep-listening shot is available coverage, not something the editor has to dig
# for in raw. This is the cure for "no long reaction shots."
LISTEN_MIN_TURN_MS = 2500    # a turn shorter than this doesn't earn a listening shot
LISTEN_MAX_MS = 8000         # cap the held slice (a long turn yields a usable handle)


def _listening_anchors(perception: dict, duration_ms: int) -> List[Anchor]:
    """A held listening/attention reaction for each on-camera person who is NOT
    speaking during another person's sustained turn. Salience = the turn-length
    warrant (the same scale ``_apply_reaction_warrant`` uses), so a long turn
    earns a full-strength listening shot and a glance does not."""
    speaking = perception.get("speaking") or []
    if not speaking:
        return []
    persons = {p.get("local_id"): p for p in (perception.get("persons") or [])
               if p.get("local_id")}
    on_cam = {s.get("subject") for s in speaking if s.get("subject")}
    for lid, p in persons.items():
        if p.get("frame_region"):
            on_cam.add(lid)
    out: List[Anchor] = []
    for subj, ta, tb in _speech_turns(speaking):
        if tb - ta < LISTEN_MIN_TURN_MS:
            continue
        for lid in sorted(x for x in on_cam if x and x != subj):
            p = persons.get(lid) or {}
            enters, exits = p.get("enters_ms"), p.get("exits_ms")
            if enters is not None and int(enters) > ta:   # not yet on camera
                continue
            if exits is not None and int(exits) < tb:      # already gone
                continue
            a, b = _hold_core(ta, tb, cap=LISTEN_MAX_MS)
            warrant = _clamp01((tb - ta) / _REACTION_LISTEN_FULL_MS)
            out.append(Anchor(
                ts_ms=(a + b) // 2, start_ms=a, end_ms=b, kind="listening",
                affordance=AFF_REACTION, salience=warrant, actor=lid,
                region=p.get("frame_region"), text="listening", flags=["listening"],
            ))
    return out


def _dedup_listening(anchors: List[Anchor]) -> List[Anchor]:
    """Drop a synthesized listening shot when the VLM already logged a reaction
    for the SAME actor over the SAME stretch (its explicit cutaway wins)."""
    logged = [a for a in anchors
              if a.affordance == AFF_REACTION and "listening" not in a.flags]
    kept: List[Anchor] = []
    for a in anchors:
        if "listening" in a.flags:
            span = max(1, a.end_ms - a.start_ms)
            if any(r.actor == a.actor and _overlap(a.start_ms, a.end_ms, r.start_ms, r.end_ms)
                   >= 0.5 * span for r in logged):
                continue
        kept.append(a)
    return kept


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
    anchors += _behavior_anchors(perception, motion)
    cutaways = _filter_redundant_broll(perception.get("cutaways") or [], perception)
    anchors += _cutaway_anchors(cutaways)
    anchors += _audio_event_anchors(audio, sentences, duration_ms)
    # Held listening shots, synthesized from the speaking-turn inverse (deduped
    # against any reaction the VLM already logged for the same actor/stretch).
    anchors += _listening_anchors(perception, duration_ms)
    anchors = _dedup_listening(anchors)

    # Reactions (from the cutaways track + non-speech audio + listening) earn
    # their salience from context (action / intensity / listening).
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
