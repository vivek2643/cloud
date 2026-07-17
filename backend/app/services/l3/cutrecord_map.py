"""
Cuts v3 (``cut_records``) -> clip-tree projection for the agentic editor.

``footage_map.build_clip_tree`` expects one cut dict per beat, each owning a
broad..sharp zoom LADDER (the shape the retired hero-cut substrate used to
hand it -- see cuts_v3_to_brain.plan.md and cleanup.plan.md B2). Cuts v3's
``cut_records`` are the current source of truth (what the Cuts tab reads),
but they carry no ladder -- just one span (``src_in_ms``/``src_out_ms``), a
``hero_ts_ms`` anchor, and a ``pace`` envelope (``min_ms``/``natural_ms``/
``max_ms``/``levels``/``remove_spans``). This module is the bridge: it maps
each ``cut_record`` row to the exact cut-dict shape ``build_clip_tree``
consumes, synthesizing the ladder deterministically in code -- mirroring the
frontend energy dial's own math (``cuts-v3-view.tsx``'s ``tightenedSpan``/
``chosenRemoveSpans``) -- so the brain's tightening matches what the editor
shows. No LLM numbers involved anywhere in this module; code owns every
derived value.

See cuts_v3_to_brain.plan.md.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

# v4_cluster_tree_cuts.plan.md section 5: the SAME per-event floors
# v4_segment.py's own _broad_window_for_event uses, so resolve_cluster's
# broadest per-event window agrees with how the cluster's own src_in_ms/
# src_out_ms were computed at ingest time. Aliased (not the plan's own
# _FOLLOW_THROUGH_FLOOR_MS/_LEAD_FLOOR_MS below, which are a DIFFERENT,
# whole-cut-level asymmetry knob, unrelated to this per-event one).
from app.services.l3.v4_segment_params import (
    CLUSTER_PRUNE_GATE as _CLUSTER_PRUNE_GATE,
    FOLLOW_THROUGH_FLOOR_MS as _EVENT_FOLLOW_THROUGH_FLOOR_MS,
    RUN_UP_FLOOR_MS as _EVENT_RUN_UP_FLOOR_MS,
)

logger = logging.getLogger(__name__)

# Bump to force footage_trees cache rebuilds when this module's projection or
# ladder-synthesis logic changes, independent of TREE_VERSION (which governs
# the shared moment-tree shape in footage_map.py).
# v2: junk cuts are no longer dropped -- carried through (labeled) with their
# continuity block, per cuts_v3_continuity.plan.md.
# v3: cluster ladder gains the rising salience gate (_prune_events) -- a dense
# cluster now compresses to its salient few as energy rises instead of keeping
# every event's window, so cached rungs must rebuild.
CUTRECORD_MAP_VERSION = 3

# broad -> sharp, matching footage_map._LEVEL_NAMES; the same five band
# centers the (now-retired) hero-cut ladders used to zoom at.
_LEVELS = ("broad", "calm", "balanced", "tight", "sharp")
_BAND_ENERGIES = (0.1, 0.3, 0.5, 0.7, 0.9)

# cuts_v4_segmentation.plan.md section 6: small floors so a punchy "before"/
# "after" rung never clips the impact/lead it's built around -- always lands a
# hair past/before the salience peak (itself coarse, ~100ms audio hops).
_FOLLOW_THROUGH_FLOOR_MS = 300
_LEAD_FLOOR_MS = 300

# Mirrors cuts-v3-view.tsx SPEECH_TRIM_MAX: even at max energy, only shave this
# fraction of a speech cut's removable dead-air/filler budget. Kept identical
# to the frontend dial so the ladder's sharpest rung matches what the editor's
# dial shows at energy 1. edso_pacing_audit_timing.plan.md SS7: raised from
# 0.85 to 1.0 -- sharp is the max-tight rung, so the only remaining ceiling on
# how much removable dead-air it shaves is the band's own energy value (see
# _BAND_ENERGIES), not an extra artificial cap layered on top of it.
_SPEECH_TRIM_MAX = 1.0

# said -> person (who's talking), done -> person (an action performed by
# someone), shown -> object (b-roll/display, no performed action). cut_records
# carries no explicit `subject` column; this is the deterministic stand-in
# until the ingest LLM output grows one (see plan "open questions").
_SUBJECT_BY_CHANNEL = {"said": "person", "done": "person", "shown": "object"}


def _pg_conn():
    import psycopg
    from app.config import get_settings
    return psycopg.connect(get_settings().database_url, autocommit=True)


# --------------------------------------------------------------------------
# Cache signature (Phase 2): ingest_run_id + a content hash of the file's rows
# --------------------------------------------------------------------------

def signatures_for(file_ids: List[str], run_id: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Public content signature per file for the ``cut_records`` source: the
    covering ingest run id + row count, so ``footage_map.get_trees`` busts its
    cache on re-ingest. None when the file has no cut_records in the resolved
    run yet.

    ``run_id`` pins the thread's covering run (migration 028); None resolves the
    latest covering run live. Because the signature embeds the run id, a pinned
    thread and an unpinned one key the ``footage_trees`` cache independently --
    no cross-contamination. Counts ALL rows (junk included -- cuts_v3_continuity
    .plan.md keeps junk in the brain's map, so a junk-only edit must also bust
    the cache)."""
    out: Dict[str, Optional[str]] = {fid: None for fid in file_ids}
    if not file_ids:
        return out
    from app.services.l3 import cuts_v3_read
    run_id = run_id or cuts_v3_read.latest_run_for_files(file_ids)
    if run_id is None:
        return out
    counts: Dict[str, int] = {}
    for row in cuts_v3_read.rows_for_run(run_id, file_ids):
        counts[row["file_id"]] = counts.get(row["file_id"], 0) + 1
    for fid, n in counts.items():
        payload = json.dumps({"run": run_id, "n": n, "v": CUTRECORD_MAP_VERSION}, sort_keys=True)
        out[fid] = hashlib.sha1(payload.encode()).hexdigest()
    return out


# --------------------------------------------------------------------------
# Ladder synthesis (Fork A, LOCKED): mirrors the frontend energy dial exactly.
# --------------------------------------------------------------------------

def _symmetric_rung(s: int, e: int, target: int, anchor_ms: Optional[int]) -> Tuple[int, int]:
    """Anchor-protected negative padding toward ``anchor_ms`` -- the original
    (V3, hero_ts_ms-centered) shrink, reused for a V4 cut whose salience has
    no directional shape (kind="span" excluded -- see _video_rung) with the
    anchor swapped to the salience peak."""
    anchor = min(max(int(anchor_ms), s), e) if anchor_ms is not None else (s + e) // 2
    in_ms = round(anchor - target / 2)
    out_ms = in_ms + target
    if in_ms < s:
        in_ms, out_ms = s, s + target
    if out_ms > e:
        out_ms, in_ms = e, e - target
    return in_ms, out_ms


def _shrink_fraction(natural: int, min_dur: int, target: int) -> float:
    """0 at energy 0 (target == natural) -> 1 at max tightening (target ==
    min_dur) -- how far through the shrink this rung's energy sits, used to
    interpolate the SLOW-moving edge of a before/after rung. (Not simply
    ``1 - target/natural``: that only reaches 1.0 at target==0, so a rung
    whose min_dur floor sits well above zero would never fully settle onto
    its floor edge, letting the peak drift outside the window at max energy.)"""
    span = natural - min_dur
    if span <= 0:
        return 0.0
    return max(0.0, min(1.0, (natural - target) / span))


def _before_rung(s: int, e: int, target: int, min_dur: int, peak_ms: Optional[int], natural: int) -> Tuple[int, int]:
    """shape="before" (point): the OUT edge anchors near peak +
    follow-through-floor and barely moves (it's already close to the peak);
    the IN edge absorbs the shrink. As energy rises the window ends closer to
    the impact with a shrinking lead-in."""
    peak = min(max(int(peak_ms), s), e) if peak_ms is not None else (s + e) // 2
    out_floor = min(e, peak + _FOLLOW_THROUGH_FLOOR_MS)
    out_ms = round(e - _shrink_fraction(natural, min_dur, target) * (e - out_floor))
    in_ms = out_ms - target
    if in_ms < s:
        in_ms, out_ms = s, s + target
    # Never clip the peak itself, whatever the floors/rounding worked out to.
    if out_ms < peak:
        out_ms, in_ms = min(e, peak), min(e, peak) - target
    return in_ms, out_ms


def _after_rung(s: int, e: int, target: int, min_dur: int, peak_ms: Optional[int], natural: int) -> Tuple[int, int]:
    """shape="after" (point): the mirror of _before_rung -- the IN edge
    anchors near peak - lead-floor and barely moves; the OUT edge (tail/
    settle) absorbs the shrink."""
    peak = min(max(int(peak_ms), s), e) if peak_ms is not None else (s + e) // 2
    in_floor = max(s, peak - _LEAD_FLOOR_MS)
    in_ms = round(s + _shrink_fraction(natural, min_dur, target) * (in_floor - s))
    out_ms = in_ms + target
    if out_ms > e:
        out_ms, in_ms = e, e - target
    # Never clip the peak itself, whatever the floors/rounding worked out to.
    if in_ms > peak:
        in_ms, out_ms = max(s, peak), max(s, peak) + target
    return in_ms, out_ms


def _span_rung(s: int, e: int, target: int, span_ms: Optional[List[Any]]) -> Tuple[int, int]:
    """salience.kind="span" (camera move): trim the head (drop the slow
    ramp-in), keep the settle -- OUT stays at the cut's own natural out
    (already the move's settle, by construction of v4_segment._camera_move_
    core); as energy rises IN moves toward the move's dynamic core start,
    never past it (never end mid-move on the other side either, since OUT
    never moves)."""
    core_s = max(s, min(int(span_ms[0]), e)) if span_ms else s
    in_ms = max(core_s, e - target)
    return in_ms, e


def _event_anchor(event: Dict[str, Any]) -> int:
    return int(event.get("peak_ms") or 0)


def _event_score(event: Dict[str, Any]) -> float:
    return float(event.get("score") or 0.0)


def _prune_events(events: List[Dict[str, Any]], energy: float) -> List[Dict[str, Any]]:
    """The rising salience gate (v4_segment_params.CLUSTER_PRUNE_GATE): keep only
    events whose salience clears ``energy * GATE * max_score`` -- weak/connective/
    noise events fall away first as the dial rises, so the survivors compress and
    separate. Relative to the cluster's OWN strongest event, so it's generic (a
    lone peak keeps one; several comparable hits keep several; an all-weak
    monotonous span keeps just its best). Never empty -- the single strongest
    event always survives. Energy 0 keeps everything (broad = whole moment)."""
    if not events:
        return events
    max_score = max(_event_score(e) for e in events)
    if max_score <= 0.0:
        return events
    thr = energy * _CLUSTER_PRUNE_GATE * max_score
    survivors = [e for e in events if _event_score(e) >= thr]
    return survivors or [max(events, key=_event_score)]


def resolve_cluster(
    events: List[Dict[str, Any]], cluster_s: int, cluster_e: int, energy: float,
    *, prune: bool = True,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Multi-event per-level resolution (v4_cluster_tree_cuts.plan.md section
    5, "the heart"): every event's own window half-width slides between
    "reach exactly halfway to its neighbor (or the cluster's own outer
    bound)" at energy 0 and "this event's own tight floor" at energy 1 --
    the SAME per-event floors (_EVENT_RUN_UP_FLOOR_MS/_EVENT_FOLLOW_THROUGH_
    FLOOR_MS) v4_segment.py's _broad_window_for_event uses. Adjacent windows
    that still touch fuse into one piece. Because each event's outward
    "capacity" is defined as exactly half the gap to its neighbor, energy 0
    windows always touch their neighbor exactly at the midpoint -- callers
    still special-case the "broad" rung to the cluster's own outer bound
    directly (see _video_rung) rather than relying on that limit, since the
    ladder never actually samples energy 0. Window half-widths shrink
    monotonically as energy rises, so the piece count this produces is
    monotonically non-decreasing (windows can only separate further, never
    re-merge) -- shape (before/after) is deliberately NOT threaded into the
    per-event asymmetry here (an internal valley between two hits has no
    clear "build/reveal" reading); each event's own natural
    run-up/follow-through floor asymmetry (500 > 300, "favor a beat past the
    peak over dwelling before it") still applies uniformly.

    Returns (pieces, removed) -- pieces are the kept spans in time order,
    removed is their complement inside [cluster_s, cluster_e]. When ``prune`` is
    set (the RENDERED ladder -- the energy dial the timeline actually plays),
    the rising salience gate (_prune_events) drops weak events as energy rises,
    so a dense/noisy cluster compresses to its genuinely-salient few (dropped
    regions become part of ``removed``) rather than keeping every event's window
    forever. ``prune=False`` keeps every event -- used only for the brain's
    piece ADDRESSING (footage_map.piece_breakdown / resolve_piece), where every
    salience event must stay individually nameable and placeable regardless of
    where the dial would gate it."""
    src = _prune_events(events, energy) if prune else events
    ordered = sorted(src, key=_event_anchor)
    n = len(ordered)
    windows: List[Tuple[int, int]] = []
    for i, ev in enumerate(ordered):
        peak = _event_anchor(ev)
        left_cap = float(peak - cluster_s) if i == 0 else (peak - _event_anchor(ordered[i - 1])) / 2.0
        right_cap = float(cluster_e - peak) if i == n - 1 else (_event_anchor(ordered[i + 1]) - peak) / 2.0
        left_cap = max(left_cap, _EVENT_RUN_UP_FLOOR_MS)
        right_cap = max(right_cap, _EVENT_FOLLOW_THROUGH_FLOOR_MS)
        w_in = _EVENT_RUN_UP_FLOOR_MS + (1.0 - energy) * (left_cap - _EVENT_RUN_UP_FLOOR_MS)
        w_out = _EVENT_FOLLOW_THROUGH_FLOOR_MS + (1.0 - energy) * (right_cap - _EVENT_FOLLOW_THROUGH_FLOOR_MS)
        in_ms = max(cluster_s, round(peak - w_in))
        out_ms = min(cluster_e, round(peak + w_out))
        windows.append((min(in_ms, peak), max(out_ms, peak)))

    pieces: List[Tuple[int, int]] = []
    for a, b in windows:
        if pieces and a <= pieces[-1][1]:
            pieces[-1] = (pieces[-1][0], max(pieces[-1][1], b))
        else:
            pieces.append((a, b))

    removed: List[Tuple[int, int]] = []
    cur = cluster_s
    for a, b in pieces:
        if a > cur:
            removed.append((cur, a))
        cur = max(cur, b)
    if cur < cluster_e:
        removed.append((cur, cluster_e))
    return pieces, removed


def _cluster_rung(row: Dict[str, Any], events: List[Dict[str, Any]], energy: float,
                  level: str, score: float) -> Dict[str, Any]:
    """The multi-event ("spans", like a speech rung) branch of _video_rung --
    section 5's "Broad: windows wide -> all touch -> 1 piece = whole
    cluster" is enforced directly (not relied on as a limit of the energy-0
    interpolation, since the ladder only ever samples energy 0.1)."""
    s, e = int(row["src_in_ms"]), int(row["src_out_ms"])
    if level == "broad":
        pieces: List[Tuple[int, int]] = [(s, e)]
    else:
        pieces, _removed = resolve_cluster(events, s, e, energy)
        pieces = [(max(s, a), min(e, b)) for a, b in pieces if min(e, b) > max(s, a)]
        if not pieces:
            pieces = [(s, e)]
    return {
        "level": level,
        "spans": [{"in_ms": a, "out_ms": b} for a, b in pieces],
        "in_ms": pieces[0][0], "out_ms": pieces[-1][1],
        "play_ms": sum(b - a for a, b in pieces),
        "score": score,
    }


def _video_rung(row: Dict[str, Any], energy: float, level: str, score: float) -> Dict[str, Any]:
    """One video rung, clamped to ``pace.min_ms`` -- energy 0 = full grounded
    span, energy 1 = the tightest safe inset, evaluated at this rung's
    band-center energy. V3 (no ``salience.kind``) keeps the original
    hero_ts_ms-centered symmetric shrink exactly (cuts-v3-view.tsx's
    ``tightenedSpan``); a V4 cut (``salience.kind`` present) collapses toward
    the salience anchor instead, asymmetrically per shape -- see
    cuts_v4_segmentation.plan.md section 6. `shape` only ever picks the
    before/after asymmetry; an unrecognized/missing shape (including VLM
    shape="none", by the plan's arbitration rule) falls through to the same
    symmetric-around-peak math as "both"/"center" -- `kind` alone (always
    code-owned, never influenced by the VLM) decides point vs span vs none.

    v4_cluster_tree_cuts.plan.md: a cluster with 2+ events (salience.events)
    dispatches to _cluster_rung instead -- a multi-piece ("spans") rung, like
    speech. A cluster of exactly one event (or a V3/legacy row with no
    "events" at all) falls straight through to the untouched single-window
    math below -- byte-identical to the pre-cluster V4/V3 output."""
    salience = row.get("salience") or {}
    events = salience.get("events") or []
    if len(events) > 1:
        return _cluster_rung(row, events, energy, level, score)

    s, e = int(row["src_in_ms"]), int(row["src_out_ms"])
    natural = e - s
    pace = row.get("pace") or {}
    raw_min = pace.get("min_ms")
    min_dur = min(int(raw_min), natural) if raw_min is not None else natural
    target = round(natural - energy * (natural - min_dur))
    if target >= natural or target <= 0:
        in_ms, out_ms = s, e
    else:
        kind = salience.get("kind")
        if kind is None:
            in_ms, out_ms = _symmetric_rung(s, e, target, row.get("hero_ts_ms"))
        elif kind == "span":
            in_ms, out_ms = _span_rung(s, e, target, salience.get("span_ms"))
        else:
            shape = salience.get("shape") or "center"
            peak_ms = salience.get("peak_ms")
            if shape == "before":
                in_ms, out_ms = _before_rung(s, e, target, min_dur, peak_ms, natural)
            elif shape == "after":
                in_ms, out_ms = _after_rung(s, e, target, min_dur, peak_ms, natural)
            else:
                in_ms, out_ms = _symmetric_rung(s, e, target, peak_ms)
    return {"level": level, "in_ms": int(in_ms), "out_ms": int(out_ms),
            "play_ms": int(out_ms - in_ms), "score": score}


def _chosen_remove_spans(spans: List[Tuple[int, int]], energy: float) -> List[Tuple[int, int]]:
    """Mirrors cuts-v3-view.tsx's ``chosenRemoveSpans``: the longest removable
    dead-air/filler spans first, up to ``energy * _SPEECH_TRIM_MAX`` of the
    total removable budget."""
    if not spans or energy <= 0:
        return []
    total = sum(b - a for a, b in spans)
    target = energy * _SPEECH_TRIM_MAX * total
    by_len = sorted(spans, key=lambda sp: sp[1] - sp[0], reverse=True)
    chosen: List[Tuple[int, int]] = []
    acc = 0
    for a, b in by_len:
        if acc >= target:
            break
        chosen.append((a, b))
        acc += b - a
    return chosen


def _kept_segments(in_ms: int, out_ms: int, removed: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Mirrors cuts-v3-view.tsx's ``keptSegments``: subtract the removed spans
    from [in_ms, out_ms] -> the ordered kept segments. Never empty."""
    rs = sorted((max(a, in_ms), min(b, out_ms)) for a, b in removed)
    rs = [(a, b) for a, b in rs if b > a]
    segs: List[Tuple[int, int]] = []
    cur = in_ms
    for a, b in rs:
        if a > cur:
            segs.append((cur, a))
        cur = max(cur, b)
    if cur < out_ms:
        segs.append((cur, out_ms))
    return segs or [(in_ms, out_ms)]


def _speech_rung(row: Dict[str, Any], energy: float, level: str, score: float) -> Dict[str, Any]:
    """One speech rung: the OUTER span stays full (words are the point -- no
    anchor-protected inset), but interior dead-air/fillers are progressively
    shaved via ``pace.remove_spans`` -> a multi-span jump-cut keep-list, exactly
    matching the frontend dial's speech behavior at this rung's energy."""
    s, e = int(row["src_in_ms"]), int(row["src_out_ms"])
    pace = row.get("pace") or {}
    remove = [(int(sp[0]), int(sp[1])) for sp in (pace.get("remove_spans") or [])]
    chosen = _chosen_remove_spans(remove, energy)
    kept = _kept_segments(s, e, chosen) if chosen else [(s, e)]
    return {
        "level": level,
        "spans": [{"in_ms": a, "out_ms": b} for a, b in kept],
        "in_ms": kept[0][0], "out_ms": kept[-1][1],
        "play_ms": sum(b - a for a, b in kept),
        "score": score,
    }


def synth_ladder(row: Dict[str, Any], score: float) -> List[Dict[str, Any]]:
    """The 5-rung broad..sharp ladder for one ``cut_record`` row, synthesized
    deterministically from its OWN span/anchor/pace -- never an LLM number.
    Video rungs zoom via anchor-protected negative padding; speech rungs stay
    full-span and thread ``pace.remove_spans`` into a keep-list instead."""
    if row.get("kind") == "speech":
        return [_speech_rung(row, en, lvl, score) for en, lvl in zip(_BAND_ENERGIES, _LEVELS)]
    return [_video_rung(row, en, lvl, score) for en, lvl in zip(_BAND_ENERGIES, _LEVELS)]


# --------------------------------------------------------------------------
# cut_record row -> the cut-dict shape build_clip_tree consumes
# --------------------------------------------------------------------------

def _legacy_score_for(row: Dict[str, Any]) -> float:
    """Fallback rank key for a cut ingested BEFORE deterministic total_quality
    existed (migration 031): the cut's OWN duration + how centered its anchor
    sits in its span. Re-ingested runs carry a real total_quality and never
    reach this."""
    s, e = int(row["src_in_ms"]), int(row["src_out_ms"])
    dur = max(1, e - s)
    hero = row.get("hero_ts_ms")
    if hero is None:
        anchor_frac = 0.5
    else:
        mid = (s + e) / 2.0
        anchor_frac = 1.0 - min(1.0, abs(int(hero) - mid) / max(1.0, dur / 2.0))
    dur_frac = min(1.0, dur / 8000.0)
    return round(0.5 * anchor_frac + 0.5 * dur_frac, 3)


def _score_for(row: Dict[str, Any]) -> float:
    """The cut's deterministic rank score: the real total_quality stamped at
    ingest (post.compute_total_quality), or the legacy geometric fallback for
    rows predating it."""
    tq = row.get("total_quality")
    return float(tq) if tq else _legacy_score_for(row)


def _people_for(row: Dict[str, Any]) -> List[dict]:
    """The Tier-1 raw appearance detail for this cut's speaker (when bound):
    the global person id plus the pass-2 appearance fingerprints
    (characteristics) so an on-demand inspection can recognise them by
    description, not just id. PIC/SND rendering itself reads speaker_person/
    visible_persons directly off the moment (see footage_map.py) -- this is
    only the richer per-person detail for `inspect_moment`-style lookups."""
    chars = row.get("characteristics") or []
    speaker_person = row.get("speaker_person")
    if not speaker_person and not chars:
        return []
    return [{
        "person_id": speaker_person,
        "voice_speaker_id": speaker_person,
        "on_camera": row.get("on_camera"),
        "characteristics": chars,
    }]


def _audio_mute_for(channel: str, row: Dict[str, Any]) -> Tuple[Optional[str], bool, List[str]]:
    """The video default-mute rule: a video (done/shown) cut whose pace envelope
    says its source sound ISN'T worth keeping (``natural_sound`` false) is muted
    by default -- matching the old substrate's "b-roll shouldn't drag in stray
    audio" policy, using cuts v3's own worth-keeping judgment as the signal
    instead of a raw speech/silence re-analysis. Said cuts leave audio/mute at
    their hero-cut default (unset/False) -- their audio IS the point."""
    if channel == "said":
        return None, False, []
    natural_sound = bool((row.get("pace") or {}).get("natural_sound"))
    if natural_sound:
        return "sound", False, []
    return "silent", True, ["muted"]


def _to_cut_dict(row: Dict[str, Any]) -> Dict[str, Any]:
    channel = row.get("channel") or ("said" if row.get("kind") == "speech" else "shown")
    score = _score_for(row)
    audio, mute, flags = _audio_mute_for(channel, row)
    return {
        "hero_id": row["id"],
        "file_id": row["file_id"],
        "channel": channel,
        # speech | video -- lets the brain (and the pace tag / retime verb) know
        # whether pacing means playback SPEED (video) or dead-air TRIM (speech).
        "kind": row.get("kind"),
        "subject": _SUBJECT_BY_CHANNEL.get(channel, "object"),
        "label": row.get("label") or "",
        "summary": row.get("summary"),
        # voice_first_identity.plan.md: all four code-derived, never LLM-
        # echoed. voice_ids = the global voice(s) heard (Pass 1 word-level
        # diarization + voice clustering); speaker_person/on_camera = the
        # speaker pass's voice->person binding + whether that person is
        # visible in THIS cut; visible_persons = every global person id on
        # screen (per-cut-occurrence face clustering, not one-per-file).
        "voice_ids": row.get("voice_ids") or [],
        "speaker_person": row.get("speaker_person"),
        "visible_persons": row.get("visible_persons") or [],
        "on_camera": row.get("on_camera"),
        "src_in_ms": int(row["src_in_ms"]),
        "src_out_ms": int(row["src_out_ms"]),
        "play_ms": int(row["src_out_ms"]) - int(row["src_in_ms"]),
        "keep_spans": None,
        "score": score,
        # The two deterministic quality scores (post.py) surfaced verbatim so
        # the brain can arrange on them: speech_quality (delivery, camera-
        # independent -> same across simultaneous angles) and total_quality
        # (speech + visual; the on-camera close-up of a beat ranks highest).
        "speech_quality": row.get("speech_quality"),
        "total_quality": row.get("total_quality"),
        "flags": flags,
        "audio": audio,
        "mute": mute,
        "people": _people_for(row),
        "framing": row.get("framing"),
        "quality": row.get("look"),
        # The full pace ENVELOPE (min_ms/natural_ms/max_ms/levels/remove_spans/
        # natural_sound), carried through untouched so the brain sees the pacing
        # ROOM per cut -- video speed levels (cross-clip normalized) or a speech
        # cut's removable dead-air/filler budget -- and `retime` can act on it.
        "pace": row.get("pace") or {},
        "ladder": synth_ladder(row, score),
        # Carried through for Phase 3 (footage_map._annotate_dups reads these
        # directly off the built moment instead of recomputing take groups).
        "take_group_id": row.get("take_group_id"),
        "take_role": row.get("take_role"),
        # cuts_v3_continuity.plan.md: junk is KEPT (labeled), not dropped, so
        # numbering + contiguity stay honest and a junk beat can still be
        # placed deliberately as a bridge. The persisted continuity block
        # (cut_no/of/prev_contiguous/next_contiguous/seam_reason_*) rides
        # straight through -- computed once at ingest, never re-derived.
        "junk": bool(row.get("junk")),
        "junk_reason": row.get("junk_reason"),
        "continuity": row.get("continuity") or {},
        # A plain camera-move phrase (static / pan / tilt / zoom / follow /
        # shaky) so the brain knows how the shot moves without raw signals.
        "camera": row.get("camera") or "unknown",
        # Which outlook group (if any) this cut's audio belongs to --
        # footage_map._annotate_outlook_groups reads this to tag "N angles
        # share this audio", a structural fact, not editorial guidance (the
        # brain still picks picture on its own). DB column stays sync_group_id.
        "sync_group_id": row.get("sync_group_id"),
        # perception_upgrade.plan.md Part C3/D: on-screen text/graphics the
        # model read off the pixels, and this cut's single strongest INSTANT
        # (code-computed, post._salience). "" / {} on a pre-migration cut --
        # these were never actually surfaced here before (footage_map's
        # screen_text/salience tags silently always rendered empty).
        "screen_text": row.get("screen_text") or "",
        "salience": row.get("salience") or {},
        # av_coupling_authoritative.plan.md: this cut's baked, authoritative
        # audio coupling. audio_file_id defaults to this cut's OWN file_id
        # (same-source) for a pre-migration row (DB column is NULL there).
        "audio_file_id": row.get("audio_file_id") or row["file_id"],
        "audio_offset_ms": int(row.get("audio_offset_ms") or 0),
        "audio_align_confidence": row.get("audio_align_confidence"),
    }


# --------------------------------------------------------------------------
# Public: cut_records -> {file_id: [cut_dict, ...]}
# --------------------------------------------------------------------------

def cut_dicts_for_files(file_ids: List[str], run_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
    """``{file_id: [cut_dict, ...]}`` for the given clips, resolved off a
    ``cut_records`` ingest run. ``run_id`` pins the thread's covering run
    (migration 028); None resolves the latest covering run live.

    Junk cuts are KEPT (labeled), not dropped -- cuts_v3_continuity.plan.md:
    the ordered sequence must stay honest for cut_no/of numbering + contiguity,
    and a junk beat is recoverable (the brain can place it deliberately as a
    connective bridge), never silently deleted. The frontend still hides junk
    in its tray by default -- display != what the brain sees. Fail-open: a
    file with no cut_records in the resolved run is simply absent -- no
    fabrication."""
    if not file_ids:
        return {}
    from app.services.l3 import cuts_v3_read
    run_id = run_id or cuts_v3_read.latest_run_for_files(file_ids)
    if run_id is None:
        return {}
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in cuts_v3_read.rows_for_run(run_id, file_ids):
        out.setdefault(row["file_id"], []).append(_to_cut_dict(row))
    return out
