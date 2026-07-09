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

logger = logging.getLogger(__name__)

# Bump to force footage_trees cache rebuilds when this module's projection or
# ladder-synthesis logic changes, independent of TREE_VERSION (which governs
# the shared moment-tree shape in footage_map.py).
# v2: junk cuts are no longer dropped -- carried through (labeled) with their
# continuity block, per cuts_v3_continuity.plan.md.
CUTRECORD_MAP_VERSION = 2

# broad -> sharp, matching footage_map._LEVEL_NAMES; the same five band
# centers the (now-retired) hero-cut ladders used to zoom at.
_LEVELS = ("broad", "calm", "balanced", "tight", "sharp")
_BAND_ENERGIES = (0.1, 0.3, 0.5, 0.7, 0.9)

# Mirrors cuts-v3-view.tsx SPEECH_TRIM_MAX: even at max energy, only shave this
# fraction of a speech cut's removable dead-air/filler budget, so a beat never
# feels clipped. Kept identical to the frontend dial so the ladder's sharpest
# rung matches what the editor's dial shows at energy 1.
_SPEECH_TRIM_MAX = 0.85

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

def _video_rung(row: Dict[str, Any], energy: float, level: str, score: float) -> Dict[str, Any]:
    """One video rung: anchor-protected negative padding toward ``hero_ts_ms``,
    clamped to ``pace.min_ms`` -- the exact math of cuts-v3-view.tsx's
    ``tightenedSpan`` (energy 0 = full grounded span, energy 1 = the tightest
    anchor-safe inset), evaluated at this rung's band-center energy."""
    s, e = int(row["src_in_ms"]), int(row["src_out_ms"])
    natural = e - s
    pace = row.get("pace") or {}
    raw_min = pace.get("min_ms")
    min_dur = min(int(raw_min), natural) if raw_min is not None else natural
    target = round(natural - energy * (natural - min_dur))
    if target >= natural or target <= 0:
        in_ms, out_ms = s, e
    else:
        hero = row.get("hero_ts_ms")
        hero = min(max(int(hero), s), e) if hero is not None else (s + e) // 2
        in_ms = round(hero - target / 2)
        out_ms = in_ms + target
        if in_ms < s:
            in_ms, out_ms = s, s + target
        if out_ms > e:
            out_ms, in_ms = e, e - target
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
    """The people on screen for this cut: the diarized speaker (voice identity)
    plus the pass-2 appearance fingerprints (characteristics) so the brain can
    recognise the same person across cuts by description, not just voice id."""
    chars = row.get("characteristics") or []
    speaker = row.get("speaker")
    if not speaker and not chars:
        return []
    return [{
        "person_id": speaker,
        "voice_speaker_id": speaker,
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
        "speaker": row.get("speaker"),
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
