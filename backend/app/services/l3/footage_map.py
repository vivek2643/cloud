"""
Footage map: the moment-tree + compact "give the model the whole library" index.

The arranger model works like Cursor over a codebase: it needs *total awareness*
of every usable moment across every clip, but the raw hero-cuts feed is 5x
redundant (the same content appears once per energy band) and far too verbose to
drop into a prompt whole. This module is the breakdown + map + retrieval layer:

  * ``build_clip_tree``   turns one clip's cut-set into a list of MOMENTS. Each
    cut now OWNS its zoom ladder (broad turn/run-up -> balanced thought -> tight
    core -> sharp punch), so a moment reads its VARIANTS straight off that ladder
    -- no cross-band geometric re-matching, no guessing which coarse cut a fine
    one belongs to. One anchor band (balanced: "one complete thought per cut") is
    all the tree needs; the rungs carry the rest. A split is just a rung with
    several spans (a jump-cut keep-list), so the legacy ATOMS slot is gone.

  * ``assemble_map``      renders every clip's moments as a compact, complete
    text index (Tier-0: what goes in the prompt) plus the machine-readable
    struct the arranger and compiler consume.

  * ``moment_detail``     returns the full record for one moment on demand
    (Tier-1: what the model retrieves when it wants to inspect a candidate).

A clip's tree is deterministic given its resolved ``cut_records`` ingest run,
so it is keyed by that run's content signature (plus a tree-logic version)
and rebuilt only when the signature changes. Best-effort throughout:
a missing artifact for a file simply yields fewer (or no) moments.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.services.l3.energy import default_energy_for

logger = logging.getLogger(__name__)

# Bump when the moment-tree shape / clustering logic changes so cached trees
# rebuild even if the underlying hero cuts (PARAMS_VERSION) did not.
# v2: every nested level (incl. tight=core) is a selectable variant; atoms are
# emitted only when a moment splits into >= 2 finer sub-units.
# v3: cuts OWN their ladder -- variants read straight off each cut's rungs from
# one anchor band; no geometric re-matching, no atoms (a split is a multi-span
# rung). Moments also carry the people/framing/quality facets.
# v4: the synthetic "moment" entity is retired. A moment IS its spine cut (the
# line / the beat), tagged with every affordance it serves and carrying COVERAGE
# (alternate framings of the same instant). The default placement span is the
# spine's own span; coverage is optional extra angles the brain can reach for.
# v5: FLAT candidates + typed edges. Each cut is an independent moment carrying
# its OUTGOING `relations` (typed, directional) to other moments and a
# `cluster_id` for the connected bundle it belongs to. No forced default
# arrangement -- the brain reads the graph and decides.
# v6: clusters carry a ZOOM LADDER -- a moment (connected bundle) is itself a
# selectable unit that zooms from the whole run of its member cuts (Broad) down
# to just the peak member (Sharp), by neighbour-atom inclusion expanding in time
# out from the peak. Mirrors the per-cut ladder one level up.
# v7: per-clip `default_energy` (genre opens the slider calm/punchy) + each
# cluster carries collapsed `rungs` -- only the meaningfully-distinct zoom steps
# (look-alike ladder rungs merged), so the UI shows real choices not duplicates.
# v8: leaner, less-opinionated resident line -- the brain reads the capture
# PRIMITIVE (what was captured), not the editorial affordance; the narrative
# `role` tag is dropped (no baked intent -> the brain decides at placement); a
# graphic's gist is a short tag shown only when speech doesn't already narrate
# it. Clusters carry the primitive mix too.
# v9: CUTS-V2. The resident line is keyed on the capture CHANNEL.SUBJECT
# (said.person / done.object / shown.graphic) -- the honest substrate, no
# affordance/role. Moment clusters are the deterministic cross-channel
# capture-moments (a shared moment_id from l3.combine). Tier-1 reads the VLM
# `atoms` track (legacy events/cutaways kept as a fallback for un-migrated clips).
# v10: the v1 affordance/primitive/modality vocabulary is fully removed -- cuts
# carry ONLY channel + subject; the map no longer reads or emits affordance/
# primitive/modality. Clusters carry a channel mix + subject mix.
# v11: DONE/SHOWN cuts now carry a full broad..sharp ladder, so their moments
# expose all five variants (not a single flat span) for the brain to zoom.
# v12: video core fractions retuned (Tight 0.4 / Sharp 0.15) -- a wider, sharper
# proportional ladder, so Sharp variants land as bangers.
# v13: windup|payoff split moved to TIGHT (Sharp is a pure banger), so video
# moments' Sharp variant is always the tightest rung (monotonic ladder).
# v14: cross-channel capture-moment CLUSTERS retired -- the tree no longer emits
# `clusters` / per-cut `cluster_id`, and the resident line drops the "· moment:X"
# tag. Grouping is the brain's job (it reads same-clip overlapping timestamps),
# and adjacent same-clip cuts weld at compile time.
# v15: CONTINUITY RUNS -- moments that are back-to-back in source time (one
# uninterrupted stretch of the clip atomized into beats) carry a shared clip-
# local `run_id`. PURELY TEMPORAL + CHANNEL-AGNOSTIC (same clip + adjacent time,
# exactly like the compile-time weld); the cut's category never enters it. The
# resident line shows "· run:rN" so the brain sees continuous footage at DECISION
# time and doesn't fragment it by accident -- a hint, not a rule (it stays free
# to break/intercut a run on purpose).
# v16: moments carry the video-cut AUDIO facet (audio kind + default `mute` for
# stray speech under a shot); the resident line shows a 'muted' tag + the cut's
# PLAY length so the brain sees dropped audio and can pace to a target length.
# v17: audio facet is speech|sound|silent -- a shot mutes ANY uncontrolled audio
# by default (talk OR off-mic/crew/action sound); the line shows 'muted(talk)' vs
# 'muted(sound)' so the brain knows when to `audio:keep` an action's own sound.
# v18: moments carry `take_group_id`/`take_role`, passed through from the cut
# dict when present (cuts-v3 substrate only; legacy hero cuts leave them None) --
# lets `_annotate_dups` read persisted take groups directly instead of
# recomputing them (see cuts_v3_to_brain.plan.md Phase 3).
# v19: moments carry `junk`/`junk_reason`/`continuity` (cuts-v3 substrate only).
# Junk is no longer dropped upstream -- it rides in the map, labeled, so the
# ordered sequence + cut_no/of numbering stay honest and a junk beat stays
# placeable as a deliberate connective bridge; the resident line renders it as
# a terse one-liner instead of the full rich line. Non-junk moments show their
# own `cut:N/of` position + weld marks (↔ weldable / ⋯ hard) to each neighbor,
# read off the persisted `continuity` (see cuts_v3_continuity.plan.md).
TREE_VERSION = 19

# Two moments are one continuous source run when the next starts within this gap
# of where the previous ended (back-to-back in the original footage). Loose
# enough to bridge the small breath between consecutive beats; tight enough that
# genuinely separate beats (a real gap) stay apart.
_RUN_GAP_MS = 500

# Band index -> energy-level name. Band 2 (energy 0.5) is the anchor: one
# complete thought per cut. Lower = wider (whole answer), higher = tighter.
_LEVEL_NAMES = ("broad", "calm", "balanced", "tight", "sharp")
_ANCHOR_BAND = 2


def _level_name(band: int) -> str:
    return _LEVEL_NAMES[max(0, min(band, len(_LEVEL_NAMES) - 1))]


def _band_of(level: Optional[str]) -> int:
    return _LEVEL_NAMES.index(level) if level in _LEVEL_NAMES else _ANCHOR_BAND


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------

def _in(cut: Dict[str, Any]) -> int:
    return int(cut.get("src_in_ms", 0))


def _out(cut: Dict[str, Any]) -> int:
    return int(cut.get("src_out_ms", 0))


def _overlap_ms(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _overlaps_any(a0: int, a1: int, spans: List[Tuple[int, int]]) -> bool:
    return any(_overlap_ms(a0, a1, b0, b1) > 0 for b0, b1 in spans)


def _variant_from_rung(rung: Dict[str, Any], hero_id: Optional[str]) -> Dict[str, Any]:
    """A selectable option for taking this content at one energy, read straight
    off a cut's owned ladder rung. A multi-span rung carries its jump-cut
    keep-list (a split/breath-excised edit), otherwise the span plays whole."""
    spans = rung.get("spans") or []
    if spans:
        in_ms = int(spans[0].get("in_ms", rung.get("in_ms", 0)))
        out_ms = int(spans[-1].get("out_ms", rung.get("out_ms", 0)))
    else:
        in_ms, out_ms = int(rung.get("in_ms", 0)), int(rung.get("out_ms", 0))
    keep = ([[int(s["in_ms"]), int(s["out_ms"])] for s in spans]
            if len(spans) > 1 else None)
    return {
        "level": rung.get("level"),
        "band": _band_of(rung.get("level")),
        "in_ms": in_ms,
        "out_ms": out_ms,
        "play_ms": int(rung.get("play_ms", out_ms - in_ms)),
        "keep_spans": keep,
        "score": float(rung.get("score", 0.0)),
        "hero_id": hero_id,
    }


def _flat_variant(cut: Dict[str, Any]) -> Dict[str, Any]:
    """Synthesize the balanced variant from a cut's flat span when it carries no
    ladder (legacy fallback cuts)."""
    return {
        "level": _level_name(_ANCHOR_BAND), "band": _ANCHOR_BAND,
        "in_ms": _in(cut), "out_ms": _out(cut),
        "play_ms": int(cut.get("play_ms", _out(cut) - _in(cut))),
        "keep_spans": cut.get("keep_spans"),
        "score": float(cut.get("score", 0.0)), "hero_id": cut.get("hero_id"),
    }


def _assign_runs(moments: List[Dict[str, Any]]) -> None:
    """Tag CONTINUOUS SOURCE RUNS in place.

    A run is a maximal chain of moments that are back-to-back in source time --
    each starts within ``_RUN_GAP_MS`` of where the previous one ended (or
    overlaps it) -- i.e. one uninterrupted stretch of the original clip that got
    atomized into several beats. This is a PURELY TEMPORAL fact about the source
    (same clip + adjacent time); it is deliberately CHANNEL-AGNOSTIC -- said,
    done and shown are weighed equally, exactly like the compile-time weld. The
    cut's category never enters the decision.

    Members of a run (>=2) get a shared clip-local ``run_id`` ("r0", "r1", ...)
    plus their position, so the brain reads them as one continuous stretch at
    DECISION time (the raw timestamps alone weren't salient enough) and doesn't
    fragment continuous footage by accident. It stays free to break a run on
    purpose. Lone moments get no tag.

    ``moments`` must be in source (in_ms) order."""
    runs: List[List[Dict[str, Any]]] = []
    cur: Optional[List[Dict[str, Any]]] = None
    cur_end = 0
    for m in moments:
        if cur is not None and int(m["in_ms"]) <= cur_end + _RUN_GAP_MS:
            cur.append(m)
            cur_end = max(cur_end, int(m["out_ms"]))
        else:
            cur = [m]
            cur_end = int(m["out_ms"])
            runs.append(cur)
    ri = 0
    for run in runs:
        if len(run) < 2:
            continue
        rid = f"r{ri}"
        ri += 1
        for pos, m in enumerate(run):
            m["run_id"] = rid
            m["run_pos"] = pos
            m["run_len"] = len(run)


# --------------------------------------------------------------------------
# Tree builder (pure: given a clip header + its anchor-band cut-set)
# --------------------------------------------------------------------------

def build_clip_tree(
    file_id: str,
    header: Dict[str, Any],
    cuts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Turn one clip's anchor-band cut-set into a moment-tree.

    Every cut owns its full zoom ladder, so each moment's variants are read
    directly off the cut's rungs -- no cross-band re-matching. Pure and
    deterministic given the inputs, so it is what the per-clip cache stores.
    """
    fid8 = file_id[:8]
    moments: List[Dict[str, Any]] = []
    ordered = sorted(cuts, key=_in)
    # Spans where SPEECH is on the track. A graphic's gist is redundant when the
    # voiceover already narrates it (the screen-recording case), so we suppress
    # the gist from the resident line whenever speech overlaps the graphic span.
    speech_spans = [
        (_in(c), _out(c)) for c in ordered
        if c.get("channel") == "said"
    ]
    for idx, cut in enumerate(ordered):
        hero_id = cut.get("hero_id")
        variants: Dict[str, Dict[str, Any]] = {}
        for rung in (cut.get("ladder") or []):
            v = _variant_from_rung(rung, hero_id)
            if v["level"]:
                variants[v["level"]] = v
        anchor = variants.get(_level_name(_ANCHOR_BAND))
        if anchor is None:                       # no ladder -> use the flat span
            anchor = _flat_variant(cut)
            variants[anchor["level"]] = anchor

        moments.append({
            "moment_id": f"{fid8}:m{idx:02d}",
            "file_id": file_id,
            # cuts-v2 substrate: the capture CHANNEL (said|done|shown) + the
            # orthogonal SUBJECT tag (person|place|object|graphic). The honest
            # what-was-captured the brain keys on.
            "channel": cut.get("channel"),
            # speech | video -- the pacing axis this cut lives on (video =
            # playback speed; speech = dead-air trim). Carried for the pace tag
            # + the retime verb.
            "kind": cut.get("kind"),
            "subject": cut.get("subject"),
            # Plain camera-move phrase for the shot (static / pan / tilt / zoom /
            # follow subject / shaky) -- surfaced in the beat line so the brain
            # knows how the shot moves.
            "camera": cut.get("camera") or "unknown",
            # Gist of an information-dense graphic (what it conveys), when present.
            "summary": cut.get("summary"),
            # True when speech overlaps this span -- the gist is redundant in the
            # resident line (the brain hears it); still kept for Tier-1.
            "summary_covered_by_speech": bool(
                cut.get("summary")
                and _overlaps_any(anchor["in_ms"], anchor["out_ms"], speech_spans)
            ),
            "speaker": cut.get("speaker"),
            "gist": cut.get("label") or "",
            "flags": cut.get("flags") or [],
            # Source-audio facet for video cuts: what's on the track ("speech"/
            # "ambient") and whether it's muted by default (stray speech under a
            # shot). Said cuts leave these unset (their audio is the point).
            "audio": cut.get("audio"),
            "mute": bool(cut.get("mute")),
            "score": float(cut.get("score", 0.0)),
            # Deterministic quality scores (post.py) carried through untouched:
            # speech_quality (delivery, ~equal across simultaneous angles) and
            # total_quality (speech + visual; the `score` above == total_quality
            # for a re-ingested cut). Let the brain "show the speaker" by ranking
            # a beat's alternate angles on visual presentation.
            "speech_quality": cut.get("speech_quality"),
            "total_quality": cut.get("total_quality"),
            "in_ms": anchor["in_ms"],
            "out_ms": anchor["out_ms"],
            "play_ms": anchor["play_ms"],
            "people": cut.get("people") or [],
            "framing": cut.get("framing"),
            "quality": cut.get("quality"),
            # Cuts-v3 take grouping, carried straight through when the cut
            # carries it (None for legacy hero cuts) -- see _annotate_dups.
            "take_group_id": cut.get("take_group_id"),
            "take_role": cut.get("take_role"),
            # Outlook group this beat belongs to (alternate cameras of one
            # moment sharing an authoritative audio track). Carried through so
            # `_annotate_outlook_groups` can tell the brain the audio is
            # decoupled from whichever angle it shows. (DB column is
            # `sync_group_id`; the map speaks only "outlook".)
            "outlook_group_id": cut.get("sync_group_id"),
            # Cuts-v3 continuity (cuts_v3_continuity.plan.md): junk stays IN
            # the map, labeled, and its persisted position/weld-to-neighbor
            # facts ride straight through (False/{} for legacy hero cuts).
            "junk": bool(cut.get("junk")),
            "junk_reason": cut.get("junk_reason"),
            "continuity": cut.get("continuity") or {},
            # Pace envelope (min/natural/max ms, cross-clip-normalized speed
            # levels, removable dead-air/filler spans) -- the pacing ROOM the
            # brain reads + `retime` acts on. {} for legacy hero cuts.
            "pace": cut.get("pace") or {},
            # perception_upgrade.plan.md: on-screen text/graphics (slide,
            # lower-third, UI, title) the model read off the pixels, and this
            # cut's single strongest INSTANT (code-computed -- see
            # post._salience; distinct from hero_ts_ms, the best STILL).
            # "" / {} on a pre-migration cut.
            "screen_text": cut.get("screen_text") or "",
            "salience": cut.get("salience") or {},
            "variants": variants,
            "atoms": [],
        })

    # Tag same-channel back-to-back beats as continuous source runs (one shot).
    _assign_runs(moments)

    return {
        "file_id": file_id,
        "name": header.get("name") or fid8,
        "duration_ms": int(header.get("duration_ms") or 0),
        "content_type": header.get("content_type"),
        "primary_axis": header.get("primary_axis"),
        # Where the energy slider should OPEN for this genre (long-form calmer,
        # short-form punchier); the editor still controls the full range.
        "default_energy": default_energy_for(header.get("content_type")),
        "mood": header.get("mood"),
        "people": header.get("people") or [],
        "topics": header.get("topics") or [],
        "logline": header.get("logline"),
        "moment_count": len(moments),
        "moments": moments,
    }


# --------------------------------------------------------------------------
# Cache (one row per file, keyed by the hero-cut content signature + TREE_VERSION)
# --------------------------------------------------------------------------

def _pg_conn():
    import psycopg
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _ensure_table(conn) -> None:
    conn.execute(
        """
        create table if not exists footage_trees (
            file_id        uuid primary key,
            source_version text not null,
            tree           jsonb not null,
            created_at     timestamptz not null default now()
        )
        """
    )


def _tree_version(sig: str) -> str:
    return f"{sig}:t{TREE_VERSION}"


def _source_signatures(file_ids: List[str], run_id: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Per-file content signature from the cuts-v3 ``cut_records`` substrate.
    A change in the signature is what tells ``get_trees`` to rebuild that
    file's cached tree. ``run_id`` pins the thread's covering ingest run (see
    migration 028)."""
    from app.services.l3 import cutrecord_map
    return cutrecord_map.signatures_for(file_ids, run_id=run_id)


def get_trees(file_ids: List[str], run_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """``{file_id: clip_tree}`` for the given clips, served from the per-file
    cache and lazily (re)built for any file whose signature changed. Files with
    no usable artifacts yet are simply absent from the result. Fail-open: a
    cache error degrades to a live build for the affected files.
    ``run_id`` pins the thread's covering ingest run (see migration 028)."""
    if not file_ids:
        return {}

    sigs = _source_signatures(file_ids, run_id=run_id)
    usable = [fid for fid in file_ids if sigs.get(fid)]
    if not usable:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    try:
        with _pg_conn() as conn:
            _ensure_table(conn)
            rows = conn.execute(
                "select file_id::text, source_version, tree from footage_trees "
                "where file_id = any(%s::uuid[])",
                (usable,),
            ).fetchall()
            cached = {fid: (sv, tree if isinstance(tree, dict) else json.loads(tree))
                      for fid, sv, tree in rows}
            for fid in usable:
                want = _tree_version(sigs[fid])
                hit = cached.get(fid)
                if hit and hit[0] == want:
                    out[fid] = hit[1]
                else:
                    missing.append(fid)
    except Exception:
        logger.exception("footage map: cache read failed; rebuilding all")
        missing = list(usable)
        out = {}

    if missing:
        built = _build_trees(missing, sigs, run_id=run_id)
        try:
            with _pg_conn() as conn:
                _ensure_table(conn)
                for fid, tree in built.items():
                    conn.execute(
                        """
                        insert into footage_trees (file_id, source_version, tree)
                        values (%s, %s, %s)
                        on conflict (file_id) do update set
                            source_version = excluded.source_version,
                            tree = excluded.tree,
                            created_at = now()
                        """,
                        (fid, _tree_version(sigs[fid]), json.dumps(tree)),
                    )
        except Exception:
            logger.exception("footage map: cache write failed (continuing)")
        out.update(built)

    return {fid: out[fid] for fid in file_ids if fid in out}


def _build_trees(file_ids: List[str], sigs: Dict[str, Optional[str]],
                 run_id: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Build moment-trees for a set of files from their cuts + headers. Cuts
    come from cuts-v3 ``cut_records``. ``run_id`` pins the thread's covering
    ingest run (migration 028)."""
    from app.services.l3.auto_edit import _clip_cards
    from app.services.l3 import cutrecord_map

    cuts_by_file = cutrecord_map.cut_dicts_for_files(file_ids, run_id=run_id)
    headers = _clip_cards(file_ids)
    out: Dict[str, Dict[str, Any]] = {}
    for fid in file_ids:
        header = headers.get(fid) or {"name": fid, "duration_ms": 0}
        tree = build_clip_tree(fid, header, cuts_by_file.get(fid, []))
        if tree["moments"]:
            out[fid] = tree
    return out


# --------------------------------------------------------------------------
# Tier-0 map (compact text + struct for the arranger prompt)
# --------------------------------------------------------------------------

def _fmt_ts(ms: int) -> str:
    s = max(0, int(ms)) // 1000
    return f"{s // 60}:{s % 60:02d}"


def _capture_tag(m: Dict[str, Any]) -> str:
    """WHAT this cut captured (cuts-v2): the CHANNEL.SUBJECT -- said.person,
    done.object, shown.graphic. The honest substrate the brain reads, with no
    editorial affordance/role bias."""
    ch = m.get("channel") or ""
    sub = m.get("subject")
    return f"{ch}.{sub}" if (ch and sub) else ch


def _short_gist(s: str, max_words: int = 6) -> str:
    """A graphic's gist, capped to a few words -- enough to register what the
    frame conveys without bloating the resident line (the full text is Tier-1)."""
    words = s.strip().split()
    if len(words) <= max_words:
        return s.strip()
    return " ".join(words[:max_words]) + "\u2026"


def _pic_who(m: Dict[str, Any], handle: Optional[str],
            alias: Optional[Dict[Any, str]], oncam: Optional[Dict[str, str]]) -> Optional[str]:
    """WHO or WHAT is on screen for this beat (Fact #1's PICTURE half) -- the
    reconciled shoot cast's shown face when known (right even where this clip's
    own on-camera flag was wrong), else this moment's own on-camera person, else
    the subject noun for a non-person capture (place/object/graphic). None when
    genuinely unknown -- PIC never invents a face it isn't told about."""
    shown, _ = _shown_and_cam(m.get("file_id"), handle, alias, oncam)
    if shown:
        return shown
    for p in (m.get("people") or []):
        if p.get("on_camera") is True:
            return p.get("person_id") or p.get("voice_speaker_id")
    if "offscreen" in (m.get("flags") or []):
        return None
    if any(p.get("on_camera") is False for p in (m.get("people") or [])):
        return None
    return m.get("subject") or None


def _framing_tag(m: Dict[str, Any]) -> str:
    """A short shot-size note for a delivery, when the perception carries one
    (never invented). '' when unknown."""
    fr = m.get("framing")
    if isinstance(fr, str) and fr.strip():
        return fr.strip()
    if isinstance(fr, dict):
        for k in ("shot", "shot_size", "size"):
            v = fr.get(k)
            if v:
                return str(v)
    return ""


def _peak_tag(m: Dict[str, Any]) -> str:
    """`peak:+X.Xs` -- this cut's single strongest INSTANT (post._salience,
    code-computed), as an offset from the cut's own start; use it for
    emphasis / punch-in / hold timing. Rendered only when the peak is
    meaningfully INTERIOR -- more than a clip-relative edge guard away from
    BOTH the start and the end, since a peak pinned to the first/last frame
    tells the brain nothing. Never rendered for a pre-migration cut ({}) or
    the no-signal fallback (score == 0.0, which is just hero_ts_ms restated,
    not a real peak). `score` itself is NEVER shown: it's the peak's height
    normalized against this cut's OWN curve range, so it reads as ~1.0 almost
    everywhere and is not a cross-cut "how salient" magnitude."""
    sal = m.get("salience") or {}
    if not sal or not sal.get("score"):
        return ""
    peak_ms = sal.get("peak_ms")
    if peak_ms is None:
        return ""
    in_ms, out_ms = m.get("in_ms", 0), m.get("out_ms", 0)
    span_ms = out_ms - in_ms
    if span_ms <= 0:
        return ""
    off_ms = peak_ms - in_ms
    # ~1 hop_ms (L1's own granularity is consistently 100ms) or 10% of the
    # span, whichever is larger -- a short cut still gets a usable guard band.
    edge_guard = max(100, span_ms // 10)
    if off_ms < edge_guard or (span_ms - off_ms) < edge_guard:
        return ""
    return f" peak:+{off_ms / 1000:.1f}s"


def _speaker_handle(m: Dict[str, Any]) -> Optional[str]:
    """The RAW, resolvable speaker handle for a said moment -- the diarized voice
    id (else the VLM person id) that the shoot identity registry is keyed on.

    Critical distinction: the moment's ``speaker`` field is a human DISPLAY LABEL
    (role/person/voice, e.g. "main subject") produced by the cast map for reading;
    it does NOT resolve to a global id. The raw id needed to alias a voice to its
    shoot-wide person lives in the ``people`` facet. Resolving off the label was
    the "speaker id gap" -- every speech line fell back to its role label and the
    brain could never match a speaker to their camera. Falls back to the display
    ``speaker`` for an unresolved cut that already kept its raw voice id there."""
    for p in (m.get("people") or []):
        h = p.get("voice_speaker_id") or p.get("person_id")
        if h:
            return h
    return m.get("speaker")


def _global_speaker(file_id: Optional[str], handle: Optional[str],
                    alias: Optional[Dict[Any, str]],
                    label: Optional[str] = None) -> Optional[str]:
    """The shoot-wide person id (Gx) for a raw voice/person handle, when the
    registry linked it across clips; else the human ``label`` (the moment's
    display speaker) when given, else the handle itself. Never nothing when a
    speaker exists."""
    if not handle:
        return label
    if alias and file_id is not None:
        gid = alias.get((file_id, handle))
        if gid:
            return gid
    return label if label is not None else handle


def _shown_and_cam(file_id: Optional[str], handle: Optional[str],
                   alias: Optional[Dict[Any, str]],
                   oncam: Optional[Dict[str, str]]) -> Tuple[Optional[str], str]:
    """(global face on screen, on/off-cam) DERIVED from the reconciled cast: whose
    FACE this clip shows, and whether the speaker is that person. This is the
    trustworthy on-camera fact -- it survives a clip whose per-clip A/V sensor was
    wrong -- so it OVERRIDES the per-moment on_camera flag when available. The
    speaker is resolved STRICTLY (raw handle -> Gx); an unresolved speaker can't
    be claimed on-camera. Returns (None, "") when the shoot cast doesn't know who
    this clip shows."""
    if not oncam or file_id is None:
        return None, ""
    shown = oncam.get(file_id)
    if not shown:
        return None, ""
    spk = alias.get((file_id, handle)) if (alias and handle) else None
    cam = "on-cam" if (spk and spk == shown) else "off-cam"
    return shown, cam


def _dur_tag(m: Dict[str, Any]) -> str:
    """The cut's PLAY length (after any breath/dead-air excision) so the brain
    can pace + honor a target length without arithmetic on the timestamps."""
    ms = int(m.get("play_ms", m["out_ms"] - m["in_ms"]))
    s = ms / 1000.0
    return f"{s:.1f}s" if s < 10 else f"{int(round(s))}s"


def _qual(score: float) -> str:
    """Compact quality/confidence token shared by PIC and alt-PIC parens."""
    return f"q.{int(round(max(0.0, score) * 100)):02d}"


def _pic_segment(m: Dict[str, Any], handle: Optional[str],
                 alias: Optional[Dict[Any, str]], oncam: Optional[Dict[str, str]]) -> str:
    """PIC leads the beat line: whose face / what scene is on screen (never the
    speaker), + shot size + quality. '?' when genuinely unresolved -- never a
    guess dressed up as a fact."""
    who = _pic_who(m, handle, alias, oncam) or "?"
    bits = [b for b in (_framing_tag(m), _qual(float(m.get("score", 0.0)))) if b]
    return f"PIC:{who} ({', '.join(bits)})"


# Video cut's own source-audio facet (speech/sound/silent) -> the SND state word
# when it plays as captured (not muted).
_AUDIO_STATE = {"speech": "talk", "sound": "ambient", "silent": "silence"}


def _snd_state(m: Dict[str, Any]) -> str:
    """The audio STATE half of SND -- 'speaking' for a said beat (the content IS
    the speech); else the video cut's own source-audio facet, folding in the
    default mute so the brain isn't surprised by a silent play. Unknown audio
    (no signal to judge) defaults to 'silence', the harmless assumption."""
    if m.get("channel") == "said":
        return "speaking"
    if m.get("mute"):
        kind = "talk" if m.get("audio") == "speech" else "ambient" if m.get("audio") == "sound" else "audio"
        return f"muted({kind})"
    return _AUDIO_STATE.get(m.get("audio"), "silence")


def _snd_segment(m: Dict[str, Any], handle: Optional[str],
                 alias: Optional[Dict[Any, str]]) -> str:
    """SND: a co-equal peer of PIC -- who is heard (identity-resolved off the RAW
    handle, not the display label -- the speaker-id gap) + the audio state."""
    state = _snd_state(m)
    if m.get("channel") != "said":
        return f"SND:{state}"
    gspk = _global_speaker(m.get("file_id"), handle, alias, label=m.get("speaker"))
    who = f"{gspk} " if gspk else ""
    return f"SND:{who}{state}"


def _alt_pic_segment(m: Dict[str, Any], alias: Optional[Dict[Any, str]],
                     oncam: Optional[Dict[str, str]]) -> str:
    """Fact #2 folded onto the beat: every OTHER picture the SAME sound is also
    available as (this beat's cross-clip take-group members), each a neutral
    `Gx→ref (shot, q)`. Never a verdict -- just where else this sound's picture
    lives. Absent when there is no co-occurrence (`_annotate_dups` sets no
    `alt_pic` in that case) or when no member resolves to a picture distinct
    from this beat's own.

    OUTLOOK vs TAKE dedupe differs, because the two are distinct kinds:
      * TAKE (retakes of the same content): collapse by on-camera PERSON and
        drop an alternate that shows the same person as this beat -- offering
        the same face twice is noise, the brain wants the distinct alternate.
      * OUTLOOK (alternate CAMERAS of one simultaneous moment): every angle is
        a real, switchable option even when it shows the same person (that's
        the whole point of multicam) or when the on-camera identity is unknown.
        Collapse by ANGLE (file) instead, and never drop on same/unknown who --
        otherwise a talking-head shot from three cameras would surface as one."""
    facts = m.get("alt_pic") or []
    if not facts:
        return ""
    is_outlook = bool(m.get("outlook_group_id")) or m.get("take_role") == "outlook"
    my_who = _pic_who(m, _speaker_handle(m), alias, oncam)
    seen = set()
    parts: List[str] = []
    for f in facts:
        shown, _ = _shown_and_cam(f.get("file"), f.get("voice"), alias, oncam)
        who = shown or _global_speaker(f.get("file"), f.get("voice"), alias)
        if is_outlook:
            key = f.get("file")
            if not key or key in seen:
                continue
            seen.add(key)
            label = who or "angle"
        else:
            if not who or who == my_who or who in seen:
                continue
            seen.add(who)
            label = who
        bits = [b for b in (f.get("framing"), _qual(float(f.get("score", 0.0)))) if b]
        retry = ", retry" if f.get("restart") else ""
        parts.append(f"{label}→{f['moment_id']} ({', '.join(bits)}{retry})")
    return f" ·alt-PIC:{', '.join(parts)}" if parts else ""


def _weld_mark(contiguous: bool, reason: Optional[str]) -> str:
    """↔ (weldable continuation) / ⋯ (hard cut) toward one neighbor, read off
    the persisted continuity seam verdict. '' when there IS no neighbor on
    that side (clip start/end, ``seam_reason`` unset) -- never a fabricated
    edge."""
    if reason is None:
        return ""
    return "↔" if contiguous else "⋯"


def _pace_tag(m: Dict[str, Any]) -> str:
    """The cut's PACING room, read straight off its persisted pace envelope --
    so the brain knows what `retime` can do here before it tries:
      * SPEECH -> ``trim<=X.Xs``: total removable dead-air/filler budget (retime
        shortens the delivery by shaving this; it NEVER changes pitch/speed).
      * VIDEO  -> ``pace:LO-HIx``: the reachable playback-speed range (levels
        cross-clip-normalized so the same rung looks smooth against neighbors),
        shown only when there is real speed room.
    '' when there is no room (a clean speech beat / a video cut pinned to 1x) or
    for a legacy hero-cut moment with no envelope."""
    pace = m.get("pace") or {}
    is_speech = m.get("kind") == "speech" or m.get("channel") == "said"
    if is_speech:
        budget = 0
        for sp in pace.get("remove_spans") or []:
            try:
                budget += int(sp[1]) - int(sp[0])
            except (IndexError, TypeError, ValueError):
                continue
        return f" · trim\u2264{budget / 1000:.1f}s" if budget > 0 else ""
    levels = pace.get("levels") or []
    if len(levels) < 2:
        return ""
    try:
        lo, hi = float(min(levels)), float(max(levels))
    except (TypeError, ValueError):
        return ""
    if hi - lo < 0.05:                      # pinned to a single speed -> no room
        return ""
    return f" · pace:{lo:.2g}-{hi:.2g}x"


def _continuity_tag(m: Dict[str, Any]) -> str:
    """This beat's position within its clip's full cut sequence (incl. junk --
    a gap in cut_no IS the signal a junk beat sits there) + weld marks toward
    each neighbor. '' for a legacy hero-cut moment (no continuity block)."""
    cont = m.get("continuity") or {}
    cut_no, of = cont.get("cut_no"), cont.get("of")
    if not cut_no or not of:
        return ""
    prev = _weld_mark(bool(cont.get("prev_contiguous")), cont.get("seam_reason_prev"))
    nxt = _weld_mark(bool(cont.get("next_contiguous")), cont.get("seam_reason_next"))
    return f" · {prev}cut:{cut_no}/{of}{nxt}"


def _moment_line(m: Dict[str, Any], *, compact: bool = False,
                 alias: Optional[Dict[Any, str]] = None,
                 oncam: Optional[Dict[str, str]] = None) -> str:
    # Junk stays IN the map (labeled), never dropped, so numbering/contiguity
    # stay honest and the brain can still place it as a deliberate bridge --
    # but it renders as a terse one-liner (id + reason + position + span) so
    # keeping it visible doesn't bloat the index with a full rich line for
    # content that's skip-by-default.
    if m.get("junk"):
        reason = (m.get("junk_reason") or "").strip() or "unspecified"
        return (f"  {m['moment_id'].split(':')[-1]} [JUNK: {reason}]{_continuity_tag(m)} "
                f"[{_fmt_ts(m['in_ms'])}-{_fmt_ts(m['out_ms'])} {_dur_tag(m)}]")
    levels = list(m["variants"].keys())
    nrg = "|".join(L for L in _LEVEL_NAMES if L in levels)
    # Resolve the speaker off the RAW handle (people facet), not the display
    # label -- so every speech line names its shoot-wide person (Gx), not a
    # role label the registry can't match.
    handle = _speaker_handle(m)
    # PIC leads (what actually lands on screen); SND is a co-equal peer, never
    # the subject -- placing a beat can no longer read as "showing the speaker".
    pic = _pic_segment(m, handle, alias, oncam)
    snd = _snd_segment(m, handle, alias)
    gist = (m.get("gist") or "").strip().replace("\n", " ")
    # Resident mode gives the model the FULL line so it picks by reading, not
    # guessing; compact (paged) mode truncates and relies on inspect_moment.
    if compact and len(gist) > 80:
        gist = gist[:77] + "..."
    # A graphic's gist, only when speech doesn't already narrate it (else it's
    # redundant -- the brain hears it). Short tag; full text via inspect_moment.
    gloss = ""
    summ = (m.get("summary") or "").strip().replace("\n", " ")
    if summ and not m.get("summary_covered_by_speech"):
        gloss = f" graphic:\"{_short_gist(summ)}\""
    # Legible on-screen text/graphics the model read off the pixels (slide,
    # lower-third, UI, title) -- distinct from the action gloss; rendered only
    # when present so text-free footage stays terse. "" on a pre-migration cut.
    scr = (m.get("screen_text") or "").strip().replace("\n", " ")
    scr_tag = f" text:\"{_short_gist(scr)}\"" if scr else ""
    # Continuity run: this beat is part of one uninterrupted same-clip shot
    # (members listed in source order); keep run members together + in order.
    run = ""
    if m.get("run_id"):
        run = f" · run:{m['run_id']}"
    cut_tag = _continuity_tag(m)
    pace_tag = _pace_tag(m)
    # Camera move, only when there's something to say (a static/unknown shot
    # adds no signal and would just bloat every line).
    cam = (m.get("camera") or "").strip()
    cam_tag = f" cam:{cam.replace(' ', '-')}" if cam and cam not in ("static", "unknown") else ""
    peak_tag = _peak_tag(m)
    # A structural fact only -- this moment's audio is shared verbatim with the
    # other angles (the outlook group's authoritative track), so picture choice
    # is entirely the brain's call and switching angles never jumps the audio.
    outlook_tag = f" outlook:{m['outlook_angle_count']}-angles" if m.get("outlook_angle_count") else ""
    alt = _alt_pic_segment(m, alias, oncam)
    return (f"  {m['moment_id'].split(':')[-1]} {_capture_tag(m)} {pic} {snd} "
            f"[{_fmt_ts(m['in_ms'])}-{_fmt_ts(m['out_ms'])} {_dur_tag(m)}] "
            f"\"{gist}\"{gloss}{scr_tag} · nrg:{nrg}{pace_tag}{cam_tag}{peak_tag}{outlook_tag}{cut_tag}{run}{alt}")


def _clip_block(tree: Dict[str, Any], *, compact: bool = False,
                alias: Optional[Dict[Any, str]] = None,
                oncam: Optional[Dict[str, str]] = None) -> str:
    head_bits = [f"{tree['duration_ms'] // 1000}s"]
    if tree.get("content_type"):
        head_bits.append(tree["content_type"])
    if tree.get("primary_axis"):
        head_bits.append(f"axis:{tree['primary_axis']}")
    if tree.get("mood"):
        head_bits.append(f"mood:{tree['mood']}")
    if tree.get("people"):
        head_bits.append("people:" + ",".join(tree["people"]))
    header = (f"CLIP {tree['file_id'][:8]} \"{tree['name']}\" · "
              + " · ".join(head_bits))
    lines = [header] + [_moment_line(m, compact=compact, alias=alias, oncam=oncam)
                        for m in tree["moments"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Cross-clip duplicate linking (same line delivered as multiple takes/angles)
# --------------------------------------------------------------------------

def _annotate_dups(trees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Link moments that deliver the SAME content across clips/angles.

    dup_groups read DIRECTLY off each moment's persisted ``take_group_id``/
    ``take_role`` (pass 2 resolved cross-clip takes from the pixels +
    transcript, then ``post._enforce_take_winner`` re-crowned the winner
    deterministically by total_quality -- and only within a same-setting take
    cluster, never among outlook angles) -- no token-overlap recompute, no IoU
    matching needed since the group id already lives on the exact moment it
    belongs to. Tags
    each linked moment in place with its ``dup_group`` plus ``alt_pic`` (Fact
    #2 folded onto the beat: every OTHER member's raw facts, so
    ``_moment_line`` can render this beat's alternates without a separate
    lookup). No winner is crowned -- the brain compares the members and
    decides. Fail-open: a moment with no take_group_id is simply not linked."""
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for t in trees:
        for m in t.get("moments", []) or []:
            gid = m.get("take_group_id")
            # Junk is skip-by-default -- keep it OUT of the take-group /
            # coverage set even if pass 2 somehow tagged one (cuts_v3_continuity
            # .plan.md); it stays independently placeable via its own ref.
            if gid and not m.get("junk"):
                by_group.setdefault(gid, []).append(m)

    summary: List[Dict[str, Any]] = []
    for gid, members in by_group.items():
        # A real choice needs >= 2 DISTINCT moments, exactly like the hero path.
        if len(members) < 2:
            continue
        for m in members:
            m["dup_group"] = gid
        member_facts = [{
            "moment_id": m["moment_id"],
            "file": m["file_id"],
            "voice": _speaker_handle(m),
            "framing": _framing_tag(m),
            "score": float(m.get("score", 0.0)),
            # cuts-v3 has no restart concept at the take-group level (that's a
            # same-clip mid-attempt retry, judged inside pass 2); never fabricated.
            "restart": False,
            "take_role": m.get("take_role"),
        } for m in members]
        by_mid = {m["moment_id"]: m for m in members}
        for f in member_facts:
            by_mid[f["moment_id"]]["alt_pic"] = [
                g for g in member_facts if g["moment_id"] != f["moment_id"]
            ]
        summary.append({
            "group_id": gid,
            "members": [m["moment_id"] for m in members],
            "member_facts": member_facts,
            "text": (members[0].get("gist") or "").strip(),
        })
    return summary


def _annotate_outlook_groups(trees: List[Dict[str, Any]]) -> None:
    """Tag each moment in an outlook group with `outlook_angle_count` -- how
    many distinct ANGLES cover THIS beat (this angle + its alt-PIC alternates),
    a purely STRUCTURAL fact (`_moment_line` renders it as `outlook:N-angles`),
    never editorial guidance. It tells the brain "these N angles' picture all
    ride the same authoritative audio track, so switching between them never
    jumps the sound" -- picture choice is still entirely the brain's call.

    Counts per-BEAT (from the beat's own alt-PIC angles), NOT how many moments
    share the project-wide group id: a group id spans the whole recording, but
    a given beat is only actually covered by the angles that were rolling then
    (a shorter angle drops out for the parts it didn't film). Runs after
    `_annotate_dups`, which populates each moment's `alt_pic`."""
    for t in trees:
        for m in t.get("moments", []) or []:
            if m.get("junk") or not m.get("outlook_group_id"):
                continue
            angles = {g.get("file") for g in (m.get("alt_pic") or []) if g.get("file")}
            angles.add(m.get("file_id"))   # this beat's own angle
            angles.discard(None)
            if len(angles) >= 2:
                m["outlook_angle_count"] = len(angles)


def _identity_maps_for(file_ids: List[str], run_id: Optional[str]) -> Tuple[Optional[Dict[str, str]], Optional[Dict[Any, str]]]:
    """(oncam, alias) for the covering ingest run's reconciled cast
    (identity_map.plan.md Phase 4), or (None, None) when there isn't one --
    an older run, a run reconciliation found nothing to bind/cluster for, or
    no covering run at all. `alias` keys are re-split from the persisted
    ``"file_id|voice"`` string into the `(file_id, voice)` tuples
    `_shown_and_cam`/`_pic_who` already expect. Fail-open: any lookup error
    degrades to (None, None), i.e. today's behavior, never breaks the map."""
    try:
        from app.services.l3 import cuts_v3_read, ingest_store
        resolved_run_id = run_id or cuts_v3_read.latest_run_for_files(file_ids)
        if resolved_run_id is None:
            return None, None
        identity_map = ingest_store.get_identity_map(resolved_run_id)
        if not identity_map:
            return None, None
        oncam = identity_map.get("oncam") or None
        raw_alias = identity_map.get("alias") or {}
        alias = {}
        for key, display in raw_alias.items():
            fid, _, voice = str(key).partition("|")
            if fid and voice:
                alias[(fid, voice)] = display
        return oncam, (alias or None)
    except Exception:
        logger.exception("footage_map: identity_map lookup failed (continuing without it)")
        return None, None


def assemble_map(file_ids: List[str], *, compact: bool = False,
                 run_id: Optional[str] = None) -> Dict[str, Any]:
    """The Tier-0 footage index for the arranger.

    Returns ``{"text", "struct", "clip_count", "moment_count", "dup_groups"}``
    where ``text`` is the one-line-per-moment index dropped into the prompt and
    ``struct`` is the machine-readable trees the arranger/compiler resolve
    placements against. ``compact`` truncates gists for the paged (over-budget)
    path; resident mode emits the full line. Moments are tagged in place with
    cross-clip duplicate links -- coverage reads INLINE on each beat as
    ``·alt-PIC`` (no separate coverage block: dedupe the information, not the
    sequence). Speakers resolve through the covering run's reconciled cast
    (identity_map.plan.md) when one exists -- ``PIC``/``alt-PIC`` then name
    real persons instead of a raw per-clip diarized id; absent a reconciled
    cast (older run, or nothing bindable/clusterable), behaves exactly as
    before this feature existed.
    """
    trees = get_trees(file_ids, run_id=run_id)
    ordered = [trees[fid] for fid in file_ids if fid in trees]
    dups = _annotate_dups(ordered)      # tags moments in `ordered` in place
    _annotate_outlook_groups(ordered)   # tags moments in `ordered` in place
    oncam, alias = _identity_maps_for(file_ids, run_id)
    text = "\n\n".join(_clip_block(t, compact=compact, alias=alias, oncam=oncam)
                       for t in ordered)
    n_moments = sum(t["moment_count"] for t in ordered)
    return {
        "text": text,
        "struct": {"clips": ordered},
        "clip_count": len(ordered),
        "moment_count": n_moments,
        "dup_groups": dups,
    }


# --------------------------------------------------------------------------
# Tier-1 retrieval (full record for one moment, on demand)
# --------------------------------------------------------------------------

def _as_doc(v: Any) -> Dict[str, Any]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str) and v:
        try:
            return json.loads(v)
        except Exception:
            return {}
    return {}


# Context window padding when retrieving raw detail for a span -- a little before
# and after the moment so the brain reads it in situ, not abruptly clipped.
_DETAIL_PAD_MS = 1200


def _span_detail(file_id: str, in_ms: int, out_ms: int) -> Dict[str, Any]:
    """The RAW source detail overlapping a moment's span -- the brain's Tier-1
    'open the file and read' depth: the verbatim transcript window (L1 dialogue
    segments). Kept OUT of the resident prompt; loaded on demand only when the
    brain inspects a candidate. Best-effort -- a missing artifact simply yields
    fewer fields."""
    lo, hi = in_ms - _DETAIL_PAD_MS, out_ms + _DETAIL_PAD_MS

    def _ov(a0: Any, a1: Any) -> bool:
        try:
            return _overlap_ms(int(a0), int(a1), lo, hi) > 0
        except Exception:
            return False

    out: Dict[str, Any] = {"transcript": []}
    try:
        with _pg_conn() as conn:
            row = conn.execute(
                "select segments from dialogue_segments where file_id = %s",
                (file_id,),
            ).fetchone()
    except Exception:
        logger.exception("moment_detail: span detail load failed for %s", file_id)
        return out
    if not row:
        return out
    segments = _as_doc(row[0])

    # Verbatim transcript window (sentence granularity) -- the words actually
    # spoken across the span, so the brain reads the line, not just the gist.
    out["transcript"] = [
        {"speaker": s.get("speaker"), "text": s.get("text"),
         "in_ms": s.get("src_in_ms", s.get("in_ms")), "out_ms": s.get("src_out_ms", s.get("out_ms"))}
        for s in (segments.get("sentence") or [])
        if _ov(s.get("src_in_ms", s.get("in_ms", 0)), s.get("src_out_ms", s.get("out_ms", 0)))
    ]
    return out


def moment_detail(file_id: str, moment_id: str) -> Optional[Dict[str, Any]]:
    """The full record for one moment -- every variant span, the parent clip's
    context, AND the raw source detail over its span (VLM detection atoms +
    transcript window). What the model retrieves to inspect a candidate deeply,
    without any of it living in the resident prompt."""
    tree = get_trees([file_id]).get(file_id)
    if not tree:
        return None
    m = next((mm for mm in tree["moments"] if mm["moment_id"] == moment_id), None)
    if m is None:
        return None
    return {
        "clip": {
            "file_id": tree["file_id"],
            "name": tree["name"],
            "duration_ms": tree["duration_ms"],
            "content_type": tree.get("content_type"),
            "primary_axis": tree.get("primary_axis"),
            "mood": tree.get("mood"),
            "people": tree.get("people"),
            "topics": tree.get("topics"),
            "logline": tree.get("logline"),
        },
        "moment": m,
        # Tier-1 raw detail over the moment's span (detection atoms + transcript).
        "source": _span_detail(file_id, int(m["in_ms"]), int(m["out_ms"])),
    }
