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

Caching mirrors ``hero_store``: a clip's tree is deterministic given its L1/L2
artifacts, so it is keyed by the same content signature (plus a tree-logic
version) and rebuilt only when the signature changes. Best-effort throughout:
a missing artifact for a file simply yields fewer (or no) moments.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.services.l3 import hero_cuts as hc
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
TREE_VERSION = 17

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
            "subject": cut.get("subject"),
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
            "in_ms": anchor["in_ms"],
            "out_ms": anchor["out_ms"],
            "play_ms": anchor["play_ms"],
            "people": cut.get("people") or [],
            "framing": cut.get("framing"),
            "quality": cut.get("quality"),
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


def get_trees(file_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """``{file_id: clip_tree}`` for the given clips, served from the per-file
    cache and lazily (re)built for any file whose signature changed. Files with
    no usable artifacts yet are simply absent from the result. Fail-open: a
    cache error degrades to a live build for the affected files."""
    if not file_ids:
        return {}

    from app.services.l3 import hero_store
    sigs = hero_store.signatures_for(file_ids)
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
        built = _build_trees(missing, sigs)
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


def _build_trees(file_ids: List[str], sigs: Dict[str, Optional[str]]) -> Dict[str, Dict[str, Any]]:
    """Build moment-trees for a set of files from their band cuts + headers."""
    from app.services.l3.auto_edit import _clip_cards
    from app.services.l3 import hero_store

    anchor_cuts = hero_store.get_anchor_cuts(file_ids)
    headers = _clip_cards(file_ids)
    out: Dict[str, Dict[str, Any]] = {}
    for fid in file_ids:
        header = headers.get(fid) or {"name": fid, "duration_ms": 0}
        tree = build_clip_tree(fid, header, anchor_cuts.get(fid, []))
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


def _people_tag(m: Dict[str, Any]) -> str:
    """Compact on/off-camera marker so the brain knows when the voice it's about
    to cut to has no on-screen speaker (an off-camera interviewer / voiceover)."""
    if "offscreen" in (m.get("flags") or []):
        return " off-cam"
    if any(p.get("on_camera") is False for p in (m.get("people") or [])):
        return " off-cam"
    return ""


def _cam_state(m: Dict[str, Any]) -> str:
    """Tri-state on-camera marker for one delivery: 'on-cam' / 'off-cam' / '' when
    there's no signal to judge. Used by the coverage groups so the brain can see
    which member actually SHOWS the speaker."""
    if "offscreen" in (m.get("flags") or []):
        return "off-cam"
    people = m.get("people") or []
    if any(p.get("on_camera") is False for p in people):
        return "off-cam"
    if any(p.get("on_camera") is True for p in people):
        return "on-cam"
    return ""


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


def _global_speaker(file_id: Optional[str], voice: Optional[str],
                    alias: Optional[Dict[Any, str]]) -> Optional[str]:
    """The shoot-wide person id for a per-clip voice, when the registry linked it
    across clips; else the raw voice (never nothing when a voice exists)."""
    if not voice:
        return None
    if alias and file_id is not None:
        gid = alias.get((file_id, voice))
        if gid:
            return gid
    return voice


def _dur_tag(m: Dict[str, Any]) -> str:
    """The cut's PLAY length (after any breath/dead-air excision) so the brain
    can pace + honor a target length without arithmetic on the timestamps."""
    ms = int(m.get("play_ms", m["out_ms"] - m["in_ms"]))
    s = ms / 1000.0
    return f"{s:.1f}s" if s < 10 else f"{int(round(s))}s"


def _audio_tag(m: Dict[str, Any]) -> str:
    """Source-audio note for a VIDEO cut. A muted cut plays SILENT by default so
    the brain isn't surprised, and the kind tells it WHY -- 'muted(talk)' is a
    stray half-sentence (usually leave silent); 'muted(sound)' is uncontrolled or
    the action's own sound (the brain may want to `audio:keep` it)."""
    if m.get("mute"):
        kind = m.get("audio")
        if kind == "speech":
            return " muted(talk)"
        if kind == "sound":
            return " muted(sound)"
        return " muted"
    return ""


def _moment_line(m: Dict[str, Any], *, compact: bool = False,
                 alias: Optional[Dict[Any, str]] = None) -> str:
    levels = list(m["variants"].keys())
    nrg = "|".join(L for L in _LEVEL_NAMES if L in levels)
    gspk = _global_speaker(m.get("file_id"), m.get("speaker"), alias)
    spk = f" {gspk}" if gspk else ""
    cam = _people_tag(m)
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
        gloss = f" shows:\"{_short_gist(summ)}\""
    # Continuity run: this beat is part of one uninterrupted same-clip shot
    # (members listed in source order); keep run members together + in order.
    run = ""
    if m.get("run_id"):
        run = f" · run:{m['run_id']}"
    dup = ""
    if m.get("dup_group"):
        dup = f" · dup:{m['dup_group']}{' retry' if m.get('dup_restart') else ''}"
    return (f"  {m['moment_id'].split(':')[-1]} {_capture_tag(m)}{spk}{cam}{_audio_tag(m)} "
            f".{int(round(m['score'] * 100)):02d} "
            f"[{_fmt_ts(m['in_ms'])}-{_fmt_ts(m['out_ms'])} {_dur_tag(m)}] "
            f"\"{gist}\"{gloss} · nrg:{nrg}{run}{dup}")


def _clip_block(tree: Dict[str, Any], *, compact: bool = False,
                alias: Optional[Dict[Any, str]] = None) -> str:
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
    lines = [header] + [_moment_line(m, compact=compact, alias=alias)
                        for m in tree["moments"]]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Cross-clip duplicate linking (same line delivered as multiple takes/angles)
# --------------------------------------------------------------------------

def _best_overlap_moment(
    moments: List[Dict[str, Any]], start_ms: int, end_ms: int
) -> Optional[Dict[str, Any]]:
    """The moment that best CORRESPONDS to [start_ms, end_ms] by temporal IoU
    (intersection over union), not raw overlap. IoU is what stops a whole-clip
    establishing/b-roll moment (e.g. a 400s "person at desk" shown-atom) from
    swallowing every speech attempt: it overlaps all of them but its IoU is tiny,
    while the aligned speech moment scores ~1.0. Returns None if nothing overlaps."""
    best: Optional[Dict[str, Any]] = None
    best_iou = 0.0
    for m in moments:
        mi, mo = int(m["in_ms"]), int(m["out_ms"])
        ov = _overlap_ms(mi, mo, start_ms, end_ms)
        if ov <= 0:
            continue
        union = max(mo, end_ms) - min(mi, start_ms)
        iou = ov / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou, best = iou, m
    return best


def _annotate_dups(trees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Link moments that deliver the SAME content across clips/angles.

    Reconciles cross-clip take groups (``takes.build_take_groups``) onto the
    moment-tree by best temporal IoU, tags each linked moment in place with its
    ``dup_group`` (and ``dup_restart`` for an abandoned take), and returns a
    render-ready summary. No winner is crowned -- the brain compares the members
    and decides. This is cross-set dependent (it changes with the file set), so
    it is computed per request and never baked into the per-file tree cache.
    Fail-open: any error yields no links.
    """
    file_ids = [t["file_id"] for t in trees]
    if len(file_ids) < 1:
        return []
    try:
        from app.services.l3 import takes  # lazy: keep footage_map import-light
        groups = takes.build_take_groups(file_ids)
    except Exception:
        logger.exception("footage map: take grouping failed; no dup links")
        return []

    by_file = {t["file_id"]: t.get("moments", []) for t in trees}
    summary: List[Dict[str, Any]] = []
    for g in groups:
        linked: Dict[str, Dict[str, Any]] = {}   # moment_id -> moment
        file_by_mid: Dict[str, str] = {}
        restart_ids: set = set()
        for a in g.attempts:
            m = _best_overlap_moment(by_file.get(a.file_id, []), a.start_ms, a.end_ms)
            if m is None:
                continue
            linked.setdefault(m["moment_id"], m)
            file_by_mid[m["moment_id"]] = a.file_id
            if a.is_restart:
                restart_ids.add(m["moment_id"])
        # A real choice needs >= 2 DISTINCT moments (one moment alone is not a
        # take decision -- the engine already collapsed its own bands).
        if len(linked) < 2:
            continue
        # No winner is crowned: members are tagged as the SAME beat and left for
        # the brain to compare (by reading text + quality) and place. An abandoned
        # (restart) take is flagged so the brain can see it, not auto-dropped.
        for mid, m in linked.items():
            m["dup_group"] = g.group_id
            if mid in restart_ids:
                m["dup_restart"] = True
        # Per-member facts so the brain can compare deliveries WITHOUT re-reading
        # the whole index: who (voice, aliased to a global person at render), on/
        # off camera, shot size, and quality score. Facts, not a verdict.
        member_facts = [{
            "moment_id": mid,
            "file": file_by_mid.get(mid, ""),
            "voice": m.get("speaker"),
            "cam": _cam_state(m),
            "framing": _framing_tag(m),
            "score": float(m.get("score", 0.0)),
            "restart": mid in restart_ids,
        } for mid, m in linked.items()]
        summary.append({
            "group_id": g.group_id,
            "members": list(linked.keys()),     # moment_ids (consumers key on this)
            "member_facts": member_facts,
            "text": (g.attempts[0].text or "").strip(),
        })
    return summary


def _dups_block(summary: List[Dict[str, Any]],
                *, alias: Optional[Dict[Any, str]] = None) -> str:
    if not summary:
        return ""
    # Coverage, not "pick one": each member is one delivery of the same beat.
    # The brain chooses which to SHOW and may reuse another as a cutaway -- no
    # directive here, the facts carry the decision. (Genre-agnostic: "beat"
    # covers a retake, a second camera, or the same line said twice.)
    lines = ["COVERAGE GROUPS (the same beat delivered more than once -- different "
             "takes and/or cameras; each member is one delivery, with who it shows "
             "and how well):"]
    for d in summary:
        gist = d["text"].replace("\n", " ")
        if len(gist) > 80:
            gist = gist[:77] + "..."
        facts = d.get("member_facts")
        if facts:
            parts = []
            for f in facts:
                gspk = _global_speaker(f.get("file"), f.get("voice"), alias)
                bits = [f["moment_id"]]
                if gspk:
                    bits.append(gspk)
                if f.get("cam"):
                    bits.append(f["cam"])
                if f.get("framing"):
                    bits.append(f["framing"])
                bits.append(f".{int(round(f.get('score', 0.0) * 100)):02d}")
                if f.get("restart"):
                    bits.append("retry")
                parts.append(" ".join(bits))
            body = " · ".join(parts)
        else:
            body = " ".join(d["members"])
        lines.append(f"  {d['group_id']} \"{gist}\": {body}")
    return "\n".join(lines)


def assemble_map(file_ids: List[str], *, compact: bool = False,
                 relations: Optional[dict] = None) -> Dict[str, Any]:
    """The Tier-0 footage index for the arranger.

    Returns ``{"text", "struct", "clip_count", "moment_count", "dup_groups"}``
    where ``text`` is the one-line-per-moment index dropped into the prompt and
    ``struct`` is the machine-readable trees the arranger/compiler resolve
    placements against. ``compact`` truncates gists for the paged (over-budget)
    path; resident mode emits the full line. Moments are tagged in place with
    cross-clip duplicate links. When ``relations`` (the shoot identity registry)
    is supplied, per-line and coverage-group speakers are aliased to their global
    person id so the same human reads consistently across clips.
    """
    alias: Optional[Dict[Any, str]] = None
    if relations:
        try:
            from app.services.l3 import relations as relations_mod
            alias = relations_mod.identity_index(relations)
        except Exception:
            logger.exception("footage map: identity index failed (continuing)")
    trees = get_trees(file_ids)
    ordered = [trees[fid] for fid in file_ids if fid in trees]
    dups = _annotate_dups(ordered)   # tags moments in `ordered` in place
    text = "\n\n".join(_clip_block(t, compact=compact, alias=alias) for t in ordered)
    dblock = _dups_block(dups, alias=alias)
    if dblock:
        text = f"{text}\n\n{dblock}"
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
    'open the file and read' depth: the VLM detection atoms (channel/subject/peak)
    and the verbatim transcript window. Kept OUT of the resident prompt; loaded on
    demand only when the brain inspects a candidate. Best-effort -- a missing
    artifact simply yields fewer fields."""
    lo, hi = in_ms - _DETAIL_PAD_MS, out_ms + _DETAIL_PAD_MS

    def _ov(a0: Any, a1: Any) -> bool:
        try:
            return _overlap_ms(int(a0), int(a1), lo, hi) > 0
        except Exception:
            return False

    out: Dict[str, Any] = {"atoms": [], "transcript": []}
    try:
        with _pg_conn() as conn:
            row = conn.execute(
                "select cp.perception, ds.segments "
                "from files f "
                "left join clip_perception cp on cp.file_id = f.id "
                "left join dialogue_segments ds on ds.file_id = f.id "
                "where f.id = %s",
                (file_id,),
            ).fetchone()
    except Exception:
        logger.exception("moment_detail: span detail load failed for %s", file_id)
        return out
    if not row:
        return out
    perception, segments = _as_doc(row[0]), _as_doc(row[1])

    # cuts-v2 substrate: the VLM's detection atoms over the span (channel /
    # subject / peak / confidence / gist). The brain's primary Tier-1 read.
    out["atoms"] = [
        {"channel": a.get("channel"), "subject": a.get("subject"),
         "start_ms": a.get("start_ms"), "end_ms": a.get("end_ms"),
         "peak_ms": a.get("peak_ms"), "confidence": a.get("confidence"),
         "label": a.get("label"), "summary": a.get("summary")}
        for a in (perception.get("atoms") or [])
        if _ov(a.get("start_ms", 0), a.get("end_ms", 0))
    ]

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
