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
TREE_VERSION = 9

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


def _cluster_ladder(members: List[Dict[str, Any]]) -> Tuple[Dict[str, List[str]], str]:
    """The zoom ladder for one connected bundle (moment): at Broad the whole run
    of member cuts, narrowing to just the PEAK member at Sharp.

    The peak is the highest-scoring member; lower energy admits its temporal
    NEIGHBOURS, nearest-first, until the full run is in at Broad. So a moment
    behaves like every other cut -- one selectable unit with a coarse..fine
    ladder -- one level up from the individual cuts. Returns
    ``({level: [moment_id, ...]}, peak_moment_id)``."""
    by_time = sorted(members, key=lambda m: int(m["in_ms"]))
    n = len(by_time)
    peak_idx = max(range(n), key=lambda i: float(by_time[i].get("score", 0.0)))
    peak = by_time[peak_idx]
    ladder: Dict[str, List[str]] = {}
    for band in range(len(_LEVEL_NAMES)):
        # Broad (band 0) = whole run; Sharp (band 4) = the peak alone.
        k = max(1, round(n * (1.0 - band / (len(_LEVEL_NAMES) - 1))))
        lo = hi = peak_idx
        while (hi - lo + 1) < k:
            left_ok, right_ok = lo > 0, hi < n - 1
            if left_ok and right_ok:
                left_gap = int(peak["in_ms"]) - int(by_time[lo - 1]["out_ms"])
                right_gap = int(by_time[hi + 1]["in_ms"]) - int(peak["out_ms"])
                if left_gap <= right_gap:
                    lo -= 1
                else:
                    hi += 1
            elif left_ok:
                lo -= 1
            elif right_ok:
                hi += 1
            else:
                break
        ladder[_LEVEL_NAMES[band]] = [by_time[i]["moment_id"] for i in range(lo, hi + 1)]
    return ladder, peak["moment_id"]


def _distinct_ladder(ladder: Dict[str, List[str]]) -> List[Dict[str, Any]]:
    """Collapse consecutive levels with an IDENTICAL member set into one rung, so
    only meaningfully-distinct zoom steps materialize. A moment that reads the
    same from Broad through Balanced is ONE rung spanning those levels, not three
    duplicates -- the antidote to a ladder of look-alike rungs. Additive: the
    full per-level ``ladder`` stays for callers that index by level."""
    rungs: List[Dict[str, Any]] = []
    for level in _LEVEL_NAMES:
        members = list(ladder.get(level) or [])
        if rungs and rungs[-1]["members"] == members:
            rungs[-1]["levels"].append(level)
        else:
            rungs.append({"levels": [level], "members": members})
    return rungs


def _build_clusters(moments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Roll the flat moments up into their connected bundles (shared
    ``cluster_id``). Each cluster is a moment-as-unit: its member run, the peak
    member, the run's time span, and the whole-run -> peak zoom ladder. Only
    bundles of >= 2 cuts are clusters (a lone cut is already its own moment)."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for m in moments:
        cid = m.get("cluster_id")
        if cid:
            groups.setdefault(cid, []).append(m)
    clusters: List[Dict[str, Any]] = []
    for cid, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        ladder, peak = _cluster_ladder(members)
        ordered = sorted(members, key=lambda m: int(m["in_ms"]))
        affs = []
        prims = []
        channels = []
        for m in ordered:
            for a in (m.get("affordances") or []):
                if a and a not in affs:
                    affs.append(a)
            for p in (m.get("primitives") or []):
                if p and p not in prims:
                    prims.append(p)
            ch = m.get("channel")
            if ch and ch not in channels:
                channels.append(ch)
        clusters.append({
            "cluster_id": cid,
            "members": [m["moment_id"] for m in ordered],
            "peak": peak,
            "affordances": affs,
            # The capture mix across members (person+speech+graphic ...) -- what
            # the whole moment is made of, primitive-led like the per-cut line.
            "primitives": prims,
            # The channel mix (said+done+shown) -- a capture-moment is exactly a
            # cross-channel bundle, so this is what defines it.
            "channels": channels,
            "in_ms": min(int(m["in_ms"]) for m in ordered),
            "out_ms": max(int(m["out_ms"]) for m in ordered),
            "ladder": ladder,
            # Only the meaningfully-distinct zoom steps (look-alike rungs merged).
            "rungs": _distinct_ladder(ladder),
        })
    return clusters


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
        or "speech" in (c.get("primitives") or []) or "speech" in (c.get("affordances") or [])
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
            "modality": cut.get("modality"),
            "affordances": cut.get("affordances") or ([cut.get("modality")] if cut.get("modality") else []),
            # Intrinsic capture substrate (person/action/place/object/graphic/speech).
            "primitives": cut.get("primitives") or [],
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
            "score": float(cut.get("score", 0.0)),
            "in_ms": anchor["in_ms"],
            "out_ms": anchor["out_ms"],
            "play_ms": anchor["play_ms"],
            "people": cut.get("people") or [],
            "framing": cut.get("framing"),
            "quality": cut.get("quality"),
            # The connected-cluster id this cut shares with the cuts it forms a
            # moment with (a line + its reaction + illustrating b-roll). None for
            # a standalone cut. The "Moments" view groups by this.
            "cluster_id": cut.get("moment_id"),
            "variants": variants,
            "atoms": [],
        })

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
        # Connected bundles (>= 2 cuts) rolled up as moment-as-unit, each with a
        # whole-run -> peak zoom ladder. Empty for clips with no relations.
        "clusters": _build_clusters(moments),
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
    editorial affordance/role bias. Falls back to the legacy primitive/affordance
    mix for un-migrated clips that carry no channel."""
    ch = m.get("channel")
    if ch:
        sub = m.get("subject")
        return f"{ch}.{sub}" if sub else ch
    prims = [p for p in (m.get("primitives") or []) if p]
    if prims:
        return "+".join(prims)
    affs = [a for a in (m.get("affordances") or []) if a]
    if len(affs) > 1:
        return "+".join(affs)
    return m.get("modality") or (affs[0] if affs else "")


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


def _relation_tag(m: Dict[str, Any]) -> str:
    """Compact view of the cross-channel moment cluster this cut belongs to, so
    the brain sees the deterministic capture-moment grouping (a line + its
    reaction + illustrating b-roll) instead of guessing from adjacency."""
    if m.get("cluster_id"):
        return f" · moment:{str(m['cluster_id']).split(':')[-1]}"
    return ""


def _moment_line(m: Dict[str, Any], *, compact: bool = False) -> str:
    levels = list(m["variants"].keys())
    nrg = "|".join(L for L in _LEVEL_NAMES if L in levels)
    spk = f" {m['speaker']}" if m.get("speaker") else ""
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
    dup = ""
    if m.get("dup_group"):
        dup = f" · dup:{m['dup_group']}{' retry' if m.get('dup_restart') else ''}"
    return (f"  {m['moment_id'].split(':')[-1]} {_capture_tag(m)}{spk}{cam} "
            f".{int(round(m['score'] * 100)):02d} "
            f"[{_fmt_ts(m['in_ms'])}-{_fmt_ts(m['out_ms'])}] "
            f"\"{gist}\"{gloss} · nrg:{nrg}{dup}{_relation_tag(m)}")


def _short_mid(mid: str) -> str:
    return str(mid).split(":")[-1]


def _cluster_line(c: Dict[str, Any]) -> str:
    """One line for a connected bundle (moment-as-unit): its capture mix, the run
    span, the peak member, and the whole-run -> peak zoom ladder so the brain can
    place the moment as ONE thing -- the whole run loose, or just its peak when
    tight. Members are still listed above (each tagged with this cluster id), so
    nothing is hidden; this is the rolled-up unit on top of them."""
    mix = "+".join(p for p in (c.get("channels") or c.get("primitives") or c.get("affordances") or []) if p)
    rungs = []
    for L in _LEVEL_NAMES:
        ids = c["ladder"].get(L) or []
        if ids:
            rungs.append(f"{L}:{','.join(_short_mid(i) for i in ids)}")
    return (f"  moment {_short_mid(c['cluster_id'])} {mix} "
            f"[{_fmt_ts(c['in_ms'])}-{_fmt_ts(c['out_ms'])}] "
            f"peak={_short_mid(c['peak'])} · zoom {' -> '.join(rungs)}")


def _clip_block(tree: Dict[str, Any], *, compact: bool = False) -> str:
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
    lines = [header] + [_moment_line(m, compact=compact) for m in tree["moments"]]
    clusters = tree.get("clusters") or []
    if clusters:
        lines.append("  MOMENTS (connected bundles -- take the whole run loose, "
                     "or just the peak tight):")
        lines.extend(_cluster_line(c) for c in clusters)
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
        restart_ids: set = set()
        for a in g.attempts:
            m = _best_overlap_moment(by_file.get(a.file_id, []), a.start_ms, a.end_ms)
            if m is None:
                continue
            linked.setdefault(m["moment_id"], m)
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
        summary.append({
            "group_id": g.group_id,
            "members": list(linked.keys()),
            "text": (g.attempts[0].text or "").strip(),
        })
    return summary


def _dups_block(summary: List[Dict[str, Any]]) -> str:
    if not summary:
        return ""
    lines = ["SAME-BEAT GROUPS (the same spoken line captured more than once -- a "
             "retake or another camera angle). Choose the member that reads/looks "
             "best by its text and score; do not repeat the same line on the main "
             "line, but you MAY place another member as a silent reaction/cutaway "
             "over that line's audio:"]
    for d in summary:
        members = " ".join(d["members"])
        gist = d["text"].replace("\n", " ")
        if len(gist) > 80:
            gist = gist[:77] + "..."
        lines.append(f"  {d['group_id']}: {members}  \"{gist}\"")
    return "\n".join(lines)


def assemble_map(file_ids: List[str], *, compact: bool = False) -> Dict[str, Any]:
    """The Tier-0 footage index for the arranger.

    Returns ``{"text", "struct", "clip_count", "moment_count", "dup_groups"}``
    where ``text`` is the one-line-per-moment index dropped into the prompt and
    ``struct`` is the machine-readable trees the arranger/compiler resolve
    placements against. ``compact`` truncates gists for the paged (over-budget)
    path; resident mode emits the full line. Moments are tagged in place with
    cross-clip duplicate links.
    """
    trees = get_trees(file_ids)
    ordered = [trees[fid] for fid in file_ids if fid in trees]
    dups = _annotate_dups(ordered)   # tags moments in `ordered` in place
    text = "\n\n".join(_clip_block(t, compact=compact) for t in ordered)
    dblock = _dups_block(dups)
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
    'open the file and read' depth: the VLM event timeline, the content-unit /
    cutaway records, and the verbatim transcript window. Kept OUT of the resident
    prompt; loaded on demand only when the brain inspects a candidate. Best-effort
    -- a missing artifact simply yields fewer fields."""
    lo, hi = in_ms - _DETAIL_PAD_MS, out_ms + _DETAIL_PAD_MS

    def _ov(a0: Any, a1: Any) -> bool:
        try:
            return _overlap_ms(int(a0), int(a1), lo, hi) > 0
        except Exception:
            return False

    out: Dict[str, Any] = {"atoms": [], "events": [], "content_units": [],
                           "cutaways": [], "transcript": []}
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
        # Tier-1 raw detail over the moment's span (events / units / transcript).
        "source": _span_detail(file_id, int(m["in_ms"]), int(m["out_ms"])),
    }
