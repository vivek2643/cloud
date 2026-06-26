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

logger = logging.getLogger(__name__)

# Bump when the moment-tree shape / clustering logic changes so cached trees
# rebuild even if the underlying hero cuts (PARAMS_VERSION) did not.
# v2: every nested level (incl. tight=core) is a selectable variant; atoms are
# emitted only when a moment splits into >= 2 finer sub-units.
# v3: cuts OWN their ladder -- variants read straight off each cut's rungs from
# one anchor band; no geometric re-matching, no atoms (a split is a multi-span
# rung). Moments also carry the people/framing/quality facets.
TREE_VERSION = 3

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
    for idx, cut in enumerate(sorted(cuts, key=_in)):
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
            "variants": variants,
            "atoms": [],
        })

    return {
        "file_id": file_id,
        "name": header.get("name") or fid8,
        "duration_ms": int(header.get("duration_ms") or 0),
        "content_type": header.get("content_type"),
        "primary_axis": header.get("primary_axis"),
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


def _affordance_tag(m: Dict[str, Any]) -> str:
    """Show the full modality mix for multi-affordance moments (speech+reaction),
    else the single modality -- so the brain sees a cut carries more than one
    thing to land on."""
    affs = [a for a in (m.get("affordances") or []) if a]
    if len(affs) > 1:
        return "+".join(affs)
    return m.get("modality") or (affs[0] if affs else "")


def _people_tag(m: Dict[str, Any]) -> str:
    """Compact on/off-camera marker so the brain knows when the voice it's about
    to cut to has no on-screen speaker (an off-camera interviewer / voiceover)."""
    if "offscreen" in (m.get("flags") or []):
        return " off-cam"
    if any(p.get("on_camera") is False for p in (m.get("people") or [])):
        return " off-cam"
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
    dup = ""
    if m.get("dup_group"):
        dup = f" · dup:{m['dup_group']}{'*' if m.get('dup_best') else ''}"
    return (f"  {m['moment_id'].split(':')[-1]} {_affordance_tag(m)}{spk}{cam} "
            f".{int(round(m['score'] * 100)):02d} "
            f"[{_fmt_ts(m['in_ms'])}-{_fmt_ts(m['out_ms'])}] "
            f"\"{gist}\" · nrg:{nrg}{dup}")


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
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Cross-clip duplicate linking (same line delivered as multiple takes/angles)
# --------------------------------------------------------------------------

def _take_key(m: Dict[str, Any], restart_ids: set) -> Tuple[bool, float, float, float]:
    """Deterministic best-take order for a moment: not-a-retry, then speaker
    on-camera, then delivery fluency, then score. Mirrors hero_cuts._take_rank so
    the engine's pick is consistent whether read off heroes or the moment-tree."""
    q = m.get("quality") or {}
    on_cam = q.get("on_camera")
    deliv = q.get("delivery")
    return (
        m["moment_id"] not in restart_ids,
        float(on_cam) if on_cam is not None else 0.5,
        float(deliv) if deliv is not None else 0.5,
        float(m.get("score", 0.0)),
    )


def _best_overlap_moment(
    moments: List[Dict[str, Any]], start_ms: int, end_ms: int
) -> Optional[Dict[str, Any]]:
    """The moment whose span overlaps [start_ms, end_ms] the most (or None)."""
    best: Optional[Dict[str, Any]] = None
    best_ov = 0
    for m in moments:
        ov = _overlap_ms(int(m["in_ms"]), int(m["out_ms"]), start_ms, end_ms)
        if ov > best_ov:
            best_ov, best = ov, m
    return best if best_ov > 0 else None


def _annotate_dups(trees: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Link moments that deliver the SAME content across clips/angles.

    Reconciles cross-clip take groups (``takes.build_take_groups``) onto the
    moment-tree by time overlap, tags each linked moment in place with its
    ``dup_group`` and whether it's the engine's pre-picked best take, and returns
    a render-ready summary. This is cross-set dependent (it changes with the file
    set), so it is computed per request and never baked into the per-file tree
    cache. Fail-open: any error yields no links.
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
        # Engine best take, deterministic: avoid abandoned (restart) takes, then
        # prefer the speaker ON CAMERA, then the cleaner DELIVERY, then score.
        best = max(linked.values(), key=lambda m: _take_key(m, restart_ids))
        for mid, m in linked.items():
            m["dup_group"] = g.group_id
            m["dup_best"] = (mid == best["moment_id"])
        summary.append({
            "group_id": g.group_id,
            "best": best["moment_id"],
            "members": list(linked.keys()),
            "text": (g.attempts[0].text or "").strip(),
        })
    return summary


def _dups_block(summary: List[Dict[str, Any]]) -> str:
    if not summary:
        return ""
    lines = ["DUPLICATE GROUPS (same line, multiple takes/angles -- place ONE "
             "per group; the * is the engine's best take, override only if "
             "another reads/looks better):"]
    for d in summary:
        members = " ".join(
            (mid + "*" if mid == d["best"] else mid) for mid in d["members"]
        )
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

def moment_detail(file_id: str, moment_id: str) -> Optional[Dict[str, Any]]:
    """The full record for one moment -- every variant span, every atom, and the
    parent clip's context. What the model retrieves to inspect a candidate."""
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
    }
