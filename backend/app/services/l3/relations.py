"""
Cross-clip relations -- the shoot-level truths the per-clip senses can't see.

Each clip's awareness is complete WITHIN the clip (lanes, seams, peaks, cast),
but a shoot is a set of clips, and two truths live only BETWEEN them:

  * TIME:     two clips may be recordings of the SAME live moment (two cameras
              on one conversation). No clock sync exists or is wanted -- but the
              speech itself is a fingerprint: when several matched lines across
              two files agree on one start-time delta, the clips are co-temporal
              at that offset ("CLIP B ~= CLIP A + 12.4s").
  * IDENTITY: person/voice ids are per-clip (p1/S0 in one file has no relation
              to p1/S0 in another). When the same VOICE delivers the same LINE
              in two files (a take-group match), those per-clip voices are the
              same human -- union them into one global identity, and pull in the
              per-clip person link (cast) so the identity carries a face+role.

Everything here is DERIVED from artifacts that already exist -- take groups
(`l3.takes`), the voice<->person cast (`l3.cast`), and the VLM person cards.
No new perception pass, no embeddings, no model calls: fuse and surface.
Fail-open and pure-ish (one batched DB read); empty inputs -> empty relations.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3 import cast as cast_mod
from app.services.l3 import takes

logger = logging.getLogger(__name__)

# Co-temporal test: at least this many independent matched lines must agree...
MIN_OFFSET_MATCHES = 2
# ...within this spread (ms). Transcript boundaries drift a few hundred ms
# between two cameras' cuts of the same utterance; true retakes scatter by
# tens of seconds, so this separates "same moment" from "same line again".
OFFSET_AGREE_MS = 1500


def _fid8(s: str) -> str:
    return (s or "")[:8]


def _median(xs: List[int]) -> int:
    ys = sorted(xs)
    n = len(ys)
    return ys[n // 2] if n % 2 else (ys[n // 2 - 1] + ys[n // 2]) // 2


# --------------------------------------------------------------------------
# Load once: perception + words per clip (shared by offsets and identity)
# --------------------------------------------------------------------------

def _load_clips(file_ids: List[str]) -> Dict[str, Tuple[Optional[dict], List[dict]]]:
    """file_id -> (perception, flat words). One batched read."""
    if not file_ids:
        return {}
    with takes._pg_conn() as conn:
        rows = conn.execute(
            """
            select f.id::text, cp.perception, t.segments
              from files f
              left join clip_perception cp on cp.file_id = f.id
              left join transcripts t      on t.file_id  = f.id
             where f.id = any(%s::uuid[])
            """,
            (file_ids,),
        ).fetchall()
    out: Dict[str, Tuple[Optional[dict], List[dict]]] = {}
    for fid, perception, segments in rows:
        words: List[dict] = []
        for seg in takes._as_list(segments):
            words.extend(seg.get("words", []) or [])
        out[fid] = (takes._as_doc(perception), words)
    return out


# --------------------------------------------------------------------------
# TIME: co-temporal offsets from matched speech
# --------------------------------------------------------------------------

def _cross_file_pairs(group: takes.TakeGroup) -> List[Tuple[takes.Attempt, takes.Attempt]]:
    """One (a, b) pair per file pair in the group, earliest attempt per file,
    ordered by file_id so the delta sign is stable."""
    first: Dict[str, takes.Attempt] = {}
    for att in group.attempts:
        cur = first.get(att.file_id)
        if cur is None or att.start_ms < cur.start_ms:
            first[att.file_id] = att
    fids = sorted(first)
    return [(first[fa], first[fb])
            for i, fa in enumerate(fids) for fb in fids[i + 1:]]


def derive_offsets(groups: List[takes.TakeGroup]) -> List[dict]:
    """Per file pair: if >= MIN_OFFSET_MATCHES matched lines agree on one start
    -time delta (within OFFSET_AGREE_MS), the clips are co-temporal recordings
    of the same moment at that offset. Scattered deltas = separate retakes ->
    no relation (that case already lives in the dup groups)."""
    deltas: Dict[Tuple[str, str], List[int]] = {}
    for g in groups:
        for a, b in _cross_file_pairs(g):
            deltas.setdefault((a.file_id, b.file_id), []).append(b.start_ms - a.start_ms)

    out: List[dict] = []
    for (fa, fb), ds in deltas.items():
        if len(ds) < MIN_OFFSET_MATCHES:
            continue
        med = _median(ds)
        agreeing = [d for d in ds if abs(d - med) <= OFFSET_AGREE_MS]
        if len(agreeing) < MIN_OFFSET_MATCHES:
            continue
        # Confidence grows with agreeing lines and falls with dissenters.
        conf = round(min(1.0, len(agreeing) / 4.0) * (len(agreeing) / len(ds)), 2)
        out.append({"file_a": fa, "file_b": fb, "offset_ms": _median(agreeing),
                    "matches": len(agreeing), "confidence": conf})
    out.sort(key=lambda o: -o["confidence"])
    return out


# --------------------------------------------------------------------------
# IDENTITY: one human across files, from matched voices + the per-clip cast
# --------------------------------------------------------------------------

def derive_identities(groups: List[takes.TakeGroup],
                      clips: Dict[str, Tuple[Optional[dict], List[dict]]]) -> List[dict]:
    """Union (file, voice) nodes that delivered the same line, then dress each
    identity with the per-clip cast link (person id / role) and the VLM person
    card description, so the brain sees ONE human across files."""
    # Union-find over (file_id, voice) nodes.
    parent: Dict[Tuple[str, str], Tuple[str, str]] = {}

    def find(x: Tuple[str, str]) -> Tuple[str, str]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Tuple[str, str], b: Tuple[str, str]) -> None:
        parent[find(a)] = find(b)

    for g in groups:
        voiced = [(a.file_id, a.speaker) for a in g.attempts if a.speaker]
        for i in range(1, len(voiced)):
            if voiced[i][0] != voiced[0][0]:      # only cross-file evidence links
                union(voiced[0], voiced[i])

    clusters: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for node in list(parent):
        clusters.setdefault(find(node), []).append(node)

    casts: Dict[str, cast_mod.ClipCast] = {}
    persons_by_fid: Dict[str, Dict[str, dict]] = {}
    for fid, (perception, words) in clips.items():
        casts[fid] = cast_mod.build_cast(perception, words)
        persons_by_fid[fid] = {
            p.get("local_id"): p for p in (perception or {}).get("persons") or []
            if p.get("local_id")}

    out: List[dict] = []
    n = 0
    for nodes in clusters.values():
        files = {f for f, _ in nodes}
        if len(files) < 2:                        # identity is a CROSS-clip fact
            continue
        n += 1
        members: List[dict] = []
        desc: Optional[str] = None
        for fid, voice in sorted(nodes):
            m = casts[fid].resolve(voice) if fid in casts else None
            person = m.person_id if m else None
            role = m.role if m else None
            if person and desc is None:
                card = persons_by_fid.get(fid, {}).get(person) or {}
                desc = card.get("canonical_description")
            members.append({"file": fid, "voice": voice,
                            "person": person, "role": role})
        out.append({"global_id": f"G{n}", "members": members, "description": desc})
    return out


# --------------------------------------------------------------------------
# Entry + rendering
# --------------------------------------------------------------------------

def build_relations(file_ids: List[str]) -> dict:
    """The shoot-level relation set: co-temporal offsets + global identities.
    {} when clips share nothing (or there is only one clip). Fail-open."""
    if len(file_ids or []) < 2:
        return {}
    try:
        groups = takes.build_take_groups(file_ids)
        cross = [g for g in groups if len({a.file_id for a in g.attempts}) >= 2]
        if not cross:
            return {}
        clips = _load_clips(file_ids)
        offsets = derive_offsets(cross)
        identities = derive_identities(cross, clips)
        if not offsets and not identities:
            return {}
        return {"offsets": offsets, "identities": identities}
    except Exception:
        logger.exception("relations: build failed (continuing without)")
        return {}


def render_relations(relations: dict) -> str:
    """The brain-facing digest -- prepended to source_awareness so cross-clip
    facts arrive BEFORE the per-clip blocks they connect. '' when empty."""
    if not relations:
        return ""
    lines = ["CROSS-CLIP RELATIONS (facts that hold BETWEEN clips):"]
    for o in relations.get("offsets") or []:
        sign = "+" if o["offset_ms"] >= 0 else "-"
        lines.append(
            f"  co-temporal: CLIP {_fid8(o['file_b'])} ~= CLIP {_fid8(o['file_a'])} "
            f"{sign}{abs(o['offset_ms']) / 1000.0:.1f}s (conf {o['confidence']:.2f}, "
            f"{o['matches']} matched lines) -- the SAME live moment from two "
            f"recordings; you can cut between them mid-beat.")
    for ident in relations.get("identities") or []:
        parts = []
        for m in ident.get("members") or []:
            who = m.get("person") or "?"
            role = f" ({m['role']})" if m.get("role") else ""
            parts.append(f"{_fid8(m['file'])}:{who}/{m.get('voice')}{role}")
        desc = f'  "{ident["description"]}"' if ident.get("description") else ""
        lines.append(f"  same person {ident['global_id']}: " + " = ".join(parts) + desc)
    return "\n".join(lines)


def to_json(relations: dict) -> str:
    return json.dumps(relations, default=str)
