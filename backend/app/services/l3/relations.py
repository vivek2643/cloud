"""
Cross-clip identity -- one global person across the clips of a shoot.

Per-clip diarization gives every voice a LOCAL label (S0/S1/...) that means
nothing across clips: the same human is S1 in one clip and S0 in another, and a
messy clip can even invent extra voices. The only signal that survives between
clips is SPEECH -- when two clips contain the SAME spoken line, whoever delivers
it is the same person in both. So identity is built from that alone:

  1. MATCH lines across clips by CONTENT (token containment), ignoring the
     per-clip voice label entirely -- because the labels aren't comparable.
  2. In each matched line, the dominant voice in clip A corresponds to the
     dominant voice in clip B: one vote that (A, vA) and (B, vB) are one human.
  3. Per clip PAIR, resolve those votes into a ONE-TO-ONE voice matching -- a
     two-person conversation yields TWO correspondences, never one voice mapping
     to two -- keeping only correspondences corroborated by >= MIN_LINK_VOTES
     shared lines. Union the surviving correspondences into global identities.

The union is CONSTRAINED by one hard truth (not a heuristic, not a fallback):
two DISTINCT voices in the SAME clip are two different people, so any merge that
would place two of a clip's voices into one identity is rejected. That single
constraint is what stops a noisy clip from transitively collapsing everyone into
one person -- so no appearance-matching, no time offsets, no take-group speaker
gate is needed or wanted (those either fuse look-alikes or lean on labels being
comparable across clips). Speech + same-clip-distinctness is the whole method.

Everything is DERIVED from artifacts that already exist -- transcript sentences
/ VLM content units and the voice<->person cast (`l3.cast`). No new perception
pass, no embeddings, no model calls. Fail-open: <2 clips or no cross-clip line
-> {} (a person seen in only one clip is honestly not a cross-clip identity).
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

from app.services.l3 import cast as cast_mod
from app.services.l3 import takes

logger = logging.getLogger(__name__)

# A voice<->voice correspondence between two clips is only trusted once this many
# distinct shared lines agree on it, so a single loose content match can't invent
# a cross-clip link. Genuine multicam/overlapping recordings of one conversation
# share far more than this; the threshold just discards the noise floor.
MIN_LINK_VOTES = 2

Node = Tuple[str, str]        # (file_id, voice)


def _fid8(s: str) -> str:
    return (s or "")[:8]


# --------------------------------------------------------------------------
# Load: perception + words per clip (cast dressing) and attempts (line matching)
# --------------------------------------------------------------------------

def _load_clips(file_ids: List[str]) -> Dict[str, Tuple[Optional[dict], List[dict]]]:
    """file_id -> (perception, flat words). One batched read; used to build the
    per-clip cast that dresses each identity member with its person/role/on-cam."""
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


def _load_attempts(file_ids: List[str]) -> List[takes.Attempt]:
    """Every spoken-line attempt across the clips in scope (one batched read).
    These carry content tokens + the dominant diarized voice -- the raw material
    the matcher links across clips."""
    if not file_ids:
        return []
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
    out: List[takes.Attempt] = []
    for fid, perception, segments in rows:
        out.extend(takes._attempts_for_clip(fid, takes._as_doc(perception),
                                             takes._as_list(segments)))
    return out


# --------------------------------------------------------------------------
# IDENTITY
# --------------------------------------------------------------------------

def _cluster_by_content(attempts: List[takes.Attempt]) -> List[List[takes.Attempt]]:
    """Greedy same-LINE clustering on content-token containment ONLY (no speaker
    gate): each cluster is one spoken line, gathering that line's delivery from
    every clip that contains it. Mirrors `takes.cluster_attempts` but drops the
    speaker constraint, because identity is exactly the label-agnostic question
    of WHICH voice each clip used for the same line."""
    clusters: List[List[takes.Attempt]] = []
    seeds: List[frozenset] = []
    for att in attempts:
        toks = att.tokens
        if not toks:
            continue
        best = -1
        best_score = takes.CONTAINMENT_THRESHOLD
        for gi in range(len(clusters)):
            seed = seeds[gi]
            if len(toks & seed) < takes.MIN_SHARED_TOKENS:
                continue
            score = takes._containment(toks, seed)
            if score >= best_score:
                best, best_score = gi, score
        if best < 0:
            clusters.append([att])
            seeds.append(toks)
        else:
            clusters[best].append(att)
    return clusters


def _pair_votes(clusters: List[List[takes.Attempt]]) -> Dict[Tuple[str, str], Counter]:
    """{(file_a, file_b) -> Counter[(voice_a, voice_b)]}: for every matched line,
    the dominant voice each clip used casts one vote linking those two voices.
    file_a < file_b so the vote direction is stable."""
    votes: Dict[Tuple[str, str], Counter] = {}
    for cl in clusters:
        # Dominant voice per file for THIS line (a file may hold retakes).
        per_file: Dict[str, Counter] = {}
        for a in cl:
            if a.speaker:
                per_file.setdefault(a.file_id, Counter())[a.speaker] += 1
        reps = {f: c.most_common(1)[0][0] for f, c in per_file.items()}
        files = sorted(reps)
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                fa, fb = files[i], files[j]
                votes.setdefault((fa, fb), Counter())[(reps[fa], reps[fb])] += 1
    return votes


def _match_pair(cnt: Counter) -> List[Tuple[int, str, str]]:
    """One-to-one voice matching for a single clip pair: take the best-voted
    correspondences greedily, never reusing a voice on either side, keeping only
    those with >= MIN_LINK_VOTES. A two-person conversation yields two edges
    (one per person); a spurious single match is filtered out."""
    used_a: set = set()
    used_b: set = set()
    edges: List[Tuple[int, str, str]] = []
    for (va, vb), v in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        if v < MIN_LINK_VOTES:
            break
        if va in used_a or vb in used_b:
            continue
        used_a.add(va)
        used_b.add(vb)
        edges.append((v, va, vb))
    return edges


def derive_identities(attempts: List[takes.Attempt],
                      clips: Dict[str, Tuple[Optional[dict], List[dict]]]) -> List[dict]:
    """Global people across the shoot from matched SPEECH alone.

    Content-cluster the lines (label-agnostic), vote per clip pair, resolve each
    pair to a one-to-one voice matching, then union the correspondences under the
    hard same-clip-distinctness constraint. Each surviving cluster that spans >=2
    clips is one global person, dressed with the per-clip cast (person/role/on
    -camera) and a canonical appearance description.
    """
    if not attempts:
        return []

    clusters = _cluster_by_content(attempts)
    votes = _pair_votes(clusters)

    # Correspondence edges across all clip pairs, strongest first.
    edges: List[Tuple[int, Node, Node]] = []
    for (fa, fb), cnt in votes.items():
        for v, va, vb in _match_pair(cnt):
            edges.append((v, (fa, va), (fb, vb)))
    edges.sort(key=lambda e: -e[0])

    # Constrained union-find. Each cluster tracks {file -> voice}; a merge is
    # rejected when the two clusters disagree on any file's voice, because two
    # voices in one clip are two people (the constraint that prevents collapse).
    parent: Dict[Node, Node] = {}
    files_of: Dict[Node, Dict[str, str]] = {}

    def touch(n: Node) -> None:
        if n not in parent:
            parent[n] = n
            files_of[n] = {n[0]: n[1]}

    def find(n: Node) -> Node:
        touch(n)
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a: Node, b: Node) -> None:
        touch(a)
        touch(b)
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        fa, fb = files_of[ra], files_of[rb]
        for f, v in fb.items():
            if fa.get(f, v) != v:                 # same clip, different voice
                return                            # -> cannot be one person; skip
        parent[rb] = ra
        fa.update(fb)
        files_of.pop(rb, None)

    for _, a, b in edges:
        union(a, b)

    groups: Dict[Node, List[Node]] = {}
    for n in list(parent):
        groups.setdefault(find(n), []).append(n)

    # Cast for dressing members with person/role/on-camera + a description.
    casts: Dict[str, cast_mod.ClipCast] = {}
    persons_by_fid: Dict[str, Dict[str, dict]] = {}
    for fid, (perception, words) in clips.items():
        casts[fid] = cast_mod.build_cast(perception, words)
        persons_by_fid[fid] = {
            p.get("local_id"): p for p in ((perception or {}).get("persons") or [])
            if p.get("local_id")}

    out: List[dict] = []
    n = 0
    for _, nodes in sorted(groups.items(), key=lambda kv: sorted(kv[1])):
        if len({f for f, _ in nodes}) < 2:        # identity is a CROSS-clip fact
            continue
        n += 1
        members: List[dict] = []
        desc: Optional[str] = None
        for fid, voice in sorted(nodes):
            m = casts[fid].resolve(voice) if fid in casts else None
            person = m.person_id if m else None
            role = m.role if m else None
            on_camera = m.on_camera if m else None
            if person and desc is None:
                desc = (persons_by_fid.get(fid, {}).get(person) or {}).get(
                    "canonical_description")
            members.append({"file": fid, "voice": voice, "person": person,
                            "role": role, "on_camera": on_camera})
        out.append({"global_id": f"G{n}", "members": members, "description": desc})
    return out


# --------------------------------------------------------------------------
# Entry + rendering
# --------------------------------------------------------------------------

def build_relations(file_ids: List[str]) -> dict:
    """The shoot-level relation set: global person identities across the clips.
    {} when there is only one clip or no cross-clip person can be established.
    Fail-open."""
    if len(file_ids or []) < 2:
        return {}
    try:
        clips = _load_clips(file_ids)
        attempts = _load_attempts(file_ids)
        identities = derive_identities(attempts, clips)
        if not identities:
            return {}
        return {"identities": identities}
    except Exception:
        logger.exception("relations: build failed (continuing without)")
        return {}


def _member_str(m: dict) -> str:
    """One member of a global person, in a single file: file8[:who][ on/off-cam]."""
    who = m.get("voice") or m.get("person") or "?"
    cam = ""
    if m.get("on_camera") is True:
        cam = " on-cam"
    elif m.get("on_camera") is False:
        cam = " off-cam"
    return f"{_fid8(m['file'])}:{who}{cam}"


def render_relations(relations: dict) -> str:
    """The brain-facing digest of WHO is in the shoot -- prepended to
    source_awareness so the global people are named BEFORE the per-clip blocks
    that reference them. Identity only. '' when empty."""
    if not relations:
        return ""
    idents = relations.get("identities") or []
    if not idents:
        return ""
    lines = ["PEOPLE OF THE SHOOT (one person across clips; use the global id "
             "when you talk about them):"]
    for ident in idents:
        parts = [_member_str(m) for m in (ident.get("members") or [])]
        desc = f' "{ident["description"]}"' if ident.get("description") else ""
        lines.append(f"  {ident['global_id']}{desc}: " + " = ".join(parts))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Registry lookups -- alias a per-clip voice/person to its global id.
# --------------------------------------------------------------------------

def identity_index(relations: dict) -> Dict[Tuple[str, str], str]:
    """{(file_id, handle) -> global_id} where handle is a voice id or a person id
    -- so a caller can look up by whichever handle it holds. Empty when there are
    no identities."""
    idx: Dict[Tuple[str, str], str] = {}
    for ident in (relations or {}).get("identities") or []:
        gid = ident["global_id"]
        for m in ident.get("members") or []:
            fid = m.get("file")
            if not fid:
                continue
            for handle in (m.get("voice"), m.get("person")):
                if handle:
                    idx[(fid, handle)] = gid
    return idx


def global_id_of(relations: dict, file_id: str, handle: Optional[str]) -> Optional[str]:
    """The global person id for a per-clip voice OR person handle in a file
    (None when that handle isn't part of any cross-clip identity)."""
    if not handle:
        return None
    return identity_index(relations).get((file_id, handle))


def local_of(relations: dict, file_id: str, global_id: str) -> Optional[dict]:
    """The member record (voice/person/on_camera) that a global person maps to
    within one file -- the inverse used to resolve 'presence:G2' to a local lane.
    None when that person doesn't appear in the file."""
    for ident in (relations or {}).get("identities") or []:
        if ident.get("global_id") != global_id:
            continue
        for m in ident.get("members") or []:
            if m.get("file") == file_id:
                return m
    return None


def to_json(relations: dict) -> str:
    return json.dumps(relations, default=str)
