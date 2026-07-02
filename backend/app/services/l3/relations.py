"""
Cross-clip relations -- the shoot-level truth the per-clip senses can't see.

Each clip's awareness is complete WITHIN the clip (lanes, seams, peaks, cast),
but a shoot is a set of clips, and one truth lives only BETWEEN them:

  * IDENTITY: person/voice ids are per-clip (p1/S0 in one file has no relation
              to p1/S0 in another). Two kinds of evidence link them into ONE
              global person across the shoot:
                - SPEECH (strong): the same VOICE delivers the same LINE in two
                  files (a take-group match). Grounded in transcript+diarization
                  agreement; this is the backbone.
                - TRAITS (weak, best-effort): two clips that share NO spoken line
                  can only be matched on the VLM's appearance description
                  ("bald head, gray beard"). This is fuzzy -- prose overlap, not
                  a voiceprint or face embedding -- so it never overrides a
                  speech link and is always surfaced as low-confidence.

Everything here is DERIVED from artifacts that already exist -- take groups
(`l3.takes`), the voice<->person cast (`l3.cast`), and the VLM person cards.
No new perception pass, no embeddings, no model calls: fuse and surface.

There is deliberately NO time/offset ontology in the brain-facing digest: the
editor reasons about coverage as GROUPS (footage_map) and about reactions as
plausible windows near a group member (scan_source), not about a shoot-wide
clock we cannot honestly reconstruct. ``derive_offsets`` remains below purely as
an internal diagnostic (and for its unit tests); it is not rendered.

Fail-open and pure-ish (one batched DB read); empty inputs -> empty relations.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from app.services.l3 import cast as cast_mod
from app.services.l3 import takes

logger = logging.getLogger(__name__)

# Co-temporal test: at least this many independent matched lines must agree...
MIN_OFFSET_MATCHES = 2
# ...within this spread (ms). Transcript boundaries drift a few hundred ms
# between two cameras' cuts of the same utterance; true retakes scatter by
# tens of seconds, so this separates "same moment" from "same line again".
OFFSET_AGREE_MS = 1500

# --- Trait-fallback identity (weak evidence) -------------------------------
# Two persons in different clips are TENTATIVELY the same human when their VLM
# appearance descriptions overlap this much (Jaccard over content tokens) AND
# share at least this many tokens. Kept intentionally strict-ish to limit false
# merges, but treated as low-confidence regardless.
TRAIT_SIM_THRESHOLD = 0.5
MIN_TRAIT_TOKENS = 3
_TRAIT_CONF = 0.3          # confidence stamped on a trait-only identity
_SPEECH_CONF = 0.85        # confidence stamped on a speech-backed identity
_WORD_RE = re.compile(r"[a-z0-9']+")
# Generic appearance words that would glue any two people together.
_TRAIT_STOP = {
    "a", "an", "the", "with", "and", "of", "in", "on", "wearing", "person",
    "man", "woman", "male", "female", "adult", "young", "middle", "aged",
    "has", "is", "who", "appears", "to", "be", "seen", "shirt", "top",
}


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

Node = Tuple[str, str]        # (file_id, key) where key is "p:<person>" or "v:<voice>"


def _trait_tokens(card: dict) -> FrozenSet[str]:
    """Content tokens of a person card's durable appearance -- canonical
    description plus any durable-trait strings -- with generic filler removed."""
    parts: List[str] = []
    d = card.get("canonical_description")
    if d:
        parts.append(str(d))
    durable = card.get("durable")
    if isinstance(durable, dict):
        parts.extend(str(v) for v in durable.values() if v)
    elif isinstance(durable, (list, tuple)):
        parts.extend(str(v) for v in durable if v)
    elif durable:
        parts.append(str(durable))
    toks = {t for t in _WORD_RE.findall(" ".join(parts).lower())
            if t not in _TRAIT_STOP and len(t) > 1}
    return frozenset(toks)


def _trait_sim(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    """Jaccard overlap of two token sets (0 when either is empty)."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def derive_identities(groups: List[takes.TakeGroup],
                      clips: Dict[str, Tuple[Optional[dict], List[dict]]]) -> List[dict]:
    """One global person per human across the shoot, from two evidence kinds.

    SPEECH (strong): union the per-clip voices that delivered the same line
    (take-group match) -- the reliable backbone. TRAITS (weak): additionally
    union persons in different clips whose VLM appearance descriptions overlap,
    so a person who never shares a spoken line can still be recognised across
    clips. Trait links are best-effort and cannot manufacture a stronger claim
    than the speech backbone -- an identity is only stamped ``basis:'speech'``
    (confidence ~0.85) when a real speech edge lives inside it; a trait-only
    cluster is ``basis:'traits'`` (confidence ~0.3), surfaced as such.

    Nodes are keyed on the PERSON (VLM local_id) when a voice resolves to one,
    else on the raw voice, so both evidence kinds share one union space.
    """
    casts: Dict[str, cast_mod.ClipCast] = {}
    persons_by_fid: Dict[str, Dict[str, dict]] = {}
    for fid, (perception, words) in clips.items():
        casts[fid] = cast_mod.build_cast(perception, words)
        persons_by_fid[fid] = {
            p.get("local_id"): p for p in ((perception or {}).get("persons") or [])
            if p.get("local_id")}

    parent: Dict[Node, Node] = {}

    def find(x: Node) -> Node:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Node, b: Node) -> None:
        parent[find(a)] = find(b)

    def node_for(fid: str, voice: Optional[str]) -> Optional[Node]:
        """The union node for a diarized voice: its resolved person if we have
        one (so trait links land on the same node), else the raw voice."""
        if not voice:
            return None
        m = casts[fid].resolve(voice) if fid in casts else None
        if m and m.person_id:
            return (fid, f"p:{m.person_id}")
        return (fid, f"v:{voice}")

    # 1) SPEECH edges (strong): same voice, same line, different files.
    speech_edges: List[Tuple[Node, Node]] = []
    for g in groups:
        voiced = [node_for(a.file_id, a.speaker) for a in g.attempts if a.speaker]
        voiced = [v for v in voiced if v is not None]
        for i in range(1, len(voiced)):
            if voiced[i][0] != voiced[0][0]:      # cross-file only
                union(voiced[0], voiced[i])
                speech_edges.append((voiced[0], voiced[i]))

    # 2) TRAIT edges (weak): persons in different files whose appearance
    #    descriptions overlap. Every person node exists here even if silent.
    trait_toks: Dict[Node, FrozenSet[str]] = {}
    for fid, cards in persons_by_fid.items():
        for pid, card in cards.items():
            toks = _trait_tokens(card)
            if toks:
                trait_toks[(fid, f"p:{pid}")] = toks
    tnodes = list(trait_toks)
    for i in range(len(tnodes)):
        for j in range(i + 1, len(tnodes)):
            na, nb = tnodes[i], tnodes[j]
            if na[0] == nb[0]:                    # same clip -> not a cross fact
                continue
            ta, tb = trait_toks[na], trait_toks[nb]
            if len(ta & tb) < MIN_TRAIT_TOKENS:
                continue
            if _trait_sim(ta, tb) >= TRAIT_SIM_THRESHOLD:
                union(na, nb)

    # 3) Gather clusters (a person node with no edges is its own singleton and
    #    is dropped below -- identity is a CROSS-clip fact).
    clusters: Dict[Node, List[Node]] = {}
    for node in list(parent):
        clusters.setdefault(find(node), []).append(node)

    speech_roots = {find(a) for a, _ in speech_edges} | {find(b) for _, b in speech_edges}

    out: List[dict] = []
    n = 0
    for root, nodes in clusters.items():
        files = {f for f, _ in nodes}
        if len(files) < 2:
            continue
        n += 1
        basis = "speech" if root in speech_roots else "traits"
        members: List[dict] = []
        desc: Optional[str] = None
        for fid, key in sorted(nodes):
            person: Optional[str] = None
            voice: Optional[str] = None
            role: Optional[str] = None
            on_camera: Optional[bool] = None
            if key.startswith("p:"):
                person = key[2:]
                m = next((cm for cm in casts[fid].members
                          if cm.person_id == person), None) if fid in casts else None
                if m:
                    voice, role, on_camera = m.voice_speaker_id, m.role, m.on_camera
                if desc is None:
                    desc = (persons_by_fid.get(fid, {}).get(person) or {}).get(
                        "canonical_description")
            else:                                 # v:<voice>
                voice = key[2:]
                m = casts[fid].resolve(voice) if fid in casts else None
                if m:
                    role, on_camera = m.role, m.on_camera
            members.append({"file": fid, "person": person, "voice": voice,
                            "role": role, "on_camera": on_camera})
        out.append({"global_id": f"G{n}", "members": members, "description": desc,
                    "basis": basis,
                    "confidence": _SPEECH_CONF if basis == "speech" else _TRAIT_CONF})
    return out


# --------------------------------------------------------------------------
# Entry + rendering
# --------------------------------------------------------------------------

def build_relations(file_ids: List[str]) -> dict:
    """The shoot-level relation set: global person identities across the clips.
    {} when there is only one clip or no cross-clip person can be established.
    Offsets are computed for internal diagnostics only (never surfaced).
    Fail-open."""
    if len(file_ids or []) < 2:
        return {}
    try:
        groups = takes.build_take_groups(file_ids)
        cross = [g for g in groups if len({a.file_id for a in g.attempts}) >= 2]
        clips = _load_clips(file_ids)
        identities = derive_identities(cross, clips)
        offsets = derive_offsets(cross)          # internal only; not rendered
        if not identities:
            return {}
        return {"identities": identities, "offsets": offsets}
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
    that reference them. Identity only; no time/offset ontology. '' when empty."""
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
        conf = ident.get("confidence")
        basis = ident.get("basis")
        tag = ""
        if basis == "traits":
            tag = f" (appearance-matched, low confidence {conf:.2f})"
        elif conf is not None:
            tag = f" (conf {conf:.2f})"
        lines.append(f"  {ident['global_id']}{desc}: " + " = ".join(parts) + tag)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Registry lookups -- alias a per-clip voice/person to its global id.
# --------------------------------------------------------------------------

def identity_index(relations: dict) -> Dict[Tuple[str, str], str]:
    """{(file_id, key) -> global_id} where key is a voice id, a person id, or
    'voice:'/'person:' either way -- so a caller can look up by whichever handle
    it holds. Empty when there are no identities."""
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
