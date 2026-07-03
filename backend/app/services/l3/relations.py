"""
Cross-clip identity -- one global person (face + voice) across a shoot's clips.

Two independent signals describe a shoot's people, each reliable at ONE thing and
useless at the other. We use each for its own job and bridge them carefully:

  * SPEECH clusters VOICES. Per-clip diarization labels (S0/S1) don't survive
    across clips, but the SAME spoken line delivered in two clips is the same
    human. Matching lines by content -> per-clip-pair one-to-one voice
    correspondence (>= MIN_LINK_VOTES shared lines) -> constrained union gives
    cross-clip VOICE clusters. Constrained by one hard truth: two distinct
    voices in ONE clip are two people, so no merge may co-locate a clip's voices.

  * APPEARANCE clusters FACES. Each clip's on-camera subject has a VLM
    description; clustering those descriptions across clips gives cross-clip
    FACE clusters. (This is appearance doing its OWN job -- grouping visible
    people -- never a fallback that decides voice identity.)

  * A face cluster and a voice cluster are the SAME person only through a
    HIGH-CONFIDENCE per-clip A/V link (the cast's voice<->face link >= the
    bridge bar). Weak links (a clip whose VLM "speaking" spans are inflated, so
    the wrong voice looks on-camera) are IGNORED as bridges -- they can't corrupt
    identity. `on_camera` for a voice is then DERIVED, not trusted per clip: a
    voice is on camera in a clip iff its global person owns that clip's on-camera
    face. One clip's bad sensor can no longer glue a voice to the wrong face.

Everything is DERIVED from artifacts that already exist -- transcript / VLM
content units and the per-clip cast (`l3.cast`). No new perception pass, no
embeddings, no model calls. Fail-open: <2 clips or no cross-clip person -> {}.
Where a face never gets a confident voice (or a voice is never confidently seen)
we SAY so (`warnings`) rather than guessing.
"""
from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Dict, FrozenSet, List, Optional, Tuple

from app.services.l3 import cast as cast_mod
from app.services.l3 import takes

logger = logging.getLogger(__name__)

# A voice<->voice correspondence between two clips is only trusted once this many
# distinct shared lines agree on it, so a single loose content match can't invent
# a cross-clip link. Genuine multicam/overlapping recordings of one conversation
# share far more than this; the threshold just discards the noise floor.
MIN_LINK_VOTES = 2

# A per-clip voice<->face A/V link must be at least this confident to BRIDGE a
# face cluster to a voice cluster. Below it the link is ignored for identity (the
# face/voice still exist; they just don't get fused on that weak evidence).
BRIDGE_CONF = cast_mod._LINK_CONFIDENT      # 0.6

# Two on-camera faces in different clips are the same visual person when their VLM
# appearance descriptions overlap this much (Jaccard over distinctive tokens) and
# share at least this many tokens. Used ONLY to cluster faces, never voices.
FACE_SIM_THRESHOLD = 0.34
MIN_FACE_TOKENS = 2
_WORD_RE = re.compile(r"[a-z0-9']+")
# Generic filler stripped before face matching so build/skin/age can't glue two
# different people; the distinctive tokens (bald, curly, beard, moustache, hair,
# glasses, colours) are what carry a face's identity.
_FACE_STOP = {
    "a", "an", "the", "with", "and", "of", "in", "on", "his", "her", "their",
    "man", "woman", "male", "female", "person", "guy", "adult", "young", "old",
    "middle", "aged", "age", "years", "year", "20s", "30s", "40s", "50s", "60s",
    "average", "build", "medium", "light", "skin", "tone", "complexion",
    "wearing", "wears", "has", "is", "who", "appears", "to", "be", "seen",
    "slightly", "somewhat", "very", "quite", "short", "tall",
}

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
    those with >= MIN_LINK_VOTES AND that are UNAMBIGUOUSLY dominant -- the vote
    must strictly beat the runner-up for BOTH voices. Two co-temporal cameras of
    one conversation share dozens of lines on the true pairing and only a noisy
    handful on any other, so the true edges dominate; two clips that merely share
    stray filler phrases tie across pairings and are dropped (they must NOT invent
    a cross-clip voice link -- appearance will link those people, not this)."""
    best_a: Dict[str, int] = {}
    best_b: Dict[str, int] = {}
    for (va, vb), v in cnt.items():
        best_a[va] = max(best_a.get(va, 0), v)
        best_b[vb] = max(best_b.get(vb, 0), v)
    used_a: set = set()
    used_b: set = set()
    edges: List[Tuple[int, str, str]] = []
    for (va, vb), v in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])):
        if v < MIN_LINK_VOTES:
            break
        if va in used_a or vb in used_b:
            continue
        runner_a = max([w for (a, _b), w in cnt.items() if a == va and _b != vb] or [0])
        runner_b = max([w for (_a, b), w in cnt.items() if b == vb and _a != va] or [0])
        if v <= runner_a or v <= runner_b:        # tied/ambiguous -> not a real link
            continue
        used_a.add(va)
        used_b.add(vb)
        edges.append((v, va, vb))
    return edges


def _voice_clusters(attempts: List[takes.Attempt]) -> List[List[Node]]:
    """Cross-clip VOICE clusters from matched speech: content-cluster the lines,
    vote per clip pair, resolve each pair to a one-to-one matching, then union
    under the hard same-clip-distinctness constraint (two voices in one clip are
    two people -- no merge may co-locate them). Returns lists of (file, voice)."""
    clusters = _cluster_by_content(attempts)
    votes = _pair_votes(clusters)
    edges: List[Tuple[int, Node, Node]] = []
    for (fa, fb), cnt in votes.items():
        for v, va, vb in _match_pair(cnt):
            edges.append((v, (fa, va), (fb, vb)))
    edges.sort(key=lambda e: -e[0])

    parent: Dict[Node, Node] = {}
    files_of: Dict[Node, Dict[str, str]] = {}

    def find(n: Node) -> Node:
        if n not in parent:
            parent[n] = n
            files_of[n] = {n[0]: n[1]}
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a: Node, b: Node) -> None:
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
    return list(groups.values())


# --- Face side: cluster on-camera faces by APPEARANCE -----------------------

def _appearance_tokens(desc: Optional[str]) -> FrozenSet[str]:
    """Distinctive content tokens of a face description (generic filler removed,
    grey/gray normalized) -- the surface two clips' faces are matched on."""
    if not desc:
        return frozenset()
    toks = set()
    for t in _WORD_RE.findall(desc.lower()):
        if t == "grey":
            t = "gray"
        if t not in _FACE_STOP and len(t) > 1:
            toks.add(t)
    return frozenset(toks)


def _face_sim(a: FrozenSet[str], b: FrozenSet[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _primary_face(perception: Optional[dict],
                  cast: cast_mod.ClipCast) -> Optional[Tuple[str, str]]:
    """The clip's on-camera SUBJECT (person id, appearance): the visible person
    with the most on-camera speaking coverage, else the first VLM person. This is
    the reliable 'whose face is on screen' -- independent of any voice link."""
    persons = [p for p in ((perception or {}).get("persons") or []) if p.get("local_id")]
    if not persons:
        return None
    speaking = cast_mod._speaking_by_person((perception or {}).get("speaking") or [])

    def coverage(pid: str) -> int:
        return sum(b - a for a, b in speaking.get(pid, []))

    primary = max(persons, key=lambda p: coverage(p.get("local_id", "")))
    return (primary["local_id"], primary.get("canonical_description") or "")


def _reconcile(attempts: List[takes.Attempt],
               clips: Dict[str, Tuple[Optional[dict], List[dict]]]) -> dict:
    """Fuse VOICE clusters (speech) and FACE clusters (appearance) into global
    people via high-confidence A/V bridges, then DERIVE on_camera. Returns
    {"identities": [...], "warnings": [...]}."""
    casts: Dict[str, cast_mod.ClipCast] = {}
    faces: Dict[str, Tuple[str, str]] = {}          # file -> (pid, appearance)
    face_tokens: Dict[Node, FrozenSet[str]] = {}    # ("f", file, pid) surrogate via (file,pid)
    for fid, (perception, words) in clips.items():
        casts[fid] = cast_mod.build_cast(perception, words)
        pf = _primary_face(perception, casts[fid])
        if pf:
            faces[fid] = pf
            face_tokens[(fid, pf[0])] = _appearance_tokens(pf[1])

    # Nodes live in one union space: voices tagged ("v", file, voice), faces
    # ("f", file, pid). The union is CONSTRAINED: it refuses any merge that would
    # place two voices of the SAME clip in one person -- the hard truth that keeps
    # a single wrong edge from collapsing the whole cast.
    parent: Dict[tuple, tuple] = {}
    files_of: Dict[tuple, Dict[str, str]] = {}      # root -> {file: voice} it owns

    def touch(n: tuple) -> None:
        if n not in parent:
            parent[n] = n
            files_of[n] = {n[1]: n[2]} if n[0] == "v" else {}

    def find(n: tuple) -> tuple:
        touch(n)
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n

    def union(a: tuple, b: tuple) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return True
        fa, fb = files_of[ra], files_of[rb]
        for f, v in fb.items():
            if fa.get(f, v) != v:                   # two voices of one clip -> refuse
                return False
        parent[rb] = ra
        fa.update(fb)
        files_of.pop(rb, None)
        return True

    # 1) VOICE clusters (each already <=1 voice per clip; unions never conflict).
    for cluster in _voice_clusters(attempts):
        vnodes = [("v", f, v) for (f, v) in cluster]
        for other in vnodes[1:]:
            union(vnodes[0], other)

    # 2) FACE clusters (appearance similarity, greedy).
    face_list = list(face_tokens.items())
    for i in range(len(face_list)):
        (fa, pa), ta = face_list[i]
        for j in range(i + 1, len(face_list)):
            (fb, pb), tb = face_list[j]
            if fa == fb:
                continue
            if len(ta & tb) >= MIN_FACE_TOKENS and _face_sim(ta, tb) >= FACE_SIM_THRESHOLD:
                union(("f", fa, pa), ("f", fb, pb))

    # 3) BRIDGES -- fuse a face cluster to a voice cluster, resolved PER VOICE
    #    CLUSTER, one-to-one, by confidence. A clip's on-camera face casts a claim
    #    on the voice it links to (strength = A/V link confidence). Each voice
    #    cluster then goes to its single STRONGEST claimant face. This is what
    #    survives one clip's sensor error: a wrong-but-confident link (a clip whose
    #    VLM "speaking" spans are inflated, so the loud OTHER voice looks
    #    on-camera) loses to the correct face that owns that voice elsewhere.
    claims: Dict[tuple, List[Tuple[float, str, str, str]]] = {}
    for fid, (pid, _appr) in faces.items():
        m = next((x for x in casts[fid].members
                  if x.person_id == pid and x.voice_speaker_id), None)
        if m and m.voice_speaker_id:
            vrep = find(("v", fid, m.voice_speaker_id))
            claims.setdefault(vrep, []).append(
                (float(m.av_link_confidence), fid, pid, m.voice_speaker_id))

    contradiction_files: set = set()
    weak_link_files: List[str] = []
    for vrep, cl in claims.items():
        cl.sort(key=lambda c: -c[0])
        conf, fid, pid, voice = cl[0]
        if conf < BRIDGE_CONF:
            weak_link_files.append(fid)
            continue
        union(("f", fid, pid), ("v", fid, voice))               # strongest claim wins
        winner_face = find(("f", fid, pid))
        for c2, f2, p2, _v2 in cl[1:]:                          # losing claims
            if c2 >= BRIDGE_CONF and find(("f", f2, p2)) != winner_face:
                contradiction_files.add(f2)                     # this clip's link was wrong

    # 4) Voice completion by elimination: in a clip whose on-camera face has no
    #    voice yet, if exactly ONE of the clip's voices is still unbridged (the
    #    rest already belong to OTHER people), that voice must be the on-camera
    #    person's -- assign it. Recovers the true on-camera voice of a clip whose
    #    own A/V link was wrong (its partner's voice was identified via the other
    #    camera), never using the bad link.
    voices_by_file: Dict[str, set] = {}
    for a in attempts:
        if a.speaker:
            voices_by_file.setdefault(a.file_id, set()).add(a.speaker)

    def _has_face(root: tuple) -> bool:
        return any(n[0] == "f" and find(n) == root for n in list(parent))

    for fid, (pid, _a) in faces.items():
        vs = voices_by_file.get(fid, set())
        if not vs:
            continue
        face_root = find(("f", fid, pid))
        if any(find(("v", fid, v)) == face_root for v in vs):   # face already voiced here
            continue
        free = [v for v in vs if not _has_face(find(("v", fid, v)))]
        if len(free) == 1:
            union(("v", fid, free[0]), ("f", fid, pid))

    # 5) Gather components -> global people.
    comps: Dict[tuple, List[tuple]] = {}
    for n in list(parent):
        comps.setdefault(find(n), []).append(n)

    def _consensus_appearance(pids: List[Tuple[str, str]]) -> Optional[str]:
        """Longest description among a person's faces (a proxy for the most
        detailed / canonical); all faces in a cluster are already similar."""
        descs = [faces[f][1] for (f, p) in pids if f in faces and faces[f][0] == p]
        descs = [d for d in descs if d]
        return max(descs, key=len) if descs else None

    identities: List[dict] = []
    warnings: List[str] = []
    n_id = 0
    for _root, nodes in sorted(comps.items(), key=lambda kv: str(sorted(kv[1]))):
        vnodes = [(f, v) for (t, f, v) in nodes if t == "v"]
        fnodes = [(f, p) for (t, f, p) in nodes if t == "f"]
        files = {f for f, _ in vnodes} | {f for f, _ in fnodes}
        if len(files) < 2:                        # identity is a CROSS-clip fact
            continue
        n_id += 1
        gid = f"G{n_id}"
        oncam_files = {f for f, _ in fnodes}        # clips where THIS person's face is on screen
        voice_by_file = {f: v for f, v in vnodes}
        role = None
        for f, v in vnodes:
            r = (casts[f].resolve(v).role if casts[f].resolve(v) else None) if f in casts else None
            role = role or r
        members: List[dict] = []
        for f in sorted(files):
            pid = faces[f][0] if f in oncam_files and f in faces else None
            members.append({
                "file": f,
                "voice": voice_by_file.get(f),
                "person": pid,
                "role": role,
                "on_camera": True if f in oncam_files else (
                    False if f in voice_by_file else None),
            })
        desc = _consensus_appearance(fnodes)
        if not fnodes:
            warnings.append(f"{gid}: heard in {len(files)} clip(s) but never "
                            f"confidently seen on camera.")
        identities.append({"global_id": gid, "members": members, "description": desc})

    # Faces that never bridged to any voice: visible but unlinked -> say so.
    for fid, (pid, _a) in faces.items():
        root = find(("f", fid, pid))
        if not any(t == "v" for (t, _f, _v) in comps.get(root, [])):
            warnings.append(f"{_fid8(fid)}: on-camera face never confidently "
                            f"linked to a voice.")
    for fid in sorted(contradiction_files):
        warnings.append(f"{_fid8(fid)}: its A/V link claimed a voice that belongs "
                        f"to another person (likely inflated visual-speaking); "
                        f"overridden -- on-camera voice derived by elimination.")
    for fid in sorted(set(weak_link_files) - contradiction_files):
        warnings.append(f"{_fid8(fid)}: voice<->face link below the bridge bar; "
                        f"on-camera speaker there was derived, not trusted.")
    return {"identities": identities, "warnings": warnings}


def derive_identities(attempts: List[takes.Attempt],
                      clips: Dict[str, Tuple[Optional[dict], List[dict]]]) -> List[dict]:
    """Global people across the shoot (reconciled face + voice). Thin wrapper over
    ``_reconcile`` returning just the identity list (see ``build_relations`` for
    the warnings)."""
    if not attempts and not clips:
        return []
    return _reconcile(attempts, clips)["identities"]


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
        recon = _reconcile(attempts, clips)
        if not recon["identities"]:
            return {}
        return {"identities": recon["identities"],
                "warnings": recon.get("warnings") or []}
    except Exception:
        logger.exception("relations: build failed (continuing without)")
        return {}


def render_relations(relations: dict) -> str:
    """The shoot ORIENTATION header -- the first thing the brain reads, naming
    every global person and, per person, the clips that SHOW their face vs. the
    clips where they're only HEARD off-camera. Prepended to source_awareness so
    the people (and which camera holds each) are known BEFORE the per-clip blocks.
    Honest: any identity uncertainty is stated as a NOTE, never hidden. '' when
    there are no cross-clip people."""
    if not relations:
        return ""
    idents = relations.get("identities") or []
    if not idents:
        return ""
    lines = ["PEOPLE OF THE SHOOT (one person across all clips -- use the global "
             "id; each person's own camera vs. where they're only heard):"]
    for ident in idents:
        members = ident.get("members") or []
        on = [_fid8(m["file"]) for m in members if m.get("on_camera") is True]
        off = [_fid8(m["file"]) for m in members if m.get("on_camera") is False]
        desc = f' "{ident["description"]}"' if ident.get("description") else ""
        bits = []
        if on:
            bits.append("on camera in " + ", ".join(sorted(on)))
        if off:
            bits.append("heard off-camera in " + ", ".join(sorted(off)))
        tail = ("; ".join(bits)) if bits else "presence unresolved"
        lines.append(f"  {ident['global_id']}{desc}: {tail}")
    for w in (relations.get("warnings") or []):
        lines.append(f"  NOTE {w}")
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


def oncam_global_by_file(relations: dict) -> Dict[str, str]:
    """{file_id -> global_id} of the person whose FACE is on camera in that clip
    (the on-camera member). Lets a consumer answer 'whose face shows here?'
    regardless of who is speaking -- the key fact for reaction/listening cuts."""
    out: Dict[str, str] = {}
    for ident in (relations or {}).get("identities") or []:
        gid = ident.get("global_id")
        for m in ident.get("members") or []:
            if m.get("on_camera") is True and m.get("file"):
                out[m["file"]] = gid
    return out


def validate(relations: dict) -> List[str]:
    """Substrate invariants over the reconciled cast -- a safety net that turns a
    silent structural corruption into an explicit, honest warning (never raises).
    Checks the facts the rest of the pipeline RELIES on:

      * one clip's face is shown by at most ONE global person (a clip can't be
        two people's on-camera coverage at once);
      * one (clip, voice) belongs to at most ONE global person (a voice is one
        human);
      * a global person is on-camera in a clip via at most one voice.

    Returns the list of violation strings (empty when the cast is coherent)."""
    problems: List[str] = []
    idents = (relations or {}).get("identities") or []
    oncam_owner: Dict[str, str] = {}
    voice_owner: Dict[Tuple[str, str], str] = {}
    for ident in idents:
        gid = ident.get("global_id")
        seen_oncam_files: set = set()
        for m in ident.get("members") or []:
            fid = m.get("file")
            if not fid:
                continue
            if m.get("on_camera") is True:
                prev = oncam_owner.get(fid)
                if prev and prev != gid:
                    problems.append(f"clip {_fid8(fid)} is on-camera for both "
                                    f"{prev} and {gid} (two faces claimed).")
                oncam_owner[fid] = gid
                if fid in seen_oncam_files:
                    problems.append(f"{gid} is on-camera in {_fid8(fid)} twice.")
                seen_oncam_files.add(fid)
            v = m.get("voice")
            if v:
                prev_v = voice_owner.get((fid, v))
                if prev_v and prev_v != gid:
                    problems.append(f"voice {_fid8(fid)}:{v} belongs to both "
                                    f"{prev_v} and {gid}.")
                voice_owner[(fid, v)] = gid
    return problems


def to_json(relations: dict) -> str:
    return json.dumps(relations, default=str)
