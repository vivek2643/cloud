"""
Cross-file face-track clustering (asd_identity.plan.md): replaces identity/
reconcile.py's LLM-categorical-label clustering entirely. Face embeddings
(ArcFace, from L1's app.services.l1.active_speaker.FaceTrack.embedding) are
a robust MACHINE identity signal -- clustering on them, rather than on noisy
per-cut LLM appearance labels that flip run-to-run, is the whole point of
this plan (a 2-person podcast used to over-split into a dozen "persons"
because "beard: yes"/"shirt: blue" labels aren't stable; embeddings don't
drift the way free-text labels do).

Deterministic union-find over (file_id, track_id) nodes, cosine similarity,
CONSERVATIVE threshold -- mirrors identity/voices.py's stance exactly:
over-splitting (the same person shows up as two cast rows) is the safe
failure, never a wrong merge.

Person characterization (display/tags) is explicitly DE-SCOPED for this
pass (asd_identity.plan.md SS3: "No characterize.py, no dedicated call") --
`cluster`'s persons carry only the deterministic globals (person id,
appearance_count, is_major, owned_voices). `_cast_line` (footage_map.py)
already falls back to the bare person_id when `display` is unset, so this
needs no consumer-side change.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from app.services.l1.active_speaker import FaceTrack
from app.services.l3.lattice import resolve_speech_span_ms

# Conservative: ArcFace cosine similarity must be at least this high to merge
# two tracks into one global person. MobileFaceNet-backbone (buffalo_sc)
# same-identity similarities commonly run ~0.4-0.6+; this sits on the
# stricter end deliberately (identity/voices.py's "over-split is the safe
# failure" stance) -- tune once real footage validates it (asd_identity.
# plan.md SS12's open question).
FACE_MERGE_COSINE = 0.45

# A cut showing more than this many distinct persons at once is a crowd --
# list only the most prominent few (by mean face area) rather than an
# exhaustive, unreadable roster. Mirrors identity/reconcile.py's old
# CROWD_SIZE spirit, though here it caps the per-cut LISTING only --
# clustering itself is unaffected by any one cut's crowd status.
MAX_VISIBLE_PER_CUT = 6


def _cosine(a: List[float], b: List[float]) -> float:
    """-1.0 (never a match) for empty/mismatched-length/zero vectors --
    never divides by zero, never fabricates a similarity."""
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na <= 0 or nb <= 0:
        return -1.0
    return dot / (na * nb)


def cluster(
    face_tracks_by_file: Dict[str, List[FaceTrack]], threshold: float = FACE_MERGE_COSINE,
) -> Tuple[Dict[Tuple[str, int], str], Dict[str, Dict[str, Any]]]:
    """(file_id, track_id) -> global person id ("P0", "P1", ...), plus a
    thin per-person record ({person_id, appearance_count, is_major,
    owned_voices}) -- the cast table's deterministic core. Deterministic
    union-find, conservative cosine threshold: a track with no embedding or
    no close match simply stays its own unmerged person (over-split-safe,
    never a wrong merge). Person ids are assigned by each cluster's minimum
    (file_id, track_id) key, stable across re-runs of the same tracks. The
    cast table is UNCAPPED -- every reconciled person is a full row
    (`is_major=True`), mirroring identity/reconcile.py's prior stance."""
    nodes: List[Tuple[str, int]] = sorted(
        (fid, tr.track_id) for fid, tracks in face_tracks_by_file.items() for tr in tracks
    )
    track_by_node: Dict[Tuple[str, int], FaceTrack] = {
        (fid, tr.track_id): tr for fid, tracks in face_tracks_by_file.items() for tr in tracks
    }

    parent: Dict[Tuple[str, int], Tuple[str, int]] = {n: n for n in nodes}

    def find(x: Tuple[str, int]) -> Tuple[str, int]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Tuple[str, int], b: Tuple[str, int]) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    emb_nodes = [n for n in nodes if track_by_node[n].embedding]
    for i in range(len(emb_nodes)):
        fid_a = emb_nodes[i][0]
        vec_a = track_by_node[emb_nodes[i]].embedding
        for j in range(i + 1, len(emb_nodes)):
            fid_b = emb_nodes[j][0]
            if fid_a == fid_b:
                continue  # two tracks in the SAME file are never one person
            vec_b = track_by_node[emb_nodes[j]].embedding
            if _cosine(vec_a, vec_b) >= threshold:
                union(emb_nodes[i], emb_nodes[j])

    clusters: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}
    for n in nodes:
        clusters.setdefault(find(n), []).append(n)

    ordered_roots = sorted(clusters.keys(), key=lambda r: min(clusters[r]))
    track_to_person: Dict[Tuple[str, int], str] = {}
    persons: Dict[str, Dict[str, Any]] = {}
    for i, root in enumerate(ordered_roots):
        pid = f"P{i}"
        members = clusters[root]
        for n in members:
            track_to_person[n] = pid
        persons[pid] = {
            "person_id": pid,
            "appearance_count": sum(len(track_by_node[n].frames) for n in members),
            "is_major": True,
            "owned_voices": [],   # filled in later by identity/apply.py, once bind_asd.bind resolves voices
        }
    return track_to_person, persons


def _cut_span_ms(cut: Any, lattice: Any,
                  span_override: Optional[Dict[str, Tuple[int, int]]] = None) -> Optional[Tuple[int, int]]:
    """Best-effort (s, e) ms for one Pass2Cut-shaped object (duck-typed --
    file_id/kind/word_span/atom_ids -- this module stays decoupled from the
    pass2 schema, same pattern the rest of identity/ already uses).

    ``span_override`` (cuts_v4_segmentation.plan.md): {source_ref: (s, e)} for
    a V4 video cut, whose real span is the segmenter's own tight span, NOT the
    bounding box of the (coarser, informational-only) atoms it happens to
    overlap -- see v4_segment.segment_video's module docstring. None/absent
    for every cut falls through to the atom-membership resolution below,
    identical to today's behavior (the V3 path, and any cut with no override)."""
    if span_override and cut.source_ref in span_override:
        return span_override[cut.source_ref]
    if lattice is None:
        return None
    if cut.kind == "speech" and cut.word_span:
        return resolve_speech_span_ms(lattice.words, lattice.atoms, tuple(cut.word_span), [])
    if cut.kind == "video" and cut.atom_ids:
        atoms_by_id = {a.atom_id: a for a in lattice.atoms}
        members = [atoms_by_id[i] for i in cut.atom_ids if i in atoms_by_id]
        if not members:
            return None
        return min(a.start_ms for a in members), max(a.end_ms for a in members)
    return None


def visible_persons_by_cut(
    track_to_person: Dict[Tuple[str, int], str],
    face_tracks_by_file: Dict[str, List[FaceTrack]],
    cuts: List[Any],
    lattices: Dict[str, Any],
    span_override: Optional[Dict[str, Tuple[int, int]]] = None,
) -> Dict[Tuple[str, str], List[str]]:
    """(file_id, source_ref) -> sorted, distinct global person ids visible
    in that cut. Deterministic, from tracks: a track counts as visible in a
    cut when at least one of its sampled `frames[]` boxes falls inside the
    cut's resolved ms span. Capped to MAX_VISIBLE_PER_CUT persons (ranked
    by mean face-box area, most prominent first) so a crowd shot stays a
    readable signal rather than an exhaustive roster.

    ``span_override``: see ``_cut_span_ms`` -- {source_ref: (s, e)} for a V4
    ingest's video cuts, whose real span is the segmenter's own, not the
    atom-membership bounding box. None/empty for a V3 run (today's behavior)."""
    out: Dict[Tuple[str, str], List[str]] = {}
    cuts_by_file: Dict[str, List[Any]] = {}
    for cut in cuts:
        cuts_by_file.setdefault(cut.file_id, []).append(cut)

    for fid, file_cuts in cuts_by_file.items():
        tracks = face_tracks_by_file.get(fid, [])
        lattice = lattices.get(fid)
        for cut in file_cuts:
            span = _cut_span_ms(cut, lattice, span_override)
            if span is None:
                continue
            s, e = span
            areas: Dict[str, List[int]] = {}
            for tr in tracks:
                person = track_to_person.get((fid, tr.track_id))
                if person is None:
                    continue
                in_span = [f for f in tr.frames if s <= f.t_ms < e]
                if not in_span:
                    continue
                areas.setdefault(person, []).extend(f.box[2] * f.box[3] for f in in_span)
            if not areas:
                continue
            ranked = sorted(areas.items(), key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True)
            out[(fid, cut.source_ref)] = sorted(p for p, _ in ranked[:MAX_VISIBLE_PER_CUT])
    return out
