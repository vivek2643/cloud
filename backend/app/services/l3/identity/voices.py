"""
Cross-clip voice clustering (voice_first_identity.plan.md Phase B): cluster
per-file diarized speaker voiceprints (L1 diarization Phase A embeddings)
into GLOBAL voices V0..Vn, so the same person's voice across different clips
resolves to one identity -- the cross-clip spine `identity/voice_id.py` keys
off (voice_id_pass.plan.md).

Deterministic cosine-similarity agglomeration, CONSERVATIVE threshold.
Identity anchor is the PERSON (face; see `identity/reconcile.py`), and the
voice binds TO the person (`identity/voice_id.py`) -- so if voice
clustering OVER-SPLITS (the same person sounds different in two clips), both
fragments still bind to the same person later and self-heal. The only
dangerous direction is voice MERGING two different people's voices, so this
stays conservative -- same "over-split is the safe failure" stance
`identity/reconcile.py` already takes for faces.

Outlook-group members share ONE authoritative audio track (`sync.lattice_
merge.authoritative_view` already re-based their turns onto it, before pass 1
ever ran), so a group's shared speaker labels are unified WITHOUT an
embedding comparison -- they are the same recording by construction. This
unification works off the LABEL ROSTER (every local speaker a file actually
carries, `all_speakers_by_file` -- typically Pass 1's per-file union of
`SpeechCut.speaker_ids`), not off which speakers happen to have an
embedding: a non-authoritative angle in a group usually has NO diarization
of its own (its speech was swapped for the authoritative source's before
pass 1 ever ran, so it was never separately diarized/embedded) but still
needs its share of the group's speaker labels to resolve to the same voice.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Conservative: cosine similarity must be at least this high to merge two
# voiceprints into one global voice. High bar on purpose -- see module
# docstring on why merging is the dangerous direction, not over-splitting.
VOICE_MERGE_COSINE = 0.75


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


def cluster_voices(
    embeddings_by_file: Dict[str, Dict[str, List[float]]],
    groups: Dict[str, Dict[str, Any]],
    all_speakers_by_file: Optional[Dict[str, List[str]]] = None,
    threshold: float = VOICE_MERGE_COSINE,
) -> Dict[Tuple[str, str], str]:
    """(file_id, local_speaker) -> global voice id ("V0", "V1", ...), for
    EVERY speaker in the roster -- `all_speakers_by_file` when given
    (typically Pass 1's per-file union of `SpeechCut.speaker_ids`: the
    complete set of speakers that actually talk, embedding or not), else
    falls back to just the embedding-bearing speakers (convenient for
    embedding-only tests/callers). `groups`: the SAME outlook-group dict
    `identity/apply.py`/`sync.lattice_merge` use ({group_id: {"auth":
    file_id, "members": {file_id, ...}}}).

    Deterministic union-find over the full roster: outlook-group members'
    matching local-speaker labels are unified first (label identity alone,
    no embedding needed -- same audio by construction, see module
    docstring), then any two (file, local_speaker) pairs in DIFFERENT files
    that BOTH have voiceprints with cosine similarity >= `threshold` are
    merged. A roster speaker with no embedding and no group-mate simply
    stays its own singleton voice (identity_map.plan.md's "over-split is
    the safe failure" stance -- never a wrong merge, never dropped). Voice
    ids are assigned "V0", "V1", ... ordered by each cluster's minimum
    (file_id, local_speaker) key, stable across re-runs of the same
    embeddings/roster."""
    roster = all_speakers_by_file or {
        fid: list(embs.keys()) for fid, embs in embeddings_by_file.items()
    }
    nodes: List[Tuple[str, str]] = sorted(
        (fid, spk) for fid, spks in roster.items() for spk in set(spks)
    )
    parent: Dict[Tuple[str, str], Tuple[str, str]] = {n: n for n in nodes}

    def find(x: Tuple[str, str]) -> Tuple[str, str]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: Tuple[str, str], b: Tuple[str, str]) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Deterministic tie-break: smaller root wins, independent of union order.
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    # 1. Outlook-group unification, over the ROSTER (not just embedding-
    # bearing speakers) -- every member file of a group shares one
    # authoritative audio track, so identically-named local speaker labels
    # across members are the SAME voice by construction, whether or not
    # that particular member was ever separately diarized/embedded.
    for grp in groups.values():
        members = sorted(grp.get("members") or [])
        member_speakers: Dict[str, set] = {fid: set(roster.get(fid, [])) for fid in members}
        all_speakers = sorted({spk for spks in member_speakers.values() for spk in spks})
        for spk in all_speakers:
            owners = [fid for fid in members if spk in member_speakers[fid]]
            for prev, nxt in zip(owners, owners[1:]):
                union((prev, spk), (nxt, spk))

    # 2. Cross-clip cosine clustering, restricted to nodes that actually
    # have a voiceprint, conservative threshold, fixed deterministic order.
    emb_nodes = [n for n in nodes if n[1] in embeddings_by_file.get(n[0], {})]
    for i in range(len(emb_nodes)):
        fid_a, spk_a = emb_nodes[i]
        vec_a = embeddings_by_file[fid_a][spk_a]
        for j in range(i + 1, len(emb_nodes)):
            fid_b, spk_b = emb_nodes[j]
            if fid_a == fid_b:
                continue  # two local speakers in the SAME file are never one voice
            vec_b = embeddings_by_file[fid_b][spk_b]
            if _cosine(vec_a, vec_b) >= threshold:
                union(emb_nodes[i], emb_nodes[j])

    clusters: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    for n in nodes:
        clusters.setdefault(find(n), []).append(n)

    ordered_roots = sorted(clusters.keys(), key=lambda r: min(clusters[r]))
    voice_of: Dict[Tuple[str, str], str] = {}
    for i, root in enumerate(ordered_roots):
        vid = f"V{i}"
        for n in clusters[root]:
            voice_of[n] = vid
    return voice_of


def assign_voices(
    embeddings_by_file: Dict[str, Dict[str, List[float]]],
    groups: Dict[str, Dict[str, Any]],
    all_speakers_by_file: Dict[str, List[str]],
    threshold: float = VOICE_MERGE_COSINE,
) -> Dict[Tuple[str, str], str]:
    """Thin, explicit alias for `cluster_voices` with the full speaker
    roster required (not optional) -- the entry point `identity/apply.py`
    calls: every (file_id, local_speaker) pass 1 says actually spoke gets a
    global voice id, covered or not by an embedding."""
    return cluster_voices(embeddings_by_file, groups, all_speakers_by_file, threshold)
