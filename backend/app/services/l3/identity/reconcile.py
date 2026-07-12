"""
Cross-file person unification (identity_map.plan.md Phase 2): each camera
frames (assumed) one person; this module fingerprints that person from the
LLM's structured `appearance` categories, clusters cameras whose
fingerprints genuinely agree into one `person_id`, and composes the
`oncam`/`alias` maps `footage_map.py` renders into the brain's index.

Deterministic matching over LLM-authored categorical data -- code owns the
clustering/assignment, the LLM only ever describes what it sees (Phase 0).
Never forced: a camera whose fingerprint is too sparse or that disagrees
with everyone stays its own person (over-splitting is the safe failure
mode, not a wrong merge -- see the plan's "Known 1%").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

STABLE_FIELDS = (
    "apparent_gender", "apparent_age_band", "hair", "hair_color",
    "facial_hair", "glasses", "skin_tone", "build",
)
# A merge needs at least this many fields where BOTH sides actually have
# evidence (not just zero disagreement on nothing) -- otherwise two totally
# undescribed cameras would trivially "match" on emptiness.
MIN_SHARED_FIELDS = 3


def build_fingerprint(appearances: List[dict]) -> Dict[str, str]:
    """Majority vote per stable field across every `appearance` dict
    collected for one file (one per person-occurrence across its cuts).
    `None`/`"unsure"` never vote. A tie for most-common value leaves that
    field unset (no clear majority, per the plan) rather than picking
    arbitrarily."""
    fp: Dict[str, str] = {}
    for f in STABLE_FIELDS:
        counts: Dict[str, int] = {}
        for a in appearances:
            v = (a or {}).get(f)
            if not v or v == "unsure":
                continue
            counts[v] = counts.get(v, 0) + 1
        if not counts:
            continue
        best = max(counts.values())
        winners = [v for v, c in counts.items() if c == best]
        if len(winners) == 1:
            fp[f] = winners[0]
    return fp


def _distance(a: Dict[str, str], b: Dict[str, str]) -> Tuple[int, int]:
    """(disagreements, shared_set_fields): fields where BOTH sides have
    evidence count as "shared"; unset on either side contributes neither
    agreement nor disagreement (no evidence, not proof of a match)."""
    shared = 0
    disagree = 0
    for f in STABLE_FIELDS:
        va, vb = a.get(f), b.get(f)
        if va is None or vb is None:
            continue
        shared += 1
        if va != vb:
            disagree += 1
    return disagree, shared


def cluster_files(
    fingerprints: Dict[str, Dict[str, str]], min_shared: int = MIN_SHARED_FIELDS,
) -> Dict[str, str]:
    """`file_id -> person_id` via union-find over pairwise zero-disagreement,
    enough-shared-evidence merges (deterministic agglomeration, Phase 2b).
    `person_id`s are assigned `P0, P1, ...` ordered by each cluster's
    minimum `file_id` -- stable across re-runs of the same file set."""
    parent: Dict[str, str] = {fid: fid for fid in fingerprints}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        # Deterministic tie-break: smaller root wins, independent of union order.
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    ids = sorted(fingerprints.keys())
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            disagree, shared = _distance(fingerprints[a], fingerprints[b])
            if disagree == 0 and shared >= min_shared:
                union(a, b)

    clusters: Dict[str, List[str]] = {}
    for fid in ids:
        clusters.setdefault(find(fid), []).append(fid)

    ordered_roots = sorted(clusters.keys(), key=lambda r: min(clusters[r]))
    file_person: Dict[str, str] = {}
    for i, root in enumerate(ordered_roots):
        pid = f"P{i}"
        for fid in clusters[root]:
            file_person[fid] = pid
    return file_person


@dataclass
class Person:
    person_id: str
    display: str
    fingerprint: Dict[str, str] = field(default_factory=dict)
    file_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id, "display": self.display,
            "fingerprint": self.fingerprint, "file_ids": self.file_ids,
        }


def build_persons(
    file_person: Dict[str, str],
    fingerprints: Dict[str, Dict[str, str]],
    descriptions_by_file: Dict[str, List[str]],
) -> Dict[str, Person]:
    """One `Person` per cluster: display label = the longest raw
    `description` prose seen across the cluster's files (display only,
    never an identity claim -- the plan's own framing); fingerprint =
    the union of the member files' own fields (no conflicts possible,
    clustering already required zero disagreement)."""
    by_person: Dict[str, List[str]] = {}
    for fid, pid in file_person.items():
        by_person.setdefault(pid, []).append(fid)

    persons: Dict[str, Person] = {}
    for pid, fids in by_person.items():
        descs = [d for fid in fids for d in descriptions_by_file.get(fid, []) if d]
        display = max(descs, key=len) if descs else pid
        merged_fp: Dict[str, str] = {}
        for fid in fids:
            for k, v in fingerprints.get(fid, {}).items():
                merged_fp.setdefault(k, v)
        persons[pid] = Person(person_id=pid, display=display, fingerprint=merged_fp, file_ids=sorted(fids))
    return persons


def compose_maps(
    file_person: Dict[str, str],
    persons: Dict[str, Person],
    bound_voice_by_file: Dict[str, Optional[str]],
    groups: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, str], Dict[Tuple[str, str], str]]:
    """(oncam, alias) -- Phase 2c. `oncam[file_id]` = the display name of
    whoever that camera's picture shows. `alias[(file_id, voice)]` = the
    display name behind a voice AS HEARD ON that file: a file's own bound
    voice resolves directly; another voice heard in the same outlook group
    resolves via whichever OTHER member is bound to it (the group's cameras
    collectively cover every voice in the moment). A voice with no
    resolvable person anywhere is simply absent (stays unresolved, same as
    today)."""
    oncam = {fid: persons[pid].display for fid, pid in file_person.items() if pid in persons}

    alias: Dict[Tuple[str, str], str] = {}
    for fid, voice in bound_voice_by_file.items():
        if voice and fid in file_person:
            alias[(fid, voice)] = persons[file_person[fid]].display

    for grp in groups.values():
        members = list(grp["members"])
        for fid in members:
            for other_fid in members:
                if other_fid == fid:
                    continue
                other_voice = bound_voice_by_file.get(other_fid)
                if other_voice and other_fid in file_person:
                    alias.setdefault((fid, other_voice), persons[file_person[other_fid]].display)

    return oncam, alias


def reconcile(
    appearances_by_file: Dict[str, List[dict]],
    descriptions_by_file: Dict[str, List[str]],
    bound_voice_by_file: Dict[str, Optional[str]],
    groups: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """The whole Phase 2 pipeline: fingerprint -> cluster -> compose. Returns
    the persisted-shape payload's `persons`/`file_person` pieces plus the
    derived `oncam`/`alias` maps (as a `{file_id: voice}`-keyed dict for
    `alias`, since jsonb can't key on tuples -- `identity/apply.py`/
    `footage_map.py` handle the `"file_id|voice"` string encoding)."""
    fingerprints = {fid: build_fingerprint(apps) for fid, apps in appearances_by_file.items()}
    file_person = cluster_files(fingerprints)
    persons = build_persons(file_person, fingerprints, descriptions_by_file)
    oncam, alias = compose_maps(file_person, persons, bound_voice_by_file, groups)
    return {
        "persons": [p.to_dict() for p in persons.values()],
        "file_person": file_person,
        "oncam": oncam,
        "alias": alias,
    }
