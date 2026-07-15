"""
Cross-cut person unification (voice_first_identity.plan.md Phase D,
superseding identity_map.plan.md Phase 2's per-FILE fingerprinting): each
PERSON-OCCURRENCE -- one visible-person entry within one cut's `people[]`
list -- is fingerprinted from the LLM's structured `appearance` categories
and clustered into global `person_id`s by the SAME deterministic zero-
disagreement matching identity_map.plan.md always used, just fed
occurrences instead of whole files. This drops the "one camera frames one
person" assumption entirely: a single cut showing two people now clusters
each of them independently, and BOTH resolve to real global persons.

Deterministic matching over LLM-authored categorical data -- code owns the
clustering/ranking, the LLM only ever describes what it sees (pass 2's
`people`/`appearance`). Never forced: an occurrence whose fingerprint is too
sparse or that disagrees with everyone stays its own person (over-splitting
is the safe failure mode, not a wrong merge -- see identity_map.plan.md's
"Known 1%", carried over unchanged).

Every resulting person becomes a full, named cast-table member -- the cast
table is UNCAPPED (as large as the shoot's real, distinct-clustered cast),
so a five-person panel keeps all five just as a two-host podcast keeps both.
A PROMINENCE signal (how often a person appears, whether they co-occur with
speech, how large the cut's framing tends to be) is still computed for
ordering, but no real person is ever hidden behind a fixed ceiling. A single
cut with too many simultaneous people (a crowd shot) is excluded from
clustering entirely --
no basis to tell individuals apart occurrence by occurrence, and forcing an
id per face there would be noise, not signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

STABLE_FIELDS = (
    "apparent_gender", "apparent_age_band", "hair", "hair_color",
    "facial_hair", "glasses", "skin_tone", "build",
)
# Fields the clustering actually COMPARES. `apparent_age_band` is deliberately
# excluded: Flash-Lite emits it as free text with no fixed vocabulary ("30s",
# "30-40", "adult", "young_adult", "middle_aged" all seen for the same man), so
# it splits far more people than it ever correctly separates. It's still kept
# in fingerprints for display -- just never used as a merge signal.
DISTANCE_FIELDS = (
    "apparent_gender", "hair", "hair_color",
    "facial_hair", "glasses", "skin_tone", "build",
)
# A merge needs at least this many fields where BOTH sides actually have
# evidence (not just zero disagreement on nothing) -- otherwise two totally
# undescribed occurrences would trivially "match" on emptiness.
MIN_SHARED_FIELDS = 3
# Two clusters merge when at most this FRACTION of their shared, evidenced
# fields disagree (after value normalization). Strict zero-disagreement was too
# brittle for per-frame LLM labels -- one flipped field ("beard"->"full_beard",
# "bald"->"short") spawned a whole new person. A small tolerance absorbs that
# wording noise; centroid (average-linkage) agglomeration below keeps it from
# chaining two genuinely different people together.
MAX_DISAGREE_FRAC = 0.25

# Canonical value buckets: collapse the LLM's synonymous / near-synonymous
# labels so the same attribute described two ways stops looking like a
# disagreement. Anything unmapped passes through unchanged (still comparable
# to an identical unmapped value); values that reduce to "" drop out entirely.
_VALUE_NORMALIZERS: Dict[str, Dict[str, str]] = {
    "apparent_gender": {
        "man": "male", "m": "male", "woman": "female", "f": "female",
    },
    "skin_tone": {
        "very_light": "light", "fair": "light", "pale": "light", "white": "light",
        "tan": "medium", "olive": "medium", "brown": "medium", "beige": "medium",
        "deep": "dark", "black": "dark",
    },
    "build": {
        "thin": "slim", "lean": "slim", "slender": "slim", "skinny": "slim",
        "medium": "average", "normal": "average", "athletic": "average",
        "large": "heavy", "broad": "heavy", "stocky": "heavy", "big": "heavy",
        "muscular": "heavy", "overweight": "heavy",
    },
    "hair": {
        "shaved": "bald", "none": "bald", "buzz": "short", "buzzcut": "short",
        "very_short": "short", "cropped": "short", "medium_length": "medium",
    },
    "hair_color": {
        "gray": "grey", "silver": "grey", "white": "grey", "salt_and_pepper": "grey",
        "black": "dark", "brown": "dark", "brunette": "dark", "dark_brown": "dark",
        "blond": "blonde", "light": "blonde",
        "auburn": "red", "ginger": "red",
        "none": "", "bald": "",   # a bald head has no hair color to compare
    },
    "facial_hair": {
        "clean_shaven": "none", "clean": "none", "shaven": "none",
        "light_stubble": "stubble", "heavy_stubble": "stubble", "scruff": "stubble",
        "moustache": "mustache",
        "full_beard": "beard", "short_beard": "beard", "grey_beard": "beard",
        "gray_beard": "beard", "thick_beard": "beard", "goatee": "beard",
    },
    "glasses": {
        "eyeglasses": "glasses", "spectacles": "glasses", "specs": "glasses",
        "yes": "glasses", "no": "none",
    },
}


def _norm_value(field: str, value: Optional[str]) -> Optional[str]:
    """Canonicalize one categorical value; returns None when the value carries
    no comparable evidence (unset, "unsure", or normalized to empty)."""
    if not value or value == "unsure":
        return None
    v = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    v = _VALUE_NORMALIZERS.get(field, {}).get(v, v)
    return v or None


def _normalize_fp(fp: Dict[str, str]) -> Dict[str, str]:
    """A fingerprint reduced to DISTANCE_FIELDS with canonicalized values --
    the representation clustering compares (age_band and empties dropped)."""
    out: Dict[str, str] = {}
    for f in DISTANCE_FIELDS:
        nv = _norm_value(f, fp.get(f))
        if nv is not None:
            out[f] = nv
    return out
# A cut with more than this many simultaneously visible people is a CROWD --
# no basis to tell individuals apart occurrence by occurrence, so none of
# them are clustered or assigned a person id (identity_map.plan.md's "never
# fabricate" stance, extended to crowds).
CROWD_SIZE = 6
# The cast table is deliberately UNCAPPED: every reconciled person is a full,
# named cast-table member (`is_major=True`), no matter how many the shoot has.
# There used to be a top-N cap here (the "majors vs anonymous others" split),
# but a fixed ceiling can't fit generic footage -- a five-person panel is as
# valid as a two-host podcast. Correctness now rests entirely on the clustering
# not over-splitting (its conservative, evidence-gated matching), not on hiding
# real people behind a constant.

# (file_id, source_ref, person_index within that cut's people[] list) --
# uniquely identifies one visible-person sighting.
OccurrenceKey = Tuple[str, str, int]


@dataclass
class Occurrence:
    """One visible-person sighting: a `PersonLook` entry inside one cut,
    plus just enough cut-level context (does this cut carry speech, how big
    was its subject framing) for the prominence ranking below. Built by
    `identity/apply.py` from the project's `Pass2Cut`s -- this module stays
    decoupled from the `pass2` schema on purpose (same pattern `bind.py`/
    the old `reconcile.py` already used)."""
    file_id: str
    source_ref: str
    person_index: int
    appearance: Dict[str, Any]
    description: str
    is_speech_cut: bool
    subject_box_area: Optional[float]   # None when the cut had no subject_box


def build_fingerprint(appearances: List[dict]) -> Dict[str, str]:
    """Majority vote per stable field across the given `appearance` dicts.
    `None`/`"unsure"` never vote. A tie for most-common value leaves that
    field unset (no clear majority) rather than picking arbitrarily. Used
    both for a single occurrence (a list of length 1 -- majority vote over
    one item just keeps whatever's set, cleaned of unsure/None) and, if a
    caller ever wants it, an aggregate over several."""
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
    """(disagreements, shared_set_fields) over DISTANCE_FIELDS: fields where
    BOTH sides have evidence count as "shared"; unset on either side
    contributes neither agreement nor disagreement (no evidence, not proof of
    a match). Inputs are assumed already value-normalized (`_normalize_fp`)."""
    shared = 0
    disagree = 0
    for f in DISTANCE_FIELDS:
        va, vb = a.get(f), b.get(f)
        if va is None or vb is None:
            continue
        shared += 1
        if va != vb:
            disagree += 1
    return disagree, shared


def _centroid(members: List[Dict[str, str]]) -> Dict[str, str]:
    """Majority-vote fingerprint over a cluster's normalized members. A tie
    for most-common value leaves that field unset (no clear majority), so a
    noisy outlier can't drag the centroid onto a wrong value."""
    fp: Dict[str, str] = {}
    for f in DISTANCE_FIELDS:
        counts: Dict[str, int] = {}
        for m in members:
            v = m.get(f)
            if v is None:
                continue
            counts[v] = counts.get(v, 0) + 1
        if not counts:
            continue
        best = max(counts.values())
        winners = [v for v, c in counts.items() if c == best]
        if len(winners) == 1:
            fp[f] = winners[0]
    return fp


def cluster_occurrences(
    fingerprints: Dict[OccurrenceKey, Dict[str, str]],
    min_shared: int = MIN_SHARED_FIELDS,
    max_disagree_frac: float = MAX_DISAGREE_FRAC,
) -> Dict[OccurrenceKey, str]:
    """`occurrence_key -> person_id` via greedy centroid (average-linkage)
    agglomeration over value-normalized fingerprints. Each step merges the two
    clusters whose centroids are most compatible -- enough shared evidence
    (`min_shared`) and at most `max_disagree_frac` of those shared fields
    disagreeing -- recomputing the centroid after each merge. Comparing
    against the running centroid (not any single member) is what stops
    chaining: once a cluster settles on "bald / grey / bearded", an occurrence
    that only half-matches can't quietly bridge it to a different person.

    Still conservative by design (over-split, never wrong-merge): a fingerprint
    too sparse to share `min_shared` fields with anyone stays its own person.
    `person_id`s are `P0, P1, ...` ordered by each cluster's minimum occurrence
    key, stable across re-runs of the same cuts."""
    keys = sorted(fingerprints.keys())
    norm: Dict[OccurrenceKey, Dict[str, str]] = {k: _normalize_fp(fingerprints[k]) for k in keys}

    # Each cluster is identified by its minimum member key (deterministic).
    members: Dict[OccurrenceKey, List[OccurrenceKey]] = {k: [k] for k in keys}
    centroid: Dict[OccurrenceKey, Dict[str, str]] = {k: dict(norm[k]) for k in keys}

    while True:
        best: Optional[Tuple[float, int, OccurrenceKey, OccurrenceKey]] = None
        cids = sorted(members.keys())
        for i in range(len(cids)):
            for j in range(i + 1, len(cids)):
                a, b = cids[i], cids[j]
                disagree, shared = _distance(centroid[a], centroid[b])
                if shared < min_shared:
                    continue
                frac = disagree / shared
                if frac <= max_disagree_frac:
                    cand = (frac, disagree, a, b)
                    if best is None or cand < best:
                        best = cand
        if best is None:
            break
        _, _, a, b = best
        keep, drop = (a, b) if a < b else (b, a)
        members[keep].extend(members[drop])
        del members[drop]
        centroid[keep] = _centroid([norm[k] for k in members[keep]])
        del centroid[drop]

    ordered_roots = sorted(members.keys(), key=lambda r: min(members[r]))
    occurrence_person: Dict[OccurrenceKey, str] = {}
    for i, root in enumerate(ordered_roots):
        pid = f"P{i}"
        for k in members[root]:
            occurrence_person[k] = pid
    return occurrence_person


@dataclass
class Person:
    person_id: str
    display: str
    fingerprint: Dict[str, str] = field(default_factory=dict)
    file_ids: List[str] = field(default_factory=list)
    appearance_count: int = 0
    is_major: bool = False
    # Filled later by identity/apply.py, once voice->person binding
    # (identity/voice_id.py, voice_id_pass.plan.md) has resolved which
    # voice(s), if any, this person was confirmed speaking. Empty here --
    # Phase D never touches voices at all.
    owned_voices: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "person_id": self.person_id, "display": self.display,
            "fingerprint": self.fingerprint, "file_ids": self.file_ids,
            "appearance_count": self.appearance_count, "is_major": self.is_major,
            "owned_voices": self.owned_voices,
        }


def _prominence_key(agg: Dict[str, Any]) -> Tuple[int, int, float]:
    """(appearance_count, speech_presence_count, avg_subject_box_area),
    descending -- the plan's own three prominence signals, as a sort key. A
    person who appears often, alongside speech, in a prominent framing is a
    "major"; a background/incidental appearance is not."""
    areas = agg["subject_box_areas"]
    avg_area = sum(areas) / len(areas) if areas else 0.0
    return (agg["appearance_count"], agg["speech_presence_count"], avg_area)


def build_persons(occurrence_person: Dict[OccurrenceKey, str], occurrences: List[Occurrence]) -> Dict[str, Person]:
    """One `Person` per cluster: display label = the longest raw
    `description` prose seen across the cluster's occurrences (display
    only, never an identity claim); fingerprint = the union of the member
    occurrences' own fields (no conflicts possible, clustering already
    required zero disagreement). Every cluster is marked `is_major=True`
    (the cast table is uncapped), in place on the returned dict."""
    by_key: Dict[OccurrenceKey, Occurrence] = {
        (occ.file_id, occ.source_ref, occ.person_index): occ for occ in occurrences
    }
    agg_by_person: Dict[str, Dict[str, Any]] = {}
    for key, pid in occurrence_person.items():
        occ = by_key[key]
        agg = agg_by_person.setdefault(pid, {
            "file_ids": set(), "descriptions": [], "fingerprints": [],
            "appearance_count": 0, "speech_presence_count": 0, "subject_box_areas": [],
        })
        agg["file_ids"].add(occ.file_id)
        if occ.description:
            agg["descriptions"].append(occ.description)
        agg["fingerprints"].append(build_fingerprint([occ.appearance]))
        agg["appearance_count"] += 1
        if occ.is_speech_cut:
            agg["speech_presence_count"] += 1
        if occ.subject_box_area is not None:
            agg["subject_box_areas"].append(occ.subject_box_area)

    persons: Dict[str, Person] = {}
    for pid, agg in agg_by_person.items():
        display = max(agg["descriptions"], key=len) if agg["descriptions"] else pid
        merged_fp: Dict[str, str] = {}
        for fp in agg["fingerprints"]:
            for k, v in fp.items():
                merged_fp.setdefault(k, v)
        persons[pid] = Person(person_id=pid, display=display, fingerprint=merged_fp,
                              file_ids=sorted(agg["file_ids"]), appearance_count=agg["appearance_count"])

    # Uncapped cast: every reconciled person is a full cast-table member. No
    # one is demoted to an anonymous "other" (`_prominence_key` remains the
    # definition of prominence, kept for ordering / diagnostics).
    for pid in agg_by_person.keys():
        persons[pid].is_major = True
    return persons


def visible_persons_by_cut(occurrence_person: Dict[OccurrenceKey, str]) -> Dict[Tuple[str, str], List[str]]:
    """(file_id, source_ref) -> sorted, distinct global person ids visible
    in that cut -- replaces the old one-person-per-FILE `oncam` map with a
    per-CUT set, since a cut can now show several people at once."""
    out: Dict[Tuple[str, str], set] = {}
    for (fid, ref, _idx), pid in occurrence_person.items():
        out.setdefault((fid, ref), set()).add(pid)
    return {k: sorted(v) for k, v in out.items()}


def reconcile(occurrences: List[Occurrence]) -> Dict[str, Any]:
    """The whole Phase D pipeline: crowd-exclude -> fingerprint -> cluster
    -> rank majors -> compose per-cut visible_persons. Returns
    ``{"persons": [Person.to_dict(), ...], "visible_persons":
    {(file_id, source_ref): [person_id, ...]}}``. An occurrence in a crowd
    cut (more than `CROWD_SIZE` simultaneous people) is dropped before
    fingerprinting -- it never becomes a person and never appears in
    `visible_persons` for that cut."""
    by_cut: Dict[Tuple[str, str], List[Occurrence]] = {}
    for occ in occurrences:
        by_cut.setdefault((occ.file_id, occ.source_ref), []).append(occ)
    kept = [occ for occs in by_cut.values() if len(occs) <= CROWD_SIZE for occ in occs]

    fingerprints: Dict[OccurrenceKey, Dict[str, str]] = {
        (occ.file_id, occ.source_ref, occ.person_index): build_fingerprint([occ.appearance])
        for occ in kept
    }
    occurrence_person = cluster_occurrences(fingerprints)
    persons = build_persons(occurrence_person, kept)
    visible = visible_persons_by_cut(occurrence_person)
    return {
        "persons": [p.to_dict() for p in persons.values()],
        "visible_persons": visible,
    }
