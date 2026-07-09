"""
Cuts v3, Pass 2a: IDENTITY + take resolution, vision, cached prefix +
numbered images, sharded only by image/cut budget (never by clip -- see
``build_identity_shards``).

Split out of the original monolithic pass 2 (see ``pass2b.py`` for the
sibling half) after real ingest runs showed the model getting unreliable
past ~15-40 cuts of output in one call -- not a truncation, a complexity
cliff in generating one very large nested JSON structure. Pass 2a keeps
only the fields that genuinely need cross-cut comparison (take vs outlook
requires seeing multiple clips' pixels together) or gate a cut's very
identity (kind, atom_ids, word_span, junk); the purely-per-cut visual
judgment (framing/look/captions/taste) moves to pass2b.py, which has no
take-style co-location constraint and can be sharded far more finely.

Design note (carried over from the original pass2.py, still true here):
each output cut carries a ``source_ref`` pointing back at the pass-1 unit
it originated from (``speech_cut[i]`` / ``video_group[i]``), using the same
ref strings ``image_plan.py`` used for the numbered images. A video cut's
``atom_ids`` may be a sub-range of its source group's atoms (splitting a
tentative group back at atom edges is allowed); a speech cut's ``word_span``
is pass-through (pass 1's speech grouping is final) but re-emitted here so
pass 2a can flag it junk without a second round trip.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.services.l3.image_plan import PlannedFrame
from app.services.l3.lattice import Lattice, resolve_speech_span_ms
from app.services.l3.pass1 import Pass1Output, build_pass1_blocks, render_pass1_output
from app.services.l3.pass2_params import MAX_CUTS_PER_SHARD, MAX_IMAGES_PER_SHARD
from app.services.llm import client as ic
from app.services.llm.base import image_block, text_block

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Output schema -- identity + take resolution only (see pass2b.py for the
# visual-judgment half; pass2.py holds the MERGED final Pass2Cut/Pass2Output
# both halves assemble into)
# --------------------------------------------------------------------------

# Reasonable model wordings for "a non-winning take" beyond the literal
# "take" -- normalized rather than rejected, so a harmless naming choice
# doesn't burn a re-ask (or worse, survive as an unrecognized value
# downstream). Genuinely invalid values still fail validation below.
_TAKE_ROLE_ALIASES = {
    "alt": "take", "alternate": "take", "alternative": "take",
    "sibling": "take", "loser": "take", "other": "take",
}

# The model intermittently echoes pass 1's OWN unit name into the kind enum --
# "video_tentative_group" (the pass-1 group name) instead of the canonical
# "video", or "speech_cut" instead of "speech". It's unambiguous (the ref
# prefix already pins the kind), so normalize it at parse time rather than
# burn a re-ask on a pure naming tic. Observed twice-in-a-row on one real
# Reel-trail shard, which the one-re-ask loop then couldn't clear.
_KIND_ALIASES = {
    "video_tentative_group": "video", "video_group": "video", "vid": "video",
    "speech_cut": "speech", "speech_group": "speech", "spoken": "speech",
}

# channel = the delivery CATEGORY (the model's call): "said" (spoken), "done"
# (an action is performed/demonstrated on screen), "shown" (b-roll / an object /
# scenery / a display, no performed action). Fold common synonyms rather than
# burn a re-ask on a naming choice; unknown values default to "shown" in post.
_CHANNEL_ALIASES = {
    "speech": "said", "spoken": "said", "dialogue": "said", "talking": "said",
    "action": "done", "demo": "done", "demonstration": "done", "performed": "done",
    "b-roll": "shown", "broll": "shown", "b_roll": "shown", "insert": "shown",
    "display": "shown", "scenery": "shown", "object": "shown",
}


class IdentityCut(BaseModel):
    source_ref: str                 # e.g. "speech_cut[2]" / "video_group[0]" -- joins back to pass 1
    kind: str                        # "speech" | "video"
    file_id: str
    word_span: Tuple[int, int] | None = None    # speech cuts only
    atom_ids: List[int] | None = None           # video cuts only
    label: str
    summary: str
    speaker: str | None = None
    on_camera: bool | None = None
    junk: bool = False
    junk_reason: str = ""
    natural_sound: bool = False
    take_group_id: str | None = None
    take_role: str | None = None    # "take" | "outlook" | "winner"
    channel: str | None = None      # "said" | "done" | "shown"

    @field_validator("take_role", mode="before")
    @classmethod
    def _normalize_take_role(cls, v: Any) -> Any:
        if isinstance(v, str):
            key = v.strip().lower()
            return _TAKE_ROLE_ALIASES.get(key, key)
        return v

    @field_validator("channel", mode="before")
    @classmethod
    def _normalize_channel(cls, v: Any) -> Any:
        if isinstance(v, str):
            key = v.strip().lower()
            return _CHANNEL_ALIASES.get(key, key)
        return v

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, v: Any) -> Any:
        if isinstance(v, str):
            key = v.strip().lower()
            return _KIND_ALIASES.get(key, key)
        return v

    @model_validator(mode="after")
    def _kind_matches_locator(self) -> "IdentityCut":
        # word_span/atom_ids may legitimately be omitted -- they're
        # pass-through from pass 1 and backfilled deterministically by
        # backfill_locators (the model echoing them verbatim was the single
        # biggest source of output-complexity failures on big shards).
        if self.take_role is not None and self.take_role not in ("take", "outlook", "winner"):
            raise ValueError(f"invalid take_role {self.take_role!r}")
        return self


class IdentityOutput(BaseModel):
    # Every field here has a safe default, so a model response that wraps
    # its real answer under an unexpected top-level key would otherwise
    # validate cleanly as an empty result -- see pass1.Pass1Output's
    # identical config for the full rationale (observed in the wild).
    model_config = ConfigDict(extra="forbid")

    cuts: List[IdentityCut] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

_SYSTEM = (
    "You already did pass 1 for this project: you saw every clip's "
    "transcript and video-atom table, and grouped them into speech cuts, "
    "take candidates, and tentative video groups (repeated below verbatim). "
    "Now you see the actual pixels: numbered stills, each captioned with "
    "which clip/timestamp/pass-1 unit it belongs to.\n\n"
    "This is an IDENTITY pass -- do not judge framing, look/grade, caption "
    "placement, or pace here; another pass handles those. SCOPE: this call "
    "covers ONLY the clips whose images you are shown below -- the pass-1 "
    "result may mention other clips, but those are handled in separate "
    "calls; never emit a cut for a clip you were not shown images for. "
    "Your only job:\n\n"
    "For every in-scope speech_cut and every in-scope video_tentative_group, emit ONE final "
    "cut record (a tentative video group MAY be split back into multiple "
    "cuts along its existing atom_ids if the pixels show it isn't one "
    "moment -- never invent a boundary inside an atom).\n\n"
    "Do NOT echo word_span (it is derived from your source_ref by code). "
    "Do NOT echo atom_ids either, EXCEPT when you split a video group into "
    "several cuts -- then each piece must list the atom_ids it owns, the "
    "pieces together must use every atom_id of the group exactly once, "
    "never zero times and never twice. For every "
    "take_candidate, resolve it: is it a TAKE (same words, same setting) or "
    "an OUTLOOK (same words, different setting)? Pick a winner; keep the "
    "rest stacked under it. A take boundary is always a hard split. Set "
    "take_role to EXACTLY one of \"winner\", \"take\" (a non-winning take of "
    "the same setting), or \"outlook\" (different setting) -- no other "
    "value.\n\n"
    "Per cut, judge from the pixels: label, summary (a best guess from image "
    "+ transcript is fine, and expected), on_camera (does the visible person "
    "match the diarized speaker), channel, junk (+reason), and natural_sound "
    "(does the cut carry sound worth keeping). CHANNEL is the delivery category: "
    "for a video cut set \"done\" when an action is performed / demonstrated on "
    "screen (a swing, a catch, pouring, assembling, a gesture that IS the "
    "content) or \"shown\" when it is b-roll / an object / scenery / a display "
    "with no performed action (speech cuts are always \"said\" -- you may omit "
    "channel for them). JUNK is BINARY and by MEANING: set "
    "junk=true ONLY when the cut is clearly not part of the piece (a camera "
    "cue like 'and go'/'3-2-1'/'take three', pre-roll setup, obvious dead "
    "air). Junk is recoverable -- it's hidden into a Discarded tray, not "
    "deleted -- but keep the bar HIGH: if there is ANY doubt a cut might be "
    "wanted, leave junk=false. Never mark an ACTION/motion payoff junk. A "
    "label must name what the cut SHOWS (e.g. 'forehand swing', 'catches the "
    "ball'), never a mechanical 'settle'/'trailing frames'.\n\n"
    "Reference every cut by source_ref using the SAME ref string pass 1 (and "
    "the image captions) used for it -- speech_cut[i] or video_group[i], "
    "VERBATIM. Never invent a new ref: every take member is already its own "
    "speech_cut, so takes are resolved by setting take_group_id/take_role on "
    "those existing speech_cut refs, never by emitting extra cuts."
)


def build_identity_shard_blocks(
    planned_frames: List[PlannedFrame], images_b64: Dict[Tuple[str, int], str],
) -> List[Dict[str, Any]]:
    """Numbered [caption, image] block pairs for one shard, in stable
    (file_id, ts_ms) order. Frames with no extracted image (not yet pulled,
    or extraction failed upstream) are skipped rather than sent blank."""
    ordered = sorted(planned_frames, key=lambda f: (f.file_id, f.ts_ms))
    blocks: List[Dict[str, Any]] = []
    for n, f in enumerate(ordered, start=1):
        b64 = images_b64.get((f.file_id, f.ts_ms))
        if b64 is None:
            continue
        blocks.append(text_block(f"IMG {n} = clip {f.file_id}, {f.ts_ms / 1000:.1f}s, {f.ref}"))
        blocks.append(image_block(b64))
    return blocks


# --------------------------------------------------------------------------
# Sharding: whole clips per shard, take-group members co-located (hard
# constraint -- take-vs-outlook needs direct visual comparison), bin-packed
# to MAX_IMAGES_PER_SHARD / MAX_CUTS_PER_SHARD (soft -- an oversized cluster
# gets its own shard rather than being split; co-location is never
# negotiable, see cuts_v3.plan.md sec. 5).
# --------------------------------------------------------------------------

def _cluster_files(pass1: Pass1Output, file_ids: List[str]) -> List[List[str]]:
    parent = {fid: fid for fid in file_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for tc in pass1.take_candidates:
        members = [m.file_id for m in tc.members if m.file_id in parent]
        for m in members[1:]:
            union(members[0], m)

    clusters: Dict[str, List[str]] = {}
    for fid in file_ids:
        clusters.setdefault(find(fid), []).append(fid)
    return list(clusters.values())


def _cut_counts_by_file(pass1: Pass1Output) -> Dict[str, int]:
    """How many final IdentityCut records a file will need -- one per
    speech_cut plus one per video_tentative_group. Independent of image
    count, and the thing that actually drives a shard's OUTPUT size."""
    counts: Dict[str, int] = {}
    for sc in pass1.speech_cuts:
        counts[sc.file_id] = counts.get(sc.file_id, 0) + 1
    for vg in pass1.video_tentative_groups:
        counts[vg.file_id] = counts.get(vg.file_id, 0) + 1
    return counts


def build_identity_shards(pass1: Pass1Output, planned_frames: List[PlannedFrame]) -> List[List[str]]:
    image_counts: Dict[str, int] = {}
    for f in planned_frames:
        image_counts[f.file_id] = image_counts.get(f.file_id, 0) + 1
    cut_counts = _cut_counts_by_file(pass1)
    file_ids = list(image_counts.keys())
    if not file_ids:
        return []

    clusters = _cluster_files(pass1, file_ids)

    def images_of(cluster: List[str]) -> int:
        return sum(image_counts.get(f, 0) for f in cluster)

    def cuts_of(cluster: List[str]) -> int:
        return sum(cut_counts.get(f, 0) for f in cluster)

    sized = sorted(clusters, key=images_of, reverse=True)

    shards: List[List[str]] = []
    shard_images: List[int] = []
    shard_cuts: List[int] = []
    for cluster in sized:
        images, cuts = images_of(cluster), cuts_of(cluster)
        placed = False
        for i, shard in enumerate(shards):
            if shard_images[i] + images <= MAX_IMAGES_PER_SHARD and shard_cuts[i] + cuts <= MAX_CUTS_PER_SHARD:
                shard.extend(cluster)
                shard_images[i] += images
                shard_cuts[i] += cuts
                placed = True
                break
        if not placed:
            if images > MAX_IMAGES_PER_SHARD or cuts > MAX_CUTS_PER_SHARD:
                logger.warning("pass2a: take-group cluster (%d images, %d cuts) exceeds shard budget "
                               "(%d images, %d cuts) -- co-location is non-negotiable, sending oversized",
                               images, cuts, MAX_IMAGES_PER_SHARD, MAX_CUTS_PER_SHARD)
            shards.append(list(cluster))
            shard_images.append(images)
            shard_cuts.append(cuts)
    return shards


# --------------------------------------------------------------------------
# Semantic checks pydantic can't express -- all observed against the real
# API, all folded into the re-ask loop instead of only surfacing as opaque
# failures downstream in post.assemble_cut_records.
# --------------------------------------------------------------------------

def _ref_index(ref: str, prefix: str) -> Optional[int]:
    if ref.startswith(prefix) and ref.endswith("]"):
        idx = ref[len(prefix):-1]
        if idx.isdigit():
            return int(idx)
    return None


def backfill_locators(output: IdentityOutput, pass1: Pass1Output) -> IdentityOutput:
    """Deterministically fill/normalize every cut's word_span/atom_ids from
    pass 1 by source_ref. The model was originally required to echo these
    verbatim; observed against the real API, that echo was the single
    biggest output-complexity failure (66 cuts -> 41 validation errors,
    twice). They carry zero judgment -- pass 1's grouping is final -- so
    code owns them now:

      * speech cut  -> word_span := pass1.speech_cuts[i].word_span, always.
      * video cut, ref emitted by exactly ONE cut -> atom_ids := the whole
        group's atom_ids (no split, nothing to decide).
      * video cut, ref emitted by SEVERAL cuts (a split) -> each piece keeps
        its own atom_ids (that IS judgment); _split_groups_partition_atoms
        validates the pieces partition the group exactly."""
    video_ref_counts: Dict[str, int] = {}
    for cut in output.cuts:
        if cut.kind == "video":
            video_ref_counts[cut.source_ref] = video_ref_counts.get(cut.source_ref, 0) + 1

    new_cuts: List[IdentityCut] = []
    for cut in output.cuts:
        update: Dict[str, Any] = {}
        if cut.kind == "speech":
            i = _ref_index(cut.source_ref, "speech_cut[")
            if i is not None and i < len(pass1.speech_cuts):
                update["word_span"] = tuple(pass1.speech_cuts[i].word_span)
        elif cut.kind == "video" and video_ref_counts.get(cut.source_ref) == 1:
            gi = _ref_index(cut.source_ref, "video_group[")
            if gi is not None and gi < len(pass1.video_tentative_groups):
                update["atom_ids"] = list(pass1.video_tentative_groups[gi].atom_ids)
        new_cuts.append(cut.model_copy(update=update) if update else cut)
    return IdentityOutput(cuts=new_cuts)


def _locators_resolved(output: IdentityOutput) -> Optional[str]:
    """After backfill, every cut must have its locator: a speech cut missing
    word_span means its ref didn't resolve; a video cut missing atom_ids
    means it's one piece of a SPLIT group that didn't say which atoms it
    owns (the one locator that IS the model's judgment)."""
    for cut in output.cuts:
        if cut.kind == "speech" and cut.word_span is None:
            return (f"{cut.source_ref!r} resolved to no word_span -- its ref must name an "
                    f"existing pass-1 speech_cut")
        if cut.kind == "video" and not cut.atom_ids:
            return (f"{cut.source_ref!r} has no atom_ids -- when you split a video group "
                    f"into several cuts, every piece must list the atom_ids it owns")
    return None


def _split_groups_partition_atoms(output: IdentityOutput, pass1: Pass1Output) -> Optional[str]:
    """When a video group is split into several cuts, the pieces' atom_ids
    must partition the group's atoms exactly -- no atom lost, none invented.
    (The no-duplicates half is _no_duplicate_atoms; this checks the union.)"""
    by_ref: Dict[str, List[IdentityCut]] = {}
    for cut in output.cuts:
        if cut.kind == "video":
            by_ref.setdefault(cut.source_ref, []).append(cut)
    for ref, cuts in by_ref.items():
        if len(cuts) < 2:
            continue
        gi = _ref_index(ref, "video_group[")
        if gi is None or gi >= len(pass1.video_tentative_groups):
            continue
        expected = set(pass1.video_tentative_groups[gi].atom_ids)
        got = {a for c in cuts for a in (c.atom_ids or [])}
        if got != expected:
            missing = sorted(expected - got)
            extra = sorted(got - expected)
            return (f"the cuts splitting {ref!r} don't partition its atoms exactly -- "
                    f"missing atom_ids {missing}, unexpected {extra}; every atom of the "
                    f"group must land in exactly one piece")
    return None


def _source_refs_exist(output: IdentityOutput, pass1: Pass1Output) -> Optional[str]:
    """Observed against the real API: the model INVENTED refs (e.g.
    "take[intro_greeting]_take1") for cuts it wanted to emit around a take,
    instead of using the pass-1 ref strings. Nothing downstream can join
    such a ref -- image_plan planned no frames for it, so pass 2b's batch
    comes up imageless and the whole run dies with an unrelated-looking
    "no images resolved" error. Every source_ref must be a literal
    speech_cut[i] / video_group[i] that pass 1 actually emitted."""
    n_speech, n_video = len(pass1.speech_cuts), len(pass1.video_tentative_groups)
    for cut in output.cuts:
        ref = cut.source_ref
        for prefix, n in (("speech_cut[", n_speech), ("video_group[", n_video)):
            if ref.startswith(prefix) and ref.endswith("]"):
                idx_str = ref[len(prefix):-1]
                if idx_str.isdigit() and int(idx_str) < n:
                    break
        else:
            return (f"source_ref {ref!r} is not a ref pass 1 emitted -- every cut must "
                    f"reference EXACTLY one of speech_cut[0..{max(n_speech - 1, 0)}] or "
                    f"video_group[0..{max(n_video - 1, 0)}], verbatim; never invent a new "
                    f"ref (take members are already their own speech_cuts)")
    return None


def _no_duplicate_atoms(output: IdentityOutput) -> Optional[str]:
    """Every atom_id must end up in exactly one output cut. Observed against
    the real API: the model occasionally double-counts an atom when
    splitting a tentative video group into multiple final cuts."""
    seen: Dict[Tuple[str, int], str] = {}
    for cut in output.cuts:
        if cut.kind != "video" or not cut.atom_ids:
            continue
        for atom_id in cut.atom_ids:
            key = (cut.file_id, atom_id)
            if key in seen:
                return (f"atom_id {atom_id} of file {cut.file_id} appears in both "
                       f"{seen[key]!r} and {cut.source_ref!r} -- split a group's atom_ids "
                       f"between the pieces, never duplicate one across them")
            seen[key] = cut.source_ref
    return None


def _kind_matches_source_ref(output: IdentityOutput) -> Optional[str]:
    """Observed against the real API: a cut keeps its pass-1 ref name
    (e.g. "speech_cut[10]") but gets emitted with the WRONG kind (e.g.
    "video") -- word_span/atom_ids then don't resolve, surfacing downstream
    in post.assemble_cut_records as a much less actionable error. Catching
    the mismatch here, against the ref's own naming, folds it into the
    re-ask loop instead."""
    for cut in output.cuts:
        if cut.source_ref.startswith("speech_cut[") and cut.kind != "speech":
            return (f"{cut.source_ref!r} is a speech_cut but was emitted with "
                    f"kind={cut.kind!r} -- speech_cut refs must always have kind=\"speech\"")
        if cut.source_ref.startswith("video_group[") and cut.kind != "video":
            return (f"{cut.source_ref!r} is a video_group but was emitted with "
                    f"kind={cut.kind!r} -- video_group refs must always have kind=\"video\"")
    return None


def _no_overlapping_word_spans(output: IdentityOutput) -> Optional[str]:
    """Observed against the real API: two speech cuts in the same file with
    identical/overlapping word_span ranges (a duplicate or a pass-1 grouping
    mistake pass 2a echoed through) -- this only ever surfaces downstream as
    a raw ms-coverage overlap in post.assemble_cut_records, which doesn't
    say WHICH two cuts or why. Checking word indices directly here is both
    cheaper (no lattice needed) and a clearer message fed back on re-ask."""
    by_file: Dict[str, List[Tuple[int, int, str]]] = {}
    for cut in output.cuts:
        if cut.kind != "speech" or not cut.word_span:
            continue
        by_file.setdefault(cut.file_id, []).append((cut.word_span[0], cut.word_span[1], cut.source_ref))
    for file_id, spans in by_file.items():
        spans.sort()
        for (a0, b0, r0), (a1, b1, r1) in zip(spans, spans[1:]):
            if a1 <= b0:
                return (f"{r0!r} words[{a0}-{b0}] and {r1!r} words[{a1}-{b1}] overlap in "
                       f"{file_id} -- speech cuts must partition non-overlapping word ranges")
    return None


def _resolve_cut_span_ms(cut: IdentityCut, lattices: Dict[str, Lattice]) -> Optional[Tuple[int, int]]:
    """Best-effort (s, e) in ms for one cut, same resolution post.py uses --
    word/atom edges, clamped so a speech cut's silence cushion can never
    reach into a neighboring atom's span (see resolve_speech_span_ms).
    Silence data is skipped here (empty list) since the clamp alone is what
    prevents the overlap this check exists to catch; a few ms of precision
    beyond that doesn't matter for a GROSS-overlap check like this one."""
    lattice = lattices.get(cut.file_id)
    if lattice is None:
        return None
    if cut.kind == "speech" and cut.word_span:
        return resolve_speech_span_ms(lattice.words, lattice.atoms, cut.word_span, [])
    if cut.kind == "video" and cut.atom_ids:
        atoms_by_id = {a.atom_id: a for a in lattice.atoms}
        members = [atoms_by_id[i] for i in cut.atom_ids if i in atoms_by_id]
        if not members:
            return None
        return min(a.start_ms for a in members), max(a.end_ms for a in members)
    return None


def _no_cross_kind_ms_overlap(output: IdentityOutput, lattices: Dict[str, Lattice]) -> Optional[str]:
    """A speech cut and a video cut in the same file resolving to
    overlapping ms spans -- neither _no_duplicate_atoms (video-only, by
    atom id) nor _no_overlapping_word_spans (speech-only, by word index)
    catches this, since it's the CROSS-kind case. Observed against the real
    API surfacing only as an opaque ms-overlap failure in
    post.assemble_cut_records; resolving spans here catches it earlier,
    with a message that names both cuts."""
    by_file: Dict[str, List[Tuple[int, int, str]]] = {}
    for cut in output.cuts:
        span = _resolve_cut_span_ms(cut, lattices)
        if span is None:
            continue
        by_file.setdefault(cut.file_id, []).append((span[0], span[1], cut.source_ref))
    for file_id, spans in by_file.items():
        spans.sort()
        for (s0, e0, r0), (s1, e1, r1) in zip(spans, spans[1:]):
            if s1 < e0:
                return (f"{r0!r} [{s0}-{e0}]ms and {r1!r} [{s1}-{e1}]ms overlap in "
                       f"{file_id} -- every cut's resolved span must be disjoint")
    return None


def _resolved_file_id(cut: IdentityCut, pass1: Pass1Output) -> Optional[str]:
    """The clip a cut TRULY belongs to, resolved from its source_ref against
    pass 1 -- authoritative, since the model's own file_id field can be wrong.
    None when the ref doesn't resolve to a pass-1 unit."""
    if cut.kind == "speech":
        i = _ref_index(cut.source_ref, "speech_cut[")
        if i is not None and i < len(pass1.speech_cuts):
            return pass1.speech_cuts[i].file_id
    elif cut.kind == "video":
        gi = _ref_index(cut.source_ref, "video_group[")
        if gi is not None and gi < len(pass1.video_tentative_groups):
            return pass1.video_tentative_groups[gi].file_id
    return None


def _drop_out_of_shard_cuts(output: IdentityOutput, pass1: Pass1Output,
                            shard_files: set) -> Tuple[IdentityOutput, int]:
    """Drop cuts that belong to a clip OUTSIDE this shard, returning
    (filtered_output, n_dropped). The cached pass-1 render lists EVERY clip's
    units, so a shard -- especially an oversized take-group cluster the model
    can't hold in one call -- sometimes emits cuts for clips it wasn't shown.
    Those clips are handled by their OWN shard, so the strays here are pure
    duplicates: discarding them (rather than failing the whole call) is what
    lets big multicam projects ingest, and still keeps post from ever seeing
    cross-shard duplicate spans. Clip ownership is resolved from source_ref
    (authoritative), falling back to the model's file_id if a ref doesn't
    resolve. A kept cut whose model file_id disagrees with its source_ref is
    corrected to the resolved clip so post attributes it correctly."""
    kept: List[IdentityCut] = []
    for c in output.cuts:
        owner = _resolved_file_id(c, pass1) or c.file_id
        if owner not in shard_files:
            continue
        kept.append(c if c.file_id == owner else c.model_copy(update={"file_id": owner}))
    dropped = len(output.cuts) - len(kept)
    return output.model_copy(update={"cuts": kept}), dropped


def _pass2a_semantic_checks(output: IdentityOutput, pass1: Pass1Output,
                            lattices: Dict[str, Lattice], shard_files: set) -> Optional[str]:
    """Run against the BACKFILLED output (see backfill_locators) -- locator
    checks are meaningless before the deterministic fill. Out-of-shard strays
    are filtered out FIRST (see _drop_out_of_shard_cuts) so they can't trip a
    spurious re-ask; run_identity_shard applies the same filter to the
    persisted output."""
    output = backfill_locators(output, pass1)
    output, _ = _drop_out_of_shard_cuts(output, pass1, shard_files)
    return (_source_refs_exist(output, pass1)
           or _kind_matches_source_ref(output)
           or _locators_resolved(output)
           or _split_groups_partition_atoms(output, pass1)
           or _no_duplicate_atoms(output)
           or _no_overlapping_word_spans(output)
           or _no_cross_kind_ms_overlap(output, lattices))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_identity_shard(
    file_rows: List[Tuple[str, str, int, Lattice]],
    pass1_output: Pass1Output,
    shard_frames: List[PlannedFrame],
    images_b64: Dict[Tuple[str, int], str],
) -> ic.Completion:
    """One pass-2a shard call. ``file_rows`` + ``pass1_output`` render the
    SAME cached prefix on every shard in a run (identical blocks -> cache
    hits after the first); ``shard_frames``/``images_b64`` are this shard's
    images only, appended uncached. Raises ``ValueError`` if the shard has
    no resolvable images."""
    cached_blocks = build_pass1_blocks(file_rows) + [text_block(render_pass1_output(pass1_output))]
    image_blocks = build_identity_shard_blocks(shard_frames, images_b64)
    if not image_blocks:
        raise ValueError("run_identity_shard: no images resolved for this shard")
    lattices = {fid: lattice for fid, _name, _dur, lattice in file_rows}
    shard_files = {f.file_id for f in shard_frames}
    completion = ic.complete("pass2", _SYSTEM, cached_blocks, IdentityOutput,
                             extra_blocks=image_blocks, max_tokens=20000,
                             extra_check=lambda output: _pass2a_semantic_checks(
                                 output, pass1_output, lattices, shard_files))
    # The persisted output must carry ONLY this shard's clips. The semantic
    # check already filters strays before validating, but ic.complete returns
    # the model's RAW payload, so drop them here too -- otherwise post would
    # see the same cut from two shards as an identical-span overlap.
    filtered, dropped = _drop_out_of_shard_cuts(
        IdentityOutput.model_validate(completion.data), pass1_output, shard_files)
    if dropped:
        logger.warning("pass2a: dropped %d out-of-shard cut(s) from a %d-clip shard "
                       "(each handled by its own shard)", dropped, len(shard_files))
        completion.data = filtered.model_dump()
    return completion
