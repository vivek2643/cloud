"""
Cuts v3, Pass 2: vision, cached prefix + numbered images, sharded only by
image-token budget (never by clip -- see ``build_shards``).

Design note (not fully spelled out in cuts_v3.plan.md section 5, decided
here): each output cut carries a ``source_ref`` pointing back at the pass-1
unit it originated from (``speech_cut[i]`` / ``video_group[i]``), using the
exact same ref strings ``image_plan.py`` used for the numbered images. A
video cut's ``atom_ids`` may be a sub-range of its source group's atoms (the
plan allows splitting a tentative group back at atom edges); a speech cut's
``word_span`` is currently pass-through (pass 1's speech grouping is final
per the plan's North Star #2) but is re-emitted here so pass 2 can flag a
speech cut as junk without a second round trip.
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
# Output schema -- the complete per-cut record (see cuts_v3.plan.md sec. 5)
# --------------------------------------------------------------------------

class Framing(BaseModel):
    subject_box: Tuple[float, float, float, float] | None = None   # normalized x,y,w,h
    crop_16x9: Tuple[float, float, float, float] | None = None
    crop_9x16: Tuple[float, float, float, float] | None = None
    crop_1x1: Tuple[float, float, float, float] | None = None
    rotation_deg: float = 0.0


class Look(BaseModel):
    graded: bool = False
    palette: List[str] = Field(default_factory=list)
    exposure_flags: List[str] = Field(default_factory=list)


class TasteFences(BaseModel):
    max_tasteful_speed: float = 1.0
    min_tasteful_speed: float = 1.0


# Reasonable model wordings for "a non-winning take" beyond the literal
# "take" -- normalized rather than rejected, so a harmless naming choice
# doesn't burn a re-ask (or worse, survive as an unrecognized value
# downstream). Genuinely invalid values still fail validation below.
_TAKE_ROLE_ALIASES = {
    "alt": "take", "alternate": "take", "alternative": "take",
    "sibling": "take", "loser": "take", "other": "take",
}


class Pass2Cut(BaseModel):
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
    framing: Framing = Field(default_factory=Framing)
    look: Look = Field(default_factory=Look)
    caption_zones: List[Tuple[float, float, float, float]] = Field(default_factory=list)
    taste_fences: TasteFences = Field(default_factory=TasteFences)
    readability_ms: int = 0
    natural_sound: bool = False
    take_group_id: str | None = None
    take_role: str | None = None    # "take" | "outlook" | "winner"

    @field_validator("take_role", mode="before")
    @classmethod
    def _normalize_take_role(cls, v: Any) -> Any:
        if isinstance(v, str):
            key = v.strip().lower()
            return _TAKE_ROLE_ALIASES.get(key, key)
        return v

    @model_validator(mode="after")
    def _kind_matches_locator(self) -> "Pass2Cut":
        if self.kind == "speech" and self.word_span is None:
            raise ValueError("speech cut missing word_span")
        if self.kind == "video" and not self.atom_ids:
            raise ValueError("video cut missing atom_ids")
        if self.take_role is not None and self.take_role not in ("take", "outlook", "winner"):
            raise ValueError(f"invalid take_role {self.take_role!r}")
        return self


class Pass2Output(BaseModel):
    # See Pass1Output's identical config for why: a single defaulted field
    # ("cuts") means a response wrapped under an unexpected top-level key
    # would otherwise "validate" as an empty result instead of failing loud.
    model_config = ConfigDict(extra="forbid")

    cuts: List[Pass2Cut] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

_SYSTEM = (
    "You already did pass 1 for this project: you saw every clip's "
    "transcript and video-atom table, and grouped them into speech cuts, "
    "take candidates, and tentative video groups (repeated below verbatim). "
    "Now you see the actual pixels: numbered stills, each captioned with "
    "which clip/timestamp/pass-1 unit it belongs to.\n\n"
    "For every speech_cut and every video_tentative_group, emit ONE final "
    "cut record (a tentative video group MAY be split back into multiple "
    "cuts along its existing atom_ids if the pixels show it isn't one "
    "moment -- never invent a boundary inside an atom). Every atom_id you "
    "were given must end up in EXACTLY ONE output cut, never zero and never "
    "two -- if you split a group, partition its atom_ids, don't duplicate "
    "any of them across the pieces. For every "
    "take_candidate, resolve it: is it a TAKE (same words, same setting) or "
    "an OUTLOOK (same words, different setting)? Pick a winner; keep the "
    "rest stacked under it. A take boundary is always a hard split. Set "
    "take_role to EXACTLY one of \"winner\", \"take\" (a non-winning take of "
    "the same setting), or \"outlook\" (different setting) -- no other "
    "value.\n\n"
    "Per cut, judge from the pixels: label, summary (a best guess from image "
    "+ transcript is fine, and expected), on_camera (does the visible person "
    "match the diarized speaker), junk (+reason), framing (subject_box + "
    "three crops + rotation_deg for orientation/horizon), look (graded vs "
    "log/flat, palette, exposure flags), caption_zones (boxes clear of the "
    "subject on BOTH the hero frame and any drift frame you were shown), "
    "taste fences (max/min tasteful playback speed) and readability_ms (how "
    "long a viewer needs to read this frame), and natural_sound (does the "
    "cut carry sound worth keeping).\n\n"
    "Reference every cut by source_ref using the SAME ref string pass 1 (and "
    "the image captions) used for it -- speech_cut[i] or video_group[i]."
)


def build_pass2_shard_blocks(
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
# constraint), bin-packed to MAX_IMAGES_PER_SHARD (soft -- an oversized
# cluster gets its own shard rather than being split; co-location is never
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
    """How many final Pass2Cut records a file will need -- one per
    speech_cut plus one per video_tentative_group. Independent of image
    count, and the thing that actually drives a shard's OUTPUT size."""
    counts: Dict[str, int] = {}
    for sc in pass1.speech_cuts:
        counts[sc.file_id] = counts.get(sc.file_id, 0) + 1
    for vg in pass1.video_tentative_groups:
        counts[vg.file_id] = counts.get(vg.file_id, 0) + 1
    return counts


def build_shards(pass1: Pass1Output, planned_frames: List[PlannedFrame]) -> List[List[str]]:
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
                logger.warning("pass2: take-group cluster (%d images, %d cuts) exceeds shard budget "
                               "(%d images, %d cuts) -- co-location is non-negotiable, sending oversized",
                               images, cuts, MAX_IMAGES_PER_SHARD, MAX_CUTS_PER_SHARD)
            shards.append(list(cluster))
            shard_images.append(images)
            shard_cuts.append(cuts)
    return shards


# --------------------------------------------------------------------------
# Semantic check pydantic can't express: every atom_id must end up in
# exactly one output cut. Observed against the real API: the model
# occasionally double-counts an atom when splitting a tentative video group
# into multiple final cuts, which manifests downstream as a coverage overlap
# in post.assemble_cut_records -- catching it here folds it into the same
# one-re-ask-then-fail-loud path instead of failing the whole ingest run.
# --------------------------------------------------------------------------

def _no_duplicate_atoms(output: Pass2Output) -> Optional[str]:
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


def _kind_matches_source_ref(output: Pass2Output) -> Optional[str]:
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


def _no_overlapping_word_spans(output: Pass2Output) -> Optional[str]:
    """Observed against the real API: two speech cuts in the same file with
    identical/overlapping word_span ranges (a duplicate or a pass-1 grouping
    mistake pass 2 echoed through) -- this only ever surfaces downstream as
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


def _resolve_cut_span_ms(cut: "Pass2Cut", lattices: Dict[str, Lattice]) -> Optional[Tuple[int, int]]:
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


def _no_cross_kind_ms_overlap(output: Pass2Output, lattices: Dict[str, Lattice]) -> Optional[str]:
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


def _pass2_semantic_checks(output: Pass2Output, lattices: Dict[str, Lattice]) -> Optional[str]:
    return (_kind_matches_source_ref(output)
           or _no_duplicate_atoms(output)
           or _no_overlapping_word_spans(output)
           or _no_cross_kind_ms_overlap(output, lattices))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_pass2_shard(
    file_rows: List[Tuple[str, str, int, Lattice]],
    pass1_output: Pass1Output,
    shard_frames: List[PlannedFrame],
    images_b64: Dict[Tuple[str, int], str],
) -> ic.Completion:
    """One pass-2 shard call. ``file_rows`` + ``pass1_output`` render the SAME
    cached prefix on every shard in a run (identical blocks -> cache hits
    after the first); ``shard_frames``/``images_b64`` are this shard's images
    only, appended uncached. Raises ``ValueError`` if the shard has no
    resolvable images."""
    cached_blocks = build_pass1_blocks(file_rows) + [text_block(render_pass1_output(pass1_output))]
    image_blocks = build_pass2_shard_blocks(shard_frames, images_b64)
    if not image_blocks:
        raise ValueError("run_pass2_shard: no images resolved for this shard")
    lattices = {fid: lattice for fid, _name, _dur, lattice in file_rows}
    # A shard's cut count isn't budget-capped (only its IMAGE count is -- see
    # build_shards), and each cut's full record (framing/look/caption_zones/
    # taste fences/...) is verbose JSON -- a real ~70-cut project shard has
    # measurably needed >16k output tokens. Generous on purpose; ic.complete
    # still re-asks with more room (up to a hard ceiling) if this undershoots.
    return ic.complete("pass2", _SYSTEM, cached_blocks, Pass2Output,
                       extra_blocks=image_blocks, max_tokens=32000,
                       extra_check=lambda output: _pass2_semantic_checks(output, lattices))
