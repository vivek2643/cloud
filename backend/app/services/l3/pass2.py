"""
Cuts, Pass 2: ONE per-cut vision call -- identity (label/summary/channel/
natural_sound/junk) + full visual judgment (framing/look/captions/taste/
people), merged into a single ``CutJudgment`` the model emits per cut
(pass2_merge.plan.md). Batches are pure size-based chunking (no co-location
constraint) and images are sent to the model exactly ONCE per batch.
``speaker``/``on_camera`` are NOT part of this call (voice_first_identity.
plan.md): who is talking is Pass 1's word-level ``speaker_ids``, who is on
camera is derived code-side by ``identity/apply.py`` from voice->person
binding + this call's own ``people`` list.

History: this used to be two calls -- ``pass2a.py`` (identity + take
resolution, shards co-located by take group so the model could compare
members' pixels side by side) and ``pass2b.py`` (visual judgment only, no
cross-cut dependency, pure chunking) -- split after real ingest runs showed
the model getting unreliable past ~15-40 cuts of output in one call. The
fold back to one call is safe now because the one thing that genuinely
needed cross-cut pixels -- resolving a take group's members into
take/winner/outlook -- moved to deterministic code (``apply_take_groups``,
fed by pass 1's ``take_candidates``); everything else pass 2a carried
(label, summary, channel, on_camera, natural_sound, speaker, junk) is
per-cut and never needed another cut's pixels at all. Two calls were paying
to send the SAME frames twice for a co-location requirement that no longer
exists.

``Pass2Cut``/``Pass2Output`` (below) are the final MERGED per-cut record
``post.py`` consumes -- UNCHANGED in shape from before the fold, so every
downstream consumer (``identity/apply.py``, ``post.assemble_cut_records``,
``footage_map``, ``observe``) is untouched. ``CutJudgment``/
``Pass2BatchOutput`` are the model's own per-call response schema (no
``take_group_id``/``take_role`` -- the model never resolves takes anymore);
``to_pass2_cuts`` is the direct (no-merge-needed) conversion from one to the
other, and ``apply_take_groups``/``apply_junk_suspects`` stamp the
code-owned fields on afterward.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import get_settings
from app.services.l3.image_plan import PlannedFrame
from app.services.l3.lattice import Lattice, resolve_speech_span_ms
from app.services.l3.pass1 import Pass1Output, build_pass1_blocks, render_pass1_output
from app.services.l3.pass2_params import MAX_CUTS_PER_PASS2_BATCH
from app.services.llm import client as ic
from app.services.llm.base import image_block, text_block

__all__ = [
    "Framing", "Look", "TasteFences", "Appearance", "PersonLook",
    "CutJudgment", "Pass2BatchOutput", "Pass2Cut", "Pass2Output",
    "to_pass2_cuts", "apply_junk_suspects", "apply_take_groups",
    "build_pass2_batches", "run_pass2_batch",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Visual-judgment schema pieces (moved from the old pass2b.py verbatim --
# still exactly what the identity map / arrange / grade layers consume)
# --------------------------------------------------------------------------

# Ordinal shot-size vocabulary, tightest -> widest (+ "unsure"). A CATEGORY
# the model owns (deterministic-keep rule); code turns it into the ordinal
# tightness term of total_quality (post._shot_tightness). Kept as a closed set
# so that ranking is well-defined; anything off-list reads as "unsure".
SHOT_SIZES = (
    "extreme_close_up", "close_up", "medium_close_up", "medium",
    "medium_wide", "wide", "extreme_wide", "unsure",
)

# Technical shot stability -- a category a SINGLE still often can't judge
# (perception_upgrade.plan.md Part C2); the 2nd (early/late) frame makes it
# answerable by showing whether the frame drifted/blurred/exposure-shifted
# between the two moments. Purely descriptive; code (not the model) decides
# what, if anything, to do with it (see post.py's optional visual-score
# penalty). Closed set so "unsure" is the only off-list fallback.
SHOT_QUALITY = (
    "stable", "shaky", "whip", "soft_focus", "racking_focus", "exposure_shift", "unsure",
)

# cuts_v4_segmentation.plan.md section 5: a coarse SEMANTIC prior on which side
# of a moment carries the value -- NEVER a timestamp (the VLM only ever sees
# ~2 frames, so it can name "what kind of moment" but never "when"; location
# stays fully deterministic, owned by v4_segment.py). Used only in a V4 ingest
# run's shape-aware ladder (cutrecord_map._video_rung); ignored entirely on V3.
SHAPES = ("before", "after", "both", "center", "none")


class Framing(BaseModel):
    subject_box: Tuple[float, float, float, float] | None = None   # normalized x,y,w,h
    crop_16x9: Tuple[float, float, float, float] | None = None
    crop_9x16: Tuple[float, float, float, float] | None = None
    crop_1x1: Tuple[float, float, float, float] | None = None
    rotation_deg: float = 0.0
    # How tight the framing is on the subject -- one of SHOT_SIZES. Purely a
    # category from the pixels; code (not the model) maps it to a number.
    shot_size: str = "unsure"
    # Technical stability -- one of SHOT_QUALITY. Optional/prompt-nudged only
    # (Flash-Lite guardrail: never made required -- see gemini_schema).
    shot_quality: str = "unsure"


class WhiteReference(BaseModel):
    """A candidate neutral object/region for white-balance anchoring
    (color_grading.plan.md SS2.3). The model only PROPOSES this from the
    pixels -- deterministic code in the correct layer verifies it actually
    reads as neutral (low Lab a*/b* cast) before ever trusting it, same
    "model proposes, code decides" split as `shot_size`."""
    present: bool = False
    region: Tuple[float, float, float, float] | None = None   # normalized x,y,w,h
    object: str | None = None   # brief description, e.g. "white wall", "grey card", "white shirt"


class Look(BaseModel):
    graded: bool = False
    palette: List[str] = Field(default_factory=list)
    exposure_flags: List[str] = Field(default_factory=list)
    white_reference: WhiteReference = Field(default_factory=WhiteReference)


class TasteFences(BaseModel):
    max_tasteful_speed: float = 1.0
    min_tasteful_speed: float = 1.0


class Appearance(BaseModel):
    """Structured, STABLE-ONLY identity traits (identity_map.plan.md Phase 0):
    categorical fields code can match exactly, instead of parsing free prose.
    Deliberately excludes anything volatile (clothing, pose, current action)
    -- those change shot to shot and would poison the cross-file fingerprint
    `identity/reconcile.py` clusters on. Every field is optional/"unsure";
    the model states what it can actually see, never guesses to fill a slot."""
    model_config = ConfigDict(extra="forbid")

    apparent_gender: str | None = None    # "male" | "female" | "unsure"
    apparent_age_band: str | None = None  # "child"|"teen"|"20s"|"30s"|"40s"|"50s"|"60s+"|"unsure"
    hair: str | None = None               # "bald"|"very_short"|"short"|"medium"|"long"|"unsure"
    hair_color: str | None = None         # "black"|"brown"|"blonde"|"grey"|"white"|"red"|"unsure"
    facial_hair: str | None = None        # "none"|"stubble"|"moustache"|"beard"|"goatee"|"unsure"
    glasses: str | None = None            # "yes"|"no"|"unsure"
    skin_tone: str | None = None          # "light"|"medium"|"tan"|"dark"|"unsure"
    build: str | None = None              # "slim"|"average"|"heavy"|"unsure"


class PersonLook(BaseModel):
    """A concise visual fingerprint of one person visible in the cut -- enough
    to recognise the same person across cuts by eye, never an identity claim.
    Categorical/descriptive only (the model owns appearance; it never assigns
    scores or ids)."""
    model_config = ConfigDict(extra="forbid")

    description: str                       # e.g. "man, short dark hair, beard, grey hoodie"
    appearance: Appearance = Field(default_factory=Appearance)  # stable-trait fingerprint, see Appearance
    position: str | None = None            # rough frame position: "left" | "center" | "right"


# --------------------------------------------------------------------------
# CutJudgment -- the MODEL'S per-call response schema (moved from the old
# pass2a.py's IdentityCut, plus the visual fields above; NO take_group_id/
# take_role -- D1/D2 (pass2_merge.plan.md): the model never resolves takes,
# code owns that end to end via `apply_take_groups`).
# --------------------------------------------------------------------------

# The model intermittently echoes pass 1's OWN unit name into the kind enum --
# "video_tentative_group" (the pass-1 group name) instead of the canonical
# "video", or "speech_cut" instead of "speech". It's unambiguous (the ref
# prefix already pins the kind), so normalize it at parse time rather than
# burn a re-ask on a pure naming tic. Observed twice-in-a-row on one real
# Reel-trail shard, which the one-re-ask loop then couldn't clear.
_KIND_ALIASES = {
    "video_tentative_group": "video", "video_group": "video", "vid": "video",
    "video_cut": "video",
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

# Per-cut people ceiling: list at most this many people PER CUT (the most
# prominent), so a cut with a lot of people (a crowd, a busy background) stays a
# tight, readable signal instead of an exhaustive roster the brain must wade
# through. This is a PER-CUT cap only -- the project-wide cast table
# (identity/reconcile.py) is uncapped and can hold as many distinct people as
# the shoot really has; it just never sees more than this many from any single
# cut. The prompt asks the model to order people most-prominent-first; this is
# the deterministic backstop that enforces the ceiling regardless.
MAX_PEOPLE_PER_CUT = 4


class CutJudgment(BaseModel):
    source_ref: str                 # e.g. "speech_cut[2]" / "video_group[0]" -- joins back to pass 1
    kind: str                        # "speech" | "video"
    file_id: str
    word_span: Tuple[int, int] | None = None    # speech cuts only
    atom_ids: List[int] | None = None           # video cuts only
    label: str
    summary: str
    channel: str | None = None      # "said" | "done" | "shown"
    natural_sound: bool = False
    junk: bool = False
    junk_reason: str = ""
    framing: Framing = Field(default_factory=Framing)
    look: Look = Field(default_factory=Look)
    caption_zones: List[Tuple[float, float, float, float]] = Field(default_factory=list)
    taste_fences: TasteFences = Field(default_factory=TasteFences)
    readability_ms: int = 0
    # Every person visible in this cut, described well enough to re-identify by
    # eye across cuts (for identity_map + "show the speaker" arrange decisions).
    # Empty for a cut with no people on screen. voice_first_identity.plan.md
    # Part C: `speaker`/`on_camera` (the old per-still LLM guesses) are GONE --
    # who is talking (SpeechCut.speaker_ids, deterministic from word-level
    # diarization) and who is on camera (identity/apply.py, derived from
    # voice->person binding + this `people` list) are now both code-owned.
    people: List[PersonLook] = Field(default_factory=list)
    # perception_upgrade.plan.md Part C3: any legible on-screen text/graphics
    # (title, lower-third, slide, UI) -- "" when none. Optional/prompt-nudged
    # only (Flash-Lite guardrail: never required).
    screen_text: str = ""
    # cuts_v4_segmentation.plan.md section 5: which side of the cut's key
    # moment carries the value -- see SHAPES. V3 ingests never read this field.
    shape: str = "center"

    @field_validator("channel", mode="before")
    @classmethod
    def _normalize_channel(cls, v: Any) -> Any:
        if isinstance(v, str):
            key = v.strip().lower()
            return _CHANNEL_ALIASES.get(key, key)
        return v

    @field_validator("shape", mode="before")
    @classmethod
    def _normalize_shape(cls, v: Any) -> Any:
        # Off-list/missing -> the safe "center" default (section 5: "Default
        # when VLM missing or low-info: center") rather than a schema error --
        # shape is a coarse prior, never worth burning a re-ask over.
        if isinstance(v, str) and v.strip().lower() in SHAPES:
            return v.strip().lower()
        return "center"

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, v: Any) -> Any:
        if isinstance(v, str):
            key = v.strip().lower()
            return _KIND_ALIASES.get(key, key)
        return v

    @model_validator(mode="after")
    def _kind_from_ref(self) -> "CutJudgment":
        # kind is FULLY determined by the source_ref prefix (speech_cut[ ->
        # speech, video_group[ -> video), so DERIVE it here instead of trusting
        # the model's kind field. Flash-Lite routinely confuses kind (structural)
        # with channel (semantic) and emits kind="said"/"shown" on an otherwise
        # valid ref; left uncorrected that trips the ref/kind semantic check and
        # burns the single re-ask on a field we can compute -- observed
        # hard-failing whole runs (video_group[..] kind="shown", speech_cut[..]
        # kind="said"). An unrecognized prefix is left as-is; _source_refs_exist
        # rejects a genuinely invalid ref separately.
        if self.source_ref.startswith("speech_cut["):
            self.kind = "speech"
        elif self.source_ref.startswith("video_group["):
            self.kind = "video"
        return self


class Pass2BatchOutput(BaseModel):
    # A response wrapped under an unexpected top-level key would otherwise
    # "validate" as an empty result instead of failing loud -- observed in
    # the wild once (a whole payload nested under a literal "$PARAMETER_NAME"
    # key). extra="forbid" turns that into a loud schema violation instead.
    model_config = ConfigDict(extra="forbid")

    cuts: List[CutJudgment] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Pass2Cut / Pass2Output -- the final MERGED per-cut record post.py consumes.
# Unchanged in shape from before the fold (byte-compatible downstream).
# --------------------------------------------------------------------------

class Pass2Cut(BaseModel):
    source_ref: str                 # e.g. "speech_cut[2]" / "video_group[0]" -- joins back to pass 1
    kind: str                        # "speech" | "video"
    file_id: str
    word_span: Tuple[int, int] | None = None    # speech cuts only
    atom_ids: List[int] | None = None           # video cuts only
    label: str
    summary: str
    # voice_first_identity.plan.md Phase C/G: `speaker`/`on_camera` are no
    # longer LLM-echoed. `on_camera` stays as a field, but is now PURELY
    # code-derived (identity/apply.py sets it after voice->person binding
    # resolves; None until then, and None forever for an unbound cut).
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
    channel: str | None = None      # "said" | "done" | "shown" (video: done|shown)
    # Per-person visual fingerprints -- appearance descriptors used to
    # recognise the same person across cuts. List of {description, position,
    # appearance}. See PersonLook.
    people: List[dict] = Field(default_factory=list)
    # perception_upgrade.plan.md Part C3: on-screen text/graphics, "" if none.
    screen_text: str = ""
    # voice_first_identity.plan.md Phase C: the GLOBAL voice(s) heard in this
    # cut, deterministic from Pass 1's word-level SpeechCut.speaker_ids
    # mapped through voice clustering (identity/voices.py) -- backfilled in
    # to_pass2_cuts, replacing the removed LLM `speaker` guess. Empty for a
    # video cut or a speech cut with no resolvable voice.
    voice_ids: List[str] = Field(default_factory=list)
    # The global PERSON id (Px) the speaker pass bound the speaking voice
    # to -- set later by identity/apply.py (Phase F/G), once the whole
    # project's cast is reconciled (needs every batch's cuts assembled
    # first, so it can't be known at to_pass2_cuts time). None until then,
    # and None forever for a voice with no confident binding -- honest
    # ignorance, never a guess.
    speaker_person: str | None = None
    # Every global PERSON id (Px) visible on screen in this cut -- from the
    # per-cut-occurrence face clustering (identity/reconcile.py Phase D),
    # set by identity/apply.py alongside speaker_person. Replaces the old
    # one-person-per-FILE `oncam` assumption: a cut can now show several
    # people at once. Empty until identity/apply.py runs, and empty forever
    # for a cut with no confidently-clustered face.
    visible_persons: List[str] = Field(default_factory=list)
    # cuts_v4_segmentation.plan.md section 5: carried straight through from
    # CutJudgment. Ignored on a V3 ingest run.
    shape: str = "center"


class Pass2Output(BaseModel):
    cuts: List[Pass2Cut] = Field(default_factory=list)


def to_pass2_cuts(
    judgments: List[CutJudgment],
    pass1: Optional[Pass1Output] = None,
    voice_of: Optional[Dict[Tuple[str, str], str]] = None,
) -> List[Pass2Cut]:
    """A batch's validated, backfilled CutJudgments -> the final Pass2Cut
    shape post.py consumes. take_group_id/take_role start unset -- code
    (`apply_take_groups`) stamps them from pass 1's take_candidates. This is
    the direct replacement for the old `merge_identity_and_visual`: there's
    nothing to MERGE anymore (one call already emits the whole record), just
    a straight field copy, `people` flattened to plain dicts same as before.

    voice_first_identity.plan.md Phase C: `voice_ids` is backfilled HERE
    (not in `backfill_locators`, since it needs `word_span` already
    resolved) from pass 1's own `SpeechCut.speaker_ids` for this cut's
    (file_id, word_span), mapped through `voice_of` (identity/voices.
    assign_voices, computed once up front from L1 embeddings -- available
    before pass 2 ever runs). `speaker_person`/`on_camera` stay unset here:
    they need the whole project's voice->person binding
    (identity/voice_id.py, voice_id_pass.plan.md), which can't run until
    every batch's cuts are assembled -- `identity/apply.py` sets them
    afterward. `pass1`/
    `voice_of` default to None/empty so existing callers (and every test
    that doesn't care about voice identity) are unaffected -- voice_ids is
    then simply empty for every cut, same as before this field existed."""
    speaker_ids_by_span: Dict[Tuple[str, Tuple[int, int]], List[str]] = {}
    if pass1 is not None:
        for sc in pass1.speech_cuts:
            speaker_ids_by_span[(sc.file_id, tuple(sc.word_span))] = sc.speaker_ids
    voice_of = voice_of or {}

    out: List[Pass2Cut] = []
    for j in judgments:
        voice_ids: List[str] = []
        if j.kind == "speech" and j.word_span is not None:
            local_ids = speaker_ids_by_span.get((j.file_id, tuple(j.word_span))) or []
            # Preserve Pass 1's DOMINANT-FIRST order (SpeechCut.speaker_ids is
            # already ordered by spoken time, sub-threshold voices dropped --
            # see pass1._speaker_ids_for_span); dedupe order-preservingly rather
            # than sorted() so voice_ids[0] stays the beat's dominant voice, which
            # identity/apply._rewrite_cuts credits as speaker_person.
            seen: set = set()
            for s in local_ids:
                v = voice_of.get((j.file_id, s))
                if v is not None and v not in seen:
                    seen.add(v)
                    voice_ids.append(v)
        out.append(Pass2Cut(
            source_ref=j.source_ref, kind=j.kind, file_id=j.file_id,
            word_span=j.word_span, atom_ids=j.atom_ids,
            label=j.label, summary=j.summary,
            junk=j.junk, junk_reason=j.junk_reason,
            framing=j.framing, look=j.look, caption_zones=j.caption_zones,
            taste_fences=j.taste_fences, readability_ms=j.readability_ms,
            natural_sound=j.natural_sound, channel=j.channel,
            people=[p.model_dump() for p in j.people[:MAX_PEOPLE_PER_CUT]],
            screen_text=j.screen_text,
            voice_ids=voice_ids, shape=j.shape,
        ))
    return out


def apply_junk_suspects(pass2: Pass2Output, pass1: Pass1Output, lattices: Dict[str, Lattice]) -> Pass2Output:
    """Deterministically carry pass 1's semantic junk calls onto the final
    cuts. A cut fully contained in a pass-1 junk_suspect span (a leading
    camera cue that the coverage-fill surfaced as its own recovered cut, dead
    air, etc.) is marked junk -- rather than trusting pass 2 to re-flag it
    from a single still. Junk is binary and RECOVERABLE (hidden into the
    Discarded tray, never deleted), so nothing is lost. Conservative: only
    EXACT containment, never a partial overlap (which might clip real
    content).

    cuts_v4_only.plan.md: video-junk matching is ms-span overlap, not
    atom_ids -- a V4 video cut carries no atom_ids to intersect against (the
    old `atom_ids <= suspect.atom_ids` test was already permanently
    vacuous). A junk suspect's own `atom_ids` (still emitted by pass 1 --
    atoms remain the non-speech scaffolding) are resolved to their ms span
    via `lattices`, the same edge-resolution `post._junk_suspect_spans`
    uses; a video cut's span comes straight from its own
    `pass1.video_tentative_groups` entry (the segmenter's own ground truth),
    looked up by source_ref via `_ref_index`."""
    speech_susp: Dict[str, List[Tuple[int, int, str]]] = {}
    video_susp: Dict[str, List[Tuple[int, int, str]]] = {}
    for js in pass1.junk_suspects:
        if js.word_span is not None:
            speech_susp.setdefault(js.file_id, []).append((js.word_span[0], js.word_span[1], js.reason))
        elif js.atom_ids:
            lattice = lattices.get(js.file_id)
            if lattice is None:
                continue
            atoms_by_id = {a.atom_id: a for a in lattice.atoms}
            members = [atoms_by_id[i] for i in js.atom_ids if i in atoms_by_id]
            if members:
                span = (min(a.start_ms for a in members), max(a.end_ms for a in members))
                video_susp.setdefault(js.file_id, []).append((span[0], span[1], js.reason))

    n = 0
    for c in pass2.cuts:
        if c.junk:
            continue
        hit = None
        if c.kind == "speech" and c.word_span is not None:
            a, b = c.word_span
            hit = next((r for (sa, sb, r) in speech_susp.get(c.file_id, []) if sa <= a and b <= sb), None)
        elif c.kind == "video":
            gi = _ref_index(c.source_ref, "video_group[")
            if gi is not None and gi < len(pass1.video_tentative_groups):
                vg = pass1.video_tentative_groups[gi]
                if vg.src_in_ms is not None and vg.src_out_ms is not None:
                    cs, ce = vg.src_in_ms, vg.src_out_ms
                    hit = next((r for (ss, se, r) in video_susp.get(c.file_id, []) if ss <= cs and ce <= se), None)
        if hit is not None:
            c.junk = True
            c.junk_reason = c.junk_reason or hit
            n += 1
    if n:
        logger.info("pass2: %d cut(s) marked junk from pass-1 suspects", n)
    return pass2


def apply_take_groups(pass2: Pass2Output, pass1: Pass1Output) -> Pass2Output:
    """Stamp take_group_id/take_role onto every cut from pass 1's
    take_candidates -- fully deterministic (pass2_merge.plan.md D1/D2): the
    model never resolves takes anymore (no take_group_id/take_role in
    CutJudgment's schema at all), so this is the ONLY place those fields are
    ever set. Generalizes the old `apply_outlook_roles`:

      * group_id prefixed "outlook:" -> take_role="outlook" (alternate
        camera, never a retake -- these members were declared simultaneous
        by the sync/outlook machinery, not judged from pixels).
      * every other group -> take_role="take", all members starting equal;
        `post._enforce_take_winner` crowns the highest-total_quality member
        of each same-setting cluster as "winner" afterward -- replacing
        pass 2's own winner call entirely, same as it always has.

    A cut whose (file_id, word_span) isn't any take_candidate's member is
    untouched (take_group_id/take_role stay None -- it's not part of any
    take or outlook group). Mirrors `apply_junk_suspects`: pass 1 owns the
    meaning, code carries it deterministically onto the final cuts."""
    member_group: Dict[Tuple[str, Tuple[int, int]], Tuple[str, str]] = {}
    for tc in pass1.take_candidates:
        role = "outlook" if str(tc.group_id).startswith("outlook:") else "take"
        for m in tc.members:
            member_group[(m.file_id, tuple(m.word_span))] = (tc.group_id, role)
    if not member_group:
        return pass2
    n = 0
    for c in pass2.cuts:
        if c.kind != "speech" or c.word_span is None:
            continue
        hit = member_group.get((c.file_id, tuple(c.word_span)))
        if hit is None:
            continue
        gid, role = hit
        if c.take_group_id != gid or c.take_role != role:
            c.take_group_id = gid
            c.take_role = role
            n += 1
    if n:
        logger.info("pass2: %d cut(s) stamped with take/outlook group from pass-1 take_candidates", n)
    return pass2


# --------------------------------------------------------------------------
# Prompt (merged from the old pass2a.py identity brief + pass2b.py visual
# brief; all take/winner/outlook instructions removed -- code owns take
# resolution now, see apply_take_groups above)
# --------------------------------------------------------------------------

_SYSTEM_PREFIX = (
    "You already did pass 1 for this project: you saw every clip's "
    "transcript and video-atom table, and grouped them into speech cuts and "
    "tentative video groups (repeated below verbatim). Now you see the "
    "actual pixels: numbered stills, each captioned with which clip/"
    "timestamp/pass-1 unit it belongs to.\n\n"
    "SCOPE: this call covers ONLY the clips whose images you are shown "
    "below -- the pass-1 result may mention other clips, but those are "
    "handled in separate calls; never emit a cut for a clip you were not "
    "shown images for.\n\n"
    "You may see more than one frame for one cut. Two frames labelled "
    "early/late in their captions are a PAIR, the SAME cut at two different "
    "moments -- read them TOGETHER and describe what CHANGES between them "
    "(a gesture completing, a camera move, an expression shifting, an "
    "object entering/leaving frame) -- that change IS the content worth "
    "judging. More than two frames for one cut are separate glimpses across "
    "a busier moment, not a pair -- read across all of them for the overall "
    "arc, still one judgment. NEVER emit more than one cut record for one "
    "source_ref just because it has multiple frames; it is still exactly "
    "one cut (see the one-record-per-ref rule below).\n\n"
)

# cuts_v4_only.plan.md: video cuts are always V4-shaped now -- a
# video_tentative_group's boundary is already finished by the segmenter
# before any pixel was seen, so the model's job on it is only to label it,
# never to adjust or split its span.
_V4_VIDEO_CLAUSE = (
    "For every in-scope speech_cut, emit ONE final cut record. Each "
    "in-scope video_tentative_group is ALREADY a finished cut -- its "
    "boundary was chosen from the signals before any pixel was ever seen, "
    "so your job on it is only to LABEL it (label/summary/channel/shape/"
    "people) and decide keep-vs-junk for the WHOLE cut: emit exactly one "
    "cut per video_group[i], never split or merge one, and never invent a "
    "boundary inside it -- if part of it isn't wanted, junk the whole cut, "
    "don't try to re-cut it.\n\n"
    "Do NOT echo word_span (it is derived from your source_ref by code); a "
    "video cut carries no atom_ids to echo either.\n\n"
)

_SYSTEM_SUFFIX = (
    "Per cut, judge from the pixels and transcript:\n"
    "  - label, summary (a best guess from image + transcript is fine, and "
    "expected). A label must name what the cut SHOWS (e.g. 'forehand "
    "swing', 'catches the ball'), never a mechanical 'settle'/'trailing "
    "frames'. summary must describe WHAT IS HAPPENING -- the action, the "
    "beat, the on-screen event, and (when you saw two frames) what CHANGES "
    "across them -- never a static description of a still ('a man in a "
    "kitchen' is not a summary; 'he cracks an egg into the bowl and starts "
    "whisking' is).\n"
    "  - channel: the delivery CATEGORY -- for a video cut set \"done\" when "
    "an action is performed/demonstrated on screen (a swing, a catch, "
    "pouring, assembling, a gesture that IS the content) or \"shown\" when "
    "it is b-roll/an object/scenery/a display with no performed action "
    "(speech cuts are always \"said\" -- you may omit channel for them)\n"
    "  - natural_sound (does the cut carry sound worth keeping)\n"
    "  - shape: which side of the cut's key moment carries the value -- "
    "\"before\" (everything before it matters; it's a build-to-impact, end "
    "ON it), \"after\" (everything after it matters; it's a reveal/payoff, "
    "keep the tail), \"both\" (the moment and both sides matter), \"center\" "
    "(no strong asymmetry -- the safe default), \"none\" (nothing to trim "
    "to, e.g. a screen recording or uniform footage). This is a coarse call "
    "from what kind of moment it is, NEVER a timestamp -- you are not "
    "judging exactly when it happens, only which side of it matters more.\n"
    "  - junk (+reason): BINARY, by MEANING -- set junk=true ONLY when the "
    "cut is clearly not part of the piece (a camera cue like 'and go'/"
    "'3-2-1'/'take three', pre-roll setup, obvious dead air). Junk is "
    "recoverable -- hidden into a Discarded tray, not deleted -- but keep "
    "the bar HIGH: if there is ANY doubt a cut might be wanted, leave "
    "junk=false. Never mark an ACTION/motion payoff junk.\n"
    "  - framing: subject_box, plus the best crop for each delivery shape "
    "(crop_16x9, crop_9x16, crop_1x1), each recomposed to keep the subject "
    "and eyeline in frame for that aspect (not a centre-crop of the "
    "landscape), rotation_deg only for a visibly tilted shot (else 0), and "
    "shot_size -- how tight the frame is on the subject, exactly one of: "
    "extreme_close_up, close_up, medium_close_up, medium, medium_wide, "
    "wide, extreme_wide (use unsure only if there is no clear subject), and "
    "shot_quality -- technical stability, exactly one of: stable, shaky "
    "(handheld wobble), whip (a fast pan/tilt blur), soft_focus, "
    "racking_focus (focus visibly pulling), exposure_shift (visibly "
    "brightening/darkening); if you saw two frames judge it from how the "
    "shot behaves BETWEEN them, one still is unsure by default\n"
    "  - look: graded vs log/flat, palette, exposure_flags, and its "
    "white_reference, which is a field NESTED INSIDE look (look."
    "white_reference, NOT a top-level field): if some object in frame is "
    "genuinely neutral-colored (a white/grey wall, a grey card, white "
    "paper, a plain white garment -- NOT skin, NOT anything colored or "
    "patterned) and evenly lit, set look.white_reference.present=true with "
    "its region (normalized x,y,w,h) and a short object description; "
    "otherwise present=false and leave region/object empty. Only propose it "
    "when genuinely confident -- this is a candidate the code will verify, "
    "not a guess to force.\n"
    "  - caption_zones (normalized boxes clear of the subject across every "
    "image you were shown for that cut), taste fences (max/min tasteful "
    "playback speed for this content), and readability_ms (how long a "
    "viewer needs to read this frame if it holds as a still)\n\n"
    "Also list `people`: the people who MATTER in this cut -- at most the "
    "FOUR most prominent (largest / most central / in the foreground), listed "
    "MOST-PROMINENT FIRST -- each with a concise description that would let "
    "someone recognise them again across cuts (apparent gender/age, hair, "
    "facial hair, clothing/colour, anything distinctive) and their rough frame "
    "position (left/center/right). If the cut is a crowd, list only its few "
    "foreground figures and say 'crowd'/'group' in the summary rather than "
    "trying to enumerate everyone. No people on screen -> empty list. Describe "
    "appearance only; never guess names, never judge who is speaking, and "
    "never assign any score.\n\n"
    "Each person ALSO gets a structured `appearance` (nested inside that "
    "person, alongside description/position): apparent_gender, "
    "apparent_age_band, hair, hair_color, facial_hair, glasses, skin_tone, "
    "build -- each one of its listed categories, or omitted/\"unsure\" if "
    "not clearly visible. These are for matching the SAME PERSON across "
    "different cuts and different camera angles, so use ONLY traits that "
    "stay stable shot to shot: never clothing, never pose, never what they "
    "are doing right now -- those belong in `description`/`position`, not "
    "`appearance`. Leave a field unset rather than guess.\n\n"
    "screen_text: any legible on-screen text or graphics (a title, "
    "lower-third, slide, UI element, screen share) -- transcribe what it "
    "says; note briefly if it changes between the frames you were shown. "
    "Empty string if there is no on-screen text/graphic.\n\n"
    "Reference every cut by source_ref using the SAME ref string pass 1 "
    "(and the image captions) used for it -- speech_cut[i] or "
    "video_group[i], VERBATIM. Never invent a new ref."
)

def system_prompt() -> str:
    """The per-call system prompt. cuts_v4_only.plan.md: V4 is the only cuts
    path, so this always returns the V4 video-handling clause ("already a
    finished cut, never split"). Everything else -- speech grouping,
    framing, look, junk, shape, people -- is unchanged."""
    return _SYSTEM_PREFIX + _V4_VIDEO_CLAUSE + _SYSTEM_SUFFIX

# gemini_pass2.plan.md Phase 3: appended to the base system prompt ONLY when
# ingest_pass2_provider=="gemini" -- never mutates the base prompt (that
# would perturb the proven Claude path). The Anthropic path relies on
# FORCED tool-use (tool_choice pinned to the one schema tool) to guarantee a
# non-empty, fully-populated response; Gemini's structured-output path has
# no equivalent forcing mechanism, so an unconstrained call was observed
# satisfying the schema trivially with `cuts: []`. This suffix states the
# contract explicitly; `ingest_gemini.gemini_schema` backs it with an
# enforced `cuts` minItems=1 + required, so a genuinely empty response is a
# schema violation (re-asked), not a silent success.
_GEMINI_REINFORCE = (
    "\n\nOUTPUT CONTRACT (STRICT): Return a JSON object {\"cuts\": [ ... ]}. "
    "Emit EXACTLY ONE cut object per source_ref you were shown -- never an "
    "empty list, never skip a ref. Every cut MUST include a non-empty label "
    "and summary, plus framing (subject_box + shot_size) and look. Fill "
    "every required field from the pixels; use the 'unsure' category rather "
    "than omitting a field.\n"
    "ALWAYS set framing.subject_box: the normalized [x, y, w, h] box around "
    "the shot's MAIN compositional subject. Every shot has one -- for b-roll, "
    "scenery, or product footage with no person, box the dominant focal "
    "element (the hero object, the focal landscape feature, the horizon "
    "subject). Never leave subject_box null.\n"
    "In `people`, list EVERY person visible on screen (one entry each); leave "
    "it empty ONLY when there is genuinely no person anywhere in frame."
)


def gemini_system_prompt() -> str:
    """The exact system string a Gemini-provider batch call sends
    (``system_prompt()`` + the reinforcement suffix). P4 (gemini_pass2.plan.md):
    ``ingest.py`` bakes this into the per-run ``CachedContent`` so caching
    doesn't silently drop the reinforcement -- a cached call's per-call
    config never re-sends ``system_instruction`` (see
    ``ingest_gemini._build_config``), so whatever was baked in at cache
    creation is the ONLY system prompt those calls ever get."""
    return system_prompt() + _GEMINI_REINFORCE


def build_pass2_batch_blocks(
    planned_frames: List[PlannedFrame], images_b64: Dict[Tuple[str, int], str],
) -> List[Dict[str, Any]]:
    """Numbered [caption, image] block pairs for one batch, in
    (file_id, ref, ts_ms) order -- perception_upgrade.plan.md Part B: sorting
    by ref (not just file_id/ts_ms) keeps a ref's early/late pair ADJACENT in
    the sequence the model reads, rather than interleaved with other cuts'
    frames that merely happen to fall between them in time. Frames with no
    extracted image (not yet pulled, or extraction failed upstream) are
    skipped rather than sent blank. Each caption is labelled with its phase
    (early/late) so the model knows which frames are a pair to read
    TOGETHER; a "only" phase (the common case -- most cuts stay single-frame)
    gets no suffix at all."""
    ordered = sorted(planned_frames, key=lambda f: (f.file_id, f.ref, f.ts_ms))
    blocks: List[Dict[str, Any]] = []
    for n, f in enumerate(ordered, start=1):
        b64 = images_b64.get((f.file_id, f.ts_ms))
        if b64 is None:
            continue
        suffix = f" ({f.phase})" if f.phase in ("early", "late") else ""
        blocks.append(text_block(f"IMG {n} = clip {f.file_id}, {f.ts_ms / 1000:.1f}s, {f.ref}{suffix}"))
        blocks.append(image_block(b64))
    return blocks


# --------------------------------------------------------------------------
# Batching: pure size-based chunking of the image-bearing cuts (one
# speech_cut[i]/video_group[j] each), in stable (file_id, ts_ms) order. NO
# take co-location -- pass2_merge.plan.md Phase 1 moved take-grouping to
# deterministic code, so there is no longer any reason for a take's members
# to share a batch. A batch's own refs' planned frames trivially stay
# together since chunking is by whole ref, never by individual frame.
# --------------------------------------------------------------------------

def build_pass2_batches(
    pass1_output: Pass1Output, planned_frames: List[PlannedFrame],
    max_per_batch: int = MAX_CUTS_PER_PASS2_BATCH,
) -> List[List[str]]:
    """Partition the image-bearing pass-1 units into batches of source_refs."""
    seen: set = set()
    refs: List[str] = []
    for f in sorted(planned_frames, key=lambda f: (f.file_id, f.ts_ms)):
        if f.ref in seen:
            continue
        if f.ref.startswith("speech_cut[") or f.ref.startswith("video_group["):
            refs.append(f.ref)
            seen.add(f.ref)
    return [refs[i:i + max_per_batch] for i in range(0, len(refs), max_per_batch)]


# --------------------------------------------------------------------------
# Semantic checks pydantic can't express -- all observed against the real
# API, all folded into the re-ask loop instead of only surfacing as opaque
# failures downstream in post.assemble_cut_records. (Moved verbatim from the
# old pass2a.py, retyped onto CutJudgment/Pass2BatchOutput.)
# --------------------------------------------------------------------------

def _ref_index(ref: str, prefix: str) -> Optional[int]:
    if ref.startswith(prefix) and ref.endswith("]"):
        idx = ref[len(prefix):-1]
        if idx.isdigit():
            return int(idx)
    return None


def backfill_locators(output: Pass2BatchOutput, pass1: Pass1Output) -> Pass2BatchOutput:
    """Deterministically fill/normalize every cut's word_span from pass 1 by
    source_ref. The model was originally required to echo these verbatim;
    observed against the real API, that echo was the single biggest
    output-complexity failure (66 cuts -> 41 validation errors, twice). They
    carry zero judgment -- pass 1's grouping is final -- so code owns them
    now:

      * speech cut -> word_span := pass1.speech_cuts[i].word_span, always.
      * video cut (own span, no atoms) -> nothing to backfill; atom_ids
        stays None (there ARE none -- cuts_v4_only.plan.md) and the cut's
        real span rides separately on v4_meta_by_ref, keyed by this same
        ref (see post.py/ingest.py)."""
    new_cuts: List[CutJudgment] = []
    for cut in output.cuts:
        update: Dict[str, Any] = {}
        if cut.kind == "speech":
            i = _ref_index(cut.source_ref, "speech_cut[")
            if i is not None and i < len(pass1.speech_cuts):
                update["word_span"] = tuple(pass1.speech_cuts[i].word_span)
        new_cuts.append(cut.model_copy(update=update) if update else cut)
    return Pass2BatchOutput(cuts=new_cuts)


def _locators_resolved(output: Pass2BatchOutput, pass1: Pass1Output) -> Optional[str]:
    """After backfill, every speech cut must have its locator: word_span
    missing means its ref didn't resolve. A video cut carries no atom_ids by
    design -- its span lives on v4_meta_by_ref -- so there's nothing to
    check for it."""
    for cut in output.cuts:
        if cut.kind == "speech" and cut.word_span is None:
            return (f"{cut.source_ref!r} resolved to no word_span -- its ref must name an "
                    f"existing pass-1 speech_cut")
    return None


def _source_refs_exist(output: Pass2BatchOutput, pass1: Pass1Output) -> Optional[str]:
    """Observed against the real API: the model INVENTED refs (e.g.
    "take[intro_greeting]_take1") for cuts it wanted to emit around a take,
    instead of using the pass-1 ref strings. Nothing downstream can join
    such a ref -- image_plan planned no frames for it, so the batch comes up
    imageless and the whole run dies with an unrelated-looking "no images
    resolved" error. Every source_ref must be a literal speech_cut[i] /
    video_group[i] that pass 1 actually emitted."""
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
                    f"ref")
    return None


def _kind_matches_source_ref(output: Pass2BatchOutput) -> Optional[str]:
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


def _no_overlapping_word_spans(output: Pass2BatchOutput) -> Optional[str]:
    """Observed against the real API: two speech cuts in the same file with
    identical/overlapping word_span ranges (a duplicate or a pass-1 grouping
    mistake echoed through) -- this only ever surfaces downstream as a raw
    ms-coverage overlap in post.assemble_cut_records, which doesn't say
    WHICH two cuts or why. Checking word indices directly here is both
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


def _resolve_cut_span_ms(cut: CutJudgment, lattices: Dict[str, Lattice]) -> Optional[Tuple[int, int]]:
    """Best-effort (s, e) in ms for a speech cut, same resolution post.py
    uses -- word/atom edges, clamped so a speech cut's silence cushion can
    never reach into a neighboring atom's span (see resolve_speech_span_ms).
    Silence data is skipped here (empty list) since the clamp alone is what
    prevents the overlap this check exists to catch; a few ms of precision
    beyond that doesn't matter for a GROSS-overlap check like this one.

    cuts_v4_only.plan.md: a video cut's real span lives on v4_meta_by_ref,
    not here -- this always returns None for kind=="video" (out of scope
    for this plan to wire up; _no_cross_kind_ms_overlap below is a no-op for
    the video side of the check as a result, matching the already-current
    V4 runtime behavior, not a new regression)."""
    lattice = lattices.get(cut.file_id)
    if lattice is None:
        return None
    if cut.kind == "speech" and cut.word_span:
        return resolve_speech_span_ms(lattice.words, lattice.atoms, cut.word_span, [])
    return None


def _no_cross_kind_ms_overlap(output: Pass2BatchOutput, lattices: Dict[str, Lattice]) -> Optional[str]:
    """A speech cut and a video cut in the same file resolving to
    overlapping ms spans; _no_overlapping_word_spans only checks
    speech-vs-speech, so this catches the CROSS-kind case. Observed against
    the real API surfacing only as an opaque ms-overlap failure in
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


def _resolved_file_id(cut: CutJudgment, pass1: Pass1Output) -> Optional[str]:
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


def _drop_out_of_batch_cuts(output: Pass2BatchOutput, pass1: Pass1Output,
                            batch_refs: set) -> Tuple[Pass2BatchOutput, int]:
    """Drop cuts whose source_ref is OUTSIDE this batch, returning
    (filtered_output, n_dropped). The cached pass-1 render lists EVERY cut, so
    a batch sometimes emits cuts for refs it wasn't shown images for. Each ref
    is handled by exactly one batch, so the strays here are pure duplicates:
    discarding them (rather than failing the whole call) is what lets big
    multicam projects ingest, and keeps post from seeing cross-batch duplicate
    spans. Scoping by ref (not clip) is precise now that one clip's cuts can
    land in different batches. A kept cut whose model file_id disagrees with
    the clip its source_ref resolves to is corrected so post attributes it
    right."""
    kept: List[CutJudgment] = []
    for c in output.cuts:
        if c.source_ref not in batch_refs:
            continue
        owner = _resolved_file_id(c, pass1)
        kept.append(c if owner is None or c.file_id == owner
                    else c.model_copy(update={"file_id": owner}))
    dropped = len(output.cuts) - len(kept)
    return output.model_copy(update={"cuts": kept}), dropped


def _pass2_semantic_checks(output: Pass2BatchOutput, pass1: Pass1Output,
                           lattices: Dict[str, Lattice], batch_refs: set) -> Optional[str]:
    """Run against the BACKFILLED output (see backfill_locators) -- locator
    checks are meaningless before the deterministic fill. Out-of-batch strays
    are filtered out FIRST (see _drop_out_of_batch_cuts) so they can't trip a
    spurious re-ask; run_pass2_batch applies the same filter to the persisted
    output."""
    output = backfill_locators(output, pass1)
    output, _ = _drop_out_of_batch_cuts(output, pass1, batch_refs)
    return (_source_refs_exist(output, pass1)
           or _kind_matches_source_ref(output)
           or _locators_resolved(output, pass1)
           or _no_overlapping_word_spans(output)
           or _no_cross_kind_ms_overlap(output, lattices))


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def run_pass2_batch(
    file_rows: List[Tuple[str, str, int, Lattice]],
    pass1_output: Pass1Output,
    batch_frames: List[PlannedFrame],
    images_b64: Dict[Tuple[str, int], str],
) -> ic.Completion:
    """One pass-2 batch call: identity + full visual judgment, images sent
    ONCE. The cached prefix is SCOPED to this batch: ``file_rows`` are the
    batch's clips only and the pass-1 render is trimmed to the batch's refs
    (see render_pass1_output), so the model can only see -- and so only
    emit -- the cuts it was actually shown. ``batch_frames``/``images_b64``
    are this batch's images, appended uncached. Raises ``ValueError`` if the
    batch has no resolvable images.

    gemini_pass2.plan.md: when ``ingest_pass2_provider=="gemini"``, the
    reinforcement suffix is appended to the system prompt (never mutates the
    base prompt itself, so the proven Claude path is untouched) and, if a
    per-run Gemini ``CachedContent`` is active (P4), the STABLE
    ``build_pass1_blocks`` prefix is left out of this call entirely -- it's
    already baked into that cache; re-sending it here would erase the cost
    win caching exists for. Only the per-batch-trimmed ``render_pass1_output``
    (which differs batch to batch) is ever sent fresh."""
    batch_refs = {f.ref for f in batch_frames}
    settings = get_settings()
    system = system_prompt()
    stable_blocks = build_pass1_blocks(file_rows)
    if settings.ingest_pass2_provider == "gemini":
        system = system + _GEMINI_REINFORCE
        from app.services.llm import ingest_gemini as ig
        if ig.get_pass2_cache_handle():
            stable_blocks = []
    cached_blocks = stable_blocks + [text_block(render_pass1_output(pass1_output, batch_refs))]
    image_blocks = build_pass2_batch_blocks(batch_frames, images_b64)
    if not image_blocks:
        raise ValueError("run_pass2_batch: no images resolved for this batch")
    lattices = {fid: lattice for fid, _name, _dur, lattice in file_rows}
    completion = ic.complete("pass2", system, cached_blocks, Pass2BatchOutput,
                             extra_blocks=image_blocks, max_tokens=32000,
                             extra_check=lambda output: _pass2_semantic_checks(
                                 output, pass1_output, lattices, batch_refs))
    # The persisted output must carry ONLY this batch's refs. The semantic
    # check already filters strays before validating, but ic.complete returns
    # the model's RAW payload, so drop them here too -- otherwise post would
    # see the same cut from two batches as an identical-span overlap.
    filtered, dropped = _drop_out_of_batch_cuts(
        Pass2BatchOutput.model_validate(completion.data), pass1_output, batch_refs)
    if dropped:
        logger.warning("pass2: dropped %d out-of-batch cut(s) from a %d-ref batch "
                       "(each handled by its own batch)", dropped, len(batch_refs))
        completion.data = filtered.model_dump()
    return completion
