"""
L2 perception artifact schema.

This is the contract for what Gemini returns for one short, single-take clip.
It is passed directly to the SDK as a `response_schema` (so Gemini emits typed
JSON) and re-validated on the way back, so the same definition documents the
prompt, constrains the model, and parses the result.

Design rules baked into the shape (see the perception spec discussion):
  * Universal spine + optional modules. Every clip gets clip-level fields; the
    `persons`/`speaking`/`atoms`/... tracks are lists that are simply empty when
    they don't apply (a sunset clip has no persons, no speech, just shown atoms).
  * Single take. There are no shots/scenes -- the whole clip is one continuous
    camera take, so "structure" is a *timeline of detection atoms*, never a cut list.
  * Detection only (cuts v2). The video track is logged as `atoms`: per-beat
    CHANNEL (done/shown) + SUBJECT (person/place/object/graphic) + peak +
    confidence. No editorial buckets, roles, or relations -- a downstream engine
    decides use. Said comes from L1 transcript, Heard from the audio envelope.
  * Stable local ids. People are `p1`, `p2`, ... within this clip; atoms
    reference those ids (`actor`) so subjects link to people. The ids are
    clip-local; cross-video identity is resolved later from the durable traits.
  * Controlled vocabularies (enums) wherever comparability matters, free text
    where nuance matters.

All timestamps are integer milliseconds from the start of the clip.
"""
from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.services.l3 import vocab


def _to_unit(v):
    """Coerce a model-supplied score into 0..1.

    We deliberately drop ``ge``/``le`` Field bounds (Option B): without
    constrained decoding the model occasionally returns out-of-range values
    (e.g. a 1-5 rubric number bleeding into a 0..1 field). Clamp instead of
    failing validation: map (1, 5] -> /5, then clamp to [0, 1].
    """
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x > 1.0:
        x = x / 5.0 if x <= 5.0 else 1.0
    return max(0.0, min(1.0, x))

# v2 adds coarse spatial framing signals (subject `region`s + clip
# `frame_orientation`) used by the editor's auto-reframe/crop.
# v6 (cuts v2) makes the detection-only `atoms` track the sole video-capture
# substrate: per-beat CHANNEL (done/shown) + SUBJECT (person/place/object/
# graphic) + peak + confidence. The old v1 editorial tracks (events / reactions
# / cutaways / content_units / relations / role) have been REMOVED -- the cut
# pipeline reads atoms (+ Said from L1, Heard from the audio envelope) only.
# v7 adds a single clip-level `valence` (emotional tone) -- the one feel signal
# a VLM must supply that L1 cannot derive; the rest of "feel" (pace, energy,
# pauses) is computed downstream from L1. Null-safe: old docs simply lack it.
SCHEMA_VERSION = 7


# The editing vocabulary (vocab.py) is the single source of truth.
# v6 cuts-v2 substrate: the VLM detects video-channel beats as ATOMS. Channel is
# only the two video channels here (audio Said comes from L1 transcript; Heard
# from the RMS envelope -- the VLM never emits them). Subject is orthogonal.
Channel = Enum("Channel", [(c, c) for c in (vocab.CHANNEL_DONE, vocab.CHANNEL_SHOWN)], type=str)
Subject = Enum("Subject", [(s, s) for s in vocab.SUBJECTS], type=str)


def _coerce_channel(v):
    """Lenient atom channel: only the two VIDEO channels are valid here; an
    out-of-vocab or audio value (said/heard) is dropped to None rather than
    failing the whole clip's parse."""
    if v is None:
        return None
    s = str(getattr(v, "value", v)).strip().lower()
    return s if s in (vocab.CHANNEL_DONE, vocab.CHANNEL_SHOWN) else None


def _coerce_subject(v):
    if v is None:
        return None
    s = str(getattr(v, "value", v)).strip().lower()
    return s if s in vocab.SUBJECT_SET else None


def _coerce_enum(enum_cls):
    """Build a lenient `mode='before'` coercer for a closed string Enum.

    The schema is carried in-prompt (no constrained decoding), so the model
    routinely emits near-misses -- a separator swap ('medium-wide' for
    'medium_wide'), stray case/whitespace, or a value just out of vocab. Rather
    than reject the WHOLE clip's perception on one cosmetic mismatch (recall-
    first), normalize separators and drop a genuinely unknown value to None.
    """
    values = {e.value for e in enum_cls}

    def _coerce(v):
        if v is None or isinstance(v, enum_cls):
            return v
        s = str(getattr(v, "value", v)).strip().lower().replace("-", "_").replace(" ", "_")
        return s if s in values else None

    return _coerce


# --------------------------------------------------------------------------
# Controlled vocabularies
# --------------------------------------------------------------------------

class ContentType(str, Enum):
    talking_head = "talking_head"
    interview = "interview"
    vlog = "vlog"
    tutorial = "tutorial"
    demo = "demo"
    product = "product"
    performance = "performance"
    action = "action"
    scenic = "scenic"
    broll = "broll"
    screen_recording = "screen_recording"
    event = "event"
    other = "other"


class PrimaryAxis(str, Enum):
    """What carries the clip -- i.e. which cut-cost channel should dominate."""
    speech = "speech"
    action = "action"
    visual = "visual"
    performance = "performance"


class Valence(str, Enum):
    """The clip's overall EMOTIONAL TONE -- the one feel signal only a viewer can
    read (L1 gives pace/energy/pauses; this gives colour). Keep it coarse; it
    tints how the edit feels, it does not decide cuts."""
    positive = "positive"   # upbeat, warm, celebratory, funny
    neutral = "neutral"     # informational, matter-of-fact
    negative = "negative"   # sad, frustrated, critical
    tense = "tense"         # anxious, high-stakes, suspenseful
    somber = "somber"       # solemn, reflective, heavy, quiet-serious


class CutSensitivity(str, Enum):
    """How forgiving the clip is of cuts overall."""
    high = "high"      # delicate: dialogue / continuous action, few safe seams
    medium = "medium"
    low = "low"        # forgiving: static b-roll, scenery


class TimeOfDay(str, Enum):
    day = "day"
    night = "night"
    golden_hour = "golden_hour"
    dawn_dusk = "dawn_dusk"
    indoor_artificial = "indoor_artificial"
    unsure = "unsure"


class InteriorExterior(str, Enum):
    interior = "interior"
    exterior = "exterior"
    unsure = "unsure"


class ShotSize(str, Enum):
    extreme_close_up = "extreme_close_up"
    close_up = "close_up"
    medium_close_up = "medium_close_up"
    medium = "medium"
    medium_wide = "medium_wide"
    wide = "wide"
    extreme_wide = "extreme_wide"
    unsure = "unsure"


class CameraAngle(str, Enum):
    eye_level = "eye_level"
    low = "low"
    high = "high"
    overhead = "overhead"
    dutch = "dutch"
    unsure = "unsure"


class CameraMovement(str, Enum):
    static = "static"
    pan = "pan"
    tilt = "tilt"
    push_in = "push_in"
    pull_out = "pull_out"
    zoom = "zoom"
    handheld = "handheld"
    follow = "follow"
    whip = "whip"
    orbit = "orbit"
    unsure = "unsure"


class GazeDirection(str, Enum):
    to_camera = "to_camera"
    off_camera = "off_camera"
    at_person = "at_person"
    at_object = "at_object"
    down = "down"
    around = "around"
    unsure = "unsure"


class FrameOrientation(str, Enum):
    """How the footage must be ROTATED to sit upright. `upright` is the norm;
    the others flag footage shot sideways/upside-down whose rotation metadata is
    missing or wrong, so the editor can correct it (orthogonal only)."""
    upright = "upright"
    rotate_cw90 = "rotate_cw90"     # rotate 90 clockwise to make upright
    rotate_ccw90 = "rotate_ccw90"   # rotate 90 counter-clockwise
    rotate_180 = "rotate_180"


class Region(BaseModel):
    """Coarse normalized location of a subject in the frame (origin top-left,
    0..1 of width/height). Approximate is fine -- a loose box around the
    head/subject, NOT a tight detection. The editor uses the box CENTER to keep
    the subject in frame when reframing to another aspect (e.g. a 9:16 reel)."""
    x: float = Field(description="left edge, fraction of width")
    y: float = Field(description="top edge, fraction of height")
    w: float = Field(description="width, fraction of frame width")
    h: float = Field(description="height, fraction of frame height")

    @field_validator("x", "y", "w", "h", mode="before")
    @classmethod
    def _clamp_region(cls, v):
        if v is None:
            return 0.0
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.0


# --------------------------------------------------------------------------
# Clip-level (the universal spine -- present for every clip)
# --------------------------------------------------------------------------

class Look(BaseModel):
    time_of_day: Optional[TimeOfDay] = None
    interior_exterior: Optional[InteriorExterior] = None

    @field_validator("time_of_day", "interior_exterior", mode="before")
    @classmethod
    def _coerce_look_enums(cls, v, info):
        return _coerce_enum({"time_of_day": TimeOfDay, "interior_exterior": InteriorExterior}[info.field_name])(v)
    light_quality: Optional[str] = Field(
        None, description="e.g. 'soft diffused', 'hard direct sun', 'mixed/practical', 'low-key'"
    )
    light_direction: Optional[str] = Field(None, description="e.g. 'frontal', 'back-lit', 'side/left'")
    color_palette: Optional[str] = Field(None, description="dominant colors / grade, e.g. 'warm teal-orange'")
    mood: Optional[str] = None


class SettingObject(BaseModel):
    name: str
    detail: Optional[str] = Field(None, description="make/model/state -- only when it matters editorially")
    is_subject: bool = Field(
        False, description="True only when the clip is ABOUT this object (e.g. a product/car review)"
    )


class Setting(BaseModel):
    location: Optional[str] = Field(None, description="where this is, e.g. 'home kitchen', 'city street', 'forest trail'")
    background: List[str] = Field(default_factory=list, description="notable background elements")
    objects: List[SettingObject] = Field(default_factory=list, description="foreground objects worth indexing")


class Editability(BaseModel):
    primary_axis: Optional[PrimaryAxis] = None
    cut_sensitivity: Optional[CutSensitivity] = None

    @field_validator("primary_axis", "cut_sensitivity", mode="before")
    @classmethod
    def _coerce_edit_enums(cls, v, info):
        return _coerce_enum({"primary_axis": PrimaryAxis, "cut_sensitivity": CutSensitivity}[info.field_name])(v)
    best_use: List[str] = Field(
        default_factory=list,
        description="how an editor would reach for this clip, e.g. ['establishing', 'reaction insert', 'soundbite']",
    )


# --------------------------------------------------------------------------
# Camera craft (timeline -- a single take can still pan, push, reframe)
# --------------------------------------------------------------------------

class CameraSpan(BaseModel):
    start_ms: int
    end_ms: int
    shot_size: Optional[ShotSize] = None
    angle: Optional[CameraAngle] = None
    movement: Optional[CameraMovement] = None
    subject_focus: Optional[str] = Field(None, description="what the framing favors, e.g. 'p1 face', 'the car', 'horizon'")
    is_deliberate: Optional[bool] = Field(
        None, description="True for an intentional move (push-in, planned pan); False for incidental wobble"
    )

    @field_validator("shot_size", "angle", "movement", mode="before")
    @classmethod
    def _coerce_camera_enums(cls, v, info):
        return _coerce_enum({"shot_size": ShotSize, "angle": CameraAngle, "movement": CameraMovement}[info.field_name])(v)


# --------------------------------------------------------------------------
# Persons (optional module -- empty when no people)
# --------------------------------------------------------------------------

class PersonDurable(BaseModel):
    """Traits that survive across clips -- the cross-video matching key. Describe
    what you can SEE; omit (leave null) what you genuinely can't tell."""
    gender_presentation: Optional[str] = None
    age_band: Optional[str] = Field(None, description="e.g. 'child', 'teen', '20s', '30s', '40s', '50s+'")
    skin_tone: Optional[str] = None
    build: Optional[str] = Field(None, description="e.g. 'slim', 'average', 'broad', 'tall'")
    face_shape: Optional[str] = None
    hair_color: Optional[str] = None
    eye_color: Optional[str] = None
    distinctive_marks: List[str] = Field(
        default_factory=list,
        description="the strongest re-id signal: moles, scars, tattoos, freckles, dimples, gap teeth, etc.",
    )


class PersonMutable(BaseModel):
    """Session-specific appearance -- useful for this edit, useless for re-id."""
    hair_style: Optional[str] = None
    facial_hair: Optional[str] = None
    glasses: Optional[str] = None
    headwear: Optional[str] = None
    wardrobe: List[str] = Field(default_factory=list, description="visible clothing, top to bottom")
    accessories: List[str] = Field(default_factory=list)


class Person(BaseModel):
    local_id: str = Field(description="clip-local id: 'p1', 'p2', ... referenced by events/interactions")
    canonical_description: Optional[str] = Field(
        None, description="one-line natural-language identikit a human could use to pick this person out"
    )
    role: Optional[str] = Field(None, description="e.g. 'main subject', 'interviewer', 'passerby'")
    durable: Optional[PersonDurable] = None
    mutable: Optional[PersonMutable] = None
    enters_ms: Optional[int] = None
    exits_ms: Optional[int] = None
    best_face_ms: Optional[int] = Field(None, description="timestamp of the clearest, most front-on look at the face")
    frame_region: Optional[Region] = Field(
        None, description="where this person typically sits in the frame (coarse box around their head/torso); used to keep them in frame when reframing"
    )
    screen_time_note: Optional[str] = None
    # NOTE: filled in post-hoc by audio/visual fusion against L1 diarization.
    # The model should leave this null.
    voice_speaker_id: Optional[str] = None
    av_link_confidence: Optional[float] = None


# --------------------------------------------------------------------------
# Temporal tracks (sparse, timestamped events)
# --------------------------------------------------------------------------

class GazeSpan(BaseModel):
    start_ms: int
    end_ms: int
    subject: str = Field(description="person local_id")
    direction: Optional[GazeDirection] = None
    target: Optional[str] = Field(None, description="person local_id or object the gaze lands on")


class SpeakingSpan(BaseModel):
    """When a visible person is clearly speaking on camera (mouth moving in
    speech). This is the VLM's *independent* observation -- it is never told the
    diarization answer. Code later intersects these spans with L1 voice activity
    to derive the audio<->visual identity link, so the worst case is 'no link',
    never a hallucinated one."""
    start_ms: int
    end_ms: int
    subject: str = Field(description="person local_id who is visibly speaking")
    region: Optional[Region] = Field(
        None, description="where the speaker is in the frame while speaking (coarse box around their head); used to keep them in frame when reframing"
    )


# --------------------------------------------------------------------------
# Take selection (span-level): localized quality + retries
#
# Quality is NEVER a clip-level scalar. It is a property of a SPAN of content.
# The VLM localizes its quality judgements in time exactly like every other
# track here, so downstream code can compare any part of one clip to any part
# of another (across clips OR within one clip) without a second VLM pass.
# --------------------------------------------------------------------------

class QualityDimension(str, Enum):
    energy = "energy"            # performance energy / engagement
    fluency = "fluency"         # smoothness of delivery (stumbles, restarts)
    naturalness = "naturalness"  # natural vs awkward/stiff/over-rehearsed
    technical = "technical"      # framing/focus/visible craft issues you can see


class TakeQualityEvent(BaseModel):
    """A localized, rubric-anchored quality judgement over a span. Emit these
    where quality is notable (good OR bad); they overlap content_units and let
    code score any window. Anchor scores to the rubric you are given; prefer
    citing concrete evidence over a bare number."""
    start_ms: int
    end_ms: int
    dimension: QualityDimension
    score: int = Field(description="1=poor .. 3=acceptable .. 5=excellent")
    evidence: Optional[str] = Field(None, description="what you observed, concretely")

    @field_validator("score", mode="before")
    @classmethod
    def _clamp_score(cls, v):
        try:
            return max(1, min(5, int(round(float(v)))))
        except (TypeError, ValueError):
            return 3


class RestartMarker(BaseModel):
    """A point where a take is abandoned and re-attempted within THIS clip
    (a flub + retry). Splits one clip into multiple attempts of the same
    content."""
    ms: int = Field(description="where the restart happens (start of the retry)")
    cue: Optional[str] = Field(None, description="verbatim cue if any, e.g. 'sorry, let me redo that'")
    restarts_unit: Optional[str] = Field(
        None, description="unit_id this is another attempt of, when identifiable"
    )


# --------------------------------------------------------------------------
# v6 cuts-v2: capture ATOMS (detection only)
# --------------------------------------------------------------------------

class CaptureAtom(BaseModel):
    """One thing the camera captured on a single video CHANNEL -- detection, not
    judgment. No editorial bucket, no role, no relation. Emit a DONE atom for a
    physical action / change over time (a kick, a pour, a screen-rec UI changing)
    and a SHOWN atom for a held subject worth seeing (a face, the product, the
    scenery, a static title card). Tag the SUBJECT (person/place/object/graphic).
    Keep recall high; a `confidence` gate trims downstream.

    Audio channels are NOT emitted here: Said comes from the transcript, Heard
    from the audio envelope."""
    id: Optional[str] = Field(None, description="clip-local id ('a1', ...)")
    channel: Optional[Channel] = Field(
        None,
        description=(
            "'done' = an action / change unfolding over time (peak = the impact "
            "instant); 'shown' = a held subject to look at (peak = the clearest "
            "representative frame). A screen-rec demo whose UI is changing is "
            "'done'; a static chart/title is 'shown'."
        ),
    )
    subject: Optional[Subject] = Field(
        None,
        description=(
            "what it is ABOUT: 'person' (a human), 'place' (scenery/setting), "
            "'object' (a thing/product/detail), or 'graphic' (on-screen text/"
            "chart/UI). Orthogonal to the channel."
        ),
    )
    start_ms: int
    end_ms: int
    peak_ms: Optional[int] = Field(
        None, description="the impact (done) or clearest reveal (shown) instant within the span"
    )
    actor: Optional[str] = Field(None, description="person local_id when the subject is a known person")
    label: str = Field(description="short human-facing label for the card, e.g. 'pours coffee', 'mountain vista'")
    summary: Optional[str] = Field(
        None,
        description=(
            "for an information-dense graphic only (slide/chart/list/UI), one "
            "line on what it CONVEYS -- not verbatim OCR. Null otherwise."
        ),
    )
    content_key: Optional[str] = Field(
        None, description="canonical identity of WHAT is delivered, so retakes of the same beat group"
    )
    confidence: Optional[float] = Field(
        None, description="0..1 how sure you are this footage was shot to deliver this (keep recall high)"
    )
    region: Optional[Region] = Field(
        None, description="coarse box of the subject in frame; used to keep it in frame when reframing"
    )

    @field_validator("channel", mode="before")
    @classmethod
    def _norm_channel(cls, v):
        return _coerce_channel(v)

    @field_validator("subject", mode="before")
    @classmethod
    def _norm_subject(cls, v):
        return _coerce_subject(v)

    @field_validator("confidence", mode="before")
    @classmethod
    def _norm_conf(cls, v):
        return _to_unit(v)


# --------------------------------------------------------------------------
# Root artifact
# --------------------------------------------------------------------------

class ClipPerception(BaseModel):
    schema_version: int = SCHEMA_VERSION
    content_type: Optional[ContentType] = None
    frame_orientation: Optional[FrameOrientation] = Field(
        None, description="how the footage must be rotated to sit upright (almost always 'upright'); flag sideways/flipped footage so the editor can correct it"
    )

    valence: Optional[Valence] = Field(
        None,
        description=(
            "the clip's overall EMOTIONAL TONE: positive (upbeat/warm/funny), "
            "neutral (informational), negative (sad/frustrated/critical), tense "
            "(anxious/high-stakes), or somber (solemn/reflective/heavy). Coarse "
            "colour, not a cut decision; null if genuinely unreadable."
        ),
    )

    @field_validator("content_type", "frame_orientation", "valence", mode="before")
    @classmethod
    def _coerce_clip_enums(cls, v, info):
        return _coerce_enum({"content_type": ContentType, "frame_orientation": FrameOrientation, "valence": Valence}[info.field_name])(v)
    logline: Optional[str] = Field(None, description="one sentence: what this clip is and what happens in it")
    synopsis: Optional[str] = Field(None, description="a short chronological paragraph describing the take start to finish")
    topics: List[str] = Field(default_factory=list)

    look: Optional[Look] = None
    setting: Optional[Setting] = None
    editability: Optional[Editability] = None

    camera_craft: List[CameraSpan] = Field(default_factory=list)
    persons: List[Person] = Field(default_factory=list)

    gaze: List[GazeSpan] = Field(default_factory=list)
    speaking: List[SpeakingSpan] = Field(default_factory=list)

    # v6 cuts-v2 substrate: detection-only video-channel atoms (done/shown) the
    # active cut pipeline reads (alongside Said from L1 + Heard from the audio
    # envelope). This is the single capture track the cut builder consumes.
    atoms: List[CaptureAtom] = Field(default_factory=list)

    @field_validator("atoms", mode="before")
    @classmethod
    def _drop_bad_atoms(cls, v):
        """Drop atoms with no usable video channel (e.g. the model put 'said'
        here) rather than failing the whole clip's parse."""
        if not isinstance(v, list):
            return v
        return [a for a in v if isinstance(a, dict) and _coerce_channel(a.get("channel"))]

    # Take selection (span-level): localized quality + retry markers.
    take_quality_events: List[TakeQualityEvent] = Field(default_factory=list)
    restart_markers: List[RestartMarker] = Field(default_factory=list)

    notes: Optional[str] = Field(None, description="caveats, low-confidence calls, anything ambiguous")
