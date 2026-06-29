"""
L2 perception artifact schema.

This is the contract for what Gemini returns for one short, single-take clip.
It is passed directly to the SDK as a `response_schema` (so Gemini emits typed
JSON) and re-validated on the way back, so the same definition documents the
prompt, constrains the model, and parses the result.

Design rules baked into the shape (see the perception spec discussion):
  * Universal spine + optional modules. Every clip gets clip-level fields; the
    `persons`/`events`/`reactions`/... tracks are lists that are simply empty
    when they don't apply (a sunset clip has no persons, no speech, no events).
  * Single take. There are no shots/scenes -- the whole clip is one continuous
    camera take, so "structure" is a *timeline of events*, never a cut list.
  * Sparse, timestamped events. Tracks emit events at the moments they happen
    (ms, video-relative); downstream code rasterizes them into the dense 100 ms
    cut-cost grid. The model never emits dense per-frame arrays.
  * Stable local ids. People are `p1`, `p2`, ... within this clip; events and
    interactions reference those ids so actors link to actions. The ids are
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
# `frame_orientation`) used by the editor's auto-reframe/crop. Older v1 rows
# simply lack them and fall back to centered framing.
# v4 adds the typed relation graph (`relations`) plus node-level intent
# (`role`) and grounding (`topic`/`entity`) on the cut-bearing tracks, and gives
# `reactions`/`cutaways` stable local ids so they can be relation endpoints.
# v5 adds the intrinsic CAPTURE PRIMITIVE (`primitive`: person/action/place/
# object/graphic/speech) + a `confidence` on each cut-bearing beat, and a
# `summary` (semantic gist) for information-dense graphics (don't OCR a slide --
# say what it conveys).
# v6 (cuts v2) adds the detection-only `atoms` track: per-beat CHANNEL
# (done/shown) + SUBJECT (person/place/object/graphic) + peak + confidence. This
# is the substrate the v2 cut pipeline reads; the v1 tracks above (events/
# reactions/cutaways/content_units/relations/role) are retained but no longer
# read by the active path.
SCHEMA_VERSION = 6


# The editing vocabulary (vocab.py) is the single source of truth. Render the
# closed relation/role sets as enums so they constrain the model's JSON and
# self-document in the prompt schema, without re-typing the strings here.
RelationType = Enum("RelationType", [(r, r) for r in vocab.RELATIONS], type=str)
Role = Enum("Role", [(r, r) for r in vocab.ROLES], type=str)
# The intrinsic capture substrate the VLM states per cut-bearing beat: what the
# frame/track is about, independent of how it's later used (see vocab.py).
CapturePrimitive = Enum(
    "CapturePrimitive", [(p, p) for p in vocab.CAPTURE_PRIMITIVES], type=str
)
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


def _coerce_primitive(v):
    """Lenient capture primitive: drop an out-of-vocab value to None rather than
    failing the whole clip's parse (the schema is prompt-guided, not constrained)."""
    if v is None:
        return None
    s = str(getattr(v, "value", v)).strip().lower()
    return s if s in vocab.CAPTURE_PRIMITIVE_SET else None


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


def _coerce_role(v):
    """Lenient role: the schema is carried in-prompt (no constrained decoding),
    so the model occasionally emits an out-of-vocab role (a relation name, a
    'question'). Drop it to None rather than failing the WHOLE clip's parse."""
    if v is None:
        return None
    s = str(getattr(v, "value", v)).strip().lower()
    return s if s in vocab.ROLE_SET else None


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


class EventChange(str, Enum):
    """The editorial moment an event marks -- these are the points a cut wants
    to respect (don't cut mid-reveal) or land on (cut on an exit)."""
    enters_frame = "enters_frame"
    exits_frame = "exits_frame"
    action_starts = "action_starts"
    action_peak = "action_peak"
    action_ends = "action_ends"
    holds = "holds"
    reveal = "reveal"
    setup = "setup"


class ReactionType(str, Enum):
    smile = "smile"
    laugh = "laugh"
    surprise = "surprise"
    frown = "frown"
    cry = "cry"
    nod = "nod"
    shake_head = "shake_head"
    eye_widen = "eye_widen"
    eyebrow_raise = "eyebrow_raise"
    look_away = "look_away"
    other = "other"


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


class GraphicKind(str, Enum):
    on_screen_text = "on_screen_text"
    caption = "caption"
    lower_third = "lower_third"
    sign = "sign"
    ui_element = "ui_element"
    logo = "logo"
    other = "other"


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

def _lenient_event_change(v):
    """Coerce an unknown ``change`` value to None instead of failing the whole
    doc. Without constrained decoding the model sometimes puts a cutaway kind
    ('reaction', 'gaze') here; drop it rather than reject the entire perception."""
    if v is None or isinstance(v, EventChange):
        return v
    try:
        return EventChange(v)
    except ValueError:
        return None


class Event(BaseModel):
    """One beat on the timeline. `actor` is optional so the timeline generalizes
    beyond people (a door opening, a car passing). Multi-actor moments are split
    into one event per actor and tied together by a shared `interaction_id`."""
    id: str
    start_ms: int
    end_ms: int
    description: str = Field(description="what happens, concretely: 'p1 opens the car door and steps out'")
    actor: Optional[str] = Field(None, description="person local_id, or null for non-person events")
    target: Optional[str] = Field(None, description="person local_id or object name the action is directed at")
    change: Optional[EventChange] = None

    @field_validator("change", mode="before")
    @classmethod
    def _coerce_change(cls, v):
        return _lenient_event_change(v)
    interaction_id: Optional[str] = Field(None, description="links the per-actor events of one shared moment")
    region: Optional[Region] = Field(
        None, description="where in the frame this beat happens (coarse box); used to reframe onto the action"
    )
    role: Optional[Role] = Field(None, description="narrative intent of this beat, when clear (e.g. 'establishing', 'climax')")
    confidence: Optional[float] = Field(
        None, description="0..1 how sure you are this is a real, usable beat (keep recall high)"
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _norm_conf(cls, v):
        return _to_unit(v)

    @field_validator("role", mode="before")
    @classmethod
    def _norm_role(cls, v):
        return _coerce_role(v)


class Interaction(BaseModel):
    id: str
    start_ms: int
    end_ms: int
    kind: Optional[str] = Field(None, description="e.g. 'conversation', 'handshake', 'hug', 'hand-off'")
    participants: List[str] = Field(default_factory=list, description="person local_ids involved")
    description: Optional[str] = None


class Reaction(BaseModel):
    id: Optional[str] = Field(None, description="clip-local id ('rx1', ...) so a relation can point at this reaction")
    start_ms: int
    end_ms: int
    subject: str = Field(description="person local_id reacting")
    type: Optional[ReactionType] = None
    intensity: Optional[float] = Field(None, description="0..1")
    trigger: Optional[str] = Field(None, description="what prompted it, e.g. 'p2's joke', 'the reveal'")
    role: Optional[Role] = Field(None, description="narrative intent of this beat, when clear (e.g. 'listener', 'climax')")

    @field_validator("intensity", mode="before")
    @classmethod
    def _norm_intensity(cls, v):
        return _to_unit(v)

    @field_validator("role", mode="before")
    @classmethod
    def _norm_role(cls, v):
        return _coerce_role(v)


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


class EnvironmentEvent(BaseModel):
    """Non-person world changes worth a cut point: light shifts, weather,
    something entering/leaving frame, a vehicle passing."""
    start_ms: int
    end_ms: int
    description: str
    change: Optional[EventChange] = None

    @field_validator("change", mode="before")
    @classmethod
    def _coerce_change(cls, v):
        return _lenient_event_change(v)


class GraphicTextEvent(BaseModel):
    start_ms: int
    end_ms: int
    kind: Optional[GraphicKind] = None
    text: Optional[str] = Field(None, description="verbatim text if legible")
    summary: Optional[str] = Field(
        None,
        description=(
            "for an information-dense graphic (slide/chart/list/UI), what it "
            "CONVEYS in one line -- the gist, not a transcription of every word."
        ),
    )


# --------------------------------------------------------------------------
# Editorial cutaways (sparse overlay layer -- the only feed source for
# reactions / b-roll / inserts in the anchor pipeline when populated)
# --------------------------------------------------------------------------

class CutawayAffordance(str, Enum):
    reaction = "reaction"
    broll = "broll"
    insert = "insert"


class CutawayKind(str, Enum):
    reaction = "reaction"
    gaze = "gaze"
    broll_hold = "broll_hold"
    broll_move = "broll_move"
    reveal = "reveal"
    graphic = "graphic"
    environment = "environment"
    interaction = "interaction"


# Map common model drift onto a real cutaway kind so one mislabeled cutaway
# (e.g. kind='action', bleeding from the new `primitive` field) degrades to a
# neutral hold instead of failing the WHOLE clip's parse (recall-first).
_CUTAWAY_KIND_SYNONYM = {
    "action": "broll_move",
    "motion": "broll_move",
    "person": "reaction",
    "place": "broll_hold",
    "object": "broll_hold",
    "broll": "broll_hold",
    "insert": "reveal",
    "speech": "broll_hold",
}


def _coerce_cutaway_kind(v):
    if v is None:
        return v
    s = str(getattr(v, "value", v)).strip().lower()
    if s in {k.value for k in CutawayKind}:
        return s
    return _CUTAWAY_KIND_SYNONYM.get(s, CutawayKind.broll_hold.value)


class CutawayMoment(BaseModel):
    """One overlay moment an editor would cut TO -- not every visible change."""
    id: Optional[str] = Field(None, description="clip-local id ('cx1', ...) so a relation can point at this cutaway")
    start_ms: int
    end_ms: int
    kind: CutawayKind
    affordance: CutawayAffordance
    primitive: Optional[CapturePrimitive] = Field(
        None,
        description=(
            "what is captured here, independent of the affordance: 'person' (a "
            "human shot -- a reaction is a person shot), 'place' (scenery/"
            "establishing), 'object' (a thing/detail), 'graphic' (on-screen text/"
            "chart/reveal), or 'action'. Distinguish place vs object vs person "
            "for b-roll instead of leaving it generic."
        ),
    )
    subject: Optional[str] = Field(None, description="person local_id when relevant")
    label: str = Field(description="short human-facing label for the card")
    summary: Optional[str] = Field(
        None,
        description=(
            "for an information-dense graphic (a slide, chart, list, UI), a short "
            "summary of what it CONVEYS -- not verbatim OCR. Null for plain shots."
        ),
    )
    trigger: Optional[str] = Field(None, description="what prompted a reaction")
    intensity: Optional[float] = Field(None, description="0..1")
    editorial_role: Optional[str] = Field(
        None,
        description="e.g. listener_reaction, establishing, product_reveal",
    )
    role: Optional[Role] = Field(None, description="narrative intent of this beat, when clear (e.g. 'establishing', 'cta')")
    topic: Optional[str] = Field(None, description="the subject/topic this beat is about, for grounding 'illustrates' links")
    entity: Optional[str] = Field(None, description="the concrete thing shown (a noun: 'coffee cup', 'logo', 'mountain')")
    salience_hint: Optional[float] = Field(
        None, description="how cut-worthy (higher = stronger), 0..1"
    )
    confidence: Optional[float] = Field(
        None, description="0..1 how sure you are this is a real, usable cutaway (keep recall high)"
    )
    peak_ms: Optional[int] = Field(None, description="peak frame for reactions")

    @field_validator("intensity", "salience_hint", "confidence", mode="before")
    @classmethod
    def _norm_unit(cls, v):
        return _to_unit(v)

    @field_validator("kind", mode="before")
    @classmethod
    def _norm_kind(cls, v):
        return _coerce_cutaway_kind(v)

    @field_validator("primitive", mode="before")
    @classmethod
    def _norm_primitive(cls, v):
        return _coerce_primitive(v)

    @field_validator("role", mode="before")
    @classmethod
    def _norm_role(cls, v):
        return _coerce_role(v)


# --------------------------------------------------------------------------
# Take selection (span-level): content units + localized quality + retries
#
# Quality is NEVER a clip-level scalar. It is a property of a SPAN of content.
# The VLM localizes its quality judgements in time exactly like every other
# track here, so downstream code can compare any part of one clip to any part
# of another (across clips OR within one clip) without a second VLM pass.
# --------------------------------------------------------------------------

class ContentKind(str, Enum):
    speech = "speech"        # a spoken line / sentence / utterance
    action = "action"        # a physical action beat
    visual = "visual"        # a held composition / b-roll moment
    performance = "performance"


class ContentUnit(BaseModel):
    """One span that delivers ONE unit of content -- the atom of take
    selection. For speech this is a sentence/line; for action, one beat. Two
    deliveries of the SAME content (across clips, or a retry within this clip)
    must share a comparable `content_key` so they can be grouped later."""
    unit_id: str = Field(description="clip-local id: 'u1', 'u2', ...")
    start_ms: int
    end_ms: int
    kind: Optional[ContentKind] = None
    content_key: Optional[str] = Field(
        None,
        description=(
            "Normalized identity of WHAT is delivered, so the same content "
            "matches across takes. For speech: the spoken line, lower-cased, "
            "stripped of fillers/false-starts. For action/visual: a short "
            "canonical description ('p1 pours the coffee')."
        ),
    )
    label: Optional[str] = Field(None, description="short human-facing label")
    primitive: Optional[CapturePrimitive] = Field(
        None,
        description=(
            "what this unit fundamentally IS, captured: 'speech' (a spoken line), "
            "'action' (a physical beat), 'person' (a held person shot), 'place' "
            "(scenery/environment), 'object' (a thing/detail), or 'graphic' "
            "(on-screen text/chart). Independent of how it might later be used."
        ),
    )
    role: Optional[Role] = Field(None, description="narrative intent of this unit, when clear (e.g. 'hook', 'answer', 'cta')")
    topic: Optional[str] = Field(None, description="the subject/topic this unit is about, for grouping and 'illustrates' links")
    entity: Optional[str] = Field(None, description="the concrete thing/person this unit centers on, when there is a clear one")
    confidence: Optional[float] = Field(
        None, description="0..1 how sure you are this is a real, usable beat (keep recall high -- include moderate ones)"
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _norm_conf(cls, v):
        return _to_unit(v)

    @field_validator("primitive", mode="before")
    @classmethod
    def _norm_primitive(cls, v):
        return _coerce_primitive(v)

    @field_validator("role", mode="before")
    @classmethod
    def _norm_role(cls, v):
        return _coerce_role(v)


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
# Typed relation graph (how the logged beats connect)
# --------------------------------------------------------------------------

class Relation(BaseModel):
    """One typed edge between two logged beats. Endpoints reference any id you
    emitted: an event `id`, a `content_units` unit_id, a `cutaways` id, or a
    `reactions` id. This is how downstream editing reasons about real
    relationships (a reaction to a line, b-roll that illustrates a topic, a
    setup that leads into a payoff) instead of guessing from time overlap.

    Directed edges read from_id -> to_id:
      * responds_to  : reaction/answer  ->  the line or action that triggered it
      * illustrates  : a visual/insert  ->  the topic/line/noun it shows
      * leads_into   : a setup/windup   ->  its payoff/impact
      * answers      : an answer line   ->  the question it answers
      * continues    : a beat           ->  the next beat in the same scene
    Symmetric edges (order carries no meaning):
      * take_of      : two deliveries of the SAME content (place at most one)
      * same_instant : two simultaneous coverages/angles of one beat
    """
    type: RelationType
    from_id: str = Field(description="id this edge points FROM (event/unit/cutaway/reaction id)")
    to_id: str = Field(description="id this edge points TO")
    note: Optional[str] = Field(None, description="optional short reason, e.g. 'laughs at the punchline'")


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

    @field_validator("content_type", "frame_orientation", mode="before")
    @classmethod
    def _coerce_clip_enums(cls, v, info):
        return _coerce_enum({"content_type": ContentType, "frame_orientation": FrameOrientation}[info.field_name])(v)
    logline: Optional[str] = Field(None, description="one sentence: what this clip is and what happens in it")
    synopsis: Optional[str] = Field(None, description="a short chronological paragraph describing the take start to finish")
    topics: List[str] = Field(default_factory=list)

    look: Optional[Look] = None
    setting: Optional[Setting] = None
    editability: Optional[Editability] = None

    camera_craft: List[CameraSpan] = Field(default_factory=list)
    persons: List[Person] = Field(default_factory=list)

    events: List[Event] = Field(default_factory=list)
    interactions: List[Interaction] = Field(default_factory=list)
    reactions: List[Reaction] = Field(default_factory=list)
    gaze: List[GazeSpan] = Field(default_factory=list)
    speaking: List[SpeakingSpan] = Field(default_factory=list)
    environment_events: List[EnvironmentEvent] = Field(default_factory=list)
    graphic_text_events: List[GraphicTextEvent] = Field(default_factory=list)

    # Sparse overlay catalog: reactions, b-roll handles, inserts worth cutting to.
    # When non-empty, downstream overlay anchors read ONLY this list.
    cutaways: List[CutawayMoment] = Field(default_factory=list)

    # v6 cuts-v2 substrate: detection-only video-channel atoms (done/shown) the
    # active cut pipeline reads. When present, the v1 cutaways/content_units/
    # events tracks above are ignored by the cut builder.
    atoms: List[CaptureAtom] = Field(default_factory=list)

    @field_validator("atoms", mode="before")
    @classmethod
    def _drop_bad_atoms(cls, v):
        """Drop atoms with no usable video channel (e.g. the model put 'said'
        here) rather than failing the whole clip's parse."""
        if not isinstance(v, list):
            return v
        return [a for a in v if isinstance(a, dict) and _coerce_channel(a.get("channel"))]

    # Take selection (span-level). Empty for clips with no comparable content.
    content_units: List[ContentUnit] = Field(default_factory=list)
    take_quality_events: List[TakeQualityEvent] = Field(default_factory=list)
    restart_markers: List[RestartMarker] = Field(default_factory=list)

    # Typed graph over the logged beats (events / units / cutaways / reactions).
    # Empty when nothing connects (a single static b-roll clip).
    relations: List[Relation] = Field(default_factory=list)

    @field_validator("relations", mode="before")
    @classmethod
    def _drop_bad_relations(cls, v):
        """Keep only well-formed, in-vocab edges -- a stray relation type (the
        model is prompt-guided, not constrained) drops that one edge instead of
        failing the whole clip's parse."""
        if not isinstance(v, list):
            return v
        out = []
        for r in v:
            if not isinstance(r, dict):
                continue
            t = str(r.get("type", "")).strip().lower()
            if t in vocab.RELATION_SET and r.get("from_id") and r.get("to_id"):
                out.append({**r, "type": t})
        return out

    notes: Optional[str] = Field(None, description="caveats, low-confidence calls, anything ambiguous")
