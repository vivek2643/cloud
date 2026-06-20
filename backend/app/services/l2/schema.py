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
SCHEMA_VERSION = 3


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


class Interaction(BaseModel):
    id: str
    start_ms: int
    end_ms: int
    kind: Optional[str] = Field(None, description="e.g. 'conversation', 'handshake', 'hug', 'hand-off'")
    participants: List[str] = Field(default_factory=list, description="person local_ids involved")
    description: Optional[str] = None


class Reaction(BaseModel):
    start_ms: int
    end_ms: int
    subject: str = Field(description="person local_id reacting")
    type: Optional[ReactionType] = None
    intensity: Optional[float] = Field(None, description="0..1")
    trigger: Optional[str] = Field(None, description="what prompted it, e.g. 'p2's joke', 'the reveal'")

    @field_validator("intensity", mode="before")
    @classmethod
    def _norm_intensity(cls, v):
        return _to_unit(v)


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


class CutawayMoment(BaseModel):
    """One overlay moment an editor would cut TO -- not every visible change."""
    start_ms: int
    end_ms: int
    kind: CutawayKind
    affordance: CutawayAffordance
    subject: Optional[str] = Field(None, description="person local_id when relevant")
    label: str = Field(description="short human-facing label for the card")
    trigger: Optional[str] = Field(None, description="what prompted a reaction")
    intensity: Optional[float] = Field(None, description="0..1")
    editorial_role: Optional[str] = Field(
        None,
        description="e.g. listener_reaction, establishing, product_reveal",
    )
    salience_hint: Optional[float] = Field(
        None, description="how cut-worthy (higher = stronger), 0..1"
    )
    peak_ms: Optional[int] = Field(None, description="peak frame for reactions")

    @field_validator("intensity", "salience_hint", mode="before")
    @classmethod
    def _norm_unit(cls, v):
        return _to_unit(v)


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
# Root artifact
# --------------------------------------------------------------------------

class ClipPerception(BaseModel):
    schema_version: int = SCHEMA_VERSION
    content_type: Optional[ContentType] = None
    frame_orientation: Optional[FrameOrientation] = Field(
        None, description="how the footage must be rotated to sit upright (almost always 'upright'); flag sideways/flipped footage so the editor can correct it"
    )
    logline: Optional[str] = Field(None, description="one sentence: what this clip is and what happens in it")
    synopsis: Optional[str] = Field(None, description="a short chronological paragraph describing the take start to finish")
    topics: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)

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

    # Take selection (span-level). Empty for clips with no comparable content.
    content_units: List[ContentUnit] = Field(default_factory=list)
    take_quality_events: List[TakeQualityEvent] = Field(default_factory=list)
    restart_markers: List[RestartMarker] = Field(default_factory=list)

    notes: Optional[str] = Field(None, description="caveats, low-confidence calls, anything ambiguous")
