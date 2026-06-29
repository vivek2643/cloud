"""
The locked editing vocabulary -- the single source of truth for the whole cut
pipeline (L2 perception, L3 assembly, the brain-facing map, the frontend).

The model is LAYERED: the substrate is intrinsic (what the camera/mic captured,
true regardless of what you're making); everything editorial is DERIVED on top.

  * CAPTURE PRIMITIVES -- *what was captured*. The intrinsic, intent-free
    substrate: visual (person / action / place / object / graphic) + audio
    (speech; music/sfx deferred). These describe the frame/track content and
    never depend on the target edit. This is the honest atom of a cut.

  * CAPTURE CHANNELS (cuts v2) -- the honest substrate the active pipeline reads:
    SAID / DONE / SHOWN / HEARD, with an orthogonal SUBJECT tag on the video
    channels. Detection only; the editor and a downstream engine decide use.

  * AFFORDANCES -- *why an editor reaches for a shot* (speech/action/reaction/
    broll/insert). The editor-facing VIEW layer (the UI's filter tabs). Frozen
    at five; a new detector MAPS into one (see ``SOURCE_AFFORDANCE``), never adds
    one. Each affordance maps DOWN to a capture primitive (``AFFORDANCE_PRIMITIVE``)
    and onto a v2 channel (``LEGACY_AFFORDANCE_CHANNEL``).

Everything downstream imports these names; nothing redefines them. Pure module
(no deps) so both L2 and L3 can import it freely.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

# --------------------------------------------------------------------------
# Capture primitives -- WHAT was captured (the intrinsic, intent-free substrate)
# --------------------------------------------------------------------------
# Visual: what the frame is *about*.
PRIM_PERSON = "person"     # a human in frame (the subject), whether speaking or not
PRIM_ACTION = "action"     # a physical event / motion / business (something done)
PRIM_PLACE = "place"       # an environment / establishing / scenery
PRIM_OBJECT = "object"     # a thing / detail / close-up
PRIM_GRAPHIC = "graphic"   # on-screen text / title / chart / info (carries a gist)
# Audio: what the track carries. (music / sfx are a deferred pipeline.)
PRIM_SPEECH = "speech"     # dialogue / voiceover

VISUAL_PRIMITIVES: Tuple[str, ...] = (
    PRIM_PERSON, PRIM_ACTION, PRIM_PLACE, PRIM_OBJECT, PRIM_GRAPHIC,
)
AUDIO_PRIMITIVES: Tuple[str, ...] = (PRIM_SPEECH,)
CAPTURE_PRIMITIVES: Tuple[str, ...] = VISUAL_PRIMITIVES + AUDIO_PRIMITIVES
CAPTURE_PRIMITIVE_SET: FrozenSet[str] = frozenset(CAPTURE_PRIMITIVES)

# Derived VIEWS -- not captured things; editor-facing lenses computed from a
# primitive plus a relation/use. Kept for the UI's filter tabs; NEVER primitives.
#   reaction = a PERSON shot + a `responds_to` relation (meaning is relational)
#   broll    = a place/object/person USED as supplementary footage (use, not capture)
#   moment   = a COMPOSITE (a connected cluster of cuts)
VIEW_REACTION = "reaction"
VIEW_BROLL = "broll"
VIEW_MOMENT = "moment"
DERIVED_VIEWS: Tuple[str, ...] = (VIEW_REACTION, VIEW_BROLL, VIEW_MOMENT)
DERIVED_VIEW_SET: FrozenSet[str] = frozenset(DERIVED_VIEWS)


def is_capture_primitive(x: str) -> bool:
    return x in CAPTURE_PRIMITIVE_SET


def is_derived_view(x: str) -> bool:
    return x in DERIVED_VIEW_SET


# --------------------------------------------------------------------------
# v2 CAPTURE CHANNELS -- the honest substrate (supersedes the primitives above).
# --------------------------------------------------------------------------
# A captured shot delivers meaning through exactly one of four channels, derived
# from the two physical tracks (audio|video) x how each carries (event|held):
#
#                 event / language            held / continuous
#   audio   SAID  (speech)                    HEARD (music / sfx / ambient)
#   video   DONE  (action, change over time)  SHOWN (a held subject)
#
# SUBJECT (person/place/object/graphic) is an ORTHOGONAL tag that rides on the
# video channels -- a screen-rec demo is Done.graphic (changing), a title card is
# Shown.graphic (static). This is the single source of truth for v2; the
# affordance/primitive sets above are kept only until the v1 path is deleted.
CHANNEL_SAID = "said"      # audio . event   -- a spoken line / voiceover
CHANNEL_DONE = "done"      # video . event   -- a physical action / change
CHANNEL_SHOWN = "shown"    # video . held    -- a held subject to be seen
CHANNEL_HEARD = "heard"    # audio . held    -- non-speech sound (built, suppressed)

CHANNELS: Tuple[str, ...] = (CHANNEL_SAID, CHANNEL_DONE, CHANNEL_SHOWN, CHANNEL_HEARD)
CHANNEL_SET: FrozenSet[str] = frozenset(CHANNELS)
# What the editor UI and brain actually see. Heard is detected but withheld
# (no real model yet); naming it gives non-speech audio an honest home instead
# of leaking through as a bogus "person" cut.
SURFACED_CHANNELS: Tuple[str, ...] = (CHANNEL_SAID, CHANNEL_DONE, CHANNEL_SHOWN)
SURFACED_CHANNEL_SET: FrozenSet[str] = frozenset(SURFACED_CHANNELS)

CHANNEL_LABEL: Dict[str, str] = {
    CHANNEL_SAID: "said",
    CHANNEL_DONE: "done",
    CHANNEL_SHOWN: "shown",
    CHANNEL_HEARD: "heard",
}

# Subjects -- WHAT a video channel is about (orthogonal tag, not a channel).
SUBJECT_PERSON = "person"
SUBJECT_PLACE = "place"
SUBJECT_OBJECT = "object"
SUBJECT_GRAPHIC = "graphic"
SUBJECTS: Tuple[str, ...] = (SUBJECT_PERSON, SUBJECT_PLACE, SUBJECT_OBJECT, SUBJECT_GRAPHIC)
SUBJECT_SET: FrozenSet[str] = frozenset(SUBJECTS)


def is_channel(x: str) -> bool:
    return x in CHANNEL_SET


def is_surfaced_channel(x: str) -> bool:
    return x in SURFACED_CHANNEL_SET


def is_subject(x: str) -> bool:
    return x in SUBJECT_SET


def channel_label(x: str) -> str:
    return CHANNEL_LABEL.get((x or "").lower(), x or "")


# --------------------------------------------------------------------------
# Affordances (closed set of FIVE -- the editor-facing VIEW layer)
# --------------------------------------------------------------------------
AFF_SPEECH = "speech"      # hear what is said (sync audio is the point)
AFF_ACTION = "action"      # see something done / happen (incl. gestures, business)
AFF_REACTION = "reaction"  # see someone respond / feel / attend (incl. listening)
AFF_BROLL = "broll"        # see a place / thing / texture (establishing, scenery)
AFF_INSERT = "insert"      # register a graphic / reveal / on-screen text

AFFORDANCES: Tuple[str, ...] = (
    AFF_SPEECH, AFF_ACTION, AFF_REACTION, AFF_BROLL, AFF_INSERT,
)
AFFORDANCE_SET: FrozenSet[str] = frozenset(AFFORDANCES)

# Human-facing label per affordance (UI badges / tabs).
AFFORDANCE_LABEL: Dict[str, str] = {
    AFF_SPEECH: "speech",
    AFF_ACTION: "action",
    AFF_REACTION: "reaction",
    AFF_BROLL: "b-roll",
    AFF_INSERT: "insert",
}

# THE rule that stops the list growing: every detection KIND maps into exactly
# one affordance. Add a detector -> add a row here, never a new affordance.
#   - behavior (incidental physical business from the event timeline) IS action.
#   - listening (held attention from the speaking-turn inverse) IS a reaction.
SOURCE_AFFORDANCE: Dict[str, str] = {
    "speech": AFF_SPEECH,
    "action": AFF_ACTION,
    "action_beat": AFF_ACTION,
    "performance": AFF_ACTION,
    "behavior": AFF_ACTION,        # coffee sip / gesture / handling an object
    "reaction": AFF_REACTION,
    "expression": AFF_REACTION,
    "listening": AFF_REACTION,     # deep listening / held attention
    "audio_event": AFF_REACTION,   # off-screen laugh / gasp / applause
    "broll": AFF_BROLL,
    "hold": AFF_BROLL,
    "move": AFF_BROLL,
    "insert": AFF_INSERT,
    "reveal": AFF_INSERT,
    "graphic": AFF_INSERT,
}


def affordance_for(kind: str) -> str:
    """Map a detection ``kind`` to its (closed-set) affordance. Unknown kinds
    fall back to b-roll (a neutral visual), never a new bucket."""
    return SOURCE_AFFORDANCE.get((kind or "").lower(), AFF_BROLL)


# Each editor-facing affordance maps DOWN to the capture primitive beneath it.
# reaction -> person (the reaction-ness is the `responds_to` relation, not the
# capture); broll -> place (coarse; refined to place/object once L2 emits the
# subject); insert -> graphic.
AFFORDANCE_PRIMITIVE: Dict[str, str] = {
    AFF_SPEECH: PRIM_SPEECH,
    AFF_ACTION: PRIM_ACTION,
    AFF_REACTION: PRIM_PERSON,
    AFF_BROLL: PRIM_PLACE,
    AFF_INSERT: PRIM_GRAPHIC,
}


def primitive_for_affordance(aff: str) -> str:
    """The capture primitive beneath an editor-facing affordance."""
    return AFFORDANCE_PRIMITIVE.get((aff or "").lower(), PRIM_PLACE)


def primitives_for(affordances: List[str]) -> List[str]:
    """The distinct capture primitive(s) a cut delivers, from its affordance(s).
    The intrinsic 'what was captured' layer beneath the editorial view(s)."""
    out: List[str] = []
    for a in affordances or ():
        p = primitive_for_affordance(a)
        if p not in out:
            out.append(p)
    return out


# Migration bridge (v1 -> v2): map an old editorial affordance onto its v2
# channel so any cached/legacy cut can still be displayed under the new tabs.
# reaction and b-roll were both "a held subject" (Shown); insert (a graphic) is
# Shown by default. Defined here (after the AFF_* constants) to avoid a forward
# reference.
LEGACY_AFFORDANCE_CHANNEL: Dict[str, str] = {
    AFF_SPEECH: CHANNEL_SAID,
    AFF_ACTION: CHANNEL_DONE,
    AFF_REACTION: CHANNEL_SHOWN,
    AFF_BROLL: CHANNEL_SHOWN,
    AFF_INSERT: CHANNEL_SHOWN,
}


def channel_for_affordance(aff: str) -> str:
    """v1->v2 bridge: the channel an old affordance maps onto (Shown default)."""
    return LEGACY_AFFORDANCE_CHANNEL.get((aff or "").lower(), CHANNEL_SHOWN)


# A physical/visual beat (vs. passive texture). Used to decide what counts as a
# real "moment" beat and what is mere ambience (a listener, an ambient sound).
PHYSICAL_AFFORDANCES: FrozenSet[str] = frozenset(
    {AFF_ACTION, AFF_BROLL, AFF_INSERT}
)

def is_affordance(x: str) -> bool:
    return x in AFFORDANCE_SET
