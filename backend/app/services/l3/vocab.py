"""
The locked editing vocabulary -- the single source of truth for the whole cut
pipeline (L2 perception, L3 assembly, the brain-facing map, the frontend).

The model is LAYERED: the substrate is intrinsic (what the camera/mic captured,
true regardless of what you're making); everything editorial is DERIVED on top.

  * CAPTURE PRIMITIVES -- *what was captured*. The intrinsic, intent-free
    substrate: visual (person / action / place / object / graphic) + audio
    (speech; music/sfx deferred). These describe the frame/track content and
    never depend on the target edit. This is the honest atom of a cut.

  * AFFORDANCES -- *why an editor reaches for a shot* (speech/action/reaction/
    broll/insert). The editor-facing VIEW layer (the UI's filter tabs). Frozen
    at five; a new detector MAPS into one (see ``SOURCE_AFFORDANCE``), never adds
    one. Each affordance maps DOWN to a capture primitive (``AFFORDANCE_PRIMITIVE``)
    -- and two of them (reaction, broll) are *derived views*, not primitives:
    a reaction is a PERSON shot + a `responds_to` relation; b-roll is a
    place/object USED as supplementary footage. Their meaning lives in relations
    and use, not in the capture.

  * RELATIONS -- *how two cuts connect*. A typed, (mostly) directed graph over
    cuts. Extracted at the source (the VLM states them) rather than guessed from
    time overlap, so the brain reasons about real relationships.

  * ROLES -- *what a cut is FOR* in a narrative (hook, answer, ...). Node-level
    intent -- assigned at PLACEMENT (the brain), not baked into the capture.

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


# A physical/visual beat (vs. passive texture). Used to decide what counts as a
# real "moment" beat and what is mere ambience (a listener, an ambient sound).
PHYSICAL_AFFORDANCES: FrozenSet[str] = frozenset(
    {AFF_ACTION, AFF_BROLL, AFF_INSERT}
)

# --------------------------------------------------------------------------
# Relations (closed set of SEVEN typed edges between cuts)
# --------------------------------------------------------------------------
REL_TAKE_OF = "take_of"           # same content, alternate take / angle (pick one)
REL_RESPONDS_TO = "responds_to"   # a reaction <- the line/action that triggered it
REL_ILLUSTRATES = "illustrates"   # a b-roll/insert <- the topic/noun it shows
REL_LEADS_INTO = "leads_into"     # setup -> payoff, windup -> impact
REL_SAME_INSTANT = "same_instant" # simultaneous coverage / angles of one beat
REL_ANSWERS = "answers"           # an answer line <- the question it answers
REL_CONTINUES = "continues"       # next-in-time within one continuous scene

RELATIONS: Tuple[str, ...] = (
    REL_TAKE_OF, REL_RESPONDS_TO, REL_ILLUSTRATES, REL_LEADS_INTO,
    REL_SAME_INSTANT, REL_ANSWERS, REL_CONTINUES,
)
RELATION_SET: FrozenSet[str] = frozenset(RELATIONS)

# Directed edges read "from_id -> to_id" with the meaning below; the rest are
# symmetric (order carries no meaning).
DIRECTED_RELATIONS: FrozenSet[str] = frozenset(
    {REL_RESPONDS_TO, REL_ILLUSTRATES, REL_LEADS_INTO, REL_ANSWERS, REL_CONTINUES}
)
SYMMETRIC_RELATIONS: FrozenSet[str] = frozenset(
    {REL_TAKE_OF, REL_SAME_INSTANT}
)

# Relations whose two endpoints are ALTERNATIVES (place at most one). Everything
# else may legitimately co-exist on the timeline.
ALTERNATIVE_RELATIONS: FrozenSet[str] = frozenset({REL_TAKE_OF})

# Relations that bind cuts into ONE moment cluster (a connected beat the editor
# would treat as a unit). take_of is NOT here -- alternates are one slot, not a
# multi-shot beat.
MOMENT_RELATIONS: FrozenSet[str] = frozenset(
    {REL_RESPONDS_TO, REL_ILLUSTRATES, REL_LEADS_INTO, REL_SAME_INSTANT, REL_ANSWERS}
)

# --------------------------------------------------------------------------
# Roles (node-level narrative intent -- what a cut is FOR)
# --------------------------------------------------------------------------
ROLE_HOOK = "hook"               # the opener that earns the watch
ROLE_ANSWER = "answer"           # the payoff line to a question
ROLE_CTA = "cta"                 # a call to action (ad / promo)
ROLE_ESTABLISHING = "establishing"  # sets place / context
ROLE_CLIMAX = "climax"           # the emotional / narrative peak
ROLE_LISTENER = "listener"       # a held listening / attention shot

ROLES: Tuple[str, ...] = (
    ROLE_HOOK, ROLE_ANSWER, ROLE_CTA, ROLE_ESTABLISHING, ROLE_CLIMAX, ROLE_LISTENER,
)
ROLE_SET: FrozenSet[str] = frozenset(ROLES)


def is_affordance(x: str) -> bool:
    return x in AFFORDANCE_SET


def is_relation(x: str) -> bool:
    return x in RELATION_SET


def is_role(x: str) -> bool:
    return x in ROLE_SET
