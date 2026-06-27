"""
The locked editing vocabulary -- the single source of truth for the whole cut
pipeline (L2 perception, L3 assembly, the brain-facing map, the frontend).

Three closed sets, defined by EDITORIAL MEANING, never by detection method:

  * AFFORDANCES -- *why an editor reaches for a shot*. Frozen at five. A new
    detector (motion, the event timeline, the cutaway track, audio) never adds an
    affordance; it MAPS into one of these (see ``SOURCE_AFFORDANCE``). This is the
    permanent guard against the bucket list ever growing.

  * RELATIONS -- *how two cuts connect*. A typed, (mostly) directed graph over
    cuts. These are extracted at the source (the VLM states them) rather than
    guessed from time overlap, so the brain reasons about real relationships.

  * ROLES -- *what a cut is FOR* in a narrative (hook, answer, ...). Node-level
    intent the brain needs to build structure, not just adjacency.

Everything downstream imports these names; nothing redefines them. Pure module
(no deps) so both L2 and L3 can import it freely.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

# --------------------------------------------------------------------------
# Affordances (closed set of FIVE -- the only editorial buckets that exist)
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
