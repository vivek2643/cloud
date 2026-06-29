"""
The locked capture vocabulary -- the single source of truth for the whole cut
pipeline (L2 perception, L3 assembly, the brain-facing map, the frontend).

Cuts v2 stands on ONE model: every captured shot delivers meaning through
exactly one of four CHANNELS, derived from the two physical tracks (audio|video)
x how each carries (event|held):

                event / language            held / continuous
  audio   SAID  (speech)                    HEARD (music / sfx / ambient)
  video   DONE  (action, change over time)  SHOWN (a held subject)

SUBJECT (person/place/object/graphic) is an ORTHOGONAL tag that rides on the two
video channels -- a screen-rec demo is Done.graphic (changing), a title card is
Shown.graphic (static). There is no editorial affordance/primitive/role layer:
detection only; the editor and the brain decide use.

Pure module (no deps) so both L2 and L3 can import it freely.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Tuple

# --------------------------------------------------------------------------
# Capture channels -- the honest substrate (the only cut vocabulary).
# --------------------------------------------------------------------------
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
