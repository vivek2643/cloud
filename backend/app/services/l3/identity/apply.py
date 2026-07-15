"""
Deterministic identity resolution -- final assembly step (asd_identity.
plan.md). By the time `run(...)` is called, ingest.py has already done ALL
the real work: `identity/faces.py` clustered face-track embeddings into
global persons + per-cut `visible_persons`, and `identity/bind_asd.py`
intersected diarization turns against ASD-speaking intervals to resolve
`owner_by_voice`. This module is pure assembly -- attach each person's
owned voices, rewrite cuts with `speaker_person`/`on_camera`, and compose
the persisted cast payload. No model call anywhere in this file, no model
call anywhere upstream of it either (ASD_identity.plan.md's whole point:
identity is CV + code now, not a per-project LLM pass).

Supersedes the old motion-correlation bind (`identity/bind.py`, deleted)
and the Gemini-clip-judging bind (`identity/voice_id.py`, deleted) both:
"the person talking moves more" was a whole-frame proxy that broke on
animated listeners / still talkers; a per-clip Gemini call was accurate but
slow, metered, and stochastic. ASD (local, deterministic, cheap) replaces
both.

The single entry point, `run(...)`, returns `(new_pass2_output, persisted_
payload)`. Never fabricates: a voice with no confident binding leaves its
speech cuts' `speaker_person`/`on_camera` unset (owner unknown), never
forced to a guess.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.pass2 import Pass2Cut, Pass2Output

logger = logging.getLogger(__name__)


def _owned_voices_by_person(owner_by_voice: Dict[str, Optional[str]]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for voice, person in owner_by_voice.items():
        if person is not None:
            out.setdefault(person, []).append(voice)
    for vlist in out.values():
        vlist.sort()
    return out


def _rewrite_cuts(
    cuts: List[Pass2Cut],
    owner_by_voice: Dict[str, Optional[str]],
    visible_persons: Dict[Tuple[str, str], List[str]],
) -> List[Pass2Cut]:
    """Per cut: `visible_persons` (identity/faces.py) always rides through.
    `speaker_person` is the owner of this cut's DOMINANT voice --
    `voice_ids[0]`, which pass1/pass2 order by spoken time (see
    pass1._speaker_ids_for_span, pass2.to_pass2_cuts) -- so a beat is credited
    to whoever actually holds the floor, never to a brief interjector who just
    happened to be diarized. We look ONLY at the dominant voice on purpose: if
    it's off-camera/unbound, `speaker_person` stays None (an honest "the person
    speaking here is unknown") rather than falling through to a minor voice's
    owner and mislabelling the beat. A video cut (no voice_ids) also leaves it
    None. `on_camera` is then PURELY `speaker_person in visible_persons` --
    never a guess, and only ever set when both facts are actually known."""
    out: List[Pass2Cut] = []
    for cut in cuts:
        vis = visible_persons.get((cut.file_id, cut.source_ref), [])
        speaker_person = owner_by_voice.get(cut.voice_ids[0]) if cut.voice_ids else None
        on_camera = (speaker_person in vis) if speaker_person is not None else None
        out.append(cut.model_copy(update={
            "visible_persons": vis, "speaker_person": speaker_person, "on_camera": on_camera,
        }))
    return out


def run(
    pass2_output: Pass2Output,
    voice_of: Dict[Tuple[str, str], str],
    persons: Dict[str, Dict[str, Any]],
    visible_persons: Dict[Tuple[str, str], List[str]],
    owner_by_voice: Dict[str, Optional[str]],
    unbound_voices: set,
) -> Tuple[Pass2Output, Dict[str, Any]]:
    """Final assembly. `voice_of` is identity/voices.assign_voices's output
    (the full voice roster, used here to make sure a voice with zero ASD
    votes at all still ends up off-camera rather than silently missing).
    `persons`/`visible_persons` are identity/faces.py's clustering output;
    `owner_by_voice`/`unbound_voices` are identity/bind_asd.bind's output.
    `persons` is mutated in place (owned_voices attached) and also returned
    inside the payload -- the caller doesn't need it back separately."""
    cuts = pass2_output.cuts
    all_voices = set(voice_of.values())
    off_camera_voices = (all_voices - set(owner_by_voice.keys())) | unbound_voices

    owned_voices = _owned_voices_by_person(owner_by_voice)
    for pid, vlist in owned_voices.items():
        if pid in persons:
            persons[pid]["owned_voices"] = vlist

    new_cuts = _rewrite_cuts(cuts, owner_by_voice, visible_persons)

    payload = {
        "persons": list(persons.values()),
        "voice_owner": {v: p for v, p in owner_by_voice.items() if p is not None},
        "off_camera_voices": sorted(off_camera_voices),
    }
    if payload["persons"]:
        logger.info("identity: %d person(s), %d voice(s) bound, %d off-camera",
                   len(payload["persons"]), len(payload["voice_owner"]), len(off_camera_voices))
    return Pass2Output(cuts=new_cuts), payload
