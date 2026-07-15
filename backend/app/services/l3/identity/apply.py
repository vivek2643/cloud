"""
Deterministic identity resolution (voice_first_identity.plan.md Phase D,
voice_id_pass.plan.md Part B). Clusters visible faces into persons
(identity/reconcile.py), then -- given the clip verdicts Track B (ingest.py)
already produced IN PARALLEL with Pass 2 (identity/voice_id.py) -- maps each
"speaking" clip to the global person visible in that cut, tallies votes, and
derives on/off-camera per cut. This module's own work is now instant and
code-only; the model call that used to live here (identity/speaker_pass.py)
moved upstream into Track A/B so it no longer sits on Pass 2's critical path.

Supersedes the old motion-correlation bind (`identity/bind.py`, deleted)
entirely: "the person talking moves more" was a whole-frame proxy that broke
on animated listeners / still talkers.

The single entry point, `run(...)`, does the whole pipeline and returns
`(new_pass2_output, persisted_payload)`. Never fabricates: a voice with no
confident binding leaves its speech cuts' `speaker_person`/`on_camera`
unset (owner unknown), never forced to a guess.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.identity import reconcile, voice_id
from app.services.l3.identity.reconcile import Occurrence
from app.services.l3.lattice import Lattice
from app.services.l3.pass2 import Pass2Cut, Pass2Output

logger = logging.getLogger(__name__)


def _occurrences_from_cuts(cuts: List[Pass2Cut]) -> List[Occurrence]:
    """Every visible-person sighting across the project's cuts -- the
    per-cut-occurrence input identity/reconcile.py's Phase D clustering
    needs (dropping the old one-person-per-FILE assumption)."""
    out: List[Occurrence] = []
    for cut in cuts:
        box = cut.framing.subject_box if cut.framing else None
        area = float(box[2]) * float(box[3]) if box else None
        for i, p in enumerate(cut.people or []):
            out.append(Occurrence(
                file_id=cut.file_id, source_ref=cut.source_ref, person_index=i,
                appearance=(p.get("appearance") or {}), description=p.get("description") or "",
                is_speech_cut=(cut.kind == "speech"), subject_box_area=area,
            ))
    return out


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
    """Per cut: `visible_persons` from Phase D always rides through.
    `speaker_person` is the first of this cut's `voice_ids` (deterministic,
    Pass 1 word-level diarization + voice clustering -- see pass2.
    to_pass2_cuts) that resolved to a confident owner; a video cut (no
    voice_ids) or an unbound voice leaves it None. `on_camera` is then
    PURELY `speaker_person in visible_persons` -- never a guess, and only
    ever set when both facts are actually known."""
    out: List[Pass2Cut] = []
    for cut in cuts:
        vis = visible_persons.get((cut.file_id, cut.source_ref), [])
        speaker_person = None
        for v in cut.voice_ids:
            owner = owner_by_voice.get(v)
            if owner is not None:
                speaker_person = owner
                break
        on_camera = (speaker_person in vis) if speaker_person is not None else None
        out.append(cut.model_copy(update={
            "visible_persons": vis, "speaker_person": speaker_person, "on_camera": on_camera,
        }))
    return out


def run(
    pass2_output: Pass2Output,
    lattices: Dict[str, Lattice],
    voice_of: Dict[Tuple[str, str], str],
    verdicts: List[voice_id.ClipVerdict],
) -> Tuple[Pass2Output, Dict[str, Any]]:
    """The whole Phase D + voice-ID-pass Part B pipeline. `voice_of` is
    identity/voices.assign_voices's output (the full voice roster, used
    here to make sure a voice with zero clip verdicts at all still ends up
    off-camera rather than silently missing). `verdicts` is Track B's
    (ingest.py) cast-blind lip-sync judgments -- already computed in
    PARALLEL with pass 2 (identity/voice_id.py's select_clips ->
    extraction -> run_voice_id_pass) -- this function only does the
    instant, code-only Part B: map each "speaking" clip to the global
    person reconcile finds visible in that cut, tally votes, derive
    on/off-camera."""
    cuts = pass2_output.cuts
    occurrences = _occurrences_from_cuts(cuts)
    face_result = reconcile.reconcile(occurrences)
    visible_persons: Dict[Tuple[str, str], List[str]] = face_result["visible_persons"]
    persons_by_id: Dict[str, Dict[str, Any]] = {p["person_id"]: p for p in face_result["persons"]}

    cuts_by_file: Dict[str, List[Pass2Cut]] = {}
    for c in cuts:
        cuts_by_file.setdefault(c.file_id, []).append(c)

    owner_by_voice, unbound = voice_id.bind_from_verdicts(verdicts, visible_persons, cuts_by_file, lattices)
    all_voices = set(voice_of.values())
    off_camera_voices = (all_voices - set(owner_by_voice.keys())) | unbound

    owned_voices = _owned_voices_by_person(owner_by_voice)
    for pid, vlist in owned_voices.items():
        if pid in persons_by_id:
            persons_by_id[pid]["owned_voices"] = vlist

    new_cuts = _rewrite_cuts(cuts, owner_by_voice, visible_persons)

    payload = {
        "persons": list(persons_by_id.values()),
        "voice_owner": {v: p for v, p in owner_by_voice.items() if p is not None},
        "off_camera_voices": sorted(off_camera_voices),
    }
    if payload["persons"]:
        logger.info("identity: %d person(s), %d voice(s) bound, %d off-camera",
                   len(payload["persons"]), len(payload["voice_owner"]), len(off_camera_voices))
    return Pass2Output(cuts=new_cuts), payload
