"""
Deterministic identity resolution (voice_first_identity.plan.md, Phases D/E/F
orchestrated -- Phase B's voice clustering runs earlier in `ingest.py`,
before pass 2, since `pass2.to_pass2_cuts` needs the global voice map to
backfill `voice_ids`; this module receives it as `voice_of`). Clusters
visible faces into persons (identity/reconcile.py), plans the close-burst
frames that reveal who's speaking (identity/speaker_frames.py), runs the
small per-voice binding pass (identity/speaker_pass.py), then derives
on/off-camera per cut and composes the persisted cast payload.

Supersedes the old motion-correlation bind (`identity/bind.py`, now
deleted) entirely: "the person talking moves more" was a whole-frame
proxy that broke on animated listeners / still talkers. This module never
guesses a talker from motion -- it looks at actual mouth movement in
close, loud, sharp, isolated frames.

The single entry point, `run(...)`, does the whole pipeline and returns
`(new_pass2_output, persisted_payload)`. Never fabricates: a voice with no
confident binding leaves its speech cuts' `speaker_person`/`on_camera`
unset (owner unknown), never forced to a guess.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3 import frames as fr
from app.services.l3.identity import reconcile, speaker_frames, speaker_pass
from app.services.l3.identity.reconcile import Occurrence
from app.services.l3.image_plan import PlannedFrame
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


def _planned_frames_for_bursts(bursts: List[speaker_frames.Burst]) -> List[PlannedFrame]:
    seen: set = set()
    out: List[PlannedFrame] = []
    for b in bursts:
        for ts in b.ts_ms:
            key = (b.file_id, ts)
            if key in seen:
                continue
            seen.add(key)
            out.append(PlannedFrame(file_id=b.file_id, ts_ms=ts, reason="speaker_binding",
                                    ref=b.window_id))
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
    turns_by_file: Dict[str, List[Tuple[int, int, str]]],
    audio_by_file: Dict[str, dict],
    motion_by_file: Dict[str, dict],
    proxy_key_by_file: Dict[str, str],
) -> Tuple[Pass2Output, Dict[str, Any]]:
    """The whole Phase D/E/F pipeline. `voice_of` is identity/voices.
    assign_voices's output, already computed once up front (before pass 2
    ran) and reused here so voice identity is consistent project-wide.
    `turns_by_file` is each file's `Lattice.turns` (re-based onto the
    outlook group's authoritative clock where applicable, same as `voice_
    of`'s own input) -- identity/speaker_frames.py's raw material for
    finding clean speaking windows."""
    cuts = pass2_output.cuts
    occurrences = _occurrences_from_cuts(cuts)
    face_result = reconcile.reconcile(occurrences)
    visible_persons: Dict[Tuple[str, str], List[str]] = face_result["visible_persons"]
    persons_by_id: Dict[str, Dict[str, Any]] = {p["person_id"]: p for p in face_result["persons"]}

    cuts_by_file: Dict[str, List[Pass2Cut]] = {}
    for c in cuts:
        cuts_by_file.setdefault(c.file_id, []).append(c)

    bursts, planning_off_camera = speaker_frames.plan_bursts(
        turns_by_file, voice_of, cuts_by_file, lattices, visible_persons,
        audio_by_file, motion_by_file,
    )

    planned_frames = _planned_frames_for_bursts(bursts)
    images_b64 = fr.extract_for_planned_frames(planned_frames, proxy_key_by_file) if planned_frames else {}

    owner_by_voice = speaker_pass.bind_voices(bursts, images_b64, persons_by_id)
    off_camera_voices = set(planning_off_camera) | {v for v, p in owner_by_voice.items() if p is None}

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
