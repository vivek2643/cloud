"""Wire the existing l3 identity reconciliation onto vcut's shared-L1 inputs
(identity_map_vcut.plan.md).

No new identity logic: this module assembles the reconciliation inputs from the
SHARED L1 signals (diarization voiceprints, ASD face tracks, diarization turns,
outlook groups) -- exactly the reads l3/ingest.py:129-167,318-323 do -- and
calls the existing ``l3.identity.apply.run(...)`` UNCHANGED, then persists via
``l3.ingest_store.set_identity_map``. Phase 1 (CAST payload only): the run is
fed an EMPTY cut list, so ``apply.run``'s per-cut rewrite is a no-op and the
payload is byte-shape-identical to the old l3 pipeline (validated read-only in
the plan, §1). Reconciliation for vcut needs L1 alone -- no vcut Pass-2 traits.

Fail-open: any error is logged and swallowed. ``identity_map`` simply stays NULL
(the same byte-identical fallback ``footage_map`` already handles: no CAST line,
PIC/SND carry no ids) and the run stays 'ready' with its cuts untouched.
"""
from __future__ import annotations

import logging
from typing import Dict, List

from app.services.l1 import active_speaker as asd
from app.services.l3 import diarize as l3diarize
from app.services.l3 import ingest_store as l3store
from app.services.l3.identity import apply as identity_apply
from app.services.l3.identity import bind_asd as identity_bind_asd
from app.services.l3.identity import faces as identity_faces
from app.services.l3.identity import voices as identity_voices
from app.services.l3.pass2 import Pass2Output
from app.services.vcut.speech.outlooks import load_sync_groups

logger = logging.getLogger(__name__)


def _pg_conn():
    from app.services import db
    return db.connection()


def _embeddings_for_files(file_ids: List[str]) -> Dict[str, Dict[str, list]]:
    """file_id -> {local_speaker: embedding} from L1 diarization's voiceprints
    (transcripts.speaker_embeddings). Mirrors l3/ingest._embeddings_for_files.
    A file with no embeddings is simply absent -- its speakers fall back to
    unclustered singleton voices, never a hard failure."""
    if not file_ids:
        return {}
    with _pg_conn() as conn:
        rows = conn.execute(
            "select file_id::text, speaker_embeddings from transcripts "
            "where file_id = any(%s::uuid[])", (file_ids,)).fetchall()
    return {fid: (emb or {}) for fid, emb in rows if emb}


def _face_tracks_for_files(file_ids: List[str]) -> Dict[str, List[asd.FaceTrack]]:
    """file_id -> its L1 active-speaker face tracks (face_tracks.tracks).
    Mirrors l3/ingest._face_tracks_for_files. A file with no row (L1 ASD hasn't
    run, or found no legible faces) is simply absent."""
    if not file_ids:
        return {}
    with _pg_conn() as conn:
        rows = conn.execute(
            "select file_id::text, tracks from face_tracks "
            "where file_id = any(%s::uuid[])", (file_ids,)).fetchall()
    return {fid: [asd.FaceTrack.from_dict(t) for t in (tracks or [])]
            for fid, tracks in rows}


def _groups_for_files(file_ids: List[str]) -> Dict[str, dict]:
    """Outlook sync groups shaped for identity/voices (which reads ONLY
    grp["members"], a list of file_ids). Adapts vcut's own load_sync_groups."""
    return {gid: {"members": list(g.members.keys())}
            for gid, g in load_sync_groups(file_ids).items()}


def reconcile_and_store(ingest_run_id: str, file_ids: List[str]) -> None:
    """Compute this run's identity_map from L1 signals (voices/faces/ASD) and
    persist it, exactly like l3/ingest.py:447-450. Fail-open: any error is
    logged and swallowed -- the run stays 'ready' with identity_map NULL, the
    same byte-identical fallback footage_map already handles."""
    try:
        embeddings_by_file = _embeddings_for_files(file_ids)
        face_tracks_by_file = _face_tracks_for_files(file_ids)
        groups = _groups_for_files(file_ids)

        turns_by_file: Dict[str, list] = {}
        all_speakers_by_file: Dict[str, list] = {}
        for fid in file_ids:
            _text, spk_ids, turns = l3diarize.load_turns(fid)
            turns_by_file[fid] = turns
            all_speakers_by_file[fid] = spk_ids

        voice_of = identity_voices.assign_voices(
            embeddings_by_file, groups, all_speakers_by_file)
        track_to_person, persons = identity_faces.cluster(face_tracks_by_file)
        owner_by_voice, unbound = identity_bind_asd.bind(
            turns_by_file, voice_of, face_tracks_by_file, track_to_person)

        # Phase 1: CAST payload only. Empty cuts -> no-op rewrite, exact payload
        # (the payload depends only on persons/owner_by_voice/unbound/voice_of).
        _out, identity_map = identity_apply.run(
            Pass2Output(cuts=[]), voice_of, persons, {}, owner_by_voice, unbound)
        if identity_map.get("persons"):
            l3store.set_identity_map(ingest_run_id, identity_map)
            logger.info("vcut identity: run %s -> %d person(s), %d voice(s) bound",
                        ingest_run_id, len(identity_map["persons"]),
                        len(identity_map["voice_owner"]))
    except Exception:
        logger.exception("vcut identity: reconciliation failed for run %s "
                         "(identity_map left NULL -- cuts unaffected)", ingest_run_id)
