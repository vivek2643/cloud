"""
Cuts orchestrator: pass1 -> image_plan -> frame extraction -> pass2
(identity + full visual judgment in one call per batch, batches run
concurrently) -> post -> persist + hero frames. One call per project;
re-runnable (each call is a fresh ``ingest_run`` row -- nothing is patched
in place). See cuts_v3.plan.md section 6 and the build-order table (step E).

Pass 2's batches run IN PARALLEL (ThreadPoolExecutor), not sequentially:
they only share a read-only cached prefix, so nothing stops them firing at
the same time -- this is a pure wall-clock win. Batching is pure size-based
chunking, no co-location constraint at all (pass2_merge.plan.md): the one
thing that used to need cross-cut pixels -- resolving a take group's members
into take/winner/outlook -- is now deterministic code
(``pass2.apply_take_groups``, fed by pass 1's own ``take_candidates``), so
there is no reason for a take's members to share a batch or for images to be
sent to the model more than once.

Identity resolution (asd_identity.plan.md) is entirely code + CV now, no
model call anywhere: L1's ``active_speaker`` pass already detected+tracked+
embedded faces and scored each track's ASD-speaking timeline, once per
file, persisted, and reused here (no per-project API cost, no "Track B"
background thread to launch/join -- that whole voice_id_pass.plan.md
mechanism is gone). ``identity/faces.py`` clusters those tracks into global
persons; ``identity/bind_asd.py`` intersects diarization turns against
ASD-speaking intervals to resolve voice ownership; ``identity/apply.py``
does the final cut rewrite + payload assembly.

THIS MODULE SPENDS REAL MONEY once invoked: ``run_ingest`` makes one real
pass-1 API call and one real pass-2 call per batch (identity resolution is
free -- it only reads already-computed L1 data). Every deterministic step
it wires together (pass1's prompt building, image_plan, pass2's prompt
building, post's assembly) already has its own zero-cost test coverage;
``scripts/test_ingest.py`` mocks every one of those seams so the
ORCHESTRATION ITSELF (stage sequencing, status transitions, error handling)
is verified without spending anything. Actually calling ``run_ingest``
against a real project is a separate, explicit decision.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.services import correlation
from app.services.jobs import app
from app.services.l1 import active_speaker
from app.services.l1.snapshot import build_l1_snapshot
from app.services.l3 import frames as fr
from app.services.l3 import image_plan as ip
from app.services.l3 import ingest_store as store
from app.services.l3 import pass1
from app.services.l3 import pass2
from app.services.l3 import post
from app.services.l3 import v4_segment as v4seg
from app.services.l3.identity import apply as identity_apply
from app.services.l3.identity import bind_asd as identity_bind_asd
from app.services.l3.identity import faces as identity_faces
from app.services.l3.identity import voices as identity_voices
from app.services.l3.lattice import Lattice, resolve_speech_span_ms
from app.services.l3.sync import store as sync_store
from app.services.l3.sync.lattice_merge import apply_outlook_groups, outlook_groups
from app.services.l3.pass2_params import MAX_PARALLEL_PASS2_BATCHES, STILL_WIDTH_PX
from app.services.processing import _download_from_r2, _upload_to_r2

logger = logging.getLogger(__name__)


def _user_id_for_project(project_id: str) -> Optional[str]:
    """Pillar 7: for correlation-scope logging only -- best-effort, never
    raises (an unknown owner just logs as "-")."""
    try:
        with pass1._pg_conn() as conn:
            row = conn.execute(
                "select user_id::text from projects where id = %s", (project_id,)
            ).fetchone()
        return row[0] if row else None
    except Exception:
        return None


def _onsets_for_files(file_ids: List[str]) -> Dict[str, List[int]]:
    """file_id -> its audio_features.onsets_ms, a targeted query rather than
    reuse of build_l1_snapshot -- that function is shared by several
    routers/audit_log consumers that read a summary view (onset_count, not
    the raw list); adding the full onsets list there would bloat every one
    of those payloads for a value only post._salience needs (perception_
    upgrade.plan.md Part D)."""
    if not file_ids:
        return {}
    with pass1._pg_conn() as conn:
        rows = conn.execute(
            "select file_id::text, onsets_ms from audio_features where file_id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()
    return {fid: (onsets or []) for fid, onsets in rows}


def _load_signals(
    file_ids: List[str],
) -> Tuple[Dict[str, dict], Dict[str, dict], Dict[str, list], Dict[str, dict]]:
    motion_by_file: Dict[str, dict] = {}
    scene_by_file: Dict[str, dict] = {}
    silences_by_file: Dict[str, list] = {}
    audio_by_file: Dict[str, dict] = {}
    onsets_by_file = _onsets_for_files(file_ids)
    for fid in file_ids:
        snap = build_l1_snapshot(fid)
        motion_by_file[fid] = snap.get("motion_dynamics") or {}
        scene_by_file[fid] = snap.get("scene_cuts") or {}
        af = snap.get("audio_features") or {}
        silences_by_file[fid] = af.get("silence_intervals") or []
        # rms_db (dB energy envelope) + its hop feed the speech_quality loudness
        # term in post.assemble_cut_records; empty when a clip has no audio_features.
        # onsets_ms feeds post._salience's proximity-bump term (Part D). is_musical
        # gates whether onsets_ms is trusted as an event signal at all (only
        # meaningful for musical clips) -- v4_segment.segment_video's novelty
        # curve reads it the same way (cuts_v4_segmentation.plan.md).
        audio_by_file[fid] = {"rms_db": af.get("rms_db") or [], "hop_ms": af.get("prosody_hop_ms") or 0,
                              "onsets_ms": onsets_by_file.get(fid) or [],
                              "is_musical": bool(af.get("is_musical"))}
    return motion_by_file, scene_by_file, silences_by_file, audio_by_file


def _embeddings_for_files(file_ids: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """file_id -> {local_speaker: embedding} from L1 diarization's captured
    voiceprints (voice_first_identity.plan.md Phase A) -- the cross-clip
    identity spine identity/voices.assign_voices clusters on. A file with no
    embeddings (diarization never ran, or the pyannote build in use
    couldn't produce them) is simply absent -- its speakers fall back to
    unclustered singleton voices, never a hard failure."""
    if not file_ids:
        return {}
    with pass1._pg_conn() as conn:
        rows = conn.execute(
            "select file_id::text, speaker_embeddings from transcripts where file_id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()
    return {fid: (embeddings or {}) for fid, embeddings in rows if embeddings}


def _proxy_keys_for_files(file_ids: List[str]) -> Dict[str, str]:
    with pass1._pg_conn() as conn:
        rows = conn.execute(
            "select id::text, r2_proxy_key from files where id = any(%s::uuid[])", (file_ids,),
        ).fetchall()
    return {fid: key for fid, key in rows if key}


def _face_tracks_for_files(file_ids: List[str]) -> Dict[str, List[active_speaker.FaceTrack]]:
    """file_id -> its L1 active-speaker pass's face tracks (asd_identity.
    plan.md), persisted once per file and reused across every ingest -- no
    per-project compute, just a read. A file with no row yet (the L1 pass
    hasn't run, or found no legible faces) is simply absent -- its cuts
    fall back to id-less PIC/SND, never a hard failure."""
    if not file_ids:
        return {}
    with pass1._pg_conn() as conn:
        rows = conn.execute(
            "select file_id::text, tracks from face_tracks where file_id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()
    return {fid: [active_speaker.FaceTrack.from_dict(t) for t in (tracks or [])] for fid, tracks in rows}


def _extract_and_upload_heroes(records: List[post.CutRecord], record_ids: List[str],
                               proxy_keys: Dict[str, str]) -> None:
    by_file: Dict[str, List[Tuple[str, post.CutRecord]]] = {}
    for rid, rec in zip(record_ids, records):
        by_file.setdefault(rec.file_id, []).append((rid, rec))

    for file_id, items in by_file.items():
        proxy_key = proxy_keys.get(file_id)
        if not proxy_key:
            logger.warning("ingest: no proxy for %s -- skipping %d hero frame(s)", file_id, len(items))
            continue
        with tempfile.TemporaryDirectory() as tmp:
            proxy_path = os.path.join(tmp, "proxy.mp4")
            _download_from_r2(proxy_key, proxy_path)
            for rid, rec in items:
                still_path = os.path.join(tmp, f"{rid}.jpg")
                fr.extract_still(proxy_path, rec.hero_ts_ms, still_path, width=STILL_WIDTH_PX)
                hero_key = f"heroes/{file_id}/{rid}.jpg"
                _upload_to_r2(still_path, hero_key, "image/jpeg")
                store.set_hero_key(rid, hero_key)


def run_ingest(project_id: str) -> str:
    """Run the full cuts ingest for one project. Returns the new
    ingest_run id. On any failure the run is marked ``failed`` with the
    reason and the exception re-raised -- no partial result is ever left
    looking like a success (the plan's "no fallback" rule)."""
    settings = get_settings()
    ingest_run_id = store.create_ingest_run(project_id, settings.ingest_pass1_model, settings.ingest_pass2_model)
    # scale_architecture.plan.md Pillar 7: every log line for this run
    # carries project_id/ingest_run_id/user_id from here on (correlation.
    # scope), not threaded through every logger.info call by hand.
    with correlation.scope(project_id=project_id, ingest_run_id=ingest_run_id,
                            user_id=_user_id_for_project(project_id)):
        return _run_ingest(project_id, settings, ingest_run_id)


def _run_ingest(project_id: str, settings, ingest_run_id: str) -> str:
    # scale_architecture.plan.md Pillar 7: per-stage wall-clock breakdown,
    # persisted on the run row + logged as a one-line scoreboard, win or
    # fail. `t_stage` is a rolling checkpoint reset at each stage boundary.
    t_start = time.monotonic()
    t_stage = t_start
    timings_ms: Dict[str, float] = {}
    try:
        file_rows = pass1.load_project_file_rows(project_id)
        if not file_rows:
            raise ValueError(f"project {project_id} has no ingest-ready files")

        # Pin whatever OUTLOOK groups exist for this project's files RIGHT NOW
        # (a later re-align must not mutate this run) and swap each angle's
        # Lattice speech side onto its group's authoritative source. A project
        # with no declared groups is untouched here (`outlook_by_file` empty ->
        # `file_rows`/hints/ids all pass through as-is): the no-regression
        # guarantee.
        pre_file_ids = [fid for fid, _name, _dur, _lat in file_rows]
        outlook_by_file = sync_store.sync_groups_for_files(pre_file_ids)
        # EVERY member of a declared group is an outlook of the others (a
        # different camera on one moment) -- speech-swapped, beat-mirrored, and
        # grouped as outlooks. Alignment confidence is metadata only, never an
        # exclusion: demoting a low-confidence member to an independent clip is
        # exactly what let pass 1 mis-group it as a TAKE with its own audio.
        groups = outlook_groups(outlook_by_file)
        file_rows, outlook_hints, outlook_file_ids = apply_outlook_groups(file_rows, outlook_by_file, groups)
        outlook_group_by_file = {fid: gid for gid, grp in groups.items() for fid in grp["members"]}

        lattices: Dict[str, Lattice] = {fid: lat for fid, _name, _dur, lat in file_rows}
        file_ids = list(lattices.keys())
        motion_by_file, scene_by_file, silences_by_file, audio_by_file = _load_signals(file_ids)
        proxy_keys = _proxy_keys_for_files(file_ids)

        store.set_status(ingest_run_id, "pass1")
        pass1_completion = pass1.run_pass1(file_rows, outlook_hints)
        pass1_output = pass1.Pass1Output.model_validate(pass1_completion.data)
        # Outlook angles: mirror the authoritative angle's speech beats onto
        # every angle in the group BEFORE enforcement, so each angle carries the
        # identical spans. Enforcement then runs per angle (synced=True) and
        # yields byte-identical final speech cuts.
        pass1_output = pass1.replicate_outlook_speech(pass1_output, lattices, groups)
        # Deterministic boundary repair (split cuts crossing atom-owned gaps,
        # realign take members, and drop any take-candidate that is really one
        # outlook group's angles) -- the model owns meaning, the lattice owns
        # boundaries. The ENFORCED output is what gets persisted.
        pass1_output = pass1.enforce_lattice_partition(
            pass1_output, lattices, outlook_file_ids, outlook_group_by_file)
        # Now that each angle's cuts are aligned, link them into take_candidates
        # -- pass 2a resolves each as an outlook (no winner), feeding
        # footage_map/observe's alt-PIC angle switching.
        pass1_output = pass1.group_outlooks(pass1_output, groups)

        # cuts_v4_only.plan.md: the deterministic, signal-driven segmenter is
        # the ONLY video-cut path now -- pass 1 no longer emits its own video
        # groups at all (see pass1.py, the model prompt only asks for
        # speech). v4_meta_by_ref carries each cut's real span + salience +
        # density to image_plan/identity/post below.
        v4_meta_by_ref: Dict[str, Dict[str, Any]] = {}
        speech_spans_by_file: Dict[str, List[Tuple[int, int]]] = {}
        for sc in pass1_output.speech_cuts:
            lat = lattices.get(sc.file_id)
            if lat is None or not lat.words:
                continue
            s, e = resolve_speech_span_ms(lat.words, lat.atoms, tuple(sc.word_span),
                                          silences_by_file.get(sc.file_id, []))
            if e > s:
                speech_spans_by_file.setdefault(sc.file_id, []).append((s, e))

        video_tentative_groups: List[pass1.VideoTentativeGroup] = []
        for fid in file_ids:
            lat = lattices[fid]
            for vc in v4seg.segment_video(
                file_id=fid, duration_ms=lat.duration_ms,
                speech_spans=sorted(speech_spans_by_file.get(fid, [])),
                motion=motion_by_file.get(fid) or {}, audio=audio_by_file.get(fid) or {},
                scene=scene_by_file.get(fid) or {},
            ):
                gi = len(video_tentative_groups)
                # v4_cuts_as_primitive.plan.md section 3.1: the V4 cut IS
                # the primitive -- no atoms, its own span carried directly
                # on the group (never re-derived from atom membership).
                video_tentative_groups.append(pass1.VideoTentativeGroup(
                    file_id=fid, atom_ids=[], src_in_ms=vc.src_in_ms, src_out_ms=vc.src_out_ms))
                v4_meta_by_ref[f"video_group[{gi}]"] = {
                    "src_in_ms": vc.src_in_ms, "src_out_ms": vc.src_out_ms,
                    "salience": dict(vc.salience), "density": vc.density,
                }
        logger.info("ingest: v4 segmenter produced %d video cut(s) across %d file(s)",
                   len(video_tentative_groups), len(file_ids))
        pass1_output = pass1_output.model_copy(update={"video_tentative_groups": video_tentative_groups})

        store.record_pass1_result(ingest_run_id, pass1_output.model_dump(), pass1_completion.usage,
                                  pass1_output.project_summary)

        # Cross-clip voice clustering (voice_first_identity.plan.md Phase B),
        # computed ONCE here -- after pass 1 (needs its word-level speaker_ids
        # roster) but BEFORE pass 2 (whose to_pass2_cuts backfills voice_ids
        # from this SAME map, so every batch agrees on one global voice
        # identity). `turns_by_file` feeds identity/bind_asd.py later;
        # `groups` is the SAME outlook grouping the speech lattice merge
        # already used, so voice identity and speech boundaries share one
        # notion of "these cameras are one moment."
        embeddings_by_file = _embeddings_for_files(file_ids)
        all_speakers_by_file: Dict[str, List[str]] = {}
        for sc in pass1_output.speech_cuts:
            all_speakers_by_file.setdefault(sc.file_id, []).extend(sc.speaker_ids)
        voice_of = identity_voices.assign_voices(embeddings_by_file, groups, all_speakers_by_file)
        turns_by_file = {fid: lat.turns for fid, lat in lattices.items()}

        timings_ms["pass1"] = (time.monotonic() - t_stage) * 1000
        t_stage = time.monotonic()

        store.set_status(ingest_run_id, "images")
        planned_frames = ip.build_image_plan(pass1_output, lattices, motion_by_file, scene_by_file,
                                             silences_by_file, v4_meta_by_ref=v4_meta_by_ref or None)
        images_b64 = fr.extract_for_planned_frames(planned_frames, proxy_keys)

        timings_ms["extract"] = (time.monotonic() - t_stage) * 1000
        t_stage = time.monotonic()

        # Still reported as "pass2" -- the DB status column's check
        # constraint only knows the original stage name, and adding a new
        # one isn't worth a migration for what's purely a progress label.
        store.set_status(ingest_run_id, "pass2")
        batches = pass2.build_pass2_batches(pass1_output, planned_frames)
        batch_args = []
        for batch_refs in batches:
            ref_set = set(batch_refs)
            batch_frames = [f for f in planned_frames if f.ref in ref_set]
            batch_file_ids = {f.file_id for f in batch_frames}
            batch_file_rows = [row for row in file_rows if row[0] in batch_file_ids]
            batch_args.append((batch_file_rows, batch_frames))

        all_cuts: List[pass2.Pass2Cut] = []
        # gemini_pass2.plan.md P4: a per-run Gemini CachedContent, gated to
        # the gemini provider only. On the default Anthropic path this
        # whole block is skipped -- ingest_gemini (and google-genai) is
        # never imported, `cache_ctx`/`submit_batch` stay the plain
        # no-cache versions, byte-identical to before P4 existed. Cost
        # optimization only; a failed/skipped cache just means every batch
        # runs uncached (see create_pass2_cache's own graceful-degradation
        # contract) -- never a correctness requirement.
        cache_ctx = nullcontext()
        pass2_cache_name = None
        # Pillar 7: wall clock per batch, not just the stage total -- batches
        # run concurrently, so "pass2=Xs" alone hides whether that's one slow
        # batch or a genuinely long stage. `batch_timings_ms.append` is a
        # plain list append from multiple threads, safe under the GIL.
        batch_timings_ms: List[float] = []

        def _timed_pass2_batch(rows, out, frames, imgs):
            t0 = time.monotonic()
            try:
                return pass2.run_pass2_batch(rows, out, frames, imgs)
            finally:
                batch_timings_ms.append((time.monotonic() - t0) * 1000)

        submit_batch = lambda pool, rows, frames: pool.submit(  # noqa: E731
            _timed_pass2_batch, rows, pass1_output, frames, images_b64)

        if settings.ingest_pass2_provider == "gemini":
            from app.services.llm import ingest_gemini as ig
            pass2_cache_name = ig.create_pass2_cache(
                pass2.gemini_system_prompt(), pass1.build_pass1_blocks(file_rows))
            if pass2_cache_name:
                cache_ctx = ig.pass2_cache_scope(pass2_cache_name)
                submit_batch = lambda pool, rows, frames: ig.submit_with_cache_context(  # noqa: E731
                    pool, _timed_pass2_batch, rows, pass1_output, frames, images_b64)

        try:
            # Batches only share a read-only cached prefix -- no reason to
            # make one wait on another's response. Concurrency here is a
            # pure wall-clock win (see MAX_PARALLEL_PASS2_BATCHES); a
            # failure in any batch still propagates and fails the whole
            # run, same as sequential did.
            with cache_ctx, ThreadPoolExecutor(
                    max_workers=min(MAX_PARALLEL_PASS2_BATCHES, len(batch_args)) or 1) as pool:
                futures = [submit_batch(pool, rows, frames) for rows, frames in batch_args]
                for future in futures:
                    completion = future.result()
                    store.accumulate_pass2_usage(ingest_run_id, completion.usage)
                    batch_output = pass2.Pass2BatchOutput.model_validate(completion.data)
                    # locators (word_span/atom_ids) are code-derived from pass 1,
                    # not echoed by the model -- see pass2.backfill_locators.
                    batch_output = pass2.backfill_locators(batch_output, pass1_output)
                    all_cuts.extend(pass2.to_pass2_cuts(batch_output.cuts, pass1_output, voice_of))
        finally:
            if pass2_cache_name:
                from app.services.llm import ingest_gemini as ig
                ig.delete_pass2_cache(pass2_cache_name)

        timings_ms["pass2"] = (time.monotonic() - t_stage) * 1000
        timings_ms["pass2_max_batch"] = max(batch_timings_ms) if batch_timings_ms else 0.0
        t_stage = time.monotonic()

        pass2_output = pass2.Pass2Output(cuts=all_cuts)
        pass2_output = pass2.apply_junk_suspects(pass2_output, pass1_output, lattices)
        # take_group_id/take_role are entirely code-owned now (pass2_merge.
        # plan.md D1/D2): the model never resolves takes, so this is the
        # ONLY place they're ever set -- both declared-outlook members
        # (alternate angles, never retakes) and genuine same-setting takes
        # (post._enforce_take_winner crowns the winner right after post.py's
        # assemble_cut_records call below).
        pass2_output = pass2.apply_take_groups(pass2_output, pass1_output)

        # Deterministic identity resolution (asd_identity.plan.md): cluster
        # this project's files' L1 face tracks into global persons, resolve
        # each cut's visible_persons from those tracks, intersect diarization
        # turns against ASD-speaking intervals to bind voices to persons, then
        # rewrite cuts with speaker_person/on_camera. Must run before
        # assemble_cut_records below -- on_camera also feeds total_quality
        # there, so the rewrite only counts if it lands first. `voice_of` is
        # the SAME voice map pass 2's own voice_ids backfill already used, so
        # identity stays one consistent notion project-wide. No model call
        # anywhere in this block -- everything it reads was already computed
        # (face tracks at L1, cuts by pass 2).
        face_tracks_by_file = _face_tracks_for_files(file_ids)
        track_to_person, persons = identity_faces.cluster(face_tracks_by_file)
        # cuts_v4_only.plan.md: a video cut's real span is the segmenter's
        # own -- see identity/faces._cut_span_ms. `or None` below just means
        # "this project happened to have zero video cuts" (all speech), not
        # a fallback path.
        v4_span_override = {ref: (meta["src_in_ms"], meta["src_out_ms"])
                            for ref, meta in v4_meta_by_ref.items()}
        visible_persons = identity_faces.visible_persons_by_cut(
            track_to_person, face_tracks_by_file, pass2_output.cuts, lattices,
            span_override=v4_span_override or None)
        owner_by_voice, unbound_voices = identity_bind_asd.bind(
            turns_by_file, voice_of, face_tracks_by_file, track_to_person)
        pass2_output, identity_map = identity_apply.run(
            pass2_output, voice_of, persons, visible_persons, owner_by_voice, unbound_voices)
        if identity_map.get("persons"):
            store.set_identity_map(ingest_run_id, identity_map)

        timings_ms["identity"] = (time.monotonic() - t_stage) * 1000
        t_stage = time.monotonic()

        store.set_status(ingest_run_id, "post")
        records = post.assemble_cut_records(pass2_output, lattices, motion_by_file, silences_by_file,
                                            junk_suspects=pass1_output.junk_suspects,
                                            audio_by_file=audio_by_file,
                                            synced_file_ids=outlook_file_ids,
                                            sync_group_by_file=outlook_group_by_file,
                                            sync_info_by_file=outlook_by_file,
                                            v4_meta_by_ref=v4_meta_by_ref or None)

        store.delete_cut_records_for_run(ingest_run_id)
        record_ids = store.insert_cut_records(ingest_run_id, records)
        _extract_and_upload_heroes(records, record_ids, proxy_keys)

        timings_ms["post"] = (time.monotonic() - t_stage) * 1000
        timings_ms["total"] = (time.monotonic() - t_start) * 1000
        store.set_timings(ingest_run_id, timings_ms)
        logger.info(
            "ingest run %s scoreboard: pass1=%.1fs extract=%.1fs pass2=%.1fs(max batch=%.1fs) "
            "identity=%.1fs post=%.1fs total=%.1fs",
            ingest_run_id,
            timings_ms.get("pass1", 0.0) / 1000, timings_ms.get("extract", 0.0) / 1000,
            timings_ms.get("pass2", 0.0) / 1000, timings_ms.get("pass2_max_batch", 0.0) / 1000,
            timings_ms.get("identity", 0.0) / 1000, timings_ms.get("post", 0.0) / 1000,
            timings_ms.get("total", 0.0) / 1000,
        )

        store.set_status(ingest_run_id, "ready")
    except Exception as e:
        logger.exception("ingest run %s failed", ingest_run_id)
        timings_ms["total"] = (time.monotonic() - t_start) * 1000
        try:
            store.set_timings(ingest_run_id, timings_ms)
        except Exception:
            logger.warning("ingest run %s: failed to persist partial timings", ingest_run_id, exc_info=True)
        store.set_status(ingest_run_id, "failed", error=str(e))
        raise
    return ingest_run_id


@app.task(name="l3_cuts_ingest", queue="ingest", retry=False)
def l3_cuts_ingest(project_id: str) -> None:
    run_ingest(project_id)


@app.task(name="l3_cuts_v3_ingest", queue="ingest", retry=False)
def l3_cuts_v3_ingest(project_id: str) -> None:
    """cuts_v4_only.plan.md Phase 3 (risk R1): temporary alias for
    ``l3_cuts_ingest`` under the pre-rename task name, so any job still
    enqueued under the old name from before this deploy keeps running.
    ``defer_ingest`` below only ever enqueues the new name -- remove this
    once the ``ingest`` queue has drained past this release."""
    run_ingest(project_id)


def defer_ingest(project_id: str) -> None:
    """Enqueue a cuts ingest run on the network-bound ``ingest`` procrastinate
    queue (its own worker, decoupled from GPU ingest). Not auto-retried: each
    attempt is a real, costed API call, and a failure here is almost always a
    schema/prompt problem worth looking at, not a transient one."""
    from procrastinate import App, PsycopgConnector

    enqueue_app = App(connector=PsycopgConnector(
        conninfo=get_settings().database_url, min_size=1, max_size=2))
    with enqueue_app.open():
        enqueue_app.configure_task("l3_cuts_ingest", queue="ingest").defer(project_id=project_id)


def run_many(project_ids: List[str], max_workers: int = 4) -> Dict[str, Any]:
    """Run cuts ingest for several projects concurrently. Different
    projects share no state -- separate API calls, separate DB rows,
    separate R2 keys -- so this is a pure wall-clock win with none of the
    prompt-cache tradeoff that parallelizing shards WITHIN one project would
    have (those deliberately stay sequential, see run_ingest). Never raises
    itself: returns {project_id: ingest_run_id} on success or
    {project_id: the raised Exception} on failure, so one project's failure
    doesn't stop the others from being reported."""
    results: Dict[str, Any] = {}
    with ThreadPoolExecutor(max_workers=min(max_workers, len(project_ids))) as pool:
        futures = {pool.submit(run_ingest, pid): pid for pid in project_ids}
        for future in futures:
            project_id = futures[future]
            try:
                results[project_id] = future.result()
            except Exception as e:
                results[project_id] = e
    return results
