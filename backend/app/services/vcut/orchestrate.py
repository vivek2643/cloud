"""
seam_cut_pipeline.plan.md section 11 -- orchestration: two procrastinate
tasks on the existing `ingest` queue (``vcut_ingest`` foreground path,
``vcut_enrich`` background), mirroring app.services.l3.ingest's own task/
defer pattern (own fresh App+PsycopgConnector per defer call, ``ingest``
queue, ``retry=False`` -- every attempt is a real, costed API call). Lock
keys are namespaced ``vcut_ingest:<project_id>``/``vcut_enrich:<run_id>``,
distinct from the old pipeline's ``ingest:<project_id>``/
``scene_enrich:<run_id>``, so the two pipelines never lock against each
other for the same project.

Reuses app.services.l3.ingest_store directly (create_ingest_run/set_status/
accumulate_pass2_usage/insert_cut_records -- section 8/10 name these
explicitly) and post.CutRecord via store.py. Nothing else from
app.services.l3 (principle 7).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from app.config import get_settings
from app.services.jobs import app
from app.services.l3 import ingest_store as l3store
from app.services.vcut import pass1 as p1
from app.services.vcut import pass2 as p2
from app.services.vcut import qplan
from app.services.vcut import resolve as rv
from app.services.vcut import sampling
from app.services.vcut import spans as sp
from app.services.vcut import store
from app.services.vcut import subclip
from app.services.vcut.params import DEFAULT_ENERGY, SUBCLIP_MIN_MS, VCUT_VIDEO_WIDTH_PX

logger = logging.getLogger(__name__)


def _pg_conn():
    from app.services import db
    return db.connection()


def _project_file_rows(project_id: str) -> List[Tuple[str, str, int, str]]:
    """(file_id, filename, duration_ms, proxy_key) for every video file in
    this project -- vcut's own minimal loader (no lattice/atoms needed,
    unlike l3.pass1.load_project_file_rows -- principle 7's isolation
    constraint). A file with no proxy at all is silently skipped."""
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select f.id::text, f.filename, coalesce(f.duration_seconds, 0),
                   coalesce(f.r2_proxy_key, f.r2_key)
              from files f
             where f.id = any(select unnest(source_file_ids) from projects where id = %s)
               and f.file_type = 'video'
            """,
            (project_id,),
        ).fetchall()
    return [(fid, name, int(round((dur or 0) * 1000)), proxy_key)
            for fid, name, dur, proxy_key in rows if proxy_key]


def _build_seam_cache(file_ids: List[str]) -> Dict[str, dict]:
    """{file_id: {hop_ms, S, action_energy, frame_diff}} (section 4). The
    action_energy/frame_diff tracks are SeamSignals' own clip-normalized
    act_n/fd_n -- reused as-is, not recomputed a second time. A file whose
    seam computation fails contributes no cuts (best-effort, logged) rather
    than aborting the whole run."""
    from app.services.seam import build_seam_signals, compute_seam_curve

    seam_cache: Dict[str, dict] = {}
    for file_id in file_ids:
        try:
            signals = build_seam_signals(file_id)
            curve = compute_seam_curve(signals)
        except Exception:
            logger.exception("vcut: seam computation failed for %s (no cuts will resolve for it)", file_id)
            continue
        seam_cache[file_id] = {
            "hop_ms": curve.hop_ms, "S": curve.S,
            "action_energy": signals.act_n, "frame_diff": signals.fd_n,
        }
    return seam_cache


def _add_dims_to_seam_cache(seam_cache: Dict[str, dict], proxy_key_by_file: Dict[str, str]) -> None:
    """reframe_vcut_geometry.plan.md section 2: mutate ``seam_cache`` in
    place, adding {src_w, src_h} (the proxy's own probed dims) to each
    file entry that resolved a seam curve. Threading this onto the SAME
    dict store.build_cut_records already reads means zero extra plumbing
    -- seam_cache round-trips through persist_seam_and_plan/load_seam_
    and_plan as-is, so the energy re-resolve path (routers/projects.py)
    gets these for free too. Sequential per file (a natural follow-up to
    parallelize, like _build_seam_cache/sampling.sample_frames_for_files,
    if this proves to be a bottleneck; not done here to keep this change
    minimal). A probe failure just leaves that file's entry without dims
    -- store.py then falls back to an empty framing for its cuts."""
    for file_id, entry in seam_cache.items():
        proxy_key = proxy_key_by_file.get(file_id)
        if not proxy_key:
            continue
        dims = subclip.probe_proxy_dims(proxy_key)
        if dims:
            entry["src_w"], entry["src_h"] = dims


def _words_by_file(prompt_rows: List[Tuple[str, str, int]]) -> Dict[str, List[Tuple[int, int, str]]]:
    """{file_id: [(start_ms, end_ms, text), ...]} -- qplan.py's local
    transcript-context input (section 4.1), read via the SAME loader the
    speech channel itself uses (app.services.vcut.speech.inputs, vcut-
    internal, not L3 -- principle 7's isolation constraint is unaffected).
    A plain read (no model cost); a file with no transcript at all is
    simply absent, same as load_project_inputs' own contract."""
    from app.services.vcut.speech.inputs import load_project_inputs

    inputs_by_file = load_project_inputs(prompt_rows)
    return {
        file_id: [(w.start_ms, w.end_ms, w.text) for w in fi.words]
        for file_id, fi in inputs_by_file.items()
    }


def _create_vcut_cache(
    prompt_rows: List[Tuple[str, str, int]],
    non_speech_by_file: Dict[str, List[Tuple[int, int]]],
    images_by_key: Dict[Tuple[str, int], str],
    settings,
) -> Optional[Dict[str, Any]]:
    """vcut_pass2_rich.plan.md section 3.1: the shared frame CachedContent
    both Pass 1 and Pass 2 ride, created ONCE from every sampled non-speech
    frame. None on failure/no frames -- both passes then fall back to
    uncached (Pass 1: inline frames every call; Pass 2: re-extracted hero
    frames), never a hard failure. Created with vcut_pass1_model (the
    model Pass 1 rides it with); the handle records that model so Pass 2
    can call the SAME model when reusing this cache, even if
    vcut_pass2_model has since been configured differently."""
    if not images_by_key:
        return None
    from datetime import datetime, timezone

    from app.services.llm.ingest_gemini import create_pass2_cache

    blocks = p1.build_frame_blocks(prompt_rows, non_speech_by_file, images_by_key)
    name = create_pass2_cache(
        p1.NEUTRAL_SYSTEM, blocks, model=settings.vcut_pass1_model, ttl_seconds=settings.vcut_cache_ttl_s)
    if not name:
        return None
    return {
        "name": name, "model": settings.vcut_pass1_model,
        "created_at": datetime.now(timezone.utc).isoformat(), "ttl_s": settings.vcut_cache_ttl_s,
    }


def _run_speech_channel(project_id: str, ingest_run_id: str, seam_cache: Dict[str, dict]) -> int:
    """speech_cuts_pipeline.plan.md section 12/13: vcut's own transcript-
    first speech pass. NO FALLBACK -- ANY exception here (segmentation,
    delivery scoring, frame analysis, insert) propagates and fails the whole
    run. A speech-channel bug must surface loudly, never be masked by copying
    a prior run's speech cuts (the old copy_prior stopgap): a broken speech
    pass silently substituting old cuts is exactly the degraded behavior we
    refuse. The video cuts are already inserted by this point; the run's
    outer handler marks it 'failed' on the propagated exception.

    vcut_moment_energy.plan.md section 8: persists which path ran
    (speech_channel_status) -- source='pipeline' on success, source='failed'
    on the propagated error -- so the outcome stays observable in GET /cuts
    and logs."""
    from app.services.vcut.speech.orchestrate import run_speech_channel
    try:
        n = run_speech_channel(project_id, ingest_run_id, seam_cache)
    except Exception as e:
        store.persist_speech_channel_status(ingest_run_id, {"source": "failed", "error": str(e)})
        raise
    store.persist_speech_channel_status(ingest_run_id, {"source": "pipeline", "error": None})
    return n


_VideoCleanupEntry = Tuple[subclip.SubClip, Optional[Dict[str, str]], Optional[str]]


def _prepare_video_inputs(
    file_rows: List[Tuple[str, str, int, str]],
    non_speech_by_file: Dict[str, List[Tuple[int, int]]],
    settings,
) -> Tuple[Dict[str, p1.VideoHandle], List[_VideoCleanupEntry]]:
    """pass1_video_input.plan.md section 7, Phase 1: cut + upload each
    file's non-speech sub-clip, sequentially (a natural follow-up to
    parallelize -- like _build_seam_cache/sampling.sample_frames_for_files
    already do -- if this proves to be a bottleneck; not done here to keep
    this Phase-1 change minimal). vcut_pass2_video_specifics.plan.md section
    3.2: after a successful upload, also create that file's per-file video
    CachedContent (shared by Pass 1 and Pass 2) -- a cache-creation failure
    or below-min-token-floor clip degrades to None (VideoHandle.cache_name),
    never a hard failure; that file's calls just send the video inline
    instead, uncached.

    Returns ({file_id: VideoHandle} for files where BOTH the cut AND the
    upload succeeded) and the full cleanup list (every SubClip actually cut,
    paired with its upload handle and cache name, either of which may be
    None) so the caller can always tear down local temp files + uploaded
    Files API objects + CachedContents regardless of outcome.

    NO FALLBACK: there is no frames substitution any more. A file whose cut
    produced nothing, or whose upload returned None, is simply left ABSENT
    from the returned dict; run_vcut_ingest then hard-fails the whole run on
    any missing file (it names them). An unexpected exception here (ffmpeg
    crash, network error) is no longer swallowed -- it propagates and aborts
    the run, rather than silently degrading that file to sampled frames."""
    from app.services.llm.ingest_gemini import create_video_cache, upload_video

    video_by_file: Dict[str, p1.VideoHandle] = {}
    cleanup: List[_VideoCleanupEntry] = []
    for file_id, _name, _dur, proxy_key in file_rows:
        sub = subclip.cut_non_speech_subclip(
            proxy_key, non_speech_by_file.get(file_id) or [],
            fps=settings.vcut_video_fps, width_px=VCUT_VIDEO_WIDTH_PX,
        )
        if sub is None:
            # Left absent -> run_vcut_ingest's missing-file check hard-fails.
            continue
        handle = upload_video(sub.path)
        cache_name: Optional[str] = None
        if handle is not None:
            cache_name = create_video_cache(
                p1.NEUTRAL_SYSTEM, handle["uri"], model=settings.vcut_pass1_model,
                ttl_seconds=settings.vcut_cache_ttl_s, fps=settings.vcut_video_fps,
            )
        cleanup.append((sub, handle, cache_name))
        if handle is not None:
            video_by_file[file_id] = p1.VideoHandle(
                file_uri=handle["uri"], segments=sub.segments, cache_name=cache_name)
        # handle is None -> file stays absent -> caller hard-fails (clip still
        # tracked in cleanup above so its temp file is torn down).
    return video_by_file, cleanup


def _cleanup_video_inputs(cleanup: List[_VideoCleanupEntry]) -> None:
    """Best-effort teardown of every uploaded Files API object + per-file
    video CachedContent + local temp clip from a video-mode run, called in
    run_vcut_ingest's finally block regardless of success/failure."""
    from app.services.llm.ingest_gemini import delete_pass2_cache, delete_uploaded_file

    for sub, handle, cache_name in cleanup:
        if cache_name is not None:
            delete_pass2_cache(cache_name)
        if handle is not None:
            delete_uploaded_file(handle.get("name"))
        subclip.cleanup_subclip(sub)


def run_vcut_ingest(project_id: str) -> str:
    """The vcut_ingest foreground path (section 11): spans -> sample/cut ->
    (shared frame cache, "frames" mode only) -> Pass 1 -> persist artifacts
    -> resolve at default energy -> write cut_records -> 'ready' -> enqueue
    vcut_enrich. Pass 1's input medium is settings.vcut_pass1_input_mode
    ("frames", default, or "video" -- pass1_video_input.plan.md Phase 1,
    per-file non-speech sub-clips, uploaded Files API handles torn down in
    the finally block below regardless of outcome). On any failure the run
    is marked 'failed' with the reason and the exception re-raised, same
    contract as l3.ingest.run_ingest. THIS FUNCTION SPENDS REAL MONEY: one
    Gemini Pass-1 call per file per project."""
    settings = get_settings()
    ingest_run_id = l3store.create_ingest_run(
        project_id, f"vcut:{settings.vcut_pass1_model}", f"vcut:{settings.vcut_pass2_model}")
    video_cleanup: List[_VideoCleanupEntry] = []
    try:
        file_rows = _project_file_rows(project_id)
        if not file_rows:
            raise ValueError(f"project {project_id} has no video files with a proxy")
        file_ids = [r[0] for r in file_rows]
        proxy_key_by_file = {fid: key for fid, _name, _dur, key in file_rows}
        duration_by_file = {fid: dur for fid, _name, dur, _key in file_rows}
        prompt_rows = [(fid, name, duration_by_file[fid]) for fid, name, _dur, _key in file_rows]

        l3store.set_status(ingest_run_id, "images")
        non_speech_by_file = {fid: sp.non_speech_spans(fid, duration_by_file[fid]) for fid in file_ids}
        seam_cache = _build_seam_cache(file_ids)
        _add_dims_to_seam_cache(seam_cache, proxy_key_by_file)

        video_by_file: Dict[str, p1.VideoHandle] = {}
        images_by_key: Dict[Tuple[str, int], str] = {}
        cache_handle: Optional[Dict[str, Any]] = None
        if settings.vcut_pass1_input_mode == "video":
            video_by_file, video_cleanup = _prepare_video_inputs(file_rows, non_speech_by_file, settings)
            # pass1_video_input.plan.md section 7: no shared Gemini
            # CachedContent in video mode (that's the real, costed resource
            # to drop) -- Pass 2 (Phase 1) falls back to its existing
            # uncached hero-frame re-extraction path; Phase 2 restores a
            # cache via the per-file video cache.
            #
            # NO FALLBACK -- but "no video handle" is NOT always a prep failure.
            # A file with too little non-speech content to cut (all-speech, or
            # a total below SUBCLIP_MIN_MS) legitimately yields no sub-clip: it
            # simply has nothing for Pass 1's VISUAL channel (the speech channel
            # owns it), so it is EXCLUDED from video Pass 1, not failed. Only a
            # file that HAS enough non-speech content yet still produced no
            # handle is a genuine ffmpeg/R2 error -> hard-fail (we still refuse
            # to substitute sampled frames for a broken video input).
            expected_video = [
                fid for fid in file_ids
                if sum(e - s for s, e in (non_speech_by_file.get(fid) or []) if e > s) >= SUBCLIP_MIN_MS
            ]
            missing_file_ids = [fid for fid in expected_video if fid not in video_by_file]
            if missing_file_ids:
                raise RuntimeError(
                    f"vcut video mode: video prep produced no handle for {len(missing_file_ids)}/"
                    f"{len(expected_video)} file(s) with non-speech content {missing_file_ids} "
                    f"-- refusing to substitute sampled frames")
            # Pass 1 (video) runs ONLY on files that actually produced a handle;
            # content-less files would otherwise send an empty visual call.
            pass1_file_rows = [r for r in prompt_rows if r[0] in video_by_file]
        else:
            images_by_key = sampling.sample_frames_for_files(non_speech_by_file, proxy_key_by_file)
            cache_handle = _create_vcut_cache(prompt_rows, non_speech_by_file, images_by_key, settings)
            pass1_file_rows = prompt_rows

        store.persist_vcut_cache(ingest_run_id, cache_handle)
        cache_name = cache_handle["name"] if cache_handle else None

        l3store.set_status(ingest_run_id, "pass1")
        plan, usage = p1.run_pass1(
            pass1_file_rows, non_speech_by_file, images_by_key,
            cached_content=cache_name, video_by_file=video_by_file,
        )
        l3store.accumulate_pass2_usage(ingest_run_id, usage)

        l3store.set_status(ingest_run_id, "post")
        resolved = rv.resolve_cuts(plan, seam_cache, energy=DEFAULT_ENERGY)
        video_ids = store.insert_video_cuts(ingest_run_id, resolved, seam_cache)

        n_speech = _run_speech_channel(project_id, ingest_run_id, seam_cache)

        # vcut_pass2_video_specifics.plan.md section 4/6.1: the question
        # planner (one cheap text-only call) runs for BOTH input modes --
        # it writes question_ids/custom_questions onto the plan's flags,
        # which pass2.run_enrich(_inline) (video, here) or the deferred
        # vcut_enrich task (frames) then answers. Never a hard failure
        # (qplan.plan_questions itself never raises).
        words_by_file = _words_by_file(prompt_rows)
        plan, qplan_usage = qplan.plan_questions(plan, words_by_file)
        if qplan_usage:
            l3store.accumulate_pass2_usage(ingest_run_id, qplan_usage)
        store.persist_seam_and_plan(ingest_run_id, seam_cache, plan.to_dict())

        if settings.vcut_pass1_input_mode == "video":
            # section 6.1: run enrich INLINE, before _cleanup_video_inputs
            # tears the per-file video caches/uploads down below -- a
            # deferred call risks the CachedContent's TTL expiring between
            # passes. Re-resolving + rewriting cut_records here is what
            # makes the shown cuts carry their composed specifics
            # immediately (section 5.4/7.3) instead of waiting on the
            # background vcut_enrich task.
            plan, enrich_usage = p2.run_enrich_inline(plan, video_by_file)
            if enrich_usage:
                l3store.accumulate_pass2_usage(ingest_run_id, enrich_usage)
            store.persist_seam_and_plan(ingest_run_id, seam_cache, plan.to_dict())
            resolved = rv.resolve_cuts(plan, seam_cache, energy=DEFAULT_ENERGY)
            video_ids = store.insert_video_cuts(ingest_run_id, resolved, seam_cache)

        l3store.set_status(ingest_run_id, "ready")
        logger.info("vcut_ingest run %s: %d video cut(s), %d speech cut(s), %d file(s)",
                   ingest_run_id, len(video_ids), n_speech, len(file_ids))
        if settings.vcut_pass1_input_mode != "video":
            # Video mode already enriched inline above -- deferring vcut_
            # enrich here too would redundantly re-answer every flagged
            # moment from hero-still frames (pass2.run_enrich is a FRAMES-
            # mode function now, section 6.1), spending real money twice
            # for no benefit. Frames mode still defers it, unchanged.
            defer_vcut_enrich(project_id, ingest_run_id)
    except Exception as e:
        logger.exception("vcut_ingest run %s failed", ingest_run_id)
        l3store.set_status(ingest_run_id, "failed", error=str(e))
        raise
    finally:
        _cleanup_video_inputs(video_cleanup)
    return ingest_run_id


@app.task(name="vcut_ingest", queue="ingest", retry=False)
def vcut_ingest(project_id: str) -> None:
    run_vcut_ingest(project_id)


@app.task(name="vcut_enrich", queue="ingest", retry=False)
def vcut_enrich(project_id: str, ingest_run_id: str) -> None:
    """Background scene-specificity enrichment (section 10). Fired right
    after run_vcut_ingest sets status 'ready', so cuts are already visible;
    this only sharpens their descriptions afterward. Fail-open: never
    affects whether the run's cuts are usable."""
    from app.services.vcut import pass2 as p2
    try:
        p2.run_enrich(project_id, ingest_run_id)
    except Exception:
        logger.exception("vcut_enrich: enrichment failed for project %s run %s "
                         "(cuts remain usable without specifics)", project_id, ingest_run_id)


def defer_vcut_enrich(project_id: str, ingest_run_id: str) -> None:
    """Best-effort: a failure to enqueue is logged and swallowed -- cuts are
    already visible and usable; missing specifics is a quality gap, never a
    run failure (mirrors l3.ingest.defer_scene_enrich exactly)."""
    try:
        from procrastinate import App, PsycopgConnector

        enqueue_app = App(connector=PsycopgConnector(
            conninfo=get_settings().database_url, min_size=1, max_size=2))
        with enqueue_app.open():
            enqueue_app.configure_task(
                "vcut_enrich", queue="ingest",
                queueing_lock=f"vcut_enrich:{ingest_run_id}",
            ).defer(project_id=project_id, ingest_run_id=ingest_run_id)
    except Exception:
        logger.exception("could not enqueue vcut enrichment for run %s", ingest_run_id)


def defer_vcut_ingest(project_id: str, user_id: str) -> None:
    """Enqueue a vcut ingest run on the shared ``ingest`` procrastinate
    queue. Not auto-retried: each attempt is a real, costed API call.
    ``lock``/``queueing_lock`` are namespaced separately from
    l3.ingest.defer_ingest's ``ingest:<project_id>`` so the two pipelines
    never block each other for the same project."""
    from procrastinate import App, PsycopgConnector

    from app.services import fairness

    priority = fairness.check_ingest_capacity(user_id)

    enqueue_app = App(connector=PsycopgConnector(
        conninfo=get_settings().database_url, min_size=1, max_size=2))
    with enqueue_app.open():
        enqueue_app.configure_task(
            "vcut_ingest", queue="ingest", priority=priority,
            lock=f"vcut_ingest:{project_id}", queueing_lock=f"vcut_ingest:{project_id}",
        ).defer(project_id=project_id)
