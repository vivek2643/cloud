"""
L2 orchestrator: the Gemini perception pass for one short clip.

Flow (one procrastinate task per file):
  1. Gate: skip if disabled, no API key, or the clip is longer than the limit.
  2. Pull the L1 transcript + diarization as timing scaffolding.
  3. Download the 1080p proxy (cheap, plenty for perception).
  4. One JSON Gemini call (prompt-carried schema, validated client-side) -> ClipPerception.
  5. Fuse: map visual person ids <-> diarization speaker ids by overlap.
  6. Persist to clip_perception + flip files.l2_status.

L2 is its own task (not an L1 stage) so a Gemini hiccup retries independently
and never blocks or fails the L1 index. It is enqueued by L1 on completion.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import os
import tempfile
import traceback
from typing import Dict, List, Optional, Tuple

import psycopg
from procrastinate import RetryStrategy

from app.config import get_settings
from app.services.jobs import app
from app.services.l2 import gemini_video, prompt as l2_prompt
from app.services.l2.schema import ClipPerception, SCHEMA_VERSION
from app.services.processing import _download_from_r2
from app.services.supabase_client import get_supabase

logger = logging.getLogger(__name__)

STAGE = "l2_perception"

# Merge consecutive same-speaker words into one turn unless the silence between
# them exceeds this (likely a real handover or pause).
_TURN_GAP_MS = 800
# Don't blow up the prompt on a chatty 5-minute clip.
_MAX_TRANSCRIPT_CHARS = 24000
# Below this overlap fraction we don't trust a person<->voice link.
_AV_LINK_MIN_CONFIDENCE = 0.45


# --------------------------------------------------------------------------
# DB helpers (small, local -- avoids importing the heavy L1 pipeline module)
# --------------------------------------------------------------------------

def _pg_conn() -> psycopg.Connection:
    settings = get_settings()
    return psycopg.connect(settings.database_url, autocommit=True)


def _set_l2_status(file_id: str, status: str) -> None:
    try:
        get_supabase().table("files").update({"l2_status": status}).eq("id", file_id).execute()
    except Exception:
        logger.exception("L2: failed to set l2_status=%s for %s", status, file_id)


def _stage_begin(conn: psycopg.Connection, file_id: str) -> None:
    conn.execute(
        """
        insert into processing_jobs (file_id, stage, status, started_at, attempts)
        values (%s, %s, 'running', now(), 1)
        on conflict (file_id, stage) do update set
            status = 'running', started_at = now(),
            attempts = processing_jobs.attempts + 1, error = null
        """,
        (file_id, STAGE),
    )


def _stage_done(conn: psycopg.Connection, file_id: str) -> None:
    conn.execute(
        "update processing_jobs set status='done', finished_at=now(), error=null "
        "where file_id=%s and stage=%s",
        (file_id, STAGE),
    )


def _stage_fail(conn: psycopg.Connection, file_id: str, err: str) -> None:
    conn.execute(
        "update processing_jobs set status='failed', finished_at=now(), error=%s "
        "where file_id=%s and stage=%s",
        (err[:8000], file_id, STAGE),
    )


def _file_meta(file_id: str) -> Optional[Tuple[str, str, float]]:
    """(r2_key_to_use, mime, duration_seconds) or None if the file is gone.

    Prefers the client analysis proxy A (480p@1fps + audio -- purpose-built for
    perception and available seconds after upload starts), then the 1080p
    editing proxy, then the raw upload. Perception at 1fps is intentional: the
    VLM samples frames, not motion, so the tiny proxy is plenty and lets L2 run
    without waiting on the multi-GB raw.
    """
    with _pg_conn() as conn:
        row = conn.execute(
            "select r2_key, r2_proxy_a_key, r2_proxy_key, duration_seconds from files where id = %s",
            (file_id,),
        ).fetchone()
    if not row:
        return None
    raw_key, proxy_a_key, proxy_key, duration = row
    analysis_key = proxy_a_key or proxy_key
    key = analysis_key or raw_key
    if not key:
        return None
    if analysis_key:
        mime = "video/mp4"  # proxies are always normalized H.264/mp4
    else:
        guessed, _ = mimetypes.guess_type(key)
        mime = guessed if (guessed or "").startswith("video/") else "video/mp4"
    return key, mime, float(duration or 0.0)


# --------------------------------------------------------------------------
# L1 context: transcript text + speaker turns (for grounding + AV fusion)
# --------------------------------------------------------------------------

def _load_transcript_context(
    file_id: str,
) -> Tuple[Optional[str], List[str], List[Tuple[int, int, str]]]:
    """Return (transcript_text, speaker_ids, speaker_turns).

    speaker_turns are (start_ms, end_ms, speaker) merged from diarized words and
    are reused both for the prompt and for audio/visual fusion. The merge lives
    in the shared `l3.diarize` leaf so L3 angle routing uses the identical turns.
    """
    from app.services.l3.diarize import load_turns

    return load_turns(
        file_id, turn_gap_ms=_TURN_GAP_MS, max_chars=_MAX_TRANSCRIPT_CHARS
    )


def _load_editorial_context(file_id: str, duration_s: float) -> dict:
    """L1 dialogue counts passed to the VLM so cutaway density scales with beats."""
    ctx: dict = {"duration_ms": int(duration_s * 1000), "sentence_count": 0, "topic_count": 0}
    try:
        with _pg_conn() as conn:
            row = conn.execute(
                "select segments from dialogue_segments where file_id = %s", (file_id,)
            ).fetchone()
        if row and row[0]:
            doc = row[0] if isinstance(row[0], dict) else json.loads(row[0])
            ctx["sentence_count"] = len(doc.get("sentence") or [])
            ctx["topic_count"] = len(doc.get("topic") or [])
    except Exception:
        logger.exception("L2: dialogue context unavailable for %s", file_id)
    return ctx


# --------------------------------------------------------------------------
# Audio/visual speaker fusion
# --------------------------------------------------------------------------

def _overlap_ms(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def _fuse_speakers(
    perception: ClipPerception, speaker_turns: List[Tuple[int, int, str]]
) -> None:
    """Bridge the two *independent* identity spaces -- VLM person ids (p1, p2)
    and diarization speaker ids (S0, S1) -- entirely in code, AFTER both ran.

    The VLM was never shown the diarization answer; it only logged, as a visual
    observation, when each person was visibly speaking. We intersect those spans
    with L1's voice-activity turns and attach the best-overlapping speaker to
    each person. Because the two signals are independent, the failure mode is a
    *missing* link (low overlap -> nothing written), never a hallucinated one.

    Mutates `perception` in place, setting voice_speaker_id + av_link_confidence
    on persons that clear the confidence floor.
    """
    if not speaker_turns or not perception.persons or not perception.speaking:
        return

    # Group the VLM's on-camera speaking spans by visual person.
    speaking_by_person: Dict[str, List[Tuple[int, int]]] = {}
    for s in perception.speaking:
        speaking_by_person.setdefault(s.subject, []).append((s.start_ms, s.end_ms))

    for person in perception.persons:
        spans = speaking_by_person.get(person.local_id)
        if not spans:
            continue
        total = sum(max(0, e - s) for s, e in spans) or 1
        scores: Dict[str, int] = {}
        for span in spans:
            for t_start, t_end, spk in speaker_turns:
                scores[spk] = scores.get(spk, 0) + _overlap_ms(span, (t_start, t_end))
        if not scores:
            continue
        best_spk, best_overlap = max(scores.items(), key=lambda kv: kv[1])
        confidence = round(best_overlap / total, 3)
        if confidence >= _AV_LINK_MIN_CONFIDENCE:
            person.voice_speaker_id = best_spk
            person.av_link_confidence = confidence


def _flag_offscreen_dialogue(file_id: str, perception: ClipPerception) -> None:
    """Mark Dialogues-lens segments whose diarized speaker never links to a
    VISIBLE on-camera person as `offscreen` (the off-screen crew/director).

    Runs after `_fuse_speakers`, so we know which `S#` ids belong to people the
    VLM actually saw speaking. We only act when there is at least one confident
    on-camera link -- otherwise we have no basis to call anyone off-camera and
    leave the lens untouched (fail to a missing flag, never a wrong one). The
    flag is additive and non-destructive; the UI hides flagged clips by default.
    """
    oncam = {
        p.voice_speaker_id
        for p in perception.persons
        if p.voice_speaker_id and (p.av_link_confidence or 0.0) >= _AV_LINK_MIN_CONFIDENCE
    }
    if not oncam:
        return

    with _pg_conn() as conn:
        row = conn.execute(
            "select segments from dialogue_segments where file_id = %s", (file_id,)
        ).fetchone()
        if not row or not row[0]:
            return
        doc = row[0] if isinstance(row[0], dict) else json.loads(row[0])
        changed = False
        for level in ("sentence", "topic"):
            for seg in doc.get(level) or []:
                spk = seg.get("speaker")
                if spk and spk not in oncam:
                    flags = seg.setdefault("flags", [])
                    if "offscreen" not in flags:
                        flags.append("offscreen")
                        changed = True
        if changed:
            conn.execute(
                "update dialogue_segments set segments = %s::jsonb where file_id = %s",
                (json.dumps(doc), file_id),
            )
            logger.info("L2: flagged off-camera dialogue for %s (on-camera=%s)", file_id, sorted(oncam))


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------

def _persist(file_id: str, perception: ClipPerception, usage: dict, model: str) -> None:
    doc = perception.model_dump(mode="json")
    with _pg_conn() as conn:
        conn.execute(
            """
            insert into clip_perception (file_id, schema_version, model, perception, usage, created_at)
            values (%s, %s, %s, %s::jsonb, %s::jsonb, now())
            on conflict (file_id) do update set
                schema_version = excluded.schema_version,
                model = excluded.model,
                perception = excluded.perception,
                usage = excluded.usage,
                created_at = now()
            """,
            (file_id, SCHEMA_VERSION, model, json.dumps(doc), json.dumps(usage or {})),
        )


def _persist_unparsed(file_id: str, raw_text: str, usage: dict, model: str) -> None:
    """Best-effort: keep the raw model output so a bad parse is debuggable."""
    with _pg_conn() as conn:
        conn.execute(
            """
            insert into clip_perception (file_id, schema_version, model, perception, usage, created_at)
            values (%s, %s, %s, %s::jsonb, %s::jsonb, now())
            on conflict (file_id) do update set
                schema_version = excluded.schema_version,
                model = excluded.model,
                perception = excluded.perception,
                usage = excluded.usage,
                created_at = now()
            """,
            (
                file_id,
                SCHEMA_VERSION,
                model,
                json.dumps({"_parse_error": True, "_raw_text": raw_text[:200000]}),
                json.dumps(usage or {}),
            ),
        )


# --------------------------------------------------------------------------
# Top-level procrastinate task
# --------------------------------------------------------------------------

@app.task(name="l2_perception", queue="l2", retry=RetryStrategy(max_attempts=2, exponential_wait=8))
def l2_perception(file_id: str) -> None:
    settings = get_settings()

    if not settings.enable_l2_perception:
        logger.info("L2 disabled; skipping %s", file_id)
        return
    if not settings.gemini_api_key:
        logger.info("L2: no GEMINI_API_KEY; skipping %s", file_id)
        return

    meta = _file_meta(file_id)
    if meta is None:
        logger.info("L2: file %s gone; skipping.", file_id)
        return
    r2_key, mime, duration_s = meta

    if duration_s <= 0 or duration_s > settings.l2_max_duration_seconds:
        logger.info(
            "L2: %s is %.1fs (limit %ds); skipping deep perception.",
            file_id, duration_s, settings.l2_max_duration_seconds,
        )
        _set_l2_status(file_id, "skipped")
        return

    _set_l2_status(file_id, "running")
    with _pg_conn() as conn:
        _stage_begin(conn, file_id)

    try:
        transcript_text, speaker_ids, speaker_turns = _load_transcript_context(file_id)
        editorial_context = _load_editorial_context(file_id, duration_s)
        user_prompt = l2_prompt.build_user_prompt(
            duration_seconds=duration_s,
            transcript_text=transcript_text,
            speaker_ids=speaker_ids,
            editorial_context=editorial_context,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            local = os.path.join(tmpdir, "clip.mp4")
            logger.info("L2: downloading %s for file %s", r2_key, file_id)
            _download_from_r2(r2_key, local)

            result = gemini_video.analyze_video(
                video_path=local,
                mime_type=mime,
                system_instruction=l2_prompt.SYSTEM_INSTRUCTION,
                prompt=user_prompt,
                response_schema=ClipPerception,
            )

        if result.parsed is None:
            logger.error("L2: Gemini returned unparseable output for %s", file_id)
            _persist_unparsed(file_id, result.raw_text, result.usage, result.model)
            _set_l2_status(file_id, "failed")
            with _pg_conn() as conn:
                _stage_fail(conn, file_id, "perception JSON did not validate against schema")
            return

        perception = result.parsed
        usage = dict(result.usage or {})
        _fuse_speakers(perception, speaker_turns)
        _persist(file_id, perception, usage, result.model)

        # Refine the Dialogues lens now that we know who is actually on camera:
        # any diarized voice with no visible speaker is off-screen crew. Best-
        # effort -- a failure here must never fail the L2 perception task.
        try:
            _flag_offscreen_dialogue(file_id, perception)
        except Exception:
            logger.exception("L2: off-camera dialogue flagging failed for %s", file_id)

        _set_l2_status(file_id, "ready")
        with _pg_conn() as conn:
            _stage_done(conn, file_id)

        logger.info(
            "L2 complete for %s (%d persons, %d atoms, %d tok out)",
            file_id,
            len(perception.persons),
            len(perception.atoms),
            usage.get("output_tokens", 0),
        )
    except psycopg.errors.ForeignKeyViolation:
        logger.info("L2: file %s deleted mid-run; abandoning cleanly.", file_id)
        return
    except Exception as e:
        logger.exception("L2 failed for %s", file_id)
        _set_l2_status(file_id, "failed")
        try:
            with _pg_conn() as conn:
                _stage_fail(conn, file_id, f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        except Exception:
            pass
        raise


def reenqueue_l2(file_id: str) -> str:
    """Force a fresh L2 perception run for an EXISTING clip, on demand.

    Unlike ``enqueue_l2_if_eligible`` (idempotent -- skips anything already
    queued/running/ready/skipped), this deliberately RE-runs so a schema bump
    (e.g. adding ``valence``) can be backfilled. Re-running L2 cascades on its
    own: the task re-defers thought segmentation + the hero-cuts precompute, so
    the whole footage map rebuilds off the fresh perception.

    Applies the same config/duration gate as the normal path. Returns one of
    "queued" | "skipped" | "disabled" | "gone" | "error" (the resulting state).
    """
    settings = get_settings()
    if not settings.enable_l2_perception or not settings.gemini_api_key:
        return "disabled"

    meta = _file_meta(file_id)
    if meta is None:
        return "gone"
    _, _, duration_s = meta
    if duration_s <= 0 or duration_s > settings.l2_max_duration_seconds:
        _set_l2_status(file_id, "skipped")
        return "skipped"

    try:
        l2_perception.defer(file_id=file_id)
        _set_l2_status(file_id, "queued")
        logger.info("L2: re-enqueued perception for %s (backfill, %.1fs)", file_id, duration_s)
        return "queued"
    except Exception:
        logger.exception("L2: failed to re-enqueue perception for %s", file_id)
        return "error"


def stale_perception_file_ids(user_id: str) -> List[str]:
    """A user's video/audio clips whose stored perception predates the current
    SCHEMA_VERSION (or was never perceived) -- the backfill candidates."""
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select f.id::text
              from files f
              left join clip_perception cp on cp.file_id = f.id
             where f.user_id = %s
               and f.file_type in ('video', 'audio')
               and (cp.schema_version is null or cp.schema_version < %s)
            """,
            (user_id, SCHEMA_VERSION),
        ).fetchall()
    return [r[0] for r in rows]


def enqueue_l2_if_eligible(file_id: str, duration_seconds: float) -> None:
    """Called by L1 on completion. Defers the perception task when the clip is
    short enough and L2 is configured; otherwise marks it skipped."""
    settings = get_settings()
    if not settings.enable_l2_perception or not settings.gemini_api_key:
        return
    if duration_seconds <= 0 or duration_seconds > settings.l2_max_duration_seconds:
        _set_l2_status(file_id, "skipped")
        return
    # Idempotent: L2 is now fired early (right after the speech track) so an L1
    # retry -- or the no-audio fallback path -- must not enqueue it twice. Only
    # a never-run file (null l2_status) is eligible.
    try:
        with _pg_conn() as conn:
            row = conn.execute(
                "select l2_status from files where id = %s", (file_id,)
            ).fetchone()
        if row and (row[0] or "") in ("queued", "running", "ready", "skipped"):
            return
    except Exception:
        logger.exception("L2: eligibility check failed for %s; enqueuing anyway", file_id)
    try:
        l2_perception.defer(file_id=file_id)
        _set_l2_status(file_id, "queued")
        logger.info("L2: enqueued perception for %s (%.1fs)", file_id, duration_seconds)
    except Exception:
        logger.exception("L2: failed to enqueue perception for %s", file_id)
