"""Podcast-only driver for the voice-first identity layer.

Scoped variant of revoice_all.py: re-diarize ONLY the podcast project's files
(now that pyannote is installed locally) so `transcripts.speaker_embeddings`
gets populated with real voiceprints, then re-ingest ONLY the podcast project on
the new pipeline. Serial (the pyannote pipeline is a process-wide singleton).

Daemonized (double-fork + setsid) so it survives the launching shell / agent
turn end. Logs to /tmp/revoice_podcast.log, pid in /tmp/revoice_podcast.pid.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg  # noqa: E402
from app.config import get_settings  # noqa: E402

_LOG_PATH = "/tmp/revoice_podcast.log"
_PID_PATH = "/tmp/revoice_podcast.pid"

PODCAST_PID = "57b689b3-39db-4cb4-8385-9e87a996fe9a"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _daemonize() -> None:
    if os.fork() > 0:
        os._exit(0)
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    sys.stdout.flush()
    sys.stderr.flush()
    fd = os.open(_LOG_PATH, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.dup2(fd, sys.stdout.fileno())
    os.dup2(fd, sys.stderr.fileno())
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, sys.stdin.fileno())
    with open(_PID_PATH, "w") as f:
        f.write(str(os.getpid()))


def _keep_awake() -> None:
    try:
        subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        pass


def _extract_wav(src: str, dst: str) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", dst],
        capture_output=True,
    )
    return r.returncode == 0 and os.path.exists(dst)


def _podcast_files():
    """(file_id, media_key) for every distinct file in the podcast's latest
    ingest run that has a transcript."""
    with psycopg.connect(get_settings().database_url) as c:
        run = c.execute(
            "select id::text from ingest_runs where project_id::text=%s and status='ready' "
            "order by created_at desc limit 1", (PODCAST_PID,)
        ).fetchone()
        if not run:
            return []
        fids = [r[0] for r in c.execute(
            "select distinct file_id::text from cut_records where ingest_run_id::text=%s", (run[0],)
        ).fetchall()]
        out = []
        for fid in fids:
            row = c.execute(
                "select coalesce(f.r2_proxy_a_key, f.r2_proxy_key) "
                "from files f join transcripts t on t.file_id=f.id "
                "where f.id::text=%s and f.file_type='video' and f.l1_status='ready'",
                (fid,)
            ).fetchone()
            if row and row[0]:
                out.append((fid, row[0]))
        return out


def phase1_rediarize() -> None:
    from app.services.l1.pipeline import _stage6_diarization
    from app.services.processing import _download_from_r2

    targets = _podcast_files()
    _log(f"PHASE 1: re-diarizing {len(targets)} podcast files to populate voiceprints")
    ok = fail = 0
    for i, (fid, key) in enumerate(targets, 1):
        with tempfile.TemporaryDirectory() as tmp:
            media = os.path.join(tmp, "src.mp4")
            wav = os.path.join(tmp, "audio.wav")
            try:
                _download_from_r2(key, media)
                if not _extract_wav(media, wav):
                    raise RuntimeError("ffmpeg wav extract failed")
                with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
                    _stage6_diarization(fid, wav, conn)
                    emb = conn.execute(
                        "select speaker_embeddings from transcripts where file_id=%s", (fid,)
                    ).fetchone()
                ok += 1
                n_vp = len(emb[0]) if emb and emb[0] else 0
                _log(f"  [{i}/{len(targets)}] ok {fid[:8]} -> {n_vp} voiceprint(s)")
            except Exception as e:  # noqa: BLE001
                fail += 1
                _log(f"  [{i}/{len(targets)}] FAIL {fid[:8]}: {type(e).__name__}: {e}")
    _log(f"PHASE 1 complete: {ok} ok, {fail} failed")


def phase2_ingest() -> None:
    from app.services.l3 import ingest
    _log(f"PHASE 2: re-ingesting podcast {PODCAST_PID[:8]}")
    p0 = time.time()
    try:
        run_id = ingest.run_ingest(PODCAST_PID)
        _log(f"PHASE 2 OK -> run {str(run_id)[:8]} ({(time.time() - p0) / 60:.1f} min)")
    except Exception as e:  # noqa: BLE001
        _log(f"PHASE 2 FAIL: {type(e).__name__}: {e}")


def main() -> None:
    t0 = time.time()
    _keep_awake()
    phase1_rediarize()
    phase2_ingest()
    _log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    if "--daemon" in sys.argv:
        _daemonize()
    main()
