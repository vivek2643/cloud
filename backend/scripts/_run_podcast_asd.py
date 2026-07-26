"""Podcast-only driver for the new ASD identity layer (asd_identity.plan.md).

Runs the L1 active-speaker pass (face detect+track+embed+ASD) for every file in
the podcast's latest ready ingest run -- populating face_tracks -- then
re-ingests the podcast on the new pipeline so cut_records pick up the ASD-bound
identity. Mirrors the l1_active_speaker task body (pipeline.py) so it runs the
exact production code path, just locally/synchronously (the ASD pass is
CPU-only, so no GPU/fleet is required).

Daemonized so it survives the launching shell. Logs to /tmp/podcast_asd.log,
pid in /tmp/podcast_asd.pid.
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

_LOG_PATH = "/tmp/podcast_asd.log"
_PID_PATH = "/tmp/podcast_asd.pid"
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


def _podcast_file_ids():
    with psycopg.connect(get_settings().database_url) as c:
        run = c.execute(
            "select id::text from ingest_runs where project_id::text=%s and status='ready' "
            "order by created_at desc limit 1", (PODCAST_PID,)
        ).fetchone()
        if not run:
            return []
        return [r[0] for r in c.execute(
            "select distinct file_id::text from cut_records where ingest_run_id::text=%s", (run[0],)
        ).fetchall()]


def phase1_active_speaker() -> None:
    from app.services.l1.pipeline import (
        _pg_conn, _run_stage, _stage_active_speaker,
    )
    from app.services.processing import _download_from_r2

    fids = _podcast_file_ids()
    _log(f"PHASE 1: active-speaker pass for {len(fids)} podcast files")
    ok = fail = 0
    for i, fid in enumerate(fids, 1):
        t0 = time.time()
        try:
            with _pg_conn() as conn:
                row = conn.execute(
                    "select r2_proxy_key from files where id::text=%s", (fid,)
                ).fetchone()
            key = row[0] if row else None
            if not key:
                _log(f"  [{i}/{len(fids)}] SKIP {fid[:8]}: no editing proxy")
                continue
            with tempfile.TemporaryDirectory() as tmp:
                proxy = os.path.join(tmp, "proxy.mp4")
                _download_from_r2(key, proxy)
                with _pg_conn() as conn:
                    _run_stage(conn, fid, "active_speaker",
                               _stage_active_speaker, fid, proxy, conn)
                    n = conn.execute(
                        "select jsonb_array_length(tracks) from face_tracks where file_id::text=%s",
                        (fid,)
                    ).fetchone()
            ok += 1
            _log(f"  [{i}/{len(fids)}] ok {fid[:8]} -> {n[0] if n else 0} track(s) "
                 f"({time.time()-t0:.0f}s)")
        except Exception as e:  # noqa: BLE001
            fail += 1
            _log(f"  [{i}/{len(fids)}] FAIL {fid[:8]}: {type(e).__name__}: {e}")
    _log(f"PHASE 1 complete: {ok} ok, {fail} failed")


def phase2_ingest() -> None:
    from app.services.l3 import ingest
    _log(f"PHASE 2: re-ingesting podcast {PODCAST_PID[:8]}")
    p0 = time.time()
    try:
        run_id = ingest.run_ingest(PODCAST_PID)
        _log(f"PHASE 2 OK -> run {str(run_id)[:8]} ({(time.time()-p0)/60:.1f} min)")
    except Exception as e:  # noqa: BLE001
        _log(f"PHASE 2 FAIL: {type(e).__name__}: {e}")


def main() -> None:
    t0 = time.time()
    _keep_awake()
    if "--ingest-only" not in sys.argv:
        phase1_active_speaker()
    phase2_ingest()
    _log(f"ALL DONE in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    if "--daemon" in sys.argv:
        _daemonize()
    main()
