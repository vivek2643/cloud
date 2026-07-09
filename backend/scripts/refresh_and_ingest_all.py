"""One-off driver: bring EVERY project up to the latest cuts-v3 pipeline.

Phase 1 -- L1 motion refresh. The camera-move tag needs the signed camera
velocity series (camera_dx/dy/zoom) that only exists for files whose motion
stage ran after migration 033. Re-run motion_dynamics for every file still on
the old row, off its CHEAPEST existing source (proxy_b if present, else the
1080p editing proxy) so we never pull a multi-GB raw. The stage upsert
overwrites the whole row deterministically, so all other L1 signals are
unchanged -- it just fills in the camera series.

Phase 2 -- re-ingest every distinct project so quality scores, framing, and
the camera tag land on every cut.

Idempotent-ish: re-running Phase 1 just recomputes (harmless); Phase 2 always
creates a fresh ingest_run (the UI reads the latest ready one).
"""
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg
from app.config import get_settings


_LOG_PATH = "/tmp/refresh_ingest_all.log"
_PID_PATH = "/tmp/refresh_ingest_all.pid"


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _daemonize() -> None:
    """Double-fork + setsid so the run is owned by launchd (pid 1), fully
    detached from the launching shell's session -- otherwise the agent harness
    reaps it at turn-end. stdout/stderr are redirected to the log file. macOS
    ships no `setsid` binary, so we do it in-process."""
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


def _distinct_projects_and_need():
    """(project_ids in source order, [(file_id, source_key, duration_s)] to refresh)."""
    with psycopg.connect(get_settings().database_url) as c:
        rows = c.execute("select id::text, source_file_ids from projects").fetchall()
        seen = {}
        for pid, sf in rows:
            key = tuple(sorted(str(x) for x in (sf or [])))
            if key:
                seen.setdefault(key, []).append(pid)
        project_ids = [pids[0] for _key, pids in seen.items()]
        all_files = sorted(set(f for key in seen for f in key))

        have = set(
            r[0] for r in c.execute(
                "select file_id::text from motion_dynamics where camera_dx <> '[]'::jsonb"
            ).fetchall()
        )
        frows = c.execute(
            "select id::text, r2_proxy_b_key, r2_proxy_key, duration_seconds "
            "from files where id = any(%s::uuid[]) and file_type = 'video'",
            (all_files,),
        ).fetchall()
        need = []
        for fid, pb, pk, dur in frows:
            if fid in have:
                continue
            src = pb or pk
            if not src:
                _log(f"  WARN {fid[:8]} has no proxy source; skipping motion refresh")
                continue
            need.append((fid, src, float(dur or 0.0)))
    return project_ids, need


def _refresh_one_inproc(file_id: str, source_key: str, duration_s: float) -> None:
    """Actual motion refresh -- imported heavy libs lazily so this only pays
    the OpenCV import cost inside the isolated child process."""
    from app.services.processing import _download_from_r2
    from app.services.l1.pipeline import _pg_conn, _stage9_motion_dynamics
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "motion_src.mp4")
        _download_from_r2(source_key, path)
        with _pg_conn() as conn:
            _stage9_motion_dynamics(file_id, path, duration_s, conn)


def phase1_refresh(need, timeout_s: int = 600) -> None:
    """Refresh each file's motion in its OWN subprocess, one at a time. The
    optical-flow pass segfaults when several run concurrently in one process
    (shared native OpenCV state), so isolation + serial execution is the safe
    path -- a crash/timeout on one file is contained and we move on."""
    _log(f"PHASE 1: refreshing L1 motion (camera series) for {len(need)} files "
         f"(1 subprocess each, {timeout_s}s timeout)")
    ok = fail = 0
    for i, (fid, src, dur) in enumerate(need, 1):
        _log(f"  [{i}/{len(need)}] refreshing {fid[:8]} ...")
        cmd = [sys.executable, os.path.abspath(__file__), "--one", fid, src, str(dur)]
        try:
            r = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               capture_output=True, text=True, timeout=timeout_s)
            if r.returncode == 0:
                ok += 1
                _log(f"  [{i}/{len(need)}] ok {fid[:8]}")
            else:
                fail += 1
                tail = (r.stderr or r.stdout or "").strip().splitlines()[-3:]
                _log(f"  [{i}/{len(need)}] FAILED {fid[:8]} rc={r.returncode}: {' | '.join(tail)}")
        except subprocess.TimeoutExpired:
            fail += 1
            _log(f"  [{i}/{len(need)}] TIMEOUT {fid[:8]} after {timeout_s}s")
    _log(f"PHASE 1 complete: {ok} ok, {fail} failed")


def phase2_ingest(project_ids, workers: int = 3) -> None:
    from app.services.l3 import ingest
    _log(f"PHASE 2: re-ingesting {len(project_ids)} projects ({workers} workers)")
    results = ingest.run_many(project_ids, max_workers=workers)
    ok, fail = 0, 0
    for pid, res in results.items():
        if isinstance(res, Exception):
            fail += 1
            _log(f"  FAILED {pid[:8]}: {type(res).__name__}: {res}")
        else:
            ok += 1
            _log(f"  ok {pid[:8]} -> run {str(res)[:8]}")
    _log(f"PHASE 2 complete: {ok} ok, {fail} failed")


def _keep_awake() -> None:
    """Hold off idle/system sleep for as long as THIS process lives (macOS).
    Bound to our pid so it dies with us; best-effort."""
    try:
        subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        pass


def main() -> None:
    t0 = time.time()
    _keep_awake()
    project_ids, need = _distinct_projects_and_need()
    _log(f"plan: {len(project_ids)} projects, {len(need)} files need motion refresh")
    phase1_refresh(need)
    phase2_ingest(project_ids)
    _log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    # Isolated single-file worker: `--one <file_id> <source_key> <duration_s>`.
    # Runs in its own process so a native optical-flow crash can't take down
    # the parent driver.
    if len(sys.argv) >= 2 and sys.argv[1] == "--one":
        _fid, _src, _dur = sys.argv[2], sys.argv[3], float(sys.argv[4])
        _refresh_one_inproc(_fid, _src, _dur)
        sys.exit(0)
    if "--daemon" in sys.argv:
        _daemonize()
    main()
