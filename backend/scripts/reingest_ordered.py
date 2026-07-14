"""Ordered, SERIAL re-ingest of every project on the new Gemini perception.

Unlike refresh_and_ingest_all.py (parallel, deduped), this runs projects ONE
AT A TIME in a chosen order so each finished project can be checked as it lands,
starting with the reel + podcast. L1 is already current (motion camera-series
refresh found 0 files needing it), so this only re-runs the L3 cuts-v3 ingest
(Pass 1 + Gemini Pass 2 + post) per project.

All 13 project rows are included; the 9 UNIQUE footage sets are front-loaded and
the 4 exact-duplicate rows come last, so every row ends with a fresh run while
the interesting content is done first.

Daemonized (double-fork + setsid) so it survives the launching shell; logs to
/tmp/reingest_ordered.log, pid in /tmp/reingest_ordered.pid.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_LOG_PATH = "/tmp/reingest_ordered.log"
_PID_PATH = "/tmp/reingest_ordered.pid"

# (project_id, human label) -- ORDER MATTERS. Unique footage first, dups last.
ORDER = [
    ("72d87ca9-", "montage reel (MVI)"),
    ("57b689b3-", "podcast (4 angles)"),
    ("a294f9da-", "drone / b-roll reel (DJI)"),
    ("642e9587-", "single short (video186)"),
    ("8621c012-", "short (video189/186)"),
    ("7ef4663d-", "canon shoot A_0009 (5)"),
    ("f48da65f-", "canon shoot A_0004 (9)"),
    ("f52e7ee1-", "canon shoot A_0004 (14)"),
    ("41fb01fc-", "canon shoot A_0017 (11)"),
    ("a596ea5f-", "DUP of drone reel"),
    ("94f92040-", "DUP of A_0009 shoot"),
    ("91688328-", "DUP of A_0017 shoot"),
    ("5cd8f004-", "DUP of short"),
]


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


def _resolve_ids(prefixes):
    """Map each 8-char prefix to the full project uuid (order preserved)."""
    import psycopg
    from app.config import get_settings
    with psycopg.connect(get_settings().database_url) as c:
        rows = c.execute("select id::text from projects").fetchall()
    all_ids = [r[0] for r in rows]
    out = []
    for pref, label in prefixes:
        match = [i for i in all_ids if i.startswith(pref.rstrip("-"))]
        if match:
            out.append((match[0], label))
        else:
            _log(f"  WARN no project matches prefix {pref} ({label}); skipping")
    return out


def main() -> None:
    t0 = time.time()
    _keep_awake()
    from app.services.l3 import ingest
    plan = _resolve_ids(ORDER)
    _log(f"ORDERED SERIAL re-ingest: {len(plan)} projects, one at a time")
    ok = fail = 0
    for i, (pid, label) in enumerate(plan, 1):
        _log(f"[{i}/{len(plan)}] START {pid[:8]} -- {label}")
        p0 = time.time()
        try:
            run_id = ingest.run_ingest(pid)
            ok += 1
            _log(f"[{i}/{len(plan)}] OK    {pid[:8]} -> run {str(run_id)[:8]} "
                 f"({(time.time() - p0) / 60:.1f} min) -- {label}")
        except Exception as e:  # noqa: BLE001
            fail += 1
            _log(f"[{i}/{len(plan)}] FAIL  {pid[:8]}: {type(e).__name__}: {e} -- {label}")
    _log(f"ALL DONE: {ok} ok, {fail} failed in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    if "--daemon" in sys.argv:
        _daemonize()
    main()
