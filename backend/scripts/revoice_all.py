"""One-off driver for the voice-first identity layer (voice_first_identity.plan.md).

Phase 1 -- RE-DIARIZE every file that has a transcript, so `transcripts.
speaker_embeddings` gets populated (older files were diarized before Phase A
existed, so their voiceprints are empty and identity/voices.py can only fall
back to per-file/group voices). We call `_stage6_diarization` DIRECTLY (not via
`_run_stage`) so the already-"done" diarization stage is forced to re-run; it
self-loads the transcript's words, re-runs pyannote WITH `return_embeddings`,
and rewrites segments + speaker_embeddings in one shot. Serial: the pyannote
pipeline is a process-wide singleton, so one file at a time is the safe path.

Phase 2 -- re-ingest every project (podcast FIRST, then the rest) on the new
pipeline, one at a time, logging per-project wall time.

Daemonized (double-fork + setsid, like refresh_and_ingest_all.py) so it is owned
by launchd and survives the launching shell / agent-harness turn end. Logs to
/tmp/revoice_all.log, pid in /tmp/revoice_all.pid.
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

_LOG_PATH = "/tmp/revoice_all.log"
_PID_PATH = "/tmp/revoice_all.pid"

# Project order: PODCAST FIRST (the one being checked), then unique footage,
# then exact-duplicate rows last. Prefixes resolved to full uuids at runtime.
ORDER = [
    ("57b689b3-", "podcast (4 angles)"),
    ("72d87ca9-", "montage reel (MVI)"),
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


def _extract_wav(src: str, dst: str) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", dst],
        capture_output=True,
    )
    return r.returncode == 0 and os.path.exists(dst)


def _files_to_rediarize():
    """Distinct video files with a transcript: (file_id, audio-or-video key)."""
    with psycopg.connect(get_settings().database_url) as c:
        rows = c.execute(
            """
            select distinct f.id::text,
                   coalesce(f.r2_proxy_a_key, f.r2_proxy_key) as media_key
              from files f
              join transcripts t on t.file_id = f.id
             where f.file_type = 'video' and f.l1_status = 'ready'
            """
        ).fetchall()
    return [(fid, key) for fid, key in rows if key]


def phase1_rediarize() -> None:
    from app.services.l1.pipeline import _stage6_diarization
    from app.services.processing import _download_from_r2

    targets = _files_to_rediarize()
    _log(f"PHASE 1: re-diarizing {len(targets)} files to populate voiceprints")
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


def _resolve_order():
    with psycopg.connect(get_settings().database_url) as c:
        all_ids = [r[0] for r in c.execute("select id::text from projects").fetchall()]
    out = []
    for pref, label in ORDER:
        match = [i for i in all_ids if i.startswith(pref.rstrip("-"))]
        if match:
            out.append((match[0], label))
        else:
            _log(f"  WARN no project matches prefix {pref} ({label})")
    return out


def phase2_ingest() -> None:
    from app.services.l3 import ingest
    plan = _resolve_order()
    _log(f"PHASE 2: re-ingesting {len(plan)} projects, one at a time")
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
    _log(f"PHASE 2 complete: {ok} ok, {fail} failed")


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
