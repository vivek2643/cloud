"""Live smoke test of VIDEO-mode Pass 1 on ONE project, so we prove the new
pipeline end-to-end (ffmpeg sub-clip -> Files API upload -> Gemini video call ->
t_ms remap) before committing a full 21-project video batch.

Run with:  VCUT_PASS1_INPUT_MODE=video .venv/bin/python scripts/_smoke_video_one.py <project_id>

Surfaces every fallback instead of trusting it: captures the vcut logger and
reports whether ANY file fell back to frames ("falling back to frames") and
whether the speech channel fell back to copy_prior. A clean run = 0 frames
fallbacks + speech_source=pipeline.
"""
import io
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_settings  # noqa: E402
from app.services import db  # noqa: E402
from app.services.vcut import pass2 as p2  # noqa: E402
from app.services.vcut.orchestrate import run_vcut_ingest  # noqa: E402


def _counts(rid):
    with db.connection() as conn:
        return conn.execute(
            """
            select count(*),
                   count(*) filter (where kind = 'video'),
                   count(*) filter (where kind = 'speech'),
                   count(*) filter (where take_role = 'outlook'),
                   count(*) filter (where take_role = 'winner')
              from cut_records where ingest_run_id = %s
            """,
            (rid,),
        ).fetchone()


def _speech_source(rid):
    with db.connection_dict_row() as conn:
        row = conn.execute(
            "select speech_channel_status from ingest_runs where id = %s", (rid,)
        ).fetchone()
    return ((row["speech_channel_status"] if row else None) or {}).get("source")


def main():
    pid = sys.argv[1]
    settings = get_settings()
    print(f"SMOKE pid={pid} pass1_input_mode={settings.vcut_pass1_input_mode} "
          f"fps={settings.vcut_video_fps} media_res={settings.vcut_video_media_resolution}",
          flush=True)
    if settings.vcut_pass1_input_mode != "video":
        print("!! NOT in video mode -- set VCUT_PASS1_INPUT_MODE=video", flush=True)

    # Tap the vcut logger so fallbacks are provable, not assumed.
    buf = io.StringIO()
    h = logging.StreamHandler(buf)
    h.setLevel(logging.INFO)
    root = logging.getLogger("app.services.vcut")
    root.addHandler(h)
    root.setLevel(logging.INFO)

    t0 = time.time()
    rid = run_vcut_ingest(pid)
    try:
        p2.run_enrich(pid, rid)
    except Exception as e:  # noqa: BLE001
        print(f"  enrich failed (cuts still visible): {e!r}", flush=True)
    dt = time.time() - t0

    logs = buf.getvalue()
    frames_fallbacks = logs.count("falling back to frames")
    speech_fb = "falling back to copy_prior" in logs
    source = _speech_source(rid)
    c = _counts(rid)

    print(f"\nRUN {rid}  {dt:.0f}s", flush=True)
    print(f"  counts total/video/speech/outlook/winner = {c}", flush=True)
    print(f"  pass1 video->frames fallbacks: {frames_fallbacks}  "
          f"(0 = every file used video)", flush=True)
    print(f"  speech_source: {source}  (want 'pipeline'; copy_prior={speech_fb})", flush=True)
    verdict = "CLEAN (true new pipeline, no fallbacks)" if (
        frames_fallbacks == 0 and source == "pipeline") else "HAS FALLBACKS -- see below"
    print(f"  VERDICT: {verdict}", flush=True)
    if verdict.startswith("HAS"):
        print("\n---- vcut log tail ----", flush=True)
        print("\n".join(logs.splitlines()[-40:]), flush=True)


if __name__ == "__main__":
    main()
