"""Throwaway: re-ingest the podcast on the voice-id-pass pipeline and dump the
resulting cast table + on/off-camera coverage. Logs to /tmp/podcast_voiceid.log.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg  # noqa: E402
from app.config import get_settings  # noqa: E402

PID = "57b689b3-39db-4cb4-8385-9e87a996fe9a"


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> None:
    from app.services.l3 import ingest

    _log(f"re-ingesting podcast {PID[:8]} on voice-id-pass pipeline")
    t0 = time.time()
    run_id = ingest.run_ingest(PID)
    _log(f"ingest OK -> run {str(run_id)[:8]} in {(time.time()-t0)/60:.1f} min")

    with psycopg.connect(get_settings().database_url) as c:
        im = c.execute(
            "select identity_map from ingest_runs where id::text=%s", (str(run_id),)
        ).fetchone()
        payload = (im[0] if im else None) or {}
        persons = payload.get("persons") or []
        voice_owner = payload.get("voice_owner") or {}
        off_cam = payload.get("off_camera_voices") or []

        _log("=== CAST TABLE ===")
        _log(f"persons: {len(persons)}")
        for p in persons:
            _log(f"  {p.get('person_id')}: major={p.get('is_major')} "
                 f"owned_voices={p.get('owned_voices')} desc={ (p.get('description') or '')[:80] }")
        _log(f"voice_owner (voice->person): {voice_owner}")
        _log(f"off_camera_voices: {off_cam}")

        _log("=== PER-CUT COVERAGE ===")
        rows = c.execute(
            "select kind, speaker_person, on_camera from cut_records where ingest_run_id::text=%s",
            (str(run_id),),
        ).fetchall()
        speech = [r for r in rows if r[0] == "speech"]
        with_spk = [r for r in speech if r[1] is not None]
        oncam = [r for r in speech if r[2] is True]
        offcam = [r for r in speech if r[2] is False]
        _log(f"speech cuts: {len(speech)} | with speaker_person: {len(with_spk)} "
             f"| on_camera=True: {len(oncam)} | on_camera=False: {len(offcam)} "
             f"| on_camera=None: {len(speech)-len(oncam)-len(offcam)}")


if __name__ == "__main__":
    main()
