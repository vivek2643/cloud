"""
Backfill the Dialogues lens for files that were L1-processed before the
`dialogue_segments` stage existed.

For each file that has a transcript but no dialogue_segments row, we pull the
proxy from R2, extract a 16k mono WAV (for silence-snapped cuts), run the
segmenter, and upsert. Best-effort per file.

Run:  python backend/scripts/backfill_dialogues.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import psycopg  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.l1 import dialogue_segments as dlg_mod  # noqa: E402
from app.services.l1.pipeline import _flatten_words  # noqa: E402
from app.services.processing import _download_from_r2  # noqa: E402


def _extract_wav(src: str, dst: str) -> bool:
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", dst],
        capture_output=True,
    )
    return r.returncode == 0 and os.path.exists(dst)


def main(force: bool = False):
    settings = get_settings()
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        rows = conn.execute(
            """
            select f.id, f.name, f.r2_proxy_key, f.r2_key
              from files f
              join transcripts t on t.file_id = f.id
             where f.l1_status = 'ready'
            """
        ).fetchall()

        done = {
            r[0] for r in conn.execute("select file_id from dialogue_segments").fetchall()
        }

        targets = [r for r in rows if force or str(r[0]) not in {str(d) for d in done}]
        print(f"{len(rows)} ready files with transcripts; {len(targets)} to backfill")

        for fid, name, proxy_key, raw_key in targets:
            key = proxy_key or raw_key
            if not key:
                print(f"  skip {name}: no media key")
                continue
            seg_row = conn.execute(
                "select segments from transcripts where file_id = %s", (fid,)
            ).fetchone()
            words = _flatten_words(seg_row[0]) if seg_row and seg_row[0] else []
            if not words:
                print(f"  skip {name}: no words")
                continue

            with tempfile.TemporaryDirectory() as tmp:
                media = os.path.join(tmp, "proxy.mp4")
                wav = os.path.join(tmp, "audio.wav")
                try:
                    _download_from_r2(key, media)
                except Exception as e:
                    print(f"  skip {name}: download failed ({e})")
                    continue
                wav_path = wav if _extract_wav(media, wav) else None
                result = dlg_mod.build_dialogue_segments(words, wav_path)

            conn.execute(
                """
                insert into dialogue_segments (file_id, schema_version, segments)
                values (%s, %s, %s::jsonb)
                on conflict (file_id) do update set
                    schema_version = excluded.schema_version,
                    segments = excluded.segments,
                    created_at = now()
                """,
                (fid, dlg_mod.SCHEMA_VERSION, json.dumps(result)),
            )
            print(
                f"  ok {name}: {len(result['sentence'])} sentence, "
                f"{len(result['topic'])} topic"
                + ("" if wav_path else "  [no-audio fallback]")
            )

    print("backfill done")


if __name__ == "__main__":
    main(force="--force" in sys.argv)
