"""
Backfill: re-run speaker diarization with the NEURAL backend on already-analyzed
clips, then re-run the (cheap, no-Gemini) L2 audio/visual speaker fusion so the
person<->voice links match the new speaker ids.

Why: the classical MFCC+pitch diarizer collapses two same-gender speakers into a
single "S0" (silhouette < gate), which made a two-way interview look like a
monologue downstream. The neural d-vector backend separates them. This script
reuses each clip's proxy audio + existing transcript words; it does NOT re-run
Whisper or Gemini.

For each file:
  1. download the proxy, demux a 16 kHz wav
  2. diarize(words, backend="neural") -> per-word speaker ids
  3. write speakers back into transcripts.segments
  4. reload the L2 perception, clear stale voice links, re-fuse against the new
     turns, and persist

Usage:
  cd backend && .venv/bin/python scripts/rediarize.py                # all clips with a transcript
  cd backend && .venv/bin/python scripts/rediarize.py <file_id> ...  # specific clips
  cd backend && .venv/bin/python scripts/rediarize.py --thread <id>  # a thread's clips
  add --min-speakers 2 to force >=2 clusters (e.g. known multicam interview)
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.l1 import diarization as diar_mod  # noqa: E402
from app.services.processing import _download_from_r2  # noqa: E402


def _pg_conn():
    import psycopg
    from app.config import get_settings
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _demux_wav(src: str, out: str) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out],
        check=True, capture_output=True, timeout=600,
    )


def _resolve_file_ids(args) -> list:
    with _pg_conn() as c:
        if args.thread:
            from app.services.l3 import store
            t = store.get_thread(args.thread)
            ids = t["file_ids"] if t else []
        elif args.file_ids:
            ids = args.file_ids
        else:
            ids = [r[0] for r in c.execute(
                "select f.id::text from files f join transcripts t on t.file_id=f.id"
            ).fetchall()]
    return ids


def _flatten(segments: list):
    flat, refs = [], []
    for seg in segments:
        for w in seg.get("words") or []:
            flat.append(w)
            refs.append(w)
    return flat, refs


def _refuse(file_id: str, min_conf_clear: bool = True) -> str:
    """Re-run L2 speaker fusion against the freshly-written diarization. Returns
    a short status string. No Gemini; pure overlap math on existing perception."""
    from app.services.l2.perception import _fuse_speakers
    from app.services.l2.schema import ClipPerception
    from app.services.l3.diarize import load_turns

    with _pg_conn() as c:
        row = c.execute(
            "select perception, model, usage from clip_perception where file_id=%s",
            (file_id,),
        ).fetchone()
    if not row or not row[0]:
        return "no perception"
    doc = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    if doc.get("_parse_error"):
        return "perception unparsed; skipped"
    try:
        perc = ClipPerception.model_validate(doc)
    except Exception as e:  # noqa: BLE001
        return f"perception revalidate failed ({type(e).__name__}); diarization still updated"

    # Clear stale links so fusion can't keep an old S0 that no longer fits.
    for p in perc.persons:
        p.voice_speaker_id = None
        p.av_link_confidence = None

    _txt, _spk, turns = load_turns(file_id)
    _fuse_speakers(perc, turns)

    with _pg_conn() as c:
        c.execute(
            "update clip_perception set perception=%s::jsonb where file_id=%s",
            (json.dumps(perc.model_dump(mode="json")), file_id),
        )
    links = [(p.local_id, p.voice_speaker_id) for p in perc.persons if p.voice_speaker_id]
    return f"refused -> links {links}" if links else "refused -> no confident links"


def rediarize_one(file_id: str, min_speakers: int, max_speakers: int) -> None:
    with _pg_conn() as c:
        frow = c.execute("select name, r2_proxy_key, r2_key from files where id=%s",
                         (file_id,)).fetchone()
        trow = c.execute("select segments from transcripts where file_id=%s",
                         (file_id,)).fetchone()
    if not frow:
        print(f"  {file_id[:8]}  SKIP (no file row)"); return
    name, proxy_key, raw_key = frow
    if not trow or not trow[0]:
        print(f"  {file_id[:8]} {name:22} SKIP (no transcript)"); return
    segments = trow[0] if isinstance(trow[0], list) else json.loads(trow[0])
    flat, refs = _flatten(segments)
    if not flat:
        print(f"  {file_id[:8]} {name:22} SKIP (no words)"); return

    src_key = proxy_key or raw_key
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src"); wav = os.path.join(td, "a.wav")
        _download_from_r2(src_key, src)
        _demux_wav(src, wav)
        res = diar_mod.diarize(wav, flat, backend="neural",
                               min_speakers=min_speakers, max_speakers=max_speakers)

    spk = res.speaker_by_word
    if not spk or len(spk) != len(refs):
        print(f"  {file_id[:8]} {name:22} diarize returned nothing usable"); return
    for w, s in zip(refs, spk):
        if s is not None:
            w["speaker"] = s
    with _pg_conn() as c:
        c.execute("update transcripts set segments=%s::jsonb where file_id=%s",
                  (json.dumps(segments), file_id))

    # turn count per speaker for a quick sanity readout
    from collections import Counter
    by_spk = Counter(t["speaker"] for t in res.turns)
    fused = _refuse(file_id)
    print(f"  {file_id[:8]} {name:22} -> {res.num_speakers} speaker(s) "
          f"{dict(by_spk)} | {fused}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file_ids", nargs="*")
    ap.add_argument("--thread")
    ap.add_argument("--min-speakers", type=int, default=1)
    ap.add_argument("--max-speakers", type=int, default=8,
                    help="upper bound on clusters (cap to the real cast size to "
                         "stop one voice splitting into spurious extras)")
    args = ap.parse_args()

    ids = _resolve_file_ids(args)
    if not ids:
        print("no files to process"); return
    print(f"re-diarizing {len(ids)} clip(s) with the neural backend "
          f"(min_speakers={args.min_speakers}, max_speakers={args.max_speakers}):")
    for fid in ids:
        try:
            rediarize_one(fid, args.min_speakers, args.max_speakers)
        except Exception as e:  # noqa: BLE001
            print(f"  {fid[:8]}  FAILED: {type(e).__name__}: {e}")
    print("done.")


if __name__ == "__main__":
    main()
