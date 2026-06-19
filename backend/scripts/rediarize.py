"""
Backfill: re-run speaker diarization on already-analyzed clips (default backend =
the configured DIARIZATION_BACKEND, i.e. pyannote), rebuild the Dialogues lens
from the new speakers, then re-run the (cheap, no-Gemini) L2 audio/visual speaker
fusion + off-camera flagging so everything matches the new speaker ids.

Why: the old homemade clusterer mislabels short words (a sentence tail flips to a
phantom S0/S1) and collapses same-gender speakers. The pyannote backend is a real
diarization pipeline (VAD + neural segmentation + overlap-aware resegmentation).
This script reuses each clip's proxy audio + existing transcript words; it does
NOT re-run Whisper or Gemini.

For each file:
  1. download the proxy, demux a 16 kHz wav
  2. diarize(words, backend=...) -> per-word speaker ids (smoothed)
  3. write speakers back into transcripts.segments
  4. rebuild dialogue_segments (sentence + topic) from the new words
  5. reload the L2 perception, clear stale voice links, re-fuse against the new
     turns, re-apply the off-camera flag, and persist

Usage:
  cd backend && .venv/bin/python scripts/rediarize.py                # all clips with a transcript
  cd backend && .venv/bin/python scripts/rediarize.py <file_id> ...  # specific clips
  cd backend && .venv/bin/python scripts/rediarize.py --thread <id>  # a thread's clips
  add --min-speakers 2 to force >=2 clusters (e.g. known multicam interview)
  add --backend neural to force the CPU fallback (e.g. no HF token locally)
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
    # Re-apply the off-camera flag now that the on-camera voice set is known.
    try:
        from app.services.l2.perception import _flag_offscreen_dialogue
        _flag_offscreen_dialogue(file_id, perc)
    except Exception as e:  # noqa: BLE001
        print(f"    (offscreen flagging skipped: {type(e).__name__}: {e})")
    links = [(p.local_id, p.voice_speaker_id) for p in perc.persons if p.voice_speaker_id]
    return f"refused -> links {links}" if links else "refused -> no confident links"


def _rebuild_dialogue(file_id: str, flat: list, wav_path: str) -> None:
    """Rebuild the Dialogues-lens selects from the freshly-diarized words so the
    smoothed speakers + fragment-rejoin take effect on already-analyzed clips."""
    from app.services.l1 import dialogue_segments as dlg_mod

    words = [
        {
            "start_ms": w.get("start_ms", 0), "end_ms": w.get("end_ms", 0),
            "text": w.get("text", ""), "is_filler": w.get("is_filler", False),
            "speaker": w.get("speaker"),
        }
        for w in flat
    ]
    result = dlg_mod.build_dialogue_segments(words, wav_path)
    with _pg_conn() as c:
        c.execute(
            """
            insert into dialogue_segments (file_id, schema_version, segments)
            values (%s, %s, %s::jsonb)
            on conflict (file_id) do update set
                schema_version = excluded.schema_version,
                segments = excluded.segments,
                created_at = now()
            """,
            (file_id, dlg_mod.SCHEMA_VERSION, json.dumps(result)),
        )


def rediarize_one(file_id: str, min_speakers: int, max_speakers: int,
                  backend: str) -> None:
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
        res = diar_mod.diarize(wav, flat, backend=backend,
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
        # Rebuild the Dialogues lens while the wav is still on disk (needs it for
        # silence-snapped cut points).
        _rebuild_dialogue(file_id, flat, wav)

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
    ap.add_argument("--backend", default=None,
                    help="diarization backend: pyannote | neural | mfcc "
                         "(default: the configured DIARIZATION_BACKEND)")
    args = ap.parse_args()

    from app.config import get_settings
    backend = args.backend or get_settings().diarization_backend

    ids = _resolve_file_ids(args)
    if not ids:
        print("no files to process"); return
    print(f"re-diarizing {len(ids)} clip(s) with the {backend!r} backend "
          f"(min_speakers={args.min_speakers}, max_speakers={args.max_speakers}):")
    for fid in ids:
        try:
            rediarize_one(fid, args.min_speakers, args.max_speakers, backend)
        except Exception as e:  # noqa: BLE001
            print(f"  {fid[:8]}  FAILED: {type(e).__name__}: {e}")
    print("done.")


if __name__ == "__main__":
    main()
