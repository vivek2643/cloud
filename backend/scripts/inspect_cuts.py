"""
Inspect the actual hero cuts produced for the files in a folder, file by file.

Reuses the production path (compute_file_cache) so what prints is exactly what
the feed would serve, plus the raw anchors so we can see WHERE each cut came
from (which anchor source minted it).

Usage:
    .venv/bin/python scripts/inspect_cuts.py --folder "Reel trail"
    .venv/bin/python scripts/inspect_cuts.py --folder "Reel trail" --energy 0.5
    .venv/bin/python scripts/inspect_cuts.py --only <file_id>
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import psycopg  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.l3 import hero_cuts as hc  # noqa: E402
from app.services.l3 import anchors as anc  # noqa: E402
from app.services.l3 import score_span as ss  # noqa: E402


def _conn():
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _files_in_folder(conn, folder: str):
    return conn.execute(
        """
        select f.id::text, coalesce(f.name, f.id::text),
               coalesce(f.duration_seconds, 0)
          from files f
          join folders d on d.id = f.folder_id
         where lower(d.name) = lower(%s)
           and f.file_type in ('video','audio')
         order by coalesce(f.duration_seconds,0) asc
        """,
        (folder,),
    ).fetchall()


def _fmt(ms: int) -> str:
    return f"{ms/1000:.2f}s"


def inspect_file(fid: str, name: str, dur_s: float, energy: float) -> None:
    print("\n" + "=" * 88)
    print(f"FILE  {name}   ({dur_s:.0f}s)   {fid}")
    print("=" * 88)

    inputs = hc._load_inputs([fid])
    clip = inputs.get(fid)
    if clip is None:
        print("  (no stored artifacts)")
        return

    # Raw anchors -- the source signals before the beat engine.
    anchors = anc.gather_anchors(
        duration_ms=clip.duration_ms, dialogue=clip.dialogue,
        perception=clip.perception, motion=clip.motion, audio=clip.audio)
    by_aff = Counter(a.affordance for a in anchors)
    by_kind = Counter(a.kind for a in anchors)
    print(f"\n  ANCHORS ({len(anchors)}): "
          + ", ".join(f"{k}={v}" for k, v in sorted(by_aff.items())))
    print("    kinds: " + ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items())))

    # Produced cuts via the real per-file path.
    params = hc.energy_to_params(energy)
    source = ss.load_sources([fid]).get(fid)
    raw = hc._file_heroes(clip, source, params)
    by_chan = Counter(h.channel_of() for h in raw)
    by_subj = Counter((h.subject or "-") for h in raw)
    print(f"\n  CUTS ({len(raw)}) @ energy={energy}  "
          f"channels: " + ", ".join(f"{k}={v}" for k, v in sorted(by_chan.items())))
    print("    subjects: " + ", ".join(f"{k}={v}" for k, v in sorted(by_subj.items())))
    print()
    for h in sorted(raw, key=lambda h: h.src_in_ms):
        dur = h.src_out_ms - h.src_in_ms
        tag = f"{h.channel_of()}.{h.subject or '-'}"
        mom = f" moment={h.moment_id[-4:]}" if h.moment_id else ""
        split = f" split={len(h.keep_spans)}" if h.keep_spans else ""
        label = (h.label or "")[:52]
        print(f"    [{tag:<14}] {_fmt(h.src_in_ms):>8}-{_fmt(h.src_out_ms):<8} "
              f"({_fmt(dur):>6}) s={h.score:.2f}{mom}{split}  {label}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="Reel trail")
    ap.add_argument("--only", default=None)
    ap.add_argument("--energy", type=float, default=0.5)
    args = ap.parse_args()

    with _conn() as conn:
        if args.only:
            rows = conn.execute(
                "select id::text, coalesce(name,id::text), coalesce(duration_seconds,0) "
                "from files where id = %s", (args.only,)).fetchall()
        else:
            rows = _files_in_folder(conn, args.folder)

    print(f"folder={args.folder!r}  files={len(rows)}  energy={args.energy}")
    for fid, name, dur in rows:
        inspect_file(fid, name, dur, args.energy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
