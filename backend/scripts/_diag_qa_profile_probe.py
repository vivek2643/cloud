#!/usr/bin/env python3
"""color_qa_harness.plan.md B.2 Step 0 (the Part B spike): ffprobe
transfer-tag survey over the corpus's ORIGINAL source files, not proxies
(Risk 4 -- proxies are transcoded and may strip/normalize transfer tags, so
probing them would silently overstate how much real metadata exists).
Read-only: downloads each original once, runs `ffprobe -show_streams`
(container metadata only, no decode), and tabulates `color_transfer`
(e.g. `arib-std-b67`=HLG, `smpte2084`=PQ, `bt709`, unset/`unknown`),
`color_primaries`, `color_space`, and codec -- answering B.2's open
question: is real transfer-function metadata available for the IDT to key
off, or are we heuristic-only (`color_stats.is_log_flat`)?

Also cross-tabulates against each file's `is_log_flat` heuristic (from
`color_stats`, already computed at ingest) so a mismatch is visible directly
(e.g. a `bt709`-tagged file the heuristic nonetheless calls log/flat --
Risk 2's false-positive case).

Usage: PYTHONPATH=. .venv/bin/python scripts/_diag_qa_profile_probe.py
       [--corpus <path to corpus.json, default _out/qa/corpus.json>]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import psycopg  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.l3.grade.measure import fetch_color_stats  # noqa: E402
from app.services.processing import _download_from_r2  # noqa: E402

CORPUS_PATH = os.path.join(HERE, "_out", "qa", "corpus.json")
OUT_PATH = os.path.join(HERE, "_out", "qa", "profile_probe.json")


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _file_ids_from_corpus(corpus: List[Dict[str, Any]]) -> List[str]:
    ids = set()
    for entry in corpus:
        ids.update(entry.get("file_ids") or [])
    return sorted(ids)


def _probe_stream(path: str) -> Optional[Dict[str, Any]]:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", "-select_streams", "v:0", path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    streams = data.get("streams") or []
    return streams[0] if streams else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=CORPUS_PATH)
    args = parser.parse_args()

    if not os.path.exists(args.corpus):
        raise SystemExit(f"no corpus manifest at {args.corpus} -- run _diag_qa_corpus.py first")
    with open(args.corpus) as f:
        corpus = json.load(f)
    file_ids = _file_ids_from_corpus(corpus)
    if not file_ids:
        raise SystemExit("corpus manifest has no file_ids")

    with _pg() as c:
        rows = c.execute(
            "select id::text, r2_key, filename from files where id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()
    key_by_file = {r[0]: r[1] for r in rows}
    name_by_file = {r[0]: r[2] for r in rows}

    color_stats = fetch_color_stats(file_ids)

    records: List[Dict[str, Any]] = []
    transfer_counter: Counter = Counter()
    with tempfile.TemporaryDirectory(prefix="edso_qa_probe_") as tmp:
        for i, fid in enumerate(file_ids):
            key = key_by_file.get(fid)
            print(f"[{i + 1}/{len(file_ids)}] {fid[:8]} ({name_by_file.get(fid) or '?'})", flush=True)
            if not key:
                records.append({"file_id": fid, "error": "no r2_key on file row"})
                continue
            ext = os.path.splitext(key)[1] or ".mp4"
            local = os.path.join(tmp, f"{fid}{ext}")
            try:
                _download_from_r2(key, local)
            except Exception as e:
                records.append({"file_id": fid, "error": f"download failed: {e}"})
                continue
            stream = _probe_stream(local)
            try:
                os.remove(local)
            except OSError:
                pass
            if stream is None:
                records.append({"file_id": fid, "error": "ffprobe found no video stream"})
                continue

            transfer = stream.get("color_transfer", "unset")
            transfer_counter[transfer] += 1
            cs = (color_stats.get(fid) or {})
            records.append({
                "file_id": fid,
                "name": name_by_file.get(fid),
                "codec_name": stream.get("codec_name"),
                "color_transfer": transfer,
                "color_primaries": stream.get("color_primaries", "unset"),
                "color_space": stream.get("color_space", "unset"),
                "width": stream.get("width"),
                "height": stream.get("height"),
                "is_log_flat_heuristic": bool(cs.get("is_log_flat")),
            })

    print("\n==== color_transfer tabulation ====")
    for tag, count in transfer_counter.most_common():
        print(f"  {tag:20} {count}")

    tagged_log = [r for r in records if r.get("color_transfer") not in (None, "unset", "bt709", "unknown")]
    print(f"\n{len(tagged_log)}/{len(records)} files carry a non-bt709/unset transfer tag ffprobe could read.")

    mismatches = [
        r for r in records
        if r.get("is_log_flat_heuristic") and r.get("color_transfer") in ("bt709",)
    ]
    if mismatches:
        print(f"\n{len(mismatches)} file(s) tagged bt709 but flagged is_log_flat by the heuristic "
             f"(Risk 2's false-positive case):")
        for r in mismatches:
            print(f"  {r['file_id'][:8]}  {r.get('name')}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({"records": records, "transfer_tabulation": dict(transfer_counter)}, f, indent=2)
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
