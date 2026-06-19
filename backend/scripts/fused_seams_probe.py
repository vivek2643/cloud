#!/usr/bin/env python3
"""
Numeric sanity probe for the fused seam field (READ-ONLY).

For a file it prints:
  1. The veto check  -- mean fused cost DURING speech (should be high: don't
     cut) vs OUTSIDE speech (should be low: safe to cut). A working field shows
     speech >> silence.
  2. The top fused seams -- the best ranked cut instants and what fed them.
  3. A snap demo -- a rough action window that *overlaps a spoken word* is snapped
     by the field; the result should move OFF the word (fixing the old bug where
     action cuts bled background dialogue).

    python scripts/fused_seams_probe.py --file-id <uuid> [--energy 0.5]
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import viz_cut_grid as viz  # noqa: E402
from app.services.l1 import fused_seams as fs  # noqa: E402


def _ch(channels, label):
    for c in channels:
        if c["label"] == label:
            return c
    return None


def probe(file_id: str, energy: float) -> None:
    data = viz._load_from_db(file_id)
    dur = data["duration_ms"]
    dlg = data["dialogue"] or {}
    chans = data["channels"]
    action = _ch(chans, "action")
    camera = _ch(chans, "camera/distortion")
    beat = _ch(chans, "beat")

    field = fs.compute_fused_field(
        duration_ms=dur, energy=energy,
        dialogue_cost=dlg.get("cut_cost"), dialogue_hop=dlg.get("hop_ms", 100),
        dialogue_points=dlg.get("cut_points"),
        camera_cost=(camera or {}).get("cost"),
        action_cost=(action or {}).get("cost"),
        action_points=(action or {}).get("points"),
        motion_hop=(action or camera or {}).get("hop_ms", 100),
        beat_cost=(beat or {}).get("cost"),
        beat_points=(beat or {}).get("points"),
        beat_hop=(beat or {}).get("hop_ms", 100),
    )
    hop = field.hop_ms
    n = len(field.cost)

    print(f"\n=== {data['name']}  ({dur/1000:.1f}s, energy={energy}) ===")

    # 1) veto check
    speech_hops = [False] * n
    for w in dlg.get("words", []):
        if w.get("is_filler"):
            continue
        for i in range(w["start_ms"] // hop, w["end_ms"] // hop + 1):
            if 0 <= i < n:
                speech_hops[i] = True
    sp = [field.cost[i] for i in range(n) if speech_hops[i]]
    si = [field.cost[i] for i in range(n) if not speech_hops[i]]
    avg = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    print(f"  veto: mean fused cost  in-speech={avg(sp):.3f}  out-speech={avg(si):.3f}"
          f"   (want in >> out)")

    # 2) top seams
    print("  top seams (best cut instants):")
    for s in field.seams[:10]:
        print(f"    {s.ts_ms/1000:6.2f}s  q={s.q:.3f}  {s.kind:<14} {','.join(s.sources)}")

    # 3) snap demo: pick a rough window that starts inside a real word
    words = [w for w in dlg.get("words", []) if not w.get("is_filler")]
    if words:
        w = words[len(words) // 2]
        raw_in = (w["start_ms"] + w["end_ms"]) // 2          # mid-word (bad cut!)
        raw_out = min(dur, raw_in + 2000)
        in_ms, out_ms = fs.snap_bounds(field, raw_in, raw_out,
                                       energy=energy, duration_ms=dur)
        print(f"  snap demo: rough in={raw_in/1000:.2f}s (mid-word '{w['text']}') "
              f"-> snapped in={in_ms/1000:.2f}s "
              f"(moved {abs(in_ms-raw_in)}ms, q {field.q_at(raw_in):.2f}->{field.q_at(in_ms):.2f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-id", required=True)
    ap.add_argument("--energy", type=float, default=0.5)
    probe(*[(a := ap.parse_args()).file_id], a.energy)


if __name__ == "__main__":
    main()
