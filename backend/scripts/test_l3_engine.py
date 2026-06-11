"""
Self-contained sanity tests for the L3 deterministic cut-engine.

Builds synthetic cut-cost grids (no DB, no LLM, no video) shaped like a real
dialogue clip -- speech plateaus at cost ~1.0 with clean valleys between
sentences -- and asserts the engine's core promises:

  1. snap_cut moves a mid-word cut into the nearest valley.
  2. query_seams prefers a clean nearby seam over a cleaner distant one.
  3. make_segment snaps both ends + flags degenerate spans.
  4. timeline_status totals durations and flags jump cuts.
  5. fit_duration trims lowest-priority segments onto seams to hit a target.

Run:  cd backend && python3 scripts/test_l3_engine.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.l3.engine import (  # noqa: E402
    ClipGrids,
    fit_duration,
    make_segment,
    query_seams,
    snap_cut,
    timeline_status,
)

HOP = 100  # ms


def synthetic_dialogue_clip(file_id: str = "clipA", duration_ms: int = 30_000) -> ClipGrids:
    """Speech blocks (cost 1.0) separated by gap valleys; valley centers carry
    discrete word_gap snap points, mirroring what L1 actually emits."""
    n = duration_ms // HOP
    cost = [1.0] * n
    valleys_ms = [4_000, 9_000, 15_000, 21_000, 27_000]
    for v in valleys_ms:
        c = v // HOP
        for off, val in ((-3, 0.6), (-2, 0.35), (-1, 0.15), (0, 0.05), (1, 0.15), (2, 0.35), (3, 0.6)):
            i = c + off
            if 0 <= i < n:
                cost[i] = min(cost[i], val)
    grids = ClipGrids(file_id=file_id, duration_ms=duration_ms)
    grids.channels["dialogue"] = (HOP, cost)
    grids.points["dialogue"] = [
        {"ts_ms": v, "kind": "sentence_end", "score": 0.9} for v in valleys_ms
    ]
    return grids


def main() -> None:
    g = synthetic_dialogue_clip()
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    print("1) snap_cut rescues a mid-word cut")
    snapped = snap_cut(g, 8_500, axis="speech")  # mid-speech, valley at 9000
    check("snaps to the 9000ms valley", snapped["ts_ms"] == 9_000, f"got {snapped}")
    check("seam is clean", snapped["cost"] <= 0.1 and not snapped["dirty"])

    print("2) query_seams prefers near-clean over far-cleaner")
    seams = query_seams(g, 14_200, axis="speech", window_ms=2_500)
    check("top seam is the 15000ms valley", seams and seams[0]["ts_ms"] == 15_000,
          f"got {[s['ts_ms'] for s in seams[:3]]}")

    print("3) snap inside dead-center speech is honest about dirtiness")
    bad = snap_cut(g, 12_000, axis="speech", window_ms=500)  # nearest valley 3000ms away
    check("flags dirty + warns", bad["dirty"] and "warning" in bad, f"got {bad}")

    print("4) make_segment snaps both ends")
    seg = make_segment(g, 3_700, 9_300, axis="speech", content="first soundbite", priority=1)
    check("in->4000 / out->9000", seg["in_ms"] == 4_000 and seg["out_ms"] == 9_000,
          f"got {seg['in_ms']}->{seg['out_ms']}")
    check("costs recorded", seg["cut_in_cost"] <= 0.1 and seg["cut_out_cost"] <= 0.1)

    print("5) timeline_status flags jump cuts + totals")
    seg2 = make_segment(g, 9_800, 15_200, axis="speech", content="second", priority=3)
    status = timeline_status([seg, seg2])
    check("total duration sums", status["total_ms"] ==
          (seg["out_ms"] - seg["in_ms"]) + (seg2["out_ms"] - seg2["in_ms"]))
    check("jump cut flagged", any("jump cut" in w for w in status["warnings"]),
          f"warnings={status['warnings']}")

    print("6) fit_duration trims low priority onto seams")
    seg3 = make_segment(g, 15_000, 27_000, axis="speech", content="long b-roll", priority=5)
    timeline = [seg, seg2, seg3]
    before = timeline_status(timeline)["total_ms"]
    target = before - 5_000
    fitted, report = fit_duration(timeline, {"clipA": g}, target_ms=target)
    after = timeline_status(fitted)["total_ms"]
    check("hit target within tolerance", report["fitted"] and after <= target + 500,
          f"before={before} target={target} after={after}")
    check("trimmed the low-priority segment first",
          report["moves"] and report["moves"][0]["seg_id"] == seg3["seg_id"],
          f"moves={report['moves']}")
    snapped_out = report["moves"][0]["new_out_ms"] if report["moves"] else None
    check("new out landed on a valley seam", snapped_out in (21_000, 15_000, 27_000),
          f"got {snapped_out}")
    check("original timeline untouched",
          timeline[2]["out_ms"] == seg3["out_ms"])

    print("7) tool executor: full simulated agent session (no DB, no LLM)")
    import json

    from app.services.l3.tools import EditSession, execute_tool

    session = EditSession(thread_id="t-test", file_ids=["clipA"], catalog=[])
    session._grids["clipA"] = g  # pre-seed so no DB load happens

    def call(name: str, **args) -> dict:
        return json.loads(execute_tool(session, name, args))

    r = call("set_brief", goal="30s teaser", target_duration_s=12,
             assumptions=["assumed energetic tone"])
    check("set_brief", r.get("ok") is True)

    r = call("set_outline", beats=[
        {"beat_id": "b1", "purpose": "hook", "intent": "open mid-speech"},
        {"beat_id": "b2", "purpose": "payoff", "intent": "land the point"},
    ])
    check("set_outline", r.get("beat_count") == 2)

    r1 = call("add_segment", file_id="clipA", in_ms=3_700, out_ms=9_300,
              axis="speech", beat_id="b1", content="first soundbite", priority=1)
    r2 = call("add_segment", file_id="clipA", in_ms=14_800, out_ms=27_200,
              axis="speech", beat_id="b2", content="long payoff", priority=4)
    check("segments snapped on add",
          r1["segment"]["in_ms"] == 4_000 and r2["segment"]["out_ms"] == 27_000,
          f"{r1['segment']['in_ms']}->{r1['segment']['out_ms']}, "
          f"{r2['segment']['in_ms']}->{r2['segment']['out_ms']}")

    status = call("timeline_status")
    check("status sees 2 segments", status["segment_count"] == 2)

    r = call("fit_duration", target_s=12)
    check("fit hits 12s", r["report"]["fitted"] and r["status"]["total_s"] <= 12.5,
          f"total={r['status']['total_s']}s moves={r['report']['moves']}")

    r = call("ask_user", questions=[
        {"q_id": "q1", "question": "End on the laugh or the line?", "default": "the laugh"},
    ])
    check("ask_user records questions", r.get("paused") is True
          and session.document["open_questions"][0]["q_id"] == "q1")

    r = call("finalize", summary="12s two-beat teaser, dialogue-snapped.")
    check("finalize stamps diagnostics", r.get("finalized") is True
          and session.document["diagnostics"]["segment_count"] == 2)

    r = call("update_segment", seg_id="nope", out_ms=1)
    check("unknown seg_id is a soft error", "error" in r)

    r = call("read_clip", file_id="not-in-scope")
    check("scope enforced on read_clip", "error" in r)

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): {failures}")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
