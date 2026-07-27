"""
Cross-regime eval harness for the V4 deterministic video segmenter
(cuts_content_first_segmentation.plan.md Part 7 -- "BUILD FIRST").

``segment_video`` is a pure function of already-loaded L1 signals (no DB, no
model call -- see v4_segment.py's own module docstring), so this harness
loads REAL persisted motion_dynamics/audio_features/scene_cuts rows for a
small, regime-diverse set of already-ingested files, runs the segmenter
exactly as ingest.py does, and scores the result.

Ground truth substitute: the plan's primary instruction is "hand-mark good
boundaries", which requires literally watching footage -- not something this
script (or the agent that wrote it) can do. The plan itself offers a
fallback for exactly this situation: "(or a rubric: no dead lead-in;
boundary aligns to a real content change; no cut through a coherent
action)". This harness implements that rubric, fully computed from the same
signals ``segment_video`` itself consumes -- no hand labeling. Three rubric
checks per cut, using the SAME clip-relative helpers (post._series_lohi/
_norm_in_clip/_span_slice) and the same DEAD_ENERGY_FLOOR/
REGIME_MAGNITUDE_MOVE_MIN constants v4_segment.py itself uses, so "content
change" and "dead" mean exactly what they mean inside the segmenter:

  1. no dead lead-in  -- the cut's own first stretch must clear DEAD_ENERGY_
     FLOOR on action or rms, or carry real camera magnitude.
  2. boundary aligns to a real content change -- at least one edge sits
     within a tolerance of a discrete signal mark (action_points,
     transition_points, composition_points, or a camera-move-core edge).
  3. no cut through a coherent action -- neither edge falls strictly inside
     a content-bearing sustained camera move (a real move should be kept
     whole, never split).

Regime coverage in this codebase's real DB is honest, not exhaustive: no
periodic/turntable, screen-recording, or music-bed footage happens to be
ingested here (see FIXTURES below and its trailing note) -- the harness
covers what's actually available (aerial/drone-locked, cinema gimbal/
handheld, iPhone handheld, long single takes, and two clips with real
persisted composition_points for Part 2). Re-run before/after each plan
Part; read the printed deltas by hand -- no stored baseline file, per the
plan's own "run before/after" instruction (not "diff two saved runs").

Run:  .venv/bin/python scripts/eval_cuts.py [-v]
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import v4_segment as v4  # noqa: E402
from app.services.l3.post import _norm_in_clip, _series_lohi, _span_slice  # noqa: E402
from app.services.l3.v4_segment_params import (  # noqa: E402
    DEAD_ENERGY_FLOOR, REGIME_COHERENCE_MOVE_MIN, REGIME_MAGNITUDE_MOVE_MIN,
)

# --------------------------------------------------------------------------
# Fixtures -- real, already-ingested files spanning regimes (Part 7: "Reuse
# real projects incl. factory 873d4e35"). project 873d4e35-030f-4540-82f8-
# e36ce52b386d IS the "factory" project the plan names by id -- 41 DJI aerial
# clips over an industrial site; the two picked below are representative.
# --------------------------------------------------------------------------

FIXTURES: List[Tuple[str, str]] = [
    ("aerial_drone_factory_a", "14e9e688-f630-4775-86dd-13b9044579a1"),   # DJI_0012, factory 873d4e35
    ("aerial_drone_factory_b", "febaa5e0-3913-4ef7-8d0d-39da34a6398a"),   # DJI_0050, factory 873d4e35
    ("cinema_gimbal_a", "1f0dbe7e-784b-4233-922f-af8a4540c942"),         # A_0004D462, Canon cinema
    ("cinema_gimbal_b", "0a76f68b-57ee-4937-9c63-f74148d19e7a"),         # A_0004D472, Canon cinema
    ("handheld_iphone_a", "9f21c992-31fd-49f2-bc3a-ae3b0e615ac6"),       # IMG_4628.MOV
    ("handheld_iphone_composition", "01e783cc-9fdf-4c9f-b669-b5c11e2adf33"),  # IMG_4629.MOV, has composition_points
    ("long_single_take_a", "1aedb093-9259-4deb-aa45-c8d5fba6def0"),      # C0965-003.MP4, 701s
    ("long_single_take_b", "48c93cef-0db6-4898-a45d-a343377682b8"),      # MVI_7749-002.MP4, 705s
    ("vertical_long_composition", "6cc2afc6-b08d-4fcd-9608-6cfffd03550b"),  # video1864710564.mp4, has shot+composition points
]
# No periodic (turntable/reps/waves), screen-recording, or music-bed clip is
# currently ingested in this DB -- Part 6 (period-aware cut) and the
# music/beat regime are exercised by scripts/test_v4_segment.py's synthetic
# unit tests instead, not this harness.

LEAD_IN_CHECK_MS = 400
ALIGN_TOLERANCE_MS = 400


def _load_fixture(file_id: str) -> Optional[Dict[str, Any]]:
    from app.services import db

    with db.connection_dict_row() as conn:
        f = conn.execute(
            "select duration_seconds from files where id = %s", (file_id,)
        ).fetchone()
        if not f or not f["duration_seconds"]:
            return None
        m = conn.execute(
            "select hop_ms, action_energy, camera_motion, camera_coherence, "
            "camera_stability, blur, action_points, transition_points, "
            "camera_dx, camera_dy, camera_zoom from motion_dynamics where file_id = %s",
            (file_id,),
        ).fetchone()
        if not m:
            return None
        af = conn.execute(
            "select rms_db, prosody_hop_ms, onsets_ms, is_musical from audio_features "
            "where file_id = %s", (file_id,),
        ).fetchone()
        sc = conn.execute(
            "select hop_ms, shot_points, composition_points from scene_cuts "
            "where file_id = %s", (file_id,),
        ).fetchone()

    motion = dict(m)
    audio = ({"rms_db": af["rms_db"] or [], "hop_ms": af["prosody_hop_ms"] or 0,
              "onsets_ms": af["onsets_ms"] or [], "is_musical": bool(af["is_musical"])}
             if af else {})
    scene = dict(sc) if sc else {}
    return {
        "file_id": file_id, "duration_ms": int(f["duration_seconds"] * 1000),
        "motion": motion, "audio": audio, "scene": scene,
    }


# --------------------------------------------------------------------------
# Rubric (fallback for hand-marked ground truth -- see module docstring)
# --------------------------------------------------------------------------

def _content_marks(motion: dict, scene: dict, hop_ms: int, duration_ms: int,
                    ae_lohi: Tuple[Optional[float], Optional[float]]) -> List[int]:
    """Every discrete "something real happens here" instant this fixture's
    signals carry: action_points, transition_points, composition_points
    (Part 2+), every sustained camera-move-core's onset/offset (Part 1's own
    clip-relative gate, via v4._clip_move_threshold -- so this stays honest
    as that gate changes), and every action-run/lull boundary (Part 3, via
    v4._action_runs_and_lulls -- reusing the segmenter's OWN mechanism
    rather than re-deriving a second, possibly-diverging notion of "a
    content edge")."""
    marks = [int(p["ts_ms"]) for p in (motion.get("action_points") or []) if "ts_ms" in p]
    marks += [int(p["ts_ms"]) for p in (motion.get("transition_points") or []) if "ts_ms" in p]
    marks += [int(p["ts_ms"]) for p in (scene.get("composition_points") or []) if "ts_ms" in p]
    marks += [int(p["ts_ms"]) for p in (scene.get("shot_points") or []) if "ts_ms" in p]

    move_threshold = v4._clip_move_threshold(motion)
    dx = motion.get("camera_dx") or []
    dy = motion.get("camera_dy") or []
    dz = motion.get("camera_zoom") or []
    n = max(len(dx), len(dy), len(dz))
    moving = [
        (abs(dx[i] if i < len(dx) else 0.0) + abs(dy[i] if i < len(dy) else 0.0)
         + abs(dz[i] if i < len(dz) else 0.0)) >= move_threshold
        for i in range(n)
    ]
    i = 0
    while i < n:
        if not moving[i]:
            i += 1
            continue
        j = i
        while j < n and moving[j]:
            j += 1
        marks.append(i * hop_ms)
        marks.append(j * hop_ms)
        i = j

    for run_s, run_e in v4._action_runs_and_lulls((0, duration_ms), motion, hop_ms, ae_lohi):
        marks.append(run_s)
        marks.append(run_e)
    return marks


def _has_energy(motion: dict, audio_rms_at_hop: List[float], hop_ms: int, s: int, e: int,
                 ae_lohi: Tuple[Optional[float], Optional[float]],
                 rms_lohi: Tuple[Optional[float], Optional[float]]) -> bool:
    action = _span_slice(motion.get("action_energy") or [], hop_ms, s, e)
    ae_lo, ae_hi = ae_lohi
    if any((_norm_in_clip(v, ae_lo, ae_hi) or 0.0) >= DEAD_ENERGY_FLOOR for v in action):
        return True
    rms = _span_slice(audio_rms_at_hop, hop_ms, s, e)
    rms_lo, rms_hi = rms_lohi
    if any((_norm_in_clip(v, rms_lo, rms_hi) or 0.0) >= DEAD_ENERGY_FLOOR for v in rms):
        return True
    dx = _span_slice(motion.get("camera_dx") or [], hop_ms, s, e)
    dy = _span_slice(motion.get("camera_dy") or [], hop_ms, s, e)
    dz = _span_slice(motion.get("camera_zoom") or [], hop_ms, s, e)
    n = max(len(dx), len(dy), len(dz))
    for i in range(n):
        mag = ((abs(dx[i]) if i < len(dx) else 0.0) + (abs(dy[i]) if i < len(dy) else 0.0)
               + (abs(dz[i]) if i < len(dz) else 0.0))
        if mag >= REGIME_MAGNITUDE_MOVE_MIN:
            return True
    return False


def _move_cores_containing_interior(motion: dict, hop_ms: int, cut_in: int, cut_out: int) -> bool:
    """True when either edge of [cut_in, cut_out) falls STRICTLY inside a
    sustained camera-move core spanning the whole file (never at its own
    onset/offset, which is a legitimate cut edge). Part 1's own clip-relative
    magnitude gate (v4._clip_move_threshold), so this rubric check stays
    honest as that gate changes."""
    move_threshold = v4._clip_move_threshold(motion)
    dx = motion.get("camera_dx") or []
    dy = motion.get("camera_dy") or []
    dz = motion.get("camera_zoom") or []
    coh = motion.get("camera_coherence") or []
    n = max(len(dx), len(dy), len(dz))
    moving = [
        (abs(dx[i] if i < len(dx) else 0.0) + abs(dy[i] if i < len(dy) else 0.0)
         + abs(dz[i] if i < len(dz) else 0.0)) >= move_threshold
        and (coh[i] if i < len(coh) else 0.0) >= REGIME_COHERENCE_MOVE_MIN
        for i in range(n)
    ]
    i = 0
    while i < n:
        if not moving[i]:
            i += 1
            continue
        j = i
        while j < n and moving[j]:
            j += 1
        core_s, core_e = i * hop_ms, j * hop_ms
        if core_e - core_s >= 500:
            for edge in (cut_in, cut_out):
                if core_s < edge < core_e:
                    return True
        i = j
    return False


def score_cuts(fixture: Dict[str, Any], cuts: List[v4.VideoCut]) -> Dict[str, Any]:
    motion, audio, scene = fixture["motion"], fixture["audio"], fixture["scene"]
    hop_ms = int(motion.get("hop_ms") or 0)
    duration_ms = fixture["duration_ms"]
    if not cuts or hop_ms <= 0:
        return {"n_cuts": len(cuts), "coverage_pct": 0.0, "dead_lead_in_rate": None,
                "content_aligned_rate": None, "action_split_rate": None, "avg_cut_ms": 0.0}

    ae_lohi = _series_lohi(motion.get("action_energy") or [])
    rms = audio.get("rms_db") or []
    rms_hop_ms = int(audio.get("hop_ms") or 0)
    rms_lohi = _series_lohi(rms)
    n_motion = (duration_ms // hop_ms) + 1
    rms_at_hop = ([rms[min(len(rms) - 1, (i * hop_ms) // rms_hop_ms)] for i in range(n_motion)]
                  if rms and rms_hop_ms > 0 else [])

    marks = sorted(_content_marks(motion, scene, hop_ms, duration_ms, ae_lohi))

    dead_lead_ins = 0
    aligned = 0
    split_through_action = 0
    total_cut_ms = 0
    for c in cuts:
        total_cut_ms += c.src_out_ms - c.src_in_ms
        lead_end = min(c.src_out_ms, c.src_in_ms + LEAD_IN_CHECK_MS)
        if not _has_energy(motion, rms_at_hop, hop_ms, c.src_in_ms, lead_end, ae_lohi, rms_lohi):
            dead_lead_ins += 1
        if marks and any(abs(c.src_in_ms - m) <= ALIGN_TOLERANCE_MS
                          or abs(c.src_out_ms - m) <= ALIGN_TOLERANCE_MS for m in marks):
            aligned += 1
        if _move_cores_containing_interior(motion, hop_ms, c.src_in_ms, c.src_out_ms):
            split_through_action += 1

    n = len(cuts)
    return {
        "n_cuts": n,
        "coverage_pct": round(100.0 * total_cut_ms / max(1, duration_ms), 1),
        "dead_lead_in_rate": round(dead_lead_ins / n, 2),
        "content_aligned_rate": (round(aligned / n, 2) if marks else None),
        "action_split_rate": round(split_through_action / n, 2),
        "avg_cut_ms": round(total_cut_ms / n, 0),
    }


def run(verbose: bool = False) -> None:
    rows = []
    for label, file_id in FIXTURES:
        fixture = _load_fixture(file_id)
        if fixture is None:
            print(f"skip  {label} ({file_id}): no persisted L1 signals")
            continue
        cuts = v4.segment_video(
            file_id=fixture["file_id"], duration_ms=fixture["duration_ms"],
            speech_spans=[], motion=fixture["motion"], audio=fixture["audio"],
            scene=fixture["scene"],
        )
        stats = score_cuts(fixture, cuts)
        rows.append((label, stats))
        if verbose:
            for c in cuts:
                print(f"    {label}: [{c.src_in_ms:>7}-{c.src_out_ms:<7}] "
                      f"kind={c.salience['kind']:<6} events={len(c.salience['events'])}")

    header = f"{'fixture':<30} {'cuts':>5} {'cov%':>6} {'deadlead':>9} {'aligned':>8} {'actsplit':>9} {'avgms':>7}"
    print(header)
    print("-" * len(header))
    for label, s in rows:
        def _fmt(v):
            return "n/a" if v is None else v
        print(f"{label:<30} {s['n_cuts']:>5} {s['coverage_pct']:>6} "
              f"{_fmt(s['dead_lead_in_rate']):>9} {_fmt(s['content_aligned_rate']):>8} "
              f"{_fmt(s['action_split_rate']):>9} {s['avg_cut_ms']:>7}")

    zero_cut = [label for label, s in rows if s["n_cuts"] == 0]
    if zero_cut:
        print(f"\nzero-cut fixtures (fail per the plan's coverage framing): {zero_cut}")


if __name__ == "__main__":
    run(verbose="-v" in sys.argv or "--verbose" in sys.argv)
