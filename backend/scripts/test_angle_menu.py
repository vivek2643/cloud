"""
Regression test for the focus-driven angle facts layer (l3.focus + l3.angle_menu
+ the l3.sync candidate pre-screen).

All DB-free:
  * the focus extractors and the angle menu are pure functions over synthetic
    signals / perceptions, so the speaker-follow behavior is pinned without any
    video or database;
  * the audio pre-screen is exercised on the SAME frozen interview envelopes the
    sync regression uses, asserting it NOMINATES the one true synced pair and
    ranks it above the distinct-content clips.

What this guards:
  * dialogue focus -> the menu flags the SPEAKER camera vs. the LISTENER camera
    per turn, using each camera's OWN speaking spans (no cross-clip identity),
    and surfaces a reaction on the listener -- the anti-collapse signal;
  * the sync offset re-bases the angle's clock correctly;
  * the focus interface is genre-general (action focus produces beats with no
    dialogue at all);
  * single-camera scope yields NO synced angle (the menu pipeline stays dormant).

Run:  cd backend && python3 scripts/test_angle_menu.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.l3 import angle_menu, focus, sync  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sync_envs_interview.json")


def _role(row, file_id):
    for s in row.shots:
        if s.file_id == file_id:
            return s
    raise AssertionError(f"{file_id} not in row")


def test_speaker_focus_follows_the_talker() -> None:
    """Two cameras, no shared identity labels: camera A frames whoever speaks
    first, camera B frames whoever speaks second. The menu must mark the speaker
    camera 'speaker' and the other 'listener', and surface B's reaction in the
    first turn."""
    # Diarized turns on the spine clip (S1 then S0) -> the focus timeline.
    sig = focus.FocusSignals(
        file_id="A",
        turns=[(0, 5000, "S1"), (5000, 10000, "S0")],
        action_points=[], sections=[],
    )
    intervals = focus.focus_timeline("speaker", sig, 0, 10000)
    assert [iv.label for iv in intervals] == ["S1", "S0"], intervals

    perceptions = {
        "A": {
            "persons": [{"local_id": "p1"}],
            "speaking": [{"start_ms": 0, "end_ms": 5000, "subject": "p1"}],
            "camera_craft": [{"start_ms": 0, "end_ms": 10000, "shot_size": "medium_close_up"}],
        },
        "B": {
            "persons": [{"local_id": "p1"}],
            "speaking": [{"start_ms": 5000, "end_ms": 10000, "subject": "p1"}],
            "reactions": [{"start_ms": 2000, "end_ms": 3000, "subject": "p1",
                           "type": "nod", "intensity": 0.7}],
            "camera_craft": [{"start_ms": 0, "end_ms": 10000, "shot_size": "close_up"}],
        },
    }
    rows = angle_menu.build_angle_menu("A", {"B": 0}, intervals, perceptions)
    assert len(rows) == 2

    r0 = rows[0]  # turn S1
    assert _role(r0, "A").role == "speaker", _role(r0, "A")
    assert _role(r0, "B").role == "listener", _role(r0, "B")
    assert _role(r0, "B").reaction and "nod" in _role(r0, "B").reaction

    r1 = rows[1]  # turn S0
    assert _role(r1, "A").role == "listener", _role(r1, "A")
    assert _role(r1, "B").role == "speaker", _role(r1, "B")
    print("  OK  speaker focus marks speaker vs listener camera per turn (+ reaction)")


def test_sync_offset_rebases_the_angle_clock() -> None:
    """B starts 2 s after A in A's clock, so the same instant in B is (A-2000).
    A speaking-span placed in B's clock must line up after re-basing."""
    sig = focus.FocusSignals("A", turns=[(4000, 6000, "S0")], action_points=[], sections=[])
    intervals = focus.focus_timeline("speaker", sig, 4000, 6000)
    perceptions = {
        "A": {"persons": [{"local_id": "p1"}]},
        # In B's OWN clock the speech is at 2000-4000; with offset +2000 that maps
        # to A-time 4000-6000, the focus window. So B must read as 'speaker'.
        "B": {"persons": [{"local_id": "p1"}],
              "speaking": [{"start_ms": 2000, "end_ms": 4000, "subject": "p1"}]},
    }
    rows = angle_menu.build_angle_menu("A", {"B": 2000}, intervals, perceptions)
    assert _role(rows[0], "B").role == "speaker", _role(rows[0], "B")
    print("  OK  verified sync offset re-bases the angle clock correctly")


def test_action_focus_is_dialogue_free() -> None:
    """The focus interface is not interview-specific: an action spine with zero
    dialogue still yields beats delimited by motion impacts."""
    sig = focus.FocusSignals("A", turns=[], action_points=[{"ts_ms": 3000}, {"ts_ms": 7000}],
                             sections=[])
    assert focus.infer_focus_kind(sig) == "action"
    intervals = focus.focus_timeline("action", sig, 0, 10000)
    assert [iv.kind for iv in intervals] == ["action", "action", "action"], intervals
    assert intervals[0].start_ms == 0 and intervals[1].start_ms == 3000
    print("  OK  action focus produces motion beats with no transcript")


def test_prescreen_then_verify_keeps_only_the_true_pair() -> None:
    """The full discovery pipeline, DB-free: the cheap pre-screen must NOMINATE
    the real synced pair (ranked top), and the airtight verify must then keep
    ONLY it -- the distinct-content clips a permissive screen also nominates are
    all rejected by align_envs."""
    raw = json.load(open(FIXTURE))
    envs = {name: d["sync_env"] for name, d in raw.items()}
    ref = "C0965"
    others = ["MVI_7749", "C0962", "C0964", "C0963", "C0967", "MVI_7752"]

    # Pre-screen: coarse score per pair, nominate everything above the floor.
    decim = sync._decim_for(*(len(envs[n]) for n in envs))
    scored = {n: sync._coarse_corr(envs[n], envs[ref], decim) for n in others}
    nominated = [n for n, c in scored.items() if c >= sync.CANDIDATE_CORR_FLOOR]
    assert "MVI_7749" in nominated, f"true pair not nominated: {scored}"
    assert max(scored, key=scored.get) == "MVI_7749", (
        f"true pair should rank highest in the screen, got {scored}"
    )

    # Verify: only the true pair survives the airtight gates.
    survivors = [n for n in nominated
                 if sync.align_envs((500, envs[n], None), (500, envs[ref], None)) is not None]
    assert survivors == ["MVI_7749"], f"verification should keep only the true pair, got {survivors}"
    print(f"  OK  pre-screen+verify keeps only the true pair "
          f"(screen {scored['MVI_7749']:.3f}, {len(nominated)} nominated, 1 survivor)")


def test_single_camera_has_no_synced_block() -> None:
    """No verified angle -> empty summary -> dormant pipeline (byte-identical to
    the pre-multicam path for single-source projects)."""
    assert angle_menu.render_synced_angles_text([]) == ""
    print("  OK  single-camera scope surfaces no SYNCED ANGLES block")


def main() -> None:
    print("angle facts layer regression:")
    test_speaker_focus_follows_the_talker()
    test_sync_offset_rebases_the_angle_clock()
    test_action_focus_is_dialogue_free()
    test_prescreen_then_verify_keeps_only_the_true_pair()
    test_single_camera_has_no_synced_block()
    print("ALL PASS")


if __name__ == "__main__":
    main()
