#!/usr/bin/env python3
"""Tests for color_phase1.plan.md Part 1d's _diag_qa_label_sheet.py -- pure
dict/string logic (uid, split, selection, cell rendering), no DB / ffmpeg /
real frame files.

Run:  .venv/bin/python scripts/test_diag_qa_label_sheet.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import _diag_qa_label_sheet as ls  # noqa: E402


def _shot(project_id="p1", thread_id="t1", shot_key="a000", **kw) -> dict:
    d = {
        "project_id": project_id, "project_label": kw.pop("project_label", "P"),
        "thread_id": thread_id, "shot_key": shot_key,
        "look_engine": kw.pop("look_engine", None), "is_log_flat": kw.pop("is_log_flat", False),
        "frames": kw.pop("frames", [{"is_hero": True, "raw_path": "raw.jpg", "graded_path": "graded.jpg"}]),
    }
    d.update(kw)
    return d


def _score(shot: dict, content_type="photographic", exposure_median=0.42, metrics=None) -> dict:
    m = dict(metrics or {})
    m.setdefault("exposure_band", {"verdict": "pass", "value": exposure_median, "median": exposure_median})
    return {
        "project_id": shot["project_id"], "thread_id": shot["thread_id"], "shot_key": shot["shot_key"],
        "content_type": content_type, "metrics": m,
    }


def test_shot_uid_distinguishes_same_shot_key_across_threads():
    a = _shot(project_id="p1", thread_id="t1", shot_key="a000")
    b = _shot(project_id="p2", thread_id="t2", shot_key="a000")
    assert ls._shot_uid(a) != ls._shot_uid(b)
    print("ok  _shot_uid: same shot_key in different threads/projects doesn't collide")


def test_split_for_is_deterministic():
    uid = "p1::t1::a000"
    assert ls._split_for(uid) == ls._split_for(uid)
    print("ok  _split_for: same uid always gets the same split")


def test_split_for_is_roughly_70_30():
    splits = [ls._split_for(f"p::t::shot{i}") for i in range(2000)]
    cal_frac = sum(1 for s in splits if s == "calibration") / len(splits)
    assert 0.60 < cal_frac < 0.80, cal_frac
    print(f"ok  _split_for: ~70/30 split holds at scale (calibration={cal_frac:.2f})")


def test_select_shots_keeps_both_shots_sharing_a_bare_shot_key():
    a = _shot(project_id="p1", thread_id="t1", shot_key="a000")
    b = _shot(project_id="p2", thread_id="t2", shot_key="a000")
    samples = [a, b]
    scoreboard_by_uid = {ls._shot_uid(a): _score(a), ls._shot_uid(b): _score(b)}
    selected = ls._select_shots(samples, scoreboard_by_uid)
    assert len(selected) == 2, selected
    print("ok  _select_shots: two distinct shots sharing a bare shot_key both survive")


def test_select_shots_respects_max_shots_cap():
    samples = [_shot(project_id=f"p{i}", thread_id=f"t{i}", shot_key="a000") for i in range(ls.MAX_SHOTS + 15)]
    scoreboard_by_uid = {ls._shot_uid(s): _score(s, exposure_median=0.42) for s in samples}
    selected = ls._select_shots(samples, scoreboard_by_uid)
    assert len(selected) <= ls.MAX_SHOTS, len(selected)
    print(f"ok  _select_shots: selection stays within MAX_SHOTS ({len(selected)} <= {ls.MAX_SHOTS})")


def test_select_shots_prioritizes_log_and_synthetic_and_look_extremes():
    log_shot = _shot(project_id="log", thread_id="t", shot_key="a000", is_log_flat=True)
    synth_shot = _shot(project_id="synth", thread_id="t", shot_key="a000")
    mono_look_shot = _shot(project_id="mono", thread_id="t", shot_key="a000", look_engine={"sat": 0.05})
    warm_look_shot = _shot(project_id="warm", thread_id="t", shot_key="a000", look_engine={"sat": 1.4})
    filler = [_shot(project_id=f"filler{i}", thread_id="t", shot_key="a000") for i in range(3)]
    samples = [log_shot, synth_shot, mono_look_shot, warm_look_shot] + filler
    scoreboard_by_uid = {
        ls._shot_uid(log_shot): _score(log_shot),
        ls._shot_uid(synth_shot): _score(synth_shot, content_type="synthetic"),
        ls._shot_uid(mono_look_shot): _score(mono_look_shot),
        ls._shot_uid(warm_look_shot): _score(warm_look_shot),
        **{ls._shot_uid(s): _score(s) for s in filler},
    }
    selected = ls._select_shots(samples, scoreboard_by_uid)
    uids = {ls._shot_uid(s) for s, _h, _sc in selected}
    assert ls._shot_uid(log_shot) in uids, "log shot must be included"
    assert ls._shot_uid(synth_shot) in uids, "synthetic shot must be included"
    assert ls._shot_uid(mono_look_shot) in uids, "mono look shot must be included"
    assert ls._shot_uid(warm_look_shot) in uids, "warm/saturated look shot must be included"
    print("ok  _select_shots: log, synthetic, and both look extremes are always included")


def test_select_shots_skips_candidates_missing_hero_or_score():
    no_frames = _shot(project_id="p1", thread_id="t1", shot_key="a000", frames=[])
    unscored = _shot(project_id="p2", thread_id="t2", shot_key="a000")
    samples = [no_frames, unscored]
    scoreboard_by_uid = {}  # neither is scored
    selected = ls._select_shots(samples, scoreboard_by_uid)
    assert selected == []
    print("ok  _select_shots: shots with no hero frame or no scoreboard entry are skipped")


def test_metric_cell_handles_missing_metric():
    out = ls._metric_cell("saturation_band", None)
    assert "--" in out and "saturation_band" in out
    print("ok  _metric_cell: a missing metric renders a placeholder, doesn't crash")


def test_metric_cell_renders_verdict_class():
    out = ls._metric_cell("exposure_band", {"verdict": "fail", "value": 0.9})
    assert "metric-fail" in out and "0.900" in out
    print("ok  _metric_cell: verdict maps to a CSS class and the value is formatted")


def main():
    test_shot_uid_distinguishes_same_shot_key_across_threads()
    test_split_for_is_deterministic()
    test_split_for_is_roughly_70_30()
    test_select_shots_keeps_both_shots_sharing_a_bare_shot_key()
    test_select_shots_respects_max_shots_cap()
    test_select_shots_prioritizes_log_and_synthetic_and_look_extremes()
    test_select_shots_skips_candidates_missing_hero_or_score()
    test_metric_cell_handles_missing_metric()
    test_metric_cell_renders_verdict_class()
    print("\nall _diag_qa_label_sheet tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
