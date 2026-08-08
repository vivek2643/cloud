#!/usr/bin/env python3
"""Tests for observe.inspect_cut (brain_perception_upgrade.plan.md Change 1,
Mechanism B) -- the on-demand rich signal payload behind a Beat Index line's
"sig:" breadcrumb. Builds a real moment-tree map, hand-builds an EditContext
(no DB), and stubs observe._fetch_signal_window (the one DB touch) with
synthetic L1 arrays.

Run:  .venv/bin/python scripts/test_observe_inspect_cut.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import footage_map as fm  # noqa: E402
from app.services.l3 import observe  # noqa: E402
from app.services.l3.arrange import _MapIndex  # noqa: E402

# build_clip_tree calls _said_text_for_span -> _sentences_for_file for every
# "said" cut, which would otherwise hit a real DB. Stub it process-wide, same
# as test_footage_map.py/test_observe_act.py.
fm._sentences_for_file = lambda file_id: ()

_FILE_ID = "ffffffff-1111"


def _cut(hero_id, in_ms, out_ms, **extra):
    d = {
        "hero_id": hero_id, "file_id": _FILE_ID, "channel": "shown",
        "subject": "object", "label": "watch this", "src_in_ms": in_ms, "src_out_ms": out_ms,
        "play_ms": out_ms - in_ms, "keep_spans": None, "score": 0.6, "speaker": None,
        "flags": [], "take_count": 1, "ladder": None,
    }
    d.update(extra)
    return d


def _map(cuts):
    tree = fm.build_clip_tree(_FILE_ID, {"name": "T", "duration_ms": 20000}, cuts)
    return {"clips": [tree]}


def _ctx(struct):
    return observe.EditContext(
        file_ids=[_FILE_ID], index=_MapIndex(struct), map_struct=struct,
        durations={_FILE_ID: 20000}, dup_groups=[])


def _signals(motion=None, audio=None, scene=None):
    return {"motion": motion or {}, "audio": audio or {}, "scene": scene or {}}


def test_inspect_cut_resolves_by_ref_and_windows_to_the_true_span():
    struct = _map([_cut("f:m0", 1000, 5000)])
    ctx = _ctx(struct)
    signals = _signals(
        motion={"hop_ms": 100, "action_energy": [0.1] * 200, "action_points": []},
        audio={"hop_ms": 100, "rms_db": [-40.0] * 200, "silence_intervals": []},
    )
    with mock.patch.object(observe, "_fetch_signal_window", return_value=signals):
        result = observe.inspect_cut(ctx, ref="ffffffff:m00")
    assert result["ref"] == "ffffffff:m00", result
    assert result["file"] == "ffffffff", result
    assert result["span_ms"] == 4000, result
    assert result["hop_ms"] == 100, result
    print("ok  test_inspect_cut_resolves_by_ref_and_windows_to_the_true_span")


def test_inspect_cut_resolves_by_seg_id_via_placed_document():
    struct = _map([_cut("f:m0", 1000, 5000)])
    ctx = _ctx(struct)
    document = {"timeline": [{"seg_id": "s1", "ref": "ffffffff:m00", "in_ms": 0, "out_ms": 4000}]}
    signals = _signals()
    with mock.patch.object(observe, "_fetch_signal_window", return_value=signals):
        result = observe.inspect_cut(ctx, seg_id="s1", document=document)
    assert result["ref"] == "ffffffff:m00", result
    print("ok  test_inspect_cut_resolves_by_seg_id_via_placed_document")


def test_inspect_cut_unknown_seg_id_with_no_matching_ref_errors():
    struct = _map([_cut("f:m0", 1000, 5000)])
    ctx = _ctx(struct)
    document = {"timeline": []}
    result = observe.inspect_cut(ctx, seg_id="missing", document=document)
    assert "error" in result, result
    print("ok  test_inspect_cut_unknown_seg_id_with_no_matching_ref_errors")


def test_inspect_cut_unknown_ref_errors():
    struct = _map([_cut("f:m0", 1000, 5000)])
    ctx = _ctx(struct)
    result = observe.inspect_cut(ctx, ref="ffffffff:m99")
    assert "error" in result, result
    print("ok  test_inspect_cut_unknown_ref_errors")


def test_inspect_cut_curve_never_exceeds_max_samples():
    # 20s span, 100ms hop -> 200 raw samples, must downsample to <= 24.
    struct = _map([_cut("f:m0", 0, 20000)])
    ctx = _ctx(struct)
    signals = _signals(
        motion={"hop_ms": 100, "action_energy": [float(i % 10) / 10 for i in range(200)], "action_points": []},
        audio={"hop_ms": 100, "rms_db": [-40.0 + (i % 20) for i in range(200)], "silence_intervals": []},
    )
    with mock.patch.object(observe, "_fetch_signal_window", return_value=signals):
        result = observe.inspect_cut(ctx, ref="ffffffff:m00")
    assert len(result["action"]["curve"]) <= observe._INSPECT_MAX_SAMPLES, result["action"]["curve"]
    assert len(result["audio"]["curve"]) <= observe._INSPECT_MAX_SAMPLES, result["audio"]["curve"]
    print("ok  test_inspect_cut_curve_never_exceeds_max_samples")


def test_inspect_cut_action_hits_capped_and_offsets_within_span():
    ae = [0.1] * 200
    for i in range(5, 195, 10):    # 19 well-separated interior spikes
        ae[i] = 0.9
    struct = _map([_cut("f:m0", 0, 20000)])
    ctx = _ctx(struct)
    signals = _signals(motion={"hop_ms": 100, "action_energy": ae, "action_points": []})
    with mock.patch.object(observe, "_fetch_signal_window", return_value=signals):
        result = observe.inspect_cut(ctx, ref="ffffffff:m00")
    hits = result["action"]["hits"]
    assert len(hits) <= observe._INSPECT_MAX_HITS, hits
    assert all(0 <= h["off"] < 20000 for h in hits), hits
    assert hits == sorted(hits, key=lambda h: h["off"]), hits
    print("ok  test_inspect_cut_action_hits_capped_and_offsets_within_span")


def test_inspect_cut_shots_capped_and_offsets_within_span():
    scene = {
        "hop_ms": 100,
        "shot_points": [{"ts_ms": t, "kind": "shot_cut", "score": 1.0} for t in range(500, 19500, 900)],
        "composition_points": [],
    }
    struct = _map([_cut("f:m0", 0, 20000)])
    ctx = _ctx(struct)
    signals = _signals(scene=scene)
    with mock.patch.object(observe, "_fetch_signal_window", return_value=signals):
        result = observe.inspect_cut(ctx, ref="ffffffff:m00")
    cuts = result["shots"]["cuts"]
    assert len(cuts) <= observe._INSPECT_MAX_SHOTS, cuts
    assert all(0 <= c["off"] < 20000 for c in cuts), cuts
    assert cuts == sorted(cuts, key=lambda c: c["off"]), cuts
    print("ok  test_inspect_cut_shots_capped_and_offsets_within_span")


def test_inspect_cut_silence_offsets_relative_to_cut_start():
    struct = _map([_cut("f:m0", 2000, 6000)])
    ctx = _ctx(struct)
    signals = _signals(audio={
        "hop_ms": 100, "rms_db": [-40.0] * 200,
        "silence_intervals": [{"start_ms": 3000, "end_ms": 3500}],
    })
    with mock.patch.object(observe, "_fetch_signal_window", return_value=signals):
        result = observe.inspect_cut(ctx, ref="ffffffff:m00")
    assert result["audio"]["silence"] == [{"off": 1000, "dur": 500}], result["audio"]["silence"]
    print("ok  test_inspect_cut_silence_offsets_relative_to_cut_start")


def test_inspect_cut_empty_channels_when_file_has_no_l1_rows():
    struct = _map([_cut("f:m0", 1000, 5000)])
    ctx = _ctx(struct)
    with mock.patch.object(observe, "_fetch_signal_window", return_value=_signals()):
        result = observe.inspect_cut(ctx, ref="ffffffff:m00")
    assert result["action"]["curve"] == [] and result["action"]["hits"] == [], result["action"]
    assert result["audio"]["curve"] == [] and result["audio"]["changes"] == [] \
        and result["audio"]["silence"] == [], result["audio"]
    assert result["shots"]["cuts"] == [], result["shots"]
    # hop_ms falls back to a default (never crashes) when neither table has a row.
    assert result["hop_ms"] > 0, result
    print("ok  test_inspect_cut_empty_channels_when_file_has_no_l1_rows")


# --------------------------------------------------------------------------
# brain_cut_specifics_wiring.plan.md section 3: the full, untruncated
# scene_specifics blob on demand -- unlike the beat line's capped spec: tag.
# --------------------------------------------------------------------------

def test_inspect_cut_includes_specifics_when_scene_specifics_present():
    struct = _map([_cut("f:m0", 1000, 5000, scene_specifics={"subject": "a dog", "action": "runs"})])
    ctx = _ctx(struct)
    with mock.patch.object(observe, "_fetch_signal_window", return_value=_signals()):
        result = observe.inspect_cut(ctx, ref="ffffffff:m00")
    assert result["specifics"] == {"subject": "a dog", "action": "runs"}, result
    print("ok  test_inspect_cut_includes_specifics_when_scene_specifics_present")


def test_inspect_cut_omits_specifics_key_when_scene_specifics_empty():
    struct = _map([_cut("f:m0", 1000, 5000)])  # no scene_specifics at all
    ctx = _ctx(struct)
    with mock.patch.object(observe, "_fetch_signal_window", return_value=_signals()):
        result = observe.inspect_cut(ctx, ref="ffffffff:m00")
    assert "specifics" not in result, result
    print("ok  test_inspect_cut_omits_specifics_key_when_scene_specifics_empty")


def test_inspect_cut_specifics_drops_empty_fields():
    spec = {"subject": "a dog", "action": "", "count": None, "tags": [], "notable_object": "leash"}
    struct = _map([_cut("f:m0", 1000, 5000, scene_specifics=spec)])
    ctx = _ctx(struct)
    with mock.patch.object(observe, "_fetch_signal_window", return_value=_signals()):
        result = observe.inspect_cut(ctx, ref="ffffffff:m00")
    assert result["specifics"] == {"subject": "a dog", "notable_object": "leash"}, result
    print("ok  test_inspect_cut_specifics_drops_empty_fields")


def test_inspect_cut_specifics_includes_the_full_untruncated_moments_list():
    moments = [{"t_ms": 1000 * i, "summary": f"moment {i}"} for i in range(8)]
    spec = {"subject": "barista", "moments": moments}
    struct = _map([_cut("f:m0", 0, 20000, scene_specifics=spec)])
    ctx = _ctx(struct)
    with mock.patch.object(observe, "_fetch_signal_window", return_value=_signals()):
        result = observe.inspect_cut(ctx, ref="ffffffff:m00")
    # unlike footage_map._specific_tag's capped-at-5 beat-line rendering,
    # inspect_cut carries every moment -- the whole point of "on demand".
    assert len(result["specifics"]["moments"]) == 8, result["specifics"]["moments"]
    print("ok  test_inspect_cut_specifics_includes_the_full_untruncated_moments_list")


def test_inspect_cut_specifics_legacy_shape_passes_through_too():
    struct = _map([_cut("f:m0", 1000, 5000,
                        scene_specifics={"specific": "CNC lathe turning a steel shaft", "label": "milling"})])
    ctx = _ctx(struct)
    with mock.patch.object(observe, "_fetch_signal_window", return_value=_signals()):
        result = observe.inspect_cut(ctx, ref="ffffffff:m00")
    assert result["specifics"] == {"specific": "CNC lathe turning a steel shaft", "label": "milling"}, result
    print("ok  test_inspect_cut_specifics_legacy_shape_passes_through_too")


def main():
    test_inspect_cut_resolves_by_ref_and_windows_to_the_true_span()
    test_inspect_cut_resolves_by_seg_id_via_placed_document()
    test_inspect_cut_unknown_seg_id_with_no_matching_ref_errors()
    test_inspect_cut_unknown_ref_errors()
    test_inspect_cut_curve_never_exceeds_max_samples()
    test_inspect_cut_action_hits_capped_and_offsets_within_span()
    test_inspect_cut_shots_capped_and_offsets_within_span()
    test_inspect_cut_silence_offsets_relative_to_cut_start()
    test_inspect_cut_empty_channels_when_file_has_no_l1_rows()
    test_inspect_cut_includes_specifics_when_scene_specifics_present()
    test_inspect_cut_omits_specifics_key_when_scene_specifics_empty()
    test_inspect_cut_specifics_drops_empty_fields()
    test_inspect_cut_specifics_includes_the_full_untruncated_moments_list()
    test_inspect_cut_specifics_legacy_shape_passes_through_too()
    print("\nall inspect_cut tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
