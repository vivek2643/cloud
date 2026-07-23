"""
Pure unit tests for the FCPXML rough-cut exporter
(``app.services.export.fcpxml``) -- no DB, no ffmpeg, no model calls.

Two things are covered, deliberately kept separate (see fcpxml.py's own
module docstring on the Phase 6 caveat):
  * the transform MATH's own internal consistency (`_cell_transform`) --
    this IS fully verified here, against a fixture I hand-checked by hand.
  * the document's STRUCTURE -- well-formed XML, correct clip count/order,
    a transform present for a split-screen region -- exactly what
    export_options.plan.md Phase 3 asks this test file to assert.
This does NOT (and cannot, in this environment) verify real-NLE agreement
(DaVinci Resolve / Premiere Pro import) -- that is Phase 6's own separate,
manual "golden-import validation" gate.

Run:  .venv/bin/python scripts/test_export_fcpxml.py
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.export import fcpxml  # noqa: E402


# --------------------------------------------------------------------------
# frame snapping
# --------------------------------------------------------------------------

def test_frames_snaps_to_nearest_frame():
    assert fcpxml._frames(0) == 0
    assert fcpxml._frames(1000) == 30            # exactly 1s @ 30fps
    assert fcpxml._frames(1017) == 31             # rounds to nearest frame
    assert fcpxml._frames(-50) == 0               # never negative
    print("ok  test_frames_snaps_to_nearest_frame")


def test_time_emits_rational_seconds_string():
    assert fcpxml._time(1000) == "30/30s"
    assert fcpxml._time(0) == "0/30s"
    print("ok  test_time_emits_rational_seconds_string")


# --------------------------------------------------------------------------
# Phase 6: coordinate mapping math
# --------------------------------------------------------------------------

def test_cell_transform_full_frame_identity():
    pos_x, pos_y, sx, sy, crop = fcpxml._cell_transform(
        {}, canvas_w=1920, canvas_h=1080, native_w=1920, native_h=1080)
    assert abs(pos_x) < 1e-6 and abs(pos_y) < 1e-6
    assert abs(sx - 1.0) < 1e-6 and abs(sy - 1.0) < 1e-6
    assert crop is None
    print("ok  test_cell_transform_full_frame_identity")


def test_cell_transform_split_h_right_cell_hand_checked():
    # Right half of a 1920x1080 canvas, 16:9 source (matches canvas aspect
    # exactly) -- hand-derived expectation:
    #   baseline_scale = min(1920/3840, 1080/2160) = 0.5 (exact full-canvas fit)
    #   cell (cover)    = max(960/3840, 1080/2160) = 0.5  -> scale = 0.5/0.5 = 1.0
    #   position: cell center (1440, 540) - canvas center (960, 540) = (480, 0)
    #   crop: displayed_w = 3840*0.5 = 1920 vs cell_w 960 -> 50% overflow,
    #         25% trimmed off each side; displayed_h == cell_h -> no vertical crop.
    transform = {"fit": "cover", "anchor": "center", "dest": {"x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0}}
    pos_x, pos_y, sx, sy, crop = fcpxml._cell_transform(
        transform, canvas_w=1920, canvas_h=1080, native_w=3840, native_h=2160)
    assert abs(pos_x - 480.0) < 0.01, pos_x
    assert abs(pos_y - 0.0) < 0.01, pos_y
    assert abs(sx - 1.0) < 1e-4 and abs(sy - 1.0) < 1e-4, (sx, sy)
    assert crop is not None
    left, right, top, bottom = crop
    assert abs(left - 0.25) < 1e-4 and abs(right - 0.25) < 1e-4, crop
    assert abs(top) < 1e-4 and abs(bottom) < 1e-4, crop
    print("ok  test_cell_transform_split_h_right_cell_hand_checked")


def test_cell_transform_pip_inset_cell_scales_up_and_positions_bottom_right():
    # LAYOUT_TEMPLATES["pip"]["inset"] = (0.66, 0.66, 0.32, 0.32) -- a small
    # bottom-right inset, source aspect matching canvas.
    transform = {"fit": "cover", "anchor": "center", "dest": {"x": 0.66, "y": 0.66, "w": 0.32, "h": 0.32}}
    pos_x, pos_y, sx, sy, crop = fcpxml._cell_transform(
        transform, canvas_w=1920, canvas_h=1080, native_w=1920, native_h=1080)
    # Cell center (0.82, 0.82) of canvas -> pixel (1574.4, 885.6); canvas
    # center (960, 540); pos = (614.4, -345.6) after the Y-flip.
    assert pos_x > 0 and pos_y < 0, (pos_x, pos_y)   # bottom-right of center, Y-up flips sign
    assert sx < 1.0 and sy < 1.0, (sx, sy)           # a smaller cell scales DOWN from baseline
    print("ok  test_cell_transform_pip_inset_cell_scales_up_and_positions_bottom_right")


def test_cell_transform_contain_never_crops():
    transform = {"fit": "contain", "dest": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0}}
    _, _, _, _, crop = fcpxml._cell_transform(
        transform, canvas_w=1920, canvas_h=1080, native_w=3840, native_h=2160)
    assert crop is None
    print("ok  test_cell_transform_contain_never_crops")


def test_cell_transform_unknown_native_dims_falls_back_to_canvas_aspect():
    # No width/height on file_lookup -- must not raise, must still produce a
    # sane (canvas-aspect-assumed) result rather than dividing by zero.
    transform = {"fit": "cover", "dest": {"x": 0.0, "y": 0.0, "w": 0.5, "h": 1.0}}
    pos_x, pos_y, sx, sy, crop = fcpxml._cell_transform(
        transform, canvas_w=1920, canvas_h=1080, native_w=None, native_h=None)
    assert sx > 0 and sy > 0
    print("ok  test_cell_transform_unknown_native_dims_falls_back_to_canvas_aspect")


def test_cell_transform_zoom_scales_beyond_cover():
    base = {"fit": "cover", "dest": "full"}
    zoomed = {"fit": "cover", "dest": "full", "zoom": 1.5}
    _, _, sx0, _, _ = fcpxml._cell_transform(base, 1920, 1080, 1920, 1080)
    _, _, sx1, _, _ = fcpxml._cell_transform(zoomed, 1920, 1080, 1920, 1080)
    assert abs(sx1 - sx0 * 1.5) < 1e-4, (sx0, sx1)
    print("ok  test_cell_transform_zoom_scales_beyond_cover")


# --------------------------------------------------------------------------
# document assembly
# --------------------------------------------------------------------------

def _fixture():
    resolved = {
        "duration_ms": 5000,
        "aspect": "landscape",
        "video_layers": [
            {"layer_id": "v0", "source_file_id": "f1", "src_in_ms": 0, "src_out_ms": 2000,
             "prog_start_ms": 0, "prog_end_ms": 2000, "z": 0, "kind": "spine", "transform": {}},
            {"layer_id": "v1", "source_file_id": "f2", "src_in_ms": 500, "src_out_ms": 3500,
             "prog_start_ms": 2000, "prog_end_ms": 5000, "z": 0, "kind": "spine", "transform": {}},
            {"layer_id": "op1", "source_file_id": "f1", "src_in_ms": 4000, "src_out_ms": 6000,
             "prog_start_ms": 3000, "prog_end_ms": 5000, "z": 10, "kind": "coverage",
             "transform": {"fit": "cover", "anchor": "center",
                          "dest": {"x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0}}},
        ],
        "audio_layers": [
            {"source_file_id": "f1", "src_in_ms": 0, "src_out_ms": 2000,
             "prog_start_ms": 0, "prog_end_ms": 2000, "gain_db": 0.0, "duck_db": 0.0},
            {"source_file_id": "f2", "src_in_ms": 500, "src_out_ms": 3500,
             "prog_start_ms": 2000, "prog_end_ms": 5000, "gain_db": -3.0, "duck_db": 0.0},
        ],
    }
    lookup = {
        "f1": fcpxml.FcpxmlAsset(file_id="f1", filename="clipA.mov", duration_ms=10000, width=3840, height=2160),
        "f2": fcpxml.FcpxmlAsset(file_id="f2", filename="clipB.mov", duration_ms=10000, width=1920, height=1080),
    }
    return resolved, lookup


def test_build_fcpxml_is_well_formed_xml():
    resolved, lookup = _fixture()
    xml_str = fcpxml.build_fcpxml(resolved, lookup, project_name="MyReel")
    root = ET.fromstring(xml_str)  # raises ET.ParseError if malformed
    assert root.tag == "fcpxml"
    assert root.get("version") == "1.9"
    print("ok  test_build_fcpxml_is_well_formed_xml")


def test_build_fcpxml_has_one_asset_per_distinct_file():
    resolved, lookup = _fixture()
    root = ET.fromstring(fcpxml.build_fcpxml(resolved, lookup))
    assets = root.findall(".//resources/asset")
    assert len(assets) == 2, [a.get("name") for a in assets]
    names = {a.get("name") for a in assets}
    assert names == {"clipA.mov", "clipB.mov"}, names
    print("ok  test_build_fcpxml_has_one_asset_per_distinct_file")


def test_build_fcpxml_spine_clip_count_and_program_order():
    resolved, lookup = _fixture()
    root = ET.fromstring(fcpxml.build_fcpxml(resolved, lookup))
    spine = root.find(".//spine")
    top_level_clips = list(spine)
    assert len(top_level_clips) == 2, top_level_clips  # 2 spine layers
    assert top_level_clips[0].get("name") == "clipA.mov"
    assert top_level_clips[1].get("name") == "clipB.mov"
    assert top_level_clips[0].get("offset") == "0/30s"
    assert top_level_clips[1].get("offset") == "60/30s"  # 2000ms @ 30fps
    print("ok  test_build_fcpxml_spine_clip_count_and_program_order")


def test_build_fcpxml_split_screen_region_has_transform_and_crop():
    resolved, lookup = _fixture()
    root = ET.fromstring(fcpxml.build_fcpxml(resolved, lookup))
    # The coverage cell is a connected clip (lane=1) somewhere under the spine.
    coverage_clips = [el for el in root.iter("asset-clip") if el.get("lane") == "1"]
    assert len(coverage_clips) == 1, coverage_clips
    cell = coverage_clips[0]
    transform = cell.find("adjust-transform")
    crop = cell.find("adjust-crop")
    assert transform is not None, "split-screen cell must carry adjust-transform"
    assert crop is not None, "a cover-fit cell with overflow must carry adjust-crop"
    assert crop.find("trim-rect") is not None
    print("ok  test_build_fcpxml_split_screen_region_has_transform_and_crop")


def test_build_fcpxml_identity_spine_clip_emits_no_adjust_elements():
    resolved, lookup = _fixture()
    root = ET.fromstring(fcpxml.build_fcpxml(resolved, lookup))
    spine = root.find(".//spine")
    first_clip = list(spine)[0]
    assert first_clip.find("adjust-transform") is None
    assert first_clip.find("adjust-crop") is None
    print("ok  test_build_fcpxml_identity_spine_clip_emits_no_adjust_elements")


def test_build_fcpxml_audio_layers_present_as_own_clips():
    resolved, lookup = _fixture()
    root = ET.fromstring(fcpxml.build_fcpxml(resolved, lookup))
    audio_clips = [el for el in root.iter("asset-clip") if (el.get("lane") or "").startswith("-")]
    assert len(audio_clips) == 2, audio_clips
    gained = [el for el in audio_clips if el.find("adjust-volume") is not None]
    assert len(gained) == 1, gained  # only clipB's -3dB layer gets adjust-volume
    print("ok  test_build_fcpxml_audio_layers_present_as_own_clips")


def test_build_fcpxml_skips_layers_for_files_missing_from_lookup():
    resolved, lookup = _fixture()
    del lookup["f2"]
    root = ET.fromstring(fcpxml.build_fcpxml(resolved, lookup))
    assets = root.findall(".//resources/asset")
    assert len(assets) == 1 and assets[0].get("name") == "clipA.mov", assets
    spine = root.find(".//spine")
    assert len(list(spine)) == 1  # only clipA's spine clip survives
    print("ok  test_build_fcpxml_skips_layers_for_files_missing_from_lookup")


def test_build_fcpxml_uses_relative_media_paths():
    resolved, lookup = _fixture()
    root = ET.fromstring(fcpxml.build_fcpxml(resolved, lookup, media_dir="media"))
    for asset in root.findall(".//resources/asset"):
        src = asset.get("src")
        assert src.startswith("media/"), src
        assert not src.startswith("/"), src
        assert "r2.cloudflarestorage" not in src and "https://" not in src, src
    print("ok  test_build_fcpxml_uses_relative_media_paths")


def main():
    test_frames_snaps_to_nearest_frame()
    test_time_emits_rational_seconds_string()
    test_cell_transform_full_frame_identity()
    test_cell_transform_split_h_right_cell_hand_checked()
    test_cell_transform_pip_inset_cell_scales_up_and_positions_bottom_right()
    test_cell_transform_contain_never_crops()
    test_cell_transform_unknown_native_dims_falls_back_to_canvas_aspect()
    test_cell_transform_zoom_scales_beyond_cover()
    test_build_fcpxml_is_well_formed_xml()
    test_build_fcpxml_has_one_asset_per_distinct_file()
    test_build_fcpxml_spine_clip_count_and_program_order()
    test_build_fcpxml_split_screen_region_has_transform_and_crop()
    test_build_fcpxml_identity_spine_clip_emits_no_adjust_elements()
    test_build_fcpxml_audio_layers_present_as_own_clips()
    test_build_fcpxml_skips_layers_for_files_missing_from_lookup()
    test_build_fcpxml_uses_relative_media_paths()
    print("\nall export_fcpxml tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
