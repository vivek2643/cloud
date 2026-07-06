"""
Tests for cuts-v3 JPEG still extraction (``app.services.l3.frames``). Uses a
real ffmpeg-synthesized clip (a local temp file -- no R2, no network, no
model calls) to validate the actual extraction; R2-touching functions are
exercised with a monkeypatched ``extract_stills_from_r2`` instead.

Run:  .venv/bin/python scripts/test_frames.py
"""
from __future__ import annotations

import base64
import hashlib
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import frames as fr  # noqa: E402
from app.services.l3.image_plan import PlannedFrame  # noqa: E402


def _make_synthetic_clip(path: str, duration_s: float = 3.0) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", f"testsrc=size=320x240:rate=10:duration={duration_s}",
            "-pix_fmt", "yuv420p", path,
        ],
        check=True, capture_output=True, timeout=30,
    )


def test_extract_still_produces_valid_jpeg():
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "clip.mp4")
        _make_synthetic_clip(clip)
        out = os.path.join(tmp, "still.jpg")
        fr.extract_still(clip, 500, out, width=160)
        assert os.path.exists(out) and os.path.getsize(out) > 0
        with open(out, "rb") as f:
            head = f.read(3)
        assert head == b"\xff\xd8\xff", "not a JPEG (missing SOI marker)"
    print("ok  test_extract_still_produces_valid_jpeg")


def test_extract_still_respects_width():
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "clip.mp4")
        _make_synthetic_clip(clip)
        out = os.path.join(tmp, "still.jpg")
        fr.extract_still(clip, 500, out, width=160)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width", "-of", "csv=p=0", out],
            check=True, capture_output=True, text=True, timeout=10,
        )
        assert probe.stdout.strip() == "160", probe.stdout
    print("ok  test_extract_still_respects_width")


def test_different_timestamps_yield_different_stills():
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "clip.mp4")
        _make_synthetic_clip(clip)
        stills = fr.extract_stills(clip, [100, 2800], width=160)
        assert set(stills.keys()) == {100, 2800}
        h1 = hashlib.sha256(base64.b64decode(stills[100])).hexdigest()
        h2 = hashlib.sha256(base64.b64decode(stills[2800])).hexdigest()
        assert h1 != h2, "expected visibly different frames at very different timestamps"
    print("ok  test_different_timestamps_yield_different_stills")


def test_extract_stills_dedupes_timestamps():
    with tempfile.TemporaryDirectory() as tmp:
        clip = os.path.join(tmp, "clip.mp4")
        _make_synthetic_clip(clip)
        stills = fr.extract_stills(clip, [500, 500, 500], width=160)
        assert len(stills) == 1, stills
    print("ok  test_extract_stills_dedupes_timestamps")


def test_file_to_b64_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        p = os.path.join(tmp, "x.bin")
        with open(p, "wb") as f:
            f.write(b"hello-bytes")
        b64 = fr.file_to_b64(p)
        assert base64.b64decode(b64) == b"hello-bytes"
    print("ok  test_file_to_b64_round_trips")


def test_extract_for_planned_frames_groups_by_file():
    calls = []
    orig = fr.extract_stills_from_r2

    def fake(proxy_key, ts_list, width=768):
        calls.append((proxy_key, sorted(ts_list)))
        return {ts: f"b64-{proxy_key}-{ts}" for ts in ts_list}

    fr.extract_stills_from_r2 = fake
    try:
        pf = [
            PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]"),
            PlannedFrame("f1", 200, "speech_cut", "speech_cut[1]"),
            PlannedFrame("f2", 300, "speech_cut", "speech_cut[0]"),
        ]
        out = fr.extract_for_planned_frames(pf, {"f1": "proxies/f1/proxy.mp4", "f2": "proxies/f2/proxy.mp4"})
    finally:
        fr.extract_stills_from_r2 = orig
    assert out[("f1", 100)] == "b64-proxies/f1/proxy.mp4-100"
    assert out[("f1", 200)] == "b64-proxies/f1/proxy.mp4-200"
    assert out[("f2", 300)] == "b64-proxies/f2/proxy.mp4-300"
    assert len(calls) == 2, calls
    print("ok  test_extract_for_planned_frames_groups_by_file")


def test_extract_for_planned_frames_skips_files_without_proxy_key():
    calls = []
    orig = fr.extract_stills_from_r2
    fr.extract_stills_from_r2 = lambda *a, **k: (calls.append(a), {})[1]
    try:
        pf = [PlannedFrame("f1", 100, "speech_cut", "speech_cut[0]")]
        out = fr.extract_for_planned_frames(pf, {})
    finally:
        fr.extract_stills_from_r2 = orig
    assert out == {}
    assert calls == []
    print("ok  test_extract_for_planned_frames_skips_files_without_proxy_key")


def main():
    test_extract_still_produces_valid_jpeg()
    test_extract_still_respects_width()
    test_different_timestamps_yield_different_stills()
    test_extract_stills_dedupes_timestamps()
    test_file_to_b64_round_trips()
    test_extract_for_planned_frames_groups_by_file()
    test_extract_for_planned_frames_skips_files_without_proxy_key()
    print("\nall frames tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
