#!/usr/bin/env python3
"""Tests for app.services.l1.frame_descriptors (brain_mirror Phase 1) --
the pHash primitive, the compute/write-guard semantics, and the persist/read
helpers. No real DB / R2 / ffmpeg: the decode is monkeypatched with synthetic
frames and the DB is a tiny fake.

Run:  .venv/bin/python scripts/test_frame_descriptors.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import numpy as np  # noqa: E402

from app.services.l1 import frame_descriptors as fd  # noqa: E402


# --- Synthetic frames ----------------------------------------------------

def _solid(value, side=64):
    """A flat BGR frame of a single gray level."""
    return np.full((side, side, 3), value, dtype=np.uint8)


def _gradient(side=64, shift=0):
    """A horizontal luma gradient; `shift` rolls it to make a *similar* frame."""
    row = ((np.arange(side) + shift) % side) * (255.0 / side)
    img = np.repeat(row[np.newaxis, :], side, axis=0).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def _checker(side=64, sq=8):
    """A high-frequency checkerboard -- a valid, arbitrary frame for the
    compute/range tests."""
    yy, xx = np.mgrid[0:side, 0:side]
    img = (((yy // sq) + (xx // sq)) % 2 * 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def _natural(seed, side=64):
    """A broadband "natural-like" frame: a random 8x8 upscaled to `side`.

    pHash is designed for natural images with energy spread across many
    frequencies -- the median threshold then lands most coefficients well away
    from the boundary, so the bits are stable and discriminative. (A clean
    single-frequency synthetic frame is pathological: nearly every AC coefficient
    hugs a ~0 median, so numerical noise, not content, decides the bits and two
    visibly different clean frames read as near-identical.)"""
    import cv2
    rs = np.random.RandomState(seed)
    small = rs.randint(0, 256, size=(8, 8)).astype(np.uint8)
    img = cv2.resize(small, (side, side), interpolation=cv2.INTER_CUBIC)
    return np.stack([img, img, img], axis=-1)


def _natural_nudged(seed, side=64):
    """`_natural(seed)` with ONE low-frequency cell nudged -- a small, real
    perceptual change (a nearby framing), not a global brightness shift."""
    import cv2
    rs = np.random.RandomState(seed)
    small = rs.randint(0, 256, size=(8, 8)).astype(np.int16)
    for (r, c) in ((2, 2), (3, 4), (5, 3)):
        small[r, c] = int(np.clip(small[r, c] + 90, 0, 255))
    img = cv2.resize(small.astype(np.uint8), (side, side), interpolation=cv2.INTER_CUBIC)
    return np.stack([img, img, img], axis=-1)


# --- _phash64 ------------------------------------------------------------

def test_phash_is_deterministic():
    frame = _natural(7)
    assert fd._phash64(frame) == fd._phash64(frame.copy())
    print("ok  test_phash_is_deterministic")


def test_phash_is_64_bits():
    h = fd._phash64(_checker())
    assert 0 <= h <= (1 << 64) - 1, h
    print("ok  test_phash_is_64_bits")


def test_similar_frames_small_hamming():
    a = fd._phash64(_natural(7))
    b = fd._phash64(_natural_nudged(7))  # same frame, one cell nudged
    dist = fd.hamming64(a, b)
    assert dist <= 8, f"similar frames should be close, got {dist} bits"
    print(f"ok  test_similar_frames_small_hamming (dist={dist})")


def test_different_frames_large_hamming():
    a = fd._phash64(_natural(7))
    b = fd._phash64(_natural(42))  # independent content
    dist = fd.hamming64(a, b)
    assert dist >= 12, f"different content should be far, got {dist} bits"
    print(f"ok  test_different_frames_large_hamming (dist={dist})")


def test_identical_frames_zero_hamming():
    a = fd._phash64(_checker())
    assert fd.hamming64(a, a) == 0
    print("ok  test_identical_frames_zero_hamming")


# --- compute_frame_descriptors (decode monkeypatched) --------------------

def _patch_decode(monkey_frames):
    """Return a fake _decode_bgr_frames that yields the given frames."""
    def _fake(video_path, w, h, fps):
        for f in monkey_frames:
            yield f
    return _fake


def test_compute_ascending_t_ms_and_hop(monkeypatch_frames=None):
    frames = [_gradient(shift=0), _gradient(shift=4), _checker(), _solid(120)]
    orig = fd.scene_mod._decode_bgr_frames
    fd.scene_mod._decode_bgr_frames = _patch_decode(frames)
    try:
        res = fd.compute_frame_descriptors("x.mp4", duration_ms=2000, fps=2)
    finally:
        fd.scene_mod._decode_bgr_frames = orig
    assert res.has_frames is True
    assert res.hop_ms == 500, res.hop_ms
    ts = [p["t_ms"] for p in res.phashes]
    assert ts == [0, 500, 1000, 1500], ts
    assert all(len(p["h"]) == 16 for p in res.phashes), res.phashes
    print("ok  test_compute_ascending_t_ms_and_hop")


def test_compute_fail_open_on_decode_error():
    def _boom(video_path, w, h, fps):
        raise RuntimeError("ffmpeg exploded")
        yield  # pragma: no cover
    orig = fd.scene_mod._decode_bgr_frames
    fd.scene_mod._decode_bgr_frames = _boom
    try:
        res = fd.compute_frame_descriptors("x.mp4", duration_ms=2000, fps=2)
    finally:
        fd.scene_mod._decode_bgr_frames = orig
    assert res.has_frames is False
    assert res.phashes == []
    print("ok  test_compute_fail_open_on_decode_error")


def test_compute_no_frames_is_has_frames_false():
    orig = fd.scene_mod._decode_bgr_frames
    fd.scene_mod._decode_bgr_frames = _patch_decode([])
    try:
        res = fd.compute_frame_descriptors("x.mp4", duration_ms=0, fps=2)
    finally:
        fd.scene_mod._decode_bgr_frames = orig
    assert res.has_frames is False
    assert res.phashes == []
    print("ok  test_compute_no_frames_is_has_frames_false")


# --- Persist + read helpers (fake DB) ------------------------------------

class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _FakeConn:
    """Records upserts into an in-memory dict keyed on file_id, and answers
    `select ... from frame_descriptors where file_id = any(%s)` from it."""

    def __init__(self):
        self.store = {}  # file_id -> phashes (list of {t_ms, h})

    def execute(self, sql, params):
        s = " ".join(sql.split())
        if s.startswith("insert into frame_descriptors"):
            import json
            file_id, _hop, phashes_json, _sv = params
            self.store[str(file_id)] = json.loads(phashes_json)
            return _FakeCursor([])
        if s.startswith("select file_id, phashes from frame_descriptors"):
            (ids,) = params
            rows = [(fid, self.store[fid]) for fid in ids if fid in self.store]
            return _FakeCursor(rows)
        raise AssertionError(f"unexpected SQL: {s}")


def test_upsert_then_load_roundtrips_sorted_ints():
    conn = _FakeConn()
    res = fd.FrameDescriptors(
        has_frames=True, hop_ms=500,
        phashes=[{"t_ms": 1000, "h": "00000000000000ff"},
                 {"t_ms": 0, "h": "0000000000000001"}],
    )
    fd.upsert_frame_descriptors(conn, "file-A", res)
    loaded = fd.load_descriptors(conn, ["file-A", "file-missing"])
    assert "file-missing" not in loaded, loaded
    assert loaded["file-A"] == [(0, 1), (1000, 255)], loaded["file-A"]
    print("ok  test_upsert_then_load_roundtrips_sorted_ints")


def test_load_descriptors_empty_ids():
    assert fd.load_descriptors(_FakeConn(), []) == {}
    print("ok  test_load_descriptors_empty_ids")


def test_nearest_phash_picks_closest_within_tolerance():
    samples = [(0, 10), (500, 20), (1000, 30)]
    assert fd.nearest_phash(samples, 480) == 20  # closest is 500
    assert fd.nearest_phash(samples, 240) == 10  # 240->0 (240) beats 500 (260)
    assert fd.nearest_phash(samples, 1000) == 30
    print("ok  test_nearest_phash_picks_closest_within_tolerance")


def test_nearest_phash_none_when_too_far():
    samples = [(0, 10), (500, 20)]
    assert fd.nearest_phash(samples, 5000, nearest_ms=600) is None
    assert fd.nearest_phash([], 0) is None
    print("ok  test_nearest_phash_none_when_too_far")


def test_load_phash_at_convenience():
    conn = _FakeConn()
    res = fd.FrameDescriptors(
        has_frames=True, hop_ms=500,
        phashes=[{"t_ms": 0, "h": "000000000000000a"},
                 {"t_ms": 500, "h": "0000000000000014"}],
    )
    fd.upsert_frame_descriptors(conn, "file-A", res)
    assert fd.load_phash_at(conn, "file-A", 490) == 20  # nearest 500
    assert fd.load_phash_at(conn, "file-A", 9999) is None  # too far -> fail open
    assert fd.load_phash_at(conn, "unknown", 0) is None
    print("ok  test_load_phash_at_convenience")


def main():
    test_phash_is_deterministic()
    test_phash_is_64_bits()
    test_similar_frames_small_hamming()
    test_different_frames_large_hamming()
    test_identical_frames_zero_hamming()
    test_compute_ascending_t_ms_and_hop()
    test_compute_fail_open_on_decode_error()
    test_compute_no_frames_is_has_frames_false()
    test_upsert_then_load_roundtrips_sorted_ints()
    test_load_descriptors_empty_ids()
    test_nearest_phash_picks_closest_within_tolerance()
    test_nearest_phash_none_when_too_far()
    test_load_phash_at_convenience()
    print("\nall frame_descriptors tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
