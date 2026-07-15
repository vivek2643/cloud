"""
Tests for the L1 active-speaker pass (app.services.l1.active_speaker) --
no real video, no real network. The detect/embed model (insightface) and
the video decode (cv2.VideoCapture) are faked at their exact boundary so
the REAL tracking/correlation/merge algorithms run under test, matching
this codebase's established mock-the-SDK-boundary style.

Run:  .venv/bin/python scripts/test_active_speaker.py
"""
from __future__ import annotations

import os
import sys
from unittest import mock

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l1 import active_speaker as asd  # noqa: E402


# --------------------------------------------------------------------------
# _iou
# --------------------------------------------------------------------------

def test_iou_identical_boxes_is_one():
    assert asd._iou((0, 0, 100, 100), (0, 0, 100, 100)) == 1.0
    print("ok  test_iou_identical_boxes_is_one")


def test_iou_disjoint_boxes_is_zero():
    assert asd._iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0
    print("ok  test_iou_disjoint_boxes_is_zero")


def test_iou_partial_overlap_between_zero_and_one():
    iou = asd._iou((0, 0, 10, 10), (5, 5, 10, 10))
    assert 0.0 < iou < 1.0, iou
    print("ok  test_iou_partial_overlap_between_zero_and_one")


# --------------------------------------------------------------------------
# _track_embedding
# --------------------------------------------------------------------------

def test_track_embedding_means_and_renormalizes():
    embs = [[1.0, 0.0], [0.0, 1.0]]
    out = asd._track_embedding(embs)
    norm = sum(v * v for v in out) ** 0.5
    assert abs(norm - 1.0) < 1e-6, out
    assert out[0] == out[1], out   # symmetric inputs -> symmetric mean
    print("ok  test_track_embedding_means_and_renormalizes")


def test_track_embedding_empty_is_empty():
    assert asd._track_embedding([]) == []
    print("ok  test_track_embedding_empty_is_empty")


def test_track_embedding_ignores_mismatched_dims():
    embs = [[1.0, 0.0, 0.0], [1.0, 0.0]]  # second is the wrong dim, dropped
    out = asd._track_embedding(embs)
    assert len(out) == 3, out
    print("ok  test_track_embedding_ignores_mismatched_dims")


# --------------------------------------------------------------------------
# _best_crop_ms
# --------------------------------------------------------------------------

def test_best_crop_ms_picks_the_largest_box():
    frames = [asd.FaceFrame(t_ms=0, box=(0, 0, 10, 10)), asd.FaceFrame(t_ms=100, box=(0, 0, 50, 50)),
             asd.FaceFrame(t_ms=200, box=(0, 0, 20, 20))]
    assert asd._best_crop_ms(frames) == 100
    print("ok  test_best_crop_ms_picks_the_largest_box")


def test_best_crop_ms_empty_is_zero():
    assert asd._best_crop_ms([]) == 0
    print("ok  test_best_crop_ms_empty_is_zero")


# --------------------------------------------------------------------------
# _box_at
# --------------------------------------------------------------------------

def test_box_at_returns_nearest_within_tolerance():
    frames = [asd.FaceFrame(t_ms=0, box=(1, 1, 1, 1)), asd.FaceFrame(t_ms=1000, box=(2, 2, 2, 2))]
    assert asd._box_at(frames, 950, tol_ms=200) == (2, 2, 2, 2)
    print("ok  test_box_at_returns_nearest_within_tolerance")


def test_box_at_none_outside_tolerance():
    frames = [asd.FaceFrame(t_ms=0, box=(1, 1, 1, 1))]
    assert asd._box_at(frames, 5000, tol_ms=200) is None
    print("ok  test_box_at_none_outside_tolerance")


def test_box_at_empty_frames_is_none():
    assert asd._box_at([], 0, tol_ms=200) is None
    print("ok  test_box_at_empty_frames_is_none")


# --------------------------------------------------------------------------
# _mouth_crop
# --------------------------------------------------------------------------

def test_mouth_crop_returns_correctly_sized_grayscale_patch():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    crop = asd._mouth_crop(frame, (10, 10, 80, 80))
    assert crop is not None
    assert crop.shape == (asd.MOUTH_CROP_SIZE, asd.MOUTH_CROP_SIZE)
    print("ok  test_mouth_crop_returns_correctly_sized_grayscale_patch")


def test_mouth_crop_none_for_degenerate_box():
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    assert asd._mouth_crop(frame, (10, 10, 0, 0)) is None
    print("ok  test_mouth_crop_none_for_degenerate_box")


def test_mouth_crop_clamps_a_box_hanging_off_the_frame():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # Box extends past the 100x100 frame edge, but its mouth-region (lower
    # 45%) still has some overlap with the frame -- must clamp, not crash.
    crop = asd._mouth_crop(frame, (50, 50, 60, 60))
    assert crop is not None
    assert crop.shape == (asd.MOUTH_CROP_SIZE, asd.MOUTH_CROP_SIZE)
    print("ok  test_mouth_crop_clamps_a_box_hanging_off_the_frame")


def test_mouth_crop_none_when_the_mouth_region_is_entirely_off_frame():
    # A box hanging off far enough that even its lower 45% has zero overlap
    # with the frame -- correctly None, never a crash on an empty crop.
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert asd._mouth_crop(frame, (80, 80, 60, 60)) is None
    print("ok  test_mouth_crop_none_when_the_mouth_region_is_entirely_off_frame")


# --------------------------------------------------------------------------
# _pearson
# --------------------------------------------------------------------------

def test_pearson_perfectly_correlated_is_one():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [2.0, 4.0, 6.0, 8.0]
    assert abs(asd._pearson(a, b) - 1.0) < 1e-9
    print("ok  test_pearson_perfectly_correlated_is_one")


def test_pearson_constant_series_is_zero_not_nan():
    assert asd._pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0
    print("ok  test_pearson_constant_series_is_zero_not_nan")


def test_pearson_mismatched_length_is_zero():
    assert asd._pearson([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0
    print("ok  test_pearson_mismatched_length_is_zero")


# --------------------------------------------------------------------------
# _merge_speaking_windows
# --------------------------------------------------------------------------

def test_merge_speaking_windows_joins_touching_windows():
    windows = [(0, 1000, 0.5), (1000, 2000, 0.7), (5000, 6000, 0.4)]
    out = asd._merge_speaking_windows(windows, hop_ms=200)
    assert len(out) == 2, out
    assert out[0].start_ms == 0 and out[0].end_ms == 2000
    assert abs(out[0].score - 0.6) < 1e-9
    print("ok  test_merge_speaking_windows_joins_touching_windows")


def test_merge_speaking_windows_empty_is_empty():
    assert asd._merge_speaking_windows([], hop_ms=200) == []
    print("ok  test_merge_speaking_windows_empty_is_empty")


# --------------------------------------------------------------------------
# _score_track_speaking
# --------------------------------------------------------------------------

def test_score_track_speaking_binds_when_motion_tracks_audio():
    # A track whose mouth-motion energy rises and falls WITH the audio --
    # the talker case.
    hop_ms = 200
    motion = [(i * hop_ms, v) for i, v in enumerate([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])]
    audio = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    out = asd._score_track_speaking(motion, audio, hop_ms)
    assert len(out) >= 1, out
    print("ok  test_score_track_speaking_binds_when_motion_tracks_audio")


def test_score_track_speaking_empty_when_face_is_still():
    # Audio is loud throughout but the face barely moves -- a listener, not
    # a talker. Motion never clears CORR_MIN_MOTION.
    hop_ms = 200
    motion = [(i * hop_ms, 0.001) for i in range(8)]
    audio = [1.0, 0.9, 1.0, 0.8, 1.0, 0.9, 1.0, 0.8]
    out = asd._score_track_speaking(motion, audio, hop_ms)
    assert out == [], out
    print("ok  test_score_track_speaking_empty_when_face_is_still")


def test_score_track_speaking_too_short_series_is_empty():
    assert asd._score_track_speaking([(0, 1.0), (200, 1.0)], [1.0, 1.0], 200) == []
    print("ok  test_score_track_speaking_too_short_series_is_empty")


def test_score_track_speaking_no_audio_is_empty():
    motion = [(i * 200, 1.0) for i in range(5)]
    assert asd._score_track_speaking(motion, [], 200) == []
    print("ok  test_score_track_speaking_no_audio_is_empty")


# --------------------------------------------------------------------------
# FaceTrack.to_dict / from_dict roundtrip
# --------------------------------------------------------------------------

def test_face_track_roundtrips_through_dict():
    tr = asd.FaceTrack(
        track_id=3, embedding=[0.1, 0.2, 0.3],
        frames=[asd.FaceFrame(t_ms=0, box=(1, 2, 3, 4))],
        speaking=[asd.SpeakingInterval(start_ms=0, end_ms=1000, score=0.8)],
        best_crop_ms=500,
    )
    back = asd.FaceTrack.from_dict(tr.to_dict())
    assert back.track_id == 3
    assert back.embedding == [0.1, 0.2, 0.3]
    assert back.frames[0].t_ms == 0 and back.frames[0].box == (1, 2, 3, 4)
    assert back.speaking[0].score == 0.8
    assert back.best_crop_ms == 500
    print("ok  test_face_track_roundtrips_through_dict")


# --------------------------------------------------------------------------
# _detect_and_track (fake cv2.VideoCapture + fake insightface FaceAnalysis)
# --------------------------------------------------------------------------

class _FakeDet:
    """Mirrors insightface's own Face object shape: `.bbox` is [x1, y1, x2,
    y2] (two corner points, NOT [x, y, w, h]) -- active_speaker.py converts
    to (x, y, w, h) itself."""

    def __init__(self, bbox, embedding):
        self.bbox = bbox
        self.embedding = embedding


class _FakeCapture:
    """Enough of cv2.VideoCapture's surface for _detect_and_track /
    _dense_motion_energy: a fixed frame count/fps (so duration is known)
    and a frame available at every step_ms tick up to that duration."""

    def __init__(self, n_frames: int, fps: float, frame_shape=(100, 100, 3)):
        self._n_frames = n_frames
        self._fps = fps
        self._frame_shape = frame_shape
        self._pos_ms = 0.0

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return self._n_frames
        if prop == cv2.CAP_PROP_FPS:
            return self._fps
        return 0

    def set(self, prop, value):
        self._pos_ms = value

    def read(self):
        duration_ms = self._n_frames / self._fps * 1000
        if self._pos_ms > duration_ms:
            return False, None
        return True, np.zeros(self._frame_shape, dtype=np.uint8)

    def release(self):
        pass


def test_detect_and_track_links_close_detections_into_one_track():
    # Same box (perfect IoU) at every sampled tick across a 1s clip at
    # TRACK_SAMPLE_FPS=5 -> one continuous track of 5 detections.
    fps = asd.TRACK_SAMPLE_FPS
    n_frames = int(fps)   # 1 second of footage
    fake_cap = _FakeCapture(n_frames=n_frames, fps=fps)

    fake_app = mock.Mock()
    fake_app.get.return_value = [_FakeDet(bbox=[10, 10, 60, 60], embedding=[1.0, 0.0])]

    with mock.patch.object(asd.cv2, "VideoCapture", return_value=fake_cap):
        tracks = asd._detect_and_track("fake.mp4", fake_app)

    assert len(tracks) == 1, tracks
    assert len(tracks[0].frames) == n_frames, tracks[0].frames
    print("ok  test_detect_and_track_links_close_detections_into_one_track")


def test_detect_and_track_starts_a_new_track_for_a_distant_detection():
    fps = asd.TRACK_SAMPLE_FPS
    n_frames = int(fps)
    fake_cap = _FakeCapture(n_frames=n_frames, fps=fps, frame_shape=(500, 500, 3))

    # Two spatially-disjoint boxes present in every sampled frame -- two
    # distinct people, never linked into one track (IoU ~= 0 between them).
    fake_app = mock.Mock()
    fake_app.get.return_value = [
        _FakeDet(bbox=[0, 0, 60, 60], embedding=[1.0, 0.0]),
        _FakeDet(bbox=[400, 400, 460, 460], embedding=[0.0, 1.0]),
    ]

    with mock.patch.object(asd.cv2, "VideoCapture", return_value=fake_cap):
        tracks = asd._detect_and_track("fake.mp4", fake_app)

    assert len(tracks) == 2, tracks
    print("ok  test_detect_and_track_starts_a_new_track_for_a_distant_detection")


def test_detect_and_track_drops_a_track_shorter_than_min_frames():
    fps = asd.TRACK_SAMPLE_FPS
    n_frames = int(fps)
    fake_cap = _FakeCapture(n_frames=n_frames, fps=fps)

    call_count = {"n": 0}

    def fake_get(frame):
        call_count["n"] += 1
        # Only the FIRST sampled frame has a detection -- a one-off blip,
        # under MIN_TRACK_FRAMES, must not survive to the final result.
        if call_count["n"] == 1:
            return [_FakeDet(bbox=[10, 10, 60, 60], embedding=[1.0, 0.0])]
        return []

    fake_app = mock.Mock()
    fake_app.get.side_effect = fake_get

    with mock.patch.object(asd.cv2, "VideoCapture", return_value=fake_cap):
        tracks = asd._detect_and_track("fake.mp4", fake_app)

    assert tracks == [], tracks
    print("ok  test_detect_and_track_drops_a_track_shorter_than_min_frames")


def test_detect_and_track_no_faces_is_empty():
    fake_cap = _FakeCapture(n_frames=int(asd.TRACK_SAMPLE_FPS), fps=asd.TRACK_SAMPLE_FPS)
    fake_app = mock.Mock()
    fake_app.get.return_value = []
    with mock.patch.object(asd.cv2, "VideoCapture", return_value=fake_cap):
        assert asd._detect_and_track("fake.mp4", fake_app) == []
    print("ok  test_detect_and_track_no_faces_is_empty")


# --------------------------------------------------------------------------
# compute_face_tracks fail-open contract
# --------------------------------------------------------------------------

def test_compute_face_tracks_empty_when_insightface_unavailable():
    with mock.patch.object(asd, "_get_face_app", return_value=None):
        assert asd.compute_face_tracks("fake.mp4") == []
    print("ok  test_compute_face_tracks_empty_when_insightface_unavailable")


def test_compute_face_tracks_never_raises_on_internal_error():
    with mock.patch.object(asd, "_get_face_app", side_effect=RuntimeError("boom")):
        assert asd.compute_face_tracks("fake.mp4") == []
    print("ok  test_compute_face_tracks_never_raises_on_internal_error")


def main():
    test_iou_identical_boxes_is_one()
    test_iou_disjoint_boxes_is_zero()
    test_iou_partial_overlap_between_zero_and_one()
    test_track_embedding_means_and_renormalizes()
    test_track_embedding_empty_is_empty()
    test_track_embedding_ignores_mismatched_dims()
    test_best_crop_ms_picks_the_largest_box()
    test_best_crop_ms_empty_is_zero()
    test_box_at_returns_nearest_within_tolerance()
    test_box_at_none_outside_tolerance()
    test_box_at_empty_frames_is_none()
    test_mouth_crop_returns_correctly_sized_grayscale_patch()
    test_mouth_crop_none_for_degenerate_box()
    test_mouth_crop_clamps_a_box_hanging_off_the_frame()
    test_mouth_crop_none_when_the_mouth_region_is_entirely_off_frame()
    test_pearson_perfectly_correlated_is_one()
    test_pearson_constant_series_is_zero_not_nan()
    test_pearson_mismatched_length_is_zero()
    test_merge_speaking_windows_joins_touching_windows()
    test_merge_speaking_windows_empty_is_empty()
    test_score_track_speaking_binds_when_motion_tracks_audio()
    test_score_track_speaking_empty_when_face_is_still()
    test_score_track_speaking_too_short_series_is_empty()
    test_score_track_speaking_no_audio_is_empty()
    test_face_track_roundtrips_through_dict()
    test_detect_and_track_links_close_detections_into_one_track()
    test_detect_and_track_starts_a_new_track_for_a_distant_detection()
    test_detect_and_track_drops_a_track_shorter_than_min_frames()
    test_detect_and_track_no_faces_is_empty()
    test_compute_face_tracks_empty_when_insightface_unavailable()
    test_compute_face_tracks_never_raises_on_internal_error()
    print("\nall active-speaker tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
