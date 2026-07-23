"""
L1 active-speaker pass (asd_identity.plan.md): detect+track faces and score
each track's ASD-speaking timeline, off the canonical 1080p editing proxy
(the client A/B analysis proxies are unusable here -- proxy_a is 1fps video,
proxy_b has no audio at all; only `proxies/{file_id}/proxy.mp4` carries real
video+audio+fps). Runs once per file, persisted, reused across every ingest
-- replaces the old per-project Gemini "is this person speaking?" pass
(identity/voice_id.py, deleted) with a deterministic, local, CPU signal.

Pipeline (all local, CPU):
  1. Detect + track faces (SCRFD via insightface) at TRACK_SAMPLE_FPS -> one
     track per face that recurs across consecutive sampled frames (IoU
     linking). Small/spurious tracks are dropped.
  2. Embed each track (ArcFace, mean over its detections' embeddings,
     re-normalized) -- the cross-file identity signal identity/faces.py
     clusters on, mirroring identity/voices.py's voiceprint clustering.
  3. Active-speaker score each track: a deterministic audio-visual
     correlation proxy, NOT a trained model. For each track, sample its
     mouth-region motion energy (frame-to-frame pixel difference in the
     lower-face crop) at a DENSER rate (ASD_SAMPLE_FPS) than the track/embed
     pass needs, correlate it against the audio's RMS envelope over a short
     sliding window, and merge windows that are BOTH well-correlated AND
     show real motion into `speaking` intervals. A listener's mouth is
     mostly still (no motion to correlate); a talker's mouth motion tracks
     the audio envelope. This is a v0 heuristic, not SOTA ASD accuracy --
     the `FaceTrack.speaking` shape is the seam a trained model (Light-ASD/
     TalkNet) can drop into later without touching any downstream code.

Best-effort throughout, same fail-open contract as diarization.diarize():
`compute_face_tracks` never raises. No faces / insightface unavailable /
any decode failure -> an empty list, never a hard fail -- identity simply
degrades to id-less PIC/SND for that file, exactly like a project with no
reconciled cast today.
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.services import limits

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

FFMPEG_TIMEOUT_S = 60

# --- Detect + track ---------------------------------------------------------
TRACK_SAMPLE_FPS = 5.0        # face detect+embed cadence (expensive: one insightface forward pass/sample)
IOU_MATCH_MIN = 0.3           # min IoU to extend a track from one sampled frame to the next
MAX_TRACK_GAP_S = 1.0         # a track that misses this long closes (and gates dense-pass "is it active" lookups)
MIN_TRACK_FRAMES = 3          # drop spurious 1-2-detection tracks
MIN_FACE_PX = 40              # ignore detections smaller than this (too small for a legible mouth anyway)
DET_SIZE = (320, 320)         # insightface detector input size

# --- ASD (deterministic AV-correlation) -------------------------------------
ASD_SAMPLE_FPS = 15.0         # denser cadence for mouth-motion scoring (cheap: crop + pixel diff, no detector)
MOUTH_CROP_SIZE = 32          # mouth-region crop is resized to this (size-invariant, cheap diff)
CORR_WINDOW_MS = 1000         # sliding-window width for the motion/audio correlation
CORR_MIN_SCORE = 0.3          # min Pearson correlation to call a window "speaking"
CORR_MIN_MOTION = 0.15        # min normalized motion in a window -- a still face never "speaks"
                               # regardless of a coincidentally-correlated near-zero signal


@dataclass
class FaceFrame:
    t_ms: int
    box: Tuple[int, int, int, int]   # x, y, w, h, proxy pixel space


@dataclass
class SpeakingInterval:
    start_ms: int
    end_ms: int
    score: float


@dataclass
class FaceTrack:
    track_id: int
    embedding: List[float] = field(default_factory=list)
    frames: List[FaceFrame] = field(default_factory=list)
    speaking: List[SpeakingInterval] = field(default_factory=list)
    best_crop_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "embedding": self.embedding,
            "frames": [{"t_ms": f.t_ms, "box": list(f.box)} for f in self.frames],
            "speaking": [{"start_ms": s.start_ms, "end_ms": s.end_ms, "score": s.score} for s in self.speaking],
            "best_crop_ms": self.best_crop_ms,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "FaceTrack":
        return FaceTrack(
            track_id=int(d["track_id"]),
            embedding=[float(x) for x in (d.get("embedding") or [])],
            frames=[FaceFrame(t_ms=int(f["t_ms"]), box=tuple(int(v) for v in f["box"]))
                   for f in (d.get("frames") or [])],
            speaking=[SpeakingInterval(start_ms=int(s["start_ms"]), end_ms=int(s["end_ms"]), score=float(s["score"]))
                     for s in (d.get("speaking") or [])],
            best_crop_ms=int(d.get("best_crop_ms") or 0),
        )


@dataclass
class _RawTrack:
    """Detect+track's own working state -- collapsed into a `FaceTrack`
    (embedding computed, speaking scored) once the dense ASD pass runs."""
    track_id: int
    last_box: Tuple[int, int, int, int]
    last_t_ms: int
    frames: List[FaceFrame] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_face_tracks(video_path: str) -> List[FaceTrack]:
    """The whole pass. Never raises -- see module docstring's fail-open
    contract."""
    try:
        return _compute_face_tracks(video_path)
    except Exception:
        logger.exception("active_speaker: face-track pass failed for %s", video_path)
        return []


def _compute_face_tracks(video_path: str) -> List[FaceTrack]:
    face_app = _get_face_app()
    if face_app is None:
        logger.warning("active_speaker: insightface unavailable; skipping %s.", video_path)
        return []

    raw_tracks = _detect_and_track(video_path, face_app)
    if not raw_tracks:
        return []

    duration_ms = max((f.t_ms for tr in raw_tracks for f in tr.frames), default=0) + 1000
    motion_by_track = _dense_motion_energy(video_path, raw_tracks, duration_ms)

    audio_rms: List[float] = []
    hop_ms = int(1000 / ASD_SAMPLE_FPS)
    with tempfile.TemporaryDirectory() as tmp:
        wav_path = os.path.join(tmp, "audio.wav")
        try:
            _demux_wav(video_path, wav_path)
            audio_rms = _audio_rms_envelope(wav_path, hop_ms)
        except Exception:
            logger.warning("active_speaker: audio demux/RMS failed for %s; speaking will be empty.", video_path)

    tracks: List[FaceTrack] = []
    for tr in raw_tracks:
        speaking = (_score_track_speaking(motion_by_track.get(tr.track_id, []), audio_rms, hop_ms)
                   if audio_rms else [])
        tracks.append(FaceTrack(
            track_id=tr.track_id, embedding=_track_embedding(tr.embeddings), frames=tr.frames,
            speaking=speaking, best_crop_ms=_best_crop_ms(tr.frames),
        ))
    return tracks


# ---------------------------------------------------------------------------
# Model loading (lazy, process-wide, CPU -- see asd_identity.plan.md SS2:
# "Face detect/embed (insightface + onnxruntime) run on CPU")
# ---------------------------------------------------------------------------

_FACE_APP = None  # lazily-created insightface FaceAnalysis, or False = tried and failed


def _get_face_app():
    global _FACE_APP
    if _FACE_APP is None:
        try:
            from insightface.app import FaceAnalysis

            app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=DET_SIZE)
            _FACE_APP = app
            logger.info("insightface FaceAnalysis (buffalo_sc, CPU) loaded.")
        except Exception:
            logger.warning("insightface backend unavailable.", exc_info=True)
            _FACE_APP = False
    return _FACE_APP or None


# ---------------------------------------------------------------------------
# Detect + track (SCRFD via insightface, IoU linking)
# ---------------------------------------------------------------------------

def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _sample_frames(cap: "cv2.VideoCapture", step_ms: int):
    """Yield (t_ms, frame) on a fixed 0, step_ms, 2*step_ms, ... grid by
    decoding the clip SEQUENTIALLY: grab() cheaply walks every frame and
    retrieve() only fully decodes the ones that land on the grid.

    This replaces a per-sample `cap.set(CAP_PROP_POS_MSEC)` seek, which forces
    ffmpeg to re-seek to a keyframe and re-decode from there on EVERY sample --
    O(n^2)-ish on a long H.264 proxy (minutes-long clips took longer than real
    time, and multi-hour footage on the fleet would be unusable). The emitted
    timestamps are the same grid points the seek-per-sample loop used, so the
    sampling (and every downstream ASD/audio alignment) is unchanged -- only
    the decode strategy is."""
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if src_fps <= 0:
        src_fps = 30.0  # container without reliable fps; assume ~30 for a sane grid
    next_ms = 0.0
    idx = -1
    while True:
        if not cap.grab():
            return
        idx += 1
        cur_ms = idx / src_fps * 1000.0
        if cur_ms + 1e-6 < next_ms:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            next_ms += step_ms
            continue
        yield int(round(next_ms)), frame
        next_ms += step_ms
        # If frames were dropped and we've already passed later grid points,
        # skip them rather than emitting a burst for one decoded frame.
        while next_ms <= cur_ms:
            next_ms += step_ms


def _detect_and_track(video_path: str, face_app) -> List[_RawTrack]:
    """One pass at TRACK_SAMPLE_FPS: detect faces (+ embeddings, insightface
    gives both from one forward pass), IoU-link consecutive detections into
    tracks, close a track once it's gone unmatched for MAX_TRACK_GAP_S.
    Drops tracks shorter than MIN_TRACK_FRAMES (spurious one-off
    detections, not a real recurring face)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
    try:
        step_ms = int(1000 / TRACK_SAMPLE_FPS)
        active: List[_RawTrack] = []
        finished: List[_RawTrack] = []
        next_id = 0
        max_gap_ms = int(MAX_TRACK_GAP_S * 1000)

        for t_ms, frame in _sample_frames(cap, step_ms):
            dets: List[Tuple[Tuple[int, int, int, int], List[float]]] = []
            for f in face_app.get(frame):
                x1, y1, x2, y2 = [int(v) for v in f.bbox]
                w, h = x2 - x1, y2 - y1
                if w < MIN_FACE_PX or h < MIN_FACE_PX:
                    continue
                dets.append(((x1, y1, w, h), [float(v) for v in f.embedding]))

            matched: set = set()
            for box, emb in dets:
                best_tr, best_iou = None, 0.0
                for tr in active:
                    if tr.track_id in matched:
                        continue
                    iou = _iou(tr.last_box, box)
                    if iou > best_iou:
                        best_iou, best_tr = iou, tr
                if best_tr is not None and best_iou >= IOU_MATCH_MIN:
                    best_tr.last_box, best_tr.last_t_ms = box, t_ms
                    best_tr.frames.append(FaceFrame(t_ms=t_ms, box=box))
                    best_tr.embeddings.append(emb)
                    matched.add(best_tr.track_id)
                else:
                    tr = _RawTrack(track_id=next_id, last_box=box, last_t_ms=t_ms,
                                   frames=[FaceFrame(t_ms=t_ms, box=box)], embeddings=[emb])
                    next_id += 1
                    active.append(tr)
                    matched.add(tr.track_id)

            still_active, closed = [], []
            for tr in active:
                (closed if (t_ms - tr.last_t_ms) > max_gap_ms else still_active).append(tr)
            finished.extend(closed)
            active = still_active

        finished.extend(active)
        return [tr for tr in finished if len(tr.frames) >= MIN_TRACK_FRAMES]
    finally:
        cap.release()


def _track_embedding(embeddings: List[List[float]]) -> List[float]:
    """Mean ArcFace embedding over a track's detections, L2-renormalized --
    the standard way to collapse several observations of the same face into
    one comparable vector for identity/faces.py's cross-file clustering."""
    if not embeddings:
        return []
    dim = len(embeddings[0])
    usable = [e for e in embeddings if len(e) == dim]
    if not usable:
        return []
    mean = [sum(e[i] for e in usable) / len(usable) for i in range(dim)]
    norm = sum(v * v for v in mean) ** 0.5
    return [v / norm for v in mean] if norm > 0 else mean


def _best_crop_ms(frames: List[FaceFrame]) -> int:
    """The timestamp of this track's LARGEST detection -- the closest/most
    legible crop, a reasonable proxy for "best" without a sharpness pass."""
    if not frames:
        return 0
    return max(frames, key=lambda f: f.box[2] * f.box[3]).t_ms


# ---------------------------------------------------------------------------
# Dense mouth-motion pass (shared one video decode across every active track)
# ---------------------------------------------------------------------------

def _box_at(frames: List[FaceFrame], t_ms: int, tol_ms: int) -> Optional[Tuple[int, int, int, int]]:
    """The track's nearest coarse detection to `t_ms`, or None when the
    nearest is farther than `tol_ms` away (the track isn't actually active
    at this instant -- e.g. it hasn't started yet, already ended, or the
    face left frame for a beat)."""
    if not frames:
        return None
    nearest = min(frames, key=lambda f: abs(f.t_ms - t_ms))
    return nearest.box if abs(nearest.t_ms - t_ms) <= tol_ms else None


def _mouth_crop(frame: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Grayscale, size-normalized crop of the LOWER portion of a face box
    (mouth/jaw) -- cheap frame-to-frame motion energy there is a real
    lip-articulation proxy without needing exact mouth landmarks."""
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    H, W = frame.shape[:2]
    y0 = max(0, min(H, y + int(h * 0.55)))
    y1 = max(0, min(H, y + h))
    x0 = max(0, min(W, x))
    x1 = max(0, min(W, x + w))
    if y1 <= y0 or x1 <= x0:
        return None
    gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (MOUTH_CROP_SIZE, MOUTH_CROP_SIZE))


def _dense_motion_energy(
    video_path: str, tracks: List[_RawTrack], duration_ms: int,
) -> Dict[int, List[Tuple[int, float]]]:
    """One dense pass (ASD_SAMPLE_FPS) over the whole clip -- at each
    timestamp, every track active there (per `_box_at`) gets its mouth crop
    diffed against its own previous crop. Sharing one decode loop across all
    simultaneously-active tracks avoids re-decoding the same frames once per
    track. Returns {track_id: [(t_ms, motion_energy), ...]}."""
    if not tracks:
        return {}
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    tol_ms = int(MAX_TRACK_GAP_S * 1000)
    step_ms = max(1, int(1000 / ASD_SAMPLE_FPS))
    frames_by_track = {tr.track_id: tr.frames for tr in tracks}
    prev_crop: Dict[int, np.ndarray] = {}
    energy: Dict[int, List[Tuple[int, float]]] = {tr.track_id: [] for tr in tracks}
    try:
        # Same fixed step_ms grid as before, but via one sequential decode
        # (see _sample_frames) instead of a seek per sample -- at ASD_SAMPLE_FPS
        # the old per-sample POS_MSEC seek dominated the whole pass.
        for t_ms, frame in _sample_frames(cap, step_ms):
            for tr in tracks:
                box = _box_at(frames_by_track[tr.track_id], t_ms, tol_ms)
                if box is None:
                    continue
                crop = _mouth_crop(frame, box)
                if crop is None:
                    continue
                if tr.track_id in prev_crop:
                    diff = float(np.abs(crop.astype(np.int16) - prev_crop[tr.track_id].astype(np.int16)).mean())
                    energy[tr.track_id].append((t_ms, diff))
                prev_crop[tr.track_id] = crop
    finally:
        cap.release()
    return energy


# ---------------------------------------------------------------------------
# Audio RMS envelope
# ---------------------------------------------------------------------------

def _demux_wav(video_path: str, out_path: str) -> None:
    with limits.ffmpeg_slot():
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", out_path],
            check=True, capture_output=True, timeout=FFMPEG_TIMEOUT_S,
        )


def _audio_rms_envelope(wav_path: str, hop_ms: int) -> List[float]:
    """RMS energy at a fixed `hop_ms` grid starting at t=0 -- deliberately
    NOT the bounded/downsampled prosody envelope audio_features.py computes
    (that's tuned for grading display, not tight AV correlation); this is
    its own fine grid matching the dense motion pass's own sample times."""
    import librosa

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    if y.size == 0:
        return []
    hop = max(1, int(sr * hop_ms / 1000))
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    return [float(v) for v in rms]


# ---------------------------------------------------------------------------
# AV correlation -> speaking intervals
# ---------------------------------------------------------------------------

def _pearson(a: List[float], b: List[float]) -> float:
    n = len(a)
    if n < 2 or n != len(b):
        return 0.0
    ma, mb = sum(a) / n, sum(b) / n
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    num = sum(x * y for x, y in zip(da, db))
    den = (sum(x * x for x in da) ** 0.5) * (sum(y * y for y in db) ** 0.5)
    return num / den if den > 0 else 0.0


def _merge_speaking_windows(windows: List[Tuple[int, int, float]], hop_ms: int) -> List[SpeakingInterval]:
    """Merge windows that touch or overlap (allowing one hop's gap, same
    spirit as identity/windows.py's old interval merge) into intervals,
    averaging the score of whatever windows landed inside."""
    if not windows:
        return []
    ordered = sorted(windows)
    merged: List[List[Any]] = [[ordered[0][0], ordered[0][1], [ordered[0][2]]]]
    for s, e, sc in ordered[1:]:
        if s <= merged[-1][1] + hop_ms:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2].append(sc)
        else:
            merged.append([s, e, [sc]])
    return [SpeakingInterval(start_ms=s, end_ms=e, score=sum(scs) / len(scs)) for s, e, scs in merged]


def _score_track_speaking(
    motion_series: List[Tuple[int, float]], audio_rms: List[float], hop_ms: int,
) -> List[SpeakingInterval]:
    """Sliding-window correlate a track's (clip-relative-normalized) mouth-
    motion energy against the audio RMS envelope at the same t_ms grid. A
    window counts as "speaking" only when BOTH the correlation clears
    CORR_MIN_SCORE AND the motion itself clears CORR_MIN_MOTION -- the
    second guard exists because a perfectly STILL signal can spuriously
    correlate with near-zero variance; real speech needs visible motion to
    track, not just numerical agreement on silence."""
    if len(motion_series) < 3 or not audio_rms:
        return []
    values = [v for _, v in motion_series]
    lo, hi = min(values), max(values)
    if hi <= lo:
        return []
    norm_motion = [(v - lo) / (hi - lo) for v in values]
    times = [t for t, _ in motion_series]

    win_n = max(3, int(CORR_WINDOW_MS / hop_ms))
    windows: List[Tuple[int, int, float]] = []
    for i in range(len(times) - win_n + 1):
        j = i + win_n
        motion_win = norm_motion[i:j]
        audio_win: List[float] = []
        ok = True
        for t in times[i:j]:
            idx = t // hop_ms
            if idx >= len(audio_rms):
                ok = False
                break
            audio_win.append(audio_rms[idx])
        if not ok:
            continue
        corr = _pearson(motion_win, audio_win)
        mean_motion = sum(motion_win) / len(motion_win)
        if corr >= CORR_MIN_SCORE and mean_motion >= CORR_MIN_MOTION:
            windows.append((times[i], times[j - 1], corr))

    return _merge_speaking_windows(windows, hop_ms)
