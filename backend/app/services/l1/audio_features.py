"""
L1 Stage 5: Whole-file audio features.

  - integrated LUFS + true-peak via ffmpeg loudnorm 2-pass JSON output
  - silence intervals via pydub
  - musicality detection via spectral flatness and onset-envelope variance
  - if musical: beat-track BPM + onset grid via librosa
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    integrated_lufs: float
    true_peak_db: float
    is_musical: bool
    bpm: float = 0.0
    onsets_ms: List[int] = field(default_factory=list)
    silence_intervals: List[dict] = field(default_factory=list)
    # Prosody / rhythm signals for cut-timing (snap cuts to emphasis + pauses).
    energy_peaks_ms: List[int] = field(default_factory=list)   # RMS emphasis peaks
    pause_map: List[dict] = field(default_factory=list)        # {start_ms,end_ms}
    rms_db: List[float] = field(default_factory=list)          # coarse energy envelope
    pitch_hz: List[float] = field(default_factory=list)        # coarse f0 contour (0=unvoiced)
    prosody_hop_ms: int = 0                                    # hop for rms_db / pitch_hz
    # Fixed-hop, normalized energy envelope for cross-file simultaneity (multicam)
    # detection. Same recording from two cameras -> near-identical sync_env.
    sync_env: List[float] = field(default_factory=list)
    sync_hop_ms: int = 0


def _ffmpeg_loudnorm_pass1(wav_path: str) -> tuple[float, float]:
    """Returns (integrated_lufs, true_peak_db) via ffmpeg loudnorm dryrun."""
    cmd = [
        "ffmpeg", "-i", wav_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json",
        "-f", "null", "-",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    stderr = result.stderr
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, re.DOTALL)
    if not match:
        logger.warning("ffmpeg loudnorm output not parseable; defaulting to 0/0")
        return 0.0, 0.0
    data = json.loads(match.group(0))
    try:
        return float(data["input_i"]), float(data["input_tp"])
    except (KeyError, ValueError):
        return 0.0, 0.0


def _detect_silence(wav_path: str) -> List[dict]:
    from pydub import AudioSegment, silence

    seg = AudioSegment.from_file(wav_path)
    noise_floor = seg.dBFS - 16 if seg.dBFS != float("-inf") else -40
    ranges = silence.detect_silence(
        seg,
        min_silence_len=400,
        silence_thresh=noise_floor,
    )
    return [{"start_ms": int(s), "end_ms": int(e)} for s, e in ranges]


def _detect_musicality(wav_path: str) -> tuple[bool, float, List[int]]:
    """
    Heuristic: high spectral flatness + low onset-envelope variance => musical.
    Returns (is_musical, bpm, onsets_ms).
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    if y.size == 0:
        return False, 0.0, []

    flatness = float(librosa.feature.spectral_flatness(y=y).mean())
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_var = float(onset_env.var()) if onset_env.size else 0.0

    # Calibrated thresholds: speech has low flatness (<0.05) and high
    # onset-envelope variance. Music sits in the opposite regime.
    is_musical = flatness > 0.08 and onset_var > 1e-2

    if not is_musical:
        return False, 0.0, []

    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    onsets_ms = [int(librosa.frames_to_time(f, sr=sr) * 1000) for f in beat_frames]

    # librosa.beat.beat_track sometimes returns tempo as an array; normalize.
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0]) if len(tempo) else 0.0
    else:
        tempo = float(tempo)
    return True, tempo, onsets_ms


# Storage bound: downsample the continuous envelope/contour to at most this many
# points regardless of clip length, so a long video can't bloat the JSONB.
PROSODY_MAX_POINTS = 600
# A pause is a contiguous low-energy run at least this long.
PAUSE_MIN_MS = 250

# Cross-file sync fingerprint: fixed hop (so two files share a time grid) and a
# hard cap on length so correlating a corpus stays cheap.
SYNC_HOP_MS = 500
SYNC_MAX_POINTS = 2400  # ~20 min at 500ms


def _compute_prosody(wav_path: str) -> dict:
    """Energy envelope, pitch contour, emphasis peaks and a pause map.

    All librosa, CPU, single load of the 16k WAV. Pitch uses YIN (fast) rather
    than PYIN (accurate but minutes-slow) so this stays a few seconds even on
    long clips. Returns plain lists ready for JSONB storage.
    """
    import librosa
    import numpy as np

    out = {
        "energy_peaks_ms": [], "pause_map": [],
        "rms_db": [], "pitch_hz": [], "prosody_hop_ms": 0,
        "sync_env": [], "sync_hop_ms": 0,
    }
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
    except Exception:
        logger.exception("Prosody: failed to load %s", wav_path)
        return out
    if y.size == 0:
        return out

    # --- RMS energy envelope at 50ms frames ---
    hop = int(sr * 0.05)
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    if rms.size == 0:
        return out
    times_ms = (librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=hop) * 1000)
    rms_db = 20.0 * np.log10(rms + 1e-6)

    # Reference loudness = the loud (speech) part of the clip, robust to how much
    # silence the clip contains. Used to scale peak + pause thresholds in dB.
    speech_ref = float(np.percentile(rms_db, 90))

    # --- emphasis peaks (local dB maxima well above neighbours) ---
    try:
        delta = max(3.0, float(np.std(rms_db)) * 0.4)
        peaks = librosa.util.peak_pick(
            rms_db, pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=delta, wait=8
        )
        # Keep only genuinely loud peaks (ignore bumps in the noise floor).
        out["energy_peaks_ms"] = [
            int(times_ms[p]) for p in peaks if rms_db[p] >= speech_ref - 12.0
        ]
    except Exception:
        out["energy_peaks_ms"] = []

    # --- pause map (contiguous quiet runs relative to speech level) ---
    floor = speech_ref - 25.0
    out["pause_map"] = _runs_below(rms_db, times_ms, floor, PAUSE_MIN_MS, hop_ms=50)

    # --- coarse pitch contour via YIN at 100ms hop ---
    phop = int(sr * 0.1)
    try:
        f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr, frame_length=2048, hop_length=phop)
        f0 = np.where(np.isfinite(f0), f0, 0.0)
    except Exception:
        f0 = np.zeros(0)

    # --- downsample both series to a bounded length, on a common hop ---
    dur_ms = float(times_ms[-1]) if times_ms.size else 0.0
    hop_ms = max(100, int(np.ceil((dur_ms / PROSODY_MAX_POINTS))) if dur_ms else 100)
    out["prosody_hop_ms"] = hop_ms
    out["rms_db"] = _resample_series(rms_db, times_ms, hop_ms, dur_ms)
    if f0.size:
        f0_times = librosa.frames_to_time(np.arange(f0.size), sr=sr, hop_length=phop) * 1000
        out["pitch_hz"] = _resample_series(f0, f0_times, hop_ms, dur_ms)

    # --- fixed-hop normalized sync envelope (cross-file simultaneity) ---
    if dur_ms > 0 and times_ms.size > 1:
        grid = np.arange(0.0, min(dur_ms, SYNC_MAX_POINTS * SYNC_HOP_MS), SYNC_HOP_MS)
        env = np.interp(grid, times_ms, rms_db)
        lo, hi = float(env.min()), float(env.max())
        if hi > lo:
            env = (env - lo) / (hi - lo)
        else:
            env = np.zeros_like(env)
        out["sync_env"] = [round(float(x), 3) for x in env]
        out["sync_hop_ms"] = SYNC_HOP_MS
    return out


def _runs_below(values, times_ms, thresh: float, min_ms: int, hop_ms: int) -> List[dict]:
    runs: List[dict] = []
    start = None
    for i, v in enumerate(values):
        if v < thresh:
            if start is None:
                start = i
        else:
            if start is not None:
                s_ms = int(times_ms[start]); e_ms = int(times_ms[i])
                if e_ms - s_ms >= min_ms:
                    runs.append({"start_ms": s_ms, "end_ms": e_ms})
                start = None
    if start is not None:
        s_ms = int(times_ms[start]); e_ms = int(times_ms[-1]) + hop_ms
        if e_ms - s_ms >= min_ms:
            runs.append({"start_ms": s_ms, "end_ms": e_ms})
    return runs


def _resample_series(values, times_ms, hop_ms: int, dur_ms: float) -> List[float]:
    import numpy as np
    if values is None or len(values) == 0 or dur_ms <= 0:
        return []
    grid = np.arange(0.0, dur_ms + hop_ms, hop_ms)
    sampled = np.interp(grid, times_ms, values)
    return [round(float(x), 1) for x in sampled]


def compute_audio_features(wav_path: str) -> AudioFeatures:
    lufs, tp = _ffmpeg_loudnorm_pass1(wav_path)
    silences = _detect_silence(wav_path)
    is_musical, bpm, onsets = _detect_musicality(wav_path)
    prosody = _compute_prosody(wav_path)
    return AudioFeatures(
        integrated_lufs=lufs,
        true_peak_db=tp,
        is_musical=is_musical,
        bpm=bpm,
        onsets_ms=onsets,
        silence_intervals=silences,
        energy_peaks_ms=prosody["energy_peaks_ms"],
        pause_map=prosody["pause_map"],
        rms_db=prosody["rms_db"],
        pitch_hz=prosody["pitch_hz"],
        prosody_hop_ms=prosody["prosody_hop_ms"],
        sync_env=prosody["sync_env"],
        sync_hop_ms=prosody["sync_hop_ms"],
    )
