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


def compute_audio_features(wav_path: str) -> AudioFeatures:
    lufs, tp = _ffmpeg_loudnorm_pass1(wav_path)
    silences = _detect_silence(wav_path)
    is_musical, bpm, onsets = _detect_musicality(wav_path)
    return AudioFeatures(
        integrated_lufs=lufs,
        true_peak_db=tp,
        is_musical=is_musical,
        bpm=bpm,
        onsets_ms=onsets,
        silence_intervals=silences,
    )
