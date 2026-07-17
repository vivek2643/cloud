"""
L1 Stage 5: Whole-file audio features.

  - integrated LUFS + true-peak via ffmpeg loudnorm 2-pass JSON output
  - silence intervals via pydub
  - musicality detection via spectral flatness and onset-envelope variance
  - if musical: beat-track BPM + onset grid via librosa
  - coarse prosody: RMS energy envelope, sampled on a bounded hop, for grading
    a speech boundary by more than transcript/gap-length alone
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AudioFeatures:
    integrated_lufs: float
    true_peak_db: float
    is_musical: bool
    bpm: float = 0.0
    onsets_ms: List[int] = field(default_factory=list)
    silence_intervals: List[dict] = field(default_factory=list)
    # Coarse energy envelope (dB), sampled every prosody_hop_ms -- grades a
    # speech boundary's cleanliness by more than transcript/gap-length alone.
    rms_db: List[float] = field(default_factory=list)
    prosody_hop_ms: int = 0
    # audio_and_audit.plan.md Phase 4: coarse musical structure, from the SAME
    # onset/beat signal already computed for musicality -- a prior for pacing
    # ("land the drop on the climax"), never asserted as ground truth. Empty/
    # None when undetected (not musical, or too short/ambiguous to segment).
    sections: List[dict] = field(default_factory=list)
    drop_ms: Optional[int] = None


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


def _detect_structure(onset_env, onset_times_ms, beat_times_ms,
                      bpm: float) -> Tuple[List[dict], Optional[int]]:
    """Coarse musical structure from the SAME onset/beat signal already
    computed for musicality detection (audio_and_audit.plan.md Phase 4) --
    approximate, a prior for the brain to target ('land the drop on the
    climax'), never asserted as ground truth. Pure numpy (no librosa/file
    I/O) so it's testable with synthetic arrays.

      * drop: the beat-grid instant nearest the onset-strength curve's
        largest single value -- snapped to a real beat (an addressable
        instant), not an arbitrary frame, mirroring the video salience
        model's own "largest local contrast" idea on this one axis.
      * sections: the onset-strength envelope chunked into coarse ~8-bar
        windows, each boundary relocated to the biggest novelty (frame-to-
        frame change) within one bar of it and snapped to the bar grid --
        no genre assumptions, never a beat-level (too fine) or whole-track
        (too coarse) boundary.

    Empty sections / None drop on any degenerate input (too short, no beat
    grid, unmusical bpm) -- never a fabricated boundary."""
    import numpy as np

    onset_env = np.asarray(onset_env, dtype=float)
    onset_times_ms = np.asarray(onset_times_ms, dtype=float)
    beat_times_ms = np.asarray(beat_times_ms, dtype=float)
    if onset_env.size == 0 or beat_times_ms.size < 4 or bpm <= 0:
        return [], None

    peak_ms = float(onset_times_ms[int(np.argmax(onset_env))])
    drop_ms = int(beat_times_ms[int(np.argmin(np.abs(beat_times_ms - peak_ms)))])

    bar_ms = 4.0 * (60000.0 / bpm)
    total_ms = float(onset_times_ms[-1]) if onset_times_ms.size else 0.0
    window_ms = 8.0 * bar_ms
    if bar_ms <= 0 or total_ms < window_ms * 1.5:
        return [], drop_ms

    novelty = np.abs(np.diff(onset_env, prepend=onset_env[0]))
    bounds_ms = [0.0]
    cursor = window_ms
    while cursor < total_ms - window_ms * 0.5:
        # Relocate the coarse boundary to the biggest novelty within one bar
        # of it, so a section starts on a real change, not a fixed clock tick.
        lo = int(np.searchsorted(onset_times_ms, cursor - bar_ms))
        hi = int(np.searchsorted(onset_times_ms, cursor + bar_ms))
        boundary_ms = float(onset_times_ms[lo + int(np.argmax(novelty[lo:hi]))]) \
            if hi > lo else cursor
        nearest_bar = round(boundary_ms / bar_ms) * bar_ms
        if nearest_bar - bounds_ms[-1] >= bar_ms:      # never a degenerate sliver
            bounds_ms.append(nearest_bar)
        cursor += window_ms
    bounds_ms.append(total_ms)

    sections = [
        {"start_ms": int(bounds_ms[i]), "end_ms": int(bounds_ms[i + 1])}
        for i in range(len(bounds_ms) - 1)
        if bounds_ms[i + 1] - bounds_ms[i] >= bar_ms
    ]
    return sections, drop_ms


def _detect_musicality(wav_path: str) -> Tuple[bool, float, List[int], List[dict], Optional[int]]:
    """
    Heuristic: high spectral flatness + low onset-envelope variance => musical.
    Returns (is_musical, bpm, onsets_ms, sections, drop_ms).
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    if y.size == 0:
        return False, 0.0, [], [], None

    flatness = float(librosa.feature.spectral_flatness(y=y).mean())
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_var = float(onset_env.var()) if onset_env.size else 0.0

    # Calibrated thresholds: speech has low flatness (<0.05) and high
    # onset-envelope variance. Music sits in the opposite regime.
    is_musical = flatness > 0.08 and onset_var > 1e-2

    if not is_musical:
        return False, 0.0, [], [], None

    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    onsets_ms = [int(librosa.frames_to_time(f, sr=sr) * 1000) for f in beat_frames]

    # librosa.beat.beat_track sometimes returns tempo as an array; normalize.
    if hasattr(tempo, "__len__"):
        tempo = float(tempo[0]) if len(tempo) else 0.0
    else:
        tempo = float(tempo)

    onset_times_ms = librosa.frames_to_time(np.arange(onset_env.size), sr=sr) * 1000.0
    sections, drop_ms = _detect_structure(onset_env, onset_times_ms, onsets_ms, tempo)
    return True, tempo, onsets_ms, sections, drop_ms


# Storage bound: downsample the continuous envelope to at most this many points
# regardless of clip length, so a long video can't bloat the JSONB.
PROSODY_MAX_POINTS = 600


def _compute_prosody(wav_path: str) -> dict:
    """Coarse RMS energy envelope (dB), sampled on a bounded hop -- a prosody
    signal the speech-boundary grader consumes. All librosa, CPU, single load
    of the 16k WAV. Returns plain lists ready for JSONB storage.
    """
    import librosa
    import numpy as np

    out = {"rms_db": [], "prosody_hop_ms": 0}
    try:
        y, sr = librosa.load(wav_path, sr=16000, mono=True)
    except Exception:
        logger.exception("Prosody: failed to load %s", wav_path)
        return out
    if y.size == 0:
        return out

    # RMS energy envelope at 50ms frames, then downsampled to a bounded hop.
    hop = int(sr * 0.05)
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    if rms.size == 0:
        return out
    times_ms = (librosa.frames_to_time(np.arange(rms.size), sr=sr, hop_length=hop) * 1000)
    rms_db = 20.0 * np.log10(rms + 1e-6)

    dur_ms = float(times_ms[-1]) if times_ms.size else 0.0
    hop_ms = max(100, int(np.ceil((dur_ms / PROSODY_MAX_POINTS))) if dur_ms else 100)
    out["prosody_hop_ms"] = hop_ms
    out["rms_db"] = _resample_series(rms_db, times_ms, hop_ms, dur_ms)
    return out


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
    is_musical, bpm, onsets, sections, drop_ms = _detect_musicality(wav_path)
    prosody = _compute_prosody(wav_path)
    return AudioFeatures(
        integrated_lufs=lufs,
        true_peak_db=tp,
        is_musical=is_musical,
        bpm=bpm,
        onsets_ms=onsets,
        silence_intervals=silences,
        rms_db=prosody["rms_db"],
        prosody_hop_ms=prosody["prosody_hop_ms"],
        sections=sections,
        drop_ms=drop_ms,
    )
