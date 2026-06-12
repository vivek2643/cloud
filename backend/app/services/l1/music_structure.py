"""
L1 (music-only): deep musical-structure analysis.

This is the "heavy" analysis that turns an uploaded song into something the L3
editor can cut a montage TO -- not just raw beats, but the musical scaffolding:

  - tempo + a beat grid (librosa beat tracking)
  - a downbeat / bar grid (where each musical bar starts)  [4/4 assumption]
  - sections / phrases (intro, verse, chorus, drop ...) via spectral segmentation
  - an energy / intensity envelope (so you cut on the build and the drop)
  - musical key (Krumhansl-Schmugler profile correlation)
  - a PHRASE cut-cost grid: a "cut ON these instants" channel whose cost dips at
    downbeats and section boundaries -- the musically strong places to cut.

One librosa load of a 22.05 kHz mono signal, CPU only. Every sub-feature is
wrapped so a single failure degrades that field to empty rather than failing the
whole stage. Speech tools (Whisper/diarization/prosody) are NOT run here -- a
song is not dialogue.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.services.l1.cut_grid_common import hit_cost_curve

logger = logging.getLogger(__name__)

# Sample rate for the musical pass. 22.05 kHz is the librosa default and is
# plenty for chroma/onset/segmentation while staying cheap.
MUSIC_SR = 22050

# Energy envelope: bound the stored curve regardless of song length.
ENERGY_MAX_POINTS = 800

# Phrase cut grid: dips at downbeats and section boundaries.
PHRASE_HOP_MS = 100
PHRASE_TOL_MS = 90  # a hair wider than the beat window -- bar/section feel

# Target one section boundary per ~18s, clamped to a sane band.
SECTION_TARGET_S = 18.0
SECTION_MIN = 2
SECTION_MAX = 10

# Krumhansl-Kessler key profiles (major / minor), correlated against the
# 12-bin chroma mean to estimate the tonic + mode.
_KK_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KK_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


@dataclass
class MusicStructure:
    has_music: bool = False
    bpm: float = 0.0
    key: Optional[str] = None  # e.g. "A minor", "F# major"
    beat_times_ms: List[int] = field(default_factory=list)
    downbeat_times_ms: List[int] = field(default_factory=list)
    sections: List[Dict] = field(default_factory=list)  # {start_ms,end_ms,label}
    energy_hop_ms: int = 0
    energy: List[float] = field(default_factory=list)    # 0..1 intensity curve
    # Phrase cut grid (cut ON downbeats / section boundaries).
    phrase_cut_hop_ms: int = PHRASE_HOP_MS
    phrase_cut_cost: List[float] = field(default_factory=list)
    phrase_cut_points: List[Dict] = field(default_factory=list)  # {ts_ms,kind,score}


def _corr(a: List[float], b: List[float]) -> float:
    """Pearson correlation of two equal-length vectors."""
    n = len(a)
    if n == 0 or n != len(b):
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = sum((a[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((b[i] - mb) ** 2 for i in range(n)) ** 0.5
    if da == 0 or db == 0:
        return 0.0
    return num / (da * db)


def _estimate_key(chroma_mean: List[float]) -> Optional[str]:
    """Correlate the mean chroma against rotated major/minor KK profiles."""
    if not chroma_mean or len(chroma_mean) != 12 or sum(chroma_mean) <= 0:
        return None
    best_score = -2.0
    best_name: Optional[str] = None
    for tonic in range(12):
        rot = chroma_mean[tonic:] + chroma_mean[:tonic]
        maj = _corr(rot, _KK_MAJOR)
        minr = _corr(rot, _KK_MINOR)
        if maj >= best_score:
            best_score, best_name = maj, f"{_PITCH_NAMES[tonic]} major"
        if minr >= best_score:
            best_score, best_name = minr, f"{_PITCH_NAMES[tonic]} minor"
    return best_name


def _phase_downbeats(beats_ms: List[int], onset_at_beat: List[float], meter: int = 4) -> List[int]:
    """Pick which of the `meter` beat phases is the downbeat by maximizing the
    onset energy that lands on bar-start beats. Assumes a constant 4/4 meter --
    a pragmatic heuristic (true downbeat tracking needs a dedicated model)."""
    if not beats_ms:
        return []
    if len(beats_ms) != len(onset_at_beat) or meter <= 1:
        return beats_ms[::meter] if beats_ms else []
    best_phase, best_energy = 0, -1.0
    for phase in range(meter):
        e = sum(onset_at_beat[i] for i in range(phase, len(beats_ms), meter))
        if e > best_energy:
            best_energy, best_phase = e, phase
    return [beats_ms[i] for i in range(best_phase, len(beats_ms), meter)]


def _label_sections(boundaries_ms: List[int], duration_ms: int, seg_feats) -> List[Dict]:
    """Turn boundary timestamps into [{start_ms,end_ms,label}] spans, labeling
    repeated material with the same letter (A/B/C...) via cheap clustering of
    each segment's mean feature vector."""
    import numpy as np

    edges = [0] + [b for b in boundaries_ms if 0 < b < duration_ms] + [duration_ms]
    edges = sorted(set(edges))
    spans: List[Tuple[int, int]] = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
    spans = [(s, e) for s, e in spans if e - s >= 1000]  # drop sub-1s slivers
    if not spans:
        return []

    labels = [None] * len(spans)
    try:
        # Mean feature per span -> greedy nearest-centroid clustering so a chorus
        # that recurs gets the same letter.
        n_frames = seg_feats.shape[1]
        means = []
        for s, e in spans:
            i0 = int(n_frames * s / duration_ms)
            i1 = max(i0 + 1, int(n_frames * e / duration_ms))
            means.append(seg_feats[:, i0:i1].mean(axis=1))
        means = np.array(means)
        centroids: List = []
        for i, m in enumerate(means):
            assigned = None
            for ci, c in enumerate(centroids):
                denom = (np.linalg.norm(m) * np.linalg.norm(c)) or 1.0
                if float(np.dot(m, c) / denom) > 0.92:
                    assigned = ci
                    break
            if assigned is None:
                centroids.append(m)
                assigned = len(centroids) - 1
            labels[i] = assigned
    except Exception:
        labels = list(range(len(spans)))

    out: List[Dict] = []
    for (s, e), lab in zip(spans, labels):
        letter = chr(ord("A") + (lab if isinstance(lab, int) else 0) % 26)
        out.append({"start_ms": int(s), "end_ms": int(e), "label": letter})
    return out


def compute_music_structure(audio_path: str, *, duration_ms: int = 0) -> MusicStructure:
    """Full musical-structure pass over an audio file. `audio_path` may be the
    raw upload or a demuxed wav -- librosa handles common formats."""
    try:
        import librosa
        import numpy as np
    except Exception:
        logger.exception("music_structure: librosa/numpy unavailable")
        return MusicStructure(has_music=False)

    try:
        y, sr = librosa.load(audio_path, sr=MUSIC_SR, mono=True)
    except Exception:
        logger.exception("music_structure: failed to load %s", audio_path)
        return MusicStructure(has_music=False)
    if y.size == 0:
        return MusicStructure(has_music=False)

    dur_ms = duration_ms or int(librosa.get_duration(y=y, sr=sr) * 1000)
    ms = MusicStructure(has_music=True)

    # --- tempo + beats ---
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    try:
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        tempo = float(tempo[0]) if hasattr(tempo, "__len__") and len(tempo) else float(tempo)
        ms.bpm = round(tempo, 1)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        ms.beat_times_ms = [int(t * 1000) for t in beat_times]
    except Exception:
        logger.exception("music_structure: beat tracking failed")
        beat_frames = np.array([], dtype=int)

    # --- downbeats (4/4 phase pick from onset energy on each beat) ---
    try:
        if len(beat_frames):
            onset_at_beat = [float(onset_env[min(f, len(onset_env) - 1)]) for f in beat_frames]
            ms.downbeat_times_ms = _phase_downbeats(ms.beat_times_ms, onset_at_beat, meter=4)
    except Exception:
        logger.exception("music_structure: downbeat estimation failed")

    # --- energy / intensity envelope (RMS, normalized 0..1) ---
    try:
        hop = int(sr * 0.1)
        rms = librosa.feature.rms(y=y, hop_length=hop)[0]
        if rms.size:
            n_keep = min(ENERGY_MAX_POINTS, rms.size)
            idx = np.linspace(0, rms.size - 1, n_keep).astype(int)
            env = rms[idx]
            lo, hi = float(env.min()), float(env.max())
            env = (env - lo) / (hi - lo) if hi > lo else np.zeros_like(env)
            ms.energy = [round(float(x), 3) for x in env]
            ms.energy_hop_ms = max(1, int(dur_ms / max(1, n_keep)))
    except Exception:
        logger.exception("music_structure: energy envelope failed")

    # --- key (Krumhansl on mean chroma) ---
    seg_feats = None
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        ms.key = _estimate_key([float(x) for x in chroma.mean(axis=1)])
    except Exception:
        logger.exception("music_structure: key estimation failed")
        chroma = None

    # --- sections / phrases (spectral segmentation) ---
    try:
        if chroma is None:
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # Stack normalized chroma + mfcc as the segmentation feature matrix.
        feats = np.vstack([
            librosa.util.normalize(chroma, axis=0),
            librosa.util.normalize(mfcc, axis=0),
        ])
        seg_feats = feats
        # beat-synchronize to make boundaries land on musical time when we can.
        if len(beat_frames) > SECTION_MIN + 1:
            feats_sync = librosa.util.sync(feats, beat_frames, aggregate=np.median)
            frame_to_ms = ms.beat_times_ms
        else:
            feats_sync = feats
            frame_to_ms = None

        k = int(round((dur_ms / 1000.0) / SECTION_TARGET_S))
        k = max(SECTION_MIN, min(SECTION_MAX, k))
        k = min(k, feats_sync.shape[1] - 1) if feats_sync.shape[1] > 1 else 1
        if k >= 2:
            bounds = librosa.segment.agglomerative(feats_sync, k)
            if frame_to_ms is not None:
                boundaries_ms = [
                    frame_to_ms[b] for b in bounds if 0 <= b < len(frame_to_ms)
                ]
            else:
                bt = librosa.frames_to_time(bounds, sr=sr)
                boundaries_ms = [int(t * 1000) for t in bt]
            ms.sections = _label_sections(boundaries_ms, dur_ms, seg_feats)
    except Exception:
        logger.exception("music_structure: section segmentation failed")

    # --- phrase cut grid: dip at downbeats + section boundaries ---
    try:
        section_starts = [s["start_ms"] for s in ms.sections if s["start_ms"] > 0]
        hits = sorted(set(ms.downbeat_times_ms) | set(section_starts))
        if hits and dur_ms > 0:
            ms.phrase_cut_cost = hit_cost_curve(hits, dur_ms, PHRASE_HOP_MS, PHRASE_TOL_MS)
            section_set = set(section_starts)
            ms.phrase_cut_points = [
                {
                    "ts_ms": h,
                    "kind": "section" if h in section_set else "downbeat",
                    "score": 1.0 if h in section_set else 0.7,
                }
                for h in hits
            ]
    except Exception:
        logger.exception("music_structure: phrase cut grid failed")

    return ms
