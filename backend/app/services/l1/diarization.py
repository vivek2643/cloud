"""
L1 Stage 6: speaker diarization (who-says-what), CPU-only.

The editor needs a *sense of speaker identity* -- "the person who said X also
said Y" -- so it can keep one speaker on the audio bed while cutting to other
angles, avoid stitching two people into one utterance, and structure
interview/podcast edits. It does NOT need to name people; a stable per-file
label ("S0", "S1", ...) is enough.

Design goals (per the universal-editor plan):
  - No GPU. No model download. Negligible timing impact.
  - Reuse Whisper's word timings as the segmentation -- we only need to label
    speech we already found, not re-detect speech.
  - Pluggable backend so a stronger embedding (ECAPA / pyannote) can be dropped
    in later via `DIARIZATION_BACKEND` without touching callers.

Default backend = "mfcc": build short windows over the word stream, embed each
with MFCC + pitch statistics (librosa only), pick the speaker count via
silhouette, and cluster (agglomerative). It is approximate but fast and
dependency-free, which is exactly what we want for a *soft* signal.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SR = 16000

# Window building: merge consecutive words into ~1.5s embedding windows, but
# break a window whenever there's a real silence gap (likely a speaker change).
DEFAULT_WINDOW_MS = 1500
DEFAULT_MIN_GAP_MS = 400

# Clustering controls.
DEFAULT_MAX_SPEAKERS = 8
# Below this silhouette score we treat the file as a single speaker rather than
# inventing a spurious second cluster out of noise.
SILHOUETTE_MIN = 0.12

# A window's audio must be at least this long to yield a stable MFCC vector.
MIN_WINDOW_AUDIO_MS = 200


@dataclass
class DiarizationResult:
    # One label per input word (aligned by index); None if undiarizable.
    speaker_by_word: List[Optional[str]] = field(default_factory=list)
    num_speakers: int = 0
    turns: List[Dict] = field(default_factory=list)  # [{start_ms,end_ms,speaker}]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def diarize(
    wav_path: str,
    words: Sequence[dict],
    *,
    backend: str = "mfcc",
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
    window_ms: int = DEFAULT_WINDOW_MS,
    min_gap_ms: int = DEFAULT_MIN_GAP_MS,
) -> DiarizationResult:
    """Label each word with a per-file speaker id ("S0", "S1", ...).

    `words` is the flat, chronological word list (dicts with start_ms/end_ms).
    Never raises: on any failure it returns an empty result so the caller can
    treat diarization as best-effort.
    """
    word_list = [w for w in words]
    n = len(word_list)
    if n == 0:
        return DiarizationResult()

    try:
        windows = _build_windows(word_list, window_ms, min_gap_ms)
        if not windows:
            return DiarizationResult(speaker_by_word=[None] * n)

        feats, valid_window_idx = _embed_windows(wav_path, windows, backend)
        if feats is None or feats.shape[0] == 0:
            # Couldn't embed anything -> single speaker fallback.
            return _single_speaker(word_list, windows)

        labels = _cluster(feats, max_speakers)
        return _assign(word_list, windows, valid_window_idx, labels)
    except Exception:
        logger.exception("Diarization failed for %s; leaving speakers unset.", wav_path)
        return DiarizationResult(speaker_by_word=[None] * n)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def _build_windows(
    words: Sequence[dict], window_ms: int, min_gap_ms: int
) -> List[Tuple[int, int, List[int]]]:
    """Group consecutive words into (start_ms, end_ms, [word_indices]) windows.

    A new window starts when the running window already spans `window_ms`, or
    when there is a silence gap > `min_gap_ms` (a likely turn boundary).
    Every word is assigned to exactly one window.
    """
    windows: List[Tuple[int, int, List[int]]] = []
    cur_idx: List[int] = []
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None

    for i, w in enumerate(words):
        s = int(w.get("start_ms", 0))
        e = int(w.get("end_ms", 0))
        if e < s:
            e = s
        if cur_start is None:
            cur_start, cur_end, cur_idx = s, e, [i]
            continue
        gap = s - cur_end
        dur = cur_end - cur_start
        if gap > min_gap_ms or dur >= window_ms:
            windows.append((cur_start, cur_end, cur_idx))
            cur_start, cur_end, cur_idx = s, e, [i]
        else:
            cur_end = max(cur_end, e)
            cur_idx.append(i)
    if cur_idx:
        windows.append((cur_start or 0, cur_end or 0, cur_idx))
    return windows


# ---------------------------------------------------------------------------
# Embedding (default: MFCC + pitch statistics, librosa-only)
# ---------------------------------------------------------------------------

def _embed_windows(
    wav_path: str,
    windows: List[Tuple[int, int, List[int]]],
    backend: str,
) -> Tuple[Optional[np.ndarray], List[int]]:
    """Return (feature_matrix, valid_window_indices) for embeddable windows."""
    if backend != "mfcc":
        # Future backends (ecapa / pyannote / resemblyzer) plug in here. We fall
        # back to MFCC so an unknown setting never silently disables diarization.
        logger.warning("Unknown diarization backend %r; using mfcc.", backend)

    import librosa

    y, _sr = librosa.load(wav_path, sr=SR, mono=True)
    if y.size == 0:
        return None, []

    vecs: List[np.ndarray] = []
    valid: List[int] = []
    for wi, (s_ms, e_ms, _idxs) in enumerate(windows):
        a = max(0, int(s_ms * SR / 1000))
        b = min(y.size, int(e_ms * SR / 1000))
        if b - a < int(MIN_WINDOW_AUDIO_MS * SR / 1000):
            continue
        seg = y[a:b]
        vec = _mfcc_pitch_vector(seg, librosa)
        if vec is None:
            continue
        vecs.append(vec)
        valid.append(wi)

    if not vecs:
        return None, []
    return np.vstack(vecs).astype(np.float32), valid


def _mfcc_pitch_vector(seg: np.ndarray, librosa) -> Optional[np.ndarray]:
    """A compact, speaker-discriminative vector for one short window.

    MFCC mean+std captures vocal-tract timbre; pitch (f0) median+std separates
    speakers MFCCs alone confuse (e.g. similar timbre, different register).
    """
    try:
        n_fft = 512 if seg.size >= 512 else int(2 ** np.floor(np.log2(max(seg.size, 16))))
        hop = max(128, n_fft // 4)
        mfcc = librosa.feature.mfcc(y=seg, sr=SR, n_mfcc=20, n_fft=n_fft, hop_length=hop)
        if mfcc.size == 0:
            return None
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)
        try:
            delta = librosa.feature.delta(mfcc)
            delta_mean = delta.mean(axis=1)
        except Exception:
            delta_mean = np.zeros_like(mfcc_mean)

        # Pitch statistics over voiced frames.
        pitch_feat = np.zeros(2, dtype=np.float32)
        try:
            f0 = librosa.yin(seg, fmin=65, fmax=400, sr=SR,
                             frame_length=n_fft, hop_length=hop)
            voiced = f0[np.isfinite(f0)]
            if voiced.size:
                pitch_feat = np.array(
                    [float(np.median(voiced)), float(np.std(voiced))], dtype=np.float32
                )
        except Exception:
            pass

        return np.concatenate([mfcc_mean, mfcc_std, delta_mean, pitch_feat]).astype(np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def _cluster(feats: np.ndarray, max_speakers: int) -> np.ndarray:
    """Cluster window embeddings, choosing K by silhouette. Returns a label per
    row of `feats`."""
    n = feats.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int)

    # Standardize so MFCC and pitch scales are comparable.
    mu = feats.mean(axis=0)
    sd = feats.std(axis=0)
    sd[sd == 0] = 1.0
    x = (feats - mu) / sd

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    k_max = max(1, min(max_speakers, n - 1))
    best_k = 1
    best_score = -1.0
    best_labels = np.zeros(n, dtype=int)
    for k in range(2, k_max + 1):
        try:
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
            score = float(silhouette_score(x, labels))
        except Exception:
            continue
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels

    if best_k == 1 or best_score < SILHOUETTE_MIN:
        return np.zeros(n, dtype=int)
    return best_labels


# ---------------------------------------------------------------------------
# Assignment + relabelling
# ---------------------------------------------------------------------------

def _single_speaker(
    words: Sequence[dict], windows: List[Tuple[int, int, List[int]]]
) -> DiarizationResult:
    spk = ["S0"] * len(words)
    turns = [{
        "start_ms": int(words[0].get("start_ms", 0)),
        "end_ms": int(words[-1].get("end_ms", 0)),
        "speaker": "S0",
    }] if words else []
    return DiarizationResult(speaker_by_word=spk, num_speakers=1, turns=turns)


def _assign(
    words: Sequence[dict],
    windows: List[Tuple[int, int, List[int]]],
    valid_window_idx: List[int],
    labels: np.ndarray,
) -> DiarizationResult:
    n = len(words)

    # window index -> raw cluster label (None for windows we couldn't embed).
    win_label: List[Optional[int]] = [None] * len(windows)
    for slot, wi in enumerate(valid_window_idx):
        win_label[wi] = int(labels[slot])

    # Fill embed-less windows from the nearest embedded neighbour.
    _fill_gaps(win_label)

    # Stable names by first appearance in time order.
    raw_to_name: Dict[int, str] = {}
    for lbl in win_label:
        if lbl is not None and lbl not in raw_to_name:
            raw_to_name[lbl] = f"S{len(raw_to_name)}"

    speaker_by_word: List[Optional[str]] = [None] * n
    for wi, (_s, _e, idxs) in enumerate(windows):
        lbl = win_label[wi]
        name = raw_to_name.get(lbl) if lbl is not None else None
        for i in idxs:
            speaker_by_word[i] = name

    turns = _build_turns(words, speaker_by_word)
    num_speakers = len({v for v in speaker_by_word if v})
    return DiarizationResult(
        speaker_by_word=speaker_by_word,
        num_speakers=num_speakers,
        turns=turns,
    )


def _fill_gaps(win_label: List[Optional[int]]) -> None:
    """In-place: give windows with no embedding the label of the nearest
    embedded window (prefer the previous one)."""
    last: Optional[int] = None
    for i in range(len(win_label)):
        if win_label[i] is not None:
            last = win_label[i]
        elif last is not None:
            win_label[i] = last
    # Backward pass for any leading gaps.
    nxt: Optional[int] = None
    for i in range(len(win_label) - 1, -1, -1):
        if win_label[i] is not None:
            nxt = win_label[i]
        elif nxt is not None:
            win_label[i] = nxt


def _build_turns(
    words: Sequence[dict], speaker_by_word: List[Optional[str]]
) -> List[Dict]:
    turns: List[Dict] = []
    for w, spk in zip(words, speaker_by_word):
        if spk is None:
            continue
        s = int(w.get("start_ms", 0))
        e = int(w.get("end_ms", 0))
        if turns and turns[-1]["speaker"] == spk:
            turns[-1]["end_ms"] = max(turns[-1]["end_ms"], e)
        else:
            turns.append({"start_ms": s, "end_ms": e, "speaker": spk})
    return turns
