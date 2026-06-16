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

Backends (set via DIARIZATION_BACKEND):
  * "neural" (default): a pretrained neural speaker embedding (Resemblyzer's
    GE2E d-vector -- same family as ECAPA x-vectors). This is what actually
    separates SAME-GENDER speakers, which the cheap classical features cannot;
    the model ships with the package (no download), runs on CPU in seconds, and
    needs no GPU. We embed each window, then cluster as before.
  * "mfcc": the original classical fallback -- MFCC + pitch statistics (librosa
    only), no model. Approximate but fully dependency-free; used automatically
    if the neural backend can't be imported, so diarization never hard-fails.

Either way the window embeddings are clustered (agglomerative, K chosen by
silhouette). With strong neural embeddings the silhouette is meaningful again,
so the "collapse to one speaker" gate stops firing on real two-person audio. A
`min_speakers` hint (e.g. from the VLM seeing two people, or from a verified
synced second camera) can additionally force K>=2 when the caller already knows
the clip is multi-speaker.
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
# Parsimony: adding a speaker must improve silhouette by at least this margin,
# else prefer the smaller count (Occam -- avoid splitting one voice into many).
SILHOUETTE_MARGIN = 0.03
# A cluster smaller than this fraction of windows (and below the absolute floor)
# is treated as noise and merged into its nearest neighbour -- kills the spurious
# 2-3 window "speakers" the embedding sometimes spawns.
MIN_CLUSTER_FRACTION = 0.10
MIN_CLUSTER_WINDOWS = 3

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
    backend: str = "neural",
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
    min_speakers: int = 1,
    window_ms: int = DEFAULT_WINDOW_MS,
    min_gap_ms: int = DEFAULT_MIN_GAP_MS,
) -> DiarizationResult:
    """Label each word with a per-file speaker id ("S0", "S1", ...).

    `words` is the flat, chronological word list (dicts with start_ms/end_ms).
    `min_speakers` forces at least that many clusters when the caller already
    knows the clip is multi-speaker (skips the single-speaker collapse gate).
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

        feats, valid_window_idx, neural = _embed_windows(wav_path, windows, backend)
        if feats is None or feats.shape[0] == 0:
            # Couldn't embed anything -> single speaker fallback.
            return _single_speaker(word_list, windows)

        labels = _cluster(feats, max_speakers, min_speakers=min_speakers,
                          normalized=neural)
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

_VOICE_ENCODER = None  # lazily-created, process-wide Resemblyzer encoder


def _get_voice_encoder():
    """Load the neural speaker-embedding model once per process. Returns None
    if Resemblyzer/torch isn't available, so the caller can fall back to MFCC."""
    global _VOICE_ENCODER
    if _VOICE_ENCODER is None:
        try:
            from resemblyzer import VoiceEncoder
            _VOICE_ENCODER = VoiceEncoder("cpu", verbose=False)
        except Exception:
            logger.warning("Neural diarization backend unavailable; falling back to mfcc.")
            _VOICE_ENCODER = False  # sentinel: tried and failed
    return _VOICE_ENCODER or None


def _embed_windows(
    wav_path: str,
    windows: List[Tuple[int, int, List[int]]],
    backend: str,
) -> Tuple[Optional[np.ndarray], List[int], bool]:
    """Return (feature_matrix, valid_window_indices, is_neural).

    `is_neural` tells the clusterer the vectors are L2-normalized d-vectors (so
    it normalizes instead of z-scoring). Falls back to MFCC when the neural
    backend is requested but can't load, so diarization never hard-fails."""
    import librosa

    y, _sr = librosa.load(wav_path, sr=SR, mono=True)
    if y.size == 0:
        return None, [], False

    encoder = None
    if backend not in ("mfcc",):
        encoder = _get_voice_encoder()
        if encoder is None and backend != "neural":
            logger.warning("Unknown diarization backend %r; using mfcc.", backend)

    vecs: List[np.ndarray] = []
    valid: List[int] = []
    floor = int(MIN_WINDOW_AUDIO_MS * SR / 1000)
    for wi, (s_ms, e_ms, _idxs) in enumerate(windows):
        a = max(0, int(s_ms * SR / 1000))
        b = min(y.size, int(e_ms * SR / 1000))
        if b - a < floor:
            continue
        seg = y[a:b]
        vec = _neural_vector(seg, encoder) if encoder is not None else _mfcc_pitch_vector(seg, librosa)
        if vec is None:
            continue
        vecs.append(vec)
        valid.append(wi)

    if not vecs:
        return None, [], False
    return np.vstack(vecs).astype(np.float32), valid, encoder is not None


def _neural_vector(seg: np.ndarray, encoder) -> Optional[np.ndarray]:
    """One L2-normalized d-vector for a window via the neural encoder."""
    try:
        return encoder.embed_utterance(seg.astype(np.float32)).astype(np.float32)
    except Exception:
        return None


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

def _cluster(
    feats: np.ndarray,
    max_speakers: int,
    *,
    min_speakers: int = 1,
    normalized: bool = False,
) -> np.ndarray:
    """Cluster window embeddings, choosing K by silhouette. Returns a label per
    row of `feats`.

    `normalized=True` (neural d-vectors) -> L2-normalize so ward clustering acts
    on cosine geometry; otherwise z-score the classical MFCC features.
    `min_speakers>=2` forces at least that many clusters and skips the
    single-speaker collapse gate (use when the clip is known multi-speaker)."""
    n = feats.shape[0]
    min_k = max(1, int(min_speakers))
    if n < 2 or n <= min_k:
        # Can't form the requested clusters; honor the floor if it's >1.
        return np.zeros(n, dtype=int)

    if normalized:
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x = feats / norms
    else:
        mu = feats.mean(axis=0)
        sd = feats.std(axis=0)
        sd[sd == 0] = 1.0
        x = (feats - mu) / sd

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    k_lo = max(2, min_k)
    k_max = max(k_lo, min(max_speakers, n - 1))
    scored: Dict[int, Tuple[float, np.ndarray]] = {}
    for k in range(k_lo, k_max + 1):
        try:
            labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(x)
            scored[k] = (float(silhouette_score(x, labels)), labels)
        except Exception:
            continue
    if not scored:
        return np.zeros(n, dtype=int)

    best_score = max(s for s, _ in scored.values())
    # Parsimony: take the SMALLEST k whose silhouette is within a margin of the
    # best, so one voice isn't split into several near-equivalent clusters.
    best_k = min(k for k, (s, _) in scored.items() if s >= best_score - SILHOUETTE_MARGIN)
    best_labels = scored[best_k][1]

    # A known multi-speaker clip keeps K>=min_k regardless of the collapse gate.
    if min_k >= 2:
        return _merge_tiny(x, best_labels) if best_k >= 2 else np.zeros(n, dtype=int)
    if best_k == 1 or best_score < SILHOUETTE_MIN:
        return np.zeros(n, dtype=int)
    return _merge_tiny(x, best_labels)


def _merge_tiny(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Fold clusters too small to be a real speaker into their nearest surviving
    cluster (by centroid). Prevents spurious 2-3 window 'speakers'."""
    n = len(labels)
    floor = max(MIN_CLUSTER_WINDOWS, int(MIN_CLUSTER_FRACTION * n))
    counts = {lbl: int((labels == lbl).sum()) for lbl in set(labels.tolist())}
    big = [lbl for lbl, c in counts.items() if c >= floor]
    if len(big) <= 1:
        # Everything is "tiny" relative to the floor; keep the largest cluster
        # only when it dominates, else fall back to the original labels.
        return labels
    centroids = {lbl: x[labels == lbl].mean(axis=0) for lbl in big}
    out = labels.copy()
    for lbl, c in counts.items():
        if c >= floor:
            continue
        idx = np.where(labels == lbl)[0]
        for i in idx:
            nearest = min(big, key=lambda b: float(np.linalg.norm(x[i] - centroids[b])))
            out[i] = nearest
    # Re-pack labels to 0..k-1 in first-appearance order is handled in _assign.
    return out


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
