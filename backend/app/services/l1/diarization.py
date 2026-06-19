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
  * "pyannote" (default): pyannote.audio 3.1 -- a full diarization pipeline
    (VAD -> neural segmentation -> embedding -> clustering -> overlap-aware
    resegmentation). This is real, SOTA-class diarization and the authority
    when available. It runs on GPU via `ml_device` and needs an HF token plus a
    one-time license acceptance for the gated models; on any failure (no token,
    not installed, CPU-only local dev) it falls back to the embedding backend
    below, so diarization never hard-fails. We run pyannote on the WAV and then
    attach its speaker timeline to Whisper's words by maximum temporal overlap.
  * "neural": a pretrained neural speaker embedding (Resemblyzer's
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

# Post-clustering label smoothing. A speaker "run" this short that is flanked by
# the SAME other speaker (or sits at a clip edge next to one speaker) is almost
# always a clustering blip -- e.g. the last word of a sentence flipped to a
# phantom second speaker -- so we fold it back into the surrounding voice.
SMOOTH_MAX_RUN_MS = 700
SMOOTH_MAX_RUN_WORDS = 2


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

    # Strong backend first: pyannote (GPU). On any failure -- not installed, no
    # HF token, model not licensed, runtime error -- we fall through to the
    # embedding path below so diarization is never a hard dependency on it.
    if backend == "pyannote":
        try:
            res = _diarize_pyannote(wav_path, word_list, min_speakers, max_speakers)
            if res is not None:
                return res
            logger.warning("pyannote backend unavailable for %s; using embedding fallback.", wav_path)
        except Exception:
            logger.exception("pyannote diarization failed for %s; using embedding fallback.", wav_path)
        backend = "neural"

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
# Backend: pyannote.audio 3.1 (VAD + neural segmentation + clustering)
# ---------------------------------------------------------------------------

_PYANNOTE_PIPELINE = None  # lazily-created, process-wide pipeline (or False)
_PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"


def _get_pyannote_pipeline():
    """Load the pyannote diarization pipeline once per process, on GPU when one
    is present. Returns None (so the caller can fall back) when pyannote isn't
    installed, the HF token is missing, or the gated model isn't licensed."""
    global _PYANNOTE_PIPELINE
    if _PYANNOTE_PIPELINE is None:
        try:
            import torch
            from pyannote.audio import Pipeline

            from app.config import get_settings
            from app.services.ml_device import torch_device

            token = get_settings().huggingface_token or None
            # PyTorch 2.6 flipped torch.load's default to weights_only=True, which
            # rejects pyannote's official checkpoints (they pickle a TorchVersion
            # global -> UnpicklingError). The weights come from the gated HF repo
            # we just authenticated against, so a full load is safe; force
            # weights_only=False for the duration of the load, then restore.
            _orig_torch_load = torch.load

            def _trusting_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return _orig_torch_load(*args, **kwargs)

            torch.load = _trusting_load
            try:
                pipe = Pipeline.from_pretrained(_PYANNOTE_MODEL, use_auth_token=token)
            finally:
                torch.load = _orig_torch_load
            if pipe is None:
                # pyannote returns None (not an exception) when the model is
                # gated and the token is missing or hasn't accepted the license.
                raise RuntimeError(
                    f"{_PYANNOTE_MODEL} not accessible -- set HF_TOKEN and accept "
                    "the model license on huggingface.co."
                )
            pipe.to(torch.device(torch_device()))
            logger.info("pyannote diarization pipeline loaded (%s).", torch_device())
            _PYANNOTE_PIPELINE = pipe
        except Exception:
            logger.warning("pyannote backend unavailable; will fall back.", exc_info=True)
            _PYANNOTE_PIPELINE = False  # sentinel: tried and failed
    return _PYANNOTE_PIPELINE or None


def _diarize_pyannote(
    wav_path: str,
    words: Sequence[dict],
    min_speakers: int,
    max_speakers: int,
) -> Optional[DiarizationResult]:
    """Run pyannote, then attach its speaker timeline to our words by max
    temporal overlap. Returns None if the pipeline can't be loaded/run."""
    pipe = _get_pyannote_pipeline()
    if pipe is None:
        return None

    kwargs: Dict[str, int] = {}
    if max_speakers and max_speakers > 0:
        kwargs["max_speakers"] = int(max_speakers)
    if min_speakers and min_speakers > 1:
        kwargs["min_speakers"] = int(min_speakers)

    annotation = pipe(wav_path, **kwargs)

    # (start_ms, end_ms, raw_label), chronological.
    segs: List[Tuple[int, int, str]] = [
        (int(turn.start * 1000), int(turn.end * 1000), str(label))
        for turn, _track, label in annotation.itertracks(yield_label=True)
    ]
    if not segs:
        return None
    segs.sort(key=lambda s: s[0])

    # Stable S0.. names by first appearance, matching the embedding path.
    raw_to_name: Dict[str, str] = {}
    for _s, _e, lab in segs:
        if lab not in raw_to_name:
            raw_to_name[lab] = f"S{len(raw_to_name)}"

    speaker_by_word: List[Optional[str]] = []
    for w in words:
        ws = int(w.get("start_ms", 0))
        we = int(w.get("end_ms", ws))
        if we < ws:
            we = ws
        best_lab: Optional[str] = None
        best_ov = 0
        for s, e, lab in segs:
            ov = min(we, e) - max(ws, s)
            if ov > best_ov:
                best_ov, best_lab = ov, lab
        if best_lab is None:  # word in a gap -> nearest segment by midpoint
            mid = (ws + we) // 2
            best_lab = min(segs, key=lambda s: abs(((s[0] + s[1]) // 2) - mid))[2]
        speaker_by_word.append(raw_to_name[best_lab])

    _smooth_speakers(words, speaker_by_word)
    turns = _build_turns(words, speaker_by_word)
    num_speakers = len({v for v in speaker_by_word if v})
    return DiarizationResult(
        speaker_by_word=speaker_by_word,
        num_speakers=num_speakers,
        turns=turns,
    )


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

    _smooth_speakers(words, speaker_by_word)

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


def _smooth_speakers(
    words: Sequence[dict], speaker_by_word: List[Optional[str]]
) -> None:
    """In-place: relabel tiny one-off speaker runs that are clustering noise.

    Builds maximal runs of equal labels and, for any run that is very short
    (<= SMOOTH_MAX_RUN_MS and <= SMOOTH_MAX_RUN_WORDS), folds it into the
    surrounding voice when that voice is unambiguous: either the same speaker
    sits on BOTH sides, or the run is at a clip edge with exactly one labelled
    neighbour. This kills the 'last word of a sentence flips to a phantom S1'
    failure that splits one utterance into two clips."""
    n = len(words)
    if n < 2:
        return

    runs: List[List] = []  # [start_idx, end_idx_inclusive, speaker]
    i = 0
    while i < n:
        spk = speaker_by_word[i]
        j = i
        while j + 1 < n and speaker_by_word[j + 1] == spk:
            j += 1
        runs.append([i, j, spk])
        i = j + 1
    if len(runs) < 2:
        return

    # A speaker is a "phantom" if its ENTIRE footprint across the clip is within
    # the smoothing limits -- i.e. it exists only as one of these tiny blips. We
    # only ever relabel phantoms, so a real speaker's brief turn (a one-word
    # interjection by someone who also speaks elsewhere) is never erased.
    word_count: Dict[str, int] = {}
    for spk in speaker_by_word:
        if spk is not None:
            word_count[spk] = word_count.get(spk, 0) + 1

    def _is_phantom(spk: Optional[str]) -> bool:
        return spk is not None and word_count.get(spk, 0) <= SMOOTH_MAX_RUN_WORDS

    for r, (a, b, spk) in enumerate(runs):
        if not _is_phantom(spk):
            continue
        dur = int(words[b].get("end_ms", 0)) - int(words[a].get("start_ms", 0))
        if dur > SMOOTH_MAX_RUN_MS or (b - a + 1) > SMOOTH_MAX_RUN_WORDS:
            continue
        prev_spk = runs[r - 1][2] if r > 0 else None
        next_spk = runs[r + 1][2] if r < len(runs) - 1 else None
        target: Optional[str] = None
        if prev_spk is not None and prev_spk == next_spk and prev_spk != spk:
            target = prev_spk                      # sandwiched blip
        elif prev_spk is None and next_spk not in (None, spk):
            target = next_spk                      # leading-edge blip
        elif next_spk is None and prev_spk not in (None, spk):
            target = prev_spk                      # trailing-edge blip ("matters")
        if target is not None:
            for k in range(a, b + 1):
                speaker_by_word[k] = target


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
