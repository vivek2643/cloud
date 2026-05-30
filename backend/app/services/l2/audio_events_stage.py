"""
L2 Stage C: Audio event tagging (laughter, applause, music, etc.).

Uses panns-inference (CNN14 trained on AudioSet 527 classes) as a YAMNet
substitute that's pip-installable in any PyTorch env. Output:
  - audio_features.acoustic_tags     -> top-K aggregate labels
  - audio_features.event_segments    -> [{start_ms, end_ms, tag, score}, ...]

We re-use the 16 kHz mono WAV the L1 pipeline already extracted.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# panns-inference returns logits over AudioSet's 527 classes. Below is a small
# curated allow-list of edit-useful tags. The actual model returns more; we
# filter / canonicalize here so the database stays clean.
INTERESTING_TAGS = {
    "Speech":             "speech",
    "Laughter":           "laughter",
    "Applause":           "applause",
    "Music":              "music",
    "Singing":            "singing",
    "Cheering":           "cheering",
    "Crying, sobbing":    "crying",
    "Shout":              "shouting",
    "Whispering":         "whisper",
    "Silence":            "silence",
}

WINDOW_S = 2.0      # 2-second analysis windows
HOP_S = 1.0         # 1-second hop = 50% overlap
TOP_K = 5


@dataclass
class EventSegment:
    start_ms: int
    end_ms: int
    tag: str
    score: float


class _PannsEngine:
    _tagger = None

    @classmethod
    def get(cls):
        if cls._tagger is None:
            from panns_inference import AudioTagging  # type: ignore
            logger.info("Loading panns CNN14 audio tagger (CPU)...")
            cls._tagger = AudioTagging(checkpoint_path=None, device="cpu")
            logger.info("panns tagger ready.")
        return cls._tagger


def _tag_window(audio: np.ndarray, tagger) -> tuple[List[tuple[str, float]], dict]:
    """Run inference over one window, return (top-K tags, all label scores)."""
    audio_in = audio[None, :]  # add batch dim
    clipwise_output, _ = tagger.inference(audio_in)
    scores = clipwise_output[0]
    label_to_idx = tagger.labels  # list[str]; index = class id

    pairs: List[tuple[str, float]] = []
    for tag_name, canon in INTERESTING_TAGS.items():
        if tag_name not in label_to_idx:
            continue
        idx = label_to_idx.index(tag_name)
        score = float(scores[idx])
        if score > 0.1:
            pairs.append((canon, score))
    pairs.sort(key=lambda p: -p[1])
    return pairs[:TOP_K], {}


def analyze(wav_path: str) -> tuple[List[str], List[EventSegment]]:
    """
    Slide a window across the audio, tag each window, then aggregate.
    Returns (top-K unique tags, per-window segments above threshold).
    """
    import librosa

    y, sr = librosa.load(wav_path, sr=32000, mono=True)  # CNN14 wants 32kHz
    if y.size == 0:
        return [], []

    tagger = _PannsEngine.get()
    win_samples = int(WINDOW_S * sr)
    hop_samples = int(HOP_S * sr)

    segments: List[EventSegment] = []
    tag_scores: dict[str, float] = {}

    for start in range(0, len(y) - win_samples + 1, hop_samples):
        chunk = y[start : start + win_samples]
        if chunk.size < win_samples:
            break
        pairs, _ = _tag_window(chunk, tagger)
        if not pairs:
            continue
        start_ms = int(start / sr * 1000)
        end_ms = int((start + win_samples) / sr * 1000)
        for tag, score in pairs:
            segments.append(EventSegment(start_ms, end_ms, tag, score))
            tag_scores[tag] = max(tag_scores.get(tag, 0.0), score)

    top_tags = sorted(tag_scores.keys(), key=lambda t: -tag_scores[t])[:TOP_K]
    return top_tags, segments


def serialize_segments(segments: List[EventSegment]) -> list:
    return [
        {"start_ms": s.start_ms, "end_ms": s.end_ms, "tag": s.tag, "score": s.score}
        for s in segments
    ]
