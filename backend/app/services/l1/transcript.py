"""
L1 Stage 2: Transcript + two-pass filler detection.

Pipeline:
  1. Faster-Whisper large-v3-turbo (multilingual SOTA in 2026 for CPU).
  2. initial_prompt with fillers so Whisper preserves them instead of cleaning.
  3. Pass 1 (word-list): sliding-window match of known filler stems + phrases.
  4. Pass 2 (audio-domain): scan inter-word gaps > 350ms for voiced energy
     -> catches fillers Whisper still drops. Reference: github.com/dougcalobrisi/erm.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# --- Single-word stems + multi-word filler phrases ---
# Stems match any word starting with these (so "ummmm" matches "um").
FILLER_STEMS = (
    "um", "uh", "er", "ah", "hm", "mm", "mhm", "erm", "uhh", "umm",
)
# Phrases are matched against consecutive normalized word tokens.
FILLER_PHRASES = (
    ("you", "know"),
    ("sort", "of"),
    ("kind", "of"),
    ("i", "mean"),
    ("like",),
)

# Whisper's initial_prompt: presenting fillers in-context teaches it to keep them.
WHISPER_INITIAL_PROMPT = "um, uh, er, ah, like, you know"

# --- Audio-domain gap scan parameters ---
GAP_MIN_MS = 350           # min word-gap to scan for hidden fillers
GAP_VOICED_MIN_MS = 100    # min voiced run inside a gap to count
GAP_VOICED_MAX_MS = 1500   # max voiced run (longer = probably real speech)
GAP_ENERGY_PCTL = 75       # RMS percentile threshold for "voiced"


@dataclass
class Word:
    start_ms: int
    end_ms: int
    text: str
    is_filler: bool = False


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    text: str
    words: List[Word] = field(default_factory=list)


@dataclass
class TranscriptResult:
    language: str
    text: str
    segments: List[Segment]
    fillers: List[Word]


# --- Whisper interface ----------------------------------------------------
# Behind a thin wrapper so swapping to Groq Whisper API later = one file diff.
class _WhisperEngine:
    """Lazy-loaded singleton so model weights load once per worker process."""
    _model = None

    @classmethod
    def get(cls):
        if cls._model is None:
            from faster_whisper import WhisperModel
            from app.services.ml_device import whisper_device

            device, compute_type = whisper_device()
            logger.info("Loading Whisper large-v3-turbo (%s, %s)...", compute_type, device)
            cls._model = WhisperModel(
                "large-v3-turbo",
                device=device,
                compute_type=compute_type,
            )
            logger.info("Whisper loaded.")
        return cls._model


def transcribe(wav_path: str) -> TranscriptResult:
    """Run Whisper -> two-pass filler detection -> structured TranscriptResult."""
    model = _WhisperEngine.get()
    segments_iter, info = model.transcribe(
        wav_path,
        word_timestamps=True,
        initial_prompt=WHISPER_INITIAL_PROMPT,
        vad_filter=False,
    )

    segments: List[Segment] = []
    all_words: List[Word] = []
    full_text_parts: List[str] = []

    for seg in segments_iter:
        words: List[Word] = []
        if seg.words:
            for w in seg.words:
                wm = Word(
                    start_ms=int((w.start or 0) * 1000),
                    end_ms=int((w.end or 0) * 1000),
                    text=w.word.strip() if w.word else "",
                )
                words.append(wm)
                all_words.append(wm)
        s = Segment(
            start_ms=int(seg.start * 1000),
            end_ms=int(seg.end * 1000),
            text=seg.text.strip(),
            words=words,
        )
        segments.append(s)
        full_text_parts.append(s.text)

    # Pass 1: word-list filler match on the transcript itself.
    fillers: List[Word] = []
    _mark_word_list_fillers(all_words, fillers)

    # Pass 2: audio-domain gap scan for fillers Whisper still dropped.
    try:
        fillers.extend(_audio_gap_scan(wav_path, all_words))
    except Exception:
        logger.exception("Audio gap scan failed; continuing with pass-1 fillers only.")

    return TranscriptResult(
        language=info.language or "unknown",
        text=" ".join(full_text_parts).strip(),
        segments=segments,
        fillers=fillers,
    )


# --- Pass 1: word-list match ---------------------------------------------

def _normalize(token: str) -> str:
    return "".join(ch for ch in token.lower() if ch.isalpha())


def _mark_word_list_fillers(words: List[Word], out: List[Word]) -> None:
    norm = [_normalize(w.text) for w in words]
    n = len(words)
    i = 0
    while i < n:
        # Multi-word phrase match (longest first)
        matched_phrase = False
        for phrase in sorted(FILLER_PHRASES, key=len, reverse=True):
            plen = len(phrase)
            if i + plen <= n and tuple(norm[i:i + plen]) == phrase:
                for j in range(i, i + plen):
                    if not words[j].is_filler:
                        words[j].is_filler = True
                        out.append(words[j])
                i += plen
                matched_phrase = True
                break
        if matched_phrase:
            continue
        # Single stem match
        token = norm[i]
        if token and any(token.startswith(stem) for stem in FILLER_STEMS):
            # Reject if it's an elongation of a longer real word
            if len(token) <= 6 or _looks_like_elongation(token):
                if not words[i].is_filler:
                    words[i].is_filler = True
                    out.append(words[i])
        i += 1


def _looks_like_elongation(token: str) -> bool:
    """'ummmmm' / 'uhhhh' = elongation; 'umbrella' should not match."""
    if len(token) < 3:
        return False
    # >=3 of the same character in a row anywhere is a strong elongation signal.
    for i in range(len(token) - 2):
        if token[i] == token[i + 1] == token[i + 2]:
            return True
    return False


# --- Pass 2: audio-domain gap scan ---------------------------------------

def _audio_gap_scan(wav_path: str, words: List[Word]) -> List[Word]:
    """
    For every inter-word gap > GAP_MIN_MS, look for a voiced region inside it
    that is too long to be silence but short enough to be a filler.
    """
    import librosa

    if len(words) < 2:
        return []

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    if y.size == 0:
        return []

    # RMS in ~25ms frames -> per-frame energy
    frame_len = 400   # 25ms at 16kHz
    hop_len = 160     # 10ms hop
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    if rms.size == 0:
        return []
    voiced_threshold = np.percentile(rms, GAP_ENERGY_PCTL)

    found: List[Word] = []
    for prev, nxt in zip(words[:-1], words[1:]):
        gap_start = prev.end_ms
        gap_end = nxt.start_ms
        if gap_end - gap_start < GAP_MIN_MS:
            continue
        # Scan the gap for the longest voiced run.
        start_frame = max(0, int(gap_start / 10))   # 10ms per hop
        end_frame = min(rms.size, int(gap_end / 10))
        if end_frame <= start_frame:
            continue

        run_start: Optional[int] = None
        runs: List[Tuple[int, int]] = []
        for f in range(start_frame, end_frame):
            if rms[f] >= voiced_threshold:
                if run_start is None:
                    run_start = f
            else:
                if run_start is not None:
                    runs.append((run_start, f))
                    run_start = None
        if run_start is not None:
            runs.append((run_start, end_frame))

        for rs, re in runs:
            run_len_ms = (re - rs) * 10
            if GAP_VOICED_MIN_MS <= run_len_ms <= GAP_VOICED_MAX_MS:
                found.append(Word(
                    start_ms=rs * 10,
                    end_ms=re * 10,
                    text="<filler>",
                    is_filler=True,
                ))
    return found


# --- Serialization helpers for storing in Postgres -----------------------

def serialize_segments(segments: Iterable[Segment]) -> list:
    return [
        {
            "start_ms": s.start_ms,
            "end_ms": s.end_ms,
            "text": s.text,
            "words": [
                {
                    "start_ms": w.start_ms,
                    "end_ms": w.end_ms,
                    "text": w.text,
                    "is_filler": w.is_filler,
                }
                for w in s.words
            ],
        }
        for s in segments
    ]


def serialize_fillers(fillers: Iterable[Word]) -> list:
    return [
        {"start_ms": w.start_ms, "end_ms": w.end_ms, "word": w.text}
        for w in fillers
    ]
