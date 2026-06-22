"""
L1 Stage 6: speaker diarization (who-says-what), via pyannote.audio.

The editor needs a *sense of speaker identity* -- "the person who said X also
said Y" -- so it can keep one speaker on the audio bed while cutting to other
angles, avoid stitching two people into one utterance, and structure
interview/podcast edits. It does NOT need to name people; a stable per-file
label ("S0", "S1", ...) is enough.

Backend: pyannote.audio 3.1 -- a full diarization pipeline (VAD -> neural
segmentation -> embedding -> clustering -> overlap-aware resegmentation). It
runs on GPU via `ml_device` and needs an HF token plus a one-time license
acceptance for the gated models. We run pyannote on the WAV and then attach its
speaker timeline to Whisper's words by maximum temporal overlap, then smooth
tiny one-off label blips.

Best-effort: diarization is a soft signal, not a gate. On any failure -- no
token, pyannote not installed, CPU-only dev, or a runtime error -- `diarize`
returns an empty result and the caller leaves speakers unset, so the pipeline
never hard-fails on it.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MAX_SPEAKERS = 8

# Post-clustering label smoothing. A speaker "run" this short that is flanked by
# the SAME other speaker (or sits at a clip edge next to one speaker) is almost
# always a blip -- e.g. the last word of a sentence flipped to a phantom second
# speaker -- so we fold it back into the surrounding voice.
SMOOTH_MAX_RUN_MS = 700
SMOOTH_MAX_RUN_WORDS = 2


@dataclass
class DiarizationResult:
    # One label per input word (aligned by index); None if undiarizable.
    speaker_by_word: List[Optional[str]] = field(default_factory=list)
    num_speakers: int = 0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def diarize(
    wav_path: str,
    words: Sequence[dict],
    *,
    max_speakers: int = DEFAULT_MAX_SPEAKERS,
    min_speakers: int = 1,
) -> DiarizationResult:
    """Label each word with a per-file speaker id ("S0", "S1", ...).

    `words` is the flat, chronological word list (dicts with start_ms/end_ms).
    `min_speakers` forces at least that many clusters when the caller already
    knows the clip is multi-speaker. Never raises: on any failure it returns an
    empty result so the caller can treat diarization as best-effort.
    """
    word_list = [w for w in words]
    n = len(word_list)
    if n == 0:
        return DiarizationResult()

    try:
        res = _diarize_pyannote(wav_path, word_list, min_speakers, max_speakers)
        if res is not None:
            return res
        logger.warning("pyannote unavailable for %s; leaving speakers unset.", wav_path)
    except Exception:
        logger.exception("Diarization failed for %s; leaving speakers unset.", wav_path)
    return DiarizationResult(speaker_by_word=[None] * n)


# ---------------------------------------------------------------------------
# pyannote.audio 3.1 (VAD + neural segmentation + clustering)
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
            logger.warning("pyannote backend unavailable.", exc_info=True)
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

    # Stable S0.. names by first appearance.
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
    num_speakers = len({v for v in speaker_by_word if v})
    return DiarizationResult(speaker_by_word=speaker_by_word, num_speakers=num_speakers)


# ---------------------------------------------------------------------------
# Label smoothing
# ---------------------------------------------------------------------------

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
            target = prev_spk                      # trailing-edge blip
        if target is not None:
            for k in range(a, b + 1):
                speaker_by_word[k] = target
