"""
Deterministic multicam sync detection (audio_sync.plan.md SS5): pairwise
audio cross-correlation -> per-member offset on a shared group clock +
confidence. Pure DSP, no LLM (SS3.4 "deterministic sync").

Envelope choice (SS4.2/SS14.1, "decide during build"): NOT a reuse of
`audio_features.rms_db`. That envelope's hop scales with file duration up to
`PROSODY_MAX_POINTS=600` points (`audio_features.py:135`) -- on a typical
10-30 minute multicam source that's a 1000-3000ms hop, which floors the best
achievable alignment at half that (500-1500ms). That's not "sync," it's
"vaguely nearby," and would make the feature fail its own stated goal ("the
audio never jumps"). Sync is a one-off INTERACTIVE action (SS1 "user
selects the files, hits Sync"), not a batch L1 signal, so this module
computes its own fixed-hop (20ms, roughly one video frame) envelope on
demand from each file's already-demuxed WAV -- cheap (a few seconds per
file), not persisted, and accurate enough for practical dual-system sync.
See this module's `envelope()`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

# Fixed hop for the correlation envelope -- fine enough that the residual
# alignment error after this stage is sub-frame at typical delivery frame
# rates (24-30fps => 33-42ms/frame), coarse enough to keep the correlation
# fast even on long sources.
CORR_HOP_MS = 20
# How far (seconds) two sources are allowed to have started apart and still
# be found. Generous default for hand-clapped/manually-started multicam
# setups; a file-creation-time prior could narrow this later (SS5.2 "optionally").
DEFAULT_MAX_LAG_S = 90.0
# Below this normalized-correlation peak, don't trust the auto-align (SS5.4
# confidence gate) -- surface the manual nudge UI instead of silently committing.
HIGH_CONFIDENCE_THRESHOLD = 0.3


def envelope(wav_path: str, hop_ms: int = CORR_HOP_MS) -> np.ndarray:
    """A fixed-hop, z-normalized short-time energy envelope for
    cross-correlation. Z-normalizing (zero mean, unit variance) means the
    correlation isn't dominated by absolute loudness differences between
    sources (mic gain/distance varies a lot across an external mic vs. a
    camera's on-board mic)."""
    import librosa

    y, sr = librosa.load(wav_path, sr=16000, mono=True)
    hop = max(1, int(sr * hop_ms / 1000))
    frame = hop * 2
    if len(y) < frame:
        return np.zeros(0, dtype=np.float64)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    db = (20.0 * np.log10(rms + 1e-6)).astype(np.float64)
    std = float(np.std(db))
    return (db - float(np.mean(db))) / std if std > 1e-6 else db - float(np.mean(db))


@dataclass
class PairAlignment:
    offset_ms: int   # see cross_correlate: group_ms = a_ms + 0 == b_ms + offset_ms
    confidence: float  # ~0..1, normalized correlation peak height


def cross_correlate(
    env_a: np.ndarray, env_b: np.ndarray, hop_ms: int = CORR_HOP_MS,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
) -> PairAlignment:
    """The offset that aligns `b` onto `a`'s clock: `a_ms == b_ms +
    offset_ms` for the same real-world instant (i.e. `offset_ms` is how far
    AHEAD `a`'s clock is of `b`'s -- b started `offset_ms` after a if
    positive). A sharp, high peak means the two overlapped in real time
    (same moment, e.g. multicam angles); a diffuse/low peak means they
    didn't (unrelated files, or a sequential retake that must NOT be
    grouped -- SS5's "waveform analogue of the semantic take-group logic")."""
    from scipy.signal import correlate as _correlate

    if env_a.size == 0 or env_b.size == 0:
        return PairAlignment(offset_ms=0, confidence=0.0)

    max_lag_samples = max(1, int(max_lag_s * 1000 / hop_ms))
    c = _correlate(env_a, env_b, mode="full", method="fft")
    # `c[k]` corresponds to lag = k - (len(env_b) - 1) samples: a[n] aligns
    # with b[n - lag]. Restrict the search to +/- max_lag_samples so an
    # unrelated pair can't "win" on a spurious far-away peak.
    zero_lag_idx = env_b.size - 1
    lo = max(0, zero_lag_idx - max_lag_samples)
    hi = min(c.size, zero_lag_idx + max_lag_samples + 1)
    window = c[lo:hi]
    if window.size == 0:
        return PairAlignment(offset_ms=0, confidence=0.0)
    peak_idx = lo + int(np.argmax(window))
    lag_samples = peak_idx - zero_lag_idx

    # Normalize by the actual overlap length at this lag (not the full
    # signal length) so partial-overlap pairs aren't unfairly penalized --
    # for z-normalized inputs this approximates the Pearson correlation over
    # the overlapping region, ~1.0 for a strong true match.
    overlap = min(env_a.size, env_b.size) - abs(lag_samples)
    confidence = float(c[peak_idx] / overlap) if overlap > 0 else 0.0
    confidence = max(0.0, min(1.0, confidence))

    return PairAlignment(offset_ms=lag_samples * hop_ms, confidence=confidence)


def solve_group_offsets(
    envelopes: Dict[str, np.ndarray], *, hop_ms: int = CORR_HOP_MS,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
) -> Dict[str, PairAlignment]:
    """Anchor every member to the EARLIEST-starting one (SS5.3, SS14.5's
    "group clock reference" call: earliest-start, not the authoritative
    source -- authoritative-audio pick is a separate, later decision, SS5.5,
    that shouldn't have to happen before the clock even exists). Every other
    member's offset is found by direct pairwise correlation against that
    reference; the reference itself gets `offset_ms=0, confidence=1.0`.

    `envelopes` must have >= 1 entry; a single-entry group is a degenerate
    no-op group (SS2 "single file -> sync is a no-op")."""
    if not envelopes:
        return {}
    file_ids = list(envelopes.keys())
    if len(file_ids) == 1:
        return {file_ids[0]: PairAlignment(offset_ms=0, confidence=1.0)}

    # Reference = whichever file's envelope covers the longest span (a cheap,
    # deterministic proxy for "likely captured the earliest start" without
    # needing file-creation-time metadata; any member works as a reference
    # mathematically, this just tends to maximize pairwise overlap).
    reference_id = max(file_ids, key=lambda fid: envelopes[fid].size)
    out: Dict[str, PairAlignment] = {reference_id: PairAlignment(offset_ms=0, confidence=1.0)}
    ref_env = envelopes[reference_id]
    for fid in file_ids:
        if fid == reference_id:
            continue
        # cross_correlate(a=reference, b=this file) -> offset_ms such that
        # reference_ms == this_file_ms + offset_ms, i.e. exactly this
        # member's group-clock position per the table's own definition.
        out[fid] = cross_correlate(ref_env, envelopes[fid], hop_ms=hop_ms, max_lag_s=max_lag_s)
    return out
