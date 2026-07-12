"""
Deterministic multicam sync detection (audio_sync.plan.md SS5): ALL-PAIRS
audio cross-correlation -> connected-components partition into same-audio
groups, each with per-member offsets on a shared group clock + confidence.
Pure DSP, no LLM (SS3.4 "deterministic sync").

Sync answers exactly one question -- "which files carry the same real-time
audio, and how are they aligned" -- and nothing else. It never reconstructs
cameras, detects breaks, or reasons about shoot shape: a file's group is
decided purely by which other files it overlaps (`partition_by_overlap`), so
the layer stays generic across any editing scenario. Non-overlapping files
(a post-break continuation, a voiceover, unrelated footage) simply fall into
separate groups or stay ungrouped; ordering/continuity of non-overlapping
material is a downstream editing concern, not sync's.

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
from typing import Dict, List, Tuple

import numpy as np

# Fixed hop for the correlation envelope -- fine enough that the residual
# alignment error after this stage is sub-frame at typical delivery frame
# rates (24-30fps => 33-42ms/frame), coarse enough to keep the correlation
# fast even on long sources.
CORR_HOP_MS = 20
# How far (seconds) two sources are allowed to have started apart and still
# be found. Generous default for hand-clapped/manually-started multicam
# setups; a file-creation-time prior could narrow this later (SS5.2 "optionally").
# NOTE: this only bounds the offset WITHIN a same-audio group (cameras rolling
# together start seconds apart). It deliberately does NOT try to place
# non-overlapping files (a post-break continuation, a later take) on one clock
# -- those aren't the same audio, so they're simply a different group. See
# `partition_by_overlap`.
DEFAULT_MAX_LAG_S = 90.0
# The OVERLAP GATE: below this normalized-correlation peak two sources are
# treated as NOT carrying the same real-time audio, so they belong to DIFFERENT
# groups (a sequential continuation after a break, a separate take, or unrelated
# footage). Audio that genuinely co-occurs peaks well above it; audio that
# doesn't sits far below (empirically ~0.08 for non-overlapping vs ~0.85 for a
# true simultaneous angle). This threshold is the entirety of "which files share
# the same audio" -- there is deliberately no camera/break/take/shoot-shape
# logic anywhere in this module (that would make sync non-generic).
OVERLAP_THRESHOLD = 0.3
# Back-compat alias: the /detect preview still reports a per-member
# `high_confidence` flag computed off this same number (drives the nudge UI).
HIGH_CONFIDENCE_THRESHOLD = OVERLAP_THRESHOLD


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


def all_pairs(
    envelopes: Dict[str, np.ndarray], *, hop_ms: int = CORR_HOP_MS,
    max_lag_s: float = DEFAULT_MAX_LAG_S,
) -> Dict[Tuple[str, str], PairAlignment]:
    """Every UNORDERED pair of files -> its best `PairAlignment`, keyed `(a, b)`
    with `a < b` by id and oriented to the module's one offset convention
    (`a_ms == b_ms + offset_ms`). O(n^2) FFT correlations, each reusing a file's
    single precomputed envelope -- a few seconds even for a dozen files.

    This is the honest way to discover a shoot's structure: no privileged
    reference. Under the old single-longest-file star, a file that overlapped
    the reference poorly (e.g. footage from AFTER a recording break) was forced
    onto the reference's clock at a garbage near-zero offset, silently poisoning
    the whole group. All-pairs lets every file find the files it ACTUALLY shares
    audio with, and lets the ones it doesn't fall cleanly into a separate
    group."""
    fids = sorted(f for f in envelopes if envelopes[f].size > 0)
    out: Dict[Tuple[str, str], PairAlignment] = {}
    for i in range(len(fids)):
        for j in range(i + 1, len(fids)):
            a, b = fids[i], fids[j]
            out[(a, b)] = cross_correlate(envelopes[a], envelopes[b], hop_ms=hop_ms, max_lag_s=max_lag_s)
    return out


def partition_by_overlap(
    envelopes: Dict[str, np.ndarray], *, threshold: float = OVERLAP_THRESHOLD,
    hop_ms: int = CORR_HOP_MS, max_lag_s: float = DEFAULT_MAX_LAG_S,
) -> Tuple[List[Dict[str, PairAlignment]], List[str]]:
    """Discover the same-audio groups among `envelopes` purely by which files
    carry overlapping real-time audio -- sync's ONE job. There is deliberately
    no notion of cameras, breaks, takes, or shoot shape here: a file's group is
    decided solely by which other files it shares audio with.

    Method: all-pairs cross-correlation -> a graph whose edges are the pairs
    peaking at/above `threshold` (genuine real-time overlap) -> connected
    components. Each component of >= 2 files is one group; within it every member
    gets an offset on a shared component clock (`group_ms == member_ms +
    offset_ms`, anchored at the longest-envelope member = offset 0) by summing
    pairwise offsets along the component's strong edges, plus a `confidence` =
    the peak of the edge that placed it (1.0 for the anchor). A file that
    overlaps nobody is a SINGLETON, returned in the second list -- it simply uses
    its own audio (a voiceover, a post-break continuation, an unrelated clip). It
    is never forced into a group, which is exactly what fixes the phantom-angle
    bug (non-overlapping files landing in one group with bogus offsets).

    Returns `(groups, ungrouped_file_ids)`."""
    fids = [f for f in envelopes if envelopes[f].size > 0]
    if len(fids) < 2:
        return [], list(fids)

    pairs = all_pairs({f: envelopes[f] for f in fids}, hop_ms=hop_ms, max_lag_s=max_lag_s)
    strong = {p: pa for p, pa in pairs.items() if pa.confidence >= threshold}

    # Union-find over strong (overlapping) edges -> connected components.
    parent = {f: f for f in fids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in strong:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    comps: Dict[str, List[str]] = {}
    for f in fids:
        comps.setdefault(find(f), []).append(f)

    # Both-direction adjacency over strong edges for the per-component offset
    # walk: `a_ms == b_ms + off` means from a we reach b at +off, from b we
    # reach a at -off.
    adj: Dict[str, List[Tuple[str, int, float]]] = {f: [] for f in fids}
    for (a, b), pa in strong.items():
        adj[a].append((b, pa.offset_ms, pa.confidence))
        adj[b].append((a, -pa.offset_ms, pa.confidence))

    groups: List[Dict[str, PairAlignment]] = []
    ungrouped: List[str] = []
    for members in comps.values():
        if len(members) < 2:
            ungrouped.append(members[0])
            continue
        anchor = max(members, key=lambda f: envelopes[f].size)
        offset_of: Dict[str, int] = {anchor: 0}
        conf_of: Dict[str, float] = {anchor: 1.0}
        seen = {anchor}
        frontier = [anchor]
        while frontier:
            nxt: List[str] = []
            for u in frontier:
                for v, off_uv, conf in adj[u]:
                    if v in seen:
                        continue
                    # group_ms == u_ms + offset_of[u]; u_ms == v_ms + off_uv
                    # => group_ms == v_ms + (off_uv + offset_of[u]).
                    offset_of[v] = offset_of[u] + off_uv
                    conf_of[v] = conf
                    seen.add(v)
                    nxt.append(v)
            frontier = nxt
        groups.append({
            f: PairAlignment(offset_ms=int(offset_of[f]), confidence=float(conf_of.get(f, 0.0)))
            for f in members
        })
    return groups, ungrouped
