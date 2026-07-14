"""
Cut-level A/V coupling with authoritative audio (av_coupling_authoritative.
plan.md): pure helpers deciding, PER CUT, which file's audio is authoritative
and at what offset -- baked onto the cut at assembly time instead of
re-derived lazily at render/resolve time (the old `sync.audio_route` path).

Mirrors the "fetch once, resolve pure" split `audio_route.py`/`grade.measure`
already use: `authoritative_for` is pure lookup math over an already-fetched
sync-group dict (no DB access here), and `refine_offset` is pure signal math
over already-loaded `rms_db` envelopes.

Two facts anchor a cut's coupling:
  - **Solo clip (no sync group):** authoritative = the clip's own audio ->
    `(file_id, 0)`. Identity case, byte-identical to today for ~90% of footage.
  - **Synced group:** authoritative = the group's declared authoritative
    source -> `(auth_file_id, delta)`, where `delta` starts from the group's
    globally-solved per-file offset and is then LOCALLY REFINED against this
    cut's own audio window (envelope cross-correlation) so a loose global
    offset / long-take clock drift can't show up as visible lip-sync error.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# Below this normalized-correlation peak (-1..1), the local alignment is too
# weak/ambiguous to trust -- keep the unrefined global delta rather than
# chase noise (never refine on a flat/silent/uncorrelated window).
MIN_ALIGN_CONFIDENCE = 0.3
# Cross-correlation search half-width in ms (the plan's own starting point,
# tunable) -- wide enough to absorb realistic per-cut clock drift on a long
# take without searching so far it could lock onto the wrong syllable.
DEFAULT_SEARCH_MS = 300


def authoritative_for(
    file_id: str, sync_info: Dict[str, Dict[str, Any]],
) -> Tuple[str, int]:
    """(audio_file_id, global_delta_ms) for one file, from an already-fetched
    sync-group dict (`sync.store.sync_groups_for_files`'s shape: `{file_id:
    {"authoritative_audio_file_id", "members": {file_id: {"offset_ms", ...}}}}`).

    `(file_id, 0)` (identity coupling) when: the file is in no group, the
    group declares no authoritative source, the file itself IS the
    authoritative source, or the group data is malformed/incomplete (never
    guess a delta from partial data). Otherwise the SAME signed-delta math
    `sync.audio_route.resolve_audio_routes` already used: `group_ms =
    file_ms + file_offset = auth_ms + auth_offset`, so `auth_ms = file_ms +
    file_offset - auth_offset`."""
    grp = sync_info.get(file_id)
    if not grp:
        return file_id, 0
    auth_fid = grp.get("authoritative_audio_file_id")
    if not auth_fid or auth_fid == file_id:
        return file_id, 0
    members = grp.get("members") or {}
    if file_id not in members or auth_fid not in members:
        return file_id, 0
    delta_ms = int(members[file_id]["offset_ms"]) - int(members[auth_fid]["offset_ms"])
    return auth_fid, delta_ms


def _slice(rms: List[float], hop_ms: int, s: int, e: int) -> List[float]:
    """The rms_db samples covering [s, e) -- [] for degenerate/out-of-range
    input, never an index error."""
    if not rms or hop_ms <= 0 or e <= s:
        return []
    lo, hi = max(0, s // hop_ms), min(len(rms) - 1, (e - 1) // hop_ms)
    if hi < lo:
        return []
    return rms[lo:hi + 1]


def _normalized_correlation(a: List[float], b: List[float]) -> float:
    """Pearson correlation coefficient, -1..1. 0.0 for degenerate input
    (empty, mismatched length, zero-variance) -- never divides by zero."""
    if not a or not b or len(a) != len(b):
        return 0.0
    n = len(a)
    mean_a, mean_b = sum(a) / n, sum(b) / n
    cov = sum((a[i] - mean_a) * (b[i] - mean_b) for i in range(n))
    var_a = sum((x - mean_a) ** 2 for x in a)
    var_b = sum((x - mean_b) ** 2 for x in b)
    denom = (var_a * var_b) ** 0.5
    return cov / denom if denom > 0 else 0.0


def refine_offset(
    video_rms: List[float], auth_rms: List[float], hop_ms: int, s: int, e: int,
    global_delta: int, *, search_ms: int = DEFAULT_SEARCH_MS,
) -> Tuple[int, Optional[float]]:
    """(audio_offset_ms, confidence) for one cut. Cross-correlates the video
    file's OWN loudness envelope over [s, e) against the authoritative
    file's envelope around the globally-shifted window [s+delta, e+delta),
    searching +/- `search_ms` (in `hop_ms` steps) for the residual lag that
    maximizes normalized correlation -- `audio_offset_ms = global_delta +
    residual_lag_ms`.

    Guard: no usable video envelope, or the best peak's correlation is below
    `MIN_ALIGN_CONFIDENCE`, returns the UNREFINED `global_delta` with
    `confidence=None` -- never refines on a flat/silent/ambiguous window."""
    video_slice = _slice(video_rms, hop_ms, s, e)
    if not video_slice or hop_ms <= 0:
        return global_delta, None
    n = len(video_slice)
    search_hops = max(1, search_ms // hop_ms)

    best_r_hops = 0
    best_corr = -2.0   # below any real correlation (-1..1) -- guarantees a first assignment
    for r_hops in range(-search_hops, search_hops + 1):
        shift_ms = r_hops * hop_ms
        auth_slice = _slice(auth_rms, hop_ms, s + global_delta + shift_ms, e + global_delta + shift_ms)
        if len(auth_slice) != n:
            continue
        corr = _normalized_correlation(video_slice, auth_slice)
        if corr > best_corr:
            best_corr, best_r_hops = corr, r_hops

    if best_corr < MIN_ALIGN_CONFIDENCE:
        return global_delta, None
    return global_delta + best_r_hops * hop_ms, round(best_corr, 3)
