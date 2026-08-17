"""
L1 derived signal: per-frame perceptual descriptor (pHash).

Every visual signal we persist today -- ``scene_cuts`` (histogram drift),
``motion_dynamics`` (optical flow), frame-diff -- is a *temporal derivative* over
the original source timeline: it answers "how fast is the picture changing
*here*?". A seam in a finished edit asks a different question entirely: do
segment A's out-frame and segment B's in-frame -- two arbitrary, non-adjacent
source instants -- look like the *same shot*? That needs a per-frame **state**
descriptor, not a change-rate.

This module supplies that missing primitive: a compact, comparable content
fingerprint for each sampled frame. A **64-bit DCT perceptual hash (pHash)** is
the right tool -- cheap to compute (one DCT per sampled frame), cheap to store
(8 bytes / 16 hex chars), and compared by Hamming distance (same shot with minor
motion sits a handful of bits apart; different content is far). Because a pHash
is a state at an instant, ``descriptor(fileA, tA)`` vs ``descriptor(fileB, tB)``
answers "same picture?" for *any* pair, which is exactly what seam continuity
needs. Persisted (not recomputed) because it is also reusable for later
shot-variety perception.

How it works
------------
Mirrors ``scene_cuts.py``'s pattern exactly: ffmpeg decodes the proxy down to
tiny luma frames piped straight into numpy (via ``scene_cuts._decode_bgr_frames``,
reused verbatim, inside ``limits.ffmpeg_slot()``), then the per-frame op -- a
standard DCT pHash instead of a color histogram -- runs in Python (opencv).

Best-effort, CPU-only: any decode/opencv failure returns an empty
(``has_frames=False``) result, never fails L1 -- mirrors
``scene_cuts.compute_scene_cuts``'s failure semantics. The stage's write-guard
then persists *nothing* for such files, so this signal is purely additive and
cannot break existing ingest.
"""
from __future__ import annotations

import bisect
import importlib.util
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.services.l1 import scene_cuts as scene_mod
from app.services.l1.frame_descriptors_params import (
    PHASH_LOWFREQ,
    PHASH_SIDE,
    SAMPLE_FPS,
)

logger = logging.getLogger(__name__)

# Bump when the hash/sampling shape changes so cached rows recompute even if the
# underlying proxy did not (mirrors scene_cuts.SCHEMA_VERSION discipline).
SCHEMA_VERSION = 1

# Default tolerance for "nearest sampled frame to this instant" lookups: just
# over one 500ms hop, so a seam instant always resolves to a sample when one
# exists. A read-side consumer can pass a tighter/looser value.
NEAREST_MS_DEFAULT = 600

# 64-bit mask for Hamming comparisons of two pHashes.
_MASK64 = (1 << 64) - 1


@dataclass
class FrameDescriptors:
    has_frames: bool = False
    hop_ms: int = 0
    # [{"t_ms": int, "h": "<16 hex>"}] in ascending t_ms. 16-hex 64-bit pHash --
    # never a raw image.
    phashes: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "has_frames": self.has_frames,
            "hop_ms": self.hop_ms,
            "phashes": self.phashes,
        }


def _phash64(frame_bgr, *, side: int = PHASH_SIDE, lowfreq: int = PHASH_LOWFREQ) -> int:
    """Standard 64-bit DCT perceptual hash of one BGR frame.

    grayscale -> resize to side x side -> 2D DCT -> keep the top-left
    ``lowfreq x lowfreq`` low-frequency block -> bit = coeff > median(block, DC
    excluded) -> ``lowfreq**2``-bit int (64 for the default 8x8 block).
    """
    import cv2
    import numpy as np

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    if gray.shape[0] != side or gray.shape[1] != side:
        gray = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(gray))
    block = dct[:lowfreq, :lowfreq].flatten()
    # Median over the block EXCLUDING the DC term (block[0]) -- the DC term is
    # overall brightness and would skew the threshold; the standard pHash
    # thresholds the AC coefficients' median.
    med = float(np.median(block[1:]))
    h = 0
    for coeff in block:
        h = (h << 1) | (1 if float(coeff) > med else 0)
    return h & _MASK64


def compute_frame_descriptors(
    video_path: str,
    duration_ms: int,
    *,
    fps: int = SAMPLE_FPS,
    side: int = PHASH_SIDE,
    lowfreq: int = PHASH_LOWFREQ,
) -> FrameDescriptors:
    """Decode the proxy at ``fps`` and pHash each sampled frame.

    ``t_ms = round(frame_index * 1000 / fps)`` gives the source instant of each
    sample. Best-effort: opencv unavailable, a decode failure, or no frames all
    return ``has_frames=False`` and never raise -- mirrors
    ``scene_cuts.compute_scene_cuts``.
    """
    if importlib.util.find_spec("cv2") is None or importlib.util.find_spec("numpy") is None:
        logger.warning("opencv/numpy unavailable; skipping frame descriptors.")
        return FrameDescriptors(has_frames=False)

    hop_ms = int(round(1000 / fps))
    phashes: List[Dict] = []
    try:
        for i, frame in enumerate(scene_mod._decode_bgr_frames(video_path, side, side, fps)):
            h = _phash64(frame, side=side, lowfreq=lowfreq)
            t_ms = int(round(i * 1000 / fps))
            phashes.append({"t_ms": t_ms, "h": format(h, "016x")})
    except Exception:
        logger.exception("Frame descriptor decode/pHash pass failed for %s.", video_path)
        return FrameDescriptors(has_frames=False, hop_ms=hop_ms)

    if not phashes:
        return FrameDescriptors(has_frames=False, hop_ms=hop_ms)

    return FrameDescriptors(has_frames=True, hop_ms=hop_ms, phashes=phashes)


# --- Persistence + read helpers ------------------------------------------
#
# Writer used by the pipeline stage; readers used by the (separate) read-side
# plan to fetch a frame's hash by (file_id, t_ms).

def upsert_frame_descriptors(conn, file_id: str, fd: FrameDescriptors) -> None:
    """Persist one file's descriptors. Idempotent upsert keyed on file_id --
    mirrors the ``scene_cuts`` insert. Callers should write-guard on
    ``fd.has_frames`` first (an empty result writes no row)."""
    conn.execute(
        """
        insert into frame_descriptors (file_id, hop_ms, phashes, schema_version)
        values (%s, %s, %s::jsonb, %s)
        on conflict (file_id) do update set
            hop_ms         = excluded.hop_ms,
            phashes        = excluded.phashes,
            schema_version = excluded.schema_version
        """,
        (file_id, fd.hop_ms, json.dumps(fd.phashes), SCHEMA_VERSION),
    )


def load_descriptors(conn, file_ids) -> Dict[str, List[Tuple[int, int]]]:
    """``{file_id: [(t_ms, phash_int), ...]}`` sorted ascending by ``t_ms``.

    One indexed PK lookup per file (the per-file jsonb array is loaded whole),
    with the hex hashes decoded back to ints for Hamming comparison. Files with
    no row are simply absent from the result."""
    ids = [str(f) for f in file_ids]
    if not ids:
        return {}
    out: Dict[str, List[Tuple[int, int]]] = {}
    cur = conn.execute(
        "select file_id, phashes from frame_descriptors where file_id = any(%s)",
        (ids,),
    )
    for file_id, phashes in cur.fetchall():
        samples: List[Tuple[int, int]] = []
        for entry in (phashes or []):
            try:
                samples.append((int(entry["t_ms"]), int(entry["h"], 16)))
            except (KeyError, TypeError, ValueError):
                continue
        samples.sort(key=lambda p: p[0])
        out[str(file_id)] = samples
    return out


def nearest_phash(
    samples: List[Tuple[int, int]],
    t_ms: int,
    *,
    nearest_ms: int = NEAREST_MS_DEFAULT,
) -> Optional[int]:
    """The pHash of the sampled frame closest to ``t_ms`` in a per-file list
    (as returned by ``load_descriptors``), or ``None`` if the nearest sample is
    farther than ``nearest_ms`` (treated as missing -> read-side fails open)."""
    if not samples:
        return None
    ts = [s[0] for s in samples]
    idx = bisect.bisect_left(ts, t_ms)
    best: Optional[int] = None
    best_d: Optional[int] = None
    for j in (idx - 1, idx):
        if 0 <= j < len(samples):
            d = abs(samples[j][0] - t_ms)
            if best_d is None or d < best_d:
                best_d, best = d, samples[j][1]
    if best_d is None or best_d > nearest_ms:
        return None
    return best


def load_phash_at(
    conn,
    file_id: str,
    t_ms: int,
    *,
    nearest_ms: int = NEAREST_MS_DEFAULT,
) -> Optional[int]:
    """Convenience DB reader: the nearest frame's 64-bit pHash for
    ``(file_id, t_ms)``, or ``None`` when the file has no descriptors or no
    sample within ``nearest_ms``. The read-side plan (seam continuity, mirror)
    consumes this."""
    samples = load_descriptors(conn, [file_id]).get(str(file_id), [])
    return nearest_phash(samples, t_ms, nearest_ms=nearest_ms)


def hamming64(a: int, b: int) -> int:
    """Hamming distance between two 64-bit pHashes (0 = identical picture)."""
    return bin((a ^ b) & _MASK64).count("1")
