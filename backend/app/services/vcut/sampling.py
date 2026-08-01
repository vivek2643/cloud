"""
seam_cut_pipeline.plan.md section 6 -- "medium" frame sampling over
non-speech spans only. No density knob exists anywhere in the old l3
pipeline to reuse (image_plan.py's frame budgeting is atom-aware and not
applicable here) -- SAMPLE_INTERVAL_MS/MAX_FRAMES_PER_FILE (params.py) are
net-new, vcut-only constants. Frame EXTRACTION itself reuses
app.services.l3.frames (section 6's own explicit sanction: "same helper the
old pass2 uses") -- a generic ffmpeg/R2 primitive, not L3 business logic.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from app.services.vcut.params import MAX_FRAMES_PER_FILE, SAMPLE_INTERVAL_MS, SAMPLE_WIDTH_PX


def sample_timestamps(
    spans: List[Tuple[int, int]],
    interval_ms: int = SAMPLE_INTERVAL_MS,
    max_frames: int = MAX_FRAMES_PER_FILE,
) -> List[int]:
    """One timestamp every ``interval_ms`` inside each non-speech span
    (each span's own end point included so a short trailing moment is never
    unsampled), evenly downsampled to ``max_frames`` if that's exceeded."""
    ts: List[int] = []
    for s, e in spans:
        if e <= s:
            continue
        t = s
        while t < e:
            ts.append(t)
            t += interval_ms
        if ts[-1] != e:
            ts.append(e)
    ts = sorted(set(ts))
    if len(ts) <= max_frames or max_frames <= 0:
        return ts
    step = len(ts) / max_frames
    return sorted({ts[min(len(ts) - 1, int(i * step))] for i in range(max_frames)})


def sample_frames_for_files(
    non_speech_by_file: Dict[str, List[Tuple[int, int]]],
    proxy_key_by_file: Dict[str, str],
    cache=None,
) -> Dict[Tuple[str, int], str]:
    """{(file_id, ts_ms): base64 jpeg} for every file's sampled non-speech
    timestamps -- thin glue over app.services.l3.frames.extract_for_planned_
    frames (reused per section 6), building the PlannedFrame list it needs.
    ``reason``/``ref``/``phase`` are inert for extraction (only meaningful to
    the old pipeline's own prompt-building, which vcut doesn't use)."""
    from app.services.l3.frames import extract_for_planned_frames
    from app.services.l3.image_plan import PlannedFrame

    planned = []
    for file_id, spans in non_speech_by_file.items():
        for ts_ms in sample_timestamps(spans):
            planned.append(PlannedFrame(file_id=file_id, ts_ms=ts_ms, reason="vcut_sample",
                                        ref=f"vcut[{file_id}][{ts_ms}]"))
    if not planned:
        return {}
    return extract_for_planned_frames(planned, proxy_key_by_file, width=SAMPLE_WIDTH_PX, cache=cache)
