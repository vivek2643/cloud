"""
export_options.plan.md Phase 2: SRT sidecar exporter.

Reads the SAME resolved caption events render/tasks.resolve_document already
produces (`captions.resolver.resolve_captions_for_document`) -- program-time
(post-cut, post-J/L), already word/line-grouped by captions/timing.py's
readability-aware grouper -- and formats them as a standards-compliant .srt
file. No model call, no ffmpeg; pure formatting, so the MP4's burned-in
captions and this sidecar can never drift apart (same source, same timeline).
"""
from __future__ import annotations

from typing import Any, Dict, List


def _timecode(ms: int) -> str:
    """SRT timecode: HH:MM:SS,mmm. Negative input clamps to 0 rather than
    emitting a malformed timecode (defensive -- resolved events are program-
    time and should never be negative)."""
    ms = max(0, int(ms))
    hours, rem = divmod(ms, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{millis:03d}"


def _cue_text(event: Dict[str, Any]) -> str:
    """One resolved caption event's `lines` -> SRT cue text, one text line per
    resolved caption line (already readability-grouped upstream), words
    space-joined. Only the word text is used -- colour/animation/emphasis
    live in `event["style"]`/`event["anim"]`/each word's `emphasized` flag,
    which are ASS/render-only concerns SRT has no way to express and must
    not leak into the plain text."""
    out_lines: List[str] = []
    for line in event.get("lines") or []:
        words = line.get("words") or []
        text = " ".join(w.get("text", "") for w in words if w.get("text"))
        if text:
            out_lines.append(text)
    return "\n".join(out_lines)


def build_srt(resolved_captions: List[Dict[str, Any]]) -> str:
    """`resolved.captions` (resolve_captions_for_document's output) -> a
    standards-compliant SRT string. Events are sorted by `prog_start_ms`
    (already the resolver's own order, re-sorted here defensively) and
    numbered sequentially from 1. An event with no actual word text (e.g. a
    style-only placeholder) is skipped -- never an empty cue. A degenerate
    zero/negative-duration event is bumped to a 1ms minimum span so no cue's
    timecodes are equal or inverted."""
    lines: List[str] = []
    index = 0
    for event in sorted(resolved_captions or [], key=lambda e: int(e.get("prog_start_ms") or 0)):
        text = _cue_text(event)
        if not text:
            continue
        start_ms = int(event.get("prog_start_ms") or 0)
        end_ms = int(event.get("prog_end_ms") or 0)
        if end_ms <= start_ms:
            end_ms = start_ms + 1
        index += 1
        lines.append(str(index))
        lines.append(f"{_timecode(start_ms)} --> {_timecode(end_ms)}")
        lines.append(text)
        lines.append("")  # SRT cues are separated by one blank line
    return "\n".join(lines)
