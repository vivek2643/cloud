"""
Composer / sequencer: turn a multi-section plan into one v2 (A/V split) EDL.

A single edit can mix styles: the planner emits an ordered list of sections,
each with its own style; the composer runs the right recipe per section and
lays the resulting A/V timelines end-to-end, offsetting each section's clips by
the running cursor. Single-style edits are just the one-section case.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.edl import store as edl_store
from app.services.l3.recipes import RecipeContext, SectionPlan
from app.services.l3.recipes.base import AVTimeline, PlacedClip
from app.services.l3.recipes.registry import get_recipe

logger = logging.getLogger(__name__)


@dataclass
class ComposeResult:
    edl: Dict[str, Any]
    sections: List[Dict[str, Any]] = field(default_factory=list)
    total_ms: int = 0
    vertical: bool = False
    warnings: List[str] = field(default_factory=list)


def compose(
    sections: List[SectionPlan],
    ctx: RecipeContext,
    fps: int = 30,
    resolution: Tuple[int, int] = (1920, 1080),
) -> ComposeResult:
    """Assemble each section with its recipe and stitch into a v2 EDL."""
    video_track: List[Dict[str, Any]] = []
    audio_track: List[Dict[str, Any]] = []
    sections_meta: List[Dict[str, Any]] = []
    warnings: List[str] = []
    cursor = 0
    vertical = False

    for idx, section in enumerate(sections):
        recipe = get_recipe(section.style)
        try:
            tl = recipe.assemble(section, ctx)
        except Exception as e:
            logger.exception("recipe %s failed for section %d", section.style, idx)
            warnings.append(f"Section {idx + 1} ({section.style}) failed: {type(e).__name__}: {e}")
            continue
        if tl.is_empty():
            warnings.append(f"Section {idx + 1} ({section.style}) produced no clips.")
            continue
        if section.params.get("vertical"):
            vertical = True

        sec_dur = _section_duration(tl)
        _append_track(video_track, tl.video, cursor, idx, section.style)
        _append_track(audio_track, tl.audio, cursor, idx, section.style)

        sections_meta.append({
            "index": idx,
            "style": tl.style or section.style,
            "intent": section.intent,
            "start_ms": cursor,
            "end_ms": cursor + sec_dur,
            "duration_ms": sec_dur,
            "video_clips": len(tl.video),
            "audio_clips": len(tl.audio),
        })
        cursor += sec_dur

    if not video_track and not audio_track:
        edl = edl_store.empty_av_edl(fps=fps, resolution=resolution)
        return ComposeResult(edl=edl, sections=sections_meta, total_ms=0,
                             vertical=vertical, warnings=warnings or ["Composer produced an empty timeline."])

    if vertical:
        resolution = (resolution[1], resolution[0]) if resolution[0] > resolution[1] else resolution

    edl = edl_store.build_av_edl(
        video_track=video_track,
        audio_track=audio_track,
        fps=fps,
        resolution=resolution,
        sections=sections_meta,
    )
    return ComposeResult(edl=edl, sections=sections_meta, total_ms=cursor,
                         vertical=vertical, warnings=warnings)


def _section_duration(tl: AVTimeline) -> int:
    v = max((c.timeline_out_ms for c in tl.video), default=0)
    a = max((c.timeline_out_ms for c in tl.audio), default=0)
    return max(v, a, tl.duration_ms)


def _append_track(
    track: List[Dict[str, Any]],
    clips: List[PlacedClip],
    offset: int,
    section_idx: int,
    style: str,
) -> None:
    for c in clips:
        track.append({
            "file_id": c.file_id,
            "shot_id": c.shot_id,
            "source_in_ms": c.source_in_ms,
            "source_out_ms": c.source_out_ms,
            "timeline_in_ms": c.timeline_in_ms + offset,
            "timeline_out_ms": c.timeline_out_ms + offset,
            "role_in_edit": c.role_in_edit,
            "why": c.why,
            "gain_db": c.gain_db,
            "speaker_id": c.speaker_id,
            "section": section_idx,
        })


# ---------------------------------------------------------------------------
# Critique support: a compact, text-only summary of the assembled timeline
# ---------------------------------------------------------------------------

def summarize_for_critique(result: ComposeResult, ctx: RecipeContext) -> str:
    """
    Render a compact textual summary the critic LLM can reason over WITHOUT
    pixels: per-section styles + durations, the spoken-word flow, pacing, and
    obvious flags (very short clips, repeated text, dead air on the video
    track).
    """
    edl = result.edl
    vt = edl.get("video_track") or []
    at = edl.get("audio_track") or []
    lines: List[str] = []
    lines.append(f"TOTAL: {result.total_ms / 1000:.1f}s, {len(vt)} video clips, {len(at)} audio clips")

    for s in result.sections:
        lines.append(
            f"- Section {s['index'] + 1} [{s['style']}] {s['duration_ms'] / 1000:.1f}s "
            f"({s['video_clips']} v / {s['audio_clips']} a) -- {s.get('intent', '')[:80]}"
        )

    # Pacing: clip-length histogram on the video track.
    durs = [int(c["timeline_out_ms"] - c["timeline_in_ms"]) for c in vt]
    if durs:
        avg = sum(durs) / len(durs)
        short = sum(1 for d in durs if d < 800)
        lines.append(f"PACING: avg clip {avg / 1000:.2f}s, shortest {min(durs) / 1000:.2f}s, "
                     f"longest {max(durs) / 1000:.2f}s, {short} clips < 0.8s")

    # Spoken flow: stitch the transcript text of speech-bearing clips in order.
    flow = _spoken_flow(vt, ctx)
    if flow:
        lines.append("SPOKEN FLOW: " + flow[:1200])

    # Redundancy flag: identical/near-identical adjacent shot ids.
    rep = _adjacent_repeats(vt)
    if rep:
        lines.append(f"FLAGS: {rep} adjacent clips reuse the same shot (possible redundancy)")

    return "\n".join(lines)


def _spoken_flow(video_clips: List[Dict[str, Any]], ctx: RecipeContext) -> str:
    parts: List[str] = []
    for c in video_clips:
        fid = c.get("file_id")
        fa = ctx.analyses.get(str(fid)) if fid else None
        if not fa or not fa.transcript:
            continue
        seg_text = _slice_words(fa, int(c["source_in_ms"]), int(c["source_out_ms"]))
        if seg_text:
            parts.append(seg_text)
    return " / ".join(parts).strip()


def _slice_words(fa, start_ms: int, end_ms: int) -> str:
    if not fa.transcript:
        return ""
    words = [w.text for w in fa.transcript.words
             if not w.is_filler and w.start_ms >= start_ms - 200 and w.end_ms <= end_ms + 200]
    return " ".join(words).strip()


def _adjacent_repeats(clips: List[Dict[str, Any]]) -> int:
    """Count adjacent clips that reuse the SAME shot AND overlapping source
    windows -- a true duplicate. (Consecutive sentences from one shot are not
    duplicates, so we don't flag those.)"""
    reps = 0
    prev: Optional[Dict[str, Any]] = None
    for c in clips:
        if prev is not None and c.get("shot_id") and c.get("shot_id") == prev.get("shot_id"):
            a_in, a_out = int(prev["source_in_ms"]), int(prev["source_out_ms"])
            b_in, b_out = int(c["source_in_ms"]), int(c["source_out_ms"])
            overlap = min(a_out, b_out) - max(a_in, b_in)
            if overlap > 0.5 * min(a_out - a_in, b_out - b_in):
                reps += 1
        prev = c
    return reps
