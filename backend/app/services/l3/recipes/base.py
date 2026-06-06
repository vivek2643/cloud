"""
Recipe base types + shared assembler helpers.

A Recipe consumes a SectionPlan (symbolic, from the LLM) + a RecipeContext
(deterministic data) and returns an AVTimeline whose clip timeline positions
are RELATIVE to the section start. The composer offsets sections and converts
to a v2 EDL.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.services.l3.primitives.boundaries import (
    Boundary,
    beat_grid,
    motion_boundaries,
    silence_boundaries,
    snap_to_boundary,
    speech_boundaries,
)
from app.services.l3.primitives.loader import FileAnalysis
from app.services.l3.primitives.units import EditUnit
from app.services.l3.router import FootageProfile

MIN_CLIP_MS = 500


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class PlacedClip:
    file_id: str
    source_in_ms: int
    source_out_ms: int
    timeline_in_ms: int      # relative to the SECTION start
    timeline_out_ms: int
    shot_id: Optional[str] = None
    role_in_edit: Optional[str] = None
    why: Optional[str] = None
    gain_db: Optional[float] = None

    @property
    def source_dur_ms(self) -> int:
        return max(0, self.source_out_ms - self.source_in_ms)


@dataclass
class AVTimeline:
    video: List[PlacedClip] = field(default_factory=list)
    audio: List[PlacedClip] = field(default_factory=list)
    duration_ms: int = 0
    style: str = ""

    def is_empty(self) -> bool:
        return not self.video and not self.audio


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

@dataclass
class SectionPlan:
    style: str
    intent: str = ""
    target_duration_s: Optional[float] = None
    file_ids: Optional[List[str]] = None      # footage scope for this section
    unit_ids: List[str] = field(default_factory=list)  # optional explicit LLM picks
    params: Dict[str, object] = field(default_factory=dict)


@dataclass
class RecipeContext:
    analyses: Dict[str, FileAnalysis]
    units: List[EditUnit]
    profile: FootageProfile
    units_by_id: Dict[str, EditUnit] = field(default_factory=dict)
    _bcache: Dict[str, List[Boundary]] = field(default_factory=dict)
    _beatcache: Dict[str, List[Boundary]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.units_by_id:
            self.units_by_id = {u.id: u for u in self.units}

    # -- unit scoping --
    def units_for_section(self, section: SectionPlan, modality: Optional[str] = None) -> List[EditUnit]:
        if section.unit_ids:
            picked = [self.units_by_id[uid] for uid in section.unit_ids if uid in self.units_by_id]
            if picked:
                pool = picked
            else:
                pool = list(self.units)
        else:
            pool = list(self.units)
        if section.file_ids:
            fset = set(section.file_ids)
            pool = [u for u in pool if u.file_id in fset]
        if modality:
            pool = [u for u in pool if u.modality == modality]
        return sorted(pool, key=lambda u: (u.file_id, u.in_ms))

    # -- boundary caches --
    def boundaries(self, file_id: str) -> List[Boundary]:
        if file_id not in self._bcache:
            fa = self.analyses.get(file_id)
            bs: List[Boundary] = []
            if fa:
                bs.extend(speech_boundaries(fa))
                bs.extend(silence_boundaries(fa))
                bs.extend(motion_boundaries(fa.shots))
            self._bcache[file_id] = sorted(bs, key=lambda b: b.ms)
        return self._bcache[file_id]

    def beats(self, file_id: str) -> List[Boundary]:
        if file_id not in self._beatcache:
            fa = self.analyses.get(file_id)
            self._beatcache[file_id] = beat_grid(fa) if fa else []
        return self._beatcache[file_id]

    def pick_music_file(self, exclude_file_ids: Optional[List[str]] = None) -> Optional[str]:
        """
        Choose a dedicated MUSIC file for a split-track bed.

        Only files the router classified as MUSICAL qualify -- a talking-head
        clip that merely has background ambience (is_musical=True) must NOT be
        used as a bed, or its continuous audio would desync the speech under
        reordered video. Files supplying the section's video are excluded so we
        never lay a clip's own audio under its reordered self.
        """
        from app.services.l3.router import MUSICAL  # local import avoids cycle

        exclude = set(exclude_file_ids or [])
        best: Optional[str] = None
        best_dur = -1
        for fid, fa in self.analyses.items():
            if fid in exclude:
                continue
            fp = self.profile.per_file.get(fid)
            if fp and fp.dominant_modality == MUSICAL:
                dur = int((fa.duration_seconds or 0) * 1000)
                if dur > best_dur:
                    best, best_dur = fid, dur
        return best


# ---------------------------------------------------------------------------
# Recipe interface
# ---------------------------------------------------------------------------

class Recipe(abc.ABC):
    key: str = ""
    label: str = ""

    @abc.abstractmethod
    def assemble(self, section: SectionPlan, ctx: RecipeContext) -> AVTimeline:
        ...


# ---------------------------------------------------------------------------
# Shared assembler helpers
# ---------------------------------------------------------------------------

def snap_unit_bounds(unit: EditUnit, ctx: RecipeContext) -> tuple[int, int]:
    """Snap a unit's in/out to the nearest natural boundary in its file."""
    bs = ctx.boundaries(unit.file_id)
    if unit.modality == "speech":
        in_ms = snap_to_boundary(unit.in_ms, bs, max_dist_ms=250,
                                 prefer_kinds=["speech_start", "silence"])
        out_ms = snap_to_boundary(unit.out_ms, bs, max_dist_ms=250,
                                  prefer_kinds=["speech_end", "silence"])
    else:
        in_ms = snap_to_boundary(unit.in_ms, bs, max_dist_ms=200, prefer_kinds=["shot"])
        out_ms = snap_to_boundary(unit.out_ms, bs, max_dist_ms=200, prefer_kinds=["shot"])
    if out_ms - in_ms < MIN_CLIP_MS:
        in_ms, out_ms = unit.in_ms, unit.out_ms
    return in_ms, out_ms


def trim_to_length(in_ms: int, out_ms: int, target_ms: int, ctx: RecipeContext,
                   file_id: str, anchor_ms: Optional[int] = None) -> tuple[int, int]:
    """
    Trim [in,out] down to ~target_ms, keeping the part around ``anchor_ms``
    (default: the head), and snap the new end to a boundary.
    """
    cur = out_ms - in_ms
    if cur <= target_ms or target_ms < MIN_CLIP_MS:
        return in_ms, out_ms
    start = in_ms if anchor_ms is None else max(in_ms, min(anchor_ms - target_ms // 4, out_ms - target_ms))
    new_out = start + target_ms
    bs = ctx.boundaries(file_id)
    new_out = snap_to_boundary(new_out, bs, max_dist_ms=300)
    if new_out - start < MIN_CLIP_MS:
        new_out = start + target_ms
    return start, min(new_out, out_ms)


def select_for_target(units: List[EditUnit], target_ms: Optional[int]) -> List[EditUnit]:
    """
    Greedily accumulate units (in the order given) until total source duration
    reaches target_ms. With no target, keep everything. Assumes ``units`` is
    already ordered the way they should accumulate (chronological or ranked).
    """
    if target_ms is None or target_ms <= 0:
        return list(units)
    out: List[EditUnit] = []
    total = 0
    for u in units:
        if total >= target_ms:
            break
        out.append(u)
        total += u.duration_ms
    return out


def place_coupled(clips_src: List[tuple], start_offset: int = 0) -> AVTimeline:
    """
    Build a coupled A/V timeline (video + audio from the same source per clip),
    placed back-to-back. ``clips_src`` items: (file_id, in_ms, out_ms, shot_id,
    role, why). Returns relative-positioned AVTimeline.
    """
    video: List[PlacedClip] = []
    audio: List[PlacedClip] = []
    cursor = start_offset
    for (file_id, in_ms, out_ms, shot_id, role, why) in clips_src:
        dur = max(0, out_ms - in_ms)
        if dur < MIN_CLIP_MS:
            continue
        vc = PlacedClip(
            file_id=file_id, source_in_ms=in_ms, source_out_ms=out_ms,
            timeline_in_ms=cursor, timeline_out_ms=cursor + dur,
            shot_id=shot_id, role_in_edit=role, why=why,
        )
        video.append(vc)
        audio.append(PlacedClip(
            file_id=file_id, source_in_ms=in_ms, source_out_ms=out_ms,
            timeline_in_ms=cursor, timeline_out_ms=cursor + dur,
            shot_id=shot_id, role_in_edit=role, why=why,
        ))
        cursor += dur
    return AVTimeline(video=video, audio=audio, duration_ms=cursor - start_offset)


def place_split(
    visual_clips_src: List[tuple],
    music_file_id: Optional[str],
    ctx: RecipeContext,
    music_gain_db: float = -6.0,
) -> AVTimeline:
    """
    Build a split A/V timeline: video from visuals, audio from a single music
    bed (if available) spanning the whole section. Falls back to coupled audio
    (the visuals' own audio) when there's no music file.
    """
    video: List[PlacedClip] = []
    cursor = 0
    for (file_id, in_ms, out_ms, shot_id, role, why) in visual_clips_src:
        dur = max(0, out_ms - in_ms)
        if dur < MIN_CLIP_MS:
            continue
        video.append(PlacedClip(
            file_id=file_id, source_in_ms=in_ms, source_out_ms=out_ms,
            timeline_in_ms=cursor, timeline_out_ms=cursor + dur,
            shot_id=shot_id, role_in_edit=role, why=why,
        ))
        cursor += dur
    total = cursor

    audio: List[PlacedClip] = []
    if music_file_id and total >= MIN_CLIP_MS:
        fa = ctx.analyses.get(music_file_id)
        src_dur = int((fa.duration_seconds or 0) * 1000) if fa else 0
        m_out = min(total, src_dur) if src_dur > 0 else total
        if m_out >= MIN_CLIP_MS:
            audio.append(PlacedClip(
                file_id=music_file_id, source_in_ms=0, source_out_ms=m_out,
                timeline_in_ms=0, timeline_out_ms=m_out,
                role_in_edit="music_bed", gain_db=music_gain_db,
            ))
    if not audio:
        # No music: reuse the visuals' own audio (coupled).
        for vc in video:
            audio.append(PlacedClip(
                file_id=vc.file_id, source_in_ms=vc.source_in_ms, source_out_ms=vc.source_out_ms,
                timeline_in_ms=vc.timeline_in_ms, timeline_out_ms=vc.timeline_out_ms,
                shot_id=vc.shot_id,
            ))
    return AVTimeline(video=video, audio=audio, duration_ms=total)
