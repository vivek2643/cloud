"""
The 8 concrete recipes. Each is a standard editing "recipe" built on the
shared primitives + assembler helpers, so they stay small and consistent.
"""
from __future__ import annotations

from typing import List, Optional

from app.services.l3.primitives.takes import dedup_speech_units, dedup_visual_units
from app.services.l3.primitives.units import EditUnit
from app.services.l3.recipes.base import (
    AVTimeline,
    PlacedClip,
    Recipe,
    RecipeContext,
    SectionPlan,
    MIN_CLIP_MS,
    place_coupled,
    place_split,
    select_for_target,
    snap_unit_bounds,
    trim_to_length,
)
from app.services.l3.recipes.params import RecipeParams


def _target_ms(section: SectionPlan, default_s: Optional[float] = None) -> Optional[int]:
    s = section.target_duration_s if section.target_duration_s is not None else default_s
    return int(s * 1000) if s else None


def _units_to_src(
    units: List[EditUnit],
    ctx: RecipeContext,
    max_clip_ms: Optional[int] = None,
    trim_speech: bool = False,
) -> List[tuple]:
    """
    Snap each unit to natural boundaries, optionally trim to a max length, and
    produce (file_id, in, out, shot_id, role, why) tuples for placement.

    SPEECH units are kept WHOLE by default: chopping a sentence to a fixed
    montage length cuts mid-thought (the #1 cause of incoherent talking edits).
    Only visual units get the length cap unless ``trim_speech`` is set.
    """
    out: List[tuple] = []
    for u in units:
        in_ms, out_ms = snap_unit_bounds(u, ctx)
        cap = max_clip_ms if (u.modality != "speech" or trim_speech) else None
        if cap:
            in_ms, out_ms = trim_to_length(in_ms, out_ms, cap, ctx, u.file_id, anchor_ms=in_ms)
        if out_ms - in_ms < MIN_CLIP_MS:
            continue
        out.append((u.file_id, in_ms, out_ms, u.shot_id, u.narrative_role, (u.text or "")[:140]))
    return out


def _file_ids(units: List[EditUnit]) -> List[str]:
    return list({u.file_id for u in units})


# ---------------------------------------------------------------------------

class HighlightMontage(Recipe):
    key = "highlight"
    label = "Highlight / montage"

    def assemble(self, section: SectionPlan, ctx: RecipeContext) -> AVTimeline:
        p = RecipeParams.from_section(section)
        units = ctx.units_for_section(section, modality="visual")
        units = dedup_visual_units(units)
        # Rank by combined quality + energy, keep the strongest, then restore
        # chronological order so the montage still reads coherently.
        ew = p.energy_weight(0.4)
        ranked = sorted(units, key=lambda u: (u.quality + ew * u.motion), reverse=True)
        target = _target_ms(section, default_s=20)
        # Pick enough strong units (assume ~2s each) to fill target.
        keep_n = max(3, (target // 1800)) if target else len(ranked)
        chosen = sorted(ranked[:keep_n], key=lambda u: (u.file_id, u.in_ms))
        src = _units_to_src(chosen, ctx, max_clip_ms=p.max_clip_ms(2200))
        music = ctx.pick_music_file(exclude_file_ids=_file_ids(chosen))
        if music:
            return _styled(place_split(src, music, ctx, music_gain_db=p.music_gain_db(-6.0)), self.key)
        return _styled(place_coupled(src), self.key)


class TalkingHead(Recipe):
    key = "talking_head"
    label = "Talking-head cut"

    def assemble(self, section: SectionPlan, ctx: RecipeContext) -> AVTimeline:
        units = ctx.units_for_section(section, modality="speech")
        units = dedup_speech_units(units)
        # Chronological; drop the weakest if a target forces it.
        units = sorted(units, key=lambda u: (u.file_id, u.in_ms))
        target = _target_ms(section)
        if target:
            ranked = sorted(units, key=lambda u: u.quality, reverse=True)
            chosen_ids = {u.id for u in select_for_target(ranked, target)}
            units = [u for u in units if u.id in chosen_ids]
        src = _units_to_src(units, ctx)  # no hard trim -- keep sentences whole
        return _styled(place_coupled(src), self.key)


class Trailer(Recipe):
    key = "trailer"
    label = "Trailer / teaser"

    _ROLE_ORDER = {"hook": 0, "setup": 1, "build": 2, "reveal": 3, "payoff": 4, "outro": 5}

    def assemble(self, section: SectionPlan, ctx: RecipeContext) -> AVTimeline:
        p = RecipeParams.from_section(section)
        units = ctx.units_for_section(section)
        units = dedup_speech_units(units) + dedup_visual_units(units)
        target = _target_ms(section, default_s=25)
        # Hook = highest energy/valence unit; then rising build; payoff last.
        def role_key(u: EditUnit):
            return self._ROLE_ORDER.get((u.narrative_role or "build").lower(), 2)
        ew = p.energy_weight(0.3)
        ranked = sorted(units, key=lambda u: (u.quality + ew * u.motion), reverse=True)
        chosen = select_for_target(sorted(ranked, key=role_key), target)
        # Accelerate: earlier clips longer, later clips snappier (pace-scaled).
        src: List[tuple] = []
        n = max(1, len(chosen))
        hi, lo = p.max_clip_ms(2600), p.max_clip_ms(1200)
        for i, u in enumerate(chosen):
            clip_cap = int(hi - (hi - lo) * (i / n))
            for s in _units_to_src([u], ctx, max_clip_ms=clip_cap):
                src.append(s)
        music = ctx.pick_music_file(exclude_file_ids=_file_ids(chosen))
        if music:
            return _styled(place_split(src, music, ctx, music_gain_db=p.music_gain_db(-4.0)), self.key)
        return _styled(place_coupled(src), self.key)


class BeatSyncMusicMontage(Recipe):
    key = "beat_sync"
    label = "Beat-synced music montage"

    def assemble(self, section: SectionPlan, ctx: RecipeContext) -> AVTimeline:
        music = ctx.pick_music_file()
        beats = ctx.beats(music) if music else []
        downbeats = [b.ms for b in beats if b.kind == "downbeat"] or [b.ms for b in beats]
        visuals = dedup_visual_units(ctx.units_for_section(section, modality="visual"))
        if not music or len(downbeats) < 2 or not visuals:
            # Degrade gracefully to a highlight montage.
            return HighlightMontage().assemble(section, ctx)

        ranked = sorted(visuals, key=lambda u: (u.quality + 0.4 * u.motion), reverse=True)
        target = _target_ms(section, default_s=20)

        video: List[PlacedClip] = []
        cursor = 0
        vi = 0
        for i in range(len(downbeats) - 1):
            if target and cursor >= target:
                break
            interval = downbeats[i + 1] - downbeats[i]
            if interval < MIN_CLIP_MS:
                continue
            u = ranked[vi % len(ranked)]
            vi += 1
            in_ms, out_ms = snap_unit_bounds(u, ctx)
            in_ms2, out_ms2 = trim_to_length(in_ms, out_ms, interval, ctx, u.file_id, anchor_ms=in_ms)
            dur = min(interval, out_ms2 - in_ms2)
            if dur < MIN_CLIP_MS:
                continue
            video.append(PlacedClip(
                file_id=u.file_id, source_in_ms=in_ms2, source_out_ms=in_ms2 + dur,
                timeline_in_ms=cursor, timeline_out_ms=cursor + dur,
                shot_id=u.shot_id, role_in_edit="beat", why=(u.text or "")[:120],
            ))
            cursor += dur

        total = cursor
        audio: List[PlacedClip] = []
        if total >= MIN_CLIP_MS:
            audio.append(PlacedClip(
                file_id=music, source_in_ms=downbeats[0], source_out_ms=downbeats[0] + total,
                timeline_in_ms=0, timeline_out_ms=total, role_in_edit="music_bed", gain_db=-3.0,
            ))
        return _styled(AVTimeline(video=video, audio=audio, duration_ms=total), self.key)


class VlogWalkthrough(Recipe):
    key = "vlog"
    label = "Vlog / walkthrough"

    def assemble(self, section: SectionPlan, ctx: RecipeContext) -> AVTimeline:
        speech = dedup_speech_units(ctx.units_for_section(section, modality="speech"))
        visual = dedup_visual_units(ctx.units_for_section(section, modality="visual"))
        # Chronological narrative spine from speech; sprinkle the best b-roll.
        spine = sorted(speech, key=lambda u: (u.file_id, u.in_ms))
        target = _target_ms(section)
        if target:
            ranked = sorted(spine, key=lambda u: u.quality, reverse=True)
            keep = {u.id for u in select_for_target(ranked, target)}
            spine = [u for u in spine if u.id in keep]
        units = spine if spine else sorted(visual, key=lambda u: (u.file_id, u.in_ms))
        p = RecipeParams.from_section(section)
        src = _units_to_src(units, ctx, max_clip_ms=p.max_clip_ms(6000))
        return _styled(place_coupled(src), self.key)


class SocialShort(Recipe):
    key = "social_short"
    label = "Social short (vertical, hook-first)"

    def assemble(self, section: SectionPlan, ctx: RecipeContext) -> AVTimeline:
        p = RecipeParams.from_section(section)
        units = ctx.units_for_section(section)
        units = dedup_speech_units(units) + dedup_visual_units(units)
        target = _target_ms(section, default_s=20)
        # Hard cap at 30s for a social short.
        target = min(target or 20000, 30000)
        ew = p.energy_weight(0.3)
        ranked = sorted(units, key=lambda u: (u.quality + ew * u.motion), reverse=True)
        chosen = select_for_target(ranked, target)
        if not chosen:
            return _styled(AVTimeline(style=self.key), self.key)
        # Hook-first: strongest clip up front, the rest chronological.
        hook = chosen[0]
        rest = sorted(chosen[1:], key=lambda u: (u.file_id, u.in_ms))
        ordered = [hook] + rest
        src = _units_to_src(ordered, ctx, max_clip_ms=p.max_clip_ms(2500))
        tl = place_coupled(src)
        tl.style = self.key
        # Flag vertical so the orchestrator picks a vertical render preset.
        section.params["vertical"] = True
        return tl


class TutorialExplainer(Recipe):
    key = "tutorial"
    label = "Tutorial / explainer"

    def assemble(self, section: SectionPlan, ctx: RecipeContext) -> AVTimeline:
        # Step-structured: keep speech whole + in order, preserve demos (long
        # static talking) rather than trimming them aggressively.
        speech = dedup_speech_units(ctx.units_for_section(section, modality="speech"))
        units = sorted(speech, key=lambda u: (u.file_id, u.in_ms))
        target = _target_ms(section)
        if target:
            units = select_for_target(units, target)
        src = _units_to_src(units, ctx)  # no max trim: preserve full steps
        return _styled(place_coupled(src), self.key)


class CinematicBRoll(Recipe):
    key = "cinematic_broll"
    label = "Cinematic b-roll mood"

    def assemble(self, section: SectionPlan, ctx: RecipeContext) -> AVTimeline:
        p = RecipeParams.from_section(section)
        visual = dedup_visual_units(ctx.units_for_section(section, modality="visual"))
        # Prefer stable, scenic shots (lower motion, high visual quality);
        # longer holds + a music bed for mood. energy_weight defaults negative.
        ew = p.energy_weight(-0.2)
        ranked = sorted(visual, key=lambda u: (u.quality + ew * u.motion), reverse=True)
        target = _target_ms(section, default_s=25)
        keep_n = max(3, (target // 3500)) if target else len(ranked)
        chosen = sorted(ranked[:keep_n], key=lambda u: (u.file_id, u.in_ms))
        src = _units_to_src(chosen, ctx, max_clip_ms=p.max_clip_ms(4000))
        music = ctx.pick_music_file(exclude_file_ids=_file_ids(chosen))
        if music:
            return _styled(place_split(src, music, ctx, music_gain_db=p.music_gain_db(-5.0)), self.key)
        return _styled(place_coupled(src), self.key)


def _styled(tl: AVTimeline, style: str) -> AVTimeline:
    tl.style = style
    return tl
