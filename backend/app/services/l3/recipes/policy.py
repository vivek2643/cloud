"""
Declarative style policies + one general assembler.

Every editing style is now a DATA description (a StylePolicy), not a bespoke
class. A single PolicyRecipe interprets the policy against the shared
spine/coverage + placement primitives. This kills the "monolithic style-specific
recipe" problem: a new look is a new row in STYLE_POLICIES, and the general
spine+coverage substrate is just the policy whose audio_mode is "spine_coverage".

A StylePolicy captures the few real degrees of freedom the old recipes encoded:
  - which units form the material (modality)
  - how they're ordered / selected (order + energy_weight + caps)
  - how audio is laid (coupled / music bed / beat grid / spine+coverage)
Anything the planner sets in section.params still overrides the policy default,
so the LLM keeps full per-section control.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.services.l3.primitives.takes import dedup_speech_units, dedup_visual_units
from app.services.l3.primitives.units import EditUnit
from app.services.l3.recipes.base import (
    MIN_CLIP_MS,
    AVTimeline,
    PlacedClip,
    Recipe,
    RecipeContext,
    SectionPlan,
    place_coupled,
    place_spine_coverage,
    place_split,
    select_for_target,
    snap_unit_bounds,
    trim_to_length,
)
from app.services.l3.recipes.params import RecipeParams

_ROLE_ORDER = {"hook": 0, "setup": 1, "build": 2, "reveal": 3, "payoff": 4, "outro": 5}


@dataclass
class StylePolicy:
    key: str
    label: str
    modality: str = "mixed"          # "speech" | "visual" | "mixed"
    order: str = "chronological"     # chronological|ranked_topn|role|hook_first|beat|spine
    energy_weight: float = 0.0
    seconds_per_clip: float = 2.0    # ranked_topn: N = target / this
    min_keep_n: int = 3
    clip_cap_ms: Optional[int] = None     # per-clip cap (None = keep whole)
    accelerate: bool = False              # ramp clip cap down across the section
    accelerate_lo_ms: Optional[int] = None
    audio_mode: str = "coupled"           # coupled|music_bed|beat_sync|spine_coverage
    music_gain_db: float = -6.0
    target_default_s: Optional[float] = None
    target_cap_s: Optional[float] = None
    vertical: bool = False
    trim_speech: bool = False
    # spine+coverage knobs
    coverage_ratio: float = 0.0
    max_cutaway_ms: int = 3500
    min_hold_ms: int = 1200


# ---------------------------------------------------------------------------
# The 9 presets (the former bespoke recipes, now declarative)
# ---------------------------------------------------------------------------

STYLE_POLICIES: Dict[str, StylePolicy] = {
    "spine_coverage": StylePolicy(
        key="spine_coverage", label="Spine + coverage (general)",
        modality="speech", order="spine", audio_mode="spine_coverage",
        coverage_ratio=0.35,
    ),
    "highlight": StylePolicy(
        key="highlight", label="Highlight / montage",
        modality="visual", order="ranked_topn", energy_weight=0.4,
        seconds_per_clip=1.8, clip_cap_ms=2200, audio_mode="music_bed",
        music_gain_db=-6.0, target_default_s=20,
    ),
    "talking_head": StylePolicy(
        key="talking_head", label="Talking-head cut",
        modality="speech", order="chronological", audio_mode="coupled",
    ),
    "trailer": StylePolicy(
        key="trailer", label="Trailer / teaser",
        modality="mixed", order="role", energy_weight=0.3,
        accelerate=True, clip_cap_ms=2600, accelerate_lo_ms=1200,
        audio_mode="music_bed", music_gain_db=-4.0, target_default_s=25,
    ),
    "beat_sync": StylePolicy(
        key="beat_sync", label="Beat-synced music montage",
        modality="visual", order="beat", audio_mode="beat_sync",
        target_default_s=20,
    ),
    "vlog": StylePolicy(
        key="vlog", label="Vlog / walkthrough",
        modality="speech", order="chronological", clip_cap_ms=6000,
        audio_mode="coupled",
    ),
    "social_short": StylePolicy(
        key="social_short", label="Social short (vertical, hook-first)",
        modality="mixed", order="hook_first", energy_weight=0.3,
        clip_cap_ms=2500, audio_mode="coupled", target_default_s=20,
        target_cap_s=30, vertical=True,
    ),
    "tutorial": StylePolicy(
        key="tutorial", label="Tutorial / explainer",
        modality="speech", order="chronological", audio_mode="coupled",
    ),
    "cinematic_broll": StylePolicy(
        key="cinematic_broll", label="Cinematic b-roll mood",
        modality="visual", order="ranked_topn", energy_weight=-0.2,
        seconds_per_clip=3.5, clip_cap_ms=4000, audio_mode="music_bed",
        music_gain_db=-5.0, target_default_s=25,
    ),
}


# ---------------------------------------------------------------------------
# General assembler
# ---------------------------------------------------------------------------

class PolicyRecipe(Recipe):
    """One Recipe that interprets any StylePolicy."""

    def __init__(self, policy: StylePolicy):
        self.policy = policy
        self.key = policy.key
        self.label = policy.label

    def assemble(self, section: SectionPlan, ctx: RecipeContext) -> AVTimeline:
        return _assemble(self.policy, section, ctx)


def _assemble(policy: StylePolicy, section: SectionPlan, ctx: RecipeContext) -> AVTimeline:
    p = RecipeParams.from_section(section)
    target = _target_ms(section, policy.target_default_s)
    if policy.target_cap_s:
        cap_ms = int(policy.target_cap_s * 1000)
        target = min(target or cap_ms, cap_ms)

    if policy.audio_mode == "beat_sync":
        return _assemble_beat_sync(policy, section, ctx, target, p)

    pool = _pool(policy, section, ctx)

    if policy.audio_mode == "spine_coverage":
        spine = sorted(
            [u for u in pool if u.modality == "speech"],
            key=lambda u: (u.file_id, u.in_ms),
        )
        if not spine:
            return _assemble(STYLE_POLICIES["highlight"], section, ctx)
        if target:
            ranked = sorted(spine, key=lambda u: u.quality, reverse=True)
            keep = {u.id for u in select_for_target(ranked, target)}
            spine = [u for u in spine if u.id in keep]
        tl = place_spine_coverage(
            spine, ctx,
            coverage_ratio=p.coverage_ratio(policy.coverage_ratio),
            max_cutaway_ms=p.max_cutaway_ms(policy.max_cutaway_ms),
            min_hold_ms=p.min_hold_ms(policy.min_hold_ms),
        )
        return _styled(tl, policy.key)

    # vlog-style fallback: a speech style with no speech borrows visuals.
    if not pool and policy.modality == "speech":
        pool = dedup_visual_units(ctx.units_for_section(section, modality="visual"))

    chosen = _order_and_select(policy, pool, target, p)
    if not chosen:
        return _styled(AVTimeline(style=policy.key), policy.key)

    if policy.vertical:
        section.params["vertical"] = True

    src = _build_src(policy, chosen, ctx, p)
    if policy.audio_mode == "music_bed":
        music = ctx.pick_music_file(exclude_file_ids=_file_ids(chosen))
        if music:
            return _styled(
                place_split(src, music, ctx, music_gain_db=p.music_gain_db(policy.music_gain_db)),
                policy.key,
            )
    return _styled(place_coupled(src), policy.key)


def _pool(policy: StylePolicy, section: SectionPlan, ctx: RecipeContext) -> List[EditUnit]:
    if policy.modality == "speech":
        return dedup_speech_units(ctx.units_for_section(section, modality="speech"))
    if policy.modality == "visual":
        return dedup_visual_units(ctx.units_for_section(section, modality="visual"))
    speech = dedup_speech_units(ctx.units_for_section(section, modality="speech"))
    visual = dedup_visual_units(ctx.units_for_section(section, modality="visual"))
    return speech + visual


def _order_and_select(
    policy: StylePolicy, pool: List[EditUnit], target: Optional[int], p: RecipeParams
) -> List[EditUnit]:
    if not pool:
        return []
    ew = p.energy_weight(policy.energy_weight)

    if policy.order == "chronological":
        units = sorted(pool, key=lambda u: (u.file_id, u.in_ms))
        if target:
            ranked = sorted(units, key=lambda u: u.quality, reverse=True)
            keep = {u.id for u in select_for_target(ranked, target)}
            units = [u for u in units if u.id in keep]
        return units

    if policy.order == "ranked_topn":
        ranked = sorted(pool, key=lambda u: (u.quality + ew * u.motion), reverse=True)
        if target:
            per = max(1, int(policy.seconds_per_clip * 1000))
            keep_n = max(policy.min_keep_n, target // per)
        else:
            keep_n = len(ranked)
        return sorted(ranked[:keep_n], key=lambda u: (u.file_id, u.in_ms))

    if policy.order == "role":
        ranked = sorted(pool, key=lambda u: (u.quality + ew * u.motion), reverse=True)
        ordered = sorted(ranked, key=lambda u: _ROLE_ORDER.get((u.narrative_role or "build").lower(), 2))
        return select_for_target(ordered, target)

    if policy.order == "hook_first":
        ranked = sorted(pool, key=lambda u: (u.quality + ew * u.motion), reverse=True)
        chosen = select_for_target(ranked, target)
        if not chosen:
            return []
        return [chosen[0]] + sorted(chosen[1:], key=lambda u: (u.file_id, u.in_ms))

    # default
    return sorted(pool, key=lambda u: (u.file_id, u.in_ms))


def _build_src(policy: StylePolicy, chosen: List[EditUnit], ctx: RecipeContext, p: RecipeParams) -> List[tuple]:
    if policy.accelerate and chosen and policy.clip_cap_ms:
        src: List[tuple] = []
        n = max(1, len(chosen))
        hi = p.max_clip_ms(policy.clip_cap_ms)
        lo = p.max_clip_ms(policy.accelerate_lo_ms or policy.clip_cap_ms)
        for i, u in enumerate(chosen):
            cap = int(hi - (hi - lo) * (i / n))
            src.extend(_units_to_src([u], ctx, max_clip_ms=cap))
        return src
    cap = p.max_clip_ms(policy.clip_cap_ms) if policy.clip_cap_ms else None
    return _units_to_src(chosen, ctx, max_clip_ms=cap, trim_speech=policy.trim_speech)


def _assemble_beat_sync(
    policy: StylePolicy, section: SectionPlan, ctx: RecipeContext,
    target: Optional[int], p: RecipeParams,
) -> AVTimeline:
    music = ctx.pick_music_file()
    beats = ctx.beats(music) if music else []
    downbeats = [b.ms for b in beats if b.kind == "downbeat"] or [b.ms for b in beats]
    visuals = dedup_visual_units(ctx.units_for_section(section, modality="visual"))
    if not music or len(downbeats) < 2 or not visuals:
        return _assemble(STYLE_POLICIES["highlight"], section, ctx)

    ranked = sorted(visuals, key=lambda u: (u.quality + 0.4 * u.motion), reverse=True)
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
    return _styled(AVTimeline(video=video, audio=audio, duration_ms=total), policy.key)


# ---------------------------------------------------------------------------
# Shared small helpers (ported from the old styles module)
# ---------------------------------------------------------------------------

def _target_ms(section: SectionPlan, default_s: Optional[float] = None) -> Optional[int]:
    s = section.target_duration_s if section.target_duration_s is not None else default_s
    return int(s * 1000) if s else None


def _units_to_src(
    units: List[EditUnit],
    ctx: RecipeContext,
    max_clip_ms: Optional[int] = None,
    trim_speech: bool = False,
) -> List[tuple]:
    """Snap each unit to natural boundaries, optionally cap length, and emit
    (file_id, in, out, shot_id, role, why) tuples. Speech is kept whole unless
    ``trim_speech`` -- chopping a sentence mid-thought is the #1 cause of
    incoherent talking edits."""
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


def _styled(tl: AVTimeline, style: str) -> AVTimeline:
    tl.style = style
    return tl
