"""
Tier 1 "Suggested" generation (captions.plan.md SS6): 5 caption bundles
generated FOR one edit's actual perceived signals -- energy, speaker count,
palette, aspect -- each a full `CaptionStyle` (`tier="suggested"`) plus a
rationale string. Deterministic (no LLM, no randomness) so a suggestion set
is reproducible and cheap, same "playbook" as everything else in this
feature; "Reshuffle" (SS6.5) is a seeded pick among a small alternates pool
per slot, not true randomness.

Every suggestion still flows through the SAME `resolver.resolve_captions`
pipeline as a hand-picked Standards style, so SS6.3's guarantees (safe-zone
placement, contrast, readability pacing) are enforced structurally by
`placement.py`/`colour.py`/`timing.py` -- nothing here has to re-implement
them; a "bold" suggestion can pick loud fonts/animation, never an unsafe box
or an illegible colour.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from app.services.l3.captions.colour import vibrant_accent
from app.services.l3.captions.styles import AnimationSpec, CaptionStyle, ColourSpec, PlacementSpec

# Process-local cache (SS6.5 "cache per edit version"): mirrors
# `grade.cache`'s "local disk cache, not shared across instances, fine
# since it's cheap to regenerate" contract, simpler still since there's
# nothing to bake -- just don't recompute the same signature twice in one
# process's lifetime.
_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_CACHE_MAX = 256


def edit_signature(resolved_timeline: Dict[str, Any], captions_selection: Optional[Dict[str, Any]] = None) -> str:
    """A content signature that changes exactly when the edit "materially
    changes" (SS6.5): the ordered spine layer identity, same shape the
    frontend's own reset-detection (`composite-preview.tsx`'s `shapeKey`)
    already uses for "did the plan change shape". Independent of the
    captions selection itself (a style pick shouldn't bust the suggestion
    cache -- only a genuine timeline edit should)."""
    spine = [
        v for v in (resolved_timeline.get("video_layers") or []) if v.get("kind") == "spine"
    ]
    key = "|".join(
        f"{v['source_file_id']}:{v['src_in_ms']}-{v['src_out_ms']}:{v['prog_start_ms']}"
        for v in sorted(spine, key=lambda v: v["prog_start_ms"])
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def _dominant_energy(cut_rows_all: Sequence[Dict[str, Any]]) -> str:
    grades = [(r.get("pace") or {}).get("energy_grade") for r in cut_rows_all]
    grades = [g for g in grades if g]
    if not grades:
        return "active"
    return Counter(grades).most_common(1)[0][0]


def _speaker_count(cut_rows_all: Sequence[Dict[str, Any]]) -> int:
    return len({r["speaker"] for r in cut_rows_all if r.get("speaker")})


def _is_musical_anywhere(audio_features_by_file: Dict[str, Dict[str, Any]]) -> bool:
    return any(af.get("is_musical") for af in audio_features_by_file.values())


def _content_type_best_effort(file_ids: Sequence[str]) -> Optional[str]:
    """Best-effort genre tag (SS6.1 "content_type ... -> the natural best
    fit") for the Auto rationale text only -- never load-bearing. Reuses
    `auto_edit._clip_cards` (the pass1 header cache footage_map.py itself
    reads for the same field) rather than re-deriving it; any failure here
    just means a slightly plainer rationale string, never a broken
    suggestion."""
    try:
        from app.services.l3.auto_edit import _clip_cards
        cards = _clip_cards(list(file_ids))
        types = [c.get("content_type") for c in cards.values() if c.get("content_type")]
        return Counter(types).most_common(1)[0][0] if types else None
    except Exception:
        return None


# --------------------------------------------------------------------------
# Slot archetypes + reshuffle alternates
# --------------------------------------------------------------------------

def _auto(energy: str, aspect: str, speaker_count: int, content_type: Optional[str]) -> CaptionStyle:
    if energy == "calm":
        font_id, anim_preset, intensity, colour_source = "inter_tight", "fade", 0.4, "white"
        mood = "calm"
    elif energy == "high":
        font_id, anim_preset, intensity, colour_source = "poppins_extrabold", "pop", 0.6, "high_contrast"
        mood = "high"
    else:
        font_id, anim_preset, intensity, colour_source = "inter_tight", "karaoke", 0.5, "white"
        mood = "active"
    bits = [f"{mood} energy", f"{speaker_count} speaker(s)" if speaker_count else "no diarized speaker"]
    if content_type:
        bits.append(content_type)
    rationale = "Auto — " + ", ".join(bits) + "."
    return CaptionStyle(
        style_id="sugg_auto", label="Auto", tier="suggested", font_id=font_id,
        animation=AnimationSpec(preset=anim_preset, intensity=intensity, emphasis="loudness"),
        placement=PlacementSpec(anchor="dynamic"),
        colour=ColourSpec(source=colour_source, fill="#ffffff", emphasis_fill="#ffffff"),
        rationale=rationale,
    )


_BOLD_ALTS = [
    {"font_id": "anton", "colour_source": "high_contrast"},
    {"font_id": "poppins_extrabold", "colour_source": "palette_accent"},
]


def _bold_hype(seed: int, aspect: str, beat_sync_ok: bool) -> CaptionStyle:
    alt = _BOLD_ALTS[seed % len(_BOLD_ALTS)]
    intensity = 0.85 if aspect in ("portrait", "square") else 0.7  # SS6.3 "format-gated"
    return CaptionStyle(
        style_id="sugg_bold", label="Bold / Hype", tier="suggested",
        font_id=alt["font_id"], case="upper", tracking=0.02, max_chars_per_line=24,
        animation=AnimationSpec(preset="pop", intensity=intensity, beat_sync=beat_sync_ok, emphasis="loudness"),
        placement=PlacementSpec(anchor="dynamic"),
        colour=ColourSpec(source=alt["colour_source"], fill="#ffffff", emphasis_fill="#ffe14d"),
        rationale="Bold impact — condensed caps, beat-synced pop." if beat_sync_ok
        else "Bold impact — condensed caps, high-contrast emphasis.",
    )


_CLEAN_ALTS = [
    {"font_id": "inter_tight", "anchor": "dynamic"},
    {"font_id": "inter_tight", "anchor": "lower_third"},
]


def _clean_minimal(seed: int) -> CaptionStyle:
    alt = _CLEAN_ALTS[seed % len(_CLEAN_ALTS)]
    return CaptionStyle(
        style_id="sugg_clean", label="Clean / Minimal", tier="suggested",
        font_id=alt["font_id"], max_chars_per_line=34,
        animation=AnimationSpec(preset="fade", intensity=0.3, emphasis="none"),
        placement=PlacementSpec(anchor=alt["anchor"]),
        colour=ColourSpec(source="white", fill="#ffffff", emphasis_fill="#ffffff"),
        rationale="Clean and minimal — stays out of the way of the footage.",
    )


def _editorial_premium(seed: int, grade_label: Optional[str]) -> CaptionStyle:
    font_id = "fraunces" if seed % 2 == 0 else "inter_tight"
    tail = f", matched to your {grade_label} grade" if grade_label else ""
    return CaptionStyle(
        style_id="sugg_editorial", label="Editorial / Premium", tier="suggested",
        font_id=font_id, max_chars_per_line=36,
        animation=AnimationSpec(preset="karaoke", intensity=0.4, emphasis="semantic"),
        placement=PlacementSpec(anchor="center"),
        colour=ColourSpec(source="match_grade", fill="#ffffff", emphasis_fill="#ffffff"),
        rationale=f"Editorial karaoke-fill{tail}.",
    )


def _playful_kinetic(seed: int, beat_sync_ok: bool, has_vibrant_accent: bool) -> CaptionStyle:
    intensity = 0.75 if beat_sync_ok else 0.55
    colour_source = "palette_accent" if has_vibrant_accent else "white"
    mood = "kinetic" if beat_sync_ok else "rounded"
    rationale = (
        "Playful and kinetic — bouncy accents pulled from your footage's palette."
        if has_vibrant_accent
        else f"Playful and {mood} — a soft, friendly caption style."
    )
    return CaptionStyle(
        style_id="sugg_playful", label="Playful / Kinetic", tier="suggested",
        font_id="nunito", max_chars_per_line=26,
        animation=AnimationSpec(preset="pop", intensity=intensity, beat_sync=beat_sync_ok, emphasis="loudness"),
        placement=PlacementSpec(anchor="dynamic"),
        colour=ColourSpec(source=colour_source, fill="#ffffff", emphasis_fill="#ffffff"),
        rationale=rationale,
    )


# --------------------------------------------------------------------------
# Public
# --------------------------------------------------------------------------

def generate_suggestions(
    resolved_timeline: Dict[str, Any],
    *,
    cut_records_by_file: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    audio_features_by_file: Optional[Dict[str, Dict[str, Any]]] = None,
    color_stats_by_file: Optional[Dict[str, Dict[str, Any]]] = None,
    aspect: str = "landscape",
    grade_label: Optional[str] = None,
    reshuffle_seed: int = 0,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """5 `CaptionStyle.to_dict()` bundles (Auto pinned first) + rationale.
    Pure over already-fetched signals except for the cache lookup itself
    (process-local, keyed by `edit_signature` + `reshuffle_seed`)."""
    cut_records_by_file = cut_records_by_file or {}
    audio_features_by_file = audio_features_by_file or {}
    color_stats_by_file = color_stats_by_file or {}

    sig = edit_signature(resolved_timeline) + f"|r{reshuffle_seed}|a{aspect}"
    if use_cache and sig in _CACHE:
        return _CACHE[sig]

    cut_rows_all = [r for rows in cut_records_by_file.values() for r in rows]
    energy = _dominant_energy(cut_rows_all)
    speaker_count = _speaker_count(cut_rows_all)
    beat_sync_ok = _is_musical_anywhere(audio_features_by_file)
    file_ids = list(cut_records_by_file.keys())
    content_type = _content_type_best_effort(file_ids)
    combined_palette: List[Sequence[float]] = []
    for cs in color_stats_by_file.values():
        combined_palette.extend(cs.get("palette") or [])
    has_vibrant_accent = vibrant_accent(combined_palette) is not None

    bundles = [
        _auto(energy, aspect, speaker_count, content_type),
        _bold_hype(reshuffle_seed, aspect, beat_sync_ok),
        _clean_minimal(reshuffle_seed),
        _editorial_premium(reshuffle_seed, grade_label),
        _playful_kinetic(reshuffle_seed, beat_sync_ok, has_vibrant_accent),
    ]
    out = [b.to_dict() for b in bundles]

    if use_cache:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))  # evict oldest-inserted, cheap+good-enough
        _CACHE[sig] = out
    return out
