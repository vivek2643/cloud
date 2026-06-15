"""
Editorial principles: the declarative, weighted STYLE layer.

Sibling to the spine. The spine says what is locked vs free (structure); the
principles say how taste is exercised within that freedom (style). They are
vague, weighted tendencies -- knobs, NOT rules and NOT prompt prose -- that
parameterize the deterministic baseline scorer (`baseline.py`). A principle is
one knob; a principle SET is many at once.

Storage: `document["principles"]` = [{ id, weight, scope }]. `scope` is "global"
(default) or a spine-region label for an override. Unset principles fall back to
the defaults below, so the scorer always has a value. Opus sets/overrides them
via `set_principles` (like `set_spine`); the user can edit them.
"""
from __future__ import annotations

from typing import Dict, List, Optional

# Known vocabulary -> default weight. Weights are 0..1 (tendency strength)
# unless a knob is naturally bipolar (kept 0..1 here; the scorer interprets).
DEFAULTS: Dict[str, float] = {
    # A. Picture / angle selection
    "favor_speaker": 0.8,
    "reward_reaction": 0.5,
    "shot_variety": 0.4,
    "prefer_well_framed": 0.5,
    "establishing_bias": 0.3,
    "eyeline_continuity": 0.3,
    "avoid_jump_cut": 0.6,
    # B. Pacing / rhythm
    "pace": 0.5,
    "anti_metronome": 0.5,
    "min_max_shot_length": 0.5,
    "accelerate_to_climax": 0.3,
    "hold_on_emotion": 0.4,
    "cut_to_music": 0.5,
    # C. Cut quality / continuity
    "clean_seam_bias": 0.7,
    "protect_speech": 0.9,
    "protect_action": 0.7,
    "match_action": 0.4,
    "preserve_reveals": 0.8,
    # D. Content selection
    "prefer_best_take": 0.8,
    "tighten_dead_air": 0.6,
    "keep_complete_thoughts": 0.7,
    "avoid_redundancy": 0.4,
    "honor_priority": 0.7,
    # E. Narrative / structure
    "hook_first": 0.6,
    "follow_outline": 0.7,
    "build_arc": 0.3,
    "bookend": 0.2,
    # F. Audio / mix
    "speech_intelligibility": 0.9,
    "bed_balance": 0.6,
    "smooth_seams": 0.5,
    "jl_cut_bias": 0.3,
    # G. Format / delivery
    "target_duration": 0.7,
    "platform_framing": 0.3,
}

KNOWN = set(DEFAULTS)

# The subset wired into the baseline scorer first (the rest register + default).
V1_CORE = (
    "favor_speaker",
    "reward_reaction",
    "shot_variety",
    "pace",
    "anti_metronome",
    "hook_first",
    "tighten_dead_air",
)


def normalize(entries: List[dict]) -> tuple[List[dict], List[str]]:
    """Validate incoming principle entries. Returns (clean, errors).

    Each entry: {id in KNOWN, weight: float, scope: str = 'global'}. Unknown ids
    or non-numeric weights are dropped and reported."""
    clean: List[dict] = []
    errors: List[str] = []
    for e in entries or []:
        pid = e.get("id")
        if pid not in KNOWN:
            errors.append(f"unknown principle '{pid}'")
            continue
        try:
            weight = float(e.get("weight"))
        except (TypeError, ValueError):
            errors.append(f"principle '{pid}' has a non-numeric weight")
            continue
        weight = max(-1.0, min(1.0, weight))
        clean.append({"id": pid, "weight": weight, "scope": e.get("scope") or "global"})
    return clean, errors


def weight_of(document: dict, pid: str, region: Optional[str] = None) -> float:
    """Resolve a principle's weight: region override -> global -> default."""
    entries = document.get("principles") or []
    global_val: Optional[float] = None
    for e in entries:
        if e.get("id") != pid:
            continue
        scope = e.get("scope", "global")
        if region is not None and scope == region:
            return float(e.get("weight", DEFAULTS.get(pid, 0.0)))
        if scope == "global" and global_val is None:
            global_val = float(e.get("weight", DEFAULTS.get(pid, 0.0)))
    if global_val is not None:
        return global_val
    return DEFAULTS.get(pid, 0.0)


def render_principles_text(document: dict) -> str:
    """Compact 'ACTIVE PRINCIPLES' block for the prompt (soft guidance only).

    Shows explicitly-set weights; unset knobs run at their defaults silently."""
    entries = document.get("principles") or []
    if not entries:
        return ("ACTIVE PRINCIPLES: (defaults) -- set_principles to bias the cut "
                "(e.g. favor_speaker, reward_reaction, pace, anti_metronome, hook_first).")
    parts = []
    for e in entries:
        sc = e.get("scope", "global")
        tag = f"{e['id']}={e['weight']:.2f}" + ("" if sc == "global" else f"@{sc}")
        parts.append(tag)
    return "ACTIVE PRINCIPLES: " + ", ".join(parts)
