"""
AI Pick generation (caption_style_mvp.plan.md #6): exactly 4 ranked caption
style suggestions, built by constructing every combination of the approved
catalog values (styles.FONTS x COLOURS x POSITIONS x ANIMATIONS x CASES x
SIZES x outline x shadow), rejecting combinations that violate a hard
compatibility constraint, scoring the survivors against this edit's own
perceived signals, and greedily selecting the top-scoring combo plus 3 more
under a diversity penalty and the plan's diversity constraints.

Deterministic (no LLM, no true randomness): the same edit + the same
`reshuffle_seed` always produce the same 4 picks, so a suggestion set is
reproducible and cheap. "Regenerate" (SS6.5-style) is a seeded, small score
perturbation among near-tied candidates -- never a re-roll that could
contradict a hard constraint, and never touches the user's ALREADY-APPLIED
style (that's the router/document's job, not this module's).

Every suggestion still flows through the SAME `resolver.resolve_captions`
pipeline as the Standard, so placement/colour/readability guarantees are
enforced structurally by `placement.py`/`colour.py`/`timing.py` -- nothing
here has to re-implement them.
"""
from __future__ import annotations

import hashlib
import re
import zlib
from collections import Counter
from itertools import product
from typing import Any, Dict, List, Optional, Sequence

from app.services.l3.captions.colour import footage_luma, vibrant_accent
from app.services.l3.captions.styles import (
    ANIMATIONS, CASES, COLOURS, FONTS, POSITIONS, SIZES,
    CaptionStyle, slugify_style_id,
)

# Process-local cache: mirrors `grade.cache`'s "local disk cache, not shared
# across instances, fine since it's cheap to regenerate" contract -- just
# don't recompute the same signature twice in one process's lifetime.
_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_CACHE_MAX = 256

_TIGHT_SHOTS = {"extreme_close_up", "close_up", "medium_close_up"}
_DARK_LUMA_THRESHOLD = 0.30
_BRIGHT_LUMA_THRESHOLD = 0.55
_LONG_LINE_WPM_THRESHOLD = 170.0  # a commonly-cited "fast talker" threshold
_ACCENT_PROXIMITY_THRESHOLD = 0.18  # RGB-space closeness for "similar to footage"
_HIGH_ENERGY_MAX = 2       # "at most two high-energy choices"
_MAX_FONT_REPEATS = 2      # "do not repeat the same font more than twice"
_MAX_ANIM_REPEATS = 2      # "do not repeat the same animation more than twice"
_NEAR_IDENTICAL_SIMILARITY = 0.83   # 5 of 6 comparable fields matching
_DIVERSITY_WEIGHT = 1.8
_JITTER_WEIGHT = 0.35
_TOP_N_FOR_SELECTION = 200  # see generate_suggestions -- bounds _select_four's O(pool*picked) cost


def edit_signature(resolved_timeline: Dict[str, Any], captions_selection: Optional[Dict[str, Any]] = None) -> str:
    """A content signature that changes exactly when the edit "materially
    changes": the ordered spine layer identity, same shape the frontend's
    own reset-detection (`composite-preview.tsx`'s `shapeKey`) already uses
    for "did the plan change shape". Independent of the captions selection
    itself (a style pick shouldn't bust the suggestion cache)."""
    spine = [
        v for v in (resolved_timeline.get("video_layers") or []) if v.get("kind") == "spine"
    ]
    key = "|".join(
        f"{v['source_file_id']}:{v['src_in_ms']}-{v['src_out_ms']}:{v['prog_start_ms']}"
        for v in sorted(spine, key=lambda v: v["prog_start_ms"])
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Signals -- every one backed by a real, currently-populated field. No
# fabricated "content_type" genre tag: the L2 perception layer that used to
# populate one was removed (migration 032_drop_l2_perception.sql) and
# nothing replaces it, so this module doesn't pretend to read one.
# --------------------------------------------------------------------------

def _dominant_energy(cut_rows_all: Sequence[Dict[str, Any]]) -> str:
    grades = [(r.get("pace") or {}).get("energy_grade") for r in cut_rows_all]
    grades = [g for g in grades if g]
    if not grades:
        return "active"
    return Counter(grades).most_common(1)[0][0]


def _speaker_count(cut_rows_all: Sequence[Dict[str, Any]]) -> int:
    """`speaker_person` (voice-first-identity, code-owned) -- the pre-MVP
    code read a `speaker` column that no longer exists post-migration and
    always silently returned 0; this reads the real field."""
    return len({r["speaker_person"] for r in cut_rows_all if r.get("speaker_person")})


def _is_musical_anywhere(audio_features_by_file: Dict[str, Dict[str, Any]]) -> bool:
    return any(af.get("is_musical") for af in audio_features_by_file.values())


def _dominant_shot_size(cut_rows_all: Sequence[Dict[str, Any]]) -> Optional[str]:
    sizes = [((r.get("framing") or {}).get("shot_size")) for r in cut_rows_all]
    sizes = [s for s in sizes if s]
    if not sizes:
        return None
    counts: Dict[str, int] = {}
    for s in sizes:
        counts[s] = counts.get(s, 0) + 1
    return max(sizes, key=lambda s: (counts[s], -sizes.index(s)))


def _is_face_heavy_talking_head(cut_rows_all: Sequence[Dict[str, Any]]) -> bool:
    on_camera = [r for r in cut_rows_all if r.get("on_camera")]
    if not cut_rows_all:
        return False
    on_camera_frac = len(on_camera) / len(cut_rows_all)
    tight_frac = sum(1 for r in cut_rows_all if (r.get("framing") or {}).get("shot_size") in _TIGHT_SHOTS)
    tight_frac = tight_frac / len(cut_rows_all)
    return on_camera_frac >= 0.5 and tight_frac >= 0.4


def _subjects_in_upper_frame(cut_rows_all: Sequence[Dict[str, Any]]) -> bool:
    ys = []
    for r in cut_rows_all:
        sb = (r.get("framing") or {}).get("subject_box")
        if sb and len(sb) == 4:
            ys.append(float(sb[1]))
    if not ys:
        return False
    ys.sort()
    median_y = ys[len(ys) // 2]
    return median_y < 0.22


def _footage_luma_avg(color_stats_by_file: Dict[str, Dict[str, Any]]) -> float:
    if not color_stats_by_file:
        return 0.4
    lumas = [footage_luma(cs) for cs in color_stats_by_file.values()]
    return sum(lumas) / len(lumas)


def _rgb_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def _hex_to_rgb01(hexcolor: str) -> tuple:
    h = hexcolor.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _accent_close_to(colour_id: str, color_stats_by_file: Dict[str, Dict[str, Any]]) -> bool:
    """"Similar footage" for the yellow/cyan-without-protection constraint:
    does the combined palette already carry a vibrant entry close in RGB
    space to this swatch (so the caption would visually blend in)?"""
    combined_palette: List[Sequence[float]] = []
    for cs in color_stats_by_file.values():
        combined_palette.extend(cs.get("palette") or [])
    accent = vibrant_accent(combined_palette)
    if accent is None:
        return False
    return _rgb_distance(accent, _hex_to_rgb01(COLOURS[colour_id]["hex"])) < _ACCENT_PROXIMITY_THRESHOLD


_STOPWORDS_RE = re.compile(r"^[a-zA-Z']+$")


def _speech_pace_wpm(transcripts_by_file: Dict[str, Dict[str, Any]]) -> Optional[float]:
    """Words per minute across every real (non-filler) word in every
    transcript this edit touches. None when there's no transcript data at
    all, or every segment's timestamps are unusable (never a fabricated
    number, never a crash on a malformed/incomplete word)."""
    total_words = 0
    span_ms = 0
    for t in transcripts_by_file.values():
        for seg in t.get("segments") or []:
            words = [w for w in (seg.get("words") or []) if not w.get("is_filler")]
            if not words:
                continue
            start, end = words[0].get("start_ms"), words[-1].get("end_ms")
            if start is None or end is None:
                continue
            total_words += len(words)
            span_ms += int(end) - int(start)
    if total_words == 0 or span_ms <= 0:
        return None
    return total_words / (span_ms / 60000.0)


def _word_timestamps_complete(transcripts_by_file: Dict[str, Dict[str, Any]]) -> bool:
    """False the moment ANY word is missing a usable [start_ms,end_ms) --
    Sequential Reveal needs every word's own timing to look right, not just
    most of them."""
    found_any = False
    for t in transcripts_by_file.values():
        for seg in t.get("segments") or []:
            for w in seg.get("words") or []:
                if w.get("is_filler"):
                    continue
                found_any = True
                s, e = w.get("start_ms"), w.get("end_ms")
                if s is None or e is None or int(e) <= int(s):
                    return False
    return found_any


def _compute_signals(
    cut_rows_all: Sequence[Dict[str, Any]],
    audio_features_by_file: Dict[str, Dict[str, Any]],
    color_stats_by_file: Dict[str, Dict[str, Any]],
    transcripts_by_file: Dict[str, Dict[str, Any]],
    aspect: str,
) -> Dict[str, Any]:
    energy = _dominant_energy(cut_rows_all)
    is_musical = _is_musical_anywhere(audio_features_by_file)
    speaker_count = _speaker_count(cut_rows_all)
    wpm = _speech_pace_wpm(transcripts_by_file)
    return {
        "aspect": aspect,
        "energy": energy,
        "is_musical": is_musical,
        "speaker_count": speaker_count,
        "footage_luma": _footage_luma_avg(color_stats_by_file),
        "is_face_heavy_talking_head": _is_face_heavy_talking_head(cut_rows_all),
        "subjects_in_upper_frame": _subjects_in_upper_frame(cut_rows_all),
        "dominant_shot_size": _dominant_shot_size(cut_rows_all),
        "speech_pace_wpm": wpm,
        "long_lines": bool(wpm and wpm >= _LONG_LINE_WPM_THRESHOLD),
        "word_timestamps_complete": _word_timestamps_complete(transcripts_by_file),
        "is_calm_low_energy_content": energy == "calm" and not is_musical and speaker_count <= 1,
        "accent_close": {cid: _accent_close_to(cid, color_stats_by_file) for cid in ("yellow", "cyan")},
        "cut_count": len(cut_rows_all),
    }


# --------------------------------------------------------------------------
# Combination space, hard constraints, scoring
# --------------------------------------------------------------------------

def _all_combinations() -> List[Dict[str, Any]]:
    out = []
    for font_id, colour_id, position, animation, case, size, outline_enabled, shadow_enabled in product(
        FONTS.keys(), COLOURS.keys(), POSITIONS, ANIMATIONS, CASES, SIZES, (False, True), (False, True),
    ):
        out.append({
            "font_id": font_id, "colour_id": colour_id, "position": position, "animation": animation,
            "case": case, "size": size, "outline_enabled": outline_enabled, "shadow_enabled": shadow_enabled,
        })
    return out


def _passes_hard_constraints(combo: Dict[str, Any], sig: Dict[str, Any]) -> bool:
    protected = combo["outline_enabled"] or combo["shadow_enabled"]
    if combo["colour_id"] == "charcoal" and sig["footage_luma"] < _DARK_LUMA_THRESHOLD and not protected:
        return False
    if combo["colour_id"] in ("yellow", "cyan") and sig["accent_close"].get(combo["colour_id"]) and not protected:
        return False
    if combo["position"] == "center" and sig["is_face_heavy_talking_head"]:
        return False
    if combo["position"] == "top" and sig["subjects_in_upper_frame"]:
        return False
    if combo["animation"] == "sequential_reveal" and not sig["word_timestamps_complete"]:
        return False
    if combo["animation"] == "pop" and sig["is_calm_low_energy_content"]:
        return False
    if combo["size"] == "xl" and sig["long_lines"]:
        return False
    if combo["case"] == "lower" and combo["font_id"] == "anton":
        return False
    return True


_ANIM_SCORE_BY_ENERGY = {
    "high": {"pop": 3.0, "sequential_reveal": 1.5, "active_reader": 1.0, "fade_up": 0.0},
    "calm": {"fade_up": 3.0, "active_reader": 1.5, "sequential_reveal": 1.0, "pop": 0.0},
    "active": {"active_reader": 2.5, "sequential_reveal": 2.0, "pop": 1.5, "fade_up": 1.0},
}
_FONT_SCORE_BY_ENERGY = {
    "high": {"anton": 2.0, "montserrat": 1.5, "jost": 1.0, "inter": 0.5},
    "calm": {"inter": 2.0, "jost": 1.5, "montserrat": 1.0, "anton": 0.3},
    "active": {"montserrat": 1.5, "jost": 1.3, "inter": 1.2, "anton": 1.0},
}


def _score(combo: Dict[str, Any], sig: Dict[str, Any]) -> float:
    score = 0.0
    score += _ANIM_SCORE_BY_ENERGY.get(sig["energy"], {}).get(combo["animation"], 0.5)
    score += _FONT_SCORE_BY_ENERGY.get(sig["energy"], {}).get(combo["font_id"], 0.5)

    if sig["is_musical"] and combo["animation"] in ("active_reader", "sequential_reveal"):
        score += 1.0

    if sig["is_face_heavy_talking_head"]:
        score += {"lower_third": 1.5, "bottom_dynamic": 1.0, "top": 0.3, "center": 0.0}[combo["position"]]
    if sig["aspect"] in ("portrait", "square"):
        score += {"bottom_dynamic": 0.7, "lower_third": 0.5, "center": 0.2, "top": 0.0}[combo["position"]]

    if combo["case"] == "upper" and sig["energy"] == "high":
        score += 1.0
    if combo["case"] in ("original", "sentence") and sig["energy"] == "calm":
        score += 0.5

    if sig["long_lines"]:
        score += {"small": 1.5, "regular": 1.0, "large": 0.3, "xl": 0.0}[combo["size"]]
    elif sig["energy"] == "high":
        score += {"large": 1.3, "xl": 1.0, "regular": 0.7, "small": 0.2}[combo["size"]]
    else:
        score += {"regular": 1.0, "large": 0.6, "small": 0.5, "xl": 0.2}[combo["size"]]

    if combo["colour_id"] == "white":
        score += 1.0
    elif combo["colour_id"] in ("yellow", "cyan") and sig["energy"] in ("high", "active"):
        score += 0.8
    elif combo["colour_id"] == "charcoal" and sig["footage_luma"] > _BRIGHT_LUMA_THRESHOLD:
        score += 0.6

    if combo["outline_enabled"]:
        score += 0.4
    if combo["shadow_enabled"]:
        score += 0.3
    if sig["footage_luma"] < _DARK_LUMA_THRESHOLD and combo["outline_enabled"] and combo["shadow_enabled"]:
        score += 0.5

    return score


def _jitter(combo: Dict[str, Any], reshuffle_seed: int) -> float:
    """Deterministic per-combo, per-seed perturbation -- "regenerate" reorders
    near-tied candidates without ever touching hard constraints (already
    applied before this) or swamping a genuinely strong signal-driven score.
    `zlib.crc32` (not `hashlib.sha1`): this runs per-combo, and with ~32k raw
    combinations a SHA1+hexdigest per combo measurably slowed down every
    request (see JITTER_POOL_SIZE below for the other half of that fix) --
    crc32 is far cheaper and still a fine deterministic scatter for a
    cosmetic tie-breaker, not a security context."""
    key = "|".join(str(combo[k]) for k in sorted(combo)) + f"|{reshuffle_seed}"
    return (zlib.crc32(key.encode("utf-8")) % 1000) / 1000.0 * _JITTER_WEIGHT


_LABELS = {
    "font_id": {"montserrat": "Montserrat", "anton": "Anton", "jost": "Jost", "inter": "Inter"},
    "colour_id": {k: v["label"] for k, v in COLOURS.items()},
    "position": {"lower_third": "lower third", "center": "dead center", "top": "upper third",
                 "bottom_dynamic": "bottom dynamic"},
    "animation": {"active_reader": "Active Reader", "pop": "Pop/Bounce",
                  "fade_up": "Smooth Fade Up", "sequential_reveal": "Sequential Reveal"},
}


def _rationale(combo: Dict[str, Any], sig: Dict[str, Any]) -> str:
    bits = [f"{sig['energy']} energy", _LABELS["position"][combo["position"]]]
    bits.append(_LABELS["animation"][combo["animation"]])
    if sig["speaker_count"]:
        bits.append(f"{sig['speaker_count']} speaker(s)")
    if sig["is_musical"]:
        bits.append("musical audio")
    return f"{_LABELS['font_id'][combo['font_id']]}, {_LABELS['colour_id'][combo['colour_id']]} — " + ", ".join(bits) + "."


def _diversity_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    fields = ("font_id", "colour_id", "position", "animation", "case", "size")
    return sum(1 for f in fields if a[f] == b[f]) / len(fields)


def _is_high_energy(combo: Dict[str, Any]) -> bool:
    return combo["animation"] == "pop" or combo["size"] in ("large", "xl")


def _is_restrained(combo: Dict[str, Any]) -> bool:
    return combo["animation"] == "fade_up" and combo["size"] in ("small", "regular")


def _select_four(scored: List[Dict[str, Any]], full_pool: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Greedy top-score-first selection under a diversity penalty + the
    plan's explicit diversity constraints, applied as a final reconciliation
    pass (simpler and more predictable than encoding every constraint into
    the greedy objective directly).

    `scored` is normally a fast top-N-by-score slice (see
    generate_suggestions), not the whole combination space -- correct in
    the overwhelming common case (diversity penalties only ever SUBTRACT,
    so a combo far outside the top tier by plain score could never win a
    slot). But a narrow signal profile (e.g. most animations blocked by
    hard constraints) can leave that slice dominated by near-duplicates of
    the single best-scoring combo with nothing left to diversify against.
    `full_pool` (defaults to `scored` itself) is searched ONLY when `scored`
    truly has nothing eligible left -- keeps the fast path fast without
    ever silently violating "avoid near-identical suggestions" for real."""
    full_pool = full_pool if full_pool is not None else scored
    pool = sorted(scored, key=lambda c: -c["_total_score"])
    picked: List[Dict[str, Any]] = []
    font_counts: Dict[str, int] = {}
    anim_counts: Dict[str, int] = {}

    def eligible(c: Dict[str, Any]) -> bool:
        if font_counts.get(c["font_id"], 0) >= _MAX_FONT_REPEATS:
            return False
        if anim_counts.get(c["animation"], 0) >= _MAX_ANIM_REPEATS:
            return False
        return all(_diversity_similarity(c, p) < _NEAR_IDENTICAL_SIMILARITY for p in picked)

    while len(picked) < 4:
        candidates = [c for c in pool if eligible(c) and c not in picked]
        if not candidates:
            # The fast slice has nothing eligible left -- widen to the full
            # combination space before relaxing any constraint.
            candidates = [c for c in full_pool if eligible(c) and c not in picked]
        if not candidates:
            # Still nothing -- relax the near-identical guard only (never
            # leave a slot empty; fail-open over fail-hard).
            candidates = [c for c in full_pool if c not in picked and
                          font_counts.get(c["font_id"], 0) < _MAX_FONT_REPEATS and
                          anim_counts.get(c["animation"], 0) < _MAX_ANIM_REPEATS]
        if not candidates:
            candidates = [c for c in full_pool if c not in picked]
        if not candidates:
            break

        def adjusted(c: Dict[str, Any]) -> float:
            penalty = sum(_diversity_similarity(c, p) for p in picked)
            return c["_total_score"] - _DIVERSITY_WEIGHT * penalty

        best = max(candidates, key=adjusted)
        picked.append(best)
        font_counts[best["font_id"]] = font_counts.get(best["font_id"], 0) + 1
        anim_counts[best["animation"]] = anim_counts.get(best["animation"], 0) + 1

    # Diversity reconciliation: at least one restrained, at most two
    # high-energy. Swap the lowest-scoring offender for the best-scoring
    # untaken candidate that fixes the imbalance AND still isn't
    # near-identical to any of the picks it's joining (the swap must not
    # reintroduce the very problem the greedy pass just avoided). Searches
    # `pool` first, then `full_pool`, same widen-before-relax order as above.
    def _swap_in(swap_out: Dict[str, Any], candidates_filter) -> bool:
        """True iff a replacement was actually made. Repeat caps are
        rechecked with `swap_out`'s OWN font/animation counted as already
        freed (it's leaving), so a same-font/animation replacement is still
        allowed when that's genuinely the only fix, but a swap can never
        silently push a count over the cap the greedy pass just enforced."""
        others = [p for p in picked if p is not swap_out]

        def within_caps(c: Dict[str, Any]) -> bool:
            font_cap = _MAX_FONT_REPEATS + (1 if c["font_id"] == swap_out["font_id"] else 0)
            anim_cap = _MAX_ANIM_REPEATS + (1 if c["animation"] == swap_out["animation"] else 0)
            return font_counts.get(c["font_id"], 0) < font_cap and anim_counts.get(c["animation"], 0) < anim_cap

        def find(space: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            base = [c for c in space if candidates_filter(c) and c not in picked and within_caps(c)]
            strict = [c for c in base if all(_diversity_similarity(c, p) < _NEAR_IDENTICAL_SIMILARITY for p in others)]
            return strict or base

        loose = find(pool) or find(full_pool)
        if not loose:
            return False
        winner = max(loose, key=lambda c: c["_total_score"])
        font_counts[swap_out["font_id"]] -= 1
        anim_counts[swap_out["animation"]] -= 1
        picked[picked.index(swap_out)] = winner
        font_counts[winner["font_id"]] = font_counts.get(winner["font_id"], 0) + 1
        anim_counts[winner["animation"]] = anim_counts.get(winner["animation"], 0) + 1
        return True

    if picked and not any(_is_restrained(p) for p in picked):
        _swap_in(min(picked, key=lambda c: c["_total_score"]), _is_restrained)

    high_energy_picks = [p for p in picked if _is_high_energy(p)]
    while len(high_energy_picks) > _HIGH_ENERGY_MAX:
        swapped = _swap_in(min(high_energy_picks, key=lambda c: c["_total_score"]), lambda c: not _is_high_energy(c))
        if not swapped:
            break  # pool exhausted -- stop rather than loop forever
        high_energy_picks = [p for p in picked if _is_high_energy(p)]

    return picked[:4]


# --------------------------------------------------------------------------
# Public
# --------------------------------------------------------------------------

def generate_suggestions(
    resolved_timeline: Dict[str, Any],
    *,
    cut_records_by_file: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    audio_features_by_file: Optional[Dict[str, Dict[str, Any]]] = None,
    color_stats_by_file: Optional[Dict[str, Dict[str, Any]]] = None,
    transcripts_by_file: Optional[Dict[str, Dict[str, Any]]] = None,
    aspect: str = "landscape",
    reshuffle_seed: int = 0,
    use_cache: bool = True,
) -> List[Dict[str, Any]]:
    """Exactly 4 `CaptionStyle.to_dict()` bundles + rationale, ranked
    highest-scoring first. Pure over already-fetched signals except for the
    cache lookup itself (process-local, keyed by `edit_signature` +
    `reshuffle_seed`)."""
    cut_records_by_file = cut_records_by_file or {}
    audio_features_by_file = audio_features_by_file or {}
    color_stats_by_file = color_stats_by_file or {}
    transcripts_by_file = transcripts_by_file or {}

    sig_key = edit_signature(resolved_timeline) + f"|r{reshuffle_seed}|a{aspect}"
    if use_cache and sig_key in _CACHE:
        return _CACHE[sig_key]

    cut_rows_all = [r for rows in cut_records_by_file.values() for r in rows]
    signals = _compute_signals(cut_rows_all, audio_features_by_file, color_stats_by_file, transcripts_by_file, aspect)

    valid = [c for c in _all_combinations() if _passes_hard_constraints(c, signals)]
    if not valid:
        # Every combination somehow rejected (should not happen -- "original"
        # case + fade_up + white always clears every constraint) -- fail
        # open to the single safest combo rather than returning nothing.
        valid = [{
            "font_id": "inter", "colour_id": "white", "position": "lower_third", "animation": "fade_up",
            "case": "original", "size": "regular", "outline_enabled": True, "shadow_enabled": True,
        }]
    # `_select_four`'s diversity penalty only ever SUBTRACTS from a score,
    # so a combo far outside the top tier by plain score could never win a
    # slot regardless of diversity/jitter -- score everything cheaply first
    # (no hashing), then only run the expensive greedy/diversity selection
    # (O(pool * picked) per round) over the top slice. Restrained combos
    # alone are 1/8 of the whole 32768-combo space, so a slice this size
    # keeps every diversity/reconciliation constraint reachable in practice.
    for c in valid:
        c["_total_score"] = _score(c, signals)
    top_pool = sorted(valid, key=lambda c: -c["_total_score"])[:_TOP_N_FOR_SELECTION]
    for c in top_pool:
        c["_total_score"] += _jitter(c, reshuffle_seed)

    picks = _select_four(top_pool, full_pool=valid)
    while len(picks) < 4:  # combinatorially impossible given 4^6*4 combos, but never crash
        picks.append(picks[-1] if picks else valid[0])

    bundles = []
    for i, combo in enumerate(picks):
        style_id = slugify_style_id("sugg", combo["font_id"], combo["colour_id"], combo["animation"], str(i))
        style = CaptionStyle(
            style_id=style_id, label=f"AI Pick {i + 1}", tier="suggested",
            font_id=combo["font_id"], colour_id=combo["colour_id"],
            outline_enabled=combo["outline_enabled"], shadow_enabled=combo["shadow_enabled"],
            position=combo["position"], animation=combo["animation"],
            case=combo["case"], size=combo["size"],
            rationale=_rationale(combo, signals),
        )
        bundles.append(style.to_dict())

    if use_cache:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.pop(next(iter(_CACHE)))
        _CACHE[sig_key] = bundles
    return bundles
