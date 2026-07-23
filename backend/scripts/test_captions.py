#!/usr/bin/env python3
"""Tests for caption_style_mvp.plan.md's caption-styling rewrite -- pure
functions, no DB / ffmpeg / R2 (mirrors test_grade.py's "no DB" convention,
one file covering every submodule in the captions/ package, same pattern
that file uses for grade/).

Run:  .venv/bin/python scripts/test_captions.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3.captions import ass_export  # noqa: E402
from app.services.l3.captions import colour as colour_mod  # noqa: E402
from app.services.l3.captions import placement as placement_mod  # noqa: E402
from app.services.l3.captions import resolver as resolver_mod  # noqa: E402
from app.services.l3.captions import styles as styles_mod  # noqa: E402
from app.services.l3.captions import suggest as suggest_mod  # noqa: E402
from app.services.l3.captions import timing as timing_mod  # noqa: E402
from app.services.render.compositor import CAPTION_FONTS_DIR  # noqa: E402


# --------------------------------------------------------------------------
# Catalog and API
# --------------------------------------------------------------------------

def test_exactly_four_fonts():
    assert len(styles_mod.FONTS) == 4, styles_mod.FONTS
    print("ok  catalog: exactly 4 fonts")


def test_exactly_four_colours():
    assert len(styles_mod.COLOURS) == 4, styles_mod.COLOURS
    print("ok  catalog: exactly 4 colours")


def test_exactly_four_positions():
    assert len(styles_mod.POSITIONS) == 4, styles_mod.POSITIONS
    print("ok  catalog: exactly 4 positions")


def test_exactly_four_animations():
    assert len(styles_mod.ANIMATIONS) == 4, styles_mod.ANIMATIONS
    print("ok  catalog: exactly 4 animations")


def test_exactly_four_cases():
    assert len(styles_mod.CASES) == 4, styles_mod.CASES
    print("ok  catalog: exactly 4 cases")


def test_exactly_four_sizes():
    assert len(styles_mod.SIZES) == 4, styles_mod.SIZES
    print("ok  catalog: exactly 4 sizes")


def test_exactly_one_standard():
    standards = styles_mod.list_standards()
    assert len(standards) == 1, standards
    print("ok  catalog: exactly one Standard")


def test_standard_matches_permanent_values():
    s = styles_mod.STANDARD.to_dict()
    assert s["font"]["font_id"] == "inter", s
    assert s["colour"]["colour_id"] == "white", s
    assert s["colour"]["outline_enabled"] is False, s
    assert s["colour"]["shadow_enabled"] is True, s
    assert s["position"] == "lower_third", s
    assert s["animation"] == "fade_up", s
    assert s["case"] == "original", s
    assert s["size"] == "regular", s
    print("ok  Standard: matches the plan's exact permanent values")


def test_standard_outline_disabled():
    s = styles_mod.STANDARD.to_dict()
    assert s["colour"]["outline_enabled"] is False, s
    print("ok  Standard: outline is off by default")


def test_get_standard_is_stable_regardless_of_id():
    """Exactly one Standard now -- any id (including a stale pre-MVP one)
    resolves to it, never None."""
    assert styles_mod.get_standard("std_bold_yellow") is styles_mod.STANDARD
    assert styles_mod.get_standard(None) is styles_mod.STANDARD
    assert styles_mod.get_standard() is styles_mod.STANDARD
    print("ok  get_standard: any id resolves to the one Standard")


def test_reset_restores_permanent_standard_values():
    """apply_overrides(None) on the Standard is exactly a no-op reset."""
    customized = styles_mod.apply_overrides(styles_mod.STANDARD, {"colour_id": "yellow", "outline_enabled": True})
    reset = styles_mod.apply_overrides(customized, None)
    assert reset.to_dict() == customized.to_dict()  # apply_overrides(None) is identity
    assert styles_mod.STANDARD.colour_id == "white"  # the ORIGINAL constant is untouched
    print("ok  Standard: customizing a copy never mutates the shared constant")


# --------------------------------------------------------------------------
# Style schema: overrides, from_dict, legacy parsing
# --------------------------------------------------------------------------

def test_apply_overrides_only_patches_named_fields():
    base = styles_mod.STANDARD
    out = styles_mod.apply_overrides(base, {"colour_id": "cyan", "size": "large", "unknown_field": "x"})
    assert out.colour_id == "cyan"
    assert out.size == "large"
    assert out.font_id == base.font_id  # untouched
    print("ok  apply_overrides: patches only the fields present, ignores unknown keys")


def test_apply_overrides_rejects_invalid_catalog_values():
    base = styles_mod.STANDARD
    out = styles_mod.apply_overrides(base, {"colour_id": "not_a_real_colour", "position": "bogus"})
    assert out.colour_id == base.colour_id
    assert out.position == base.position
    print("ok  apply_overrides: an invalid catalog value is ignored, not silently accepted")


def test_to_dict_from_dict_roundtrips():
    style = styles_mod.CaptionStyle(
        style_id="s1", label="Test", tier="suggested", font_id="anton", colour_id="yellow",
        outline_enabled=True, shadow_enabled=False, position="top", animation="pop",
        case="upper", size="xl",
    )
    back = styles_mod.CaptionStyle.from_dict(style.to_dict())
    assert back.to_dict() == style.to_dict()
    print("ok  CaptionStyle: to_dict/from_dict roundtrips exactly")


def test_from_dict_detects_and_normalizes_legacy_shape():
    legacy = {
        "style_id": "std_bold_yellow", "label": "Bold Yellow", "tier": "standard",
        "font": {"font_id": "anton", "case": "upper", "tracking": 0.02, "max_lines": 2, "max_chars_per_line": 22},
        "animation": {"preset": "pop", "intensity": 0.85, "beat_sync": True, "emphasis": "loudness"},
        "placement": {"anchor": "lower_third", "safe_area": True, "stability_ms": 1200},
        "colour": {"source": "white", "fill": "#ffffff", "emphasis_fill": "#ffd60a",
                   "outline": "#000000", "shadow": "#000000"},
    }
    style = styles_mod.CaptionStyle.from_dict(legacy)
    assert style.font_id == "anton"  # legacy anton -> anton (unchanged)
    assert style.animation == "pop"  # legacy pop -> pop (unchanged)
    assert style.position == "lower_third"
    assert style.case == "upper"
    assert style.outline_enabled is True and style.shadow_enabled is True  # legacy always rendered both
    print("ok  CaptionStyle.from_dict: legacy shape is detected and normalized")


def test_legacy_font_mapping_table():
    cases = {
        "inter_tight": "inter", "poppins_extrabold": "montserrat", "anton": "anton",
        "nunito": "inter", "fraunces": "inter", "permanent_marker": "inter",
    }
    for legacy_font, expected in cases.items():
        d = {"font": {"font_id": legacy_font}, "colour": {"source": "white", "fill": "#ffffff"}}
        style = styles_mod.CaptionStyle.from_dict(d)
        assert style.font_id == expected, (legacy_font, style.font_id, expected)
    print("ok  legacy font mapping: matches the plan's exact table")


def test_legacy_animation_mapping_table():
    cases = {"fade": "fade_up", "karaoke": "active_reader", "slide": "fade_up", "pop": "pop"}
    for legacy_anim, expected in cases.items():
        d = {"animation": {"preset": legacy_anim}, "colour": {"source": "white", "fill": "#ffffff"}}
        style = styles_mod.CaptionStyle.from_dict(d)
        assert style.animation == expected, (legacy_anim, style.animation, expected)
    print("ok  legacy animation mapping: matches the plan's exact table")


def test_legacy_dynamic_and_speaker_placement_maps_to_lower_third():
    for legacy_anchor in ("dynamic", "speaker"):
        d = {"placement": {"anchor": legacy_anchor}, "colour": {"source": "white", "fill": "#ffffff"}}
        style = styles_mod.CaptionStyle.from_dict(d)
        assert style.position == "lower_third", (legacy_anchor, style.position)
    print("ok  legacy dynamic/speaker placement -> lower_third")


def test_legacy_dynamic_colour_maps_to_nearest_fixed_swatch():
    # A near-yellow legacy fill (e.g. from an old palette_accent/match_grade
    # resolve) should land on "yellow", not silently default to white.
    d = {"colour": {"source": "palette_accent", "fill": "#ffec3a"}}
    style = styles_mod.CaptionStyle.from_dict(d)
    assert style.colour_id == "yellow", style.colour_id
    print("ok  legacy dynamic colour -> nearest fixed palette colour")


def test_legacy_docs_still_render_a_valid_mvp_style_end_to_end():
    """The plan's actual bar: an old document with the pre-MVP shape must
    still resolve to something render-safe -- every field a member of the
    new catalog, not a crash, not None."""
    legacy_selection = {
        "enabled": True,
        "base_style": {
            "style_id": "sugg_editorial", "label": "Editorial / Premium", "tier": "suggested",
            "font": {"font_id": "fraunces", "case": "as-is"},
            "animation": {"preset": "karaoke", "intensity": 0.4, "emphasis": "semantic"},
            "placement": {"anchor": "center"},
            "colour": {"source": "match_grade", "fill": "#e0d8c8"},
        },
    }
    style = resolver_mod.effective_style(legacy_selection)
    assert style is not None
    d = style.to_dict()
    assert d["font"]["font_id"] in styles_mod.FONTS
    assert d["colour"]["colour_id"] in styles_mod.COLOURS
    assert d["position"] in styles_mod.POSITIONS
    assert d["animation"] in styles_mod.ANIMATIONS
    print("ok  a full legacy document selection resolves to a valid MVP style")


# --------------------------------------------------------------------------
# Placement -- 4 fixed positions
# --------------------------------------------------------------------------

def test_placement_four_positions_distinct_vertical_bands():
    boxes = {p: placement_mod.resolve_placement([], position=p, aspect="landscape")["box"] for p in placement_mod.POSITIONS}
    centers = {p: box[1] + box[3] / 2 for p, box in boxes.items()}
    # top < center < bottom_dynamic < lower_third, strictly increasing
    ordered = sorted(centers.items(), key=lambda kv: kv[1])
    assert [p for p, _ in ordered] == ["top", "center", "bottom_dynamic", "lower_third"], ordered
    print("ok  placement: 4 positions land in strictly increasing vertical order")


def test_placement_falls_back_to_lower_third_when_unrecognized():
    box_unknown = placement_mod.resolve_placement([], position="not_a_position", aspect="landscape")["box"]
    box_lower = placement_mod.resolve_placement([], position="lower_third", aspect="landscape")["box"]
    assert box_unknown == box_lower
    print("ok  placement: unrecognized/missing position falls back to lower_third")


def test_placement_respects_safe_margins_per_aspect():
    for aspect in ("portrait", "square", "landscape"):
        box = placement_mod.resolve_placement([], position="lower_third", aspect=aspect, safe_area=True)["box"]
        x, y, w, h = box
        assert 0.0 <= x and x + w <= 1.0 + 1e-6, (aspect, box)
        assert 0.0 <= y and y + h <= 1.0 + 1e-6, (aspect, box)
    print("ok  placement: every position stays inside the safe rect for every aspect")


def test_placement_no_dynamic_or_per_cut_zones_used():
    """Even with caption_zones present on the cut rows, a fixed position
    must NOT route through them (plan #2: "do not use dynamic ... or
    per-cut caption_zones for these styles")."""
    rows_with_zones = [{"caption_zones": [[0.1, 0.1, 0.1, 0.1]], "framing": {}}]
    with_zones = placement_mod.resolve_placement(rows_with_zones, position="lower_third", aspect="landscape")
    without_zones = placement_mod.resolve_placement([], position="lower_third", aspect="landscape")
    assert with_zones["box"] == without_zones["box"]
    assert with_zones["source"] == "fixed"
    print("ok  placement: caption_zones on cut rows never influence a fixed position")


def test_placement_nudges_off_a_covering_subject_box():
    subject_covering_lower_third = [{"framing": {"subject_box": [0.06, 0.70, 0.88, 0.22]}}]
    box = placement_mod.resolve_placement(subject_covering_lower_third, position="lower_third", aspect="landscape")["box"]
    lower_third_default = placement_mod.resolve_placement([], position="lower_third", aspect="landscape")["box"]
    assert box != lower_third_default
    print("ok  placement: a fixed position still nudges off a subject box it lands on")


# --------------------------------------------------------------------------
# Colour -- deterministic outline/shadow
# --------------------------------------------------------------------------

def test_colour_resolve_is_fully_deterministic():
    a = colour_mod.resolve_colour("white", outline_enabled=True, shadow_enabled=False)
    b = colour_mod.resolve_colour("white", outline_enabled=True, shadow_enabled=False)
    assert a == b
    print("ok  colour: resolve_colour is a pure deterministic lookup")


def test_colour_light_swatches_get_dark_ink():
    for colour_id in ("white", "yellow", "cyan"):
        resolved = colour_mod.resolve_colour(colour_id, outline_enabled=True, shadow_enabled=True)
        assert resolved["outline"] == "#000000", (colour_id, resolved)
        assert resolved["shadow"] == "#000000", (colour_id, resolved)
    print("ok  colour: White/Yellow/Cyan get a dark outline+shadow ink")


def test_colour_charcoal_gets_light_ink():
    resolved = colour_mod.resolve_colour("charcoal", outline_enabled=True, shadow_enabled=True)
    assert resolved["outline"] == "#FFFFFF", resolved
    assert resolved["shadow"] == "#FFFFFF", resolved
    print("ok  colour: Charcoal gets a light outline+shadow ink")


def test_colour_unknown_id_falls_back_to_default():
    resolved = colour_mod.resolve_colour("not_a_real_colour")
    assert resolved["fill"] == styles_mod.COLOURS[styles_mod.DEFAULT_COLOUR_ID]["hex"]
    print("ok  colour: an unrecognized colour_id falls back to the default swatch")


def test_vibrant_accent_still_usable_as_a_ranking_signal():
    """Not removed -- suggest.py's ranking still needs it (see suggest.py's
    _accent_close_to), even though resolve_colour no longer calls it."""
    accent = colour_mod.vibrant_accent([[0.9, 0.85, 0.1]])
    assert accent is not None
    print("ok  colour: vibrant_accent is still exported for suggest.py's ranking signal")


# --------------------------------------------------------------------------
# Case handling (timing.py)
# --------------------------------------------------------------------------

def test_case_original_passes_through_unchanged():
    assert timing_mod._display_text("Hello", "original", is_first_in_event=True) == "Hello"
    print("ok  case: original passes the transcript's own casing through")


def test_case_upper():
    assert timing_mod._display_text("hello", "upper") == "HELLO"
    print("ok  case: upper")


def test_case_lower():
    assert timing_mod._display_text("HELLO", "lower") == "hello"
    print("ok  case: lower")


def test_case_sentence_capitalizes_only_first_word_of_event():
    first = timing_mod._display_text("HELLO", "sentence", is_first_in_event=True)
    second = timing_mod._display_text("WORLD", "sentence", is_first_in_event=False)
    assert first == "Hello", first
    assert second == "world", second
    print("ok  case: sentence capitalizes only the first word of the event")


def test_build_events_applies_case_to_first_word_of_each_event():
    words = [
        {"text": "hello", "start_ms": 0, "end_ms": 300},
        {"text": "there", "start_ms": 350, "end_ms": 700},
    ]
    events = timing_mod.build_events(
        words, src_in_ms=0, prog_start_ms=0, layer_prog_end_ms=2000,
        max_chars_per_line=40, max_lines=2, case="sentence",
        emphasis_mode="loudness", beat_sync=False,
    )
    assert events[0]["lines"][0]["words"][0]["text"] == "Hello"
    assert events[0]["lines"][0]["words"][1]["text"] == "there"
    print("ok  build_events: sentence case is wired through to the first word of an event")


# --------------------------------------------------------------------------
# ass_export.py -- outline/shadow zero, 4 animations, 200ms budget
# --------------------------------------------------------------------------

def _fake_event(*, position="lower_third", animation="fade_up", colour_id="white",
                 outline_enabled=False, shadow_enabled=False, words=None):
    words = words or [
        {"text": "hello", "t_in_ms": 0, "t_out_ms": 300, "emphasized": True},
        {"text": "world", "t_in_ms": 350, "t_out_ms": 700, "emphasized": False},
    ]
    style = styles_mod.CaptionStyle(
        style_id="s", label="S", font_id="inter", colour_id=colour_id,
        outline_enabled=outline_enabled, shadow_enabled=shadow_enabled,
        position=position, animation=animation, case="original", size="regular",
    )
    d = style.to_dict()
    box = placement_mod.resolve_placement([], position=position, aspect="landscape")["box"]
    return {
        "prog_start_ms": 0, "prog_end_ms": 1000, "lines": [{"words": words}],
        "box": box, "style_ref": "s", "style": d, "anim": animation,
    }


def test_ass_outline_off_is_zero_width():
    # Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,
    # OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX,
    # ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, ... -- fields[16]
    # is Outline width, fields[17] is Shadow depth (fields[15]=BorderStyle).
    ev = _fake_event(outline_enabled=False, shadow_enabled=True)
    block, _names = ass_export._build_styles_block([ev], 1080)
    style_line = [ln for ln in block.splitlines() if ln.startswith("Style:")][0]
    fields = style_line.split(",")
    outline_w = int(fields[16])
    assert outline_w == 0, style_line
    print("ok  ass_export: outline disabled -> ASS Outline field is 0")


def test_ass_shadow_off_is_zero_depth():
    ev = _fake_event(outline_enabled=True, shadow_enabled=False)
    block, _names = ass_export._build_styles_block([ev], 1080)
    style_line = [ln for ln in block.splitlines() if ln.startswith("Style:")][0]
    fields = style_line.split(",")
    shadow_d = int(fields[17])
    assert shadow_d == 0, style_line
    print("ok  ass_export: shadow disabled -> ASS Shadow field is 0")


def test_ass_outline_and_shadow_on_are_nonzero():
    ev = _fake_event(outline_enabled=True, shadow_enabled=True)
    block, _names = ass_export._build_styles_block([ev], 1080)
    style_line = [ln for ln in block.splitlines() if ln.startswith("Style:")][0]
    fields = style_line.split(",")
    assert int(fields[16]) > 0 and int(fields[17]) > 0, style_line
    print("ok  ass_export: both enabled -> nonzero Outline and Shadow")


def test_ass_font_size_matches_size_pct():
    for size, pct in styles_mod.SIZE_FRAME_PCT.items():
        style = styles_mod.CaptionStyle(style_id="s", label="S", size=size)
        ev = {**_fake_event(), "style": style.to_dict()}
        block, _ = ass_export._build_styles_block([ev], 1000)
        style_line = [ln for ln in block.splitlines() if ln.startswith("Style:")][0]
        font_size = int(style_line.split(",")[2])
        assert font_size == max(18, round(1000 * pct)), (size, font_size)
    print("ok  ass_export: font size is derived from the style's size_pct, matches preview's metric")


def test_ass_all_four_animation_presets_produce_events_block():
    for anim in styles_mod.ANIMATIONS:
        ev = _fake_event(animation=anim)
        doc = ass_export.captions_to_ass([ev], canvas_w=1080, canvas_h=1920)
        assert "[Events]" in doc and "Dialogue:" in doc, (anim, doc)
    print("ok  ass_export: all 4 animation presets render a valid events block")


def test_ass_active_reader_and_sequential_reveal_use_word_timestamps():
    """Both must emit per-word \\t(...) transforms keyed off each word's OWN
    t_in/t_out (not a fixed/shared timing) -- the actual "follow word
    timestamps" contract."""
    words = [
        {"text": "aaa", "t_in_ms": 0, "t_out_ms": 200, "emphasized": False},
        {"text": "bbb", "t_in_ms": 500, "t_out_ms": 900, "emphasized": False},
    ]
    for anim in ("active_reader", "sequential_reveal"):
        ev = _fake_event(animation=anim, words=words)
        line_text = ass_export._line_text(ev["lines"][0], ev["prog_start_ms"], ev["style"]["colour"], anim)
        # Distinct \t(...) start times present for each word's own t_in.
        assert "\\t(0," in line_text, (anim, line_text)
        assert "\\t(500," in line_text, (anim, line_text)
    print("ok  ass_export: active_reader/sequential_reveal key transforms off each word's own timestamps")


def test_ass_entry_transitions_within_200ms_budget():
    assert ass_export._FADE_UP_DUR_MS <= 200
    assert ass_export._ACTIVE_READER_ENTRY_MS <= 200
    assert 2 * ass_export._POP_RAMP_MS <= 200
    assert ass_export._SEQUENTIAL_WORD_FADE_MS <= 200
    print("ok  ass_export: every animation's entry transition constant is <=200ms")


def test_ass_pop_scale_never_exceeds_105_percent():
    assert ass_export._POP_PEAK_SCALE <= 105
    print("ok  ass_export: Pop/Bounce peak scale never exceeds 105%%")


def test_ass_fade_up_rise_within_plan_range():
    assert 10 <= ass_export._FADE_UP_RISE_PX <= 15
    print("ok  ass_export: Smooth Fade Up rise is within the plan's 10-15px range")


def test_ass_empty_events_still_produce_valid_doc():
    doc = ass_export.captions_to_ass([], canvas_w=1080, canvas_h=1920)
    assert "[Script Info]" in doc and "[V4+ Styles]" in doc
    print("ok  ass_export: empty events still produce a parseable minimal doc")


# --------------------------------------------------------------------------
# Fonts -- bundled + libass-resolvable
# --------------------------------------------------------------------------

def test_all_four_caption_fonts_bundled_on_disk():
    """caption_style_mvp.plan.md #5: "detect and report missing backend
    caption fonts during development or tests" -- this IS that detector."""
    missing = []
    for font_id, spec in styles_mod.FONTS.items():
        path = os.path.join(CAPTION_FONTS_DIR, f"caption-{font_id}.ttf")
        if not os.path.isfile(path):
            missing.append((font_id, spec["family"], path))
    assert not missing, f"missing bundled caption font TTFs: {missing}"
    print("ok  fonts: all 4 catalog fonts have a bundled caption-<id>.ttf on disk")


def test_frontend_woff2_fonts_bundled_on_disk():
    fonts_dir = os.path.join(BACKEND, "..", "frontend", "public", "fonts")
    missing = []
    for font_id in styles_mod.FONTS:
        path = os.path.join(fonts_dir, f"caption-{font_id}.woff2")
        if not os.path.isfile(path):
            missing.append((font_id, path))
    assert not missing, f"missing bundled caption font WOFF2s: {missing}"
    print("ok  fonts: all 4 catalog fonts have a bundled caption-<id>.woff2 for preview")


# --------------------------------------------------------------------------
# suggest.py -- constrained ranking
# --------------------------------------------------------------------------

_RESOLVED_TIMELINE = {"video_layers": [
    {"kind": "spine", "source_file_id": "f1", "src_in_ms": 0, "src_out_ms": 5000, "prog_start_ms": 0, "prog_end_ms": 5000},
]}


def _signal_kwargs(**overrides):
    base = dict(
        cut_records_by_file={"f1": [{
            "pace": {"energy_grade": "active"}, "speaker_person": "p1", "on_camera": True,
            "framing": {"shot_size": "medium", "subject_box": [0.3, 0.3, 0.2, 0.3]},
        }]},
        audio_features_by_file={"f1": {"is_musical": False}},
        color_stats_by_file={"f1": {"rgb_mean": [0.4, 0.4, 0.4], "palette": [[0.5, 0.5, 0.5]]}},
        transcripts_by_file={"f1": {"segments": [{"words": [
            {"text": "hello", "start_ms": 0, "end_ms": 300, "is_filler": False},
            {"text": "world", "start_ms": 350, "end_ms": 700, "is_filler": False},
        ]}]}},
    )
    base.update(overrides)
    return base


def test_suggest_returns_exactly_four():
    out = suggest_mod.generate_suggestions(_RESOLVED_TIMELINE, **_signal_kwargs(), use_cache=False)
    assert len(out) == 4, out
    print("ok  suggest: returns exactly 4 bundles")


def test_suggest_same_edit_and_seed_is_deterministic():
    a = suggest_mod.generate_suggestions(_RESOLVED_TIMELINE, **_signal_kwargs(), reshuffle_seed=3, use_cache=False)
    b = suggest_mod.generate_suggestions(_RESOLVED_TIMELINE, **_signal_kwargs(), reshuffle_seed=3, use_cache=False)
    assert a == b
    print("ok  suggest: same edit + seed -> identical results")


def test_suggest_regeneration_produces_valid_alternatives():
    a = suggest_mod.generate_suggestions(_RESOLVED_TIMELINE, **_signal_kwargs(), reshuffle_seed=0, use_cache=False)
    b = suggest_mod.generate_suggestions(_RESOLVED_TIMELINE, **_signal_kwargs(), reshuffle_seed=1, use_cache=False)
    for bundle in a + b:
        assert bundle["font"]["font_id"] in styles_mod.FONTS
        assert bundle["colour"]["colour_id"] in styles_mod.COLOURS
        assert bundle["position"] in styles_mod.POSITIONS
        assert bundle["animation"] in styles_mod.ANIMATIONS
        assert bundle["case"] in styles_mod.CASES
        assert bundle["size"] in styles_mod.SIZES
    print("ok  suggest: regeneration output is still 100%% catalog-valid")


def test_suggest_no_duplicate_style_ids():
    out = suggest_mod.generate_suggestions(_RESOLVED_TIMELINE, **_signal_kwargs(), use_cache=False)
    ids = [b["style_id"] for b in out]
    assert len(ids) == len(set(ids)), ids
    print("ok  suggest: no duplicate suggestions")


def test_suggest_missing_optional_analysis_still_returns_four():
    out = suggest_mod.generate_suggestions(_RESOLVED_TIMELINE, use_cache=False)  # everything omitted
    assert len(out) == 4, out
    print("ok  suggest: missing optional analysis still returns 4 suggestions")


def test_suggest_hard_constraint_no_sequential_reveal_without_word_timestamps():
    out = suggest_mod.generate_suggestions(
        _RESOLVED_TIMELINE, **_signal_kwargs(transcripts_by_file={}), use_cache=False,
    )
    assert all(b["animation"] != "sequential_reveal" for b in out), out
    print("ok  suggest: sequential_reveal never suggested without word timestamps")


def test_suggest_hard_constraint_no_pop_for_calm_low_energy():
    out = suggest_mod.generate_suggestions(
        _RESOLVED_TIMELINE,
        **_signal_kwargs(
            cut_records_by_file={"f1": [{"pace": {"energy_grade": "calm"}, "speaker_person": "p1", "on_camera": False, "framing": {}}]},
            audio_features_by_file={"f1": {"is_musical": False}},
        ),
        use_cache=False,
    )
    assert all(b["animation"] != "pop" for b in out), out
    print("ok  suggest: pop never suggested for calm/non-musical/single-speaker content")


def test_suggest_hard_constraint_no_lowercase_anton():
    out = suggest_mod.generate_suggestions(_RESOLVED_TIMELINE, **_signal_kwargs(), reshuffle_seed=7, use_cache=False)
    assert not any(b["case"] == "lower" and b["font"]["font_id"] == "anton" for b in out), out
    print("ok  suggest: lowercase+Anton never combined")


def test_suggest_diversity_no_font_repeated_more_than_twice():
    out = suggest_mod.generate_suggestions(_RESOLVED_TIMELINE, **_signal_kwargs(), use_cache=False)
    from collections import Counter
    counts = Counter(b["font"]["font_id"] for b in out)
    assert max(counts.values()) <= 2, counts
    print("ok  suggest: no font repeated more than twice across the 4 picks")


def test_suggest_diversity_no_animation_repeated_more_than_twice():
    out = suggest_mod.generate_suggestions(_RESOLVED_TIMELINE, **_signal_kwargs(), use_cache=False)
    from collections import Counter
    counts = Counter(b["animation"] for b in out)
    assert max(counts.values()) <= 2, counts
    print("ok  suggest: no animation repeated more than twice across the 4 picks")


def test_suggest_diversity_no_near_identical_pairs():
    out = suggest_mod.generate_suggestions(_RESOLVED_TIMELINE, **_signal_kwargs(), use_cache=False)
    fields = ("font", "colour", "position", "animation", "case", "size")

    def sig(b):
        return (b["font"]["font_id"], b["colour"]["colour_id"], b["position"], b["animation"], b["case"], b["size"])
    for i in range(4):
        for j in range(i + 1, 4):
            same = sum(1 for a, c in zip(sig(out[i]), sig(out[j])) if a == c)
            assert same / len(fields) < 0.83, (out[i]["style_id"], out[j]["style_id"])
    print("ok  suggest: no two of the 4 picks are near-identical")


def test_suggest_never_crashes_on_malformed_word_timestamps():
    out = suggest_mod.generate_suggestions(
        _RESOLVED_TIMELINE,
        **_signal_kwargs(transcripts_by_file={"f1": {"segments": [{"words": [
            {"text": "x", "start_ms": 0, "end_ms": None, "is_filler": False},
        ]}]}}),
        use_cache=False,
    )
    assert len(out) == 4
    print("ok  suggest: a malformed word timestamp never crashes generation")


# --------------------------------------------------------------------------
# resolver.py -- end-to-end pure resolve (no DB)
# --------------------------------------------------------------------------

def test_resolve_captions_end_to_end_produces_valid_events():
    style = styles_mod.STANDARD
    resolved_timeline = {"video_layers": [
        {"kind": "spine", "source_file_id": "f1", "src_in_ms": 0, "src_out_ms": 2000,
         "prog_start_ms": 0, "prog_end_ms": 2000},
    ]}
    transcripts = {"f1": {"segments": [{"words": [
        {"text": "hello", "start_ms": 0, "end_ms": 300, "is_filler": False},
        {"text": "world", "start_ms": 350, "end_ms": 700, "is_filler": False},
    ]}]}}
    events = resolver_mod.resolve_captions(
        resolved_timeline, style=style, transcripts_by_file=transcripts, aspect="landscape",
    )
    assert len(events) >= 1
    ev = events[0]
    assert ev["style"]["colour"]["colour_id"] == "white"
    assert ev["anim"] == "fade_up"
    assert len(ev["box"]) == 4
    print("ok  resolver: resolve_captions produces a valid event track for the Standard")


def test_resolve_captions_none_style_is_empty():
    assert resolver_mod.resolve_captions({"video_layers": []}, style=None) == []
    print("ok  resolver: style=None (captions off) -> no events, no crash")


def test_effective_style_none_when_disabled():
    assert resolver_mod.effective_style({"enabled": False, "style_id": "std_standard"}) is None
    assert resolver_mod.effective_style(None) is None
    print("ok  resolver: effective_style is None when captions are off/unselected")


def main():
    test_exactly_four_fonts()
    test_exactly_four_colours()
    test_exactly_four_positions()
    test_exactly_four_animations()
    test_exactly_four_cases()
    test_exactly_four_sizes()
    test_exactly_one_standard()
    test_standard_matches_permanent_values()
    test_standard_outline_disabled()
    test_get_standard_is_stable_regardless_of_id()
    test_reset_restores_permanent_standard_values()
    test_apply_overrides_only_patches_named_fields()
    test_apply_overrides_rejects_invalid_catalog_values()
    test_to_dict_from_dict_roundtrips()
    test_from_dict_detects_and_normalizes_legacy_shape()
    test_legacy_font_mapping_table()
    test_legacy_animation_mapping_table()
    test_legacy_dynamic_and_speaker_placement_maps_to_lower_third()
    test_legacy_dynamic_colour_maps_to_nearest_fixed_swatch()
    test_legacy_docs_still_render_a_valid_mvp_style_end_to_end()
    test_placement_four_positions_distinct_vertical_bands()
    test_placement_falls_back_to_lower_third_when_unrecognized()
    test_placement_respects_safe_margins_per_aspect()
    test_placement_no_dynamic_or_per_cut_zones_used()
    test_placement_nudges_off_a_covering_subject_box()
    test_colour_resolve_is_fully_deterministic()
    test_colour_light_swatches_get_dark_ink()
    test_colour_charcoal_gets_light_ink()
    test_colour_unknown_id_falls_back_to_default()
    test_vibrant_accent_still_usable_as_a_ranking_signal()
    test_case_original_passes_through_unchanged()
    test_case_upper()
    test_case_lower()
    test_case_sentence_capitalizes_only_first_word_of_event()
    test_build_events_applies_case_to_first_word_of_each_event()
    test_ass_outline_off_is_zero_width()
    test_ass_shadow_off_is_zero_depth()
    test_ass_outline_and_shadow_on_are_nonzero()
    test_ass_font_size_matches_size_pct()
    test_ass_all_four_animation_presets_produce_events_block()
    test_ass_active_reader_and_sequential_reveal_use_word_timestamps()
    test_ass_entry_transitions_within_200ms_budget()
    test_ass_pop_scale_never_exceeds_105_percent()
    test_ass_fade_up_rise_within_plan_range()
    test_ass_empty_events_still_produce_valid_doc()
    test_all_four_caption_fonts_bundled_on_disk()
    test_frontend_woff2_fonts_bundled_on_disk()
    test_suggest_returns_exactly_four()
    test_suggest_same_edit_and_seed_is_deterministic()
    test_suggest_regeneration_produces_valid_alternatives()
    test_suggest_no_duplicate_style_ids()
    test_suggest_missing_optional_analysis_still_returns_four()
    test_suggest_hard_constraint_no_sequential_reveal_without_word_timestamps()
    test_suggest_hard_constraint_no_pop_for_calm_low_energy()
    test_suggest_hard_constraint_no_lowercase_anton()
    test_suggest_diversity_no_font_repeated_more_than_twice()
    test_suggest_diversity_no_animation_repeated_more_than_twice()
    test_suggest_diversity_no_near_identical_pairs()
    test_suggest_never_crashes_on_malformed_word_timestamps()
    test_resolve_captions_end_to_end_produces_valid_events()
    test_resolve_captions_none_style_is_empty()
    test_effective_style_none_when_disabled()
    print("\nall captions tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
