"""
Resolved captions -> ASS/libass subtitle text (caption_style_mvp.plan.md #4):
the SAME `resolved.captions` track the DOM/canvas preview overlay animates,
fed through the ffmpeg compositor's `ass` filter. Parity with
`frontend/src/lib/resolve-captions.ts` is enforced by construction, not by
testing each effect twice -- the four MVP animation presets (Active Reader /
Pop-Bounce / Smooth Fade Up / Sequential Reveal) and every timing constant
below are mirrored EXACTLY there; a change to one of the `_*_MS`/`_*_PX`/
`_*_SCALE` constants below needs the matching constant updated in that file
or preview and export visibly diverge.

Text is pre-wrapped into lines by `timing.py` (max_chars_per_line/max_lines,
already computed against the style's own size-derived budget) -- this module
renders those lines verbatim (`\\N` hard breaks) rather than re-wrapping, so
preview and export never disagree about where a line breaks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# ASS numpad alignment (\an): anchors the POS point to a corner/edge of the
# text block. We always center horizontally; vertical anchor follows where
# the resolved box actually sits on the canvas.
_AN_TOP, _AN_MID, _AN_BOTTOM = 8, 5, 2

# caption_style_mvp.plan.md #3: one fixed restrained outline width, one fixed
# subtle shadow depth (ASS units) -- zero when the style's toggle is off.
_OUTLINE_WIDTH = 2
_SHADOW_DEPTH = 1

# caption_style_mvp.plan.md #4: every entry transition completes within
# 200ms. Constants below are each individually <=200ms (or split into two
# phases that together are <=200ms) -- see the docstring above re: preview
# parity.
_ACTIVE_READER_ENTRY_MS = 120          # whole-caption entry fade (block appears)
_ACTIVE_READER_DIM = 0.55              # "not currently spoken" brightness factor
_POP_START_SCALE = 80                  # %, pre-ramp baseline
_POP_PEAK_SCALE = 105                  # %, peak (never exceeds this)
_POP_SETTLE_SCALE = 100                # %, rest state
_POP_RAMP_MS = 90                      # each of the two ramp phases (90+90=180ms total)
_FADE_UP_RISE_PX = 12                  # reference pixels, within the 10-15px ask
_FADE_UP_DUR_MS = 180
_SEQUENTIAL_WORD_FADE_MS = 120         # per-word appear fade


def _ms_to_ass_time(ms: int) -> str:
    ms = max(0, int(ms))
    cs = (ms % 1000) // 10
    s = (ms // 1000) % 60
    m = (ms // 60000) % 60
    h = ms // 3600000
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _hex_to_ass_bgr(hexcolor: str, alpha: int = 0) -> str:
    """`#rrggbb` -> ASS's `&HAABBGGRR` (alpha 0 = opaque, 255 = invisible)."""
    h = (hexcolor or "#ffffff").lstrip("#")
    if len(h) != 6:
        h = "ffffff"
    r, g, b = h[0:2], h[2:4], h[4:6]
    return f"&H{alpha:02X}{b}{g}{r}&"


def _dim_hex(hexcolor: str, factor: float) -> str:
    h = (hexcolor or "#ffffff").lstrip("#")
    if len(h) != 6:
        return hexcolor
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    r, g, b = (int(c * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def _escape_ass_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")


def _style_key(style: Dict[str, Any]) -> Tuple:
    font = style["font"]
    colour = style["colour"]
    return (
        font["family"], font["weight"], colour["fill"], colour["outline"], colour["shadow"],
        colour["outline_enabled"], colour["shadow_enabled"], style.get("size_pct"),
    )


def _build_styles_block(events: List[Dict[str, Any]], canvas_h: int) -> Tuple[str, Dict[Tuple, str]]:
    """One ASS `Style:` line per distinct (font, colours, size) combo
    actually used (normally exactly one -- a document has one active caption
    style -- but never assumed)."""
    seen: Dict[Tuple, str] = {}
    lines = [
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
    ]
    for ev in events:
        style = ev["style"]
        key = _style_key(style)
        if key in seen:
            continue
        name = f"Style{len(seen)}"
        seen[key] = name
        font, colour = style["font"], style["colour"]
        size_pct = style.get("size_pct") or 0.045
        font_size = max(18, round(canvas_h * size_pct))
        primary = _hex_to_ass_bgr(colour["fill"])
        secondary = _hex_to_ass_bgr(_dim_hex(colour["fill"], _ACTIVE_READER_DIM))
        outline_c = _hex_to_ass_bgr(colour["outline"])
        back_c = _hex_to_ass_bgr(colour["shadow"], alpha=(100 if colour["shadow_enabled"] else 255))
        outline_w = _OUTLINE_WIDTH if colour["outline_enabled"] else 0
        shadow_d = _SHADOW_DEPTH if colour["shadow_enabled"] else 0
        lines.append(
            f"Style: {name},{font['family']},{font_size},{primary},{secondary},{outline_c},{back_c},"
            f"{-1 if font['weight'] >= 600 else 0},0,0,0,100,100,0,0,1,"
            f"{outline_w},{shadow_d},5,20,20,20,1"
        )
    return "\n".join(lines), seen


def _anchor_for_box(box: List[float], canvas_w: int, canvas_h: int) -> Tuple[int, int, int]:
    """(an, px, py): alignment code + the ASS `\\pos` anchor point in pixels,
    matching the resolved box's vertical thirds so `\\an`'s implicit anchor
    (top/middle/bottom of the text block) lines up with where the box
    actually sits, not just its raw top-left corner."""
    x, y, w, h = box
    cx = round((x + w / 2.0) * canvas_w)
    top_third = 1.0 / 3.0
    if y < top_third:
        return _AN_TOP, cx, round(y * canvas_h)
    if y + h > 2 * top_third:
        return _AN_BOTTOM, cx, round((y + h) * canvas_h)
    return _AN_MID, cx, round((y + h / 2.0) * canvas_h)


def _event_tags(ev: Dict[str, Any], an: int, px: int, py: int) -> Tuple[str, int, int]:
    """Event-level override tags (position + whole-line entry animation).
    Returns (tag_string, px, py) since `fade_up`'s `\\move` needs a distinct
    FROM point while everything else keeps the same (px, py)."""
    preset = ev.get("anim") or "fade_up"
    if preset == "fade_up":
        y_from = py + _FADE_UP_RISE_PX if an != _AN_TOP else py - _FADE_UP_RISE_PX
        tags = [f"\\an{an}", f"\\move({px},{y_from},{px},{py},0,{_FADE_UP_DUR_MS})", f"\\fad({_FADE_UP_DUR_MS},80)"]
        return "{" + "".join(tags) + "}", px, py
    if preset == "active_reader":
        tags = [f"\\an{an}", f"\\pos({px},{py})", f"\\fad({_ACTIVE_READER_ENTRY_MS},80)"]
        return "{" + "".join(tags) + "}", px, py
    # "pop" and "sequential_reveal" both animate per-word (see _line_text);
    # the event container itself just needs a quick, unobtrusive entry.
    tags = [f"\\an{an}", f"\\pos({px},{py})", f"\\fad({_SEQUENTIAL_WORD_FADE_MS},80)"]
    return "{" + "".join(tags) + "}", px, py


def _line_text(line: Dict[str, Any], ev_start_ms: int, style_colour: Dict[str, Any], preset: str) -> str:
    """One line's text with per-word animation tags for the presets that
    animate per-word (`active_reader`, `pop`, `sequential_reveal`);
    `fade_up` animates the whole event only, so its words render plain."""
    parts: List[str] = []
    for w in line["words"]:
        text = _escape_ass_text(w["text"])
        t0 = max(0, w["t_in_ms"] - ev_start_ms)
        t1 = max(t0, w["t_out_ms"] - ev_start_ms)

        if preset == "active_reader":
            fill_c = _hex_to_ass_bgr(style_colour["fill"])
            dim_c = _hex_to_ass_bgr(_dim_hex(style_colour["fill"], _ACTIVE_READER_DIM))
            swap_ms = min(100, max(10, t1 - t0)) if t1 > t0 else 10
            parts.append(
                f"{{\\c{dim_c}}}{{\\t({t0},{t0 + swap_ms},\\c{fill_c})}}"
                f"{text}{{\\t({t1},{t1 + swap_ms},\\c{dim_c})}} "
            )
        elif preset == "pop" and w.get("emphasized"):
            fill_c = _hex_to_ass_bgr(style_colour["fill"])
            ramp1_end = t0 + _POP_RAMP_MS
            ramp2_end = ramp1_end + _POP_RAMP_MS
            parts.append(
                f"{{\\fscx{_POP_START_SCALE}\\fscy{_POP_START_SCALE}}}"
                f"{{\\t({t0},{ramp1_end},\\fscx{_POP_PEAK_SCALE}\\fscy{_POP_PEAK_SCALE})}}"
                f"{text}"
                f"{{\\t({ramp1_end},{ramp2_end},\\fscx{_POP_SETTLE_SCALE}\\fscy{_POP_SETTLE_SCALE}\\c{fill_c})}} "
            )
        elif preset == "sequential_reveal":
            fade_end = t0 + _SEQUENTIAL_WORD_FADE_MS
            parts.append(
                f"{{\\alpha&HFF&}}{{\\t({t0},{fade_end},\\alpha&H00&)}}{text} "
            )
        else:
            parts.append(f"{text} ")
    return "".join(parts).rstrip()


def _build_events_block(events: List[Dict[str, Any]], style_names: Dict[Tuple, str], canvas_w: int, canvas_h: int) -> str:
    lines = ["[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for ev in events:
        style = ev["style"]
        style_name = style_names[_style_key(style)]
        preset = ev.get("anim") or "fade_up"
        an, px, py = _anchor_for_box(ev["box"], canvas_w, canvas_h)
        tag_str, px, py = _event_tags(ev, an, px, py)
        line_texts = [
            _line_text(line, ev["prog_start_ms"], style["colour"], preset)
            for line in ev["lines"]
        ]
        text = tag_str + "\\N".join(line_texts)
        start, end = _ms_to_ass_time(ev["prog_start_ms"]), _ms_to_ass_time(ev["prog_end_ms"])
        lines.append(f"Dialogue: 0,{start},{end},{style_name},,0,0,0,,{text}")
    return "\n".join(lines)


def captions_to_ass(events: List[Dict[str, Any]], *, canvas_w: int, canvas_h: int) -> str:
    """`resolved.captions` (already program-time-mapped, box-placed,
    colour-resolved) -> a complete `.ass` document sized to the render
    canvas. Empty events -> an empty (still valid) ASS doc, so the caller
    can always attempt the burn without a special "no captions" branch."""
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        "WrapStyle: 2\n"
        f"PlayResX: {canvas_w}\n"
        f"PlayResY: {canvas_h}\n"
        "ScaledBorderAndShadow: yes\n"
    )
    styles_block, style_names = _build_styles_block(events, canvas_h)
    if not style_names:
        # No events -> still emit a minimal default style so the doc parses.
        styles_block, style_names = _build_styles_block(
            [{"style": {
                "font": {"family": "Arial", "weight": 400},
                "colour": {"fill": "#ffffff", "outline": "#000000", "shadow": "#000000",
                           "outline_enabled": False, "shadow_enabled": False},
                "size_pct": 0.045,
            }}],
            canvas_h,
        )
    events_block = _build_events_block(events, style_names, canvas_w, canvas_h)
    return f"{header}\n{styles_block}\n\n{events_block}\n"
