"""
Resolved captions -> ASS/libass subtitle text (captions.plan.md SS12): the
SAME `resolved.captions` track the DOM/canvas preview overlay animates, fed
through `_lut3d_arg`-style local-path escaping and burned by the ffmpeg
compositor's `ass` filter. Parity is enforced by CONSTRUCTION, not by
testing each effect twice: SS16 resolved #2 restricts v1 to exactly the four
animation presets that have a faithful ASS expression (fade / pop / karaoke
/ slide) -- `styles.ANIMATION_PRESETS` is already limited to these, so
`_event_tags` below never has to approximate a fifth.

Text is pre-wrapped into lines by `timing.py` (max_chars_per_line/max_lines,
already computed against the STYLE's own budget) -- this module renders
those lines verbatim (`\\N` hard breaks) rather than re-wrapping, so preview
and export never disagree about where a line breaks.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

# ASS numpad alignment (\an): anchors the POS point to a corner/edge of the
# text block. We always center horizontally; vertical anchor follows where
# the resolved box actually sits on the canvas.
_AN_TOP, _AN_MID, _AN_BOTTOM = 8, 5, 2

_KARAOKE_SECONDARY_DIM = 0.45  # "not yet sung" text: fill dimmed toward the outline colour


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
        font["family"], font["weight"], font.get("tracking", 0.0),
        colour["fill"], colour["emphasis_fill"], colour["outline"], colour["shadow"], colour.get("box"),
    )


def _build_styles_block(events: List[Dict[str, Any]], canvas_h: int) -> Tuple[str, Dict[Tuple, str]]:
    """One ASS `Style:` line per distinct (font, colours) combo actually used
    (normally exactly one -- a document has one active caption style -- but
    never assumed, so a future per-event style override stays correct)."""
    seen: Dict[Tuple, str] = {}
    lines = [
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
    ]
    font_size = max(18, round(canvas_h * 0.045))
    for ev in events:
        style = ev["style"]
        key = _style_key(style)
        if key in seen:
            continue
        name = f"Style{len(seen)}"
        seen[key] = name
        font, colour = style["font"], style["colour"]
        primary = _hex_to_ass_bgr(colour["fill"])
        secondary = _hex_to_ass_bgr(_dim_hex(colour["fill"], _KARAOKE_SECONDARY_DIM))
        outline_c = _hex_to_ass_bgr(colour["outline"])
        back_c = _hex_to_ass_bgr(colour["box"] or colour["shadow"], alpha=(0 if colour["box"] else 100))
        border_style = 3 if colour.get("box") else 1  # 3 = opaque box, 1 = outline+shadow
        outline_w = 3 if style["colour"].get("strong_outline") else 2
        spacing = round(float(font.get("tracking", 0.0)) * font_size)
        lines.append(
            f"Style: {name},{font['family']},{font_size},{primary},{secondary},{outline_c},{back_c},"
            f"{-1 if font['weight'] >= 600 else 0},0,0,0,100,100,{spacing},0,{border_style},"
            f"{outline_w},2,5,20,20,20,1"
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


def _event_tags(ev: Dict[str, Any], an: int, px: int, py: int) -> str:
    """Event-level override tags (position + whole-line animation). Per-word
    tags (karaoke `\\kf`, pop `\\t`) are built separately in `_line_text`."""
    anim = ev["anim"]
    preset = anim.get("preset", "fade")
    tags = [f"\\an{an}", f"\\pos({px},{py})"]
    if preset == "fade":
        fad_ms = max(80, round(180 * (0.4 + float(anim.get("intensity", 0.5)))))
        tags.append(f"\\fad({fad_ms},{fad_ms})")
    elif preset == "slide":
        rise = round(40 + 60 * float(anim.get("intensity", 0.5)))
        dur = 220
        y_from = py + rise if an != _AN_TOP else py - rise
        tags = [f"\\an{an}", f"\\move({px},{y_from},{px},{py},0,{dur})", f"\\fad({dur},80)"]
    return "{" + "".join(tags) + "}"


def _line_text(line: Dict[str, Any], ev_start_ms: int, style_colour: Dict[str, Any], anim: Dict[str, Any]) -> str:
    preset = anim.get("preset", "fade")
    intensity = float(anim.get("intensity", 0.5))
    parts: List[str] = []
    for w in line["words"]:
        text = _escape_ass_text(w["text"])
        if preset == "karaoke":
            cs = max(1, round((w["t_out_ms"] - w["t_in_ms"]) / 10))
            parts.append(f"{{\\kf{cs}}}{text} ")
        elif preset == "pop" and w.get("emphasized"):
            scale = round(100 + 35 * intensity)
            emph_colour = _hex_to_ass_bgr(style_colour["emphasis_fill"])
            t_ms = max(1, w["t_in_ms"] - ev_start_ms)
            pop_dur = min(220, max(80, w["t_out_ms"] - w["t_in_ms"]))
            parts.append(
                f"{{\\t({t_ms},{t_ms + pop_dur},\\fscx{scale}\\fscy{scale}\\c{emph_colour})}}"
                f"{text}{{\\t({t_ms + pop_dur},{t_ms + pop_dur + pop_dur},\\fscx100\\fscy100)}} "
            )
        else:
            parts.append(f"{text} ")
    return "".join(parts).rstrip()


def _build_events_block(events: List[Dict[str, Any]], style_names: Dict[Tuple, str], canvas_w: int, canvas_h: int) -> str:
    lines = ["[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for ev in events:
        style = ev["style"]
        style_name = style_names[_style_key(style)]
        an, px, py = _anchor_for_box(ev["box"], canvas_w, canvas_h)
        tags = _event_tags(ev, an, px, py)
        line_texts = [
            _line_text(line, ev["prog_start_ms"], style["colour"], ev["anim"]) for line in ev["lines"]
        ]
        text = tags + "\\N".join(line_texts)
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
                "font": {"family": "Arial", "weight": 400, "tracking": 0.0},
                "colour": {"fill": "#ffffff", "emphasis_fill": "#ffffff", "outline": "#000000",
                           "shadow": "#000000", "box": None, "strong_outline": False},
            }}],
            canvas_h,
        )
    events_block = _build_events_block(events, style_names, canvas_w, canvas_h)
    return f"{header}\n{styles_block}\n\n{events_block}\n"
