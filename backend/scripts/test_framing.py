"""
Regression test for Phase-1 geometric framing (rotate / fit / zoom / anchor).

All DB-free and ffmpeg-free: the framing SOLVER (`layers.solve_transform`) and
the compositor FILTER builder (`compositor._transform_vf`) are pure functions, so
the whole contract is pinned without any media.

What this guards:
  * the automatic fit policy -- vertical/square deliveries FILL the frame (cover),
    landscape keeps the historical letterbox (contain), so existing landscape
    edits stay byte-identical;
  * the identity transform produces EXACTLY the old normalize filter (warm-cache
    safety) and collapses the cache key to '' (no cache busting);
  * explicit overrides (rotate clamped to 0/90/180/270, zoom >= 1, anchor) are
    honored and turned into the right ffmpeg chain in the canonical order
    rotate -> fit -> zoom-crop;
  * resolve() attaches a solved transform to every video layer.

The frontend resolver (`resolve-timeline.ts::solveTransform`) is a hand-verified
mirror of `solve_transform`; the canonical outputs asserted here are the parity
contract both sides must satisfy.

Run:  cd backend && python3 scripts/test_framing.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.l3 import layers  # noqa: E402
from app.services.render import compositor as C  # noqa: E402


def test_fit_policy_by_aspect() -> None:
    """Vertical/square fill; landscape letterboxes."""
    assert layers.default_fit("portrait") == "cover"
    assert layers.default_fit("square") == "cover"
    assert layers.default_fit("landscape") == "contain"
    for aspect in ("portrait", "square"):
        t = layers.solve_transform({"format": {"aspect": aspect}})
        assert t["fit"] == "cover", t
        assert t["rotate"] == 0 and t["zoom"] == 1.0 and t["anchor"] == "center"
    t = layers.solve_transform({"format": {"aspect": "landscape"}})
    assert t["fit"] == "contain", t
    print("  OK  fit policy: portrait/square=cover, landscape=contain")


def test_identity_matches_old_normalize() -> None:
    """Identity filter is byte-identical to the historical normalize, and its
    cache token collapses to '' so warm landscape segments are never re-encoded."""
    cfg = C._cfg_for("preview", "landscape")
    old = (
        f"scale=w={cfg['width']}:h={cfg['height']}:force_original_aspect_ratio=decrease,"
        f"pad={cfg['width']}:{cfg['height']}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1,fps={cfg['fps']}"
    )
    assert C._transform_vf(cfg, None) == old
    assert C._transform_vf(cfg, layers.identity_transform("landscape")) == old
    assert C._transform_key(None) == ""
    assert C._transform_key(layers.identity_transform("landscape")) == ""
    print("  OK  identity == old normalize, cache token collapses to ''")


def test_cover_and_overrides_build_canonical_chain() -> None:
    """Cover crops to the canvas; overrides apply in order rotate -> fit -> zoom."""
    pcfg = C._cfg_for("preview", "portrait")
    assert (pcfg["width"], pcfg["height"]) == (720, 1280)
    cover = layers.solve_transform({"format": {"aspect": "portrait"}})
    vf = C._transform_vf(pcfg, cover)
    assert "force_original_aspect_ratio=increase" in vf and "crop=720:1280" in vf, vf

    cfg = C._cfg_for("preview", "landscape")
    ov = layers.solve_transform(
        {"format": {"aspect": "landscape"}},
        {"rotate": 90, "zoom": 1.5, "anchor": "left", "fit": "contain"},
    )
    assert ov == {"rotate": 90, "fit": "contain", "anchor": "left", "zoom": 1.5, "dest": "full"}
    vf = C._transform_vf(cfg, ov)
    assert vf.startswith("transpose=1,"), vf            # rotate first
    assert "force_original_aspect_ratio=decrease" in vf  # contain fit
    assert "scale=w=1920:h=1080" in vf and "crop=1280:720:0:" in vf  # zoom-crop, left-anchored
    assert C._transform_key(ov) != ""                    # non-identity busts the cache
    print("  OK  cover crops to canvas; overrides build rotate->fit->zoom chain")


def test_override_clamping() -> None:
    """Rotate snaps to the orthogonal set; zoom floors at 1; junk falls back."""
    base = {"format": {"aspect": "landscape"}}
    assert layers.solve_transform(base, {"rotate": 450})["rotate"] == 90   # 450 % 360
    assert layers.solve_transform(base, {"rotate": 45})["rotate"] == 0     # not orthogonal
    assert layers.solve_transform(base, {"zoom": 0.3})["zoom"] == 1.0      # floor
    assert layers.solve_transform(base, {"zoom": "x"})["zoom"] == 1.0      # junk
    assert layers.solve_transform(base, {"anchor": "weird"})["anchor"] == "center"
    print("  OK  rotate/zoom/anchor overrides clamp safely")


def test_resolve_attaches_transform() -> None:
    """Every resolved video layer carries a solved transform."""
    doc = {
        "format": {"aspect": "portrait"},
        "timeline": [{"seg_id": "s1", "file_id": "f1", "in_ms": 0, "out_ms": 1000}],
        "operations": [{
            "type": "place_video", "op_id": "ov1", "source_file_id": "f2",
            "src_in_ms": 0, "src_out_ms": 500, "from_ms": 0, "to_ms": 500,
            "transform": {"zoom": 2.0},
        }],
    }
    r = layers.resolve(doc)
    spine = r.video_layers[0].to_dict()
    assert spine["transform"]["fit"] == "cover", spine
    cover_op = next(v.to_dict() for v in r.video_layers if v.op_id == "ov1")
    assert cover_op["transform"]["zoom"] == 2.0, cover_op
    print("  OK  resolve attaches a transform to spine + operation layers")


# --------------------------------------------------------------------------
# Phase 2: perception-grounded focus + orientation
# --------------------------------------------------------------------------

def test_focus_from_motion_centroid_only() -> None:
    """Focus resolves purely from the subject-motion centroid; empty -> None."""
    from app.services.l3 import framing

    pts = [{"ts_ms": 500, "centroid": [0.2, 0.8]}, {"ts_ms": 1500, "centroid": [0.4, 0.6]}]
    f = framing.focus_for_range(pts, 0, 2000)
    assert f["source"] == "motion" and abs(f["cx"] - 0.3) < 1e-6, f
    assert framing.focus_for_range([], 0, 1000) is None  # nothing -> centered
    print("  OK  focus from motion centroid; empty -> None")


def test_focus_builds_quoted_crop_expr() -> None:
    """A focus point produces a clamped, comma-escaped, single-quoted crop expr
    on the cover path (and the cache key changes); contain ignores focus."""
    cfg = C._cfg_for("preview", "portrait")
    t = layers.solve_transform({"format": {"aspect": "portrait"}}, {"focus": {"cx": 0.75, "cy": 0.4}})
    vf = C._transform_vf(cfg, t)
    assert "crop=720:1280:'min(max(0\\,0.7500*iw-360)\\,iw-720)'" in vf, vf
    assert C._transform_key(t).endswith("|f0.7500,0.4000"), C._transform_key(t)
    # Landscape (contain) never crops, so focus is inert there.
    lcfg = C._cfg_for("preview", "landscape")
    lt = layers.solve_transform({"format": {"aspect": "landscape"}}, {"focus": {"cx": 0.9, "cy": 0.1}})
    assert "crop=" not in C._transform_vf(lcfg, lt)
    print("  OK  focus builds quoted crop expr on cover; contain ignores it")


# --------------------------------------------------------------------------
# Phase 3: animated motion (push-in / follow) + the zoom style knob
# --------------------------------------------------------------------------

def test_motion_normalize_and_sample() -> None:
    """normalize_motion clamps + drops no-move paths; sample_motion eases."""
    m = layers.normalize_motion({"from": {"scale": 0.5, "cx": -1, "cy": 2},
                                 "to": {"scale": 1.2, "cx": 0.7, "cy": 0.5},
                                 "ease": "smooth", "dur_ms": 2000})
    assert m["from"] == {"scale": 1.0, "cx": 0.0, "cy": 1.0}, m  # clamped
    # no-move collapses to None (falls back to static zoom/focus).
    assert layers.normalize_motion({"from": {"scale": 1, "cx": .5, "cy": .5},
                                    "to": {"scale": 1, "cx": .5, "cy": .5},
                                    "dur_ms": 1000}) is None
    s = layers.sample_motion(m, 1000)  # smoothstep midpoint p=0.5
    assert abs(s["scale"] - 1.1) < 1e-6 and abs(s["cx"] - 0.35) < 1e-6, s
    assert layers.sample_motion(m, 0)["scale"] == 1.0
    assert abs(layers.sample_motion(m, 2000)["cx"] - 0.7) < 1e-6
    print("  OK  motion normalize clamps/drops no-move; sample eases")


def test_motion_styles_build() -> None:
    """punch=static zoom; push=animated scale held on focus; follow pans w/ dwell."""
    from app.services.l3 import framing

    assert framing._build_motion("punch_in", "snappy", 2000, None, None) == {"zoom": 1.12}
    push = framing._build_motion("push_in", "glide", 2000, {"cx": 0.4, "cy": 0.5}, {"cx": 0.9, "cy": 0.5})
    assert push["motion"]["from"]["scale"] < push["motion"]["to"]["scale"]
    assert push["motion"]["from"]["cx"] == push["motion"]["to"]["cx"] == 0.4  # push holds focus
    foll = framing._build_motion("follow", "snappy", 2000, {"cx": 0.2, "cy": 0.5}, {"cx": 0.8, "cy": 0.5})
    assert foll["motion"]["from"]["cx"] == 0.2 and foll["motion"]["to"]["cx"] == 0.8
    # dwell: a tiny move is held still (from == to).
    held = framing._build_motion("follow", "snappy", 2000, {"cx": 0.5, "cy": 0.5}, {"cx": 0.52, "cy": 0.5})
    assert held["motion"]["from"] == held["motion"]["to"], held
    assert framing._build_motion("static", "snappy", 2000, None, None) is None
    print("  OK  punch=static zoom; push holds focus; follow pans w/ dwell")


def test_motion_builds_zoompan_and_cover_base() -> None:
    """A motion path emits a zoompan over a centered cover base (any aspect)."""
    t = layers.solve_transform(
        {"format": {"aspect": "landscape"}},
        {"motion": {"from": {"scale": 1.0, "cx": 0.5, "cy": 0.5},
                    "to": {"scale": 1.2, "cx": 0.5, "cy": 0.5},
                    "ease": "smooth", "dur_ms": 2000}})
    vf = C._transform_vf(C._cfg_for("preview", "landscape"), t)
    # landscape would normally letterbox, but motion forces a cover base + zoompan.
    assert "force_original_aspect_ratio=increase" in vf and "zoompan=" in vf, vf
    assert "pad=" not in vf, vf
    assert "|m" in C._transform_key(t)
    print("  OK  motion -> cover base + zoompan; cache key carries the path")


def test_motion_annotate_idempotent() -> None:
    """annotate bakes motion under the chosen style and clears it when back to static."""
    from app.services.l3 import framing

    doc = {"format": {"aspect": "portrait", "motion_style": "push_in", "motion_feel": "glide"},
           "timeline": [{"seg_id": "s1", "file_id": "f1", "in_ms": 0, "out_ms": 3000}]}
    framing.annotate_document(doc)
    assert "motion" in doc["timeline"][0]["transform"], doc
    # Switch back to static: the prior motion must be cleared.
    doc["format"]["motion_style"] = "static"
    framing.annotate_document(doc)
    assert "motion" not in (doc["timeline"][0].get("transform") or {}), doc
    print("  OK  annotate bakes motion; clears it when style returns to static")


def main() -> None:
    print("geometric framing (Phase 1) regression:")
    test_fit_policy_by_aspect()
    test_identity_matches_old_normalize()
    test_cover_and_overrides_build_canonical_chain()
    test_override_clamping()
    test_resolve_attaches_transform()
    print("geometric framing (Phase 2) regression:")
    test_focus_from_motion_centroid_only()
    test_focus_builds_quoted_crop_expr()
    print("geometric framing (Phase 3) regression:")
    test_motion_normalize_and_sample()
    test_motion_styles_build()
    test_motion_builds_zoompan_and_cover_base()
    test_motion_annotate_idempotent()
    print("ALL PASS")


if __name__ == "__main__":
    main()
