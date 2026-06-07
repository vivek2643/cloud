"""
Mechanical critic: deterministic quality checks over an assembled EDL.

The LLM critique pass (director._critique) judges a *text summary* of the cut --
useful for taste, useless for catching the mechanical defects that actually make
an edit feel broken: sub-second flicker cuts, blurry or black frames on screen,
the same shot reused five times, a 12s "30s" edit, a music montage with no music.

This module checks those against L1 telemetry (blur, brightness, motion) and the
EDL geometry, returning structured, machine-readable issues plus a guidance
string the re-plan pass can act on. Pure + repeatable: same EDL -> same verdict.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.l3.primitives.loader import FileAnalysis, ShotRow

# Tunables (absolute thresholds are intentionally conservative so we flag only
# clear defects, not borderline taste calls).
MIN_CLIP_MS = 500            # below this a clip barely registers
MICRO_CLIP_MS = 800          # "flicker" cut
BLUR_MIN_THRESHOLD = 18.0    # Laplacian variance; below = visibly soft/blurry
DARK_BRIGHTNESS = 12.0       # mean luminance 0-255; below = ~black frame
MAX_SHOT_REUSE = 3           # same shot appearing more than this = repetitive
DURATION_OVER_WARN = 1.25
DURATION_OVER_ERROR = 1.6
DURATION_UNDER_WARN = 0.5
MICRO_FRACTION_WARN = 0.5    # >half the clips are flicker cuts

# Styles where rapid cutting is intentional, so micro-cut density isn't a defect.
_FAST_STYLES = {"beat_sync", "trailer", "social_short"}
_MUSIC_STYLES = {"beat_sync", "highlight", "cinematic_broll", "trailer"}


@dataclass
class CriticIssue:
    code: str
    severity: str               # "error" | "warn"
    detail: str
    clips: List[int] = field(default_factory=list)


@dataclass
class CritiqueResult:
    ok: bool                    # True if no error-severity issues
    issues: List[CriticIssue] = field(default_factory=list)
    guidance: str = ""
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "mechanical": True,
            "issues": [
                {"code": i.code, "severity": i.severity, "detail": i.detail, "clips": i.clips}
                for i in self.issues
            ],
            "guidance": self.guidance,
            "stats": self.stats,
        }


def _shot_index(analyses: Dict[str, FileAnalysis]) -> Dict[str, ShotRow]:
    idx: Dict[str, ShotRow] = {}
    for fa in analyses.values():
        for s in fa.shots:
            idx[str(s.shot_id)] = s
    return idx


def critique_edl(
    edl: Dict[str, Any],
    analyses: Dict[str, FileAnalysis],
    *,
    target_s: Optional[int] = None,
    section_styles: Optional[List[str]] = None,
) -> CritiqueResult:
    """Run all checks. `section_styles` (per the composer's sections meta) lets us
    skip pacing checks for intentionally fast styles."""
    vt = list(edl.get("video_track") or [])
    at = list(edl.get("audio_track") or [])
    shots = _shot_index(analyses)
    issues: List[CriticIssue] = []

    total_ms = 0
    for c in vt + at:
        total_ms = max(total_ms, int(c.get("timeline_out_ms") or 0))

    if not vt:
        return CritiqueResult(
            ok=False,
            issues=[CriticIssue("empty_video", "error", "The cut has no video clips.")],
            guidance="The timeline has no video. Select usable units and rebuild.",
            stats={"total_ms": total_ms, "video_clips": 0, "audio_clips": len(at)},
        )

    durs = [int(c["timeline_out_ms"] - c["timeline_in_ms"]) for c in vt]
    styles = set(section_styles or [])
    fast = bool(styles & _FAST_STYLES)

    # 1) sub-minimum clips
    too_short = [i for i, d in enumerate(durs) if d < MIN_CLIP_MS]
    if too_short:
        sev = "error" if len(too_short) > len(durs) * 0.5 else "warn"
        issues.append(CriticIssue(
            "too_short_clips", sev,
            f"{len(too_short)} clip(s) are under {MIN_CLIP_MS}ms and barely register.",
            too_short,
        ))

    # 2) flicker pacing (skip for deliberately fast styles)
    if not fast:
        micro = [i for i, d in enumerate(durs) if d < MICRO_CLIP_MS]
        if micro and len(micro) >= MICRO_FRACTION_WARN * len(durs):
            issues.append(CriticIssue(
                "frantic_pacing", "warn",
                f"{len(micro)}/{len(durs)} clips are under {MICRO_CLIP_MS}ms -- the cut "
                "may feel frantic. Hold more clips longer.",
                micro,
            ))

    # 3) blurry / 4) dark frames on screen
    blurry: List[int] = []
    dark: List[int] = []
    for i, c in enumerate(vt):
        s = shots.get(str(c.get("shot_id")))
        if not s:
            continue
        if s.blur_min is not None and s.blur_min < BLUR_MIN_THRESHOLD:
            blurry.append(i)
        if s.brightness is not None and s.brightness < DARK_BRIGHTNESS:
            dark.append(i)
    if blurry:
        issues.append(CriticIssue(
            "blurry_clips", "warn",
            f"{len(blurry)} clip(s) sit on visibly soft/blurry shots; prefer sharper takes.",
            blurry,
        ))
    if dark:
        issues.append(CriticIssue(
            "dark_clips", "warn",
            f"{len(dark)} clip(s) are near-black; drop or replace them.",
            dark,
        ))

    # 5) shot over-reuse
    reuse = Counter(str(c.get("shot_id")) for c in vt if c.get("shot_id"))
    overused = {sid: n for sid, n in reuse.items() if n > MAX_SHOT_REUSE}
    if overused:
        worst = max(overused.values())
        issues.append(CriticIssue(
            "shot_overuse", "warn",
            f"{len(overused)} shot(s) are reused up to {worst}x; vary the footage.",
            [i for i, c in enumerate(vt) if str(c.get("shot_id")) in overused],
        ))

    # 6) true adjacent duplicates (same shot, overlapping source window)
    dup = _adjacent_duplicates(vt)
    if dup:
        issues.append(CriticIssue(
            "adjacent_duplicate", "warn",
            f"{len(dup)} adjacent clip pair(s) replay the same moment back-to-back.",
            dup,
        ))

    # 7) duration vs target
    if target_s:
        target_ms = target_s * 1000
        ratio = total_ms / target_ms if target_ms else 1.0
        if ratio >= DURATION_OVER_ERROR:
            issues.append(CriticIssue(
                "too_long", "error",
                f"The cut is {total_ms/1000:.0f}s vs a {target_s}s target ({ratio:.1f}x). Tighten hard.",
            ))
        elif ratio >= DURATION_OVER_WARN:
            issues.append(CriticIssue(
                "over_target", "warn",
                f"The cut is {total_ms/1000:.0f}s vs a {target_s}s target; trim toward the target.",
            ))
        elif ratio <= DURATION_UNDER_WARN:
            issues.append(CriticIssue(
                "under_target", "warn",
                f"The cut is only {total_ms/1000:.0f}s vs a {target_s}s target; add more.",
            ))

    # 8) music style with no audio bed
    if (styles & _MUSIC_STYLES) and not at and total_ms > 6000:
        issues.append(CriticIssue(
            "no_music_bed", "warn",
            "A music-driven style produced no audio track; add a music bed or change style.",
        ))

    ok = not any(i.severity == "error" for i in issues)
    stats = {
        "total_ms": total_ms,
        "video_clips": len(vt),
        "audio_clips": len(at),
        "avg_clip_ms": int(sum(durs) / len(durs)) if durs else 0,
        "shortest_clip_ms": min(durs) if durs else 0,
        "longest_clip_ms": max(durs) if durs else 0,
    }
    return CritiqueResult(ok=ok, issues=issues, guidance=_guidance(issues), stats=stats)


def _adjacent_duplicates(clips: List[Dict[str, Any]]) -> List[int]:
    out: List[int] = []
    prev: Optional[Dict[str, Any]] = None
    for i, c in enumerate(clips):
        if prev is not None and c.get("shot_id") and c.get("shot_id") == prev.get("shot_id"):
            a_in, a_out = int(prev["source_in_ms"]), int(prev["source_out_ms"])
            b_in, b_out = int(c["source_in_ms"]), int(c["source_out_ms"])
            overlap = min(a_out, b_out) - max(a_in, b_in)
            if overlap > 0.5 * min(a_out - a_in, b_out - b_in):
                out.append(i)
        prev = c
    return out


def _guidance(issues: List[CriticIssue]) -> str:
    if not issues:
        return "No mechanical defects detected."
    # Errors first, then warnings; concise imperative bullets for the re-plan.
    ordered = [i for i in issues if i.severity == "error"] + [i for i in issues if i.severity == "warn"]
    return "Fix these concrete issues:\n" + "\n".join(f"- {i.detail}" for i in ordered)
