/**
 * Client-side caption animation interpolation (captions.plan.md SS14, SS4
 * "preview and export must render the same track"): pure functions of
 * `progMs` against a `ResolvedCaptionEvent`, deliberately mirroring the math
 * `backend/app/services/l3/captions/ass_export.py` bakes into ASS override
 * tags (`\fad`/`\t`/`\kf`/`\move`) constant-for-constant, so the DOM overlay
 * and the burned export read as the same effect. Frame-accurate (driven
 * directly by `progMs`, not a CSS animation/transition) so scrubbing while
 * paused looks identical to playing through.
 */
import type { CaptionAnimation, ResolvedCaptionEvent, CaptionWord } from "./api";

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

export function activeCaptionEvent(
  events: ResolvedCaptionEvent[] | undefined, progMs: number
): ResolvedCaptionEvent | null {
  if (!events || events.length === 0) return null;
  for (const ev of events) {
    if (progMs >= ev.prog_start_ms && progMs < ev.prog_end_ms) return ev;
  }
  return null;
}

/** `\fad(in,out)` -- mirrors ass_export.py's `_event_tags` fade duration. */
function fadeMs(intensity: number): number {
  return Math.max(80, Math.round(180 * (0.4 + intensity)));
}

/** Whole-event opacity: only "fade" and "slide" animate the container (pop/
 * karaoke animate per-word instead, same split as `_event_tags` vs
 * `_line_text` in ass_export.py). */
export function eventOpacity(ev: ResolvedCaptionEvent, progMs: number): number {
  const { preset, intensity } = ev.anim;
  const elapsed = progMs - ev.prog_start_ms;
  const remaining = ev.prog_end_ms - progMs;
  if (preset === "fade") {
    const fad = fadeMs(intensity);
    return Math.min(clamp01(elapsed / fad), clamp01(remaining / fad));
  }
  if (preset === "slide") {
    const dur = 220;
    const fadeIn = clamp01(elapsed / dur);
    const fadeOut = clamp01(remaining / 80);
    return Math.min(fadeIn, fadeOut);
  }
  return 1;
}

/** Slide entrance offset (px, positive = below rest position) -- mirrors
 * ass_export.py's `\move` rise amount and duration. */
export function eventSlideOffsetPx(ev: ResolvedCaptionEvent, progMs: number): number {
  if (ev.anim.preset !== "slide") return 0;
  const rise = 40 + 60 * ev.anim.intensity;
  const dur = 220;
  const t = clamp01((progMs - ev.prog_start_ms) / dur);
  const eased = t * (2 - t); // ease-out, matches a gentle settle
  return rise * (1 - eased);
}

/** Karaoke fill fraction (0..1) for one word at `progMs` -- mirrors
 * ass_export.py's per-word `\kf` duration (the word's own [t_in,t_out]). */
export function wordKaraokeFrac(word: CaptionWord, progMs: number): number {
  return clamp01((progMs - word.t_in_ms) / Math.max(1, word.t_out_ms - word.t_in_ms));
}

export interface PopStyle {
  scale: number;
  useEmphasisColour: boolean;
}

/** Pop/scale emphasis for one word -- mirrors ass_export.py's `_line_text`
 * pop window exactly (same ramp-up/ramp-down duration and scale formula). */
export function wordPopStyle(word: CaptionWord, progMs: number, anim: CaptionAnimation): PopStyle {
  if (anim.preset !== "pop" || !word.emphasized) return { scale: 1, useEmphasisColour: false };
  const popDur = Math.min(220, Math.max(80, word.t_out_ms - word.t_in_ms));
  const t = progMs - word.t_in_ms;
  if (t < 0 || t > 2 * popDur) return { scale: 1, useEmphasisColour: false };
  const maxScale = 1 + 0.35 * anim.intensity;
  const scale = t < popDur
    ? 1 + (maxScale - 1) * clamp01(t / popDur)
    : maxScale - (maxScale - 1) * clamp01((t - popDur) / popDur);
  return { scale, useEmphasisColour: true };
}
