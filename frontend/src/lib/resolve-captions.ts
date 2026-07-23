/**
 * Client-side caption animation interpolation (caption_style_mvp.plan.md #4,
 * "preview and export must use the same timing semantics"): pure functions
 * of `progMs` against a `ResolvedCaptionEvent`, deliberately mirroring the
 * math `backend/app/services/l3/captions/ass_export.py` bakes into ASS
 * override tags (`\fad`/`\t`/`\move`/`\alpha`) constant-for-constant, so the
 * DOM overlay and the burned export read as the same effect. Frame-accurate
 * (driven directly by `progMs`, not a CSS animation/transition) so
 * scrubbing while paused looks identical to playing through.
 *
 * The four MVP animations (no per-style intensity/beat_sync dial -- those
 * are gone from the public catalog): Active Reader (word-by-word colour
 * highlight, caption always visible), Pop/Bounce (one emphasised word per
 * line, 80% -> 105% -> 100%), Smooth Fade Up (whole caption rises ~12
 * reference px while fading in), Sequential Reveal (each word fades in at
 * its own timestamp).
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

// Every entry transition completes within 200ms -- mirrors ass_export.py's
// _ACTIVE_READER_ENTRY_MS / _POP_RAMP_MS / _FADE_UP_DUR_MS /
// _SEQUENTIAL_WORD_FADE_MS exactly.
const ACTIVE_READER_ENTRY_MS = 120;
const ACTIVE_READER_DIM = 0.55;
const POP_START_SCALE = 0.80;
const POP_PEAK_SCALE = 1.05;
const POP_SETTLE_SCALE = 1.00;
const POP_RAMP_MS = 90;
const FADE_UP_RISE_PX = 12;
const FADE_UP_DUR_MS = 180;
const SEQUENTIAL_WORD_FADE_MS = 120;

/** Whole-event opacity: "fade_up" and the generic quick entry used by
 * "active_reader"/"pop"/"sequential_reveal" containers all fade in/out at
 * the event boundary; per-word animation (pop scale, sequential alpha,
 * active_reader highlight) is layered on top, computed separately below. */
export function eventOpacity(ev: ResolvedCaptionEvent, progMs: number): number {
  const preset = ev.anim;
  const elapsed = progMs - ev.prog_start_ms;
  const remaining = ev.prog_end_ms - progMs;
  const entryMs = preset === "fade_up" ? FADE_UP_DUR_MS
    : preset === "active_reader" ? ACTIVE_READER_ENTRY_MS
    : SEQUENTIAL_WORD_FADE_MS;
  const fadeIn = clamp01(elapsed / entryMs);
  const fadeOut = clamp01(remaining / 80);
  return Math.min(fadeIn, fadeOut);
}

/** "fade_up" entrance offset (px, positive = below rest position) -- mirrors
 * ass_export.py's `\move` rise amount and duration. */
export function eventSlideOffsetPx(ev: ResolvedCaptionEvent, progMs: number): number {
  if (ev.anim !== "fade_up") return 0;
  const t = clamp01((progMs - ev.prog_start_ms) / FADE_UP_DUR_MS);
  const eased = t * (2 - t); // ease-out, matches a gentle settle
  return FADE_UP_RISE_PX * (1 - eased);
}

export interface ActiveReaderStyle {
  isCurrent: boolean;
  brightness: number; // 0..1, 1 = full fill colour, ACTIVE_READER_DIM = dimmed
}

/** "active_reader": every word is dim by default; the currently-spoken word
 * (its own [t_in,t_out] window) swaps to the full fill colour -- mirrors
 * ass_export.py's per-word `\t(...,\c...)` swap (both directions quick,
 * <=100ms, well under the 200ms entry-transition budget). */
export function wordActiveReaderStyle(word: CaptionWord, progMs: number, anim: CaptionAnimation): ActiveReaderStyle {
  if (anim !== "active_reader") return { isCurrent: false, brightness: 1 };
  const isCurrent = progMs >= word.t_in_ms && progMs < word.t_out_ms;
  return { isCurrent, brightness: isCurrent ? 1 : ACTIVE_READER_DIM };
}

/** "pop": only the one emphasised word per line bounces, 80% -> 105% ->
 * 100%, two 90ms ramp phases (180ms total, under the 200ms budget) --
 * mirrors ass_export.py's `_line_text` pop window exactly. Scale only: the
 * MVP colour catalog has one fill colour per swatch, no separate emphasis
 * colour, so there's nothing to swap besides size. */
export function wordPopScale(word: CaptionWord, progMs: number, anim: CaptionAnimation): number {
  if (anim !== "pop" || !word.emphasized) return 1;
  const t = progMs - word.t_in_ms;
  const ramp1End = POP_RAMP_MS;
  const ramp2End = 2 * POP_RAMP_MS;
  if (t < 0) return POP_START_SCALE;
  if (t > ramp2End) return POP_SETTLE_SCALE;
  if (t <= ramp1End) {
    const frac = clamp01(t / POP_RAMP_MS);
    return POP_START_SCALE + (POP_PEAK_SCALE - POP_START_SCALE) * frac;
  }
  const frac = clamp01((t - ramp1End) / POP_RAMP_MS);
  return POP_PEAK_SCALE + (POP_SETTLE_SCALE - POP_PEAK_SCALE) * frac;
}

/** "sequential_reveal": each word fades in from its own t_in_ms -- mirrors
 * ass_export.py's per-word `\alpha` transform (120ms, under the 200ms
 * budget). Words not yet reached are fully transparent (never rendered
 * early); words already revealed stay at full opacity. */
export function wordRevealOpacity(word: CaptionWord, progMs: number, anim: CaptionAnimation): number {
  if (anim !== "sequential_reveal") return 1;
  return clamp01((progMs - word.t_in_ms) / SEQUENTIAL_WORD_FADE_MS);
}
