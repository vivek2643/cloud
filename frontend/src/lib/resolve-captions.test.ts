/**
 * Tests for resolve-captions.ts (caption_style_mvp.plan.md #4): pure
 * function assertions, no DOM/React rendering. Mirrors the equivalent
 * constants/behavior asserted on the Python side in
 * backend/scripts/test_captions.py's ass_export.py tests -- this is the
 * frontend half of the "preview and export must use the same timing
 * semantics" contract; the two files can't literally share a test run
 * (different languages), so each independently pins its own side's
 * constants and behavior, with comments cross-referencing the other.
 */
import { describe, expect, it } from "vitest";
import {
  activeCaptionEvent, eventOpacity, eventSlideOffsetPx,
  wordActiveReaderStyle, wordPopScale, wordRevealOpacity,
} from "./resolve-captions";
import type { CaptionStyle, CaptionWord, ResolvedCaptionEvent } from "./api";

function makeStyle(overrides: Partial<CaptionStyle> = {}): CaptionStyle {
  return {
    style_id: "s", label: "S", tier: "standard",
    font: { font_id: "inter", family: "Inter", weight: 600, fallback_stack: "Inter, sans-serif" },
    colour: { colour_id: "white", fill: "#FFFFFF", outline_enabled: false, shadow_enabled: true, outline: "#000000", shadow: "#000000" },
    position: "lower_third", animation: "fade_up", case: "original", size: "regular",
    size_pct: 0.045, max_lines: 2, max_chars_per_line: 34,
    ...overrides,
  };
}

function makeEvent(overrides: Partial<ResolvedCaptionEvent> = {}): ResolvedCaptionEvent {
  const style = overrides.style ?? makeStyle();
  return {
    prog_start_ms: 0, prog_end_ms: 1000,
    lines: [{ words: [
      { text: "hello", t_in_ms: 0, t_out_ms: 300, emphasized: true },
      { text: "world", t_in_ms: 350, t_out_ms: 700, emphasized: false },
    ] }],
    box: [0.1, 0.7, 0.8, 0.2], style_ref: style.style_id, style, anim: style.animation,
    ...overrides,
  };
}

describe("activeCaptionEvent", () => {
  it("finds the event covering progMs", () => {
    const events = [makeEvent({ prog_start_ms: 0, prog_end_ms: 500 }), makeEvent({ prog_start_ms: 500, prog_end_ms: 1000 })];
    expect(activeCaptionEvent(events, 600)).toBe(events[1]);
  });

  it("returns null when nothing covers progMs", () => {
    const events = [makeEvent({ prog_start_ms: 0, prog_end_ms: 500 })];
    expect(activeCaptionEvent(events, 900)).toBeNull();
  });

  it("returns null for an empty/undefined event list", () => {
    expect(activeCaptionEvent(undefined, 0)).toBeNull();
    expect(activeCaptionEvent([], 0)).toBeNull();
  });
});

describe("eventOpacity -- entry transitions", () => {
  it("fade_up reaches full opacity within its entry window", () => {
    const ev = makeEvent({ style: makeStyle({ animation: "fade_up" }), anim: "fade_up", prog_start_ms: 0, prog_end_ms: 1000 });
    expect(eventOpacity(ev, 0)).toBe(0);
    expect(eventOpacity(ev, 180)).toBeCloseTo(1, 5);
  });

  it("active_reader/pop/sequential_reveal use the shared quick-entry fade", () => {
    for (const anim of ["active_reader", "pop", "sequential_reveal"] as const) {
      const ev = makeEvent({ style: makeStyle({ animation: anim }), anim, prog_start_ms: 0, prog_end_ms: 1000 });
      expect(eventOpacity(ev, 0)).toBe(0);
      expect(eventOpacity(ev, 500)).toBeCloseTo(1, 5);
    }
  });

  it("fades out near the event end", () => {
    const ev = makeEvent({ prog_start_ms: 0, prog_end_ms: 1000 });
    expect(eventOpacity(ev, 999)).toBeLessThan(1);
    expect(eventOpacity(ev, 1000)).toBe(0);
  });
});

describe("eventSlideOffsetPx -- Smooth Fade Up rise", () => {
  it("only fade_up moves the caption", () => {
    for (const anim of ["active_reader", "pop", "sequential_reveal"] as const) {
      const ev = makeEvent({ style: makeStyle({ animation: anim }), anim });
      expect(eventSlideOffsetPx(ev, 0)).toBe(0);
    }
  });

  it("settles to 0 offset by the end of the rise window", () => {
    const ev = makeEvent({ style: makeStyle({ animation: "fade_up" }), anim: "fade_up" });
    const start = eventSlideOffsetPx(ev, 0);
    const end = eventSlideOffsetPx(ev, 500);
    expect(start).toBeGreaterThan(0);
    expect(end).toBeCloseTo(0, 5);
  });

  it("rise stays within the plan's 10-15 reference pixel range", () => {
    const ev = makeEvent({ style: makeStyle({ animation: "fade_up" }), anim: "fade_up" });
    const initial = eventSlideOffsetPx(ev, 0);
    expect(initial).toBeGreaterThanOrEqual(10);
    expect(initial).toBeLessThanOrEqual(15);
  });
});

describe("wordActiveReaderStyle", () => {
  const word: CaptionWord = { text: "hi", t_in_ms: 100, t_out_ms: 400, emphasized: false };

  it("is dim outside the word's own [t_in,t_out) window", () => {
    const before = wordActiveReaderStyle(word, 50, "active_reader");
    const after = wordActiveReaderStyle(word, 500, "active_reader");
    expect(before.isCurrent).toBe(false);
    expect(after.isCurrent).toBe(false);
    expect(before.brightness).toBeLessThan(1);
  });

  it("is full brightness during the word's own window", () => {
    const during = wordActiveReaderStyle(word, 200, "active_reader");
    expect(during.isCurrent).toBe(true);
    expect(during.brightness).toBe(1);
  });

  it("is a no-op for other animations", () => {
    const out = wordActiveReaderStyle(word, 200, "fade_up");
    expect(out.isCurrent).toBe(false);
    expect(out.brightness).toBe(1);
  });
});

describe("wordPopScale -- 80% -> 105% -> 100%, never over 105%", () => {
  const word: CaptionWord = { text: "hi", t_in_ms: 100, t_out_ms: 400, emphasized: true };

  it("starts at 80%", () => {
    expect(wordPopScale(word, 100, "pop")).toBeCloseTo(0.80, 5);
  });

  it("never exceeds 105% at any sampled instant", () => {
    for (let t = 100; t <= 500; t += 5) {
      expect(wordPopScale(word, t, "pop")).toBeLessThanOrEqual(1.05 + 1e-9);
    }
  });

  it("settles at exactly 100% after both ramp phases", () => {
    expect(wordPopScale(word, 100 + 2 * 90, "pop")).toBeCloseTo(1.0, 5);
    expect(wordPopScale(word, 100 + 500, "pop")).toBeCloseTo(1.0, 5);
  });

  it("is a no-op for a non-emphasized word or a non-pop animation", () => {
    const plain: CaptionWord = { ...word, emphasized: false };
    expect(wordPopScale(plain, 150, "pop")).toBe(1);
    expect(wordPopScale(word, 150, "fade_up")).toBe(1);
  });
});

describe("wordRevealOpacity -- Sequential Reveal follows each word's own timestamp", () => {
  it("is invisible before the word's own t_in_ms", () => {
    const word: CaptionWord = { text: "hi", t_in_ms: 500, t_out_ms: 800, emphasized: false };
    expect(wordRevealOpacity(word, 499, "sequential_reveal")).toBe(0);
  });

  it("reaches full opacity within the fade window after t_in_ms", () => {
    const word: CaptionWord = { text: "hi", t_in_ms: 500, t_out_ms: 800, emphasized: false };
    expect(wordRevealOpacity(word, 620, "sequential_reveal")).toBeCloseTo(1, 5);
  });

  it("two words at different timestamps reveal independently", () => {
    const early: CaptionWord = { text: "a", t_in_ms: 0, t_out_ms: 200, emphasized: false };
    const late: CaptionWord = { text: "b", t_in_ms: 600, t_out_ms: 800, emphasized: false };
    const atT300 = { early: wordRevealOpacity(early, 300, "sequential_reveal"), late: wordRevealOpacity(late, 300, "sequential_reveal") };
    expect(atT300.early).toBeCloseTo(1, 5);
    expect(atT300.late).toBe(0);
  });

  it("is always fully opaque for other animations", () => {
    const word: CaptionWord = { text: "hi", t_in_ms: 500, t_out_ms: 800, emphasized: false };
    expect(wordRevealOpacity(word, 0, "fade_up")).toBe(1);
  });
});
