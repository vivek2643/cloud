"use client";

/**
 * Animated caption overlay (caption_style_mvp.plan.md #4): a plain DOM
 * overlay in the program monitor that reads `resolvedCaptions` (server-
 * baked, same "the backend already computed this" pattern as the grade
 * overlay) and renders the currently active caption event, frame-accurate
 * against the shared program clock (`progMs`) via `resolve-captions.ts`'s
 * pure interpolation -- no CSS animations/transitions, so scrubbing while
 * paused looks identical to playing through, and the DOM/canvas side of the
 * preview<->export parity contract is driven by the exact same math
 * `ass_export.py` bakes into ASS override tags.
 *
 * Deliberately just text + a positioned box -- no video-frame compositing --
 * so it costs nothing when captions are off: `resolvedCaptions` is simply
 * empty and this renders null.
 */
import { useEffect, useRef, useState } from "react";
import { useEditDocStore } from "@/stores/edit-doc-store";
import { useTransport } from "@/stores/transport-store";
import {
  activeCaptionEvent, eventOpacity, eventSlideOffsetPx,
  wordActiveReaderStyle, wordPopScale, wordRevealOpacity,
} from "@/lib/resolve-captions";
import type { CaptionWord, ResolvedCaptionEvent } from "@/lib/api";

// One fixed restrained outline width / subtle shadow (caption_style_mvp.
// plan.md #3) -- CSS px, not user-tunable. Mirrors ass_export.py's
// _OUTLINE_WIDTH/_SHADOW_DEPTH in spirit (different units, same "one fixed
// treatment, on/off only" contract).
const OUTLINE_WIDTH_PX = 1;

function outlineShadow(outline: string, shadow: string, outlineEnabled: boolean, shadowEnabled: boolean): string {
  const layers: string[] = [];
  if (outlineEnabled) {
    const w = OUTLINE_WIDTH_PX;
    layers.push(
      `-${w}px -${w}px 0 ${outline}`, `${w}px -${w}px 0 ${outline}`,
      `-${w}px ${w}px 0 ${outline}`, `${w}px ${w}px 0 ${outline}`,
    );
  }
  if (shadowEnabled) {
    layers.push(`2px 2px 3px ${shadow}`);
  }
  return layers.join(", ");
}

function Word({ word, ev, progMs }: { word: CaptionWord; ev: ResolvedCaptionEvent; progMs: number }) {
  const { anim, style } = ev;
  const colour = style.colour;
  const outlineShadowCss = outlineShadow(colour.outline, colour.shadow, colour.outline_enabled, colour.shadow_enabled);

  if (anim === "active_reader") {
    const { brightness } = wordActiveReaderStyle(word, progMs, anim);
    return (
      <span
        className="inline-block"
        style={{ color: colour.fill, textShadow: outlineShadowCss, filter: `brightness(${brightness})` }}
      >
        {word.text}&nbsp;
      </span>
    );
  }

  if (anim === "sequential_reveal") {
    const opacity = wordRevealOpacity(word, progMs, anim);
    return (
      <span className="inline-block" style={{ color: colour.fill, textShadow: outlineShadowCss, opacity }}>
        {word.text}&nbsp;
      </span>
    );
  }

  const scale = wordPopScale(word, progMs, anim);
  return (
    <span
      className="inline-block"
      style={{
        color: colour.fill,
        textShadow: outlineShadowCss,
        transform: scale !== 1 ? `scale(${scale})` : undefined,
        transformOrigin: "center bottom",
      }}
    >
      {word.text}&nbsp;
    </span>
  );
}

function CaptionEventView({
  ev, progMs, frameH,
}: { ev: ResolvedCaptionEvent; progMs: number; frameH: number }) {
  const opacity = eventOpacity(ev, progMs);
  if (opacity <= 0.001) return null;
  const slideOffset = eventSlideOffsetPx(ev, progMs);
  const [x, y, w, h] = ev.box;
  const font = ev.style.font;
  // Match the export metric exactly: ass_export.py sizes text at
  // canvas_h * size_pct (the style's own Size choice), so the preview uses
  // the on-screen frame height * the same factor -> identical caption
  // proportion in the monitor and the burn. Falls back to a viewport
  // estimate only until the frame is first measured.
  const sizePct = ev.style.size_pct;
  const fontSize = frameH > 0 ? `${frameH * sizePct}px` : `min(${(sizePct * 100).toFixed(1)}vw, 32px)`;

  return (
    <div
      className="pointer-events-none absolute flex items-center justify-center overflow-hidden"
      style={{
        left: `${x * 100}%`, top: `${y * 100}%`, width: `${w * 100}%`, height: `${h * 100}%`,
        opacity, transform: slideOffset ? `translateY(${slideOffset}px)` : undefined,
      }}
    >
      <div
        className="text-center leading-tight"
        style={{ fontFamily: font.fallback_stack, fontWeight: font.weight, fontSize }}
      >
        {ev.lines.map((line, i) => (
          <div key={i}>
            {line.words.map((word, wi) => (
              <Word key={wi} word={word} ev={ev} progMs={progMs} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

export function CaptionOverlay() {
  const resolvedCaptions = useEditDocStore((s) => s.resolvedCaptions);
  const progMs = useTransport((s) => s.progMs);
  const rootRef = useRef<HTMLDivElement>(null);
  const [frameH, setFrameH] = useState(0);

  const hasCaptions = !!resolvedCaptions && resolvedCaptions.length > 0;
  // Measure the full-frame root (not the caption box) so the font can size off
  // the frame height like the export does. A ResizeObserver keeps it correct
  // as the monitor resizes; re-attaches when captions toggle on.
  useEffect(() => {
    const el = rootRef.current;
    if (!el) return;
    const measure = () => setFrameH(el.clientHeight);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [hasCaptions]);

  if (!hasCaptions) return null;  // zero cost when captions are off
  const ev = activeCaptionEvent(resolvedCaptions, progMs);

  return (
    <div ref={rootRef} className="pointer-events-none absolute inset-0 overflow-hidden">
      {ev ? <CaptionEventView ev={ev} progMs={progMs} frameH={frameH} /> : null}
    </div>
  );
}
