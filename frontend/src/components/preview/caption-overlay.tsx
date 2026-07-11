"use client";

/**
 * Animated caption overlay (captions.plan.md SS4/SS14): a plain DOM overlay
 * in the program monitor that reads `resolvedCaptions` (server-baked, same
 * "the backend already computed this" pattern as the grade overlay) and
 * renders the currently active caption event, frame-accurate against the
 * shared program clock (`progMs`) via `resolve-captions.ts`'s pure
 * interpolation -- no CSS animations/transitions, so scrubbing while paused
 * looks identical to playing through, and the DOM/canvas side of the
 * preview↔export parity contract (SS4) is driven by the exact same math
 * `ass_export.py` bakes into ASS override tags.
 *
 * Deliberately just text + a positioned box -- no video-frame compositing --
 * so it costs nothing when captions are off (SS1.3): `resolvedCaptions` is
 * simply empty and this renders null.
 */
import { useEffect, useRef, useState } from "react";
import { useEditDocStore } from "@/stores/edit-doc-store";
import { useTransport } from "@/stores/transport-store";
import {
  activeCaptionEvent, eventOpacity, eventSlideOffsetPx, wordKaraokeFrac, wordPopStyle,
} from "@/lib/resolve-captions";
import type { CaptionWord, ResolvedCaptionEvent } from "@/lib/api";

function dimHex(hex: string, factor: number): string {
  const h = hex.replace("#", "");
  if (h.length !== 6) return hex;
  const r = Math.round(parseInt(h.slice(0, 2), 16) * factor);
  const g = Math.round(parseInt(h.slice(2, 4), 16) * factor);
  const b = Math.round(parseInt(h.slice(4, 6), 16) * factor);
  const toHex = (v: number) => v.toString(16).padStart(2, "0");
  return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
}

function outlineShadow(outline: string, shadow: string, strong: boolean): string {
  const w = strong ? 2 : 1;
  const layers = [
    `-${w}px -${w}px 0 ${outline}`, `${w}px -${w}px 0 ${outline}`,
    `-${w}px ${w}px 0 ${outline}`, `${w}px ${w}px 0 ${outline}`,
  ];
  layers.push(`2px 2px 3px ${shadow}`);
  return layers.join(", ");
}

function Word({ word, ev, progMs }: { word: CaptionWord; ev: ResolvedCaptionEvent; progMs: number }) {
  const { anim, style } = ev;
  const colour = style.colour;
  const outlineShadowCss = outlineShadow(colour.outline, colour.shadow, !!colour.strong_outline);

  if (anim.preset === "karaoke") {
    const frac = wordKaraokeFrac(word, progMs);
    const secondary = dimHex(colour.fill, 0.45);
    const pct = Math.round(frac * 100);
    return (
      <span
        className="inline-block"
        style={{
          backgroundImage: `linear-gradient(90deg, ${colour.fill} ${pct}%, ${secondary} ${pct}%)`,
          WebkitBackgroundClip: "text",
          backgroundClip: "text",
          color: "transparent",
          textShadow: outlineShadowCss,
        }}
      >
        {word.text}&nbsp;
      </span>
    );
  }

  const pop = wordPopStyle(word, progMs, anim);
  const colourHex = pop.useEmphasisColour ? colour.emphasis_fill : colour.fill;
  return (
    <span
      className="inline-block"
      style={{
        color: colourHex,
        textShadow: outlineShadowCss,
        transform: pop.scale !== 1 ? `scale(${pop.scale})` : undefined,
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
  const box = ev.style.colour.box;
  // Match the export metric exactly: ass_export.py sizes text at
  // canvas_h * 0.045, so the preview uses the on-screen frame height * the
  // same factor -> identical caption proportion in the monitor and the burn.
  // Falls back to a viewport estimate only until the frame is first measured.
  const fontSize = frameH > 0 ? `${frameH * 0.045}px` : "min(4.2vw, 32px)";

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
        style={{
          fontFamily: font.fallback_stack,
          fontWeight: font.weight,
          letterSpacing: `${font.tracking}em`,
          fontSize,
          background: box || undefined,
          padding: box ? "0.15em 0.4em" : undefined,
          borderRadius: box ? 4 : undefined,
        }}
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

  if (!hasCaptions) return null;  // zero cost when captions are off (SS1.3)
  const ev = activeCaptionEvent(resolvedCaptions, progMs);

  return (
    <div ref={rootRef} className="pointer-events-none absolute inset-0 overflow-hidden">
      {ev ? <CaptionEventView ev={ev} progMs={progMs} frameH={frameH} /> : null}
    </div>
  );
}
