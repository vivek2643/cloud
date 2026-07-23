"use client";

/**
 * Captions panel (caption_style_mvp.plan.md): ONE five-card gallery --
 * Standard (permanent, hand-authored, always first) + 4 AI Picks generated
 * for this edit -- not a 30-knob inspector, not the old two-tier Suggested/
 * Standards split. Every tile rides the SAME shared representative frame +
 * real sample words, frozen at each style's animation "peak pose" rather
 * than live-animating (there's no playhead on a gallery tile). Selecting
 * any tile pre-fills the customization panel below it.
 *
 * Persistence mirrors `color-grade-view.tsx`'s applyLook/flushLook exactly
 * (debounced, serialized, saves against the LIVE head version).
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, RotateCcw, Shuffle, VolumeX } from "lucide-react";
import { useEditDocStore } from "@/stores/edit-doc-store";
import { useAuthStore } from "@/stores/auth-store";
import {
  getCaptionCatalog, getCaptionSuggestions, saveEditDocument,
  type CaptionAnimation, type CaptionCase, type CaptionCatalog, type CaptionPosition,
  type CaptionSize, type CaptionStyle, type CaptionWord, type EditCaptions,
} from "@/lib/api";

const ANIMATION_LABELS: Record<CaptionAnimation, string> = {
  active_reader: "Active Reader", pop: "Pop/Bounce", fade_up: "Smooth Fade Up", sequential_reveal: "Sequential Reveal",
};
const POSITION_LABELS: Record<CaptionPosition, string> = {
  lower_third: "Lower Third", center: "Dead Center", top: "Upper Third", bottom_dynamic: "Bottom Dynamic",
};
const CASE_LABELS: Record<CaptionCase, string> = {
  original: "Original", sentence: "Sentence", upper: "UPPERCASE", lower: "lowercase",
};
const SIZE_LABELS: Record<CaptionSize, string> = {
  small: "Small", regular: "Regular", large: "Large", xl: "Extra Large",
};
// Mirrors backend/app/services/l3/captions/placement.py's
// POSITION_VERTICAL_CENTER / _BAND_H -- so a tile's peak-pose box sits
// where the real caption will actually land, not an ad-hoc guess.
const POSITION_VERTICAL_CENTER: Record<CaptionPosition, number> = {
  lower_third: 0.81, center: 0.50, top: 0.135, bottom_dynamic: 0.715,
};
const POSITION_BAND_H = 0.22;

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="mb-2 text-[11px] font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>
      {children}
    </h3>
  );
}

function outlineShadow(outline: string, shadow: string, outlineEnabled: boolean, shadowEnabled: boolean): string {
  const layers: string[] = [];
  if (outlineEnabled) {
    const w = 1;
    layers.push(
      `-${w}px -${w}px 0 ${outline}`, `${w}px -${w}px 0 ${outline}`,
      `-${w}px ${w}px 0 ${outline}`, `${w}px ${w}px 0 ${outline}`,
    );
  }
  if (shadowEnabled) layers.push(`2px 2px 3px ${shadow}`);
  return layers.join(", ");
}

function positionBoxStyle(position: CaptionPosition): React.CSSProperties {
  const center = POSITION_VERTICAL_CENTER[position] ?? POSITION_VERTICAL_CENTER.lower_third;
  const top = Math.max(0, (center - POSITION_BAND_H / 2) * 100);
  return { left: "5%", width: "90%", top: `${top}%`, height: `${POSITION_BAND_H * 100}%` };
}

/** Frozen "peak pose": Active Reader shows its middle word highlighted,
 * Pop/Bounce shows the emphasised word at its peak scale, Smooth Fade Up /
 * Sequential Reveal show the settled/fully-revealed state -- a static
 * approximation of `resolve-captions.ts`'s live interpolation at its most
 * legible instant, not a real progMs sample. */
function PeakPoseCaption({ style, words }: { style: CaptionStyle; words: CaptionWord[] }) {
  const { colour, animation, font } = style;
  const shadow = outlineShadow(colour.outline, colour.shadow, colour.outline_enabled, colour.shadow_enabled);
  const emphIdx = words.findIndex((w) => w.emphasized);
  const peakIdx = emphIdx >= 0 ? emphIdx : Math.floor(words.length / 2);
  return (
    <div
      className="pointer-events-none text-center leading-tight"
      style={{ fontFamily: font.fallback_stack, fontWeight: font.weight, fontSize: "13px" }}
    >
      {words.map((w, i) => {
        if (animation === "active_reader") {
          const isCurrent = i === peakIdx;
          return (
            <span key={i} style={{ color: colour.fill, textShadow: shadow, filter: isCurrent ? undefined : "brightness(0.55)" }}>
              {w.text}&nbsp;
            </span>
          );
        }
        if (animation === "pop") {
          const isPeak = i === peakIdx;
          return (
            <span
              key={i}
              style={{ color: colour.fill, textShadow: shadow, display: "inline-block", transform: isPeak ? "scale(1.05)" : undefined }}
            >
              {w.text}&nbsp;
            </span>
          );
        }
        return <span key={i} style={{ color: colour.fill, textShadow: shadow }}>{w.text}&nbsp;</span>;
      })}
    </div>
  );
}

function Tile({
  style, words, selected, badge, onClick,
}: {
  style: CaptionStyle;
  words: CaptionWord[];
  selected: boolean;
  badge?: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="group flex flex-col overflow-hidden rounded-lg border text-left transition-colors"
      style={{ borderColor: selected ? "var(--accent)" : "var(--border)" }}
    >
      {/* Solid black backdrop for every tile: all caption fills are light, so
          black guarantees the style is visible regardless of colour (and it
          doesn't depend on a representative frame existing). */}
      <div className="relative aspect-video w-full overflow-hidden" style={{ background: "#000000" }}>
        {words.length > 0 && (
          <div className="absolute flex items-center justify-center" style={positionBoxStyle(style.position)}>
            <PeakPoseCaption style={style} words={words} />
          </div>
        )}
        {badge && (
          <span
            className="absolute left-1.5 top-1.5 rounded px-1.5 py-0.5 text-[9px] font-medium uppercase tracking-wide"
            style={{ background: "var(--accent-soft)", color: "var(--foreground)" }}
          >
            {badge}
          </span>
        )}
      </div>
      <div className="space-y-0.5 px-2 py-1.5">
        <div className="flex items-center justify-between gap-1">
          <span className="truncate text-[11px] font-medium">{style.label}</span>
          <span className="shrink-0 text-[9px]" style={{ color: "var(--muted)" }}>{ANIMATION_LABELS[style.animation]}</span>
        </div>
        {style.rationale && (
          <p className="truncate text-[10px]" style={{ color: "var(--muted)" }} title={style.rationale}>
            {style.rationale}
          </p>
        )}
      </div>
    </button>
  );
}

export function CaptionsView() {
  const threadId = useEditDocStore((s) => s.threadId);
  const captions = useEditDocStore((s) => s.captions);
  const setCaptions = useEditDocStore((s) => s.setCaptions);
  const commitCaptions = useEditDocStore((s) => s.commitCaptions);
  const token = useAuthStore((s) => s.session?.access_token);

  const [catalog, setCatalog] = useState<CaptionCatalog | null>(null);
  const [standard, setStandard] = useState<CaptionStyle | null>(null);
  const [suggestions, setSuggestions] = useState<CaptionStyle[]>([]);
  const [sampleWords, setSampleWords] = useState<CaptionWord[]>([]);
  const [reshuffleSeed, setReshuffleSeed] = useState(0);
  const [loadingSuggestions, setLoadingSuggestions] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (!token) return;
    getCaptionCatalog(token).then(setCatalog).catch(() => {});
  }, [token]);

  useEffect(() => {
    if (!token || !threadId) return;
    let cancelled = false;
    setLoadingSuggestions(true);
    getCaptionSuggestions(threadId, token, { reshuffleSeed })
      .then((res) => {
        if (cancelled) return;
        setStandard(res.standard);
        // Regeneration only ever replaces the 4 AI picks + shared frame/words
        // (never the applied style -- that lives in `captions`, untouched here).
        setSuggestions(res.suggestions);
        setSampleWords(res.sample_words);
      })
      .catch((e) => !cancelled && setError(e instanceof Error ? e.message : "Could not load suggestions."))
      .finally(() => !cancelled && setLoadingSuggestions(false));
    return () => {
      cancelled = true;
    };
  }, [threadId, token, reshuffleSeed]);

  // --- Persistence: debounced + serialized, same contract as color-grade-view.tsx ---
  const pendingRef = useRef<EditCaptions | undefined>(undefined);
  const savingRef = useRef(false);
  const flushTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const flushCaptions = useCallback(async () => {
    if (savingRef.current) return;
    const st = useEditDocStore.getState();
    if (!st.threadId || !token) return;
    const next = pendingRef.current;
    savingRef.current = true;
    setSaving(true);
    setError(null);
    try {
      const res = await saveEditDocument(
        st.threadId,
        { base_version: st.baseVersion, timeline: st.timeline, operations: st.operations, captions: next },
        token
      );
      commitCaptions(res.version, res.document);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save captions.");
    } finally {
      savingRef.current = false;
      setSaving(false);
      if (pendingRef.current !== next) void flushCaptions();
    }
  }, [token, commitCaptions]);

  const applyCaptions = useCallback(
    (next: EditCaptions | undefined, immediate = false) => {
      setCaptions(next);
      pendingRef.current = next;
      if (flushTimer.current) clearTimeout(flushTimer.current);
      if (immediate) void flushCaptions();
      else flushTimer.current = setTimeout(() => void flushCaptions(), 300);
    },
    [setCaptions, flushCaptions]
  );

  useEffect(
    () => () => {
      if (flushTimer.current) {
        clearTimeout(flushTimer.current);
        void flushCaptions();
      }
    },
    [flushCaptions]
  );

  function selectStyle(style: CaptionStyle) {
    applyCaptions({ style_id: style.style_id, enabled: true, base_style: style, overrides: null }, true);
  }

  function turnOff() {
    applyCaptions(captions ? { ...captions, enabled: false } : { enabled: false }, true);
  }

  function resetToSelected() {
    if (!captions?.base_style) return;
    applyCaptions({ ...captions, overrides: null }, true);
  }

  function applyOverride(
    patch: Partial<{
      font_id: string; colour_id: string; outline_enabled: boolean; shadow_enabled: boolean;
      position: CaptionPosition; animation: CaptionAnimation; case: CaptionCase; size: CaptionSize;
    }>,
    immediate = false
  ) {
    if (!captions?.base_style) return;
    applyCaptions({ ...captions, overrides: { ...(captions.overrides ?? {}), ...patch } }, immediate);
  }

  const selectedStyleId = captions?.enabled ? captions.style_id : null;
  const base = captions?.base_style;
  const ov = captions?.overrides;
  const effectiveFontId = ov?.font_id ?? base?.font.font_id;
  const effectiveColourId = ov?.colour_id ?? base?.colour.colour_id;
  const effectiveOutlineEnabled = ov?.outline_enabled ?? base?.colour.outline_enabled ?? false;
  const effectiveShadowEnabled = ov?.shadow_enabled ?? base?.colour.shadow_enabled ?? false;
  const effectivePosition = ov?.position ?? base?.position;
  const effectiveAnimation = ov?.animation ?? base?.animation;
  const effectiveCase = ov?.case ?? base?.case ?? "original";
  const effectiveSize = ov?.size ?? base?.size ?? "regular";
  const hasOverrides = !!ov && Object.keys(ov).length > 0;

  if (!threadId) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <p className="text-lg font-semibold">Captions</p>
        <p className="mt-1 max-w-sm text-sm" style={{ color: "var(--muted)" }}>
          Start an edit first (Drive or the AI panel) -- captions are placed against an edit we&apos;ve already perceived.
        </p>
      </div>
    );
  }

  const galleryTiles: { style: CaptionStyle; badge?: string }[] = [
    ...(standard ? [{ style: standard, badge: "Standard" }] : []),
    ...suggestions.map((s) => ({ style: s })),
  ];

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">Captions</p>
        {captions?.enabled && (
          <button
            onClick={turnOff}
            title="Turn off captions"
            className="flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-[11px] transition-colors"
            style={{ borderColor: "var(--border)", color: "var(--muted)" }}
          >
            <VolumeX size={12} /> Turn off
          </button>
        )}
      </div>

      <section>
        <div className="mb-2 flex items-center justify-between">
          <SectionLabel>Standard + AI Picks</SectionLabel>
          <button
            onClick={() => setReshuffleSeed((s) => s + 1)}
            disabled={loadingSuggestions}
            title="Regenerate the 4 AI picks (Standard and your applied style never change)"
            className="flex items-center gap-1 rounded p-1 text-[10px] transition-colors hover:bg-[var(--accent-soft)] disabled:opacity-40"
            style={{ color: "var(--muted)" }}
          >
            <Shuffle size={11} /> Regenerate
          </button>
        </div>
        {loadingSuggestions && galleryTiles.length === 0 ? (
          <div className="flex items-center gap-1.5 py-6 text-[11px]" style={{ color: "var(--muted)" }}>
            <Loader2 size={12} className="animate-spin" /> Generating suggestions…
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {galleryTiles.map(({ style, badge }) => (
              <Tile
                key={style.style_id}
                style={style}
                words={sampleWords}
                badge={badge}
                selected={selectedStyleId === style.style_id}
                onClick={() => selectStyle(style)}
              />
            ))}
          </div>
        )}
      </section>

      {base && catalog && (
        <section>
          <div className="mb-2 flex items-center justify-between">
            <SectionLabel>Customize</SectionLabel>
            <button
              onClick={resetToSelected}
              disabled={!hasOverrides}
              title="Reset to the selected card's original values"
              className="flex items-center gap-1 rounded p-1 text-[10px] transition-colors hover:bg-[var(--accent-soft)] disabled:opacity-40"
              style={{ color: "var(--muted)" }}
            >
              <RotateCcw size={11} /> Reset
            </button>
          </div>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
            <div>
              <SectionLabel>Font</SectionLabel>
              <select
                value={effectiveFontId}
                onChange={(e) => applyOverride({ font_id: e.target.value }, true)}
                className="w-full rounded-lg border bg-transparent px-2 py-1.5 text-[11px] outline-none"
                style={{ borderColor: "var(--border)" }}
              >
                {catalog.fonts.map((f) => (
                  <option key={f.font_id} value={f.font_id}>{f.family}</option>
                ))}
              </select>
            </div>
            <div>
              <SectionLabel>Colour</SectionLabel>
              <select
                value={effectiveColourId}
                onChange={(e) => applyOverride({ colour_id: e.target.value }, true)}
                className="w-full rounded-lg border bg-transparent px-2 py-1.5 text-[11px] outline-none"
                style={{ borderColor: "var(--border)" }}
              >
                {catalog.colours.map((c) => (
                  <option key={c.colour_id} value={c.colour_id}>{c.label}</option>
                ))}
              </select>
            </div>
            <div>
              <SectionLabel>Position</SectionLabel>
              <select
                value={effectivePosition}
                onChange={(e) => applyOverride({ position: e.target.value as CaptionPosition }, true)}
                className="w-full rounded-lg border bg-transparent px-2 py-1.5 text-[11px] outline-none"
                style={{ borderColor: "var(--border)" }}
              >
                {(Object.keys(POSITION_LABELS) as CaptionPosition[]).map((p) => (
                  <option key={p} value={p}>{POSITION_LABELS[p]}</option>
                ))}
              </select>
            </div>
            <div>
              <SectionLabel>Animation</SectionLabel>
              <select
                value={effectiveAnimation}
                onChange={(e) => applyOverride({ animation: e.target.value as CaptionAnimation }, true)}
                className="w-full rounded-lg border bg-transparent px-2 py-1.5 text-[11px] outline-none"
                style={{ borderColor: "var(--border)" }}
              >
                {(Object.keys(ANIMATION_LABELS) as CaptionAnimation[]).map((a) => (
                  <option key={a} value={a}>{ANIMATION_LABELS[a]}</option>
                ))}
              </select>
            </div>
            <div>
              <SectionLabel>Case</SectionLabel>
              <select
                value={effectiveCase}
                onChange={(e) => applyOverride({ case: e.target.value as CaptionCase }, true)}
                className="w-full rounded-lg border bg-transparent px-2 py-1.5 text-[11px] outline-none"
                style={{ borderColor: "var(--border)" }}
              >
                {(Object.keys(CASE_LABELS) as CaptionCase[]).map((c) => (
                  <option key={c} value={c}>{CASE_LABELS[c]}</option>
                ))}
              </select>
            </div>
            <div>
              <SectionLabel>Size</SectionLabel>
              <select
                value={effectiveSize}
                onChange={(e) => applyOverride({ size: e.target.value as CaptionSize }, true)}
                className="w-full rounded-lg border bg-transparent px-2 py-1.5 text-[11px] outline-none"
                style={{ borderColor: "var(--border)" }}
              >
                {(Object.keys(SIZE_LABELS) as CaptionSize[]).map((s) => (
                  <option key={s} value={s}>{SIZE_LABELS[s]}</option>
                ))}
              </select>
            </div>
            <label className="flex items-center gap-1.5 text-[11px]">
              <input
                type="checkbox"
                checked={effectiveOutlineEnabled}
                onChange={(e) => applyOverride({ outline_enabled: e.target.checked }, true)}
              />
              Outline
            </label>
            <label className="flex items-center gap-1.5 text-[11px]">
              <input
                type="checkbox"
                checked={effectiveShadowEnabled}
                onChange={(e) => applyOverride({ shadow_enabled: e.target.checked }, true)}
              />
              Shadow
            </label>
          </div>
        </section>
      )}

      {error && <p className="text-[11px]" style={{ color: "var(--danger)" }}>{error}</p>}
      {saving && (
        <p className="flex items-center gap-1.5 text-[11px]" style={{ color: "var(--muted)" }}>
          <Loader2 size={11} className="animate-spin" /> Saving…
        </p>
      )}
    </div>
  );
}
