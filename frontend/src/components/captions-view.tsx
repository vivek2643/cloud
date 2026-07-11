"use client";

/**
 * Captions panel (captions.plan.md): a two-tier gallery -- "Suggested"
 * (5 bundles generated for THIS edit, SS6) over "Standards" (the universal
 * catalog, SS7) -- not a 30-knob inspector. Every tile rides the SAME
 * shared representative frame + real sample words (SS1.5/SS1.6), frozen at
 * each style's animation "peak pose" (SS1.5) rather than live-animating.
 * Selecting any tile pre-fills the Refine section below it (SS1.7).
 *
 * Persistence mirrors `color-grade-view.tsx`'s applyLook/flushLook exactly
 * (debounced, serialized, saves against the LIVE head version) -- see that
 * file's comment for why a save-per-tick was the wrong call there; the same
 * reasoning applies to a dragged/typed override here.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, Shuffle, VolumeX } from "lucide-react";
import { useEditDocStore } from "@/stores/edit-doc-store";
import { useAuthStore } from "@/stores/auth-store";
import {
  getCaptionCatalog, getCaptionSuggestions, saveEditDocument,
  type CaptionCatalog, type CaptionStyle, type CaptionWord,
  type CaptionRepresentativeFrame, type EditCaptions,
} from "@/lib/api";

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="mb-2 text-[11px] font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>
      {children}
    </h3>
  );
}

function motionLabel(style: CaptionStyle): string {
  const names: Record<string, string> = { fade: "Subtle-fade", pop: "Pop", karaoke: "Karaoke", slide: "Slide-up" };
  const base = names[style.animation.preset] ?? style.animation.preset;
  return style.animation.beat_sync ? `${base} · Beat-synced` : base;
}

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
  return [
    `-${w}px -${w}px 0 ${outline}`, `${w}px -${w}px 0 ${outline}`,
    `-${w}px ${w}px 0 ${outline}`, `${w}px ${w}px 0 ${outline}`,
    `2px 2px 3px ${shadow}`,
  ].join(", ");
}

/** Frozen "peak pose" (SS1.5): karaoke half-filled at the middle word, pop's
 * emphasized word at full scale, fade/slide just shown settled -- a static
 * approximation of `resolve-captions.ts`'s live interpolation at its most
 * legible instant, not a real progMs sample (there's no playhead on a tile). */
function PeakPoseCaption({ style, words }: { style: CaptionStyle; words: CaptionWord[] }) {
  const { colour, animation, font } = style;
  const shadow = outlineShadow(colour.outline, colour.shadow, !!colour.strong_outline);
  const peakIdx = Math.floor(words.length / 2);
  const emphIdx = words.findIndex((w) => w.emphasized);
  return (
    <div
      className="pointer-events-none text-center leading-tight"
      style={{
        fontFamily: font.fallback_stack, fontWeight: font.weight,
        letterSpacing: `${font.tracking}em`, fontSize: "13px",
        background: colour.box || undefined,
        padding: colour.box ? "0.15em 0.4em" : undefined,
        borderRadius: colour.box ? 4 : undefined,
      }}
    >
      {words.map((w, i) => {
        if (animation.preset === "karaoke") {
          const frac = i < peakIdx ? 1 : i === peakIdx ? 0.5 : 0;
          const secondary = dimHex(colour.fill, 0.45);
          return (
            <span
              key={i}
              style={{
                backgroundImage: `linear-gradient(90deg, ${colour.fill} ${frac * 100}%, ${secondary} ${frac * 100}%)`,
                WebkitBackgroundClip: "text", backgroundClip: "text", color: "transparent",
                textShadow: shadow,
              }}
            >
              {w.text}&nbsp;
            </span>
          );
        }
        const isPeakEmphasis = animation.preset === "pop" && i === (emphIdx >= 0 ? emphIdx : peakIdx);
        return (
          <span
            key={i}
            style={{
              color: isPeakEmphasis ? colour.emphasis_fill : colour.fill,
              textShadow: shadow,
              display: "inline-block",
              transform: isPeakEmphasis ? "scale(1.3)" : undefined,
            }}
          >
            {w.text}&nbsp;
          </span>
        );
      })}
    </div>
  );
}

function Tile({
  style, frame, words, selected, onClick,
}: {
  style: CaptionStyle;
  frame: CaptionRepresentativeFrame | null;
  words: CaptionWord[];
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="group flex flex-col overflow-hidden rounded-lg border text-left transition-colors"
      style={{ borderColor: selected ? "var(--accent)" : "var(--border)" }}
    >
      <div className="relative aspect-video w-full overflow-hidden" style={{ background: "var(--sidebar)" }}>
        {frame?.url && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={frame.url} alt="" className="absolute inset-0 h-full w-full object-cover" />
        )}
        {words.length > 0 && (
          <div
            className="absolute flex items-center justify-center"
            style={
              style.placement.anchor === "top"
                ? { left: "5%", top: "4%", width: "90%", height: "28%" }
                : style.placement.anchor === "center"
                ? { left: "5%", top: "38%", width: "90%", height: "28%" }
                : { left: "5%", bottom: "6%", width: "90%", height: "28%" }
            }
          >
            <PeakPoseCaption style={style} words={words} />
          </div>
        )}
      </div>
      <div className="space-y-0.5 px-2 py-1.5">
        <div className="flex items-center justify-between gap-1">
          <span className="truncate text-[11px] font-medium">{style.label}</span>
          <span className="shrink-0 text-[9px]" style={{ color: "var(--muted)" }}>{motionLabel(style)}</span>
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
  const [suggestions, setSuggestions] = useState<CaptionStyle[]>([]);
  const [repFrame, setRepFrame] = useState<CaptionRepresentativeFrame | null>(null);
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
        setSuggestions(res.suggestions);
        setRepFrame(res.representative_frame);
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

  function applyOverride(patch: Record<string, unknown>, immediate = false) {
    if (!captions?.base_style) return;
    applyCaptions({ ...captions, overrides: { ...(captions.overrides ?? {}), ...patch } }, immediate);
  }

  const selectedStyleId = captions?.enabled ? captions.style_id : null;
  const effectiveFontId =
    (captions?.overrides as { font_id?: string } | undefined)?.font_id ?? captions?.base_style?.font.font_id;
  const effectiveCase =
    (captions?.overrides as { case?: string } | undefined)?.case ?? captions?.base_style?.font.case ?? "as-is";
  const effectiveAnim =
    (captions?.overrides as { animation?: { preset?: string } } | undefined)?.animation?.preset
    ?? captions?.base_style?.animation.preset;
  const effectiveAnchor =
    (captions?.overrides as { placement?: { anchor?: string } } | undefined)?.placement?.anchor
    ?? captions?.base_style?.placement.anchor;
  const effectiveColourSource =
    (captions?.overrides as { colour?: { source?: string } } | undefined)?.colour?.source
    ?? captions?.base_style?.colour.source;

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
          <SectionLabel>Suggested for this edit</SectionLabel>
          <button
            onClick={() => setReshuffleSeed((s) => s + 1)}
            disabled={loadingSuggestions}
            title="Reshuffle (keeps Auto)"
            className="flex items-center gap-1 rounded p-1 text-[10px] transition-colors hover:bg-[var(--accent-soft)] disabled:opacity-40"
            style={{ color: "var(--muted)" }}
          >
            <Shuffle size={11} /> Reshuffle
          </button>
        </div>
        {loadingSuggestions ? (
          <div className="flex items-center gap-1.5 py-6 text-[11px]" style={{ color: "var(--muted)" }}>
            <Loader2 size={12} className="animate-spin" /> Generating suggestions…
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {suggestions.map((s) => (
              <Tile
                key={s.style_id}
                style={s}
                frame={repFrame}
                words={sampleWords}
                selected={selectedStyleId === s.style_id}
                onClick={() => selectStyle(s)}
              />
            ))}
          </div>
        )}
      </section>

      {catalog && (
        <section>
          <SectionLabel>Standards</SectionLabel>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {catalog.standards.map((s) => (
              <Tile
                key={s.style_id}
                style={s}
                frame={repFrame}
                words={sampleWords}
                selected={selectedStyleId === s.style_id}
                onClick={() => selectStyle(s)}
              />
            ))}
          </div>
        </section>
      )}

      {captions?.base_style && catalog && (
        <section className="grid grid-cols-2 gap-3 sm:grid-cols-3">
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
            <SectionLabel>Case</SectionLabel>
            <select
              value={effectiveCase}
              onChange={(e) => applyOverride({ case: e.target.value }, true)}
              className="w-full rounded-lg border bg-transparent px-2 py-1.5 text-[11px] outline-none"
              style={{ borderColor: "var(--border)" }}
            >
              <option value="as-is">As-is</option>
              <option value="upper">UPPERCASE</option>
            </select>
          </div>
          <div>
            <SectionLabel>Animation</SectionLabel>
            <select
              value={effectiveAnim}
              onChange={(e) => applyOverride({ animation: { preset: e.target.value } }, true)}
              className="w-full rounded-lg border bg-transparent px-2 py-1.5 text-[11px] outline-none"
              style={{ borderColor: "var(--border)" }}
            >
              <option value="fade">Subtle-fade</option>
              <option value="pop">Pop</option>
              <option value="karaoke">Karaoke</option>
              <option value="slide">Slide-up</option>
            </select>
          </div>
          <div>
            <SectionLabel>Placement</SectionLabel>
            <select
              value={effectiveAnchor}
              onChange={(e) => applyOverride({ placement: { anchor: e.target.value } }, true)}
              className="w-full rounded-lg border bg-transparent px-2 py-1.5 text-[11px] outline-none"
              style={{ borderColor: "var(--border)" }}
            >
              <option value="dynamic">Dynamic (safe-zone)</option>
              <option value="lower_third">Lower-third</option>
              <option value="center">Centered</option>
              <option value="top">Top</option>
            </select>
          </div>
          <div>
            <SectionLabel>Colour</SectionLabel>
            <select
              value={effectiveColourSource}
              onChange={(e) => applyOverride({ colour: { source: e.target.value } }, true)}
              className="w-full rounded-lg border bg-transparent px-2 py-1.5 text-[11px] outline-none"
              style={{ borderColor: "var(--border)" }}
            >
              <option value="white">White + soft shadow</option>
              <option value="black_box">Black-box</option>
              <option value="match_grade">Match grade</option>
              <option value="palette_accent">Accent from palette</option>
              <option value="high_contrast">High-contrast pop</option>
            </select>
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
