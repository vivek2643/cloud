/**
 * Timeline VIEW state — zoom/scroll, active tool, snapping, and per-track UI
 * meta (mute/solo/lock/height). None of this is persisted to the edit
 * document (it's not render truth); it's ephemeral editing-surface state,
 * reset per browser session. The one exception is audio-track mute, which
 * already maps onto `gain_db` on the document via the existing store path —
 * this module only tracks the VIDEO-track view-only mute + solo/lock/height.
 */
import { create } from "zustand";

export type TimelineTool = "select" | "blade" | "slip" | "slide";

export const MIN_PX_PER_SEC = 2;
export const MAX_PX_PER_SEC = 400;
const DEFAULT_PX_PER_SEC = 60;

export interface TrackMeta {
  solo?: boolean;
  lock?: boolean;
  /** View-only mute for a VIDEO track (audio mute already maps to gain_db). */
  mute?: boolean;
  heightPx?: number;
}

function clampZoom(v: number): number {
  return Math.max(MIN_PX_PER_SEC, Math.min(MAX_PX_PER_SEC, v));
}

/** A copied clip's descriptor — source + kind + a track hint for paste, per
 * timeline_nle.plan.md P0.3 ("source file + src in/out + kind + track hint"). */
export interface ClipboardEntry {
  kind: "video" | "audio";
  sourceFileId: string;
  srcInMs: number;
  srcOutMs: number;
  trackHint?: { z?: number; role?: string };
}

interface TimelineViewState {
  pxPerSec: number;
  scrollLeftPx: number;
  tool: TimelineTool;
  snapEnabled: boolean;
  /** Ms currently highlighted as the active snap target during a drag (null = none). */
  snapGuideMs: number | null;
  trackMeta: Record<string, TrackMeta>;
  /** Timeline in/out marks (program range) — P1 insert/overwrite + loop range. */
  inMarkMs: number | null;
  outMarkMs: number | null;
  /** Session-only markers (P2 will add doc persistence + a marker list panel). */
  markers: number[];
  /** Editor-local clipboard for copy/cut/paste/duplicate (P0.3). */
  clipboard: ClipboardEntry[];
  /** Insert (ripple, the default) vs overwrite for new drops/pastes (P1.2). */
  insertMode: "insert" | "overwrite";
  /** Per-spine-segment A/V link (P1.5), keyed by seg_id; absent = linked
   * (the default). The video/dialogue clips share one segment's in/out — see
   * timeline-editor.tsx's selectClip for what "unlinked" actually changes. */
  unlinkedSegIds: Record<string, boolean>;

  setZoom: (pxPerSec: number) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  zoomToFit: (viewportPx: number, totalMs: number) => void;
  setScrollLeft: (px: number) => void;
  setTool: (tool: TimelineTool) => void;
  toggleSnap: () => void;
  setSnapGuide: (ms: number | null) => void;
  setTrackMeta: (trackId: string, patch: Partial<TrackMeta>) => void;
  setInMark: (ms: number | null) => void;
  setOutMark: (ms: number | null) => void;
  addMarker: (ms: number) => void;
  removeMarker: (ms: number) => void;
  setClipboard: (entries: ClipboardEntry[]) => void;
  toggleInsertMode: () => void;
  toggleLinked: (segId: string) => void;
}

export const useTimelineView = create<TimelineViewState>((set) => ({
  pxPerSec: DEFAULT_PX_PER_SEC,
  scrollLeftPx: 0,
  tool: "select",
  snapEnabled: true,
  snapGuideMs: null,
  trackMeta: {},
  inMarkMs: null,
  outMarkMs: null,
  markers: [],
  clipboard: [],
  insertMode: "insert",
  unlinkedSegIds: {},

  setZoom: (pxPerSec) => set({ pxPerSec: clampZoom(pxPerSec) }),
  zoomIn: () => set((s) => ({ pxPerSec: clampZoom(s.pxPerSec * 1.4) })),
  zoomOut: () => set((s) => ({ pxPerSec: clampZoom(s.pxPerSec / 1.4) })),
  zoomToFit: (viewportPx, totalMs) => {
    if (totalMs <= 0 || viewportPx <= 0) return;
    const fitPxPerSec = (viewportPx / totalMs) * 1000;
    set({ pxPerSec: clampZoom(fitPxPerSec), scrollLeftPx: 0 });
  },
  setScrollLeft: (px) => set({ scrollLeftPx: Math.max(0, px) }),
  setTool: (tool) => set({ tool }),
  toggleSnap: () => set((s) => ({ snapEnabled: !s.snapEnabled })),
  setSnapGuide: (ms) => set({ snapGuideMs: ms }),
  setTrackMeta: (trackId, patch) =>
    set((s) => ({
      trackMeta: { ...s.trackMeta, [trackId]: { ...s.trackMeta[trackId], ...patch } },
    })),
  setInMark: (ms) =>
    set((s) => {
      const outMs = s.outMarkMs;
      if (ms != null && outMs != null && ms >= outMs) return { inMarkMs: ms, outMarkMs: null };
      return { inMarkMs: ms };
    }),
  setOutMark: (ms) =>
    set((s) => {
      const inMs = s.inMarkMs;
      if (ms != null && inMs != null && ms <= inMs) return { outMarkMs: ms, inMarkMs: null };
      return { outMarkMs: ms };
    }),
  addMarker: (ms) =>
    set((s) => {
      const rounded = Math.round(ms);
      if (s.markers.includes(rounded)) return {};
      return { markers: [...s.markers, rounded].sort((a, b) => a - b) };
    }),
  removeMarker: (ms) => set((s) => ({ markers: s.markers.filter((m) => m !== ms) })),
  setClipboard: (entries) => set({ clipboard: entries }),
  toggleInsertMode: () => set((s) => ({ insertMode: s.insertMode === "insert" ? "overwrite" : "insert" })),
  toggleLinked: (segId) =>
    set((s) => ({
      unlinkedSegIds: { ...s.unlinkedSegIds, [segId]: !s.unlinkedSegIds[segId] },
    })),
}));
