/**
 * Timeline VIEW state — zoom/scroll, snapping, and per-track mute. None of
 * this is persisted to the edit document (it's not render truth); it's
 * ephemeral editing-surface state, reset per browser session. Audio-track
 * mute already maps onto `gain_db` on the document via the existing store
 * path — this module only tracks the VIDEO-track view-only mute.
 */
import { create } from "zustand";

export const MIN_PX_PER_SEC = 2;
export const MAX_PX_PER_SEC = 400;
const DEFAULT_PX_PER_SEC = 60;

export interface TrackMeta {
  /** View-only mute for a VIDEO track (audio mute already maps to gain_db). */
  mute?: boolean;
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
  snapEnabled: boolean;
  /** Ms currently highlighted as the active snap target during a drag (null = none). */
  snapGuideMs: number | null;
  trackMeta: Record<string, TrackMeta>;
  /** Editor-local clipboard for copy/cut/paste/duplicate (P0.3). */
  clipboard: ClipboardEntry[];

  setZoom: (pxPerSec: number) => void;
  zoomIn: () => void;
  zoomOut: () => void;
  zoomToFit: (viewportPx: number, totalMs: number) => void;
  setScrollLeft: (px: number) => void;
  toggleSnap: () => void;
  setSnapGuide: (ms: number | null) => void;
  setTrackMeta: (trackId: string, patch: Partial<TrackMeta>) => void;
  setClipboard: (entries: ClipboardEntry[]) => void;
}

export const useTimelineView = create<TimelineViewState>((set) => ({
  pxPerSec: DEFAULT_PX_PER_SEC,
  scrollLeftPx: 0,
  snapEnabled: true,
  snapGuideMs: null,
  trackMeta: {},
  clipboard: [],

  setZoom: (pxPerSec) => set({ pxPerSec: clampZoom(pxPerSec) }),
  zoomIn: () => set((s) => ({ pxPerSec: clampZoom(s.pxPerSec * 1.4) })),
  zoomOut: () => set((s) => ({ pxPerSec: clampZoom(s.pxPerSec / 1.4) })),
  zoomToFit: (viewportPx, totalMs) => {
    if (totalMs <= 0 || viewportPx <= 0) return;
    const fitPxPerSec = (viewportPx / totalMs) * 1000;
    set({ pxPerSec: clampZoom(fitPxPerSec), scrollLeftPx: 0 });
  },
  setScrollLeft: (px) => set({ scrollLeftPx: Math.max(0, px) }),
  toggleSnap: () => set((s) => ({ snapEnabled: !s.snapEnabled })),
  setSnapGuide: (ms) => set({ snapGuideMs: ms }),
  setTrackMeta: (trackId, patch) =>
    set((s) => ({
      trackMeta: { ...s.trackMeta, [trackId]: { ...s.trackMeta[trackId], ...patch } },
    })),
  setClipboard: (entries) => set({ clipboard: entries }),
}));
