/**
 * Unified playback transport — the single source of truth for program TIME.
 *
 * The program monitor (CompositePreview) and the timeline both read `progMs` /
 * `playing` from here, so they can never disagree about where the playhead is.
 * Time is quantised to the project frame grid (30 fps, matching the backend
 * renderer), which is what makes scrubbing/seeking frame-accurate.
 *
 * Division of labour:
 *  - This store OWNS the published time + play flag + seek intents.
 *  - The preview hosts the actual playback ENGINE (rAF clock + WebAudio mixer +
 *    double-buffered video). During play it advances its own continuous clock
 *    and calls `publish(ms)` once per frame. External seeks (timeline click,
 *    step buttons, ruler scrub) bump `seekSeq`; the engine watches it and jumps
 *    its clock + media to `seekTargetMs`.
 */
import { create } from "zustand";

export const PROJECT_FPS = 30;
export const FRAME_MS = 1000 / PROJECT_FPS;

/** Snap a millisecond value to the nearest project frame boundary. */
export function snapMs(ms: number): number {
  return Math.round(ms / FRAME_MS) * FRAME_MS;
}

/** Whole frame index for a millisecond value. */
export function msToFrame(ms: number): number {
  return Math.round(ms / FRAME_MS);
}

/** Frame-accurate timecode. Hours are dropped under an hour for compactness. */
export function formatTimecode(ms: number, fps = PROJECT_FPS): string {
  const totalFrames = Math.max(0, Math.round(ms / (1000 / fps)));
  const f = totalFrames % fps;
  const totalSec = Math.floor(totalFrames / fps);
  const s = totalSec % 60;
  const m = Math.floor(totalSec / 60) % 60;
  const h = Math.floor(totalSec / 3600);
  const pad = (n: number) => String(n).padStart(2, "0");
  const tail = `${pad(m)}:${pad(s)}:${pad(f)}`;
  return h > 0 ? `${pad(h)}:${tail}` : tail;
}

interface TransportState {
  fps: number;
  frameMs: number;
  /** Published program time (frame-snapped) read by every surface. */
  progMs: number;
  playing: boolean;
  durationMs: number;
  /** Bumped on every external seek so the engine can react imperatively. */
  seekSeq: number;
  seekTargetMs: number;

  setDuration: (ms: number) => void;
  setPlaying: (p: boolean) => void;
  togglePlaying: () => void;
  /** Engine → store: publish the current (already frame-snapped) time. */
  publish: (ms: number) => void;
  /** Any surface → engine: seek to an absolute ms (snapped + clamped). */
  seek: (ms: number) => void;
  /** Nudge by N frames (negative = back). */
  step: (frames: number) => void;
  reset: () => void;
}

export const useTransport = create<TransportState>((set, get) => ({
  fps: PROJECT_FPS,
  frameMs: FRAME_MS,
  progMs: 0,
  playing: false,
  durationMs: 0,
  seekSeq: 0,
  seekTargetMs: 0,

  setDuration: (ms) =>
    set((s) => {
      const durationMs = Math.max(0, ms);
      // Keep the playhead inside the new range.
      return durationMs < s.progMs ? { durationMs, progMs: durationMs } : { durationMs };
    }),

  setPlaying: (p) => set({ playing: p }),
  togglePlaying: () => set((s) => ({ playing: !s.playing })),

  publish: (ms) => set({ progMs: ms }),

  seek: (ms) =>
    set((s) => {
      const max = s.durationMs > 0 ? s.durationMs : snapMs(ms);
      const clamped = Math.max(0, Math.min(snapMs(ms), max));
      return { progMs: clamped, seekTargetMs: clamped, seekSeq: s.seekSeq + 1 };
    }),

  step: (frames) => get().seek(get().progMs + frames * FRAME_MS),

  reset: () =>
    set((s) => ({ progMs: 0, playing: false, seekTargetMs: 0, seekSeq: s.seekSeq + 1 })),
}));
