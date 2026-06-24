/**
 * Audio engine for the composite preview — the way a real editor does it.
 *
 * Instead of seeking <audio> elements every animation frame (which drifts,
 * echoes, and stalls on remote files), we DECODE each source's audio once into
 * an AudioBuffer and SCHEDULE the timeline on the Web Audio clock:
 * `source.start(when, offset, duration)` per layer. The AudioContext is the
 * MASTER CLOCK; playback is sample-accurate and gapless, and there is NO
 * per-frame seeking, so the whole "echo / stutter" class of bugs disappears.
 *
 * The video picture follows this clock (see composite-preview): we read
 * `nowMs()` each frame to place the playhead and the muted <video> elements.
 *
 * Decoding uses `fetch(url) -> arrayBuffer -> decodeAudioData`, which needs the
 * bucket to serve CORS on GET (it does). Buffers are decoded lazily for the
 * files referenced by the current edit and evicted when they're no longer used.
 */
import { useCallback, useEffect, useRef } from "react";
import type { ResolvedAudioLayer, ResolvedTimeline } from "@/lib/api";

const FADE_S = 0.012; // 12ms equal-ish edge fade to de-click cuts
const SCHED_LEAD_S = 0.05; // start the timeline this far ahead of ctx.currentTime

function dbToGain(db: number): number {
  if (db <= -119) return 0;
  return Math.min(1, Math.pow(10, db / 20));
}

interface Scheduled {
  src: AudioBufferSourceNode;
  gain: GainNode;
}

export interface AudioEngineHandle {
  /** Lazily decode the audio for every file in the current edit. */
  prepare: () => void;
  /** Begin playback from a program position (ms). Idempotent-safe. */
  play: (fromMs: number) => void;
  /** Stop sounding; remember the current position. */
  pause: () => void;
  /** Jump to a program position; reschedules if currently playing. */
  seek: (toMs: number, playing: boolean) => void;
  /** The master program clock in ms (advances only while playing). */
  nowMs: () => number;
  setMuted: (m: boolean) => void;
  /** Hard stop + release (used on unmount / clear). */
  stop: () => void;
}

export function useAudioEngine(
  resolved: ResolvedTimeline | null,
  urls: Record<string, string>
): AudioEngineHandle {
  const ctxRef = useRef<AudioContext | null>(null);
  const masterRef = useRef<GainNode | null>(null);
  const buffers = useRef<Map<string, AudioBuffer>>(new Map());
  const decoding = useRef<Map<string, Promise<void>>>(new Map());
  const scheduled = useRef<Scheduled[]>([]);

  // Master clock state: program time `originMs` corresponds to ctx time `t0`.
  const playingRef = useRef(false);
  const originMs = useRef(0);
  const t0 = useRef(0);
  const mutedRef = useRef(false);

  const resolvedRef = useRef(resolved);
  const urlsRef = useRef(urls);
  resolvedRef.current = resolved;
  urlsRef.current = urls;

  const ensureCtx = useCallback((): AudioContext | null => {
    if (typeof window === "undefined") return null;
    if (!ctxRef.current) {
      const Ctor = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) return null;
      const ctx = new Ctor();
      const master = ctx.createGain();
      master.gain.value = mutedRef.current ? 0 : 1;
      master.connect(ctx.destination);
      ctxRef.current = ctx;
      masterRef.current = master;
    }
    return ctxRef.current;
  }, []);

  const usedFileIds = useCallback((): string[] => {
    const s = new Set<string>();
    for (const a of resolvedRef.current?.audio_layers ?? []) s.add(a.source_file_id);
    return Array.from(s).filter(Boolean);
  }, []);

  const decodeFile = useCallback(
    (fileId: string): Promise<void> => {
      if (buffers.current.has(fileId)) return Promise.resolve();
      const existing = decoding.current.get(fileId);
      if (existing) return existing;
      const ctx = ensureCtx();
      const url = urlsRef.current[fileId];
      if (!ctx || !url) return Promise.resolve();
      const p = fetch(url)
        .then((r) => r.arrayBuffer())
        .then((buf) => ctx.decodeAudioData(buf))
        .then((audio) => {
          buffers.current.set(fileId, audio);
          decoding.current.delete(fileId);
          // If we're mid-playback, fold this file's layers into the running
          // schedule now that its buffer is ready.
          if (playingRef.current) scheduleFile(fileId);
        })
        .catch(() => {
          decoding.current.delete(fileId);
        });
      decoding.current.set(fileId, p);
      return p;
    },
    // scheduleFile is stable (refs only)
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [ensureCtx]
  );

  const prepare = useCallback(() => {
    const used = new Set(usedFileIds());
    // Evict buffers we no longer need (frees the decoded PCM).
    for (const fid of Array.from(buffers.current.keys())) {
      if (!used.has(fid)) buffers.current.delete(fid);
    }
    for (const fid of used) void decodeFile(fid);
  }, [usedFileIds, decodeFile]);

  // Schedule ONE layer onto the running clock. `when` is the ctx time the layer
  // should sound; layers already past the playhead are skipped.
  const scheduleLayer = (layer: ResolvedAudioLayer) => {
    const ctx = ctxRef.current;
    const master = masterRef.current;
    const buf = buffers.current.get(layer.source_file_id);
    if (!ctx || !master || !buf) return;

    const from = originMs.current;
    const playEnd = layer.prog_end_ms;
    if (playEnd <= from) return; // entirely behind the playhead

    const playStart = Math.max(layer.prog_start_ms, from);
    const when = t0.current + (playStart - from) / 1000;
    const offset = (layer.src_in_ms + (playStart - layer.prog_start_ms)) / 1000;
    const dur = (playEnd - playStart) / 1000;
    if (dur <= 0) return;

    const srcDur = buf.duration;
    const safeOffset = Math.max(0, Math.min(offset, Math.max(0, srcDur - 0.01)));
    const safeDur = Math.max(0.01, Math.min(dur, srcDur - safeOffset));

    const g = dbToGain(layer.gain_db + layer.duck_db);
    const gain = ctx.createGain();
    const fade = Math.min(FADE_S, safeDur / 2);
    // Edge fades only at the true layer boundaries (not when we join mid-layer
    // after a seek/late-decode).
    const fadeIn = playStart <= layer.prog_start_ms + 1;
    gain.gain.setValueAtTime(fadeIn ? 0 : g, when);
    if (fadeIn) gain.gain.linearRampToValueAtTime(g, when + fade);
    gain.gain.setValueAtTime(g, Math.max(when, when + safeDur - fade));
    gain.gain.linearRampToValueAtTime(0, when + safeDur);

    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(gain);
    gain.connect(master);
    try {
      src.start(when, safeOffset, safeDur);
    } catch {
      return;
    }
    const entry: Scheduled = { src, gain };
    scheduled.current.push(entry);
    src.onended = () => {
      try {
        src.disconnect();
        gain.disconnect();
      } catch {
        /* already gone */
      }
      scheduled.current = scheduled.current.filter((e) => e !== entry);
    };
  };

  const scheduleFile = (fileId: string) => {
    for (const a of resolvedRef.current?.audio_layers ?? []) {
      if (a.source_file_id === fileId) scheduleLayer(a);
    }
  };

  const clearScheduled = () => {
    for (const e of scheduled.current) {
      try {
        e.src.onended = null;
        e.src.stop();
        e.src.disconnect();
        e.gain.disconnect();
      } catch {
        /* ignore */
      }
    }
    scheduled.current = [];
  };

  const scheduleAll = (fromMs: number) => {
    const ctx = ensureCtx();
    if (!ctx) return;
    clearScheduled();
    originMs.current = Math.max(0, fromMs);
    t0.current = ctx.currentTime + SCHED_LEAD_S;
    for (const a of resolvedRef.current?.audio_layers ?? []) {
      if (buffers.current.has(a.source_file_id)) scheduleLayer(a);
      else void decodeFile(a.source_file_id); // schedules itself when ready
    }
  };

  const play = useCallback(
    (fromMs: number) => {
      const ctx = ensureCtx();
      if (!ctx) return;
      if (ctx.state === "suspended") void ctx.resume();
      playingRef.current = true;
      scheduleAll(fromMs);
    },
    // refs/stable closures only
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [ensureCtx]
  );

  const nowMs = useCallback((): number => {
    const ctx = ctxRef.current;
    if (!playingRef.current || !ctx) return originMs.current;
    return originMs.current + Math.max(0, ctx.currentTime - t0.current) * 1000;
  }, []);

  const pause = useCallback(() => {
    originMs.current = nowMs();
    playingRef.current = false;
    clearScheduled();
  }, [nowMs]);

  const seek = useCallback(
    (toMs: number, playing: boolean) => {
      originMs.current = Math.max(0, toMs);
      if (playing) scheduleAll(toMs);
      else clearScheduled();
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const setMuted = useCallback((m: boolean) => {
    mutedRef.current = m;
    if (masterRef.current && ctxRef.current) {
      masterRef.current.gain.setValueAtTime(m ? 0 : 1, ctxRef.current.currentTime);
    }
  }, []);

  const stop = useCallback(() => {
    playingRef.current = false;
    clearScheduled();
  }, []);

  // Decode whenever the set of referenced files changes.
  useEffect(() => {
    prepare();
  }, [resolved, urls, prepare]);

  // Release the AudioContext on unmount.
  useEffect(() => {
    const bufs = buffers.current;
    const dec = decoding.current;
    return () => {
      clearScheduled();
      const ctx = ctxRef.current;
      ctxRef.current = null;
      masterRef.current = null;
      bufs.clear();
      dec.clear();
      if (ctx) void ctx.close().catch(() => {});
    };
  }, []);

  return { prepare, play, pause, seek, nowMs, setMuted, stop };
}
