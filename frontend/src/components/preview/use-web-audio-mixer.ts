/**
 * WebAudio mixer for the composite preview.
 *
 * Instead of one <audio> element per timeline segment (which thrashed the
 * decoder and comb-filtered the audio), we keep ONE pooled MediaElement source
 * per source FILE, route each through a per-file GainNode into a master bus, and
 * drive everything from the preview's program clock. Per-layer gain/duck and
 * short equal-power fades at segment boundaries are applied as GainNode ramps,
 * which de-clicks butt-splices and lets beds duck under dialogue.
 *
 * Memory note: we deliberately stream via MediaElementSource rather than
 * decoding whole multi-minute proxies into AudioBuffers (hundreds of MB each).
 */
import { useCallback, useEffect, useRef } from "react";
import type { ResolvedTimeline } from "@/lib/api";

const FADE_S = 0.012; // ~12ms de-click ramp at layer edges
const DRIFT_S = 0.18; // re-seek a file element only past this drift

function dbToGain(db: number): number {
  if (db <= -119) return 0;
  return Math.min(1, Math.pow(10, db / 20));
}

interface FileNode {
  el: HTMLAudioElement;
  src: MediaElementAudioSourceNode;
  gain: GainNode;
  wired: boolean;
}

export interface MixerHandle {
  /** Resume/realize the AudioContext from within a user gesture. */
  arm: () => void;
  /** Place all audio at program time `t` (ms). `playing` gates playback. */
  sync: (t: number, playing: boolean) => void;
  /** Stop everything (pause elements, mute bus). */
  stop: () => void;
  setMuted: (m: boolean) => void;
}

export function useWebAudioMixer(
  resolved: ResolvedTimeline | null,
  urls: Record<string, string>
): MixerHandle {
  const ctxRef = useRef<AudioContext | null>(null);
  const masterRef = useRef<GainNode | null>(null);
  const nodesRef = useRef<Map<string, FileNode>>(new Map());
  const mutedRef = useRef(false);
  const resolvedRef = useRef(resolved);
  const urlsRef = useRef(urls);
  resolvedRef.current = resolved;
  urlsRef.current = urls;

  const ensureCtx = useCallback(() => {
    if (ctxRef.current) return ctxRef.current;
    const Ctor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    const ctx = new Ctor();
    const master = ctx.createGain();
    master.gain.value = 1;
    master.connect(ctx.destination);
    ctxRef.current = ctx;
    masterRef.current = master;
    return ctx;
  }, []);

  // Build/refresh one pooled audio element per source file referenced by the
  // resolved audio layers. Tear down files no longer used.
  const ensureNodes = useCallback(() => {
    const ctx = ctxRef.current;
    const master = masterRef.current;
    if (!ctx || !master) return;
    const needed = new Set<string>();
    for (const a of resolvedRef.current?.audio_layers ?? []) needed.add(a.source_file_id);

    for (const fid of needed) {
      const url = urlsRef.current[fid];
      if (!url) continue;
      let node = nodesRef.current.get(fid);
      if (!node) {
        const el = new Audio();
        el.crossOrigin = "anonymous";
        el.preload = "auto";
        el.src = url;
        const gain = ctx.createGain();
        gain.gain.value = 0;
        let src: MediaElementAudioSourceNode;
        try {
          src = ctx.createMediaElementSource(el);
          src.connect(gain);
          gain.connect(master);
          node = { el, src, gain, wired: true };
        } catch {
          node = { el, src: null as unknown as MediaElementAudioSourceNode, gain, wired: false };
        }
        nodesRef.current.set(fid, node);
      } else if (node.el.src !== url) {
        node.el.src = url;
      }
    }

    for (const [fid, node] of nodesRef.current) {
      if (!needed.has(fid)) {
        node.el.pause();
        try {
          node.gain.disconnect();
          node.src?.disconnect();
        } catch {
          /* already gone */
        }
        nodesRef.current.delete(fid);
      }
    }
  }, []);

  useEffect(() => {
    if (ctxRef.current) ensureNodes();
  }, [resolved, urls, ensureNodes]);

  const arm = useCallback(() => {
    const ctx = ensureCtx();
    if (ctx.state === "suspended") void ctx.resume();
    ensureNodes();
  }, [ensureCtx, ensureNodes]);

  const sync = useCallback((t: number, playing: boolean) => {
    const ctx = ctxRef.current;
    if (!ctx) return;
    const resolvedNow = resolvedRef.current;
    if (!resolvedNow) return;
    const now = ctx.currentTime;

    // Resolve the target gain per FILE = the active layer over t using that
    // file (gain + duck), with an edge fade for de-click. Files with no active
    // layer fade to silence.
    const targetGain = new Map<string, number>();
    const targetPos = new Map<string, number>();
    for (const a of resolvedNow.audio_layers) {
      if (a.prog_start_ms <= t && t < a.prog_end_ms) {
        const want = (a.src_in_ms + (t - a.prog_start_ms)) / 1000;
        // equal-power edge fade
        const intoStart = (t - a.prog_start_ms) / 1000;
        const toEnd = (a.prog_end_ms - t) / 1000;
        const edge = Math.min(intoStart, toEnd);
        const fade = edge < FADE_S ? Math.sin((Math.max(0, edge) / FADE_S) * (Math.PI / 2)) : 1;
        const g = (mutedRef.current ? 0 : dbToGain(a.gain_db + a.duck_db)) * fade;
        // when a file backs multiple active layers (rare), keep the loudest
        targetGain.set(a.source_file_id, Math.max(targetGain.get(a.source_file_id) ?? 0, g));
        if (!targetPos.has(a.source_file_id)) targetPos.set(a.source_file_id, want);
      }
    }

    for (const [fid, node] of nodesRef.current) {
      const g = targetGain.get(fid) ?? 0;
      node.gain.gain.setTargetAtTime(g, now, FADE_S / 3);
      const want = targetPos.get(fid);
      if (playing && want != null) {
        if (Math.abs(node.el.currentTime - want) > DRIFT_S) {
          try {
            node.el.currentTime = want;
          } catch {
            /* not seekable yet */
          }
        }
        if (node.el.paused) void node.el.play().catch(() => {});
      } else if (!node.el.paused) {
        node.el.pause();
      }
    }
  }, []);

  const stop = useCallback(() => {
    const ctx = ctxRef.current;
    const now = ctx?.currentTime ?? 0;
    for (const node of nodesRef.current.values()) {
      node.el.pause();
      if (ctx) node.gain.gain.setTargetAtTime(0, now, FADE_S / 3);
    }
  }, []);

  const setMuted = useCallback((m: boolean) => {
    mutedRef.current = m;
  }, []);

  useEffect(() => {
    return () => {
      // eslint-disable-next-line react-hooks/exhaustive-deps
      const nodes = nodesRef.current;
      for (const node of nodes.values()) {
        node.el.pause();
        try {
          node.gain.disconnect();
          node.src?.disconnect();
        } catch {
          /* ignore */
        }
      }
      nodes.clear();
      void ctxRef.current?.close();
      ctxRef.current = null;
    };
  }, []);

  return { arm, sync, stop, setMuted };
}
