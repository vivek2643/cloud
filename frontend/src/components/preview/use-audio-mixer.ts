/**
 * Audio mixer for the composite preview.
 *
 * One pooled <audio> element per source FILE (not per segment) — this is what
 * fixed the "mixed up" audio from the old per-segment element swarm. Gain/duck
 * and short edge fades are applied directly via `element.volume`.
 *
 * IMPORTANT: we do NOT route through a WebAudio MediaElementSource. Doing so
 * taints cross-origin media (our presigned R2 URLs) and outputs SILENCE unless
 * the bucket serves CORS headers. Plain element.volume avoids that entirely.
 */
import { useCallback, useEffect, useRef } from "react";
import type { ResolvedTimeline } from "@/lib/api";

const FADE_MS = 40; // edge fade window for de-click
const DRIFT_S = 0.18; // re-seek a file element only past this drift

function dbToGain(db: number): number {
  if (db <= -119) return 0;
  return Math.min(1, Math.pow(10, db / 20));
}

export interface MixerHandle {
  arm: () => void;
  sync: (t: number, playing: boolean) => void;
  stop: () => void;
  setMuted: (m: boolean) => void;
}

export function useAudioMixer(
  resolved: ResolvedTimeline | null,
  urls: Record<string, string>
): MixerHandle {
  const els = useRef<Map<string, HTMLAudioElement>>(new Map());
  const mutedRef = useRef(false);
  const resolvedRef = useRef(resolved);
  const urlsRef = useRef(urls);
  resolvedRef.current = resolved;
  urlsRef.current = urls;

  const ensureEls = useCallback(() => {
    const needed = new Set<string>();
    for (const a of resolvedRef.current?.audio_layers ?? []) needed.add(a.source_file_id);

    for (const fid of needed) {
      const url = urlsRef.current[fid];
      if (!url) continue;
      let el = els.current.get(fid);
      if (!el) {
        el = new Audio();
        el.preload = "auto";
        el.src = url;
        els.current.set(fid, el);
      } else if (el.src !== url) {
        el.src = url;
      }
    }
    for (const [fid, el] of els.current) {
      if (!needed.has(fid)) {
        el.pause();
        els.current.delete(fid);
      }
    }
  }, []);

  useEffect(() => {
    ensureEls();
  }, [resolved, urls, ensureEls]);

  const arm = useCallback(() => {
    ensureEls();
  }, [ensureEls]);

  const sync = useCallback((t: number, playing: boolean) => {
    const resolvedNow = resolvedRef.current;
    if (!resolvedNow) return;

    // Target volume + source position per FILE: the active layer over t using
    // that file (gain + duck) with an equal-power edge fade for de-click.
    const targetVol = new Map<string, number>();
    const targetPos = new Map<string, number>();
    for (const a of resolvedNow.audio_layers) {
      if (a.prog_start_ms <= t && t < a.prog_end_ms) {
        const intoStart = t - a.prog_start_ms;
        const toEnd = a.prog_end_ms - t;
        const edge = Math.min(intoStart, toEnd);
        const fade = edge < FADE_MS ? Math.sin((Math.max(0, edge) / FADE_MS) * (Math.PI / 2)) : 1;
        const vol = (mutedRef.current ? 0 : dbToGain(a.gain_db + a.duck_db)) * fade;
        targetVol.set(a.source_file_id, Math.max(targetVol.get(a.source_file_id) ?? 0, vol));
        if (!targetPos.has(a.source_file_id))
          targetPos.set(a.source_file_id, (a.src_in_ms + (t - a.prog_start_ms)) / 1000);
      }
    }

    for (const [fid, el] of els.current) {
      const vol = targetVol.get(fid) ?? 0;
      el.volume = Math.max(0, Math.min(1, vol));
      const want = targetPos.get(fid);
      if (playing && want != null) {
        if (Math.abs(el.currentTime - want) > DRIFT_S) {
          try {
            el.currentTime = want;
          } catch {
            /* not seekable yet */
          }
        }
        if (el.paused) void el.play().catch(() => {});
      } else if (!el.paused) {
        el.pause();
      }
    }
  }, []);

  const stop = useCallback(() => {
    for (const el of els.current.values()) el.pause();
  }, []);

  const setMuted = useCallback((m: boolean) => {
    mutedRef.current = m;
  }, []);

  useEffect(() => {
    return () => {
      // eslint-disable-next-line react-hooks/exhaustive-deps
      const map = els.current;
      for (const el of map.values()) el.pause();
      map.clear();
    };
  }, []);

  return { arm, sync, stop, setMuted };
}
