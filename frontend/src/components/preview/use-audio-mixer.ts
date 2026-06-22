/**
 * Audio mixer for the composite preview.
 *
 * Two playback strategies, picked from the resolved timeline:
 *
 *  1. SEQUENTIAL (the common case — a pure spine with no overlapping beds):
 *     a double-buffered A/B pair, mirroring the video picture. The FRONT plays
 *     the current layer; the BACK preloads + pre-seeks the upcoming layer so a
 *     cut (cross-file OR a same-file jump) is just a swap — no in-place seek,
 *     so no click/stall. This is what fixes "the sound is not right" on edits
 *     that hop around one file.
 *
 *  2. MIXED (overlapping layers, e.g. a music bed under dialogue): one pooled
 *     <audio> element per source FILE, gain/duck + short edge fades applied via
 *     `element.volume`. Concurrent layers genuinely need simultaneous elements.
 *
 * IMPORTANT: we do NOT route through a WebAudio MediaElementSource. Doing so
 * taints cross-origin media (our presigned R2 URLs) and outputs SILENCE unless
 * the bucket serves CORS headers. Plain element.volume avoids that entirely.
 */
import { useCallback, useEffect, useRef } from "react";
import type { ResolvedAudioLayer, ResolvedTimeline } from "@/lib/api";

const FADE_MS = 40; // edge fade window for de-click
const DRIFT_S = 0.18; // re-seek a file element only past this drift
const PREFETCH_MS = 1500; // pre-seek the next layer this far ahead

function dbToGain(db: number): number {
  if (db <= -119) return 0;
  return Math.min(1, Math.pow(10, db / 20));
}

/** True when no two audio layers overlap in program time (sequential spine). */
function isSequential(layers: ResolvedAudioLayer[]): boolean {
  const sorted = [...layers].sort((a, b) => a.prog_start_ms - b.prog_start_ms);
  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i].prog_start_ms < sorted[i - 1].prog_end_ms - 1) return false;
  }
  return true;
}

function topAudioAt(layers: ResolvedAudioLayer[], t: number): ResolvedAudioLayer | null {
  let top: ResolvedAudioLayer | null = null;
  for (const a of layers) {
    if (a.prog_start_ms <= t && t < a.prog_end_ms) {
      if (!top || a.gain_db + a.duck_db > top.gain_db + top.duck_db) top = a;
    }
  }
  return top;
}

/** Equal-power-ish edge fade so cuts don't click. */
function edgeFade(layer: ResolvedAudioLayer, t: number): number {
  const edge = Math.min(t - layer.prog_start_ms, layer.prog_end_ms - t);
  return edge < FADE_MS ? Math.sin((Math.max(0, edge) / FADE_MS) * (Math.PI / 2)) : 1;
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
  // --- mixed (per-file) path state ---
  const els = useRef<Map<string, HTMLAudioElement>>(new Map());
  // --- sequential (A/B) path state ---
  const aRef = useRef<HTMLAudioElement | null>(null);
  const bRef = useRef<HTMLAudioElement | null>(null);
  const frontIsA = useRef(true);
  const frontLayer = useRef<string>("");
  const backLayer = useRef<string>("");

  const seqRef = useRef(false);
  const mutedRef = useRef(false);
  const resolvedRef = useRef(resolved);
  const urlsRef = useRef(urls);
  resolvedRef.current = resolved;
  urlsRef.current = urls;

  const front = () => (frontIsA.current ? aRef.current : bRef.current);
  const back = () => (frontIsA.current ? bRef.current : aRef.current);

  const ensureEls = useCallback(() => {
    const layers = resolvedRef.current?.audio_layers ?? [];
    seqRef.current = isSequential(layers);

    if (seqRef.current) {
      if (!aRef.current) {
        const a = new Audio();
        a.preload = "auto";
        aRef.current = a;
      }
      if (!bRef.current) {
        const b = new Audio();
        b.preload = "auto";
        bRef.current = b;
      }
      // Tear down the per-file pool if we switched strategies.
      for (const [fid, el] of els.current) {
        el.pause();
        els.current.delete(fid);
      }
      return;
    }

    // Mixed path: one pooled element per referenced file.
    const needed = new Set<string>();
    for (const a of layers) needed.add(a.source_file_id);
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

  // --- sequential A/B sync ---------------------------------------------------
  const syncSequential = (t: number, playing: boolean) => {
    const layers = resolvedRef.current!.audio_layers;
    const top = topAudioAt(layers, t);
    const f = front();
    const b = back();
    if (!f) return;

    if (!top) {
      // Gap — silence both.
      f.volume = 0;
      if (b) b.volume = 0;
      if (!f.paused) f.pause();
      if (b && !b.paused) b.pause();
      return;
    }

    const url = urlsRef.current[top.source_file_id];
    if (!url) return;
    const vol = (mutedRef.current ? 0 : dbToGain(top.gain_db + top.duck_db)) * edgeFade(top, t);
    const want = (top.src_in_ms + (t - top.prog_start_ms)) / 1000;

    if (frontLayer.current === top.layer_id) {
      f.volume = Math.max(0, Math.min(1, vol));
      if (playing) {
        if (Math.abs(f.currentTime - want) > DRIFT_S) {
          try {
            f.currentTime = want;
          } catch {
            /* not seekable yet */
          }
        }
        if (f.paused) void f.play().catch(() => {});
      } else if (!f.paused) {
        f.pause();
      }
    } else {
      // Cut — promote the preloaded back buffer, else seek the front.
      if (b && backLayer.current === top.layer_id && b.readyState >= 1) {
        if (Math.abs(b.currentTime - want) > DRIFT_S) {
          try {
            b.currentTime = want;
          } catch {
            /* ignore */
          }
        }
        frontIsA.current = !frontIsA.current;
        frontLayer.current = top.layer_id;
        backLayer.current = "";
        const nf = front();
        const nb = back();
        if (nf) {
          nf.volume = Math.max(0, Math.min(1, vol));
          if (playing && nf.paused) void nf.play().catch(() => {});
        }
        if (nb) {
          nb.volume = 0;
          if (!nb.paused) nb.pause();
        }
      } else {
        if (f.src !== url) f.src = url;
        frontLayer.current = top.layer_id;
        f.volume = Math.max(0, Math.min(1, vol));
        const onReady = () => {
          try {
            f.currentTime = want;
          } catch {
            /* ignore */
          }
          if (playing) void f.play().catch(() => {});
        };
        if (f.readyState >= 1) onReady();
        else f.addEventListener("loadedmetadata", onReady, { once: true });
      }
    }

    // Prefetch + pre-seek the upcoming layer into the back buffer.
    if (playing && b) {
      const next = topAudioAt(layers, t + PREFETCH_MS);
      if (next && next.layer_id !== frontLayer.current && next.layer_id !== backLayer.current) {
        const nurl = urlsRef.current[next.source_file_id];
        if (nurl) {
          if (b.src !== nurl) b.src = nurl;
          b.volume = 0;
          backLayer.current = next.layer_id;
          const seekBack = () => {
            const nwant =
              (next.src_in_ms + Math.max(0, t + PREFETCH_MS - next.prog_start_ms)) / 1000;
            try {
              b.currentTime = nwant;
            } catch {
              /* ignore */
            }
          };
          if (b.readyState >= 1) seekBack();
          else b.addEventListener("loadedmetadata", seekBack, { once: true });
        }
      }
    }
  };

  // --- mixed per-file sync ---------------------------------------------------
  const syncMixed = (t: number, playing: boolean) => {
    const resolvedNow = resolvedRef.current!;
    const targetVol = new Map<string, number>();
    const targetPos = new Map<string, number>();
    for (const a of resolvedNow.audio_layers) {
      if (a.prog_start_ms <= t && t < a.prog_end_ms) {
        const vol = (mutedRef.current ? 0 : dbToGain(a.gain_db + a.duck_db)) * edgeFade(a, t);
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
  };

  const sync = useCallback((t: number, playing: boolean) => {
    if (!resolvedRef.current) return;
    if (seqRef.current) syncSequential(t, playing);
    else syncMixed(t, playing);
    // refs only; helpers close over stable refs
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stop = useCallback(() => {
    aRef.current?.pause();
    bRef.current?.pause();
    for (const el of els.current.values()) el.pause();
  }, []);

  const setMuted = useCallback((m: boolean) => {
    mutedRef.current = m;
  }, []);

  useEffect(() => {
    return () => {
      aRef.current?.pause();
      bRef.current?.pause();
      // eslint-disable-next-line react-hooks/exhaustive-deps
      const map = els.current;
      for (const el of map.values()) el.pause();
      map.clear();
    };
  }, []);

  return { arm, sync, stop, setMuted };
}
