/**
 * Double-buffered video picture for the composite preview.
 *
 * A single <video> that reassigns `src` on every cross-file cut stalls for
 * seconds while the new proxy loads. Instead we keep TWO <video> elements: the
 * FRONT is visible/playing; the BACK silently preloads + seeks the upcoming
 * source so that at the cut we just swap which element is visible. Within a
 * single source we only seek. The picture is always muted — audio comes from
 * the WebAudio mixer.
 */
import { useCallback, useRef } from "react";
import type { ResolvedTimeline, ResolvedVideoLayer } from "@/lib/api";

const DRIFT_S = 0.2;
const PREFETCH_MS = 1500; // start loading the next source this far ahead

export interface PictureHandle {
  attachA: (el: HTMLVideoElement | null) => void;
  attachB: (el: HTMLVideoElement | null) => void;
  sync: (t: number, playing: boolean) => void;
  stop: () => void;
}

function topLayerAt(resolved: ResolvedTimeline, t: number): ResolvedVideoLayer | null {
  let top: ResolvedVideoLayer | null = null;
  for (const v of resolved.video_layers) {
    if (v.prog_start_ms <= t && t < v.prog_end_ms) {
      if (!top || v.z > top.z) top = v;
    }
  }
  return top;
}

export function useVideoPicture(
  resolved: ResolvedTimeline | null,
  urls: Record<string, string>
): PictureHandle {
  const aRef = useRef<HTMLVideoElement | null>(null);
  const bRef = useRef<HTMLVideoElement | null>(null);
  const frontIsA = useRef(true);
  const frontFile = useRef<string>("");
  const backFile = useRef<string>("");
  const resolvedRef = useRef(resolved);
  const urlsRef = useRef(urls);
  resolvedRef.current = resolved;
  urlsRef.current = urls;

  const front = () => (frontIsA.current ? aRef.current : bRef.current);
  const back = () => (frontIsA.current ? bRef.current : aRef.current);

  const setupEl = (el: HTMLVideoElement) => {
    el.muted = true;
    el.playsInline = true;
    el.preload = "auto";
  };

  const attachA = useCallback((el: HTMLVideoElement | null) => {
    aRef.current = el;
    if (el) {
      setupEl(el);
      el.style.opacity = "1";
    }
  }, []);

  const attachB = useCallback((el: HTMLVideoElement | null) => {
    bRef.current = el;
    if (el) {
      setupEl(el);
      el.style.opacity = "0";
    }
  }, []);

  const showFront = () => {
    const f = front();
    const b = back();
    if (f) f.style.opacity = "1";
    if (b) {
      b.style.opacity = "0";
      b.pause();
    }
  };

  const sync = useCallback((t: number, playing: boolean) => {
    const resolvedNow = resolvedRef.current;
    if (!resolvedNow) return;
    const top = topLayerAt(resolvedNow, t);
    const f = front();
    if (!f || !top) return;
    const url = urlsRef.current[top.source_file_id];
    if (!url) return;
    const want = (top.src_in_ms + (t - top.prog_start_ms)) / 1000;

    if (frontFile.current === top.source_file_id) {
      // same source — just keep it positioned
      if (Math.abs(f.currentTime - want) > DRIFT_S) {
        try {
          f.currentTime = want;
        } catch {
          /* not ready */
        }
      }
      if (playing && f.paused) void f.play().catch(() => {});
      else if (!playing && !f.paused) f.pause();
    } else {
      // cross-file cut: promote the back buffer (preloaded) or load front now
      const b = back();
      if (b && backFile.current === top.source_file_id && b.readyState >= 1) {
        try {
          b.currentTime = want;
        } catch {
          /* ignore */
        }
        frontIsA.current = !frontIsA.current;
        frontFile.current = top.source_file_id;
        backFile.current = "";
        showFront();
        if (playing) void front()?.play().catch(() => {});
      } else {
        // no preloaded buffer ready — load straight onto the front element
        if (f.src !== url) f.src = url;
        frontFile.current = top.source_file_id;
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

    // Prefetch the NEXT distinct source into the back buffer.
    if (playing) {
      const next = topLayerAt(resolvedNow, t + PREFETCH_MS);
      if (next && next.source_file_id !== frontFile.current) {
        const b = back();
        const nurl = urlsRef.current[next.source_file_id];
        if (b && nurl && backFile.current !== next.source_file_id) {
          b.src = nurl;
          backFile.current = next.source_file_id;
          const seekBack = () => {
            const nwant = (next.src_in_ms + Math.max(0, t + PREFETCH_MS - next.prog_start_ms)) / 1000;
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
    // refs only; showFront/front/back close over stable refs
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stop = useCallback(() => {
    aRef.current?.pause();
    bRef.current?.pause();
  }, []);

  return { attachA, attachB, sync, stop };
}
