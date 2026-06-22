/**
 * Double-buffered video picture for the composite preview.
 *
 * A single <video> that seeks on every cut stalls (black frame) while it
 * decodes to the new position — and the edit jumps around the SAME file a lot
 * (the editor picks non-contiguous moments), so most cuts are same-file seeks.
 * Instead we keep TWO <video> elements: the FRONT is visible/playing; the BACK
 * silently preloads + seeks the upcoming LAYER (same file or not) so that at the
 * boundary we just swap which element is visible — the seek stall happens
 * off-screen. We only fall back to seeking the front in place when the next
 * layer arrives faster than we could prefetch it. The picture is always muted —
 * audio comes from the mixer.
 */
import { useCallback, useRef } from "react";
import type { ResolvedTimeline, ResolvedVideoLayer } from "@/lib/api";

const DRIFT_S = 0.2;
const PREFETCH_MS = 1500; // start loading the next layer this far ahead

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
  // We key on the LAYER (a single contiguous source span), not just the file,
  // so a same-file jump-cut also swaps buffers instead of seeking in place.
  const frontLayer = useRef<string>("");
  const backLayer = useRef<string>("");
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

    if (frontLayer.current === top.layer_id) {
      // Still inside the current layer — let it free-run, drift-correct only.
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
      // Crossed a cut (cross-file OR same-file jump). Promote the back buffer
      // if it was preloaded + pre-seeked for this layer; else load front now.
      const b = back();
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
        showFront();
        if (playing) void front()?.play().catch(() => {});
      } else {
        // No preloaded buffer ready — seek straight on the front element.
        if (f.src !== url) f.src = url;
        frontLayer.current = top.layer_id;
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

    // Prefetch the NEXT layer (regardless of file) into the back buffer and
    // pre-seek it to its start, so the upcoming cut is just a swap.
    if (playing) {
      const next = topLayerAt(resolvedNow, t + PREFETCH_MS);
      if (
        next &&
        next.layer_id !== frontLayer.current &&
        next.layer_id !== backLayer.current
      ) {
        const b = back();
        const nurl = urlsRef.current[next.source_file_id];
        if (b && nurl) {
          if (b.src !== nurl) b.src = nurl;
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
    // refs only; showFront/front/back close over stable refs
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const stop = useCallback(() => {
    aRef.current?.pause();
    bRef.current?.pause();
  }, []);

  return { attachA, attachB, sync, stop };
}
