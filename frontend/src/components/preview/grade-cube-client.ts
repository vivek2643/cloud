/**
 * Fetches a baked `.cube` from the backend's `GET /api/grade/cube` (see
 * `backend/app/routers/grade.py`) for a resolved clip grade. Sends the RAW
 * CDL/creative-lut-ref/working-space values -- never a locally-computed
 * hash (see `resolve-timeline.ts::resolveClipGrade`'s docstring for why).
 * In-memory cache keyed by the request URL so repeated frames of the same
 * clip (same CDL) don't refetch; the browser's own HTTP cache backs that up
 * across page loads (the endpoint sets a long immutable Cache-Control).
 */
import type { ResolvedGrade } from "@/lib/api";
import { useAuthStore } from "@/stores/auth-store";
import { parseCubeText } from "./lut-gl";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface CubeEntry {
  grid: Float32Array;
  size: number;
}

const cache = new Map<string, CubeEntry>();
const inFlight = new Map<string, Promise<CubeEntry | null>>();

// Fired whenever an async cube fetch finishes and caches a new entry. The
// paused preview subscribes to this: the rAF draw loop only runs during
// playback, so without a repaint trigger a grade picked while paused would
// stay invisible until the cube arrived on some later frame that never comes.
const cubeLoadListeners = new Set<() => void>();

/** Subscribe to "a new cube finished loading". Returns an unsubscribe fn. */
export function subscribeCubeLoaded(cb: () => void): () => void {
  cubeLoadListeners.add(cb);
  return () => {
    cubeLoadListeners.delete(cb);
  };
}

export function gradeCubeUrl(grade: ResolvedGrade): string {
  const params = new URLSearchParams({
    cdl: JSON.stringify(grade.cdl),
    working_space: grade.working_space,
  });
  if (grade.creative_lut_ref) params.set("creative_lut_ref", grade.creative_lut_ref);
  return `${API_URL}/api/grade/cube?${params.toString()}`;
}

/** Non-blocking: returns a cached cube immediately if we have one, else
 * kicks off a fetch (deduped) and returns null for this call -- the caller
 * (the rAF draw loop) just tries again next frame once it resolves. */
export function getGradeCube(grade: ResolvedGrade | undefined): CubeEntry | null {
  if (!grade) return null;
  const url = gradeCubeUrl(grade);
  const hit = cache.get(url);
  if (hit) return hit;
  if (!inFlight.has(url)) {
    const token = useAuthStore.getState().session?.access_token;
    const p = fetch(url, {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    })
      .then((res) => (res.ok ? res.text() : null))
      .then((text) => {
        if (!text) return null;
        const parsed = parseCubeText(text);
        if (!parsed) return null;
        cache.set(url, parsed);
        cubeLoadListeners.forEach((cb) => {
          try {
            cb();
          } catch {
            /* a listener throwing must not break the fetch chain */
          }
        });
        return parsed;
      })
      .catch(() => null)
      .finally(() => {
        inFlight.delete(url);
      });
    inFlight.set(url, p);
  }
  return null;
}
