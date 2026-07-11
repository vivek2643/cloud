/**
 * Program player for the composite preview — the way browser NLEs (Canva, Veed,
 * Clipchamp) actually do it.
 *
 * We do NOT decode whole files into PCM (that stalls the start and blows up
 * memory), and we do NOT thrash one pair of <video> elements with a look-ahead
 * prefetch that overshoots short cuts. Instead we keep a small POOL of media
 * elements and treat each timeline CLIP as something we assign to a slot:
 *
 *  - Each slot is a <video> whose decoded audio is tapped into a Web Audio graph
 *    (MediaElementSource -> per-slot GainNode -> master -> destination), so we
 *    get real mixing (duck / levels / de-click fades) for free. The element is
 *    NOT `muted` — Chrome routes a muted element's audio as SILENCE into the
 *    MediaElementSource, so muting kills the graph. We rely on the user's Play
 *    gesture (sticky activation) to satisfy the autoplay policy instead.
 *  - At any program instant the VISIBLE picture is the top-z active video clip's
 *    slot; every audible clip's slot drives its gain. Picture visibility and
 *    audibility are independent, so dialogue keeps playing under a B-roll cut.
 *  - The NEXT clip is assigned to a free slot and pre-seeked to its in-point, so
 *    a cut (cross-file OR same-file jump) is just a visibility/gain swap — the
 *    seek already happened off-screen. This is what hides the seek stall.
 *
 * The master clock is `AudioContext.currentTime` (monotonic). Elements are
 * slaved to it: free-running inside a clip, drift-corrected, and re-seeked only
 * at cuts. Seeking is cheap because the proxy is encoded with ~1s keyframes.
 *
 * Requires CORS on the media GET (enabled on the bucket) so `crossOrigin` tap is
 * allowed; otherwise the MediaElementSource would output silence.
 */
import { useCallback, useEffect, useMemo, useRef } from "react";
import type {
  DestRect,
  EditAspect,
  LayerTransform,
  ResolvedGrade,
  ResolvedTimeline,
} from "@/lib/api";
import { isRect, sampleMotion } from "@/lib/resolve-timeline";
import { createLutRenderer, type LutRenderer } from "./lut-gl";
import { getGradeCube, gradeCubeUrl } from "./grade-cube-client";

const POOL_SIZE = 6;
const DRIFT_S = 0.18; // re-seek a free-running element only past this much drift
const PREWARM_MS = 1200; // assign + pre-seek a clip this long before it starts
const GAIN_TC = 0.012; // gain smoothing time-constant (de-click cuts/duck)
const START_LEAD_S = 0.04;

/** A contiguous source span to play: spine segments carry BOTH picture + sound;
 * coverage/angle are picture-only; beds are sound-only. */
interface Clip {
  id: string;
  fileId: string;
  srcInMs: number;
  progStart: number;
  progEnd: number;
  hasVideo: boolean;
  z: number;
  transform?: LayerTransform;
  // A split/PiP cell rect (normalized); null = full frame. When set, several
  // clips are visible at once, each painted into its own rect.
  dest: DestRect | null;
  hasAudio: boolean;
  gainDb: number; // base + duck, already summed (constant per layer)
  grade?: ResolvedGrade;
}

interface Slot {
  el: HTMLVideoElement;
  gain: GainNode;
  clip: Clip | null;
  needsSeek: boolean;
  wantSec: number | null; // last requested park position (applied once loadable)
  queuedSeek: number | null; // coalesced seek target while mid-seek
  // Lazily created on first graded clip -- most projects never touch grading,
  // so most slots never pay for a WebGL context.
  lutRenderer: LutRenderer | null;
  lutUrl: string | null; // gradeCubeUrl of whatever's currently loaded, for the setLut cache-key check
}

export interface ProgramPlayerHandle {
  attachContainer: (el: HTMLDivElement | null) => void;
  /** Ensure ctx/pool exist and the clip list is current. */
  prepare: () => void;
  /** Per-frame drive (called from the rAF loop and after seeks). */
  sync: (t: number, playing: boolean) => void;
  /** Begin playback from a program position (ms). */
  play: (fromMs: number) => void;
  /** Stop sounding; remember position. */
  pause: () => void;
  /** Jump to a program position; reschedules if playing. */
  seek: (toMs: number, playing: boolean) => void;
  /** Master program clock in ms (frozen until the first frame is ready). */
  nowMs: () => number;
  setMuted: (m: boolean) => void;
  /** Hard stop + park everything (used on shape change / unmount). */
  stop: () => void;
}

function dbToGain(db: number): number {
  if (db <= -119) return 0;
  return Math.min(1, Math.pow(10, db / 20));
}

/** Build the playable clip list: pair coincident spine video+audio into one
 * clip, keep coverage/angle as picture-only and beds as sound-only. */
function buildClips(resolved: ResolvedTimeline | null): Clip[] {
  if (!resolved) return [];
  const clips: Clip[] = [];
  const audio = resolved.audio_layers.slice();
  const consumed = new Set<number>();

  const matchAudio = (fileId: string, ps: number, pe: number, srcIn: number): number => {
    for (let i = 0; i < audio.length; i++) {
      if (consumed.has(i)) continue;
      const a = audio[i];
      if (
        a.source_file_id === fileId &&
        Math.abs(a.prog_start_ms - ps) < 1 &&
        Math.abs(a.prog_end_ms - pe) < 1 &&
        Math.abs(a.src_in_ms - srcIn) < 1
      ) {
        return i;
      }
    }
    return -1;
  };

  for (const v of resolved.video_layers) {
    const ai = matchAudio(v.source_file_id, v.prog_start_ms, v.prog_end_ms, v.src_in_ms);
    const a = ai >= 0 ? audio[ai] : null;
    if (a) consumed.add(ai);
    clips.push({
      id: `c_${v.layer_id}`,
      fileId: v.source_file_id,
      srcInMs: v.src_in_ms,
      progStart: v.prog_start_ms,
      progEnd: v.prog_end_ms,
      hasVideo: true,
      z: v.z,
      transform: v.transform,
      dest: isRect(v.transform?.dest) ? v.transform!.dest : null,
      hasAudio: !!a,
      gainDb: a ? a.gain_db + a.duck_db : 0,
      grade: v.grade,
    });
  }
  audio.forEach((a, i) => {
    if (consumed.has(i)) return;
    clips.push({
      id: `c_${a.layer_id}`,
      fileId: a.source_file_id,
      srcInMs: a.src_in_ms,
      progStart: a.prog_start_ms,
      progEnd: a.prog_end_ms,
      hasVideo: false,
      z: -1,
      dest: null,
      hasAudio: true,
      gainDb: a.gain_db + a.duck_db,
    });
  });
  return clips;
}

/** Position an element into its canvas cell: a split/PiP dest rect (percent
 * box) or the full frame. Mirrors the compositor's composite-into-dest step.
 * Takes any styleable element so the WebGL LUT canvas can share this with
 * the plain `<video>` path. */
function applyDestGeometry(el: HTMLElement, dest: DestRect | null) {
  if (dest) {
    el.style.inset = "auto";
    el.style.left = `${dest.x * 100}%`;
    el.style.top = `${dest.y * 100}%`;
    el.style.width = `${dest.w * 100}%`;
    el.style.height = `${dest.h * 100}%`;
  } else {
    el.style.inset = "0";
    el.style.left = "";
    el.style.top = "";
    el.style.width = "100%";
    el.style.height = "100%";
  }
}

interface Framing {
  fit: "cover" | "contain";
  focusCx: number;
  focusCy: number;
  transformCss: string;
}

const ANCHOR_FRACTIONS: Record<string, { cx: number; cy: number }> = {
  left: { cx: 0, cy: 0.5 },
  right: { cx: 1, cy: 0.5 },
  top: { cx: 0.5, cy: 0 },
  bottom: { cx: 0.5, cy: 1 },
  center: { cx: 0.5, cy: 0.5 },
};

/** Resolve fit/focus/CSS-transform ONCE, shared by both the plain `<video>`
 * CSS path (applyFrameStyle) and the WebGL LUT canvas path (which has no
 * native object-fit and must emulate it in the fragment shader) -- one
 * source of truth for preview framing instead of two that could disagree. */
function resolveFraming(clip: Clip, aspect: EditAspect): Framing {
  const t = clip.transform;
  const mid = t?.motion ? sampleMotion(t.motion, t.motion.dur_ms / 2) : null;
  const fit = clip.dest ? "cover" : mid ? "cover" : t?.fit ?? (aspect === "landscape" ? "contain" : "cover");
  const focusFrac = mid
    ? { cx: mid.cx, cy: mid.cy }
    : t?.focus
      ? { cx: t.focus.cx, cy: t.focus.cy }
      : ANCHOR_FRACTIONS[t?.anchor ?? "center"];
  const tf: string[] = [];
  if (t?.rotate) tf.push(`rotate(${t.rotate}deg)`);
  const scale = mid ? mid.scale : t?.zoom && t.zoom > 1 ? t.zoom : 1;
  if (scale > 1) tf.push(`scale(${scale})`);
  return {
    fit,
    focusCx: Math.min(1, Math.max(0, focusFrac.cx)),
    focusCy: Math.min(1, Math.max(0, focusFrac.cy)),
    transformCss: tf.join(" "),
  };
}

/** CSS framing for the visible element, mirroring the render transform chain
 * (rotate -> fit -> zoom) and the motion midpoint preview. */
function applyFrameStyle(el: HTMLVideoElement, clip: Clip, aspect: EditAspect) {
  const f = resolveFraming(clip, aspect);
  el.style.objectFit = f.fit;
  el.style.objectPosition = `${Math.round(f.focusCx * 100)}% ${Math.round(f.focusCy * 100)}%`;
  el.style.transform = f.transformCss;
  applyDestGeometry(el, clip.dest);
}

function isIdentityGrade(grade: ResolvedGrade | undefined): boolean {
  if (!grade) return true;
  if (grade.creative_lut_ref) return false;
  if (grade.soft_local?.vignette && grade.soft_local.vignette.strength > 0) return false;
  const { slope, offset, power, sat } = grade.cdl;
  const eps = 1e-9;
  return (
    slope.every((v) => Math.abs(v - 1) < eps) &&
    offset.every((v) => Math.abs(v) < eps) &&
    power.every((v) => Math.abs(v - 1) < eps) &&
    Math.abs(sat - 1) < eps
  );
}

export function useProgramPlayer(
  resolved: ResolvedTimeline | null,
  urls: Record<string, string>
): ProgramPlayerHandle {
  const ctxRef = useRef<AudioContext | null>(null);
  const masterRef = useRef<GainNode | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const poolRef = useRef<Slot[]>([]);

  const clipsRef = useRef<Clip[]>([]);
  const aspectRef = useRef<EditAspect>("landscape");
  const urlsRef = useRef(urls);
  urlsRef.current = urls;

  // Master clock: program `originMs` corresponds to ctx time `t0`.
  const playingRef = useRef(false);
  const originMs = useRef(0);
  const t0 = useRef(0);
  const mutedRef = useRef(false);
  // While non-null, the clock is HELD at this position until the visible
  // element has a decoded frame (so the playhead never runs over black).
  const pendingStart = useRef<number | null>(null);

  const clips = useMemo(() => buildClips(resolved), [resolved]);
  clipsRef.current = clips;
  aspectRef.current = resolved?.aspect ?? "landscape";

  const ensure = useCallback((): AudioContext | null => {
    if (typeof window === "undefined") return null;
    if (!ctxRef.current) {
      const Ctor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      if (!Ctor) return null;
      const ctx = new Ctor();
      const master = ctx.createGain();
      master.gain.value = mutedRef.current ? 0 : 1;
      master.connect(ctx.destination);
      ctxRef.current = ctx;
      masterRef.current = master;
    }
    const ctx = ctxRef.current;
    const master = masterRef.current;
    const container = containerRef.current;
    if (ctx && master && container && poolRef.current.length === 0) {
      for (let i = 0; i < POOL_SIZE; i++) {
        const el = document.createElement("video");
        // NOT muted: a muted element feeds SILENCE into the MediaElementSource
        // in Chrome. The user's Play gesture satisfies autoplay instead.
        el.muted = false;
        el.playsInline = true;
        el.preload = "auto";
        el.crossOrigin = "anonymous";
        el.setAttribute("aria-hidden", "true");
        el.style.position = "absolute";
        el.style.inset = "0";
        el.style.width = "100%";
        el.style.height = "100%";
        el.style.opacity = "0";
        el.style.transition = "opacity 60ms linear";
        const gain = ctx.createGain();
        gain.gain.value = 0;
        const node = ctx.createMediaElementSource(el);
        node.connect(gain);
        gain.connect(master);
        const slot: Slot = {
          el,
          gain,
          clip: null,
          needsSeek: false,
          wantSec: null,
          queuedSeek: null,
          lutRenderer: null,
          lutUrl: null,
        };
        el.addEventListener("seeked", () => {
          if (slot.queuedSeek != null) {
            const s = slot.queuedSeek;
            slot.queuedSeek = null;
            try {
              el.currentTime = s;
            } catch {
              /* not ready */
            }
          }
        });
        // A park requested before the media was loadable (readyState 0) is
        // deferred and applied here — otherwise the picture stays black while
        // PAUSED (the rAF loop isn't running to retry the seek).
        const applyParkedSeek = () => {
          if (!slot.needsSeek || slot.wantSec == null) return;
          if (el.seeking) {
            slot.queuedSeek = slot.wantSec;
            return;
          }
          try {
            el.currentTime = slot.wantSec;
            slot.needsSeek = false;
          } catch {
            /* still not ready; a later event/frame retries */
          }
        };
        el.addEventListener("loadedmetadata", applyParkedSeek);
        el.addEventListener("loadeddata", applyParkedSeek);
        el.addEventListener("canplay", applyParkedSeek);
        container.appendChild(el);
        poolRef.current.push(slot);
      }
    }
    return ctxRef.current;
  }, []);

  /** Seek a slot, coalescing if a seek is already in flight (stacking seeks on
   * a remote MP4 stalls the decoder). */
  const seekSlot = (slot: Slot, sec: number) => {
    const el = slot.el;
    if (el.seeking) {
      slot.queuedSeek = sec;
      return;
    }
    try {
      el.currentTime = sec;
      slot.queuedSeek = null;
    } catch {
      slot.queuedSeek = sec;
    }
  };

  const assignClip = (slot: Slot, clip: Clip) => {
    slot.clip = clip;
    const url = urlsRef.current[clip.fileId];
    if (url && slot.el.src !== url) slot.el.src = url;
    slot.needsSeek = true;
    slot.wantSec = null;
  };

  /** Draw (or hide) a slot's graded-preview canvas for the current frame.
   * Lazily creates the WebGL renderer on first use; falls back to showing
   * the plain (ungraded) `<video>` if WebGL2 is unavailable or the cube
   * hasn't loaded yet, so grading is a pure enhancement, never a blocker. */
  const updateLutOverlay = (slot: Slot, clip: Clip, visible: boolean): boolean => {
    const graded = visible && !isIdentityGrade(clip.grade);
    if (!graded) {
      if (slot.lutRenderer) slot.lutRenderer.canvas.style.opacity = "0";
      return false;
    }
    const cube = getGradeCube(clip.grade);
    if (!cube) return false; // still fetching -- caller shows the raw video meanwhile

    if (!slot.lutRenderer) {
      const renderer = createLutRenderer();
      if (!renderer) return false; // no WebGL2 -- permanently fall back for this slot
      const c = renderer.canvas;
      c.style.position = "absolute";
      c.style.pointerEvents = "none";
      c.style.opacity = "0";
      c.style.transition = "opacity 60ms linear";
      containerRef.current?.appendChild(c);
      slot.lutRenderer = renderer;
    }
    const renderer = slot.lutRenderer;
    const url = clip.grade ? gradeCubeUrl(clip.grade) : null;
    if (url && slot.lutUrl !== url) {
      renderer.setLut(cube.grid, cube.size, url);
      slot.lutUrl = url;
    }
    const framing = resolveFraming(clip, aspectRef.current);
    applyDestGeometry(renderer.canvas, clip.dest);
    renderer.canvas.style.transform = framing.transformCss;
    renderer.canvas.style.zIndex = String(clip.z);
    renderer.draw(
      slot.el, framing.fit, { cx: framing.focusCx, cy: framing.focusCy },
      clip.grade?.soft_local?.vignette
    );
    renderer.canvas.style.opacity = "1";
    return true;
  };

  /** Full per-frame reconcile: assign needed clips to slots, then drive each
   * slot's position/play/gain/visibility from the program time `t`. */
  const reconcile = useCallback((t: number, playing: boolean) => {
    const ctx = ensure();
    if (!ctx) return;
    const pool = poolRef.current;
    if (pool.length === 0) return;
    const cs = clipsRef.current;
    const freshById = new Map(cs.map((c) => [c.id, c] as const));

    const isActive = (c: Clip) => c.progStart <= t && t < c.progEnd;
    const isPrewarm = (c: Clip) => playing && t < c.progStart && c.progStart <= t + PREWARM_MS;
    const active = cs.filter(isActive);
    const prewarm = cs.filter(isPrewarm);
    const needed = [...active, ...prewarm];
    const neededIds = new Set(needed.map((c) => c.id));

    // What shows: split/PiP CELLS (every active picture with a dest rect) paint
    // at once; otherwise the single top-z active picture fills the frame.
    const placed = active.filter((c) => c.hasVideo && c.dest);
    let visibleIds: Set<string>;
    if (placed.length) {
      visibleIds = new Set(placed.map((c) => c.id));
    } else {
      let top: Clip | null = null;
      for (const c of active) if (c.hasVideo && (!top || c.z > top.z)) top = c;
      visibleIds = new Set(top ? [top.id] : []);
    }

    const slotFor = (id: string) => pool.find((s) => s.clip?.id === id);
    const isFree = (s: Slot) => !s.clip || !neededIds.has(s.clip.id);

    // Assign active first (must never be evicted), then pre-warm into leftovers.
    for (const clip of needed) {
      if (slotFor(clip.id)) continue;
      const slot =
        pool.find((s) => isFree(s) && s.clip?.fileId === clip.fileId) ??
        pool.find((s) => isFree(s));
      if (slot) assignClip(slot, clip);
    }

    for (const slot of pool) {
      let clip = slot.clip;
      if (!clip || !neededIds.has(clip.id)) {
        if (!slot.el.paused) slot.el.pause();
        slot.gain.gain.setTargetAtTime(0, ctx.currentTime, GAIN_TC);
        slot.el.style.opacity = "0";
        if (slot.lutRenderer) slot.lutRenderer.canvas.style.opacity = "0";
        continue;
      }
      // Refresh the slot to the LATEST resolved clip for this id. Slots are keyed
      // by a grade-independent id (`c_<layer_id>`), so a clip already loaded in a
      // slot is never re-`assignClip`d on a pure property change -- without this
      // refresh its grade/gain/transform/trim would stay pinned to the values
      // captured at first assignment (the LUT in particular freezes on the first
      // look). Grade/gain/transform/dest are read fresh below from `slot.clip`; only
      // a source file or position change needs a reseek (grade changes must not).
      const fresh = freshById.get(clip.id);
      if (fresh && fresh !== clip) {
        const fileChanged = fresh.fileId !== clip.fileId;
        const posChanged = fresh.srcInMs !== clip.srcInMs || fresh.progStart !== clip.progStart;
        slot.clip = fresh;
        clip = fresh;
        if (fileChanged) {
          const u = urlsRef.current[fresh.fileId];
          if (u) slot.el.src = u;
          slot.needsSeek = true;
        } else if (posChanged) {
          slot.needsSeek = true;
        }
      }
      const url = urlsRef.current[clip.fileId];
      if (url && slot.el.src !== url) slot.el.src = url;

      const active_ = isActive(clip);
      const rel = Math.max(0, t - clip.progStart);
      const want = (clip.srcInMs + rel) / 1000;

      if (slot.needsSeek) {
        slot.wantSec = want;
        // Only consume the seek once the element can actually accept it; before
        // HAVE_METADATA the assignment is dropped, so leave needsSeek set and let
        // the loadedmetadata/loadeddata listener (or a later frame) apply it.
        if (slot.el.readyState >= 1) {
          seekSlot(slot, want);
          slot.needsSeek = false;
        }
      } else if (active_ && playing) {
        if (Math.abs(slot.el.currentTime - want) > DRIFT_S) seekSlot(slot, want);
      }

      const shouldPlay = playing && active_;
      if (shouldPlay) {
        if (slot.el.paused) void slot.el.play().catch(() => {});
      } else if (!slot.el.paused) {
        slot.el.pause();
      }

      const g = active_ && clip.hasAudio ? dbToGain(clip.gainDb) : 0;
      slot.gain.gain.setTargetAtTime(g, ctx.currentTime, GAIN_TC);

      const isVisible = active_ && visibleIds.has(clip.id);
      if (isVisible) {
        applyFrameStyle(slot.el, clip, aspectRef.current);
        // Stack cells by layer z (PiP inset over base); split cells don't overlap.
        slot.el.style.zIndex = String(clip.z);
        slot.el.style.opacity = "1";
      } else {
        slot.el.style.opacity = "0";
      }
      // Graded clips paint through a WebGL LUT canvas on top; hides the raw
      // video underneath only once the graded frame is actually ready, so a
      // slow cube fetch never shows a black flash.
      const showingGraded = updateLutOverlay(slot, clip, isVisible);
      if (showingGraded) slot.el.style.opacity = "0";
    }
  }, [ensure]);

  const nowMs = useCallback((): number => {
    if (pendingStart.current != null) return pendingStart.current;
    const ctx = ctxRef.current;
    if (!playingRef.current || !ctx) return originMs.current;
    return originMs.current + Math.max(0, ctx.currentTime - t0.current) * 1000;
  }, []);

  const sync = useCallback(
    (t: number, playing: boolean) => {
      const ctx = ensure();
      if (!ctx) return;
      // Gated start: park + pre-seek at the pending position, and only release
      // the clock once the visible element has a decoded frame.
      if (playing && pendingStart.current != null) {
        const at = pendingStart.current;
        reconcile(at, false);
        const vis = poolRef.current.find(
          (s) => s.el.style.opacity === "1" && s.clip
        );
        const ready = !vis || (vis.el.readyState >= 2 && !vis.el.seeking);
        if (ready) {
          originMs.current = at;
          t0.current = ctx.currentTime + START_LEAD_S;
          pendingStart.current = null;
        } else {
          return; // hold the clock another frame
        }
      }
      reconcile(t, playing);
    },
    [ensure, reconcile]
  );

  const play = useCallback(
    (fromMs: number) => {
      const ctx = ensure();
      if (!ctx) return;
      if (ctx.state === "suspended") void ctx.resume();
      playingRef.current = true;
      originMs.current = Math.max(0, fromMs);
      pendingStart.current = Math.max(0, fromMs); // gate until first frame ready
    },
    [ensure]
  );

  const pause = useCallback(() => {
    originMs.current = nowMs();
    playingRef.current = false;
    pendingStart.current = null;
    for (const s of poolRef.current) if (!s.el.paused) s.el.pause();
  }, [nowMs]);

  const seek = useCallback(
    (toMs: number, playing: boolean) => {
      originMs.current = Math.max(0, toMs);
      if (playing) {
        pendingStart.current = Math.max(0, toMs); // re-gate on the new frame
      } else {
        pendingStart.current = null;
        for (const s of poolRef.current) s.needsSeek = true;
      }
    },
    []
  );

  const setMuted = useCallback((m: boolean) => {
    mutedRef.current = m;
    const ctx = ctxRef.current;
    if (masterRef.current && ctx) {
      masterRef.current.gain.setTargetAtTime(m ? 0 : 1, ctx.currentTime, 0.01);
    }
  }, []);

  const stop = useCallback(() => {
    playingRef.current = false;
    pendingStart.current = null;
    originMs.current = 0;
    for (const s of poolRef.current) {
      s.clip = null;
      s.needsSeek = false;
      s.wantSec = null;
      s.queuedSeek = null;
      try {
        s.el.pause();
        s.el.removeAttribute("src");
        s.el.load();
      } catch {
        /* ignore */
      }
      s.el.style.opacity = "0";
      if (s.lutRenderer) s.lutRenderer.canvas.style.opacity = "0";
      const ctx = ctxRef.current;
      if (ctx) s.gain.gain.setTargetAtTime(0, ctx.currentTime, 0.005);
    }
  }, []);

  const prepare = useCallback(() => {
    ensure();
  }, [ensure]);

  const attachContainer = useCallback((el: HTMLDivElement | null) => {
    containerRef.current = el;
    if (el) ensure();
  }, [ensure]);

  // Release the AudioContext + pooled elements on unmount.
  useEffect(() => {
    const pool = poolRef.current;
    return () => {
      for (const s of pool) {
        try {
          s.el.pause();
          s.el.removeAttribute("src");
          s.el.remove();
        } catch {
          /* ignore */
        }
        if (s.lutRenderer) {
          try {
            s.lutRenderer.dispose();
            s.lutRenderer.canvas.remove();
          } catch {
            /* ignore */
          }
        }
      }
      poolRef.current = [];
      const ctx = ctxRef.current;
      ctxRef.current = null;
      masterRef.current = null;
      if (ctx) void ctx.close().catch(() => {});
    };
  }, []);

  // IMPORTANT: return a STABLE handle. The consumer's rAF `loop` and its
  // play/pause effect depend on this object; if its identity changed every
  // render (which happens ~every frame while playing, as the published time
  // re-renders the monitor), those effects would tear down and re-run each
  // frame — restarting playback and re-gating the clock continuously, which
  // reads as "stuck + silent". All methods are useCallback-stable, so this memo
  // never actually changes after mount.
  return useMemo(
    () => ({
      attachContainer,
      prepare,
      sync,
      play,
      pause,
      seek,
      nowMs,
      setMuted,
      stop,
    }),
    [attachContainer, prepare, sync, play, pause, seek, nowMs, setMuted, stop]
  );
}
