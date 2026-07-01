/**
 * Client-side timeline resolver — a faithful port of the authoritative backend
 * `backend/app/services/l3/layers.py::resolve`. It compiles the SPINE
 * (`document.timeline`) + typed OPERATIONS (`document.operations`) into the flat
 * set of video/audio LAYERS that say exactly what is shown/heard at every
 * program instant.
 *
 * Keeping this in lockstep with the Python resolver is what guarantees
 * "what you preview is what you render". A parity test (resolve the same doc on
 * both sides and diff) protects against drift — see the backend parity fixture.
 */
import type {
  DestRect,
  EditAspect,
  EditDocument,
  EditOperation,
  EditSegment,
  LayerAnchor,
  LayerFit,
  LayerMotion,
  LayerTransform,
  LayoutRegion,
  MotionPoint,
  ResolvedAudioLayer,
  ResolvedTimeline,
  ResolvedVideoLayer,
} from "./api";

// Z bands so layer kinds stack predictably regardless of insertion order.
export const Z_SPINE_VIDEO = 0;
export const Z_COVERAGE = 10;
export const DEFAULT_LAYOUT = "full_frame";

export const ROLE_DIALOGUE = "dialogue";
export const ROLE_MUSIC = "music";
export const ROLE_SFX = "sfx";

/** file_id -> source duration in ms (used only to clamp split-edit extensions). */
export type Durations = Record<string, number>;

interface SpineSpan {
  seg: EditSegment;
  progStart: number;
  progEnd: number;
}

function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

function overlaps(aS: number, aE: number, bS: number, bE: number): boolean {
  return aS < bE && aE > bS;
}

/** Lay the spine segments end-to-end on the program clock. */
function spineSpans(timeline: EditSegment[]): { spans: SpineSpan[]; total: number } {
  const spans: SpineSpan[] = [];
  let t = 0;
  for (const seg of timeline) {
    const dur = Math.max(0, Math.round(seg.out_ms) - Math.round(seg.in_ms));
    spans.push({ seg, progStart: t, progEnd: t + dur });
    t += dur;
  }
  return { spans, total: t };
}

/**
 * J/L cuts: offset the AUDIO boundary at a seam from the video boundary.
 * Mutates the per-span dialogue layers in place (index-aligned with `spans`).
 */
function applySplitEdits(
  spans: SpineSpan[],
  audio: ResolvedAudioLayer[],
  operations: EditOperation[],
  durations: Durations
): void {
  const bySeg = new Map<string, number>();
  spans.forEach((s, i) => bySeg.set(s.seg.seg_id, i));

  for (const op of operations) {
    if (op.type !== "split_edit") continue;
    const i = op.seam_seg_id != null ? bySeg.get(op.seam_seg_id) : undefined;
    if (i == null || i === 0 || i >= audio.length) continue;
    const offset = Math.round(op.audio_offset_ms ?? 0);
    if (offset === 0) continue;

    const prevA = audio[i - 1];
    const curA = audio[i];
    const boundary = curA.prog_start_ms + offset;

    const prevRoom = durations[prevA.source_file_id] ?? prevA.src_out_ms;
    prevA.prog_end_ms = Math.max(prevA.prog_start_ms, boundary);
    prevA.src_out_ms = clamp(prevA.src_out_ms + offset, prevA.src_in_ms, prevRoom);

    curA.prog_start_ms = Math.max(0, boundary);
    curA.src_in_ms = clamp(curA.src_in_ms + offset, 0, curA.src_out_ms);
  }
}

/** Resolve auto-duck (beds under live dialogue) + explicit level automation. */
function applyLevels(audio: ResolvedAudioLayer[], operations: EditOperation[]): void {
  const dialogueSpans: [number, number][] = audio
    .filter((a) => a.role === ROLE_DIALOGUE && a.kind === "spine")
    .map((a) => [a.prog_start_ms, a.prog_end_ms]);

  const duckByOp = new Map<string, number>();
  for (const op of operations) {
    if (op.type === "place_audio" && op.op_id) {
      duckByOp.set(op.op_id, Number(op.duck_db ?? 0));
    }
  }

  for (const a of audio) {
    if (a.role === ROLE_DIALOGUE) continue;
    const duck = a.op_id ? duckByOp.get(a.op_id) ?? 0 : 0;
    if (
      duck < 0 &&
      dialogueSpans.some(([ds, de]) => overlaps(a.prog_start_ms, a.prog_end_ms, ds, de))
    ) {
      a.duck_db = duck;
    }
  }

  for (const op of operations) {
    if (op.type !== "level") continue;
    const role = op.role;
    const fr = Math.round(op.from_ms ?? 0);
    const to = Math.round(op.to_ms ?? 0);
    for (const a of audio) {
      if (role && a.role !== role) continue;
      if (!overlaps(a.prog_start_ms, a.prog_end_ms, fr, to)) continue;
      if (op.mute) a.gain_db = -120;
      else if (op.gain_db != null) a.gain_db = Number(op.gain_db);
    }
  }
}

/**
 * Compile spine + operations into the flat resolved layer set.
 * Mirrors `layers.resolve`. `durations` is optional (file_id -> ms); missing
 * entries skip the split-edit clamp, exactly like the backend.
 */
const ASPECTS: readonly EditAspect[] = ["landscape", "portrait", "square"];
const FITS: readonly LayerFit[] = ["cover", "contain"];
const ANCHORS: readonly LayerAnchor[] = ["center", "left", "right", "top", "bottom"];
const ROTATIONS = [0, 90, 180, 270] as const;

/** Automatic fit for a delivery aspect — vertical/square FILL, landscape fits.
 * Mirrors backend `layers.default_fit`. */
function defaultFit(aspect: EditAspect): LayerFit {
  return aspect === "portrait" || aspect === "square" ? "cover" : "contain";
}

const clamp01 = (v: number) => (v < 0 ? 0 : v > 1 ? 1 : v);

function normMotionPoint(p: unknown): MotionPoint | null {
  if (!p || typeof p !== "object") return null;
  const o = p as Record<string, unknown>;
  const scale = Number(o.scale);
  const cx = Number(o.cx);
  const cy = Number(o.cy);
  if (![scale, cx, cy].every(Number.isFinite)) return null;
  return { scale: Math.max(1, scale), cx: clamp01(cx), cy: clamp01(cy) };
}

/** Validate + clamp a motion path (drop degenerate/no-move); mirrors
 * backend `layers.normalize_motion`. */
function normalizeMotion(motion: LayerMotion | undefined): LayerMotion | null {
  if (!motion) return null;
  const from = normMotionPoint(motion.from);
  const to = normMotionPoint(motion.to);
  if (!from || !to) return null;
  const durMs = Math.max(1, Math.trunc(Number(motion.dur_ms) || 0));
  const ease: "linear" | "smooth" = motion.ease === "smooth" ? "smooth" : "linear";
  const moves =
    Math.abs(from.scale - to.scale) > 1e-4 ||
    Math.abs(from.cx - to.cx) > 1e-4 ||
    Math.abs(from.cy - to.cy) > 1e-4;
  if (!moves) return null;
  return { from, to, ease, dur_ms: durMs };
}

/** {scale,cx,cy} of a motion path at relMs into the layer; mirrors backend
 * `layers.sample_motion`. The single shared closed form preview + render use. */
export function sampleMotion(motion: LayerMotion, relMs: number): MotionPoint {
  const dur = Math.max(1, Math.trunc(motion.dur_ms || 1));
  let p = clamp01(relMs / dur);
  if (motion.ease === "smooth") p = 3 * p * p - 2 * p * p * p;
  const { from, to } = motion;
  return {
    scale: from.scale + (to.scale - from.scale) * p,
    cx: from.cx + (to.cx - from.cx) * p,
    cy: from.cy + (to.cy - from.cy) * p,
  };
}

/** Deterministic framing solver — faithful port of `layers.solve_transform`. */
function solveTransform(
  aspect: EditAspect,
  formatFit: LayerFit | undefined,
  override?: LayerTransform
): LayerTransform {
  const fit: LayerFit = formatFit && FITS.includes(formatFit) ? formatFit : defaultFit(aspect);
  const t: LayerTransform = {
    rotate: 0,
    fit,
    anchor: "center",
    zoom: 1,
    dest: "full",
  };
  if (override) {
    if (override.rotate != null) {
      const rot = (((Number(override.rotate) % 360) + 360) % 360) as 0 | 90 | 180 | 270;
      if (ROTATIONS.includes(rot)) t.rotate = rot;
    }
    if (override.fit && FITS.includes(override.fit)) t.fit = override.fit;
    if (override.anchor && ANCHORS.includes(override.anchor)) t.anchor = override.anchor;
    if (override.zoom != null && Number.isFinite(Number(override.zoom))) {
      t.zoom = Math.max(1, Number(override.zoom));
    }
    const f = override.focus;
    if (f && Number.isFinite(Number(f.cx)) && Number.isFinite(Number(f.cy))) {
      t.focus = {
        cx: Math.min(1, Math.max(0, Number(f.cx))),
        cy: Math.min(1, Math.max(0, Number(f.cy))),
      };
    }
    const m = normalizeMotion(override.motion);
    if (m) t.motion = m;
  }
  return t;
}

// --- spatial layout: split-screen / PiP templates (mirror layers.py) ---
const round4 = (v: number) => Math.round(v * 1e4) / 1e4;

const LAYOUT_TEMPLATES: Record<string, Record<string, [number, number, number, number]>> = {
  split_h: { left: [0, 0, 0.5, 1], right: [0.5, 0, 0.5, 1] },
  split_v: { top: [0, 0, 1, 0.5], bottom: [0, 0.5, 1, 0.5] },
  pip: { base: [0, 0, 1, 1], inset: [0.66, 0.66, 0.32, 0.32] },
};

/** A layout template -> {cell: dest rect}; {} for an unknown template. */
export function solveLayout(template: string): Record<string, DestRect> {
  const cells = LAYOUT_TEMPLATES[template];
  if (!cells) return {};
  const out: Record<string, DestRect> = {};
  for (const [name, [x, y, w, h]] of Object.entries(cells)) {
    out[name] = { x: round4(x), y: round4(y), w: round4(w), h: round4(h) };
  }
  return out;
}

/** True when a transform `dest` is a real sub-rect (split/PiP cell). */
export function isRect(dest: LayerTransform["dest"]): dest is DestRect {
  return (
    !!dest &&
    typeof dest === "object" &&
    ["x", "y", "w", "h"].every((k) => k in (dest as unknown as Record<string, unknown>))
  );
}

/** A sub-span [ps, pe) of a video layer, source-mapped, with an optional dest
 * rect stamped on (fit forced to cover). Mirrors layers._slice_video. */
function sliceVideo(
  v: ResolvedVideoLayer,
  ps: number,
  pe: number,
  dest: DestRect | null
): ResolvedVideoLayer {
  const tf: LayerTransform = { ...(v.transform ?? {}) };
  if (dest) {
    tf.dest = dest;
    tf.fit = "cover";
  }
  const suffix = ps === v.prog_start_ms && pe === v.prog_end_ms ? "" : `__${ps}`;
  return {
    ...v,
    layer_id: `${v.layer_id}${suffix}`,
    src_in_ms: v.src_in_ms + (ps - v.prog_start_ms),
    src_out_ms: v.src_in_ms + (pe - v.prog_start_ms),
    prog_start_ms: ps,
    prog_end_ms: pe,
    transform: tf,
  };
}

/** Stamp `dest` onto the spine picture across [f, t), slicing straddling spine
 * layers. Mirrors layers._dest_spine_window. */
function destSpineWindow(
  video: ResolvedVideoLayer[],
  f: number,
  t: number,
  dest: DestRect
): ResolvedVideoLayer[] {
  const out: ResolvedVideoLayer[] = [];
  for (const v of video) {
    if (v.kind !== "spine" || v.prog_end_ms <= f || v.prog_start_ms >= t) {
      out.push(v);
      continue;
    }
    const os = Math.max(v.prog_start_ms, f);
    const oe = Math.min(v.prog_end_ms, t);
    if (v.prog_start_ms < os) out.push(sliceVideo(v, v.prog_start_ms, os, null));
    out.push(sliceVideo(v, os, oe, dest));
    if (oe < v.prog_end_ms) out.push(sliceVideo(v, oe, v.prog_end_ms, null));
  }
  return out;
}

/** Turn each layout region into dest rects on the layers it names. Mirrors
 * layers._apply_layout_regions. */
function applyLayoutRegions(
  video: ResolvedVideoLayer[],
  regions: LayoutRegion[]
): ResolvedVideoLayer[] {
  let out = video;
  for (const r of regions) {
    const rects = solveLayout(r.template);
    if (!Object.keys(rects).length) continue;
    const f = Math.round(r.from_ms);
    const t = Math.round(r.to_ms);
    if (t <= f) continue;
    for (const [cell, sel] of Object.entries(r.cells ?? {})) {
      const rect = rects[cell];
      if (!rect) continue;
      const layer = sel?.layer;
      if (layer === "spine") {
        out = destSpineWindow(out, f, t, rect);
      } else if (layer) {
        for (const v of out) {
          if (v.op_id === layer || v.layer_id === layer) {
            v.transform = { ...(v.transform ?? {}), dest: rect, fit: "cover" };
          }
        }
      }
    }
  }
  return out;
}

export function resolveTimeline(
  document: Pick<EditDocument, "timeline" | "operations" | "format" | "layout_regions">,
  durations: Durations = {}
): ResolvedTimeline {
  const timeline = document.timeline ?? [];
  const operations = document.operations ?? [];
  const a = document.format?.aspect;
  const aspect: EditAspect = a && ASPECTS.includes(a) ? a : "landscape";
  const formatFit = document.format?.fit;
  const { spans, total } = spineSpans(timeline);

  const video: ResolvedVideoLayer[] = [];
  const audio: ResolvedAudioLayer[] = [];

  // base spine layers: coupled video + dialogue, one per span
  for (const s of spans) {
    const seg = s.seg;
    video.push({
      layer_id: `v_${seg.seg_id}`,
      source_file_id: seg.file_id,
      src_in_ms: Math.round(seg.in_ms),
      src_out_ms: Math.round(seg.out_ms),
      prog_start_ms: s.progStart,
      prog_end_ms: s.progEnd,
      z: Z_SPINE_VIDEO,
      layout: DEFAULT_LAYOUT,
      opacity: 1,
      kind: "spine",
      op_id: null,
      transform: solveTransform(aspect, formatFit, seg.transform),
    });
    audio.push({
      layer_id: `a_${seg.seg_id}`,
      role: ROLE_DIALOGUE,
      source_file_id: seg.file_id,
      src_in_ms: Math.round(seg.in_ms),
      src_out_ms: Math.round(seg.out_ms),
      prog_start_ms: s.progStart,
      prog_end_ms: s.progEnd,
      gain_db: 0,
      duck_db: 0,
      kind: "spine",
      op_id: null,
    });
  }

  applySplitEdits(spans, audio, operations, durations);

  for (const op of operations) {
    if (op.type === "place_video") {
      video.push({
        layer_id: op.op_id,
        source_file_id: op.source_file_id ?? "",
        src_in_ms: Math.round(op.src_in_ms ?? 0),
        src_out_ms: Math.round(op.src_out_ms ?? 0),
        prog_start_ms: Math.round(op.from_ms ?? 0),
        prog_end_ms: Math.round(op.to_ms ?? 0),
        z: Math.round(op.z ?? Z_COVERAGE),
        layout: op.layout ?? DEFAULT_LAYOUT,
        opacity: Number(op.opacity ?? 1),
        kind: "coverage",
        op_id: op.op_id,
        transform: solveTransform(aspect, formatFit, op.transform),
      });
    } else if (op.type === "place_audio") {
      audio.push({
        layer_id: op.op_id,
        role: op.role ?? ROLE_MUSIC,
        source_file_id: op.source_file_id ?? "",
        src_in_ms: Math.round(op.src_in_ms ?? 0),
        src_out_ms: Math.round(op.src_out_ms ?? 0),
        prog_start_ms: Math.round(op.from_ms ?? 0),
        prog_end_ms: Math.round(op.to_ms ?? 0),
        gain_db: Number(op.gain_db ?? 0),
        duck_db: 0,
        kind: op.audio_kind ?? "bed",
        op_id: op.op_id,
      });
    }
  }

  applyLevels(audio, operations);

  const regions = document.layout_regions ?? [];
  const videoOut = regions.length ? applyLayoutRegions(video, regions) : video;

  return { duration_ms: total, video_layers: videoOut, audio_layers: audio, aspect };
}
