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
  EditDocument,
  EditOperation,
  EditSegment,
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
export function resolveTimeline(
  document: Pick<EditDocument, "timeline" | "operations">,
  durations: Durations = {}
): ResolvedTimeline {
  const timeline = document.timeline ?? [];
  const operations = document.operations ?? [];
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

  return { duration_ms: total, video_layers: video, audio_layers: audio };
}
