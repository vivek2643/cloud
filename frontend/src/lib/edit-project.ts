/**
 * Generic multi-track PROJECT view of an edit document.
 *
 * The persisted/render truth is still the spine (`timeline`) + typed
 * `operations` (so backend render parity is untouched). This module presents
 * that exact same data as a GENERIC, NLE-style set of TRACKS holding CLIPS —
 * the way every video editor models a timeline — with no "coverage / angle"
 * vocabulary leaking into the UI. Each clip keeps `origin` provenance so edits
 * map straight back to the right spine segment or operation.
 *
 * Mapping (lossless, both directions):
 *   - base video track  ← spine segments (gapless, owns the program clock)
 *   - base audio track  ← the spine's coupled dialogue
 *   - upper video tracks ← `place_video` ops, stacked by z
 *   - audio tracks       ← `place_audio` ops, grouped by role
 *
 * Spatial layouts (PiP / split-screen) are intentionally NOT modelled yet, but
 * the substrate already admits them: a clip carries a `transform` slot and the
 * compositor is z-ordered, so they become additive later — not a rewrite.
 */
import type { EditAspect, EditOperation, EditSegment, EditOperationType } from "./api";

export type TrackKind = "video" | "audio";

export interface ProjectTrack {
  id: string;
  kind: TrackKind;
  label: string;
  /** The spine (video) / coupled dialogue (audio) — the gapless base. */
  isBase: boolean;
  /** Video stacking band this track represents (higher paints on top). */
  z: number;
  /** Semantic grouping key for a non-base AUDIO track ("music"/"sfx"/...);
   * undefined for video tracks and the base audio track. */
  role?: string;
}

export type ClipOrigin =
  | { kind: "spine"; segId: string }
  | { kind: "op"; opId: string; opType: EditOperationType };

export interface ProjectClip {
  id: string;
  trackId: string;
  kind: TrackKind;
  sourceFileId: string;
  srcInMs: number;
  srcOutMs: number;
  progStartMs: number;
  progEndMs: number;
  label: string;
  origin: ClipOrigin;
  /** Free horizontal repositioning (V2 cutaway / A2 audio ops). Base clips are fixed. */
  movable: boolean;
  /** Edge trimming. */
  trimmable: boolean;
  z: number;
  color: string;
  gainDb?: number;
  muted?: boolean;
}

export interface EditProject {
  tracks: ProjectTrack[];
  clips: ProjectClip[];
  durationMs: number;
  aspect: EditAspect;
}

// Generic, non-semantic palette (no coverage/angle meaning attached).
const COLOR_BASE_VIDEO = "var(--accent)";
const COLOR_BASE_AUDIO = "#2bb673";
const VIDEO_COLORS = ["#7c5cff", "#1f9ed1", "#9b6dff", "#4a8cff"];
const AUDIO_COLORS = ["#3a86e0", "#e0883a", "#2bb6a8", "#c05ad1"];

function spineDuration(seg: EditSegment): number {
  return Math.max(0, Math.round(seg.out_ms) - Math.round(seg.in_ms));
}

/**
 * Derive the generic track/clip project from the document's spine + operations.
 * Pure + cheap — call it inside a `useMemo`.
 */
export function documentToProject(
  timeline: EditSegment[],
  operations: EditOperation[],
  aspect: EditAspect
): EditProject {
  const tracks: ProjectTrack[] = [];
  const clips: ProjectClip[] = [];

  // --- base video (spine), laid end-to-end; owns the clock ---
  const baseVideo: ProjectTrack = {
    id: "V1",
    kind: "video",
    label: "V1",
    isBase: true,
    z: 0,
  };
  let t = 0;
  for (const seg of timeline) {
    const dur = spineDuration(seg);
    clips.push({
      id: `seg:${seg.seg_id}`,
      trackId: baseVideo.id,
      kind: "video",
      sourceFileId: seg.file_id,
      srcInMs: Math.round(seg.in_ms),
      srcOutMs: Math.round(seg.out_ms),
      progStartMs: t,
      progEndMs: t + dur,
      label: seg.file_id.slice(0, 4),
      origin: { kind: "spine", segId: seg.seg_id },
      movable: false,
      trimmable: true,
      z: 0,
      color: COLOR_BASE_VIDEO,
    });
    t += dur;
  }
  const durationMs = t;

  // --- upper video tracks: place_video ops, stacked by z ---
  const videoOps = operations.filter((o) => o.type === "place_video");
  const videoZs = Array.from(
    new Set(videoOps.map((o) => Math.round(o.z ?? 10)))
  ).sort((a, b) => a - b);
  const zToTrack = new Map<number, ProjectTrack>();
  videoZs.forEach((z, i) => {
    const track: ProjectTrack = {
      id: `V${i + 2}`,
      kind: "video",
      label: `V${i + 2}`,
      isBase: false,
      z,
    };
    zToTrack.set(z, track);
  });
  for (const op of videoOps) {
    const z = Math.round(op.z ?? 10);
    const track = zToTrack.get(z);
    if (!track) continue;
    const from = Math.round(op.from_ms ?? 0);
    const to = Math.round(op.to_ms ?? 0);
    const idx = Math.max(0, track.z % VIDEO_COLORS.length);
    clips.push({
      id: `op:${op.op_id}`,
      trackId: track.id,
      kind: "video",
      sourceFileId: op.source_file_id ?? "",
      srcInMs: Math.round(op.src_in_ms ?? 0),
      srcOutMs: Math.round(op.src_out_ms ?? 0),
      progStartMs: from,
      progEndMs: to,
      label: (op.source_file_id ?? "clip").slice(0, 4),
      origin: { kind: "op", opId: op.op_id, opType: op.type },
      movable: true,
      trimmable: true,
      z,
      color: VIDEO_COLORS[idx],
    });
  }

  // --- base audio: the spine's coupled dialogue (selecting maps to the seg) ---
  const baseAudio: ProjectTrack = {
    id: "A1",
    kind: "audio",
    label: "A1",
    isBase: true,
    z: 0,
  };
  let ta = 0;
  for (const seg of timeline) {
    const dur = spineDuration(seg);
    clips.push({
      id: `dlg:${seg.seg_id}`,
      trackId: baseAudio.id,
      kind: "audio",
      sourceFileId: seg.file_id,
      srcInMs: Math.round(seg.in_ms),
      srcOutMs: Math.round(seg.out_ms),
      progStartMs: ta,
      progEndMs: ta + dur,
      label: "dlg",
      origin: { kind: "spine", segId: seg.seg_id },
      movable: false,
      trimmable: false,
      z: 0,
      color: COLOR_BASE_AUDIO,
    });
    ta += dur;
  }

  // --- audio tracks: place_audio ops grouped by role ---
  const audioOps = operations.filter((o) => o.type === "place_audio");
  const roles: string[] = [];
  for (const op of audioOps) {
    const r = op.role || "music";
    if (!roles.includes(r)) roles.push(r);
  }
  const roleToTrack = new Map<string, ProjectTrack>();
  roles.forEach((role, i) => {
    roleToTrack.set(role, {
      id: `A${i + 2}`,
      kind: "audio",
      label: `A${i + 2}`,
      isBase: false,
      z: 0,
      role,
    });
  });
  audioOps.forEach((op) => {
    const role = op.role || "music";
    const track = roleToTrack.get(role);
    if (!track) return;
    const from = Math.round(op.from_ms ?? 0);
    const to = Math.round(op.to_ms ?? 0);
    const i = roles.indexOf(role);
    const muted = (op.gain_db ?? 0) <= -119;
    clips.push({
      id: `op:${op.op_id}`,
      trackId: track.id,
      kind: "audio",
      sourceFileId: op.source_file_id ?? "",
      srcInMs: Math.round(op.src_in_ms ?? 0),
      srcOutMs: Math.round(op.src_out_ms ?? 0),
      progStartMs: from,
      progEndMs: to,
      label: role,
      origin: { kind: "op", opId: op.op_id, opType: "place_audio" },
      movable: true,
      trimmable: true,
      z: 0,
      color: AUDIO_COLORS[i % AUDIO_COLORS.length],
      gainDb: op.gain_db ?? 0,
      muted,
    });
  });

  // Display order, top → bottom: upper video (high z first), base video,
  // base audio, then bed audio tracks.
  const upperVideo = Array.from(zToTrack.values()).sort((a, b) => b.z - a.z);
  const bedAudio = roles
    .map((r) => roleToTrack.get(r))
    .filter((x): x is ProjectTrack => !!x);
  tracks.push(...upperVideo, baseVideo, baseAudio, ...bedAudio);

  return { tracks, clips, durationMs, aspect };
}

// --------------------------------------------------------------------------
// Snapping (2.4): candidate program-ms targets + nearest-within-threshold.
// --------------------------------------------------------------------------

/** Every clip edge across every track, plus 0/total and any extra candidates
 * (playhead, markers, in/out marks). Excludes one clip's own edges (the clip
 * being dragged) so it never snaps to itself. */
export function collectSnapTargets(
  project: EditProject,
  opts: { excludeClipId?: string; extra?: number[] } = {}
): number[] {
  const set = new Set<number>();
  for (const c of project.clips) {
    if (c.id === opts.excludeClipId) continue;
    set.add(Math.round(c.progStartMs));
    set.add(Math.round(c.progEndMs));
  }
  set.add(0);
  set.add(Math.round(project.durationMs));
  for (const e of opts.extra ?? []) set.add(Math.round(e));
  return Array.from(set).sort((a, b) => a - b);
}

/** Nearest candidate within `thresholdPx` (screen space) of `ms`, else `ms`
 * unchanged. `pxPerMs` converts the pixel threshold into the ms domain. */
export function snapValue(
  ms: number,
  targets: number[],
  pxPerMs: number,
  thresholdPx = 8
): { value: number; snappedTo: number | null } {
  if (pxPerMs <= 0 || targets.length === 0) return { value: ms, snappedTo: null };
  const thresholdMs = thresholdPx / pxPerMs;
  let best = ms;
  let bestDist = thresholdMs;
  let snappedTo: number | null = null;
  for (const t of targets) {
    const d = Math.abs(t - ms);
    if (d <= bestDist) {
      bestDist = d;
      best = t;
      snappedTo = t;
    }
  }
  return { value: best, snappedTo };
}
