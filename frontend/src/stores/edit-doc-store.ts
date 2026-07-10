/**
 * Live working-document store for the L3 editor.
 *
 * This is the single source of truth the timeline editor MUTATES and the
 * preview READS. Edits land here instantly (no save round-trip), so the preview
 * reflects them within a frame. Saving is a separate "commit" that persists the
 * working doc as a new version via PUT /document; the agent writing a new
 * version re-seeds the baseline.
 */
import { create } from "zustand";
import type { EditAspect, EditDocument, EditOperation, EditSegment, LayoutRegion } from "@/lib/api";
import type { Durations } from "@/lib/resolve-timeline";

function docAspect(doc: EditDocument | null): EditAspect {
  const a = doc?.format?.aspect ?? doc?.brief?.aspect;
  return a === "portrait" || a === "square" ? a : "landscape";
}

const MIN_SEG_MS = 200;

function rid(prefix: string): string {
  return prefix + Math.random().toString(16).slice(2, 8);
}

function sameTimeline(a: EditSegment[], b: EditSegment[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    const x = a[i];
    const y = b[i];
    if (x.seg_id !== y.seg_id || x.file_id !== y.file_id || x.in_ms !== y.in_ms || x.out_ms !== y.out_ms)
      return false;
  }
  return true;
}

function sameOps(a: EditOperation[], b: EditOperation[]): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

/** Undo/redo snapshot. Deep-ish copy (each mutator already returns fresh
 * arrays/objects, so a shallow clone of the seg/op objects is enough). */
interface Snapshot {
  timeline: EditSegment[];
  operations: EditOperation[];
  selectedIds: string[];
}

const MAX_HISTORY = 100;

function snapshotOf(st: { timeline: EditSegment[]; operations: EditOperation[]; selectedIds: string[] }): Snapshot {
  return {
    timeline: st.timeline.map((s) => ({ ...s })),
    operations: st.operations.map((o) => ({ ...o })),
    selectedIds: [...st.selectedIds],
  };
}

interface EditDocState {
  threadId: string | null;
  /** version the working doc is based on (for optimistic-concurrency saves). */
  baseVersion: number;
  baselineTimeline: EditSegment[];
  baselineOperations: EditOperation[];
  timeline: EditSegment[];
  operations: EditOperation[];
  /** Time-scoped split-screen / PiP layouts (agent-authored; read by preview). */
  layoutRegions: LayoutRegion[];
  durations: Durations;
  /** Delivery frame shape for the preview/render (from the document format). */
  aspect: EditAspect;
  /** Multi-select: `ProjectClip.id`-shaped ("seg:<seg_id>" | "op:<op_id>"). */
  selectedIds: string[];

  /** Seed/replace baseline + working state from an authoritative document. */
  seed: (threadId: string, version: number, doc: EditDocument | null) => void;
  /** Wipe everything back to an empty document (used when starting fresh, so the
   * preview/timeline don't keep showing/playing a previous edit). */
  clear: () => void;
  /** Replace baseline after a successful save/agent write (keeps working == baseline). */
  commit: (version: number, doc: EditDocument) => void;
  revert: () => void;
  setWorking: (timeline: EditSegment[], operations: EditOperation[]) => void;
  isDirty: () => boolean;
  setDurations: (d: Durations) => void;
  mergeDurations: (d: Durations) => void;
  /** Replace the selection outright (plain click / marquee / programmatic). */
  select: (ids: string[]) => void;
  /** Add/remove one id from the selection (shift-click). */
  toggleSelect: (id: string) => void;
  clearSelection: () => void;

  // --- undo / redo (separate from revert-to-baseline) ---
  past: Snapshot[];
  future: Snapshot[];
  /** Snapshot the CURRENT state onto the undo stack and clear redo. Call once
   * per discrete edit (e.g. on pointer-down before a drag), not per intermediate
   * mutation, so a drag collapses to one history step. */
  pushHistory: () => void;
  undo: () => void;
  redo: () => void;
  canUndo: () => boolean;
  canRedo: () => boolean;

  // --- timeline (spine) mutators ---
  trim: (segId: string, edge: "in" | "out", absMs: number) => void;
  nudge: (segId: string, edge: "in" | "out", delta: number) => void;
  move: (segId: string, dir: -1 | 1) => void;
  /** Slip: move in_ms to an ABSOLUTE target (out_ms follows, keeping duration
   * fixed) — content shifts, position/duration on the program clock don't.
   * Clamped to the source's own room when its duration is known. */
  slip: (segId: string, targetInMs: number) => void;
  /** Reorder a spine segment to an absolute index (drag-to-reorder). */
  reorderSeg: (segId: string, toIndex: number) => void;
  /** Split at an absolute ms (defaults to the segment's midpoint). */
  split: (segId: string, atMs?: number) => void;
  /** Spine is gapless, so removing a segment always ripples (no "lift" for
   * the spine — there's no gap primitive in the document schema). */
  remove: (segId: string) => void;
  /** Insert a spine cut from a dragged clip (default: append). Selects it. */
  addSegment: (
    seg: { file_id: string; in_ms: number; out_ms: number },
    atIndex?: number
  ) => void;
  /** Overwrite [fromMs, toMs) of the program clock with a new spine segment. */
  overwriteSpine: (
    fromMs: number,
    toMs: number,
    seg: { file_id: string; in_ms: number; out_ms: number }
  ) => void;
  /** Add a placed V2 video cutaway / A2 audio bed from a dragged clip. Selects it. */
  addOp: (op: {
    type: "place_video" | "place_audio";
    source_file_id: string;
    src_in_ms: number;
    src_out_ms: number;
    from_ms: number;
    z?: number;
    role?: string;
    audio_kind?: string;
  }) => void;

  // --- operation mutators ---
  setGain: (opId: string, gainDb: number) => void;
  /** "Lift" delete — removes the op, leaving its slot empty (no shift). */
  removeOp: (opId: string) => void;
  /** "Ripple" delete — removes the op and shifts every LATER op on the same
   * track (video: same z; audio: same role) left by its duration. */
  rippleRemoveOp: (opId: string) => void;
  /** Split a placed op into two at an absolute PROGRAM ms (1:1 program->source
   * mapping, matching the op's own from_ms/to_ms domain). */
  splitOp: (opId: string, atProgMs: number) => void;
  /** Slip: move src_in_ms to an ABSOLUTE target (src_out_ms follows, keeping
   * duration fixed) — position/duration on the program clock stay fixed.
   * Clamped to the source's own room when known. */
  slipOp: (opId: string, targetSrcInMs: number) => void;
  /** Reposition a placed clip (place_video/place_audio) on the
   * program clock, keeping its duration. `maxMs` clamps the end to the base. */
  setOpFrom: (opId: string, fromMs: number, maxMs: number) => void;
  /** Trim a placed clip's in/out edge in PROGRAM ms; the source range shifts
   * with the moved edge so the visible content stays aligned. */
  setOpEdge: (opId: string, edge: "in" | "out", progMs: number, maxMs: number) => void;
  /** Restack a placed video clip onto another video layer (cross-track drag). */
  setOpZ: (opId: string, z: number) => void;
  /** Swap every op at z=zA with z=zB (track-header reorder — moves the WHOLE
   * layer, not one clip). */
  swapVideoZ: (zA: number, zB: number) => void;
}

export const useEditDocStore = create<EditDocState>((set, get) => ({
  threadId: null,
  baseVersion: 0,
  baselineTimeline: [],
  baselineOperations: [],
  timeline: [],
  operations: [],
  layoutRegions: [],
  durations: {},
  aspect: "landscape",
  selectedIds: [],
  past: [],
  future: [],

  seed: (threadId, version, doc) => {
    const timeline = doc?.timeline ?? [];
    const operations = doc?.operations ?? [];
    set({
      threadId,
      baseVersion: version,
      baselineTimeline: timeline,
      baselineOperations: operations,
      timeline: timeline.map((s) => ({ ...s })),
      operations: operations.map((o) => ({ ...o })),
      layoutRegions: doc?.layout_regions ?? [],
      aspect: docAspect(doc),
      selectedIds: [],
      past: [],
      future: [],
    });
  },

  clear: () =>
    set({
      threadId: null,
      baseVersion: 0,
      baselineTimeline: [],
      baselineOperations: [],
      timeline: [],
      operations: [],
      layoutRegions: [],
      aspect: "landscape",
      selectedIds: [],
      past: [],
      future: [],
    }),

  commit: (version, doc) => {
    const timeline = doc.timeline ?? [];
    const operations = doc.operations ?? [];
    set({
      baseVersion: version,
      baselineTimeline: timeline,
      baselineOperations: operations,
      timeline: timeline.map((s) => ({ ...s })),
      operations: operations.map((o) => ({ ...o })),
      layoutRegions: doc.layout_regions ?? [],
      aspect: docAspect(doc),
      past: [],
      future: [],
    });
  },

  revert: () =>
    set((st) => ({
      past: [...st.past, snapshotOf(st)].slice(-MAX_HISTORY),
      future: [],
      timeline: st.baselineTimeline.map((s) => ({ ...s })),
      operations: st.baselineOperations.map((o) => ({ ...o })),
      selectedIds: [],
    })),

  pushHistory: () =>
    set((st) => ({
      past: [...st.past, snapshotOf(st)].slice(-MAX_HISTORY),
      future: [],
    })),

  undo: () =>
    set((st) => {
      if (st.past.length === 0) return {};
      const prev = st.past[st.past.length - 1];
      return {
        past: st.past.slice(0, -1),
        future: [...st.future, snapshotOf(st)].slice(-MAX_HISTORY),
        timeline: prev.timeline,
        operations: prev.operations,
        selectedIds: prev.selectedIds,
      };
    }),

  redo: () =>
    set((st) => {
      if (st.future.length === 0) return {};
      const next = st.future[st.future.length - 1];
      return {
        future: st.future.slice(0, -1),
        past: [...st.past, snapshotOf(st)].slice(-MAX_HISTORY),
        timeline: next.timeline,
        operations: next.operations,
        selectedIds: next.selectedIds,
      };
    }),

  canUndo: () => get().past.length > 0,
  canRedo: () => get().future.length > 0,

  /** Replace only the WORKING state (baseline untouched) — e.g. loading an old
   * version to edit on top of the current head. */
  setWorking: (timeline, operations) =>
    set({
      timeline: timeline.map((s) => ({ ...s })),
      operations: operations.map((o) => ({ ...o })),
      selectedIds: [],
      past: [],
      future: [],
    }),

  isDirty: () => {
    const st = get();
    return (
      !sameTimeline(st.timeline, st.baselineTimeline) ||
      !sameOps(st.operations, st.baselineOperations)
    );
  },

  setDurations: (d) => set({ durations: d }),
  mergeDurations: (d) => set((st) => ({ durations: { ...st.durations, ...d } })),
  select: (ids) => set({ selectedIds: ids }),
  toggleSelect: (id) =>
    set((st) => ({
      selectedIds: st.selectedIds.includes(id)
        ? st.selectedIds.filter((x) => x !== id)
        : [...st.selectedIds, id],
    })),
  clearSelection: () => set({ selectedIds: [] }),

  trim: (segId, edge, absMs) =>
    set((st) => ({
      timeline: st.timeline.map((s) => {
        if (s.seg_id !== segId) return s;
        if (edge === "in") {
          return { ...s, in_ms: Math.max(0, Math.min(absMs, s.out_ms - MIN_SEG_MS)) };
        }
        return { ...s, out_ms: Math.max(s.in_ms + MIN_SEG_MS, absMs) };
      }),
    })),

  nudge: (segId, edge, delta) =>
    set((st) => ({
      timeline: st.timeline.map((s) => {
        if (s.seg_id !== segId) return s;
        if (edge === "in")
          return { ...s, in_ms: Math.max(0, Math.min(s.in_ms + delta, s.out_ms - MIN_SEG_MS)) };
        return { ...s, out_ms: Math.max(s.in_ms + MIN_SEG_MS, s.out_ms + delta) };
      }),
    })),

  move: (segId, dir) =>
    set((st) => {
      const i = st.timeline.findIndex((s) => s.seg_id === segId);
      const j = i + dir;
      if (i < 0 || j < 0 || j >= st.timeline.length) return {};
      const next = [...st.timeline];
      [next[i], next[j]] = [next[j], next[i]];
      return { timeline: next };
    }),

  slip: (segId, targetInMs) =>
    set((st) => {
      const i = st.timeline.findIndex((s) => s.seg_id === segId);
      if (i < 0) return {};
      const s = st.timeline[i];
      const dur = s.out_ms - s.in_ms;
      const room = st.durations[s.file_id];
      const maxIn = room != null ? Math.max(0, room - dur) : Infinity;
      const newIn = Math.max(0, Math.min(maxIn, Math.round(targetInMs)));
      if (newIn === s.in_ms) return {};
      const next = [...st.timeline];
      next[i] = { ...s, in_ms: newIn, out_ms: newIn + dur };
      return { timeline: next };
    }),

  reorderSeg: (segId, toIndex) =>
    set((st) => {
      const from = st.timeline.findIndex((s) => s.seg_id === segId);
      if (from < 0) return {};
      const next = [...st.timeline];
      const [moved] = next.splice(from, 1);
      const ti = Math.max(0, Math.min(Math.round(toIndex), next.length));
      next.splice(ti, 0, moved);
      // No-op guard: identical order.
      if (next.every((s, i) => s.seg_id === st.timeline[i].seg_id)) return {};
      return { timeline: next };
    }),

  split: (segId, atMs) =>
    set((st) => {
      const i = st.timeline.findIndex((s) => s.seg_id === segId);
      if (i < 0) return {};
      const s = st.timeline[i];
      // `atMs` is in the same SOURCE-ms domain as in_ms/out_ms (matching trim's
      // convention) — the caller maps a program-time click/playhead into this
      // clip's source range before calling split. Defaults to the midpoint.
      const at = atMs != null ? Math.round(atMs) : Math.round((s.in_ms + s.out_ms) / 2);
      const mid = Math.max(s.in_ms, Math.min(s.out_ms, at));
      if (mid - s.in_ms < MIN_SEG_MS || s.out_ms - mid < MIN_SEG_MS) return {};
      const a = { ...s, out_ms: mid };
      const b = { ...s, seg_id: rid("se"), in_ms: mid };
      const next = [...st.timeline];
      next.splice(i, 1, a, b);
      return { timeline: next, selectedIds: [`seg:${a.seg_id}`, `seg:${b.seg_id}`] };
    }),

  remove: (segId) =>
    set((st) => {
      if (st.timeline.length <= 1) return {};
      const clipId = `seg:${segId}`;
      return {
        timeline: st.timeline.filter((s) => s.seg_id !== segId),
        selectedIds: st.selectedIds.filter((id) => id !== clipId),
      };
    }),

  addSegment: (seg, atIndex) =>
    set((st) => {
      const inMs = Math.max(0, Math.round(seg.in_ms));
      const outMs = Math.max(inMs + MIN_SEG_MS, Math.round(seg.out_ms));
      const newSeg: EditSegment = {
        seg_id: rid("se"),
        file_id: seg.file_id,
        in_ms: inMs,
        out_ms: outMs,
      };
      const next = [...st.timeline];
      const idx =
        atIndex == null ? next.length : Math.max(0, Math.min(Math.round(atIndex), next.length));
      next.splice(idx, 0, newSeg);
      return { timeline: next, selectedIds: [`seg:${newSeg.seg_id}`] };
    }),

  /** Overwrite [fromMs, toMs) of the PROGRAM clock with a new spine segment:
   * segments fully inside the window are dropped, a segment straddling either
   * edge is trimmed to it (a segment spanning the WHOLE window is just
   * dropped rather than split in two — see timeline_nle.plan.md P1.2). The
   * new segment splices in where the window was. */
  overwriteSpine: (fromMs, toMs, seg) =>
    set((st) => {
      const from = Math.max(0, Math.round(fromMs));
      const to = Math.max(from, Math.round(toMs));
      let t = 0;
      const kept: EditSegment[] = [];
      let insertIdx = -1;
      for (const s of st.timeline) {
        const dur = Math.max(0, s.out_ms - s.in_ms);
        const ps = t;
        const pe = t + dur;
        t = pe;
        if (pe <= from || ps >= to) {
          if (insertIdx < 0 && ps >= from) insertIdx = kept.length;
          kept.push(s);
          continue;
        }
        const overlapStart = Math.max(ps, from);
        const overlapEnd = Math.min(pe, to);
        if (overlapStart > ps) kept.push({ ...s, out_ms: s.in_ms + (overlapStart - ps) });
        if (insertIdx < 0) insertIdx = kept.length;
        if (overlapEnd < pe) kept.push({ ...s, seg_id: rid("se"), in_ms: s.in_ms + (overlapEnd - ps) });
      }
      if (insertIdx < 0) insertIdx = kept.length;
      const inMs = Math.max(0, Math.round(seg.in_ms));
      const outMs = Math.max(inMs + MIN_SEG_MS, Math.round(seg.out_ms));
      const inserted: EditSegment = { seg_id: rid("se"), file_id: seg.file_id, in_ms: inMs, out_ms: outMs };
      kept.splice(insertIdx, 0, inserted);
      return { timeline: kept, selectedIds: [`seg:${inserted.seg_id}`] };
    }),

  addOp: (op) =>
    set((st) => {
      const srcIn = Math.max(0, Math.round(op.src_in_ms));
      const srcOut = Math.max(srcIn + MIN_SEG_MS, Math.round(op.src_out_ms));
      const from = Math.max(0, Math.round(op.from_ms));
      const dur = srcOut - srcIn;
      const newOp: EditOperation =
        op.type === "place_video"
          ? {
              op_id: rid("pv"),
              type: "place_video",
              source_file_id: op.source_file_id,
              src_in_ms: srcIn,
              src_out_ms: srcOut,
              from_ms: from,
              to_ms: from + dur,
              z: Math.round(op.z ?? 10),
              opacity: 1,
            }
          : {
              op_id: rid("pa"),
              type: "place_audio",
              source_file_id: op.source_file_id,
              src_in_ms: srcIn,
              src_out_ms: srcOut,
              from_ms: from,
              to_ms: from + dur,
              role: op.role ?? "music",
              audio_kind: op.audio_kind ?? "bed",
              gain_db: 0,
            };
      return { operations: [...st.operations, newOp], selectedIds: [`op:${newOp.op_id}`] };
    }),

  setGain: (opId, gainDb) =>
    set((st) => ({
      operations: st.operations.map((o) =>
        o.op_id === opId ? { ...o, gain_db: gainDb, ...(o.type === "level" ? { mute: false } : {}) } : o
      ),
    })),

  removeOp: (opId) =>
    set((st) => ({
      operations: st.operations.filter((o) => o.op_id !== opId),
      selectedIds: st.selectedIds.filter((id) => id !== `op:${opId}`),
    })),

  rippleRemoveOp: (opId) =>
    set((st) => {
      const removed = st.operations.find((o) => o.op_id === opId);
      if (!removed || removed.from_ms == null || removed.to_ms == null) {
        return {
          operations: st.operations.filter((o) => o.op_id !== opId),
          selectedIds: st.selectedIds.filter((id) => id !== `op:${opId}`),
        };
      }
      const removedFrom = removed.from_ms;
      const dur = removed.to_ms - removedFrom;
      const sameTrack = (o: EditOperation) =>
        removed.type === "place_video"
          ? o.type === "place_video" && Math.round(o.z ?? 10) === Math.round(removed.z ?? 10)
          : o.type === "place_audio" && (o.role ?? "music") === (removed.role ?? "music");
      const operations = st.operations
        .filter((o) => o.op_id !== opId)
        .map((o) => {
          if (!sameTrack(o) || o.from_ms == null || o.to_ms == null || o.from_ms < removedFrom) return o;
          return { ...o, from_ms: o.from_ms - dur, to_ms: o.to_ms - dur };
        });
      return {
        operations,
        selectedIds: st.selectedIds.filter((id) => id !== `op:${opId}`),
      };
    }),

  splitOp: (opId, atProgMs) =>
    set((st) => {
      const i = st.operations.findIndex((o) => o.op_id === opId);
      if (i < 0) return {};
      const o = st.operations[i];
      if (o.from_ms == null || o.to_ms == null) return {};
      const at = Math.round(atProgMs);
      if (at - o.from_ms < MIN_SEG_MS || o.to_ms - at < MIN_SEG_MS) return {};
      const srcIn = Math.round(o.src_in_ms ?? 0);
      // 1:1 program->source mapping (ops never change playback speed).
      const splitSrc = srcIn + (at - o.from_ms);
      const a: EditOperation = { ...o, to_ms: at, src_out_ms: splitSrc };
      const b: EditOperation = {
        ...o,
        op_id: rid(o.type === "place_video" ? "pv" : "pa"),
        from_ms: at,
        src_in_ms: splitSrc,
      };
      const next = [...st.operations];
      next.splice(i, 1, a, b);
      return { operations: next, selectedIds: [`op:${a.op_id}`, `op:${b.op_id}`] };
    }),

  slipOp: (opId, targetSrcInMs) =>
    set((st) => {
      const i = st.operations.findIndex((o) => o.op_id === opId);
      if (i < 0) return {};
      const o = st.operations[i];
      const srcIn = Math.round(o.src_in_ms ?? 0);
      const srcOut = Math.round(o.src_out_ms ?? 0);
      const dur = srcOut - srcIn;
      const room = o.source_file_id != null ? st.durations[o.source_file_id] : undefined;
      const maxIn = room != null ? Math.max(0, room - dur) : Infinity;
      const newIn = Math.max(0, Math.min(maxIn, Math.round(targetSrcInMs)));
      if (newIn === srcIn) return {};
      const next = [...st.operations];
      next[i] = { ...o, src_in_ms: newIn, src_out_ms: newIn + dur };
      return { operations: next };
    }),

  setOpFrom: (opId, fromMs, maxMs) =>
    set((st) => ({
      operations: st.operations.map((o) => {
        if (o.op_id !== opId || o.from_ms == null || o.to_ms == null) return o;
        const dur = o.to_ms - o.from_ms;
        const from = Math.max(0, Math.min(Math.round(fromMs), Math.max(0, maxMs - dur)));
        return { ...o, from_ms: from, to_ms: from + dur };
      }),
    })),

  setOpEdge: (opId, edge, progMs, maxMs) =>
    set((st) => ({
      operations: st.operations.map((o) => {
        if (o.op_id !== opId || o.from_ms == null || o.to_ms == null) return o;
        const srcIn = Math.round(o.src_in_ms ?? 0);
        const srcOut = Math.round(o.src_out_ms ?? 0);
        if (edge === "in") {
          const from = Math.max(0, Math.min(Math.round(progMs), o.to_ms - MIN_SEG_MS));
          const d = from - o.from_ms;
          return { ...o, from_ms: from, src_in_ms: Math.max(0, srcIn + d) };
        }
        const to = Math.max(o.from_ms + MIN_SEG_MS, Math.min(Math.round(progMs), maxMs));
        const d = to - o.to_ms;
        return { ...o, to_ms: to, src_out_ms: Math.max(srcIn + 1, srcOut + d) };
      }),
    })),

  setOpZ: (opId, z) =>
    set((st) => ({
      operations: st.operations.map((o) =>
        o.op_id === opId ? { ...o, z: Math.round(z) } : o
      ),
    })),

  swapVideoZ: (zA, zB) =>
    set((st) => ({
      operations: st.operations.map((o) => {
        if (o.type !== "place_video") return o;
        const z = Math.round(o.z ?? 10);
        if (z === zA) return { ...o, z: zB };
        if (z === zB) return { ...o, z: zA };
        return o;
      }),
    })),
}));
