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
import type { EditAspect, EditDocument, EditOperation, EditSegment } from "@/lib/api";
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

interface EditDocState {
  threadId: string | null;
  /** version the working doc is based on (for optimistic-concurrency saves). */
  baseVersion: number;
  baselineTimeline: EditSegment[];
  baselineOperations: EditOperation[];
  timeline: EditSegment[];
  operations: EditOperation[];
  durations: Durations;
  /** Delivery frame shape for the preview/render (from the document format). */
  aspect: EditAspect;
  selected: string | null;

  /** Seed/replace baseline + working state from an authoritative document. */
  seed: (threadId: string, version: number, doc: EditDocument | null) => void;
  /** Replace baseline after a successful save/agent write (keeps working == baseline). */
  commit: (version: number, doc: EditDocument) => void;
  revert: () => void;
  setWorking: (timeline: EditSegment[], operations: EditOperation[]) => void;
  isDirty: () => boolean;
  setDurations: (d: Durations) => void;
  mergeDurations: (d: Durations) => void;
  select: (segId: string | null) => void;

  // --- timeline mutators ---
  trim: (segId: string, edge: "in" | "out", absMs: number) => void;
  nudge: (segId: string, edge: "in" | "out", delta: number) => void;
  move: (segId: string, dir: -1 | 1) => void;
  split: (segId: string) => void;
  remove: (segId: string) => void;

  // --- operation mutators ---
  setGain: (opId: string, gainDb: number) => void;
  removeOp: (opId: string) => void;
}

export const useEditDocStore = create<EditDocState>((set, get) => ({
  threadId: null,
  baseVersion: 0,
  baselineTimeline: [],
  baselineOperations: [],
  timeline: [],
  operations: [],
  durations: {},
  aspect: "landscape",
  selected: null,

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
      aspect: docAspect(doc),
      selected: null,
    });
  },

  commit: (version, doc) => {
    const timeline = doc.timeline ?? [];
    const operations = doc.operations ?? [];
    set({
      baseVersion: version,
      baselineTimeline: timeline,
      baselineOperations: operations,
      timeline: timeline.map((s) => ({ ...s })),
      operations: operations.map((o) => ({ ...o })),
      aspect: docAspect(doc),
    });
  },

  revert: () =>
    set((st) => ({
      timeline: st.baselineTimeline.map((s) => ({ ...s })),
      operations: st.baselineOperations.map((o) => ({ ...o })),
      selected: null,
    })),

  /** Replace only the WORKING state (baseline untouched) — e.g. loading an old
   * version to edit on top of the current head. */
  setWorking: (timeline, operations) =>
    set({
      timeline: timeline.map((s) => ({ ...s })),
      operations: operations.map((o) => ({ ...o })),
      selected: null,
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
  select: (segId) => set({ selected: segId }),

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

  split: (segId) =>
    set((st) => {
      const i = st.timeline.findIndex((s) => s.seg_id === segId);
      if (i < 0) return {};
      const s = st.timeline[i];
      const mid = Math.round((s.in_ms + s.out_ms) / 2);
      if (mid - s.in_ms < MIN_SEG_MS || s.out_ms - mid < MIN_SEG_MS) return {};
      const a = { ...s, out_ms: mid };
      const b = { ...s, seg_id: rid("se"), in_ms: mid };
      const next = [...st.timeline];
      next.splice(i, 1, a, b);
      return { timeline: next };
    }),

  remove: (segId) =>
    set((st) => {
      if (st.timeline.length <= 1) return {};
      return {
        timeline: st.timeline.filter((s) => s.seg_id !== segId),
        selected: st.selected === segId ? null : st.selected,
      };
    }),

  setGain: (opId, gainDb) =>
    set((st) => ({
      operations: st.operations.map((o) =>
        o.op_id === opId ? { ...o, gain_db: gainDb, ...(o.type === "level" ? { mute: false } : {}) } : o
      ),
    })),

  removeOp: (opId) =>
    set((st) => ({ operations: st.operations.filter((o) => o.op_id !== opId) })),
}));
