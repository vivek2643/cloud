"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Scissors,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Volume2,
  VolumeX,
  Play,
  Pause,
  Undo2,
  Redo2,
  ZoomIn,
  ZoomOut,
  Maximize2,
  MousePointer2,
  Magnet,
  Lock,
  Unlock,
  Headphones,
  Bookmark,
  Copy,
} from "lucide-react";
import { type EditOperation } from "@/lib/api";
import { useEditDocStore } from "@/stores/edit-doc-store";
import { useTransport, formatTimecode, snapMs, FRAME_MS, PROJECT_FPS } from "@/stores/transport-store";
import {
  useTimelineView,
  MIN_PX_PER_SEC,
  MAX_PX_PER_SEC,
  type TrackMeta,
  type ClipboardEntry,
} from "@/stores/timeline-view";
import {
  documentToProject,
  collectSnapTargets,
  snapValue,
  type ProjectClip,
} from "@/lib/edit-project";

function fmt(ms: number) {
  const s = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Inverse of formatTimecode: "[H:]MM:SS:FF" -> ms. Returns null when the
 * text isn't a parseable timecode (the field then reverts on blur). */
function parseTimecode(text: string): number | null {
  const parts = text.trim().split(":").map((p) => p.trim());
  if (parts.length < 2 || parts.length > 4 || parts.some((p) => p === "" || Number.isNaN(Number(p))))
    return null;
  const nums = parts.map(Number);
  let h = 0, m = 0, s = 0, f = 0;
  if (nums.length === 4) [h, m, s, f] = nums;
  else if (nums.length === 3) [m, s, f] = nums;
  else [s, f] = nums;
  const totalFrames = (h * 3600 + m * 60 + s) * PROJECT_FPS + f;
  return Math.max(0, Math.round((totalFrames * 1000) / PROJECT_FPS));
}

const HEADER_W = 132;
const RULER_H = 24;
const DEFAULT_LANE_H = 36;
const MIN_LANE_H = 26;
const MAX_LANE_H = 96;
const SNAP_THRESHOLD_PX = 8;
const DUPLICATE_OFFSET_MS = 300;

/** A cut dragged from the Cuts view (see its onDragStart). */
interface DropPayload {
  file_id: string;
  in_ms: number;
  out_ms: number;
  kind?: string;
}

function parseDrop(e: React.DragEvent): DropPayload | null {
  const raw =
    e.dataTransfer.getData("application/x-cut") ||
    e.dataTransfer.getData("text/plain");
  if (!raw) return null;
  try {
    const p = JSON.parse(raw) as Partial<DropPayload>;
    if (!p.file_id || p.in_ms == null || p.out_ms == null) return null;
    return { file_id: p.file_id, in_ms: Number(p.in_ms), out_ms: Number(p.out_ms), kind: p.kind };
  } catch {
    return null;
  }
}

/** One user-created empty lane waiting for a clip to land (tracks are
 * DERIVED from ops, so an "empty track" only becomes real once an op with
 * its z/role exists — see edit-project.ts + timeline_nle.plan.md §9). */
interface PendingTrack {
  key: string;
  kind: "video" | "audio";
  z?: number;
  role?: string;
}

/** Unified shape for a rendered track row — either a real (derived) track or
 * a not-yet-real pending placeholder, so the header/lane lists never need to
 * branch on which kind of track they're drawing. */
interface RenderTrack {
  id: string;
  kind: "video" | "audio";
  label: string;
  isBase: boolean;
  z: number;
  role?: string;
  pending: boolean;
}

interface Marquee {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export function TimelineEditor({ ensureThread }: { ensureThread: () => Promise<string | null> }) {
  const timeline = useEditDocStore((s) => s.timeline);
  const operations = useEditDocStore((s) => s.operations);
  const selectedIds = useEditDocStore((s) => s.selectedIds);
  const select = useEditDocStore((s) => s.select);
  const toggleSelect = useEditDocStore((s) => s.toggleSelect);
  const clearSelection = useEditDocStore((s) => s.clearSelection);
  const trimSeg = useEditDocStore((s) => s.trim);
  const moveSeg = useEditDocStore((s) => s.move);
  const splitSeg = useEditDocStore((s) => s.split);
  const removeSeg = useEditDocStore((s) => s.remove);
  const setGain = useEditDocStore((s) => s.setGain);
  const removeOpStore = useEditDocStore((s) => s.removeOp);
  const rippleRemoveOp = useEditDocStore((s) => s.rippleRemoveOp);
  const splitOp = useEditDocStore((s) => s.splitOp);
  const setOpFrom = useEditDocStore((s) => s.setOpFrom);
  const setOpEdge = useEditDocStore((s) => s.setOpEdge);
  const setOpZ = useEditDocStore((s) => s.setOpZ);
  const reorderSeg = useEditDocStore((s) => s.reorderSeg);
  const addSegment = useEditDocStore((s) => s.addSegment);
  const addOp = useEditDocStore((s) => s.addOp);
  const pushHistory = useEditDocStore((s) => s.pushHistory);
  const undo = useEditDocStore((s) => s.undo);
  const redo = useEditDocStore((s) => s.redo);
  const canUndo = useEditDocStore((s) => s.canUndo());
  const canRedo = useEditDocStore((s) => s.canRedo());

  const pxPerSec = useTimelineView((s) => s.pxPerSec);
  const scrollLeftPx = useTimelineView((s) => s.scrollLeftPx);
  const tool = useTimelineView((s) => s.tool);
  const snapEnabled = useTimelineView((s) => s.snapEnabled);
  const snapGuideMs = useTimelineView((s) => s.snapGuideMs);
  const trackMeta = useTimelineView((s) => s.trackMeta);
  const inMarkMs = useTimelineView((s) => s.inMarkMs);
  const outMarkMs = useTimelineView((s) => s.outMarkMs);
  const markers = useTimelineView((s) => s.markers);
  const clipboard = useTimelineView((s) => s.clipboard);
  const setZoom = useTimelineView((s) => s.setZoom);
  const zoomIn = useTimelineView((s) => s.zoomIn);
  const zoomOut = useTimelineView((s) => s.zoomOut);
  const zoomToFit = useTimelineView((s) => s.zoomToFit);
  const setScrollLeft = useTimelineView((s) => s.setScrollLeft);
  const setTool = useTimelineView((s) => s.setTool);
  const toggleSnap = useTimelineView((s) => s.toggleSnap);
  const setSnapGuide = useTimelineView((s) => s.setSnapGuide);
  const setTrackMeta = useTimelineView((s) => s.setTrackMeta);
  const setInMark = useTimelineView((s) => s.setInMark);
  const setOutMark = useTimelineView((s) => s.setOutMark);
  const addMarker = useTimelineView((s) => s.addMarker);
  const removeMarker = useTimelineView((s) => s.removeMarker);
  const setClipboard = useTimelineView((s) => s.setClipboard);

  const [viewportW, setViewportW] = useState(600);
  const [pendingTracks, setPendingTracks] = useState<PendingTrack[]>([]);
  const [marquee, setMarquee] = useState<Marquee | null>(null);

  const scrollRef = useRef<HTMLDivElement>(null);
  const trackEls = useRef<Map<string, HTMLDivElement>>(new Map());
  const muteCache = useRef<Record<string, number>>({});
  const trackAudioMuteCache = useRef<Record<string, Record<string, number>>>({});

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setViewportW(el.clientWidth || 600));
    ro.observe(el);
    setViewportW(el.clientWidth || 600);
    return () => ro.disconnect();
  }, []);

  const aspect = useEditDocStore((s) => s.aspect);
  const project = useMemo(
    () => documentToProject(timeline, operations, aspect),
    [timeline, operations, aspect]
  );
  const total = project.durationMs;
  const pxPerMs = pxPerSec / 1000;
  const contentWidth = Math.max(1, total * pxPerMs);

  // Drop pending-track placeholders once a real track with the same z/role
  // exists (a clip landed there, so edit-project.ts now derives it for real).
  useEffect(() => {
    if (pendingTracks.length === 0) return;
    setPendingTracks((cur) =>
      cur.filter((p) =>
        p.kind === "video"
          ? !project.tracks.some((t) => t.kind === "video" && !t.isBase && t.z === p.z)
          : !project.tracks.some((t) => t.kind === "audio" && !t.isBase && t.role === p.role)
      )
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.tracks]);

  // Merge derived + pending tracks into one uniform shape for rendering
  // (pending video slots before base video; pending audio appended after the
  // real audio tracks). `pending: true` marks a not-yet-real placeholder lane.
  const renderTracks: RenderTrack[] = useMemo(() => {
    const baseIdx = project.tracks.findIndex((t) => t.kind === "video" && t.isBase);
    const real: RenderTrack[] = project.tracks.map((t) => ({ ...t, pending: false }));
    const pendingVideo: RenderTrack[] = pendingTracks
      .filter((p) => p.kind === "video")
      .map((p) => ({
        id: p.key, kind: "video", label: `V${p.z ?? 0}`, isBase: false, z: p.z ?? 0, pending: true,
      }));
    const pendingAudio: RenderTrack[] = pendingTracks
      .filter((p) => p.kind === "audio")
      .map((p) => ({
        id: p.key, kind: "audio", label: p.role ?? "audio", isBase: false, z: 0, role: p.role, pending: true,
      }));
    const out = [...real];
    if (pendingVideo.length) out.splice(Math.max(0, baseIdx), 0, ...pendingVideo);
    out.push(...pendingAudio);
    return out;
  }, [project.tracks, pendingTracks]);

  // Shared transport — playhead position + play state live in the same store the
  // program monitor reads, so the two surfaces are always in lockstep.
  const progMs = useTransport((s) => s.progMs);
  const playing = useTransport((s) => s.playing);
  const seek = useTransport((s) => s.seek);
  const step = useTransport((s) => s.step);
  const togglePlaying = useTransport((s) => s.togglePlaying);

  const playheadPx = pxPerMs > 0 ? progMs * pxPerMs : 0;

  // --- keep the DOM scroll position and the store in sync ---
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (Math.abs(el.scrollLeft - scrollLeftPx) > 0.5) el.scrollLeft = scrollLeftPx;
  }, [scrollLeftPx]);

  function onContentScroll() {
    const el = scrollRef.current;
    if (!el) return;
    setScrollLeft(el.scrollLeft);
  }

  // --- auto-scroll to keep the playhead in view during playback ---
  useEffect(() => {
    if (!playing || pxPerMs <= 0) return;
    const el = scrollRef.current;
    if (!el) return;
    const EDGE_PAD = 24;
    if (playheadPx > el.scrollLeft + viewportW - EDGE_PAD || playheadPx < el.scrollLeft) {
      const next = Math.max(0, playheadPx - viewportW * 0.1);
      el.scrollLeft = next;
      setScrollLeft(next);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [progMs, playing]);

  // --- Cmd/Ctrl + wheel = zoom centered on the cursor ---
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey)) return;
      e.preventDefault();
      const rect = el.getBoundingClientRect();
      const cursorX = e.clientX - rect.left;
      const msAtCursor = (el.scrollLeft + cursorX) / (pxPerSec / 1000 || 1);
      const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
      const nextPxPerSec = Math.max(MIN_PX_PER_SEC, Math.min(MAX_PX_PER_SEC, pxPerSec * factor));
      const nextPxPerMs = nextPxPerSec / 1000;
      const nextScroll = Math.max(0, msAtCursor * nextPxPerMs - cursorX);
      setZoom(nextPxPerSec);
      setScrollLeft(nextScroll);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [pxPerSec, setZoom, setScrollLeft]);

  function isLocked(trackId: string): boolean {
    return !!trackMeta[trackId]?.lock;
  }

  function clipsForIds(ids: string[]): ProjectClip[] {
    if (ids.length === 0) return [];
    const set = new Set(ids);
    return project.clips.filter((c) => set.has(c.id));
  }

  function trackRoleFor(trackId: string): string | undefined {
    return project.tracks.find((t) => t.id === trackId)?.role;
  }

  /** Track hint for copy/duplicate: preserve z only when the clip already
   * lives on a real V2+ layer. A SPINE clip's `z` is 0 (the base track) —
   * carrying that forward would collide with the base layer, so a copied/
   * duplicated spine clip instead falls back to the store's default cutaway
   * z (10), landing on its own new V2 lane per the plan's "pick op to avoid
   * disturbing spine timing." */
  function trackHintFor(c: ProjectClip): { z?: number; role?: string } {
    if (c.kind === "video") return c.origin.kind === "op" ? { z: c.z } : {};
    return { role: trackRoleFor(c.trackId) };
  }

  // --- P0.1: blade/razor at the playhead ---
  function splitAtPlayhead() {
    const spineHit = project.clips.find(
      (c) => c.origin.kind === "spine" && c.kind === "video" && progMs > c.progStartMs && progMs < c.progEndMs
    );
    const opHits = project.clips.filter(
      (c) => c.origin.kind === "op" && progMs > c.progStartMs && progMs < c.progEndMs
    );
    if (!spineHit && opHits.length === 0) return;
    pushHistory();
    if (spineHit && spineHit.origin.kind === "spine") {
      const srcMs = spineHit.srcInMs + (progMs - spineHit.progStartMs);
      splitSeg(spineHit.origin.segId, srcMs);
    }
    for (const c of opHits) {
      if (c.origin.kind === "op") splitOp(c.origin.opId, progMs);
    }
  }

  // --- P0.2: ripple vs lift delete ---
  function deleteSelected(ripple: boolean) {
    const clips = clipsForIds(selectedIds);
    if (!clips.length) return;
    pushHistory();
    for (const c of clips) {
      if (c.origin.kind === "spine") {
        removeSeg(c.origin.segId); // spine is gapless -- always ripples (no lift primitive)
      } else if (ripple) {
        rippleRemoveOp(c.origin.opId);
      } else {
        removeOpStore(c.origin.opId);
      }
    }
    clearSelection();
  }

  // --- P0.3: copy / cut / paste / duplicate ---
  function copySelection() {
    const clips = clipsForIds(selectedIds);
    if (!clips.length) return;
    const entries: ClipboardEntry[] = clips.map((c) => ({
      kind: c.kind,
      sourceFileId: c.sourceFileId,
      srcInMs: c.srcInMs,
      srcOutMs: c.srcOutMs,
      trackHint: trackHintFor(c),
    }));
    setClipboard(entries);
  }

  function cutSelection() {
    const clips = clipsForIds(selectedIds);
    if (!clips.length) return;
    copySelection();
    pushHistory();
    for (const c of clips) {
      if (c.origin.kind === "spine") removeSeg(c.origin.segId);
      else removeOpStore(c.origin.opId); // lift; matches the plain-Delete default
    }
    clearSelection();
  }

  function pasteAtPlayhead() {
    if (!clipboard.length) return;
    pushHistory();
    for (const entry of clipboard) {
      addOp({
        type: entry.kind === "video" ? "place_video" : "place_audio",
        source_file_id: entry.sourceFileId,
        src_in_ms: entry.srcInMs,
        src_out_ms: entry.srcOutMs,
        from_ms: progMs,
        z: entry.trackHint?.z,
        role: entry.trackHint?.role,
      });
    }
    void ensureThread();
  }

  function duplicateSelection() {
    const clips = clipsForIds(selectedIds);
    if (!clips.length) return;
    pushHistory();
    for (const c of clips) {
      const hint = trackHintFor(c);
      addOp({
        type: c.kind === "video" ? "place_video" : "place_audio",
        source_file_id: c.sourceFileId,
        src_in_ms: c.srcInMs,
        src_out_ms: c.srcOutMs,
        from_ms: c.progStartMs + DUPLICATE_OFFSET_MS,
        z: hint.z,
        role: hint.role,
      });
    }
    void ensureThread();
  }

  // --- P0.5: frame-nudge selected clip(s) ---
  function nudgeSelected(dir: -1 | 1, big: boolean) {
    const clips = clipsForIds(selectedIds);
    if (!clips.length) return;
    pushHistory();
    const deltaMs = dir * (big ? 10 : 1) * FRAME_MS;
    for (const c of clips) {
      if (c.origin.kind === "spine") {
        moveSeg(c.origin.segId, dir); // reorder-only; the spine has no continuous position
      } else {
        setOpFrom(c.origin.opId, snapMs(c.progStartMs + deltaMs), total);
      }
    }
  }

  // --- keyboard: transport, undo/redo, tools, snap, edit verbs ---
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = document.activeElement as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable))
        return;
      const mod = e.metaKey || e.ctrlKey;
      if (mod && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey) redo();
        else undo();
        return;
      }
      if (mod && e.key.toLowerCase() === "y") {
        e.preventDefault();
        redo();
        return;
      }
      if (mod && e.key.toLowerCase() === "k") {
        e.preventDefault();
        splitAtPlayhead();
        return;
      }
      if (mod && e.key.toLowerCase() === "c") {
        e.preventDefault();
        copySelection();
        return;
      }
      if (mod && e.key.toLowerCase() === "x") {
        e.preventDefault();
        cutSelection();
        return;
      }
      if (mod && e.key.toLowerCase() === "v") {
        e.preventDefault();
        pasteAtPlayhead();
        return;
      }
      if (mod && e.key.toLowerCase() === "d") {
        e.preventDefault();
        duplicateSelection();
        return;
      }
      if (e.key === "Delete" || e.key === "Backspace") {
        e.preventDefault();
        deleteSelected(e.shiftKey);
        return;
      }
      if (e.code === "Space") {
        e.preventDefault();
        togglePlaying();
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        step(e.shiftKey ? -10 : -1);
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        step(e.shiftKey ? 10 : 1);
      } else if (!mod && e.key.toLowerCase() === "v") {
        setTool("select");
      } else if (!mod && e.key.toLowerCase() === "b") {
        setTool("blade");
      } else if (!mod && e.key.toLowerCase() === "s") {
        toggleSnap();
      } else if (!mod && e.key.toLowerCase() === "m") {
        addMarker(useTransport.getState().progMs);
      } else if (e.key === ",") {
        e.preventDefault();
        nudgeSelected(-1, e.shiftKey);
      } else if (e.key === ".") {
        e.preventDefault();
        nudgeSelected(1, e.shiftKey);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [togglePlaying, step, undo, redo, setTool, toggleSnap, addMarker, selectedIds, clipboard, progMs, project]);

  // --- snapping helper shared by every drag gesture ---
  const snapProgramMs = useCallback(
    (rawMs: number, excludeClipId?: string): number => {
      if (!snapEnabled) {
        setSnapGuide(null);
        return rawMs;
      }
      const extra = [progMs];
      if (inMarkMs != null) extra.push(inMarkMs);
      if (outMarkMs != null) extra.push(outMarkMs);
      extra.push(...markers);
      const targets = collectSnapTargets(project, { excludeClipId, extra });
      const { value, snappedTo } = snapValue(rawMs, targets, pxPerMs, SNAP_THRESHOLD_PX);
      setSnapGuide(snappedTo);
      return value;
    },
    [snapEnabled, progMs, inMarkMs, outMarkMs, markers, project, pxPerMs, setSnapGuide]
  );

  function selectClip(clip: ProjectClip, e?: { shiftKey?: boolean }) {
    if (e?.shiftKey) toggleSelect(clip.id);
    else select([clip.id]);
  }

  function clickToSplit(clip: ProjectClip, e: React.PointerEvent) {
    if (clip.origin.kind !== "spine" || pxPerMs <= 0) return;
    const el = trackEls.current.get(clip.trackId);
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const clickProgMs = (e.clientX - rect.left) / pxPerMs;
    // Map the click's program-time position into this clip's SOURCE range
    // (spine plays at native speed, so the offset is 1:1).
    const srcMs = clip.srcInMs + (clickProgMs - clip.progStartMs);
    pushHistory();
    splitSeg(clip.origin.segId, srcMs);
  }

  // --- trim a clip's edge by dragging its handle (frame-snapped + snapped) ---
  function startTrim(clip: ProjectClip, edge: "in" | "out", e: React.PointerEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (pxPerMs <= 0 || !clip.trimmable || isLocked(clip.trackId)) return;
    selectClip(clip);
    pushHistory();
    const startX = e.clientX;
    const onMove = (ev: PointerEvent) => {
      const dMs = (ev.clientX - startX) / pxPerMs;
      if (clip.origin.kind === "spine") {
        const base = edge === "in" ? clip.srcInMs : clip.srcOutMs;
        const progBase = edge === "in" ? clip.progStartMs : clip.progEndMs;
        const snappedProg = snapProgramMs(progBase + dMs, clip.id);
        trimSeg(clip.origin.segId, edge, snapMs(base + (snappedProg - progBase)));
      } else {
        const progBase = edge === "in" ? clip.progStartMs : clip.progEndMs;
        const snappedProg = snapProgramMs(progBase + dMs, clip.id);
        setOpEdge(clip.origin.opId, edge, snapMs(snappedProg), total);
      }
    };
    const onUp = () => {
      setSnapGuide(null);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  /** Which rendered track the pointer is currently over (vertical hit-test). */
  function trackAtY(clientY: number) {
    for (const track of renderTracks) {
      const el = trackEls.current.get(track.id);
      if (!el) continue;
      const r = el.getBoundingClientRect();
      if (clientY >= r.top && clientY <= r.bottom) return track;
    }
    return null;
  }

  // --- move a placed clip freely along the program clock + across video
  //     layers (horizontal = reposition, vertical = restack onto another
  //     video track's z). Frame-snapped + edge-snapped. ---
  function startMove(clip: ProjectClip, e: React.PointerEvent) {
    if (!clip.movable || clip.origin.kind !== "op" || isLocked(clip.trackId)) return;
    e.preventDefault();
    e.stopPropagation();
    if (pxPerMs <= 0) return;
    selectClip(clip);
    pushHistory();
    const opId = clip.origin.opId;
    const startX = e.clientX;
    const startFrom = clip.progStartMs;
    const onMove = (ev: PointerEvent) => {
      const dMs = (ev.clientX - startX) / pxPerMs;
      const snappedFrom = snapProgramMs(startFrom + dMs, clip.id);
      setOpFrom(opId, snapMs(snappedFrom), total);
      // Cross-track: only V2+ video cutaways restack, only onto another non-base
      // video layer (never onto the spine or an audio track).
      if (clip.kind === "video") {
        const tgt = trackAtY(ev.clientY);
        if (tgt && tgt.kind === "video" && !tgt.isBase && tgt.z !== clip.z) {
          setOpZ(opId, tgt.z);
        }
      }
    };
    const onUp = () => {
      setSnapGuide(null);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  // --- drag a base (spine) video clip to reorder it within the spine ---
  function startReorder(clip: ProjectClip, e: React.PointerEvent) {
    if (clip.origin.kind !== "spine" || clip.kind !== "video" || isLocked(clip.trackId)) return;
    e.preventDefault();
    e.stopPropagation();
    if (pxPerMs <= 0) return;
    selectClip(clip);
    pushHistory();
    const segId = clip.origin.segId;
    const baseId = clip.trackId;
    const onMove = (ev: PointerEvent) => {
      const el = trackEls.current.get(baseId);
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const atMs = (ev.clientX - rect.left) / pxPerMs;
      // Insertion index among the OTHER base clips (count whose midpoint is
      // left of the cursor) — matches reorderSeg's post-removal splice index.
      const others = project.clips.filter(
        (c) => c.trackId === baseId && c.id !== clip.id
      );
      let idx = 0;
      for (const c of others) {
        if (atMs > (c.progStartMs + c.progEndMs) / 2) idx++;
      }
      reorderSeg(segId, idx);
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  // --- P0.4: marquee-select on empty lane space ---
  function startMarquee(e: React.PointerEvent) {
    if (tool !== "select") return;
    if (e.target !== e.currentTarget) return; // bubbled from a clip Block, not empty space
    const startX = e.clientX;
    const startY = e.clientY;
    const additive = e.shiftKey;
    setMarquee({ x0: startX, y0: startY, x1: startX, y1: startY });
    const onMove = (ev: PointerEvent) => {
      setMarquee((m) => (m ? { ...m, x1: ev.clientX, y1: ev.clientY } : m));
    };
    const onUp = (ev: PointerEvent) => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
      finishMarquee(ev.clientX, ev.clientY, additive);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  function finishMarquee(endX: number, endY: number, additive: boolean) {
    setMarquee((m) => {
      if (!m) return null;
      const rx0 = Math.min(m.x0, endX), rx1 = Math.max(m.x0, endX);
      const ry0 = Math.min(m.y0, endY), ry1 = Math.max(m.y0, endY);
      const isClick = rx1 - rx0 < 4 && ry1 - ry0 < 4;
      const hits: string[] = [];
      if (!isClick) {
        for (const clip of project.clips) {
          const laneEl = trackEls.current.get(clip.trackId);
          if (!laneEl) continue;
          const laneRect = laneEl.getBoundingClientRect();
          const cx0 = laneRect.left + clip.progStartMs * pxPerMs;
          const cx1 = laneRect.left + clip.progEndMs * pxPerMs;
          if (cx1 >= rx0 && cx0 <= rx1 && laneRect.bottom >= ry0 && laneRect.top <= ry1) hits.push(clip.id);
        }
      }
      if (hits.length) {
        select(additive ? Array.from(new Set([...selectedIds, ...hits])) : hits);
      } else if (!additive) {
        clearSelection();
      }
      return null;
    });
  }

  // --- drag cuts IN from the Cuts view ---
  // Base video lane (V1) = insert a spine cut at the drop index; an upper video
  // lane (V2+) = a placed cutaway; an audio lane = a placed bed — all at the drop time.
  const [dropTrack, setDropTrack] = useState<string | null>(null);

  function progMsAtX(trackId: string, clientX: number): number {
    const el = trackEls.current.get(trackId);
    if (!el || pxPerMs <= 0) return progMs;
    const rect = el.getBoundingClientRect();
    const raw = (clientX - rect.left) / pxPerMs;
    const snapped = snapEnabled ? snapValue(raw, collectSnapTargets(project), pxPerMs, SNAP_THRESHOLD_PX).value : raw;
    return Math.max(0, snapMs(snapped));
  }

  function spineIndexAtX(trackId: string, clientX: number): number {
    const el = trackEls.current.get(trackId);
    if (!el || pxPerMs <= 0) return timeline.length;
    const rect = el.getBoundingClientRect();
    const at = (clientX - rect.left) / pxPerMs;
    let idx = 0;
    for (const c of project.clips.filter((c) => c.trackId === trackId)) {
      if (at > (c.progStartMs + c.progEndMs) / 2) idx++;
    }
    return idx;
  }

  function onLaneDragOver(track: RenderTrack, e: React.DragEvent) {
    if (isLocked(track.id)) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    if (dropTrack !== track.id) setDropTrack(track.id);
  }

  function onLaneDrop(track: RenderTrack, e: React.DragEvent) {
    e.preventDefault();
    setDropTrack(null);
    if (isLocked(track.id)) return;
    const p = parseDrop(e);
    if (!p) return;
    pushHistory();
    if (track.kind === "video" && track.isBase) {
      addSegment(
        { file_id: p.file_id, in_ms: p.in_ms, out_ms: p.out_ms },
        spineIndexAtX(track.id, e.clientX)
      );
    } else if (track.kind === "video") {
      addOp({
        type: "place_video",
        source_file_id: p.file_id,
        src_in_ms: p.in_ms,
        src_out_ms: p.out_ms,
        from_ms: progMsAtX(track.id, e.clientX),
        z: track.z,
      });
    } else {
      addOp({
        type: "place_audio",
        source_file_id: p.file_id,
        src_in_ms: p.in_ms,
        src_out_ms: p.out_ms,
        from_ms: progMsAtX(track.id, e.clientX),
        role: track.role,
      });
    }
    // Make sure an edit session exists so the build can be saved later.
    void ensureThread();
  }

  function toggleMute(op: EditOperation) {
    pushHistory();
    const muted = (op.gain_db ?? 0) <= -119;
    if (muted) setGain(op.op_id, muteCache.current[op.op_id] ?? 0);
    else {
      muteCache.current[op.op_id] = op.gain_db ?? 0;
      setGain(op.op_id, -120);
    }
  }

  // --- track header actions (2.5 scaffolding) ---
  function toggleTrackLock(trackId: string) {
    setTrackMeta(trackId, { lock: !trackMeta[trackId]?.lock });
  }
  function toggleTrackSolo(trackId: string) {
    // View/playback flag only — coordinating the preview mixer to actually
    // silence other tracks is P1 item 4 (use-program-player integration).
    setTrackMeta(trackId, { solo: !trackMeta[trackId]?.solo });
  }
  function toggleTrackMute(track: RenderTrack) {
    const nextMuted = !trackMeta[track.id]?.mute;
    setTrackMeta(track.id, { mute: nextMuted });
    if (track.kind !== "audio" || track.pending) return; // video mute is view-only (P1)
    // Audio track mute maps onto every op on the track's gain_db, mirroring
    // the existing per-clip mute path.
    pushHistory();
    const opIds: string[] = [];
    for (const c of project.clips) {
      if (c.trackId === track.id && c.origin.kind === "op") opIds.push(c.origin.opId);
    }
    if (nextMuted) {
      const cache: Record<string, number> = {};
      for (const opId of opIds) {
        const op = operations.find((o) => o.op_id === opId);
        cache[opId] = op?.gain_db ?? 0;
        setGain(opId, -120);
      }
      trackAudioMuteCache.current[track.id] = cache;
    } else {
      const cache = trackAudioMuteCache.current[track.id] ?? {};
      for (const opId of opIds) {
        setGain(opId, cache[opId] ?? 0);
      }
    }
  }
  function startResize(trackId: string, e: React.PointerEvent) {
    e.preventDefault();
    e.stopPropagation();
    const startY = e.clientY;
    const startH = trackMeta[trackId]?.heightPx ?? DEFAULT_LANE_H;
    const onMove = (ev: PointerEvent) => {
      setTrackMeta(trackId, {
        heightPx: Math.max(MIN_LANE_H, Math.min(MAX_LANE_H, startH + (ev.clientY - startY))),
      });
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  function addVideoTrack() {
    const usedZ = new Set(project.tracks.filter((t) => t.kind === "video" && !t.isBase).map((t) => t.z));
    pendingTracks.filter((p) => p.kind === "video").forEach((p) => usedZ.add(p.z ?? 0));
    let z = 10;
    while (usedZ.has(z)) z += 1;
    setPendingTracks((cur) => [...cur, { key: `pend-v${z}`, kind: "video", z }]);
  }
  function addAudioTrack() {
    const used = new Set(
      project.tracks.filter((t) => t.kind === "audio" && !t.isBase).map((t) => t.role ?? "")
    );
    pendingTracks.filter((p) => p.kind === "audio").forEach((p) => used.add(p.role ?? ""));
    let n = 1;
    while (used.has(`bed${n}`)) n += 1;
    const role = `bed${n}`;
    setPendingTracks((cur) => [...cur, { key: `pend-a${role}`, kind: "audio", role }]);
  }

  const selectedClips = clipsForIds(selectedIds);
  const oneSelected = selectedClips.length === 1 ? selectedClips[0] : null;
  const oneOrigin = oneSelected?.origin;
  const selSeg = oneOrigin?.kind === "spine" ? timeline.find((s) => s.seg_id === oneOrigin.segId) ?? null : null;
  const selOp = oneOrigin?.kind === "op" ? operations.find((o) => o.op_id === oneOrigin.opId) ?? null : null;

  return (
    <div className="space-y-2">
      {/* Toolbar */}
      <div className="flex items-center gap-1">
        <button
          onClick={togglePlaying}
          disabled={total <= 0}
          className="rounded-full p-1.5 transition-opacity disabled:opacity-30"
          style={{ background: "var(--accent)", color: "var(--background)" }}
          title={playing ? "Pause (space)" : "Play (space)"}
        >
          {playing ? <Pause size={13} /> : <Play size={13} />}
        </button>
        <IconBtn title="Previous frame (←)" onClick={() => step(-1)}>
          <ChevronLeft size={14} />
        </IconBtn>
        <IconBtn title="Next frame (→)" onClick={() => step(1)}>
          <ChevronRight size={14} />
        </IconBtn>

        <Divider />

        <IconBtn active={tool === "select"} title="Select (V)" onClick={() => setTool("select")}>
          <MousePointer2 size={14} />
        </IconBtn>
        <IconBtn active={tool === "blade"} title="Blade (B)" onClick={() => setTool("blade")}>
          <Scissors size={14} />
        </IconBtn>
        <IconBtn active={snapEnabled} title="Snap (S)" onClick={toggleSnap}>
          <Magnet size={14} />
        </IconBtn>

        <Divider />

        <IconBtn title="Zoom out" onClick={zoomOut}>
          <ZoomOut size={14} />
        </IconBtn>
        <IconBtn title="Zoom to fit" onClick={() => zoomToFit(viewportW, total)}>
          <Maximize2 size={14} />
        </IconBtn>
        <IconBtn title="Zoom in" onClick={zoomIn}>
          <ZoomIn size={14} />
        </IconBtn>

        <Divider />

        <IconBtn title="Undo (⌘Z)" onClick={undo} disabled={!canUndo}>
          <Undo2 size={14} />
        </IconBtn>
        <IconBtn title="Redo (⌘⇧Z)" onClick={redo} disabled={!canRedo}>
          <Redo2 size={14} />
        </IconBtn>
        <IconBtn title="Split at playhead (⌘K)" onClick={splitAtPlayhead}>
          <Scissors size={14} />
        </IconBtn>
        <IconBtn title="Duplicate (⌘D)" onClick={duplicateSelection} disabled={selectedIds.length === 0}>
          <Copy size={14} />
        </IconBtn>

        <Divider />

        <IconBtn title="Add video track" onClick={addVideoTrack}>
          <TextTag>+V</TextTag>
        </IconBtn>
        <IconBtn title="Add audio track" onClick={addAudioTrack}>
          <TextTag>+A</TextTag>
        </IconBtn>
        <IconBtn title="Add marker (M)" onClick={() => addMarker(progMs)}>
          <Bookmark size={14} />
        </IconBtn>
        <IconBtn title="Set in point" onClick={() => setInMark(progMs)}>
          <TextTag>I</TextTag>
        </IconBtn>
        <IconBtn title="Set out point" onClick={() => setOutMark(progMs)}>
          <TextTag>O</TextTag>
        </IconBtn>

        <span className="ml-auto text-[11px] tabular-nums" style={{ color: "var(--muted)" }}>
          {formatTimecode(progMs)} <span style={{ opacity: 0.5 }}>/ {formatTimecode(total)}</span>
        </span>
      </div>

      {/* Header column (fixed) + scrollable ruler/lanes */}
      <div className="flex items-stretch">
        {/* Fixed track-header column */}
        <div className="shrink-0" style={{ width: HEADER_W }}>
          <div style={{ height: RULER_H }} />
          {renderTracks.map((track) => (
            <TrackHeaderRow
              key={track.id}
              track={track}
              meta={trackMeta[track.id]}
              onToggleMute={() => toggleTrackMute(track)}
              onToggleSolo={() => toggleTrackSolo(track.id)}
              onToggleLock={() => toggleTrackLock(track.id)}
              onResizeStart={(e) => startResize(track.id, e)}
            />
          ))}
        </div>

        {/* Scrollable content: ruler + lanes + playhead + snap guide + I/O band, one shared scroll */}
        <div
          ref={scrollRef}
          onScroll={onContentScroll}
          className="relative min-w-0 flex-1 overflow-x-auto overflow-y-hidden"
          style={{ touchAction: "pan-x" }}
        >
          <div className="relative" style={{ width: Math.max(contentWidth, viewportW) }}>
            <TimeRuler
              total={total}
              pxPerMs={pxPerMs}
              onSeek={seek}
              markers={markers}
              onRemoveMarker={removeMarker}
            />

            {/* In/out shaded band */}
            {inMarkMs != null && outMarkMs != null && pxPerMs > 0 && (
              <div
                className="pointer-events-none absolute top-0"
                style={{
                  left: inMarkMs * pxPerMs,
                  width: Math.max(1, (outMarkMs - inMarkMs) * pxPerMs),
                  height: RULER_H,
                  background: "var(--accent-soft)",
                }}
              />
            )}

            {/* Tracks */}
            <div>
              {renderTracks.map((track) => {
                // A pending placeholder never has real clips (its id can't
                // match any clip's trackId until a drop creates the op).
                const trackClips = project.clips.filter((c) => c.trackId === track.id);
                const isWidthTrack = track.kind === "video" && track.isBase;
                const showDropHint = isWidthTrack && trackClips.length === 0 && timeline.length === 0;
                const height = trackMeta[track.id]?.heightPx ?? DEFAULT_LANE_H;
                const locked = !!trackMeta[track.id]?.lock;
                return (
                  <div
                    key={track.id}
                    onPointerDown={startMarquee}
                    onDragOver={(e) => onLaneDragOver(track, e)}
                    onDragLeave={() => setDropTrack(null)}
                    onDrop={(e) => onLaneDrop(track, e)}
                    ref={(el) => {
                      const m = trackEls.current;
                      if (el) m.set(track.id, el);
                      else m.delete(track.id);
                    }}
                    className="relative border-b"
                    style={{
                      height,
                      borderColor: "var(--border)",
                      background: dropTrack === track.id ? "var(--accent-soft)" : "transparent",
                      outline: dropTrack === track.id ? "2px dashed var(--accent)" : "none",
                      outlineOffset: -2,
                      opacity: locked ? 0.55 : 1,
                      cursor: tool === "blade" ? "crosshair" : undefined,
                      touchAction: "none",
                    }}
                  >
                    {showDropHint && (
                      <div
                        className="pointer-events-none absolute inset-0 flex items-center justify-center text-[11px]"
                        style={{ color: "var(--muted)" }}
                      >
                        Drag cuts here to build your edit
                      </div>
                    )}
                    {trackClips.map((clip) => {
                      const selectedClip = selectedIds.includes(clip.id);
                      const bladeClick =
                        tool === "blade" && clip.origin.kind === "spine"
                          ? (e: React.PointerEvent) => {
                              e.stopPropagation();
                              clickToSplit(clip, e);
                            }
                          : undefined;
                      const bodyDrag =
                        tool === "blade"
                          ? bladeClick
                          : clip.movable
                          ? (e: React.PointerEvent) => startMove(clip, e)
                          : clip.kind === "video" && clip.origin.kind === "spine"
                          ? (e: React.PointerEvent) => startReorder(clip, e)
                          : (e: React.PointerEvent) => {
                              // Non-draggable clips (e.g. coupled dialogue) still
                              // need to stop the marquee from starting on them.
                              e.stopPropagation();
                            };
                      return (
                        <Block
                          key={clip.id}
                          left={clip.progStartMs * pxPerMs}
                          width={Math.max(8, (clip.progEndMs - clip.progStartMs) * pxPerMs)}
                          selected={selectedClip}
                          onClick={(e) => (tool === "select" ? selectClip(clip, e) : undefined)}
                          onBodyPointerDown={bodyDrag}
                          color={clip.color}
                          muted={clip.muted}
                          movable={tool === "select" && (clip.movable || (clip.kind === "video" && clip.origin.kind === "spine"))}
                          title={`${clip.label} · ${fmt(clip.progStartMs)}–${fmt(clip.progEndMs)}`}
                        >
                          {tool === "select" && clip.trimmable && (
                            <span
                              onPointerDown={(e) => startTrim(clip, "in", e)}
                              className="absolute left-0 top-0 h-full w-1.5 cursor-ew-resize"
                              style={{ background: "rgba(0,0,0,0.35)" }}
                            />
                          )}
                          <span className="pointer-events-none truncate px-2">{clip.label}</span>
                          {tool === "select" && clip.trimmable && (
                            <span
                              onPointerDown={(e) => startTrim(clip, "out", e)}
                              className="absolute right-0 top-0 h-full w-1.5 cursor-ew-resize"
                              style={{ background: "rgba(0,0,0,0.35)" }}
                            />
                          )}
                        </Block>
                      );
                    })}
                  </div>
                );
              })}
            </div>

            {/* Snap guide */}
            {snapGuideMs != null && pxPerMs > 0 && (
              <div
                className="pointer-events-none absolute top-0 z-20"
                style={{
                  left: snapGuideMs * pxPerMs,
                  top: 0,
                  bottom: 0,
                  width: 1,
                  background: "var(--accent)",
                }}
              />
            )}

            {/* Playhead — spans the ruler + every lane */}
            {pxPerMs > 0 && (
              <div className="pointer-events-none absolute bottom-0 z-10" style={{ left: playheadPx, top: 0 }}>
                <div className="h-full" style={{ width: 2, background: "var(--foreground)" }} />
                <div
                  className="absolute"
                  style={{
                    top: 0,
                    left: 1,
                    transform: "translateX(-50%)",
                    width: 0,
                    height: 0,
                    borderLeft: "4px solid transparent",
                    borderRight: "4px solid transparent",
                    borderTop: "6px solid var(--foreground)",
                  }}
                />
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Marquee overlay (screen-space; independent of the scroll transform) */}
      {marquee && (
        <div
          className="pointer-events-none fixed z-30"
          style={{
            left: Math.min(marquee.x0, marquee.x1),
            top: Math.min(marquee.y0, marquee.y1),
            width: Math.abs(marquee.x1 - marquee.x0),
            height: Math.abs(marquee.y1 - marquee.y0),
            background: "var(--accent-soft)",
            outline: "1px solid var(--accent)",
          }}
        />
      )}

      {/* Inspector: multi-select summary, or the single selected clip/layer */}
      {selectedClips.length > 1 && (
        <div className="flex flex-wrap items-center gap-2 border-t pt-2 text-xs" style={{ borderColor: "var(--border)" }}>
          <span className="font-medium">{selectedClips.length} clips selected</span>
          <div className="ml-auto flex items-center gap-1">
            <IconBtn title="Duplicate (⌘D)" onClick={duplicateSelection}><Copy size={13} /></IconBtn>
            <IconBtn title="Lift delete (Delete)" onClick={() => deleteSelected(false)} danger><Trash2 size={13} /></IconBtn>
          </div>
        </div>
      )}

      {selectedClips.length === 1 && selSeg && (
        <div className="flex flex-wrap items-center gap-1.5 border-t pt-2 text-xs" style={{ borderColor: "var(--border)" }}>
          <span style={{ color: "var(--muted)" }}>{selSeg.file_id.slice(0, 6)}</span>
          <span className="flex items-center gap-1">
            <TcField ms={selSeg.in_ms} title="In" onCommit={(ms) => { pushHistory(); trimSeg(selSeg.seg_id, "in", ms); }} />
            <span style={{ color: "var(--muted)" }}>–</span>
            <TcField ms={selSeg.out_ms} title="Out" onCommit={(ms) => { pushHistory(); trimSeg(selSeg.seg_id, "out", ms); }} />
            <span style={{ color: "var(--muted)" }}>dur</span>
            <TcField
              ms={selSeg.out_ms - selSeg.in_ms}
              title="Duration"
              onCommit={(ms) => { pushHistory(); trimSeg(selSeg.seg_id, "out", selSeg.in_ms + ms); }}
            />
          </span>
          <div className="ml-auto flex items-center gap-1">
            <IconBtn title="Move left" onClick={() => { pushHistory(); moveSeg(selSeg.seg_id, -1); }}><ChevronLeft size={14} /></IconBtn>
            <IconBtn title="Move right" onClick={() => { pushHistory(); moveSeg(selSeg.seg_id, 1); }}><ChevronRight size={14} /></IconBtn>
            <IconBtn title="Split at middle" onClick={() => { pushHistory(); splitSeg(selSeg.seg_id); }}><Scissors size={13} /></IconBtn>
            <IconBtn title="Delete cut" onClick={() => { pushHistory(); removeSeg(selSeg.seg_id); }} danger><Trash2 size={13} /></IconBtn>
          </div>
        </div>
      )}

      {selectedClips.length === 1 && selOp && (
        <div className="flex flex-wrap items-center gap-2 border-t pt-2 text-xs" style={{ borderColor: "var(--border)" }}>
          <span className="font-medium">
            {selOp.type === "place_video"
              ? "Coverage"
              : selOp.role === "sfx"
                ? "SFX"
                : "Music"}
          </span>
          <span className="flex items-center gap-1">
            <TcField
              ms={selOp.from_ms ?? 0}
              title="In"
              onCommit={(ms) => { pushHistory(); setOpEdge(selOp.op_id, "in", ms, total); }}
            />
            <span style={{ color: "var(--muted)" }}>–</span>
            <TcField
              ms={selOp.to_ms ?? 0}
              title="Out"
              onCommit={(ms) => { pushHistory(); setOpEdge(selOp.op_id, "out", ms, total); }}
            />
            <span style={{ color: "var(--muted)" }}>dur</span>
            <TcField
              ms={(selOp.to_ms ?? 0) - (selOp.from_ms ?? 0)}
              title="Duration"
              onCommit={(ms) => { pushHistory(); setOpEdge(selOp.op_id, "out", (selOp.from_ms ?? 0) + ms, total); }}
            />
          </span>
          {(selOp.type === "place_audio") && (
            <>
              <input
                type="range"
                min={-30}
                max={6}
                step={1}
                value={(selOp.gain_db ?? 0) <= -119 ? -30 : Math.max(-30, Math.min(6, selOp.gain_db ?? 0))}
                onPointerDown={() => pushHistory()}
                onChange={(e) => setGain(selOp.op_id, Number(e.target.value))}
                className="ml-auto w-24 accent-[var(--accent)]"
                title="Gain (dB)"
              />
              <button onClick={() => toggleMute(selOp)} title={(selOp.gain_db ?? 0) <= -119 ? "Unmute" : "Mute"} className="rounded p-1 hover:bg-[var(--accent-soft)]">
                {(selOp.gain_db ?? 0) <= -119 ? <VolumeX size={13} /> : <Volume2 size={13} />}
              </button>
            </>
          )}
          <button
            onClick={() => { pushHistory(); removeOpStore(selOp.op_id); clearSelection(); }}
            title="Delete layer"
            className={`rounded p-1 hover:bg-[var(--accent-soft)] ${selOp.type === "place_video" ? "ml-auto" : ""}`}
          >
            <Trash2 size={13} style={{ color: "var(--danger)" }} />
          </button>
        </div>
      )}
    </div>
  );
}

/** Typeable, frame-snapped "[H:]MM:SS:FF" field. Edits are staged locally
 * and only committed (via `onCommit`) on blur/Enter; Escape reverts. */
function TcField({ ms, onCommit, title }: { ms: number; onCommit: (ms: number) => void; title?: string }) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState(formatTimecode(ms));
  useEffect(() => {
    if (!editing) setText(formatTimecode(ms));
  }, [ms, editing]);

  function commit() {
    const parsed = parseTimecode(text);
    if (parsed != null) onCommit(snapMs(parsed));
    else setText(formatTimecode(ms));
    setEditing(false);
  }

  return (
    <input
      value={text}
      title={title}
      onFocus={() => setEditing(true)}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          (e.target as HTMLInputElement).blur();
        } else if (e.key === "Escape") {
          setText(formatTimecode(ms));
          setEditing(false);
          (e.target as HTMLInputElement).blur();
        }
      }}
      className="w-[68px] rounded border bg-transparent px-1 py-0.5 text-[10px] tabular-nums outline-none focus:border-[var(--accent)]"
      style={{ borderColor: "var(--border)" }}
    />
  );
}

/** Clickable/scrubbable time ruler + marker pips, aligned to the lane tracks. */
function TimeRuler({
  total,
  pxPerMs,
  onSeek,
  markers,
  onRemoveMarker,
}: {
  total: number;
  pxPerMs: number;
  onSeek: (ms: number) => void;
  markers: number[];
  onRemoveMarker: (ms: number) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  const seekAtClientX = useCallback(
    (clientX: number) => {
      const el = ref.current;
      if (!el || pxPerMs <= 0) return;
      const rect = el.getBoundingClientRect();
      onSeek((clientX - rect.left) / pxPerMs);
    },
    [pxPerMs, onSeek]
  );

  const onPointerDown = (e: React.PointerEvent) => {
    if (pxPerMs <= 0) return;
    e.preventDefault();
    draggingRef.current = true;
    seekAtClientX(e.clientX);
    const onMove = (ev: PointerEvent) => {
      if (draggingRef.current) seekAtClientX(ev.clientX);
    };
    const onUp = () => {
      draggingRef.current = false;
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  };

  const ticks: React.ReactNode[] = [];
  if (pxPerMs > 0 && total > 0) {
    const secPx = 1000 * pxPerMs;
    const candidates = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600];
    let stepSec = candidates[candidates.length - 1];
    for (const s of candidates) {
      if (s * secPx >= 48) {
        stepSec = s;
        break;
      }
    }
    for (let sec = 0; sec * 1000 <= total + 1; sec += stepSec) {
      const left = sec * 1000 * pxPerMs;
      ticks.push(
        <div key={sec} className="absolute top-0 flex flex-col items-start" style={{ left }}>
          <div style={{ width: 1, height: 5, background: "var(--border)" }} />
          <span className="mt-0.5 text-[9px] tabular-nums" style={{ color: "var(--muted)" }}>
            {formatTimecode(sec * 1000)}
          </span>
        </div>
      );
    }
  }

  return (
    <div
      ref={ref}
      onPointerDown={onPointerDown}
      className="relative cursor-pointer select-none"
      style={{ height: RULER_H, touchAction: "none" }}
      title="Click or drag to scrub"
    >
      {ticks}
      {pxPerMs > 0 &&
        markers.map((ms) => (
          <div
            key={ms}
            onPointerDown={(e) => {
              e.stopPropagation();
              onRemoveMarker(ms);
            }}
            title={`Marker @ ${formatTimecode(ms)} — click to remove`}
            className="absolute bottom-0"
            style={{ left: ms * pxPerMs - 3 }}
          >
            <div
              style={{
                width: 0,
                height: 0,
                borderLeft: "3px solid transparent",
                borderRight: "3px solid transparent",
                borderBottom: "5px solid var(--accent)",
              }}
            />
          </div>
        ))}
    </div>
  );
}

function TrackHeaderRow({
  track,
  meta,
  onToggleMute,
  onToggleSolo,
  onToggleLock,
  onResizeStart,
}: {
  track: RenderTrack;
  meta: TrackMeta | undefined;
  onToggleMute: () => void;
  onToggleSolo: () => void;
  onToggleLock: () => void;
  onResizeStart: (e: React.PointerEvent) => void;
}) {
  const height = meta?.heightPx ?? DEFAULT_LANE_H;
  return (
    <div
      className="relative flex items-center gap-0.5 border-b pl-2 pr-1 text-[10px]"
      style={{ height, borderColor: "var(--border)" }}
    >
      <span className="min-w-0 flex-1 truncate font-medium" style={{ color: "var(--muted)" }}>
        {track.label}
      </span>
      <button
        onClick={onToggleMute}
        title={track.kind === "audio" ? (meta?.mute ? "Unmute" : "Mute") : "Hide in preview (view-only)"}
        className="rounded p-0.5 hover:bg-[var(--accent-soft)]"
        style={meta?.mute ? { color: "var(--accent)" } : undefined}
      >
        {meta?.mute ? <VolumeX size={11} /> : <Volume2 size={11} />}
      </button>
      <button
        onClick={onToggleSolo}
        title={meta?.solo ? "Unsolo" : "Solo"}
        className="rounded p-0.5 hover:bg-[var(--accent-soft)]"
        style={meta?.solo ? { color: "var(--accent)" } : undefined}
      >
        <Headphones size={11} />
      </button>
      <button
        onClick={onToggleLock}
        title={meta?.lock ? "Unlock" : "Lock"}
        className="rounded p-0.5 hover:bg-[var(--accent-soft)]"
        style={meta?.lock ? { color: "var(--accent)" } : undefined}
      >
        {meta?.lock ? <Lock size={11} /> : <Unlock size={11} />}
      </button>
      <div
        onPointerDown={onResizeStart}
        className="absolute inset-x-0 bottom-0 h-1 cursor-row-resize"
        title="Drag to resize"
      />
    </div>
  );
}

function Block({
  left,
  width,
  selected,
  onClick,
  onBodyPointerDown,
  color,
  title,
  children,
  muted,
  movable,
}: {
  left: number;
  width: number;
  selected: boolean;
  onClick?: (e: React.MouseEvent) => void;
  onBodyPointerDown?: (e: React.PointerEvent) => void;
  color: string;
  title: string;
  children: React.ReactNode;
  muted?: boolean;
  movable?: boolean;
}) {
  return (
    <div
      onClick={onClick}
      onPointerDown={onBodyPointerDown}
      title={title}
      className={`absolute top-0 flex h-full items-center overflow-hidden rounded text-[10px] text-white ${
        movable ? "cursor-grab active:cursor-grabbing" : "cursor-pointer"
      }`}
      style={{
        left,
        width,
        background: color,
        opacity: muted ? 0.45 : 1,
        outline: selected ? "2px solid var(--foreground)" : "none",
        outlineOffset: -2,
        touchAction: "none",
      }}
    >
      {children}
    </div>
  );
}

function IconBtn({
  children,
  onClick,
  title,
  danger,
  active,
  disabled,
}: {
  children: React.ReactNode;
  onClick: () => void;
  title: string;
  danger?: boolean;
  active?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      disabled={disabled}
      className="rounded p-1 transition-colors hover:bg-[var(--accent-soft)] disabled:cursor-default disabled:opacity-30 disabled:hover:bg-transparent"
      style={{
        color: danger ? "var(--danger)" : active ? "var(--accent)" : undefined,
        background: active ? "var(--accent-soft)" : undefined,
      }}
    >
      {children}
    </button>
  );
}

function TextTag({ children }: { children: React.ReactNode }) {
  return <span className="px-0.5 text-[10px] font-semibold">{children}</span>;
}

function Divider() {
  return <div className="mx-1 h-4 w-px shrink-0" style={{ background: "var(--border)" }} />;
}
