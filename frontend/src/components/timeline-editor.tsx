"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Scissors,
  Trash2,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  ChevronDown,
  Volume2,
  VolumeX,
  Play,
  Pause,
  Undo2,
  Redo2,
  ZoomIn,
  ZoomOut,
  Maximize2,
  Magnet,
  Copy,
} from "lucide-react";
import { getFilePlaybackUrl, type EditOperation } from "@/lib/api";
import { useEditDocStore } from "@/stores/edit-doc-store";
import { useDriveStore } from "@/stores/drive-store";
import { useAuthStore } from "@/stores/auth-store";
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
  type ProjectTrack,
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
  const splitOp = useEditDocStore((s) => s.splitOp);
  const setOpFrom = useEditDocStore((s) => s.setOpFrom);
  const setOpEdge = useEditDocStore((s) => s.setOpEdge);
  const setOpZ = useEditDocStore((s) => s.setOpZ);
  const swapVideoZ = useEditDocStore((s) => s.swapVideoZ);
  const reorderSeg = useEditDocStore((s) => s.reorderSeg);
  const addSegment = useEditDocStore((s) => s.addSegment);
  const addOp = useEditDocStore((s) => s.addOp);
  const pushHistory = useEditDocStore((s) => s.pushHistory);
  const undo = useEditDocStore((s) => s.undo);
  const redo = useEditDocStore((s) => s.redo);
  const canUndo = useEditDocStore((s) => s.canUndo());
  const canRedo = useEditDocStore((s) => s.canRedo());

  const driveFiles = useDriveStore((s) => s.files);
  const fileNameById = useMemo(() => {
    const m = new Map<string, string>();
    for (const f of driveFiles) m.set(f.id, f.name);
    return m;
  }, [driveFiles]);
  function fileNameFor(fileId: string): string {
    return fileNameById.get(fileId) ?? fileId.slice(0, 6);
  }

  const pxPerSec = useTimelineView((s) => s.pxPerSec);
  const scrollLeftPx = useTimelineView((s) => s.scrollLeftPx);
  const snapEnabled = useTimelineView((s) => s.snapEnabled);
  const snapGuideMs = useTimelineView((s) => s.snapGuideMs);
  const trackMeta = useTimelineView((s) => s.trackMeta);
  const clipboard = useTimelineView((s) => s.clipboard);
  const setZoom = useTimelineView((s) => s.setZoom);
  const zoomIn = useTimelineView((s) => s.zoomIn);
  const zoomOut = useTimelineView((s) => s.zoomOut);
  const zoomToFit = useTimelineView((s) => s.zoomToFit);
  const setScrollLeft = useTimelineView((s) => s.setScrollLeft);
  const toggleSnap = useTimelineView((s) => s.toggleSnap);
  const setSnapGuide = useTimelineView((s) => s.setSnapGuide);
  const setTrackMeta = useTimelineView((s) => s.setTrackMeta);
  const setClipboard = useTimelineView((s) => s.setClipboard);

  const [viewportW, setViewportW] = useState(600);
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

  // --- delete: one behavior. Spine always ripples (it's gapless by
  // construction); ops always lift (removed, leaving their slot empty). ---
  function deleteSelected() {
    const clips = clipsForIds(selectedIds);
    if (!clips.length) return;
    pushHistory();
    for (const c of clips) {
      if (c.origin.kind === "spine") removeSeg(c.origin.segId);
      else removeOpStore(c.origin.opId);
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
      const dur = Math.max(0, entry.srcOutMs - entry.srcInMs);
      // Only an EXISTING track needs room made in it; a paste that lands on a
      // not-yet-real z/role just becomes that track's first clip.
      const trackId =
        entry.kind === "video"
          ? project.tracks.find((t) => t.kind === "video" && t.z === (entry.trackHint?.z ?? 10))?.id
          : project.tracks.find((t) => t.kind === "audio" && !t.isBase && t.role === entry.trackHint?.role)?.id;
      if (trackId) makeRoomInsert(trackId, progMs, dur);
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
        deleteSelected();
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
      } else if (!mod && e.key.toLowerCase() === "s") {
        toggleSnap();
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
  }, [togglePlaying, step, undo, redo, toggleSnap, selectedIds, clipboard, progMs, project]);

  // --- snapping helper shared by every drag gesture ---
  const snapProgramMs = useCallback(
    (rawMs: number, excludeClipId?: string): number => {
      if (!snapEnabled) {
        setSnapGuide(null);
        return rawMs;
      }
      const extra = [progMs];
      const targets = collectSnapTargets(project, { excludeClipId, extra });
      const { value, snappedTo } = snapValue(rawMs, targets, pxPerMs, SNAP_THRESHOLD_PX);
      setSnapGuide(snappedTo);
      return value;
    },
    [snapEnabled, progMs, project, pxPerMs, setSnapGuide]
  );

  /** The OTHER half of a spine segment's A/V pair ("seg:x" <-> "dlg:x"), or
   * null for anything else (ops have no pair). */
  function pairedClipId(clip: ProjectClip): string | null {
    if (clip.origin.kind !== "spine") return null;
    if (clip.id.startsWith("seg:")) return `dlg:${clip.origin.segId}`;
    if (clip.id.startsWith("dlg:")) return `seg:${clip.origin.segId}`;
    return null;
  }

  // The spine's video and its coupled dialogue always select together (a
  // spine segment's A/V is never independently splittable -- see
  // pairedClipId's own comment).
  function selectClip(clip: ProjectClip, e?: { shiftKey?: boolean }) {
    const pairId = pairedClipId(clip);
    const ids = pairId && project.clips.some((c) => c.id === pairId) ? [clip.id, pairId] : [clip.id];
    if (e?.shiftKey) {
      const allIn = ids.every((id) => selectedIds.includes(id));
      select(
        allIn
          ? selectedIds.filter((id) => !ids.includes(id))
          : Array.from(new Set([...selectedIds, ...ids]))
      );
    } else {
      select(ids);
    }
  }

  // --- trim a clip's edge by dragging its handle (frame-snapped + snapped) ---
  function startTrim(clip: ProjectClip, edge: "in" | "out", e: React.PointerEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (pxPerMs <= 0 || !clip.trimmable) return;
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
    for (const track of project.tracks) {
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
    if (!clip.movable || clip.origin.kind !== "op") return;
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
    if (clip.origin.kind !== "spine" || clip.kind !== "video") return;
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

  function onLaneDragOver(track: ProjectTrack, e: React.DragEvent) {
    e.preventDefault();
    e.dataTransfer.dropEffect = "copy";
    if (dropTrack !== track.id) setDropTrack(track.id);
  }

  // Ripple-insert: pushes every later op on the SAME track right by the new
  // clip's duration.
  function makeRoomInsert(trackId: string, atMs: number, dur: number) {
    for (const c of project.clips) {
      if (c.trackId !== trackId || c.origin.kind !== "op") continue;
      if (c.progStartMs >= atMs) setOpFrom(c.origin.opId, c.progStartMs + dur, total + dur);
    }
  }

  function onLaneDrop(track: ProjectTrack, e: React.DragEvent) {
    e.preventDefault();
    setDropTrack(null);
    const p = parseDrop(e);
    if (!p) return;
    pushHistory();
    const dur = Math.max(0, p.out_ms - p.in_ms);
    if (track.kind === "video" && track.isBase) {
      addSegment(
        { file_id: p.file_id, in_ms: p.in_ms, out_ms: p.out_ms },
        spineIndexAtX(track.id, e.clientX)
      );
    } else if (track.kind === "video") {
      const at = progMsAtX(track.id, e.clientX);
      makeRoomInsert(track.id, at, dur);
      addOp({
        type: "place_video",
        source_file_id: p.file_id,
        src_in_ms: p.in_ms,
        src_out_ms: p.out_ms,
        from_ms: at,
        z: track.z,
      });
    } else {
      const at = progMsAtX(track.id, e.clientX);
      makeRoomInsert(track.id, at, dur);
      addOp({
        type: "place_audio",
        source_file_id: p.file_id,
        src_in_ms: p.in_ms,
        src_out_ms: p.out_ms,
        from_ms: at,
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
  function toggleTrackMute(track: ProjectTrack) {
    const nextMuted = !trackMeta[track.id]?.mute;
    setTrackMeta(track.id, { mute: nextMuted });
    if (track.kind !== "audio") return; // video mute is view-only (P1)
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

        <span className="ml-auto text-[11px] tabular-nums" style={{ color: "var(--muted)" }}>
          {formatTimecode(progMs)} <span style={{ opacity: 0.5 }}>/ {formatTimecode(total)}</span>
        </span>
      </div>

      {/* Header column (fixed) + scrollable ruler/lanes */}
      <div className="flex items-stretch">
        {/* Fixed track-header column */}
        <div className="shrink-0" style={{ width: HEADER_W }}>
          <div style={{ height: RULER_H }} />
          {project.tracks.map((track, i) => {
            // Upper video tracks render highest-z first (see edit-project.ts's
            // `upperVideo` sort) -- "move up" = swap with the track ABOVE (one
            // index earlier / higher z); "move down" = the one below.
            const isReorderableVideo = track.kind === "video" && !track.isBase;
            const above = isReorderableVideo ? project.tracks[i - 1] : undefined;
            const below = isReorderableVideo ? project.tracks[i + 1] : undefined;
            const canMoveUp = isReorderableVideo && above?.kind === "video" && !above.isBase;
            const canMoveDown = isReorderableVideo && below?.kind === "video" && !below.isBase;
            return (
              <TrackHeaderRow
                key={track.id}
                track={track}
                meta={trackMeta[track.id]}
                onToggleMute={() => toggleTrackMute(track)}
                onMoveUp={canMoveUp ? () => swapVideoZ(track.z, above!.z) : undefined}
                onMoveDown={canMoveDown ? () => swapVideoZ(track.z, below!.z) : undefined}
              />
            );
          })}
        </div>

        {/* Scrollable content: ruler + lanes + playhead + snap guide, one shared scroll */}
        <div
          ref={scrollRef}
          onScroll={onContentScroll}
          className="relative min-w-0 flex-1 overflow-x-auto overflow-y-hidden"
          style={{ touchAction: "pan-x" }}
        >
          <div className="relative" style={{ width: Math.max(contentWidth, viewportW) }}>
            <TimeRuler total={total} pxPerMs={pxPerMs} onSeek={seek} />

            {/* Tracks */}
            <div>
              {project.tracks.map((track) => {
                const trackClips = project.clips.filter((c) => c.trackId === track.id);
                const isWidthTrack = track.kind === "video" && track.isBase;
                const showDropHint = isWidthTrack && trackClips.length === 0 && timeline.length === 0;
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
                      height: DEFAULT_LANE_H,
                      borderColor: "var(--border)",
                      background: dropTrack === track.id ? "var(--accent-soft)" : "transparent",
                      outline: dropTrack === track.id ? "2px dashed var(--accent)" : "none",
                      outlineOffset: -2,
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
                      const bodyDrag =
                        clip.movable
                          ? (e: React.PointerEvent) => startMove(clip, e)
                          : clip.kind === "video" && clip.origin.kind === "spine"
                          ? (e: React.PointerEvent) => startReorder(clip, e)
                          : (e: React.PointerEvent) => {
                              // Non-draggable clips (e.g. coupled dialogue) still
                              // need to stop the marquee from starting on them.
                              e.stopPropagation();
                            };
                      const isDlg = clip.kind === "audio" && clip.origin.kind === "spine";
                      const displayName = isDlg ? clip.label : fileNameFor(clip.sourceFileId);
                      const blockWidth = Math.max(8, (clip.progEndMs - clip.progStartMs) * pxPerMs);
                      return (
                        <Block
                          key={clip.id}
                          left={clip.progStartMs * pxPerMs}
                          width={blockWidth}
                          selected={selectedClip}
                          onClick={(e) => selectClip(clip, e)}
                          onBodyPointerDown={bodyDrag}
                          color={clip.color}
                          muted={clip.muted}
                          movable={clip.movable || (clip.kind === "video" && clip.origin.kind === "spine")}
                          title={`${displayName} · ${fmt(clip.progStartMs)}–${fmt(clip.progEndMs)}`}
                        >
                          {clip.trimmable && (
                            <span
                              onPointerDown={(e) => startTrim(clip, "in", e)}
                              className="absolute left-0 top-0 h-full w-1.5 cursor-ew-resize"
                              style={{ background: "rgba(0,0,0,0.35)" }}
                            />
                          )}
                          {clip.kind === "video" && !isDlg && (
                            <Filmstrip
                              fileId={clip.sourceFileId}
                              srcInMs={clip.srcInMs}
                              srcOutMs={clip.srcOutMs}
                              widthPx={blockWidth}
                            />
                          )}
                          {clip.kind === "audio" && (
                            <Waveform fileId={clip.sourceFileId} widthPx={blockWidth} />
                          )}
                          <span className="pointer-events-none relative truncate px-2">{displayName}</span>
                          {blockWidth > 64 && (
                            <span
                              className="pointer-events-none absolute right-1.5 shrink-0 text-[9px] tabular-nums"
                              style={{ color: "rgba(255,255,255,0.75)", textShadow: "0 1px 1px rgba(0,0,0,0.6)" }}
                            >
                              {fmt(clip.progEndMs - clip.progStartMs)}
                            </span>
                          )}
                          {clip.trimmable && (
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
            <IconBtn title="Delete" onClick={deleteSelected} danger><Trash2 size={13} /></IconBtn>
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

/** Clickable/scrubbable time ruler, aligned to the lane tracks. */
function TimeRuler({
  total,
  pxPerMs,
  onSeek,
}: {
  total: number;
  pxPerMs: number;
  onSeek: (ms: number) => void;
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
    </div>
  );
}

function TrackHeaderRow({
  track,
  meta,
  onToggleMute,
  onMoveUp,
  onMoveDown,
}: {
  track: ProjectTrack;
  meta: TrackMeta | undefined;
  onToggleMute: () => void;
  onMoveUp?: () => void;
  onMoveDown?: () => void;
}) {
  return (
    <div
      className="relative flex items-center gap-0.5 border-b pl-2 pr-1 text-[10px]"
      style={{ height: DEFAULT_LANE_H, borderColor: "var(--border)" }}
    >
      <span className="min-w-0 flex-1 truncate font-medium" style={{ color: "var(--muted)" }}>
        {track.label}
      </span>
      {(onMoveUp || onMoveDown) && (
        <span className="flex flex-col">
          <button
            onClick={onMoveUp}
            disabled={!onMoveUp}
            title="Move layer up"
            className="rounded hover:bg-[var(--accent-soft)] disabled:opacity-20"
          >
            <ChevronUp size={9} />
          </button>
          <button
            onClick={onMoveDown}
            disabled={!onMoveDown}
            title="Move layer down"
            className="rounded hover:bg-[var(--accent-soft)] disabled:opacity-20"
          >
            <ChevronDown size={9} />
          </button>
        </span>
      )}
      <button
        onClick={onToggleMute}
        title={track.kind === "audio" ? (meta?.mute ? "Unmute" : "Mute") : "Hide in preview (view-only)"}
        className="rounded p-0.5 hover:bg-[var(--accent-soft)]"
        style={meta?.mute ? { color: "var(--accent)" } : undefined}
      >
        {meta?.mute ? <VolumeX size={11} /> : <Volume2 size={11} />}
      </button>
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

// --------------------------------------------------------------------------
// Clip visuals (P1.3): video filmstrip + audio waveform.
//
// Best-effort: a decode/CORS failure (presigned R2 URLs aren't guaranteed to
// allow canvas pixel access / cross-origin fetch in every deployment) just
// leaves the plain color block, never breaks the timeline. Everything is
// cached module-wide by fileId (URL, decoded waveform) or {fileId, ms}
// (captured frames) so repeat clips of the same source cost nothing extra,
// and capture only starts once the block has actually scrolled into view.
// --------------------------------------------------------------------------

const playbackUrlCache = new Map<string, Promise<string>>();
function cachedPlaybackUrl(fileId: string, token: string): Promise<string> {
  let p = playbackUrlCache.get(fileId);
  if (!p) {
    p = getFilePlaybackUrl(fileId, token).then((r) => r.url);
    playbackUrlCache.set(fileId, p);
  }
  return p;
}

/** Runs `fn`s for the same file one at a time (seeking a shared <video> is
 * not safe to interleave across concurrent callers). */
const fileQueues = new Map<string, Promise<unknown>>();
function enqueueForFile<T>(fileId: string, fn: () => Promise<T>): Promise<T> {
  const prev = fileQueues.get(fileId) ?? Promise.resolve();
  const next = prev.then(fn, fn);
  fileQueues.set(fileId, next.catch(() => undefined));
  return next;
}

function useInView<T extends HTMLElement>(): [React.RefObject<T | null>, boolean] {
  const ref = useRef<T | null>(null);
  const [inView, setInView] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el || inView) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) setInView(true);
      },
      { root: null, rootMargin: "200px" }
    );
    io.observe(el);
    return () => io.disconnect();
  }, [inView]);
  return [ref, inView];
}

const frameCache = new Map<string, string>();
const videoElCache = new Map<string, HTMLVideoElement>();

function captureFrame(fileId: string, url: string, ms: number): Promise<string | null> {
  const key = `${fileId}:${ms}`;
  const cached = frameCache.get(key);
  if (cached) return Promise.resolve(cached);
  return enqueueForFile(fileId, async () => {
    const already = frameCache.get(key);
    if (already) return already;
    try {
      let v = videoElCache.get(fileId);
      if (!v) {
        v = document.createElement("video");
        v.crossOrigin = "anonymous";
        v.muted = true;
        v.preload = "metadata";
        v.src = url;
        videoElCache.set(fileId, v);
        await new Promise<void>((resolve, reject) => {
          const onMeta = () => { v!.removeEventListener("loadedmetadata", onMeta); resolve(); };
          const onErr = () => { v!.removeEventListener("error", onErr); reject(new Error("load failed")); };
          v!.addEventListener("loadedmetadata", onMeta);
          v!.addEventListener("error", onErr);
        });
      }
      await new Promise<void>((resolve, reject) => {
        const onSeeked = () => { v!.removeEventListener("seeked", onSeeked); resolve(); };
        const onErr = () => { v!.removeEventListener("error", onErr); reject(new Error("seek failed")); };
        v!.addEventListener("seeked", onSeeked);
        v!.addEventListener("error", onErr);
        v!.currentTime = Math.max(0, ms / 1000);
      });
      const canvas = document.createElement("canvas");
      canvas.width = 80;
      canvas.height = 45;
      const ctx = canvas.getContext("2d");
      if (!ctx) return null;
      ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.6);
      frameCache.set(key, dataUrl);
      return dataUrl;
    } catch {
      return null; // CORS / decode failure -- render nothing, block stays solid
    }
  });
}

const THUMB_W_PX = 40;
const MAX_THUMBS = 8;

function Filmstrip({
  fileId,
  srcInMs,
  srcOutMs,
  widthPx,
}: {
  fileId: string;
  srcInMs: number;
  srcOutMs: number;
  widthPx: number;
}) {
  const token = useAuthStore((s) => s.session?.access_token);
  const [ref, inView] = useInView<HTMLDivElement>();
  const [frames, setFrames] = useState<(string | null)[]>([]);
  const n = Math.max(1, Math.min(MAX_THUMBS, Math.floor(widthPx / THUMB_W_PX)));

  useEffect(() => {
    if (!inView || !token || widthPx < THUMB_W_PX) return;
    let cancelled = false;
    (async () => {
      try {
        const url = await cachedPlaybackUrl(fileId, token);
        const dur = Math.max(1, srcOutMs - srcInMs);
        const points = Array.from({ length: n }, (_, i) => srcInMs + ((i + 0.5) / n) * dur);
        const out: (string | null)[] = [];
        for (const ms of points) {
          if (cancelled) return;
          out.push(await captureFrame(fileId, url, ms));
        }
        if (!cancelled) setFrames(out);
      } catch {
        /* best-effort */
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [inView, token, fileId, srcInMs, srcOutMs, n, widthPx]);

  return (
    <div ref={ref} className="pointer-events-none absolute inset-0 flex overflow-hidden opacity-70">
      {frames.map((f, i) =>
        f ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img key={i} src={f} alt="" className="h-full flex-1 object-cover" style={{ minWidth: 0 }} />
        ) : null
      )}
    </div>
  );
}

const waveformCache = new Map<string, Promise<Float32Array | null>>();

function decodeWaveform(fileId: string, url: string): Promise<Float32Array | null> {
  let p = waveformCache.get(fileId);
  if (p) return p;
  p = (async () => {
    try {
      const res = await fetch(url);
      const buf = await res.arrayBuffer();
      const AC = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new AC();
      const audio = await ctx.decodeAudioData(buf);
      const raw = audio.getChannelData(0);
      const buckets = 400;
      const peaks = new Float32Array(buckets);
      const step = Math.max(1, Math.floor(raw.length / buckets));
      for (let b = 0; b < buckets; b++) {
        let max = 0;
        const start = b * step;
        const end = Math.min(raw.length, start + step);
        for (let i = start; i < end; i++) max = Math.max(max, Math.abs(raw[i]));
        peaks[b] = max;
      }
      void ctx.close();
      return peaks;
    } catch {
      return null; // CORS / decode failure -- render nothing
    }
  })();
  waveformCache.set(fileId, p);
  return p;
}

function Waveform({ fileId, widthPx }: { fileId: string; widthPx: number }) {
  const token = useAuthStore((s) => s.session?.access_token);
  const [ref, inView] = useInView<HTMLCanvasElement>();
  const [peaks, setPeaks] = useState<Float32Array | null>(null);

  useEffect(() => {
    if (!inView || !token) return;
    let cancelled = false;
    (async () => {
      const url = await cachedPlaybackUrl(fileId, token);
      const p = await decodeWaveform(fileId, url);
      if (!cancelled) setPeaks(p);
    })();
    return () => {
      cancelled = true;
    };
  }, [inView, token, fileId]);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !peaks) return;
    const dpr = window.devicePixelRatio || 1;
    const h = canvas.clientHeight || 32;
    canvas.width = Math.max(1, Math.round(widthPx * dpr));
    canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "rgba(255,255,255,0.55)";
    const mid = canvas.height / 2;
    const barW = canvas.width / peaks.length;
    for (let i = 0; i < peaks.length; i++) {
      const amp = peaks[i] * mid;
      ctx.fillRect(i * barW, mid - amp, Math.max(1, barW - 1), amp * 2);
    }
  }, [peaks, widthPx, ref]);

  return <canvas ref={ref} className="pointer-events-none absolute inset-0 h-full w-full opacity-70" />;
}
