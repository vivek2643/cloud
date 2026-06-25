"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Scissors,
  Trash2,
  ChevronLeft,
  ChevronRight,
  Save,
  RotateCcw,
  History,
  Volume2,
  VolumeX,
  Loader2,
  AlertCircle,
  Play,
  Pause,
} from "lucide-react";
import {
  saveEditDocument,
  listEditVersions,
  getEditVersion,
  type EditDocument,
  type EditOperation,
  type EditVersionListItem,
} from "@/lib/api";
import { useEditDocStore } from "@/stores/edit-doc-store";
import { useTransport, formatTimecode, snapMs } from "@/stores/transport-store";
import { documentToProject, type ProjectClip, type ProjectTrack } from "@/lib/edit-project";

function fmt(ms: number) {
  const s = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

const LABEL_W = 44;
const LANE_GAP = 8; // matches the gap-2 between a lane label and its track
const TRACK_LEFT = LABEL_W + LANE_GAP;

/** A clip dragged from the Hero Cuts / Dialogues bins (see their onDragStart). */
interface DropPayload {
  file_id: string;
  in_ms: number;
  out_ms: number;
  kind?: string;
}

function parseDrop(e: React.DragEvent): DropPayload | null {
  const raw =
    e.dataTransfer.getData("application/x-hero-cut") ||
    e.dataTransfer.getData("application/x-dialogue-segment") ||
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

export function TimelineEditor({
  threadId,
  ensureThread,
  token,
  onSaved,
}: {
  threadId: string | null;
  ensureThread: () => Promise<string | null>;
  token: string | undefined;
  onSaved: (version: number, document: EditDocument) => void;
}) {
  const timeline = useEditDocStore((s) => s.timeline);
  const operations = useEditDocStore((s) => s.operations);
  const selected = useEditDocStore((s) => s.selected);
  const baseVersion = useEditDocStore((s) => s.baseVersion);
  const select = useEditDocStore((s) => s.select);
  const trimSeg = useEditDocStore((s) => s.trim);
  const nudgeSeg = useEditDocStore((s) => s.nudge);
  const moveSeg = useEditDocStore((s) => s.move);
  const splitSeg = useEditDocStore((s) => s.split);
  const removeSeg = useEditDocStore((s) => s.remove);
  const setGain = useEditDocStore((s) => s.setGain);
  const removeOpStore = useEditDocStore((s) => s.removeOp);
  const setOpFrom = useEditDocStore((s) => s.setOpFrom);
  const setOpEdge = useEditDocStore((s) => s.setOpEdge);
  const setOpZ = useEditDocStore((s) => s.setOpZ);
  const reorderSeg = useEditDocStore((s) => s.reorderSeg);
  const addSegment = useEditDocStore((s) => s.addSegment);
  const addOp = useEditDocStore((s) => s.addOp);
  const revertStore = useEditDocStore((s) => s.revert);
  const setWorking = useEditDocStore((s) => s.setWorking);
  const commit = useEditDocStore((s) => s.commit);
  const isDirty = useEditDocStore((s) => s.isDirty);

  const dirty = useMemo(
    () => isDirty(),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [timeline, operations, baseVersion]
  );

  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [versions, setVersions] = useState<EditVersionListItem[]>([]);
  const [selectedOp, setSelectedOp] = useState<string | null>(null);
  const [trackW, setTrackW] = useState(600);

  const trackRef = useRef<HTMLDivElement>(null);
  const trackEls = useRef<Map<string, HTMLDivElement>>(new Map());
  const muteCache = useRef<Record<string, number>>({});

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setTrackW(el.clientWidth || 600));
    ro.observe(el);
    setTrackW(el.clientWidth || 600);
    return () => ro.disconnect();
  }, []);

  const aspect = useEditDocStore((s) => s.aspect);
  const project = useMemo(
    () => documentToProject(timeline, operations, aspect),
    [timeline, operations, aspect]
  );
  const total = project.durationMs;
  const pxPerMs = total > 0 ? trackW / total : 0;

  // Shared transport — playhead position + play state live in the same store the
  // program monitor reads, so the two surfaces are always in lockstep.
  const progMs = useTransport((s) => s.progMs);
  const playing = useTransport((s) => s.playing);
  const seek = useTransport((s) => s.seek);
  const step = useTransport((s) => s.step);
  const togglePlaying = useTransport((s) => s.togglePlaying);

  const playheadPx = pxPerMs > 0 ? Math.min(Math.max(progMs * pxPerMs, 0), trackW) : 0;

  // Keyboard transport: space = play/pause, ←/→ = step a frame (shift = 10).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = document.activeElement as HTMLElement | null;
      if (el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.isContentEditable))
        return;
      if (e.code === "Space") {
        e.preventDefault();
        togglePlaying();
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        step(e.shiftKey ? -10 : -1);
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        step(e.shiftKey ? 10 : 1);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [togglePlaying, step]);

  function selectClip(clip: ProjectClip) {
    if (clip.origin.kind === "spine") {
      setSelectedOp(null);
      select(clip.origin.segId);
    } else {
      select(null);
      setSelectedOp(clip.origin.opId);
    }
  }

  // --- trim a clip's edge by dragging its handle (frame-snapped) ---
  function startTrim(clip: ProjectClip, edge: "in" | "out", e: React.PointerEvent) {
    e.preventDefault();
    e.stopPropagation();
    if (pxPerMs <= 0 || !clip.trimmable) return;
    selectClip(clip);
    const startX = e.clientX;
    const onMove = (ev: PointerEvent) => {
      const dMs = (ev.clientX - startX) / pxPerMs;
      if (clip.origin.kind === "spine") {
        // source-domain trim (in_ms/out_ms); the coupled audio follows.
        const base = edge === "in" ? clip.srcInMs : clip.srcOutMs;
        trimSeg(clip.origin.segId, edge, snapMs(base + dMs));
      } else {
        // program-domain trim on a placed op.
        const base = edge === "in" ? clip.progStartMs : clip.progEndMs;
        setOpEdge(clip.origin.opId, edge, snapMs(base + dMs), total);
      }
    };
    const onUp = () => {
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
  //     video track's z). Frame-snapped. ---
  function startMove(clip: ProjectClip, e: React.PointerEvent) {
    if (!clip.movable || clip.origin.kind !== "op") return;
    e.preventDefault();
    e.stopPropagation();
    if (pxPerMs <= 0) return;
    selectClip(clip);
    const opId = clip.origin.opId;
    const startX = e.clientX;
    const startFrom = clip.progStartMs;
    const onMove = (ev: PointerEvent) => {
      const dMs = (ev.clientX - startX) / pxPerMs;
      setOpFrom(opId, snapMs(startFrom + dMs), total);
      // Cross-track: only video overlays restack, only onto another non-base
      // video layer (never onto the spine or an audio track).
      if (clip.kind === "video") {
        const tgt = trackAtY(ev.clientY);
        if (tgt && tgt.kind === "video" && !tgt.isBase && tgt.z !== clip.z) {
          setOpZ(opId, tgt.z);
        }
      }
    };
    const onUp = () => {
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
    const segId = clip.origin.segId;
    const baseId = clip.trackId;
    const onMove = (ev: PointerEvent) => {
      const el = trackEls.current.get(baseId);
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const progMs = (ev.clientX - rect.left) / pxPerMs;
      // Insertion index among the OTHER base clips (count whose midpoint is
      // left of the cursor) — matches reorderSeg's post-removal splice index.
      const others = project.clips.filter(
        (c) => c.trackId === baseId && c.id !== clip.id
      );
      let idx = 0;
      for (const c of others) {
        if (progMs > (c.progStartMs + c.progEndMs) / 2) idx++;
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

  // --- drag clips IN from the Hero Cuts / Dialogues bins ---
  // Base video lane (V1) = insert a spine cut at the drop index; an upper video
  // lane = a placed overlay; an audio lane = a placed bed — all at the drop time.
  const [dropTrack, setDropTrack] = useState<string | null>(null);

  function progMsAtX(trackId: string, clientX: number): number {
    const el = trackEls.current.get(trackId);
    if (!el || pxPerMs <= 0) return progMs;
    const rect = el.getBoundingClientRect();
    return Math.max(0, snapMs((clientX - rect.left) / pxPerMs));
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

  function onLaneDrop(track: ProjectTrack, e: React.DragEvent) {
    e.preventDefault();
    setDropTrack(null);
    const p = parseDrop(e);
    if (!p) return;
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
        role: "music",
      });
    }
    // Make sure an edit session exists so the build can be saved later.
    void ensureThread();
  }

  function toggleMute(op: EditOperation) {
    const muted = (op.gain_db ?? 0) <= -119;
    if (muted) setGain(op.op_id, muteCache.current[op.op_id] ?? 0);
    else {
      muteCache.current[op.op_id] = op.gain_db ?? 0;
      setGain(op.op_id, -120);
    }
  }

  // --- save / revert / history ---
  const save = useCallback(async () => {
    if (!token || saving) return;
    setSaving(true);
    setError(null);
    try {
      const id = threadId ?? (await ensureThread());
      if (!id) {
        setError("Could not start an edit session to save into.");
        return;
      }
      const res = await saveEditDocument(
        id,
        { base_version: baseVersion, timeline, operations },
        token
      );
      commit(res.version, res.document);
      onSaved(res.version, res.document);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Save failed.";
      setError(
        msg.includes("stale") || msg.includes("409")
          ? "The plan changed elsewhere (newer version exists). Revert to reload, then re-apply your edits."
          : msg
      );
    } finally {
      setSaving(false);
    }
  }, [token, saving, threadId, ensureThread, baseVersion, timeline, operations, commit, onSaved]);

  function revert() {
    revertStore();
    setSelectedOp(null);
    setError(null);
  }

  async function openHistory() {
    if (!threadId) return;
    setShowHistory((v) => !v);
    if (!showHistory && token) {
      try {
        const { versions } = await listEditVersions(threadId, token);
        setVersions(versions);
      } catch {
        /* ignore */
      }
    }
  }

  async function loadVersion(v: number) {
    if (!token || !threadId) return;
    try {
      const { document } = await getEditVersion(threadId, v, token);
      setWorking(document.timeline ?? [], document.operations ?? []);
      setShowHistory(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load version.");
    }
  }

  const selSeg = selected ? timeline.find((s) => s.seg_id === selected) : null;
  const selOp = selectedOp ? operations.find((o) => o.op_id === selectedOp) : null;

  return (
    <div
      className="space-y-3 rounded-2xl border p-3"
      style={{ background: "var(--background)", borderColor: "var(--border)" }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-baseline gap-2">
          <span className="text-sm font-semibold">Timeline</span>
          <span className="text-[11px] tabular-nums" style={{ color: "var(--muted)" }}>
            {timeline.length} cut{timeline.length === 1 ? "" : "s"} · {fmt(total)}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <button onClick={openHistory} disabled={!threadId} className="rounded-lg p-1.5 transition-colors hover:bg-[var(--accent-soft)] disabled:opacity-30" title="Version history">
            <History size={15} />
          </button>
          <button onClick={revert} disabled={!dirty || saving} className="rounded-lg p-1.5 transition-colors hover:bg-[var(--accent-soft)] disabled:opacity-30" title="Revert changes">
            <RotateCcw size={15} />
          </button>
          <button
            onClick={save}
            disabled={!dirty || saving}
            className="flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs font-medium transition-opacity disabled:opacity-30"
            style={{ background: "var(--accent)", color: "var(--background)" }}
            title="Save as a new version"
          >
            {saving ? <Loader2 size={13} className="animate-spin" /> : <Save size={13} />}
            Save
          </button>
        </div>
      </div>

      {showHistory && (
        <div className="rounded-lg border p-2 text-xs" style={{ borderColor: "var(--border)" }}>
          <p className="mb-1 font-medium" style={{ color: "var(--muted)" }}>
            Versions (load to edit on top of latest)
          </p>
          <div className="flex flex-col gap-1">
            {versions.map((v) => (
              <button key={v.version} onClick={() => loadVersion(v.version)} className="flex items-center justify-between rounded px-1.5 py-1 text-left transition-colors hover:bg-[var(--accent-soft)]">
                <span>v{v.version} · {v.created_by}</span>
                <span style={{ color: "var(--muted)" }}>{new Date(v.created_at).toLocaleTimeString()}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Transport strip */}
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
        <span className="ml-auto text-[11px] tabular-nums" style={{ color: "var(--muted)" }}>
          {formatTimecode(progMs)} <span style={{ opacity: 0.5 }}>/ {formatTimecode(total)}</span>
        </span>
      </div>

      {/* Ruler + lanes + playhead (shared time scale) */}
      <div className="relative">
        <TimeRuler total={total} pxPerMs={pxPerMs} onSeek={seek} />

        {/* Tracks: generic NLE lanes (top → bottom), one Block per clip. */}
        <div className="space-y-1">
          {project.tracks.map((track) => {
            const trackClips = project.clips.filter((c) => c.trackId === track.id);
            const isWidthTrack = track.isBase && track.kind === "video";
            const showDropHint =
              isWidthTrack && trackClips.length === 0 && timeline.length === 0;
            return (
              <Lane
                key={track.id}
                label={track.label}
                highlight={dropTrack === track.id}
                onDragOver={(e) => onLaneDragOver(track, e)}
                onDragLeave={() => setDropTrack(null)}
                onDrop={(e) => onLaneDrop(track, e)}
                innerRef={(el) => {
                  const m = trackEls.current;
                  if (el) m.set(track.id, el);
                  else m.delete(track.id);
                  if (isWidthTrack)
                    (trackRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
                }}
              >
                <div className="relative h-full w-full">
                  {showDropHint && (
                    <div
                      className="pointer-events-none absolute inset-0 flex items-center justify-center text-[11px]"
                      style={{ color: "var(--muted)" }}
                    >
                      Drag clips here from Hero Cuts or Dialogues to build your edit
                    </div>
                  )}
                  {trackClips.map((clip) => {
                    const selectedClip =
                      clip.origin.kind === "spine"
                        ? selected === clip.origin.segId
                        : selectedOp === clip.origin.opId;
                    const bodyDrag = clip.movable
                      ? (e: React.PointerEvent) => startMove(clip, e)
                      : clip.kind === "video" && clip.origin.kind === "spine"
                        ? (e: React.PointerEvent) => startReorder(clip, e)
                        : undefined;
                    return (
                      <Block
                        key={clip.id}
                        left={clip.progStartMs * pxPerMs}
                        width={Math.max(8, (clip.progEndMs - clip.progStartMs) * pxPerMs)}
                        selected={selectedClip}
                        onClick={() => selectClip(clip)}
                        onBodyPointerDown={bodyDrag}
                        color={clip.color}
                        muted={clip.muted}
                        movable={!!bodyDrag}
                        title={`${clip.label} · ${fmt(clip.progStartMs)}–${fmt(clip.progEndMs)}`}
                      >
                        {clip.trimmable && (
                          <span
                            onPointerDown={(e) => startTrim(clip, "in", e)}
                            className="absolute left-0 top-0 h-full w-1.5 cursor-ew-resize"
                            style={{ background: "rgba(0,0,0,0.35)" }}
                          />
                        )}
                        <span className="pointer-events-none truncate px-2">{clip.label}</span>
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
              </Lane>
            );
          })}
        </div>

        {/* Playhead — spans the ruler + every lane */}
        {pxPerMs > 0 && (
          <div
            className="pointer-events-none absolute bottom-0 z-10"
            style={{ left: TRACK_LEFT + playheadPx, top: 0 }}
          >
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

      {/* Inspector: selected clip (spine) or selected layer (op) */}
      {selSeg && (
        <div className="flex flex-wrap items-center gap-1.5 border-t pt-2 text-xs" style={{ borderColor: "var(--border)" }}>
          <span style={{ color: "var(--muted)" }}>
            {selSeg.file_id.slice(0, 6)} · {fmt(selSeg.in_ms)}–{fmt(selSeg.out_ms)}
          </span>
          <div className="ml-auto flex items-center gap-1">
            <Stepper label="in" onMinus={() => nudgeSeg(selSeg.seg_id, "in", -250)} onPlus={() => nudgeSeg(selSeg.seg_id, "in", 250)} />
            <Stepper label="out" onMinus={() => nudgeSeg(selSeg.seg_id, "out", -250)} onPlus={() => nudgeSeg(selSeg.seg_id, "out", 250)} />
            <IconBtn title="Move left" onClick={() => moveSeg(selSeg.seg_id, -1)}><ChevronLeft size={14} /></IconBtn>
            <IconBtn title="Move right" onClick={() => moveSeg(selSeg.seg_id, 1)}><ChevronRight size={14} /></IconBtn>
            <IconBtn title="Split at middle" onClick={() => splitSeg(selSeg.seg_id)}><Scissors size={13} /></IconBtn>
            <IconBtn title="Delete cut" onClick={() => removeSeg(selSeg.seg_id)} danger><Trash2 size={13} /></IconBtn>
          </div>
        </div>
      )}

      {selOp && (
        <div className="flex flex-wrap items-center gap-2 border-t pt-2 text-xs" style={{ borderColor: "var(--border)" }}>
          <span className="font-medium">
            {selOp.type === "pick_angle"
              ? "Angle"
              : selOp.type === "place_video"
                ? "Coverage"
                : selOp.role === "sfx"
                  ? "SFX"
                  : "Music"}
          </span>
          <span style={{ color: "var(--muted)" }}>{fmt(selOp.from_ms ?? 0)}–{fmt(selOp.to_ms ?? 0)}</span>
          {(selOp.type === "place_audio") && (
            <>
              <input
                type="range"
                min={-30}
                max={6}
                step={1}
                value={(selOp.gain_db ?? 0) <= -119 ? -30 : Math.max(-30, Math.min(6, selOp.gain_db ?? 0))}
                onChange={(e) => setGain(selOp.op_id, Number(e.target.value))}
                className="ml-auto w-24 accent-[var(--accent)]"
                title="Gain (dB)"
              />
              <button onClick={() => toggleMute(selOp)} title={(selOp.gain_db ?? 0) <= -119 ? "Unmute" : "Mute"} className="rounded p-1 hover:bg-[var(--accent-soft)]">
                {(selOp.gain_db ?? 0) <= -119 ? <VolumeX size={13} /> : <Volume2 size={13} />}
              </button>
            </>
          )}
          <button onClick={() => { removeOpStore(selOp.op_id); setSelectedOp(null); }} title="Delete layer" className={`rounded p-1 hover:bg-[var(--accent-soft)] ${selOp.type === "place_video" || selOp.type === "pick_angle" ? "ml-auto" : ""}`}>
            <Trash2 size={13} style={{ color: "var(--danger)" }} />
          </button>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-1.5 text-[11px]" style={{ color: "var(--danger)" }}>
          <AlertCircle size={12} className="mt-0.5 shrink-0" />
          <span>{error}</span>
        </div>
      )}
    </div>
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
    <div className="mb-1 flex items-stretch gap-2">
      <span className="shrink-0" style={{ width: LABEL_W }} />
      <div
        ref={ref}
        onPointerDown={onPointerDown}
        className="relative h-6 min-w-0 flex-1 cursor-pointer select-none"
        style={{ touchAction: "none" }}
        title="Click or drag to scrub"
      >
        {ticks}
      </div>
    </div>
  );
}

function Lane({
  label,
  children,
  innerRef,
  highlight,
  onDragOver,
  onDragLeave,
  onDrop,
}: {
  label: string;
  children: React.ReactNode;
  innerRef?: (el: HTMLDivElement | null) => void;
  highlight?: boolean;
  onDragOver?: (e: React.DragEvent) => void;
  onDragLeave?: (e: React.DragEvent) => void;
  onDrop?: (e: React.DragEvent) => void;
}) {
  return (
    <div className="flex items-stretch gap-2">
      <span
        className="flex shrink-0 items-center justify-center rounded text-[10px] font-medium"
        style={{ width: LABEL_W, color: "var(--muted)", background: "var(--accent-soft)" }}
      >
        {label}
      </span>
      <div
        ref={innerRef}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        className="relative h-9 min-w-0 flex-1 overflow-hidden rounded"
        style={{
          background: "var(--sidebar)",
          outline: highlight ? "2px dashed var(--accent)" : "none",
          outlineOffset: -2,
        }}
      >
        {children}
      </div>
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
  onClick: () => void;
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

function IconBtn({ children, onClick, title, danger }: { children: React.ReactNode; onClick: () => void; title: string; danger?: boolean }) {
  return (
    <button onClick={onClick} title={title} className="rounded p-1 transition-colors hover:bg-[var(--accent-soft)]" style={danger ? { color: "var(--danger)" } : undefined}>
      {children}
    </button>
  );
}

function Stepper({ label, onMinus, onPlus }: { label: string; onMinus: () => void; onPlus: () => void }) {
  return (
    <span className="flex items-center gap-0.5 rounded border px-1" style={{ borderColor: "var(--border)" }}>
      <button onClick={onMinus} className="px-1 text-xs hover:opacity-70">−</button>
      <span className="text-[10px]" style={{ color: "var(--muted)" }}>{label}</span>
      <button onClick={onPlus} className="px-1 text-xs hover:opacity-70">+</button>
    </span>
  );
}
