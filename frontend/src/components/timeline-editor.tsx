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
  Layers,
  Music,
} from "lucide-react";
import {
  saveEditDocument,
  listEditVersions,
  getEditVersion,
  type EditDocument,
  type EditSegment,
  type EditOperation,
  type EditVersionListItem,
} from "@/lib/api";
import { useEditDocStore } from "@/stores/edit-doc-store";

function fmt(ms: number) {
  const s = Math.max(0, Math.round(ms / 1000));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

type Span = { seg: EditSegment; start: number; end: number };

function layout(timeline: EditSegment[]): { spans: Span[]; total: number } {
  let t = 0;
  const spans: Span[] = [];
  for (const seg of timeline) {
    const dur = Math.max(0, seg.out_ms - seg.in_ms);
    spans.push({ seg, start: t, end: t + dur });
    t += dur;
  }
  return { spans, total: t };
}

export function TimelineEditor({
  threadId,
  token,
  onSaved,
}: {
  threadId: string;
  token: string | undefined;
  onSaved: (version: number, document: EditDocument) => void;
}) {
  // Live working state lives in the shared store; the preview reads the same
  // state, so every mutation here updates the program monitor instantly.
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
  const revertStore = useEditDocStore((s) => s.revert);
  const setWorking = useEditDocStore((s) => s.setWorking);
  const commit = useEditDocStore((s) => s.commit);
  const isDirty = useEditDocStore((s) => s.isDirty);

  const dirty = useMemo(
    () => isDirty(),
    // recompute whenever the working data changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [timeline, operations, baseVersion]
  );

  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [versions, setVersions] = useState<EditVersionListItem[]>([]);
  const [trackW, setTrackW] = useState(412);

  const trackRef = useRef<HTMLDivElement>(null);
  const muteCache = useRef<Record<string, number>>({});

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setTrackW(el.clientWidth || 412));
    ro.observe(el);
    setTrackW(el.clientWidth || 412);
    return () => ro.disconnect();
  }, []);

  const { spans, total } = useMemo(() => layout(timeline), [timeline]);
  const pxPerMs = total > 0 ? trackW / total : 0;

  // --- trim via drag handles ---
  function startTrim(segId: string, edge: "in" | "out", e: React.PointerEvent) {
    e.preventDefault();
    e.stopPropagation();
    const startX = e.clientX;
    const seg = timeline.find((s) => s.seg_id === segId);
    if (!seg || pxPerMs <= 0) return;
    const startIn = seg.in_ms;
    const startOut = seg.out_ms;
    const onMove = (ev: PointerEvent) => {
      const dMs = Math.round((ev.clientX - startX) / pxPerMs);
      trimSeg(segId, edge, edge === "in" ? startIn + dMs : startOut + dMs);
    };
    const onUp = () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);
  }

  // --- operations ---
  function toggleMute(op: EditOperation) {
    const muted = (op.gain_db ?? 0) <= -119;
    if (muted) {
      setGain(op.op_id, muteCache.current[op.op_id] ?? 0);
    } else {
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
      const res = await saveEditDocument(
        threadId,
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
  }, [token, saving, threadId, baseVersion, timeline, operations, commit, onSaved]);

  function revert() {
    revertStore();
    setError(null);
  }

  async function openHistory() {
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
    if (!token) return;
    try {
      const { document } = await getEditVersion(threadId, v, token);
      // Load onto the working state; baseline (head) stays put so saving makes
      // a new version on top of the latest.
      setWorking(document.timeline ?? [], document.operations ?? []);
      setShowHistory(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not load version.");
    }
  }

  const selSeg = selected ? timeline.find((s) => s.seg_id === selected) : null;

  return (
    <div className="space-y-3 rounded-2xl border p-3" style={{ background: "var(--background)", borderColor: "var(--border)" }}>
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold">Timeline editor</span>
        <div className="flex items-center gap-1">
          <button
            onClick={openHistory}
            className="rounded-lg p-1.5 transition-colors hover:bg-[var(--accent-soft)]"
            title="Version history"
          >
            <History size={15} />
          </button>
          <button
            onClick={revert}
            disabled={!dirty || saving}
            className="rounded-lg p-1.5 transition-colors hover:bg-[var(--accent-soft)] disabled:opacity-30"
            title="Revert changes"
          >
            <RotateCcw size={15} />
          </button>
          <button
            onClick={save}
            disabled={!dirty || saving}
            className="flex items-center gap-1 rounded-lg px-2.5 py-1.5 text-xs font-medium text-white transition-opacity disabled:opacity-30"
            style={{ background: "var(--accent)" }}
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
              <button
                key={v.version}
                onClick={() => loadVersion(v.version)}
                className="flex items-center justify-between rounded px-1.5 py-1 text-left transition-colors hover:bg-[var(--accent-soft)]"
              >
                <span>v{v.version} · {v.created_by}</span>
                <span style={{ color: "var(--muted)" }}>
                  {new Date(v.created_at).toLocaleTimeString()}
                </span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Spine track */}
      <div>
        <div className="mb-1 flex items-center justify-between">
          <span className="text-[11px] font-medium" style={{ color: "var(--muted)" }}>
            SPINE · {timeline.length} cut{timeline.length === 1 ? "" : "s"}
          </span>
          <span className="text-[11px] tabular-nums" style={{ color: "var(--muted)" }}>
            {fmt(total)}
          </span>
        </div>
        <div
          ref={trackRef}
          className="relative h-12 w-full overflow-hidden rounded-lg"
          style={{ background: "var(--accent-soft)" }}
        >
          {spans.map(({ seg, start, end }) => {
            const left = start * pxPerMs;
            const w = Math.max(6, (end - start) * pxPerMs);
            const isSel = selected === seg.seg_id;
            return (
              <div
                key={seg.seg_id}
                onClick={() => select(seg.seg_id)}
                className="absolute top-0 flex h-full cursor-pointer items-center overflow-hidden border-l text-[10px]"
                style={{
                  left,
                  width: w,
                  background: isSel ? "var(--accent)" : "var(--sidebar)",
                  color: isSel ? "#fff" : "var(--foreground)",
                  borderColor: "var(--border)",
                }}
                title={`${seg.file_id.slice(0, 6)} · ${fmt(seg.in_ms)}–${fmt(seg.out_ms)}`}
              >
                <span
                  onPointerDown={(e) => startTrim(seg.seg_id, "in", e)}
                  className="absolute left-0 top-0 h-full w-1.5 cursor-ew-resize"
                  style={{ background: "var(--accent)" }}
                />
                <span className="truncate px-2">{seg.file_id.slice(0, 4)}</span>
                <span
                  onPointerDown={(e) => startTrim(seg.seg_id, "out", e)}
                  className="absolute right-0 top-0 h-full w-1.5 cursor-ew-resize"
                  style={{ background: "var(--accent)" }}
                />
              </div>
            );
          })}
        </div>

        {selSeg && (
          <div className="mt-2 flex flex-wrap items-center gap-1.5 text-xs">
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
      </div>

      {/* Operations (audio gain/mute + delete) */}
      {operations.length > 0 && (
        <div className="border-t pt-2" style={{ borderColor: "var(--border)" }}>
          <span className="text-[11px] font-medium" style={{ color: "var(--muted)" }}>
            LAYERS · {operations.length}
          </span>
          <div className="mt-1 flex flex-col gap-1.5">
            {operations.map((op) => {
              const isAudio = op.type === "place_audio" || op.type === "level";
              const muted = (op.gain_db ?? 0) <= -119;
              return (
                <div key={op.op_id} className="flex items-center gap-2 text-xs">
                  <span style={{ color: "var(--accent)" }}>
                    {op.type === "place_video" ? <Layers size={12} /> : <Music size={12} />}
                  </span>
                  <span className="w-20 truncate font-medium">
                    {op.type === "place_video" ? "Coverage" : op.type === "split_edit" ? (op.kind || "Split") : op.role === "music" ? "Music" : op.role === "sfx" ? "SFX" : "Audio"}
                  </span>
                  {op.from_ms != null && op.to_ms != null && (
                    <span style={{ color: "var(--muted)" }}>{fmt(op.from_ms)}–{fmt(op.to_ms)}</span>
                  )}
                  {isAudio && (
                    <>
                      <input
                        type="range"
                        min={-30}
                        max={6}
                        step={1}
                        value={muted ? -30 : Math.max(-30, Math.min(6, op.gain_db ?? 0))}
                        onChange={(e) => setGain(op.op_id, Number(e.target.value))}
                        className="ml-auto w-20 accent-[var(--accent)]"
                        title="Gain (dB)"
                      />
                      <button onClick={() => toggleMute(op)} title={muted ? "Unmute" : "Mute"} className="rounded p-1 hover:bg-[var(--accent-soft)]">
                        {muted ? <VolumeX size={13} /> : <Volume2 size={13} />}
                      </button>
                    </>
                  )}
                  <button
                    onClick={() => removeOpStore(op.op_id)}
                    title="Delete layer"
                    className={`rounded p-1 hover:bg-[var(--accent-soft)] ${isAudio ? "" : "ml-auto"}`}
                  >
                    <Trash2 size={13} style={{ color: "var(--danger)" }} />
                  </button>
                </div>
              );
            })}
          </div>
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

function IconBtn({
  children,
  onClick,
  title,
  danger,
}: {
  children: React.ReactNode;
  onClick: () => void;
  title: string;
  danger?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className="rounded p-1 transition-colors hover:bg-[var(--accent-soft)]"
      style={danger ? { color: "var(--danger)" } : undefined}
    >
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
