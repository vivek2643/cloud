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

const LABEL_W = 44;

export function TimelineEditor({
  threadId,
  token,
  onSaved,
}: {
  threadId: string;
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
  const muteCache = useRef<Record<string, number>>({});

  useEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setTrackW(el.clientWidth || 600));
    ro.observe(el);
    setTrackW(el.clientWidth || 600);
    return () => ro.disconnect();
  }, []);

  const { spans, total } = useMemo(() => layout(timeline), [timeline]);
  const pxPerMs = total > 0 ? trackW / total : 0;

  const coverage = useMemo(() => operations.filter((o) => o.type === "place_video"), [operations]);
  const anglePicks = useMemo(() => operations.filter((o) => o.type === "pick_angle"), [operations]);
  const beds = useMemo(
    () => operations.filter((o) => o.type === "place_audio"),
    [operations]
  );
  const bedRoles = useMemo(() => {
    const roles: string[] = [];
    for (const b of beds) {
      const r = b.role || "music";
      if (!roles.includes(r)) roles.push(r);
    }
    return roles;
  }, [beds]);

  // --- trim via drag handles (spine video; audio is coupled to it) ---
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

  function selectSeg(id: string) {
    setSelectedOp(null);
    select(id);
  }
  function selectOp(id: string) {
    select(null);
    setSelectedOp(id);
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
    setSelectedOp(null);
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
          <button onClick={openHistory} className="rounded-lg p-1.5 transition-colors hover:bg-[var(--accent-soft)]" title="Version history">
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

      {/* Lanes: video on top, audio below (shared time scale) */}
      <div className="space-y-1">
        {/* Coverage (V2) — only when present */}
        {coverage.length > 0 && (
          <Lane label="V2">
            <div className="relative h-full w-full">
              {coverage.map((op) => (
                <Block
                  key={op.op_id}
                  left={(op.from_ms ?? 0) * pxPerMs}
                  width={Math.max(4, ((op.to_ms ?? 0) - (op.from_ms ?? 0)) * pxPerMs)}
                  selected={selectedOp === op.op_id}
                  onClick={() => selectOp(op.op_id)}
                  color="#7c5cff"
                  title={`Coverage ${op.source_file_id?.slice(0, 6)} · ${fmt(op.from_ms ?? 0)}–${fmt(op.to_ms ?? 0)}`}
                >
                  cover
                </Block>
              ))}
            </div>
          </Lane>
        )}

        {/* Angle switches (VA) — synced multicam re-points the spine picture */}
        {anglePicks.length > 0 && (
          <Lane label="VA">
            <div className="relative h-full w-full">
              {anglePicks.map((op) => (
                <Block
                  key={op.op_id}
                  left={(op.from_ms ?? 0) * pxPerMs}
                  width={Math.max(4, ((op.to_ms ?? 0) - (op.from_ms ?? 0)) * pxPerMs)}
                  selected={selectedOp === op.op_id}
                  onClick={() => selectOp(op.op_id)}
                  color="#1f9ed1"
                  title={`Angle ${op.source_file_id?.slice(0, 6)} · ${fmt(op.from_ms ?? 0)}–${fmt(op.to_ms ?? 0)}`}
                >
                  angle
                </Block>
              ))}
            </div>
          </Lane>
        )}

        {/* Video spine (V1) — editable */}
        <Lane label="V1" trackRef={trackRef}>
          <div className="relative h-full w-full">
            {spans.map(({ seg, start, end }) => {
              const isSel = selected === seg.seg_id;
              return (
                <Block
                  key={seg.seg_id}
                  left={start * pxPerMs}
                  width={Math.max(8, (end - start) * pxPerMs)}
                  selected={isSel}
                  onClick={() => selectSeg(seg.seg_id)}
                  color="var(--accent)"
                  title={`${seg.file_id.slice(0, 6)} · ${fmt(seg.in_ms)}–${fmt(seg.out_ms)}`}
                >
                  <span
                    onPointerDown={(e) => startTrim(seg.seg_id, "in", e)}
                    className="absolute left-0 top-0 h-full w-1.5 cursor-ew-resize"
                    style={{ background: "rgba(0,0,0,0.35)" }}
                  />
                  <span className="truncate px-2">{seg.file_id.slice(0, 4)}</span>
                  <span
                    onPointerDown={(e) => startTrim(seg.seg_id, "out", e)}
                    className="absolute right-0 top-0 h-full w-1.5 cursor-ew-resize"
                    style={{ background: "rgba(0,0,0,0.35)" }}
                  />
                </Block>
              );
            })}
          </div>
        </Lane>

        {/* Audio spine (A1) — coupled to V1, mirrors the same spans */}
        <Lane label="A1">
          <div className="relative h-full w-full">
            {spans.map(({ seg, start, end }) => (
              <Block
                key={seg.seg_id}
                left={start * pxPerMs}
                width={Math.max(8, (end - start) * pxPerMs)}
                selected={selected === seg.seg_id}
                onClick={() => selectSeg(seg.seg_id)}
                color="#2bb673"
                title={`Dialogue · ${seg.file_id.slice(0, 6)}`}
                muted
              >
                <span className="truncate px-2">dlg</span>
              </Block>
            ))}
          </div>
        </Lane>

        {/* Bed lanes (music / sfx) */}
        {bedRoles.map((role, i) => (
          <Lane key={role} label={`A${i + 2}`}>
            <div className="relative h-full w-full">
              {beds
                .filter((b) => (b.role || "music") === role)
                .map((op) => {
                  const muted = (op.gain_db ?? 0) <= -119;
                  return (
                    <Block
                      key={op.op_id}
                      left={(op.from_ms ?? 0) * pxPerMs}
                      width={Math.max(4, ((op.to_ms ?? 0) - (op.from_ms ?? 0)) * pxPerMs)}
                      selected={selectedOp === op.op_id}
                      onClick={() => selectOp(op.op_id)}
                      color={role === "sfx" ? "#e0883a" : "#3a86e0"}
                      title={`${role} · ${fmt(op.from_ms ?? 0)}–${fmt(op.to_ms ?? 0)}`}
                      muted={muted}
                    >
                      <span className="truncate px-2">{role}</span>
                    </Block>
                  );
                })}
            </div>
          </Lane>
        ))}
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

function Lane({
  label,
  children,
  trackRef,
}: {
  label: string;
  children: React.ReactNode;
  trackRef?: React.Ref<HTMLDivElement>;
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
        ref={trackRef}
        className="relative h-9 min-w-0 flex-1 overflow-hidden rounded"
        style={{ background: "var(--sidebar)" }}
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
  color,
  title,
  children,
  muted,
}: {
  left: number;
  width: number;
  selected: boolean;
  onClick: () => void;
  color: string;
  title: string;
  children: React.ReactNode;
  muted?: boolean;
}) {
  return (
    <div
      onClick={onClick}
      title={title}
      className="absolute top-0 flex h-full cursor-pointer items-center overflow-hidden rounded text-[10px] text-white"
      style={{
        left,
        width,
        background: color,
        opacity: muted ? 0.45 : 1,
        outline: selected ? "2px solid var(--foreground)" : "none",
        outlineOffset: -2,
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
