"use client";

/**
 * Colour grading panel (color_grading.plan.md SS12) -- fills the
 * project-lenses.tsx "Coming soon" placeholder. Controls: a preset look
 * gallery, an arc intensity dial, reference-image drop, .cube upload, an NL
 * steering box (routes into the SAME edit thread the AI panel talks to --
 * EDSO turns text into dial values, never raw CDL numbers, per SS10), and a
 * before/after toggle wired into the live preview via
 * `timeline-view.ts`'s `gradeBypass` flag.
 *
 * Works on top of an already-open edit session (`useEditDocStore`'s
 * `threadId`) -- grading is a refinement layer over EDSO's edit, not a
 * standalone surface, matching this whole feature's "lean AI-first" framing.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Loader2, SplitSquareHorizontal, Upload, Wand2 } from "lucide-react";
import { useEditDocStore } from "@/stores/edit-doc-store";
import { useTimelineView } from "@/stores/timeline-view";
import { useAuthStore } from "@/stores/auth-store";
import {
  getGradePresets,
  saveEditDocument,
  sendThreadMessage,
  type EditLook,
  type GradePresetSummary,
} from "@/lib/api";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/** Mean + std per RGB channel from a decoded <img>, computed entirely
 * client-side (canvas readback) -- no server upload needed just to measure
 * a reference image; only its stats are ever persisted (SS7.2). */
function computeImageStats(img: HTMLImageElement): { rgb_mean: [number, number, number]; rgb_std: [number, number, number] } {
  const canvas = document.createElement("canvas");
  const maxDim = 256; // measurement doesn't need full resolution
  const scale = Math.min(1, maxDim / Math.max(img.naturalWidth, img.naturalHeight));
  canvas.width = Math.max(1, Math.round(img.naturalWidth * scale));
  canvas.height = Math.max(1, Math.round(img.naturalHeight * scale));
  const ctx = canvas.getContext("2d")!;
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);

  const n = data.length / 4;
  let sumR = 0, sumG = 0, sumB = 0;
  for (let i = 0; i < data.length; i += 4) {
    sumR += data[i]; sumG += data[i + 1]; sumB += data[i + 2];
  }
  const meanR = sumR / n / 255, meanG = sumG / n / 255, meanB = sumB / n / 255;

  let varR = 0, varG = 0, varB = 0;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
    varR += (r - meanR) ** 2; varG += (g - meanG) ** 2; varB += (b - meanB) ** 2;
  }
  return {
    rgb_mean: [meanR, meanG, meanB],
    rgb_std: [Math.sqrt(varR / n), Math.sqrt(varG / n), Math.sqrt(varB / n)],
  };
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="mb-2 text-[11px] font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>
      {children}
    </h3>
  );
}

function Dial({ value, onChange, label }: { value: number; onChange: (v: number) => void; label: string }) {
  return (
    <div className="flex items-center gap-3">
      <input
        type="range"
        min={0}
        max={1}
        step={0.05}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-1 flex-1 cursor-pointer appearance-none rounded-full"
        style={{
          background: `linear-gradient(to right, var(--accent) ${value * 100}%, var(--border) ${value * 100}%)`,
        }}
        aria-label={label}
      />
      <span className="w-10 text-right text-[11px] tabular-nums" style={{ color: "var(--muted)" }}>
        {Math.round(value * 100)}%
      </span>
    </div>
  );
}

export function ColorGradeView() {
  const threadId = useEditDocStore((s) => s.threadId);
  const look = useEditDocStore((s) => s.look);
  const setLook = useEditDocStore((s) => s.setLook);
  const wdCommitLook = useEditDocStore((s) => s.commitLook);
  const gradeBypass = useTimelineView((s) => s.gradeBypass);
  const toggleGradeBypass = useTimelineView((s) => s.toggleGradeBypass);
  const token = useAuthStore((s) => s.session?.access_token);

  const [presets, setPresets] = useState<GradePresetSummary[]>([]);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [nlInput, setNlInput] = useState("");
  const [sending, setSending] = useState(false);
  const [sentNote, setSentNote] = useState<string | null>(null);
  const refInputRef = useRef<HTMLInputElement>(null);
  const lutInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!token) return;
    getGradePresets(token).then(setPresets).catch(() => {});
  }, [token]);

  // --- Look persistence -------------------------------------------------
  // A look change (esp. a dragged dial) must apply to the dial INSTANTLY but
  // persist at most once per gesture. Firing a save per tick raced many
  // requests against the same stale base_version -> 409s -> reverts -> the
  // dial jumped around. Here the local look updates immediately (never
  // reverts), while a single serialized, debounced save runs against the LIVE
  // head version read from the store at flush time (not a stale closure).
  const pendingRef = useRef<EditLook | undefined>(undefined);
  const savingRef = useRef(false);
  const flushTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const flushLook = useCallback(async () => {
    // One save at a time; the finally-block re-flushes if the user moved on.
    if (savingRef.current) return;
    const st = useEditDocStore.getState();
    if (!st.threadId || !token) return;
    const next = pendingRef.current;
    savingRef.current = true;
    setSaving(true);
    setError(null);
    try {
      const res = await saveEditDocument(
        st.threadId,
        // LIVE head version + working state, so concurrent edits never send a
        // stale base_version (the cause of the 409 storm).
        { base_version: st.baseVersion, timeline: st.timeline, operations: st.operations, look: next },
        token
      );
      wdCommitLook(res.version, res.document);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save the grade.");
    } finally {
      savingRef.current = false;
      setSaving(false);
      // The user changed the look again while we were saving -> persist newest.
      if (pendingRef.current !== next) void flushLook();
    }
  }, [token, wdCommitLook]);

  const applyLook = useCallback(
    (next: EditLook | undefined, immediate = false) => {
      setLook(next); // instant local: the dial reflects input and never reverts
      pendingRef.current = next;
      if (flushTimer.current) clearTimeout(flushTimer.current);
      if (immediate) {
        void flushLook();
      } else {
        flushTimer.current = setTimeout(() => void flushLook(), 300);
      }
    },
    [setLook, flushLook]
  );

  // Persist a pending debounced change if the panel unmounts mid-gesture.
  useEffect(
    () => () => {
      if (flushTimer.current) {
        clearTimeout(flushTimer.current);
        void flushLook();
      }
    },
    [flushLook]
  );

  function selectPreset(presetId: string) {
    applyLook({ ...(look ?? {}), mode: "preset", preset_id: presetId, lut_ref: null }, true);
  }

  function setArcIntensity(v: number) {
    applyLook({ ...(look ?? {}), arc_intensity: v });
  }

  function onReferenceFile(file: File) {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const stats = computeImageStats(img);
      URL.revokeObjectURL(url);
      applyLook(
        {
          ...(look ?? {}),
          mode: "reference",
          reference_stats: stats,
          match_strength: look?.match_strength ?? 0.6,
        },
        true
      );
    };
    img.onerror = () => {
      URL.revokeObjectURL(url);
      setError("Could not read that image.");
    };
    img.src = url;
  }

  async function onLutFile(file: File) {
    if (!token) return;
    setSaving(true);
    setError(null);
    try {
      const text = await file.text();
      const res = await fetch(`${API_URL}/api/grade/lut`, {
        method: "POST",
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "text/plain" },
        body: text,
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Upload failed (${res.status})`);
      }
      const { lut_ref } = await res.json();
      applyLook({ ...(look ?? {}), mode: "lut", lut_ref }, true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not upload that LUT.");
    } finally {
      setSaving(false);
    }
  }

  async function downloadExport(exportFormat: "cdl" | "ccc" | "edl") {
    if (!threadId || !token) return;
    setError(null);
    try {
      const res = await fetch(
        `${API_URL}/api/edit/threads/${threadId}/grade-export?export_format=${exportFormat}`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Export failed (${res.status})`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `grade.${exportFormat}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Export failed.");
    }
  }

  async function sendSteer() {
    if (!threadId || !token || !nlInput.trim()) return;
    setSending(true);
    setError(null);
    setSentNote(null);
    try {
      await sendThreadMessage(threadId, nlInput.trim(), token);
      setSentNote("Sent -- check the AI panel for EDSO's reply.");
      setNlInput("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not send.");
    } finally {
      setSending(false);
    }
  }

  if (!threadId) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <p className="text-lg font-semibold">Colour grading</p>
        <p className="mt-1 max-w-sm text-sm" style={{ color: "var(--muted)" }}>
          Start an edit first (Drive or the AI panel) -- grading refines an existing timeline.
        </p>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-4">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium">Colour grading</p>
        <button
          onClick={toggleGradeBypass}
          title="Toggle before/after in the preview"
          className="flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-[11px] transition-colors"
          style={{
            borderColor: gradeBypass ? "var(--accent)" : "var(--border)",
            color: gradeBypass ? "var(--accent)" : "var(--muted)",
          }}
        >
          <SplitSquareHorizontal size={12} />
          {gradeBypass ? "Showing: before" : "Showing: after"}
        </button>
      </div>

      {/* Look picker gallery */}
      <section>
        <SectionLabel>Look</SectionLabel>
        <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
          {presets.map((p) => {
            const active = look?.mode === "preset" && look.preset_id === p.preset_id;
            return (
              <button
                key={p.preset_id}
                onClick={() => selectPreset(p.preset_id)}
                title={p.description}
                className="truncate rounded-lg border px-2 py-2 text-left text-[11px] font-medium transition-colors"
                style={{
                  borderColor: active ? "var(--accent)" : "var(--border)",
                  background: active ? "var(--accent-soft)" : "transparent",
                  color: active ? "var(--foreground)" : "var(--muted)",
                }}
              >
                {p.label}
              </button>
            );
          })}
        </div>
      </section>

      {/* Reference image / LUT drop */}
      <section className="grid grid-cols-2 gap-2">
        <div>
          <SectionLabel>Reference image</SectionLabel>
          <button
            onClick={() => refInputRef.current?.click()}
            className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-dashed px-3 py-3 text-[11px]"
            style={{ borderColor: "var(--border)", color: "var(--muted)" }}
          >
            <Upload size={12} />
            {look?.mode === "reference" ? "Replace image" : "Drop / choose image"}
          </button>
          <input
            ref={refInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) onReferenceFile(f); e.target.value = ""; }}
          />
          {look?.mode === "reference" && (
            <div className="mt-2">
              <Dial
                value={look.match_strength ?? 0.6}
                onChange={(v) => applyLook({ ...look, match_strength: v })}
                label="Match strength"
              />
            </div>
          )}
        </div>
        <div>
          <SectionLabel>.cube LUT</SectionLabel>
          <button
            onClick={() => lutInputRef.current?.click()}
            className="flex w-full items-center justify-center gap-1.5 rounded-lg border border-dashed px-3 py-3 text-[11px]"
            style={{ borderColor: "var(--border)", color: "var(--muted)" }}
          >
            <Upload size={12} />
            {look?.mode === "lut" ? "Replace LUT" : "Drop / choose .cube"}
          </button>
          <input
            ref={lutInputRef}
            type="file"
            accept=".cube"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) void onLutFile(f); e.target.value = ""; }}
          />
        </div>
      </section>

      {/* Arc intensity dial */}
      <section>
        <SectionLabel>Arc intensity</SectionLabel>
        <Dial value={look?.arc_intensity ?? 0} onChange={setArcIntensity} label="Arc intensity" />
      </section>

      {/* Export bundle -- the professional round-trip (SS11) */}
      <section>
        <SectionLabel>Export grade</SectionLabel>
        <div className="flex gap-2">
          {(["ccc", "cdl", "edl"] as const).map((f) => (
            <button
              key={f}
              onClick={() => void downloadExport(f)}
              className="rounded-lg border px-2.5 py-1.5 text-[11px] font-medium uppercase transition-colors"
              style={{ borderColor: "var(--border)", color: "var(--muted)" }}
            >
              .{f}
            </button>
          ))}
        </div>
      </section>

      {/* NL steering box */}
      <section>
        <SectionLabel>Steer</SectionLabel>
        <div className="flex items-center gap-2">
          <input
            value={nlInput}
            onChange={(e) => setNlInput(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") void sendSteer(); }}
            placeholder="e.g. warmer, less teal, fix the skin"
            className="flex-1 rounded-lg border bg-transparent px-3 py-2 text-sm outline-none"
            style={{ borderColor: "var(--border)" }}
          />
          <button
            onClick={() => void sendSteer()}
            disabled={sending || !nlInput.trim()}
            className="flex items-center justify-center rounded-lg px-3 py-2 disabled:opacity-40"
            style={{ background: "var(--accent)", color: "var(--background)" }}
            title="Send to EDSO"
          >
            {sending ? <Loader2 size={14} className="animate-spin" /> : <Wand2 size={14} />}
          </button>
        </div>
        {sentNote && <p className="mt-1.5 text-[11px]" style={{ color: "var(--muted)" }}>{sentNote}</p>}
      </section>

      {error && <p className="text-[11px]" style={{ color: "var(--danger)" }}>{error}</p>}
      {saving && (
        <p className="flex items-center gap-1.5 text-[11px]" style={{ color: "var(--muted)" }}>
          <Loader2 size={11} className="animate-spin" /> Saving…
        </p>
      )}
    </div>
  );
}
