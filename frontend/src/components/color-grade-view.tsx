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
  getEditThread,
  getGradePresets,
  getGradeStatus,
  saveEditDocument,
  sendThreadMessage,
  type EditLook,
  type GradePresetSummary,
  type GradeStatus,
  type ResolvedGrade,
  type ResolvedVideoLayer,
} from "@/lib/api";
import { gradeCubeUrl, prefetchGradeCube } from "@/components/preview/grade-cube-client";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// How often the grade-status poll re-checks while a job is (expected to be)
// running, and how long we keep polling through a standalone `idle` right after
// a save before giving up (the worker usually flips idle->grading well inside
// this window).
const POLL_INTERVAL_MS = 750;
const IDLE_EXPECT_TIMEOUT_MS = 20_000;

/** Mirrors the preview player's `isIdentityGrade` (use-program-player.ts): a
 * grade the player would NOT paint (no LUT, no vignette, unit CDL) needs no
 * cube warmed, so we skip it when deciding "preview is ready". */
function isIdentityGrade(grade: ResolvedGrade | undefined): boolean {
  if (!grade) return true;
  if (grade.creative_lut_ref) return false;
  if (grade.soft_local?.vignette && grade.soft_local.vignette.strength > 0) return false;
  const cdl = grade.cdl as ResolvedGrade["cdl"] | undefined;
  if (!cdl) return true;
  const { slope, offset, power, sat } = cdl;
  const eps = 1e-9;
  return (
    slope.every((v) => Math.abs(v - 1) < eps) &&
    offset.every((v) => Math.abs(v) < eps) &&
    power.every((v) => Math.abs(v - 1) < eps) &&
    Math.abs(sat - 1) < eps
  );
}

/** Distinct, paintable grades across the resolved video layers, deduped by the
 * exact cube URL the player fetches -- so warming these is precisely what the
 * preview needs before it can show the new grade. */
function distinctPaintableGrades(layers: ResolvedVideoLayer[] | undefined): ResolvedGrade[] {
  if (!layers) return [];
  const byUrl = new Map<string, ResolvedGrade>();
  for (const layer of layers) {
    const g = layer.grade;
    if (!g || isIdentityGrade(g)) continue;
    const url = gradeCubeUrl(g);
    if (!byUrl.has(url)) byUrl.set(url, g);
  }
  return [...byUrl.values()];
}

/** Continuous progress model for the grading bar. The bar only reaches 100%
 * once the preview's LUT cubes are warmed (`ready`), never merely when the
 * background job says "done". */
type GradePhase =
  | { kind: "idle" }
  | { kind: "applying" }
  | { kind: "grading"; done: number; total: number; progress: number }
  | { kind: "warming" }
  | { kind: "ready" };

/** Map a phase to the bar's display percent + caption (null = hide the bar).
 * KEY invariant: 100% is reserved for `ready` (cubes warmed). */
function phaseDisplay(phase: GradePhase): { pct: number; caption: string } | null {
  switch (phase.kind) {
    case "idle":
      return null;
    case "applying":
      return { pct: 8, caption: "Applying…" };
    case "grading": {
      const p = Math.max(0, Math.min(1, phase.progress));
      return { pct: Math.round(10 + p * 70), caption: `Grading… ${phase.done}/${phase.total}` };
    }
    case "warming":
      return { pct: 90, caption: "Finishing…" };
    case "ready":
      return { pct: 100, caption: "Ready" };
  }
}

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

  // --- Grading progress (color_grading_upgrade.plan.md Step 1.0 §6 / Phase 4) ---
  // A CONTINUOUS bar reflecting TRUE end-to-end readiness: the real background
  // v1 grade job's progress, then a final "warming" phase that only hits 100%
  // once the preview's LUT cubes are actually fetched, so the bar disappearing
  // means the preview can genuinely show the new grade.
  const [gradePhase, setGradePhase] = useState<GradePhase>({ kind: "idle" });
  const pollTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Bumped whenever a NEW poll chain starts, so at most one chain is live: a
  // superseded chain sees its captured gen != the current one and bails,
  // preventing overlapping loops (and stale setState after unmount).
  const pollGenRef = useRef(0);
  // Absolute time (ms) after which a standalone `idle` stops being treated as
  // "a job I'm waiting for" (worker never picked it up / no job was enqueued).
  const idleDeadlineRef = useRef(0);

  // `expectJob`: true right after a look save -- we KNOW a job should appear, so
  // keep polling through the brief `idle` gap (bounded by `idleDeadlineRef`)
  // until it flips to grading/done/error. false for the panel-mount catch-up,
  // where a standalone `idle` means "nothing running" and we stop immediately.
  // `gen` is set only on recursive reschedules to keep them on the same chain.
  const pollGradeStatus = useCallback(
    async (expectJob = false, gen?: number): Promise<void> => {
      if (!threadId || !token) return;
      const myGen = gen ?? ++pollGenRef.current;
      if (gen === undefined) {
        // Fresh chain: drop any timer the previous chain scheduled and, when we
        // expect a job, arm the idle deadline.
        if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
        if (expectJob) idleDeadlineRef.current = Date.now() + IDLE_EXPECT_TIMEOUT_MS;
      }
      if (myGen !== pollGenRef.current) return; // superseded before we even ran

      const reschedule = () => {
        pollTimerRef.current = setTimeout(
          () => void pollGradeStatus(expectJob, myGen),
          POLL_INTERVAL_MS
        );
      };

      let status: GradeStatus;
      try {
        status = await getGradeStatus(threadId, token);
      } catch {
        // Transient network hiccup: retry within this chain while we still
        // expect a job; otherwise let it drop (a later trigger re-checks).
        if (expectJob && myGen === pollGenRef.current && Date.now() < idleDeadlineRef.current) {
          reschedule();
        }
        return;
      }
      if (myGen !== pollGenRef.current) return; // superseded while awaiting

      if (status.state === "grading") {
        setGradePhase({ kind: "grading", done: status.done, total: status.total, progress: status.progress });
        reschedule();
        return;
      }
      if (status.state === "error") {
        setError(status.error || "Grading failed.");
        setGradePhase({ kind: "idle" });
        return;
      }
      if (status.state === "done") {
        // Job finished. Refetch so the preview picks up the newly persisted
        // grades (without touching the working timeline/undo stack), then WARM
        // each distinct grade's cube -- the bar only completes once the preview
        // can actually paint the new look.
        setGradePhase({ kind: "warming" });
        try {
          const thread = await getEditThread(threadId, token);
          if (myGen !== pollGenRef.current) return;
          if (thread.document) wdCommitLook(thread.document_version ?? 0, thread.document);
          const grades = distinctPaintableGrades(thread.document?.resolved?.video_layers);
          await Promise.all(grades.map((g) => prefetchGradeCube(g)));
        } catch {
          // Best-effort refresh/warm; the next natural save/reload still picks
          // it up. Fall through to clear the bar either way.
        }
        if (myGen !== pollGenRef.current) return;
        setGradePhase({ kind: "ready" });
        pollTimerRef.current = setTimeout(() => {
          if (myGen === pollGenRef.current) setGradePhase({ kind: "idle" });
        }, 600);
        return;
      }
      // status.state === "idle"
      if (expectJob && Date.now() < idleDeadlineRef.current) {
        // Job enqueued but the worker hasn't flipped it to `grading` yet -- keep
        // the bar up ("Applying…") and keep polling.
        setGradePhase((p) => (p.kind === "grading" || p.kind === "warming" ? p : { kind: "applying" }));
        reschedule();
        return;
      }
      // Not expecting a job (panel-mount catch-up), or the expect window
      // elapsed with no transition -- nothing is running, so clear the bar.
      setGradePhase({ kind: "idle" });
    },
    [threadId, token, wdCommitLook]
  );

  // Catch a job already in flight (e.g. from a timeline edit made elsewhere)
  // when the panel opens/thread changes.
  useEffect(() => {
    void pollGradeStatus();
    return () => {
      // Kill any live chain (and block its stale setState) on thread change /
      // unmount, and clear the pending poll/clear timer. Mutating the ref in
      // cleanup is intentional here (we want the value live at teardown time).
      // eslint-disable-next-line react-hooks/exhaustive-deps
      pollGenRef.current++;
      if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [threadId, token]);

  const gradeDisplay = phaseDisplay(gradePhase);

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
      // A look change may have enqueued a v1 grade job (Step 1.0 §4). Poll with
      // expectJob=true so the bar rides through the brief idle->grading gap and
      // only completes once the preview's cubes are warmed.
      void pollGradeStatus(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save the grade.");
      setGradePhase({ kind: "idle" }); // save failed -- no job coming; drop the bar
    } finally {
      savingRef.current = false;
      setSaving(false);
      // The user changed the look again while we were saving -> persist newest.
      if (pendingRef.current !== next) void flushLook();
    }
  }, [token, wdCommitLook, pollGradeStatus]);

  const applyLook = useCallback(
    (next: EditLook | undefined, immediate = false) => {
      setLook(next); // instant local: the dial reflects input and never reverts
      pendingRef.current = next;
      // Instant feedback: show the bar the moment the look changes (covers the
      // debounce + save + idle gap before the job appears). Don't stomp a bar
      // that's already mid-job/warming from a prior change.
      setGradePhase((p) => (p.kind === "grading" || p.kind === "warming" ? p : { kind: "applying" }));
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

  // color_response_engine.plan.md: the gallery now mixes CDL presets and
  // engine looks (see getGradePresets) -- `entry.mode` decides which id
  // field to set and which SequenceLook.mode the resolver should apply.
  // Picking either always clears BOTH lut_ref (mode=="lut") and the other
  // mode's id field, so switching between them never leaves a stale
  // selector behind for the resolver to (mutually-exclusively) prefer.
  function selectLook(entry: GradePresetSummary) {
    if (entry.mode === "engine") {
      applyLook(
        { ...(look ?? {}), mode: "engine", look_id: entry.look_id, look_params: null, lut_ref: null, preset_id: null },
        true
      );
    } else {
      applyLook(
        { ...(look ?? {}), mode: "preset", preset_id: entry.preset_id, lut_ref: null, look_id: null, look_params: null },
        true
      );
    }
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

      {gradeDisplay && (
        <div className="space-y-1">
          <div
            className="h-1 w-full overflow-hidden rounded-full"
            style={{ background: "var(--border)" }}
            role="progressbar"
            aria-valuenow={gradeDisplay.pct}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div
              className="h-full rounded-full transition-[width] duration-300"
              style={{ width: `${gradeDisplay.pct}%`, background: "var(--accent)" }}
            />
          </div>
          <p className="text-[11px]" style={{ color: "var(--muted)" }}>
            {gradeDisplay.caption}
          </p>
        </div>
      )}

      {/* Look picker gallery */}
      <section>
        <SectionLabel>Look</SectionLabel>
        <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
          {presets.map((p) => {
            const active =
              p.mode === "engine"
                ? look?.mode === "engine" && look.look_id === p.look_id
                : look?.mode === "preset" && look.preset_id === p.preset_id;
            return (
              <button
                key={`${p.mode}:${p.mode === "engine" ? p.look_id : p.preset_id}`}
                onClick={() => selectLook(p)}
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
