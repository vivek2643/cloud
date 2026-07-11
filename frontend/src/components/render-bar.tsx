"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Download, Loader2, Clapperboard, AlertCircle, X } from "lucide-react";
import {
  createRender,
  getRender,
  type RenderJob,
  type RenderPreset,
} from "@/lib/api";

const POLL_MS = 1500;

const PRESETS: { id: RenderPreset; label: string }[] = [
  { id: "preview", label: "Preview · 720p" },
  { id: "export", label: "Export · 1080p" },
];

/**
 * Export control (editor_ui.plan.md SS2.3): relocated on top of the monitor
 * as a compact icon button. Preset choice + progress + download hide behind
 * it in a popover, instead of an always-visible strip between the monitor
 * and chat. Same render logic as before -- only the surface changed.
 */
export function RenderBar({
  threadId,
  version,
  token,
  disabled,
}: {
  threadId: string;
  version: number | null;
  token: string | undefined;
  disabled: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [preset, setPreset] = useState<RenderPreset>("preview");
  const [job, setJob] = useState<RenderJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const popRef = useRef<HTMLDivElement>(null);

  const stop = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  // Reset render state when the underlying plan version changes.
  useEffect(() => {
    stop();
    setJob(null);
    setError(null);
    setBusy(false);
  }, [version, threadId, stop]);

  useEffect(() => stop, [stop]);

  // Close the popover on an outside click.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (popRef.current && !popRef.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", onDown);
    return () => window.removeEventListener("mousedown", onDown);
  }, [open]);

  const poll = useCallback(
    (id: string) => {
      stop();
      pollRef.current = setInterval(async () => {
        if (!token) return;
        try {
          const r = await getRender(id, token);
          setJob(r);
          if (r.status === "done" || r.status === "failed" || r.status === "cancelled") {
            stop();
            setBusy(false);
            if (r.status === "failed") setError(r.error || "Render failed.");
          }
        } catch (e) {
          stop();
          setBusy(false);
          setError(e instanceof Error ? e.message : "Render polling failed.");
        }
      }, POLL_MS);
    },
    [token, stop]
  );

  async function start() {
    if (!token || busy || disabled) return;
    setError(null);
    setBusy(true);
    try {
      const r = await createRender(threadId, preset, token, version ?? undefined);
      setJob(r);
      if (r.status === "done" || r.status === "failed") {
        setBusy(false);
        if (r.status === "failed") setError(r.error || "Render failed.");
      } else {
        poll(r.id);
      }
    } catch (e) {
      setBusy(false);
      setError(e instanceof Error ? e.message : "Could not start render.");
    }
  }

  const running = job && (job.status === "queued" || job.status === "running");
  const done = job && job.status === "done" && job.output_url;

  return (
    <div ref={popRef} className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        disabled={disabled}
        className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[11px] font-medium backdrop-blur-sm transition-opacity disabled:opacity-40"
        style={{ background: "var(--accent)", color: "var(--background)" }}
        title={disabled ? "No timeline to render yet" : "Export this edit"}
      >
        {busy || running ? <Loader2 size={12} className="animate-spin" /> : <Clapperboard size={12} />}
        Export
      </button>

      {open && (
        <div
          className="absolute right-0 top-[calc(100%+6px)] z-20 w-64 space-y-2 rounded-xl border p-3 text-xs shadow-none"
          style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
        >
          <div className="flex items-center justify-between">
            <span className="font-medium">Export</span>
            <button onClick={() => setOpen(false)} className="rounded p-0.5 hover:bg-[var(--accent-soft)]">
              <X size={12} />
            </button>
          </div>

          <div className="flex items-center gap-2">
            <select
              value={preset}
              onChange={(e) => setPreset(e.target.value as RenderPreset)}
              disabled={busy || disabled}
              className="flex-1 rounded-lg border bg-transparent px-2 py-1.5 text-xs outline-none"
              style={{ borderColor: "var(--border)" }}
            >
              {PRESETS.map((p) => (
                <option key={p.id} value={p.id} style={{ background: "var(--background)" }}>
                  {p.label}
                </option>
              ))}
            </select>
            <button
              onClick={start}
              disabled={busy || disabled}
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium transition-opacity disabled:opacity-40"
              style={{ background: "var(--accent)", color: "var(--background)" }}
            >
              {busy ? <Loader2 size={12} className="animate-spin" /> : "Render"}
            </button>
          </div>

          {running && (
            <div>
              <div className="h-1.5 w-full overflow-hidden rounded-full" style={{ background: "var(--accent-soft)" }}>
                <div
                  className="h-full rounded-full transition-[width]"
                  style={{ width: `${job!.progress_pct}%`, background: "var(--accent)" }}
                />
              </div>
              <p className="mt-1 text-[11px]" style={{ color: "var(--muted)" }}>
                {job!.status === "queued" ? "Queued…" : `Rendering ${job!.progress_pct}%`}
              </p>
            </div>
          )}

          {done && (
            <a
              href={job!.output_url!}
              download
              className="flex items-center justify-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-[var(--accent-soft)]"
              style={{ borderColor: "var(--border)" }}
            >
              <Download size={13} /> Download
            </a>
          )}

          {error && (
            <div className="flex items-start gap-1.5 text-[11px]" style={{ color: "var(--danger)" }}>
              <AlertCircle size={12} className="mt-0.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
