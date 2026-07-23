"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, Download, Loader2 } from "lucide-react";
import { useAuthStore } from "@/stores/auth-store";
import { useEditDocStore } from "@/stores/edit-doc-store";
import {
  createExport,
  getExport,
  type ExportJob,
  type ExportKind,
  type ExportQuality,
} from "@/lib/api";
import { cn } from "@/lib/utils";

const POLL_MS = 1500;

const KIND_OPTIONS: { id: ExportKind; label: string; description: string }[] = [
  { id: "mp4", label: "Finished video", description: "One baked MP4 -- cuts, framing, split-screen, burned-in subtitles." },
  { id: "rough_cut", label: "Rough cut for NLE", description: "FCPXML + SRT, ready to open in DaVinci Resolve or Premiere Pro." },
  { id: "srt", label: "Subtitles only", description: "Just the .srt sidecar, word-timed." },
];

const QUALITY_OPTIONS: { id: ExportQuality; label: string }[] = [
  { id: "2160", label: "4K" },
  { id: "1080", label: "Full HD" },
  { id: "720", label: "Preview" },
  { id: "source", label: "Source" },
];

function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="mb-2 text-[11px] font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>
      {children}
    </h3>
  );
}

export function ExportView() {
  const threadId = useEditDocStore((s) => s.threadId);
  const token = useAuthStore((s) => s.session?.access_token);

  const [kind, setKind] = useState<ExportKind>("mp4");
  const [quality, setQuality] = useState<ExportQuality>("1080");
  const [includeMedia, setIncludeMedia] = useState(false);
  const [job, setJob] = useState<ExportJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stop = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => stop, [stop]);

  const poll = useCallback(
    (id: string) => {
      stop();
      pollRef.current = setInterval(async () => {
        if (!token) return;
        try {
          const r = await getExport(id, token);
          setJob(r);
          if (r.status === "done" || r.status === "failed") {
            stop();
            setBusy(false);
            if (r.status === "failed") setError(r.error || "Export failed.");
          }
        } catch (e) {
          stop();
          setBusy(false);
          setError(e instanceof Error ? e.message : "Export polling failed.");
        }
      }, POLL_MS);
    },
    [token, stop]
  );

  async function start() {
    if (!token || !threadId || busy) return;
    setError(null);
    setBusy(true);
    setJob(null);
    try {
      const r = await createExport(
        threadId,
        { kind, quality, includeMedia: kind === "rough_cut" ? includeMedia : false },
        token
      );
      setJob(r);
      if (r.status === "done" || r.status === "failed") {
        setBusy(false);
        if (r.status === "failed") setError(r.error || "Export failed.");
      } else {
        poll(r.id);
      }
    } catch (e) {
      setBusy(false);
      setError(e instanceof Error ? e.message : "Could not start export.");
    }
  }

  if (!threadId) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <p className="text-lg font-semibold">Export</p>
        <p className="mt-1 max-w-sm text-sm" style={{ color: "var(--muted)" }}>
          Start an edit first (Drive or the AI panel) -- export bakes the edit you&apos;ve already made.
        </p>
      </div>
    );
  }

  const running = job && (job.status === "queued" || job.status === "running");
  const done = job && job.status === "done" && job.output_url;

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Export</h2>
      </div>

      <div>
        <SectionLabel>What</SectionLabel>
        <div className="space-y-2">
          {KIND_OPTIONS.map((opt) => (
            <button
              key={opt.id}
              type="button"
              onClick={() => setKind(opt.id)}
              className={cn(
                "flex w-full flex-col items-start gap-0.5 rounded-lg border px-3 py-2.5 text-left transition-colors",
                kind !== opt.id && "hover:bg-[var(--border)]"
              )}
              style={{
                borderColor: kind === opt.id ? "var(--accent)" : "var(--border)",
                background: kind === opt.id ? "var(--accent-soft)" : undefined,
              }}
            >
              <span className="text-sm font-medium">{opt.label}</span>
              <span className="text-[11px]" style={{ color: "var(--muted)" }}>
                {opt.description}
              </span>
            </button>
          ))}
        </div>
      </div>

      {(kind === "mp4" || kind === "rough_cut") && (
        <div>
          <SectionLabel>Quality</SectionLabel>
          <div className="flex flex-wrap gap-2">
            {QUALITY_OPTIONS.map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setQuality(opt.id)}
                className={cn(
                  "rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors",
                  quality !== opt.id && "hover:bg-[var(--border)]"
                )}
                style={{
                  borderColor: quality === opt.id ? "var(--accent)" : "var(--border)",
                  background: quality === opt.id ? "var(--accent-soft)" : undefined,
                  color: quality === opt.id ? "var(--foreground)" : "var(--muted)",
                }}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {kind === "rough_cut" && (
        <div>
          <SectionLabel>Media</SectionLabel>
          <label className="flex items-start gap-2 text-sm">
            <input
              type="checkbox"
              checked={includeMedia}
              onChange={(e) => setIncludeMedia(e.target.checked)}
              className="mt-0.5"
            />
            <span>
              Include original media
              <span className="block text-[11px]" style={{ color: "var(--muted)" }}>
                Off (default): a tiny project-only bundle -- relink to your own footage. On: media
                is copied into the ZIP (or linked, for a very large project).
              </span>
            </span>
          </label>
        </div>
      )}

      <button
        type="button"
        onClick={start}
        disabled={busy}
        className="flex items-center justify-center gap-1.5 rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
        style={{ background: "var(--accent)", color: "var(--background)" }}
      >
        {busy && <Loader2 size={14} className="animate-spin" />}
        {busy ? "Exporting…" : "Export"}
      </button>

      {running && (
        <p className="text-[11px]" style={{ color: "var(--muted)" }}>
          {job!.status === "queued" ? "Queued…" : "Rendering…"}
        </p>
      )}

      {done && (
        <a
          href={job!.output_url!}
          download
          className="flex w-fit items-center justify-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors hover:bg-[var(--accent-soft)]"
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
  );
}
