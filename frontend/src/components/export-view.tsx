"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AlertCircle, Download, Loader2 } from "lucide-react";
import { useAuthStore } from "@/stores/auth-store";
import { useEditDocStore } from "@/stores/edit-doc-store";
import { useDriveStore } from "@/stores/drive-store";
import {
  createExport,
  getExport,
  listEditThreads,
  type EditThreadListItem,
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
  const storeThreadId = useEditDocStore((s) => s.threadId);
  const token = useAuthStore((s) => s.session?.access_token);
  const projectFiles = useDriveStore((s) => s.files);

  const [kind, setKind] = useState<ExportKind>("mp4");
  const [quality, setQuality] = useState<ExportQuality>("1080");
  const [includeMedia, setIncludeMedia] = useState(false);
  const [job, setJob] = useState<ExportJob | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Stage 3.1: Export previously had no way to find a thread on its own --
  // threadId is set only by ai-edit-panel.tsx's seed() call, so opening this
  // lens without opening the AI panel first always hit the empty state below,
  // even when the project already has an edit. Discover it ourselves: list
  // this user's threads and keep the ones whose file_ids overlap the current
  // project's files (list_threads already orders by updated_at desc, so the
  // first match is the most recent). Only runs while the store doesn't
  // already have a threadId -- once one exists (via the AI panel), that wins.
  const [discovered, setDiscovered] = useState<EditThreadListItem[]>([]);
  const [pickedThreadId, setPickedThreadId] = useState<string | null>(null);
  useEffect(() => {
    setPickedThreadId(null);
    setDiscovered([]);
    if (storeThreadId || !token || projectFiles.length === 0) return;
    let cancelled = false;
    const projectFileIds = new Set(projectFiles.map((f) => f.id));
    listEditThreads(token)
      .then((res) => {
        if (cancelled) return;
        const matches = res.threads.filter((t) => t.file_ids.some((id) => projectFileIds.has(id)));
        if (matches.length === 1) {
          setPickedThreadId(matches[0].id);
        } else if (matches.length > 1) {
          setDiscovered(matches);
        }
      })
      .catch(() => {
        // Discovery is a convenience, not a requirement -- fall through to
        // today's honest empty state rather than surfacing this as an error.
      });
    return () => {
      cancelled = true;
    };
    // projectFiles is derived from the currently-open project; re-run when it
    // changes (e.g. navigating between projects) but not on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storeThreadId, token, projectFiles.map((f) => f.id).join(",")]);

  const threadId = storeThreadId ?? pickedThreadId;

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
        // Stage 3.2: a missing token is not a reason to silently stop polling
        // -- request() already omits the Authorization header when there's
        // no token, and the backend accepts a tokenless call (dev bypass).
        // If the call genuinely fails, the catch below surfaces it.
        try {
          const r = await getExport(id, token ?? "");
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
    // Stage 3.2: dropped the `!token` no-op (see poll() above for why) --
    // a missing threadId is still a real reason not to proceed.
    if (!threadId || busy) return;
    setError(null);
    setBusy(true);
    setJob(null);
    try {
      const r = await createExport(
        threadId,
        { kind, quality, includeMedia: kind === "rough_cut" ? includeMedia : false },
        token ?? ""
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
    // Stage 3.1: more than one thread in this project touches these files --
    // ask rather than silently guessing which edit to export.
    if (discovered.length > 1) {
      return (
        <div className="mx-auto max-w-sm space-y-4 py-24 text-center">
          <p className="text-lg font-semibold">Which edit?</p>
          <p className="text-sm" style={{ color: "var(--muted)" }}>
            This project has more than one edit. Pick the one to export.
          </p>
          <div className="space-y-2 text-left">
            {discovered.map((t) => (
              <button
                key={t.id}
                type="button"
                onClick={() => setPickedThreadId(t.id)}
                className="flex w-full flex-col items-start gap-0.5 rounded-lg border px-3 py-2.5 text-left transition-colors hover:bg-[var(--border)]"
                style={{ borderColor: "var(--border)" }}
              >
                <span className="text-sm font-medium">{t.title || "Untitled edit"}</span>
                <span className="text-[11px]" style={{ color: "var(--muted)" }}>
                  {new Date(t.created_at).toLocaleDateString()} · {t.clip_count} clip{t.clip_count === 1 ? "" : "s"}
                </span>
              </button>
            ))}
          </div>
        </div>
      );
    }
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
