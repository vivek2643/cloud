"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import { useAuthStore } from "@/stores/auth-store";
import { useDriveStore, type AiTimelineData } from "@/stores/drive-store";
import {
  listProjects,
  getLatestEdl,
  renameProject,
  deleteProject,
  type ProjectSummary,
} from "@/lib/api";
import {
  Clapperboard,
  RefreshCw,
  Film,
  Sparkles,
  Pencil,
  Trash2,
  Loader2,
  Clock,
} from "lucide-react";

function fmtDuration(ms: number): string {
  const total = Math.round(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function fmtWhen(iso?: string | null): string {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diff = Date.now() - then;
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30) return `${days}d ago`;
  return new Date(iso).toLocaleDateString();
}

function authorLabel(kind: ProjectSummary["author_kind"]): { label: string; ai: boolean } {
  if (kind === "claude") return { label: "AI", ai: true };
  if (kind === "user") return { label: "Manual", ai: false };
  return { label: "System", ai: false };
}

export default function EditsPage() {
  const router = useRouter();
  const session = useAuthStore((s) => s.session);
  const openSavedEdit = useDriveStore((s) => s.openSavedEdit);

  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [openingId, setOpeningId] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    if (!session?.access_token) return;
    setLoading(true);
    setError(null);
    try {
      setProjects(await listProjects(session.access_token));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }, [session?.access_token]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function openEdit(p: ProjectSummary) {
    const token = session?.access_token;
    if (!token || openingId) return;
    setOpeningId(p.id);
    setError(null);
    try {
      const edl = await getLatestEdl(p.id, token);
      if (!edl || edl.clips.length === 0) {
        setError("This edit has no clips to open yet.");
        return;
      }
      const timeline: AiTimelineData = {
        clips: edl.clips.map((c) => ({
          shot_id: c.shot_id ?? null,
          file_id: c.file_id ?? null,
          file_name: c.file_name ?? null,
          source_in_ms: c.source_in_ms,
          source_out_ms: c.source_out_ms,
          role_in_edit: null,
          why: null,
        })),
        totalMs: edl.total_duration_ms,
        renderStatus: null,
        renderUrl: null,
        projectId: edl.project_id,
        baseVersionId: edl.version.id,
      };
      // Hydrate the docked editor (left media + bottom timeline, right
      // render/chat) and drop the user back on the Drive where it lives.
      openSavedEdit({ fileIds: p.source_file_ids, timeline });
      router.push("/drive");
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setOpeningId(null);
    }
  }

  async function handleRename(p: ProjectSummary) {
    const token = session?.access_token;
    if (!token) return;
    const next = window.prompt("Rename edit", p.name)?.trim();
    if (!next || next === p.name) return;
    setBusyId(p.id);
    try {
      await renameProject(p.id, next, token);
      setProjects((prev) => prev.map((x) => (x.id === p.id ? { ...x, name: next } : x)));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusyId(null);
    }
  }

  async function handleDelete(p: ProjectSummary) {
    const token = session?.access_token;
    if (!token) return;
    if (!window.confirm(`Delete "${p.name}"? This removes the edit and all its versions. Your source videos are not affected.`)) {
      return;
    }
    setBusyId(p.id);
    try {
      await deleteProject(p.id, token);
      setProjects((prev) => prev.filter((x) => x.id !== p.id));
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setBusyId(null);
    }
  }

  return (
    <div className="px-8 py-8">
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Clapperboard size={22} />
          <div>
            <h1 className="text-xl font-bold leading-tight">Edits</h1>
            <p className="text-xs" style={{ color: "var(--muted)" }}>
              Your saved edits. Open one to keep editing in the timeline.
            </p>
          </div>
        </div>
        <button
          onClick={() => void refresh()}
          className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition-colors hover:opacity-80"
          style={{ borderColor: "var(--border)" }}
          title="Refresh"
        >
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      {error && (
        <div
          className="mb-4 rounded-lg border px-4 py-3 text-sm"
          style={{ borderColor: "#ef4444", color: "#ef4444", background: "var(--accent-soft)" }}
        >
          {error}
        </div>
      )}

      {loading && projects.length === 0 ? (
        <div className="flex items-center gap-2 py-16 text-sm" style={{ color: "var(--muted)" }}>
          <Loader2 size={16} className="animate-spin" /> Loading edits…
        </div>
      ) : projects.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-3 py-20 text-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-full" style={{ background: "var(--sidebar)" }}>
            <Clapperboard size={24} style={{ color: "var(--muted)" }} />
          </div>
          <div className="text-sm font-medium">No saved edits yet</div>
          <p className="max-w-sm text-xs" style={{ color: "var(--muted)" }}>
            Cut a sequence in the AI editor or the timeline and it&apos;ll be saved here automatically.
          </p>
          <button
            onClick={() => router.push("/edit")}
            className="mt-2 flex items-center gap-1.5 rounded-lg px-4 py-2 text-sm font-medium text-white transition-colors"
            style={{ background: "var(--accent)" }}
          >
            <Sparkles size={15} /> Start an edit
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
          {projects.map((p) => {
            const author = authorLabel(p.author_kind);
            const opening = openingId === p.id;
            const busy = busyId === p.id;
            return (
              <div
                key={p.id}
                className="group relative flex flex-col overflow-hidden rounded-xl border transition-shadow hover:shadow-md"
                style={{ borderColor: "var(--border)", background: "var(--background)" }}
              >
                <button
                  onClick={() => void openEdit(p)}
                  disabled={opening}
                  className="relative flex aspect-video w-full items-center justify-center overflow-hidden"
                  style={{ background: "#000" }}
                  title="Open this edit"
                >
                  {p.thumbnail_url ? (
                    <Image
                      src={p.thumbnail_url}
                      alt={p.name}
                      fill
                      sizes="(max-width: 640px) 100vw, 25vw"
                      className="object-cover opacity-90 transition-opacity group-hover:opacity-100"
                      unoptimized
                    />
                  ) : (
                    <Film size={28} className="text-white/30" />
                  )}
                  <span
                    className="absolute bottom-1.5 right-1.5 rounded bg-black/70 px-1.5 py-0.5 font-mono text-[11px] text-white"
                  >
                    {fmtDuration(p.duration_ms)}
                  </span>
                  {opening && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                      <Loader2 size={22} className="animate-spin text-white" />
                    </div>
                  )}
                </button>

                <div className="flex flex-1 flex-col gap-1 p-3">
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0 truncate text-sm font-medium" title={p.name}>
                      {p.name}
                    </div>
                    <span
                      className="shrink-0 rounded-full border px-1.5 py-0 text-[10px] font-medium uppercase tracking-wide"
                      style={
                        author.ai
                          ? { borderColor: "var(--accent)", color: "var(--accent)" }
                          : { borderColor: "var(--border)", color: "var(--muted)" }
                      }
                    >
                      {author.label}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-xs" style={{ color: "var(--muted)" }}>
                    <span>{p.clip_count} clip{p.clip_count === 1 ? "" : "s"}</span>
                    <span>·</span>
                    <span>{p.version_count} version{p.version_count === 1 ? "" : "s"}</span>
                  </div>
                  <div className="mt-auto flex items-center justify-between pt-2">
                    <span className="flex items-center gap-1 text-[11px]" style={{ color: "var(--muted)" }}>
                      <Clock size={11} /> {fmtWhen(p.updated_at)}
                    </span>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => void handleRename(p)}
                        disabled={busy}
                        className="rounded-md border p-1.5 transition-colors hover:opacity-80 disabled:opacity-50"
                        style={{ borderColor: "var(--border)" }}
                        title="Rename"
                      >
                        <Pencil size={13} />
                      </button>
                      <button
                        onClick={() => void handleDelete(p)}
                        disabled={busy}
                        className="rounded-md border p-1.5 transition-colors hover:opacity-80 disabled:opacity-50"
                        style={{ borderColor: "var(--border)", color: "#ef4444" }}
                        title="Delete"
                      >
                        {busy ? <Loader2 size={13} className="animate-spin" /> : <Trash2 size={13} />}
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
