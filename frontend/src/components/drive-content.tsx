"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useDriveStore } from "@/stores/drive-store";
import { useAuthStore } from "@/stores/auth-store";
import { FileIcon } from "./file-icon";
import { formatBytes, formatDuration } from "@/lib/utils";
import { getFilePlaybackUrl, deleteFile } from "@/lib/api";
import {
  MoreHorizontal,
  Loader2,
  Play,
  Pause,
  Volume2,
  VolumeX,
  CheckCircle2,
  Circle,
  ExternalLink,
  Trash2,
} from "lucide-react";
import type { Folder, FileRecord } from "@/lib/api";

interface DriveContentProps {
  onFileContextMenu?: (file: FileRecord, e: React.MouseEvent) => void;
  onFolderContextMenu?: (folder: Folder, e: React.MouseEvent) => void;
}

export function DriveContent({ onFileContextMenu, onFolderContextMenu }: DriveContentProps) {
  const router = useRouter();
  const {
    folders,
    files,
    viewMode,
    loading,
    selectedIds,
    searchQuery,
    toggleSelected,
    removeFile,
  } = useDriveStore();
  const session = useAuthStore((s) => s.session);

  const [fileMenu, setFileMenu] = useState<{ file: FileRecord; x: number; y: number } | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  // Either a right-click or the row's "⋯" button opens our menu. If the parent
  // passed its own handler we honour it too (none currently do).
  function handleFileMenu(file: FileRecord, e: React.MouseEvent) {
    onFileContextMenu?.(file, e);
    setFileMenu({ file, x: e.clientX, y: e.clientY });
  }

  async function handleDelete(file: FileRecord) {
    setFileMenu(null);
    if (!session?.access_token) return;
    const ok = window.confirm(
      `Delete "${file.name}"?\n\nThis permanently removes the video and all of its L1/L2 analysis. This cannot be undone.`,
    );
    if (!ok) return;
    setDeletingId(file.id);
    try {
      await deleteFile(file.id, session.access_token);
      removeFile(file.id);
    } catch (err) {
      window.alert(`Could not delete the video: ${(err as Error).message}`);
    } finally {
      setDeletingId(null);
    }
  }

  // Client-side filter driven by the top search bar.
  const q = searchQuery.trim().toLowerCase();
  const visibleFolders = useMemo(
    () => (q ? folders.filter((f) => f.name.toLowerCase().includes(q)) : folders),
    [folders, q],
  );
  const visibleFiles = useMemo(
    () => (q ? files.filter((f) => f.name.toLowerCase().includes(q)) : files),
    [files, q],
  );

  if (loading) {
    return (
      <div className="flex flex-1 items-center justify-center py-20">
        <Loader2 size={24} className="animate-spin" style={{ color: "var(--accent)" }} />
      </div>
    );
  }

  if (visibleFolders.length === 0 && visibleFiles.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center py-20">
        <div className="text-4xl">{q ? "🔍" : "📁"}</div>
        <p className="mt-3 text-sm font-medium">
          {q ? "No matches" : "This folder is empty"}
        </p>
        <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
          {q ? "Try a different search" : "Drag and drop videos here or upload"}
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {viewMode === "list" ? (
        <ListView
          folders={visibleFolders}
          files={visibleFiles}
          selectedIds={selectedIds}
          deletingId={deletingId}
          onToggleSelect={toggleSelected}
          onNavigate={(id) => router.push(`/drive/folder/${id}`)}
          onOpenFile={(id) => router.push(`/file/${id}`)}
          onFileContextMenu={handleFileMenu}
          onFolderContextMenu={onFolderContextMenu}
        />
      ) : (
        <GridView
          folders={visibleFolders}
          files={visibleFiles}
          selectedIds={selectedIds}
          deletingId={deletingId}
          onToggleSelect={toggleSelected}
          onNavigate={(id) => router.push(`/drive/folder/${id}`)}
          onOpenFile={(id) => router.push(`/file/${id}`)}
          onFileContextMenu={handleFileMenu}
          onFolderContextMenu={onFolderContextMenu}
        />
      )}

      {fileMenu && (
        <FileMenu
          file={fileMenu.file}
          x={fileMenu.x}
          y={fileMenu.y}
          onClose={() => setFileMenu(null)}
          onOpen={() => {
            const id = fileMenu.file.id;
            setFileMenu(null);
            router.push(`/file/${id}`);
          }}
          onDelete={() => handleDelete(fileMenu.file)}
        />
      )}
    </div>
  );
}

// --- Context menu ---

function FileMenu({
  file,
  x,
  y,
  onClose,
  onOpen,
  onDelete,
}: {
  file: FileRecord;
  x: number;
  y: number;
  onClose: () => void;
  onOpen: () => void;
  onDelete: () => void;
}) {
  // Clamp so the menu never spills off the right/bottom edge.
  const MENU_W = 180;
  const MENU_H = 92;
  const left = Math.min(x, (typeof window !== "undefined" ? window.innerWidth : x) - MENU_W - 8);
  const top = Math.min(y, (typeof window !== "undefined" ? window.innerHeight : y) - MENU_H - 8);
  return (
    <>
      <div className="fixed inset-0 z-40" onClick={onClose} onContextMenu={(e) => { e.preventDefault(); onClose(); }} />
      <div
        className="fixed z-50 overflow-hidden rounded-lg border py-1 shadow-xl"
        style={{ left, top, minWidth: MENU_W, background: "var(--background)", borderColor: "var(--border)" }}
      >
        <div className="truncate px-3 pb-1 pt-0.5 text-[11px]" style={{ color: "var(--muted)" }}>
          {file.name}
        </div>
        <button
          onClick={onOpen}
          className="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm transition-colors hover:bg-[var(--accent-soft)]"
        >
          <ExternalLink size={15} style={{ color: "var(--muted)" }} />
          Open
        </button>
        <button
          onClick={onDelete}
          className="flex w-full items-center gap-2.5 px-3 py-2 text-left text-sm transition-colors hover:bg-[var(--accent-soft)]"
          style={{ color: "#ef4444" }}
        >
          <Trash2 size={15} />
          Delete
        </button>
      </div>
    </>
  );
}

// --- Grid View ---

function GridView({
  folders,
  files,
  selectedIds,
  deletingId,
  compact = false,
  onToggleSelect,
  onNavigate,
  onOpenFile,
  onFileContextMenu,
  onFolderContextMenu,
}: {
  folders: Folder[];
  files: FileRecord[];
  selectedIds: Set<string>;
  deletingId?: string | null;
  compact?: boolean;
  onToggleSelect: (id: string) => void;
  onNavigate: (id: string) => void;
  onOpenFile: (id: string) => void;
  onFileContextMenu?: (f: FileRecord, e: React.MouseEvent) => void;
  onFolderContextMenu?: (f: Folder, e: React.MouseEvent) => void;
}) {
  // When the AI panel is docked the main column is narrower, so we drop to
  // fewer columns so everything still fits without horizontal scroll.
  const fileGridCls = compact
    ? "grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3"
    : "grid grid-cols-2 gap-4 md:grid-cols-3 2xl:grid-cols-4";
  return (
    <div className="space-y-7">
      {folders.length > 0 && (
        <section>
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--muted)" }}>
            Folders
          </h3>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
            {folders.map((folder) => (
              <button
                key={folder.id}
                onDoubleClick={() => onNavigate(folder.id)}
                onContextMenu={(e) => { e.preventDefault(); onFolderContextMenu?.(folder, e); }}
                className="flex items-center gap-2.5 rounded-lg border p-3 text-left transition-colors hover:border-[var(--accent)]"
                style={{ borderColor: "var(--border)" }}
              >
                <FileIcon type="folder" />
                <span className="truncate text-sm">{folder.name}</span>
              </button>
            ))}
          </div>
        </section>
      )}

      {files.length > 0 && (
        <section>
          <h3 className="mb-3 text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--muted)" }}>
            Videos
          </h3>
          {/* Enlarged cards: one per row on phones, up to three on wide screens
              so each preview is big enough to watch inline. */}
          <div className={fileGridCls}>
            {files.map((file) => (
              <VideoCard
                key={file.id}
                file={file}
                selected={selectedIds.has(file.id)}
                deleting={deletingId === file.id}
                onToggleSelect={() => onToggleSelect(file.id)}
                onOpen={() => onOpenFile(file.id)}
                onContextMenu={(e) => { e.preventDefault(); onFileContextMenu?.(file, e); }}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function VideoCard({
  file,
  selected,
  deleting = false,
  onToggleSelect,
  onOpen,
  onContextMenu,
}: {
  file: FileRecord;
  selected: boolean;
  deleting?: boolean;
  onToggleSelect: () => void;
  onOpen: () => void;
  onContextMenu: (e: React.MouseEvent) => void;
}) {
  const session = useAuthStore((s) => s.session);
  const [playUrl, setPlayUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [muted, setMuted] = useState(false);
  // `desiredPlaying` is our intent; `playing` reflects what the element is
  // actually doing. A "pinned" play (via the center button) survives the
  // mouse leaving, whereas a hover-preview stops when the cursor moves away.
  const [desiredPlaying, setDesiredPlaying] = useState(false);
  const [playing, setPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const pinnedRef = useRef(false);

  const isProcessing = file.status === "processing" || file.status === "uploading";
  const isVideo = file.file_type === "video";
  const canPlay = isVideo && file.status === "ready";

  async function ensureUrl(): Promise<void> {
    if (playUrl) return;
    if (!session?.access_token) {
      setError("Sign in to play");
      return;
    }
    setError(null);
    try {
      const { url } = await getFilePlaybackUrl(file.id, session.access_token);
      setPlayUrl(url);
    } catch {
      setError("Could not load video");
    }
  }

  // Keep the element's mute flag in sync.
  useEffect(() => {
    if (videoRef.current) videoRef.current.muted = muted;
  }, [muted, playUrl]);

  // Drive play/pause from intent once a source is available.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !playUrl) return;
    if (desiredPlaying) {
      v.muted = muted;
      v.play().then(() => setPlaying(true)).catch(() => {});
    } else {
      v.pause();
      setPlaying(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [desiredPlaying, playUrl]);

  async function handleEnter() {
    if (!canPlay || pinnedRef.current) return;
    await ensureUrl();
    setDesiredPlaying(true);
  }

  function handleLeave() {
    if (pinnedRef.current) return;
    setDesiredPlaying(false);
    const v = videoRef.current;
    if (v) {
      try { v.currentTime = 0; } catch { /* ignore */ }
    }
  }

  async function handleCenterToggle(e: React.MouseEvent) {
    e.stopPropagation();
    if (!canPlay) return;
    if (desiredPlaying) {
      pinnedRef.current = false;
      setDesiredPlaying(false);
    } else {
      pinnedRef.current = true;
      await ensureUrl();
      setDesiredPlaying(true);
    }
  }

  function handleMuteToggle(e: React.MouseEvent) {
    e.stopPropagation();
    setMuted((m) => !m);
  }

  return (
    <div
      onContextMenu={onContextMenu}
      className="group relative flex flex-col overflow-hidden rounded-xl border transition-colors hover:border-[var(--accent)]"
      style={{
        borderColor: selected ? "var(--accent)" : "var(--border)",
        boxShadow: selected ? "0 0 0 1px var(--accent)" : undefined,
        background: "var(--background)",
      }}
    >
      {/* Clicking anywhere on the video selects it. The control buttons below
          stop propagation so they don't toggle selection. Dragging it drops
          the clip onto the AI editor timeline. */}
      <div
        onClick={onToggleSelect}
        onMouseEnter={handleEnter}
        onMouseLeave={handleLeave}
        draggable={canPlay}
        onDragStart={(e) => {
          if (!canPlay) return;
          const payload = JSON.stringify({
            file_id: file.id,
            file_name: file.name,
            duration_seconds: file.duration_seconds ?? undefined,
          });
          e.dataTransfer.setData("application/edso-clip", payload);
          e.dataTransfer.setData("text/plain", payload);
          e.dataTransfer.effectAllowed = "copy";
        }}
        className="relative flex aspect-video cursor-pointer items-center justify-center overflow-hidden"
        style={{ background: "#000" }}
        title={selected ? "Click to deselect" : "Click to select"}
      >
        {playUrl && (
          <video
            ref={videoRef}
            src={playUrl}
            playsInline
            loop
            muted={muted}
            className="h-full w-full bg-black object-contain"
          />
        )}
        {!playUrl && <FileIcon type={file.file_type as "video"} size={36} />}

        {/* Selection indicator (top-left). Visible when selected, and on hover. */}
        <span
          className="video-sel-indicator pointer-events-none absolute left-2 top-2 z-20 rounded-full p-0.5 transition-opacity"
          style={{
            background: "rgba(0,0,0,0.55)",
            color: selected ? "var(--accent)" : "white",
            opacity: selected ? 1 : 0,
          }}
        >
          {selected ? <CheckCircle2 size={18} /> : <Circle size={18} />}
        </span>
        {!selected && (
          <style>{`.group:hover .video-sel-indicator { opacity: 1 !important; }`}</style>
        )}

        {/* Center play / pause control. */}
        {canPlay && (
          <button
            onClick={handleCenterToggle}
            className={`absolute left-1/2 top-1/2 z-20 flex h-12 w-12 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full shadow-lg transition-all ${
              playing ? "opacity-0 group-hover:opacity-100" : "opacity-100"
            }`}
            style={{ background: "var(--accent)" }}
            title={playing ? "Pause" : "Play"}
          >
            {playing ? (
              <Pause size={22} className="text-white" fill="white" />
            ) : (
              <Play size={22} className="ml-0.5 text-white" fill="white" />
            )}
          </button>
        )}

        {/* Mute / unmute (top-right). */}
        {canPlay && (
          <button
            onClick={handleMuteToggle}
            className="absolute right-2 top-2 z-20 flex items-center justify-center rounded-full p-1.5 text-white transition-colors hover:bg-black/40"
            style={{ background: "rgba(0,0,0,0.55)" }}
            title={muted ? "Unmute" : "Mute"}
          >
            {muted ? <VolumeX size={15} /> : <Volume2 size={15} />}
          </button>
        )}

        {isProcessing && !deleting && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-2 bg-black/40">
            <Loader2 size={20} className="animate-spin text-white" />
            <span className="text-xs text-white/80">Processing…</span>
          </div>
        )}

        {deleting && (
          <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-2 bg-black/60">
            <Loader2 size={20} className="animate-spin text-white" />
            <span className="text-xs text-white/80">Deleting…</span>
          </div>
        )}

        {error && (
          <span className="absolute bottom-2 left-2 z-20 rounded bg-black/70 px-2 py-0.5 text-[11px] text-white">
            {error}
          </span>
        )}

        {isVideo && file.duration_seconds != null && !playing && (
          <span className="absolute bottom-2 right-2 z-10 rounded bg-black/70 px-1.5 py-0.5 text-[11px] font-medium text-white">
            {formatDuration(file.duration_seconds)}
          </span>
        )}
      </div>

      <div className="flex items-start gap-2 p-2.5">
        <button onClick={onOpen} className="min-w-0 flex-1 text-left" title="Open details">
          <div className="truncate text-sm font-medium">{file.name}</div>
          <div className="mt-0.5 text-xs" style={{ color: "var(--muted)" }}>
            {formatBytes(file.file_size)}
          </div>
        </button>
        <button
          onClick={(e) => { e.stopPropagation(); onContextMenu(e); }}
          className="shrink-0 rounded p-1 opacity-0 transition-opacity group-hover:opacity-100"
          style={{ color: "var(--muted)" }}
        >
          <MoreHorizontal size={16} />
        </button>
      </div>
    </div>
  );
}

// --- List View ---

function ListView({
  folders,
  files,
  selectedIds,
  deletingId,
  onToggleSelect,
  onNavigate,
  onOpenFile,
  onFileContextMenu,
  onFolderContextMenu,
}: {
  folders: Folder[];
  files: FileRecord[];
  selectedIds: Set<string>;
  deletingId?: string | null;
  onToggleSelect: (id: string) => void;
  onNavigate: (id: string) => void;
  onOpenFile: (id: string) => void;
  onFileContextMenu?: (f: FileRecord, e: React.MouseEvent) => void;
  onFolderContextMenu?: (f: Folder, e: React.MouseEvent) => void;
}) {
  return (
    <div className="overflow-hidden rounded-lg border" style={{ borderColor: "var(--border)" }}>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left" style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}>
            <th className="w-8" />
            <th className="px-4 py-2.5 font-medium">Name</th>
            <th className="px-4 py-2.5 font-medium">Size</th>
            <th className="px-4 py-2.5 font-medium">Modified</th>
            <th className="w-10" />
          </tr>
        </thead>
        <tbody>
          {folders.map((folder) => (
            <tr
              key={folder.id}
              onDoubleClick={() => onNavigate(folder.id)}
              onContextMenu={(e) => { e.preventDefault(); onFolderContextMenu?.(folder, e); }}
              className="cursor-pointer border-b transition-colors hover:bg-[var(--accent-soft)]"
              style={{ borderColor: "var(--border)" }}
            >
              <td className="w-8" />
              <td className="flex items-center gap-2.5 px-4 py-2.5">
                <FileIcon type="folder" size={16} />
                <span className="truncate">{folder.name}</span>
              </td>
              <td className="px-4 py-2.5" style={{ color: "var(--muted)" }}>—</td>
              <td className="px-4 py-2.5" style={{ color: "var(--muted)" }}>
                {new Date(folder.updated_at).toLocaleDateString()}
              </td>
              <td />
            </tr>
          ))}
          {files.map((file) => {
            const sel = selectedIds.has(file.id);
            const isDeleting = deletingId === file.id;
            return (
              <tr
                key={file.id}
                onClick={() => onOpenFile(file.id)}
                onContextMenu={(e) => { e.preventDefault(); onFileContextMenu?.(file, e); }}
                className="group cursor-pointer border-b transition-colors hover:bg-[var(--accent-soft)]"
                style={{
                  borderColor: "var(--border)",
                  background: sel ? "var(--accent-soft)" : undefined,
                  opacity: isDeleting ? 0.5 : 1,
                  pointerEvents: isDeleting ? "none" : undefined,
                }}
              >
                <td className="w-8 px-2 py-2.5">
                  <button
                    onClick={(e) => { e.stopPropagation(); onToggleSelect(file.id); }}
                    className="flex items-center justify-center"
                    style={{ color: sel ? "var(--accent)" : "var(--muted)" }}
                    title={sel ? "Deselect" : "Select for AI Editor"}
                  >
                    {sel ? <CheckCircle2 size={16} /> : <Circle size={16} />}
                  </button>
                </td>
                <td className="flex items-center gap-2.5 px-4 py-2.5">
                  <FileIcon type={file.file_type as "video"} size={16} />
                  <span className="truncate">{file.name}</span>
                  {(file.status === "processing" || isDeleting) && (
                    <Loader2 size={14} className="animate-spin" style={{ color: "var(--accent)" }} />
                  )}
                </td>
                <td className="px-4 py-2.5" style={{ color: "var(--muted)" }}>
                  {formatBytes(file.file_size)}
                </td>
                <td className="px-4 py-2.5" style={{ color: "var(--muted)" }}>
                  {new Date(file.updated_at).toLocaleDateString()}
                </td>
                <td className="px-4 py-2.5">
                  <button
                    onClick={(e) => { e.stopPropagation(); onFileContextMenu?.(file, e); }}
                    className="rounded p-0.5 opacity-0 transition-opacity group-hover:opacity-100"
                    style={{ color: "var(--muted)" }}
                  >
                    <MoreHorizontal size={14} />
                  </button>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
