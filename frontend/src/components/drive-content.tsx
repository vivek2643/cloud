"use client";

import { useRouter } from "next/navigation";
import { useDriveStore } from "@/stores/drive-store";
import { FileIcon } from "./file-icon";
import { formatBytes, formatDuration } from "@/lib/utils";
import { MoreHorizontal, Loader2, Sparkles, X, CheckCircle2, Circle } from "lucide-react";
import type { Folder, FileRecord } from "@/lib/api";

interface DriveContentProps {
  onFileContextMenu?: (file: FileRecord, e: React.MouseEvent) => void;
  onFolderContextMenu?: (folder: Folder, e: React.MouseEvent) => void;
}

export function DriveContent({ onFileContextMenu, onFolderContextMenu }: DriveContentProps) {
  const router = useRouter();
  const { folders, files, viewMode, loading, selectedIds, toggleSelected, clearSelection } = useDriveStore();

  if (loading) {
    return (
      <div className="flex flex-1 items-center justify-center py-20">
        <Loader2 size={24} className="animate-spin" style={{ color: "var(--muted)" }} />
      </div>
    );
  }

  if (folders.length === 0 && files.length === 0) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center py-20">
        <div className="text-4xl">📁</div>
        <p className="mt-3 text-sm font-medium">This folder is empty</p>
        <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
          Drag and drop files here or create a new folder
        </p>
      </div>
    );
  }

  function handleEditSelected() {
    const ids = Array.from(selectedIds);
    if (ids.length === 0) return;
    // Stash selected file metadata in sessionStorage so the chat can render
    // names without an extra round-trip. The URL still carries the IDs as the
    // source of truth (so a refresh keeps the scope).
    const idToFile = new Map(files.map((f) => [f.id, f]));
    const payload = ids
      .map((id) => idToFile.get(id))
      .filter((f): f is FileRecord => !!f)
      .map((f) => ({ id: f.id, name: f.name, file_type: f.file_type }));
    try {
      window.sessionStorage.setItem("edso_edit_scope_v1", JSON.stringify({ files: payload, ts: Date.now() }));
    } catch {
      // sessionStorage failure is non-fatal -- the URL still has the ids.
    }
    const qs = new URLSearchParams({ file_ids: ids.join(",") });
    router.push(`/edit?${qs.toString()}`);
  }

  // Anything that's selected AND is in the current files list is the user's
  // pool. We only count "video" files toward the Edit affordance because
  // the editor only operates on indexed video.
  const selectedVideoCount = files.filter(
    (f) => selectedIds.has(f.id) && f.file_type === "video",
  ).length;

  return (
    <div className="space-y-4">
      {selectedIds.size > 0 && (
        <SelectionBar
          totalSelected={selectedIds.size}
          videoSelected={selectedVideoCount}
          onEdit={handleEditSelected}
          onClear={clearSelection}
        />
      )}

      {viewMode === "list" ? (
        <ListView
          folders={folders}
          files={files}
          selectedIds={selectedIds}
          onToggleSelect={toggleSelected}
          onNavigate={(id) => router.push(`/drive/folder/${id}`)}
          onOpenFile={(id) => router.push(`/file/${id}`)}
          onFileContextMenu={onFileContextMenu}
          onFolderContextMenu={onFolderContextMenu}
        />
      ) : (
        <GridView
          folders={folders}
          files={files}
          selectedIds={selectedIds}
          onToggleSelect={toggleSelected}
          onNavigate={(id) => router.push(`/drive/folder/${id}`)}
          onOpenFile={(id) => router.push(`/file/${id}`)}
          onFileContextMenu={onFileContextMenu}
          onFolderContextMenu={onFolderContextMenu}
        />
      )}
    </div>
  );
}

// --- Selection Bar ---

function SelectionBar({
  totalSelected,
  videoSelected,
  onEdit,
  onClear,
}: {
  totalSelected: number;
  videoSelected: number;
  onEdit: () => void;
  onClear: () => void;
}) {
  return (
    <div
      className="sticky top-0 z-10 flex items-center justify-between gap-3 rounded-lg border px-3 py-2 shadow-sm"
      style={{
        borderColor: "var(--accent)",
        background: "var(--background)",
      }}
    >
      <div className="flex items-center gap-2 text-sm">
        <CheckCircle2 size={16} style={{ color: "var(--accent)" }} />
        <span className="font-medium">
          {totalSelected} selected
        </span>
        {videoSelected !== totalSelected && (
          <span className="text-xs" style={{ color: "var(--muted)" }}>
            ({videoSelected} video{videoSelected === 1 ? "" : "s"} eligible to edit)
          </span>
        )}
      </div>
      <div className="flex items-center gap-2">
        <button
          onClick={onEdit}
          disabled={videoSelected === 0}
          className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium text-white transition-colors disabled:cursor-not-allowed disabled:opacity-50"
          style={{ background: "var(--accent)" }}
          title={videoSelected === 0 ? "Select at least one video" : "Open the AI editor scoped to these videos"}
        >
          <Sparkles size={14} />
          Edit ({videoSelected})
        </button>
        <button
          onClick={onClear}
          className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition-colors hover:opacity-80"
          style={{ borderColor: "var(--border)" }}
          title="Clear selection"
        >
          <X size={14} />
          Clear
        </button>
      </div>
    </div>
  );
}

// --- Grid View ---

function GridView({
  folders,
  files,
  selectedIds,
  onToggleSelect,
  onNavigate,
  onOpenFile,
  onFileContextMenu,
  onFolderContextMenu,
}: {
  folders: Folder[];
  files: FileRecord[];
  selectedIds: Set<string>;
  onToggleSelect: (id: string) => void;
  onNavigate: (id: string) => void;
  onOpenFile: (id: string) => void;
  onFileContextMenu?: (f: FileRecord, e: React.MouseEvent) => void;
  onFolderContextMenu?: (f: Folder, e: React.MouseEvent) => void;
}) {
  return (
    <div className="space-y-6">
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
                className="flex items-center gap-2.5 rounded-lg border p-3 text-left transition-colors hover:border-blue-300"
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
            Files
          </h3>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
            {files.map((file) => (
              <FileCard
                key={file.id}
                file={file}
                selected={selectedIds.has(file.id)}
                onToggleSelect={() => onToggleSelect(file.id)}
                onClick={() => onOpenFile(file.id)}
                onContextMenu={(e) => { e.preventDefault(); onFileContextMenu?.(file, e); }}
              />
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function FileCard({
  file,
  selected,
  onToggleSelect,
  onClick,
  onContextMenu,
}: {
  file: FileRecord;
  selected: boolean;
  onToggleSelect: () => void;
  onClick: () => void;
  onContextMenu: (e: React.MouseEvent) => void;
}) {
  const isProcessing = file.status === "processing" || file.status === "uploading";

  return (
    <div
      onContextMenu={onContextMenu}
      className="group relative flex flex-col overflow-hidden rounded-lg border text-left transition-colors hover:border-blue-300"
      style={{
        borderColor: selected ? "var(--accent)" : "var(--border)",
        boxShadow: selected ? "0 0 0 1px var(--accent)" : undefined,
      }}
    >
      {/* Selection toggle (sits on top of the thumb so it's reachable without
          opening the file). Uses stopPropagation so toggling doesn't navigate. */}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onToggleSelect();
        }}
        className="absolute left-1.5 top-1.5 z-10 rounded-full p-0.5 transition-opacity"
        style={{
          background: "rgba(0,0,0,0.5)",
          color: "white",
          opacity: selected ? 1 : 0,
        }}
        // We force-show on hover for unselected items via a class below.
        title={selected ? "Deselect" : "Select for AI Editor"}
      >
        {selected ? <CheckCircle2 size={18} /> : <Circle size={18} />}
      </button>
      {/* Force-visible-on-hover when not selected */}
      {!selected && (
        <style>{`
          .group:hover > button[title="Select for AI Editor"] { opacity: 1 !important; }
        `}</style>
      )}

      <button
        onClick={onClick}
        className="flex flex-col text-left"
      >
        <div
          className="relative flex aspect-video items-center justify-center"
          style={{ background: "var(--sidebar)" }}
        >
          {file.r2_thumbnail_key ? (
            <div className="h-full w-full bg-neutral-800" />
          ) : (
            <FileIcon type={file.file_type as "video"} size={32} />
          )}

          {isProcessing && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/30">
              <Loader2 size={20} className="animate-spin text-white" />
            </div>
          )}

          {file.file_type === "video" && file.duration_seconds && (
            <span className="absolute bottom-1.5 right-1.5 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-medium text-white">
              {formatDuration(file.duration_seconds)}
            </span>
          )}
        </div>

        <div className="flex items-start gap-1.5 p-2.5">
          <div className="min-w-0 flex-1">
            <div className="truncate text-sm">{file.name}</div>
            <div className="text-xs" style={{ color: "var(--muted)" }}>
              {formatBytes(file.file_size)}
            </div>
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); onContextMenu(e); }}
            className="shrink-0 rounded p-0.5 opacity-0 transition-opacity group-hover:opacity-100"
            style={{ color: "var(--muted)" }}
          >
            <MoreHorizontal size={14} />
          </button>
        </div>
      </button>
    </div>
  );
}

// --- List View ---

function ListView({
  folders,
  files,
  selectedIds,
  onToggleSelect,
  onNavigate,
  onOpenFile,
  onFileContextMenu,
  onFolderContextMenu,
}: {
  folders: Folder[];
  files: FileRecord[];
  selectedIds: Set<string>;
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
              className="cursor-pointer border-b transition-colors hover:bg-blue-50 dark:hover:bg-blue-950/20"
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
            return (
              <tr
                key={file.id}
                onClick={() => onOpenFile(file.id)}
                onContextMenu={(e) => { e.preventDefault(); onFileContextMenu?.(file, e); }}
                className="group cursor-pointer border-b transition-colors hover:bg-blue-50 dark:hover:bg-blue-950/20"
                style={{
                  borderColor: "var(--border)",
                  background: sel ? "rgba(59,130,246,0.06)" : undefined,
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
                  {file.status === "processing" && (
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
