"use client";

import { useEffect, useCallback, useState } from "react";
import { useDriveStore } from "@/stores/drive-store";
import { useAuthStore } from "@/stores/auth-store";
import { getFolders, getFiles, createFolder } from "@/lib/api";
import { UploadZone } from "@/components/upload-zone";
import { DriveContent } from "@/components/drive-content";
import { CreateFolderDialog } from "@/components/create-folder-dialog";
import { FolderPlus, X, Search } from "lucide-react";

export default function DrivePage() {
  const session = useAuthStore((s) => s.session);
  const {
    setFolders,
    setFiles,
    setLoading,
    setCurrentFolder,
    uploads,
    selectedIds,
    clearSelection,
    searchQuery,
    setSearchQuery,
  } = useDriveStore();
  const [showNewFolder, setShowNewFolder] = useState(false);

  const loadContents = useCallback(async () => {
    if (!session?.access_token) return;
    setLoading(true);
    try {
      const [folders, files] = await Promise.all([
        getFolders(null, session.access_token),
        getFiles(null, session.access_token),
      ]);
      setFolders(folders);
      setFiles(files);
    } catch (err) {
      console.error("Failed to load drive contents:", err);
    } finally {
      setLoading(false);
    }
  }, [session?.access_token, setFolders, setFiles, setLoading]);

  useEffect(() => {
    setCurrentFolder(null);
    loadContents();
  }, [setCurrentFolder, loadContents]);

  const completedCount = uploads.filter((u) => u.status === "complete").length;
  useEffect(() => {
    if (completedCount > 0) loadContents();
  }, [completedCount, loadContents]);

  async function handleCreateFolder(name: string) {
    if (!session?.access_token) return;
    await createFolder(name, null, session.access_token);
    loadContents();
  }

  const selectedCount = selectedIds.size;

  return (
    <UploadZone>
      <div className="flex-1 p-6">
        {/* Centered project search */}
        <div className="mx-auto mb-8 w-full max-w-2xl">
          <div className="relative">
            <Search
              size={18}
              className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2"
              style={{ color: "var(--muted)" }}
            />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search projects"
              className="w-full rounded-full border bg-transparent py-2.5 pl-11 pr-10 text-sm outline-none transition-colors focus:border-[var(--accent)]"
              style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
            />
            {searchQuery && (
              <button
                onClick={() => setSearchQuery("")}
                className="absolute right-3 top-1/2 -translate-y-1/2 rounded p-0.5 transition-opacity hover:opacity-70"
                style={{ color: "var(--muted)" }}
                title="Clear search"
              >
                <X size={15} />
              </button>
            )}
          </div>
        </div>

        {/* Library actions (projects live in folders; lenses appear inside one) */}
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3 border-b pb-3" style={{ borderColor: "var(--border)" }}>
          <h1 className="text-xl font-semibold">Projects</h1>

          <div className="flex items-center gap-2">
            {selectedCount > 0 && (
              <button
                onClick={clearSelection}
                className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition-colors hover:border-[var(--accent)]"
                style={{ borderColor: "var(--border)" }}
                title="Clear selection"
              >
                <X size={16} />
                Clear ({selectedCount})
              </button>
            )}
            <button
              onClick={() => setShowNewFolder(true)}
              className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition-colors hover:border-[var(--accent)]"
              style={{ borderColor: "var(--border)" }}
            >
              <FolderPlus size={16} />
              New Project
            </button>
          </div>
        </div>

        {/* Root shows projects (folders) only — lenses live inside a folder. */}
        <DriveContent />
      </div>

      <CreateFolderDialog
        open={showNewFolder}
        onClose={() => setShowNewFolder(false)}
        onCreate={handleCreateFolder}
      />
    </UploadZone>
  );
}
