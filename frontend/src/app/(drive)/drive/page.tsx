"use client";

import { useEffect, useCallback, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { useDriveStore } from "@/stores/drive-store";
import { useAuthStore } from "@/stores/auth-store";
import { getFolders, getFiles, createFolder } from "@/lib/api";
import { UploadZone } from "@/components/upload-zone";
import { DriveContent, ProjectCard } from "@/components/drive-content";
import { CreateFolderDialog } from "@/components/create-folder-dialog";
import { readRecentProjects, type RecentProject } from "@/lib/recent-projects";
import { FolderPlus, X, Search } from "lucide-react";

export default function DrivePage() {
  const router = useRouter();
  const session = useAuthStore((s) => s.session);
  const {
    folders,
    setFolders,
    setFiles,
    setLoading,
    setCurrentFolder,
    uploads,
    selectedIds,
    clearSelection,
    searchQuery,
    setSearchQuery,
    homeView,
  } = useDriveStore();
  const [showNewFolder, setShowNewFolder] = useState(false);
  const [recents, setRecents] = useState<RecentProject[]>([]);

  // Stage 4.2: read on mount only (client-only storage, avoids a hydration
  // mismatch) -- reconciled against the loaded root folder list below so a
  // deleted project never lingers here.
  useEffect(() => {
    setRecents(readRecentProjects());
  }, []);

  // Resolve each recent id back to its live Folder (dropping any that no
  // longer exist), keeping recents' own most-recent-first order rather than
  // the folder list's. Honours the search box too, so the one search bar
  // means the same thing in both lenses.
  const visibleRecents = useMemo(() => {
    const byId = new Map(folders.map((f) => [f.id, f]));
    const q = searchQuery.trim().toLowerCase();
    return recents
      .map((r) => byId.get(r.id))
      .filter((f): f is (typeof folders)[number] => !!f)
      .filter((f) => !q || f.name.toLowerCase().includes(q));
  }, [recents, folders, searchQuery]);

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
          <h1 className="text-xl font-semibold">
            {homeView === "recents" ? "Recents" : "Projects"}
          </h1>

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

        {/* Root shows projects (folders) only — lenses live inside a folder.
            Recents is the same card grid in most-recently-opened order, so it
            reuses ProjectCard rather than growing a second card style. */}
        {homeView === "recents" ? (
          visibleRecents.length > 0 ? (
            <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-5">
              {visibleRecents.map((folder) => (
                <ProjectCard
                  key={folder.id}
                  folder={folder}
                  onOpen={() => router.push(`/drive/folder/${folder.id}`)}
                />
              ))}
            </div>
          ) : (
            <p className="py-10 text-center text-sm" style={{ color: "var(--muted)" }}>
              {searchQuery.trim()
                ? "No recent projects match your search."
                : "Projects you open will show up here."}
            </p>
          )
        ) : (
          <DriveContent />
        )}
      </div>

      <CreateFolderDialog
        open={showNewFolder}
        onClose={() => setShowNewFolder(false)}
        onCreate={handleCreateFolder}
      />
    </UploadZone>
  );
}
