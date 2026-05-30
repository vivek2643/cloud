"use client";

import { useEffect, useCallback, useRef, useState } from "react";
import { useDriveStore } from "@/stores/drive-store";
import { useAuthStore } from "@/stores/auth-store";
import { getFolders, getFiles, createFolder } from "@/lib/api";
import { UploadZone, useUploadFiles } from "@/components/upload-zone";
import { DriveContent } from "@/components/drive-content";
import { CreateFolderDialog } from "@/components/create-folder-dialog";
import { FolderPlus, Upload, Search, Sparkles, Layers, Crosshair, Star, X } from "lucide-react";

const VIDEO_EXTENSIONS = ".mp4,.mov,.avi,.mkv,.webm,.m4v,.wmv,.flv,.mxf,.mts";

type Tab = "versioned" | "crux" | "highlights";

const TABS: { id: Tab; label: string; icon: typeof Layers }[] = [
  { id: "versioned", label: "Versioned", icon: Layers },
  { id: "crux", label: "Crux", icon: Crosshair },
  { id: "highlights", label: "Highlights", icon: Star },
];

export default function DrivePage() {
  const session = useAuthStore((s) => s.session);
  const {
    setFolders,
    setFiles,
    setLoading,
    setCurrentFolder,
    uploads,
    searchQuery,
    setSearchQuery,
    selectedIds,
    files,
    openAiPanel,
    aiPanelOpen,
    clearSelection,
  } = useDriveStore();
  const [showNewFolder, setShowNewFolder] = useState(false);
  const [activeTab, setActiveTab] = useState<Tab>("versioned");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadFiles = useUploadFiles();

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

  function handleFileInputChange(e: React.ChangeEvent<HTMLInputElement>) {
    const files = e.target.files;
    if (files && files.length > 0) {
      uploadFiles(Array.from(files));
      e.target.value = "";
    }
  }

  // The AI Edit button docks the conversational editor on the right. If videos
  // are selected, scope the session to them; otherwise it draws from all videos.
  function handleAiEdit() {
    const ids = Array.from(selectedIds);
    // Stash names so the panel's scope chip can render them without a refetch.
    const idToFile = new Map(files.map((f) => [f.id, f]));
    const payload = ids
      .map((id) => idToFile.get(id))
      .filter((f): f is NonNullable<typeof f> => !!f)
      .map((f) => ({ id: f.id, name: f.name, file_type: f.file_type }));
    try {
      window.sessionStorage.setItem(
        "edso_edit_scope_v1",
        JSON.stringify({ files: payload, ts: Date.now() }),
      );
    } catch {
      // non-fatal
    }
    openAiPanel(ids);
  }

  const selectedCount = selectedIds.size;

  return (
    <UploadZone>
      <div className="flex-1 p-6">
        {/* Row 1: search + AI Edit */}
        <div className="mb-4 flex items-center gap-3">
          <div className="relative flex-1">
            <Search
              size={18}
              className="absolute left-3.5 top-1/2 -translate-y-1/2"
              style={{ color: "var(--muted)" }}
            />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search your videos…"
              className="w-full rounded-xl border py-2.5 pl-11 pr-4 text-sm outline-none transition-colors focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent-soft)]"
              style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
            />
          </div>
          <button
            onClick={handleAiEdit}
            className="flex shrink-0 items-center gap-2 rounded-xl px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition-transform hover:scale-[1.02] active:scale-95"
            style={{
              background: "var(--accent)",
              boxShadow: aiPanelOpen ? "0 0 0 2px var(--accent-soft)" : undefined,
            }}
            title={selectedCount > 0 ? "Edit selected videos with AI" : "Open the AI editor"}
          >
            <Sparkles size={16} />
            AI Edit{selectedCount > 0 ? ` (${selectedCount})` : ""}
          </button>
        </div>

        {/* Row 2: tabs + library actions */}
        <div className="mb-6 flex flex-wrap items-center justify-between gap-3 border-b pb-3" style={{ borderColor: "var(--border)" }}>
          <div className="flex items-center gap-1.5">
            {TABS.map((t) => {
              const active = t.id === activeTab;
              const Icon = t.icon;
              return (
                <button
                  key={t.id}
                  onClick={() => setActiveTab(t.id)}
                  className="flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-sm font-medium transition-colors"
                  style={{
                    background: active ? "var(--accent)" : "transparent",
                    color: active ? "#fff" : "var(--muted)",
                    border: active ? "1px solid var(--accent)" : "1px solid var(--border)",
                  }}
                >
                  <Icon size={15} />
                  {t.label}
                </button>
              );
            })}
          </div>

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
              New Folder
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors hover:border-[var(--accent)]"
              style={{ borderColor: "var(--border)", color: "var(--accent)" }}
            >
              <Upload size={16} />
              Upload
            </button>
            <input
              ref={fileInputRef}
              type="file"
              multiple
              accept={VIDEO_EXTENSIONS}
              className="hidden"
              onChange={handleFileInputChange}
            />
          </div>
        </div>

        {/* Content */}
        {activeTab === "versioned" ? (
          <DriveContent />
        ) : (
          <ComingSoon tab={activeTab} />
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

function ComingSoon({ tab }: { tab: Tab }) {
  const copy =
    tab === "crux"
      ? { icon: <Crosshair size={36} style={{ color: "var(--accent)" }} />, title: "Crux", body: "The key moments across your footage will surface here." }
      : { icon: <Star size={36} style={{ color: "var(--accent)" }} />, title: "Highlights", body: "Auto-generated highlight reels will appear here." };
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      {copy.icon}
      <p className="mt-4 text-lg font-semibold">{copy.title}</p>
      <p className="mt-1 max-w-sm text-sm" style={{ color: "var(--muted)" }}>
        {copy.body}
      </p>
      <span
        className="mt-4 rounded-full px-3 py-1 text-xs font-medium"
        style={{ background: "var(--accent-soft)", color: "var(--accent)" }}
      >
        Coming soon
      </span>
    </div>
  );
}
