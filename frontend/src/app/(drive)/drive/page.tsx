"use client";

import { useEffect, useCallback, useRef, useState } from "react";
import { useDriveStore } from "@/stores/drive-store";
import { useAuthStore } from "@/stores/auth-store";
import { getFolders, getFiles, createFolder } from "@/lib/api";
import { UploadZone, useUploadFiles } from "@/components/upload-zone";
import { DriveContent } from "@/components/drive-content";
import { CreateFolderDialog } from "@/components/create-folder-dialog";
import { FolderPlus, Upload, Layers, Crosshair, Star, X } from "lucide-react";

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
    selectedIds,
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

  const selectedCount = selectedIds.size;

  return (
    <UploadZone>
      <div className="flex-1 p-6">
        {/* Tabs + library actions */}
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
