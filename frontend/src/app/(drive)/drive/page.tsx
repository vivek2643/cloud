"use client";

import { useEffect, useCallback, useRef, useState } from "react";
import { useDriveStore } from "@/stores/drive-store";
import { useAuthStore } from "@/stores/auth-store";
import { getFolders, getFiles, createFolder } from "@/lib/api";
import { UploadZone, useUploadFiles } from "@/components/upload-zone";
import { DriveContent } from "@/components/drive-content";
import { CreateFolderDialog } from "@/components/create-folder-dialog";
import { FolderPlus, Upload } from "lucide-react";

const VIDEO_EXTENSIONS = ".mp4,.mov,.avi,.mkv,.webm,.m4v,.wmv,.flv,.mxf,.mts";

export default function DrivePage() {
  const session = useAuthStore((s) => s.session);
  const { setFolders, setFiles, setLoading, setCurrentFolder, uploads } = useDriveStore();
  const [showNewFolder, setShowNewFolder] = useState(false);
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

  return (
    <UploadZone>
      <div className="flex-1 p-6">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-xl font-semibold">My Drive</h1>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowNewFolder(true)}
              className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition-colors hover:opacity-80"
              style={{ borderColor: "var(--border)" }}
            >
              <FolderPlus size={16} />
              New Folder
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium text-white transition-colors"
              style={{ background: "var(--accent)" }}
            >
              <Upload size={16} />
              Upload Video
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
