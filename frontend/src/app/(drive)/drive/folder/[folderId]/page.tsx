"use client";

import { useEffect, useCallback, useRef, useState, use } from "react";
import { useDriveStore } from "@/stores/drive-store";
import { useAuthStore } from "@/stores/auth-store";
import { getFolders, getFiles, createFolder, getBreadcrumb, type BreadcrumbItem } from "@/lib/api";
import { UploadZone, useUploadFiles } from "@/components/upload-zone";
import { DriveContent } from "@/components/drive-content";
import { Breadcrumb } from "@/components/breadcrumb";
import { CreateFolderDialog } from "@/components/create-folder-dialog";
import { SearchEditBar } from "@/components/search-edit-bar";
import { FolderPlus, Upload } from "lucide-react";

const VIDEO_EXTENSIONS = ".mp4,.mov,.avi,.mkv,.webm,.m4v,.wmv,.flv,.mxf,.mts";

export default function FolderPage({ params }: { params: Promise<{ folderId: string }> }) {
  const { folderId } = use(params);
  const session = useAuthStore((s) => s.session);
  const { setFolders, setFiles, setLoading, setCurrentFolder, uploads } = useDriveStore();
  const [breadcrumb, setBreadcrumb] = useState<BreadcrumbItem[]>([]);
  const [showNewFolder, setShowNewFolder] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadFiles = useUploadFiles();

  const loadContents = useCallback(async () => {
    if (!session?.access_token) return;
    setLoading(true);
    try {
      const [folders, files, bc] = await Promise.all([
        getFolders(folderId, session.access_token),
        getFiles(folderId, session.access_token),
        getBreadcrumb(folderId, session.access_token),
      ]);
      setFolders(folders);
      setFiles(files);
      setBreadcrumb(bc);
    } catch (err) {
      console.error("Failed to load folder contents:", err);
    } finally {
      setLoading(false);
    }
  }, [session?.access_token, folderId, setFolders, setFiles, setLoading]);

  useEffect(() => {
    setCurrentFolder(folderId);
    loadContents();
  }, [folderId, setCurrentFolder, loadContents]);

  const completedCount = uploads.filter((u) => u.status === "complete").length;
  useEffect(() => {
    if (completedCount > 0) loadContents();
  }, [completedCount, loadContents]);

  async function handleCreateFolder(name: string) {
    if (!session?.access_token) return;
    await createFolder(name, folderId, session.access_token);
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
        <SearchEditBar />
        <div className="mb-2">
          <Breadcrumb items={breadcrumb} />
        </div>
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-xl font-semibold">
            {breadcrumb.length > 0 ? breadcrumb[breadcrumb.length - 1].name : "Folder"}
          </h1>
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
