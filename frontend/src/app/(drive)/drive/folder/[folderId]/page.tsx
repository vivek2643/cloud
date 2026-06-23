"use client";

import { useEffect, useCallback, useRef, useState, use } from "react";
import { useDriveStore } from "@/stores/drive-store";
import { useAuthStore } from "@/stores/auth-store";
import { getFolders, getFiles, getBreadcrumb, type BreadcrumbItem } from "@/lib/api";
import { UploadZone, useUploadFiles } from "@/components/upload-zone";
import { ProjectLenses } from "@/components/project-lenses";
import { SearchEditBar } from "@/components/search-edit-bar";
import { Upload, Link2, Share2 } from "lucide-react";

const VIDEO_EXTENSIONS =
  ".mp4,.mov,.avi,.mkv,.webm,.m4v,.wmv,.flv,.mxf,.mts," +
  ".mp3,.wav,.m4a,.aac,.flac,.ogg,.oga,.opus,.wma,.aiff";

export default function FolderPage({ params }: { params: Promise<{ folderId: string }> }) {
  const { folderId } = use(params);
  const session = useAuthStore((s) => s.session);
  const { setFolders, setFiles, setLoading, setCurrentFolder, setProjectStage, projectStage, uploads } =
    useDriveStore();
  const [breadcrumb, setBreadcrumb] = useState<BreadcrumbItem[]>([]);
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
    setProjectStage("media");
    loadContents();
  }, [folderId, setCurrentFolder, setProjectStage, loadContents]);

  const completedCount = uploads.filter((u) => u.status === "complete").length;
  useEffect(() => {
    if (completedCount > 0) loadContents();
  }, [completedCount, loadContents]);

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
        {projectStage === "media" && (
        <>
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-xl font-semibold">
            {breadcrumb.length > 0 ? breadcrumb[breadcrumb.length - 1].name : "Project"}
          </h1>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition-colors hover:opacity-80"
              style={{ borderColor: "var(--border)" }}
            >
              <Share2 size={16} />
              Share
            </button>
            <button
              type="button"
              className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition-colors hover:opacity-80"
              style={{ borderColor: "var(--border)" }}
            >
              <Link2 size={16} />
              Upload Link
            </button>
            <button
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition-colors hover:opacity-80"
              style={{ borderColor: "var(--border)" }}
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

        <SearchEditBar />
        </>
        )}

        <ProjectLenses />
      </div>
    </UploadZone>
  );
}
