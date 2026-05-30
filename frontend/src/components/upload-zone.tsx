"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload } from "lucide-react";
import { useDriveStore } from "@/stores/drive-store";
import { useAuthStore } from "@/stores/auth-store";
import { presignUpload, completeUpload } from "@/lib/api";

const VIDEO_ACCEPT = {
  "video/*": [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".mxf", ".mts"],
};

export function useUploadFiles() {
  const currentFolderId = useDriveStore((s) => s.currentFolderId);
  const addUpload = useDriveStore((s) => s.addUpload);
  const updateUpload = useDriveStore((s) => s.updateUpload);
  const session = useAuthStore((s) => s.session);

  const uploadFile = useCallback(
    async (file: File) => {
      const uploadId = crypto.randomUUID();
      addUpload({
        id: uploadId,
        file,
        progress: 0,
        status: "uploading",
      });

      try {
        const token = session?.access_token;
        if (!token) throw new Error("Not authenticated");

        const presign = await presignUpload(
          file.name,
          file.type || "video/mp4",
          file.size,
          currentFolderId,
          token
        );

        await new Promise<void>((resolve, reject) => {
          const xhr = new XMLHttpRequest();
          xhr.open("PUT", presign.upload_url, true);
          xhr.setRequestHeader("Content-Type", file.type || "video/mp4");

          xhr.upload.addEventListener("progress", (e) => {
            if (e.lengthComputable) {
              updateUpload(uploadId, {
                progress: Math.round((e.loaded / e.total) * 100),
              });
            }
          });

          xhr.addEventListener("load", () => {
            if (xhr.status >= 200 && xhr.status < 300) resolve();
            else reject(new Error(`Upload failed: ${xhr.status}`));
          });
          xhr.addEventListener("error", () => reject(new Error("Network error during upload")));
          xhr.addEventListener("abort", () => reject(new Error("Upload cancelled")));
          xhr.send(file);
        });

        await completeUpload(presign.file_id, token);
        updateUpload(uploadId, {
          status: "complete",
          progress: 100,
          fileId: presign.file_id,
        });
      } catch (err) {
        updateUpload(uploadId, {
          status: "error",
          error: err instanceof Error ? err.message : "Upload failed",
        });
      }
    },
    [session, currentFolderId, addUpload, updateUpload]
  );

  const uploadFiles = useCallback(
    (files: File[]) => {
      files.forEach(uploadFile);
    },
    [uploadFile]
  );

  return uploadFiles;
}

export function UploadZone({ children }: { children: React.ReactNode }) {
  const uploadFiles = useUploadFiles();

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: uploadFiles,
    accept: VIDEO_ACCEPT,
    noClick: true,
    noKeyboard: true,
  });

  return (
    <div {...getRootProps()} className="relative flex-1">
      <input {...getInputProps()} />
      {children}
      {isDragActive && (
        <div
          className="absolute inset-0 z-50 flex items-center justify-center rounded-lg border-2 border-dashed backdrop-blur-sm"
          style={{
            borderColor: "var(--accent)",
            background: "rgba(37,99,235,0.08)",
          }}
        >
          <div className="flex flex-col items-center gap-2">
            <Upload size={40} style={{ color: "var(--accent)" }} />
            <p className="text-lg font-medium" style={{ color: "var(--accent)" }}>
              Drop video files to upload
            </p>
            <p className="text-sm" style={{ color: "var(--muted)" }}>
              MP4, MOV, AVI, MKV, WebM
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
