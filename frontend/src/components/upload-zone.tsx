"use client";

import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { Upload } from "lucide-react";
import { useDriveStore } from "@/stores/drive-store";
import { useAuthStore } from "@/stores/auth-store";
import {
  presignUpload,
  completeUpload,
  createMultipartUpload,
  completeMultipartUpload,
  abortMultipartUpload,
  presignAnalysisProxies,
  completeAnalysisProxies,
  type PresignResponse,
  type MultipartCreateResponse,
  type AnalysisProxyPresignResponse,
} from "@/lib/api";
import { generateProxies } from "@/lib/proxy-gen";

export const VIDEO_ACCEPT = {
  "video/*": [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".mxf", ".mts"],
  "audio/*": [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aiff"],
};

// R2's single presigned PUT caps at 5 GiB; route anything large through
// multipart. The threshold is well under 5 GiB so we never hit EntityTooLarge.
const MULTIPART_THRESHOLD = 100 * 1024 * 1024; // 100 MiB

// frontend_project_ux.plan.md Stage 2.5: the whole upload lifecycle
// (single-PUT, multipart, analysis proxies) parameterized over an API
// binding + progress callbacks, so the authenticated drive UI and the
// anonymous public upload-link page share this ONE implementation instead
// of two copies that would silently drift (see upload.py's own 2.2 refactor
// for the backend half of this same principle).
export interface UploadApiBinding {
  presignUpload: (filename: string, contentType: string, fileSize: number) => Promise<PresignResponse>;
  completeUpload: (fileId: string) => Promise<unknown>;
  createMultipartUpload: (
    filename: string, contentType: string, fileSize: number
  ) => Promise<MultipartCreateResponse>;
  completeMultipartUpload: (fileId: string, uploadId: string) => Promise<unknown>;
  abortMultipartUpload: (fileId: string, uploadId: string) => Promise<unknown>;
  presignAnalysisProxies: (fileId: string) => Promise<AnalysisProxyPresignResponse>;
  completeAnalysisProxies: (fileId: string) => Promise<unknown>;
}

export type UploadItemStatus = "pending" | "uploading" | "complete" | "error";

export interface UploadProgressCallbacks {
  onAdd: (item: { id: string; file: File; progress: number; status: UploadItemStatus }) => void;
  onUpdate: (
    id: string,
    patch: Partial<{ progress: number; status: UploadItemStatus; error: string; fileId: string }>
  ) => void;
}

/** PUT a blob with upload progress. Pass contentType only when it was signed
 * (single-PUT); multipart part URLs are signed without a content-type. */
function putBlob(
  url: string,
  body: Blob,
  contentType: string | null,
  onProgress: (loadedBytes: number) => void
): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", url, true);
    if (contentType) xhr.setRequestHeader("Content-Type", contentType);
    xhr.upload.addEventListener("progress", (e) => {
      if (e.lengthComputable) onProgress(e.loaded);
    });
    xhr.addEventListener("load", () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else reject(new Error(`Upload failed: ${xhr.status}`));
    });
    xhr.addEventListener("error", () => reject(new Error("Network error during upload")));
    xhr.addEventListener("abort", () => reject(new Error("Upload cancelled")));
    xhr.send(body);
  });
}

/**
 * Best-effort: decode the local video once, upload the two tiny analysis
 * proxies, and register them so the server fires analysis immediately -- without
 * waiting on the multi-GB raw (see client_proxy.plan.md). Resolves to true when
 * the fast path was armed, false when we couldn't (any failure), in which case
 * the raw's own /complete triggers full analysis from the raw as before. This
 * NEVER throws, so it can't break the raw upload it runs alongside.
 */
async function armAnalysisProxies(api: UploadApiBinding, fileId: string, file: File): Promise<boolean> {
  try {
    const proxies = await generateProxies(file);
    if (!proxies) return false;
    const pres = await api.presignAnalysisProxies(fileId);
    await putBlob(pres.proxy_a_url, proxies.proxyA, "video/mp4", () => {});
    await putBlob(pres.proxy_b_url, proxies.proxyB, "video/mp4", () => {});
    await api.completeAnalysisProxies(fileId);
    return true;
  } catch (err) {
    console.warn("Analysis-proxy fast path unavailable; server will use the raw.", err);
    return false;
  }
}

/** The reusable upload orchestrator: given an API binding (which knows how
 * to reach either the authenticated /api/upload/* routes or a specific
 * upload link's /api/public/upload-links/{token}/* routes) and progress
 * callbacks, returns a function that uploads a batch of Files. */
export function useUploadFiles(api: UploadApiBinding, callbacks: UploadProgressCallbacks) {
  const uploadFile = useCallback(
    async (file: File) => {
      const uploadId = crypto.randomUUID();
      callbacks.onAdd({ id: uploadId, file, progress: 0, status: "uploading" });

      try {
        const contentType = file.type || "video/mp4";

        if (file.size > MULTIPART_THRESHOLD) {
          // Large file: chunked multipart upload (handles > 5 GiB).
          const mp = await api.createMultipartUpload(file.name, contentType, file.size);
          try {
            // Decode + upload the analysis proxies in parallel with the raw so
            // analysis can start in seconds. Video only; audio uses the raw path.
            const proxyTask = contentType.startsWith("video/")
              ? armAnalysisProxies(api, mp.file_id, file)
              : Promise.resolve(false);

            let uploadedBytes = 0;
            for (let i = 0; i < mp.part_urls.length; i++) {
              const start = i * mp.part_size;
              const end = Math.min(start + mp.part_size, file.size);
              const blob = file.slice(start, end);
              await putBlob(mp.part_urls[i], blob, null, (loaded) => {
                callbacks.onUpdate(uploadId, {
                  progress: Math.round(((uploadedBytes + loaded) / file.size) * 100),
                });
              });
              uploadedBytes += end - start;
            }
            // Let the proxy decision settle before completing the raw: if it
            // armed, raw-complete only makes the editing proxy; otherwise the
            // server runs full analysis from the raw.
            await proxyTask;
            await api.completeMultipartUpload(mp.file_id, mp.upload_id);
            callbacks.onUpdate(uploadId, { status: "complete", progress: 100, fileId: mp.file_id });
          } catch (err) {
            // Best-effort cleanup so no orphaned 'uploading' row / R2 parts linger.
            api.abortMultipartUpload(mp.file_id, mp.upload_id).catch(() => {});
            throw err;
          }
        } else {
          // Small file: single presigned PUT.
          const presign = await api.presignUpload(file.name, contentType, file.size);
          await putBlob(presign.upload_url, file, contentType, (loaded) => {
            callbacks.onUpdate(uploadId, { progress: Math.round((loaded / file.size) * 100) });
          });
          await api.completeUpload(presign.file_id);
          callbacks.onUpdate(uploadId, { status: "complete", progress: 100, fileId: presign.file_id });
        }
      } catch (err) {
        callbacks.onUpdate(uploadId, {
          status: "error",
          error: err instanceof Error ? err.message : "Upload failed",
        });
      }
    },
    [api, callbacks]
  );

  const uploadFiles = useCallback(
    (files: File[]) => {
      files.forEach(uploadFile);
    },
    [uploadFile]
  );

  return uploadFiles;
}

/** The authenticated binding: the existing /api/upload/* routes, bound to
 * the current session token and (for the two file-creating calls) the
 * currently-open folder. Used by the drive UI's UploadZone/folder page --
 * the ONLY thing that changed for them is this binding is now explicit
 * instead of hard-coded inside useUploadFiles itself. */
export function useAuthenticatedUploadApi(folderId: string | null): UploadApiBinding | null {
  const token = useAuthStore((s) => s.session?.access_token);
  if (!token) return null;
  return {
    presignUpload: (filename, contentType, fileSize) =>
      presignUpload(filename, contentType, fileSize, folderId, token),
    completeUpload: (fileId) => completeUpload(fileId, token),
    createMultipartUpload: (filename, contentType, fileSize) =>
      createMultipartUpload(filename, contentType, fileSize, folderId, token),
    completeMultipartUpload: (fileId, uploadId) => completeMultipartUpload(fileId, uploadId, token),
    abortMultipartUpload: (fileId, uploadId) => abortMultipartUpload(fileId, uploadId, token),
    presignAnalysisProxies: (fileId) => presignAnalysisProxies(fileId, token),
    completeAnalysisProxies: (fileId) => completeAnalysisProxies(fileId, token),
  };
}

// A stable (module-level, never recreated) binding for the brief window
// before a session exists. Every call rejects; useUploadFiles' own
// try/catch already turns that into a normal per-item "error" status with
// the message below, so there's no separate no-auth code path to keep in
// sync with the real one.
const NOT_AUTHENTICATED_ERROR = "Not authenticated";
const NOT_AUTHENTICATED_API: UploadApiBinding = {
  presignUpload: () => Promise.reject(new Error(NOT_AUTHENTICATED_ERROR)),
  completeUpload: () => Promise.reject(new Error(NOT_AUTHENTICATED_ERROR)),
  createMultipartUpload: () => Promise.reject(new Error(NOT_AUTHENTICATED_ERROR)),
  completeMultipartUpload: () => Promise.reject(new Error(NOT_AUTHENTICATED_ERROR)),
  abortMultipartUpload: () => Promise.reject(new Error(NOT_AUTHENTICATED_ERROR)),
  presignAnalysisProxies: () => Promise.reject(new Error(NOT_AUTHENTICATED_ERROR)),
  completeAnalysisProxies: () => Promise.reject(new Error(NOT_AUTHENTICATED_ERROR)),
};

/** Convenience wrapper matching the pre-Stage-2.5 zero-arg call shape, for
 * the two authenticated call sites (UploadZone, the folder page's manual
 * file input) -- reads the current folder + session and reports progress
 * into the shared drive store, exactly as useUploadFiles() used to do
 * internally. The public upload page does NOT use this: it supplies its
 * own link-scoped API binding and local progress state instead. */
export function useAuthenticatedUploadFiles() {
  const currentFolderId = useDriveStore((s) => s.currentFolderId);
  const addUpload = useDriveStore((s) => s.addUpload);
  const updateUpload = useDriveStore((s) => s.updateUpload);
  const api = useAuthenticatedUploadApi(currentFolderId);
  const callbacks: UploadProgressCallbacks = { onAdd: addUpload, onUpdate: updateUpload };
  return useUploadFiles(api ?? NOT_AUTHENTICATED_API, callbacks);
}

export function UploadZone({ children }: { children: React.ReactNode }) {
  const uploadFiles = useAuthenticatedUploadFiles();

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
