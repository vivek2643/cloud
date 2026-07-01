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
} from "@/lib/api";
import { generateProxies } from "@/lib/proxy-gen";

const VIDEO_ACCEPT = {
  "video/*": [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".wmv", ".flv", ".mxf", ".mts"],
  "audio/*": [".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".oga", ".opus", ".wma", ".aiff"],
};

// R2's single presigned PUT caps at 5 GiB; route anything large through
// multipart. The threshold is well under 5 GiB so we never hit EntityTooLarge.
const MULTIPART_THRESHOLD = 100 * 1024 * 1024; // 100 MiB

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
async function armAnalysisProxies(
  fileId: string,
  file: File,
  token: string
): Promise<boolean> {
  try {
    const proxies = await generateProxies(file);
    if (!proxies) return false;
    const pres = await presignAnalysisProxies(fileId, token);
    await putBlob(pres.proxy_a_url, proxies.proxyA, "video/mp4", () => {});
    await putBlob(pres.proxy_b_url, proxies.proxyB, "video/mp4", () => {});
    await completeAnalysisProxies(fileId, token);
    return true;
  } catch (err) {
    console.warn("Analysis-proxy fast path unavailable; server will use the raw.", err);
    return false;
  }
}

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
        const contentType = file.type || "video/mp4";

        if (file.size > MULTIPART_THRESHOLD) {
          // Large file: chunked multipart upload (handles > 5 GiB).
          const mp = await createMultipartUpload(
            file.name,
            contentType,
            file.size,
            currentFolderId,
            token
          );
          try {
            // Decode + upload the analysis proxies in parallel with the raw so
            // analysis can start in seconds. Video only; audio uses the raw path.
            const proxyTask = contentType.startsWith("video/")
              ? armAnalysisProxies(mp.file_id, file, token)
              : Promise.resolve(false);

            let uploadedBytes = 0;
            for (let i = 0; i < mp.part_urls.length; i++) {
              const start = i * mp.part_size;
              const end = Math.min(start + mp.part_size, file.size);
              const blob = file.slice(start, end);
              await putBlob(mp.part_urls[i], blob, null, (loaded) => {
                updateUpload(uploadId, {
                  progress: Math.round(((uploadedBytes + loaded) / file.size) * 100),
                });
              });
              uploadedBytes += end - start;
            }
            // Let the proxy decision settle before completing the raw: if it
            // armed, raw-complete only makes the editing proxy; otherwise the
            // server runs full analysis from the raw.
            await proxyTask;
            await completeMultipartUpload(mp.file_id, mp.upload_id, token);
            updateUpload(uploadId, {
              status: "complete",
              progress: 100,
              fileId: mp.file_id,
            });
          } catch (err) {
            // Best-effort cleanup so no orphaned 'uploading' row / R2 parts linger.
            abortMultipartUpload(mp.file_id, mp.upload_id, token).catch(() => {});
            throw err;
          }
        } else {
          // Small file: single presigned PUT.
          const presign = await presignUpload(
            file.name,
            contentType,
            file.size,
            currentFolderId,
            token
          );
          await putBlob(presign.upload_url, file, contentType, (loaded) => {
            updateUpload(uploadId, {
              progress: Math.round((loaded / file.size) * 100),
            });
          });
          await completeUpload(presign.file_id, token);
          updateUpload(uploadId, {
            status: "complete",
            progress: 100,
            fileId: presign.file_id,
          });
        }
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
