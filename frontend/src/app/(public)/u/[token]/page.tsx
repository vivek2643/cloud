"use client";

import { use, useCallback, useEffect, useMemo, useState } from "react";
import { useDropzone } from "react-dropzone";
import { Upload, CheckCircle2, XCircle, Loader2, FileVideo } from "lucide-react";
import {
  getPublicUploadLinkInfo,
  publicPresignUpload,
  publicCompleteUpload,
  publicCreateMultipartUpload,
  publicCompleteMultipartUpload,
  publicAbortMultipartUpload,
  publicPresignAnalysisProxies,
  publicCompleteAnalysisProxies,
  type PublicUploadLinkInfo,
} from "@/lib/api";
import {
  useUploadFiles,
  VIDEO_ACCEPT,
  type UploadApiBinding,
  type UploadItemStatus,
} from "@/components/upload-zone";
import { formatBytes } from "@/lib/utils";

interface UploadItem {
  id: string;
  file: File;
  progress: number;
  status: UploadItemStatus;
  error?: string;
}

function CenteredMessage({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-4 text-center">
      <p className="text-sm" style={{ color: "var(--muted)" }}>
        {children}
      </p>
    </div>
  );
}

export default function PublicUploadPage({ params }: { params: Promise<{ token: string }> }) {
  const { token } = use(params);
  const [info, setInfo] = useState<PublicUploadLinkInfo | null>(null);
  const [inactiveReason, setInactiveReason] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [uploads, setUploads] = useState<UploadItem[]>([]);

  useEffect(() => {
    let cancelled = false;
    getPublicUploadLinkInfo(token)
      .then((res) => {
        if (!cancelled) setInfo(res);
      })
      .catch((e) => {
        if (!cancelled) {
          setInactiveReason(e instanceof Error ? e.message : "This link is no longer active.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [token]);

  // frontend_project_ux.plan.md Stage 2.5: the SAME upload orchestration
  // (useUploadFiles) the authenticated drive UI uses, bound to this link's
  // token instead of a session token, with its own local progress state
  // instead of the shared drive store.
  const api: UploadApiBinding = useMemo(
    () => ({
      presignUpload: (filename, contentType, fileSize) =>
        publicPresignUpload(token, filename, contentType, fileSize),
      completeUpload: (fileId) => publicCompleteUpload(token, fileId),
      createMultipartUpload: (filename, contentType, fileSize) =>
        publicCreateMultipartUpload(token, filename, contentType, fileSize),
      completeMultipartUpload: (fileId, uploadId) =>
        publicCompleteMultipartUpload(token, fileId, uploadId),
      abortMultipartUpload: (fileId, uploadId) => publicAbortMultipartUpload(token, fileId, uploadId),
      presignAnalysisProxies: (fileId) => publicPresignAnalysisProxies(token, fileId),
      completeAnalysisProxies: (fileId) => publicCompleteAnalysisProxies(token, fileId),
    }),
    [token]
  );

  const onAdd = useCallback((item: { id: string; file: File; progress: number; status: UploadItemStatus }) => {
    setUploads((prev) => [...prev, item]);
  }, []);
  const onUpdate = useCallback(
    (id: string, patch: Partial<{ progress: number; status: UploadItemStatus; error: string }>) => {
      setUploads((prev) => prev.map((u) => (u.id === id ? { ...u, ...patch } : u)));
    },
    []
  );
  const callbacks = useMemo(() => ({ onAdd, onUpdate }), [onAdd, onUpdate]);

  const uploadFiles = useUploadFiles(api, callbacks);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: uploadFiles,
    accept: VIDEO_ACCEPT,
    disabled: !info,
  });

  if (loading) {
    return <CenteredMessage>Loading…</CenteredMessage>;
  }

  if (!info) {
    return <CenteredMessage>{inactiveReason ?? "This link is no longer active."}</CenteredMessage>;
  }

  return (
    <div className="mx-auto flex min-h-screen max-w-lg flex-col justify-center px-6 py-16">
      <div className="mb-8 text-center">
        <h1 className="text-xl font-semibold">{info.project_name}</h1>
        <p className="mt-1.5 text-sm" style={{ color: "var(--muted)" }}>
          Upload files to this project. No account needed.
          {info.remaining != null && (
            <>
              {" "}
              {info.remaining} upload{info.remaining === 1 ? "" : "s"} remaining.
            </>
          )}
        </p>
      </div>

      <div
        {...getRootProps()}
        className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-14 text-center transition-colors"
        style={{
          borderColor: isDragActive ? "var(--accent)" : "var(--border)",
          background: isDragActive ? "var(--accent-soft)" : "var(--sidebar)",
        }}
      >
        <input {...getInputProps()} />
        <Upload size={32} style={{ color: isDragActive ? "var(--accent)" : "var(--muted)" }} />
        <p className="text-sm font-medium">
          {isDragActive ? "Drop to upload" : "Drag files here, or click to browse"}
        </p>
        <p className="text-xs" style={{ color: "var(--muted)" }}>
          MP4, MOV, AVI, MKV, WebM, and audio files
        </p>
      </div>

      {uploads.length > 0 && (
        <ul className="mt-6 space-y-2">
          {uploads.map((u) => (
            <li
              key={u.id}
              className="flex items-center gap-2.5 rounded-lg border px-3 py-2.5"
              style={{ borderColor: "var(--border)" }}
            >
              <FileVideo size={16} className="shrink-0" style={{ color: "var(--muted)" }} />
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm">{u.file.name}</div>
                <div className="mt-0.5 text-[11px]" style={{ color: "var(--muted)" }}>
                  {u.status === "error"
                    ? u.error || "Upload failed"
                    : u.status === "complete"
                      ? formatBytes(u.file.size)
                      : `${u.progress}%`}
                </div>
                {u.status === "uploading" && (
                  <div className="mt-1.5 h-1 overflow-hidden rounded-full" style={{ background: "var(--border)" }}>
                    <div
                      className="h-full rounded-full transition-[width] duration-300"
                      style={{ width: `${u.progress}%`, background: "var(--accent)" }}
                    />
                  </div>
                )}
              </div>
              {u.status === "complete" && (
                <CheckCircle2 size={16} className="shrink-0" style={{ color: "var(--success)" }} />
              )}
              {u.status === "error" && (
                <XCircle size={16} className="shrink-0" style={{ color: "var(--danger)" }} />
              )}
              {u.status === "uploading" && (
                <Loader2 size={16} className="shrink-0 animate-spin" style={{ color: "var(--accent)" }} />
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
