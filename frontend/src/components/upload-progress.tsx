"use client";

import { useDriveStore } from "@/stores/drive-store";
import { X, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";
import { formatBytes } from "@/lib/utils";

export function UploadProgress() {
  const uploads = useDriveStore((s) => s.uploads);
  const removeUpload = useDriveStore((s) => s.removeUpload);

  if (uploads.length === 0) return null;

  const active = uploads.filter((u) => u.status === "uploading").length;
  const done = uploads.filter((u) => u.status === "complete").length;

  return (
    <div
      className="fixed bottom-4 right-4 z-50 w-80 overflow-hidden rounded-xl border shadow-lg"
      style={{ background: "var(--background)", borderColor: "var(--border)" }}
    >
      <div
        className="flex items-center justify-between px-4 py-2.5 text-sm font-medium"
        style={{ background: "var(--sidebar)" }}
      >
        <span>
          {active > 0
            ? `Uploading ${active} file${active > 1 ? "s" : ""}...`
            : `${done} upload${done !== 1 ? "s" : ""} complete`}
        </span>
      </div>

      <div className="max-h-60 overflow-y-auto">
        {uploads.map((upload) => (
          <div
            key={upload.id}
            className="flex items-center gap-3 border-t px-4 py-2.5"
            style={{ borderColor: "var(--border)" }}
          >
            <div className="min-w-0 flex-1">
              <div className="truncate text-sm">{upload.file.name}</div>
              <div className="text-xs" style={{ color: "var(--muted)" }}>
                {formatBytes(upload.file.size)}
              </div>
              {upload.status === "uploading" && (
                <div
                  className="mt-1 h-1 overflow-hidden rounded-full"
                  style={{ background: "var(--border)" }}
                >
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{
                      width: `${upload.progress}%`,
                      background: "var(--accent)",
                    }}
                  />
                </div>
              )}
              {upload.status === "error" && (
                <div className="mt-0.5 text-xs" style={{ color: "var(--danger)" }}>
                  {upload.error}
                </div>
              )}
            </div>

            <div className="shrink-0">
              {upload.status === "uploading" && (
                <Loader2 size={16} className="animate-spin" style={{ color: "var(--accent)" }} />
              )}
              {upload.status === "complete" && (
                <CheckCircle2 size={16} style={{ color: "var(--success)" }} />
              )}
              {upload.status === "error" && (
                <AlertCircle size={16} style={{ color: "var(--danger)" }} />
              )}
            </div>

            {upload.status !== "uploading" && (
              <button
                onClick={() => removeUpload(upload.id)}
                className="shrink-0 rounded p-0.5 transition-colors hover:opacity-70"
                style={{ color: "var(--muted)" }}
              >
                <X size={14} />
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
