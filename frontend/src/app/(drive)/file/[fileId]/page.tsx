"use client";

import { useEffect, useState, use } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/auth-store";
import {
  getFile,
  getFilePlaybackUrl,
  getFileDownloadUrl,
  type FileRecord,
} from "@/lib/api";
import { FileIcon } from "@/components/file-icon";
import { formatBytes, formatDuration } from "@/lib/utils";
import { ArrowLeft, Download, Loader2, ScrollText } from "lucide-react";

export default function FilePage({ params }: { params: Promise<{ fileId: string }> }) {
  const { fileId } = use(params);
  const router = useRouter();
  const session = useAuthStore((s) => s.session);
  const [file, setFile] = useState<FileRecord | null>(null);
  const [playbackUrl, setPlaybackUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      if (!session?.access_token) return;
      try {
        const f = await getFile(fileId, session.access_token);
        setFile(f);
        if (f.file_type === "video" && (f.r2_proxy_key || f.status === "ready")) {
          const { url } = await getFilePlaybackUrl(fileId, session.access_token);
          setPlaybackUrl(url);
        }
      } catch (err) {
        console.error("Failed to load file:", err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [fileId, session?.access_token]);

  async function handleDownload() {
    if (!session?.access_token || !file) return;
    const { url } = await getFileDownloadUrl(file.id, session.access_token);
    window.open(url, "_blank");
  }

  if (loading) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <Loader2 size={24} className="animate-spin" style={{ color: "var(--muted)" }} />
      </div>
    );
  }

  if (!file) {
    return (
      <div className="flex flex-1 items-center justify-center">
        <p className="text-sm" style={{ color: "var(--muted)" }}>File not found</p>
      </div>
    );
  }

  return (
    <div className="flex flex-1 flex-col">
      {/* Header */}
      <div
        className="flex items-center gap-3 border-b px-6 py-3"
        style={{ borderColor: "var(--border)" }}
      >
        <button
          onClick={() => router.back()}
          className="rounded-lg p-1.5 transition-colors hover:opacity-70"
          style={{ color: "var(--muted)" }}
        >
          <ArrowLeft size={18} />
        </button>
        <FileIcon type={file.file_type as "video"} size={18} />
        <h1 className="flex-1 truncate text-base font-medium">{file.name}</h1>
        <button
          onClick={handleDownload}
          className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition-colors hover:opacity-80"
          style={{ borderColor: "var(--border)" }}
        >
          <Download size={14} />
          Download
        </button>
      </div>

      {/* Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Preview area */}
        <div className="flex flex-1 items-center justify-center p-6" style={{ background: "#000" }}>
          {file.file_type === "video" && playbackUrl ? (
            <video
              src={playbackUrl}
              controls
              autoPlay={false}
              className="max-h-full max-w-full rounded"
              style={{ maxHeight: "calc(100vh - 200px)" }}
            />
          ) : file.file_type === "image" ? (
            <div className="flex items-center justify-center">
              <FileIcon type="image" size={64} />
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <FileIcon type={file.file_type as "video"} size={64} />
              <p className="text-sm text-white/60">No preview available</p>
            </div>
          )}
        </div>

        {/* Details sidebar */}
        <aside
          className="w-72 shrink-0 overflow-y-auto border-l p-5"
          style={{ borderColor: "var(--border)" }}
        >
          <h2 className="text-sm font-semibold">Details</h2>
          <dl className="mt-4 space-y-3 text-sm">
            <DetailRow label="Type" value={file.file_type} />
            <DetailRow label="Size" value={formatBytes(file.file_size)} />
            {file.duration_seconds && (
              <DetailRow label="Duration" value={formatDuration(file.duration_seconds)} />
            )}
            {file.width && file.height && (
              <DetailRow label="Resolution" value={`${file.width} x ${file.height}`} />
            )}
            <DetailRow label="Status" value={file.status} />
            <DetailRow label="Uploaded" value={new Date(file.created_at).toLocaleString()} />
            <DetailRow label="Filename" value={file.filename} />
          </dl>

          {file.file_type === "video" && (
            <section className="mt-6 border-t pt-5" style={{ borderColor: "var(--border)" }}>
              <h2 className="text-sm font-semibold">AI analysis</h2>
              <dl className="mt-3 space-y-2 text-sm">
                <DetailRow
                  label="L1 (auto)"
                  value={file.l1_status ?? "pending"}
                />
                <DetailRow
                  label="L2 (deeper)"
                  value={file.l2_status ?? "not run"}
                />
              </dl>

              {file.l1_status === "ready" && (
                <Link
                  href={`/logs/l1/${file.id}`}
                  className="mt-3 flex items-center gap-2 rounded-md border px-3 py-2 text-sm transition-colors hover:opacity-80"
                  style={{ borderColor: "var(--border)" }}
                >
                  <ScrollText size={14} />
                  View L1 analysis
                </Link>
              )}
            </section>
          )}
        </aside>
      </div>
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs" style={{ color: "var(--muted)" }}>{label}</dt>
      <dd className="mt-0.5 break-all">{value}</dd>
    </div>
  );
}
