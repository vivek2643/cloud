"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { useAuthStore } from "@/stores/auth-store";
import { getL1Log } from "@/lib/api";
import { ArrowLeft } from "lucide-react";

interface L1Log {
  logged_at?: string;
  file?: {
    name?: string;
    duration_seconds?: number;
    width?: number;
    height?: number;
    l1_status?: string;
  };
  summary?: Record<string, unknown>;
  transcript?: {
    language?: string;
    text?: string;
    segment_count?: number;
    filler_count?: number;
    fillers?: Array<{ word?: string; start_ms?: number; end_ms?: number }>;
    segments?: Array<Record<string, unknown>>;
  } | null;
  audio_features?: Record<string, unknown> | null;
  processing_jobs?: Array<{
    stage: string;
    status: string;
    attempts?: number;
    started_at?: string | null;
    finished_at?: string | null;
  }>;
  [k: string]: unknown;
}

export default function L1LogDetailPage() {
  const params = useParams<{ fileId: string }>();
  const session = useAuthStore((s) => s.session);
  const [log, setLog] = useState<L1Log | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!session?.access_token || !params?.fileId) return;
    getL1Log(params.fileId, session.access_token)
      .then((d) => setLog(d as L1Log))
      .catch((e) => setError((e as Error).message));
  }, [session?.access_token, params?.fileId]);

  return (
    <div className="px-8 py-8">
      <Link
        href="/logs"
        className="mb-4 inline-flex items-center gap-2 text-sm hover:underline"
        style={{ color: "var(--muted)" }}
      >
        <ArrowLeft size={14} /> Back to logs
      </Link>

      {error && (
        <div
          className="rounded-md border p-3 text-sm"
          style={{ borderColor: "#ef4444", color: "#ef4444" }}
        >
          {error}
        </div>
      )}

      {!log ? (
        <div style={{ color: "var(--muted)" }}>Loading…</div>
      ) : (
        <div className="space-y-6">
          <header>
            <h1 className="text-lg font-semibold">{log.file?.name ?? "L1 analysis"}</h1>
            <div className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
              {log.file?.width}x{log.file?.height} · {log.file?.duration_seconds?.toFixed(1)}s
              · L1: {log.file?.l1_status}
            </div>
          </header>

          {log.summary && (
            <section>
              <h2 className="mb-2 text-sm font-medium" style={{ color: "var(--muted)" }}>
                Summary
              </h2>
              <div className="grid grid-cols-3 gap-3">
                {Object.entries(log.summary).map(([k, v]) => (
                  <div
                    key={k}
                    className="rounded-md border p-3"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <div className="text-xs" style={{ color: "var(--muted)" }}>
                      {k.replaceAll("_", " ")}
                    </div>
                    <div className="mt-1 text-base font-medium tabular-nums">{String(v)}</div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {log.transcript && (
            <section>
              <h2 className="mb-2 text-sm font-medium" style={{ color: "var(--muted)" }}>
                Transcript ({log.transcript.language}, {log.transcript.segment_count} segments,{" "}
                {log.transcript.filler_count} fillers)
              </h2>
              <div
                className="rounded-md border p-3 text-sm leading-relaxed"
                style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
              >
                {log.transcript.text}
              </div>
              {log.transcript.fillers && log.transcript.fillers.length > 0 && (
                <details className="mt-2">
                  <summary
                    className="cursor-pointer text-xs"
                    style={{ color: "var(--muted)" }}
                  >
                    {log.transcript.fillers.length} fillers detected
                  </summary>
                  <pre
                    className="mt-2 max-h-64 overflow-auto rounded-md border p-2 text-xs"
                    style={{ borderColor: "var(--border)" }}
                  >
                    {JSON.stringify(log.transcript.fillers, null, 2)}
                  </pre>
                </details>
              )}
            </section>
          )}

          {log.audio_features && (
            <section>
              <h2 className="mb-2 text-sm font-medium" style={{ color: "var(--muted)" }}>
                Audio features
              </h2>
              <pre
                className="overflow-auto rounded-md border p-3 text-xs"
                style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
              >
                {JSON.stringify(log.audio_features, null, 2)}
              </pre>
            </section>
          )}

          {log.processing_jobs && log.processing_jobs.length > 0 && (
            <section>
              <h2 className="mb-2 text-sm font-medium" style={{ color: "var(--muted)" }}>
                Stage history
              </h2>
              <table className="w-full text-xs">
                <thead style={{ color: "var(--muted)" }}>
                  <tr>
                    <th className="px-3 py-2 text-left font-medium">Stage</th>
                    <th className="px-3 py-2 text-left font-medium">Status</th>
                    <th className="px-3 py-2 text-right font-medium">Attempts</th>
                    <th className="px-3 py-2 text-right font-medium">Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {log.processing_jobs.map((j) => {
                    const secs =
                      j.started_at && j.finished_at
                        ? (new Date(j.finished_at).getTime() -
                            new Date(j.started_at).getTime()) /
                          1000
                        : null;
                    return (
                    <tr
                      key={j.stage}
                      className="border-t"
                      style={{ borderColor: "var(--border)" }}
                    >
                      <td className="px-3 py-2">{j.stage}</td>
                      <td
                        className="px-3 py-2"
                        style={{
                          color:
                            j.status === "done"
                              ? "var(--accent)"
                              : j.status === "failed"
                              ? "#ef4444"
                              : "var(--muted)",
                        }}
                      >
                        {j.status}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums">{j.attempts}</td>
                      <td className="px-3 py-2 text-right tabular-nums" style={{ color: "var(--muted)" }}>
                        {secs == null ? "—" : `${secs.toFixed(1)}s`}
                      </td>
                    </tr>
                    );
                  })}
                </tbody>
              </table>
            </section>
          )}
        </div>
      )}
    </div>
  );
}
