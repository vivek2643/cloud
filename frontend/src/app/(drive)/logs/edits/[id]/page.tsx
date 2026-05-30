"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { useAuthStore } from "@/stores/auth-store";
import { getEditLog } from "@/lib/api";
import { ArrowLeft, AlertTriangle, CheckCircle2 } from "lucide-react";

interface EditLog {
  logged_at?: string;
  prompt?: string;
  status?: string;
  error?: string;
  user_id?: string;
  folder_id?: string | null;
  needs_l2_resolved?: boolean;
  fcp7_xml_chars?: number;
  stages?: {
    query?: Record<string, unknown>;
    candidates_l1_only?: Array<Record<string, unknown>>;
    candidates_after_l2?: Array<Record<string, unknown>>;
    timeline?: Array<Record<string, unknown>>;
    timeline_summary?: {
      clip_count: number;
      actual_duration_ms: number;
      actual_duration_s: number;
      target_duration_s: number | null;
      delta_ms_vs_target: number | null;
      delta_pct_vs_target: number | null;
    };
  };
  [k: string]: unknown;
}

export default function EditLogDetailPage() {
  const params = useParams<{ id: string }>();
  const session = useAuthStore((s) => s.session);
  const [log, setLog] = useState<EditLog | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!session?.access_token || !params?.id) return;
    getEditLog(params.id, session.access_token)
      .then((d) => setLog(d as EditLog))
      .catch((e) => setError((e as Error).message));
  }, [session?.access_token, params?.id]);

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
            <div className="flex items-center gap-2">
              {log.status === "ok" ? (
                <CheckCircle2 size={18} style={{ color: "var(--accent)" }} />
              ) : (
                <AlertTriangle size={18} style={{ color: "#f59e0b" }} />
              )}
              <h1 className="text-lg font-semibold">
                {log.status === "ok" ? "Edit succeeded" : `Edit ${log.status}`}
              </h1>
            </div>
            <div className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
              {log.logged_at}
            </div>
          </header>

          <section>
            <h2 className="mb-2 text-sm font-medium" style={{ color: "var(--muted)" }}>
              Prompt
            </h2>
            <div
              className="rounded-md border p-3 text-sm"
              style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
            >
              {log.prompt}
            </div>
          </section>

          {log.error && (
            <section>
              <h2 className="mb-2 text-sm font-medium" style={{ color: "#ef4444" }}>
                Error
              </h2>
              <pre
                className="overflow-x-auto rounded-md border p-3 text-xs"
                style={{ borderColor: "#ef4444", color: "#ef4444" }}
              >
                {log.error}
              </pre>
            </section>
          )}

          {log.stages?.timeline_summary && (
            <section>
              <h2 className="mb-2 text-sm font-medium" style={{ color: "var(--muted)" }}>
                Length check
              </h2>
              <div className="grid grid-cols-4 gap-3">
                {[
                  ["Target", `${log.stages.timeline_summary.target_duration_s ?? "—"}s`],
                  ["Actual", `${log.stages.timeline_summary.actual_duration_s ?? "—"}s`],
                  [
                    "Delta",
                    log.stages.timeline_summary.delta_ms_vs_target == null
                      ? "—"
                      : `${(log.stages.timeline_summary.delta_ms_vs_target / 1000).toFixed(2)}s`,
                  ],
                  ["Clips", String(log.stages.timeline_summary.clip_count)],
                ].map(([label, value]) => (
                  <div
                    key={label}
                    className="rounded-md border p-3"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <div className="text-xs" style={{ color: "var(--muted)" }}>
                      {label}
                    </div>
                    <div className="mt-1 text-base font-medium tabular-nums">{value}</div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {log.stages?.query && (
            <section>
              <h2 className="mb-2 text-sm font-medium" style={{ color: "var(--muted)" }}>
                Parsed structured query (Claude → JSON)
              </h2>
              <pre
                className="overflow-x-auto rounded-md border p-3 text-xs"
                style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
              >
                {JSON.stringify(log.stages.query, null, 2)}
              </pre>
            </section>
          )}

          {log.stages?.timeline && (
            <section>
              <h2 className="mb-2 text-sm font-medium" style={{ color: "var(--muted)" }}>
                Timeline ({log.stages.timeline.length} clips)
              </h2>
              <div
                className="overflow-hidden rounded-md border"
                style={{ borderColor: "var(--border)" }}
              >
                <table className="w-full text-xs">
                  <thead style={{ background: "var(--sidebar)", color: "var(--muted)" }}>
                    <tr>
                      <th className="px-3 py-2 text-left font-medium">#</th>
                      <th className="px-3 py-2 text-left font-medium">File</th>
                      <th className="px-3 py-2 text-right font-medium">Source in</th>
                      <th className="px-3 py-2 text-right font-medium">Source out</th>
                      <th className="px-3 py-2 text-right font-medium">Length</th>
                      <th className="px-3 py-2 text-right font-medium">Score</th>
                    </tr>
                  </thead>
                  <tbody>
                    {log.stages.timeline.map((t, i) => (
                      <tr
                        key={i}
                        className="border-t"
                        style={{ borderColor: "var(--border)" }}
                      >
                        <td className="px-3 py-2 tabular-nums">{i + 1}</td>
                        <td className="px-3 py-2">{String(t.file_name ?? "")}</td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {Number(t.source_in_ms ?? 0)} ms
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {Number(t.source_out_ms ?? 0)} ms
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {Number(t.duration_ms ?? 0)} ms
                        </td>
                        <td className="px-3 py-2 text-right tabular-nums">
                          {(Number(t.score ?? 0)).toFixed(3)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {log.stages?.candidates_l1_only && (
            <section>
              <h2 className="mb-2 text-sm font-medium" style={{ color: "var(--muted)" }}>
                Top candidates considered (top {log.stages.candidates_l1_only.length})
              </h2>
              <pre
                className="max-h-96 overflow-auto rounded-md border p-3 text-xs"
                style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
              >
                {JSON.stringify(log.stages.candidates_l1_only, null, 2)}
              </pre>
            </section>
          )}

          <section>
            <h2 className="mb-2 text-sm font-medium" style={{ color: "var(--muted)" }}>
              Full raw log
            </h2>
            <pre
              className="max-h-[600px] overflow-auto rounded-md border p-3 text-xs"
              style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
            >
              {JSON.stringify(log, null, 2)}
            </pre>
          </section>
        </div>
      )}
    </div>
  );
}
