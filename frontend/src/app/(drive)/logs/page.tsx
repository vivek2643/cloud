"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useAuthStore } from "@/stores/auth-store";
import {
  listEditLogs,
  listL1Logs,
  type EditLogListItem,
  type L1LogListItem,
} from "@/lib/api";
import { ScrollText, FileVideo, Sparkles, RefreshCw } from "lucide-react";

type Tab = "edits" | "l1";

function formatBytes(b: number) {
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / 1024 / 1024).toFixed(1)} MB`;
}

function statusColor(status: string) {
  switch (status) {
    case "ok":
      return "var(--accent)";
    case "failed":
      return "#ef4444";
    case "started":
      return "#f59e0b";
    default:
      return "var(--muted)";
  }
}

export default function LogsPage() {
  const session = useAuthStore((s) => s.session);
  const [tab, setTab] = useState<Tab>("edits");
  const [edits, setEdits] = useState<EditLogListItem[]>([]);
  const [l1, setL1] = useState<L1LogListItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    if (!session?.access_token) return;
    setLoading(true);
    setError(null);
    try {
      const [e, l] = await Promise.all([
        listEditLogs(session.access_token),
        listL1Logs(session.access_token),
      ]);
      setEdits(e.items);
      setL1(l.items);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.access_token]);

  return (
    <div className="px-8 py-8">
      <div className="mb-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <ScrollText size={22} />
          <h1 className="text-xl font-semibold">Audit logs</h1>
        </div>
        <button
          onClick={refresh}
          disabled={loading}
          className="flex items-center gap-2 rounded-md px-3 py-1.5 text-sm transition-colors"
          style={{ background: "var(--border)" }}
        >
          <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      <p className="mb-6 text-sm" style={{ color: "var(--muted)" }}>
        Every L1 indexing run and every AI edit request is written to{" "}
        <code className="rounded px-1.5 py-0.5 text-xs" style={{ background: "var(--border)" }}>
          backend/logs/
        </code>{" "}
        as a JSON file. Files survive server restarts so you can always inspect what the
        system thought of your video and how it built each timeline.
      </p>

      <div className="mb-4 flex gap-1 border-b" style={{ borderColor: "var(--border)" }}>
        <button
          onClick={() => setTab("edits")}
          className="flex items-center gap-2 px-4 py-2 text-sm transition-colors"
          style={{
            borderBottom: tab === "edits" ? "2px solid var(--accent)" : "2px solid transparent",
            color: tab === "edits" ? "var(--foreground)" : "var(--muted)",
            fontWeight: tab === "edits" ? 600 : 400,
          }}
        >
          <Sparkles size={14} />
          Edit requests ({edits.length})
        </button>
        <button
          onClick={() => setTab("l1")}
          className="flex items-center gap-2 px-4 py-2 text-sm transition-colors"
          style={{
            borderBottom: tab === "l1" ? "2px solid var(--accent)" : "2px solid transparent",
            color: tab === "l1" ? "var(--foreground)" : "var(--muted)",
            fontWeight: tab === "l1" ? 600 : 400,
          }}
        >
          <FileVideo size={14} />
          L1 analyses ({l1.length})
        </button>
      </div>

      {error && (
        <div
          className="mb-4 rounded-md border p-3 text-sm"
          style={{ borderColor: "#ef4444", color: "#ef4444" }}
        >
          {error}
        </div>
      )}

      {tab === "edits" && (
        <div className="overflow-hidden rounded-lg border" style={{ borderColor: "var(--border)" }}>
          {edits.length === 0 ? (
            <div className="px-4 py-8 text-center text-sm" style={{ color: "var(--muted)" }}>
              No edit requests yet. Try one on the AI Rough Cut page.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead style={{ background: "var(--sidebar)", color: "var(--muted)" }}>
                <tr>
                  <th className="px-4 py-2 text-left font-medium">Status</th>
                  <th className="px-4 py-2 text-left font-medium">Prompt</th>
                  <th className="px-4 py-2 text-right font-medium">Target</th>
                  <th className="px-4 py-2 text-right font-medium">Actual</th>
                  <th className="px-4 py-2 text-right font-medium">Δ</th>
                  <th className="px-4 py-2 text-left font-medium">Time</th>
                </tr>
              </thead>
              <tbody>
                {edits.map((e) => {
                  const delta =
                    e.actual_duration_s != null && e.duration_target_s != null
                      ? +(e.actual_duration_s - e.duration_target_s).toFixed(1)
                      : null;
                  return (
                    <tr
                      key={e.id}
                      className="border-t hover:bg-[var(--sidebar)]"
                      style={{ borderColor: "var(--border)" }}
                    >
                      <td className="px-4 py-2">
                        <span
                          className="rounded-full px-2 py-0.5 text-xs"
                          style={{ background: "var(--border)", color: statusColor(e.status) }}
                        >
                          {e.status}
                        </span>
                      </td>
                      <td className="px-4 py-2">
                        <Link
                          href={`/logs/edits/${e.id}`}
                          className="hover:underline"
                          style={{ color: "var(--foreground)" }}
                        >
                          {e.prompt.length > 80 ? e.prompt.slice(0, 80) + "…" : e.prompt}
                        </Link>
                      </td>
                      <td className="px-4 py-2 text-right tabular-nums" style={{ color: "var(--muted)" }}>
                        {e.duration_target_s != null ? `${e.duration_target_s}s` : "—"}
                      </td>
                      <td className="px-4 py-2 text-right tabular-nums" style={{ color: "var(--muted)" }}>
                        {e.actual_duration_s != null ? `${e.actual_duration_s}s` : "—"}
                      </td>
                      <td
                        className="px-4 py-2 text-right tabular-nums"
                        style={{
                          color:
                            delta == null
                              ? "var(--muted)"
                              : Math.abs(delta) <= 0.5
                              ? "var(--accent)"
                              : "#f59e0b",
                        }}
                      >
                        {delta == null ? "—" : delta > 0 ? `+${delta}s` : `${delta}s`}
                      </td>
                      <td className="px-4 py-2 text-xs" style={{ color: "var(--muted)" }}>
                        {new Date(e.logged_at).toLocaleString()}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}

      {tab === "l1" && (
        <div className="overflow-hidden rounded-lg border" style={{ borderColor: "var(--border)" }}>
          {l1.length === 0 ? (
            <div className="px-4 py-8 text-center text-sm" style={{ color: "var(--muted)" }}>
              No L1 analyses yet. Upload a video to trigger one.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead style={{ background: "var(--sidebar)", color: "var(--muted)" }}>
                <tr>
                  <th className="px-4 py-2 text-left font-medium">File ID</th>
                  <th className="px-4 py-2 text-right font-medium">Size</th>
                  <th className="px-4 py-2 text-left font-medium">Last analyzed</th>
                </tr>
              </thead>
              <tbody>
                {l1.map((item) => (
                  <tr
                    key={item.file_id}
                    className="border-t hover:bg-[var(--sidebar)]"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <td className="px-4 py-2">
                      <Link
                        href={`/logs/l1/${item.file_id}`}
                        className="hover:underline"
                        style={{ color: "var(--foreground)" }}
                      >
                        <code className="text-xs">{item.file_id}</code>
                      </Link>
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums" style={{ color: "var(--muted)" }}>
                      {formatBytes(item.size_bytes)}
                    </td>
                    <td className="px-4 py-2 text-xs" style={{ color: "var(--muted)" }}>
                      {new Date(item.modified_at).toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
