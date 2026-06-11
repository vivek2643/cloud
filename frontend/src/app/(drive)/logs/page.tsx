"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useAuthStore } from "@/stores/auth-store";
import { listL1Logs, type L1LogListItem } from "@/lib/api";
import { ScrollText, RefreshCw } from "lucide-react";

function statusColor(status: string | null) {
  switch (status) {
    case "ready":
      return "var(--accent)";
    case "failed":
      return "#ef4444";
    case "running":
      return "#f59e0b";
    default:
      return "var(--muted)";
  }
}

function formatSeconds(s: number | null) {
  if (s == null) return "—";
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const rem = Math.round(s % 60);
  return `${m}m ${rem}s`;
}

export default function LogsPage() {
  const session = useAuthStore((s) => s.session);
  const [l1, setL1] = useState<L1LogListItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    if (!session?.access_token) return;
    setLoading(true);
    setError(null);
    try {
      const l = await listL1Logs(session.access_token);
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
          <h1 className="text-xl font-semibold">L1 analyses</h1>
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
        Per-video <strong>L1</strong> (fast index) analyses, sourced live from the database so
        they show up no matter which worker ran them. Each row includes how long the analysis
        took, in seconds.
      </p>

      {error && (
        <div
          className="mb-4 rounded-md border p-3 text-sm"
          style={{ borderColor: "#ef4444", color: "#ef4444" }}
        >
          {error}
        </div>
      )}

      <div className="overflow-hidden rounded-lg border" style={{ borderColor: "var(--border)" }}>
        {l1.length === 0 ? (
          <div className="px-4 py-8 text-center text-sm" style={{ color: "var(--muted)" }}>
            No L1 analyses yet. Upload a video to trigger one.
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead style={{ background: "var(--sidebar)", color: "var(--muted)" }}>
              <tr>
                <th className="px-4 py-2 text-left font-medium">File</th>
                <th className="px-4 py-2 text-left font-medium">Status</th>
                <th className="px-4 py-2 text-right font-medium">Time taken</th>
                <th className="px-4 py-2 text-left font-medium">Analyzed</th>
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
                      {item.name || item.file_id}
                    </Link>
                  </td>
                  <td className="px-4 py-2">
                    <span
                      className="rounded-full px-2 py-0.5 text-xs"
                      style={{ background: "var(--border)", color: statusColor(item.l1_status) }}
                    >
                      {item.l1_status}
                    </span>
                  </td>
                  <td
                    className="px-4 py-2 text-right tabular-nums font-medium"
                    style={{ color: item.l1_seconds != null ? "var(--foreground)" : "var(--muted)" }}
                  >
                    {formatSeconds(item.l1_seconds)}
                  </td>
                  <td className="px-4 py-2 text-xs" style={{ color: "var(--muted)" }}>
                    {item.analyzed_at ? new Date(item.analyzed_at).toLocaleString() : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
