"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useAuthStore } from "@/stores/auth-store";
import {
  listEditLogs,
  listL1Logs,
  listL2Logs,
  type EditLogListItem,
  type L1LogListItem,
  type L2LogListItem,
} from "@/lib/api";
import { ScrollText, FileVideo, Layers, Sparkles, RefreshCw } from "lucide-react";

type Tab = "edits" | "l1" | "l2";

function statusColor(status: string | null) {
  switch (status) {
    case "ok":
    case "ready":
      return "var(--accent)";
    case "failed":
      return "#ef4444";
    case "started":
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
  const [tab, setTab] = useState<Tab>("l1");
  const [edits, setEdits] = useState<EditLogListItem[]>([]);
  const [l1, setL1] = useState<L1LogListItem[]>([]);
  const [l2, setL2] = useState<L2LogListItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    if (!session?.access_token) return;
    setLoading(true);
    setError(null);
    try {
      const [e, l, l2res] = await Promise.all([
        listEditLogs(session.access_token),
        listL1Logs(session.access_token),
        listL2Logs(session.access_token),
      ]);
      setEdits(e.items);
      setL1(l.items);
      setL2(l2res.items);
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
        Per-video <strong>L1</strong> (fast index) and <strong>L2</strong> (deep enrichment)
        analyses, sourced live from the database so they show up no matter which worker ran
        them. Each row includes how long the analysis took, in seconds.
      </p>

      <div className="mb-4 flex gap-1 border-b" style={{ borderColor: "var(--border)" }}>
        {([
          { id: "l1" as Tab, label: `L1 analyses (${l1.length})`, icon: FileVideo },
          { id: "l2" as Tab, label: `L2 analyses (${l2.length})`, icon: Layers },
          { id: "edits" as Tab, label: `Edit requests (${edits.length})`, icon: Sparkles },
        ]).map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className="flex items-center gap-2 px-4 py-2 text-sm transition-colors"
            style={{
              borderBottom: tab === id ? "2px solid var(--accent)" : "2px solid transparent",
              color: tab === id ? "var(--foreground)" : "var(--muted)",
              fontWeight: tab === id ? 600 : 400,
            }}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
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
                  <th className="px-4 py-2 text-left font-medium">File</th>
                  <th className="px-4 py-2 text-left font-medium">Status</th>
                  <th className="px-4 py-2 text-right font-medium">Shots</th>
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
                    <td className="px-4 py-2 text-right tabular-nums" style={{ color: "var(--muted)" }}>
                      {item.shot_count}
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
      )}

      {tab === "l2" && (
        <div className="overflow-hidden rounded-lg border" style={{ borderColor: "var(--border)" }}>
          {l2.length === 0 ? (
            <div className="px-4 py-8 text-center text-sm" style={{ color: "var(--muted)" }}>
              No L2 analyses yet. Deep enrichment runs in the background after upload.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead style={{ background: "var(--sidebar)", color: "var(--muted)" }}>
                <tr>
                  <th className="px-4 py-2 text-left font-medium">File</th>
                  <th className="px-4 py-2 text-left font-medium">Status</th>
                  <th className="px-4 py-2 text-right font-medium">Enriched shots</th>
                  <th className="px-4 py-2 text-right font-medium">Time taken</th>
                  <th className="px-4 py-2 text-left font-medium">Analyzed</th>
                </tr>
              </thead>
              <tbody>
                {l2.map((item) => (
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
                        style={{ background: "var(--border)", color: statusColor(item.l2_status) }}
                      >
                        {item.l2_status ?? "—"}
                      </span>
                    </td>
                    <td className="px-4 py-2 text-right tabular-nums" style={{ color: "var(--muted)" }}>
                      {item.enriched_shots} / {item.shot_count}
                    </td>
                    <td
                      className="px-4 py-2 text-right tabular-nums font-medium"
                      style={{ color: item.l2_seconds != null ? "var(--foreground)" : "var(--muted)" }}
                    >
                      {formatSeconds(item.l2_seconds)}
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
      )}
    </div>
  );
}
