"use client";

/**
 * Multicam sync declare/nudge panel (audio_sync.plan.md SS10). Opened via
 * `SyncButton` (search-edit-bar.tsx) once 2+ files are selected in Drive.
 * v1 UI simplification: a numeric offset field per member rather than a
 * draggable stacked-waveform view -- the `timeline-editor.tsx` `Waveform`
 * component (client-side peak decode from the playback proxy) is the
 * established reusable piece for a future drag-to-nudge upgrade; a number
 * input ships the same functional outcome (review + correct an offset)
 * without that extra canvas-interaction surface.
 */
import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Loader2, Waves, X } from "lucide-react";
import { useDriveStore } from "@/stores/drive-store";
import { useAuthStore } from "@/stores/auth-store";
import {
  createSyncGroup, detectSync,
  type SyncAlignedBy, type SyncDetectResult,
} from "@/lib/api";

export function SyncPanel() {
  const { syncPanelOpen, syncScopeFileIds, closeSyncPanel, files } = useDriveStore();
  const token = useAuthStore((s) => s.session?.access_token);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SyncDetectResult | null>(null);
  const [authoritative, setAuthoritative] = useState<string | null>(null);
  const [offsets, setOffsets] = useState<Record<string, number>>({});
  const [saving, setSaving] = useState(false);
  const [done, setDone] = useState(false);

  useEffect(() => {
    if (!syncPanelOpen || !token || syncScopeFileIds.length < 2) return;
    setLoading(true);
    setError(null);
    setResult(null);
    setDone(false);
    detectSync(syncScopeFileIds, token)
      .then((res) => {
        setResult(res);
        setAuthoritative(res.suggested_authoritative_file_id);
        setOffsets(Object.fromEntries(res.members.map((m) => [m.file_id, m.offset_ms])));
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Detection failed."))
      .finally(() => setLoading(false));
  }, [syncPanelOpen, syncScopeFileIds, token]);

  if (!syncPanelOpen) return null;

  function fileName(id: string): string {
    return files.find((f) => f.id === id)?.name ?? id.slice(0, 8);
  }

  async function confirm() {
    if (!result || !token) return;
    setSaving(true);
    setError(null);
    try {
      await createSyncGroup(
        result.members.map((m) => {
          const edited = (offsets[m.file_id] ?? m.offset_ms) !== m.offset_ms;
          return {
            file_id: m.file_id,
            offset_ms: offsets[m.file_id] ?? m.offset_ms,
            role: m.role,
            confidence: edited ? null : m.confidence,
            aligned_by: (edited ? "manual" : m.aligned_by) as SyncAlignedBy,
          };
        }),
        token,
        authoritative,
      );
      setDone(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save the sync group.");
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" style={{ background: "rgba(0,0,0,0.6)" }}>
      <div
        className="w-full max-w-lg rounded-xl border p-5"
        style={{ borderColor: "var(--border)", background: "var(--background)" }}
      >
        <div className="mb-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Waves size={16} style={{ color: "var(--accent)" }} />
            <p className="text-sm font-medium">Sync {syncScopeFileIds.length} files</p>
          </div>
          <button onClick={closeSyncPanel} className="rounded p-1 hover:bg-[var(--accent-soft)]">
            <X size={16} />
          </button>
        </div>

        {loading && (
          <div className="flex items-center justify-center gap-2 py-8 text-[13px]" style={{ color: "var(--muted)" }}>
            <Loader2 size={14} className="animate-spin" /> Analyzing audio…
          </div>
        )}

        {error && <p className="mb-3 text-[12px]" style={{ color: "var(--danger)" }}>{error}</p>}

        {result && !done && (
          <div className="space-y-4">
            {result.unusable_file_ids.length > 0 && (
              <p className="text-[11px]" style={{ color: "var(--danger)" }}>
                Could not read audio from {result.unusable_file_ids.length} file(s) -- excluded below.
              </p>
            )}
            <div className="max-h-80 space-y-2 overflow-y-auto">
              {result.members.map((m) => (
                <div
                  key={m.file_id}
                  className="flex items-center gap-2 rounded-lg border px-3 py-2"
                  style={{ borderColor: "var(--border)" }}
                >
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-[12px] font-medium">{fileName(m.file_id)}</p>
                    <p className="text-[10px]" style={{ color: "var(--muted)" }}>
                      {m.role === "audio" ? "external audio" : "camera angle"}
                    </p>
                  </div>
                  <input
                    type="number"
                    value={offsets[m.file_id] ?? m.offset_ms}
                    onChange={(e) => setOffsets((prev) => ({ ...prev, [m.file_id]: Number(e.target.value) }))}
                    className="w-24 rounded border bg-transparent px-2 py-1 text-right text-[12px] tabular-nums outline-none"
                    style={{ borderColor: "var(--border)" }}
                  />
                  <span className="text-[10px]" style={{ color: "var(--muted)" }}>ms</span>
                  <span title={m.high_confidence ? "High confidence" : "Low confidence -- check this offset"}>
                    {m.high_confidence ? (
                      <CheckCircle2 size={14} style={{ color: "var(--success)" }} />
                    ) : (
                      <AlertTriangle size={14} style={{ color: "var(--danger)" }} />
                    )}
                  </span>
                  <label className="flex items-center gap-1 text-[10px]" style={{ color: "var(--muted)" }}>
                    <input
                      type="radio"
                      name="authoritative"
                      checked={authoritative === m.file_id}
                      onChange={() => setAuthoritative(m.file_id)}
                    />
                    authoritative
                  </label>
                </div>
              ))}
            </div>
            {result.members.some((m) => !m.high_confidence) && (
              <p className="text-[11px]" style={{ color: "var(--muted)" }}>
                Low confidence on some members -- check/nudge their offsets before confirming.
              </p>
            )}
            <div className="flex justify-end gap-2">
              <button
                onClick={closeSyncPanel}
                className="rounded-lg border px-3 py-1.5 text-[12px]"
                style={{ borderColor: "var(--border)" }}
              >
                Cancel
              </button>
              <button
                onClick={() => void confirm()}
                disabled={saving}
                className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-[12px] font-medium disabled:opacity-50"
                style={{ background: "var(--accent)", color: "var(--background)" }}
              >
                {saving && <Loader2 size={12} className="animate-spin" />}
                Confirm sync
              </button>
            </div>
          </div>
        )}

        {done && (
          <div className="flex flex-col items-center gap-2 py-6 text-center">
            <CheckCircle2 size={28} style={{ color: "var(--success)" }} />
            <p className="text-[13px]">
              Synced. Ingesting these files will now use one authoritative audio source.
            </p>
            <button
              onClick={closeSyncPanel}
              className="mt-2 rounded-lg px-3 py-1.5 text-[12px] font-medium"
              style={{ background: "var(--accent)", color: "var(--background)" }}
            >
              Done
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
