"use client";

import { useEffect, useState } from "react";
import { X, Copy, Check, Loader2, Trash2 } from "lucide-react";
import { useAuthStore } from "@/stores/auth-store";
import { createUploadLink, listUploadLinks, revokeUploadLink, type UploadLink } from "@/lib/api";
import { cn } from "@/lib/utils";

interface Props {
  open: boolean;
  onClose: () => void;
  folderId: string;
}

type Expiry = "24h" | "7d" | "never";

const EXPIRY_OPTIONS: { id: Expiry; label: string; hours?: number }[] = [
  { id: "24h", label: "24 hours", hours: 24 },
  { id: "7d", label: "7 days", hours: 24 * 7 },
  { id: "never", label: "Never" },
];

function linkUrl(token: string): string {
  // frontend_project_ux.plan.md Stage 2.3: the backend returns only the raw
  // token -- the full shareable URL is built from wherever THIS dashboard
  // is actually being served, so it's correct in dev, staging, or prod
  // without a backend env var that has to be kept in sync with the
  // frontend's own deployment.
  const origin = typeof window !== "undefined" ? window.location.origin : "";
  return `${origin}/u/${token}`;
}

function linkStatus(link: UploadLink): { label: string; active: boolean } {
  if (link.revoked) return { label: "Revoked", active: false };
  if (link.expires_at && new Date(link.expires_at).getTime() < Date.now()) {
    return { label: "Expired", active: false };
  }
  if (link.max_files != null && link.used_count >= link.max_files) {
    return { label: "Limit reached", active: false };
  }
  return { label: "Active", active: true };
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      onClick={async () => {
        await navigator.clipboard.writeText(text);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      className="flex shrink-0 items-center gap-1 rounded-md border px-2 py-1 text-xs transition-colors hover:opacity-80"
      style={{ borderColor: "var(--border)" }}
      title="Copy link"
    >
      {copied ? <Check size={12} /> : <Copy size={12} />}
      {copied ? "Copied" : "Copy"}
    </button>
  );
}

export function UploadLinkDialog({ open, onClose, folderId }: Props) {
  const token = useAuthStore((s) => s.session?.access_token);
  const [links, setLinks] = useState<UploadLink[]>([]);
  const [loading, setLoading] = useState(false);
  const [expiry, setExpiry] = useState<Expiry>("7d");
  const [maxFiles, setMaxFiles] = useState("");
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || !token) return;
    setError(null);
    setLoading(true);
    listUploadLinks(folderId, token)
      .then(setLinks)
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load links."))
      .finally(() => setLoading(false));
  }, [open, folderId, token]);

  if (!open) return null;

  async function handleGenerate() {
    if (!token) return;
    setError(null);
    setGenerating(true);
    try {
      const hours = EXPIRY_OPTIONS.find((o) => o.id === expiry)?.hours;
      const parsedMax = maxFiles.trim() ? parseInt(maxFiles, 10) : undefined;
      const link = await createUploadLink(
        folderId,
        { expiresInHours: hours, maxFiles: parsedMax },
        token
      );
      setLinks((prev) => [link, ...prev]);
      setMaxFiles("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not generate link.");
    } finally {
      setGenerating(false);
    }
  }

  async function handleRevoke(linkId: string) {
    if (!token) return;
    try {
      await revokeUploadLink(linkId, token);
      setLinks((prev) => prev.map((l) => (l.id === linkId ? { ...l, revoked: true } : l)));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not revoke link.");
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div
        className="relative flex max-h-[85vh] w-full max-w-md flex-col rounded-xl border p-6 shadow-xl"
        style={{ background: "var(--background)", borderColor: "var(--border)" }}
      >
        <button
          onClick={onClose}
          className="absolute right-4 top-4 rounded p-1 transition-colors hover:opacity-70"
          style={{ color: "var(--muted)" }}
        >
          <X size={16} />
        </button>

        <h2 className="text-lg font-semibold">Upload Link</h2>
        <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
          Anyone with the link can upload files into this project. No account needed.
        </p>

        <div className="mt-4 space-y-3">
          <div>
            <label className="mb-1.5 block text-[11px] font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>
              Expires
            </label>
            <div className="flex gap-1.5">
              {EXPIRY_OPTIONS.map((opt) => (
                <button
                  key={opt.id}
                  type="button"
                  onClick={() => setExpiry(opt.id)}
                  className={cn(
                    "flex-1 rounded-lg border px-2 py-1.5 text-xs font-medium transition-colors",
                    expiry !== opt.id && "hover:bg-[var(--border)]"
                  )}
                  style={{
                    borderColor: expiry === opt.id ? "var(--accent)" : "var(--border)",
                    background: expiry === opt.id ? "var(--accent-soft)" : undefined,
                  }}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="mb-1.5 block text-[11px] font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>
              Max files
            </label>
            <input
              type="number"
              min={1}
              value={maxFiles}
              onChange={(e) => setMaxFiles(e.target.value)}
              placeholder="Unlimited"
              className="w-full rounded-lg border px-3 py-2 text-sm outline-none focus:ring-2"
              style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
            />
          </div>

          <button
            type="button"
            onClick={handleGenerate}
            disabled={generating}
            className="flex w-full items-center justify-center gap-1.5 rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
            style={{ background: "var(--accent)", color: "var(--background)" }}
          >
            {generating && <Loader2 size={14} className="animate-spin" />}
            Generate link
          </button>

          {error && (
            <p className="text-[11px]" style={{ color: "var(--danger)" }}>
              {error}
            </p>
          )}
        </div>

        <div className="mt-5 min-h-0 flex-1 overflow-y-auto border-t pt-4" style={{ borderColor: "var(--border)" }}>
          <h3 className="mb-2 text-[11px] font-medium uppercase tracking-wide" style={{ color: "var(--muted)" }}>
            Existing links
          </h3>
          {loading ? (
            <div className="flex justify-center py-6">
              <Loader2 size={18} className="animate-spin" style={{ color: "var(--accent)" }} />
            </div>
          ) : links.length === 0 ? (
            <p className="py-2 text-xs" style={{ color: "var(--muted)" }}>
              No links yet.
            </p>
          ) : (
            <ul className="space-y-2">
              {links.map((link) => {
                const status = linkStatus(link);
                return (
                  <li
                    key={link.id}
                    className="flex items-center gap-2 rounded-lg border px-2.5 py-2"
                    style={{ borderColor: "var(--border)" }}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-1.5">
                        <span
                          className="truncate text-xs"
                          style={{ color: status.active ? "var(--foreground)" : "var(--muted)" }}
                        >
                          {linkUrl(link.token)}
                        </span>
                      </div>
                      <div className="mt-0.5 flex items-center gap-1.5 text-[10px]" style={{ color: "var(--muted)" }}>
                        <span style={{ color: status.active ? "var(--success)" : "var(--danger)" }}>
                          {status.label}
                        </span>
                        <span>·</span>
                        <span>
                          {link.used_count} upload{link.used_count === 1 ? "" : "s"}
                          {link.max_files != null ? ` / ${link.max_files}` : ""}
                        </span>
                      </div>
                    </div>
                    {status.active && <CopyButton text={linkUrl(link.token)} />}
                    {!link.revoked && (
                      <button
                        type="button"
                        onClick={() => handleRevoke(link.id)}
                        className="shrink-0 rounded-md p-1.5 transition-colors hover:opacity-80"
                        style={{ color: "var(--danger)" }}
                        title="Revoke"
                      >
                        <Trash2 size={14} />
                      </button>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
