"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAuthStore } from "@/stores/auth-store";
import { useDriveStore } from "@/stores/drive-store";
import { getCutsFeed, getFilePlaybackUrl, type Cut, type FileRecord } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Scissors, Play, ChevronDown, Check } from "lucide-react";
import { EditButton } from "./search-edit-bar";

// Cuts v2 (see cuts_v2.plan.md). One video = one horizontal row of its
// deterministic, NON-OVERLAPPING partition. Every cut carries >=1 tag
// (said/done/shown); a cut appears under a filter when it INCLUDES that tag.
// No energy ladder, no take stacking, no channel tabs -- a contiguous filmstrip.

type FilterKey = "all" | "said" | "done" | "shown";

const FILTERS: { key: FilterKey; label: string }[] = [
  { key: "all", label: "All" },
  { key: "said", label: "Said" },
  { key: "done", label: "Done" },
  { key: "shown", label: "Shown" },
];

const CARD_W = 224; // uniform card width (px) -- easiest to scan

function fmtDur(ms: number): string {
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.round(s - m * 60)).padStart(2, "0")}`;
}

function cutKey(c: Cut): string {
  return `${c.file_id}:${c.src_in_ms}`;
}

function includesTag(c: Cut, key: FilterKey): boolean {
  if (key === "all") return true;
  return c.tags.includes(key);
}

export function CutsView() {
  const token = useAuthStore((s) => s.session?.access_token);
  const files = useDriveStore((s) => s.files);
  const [filter, setFilter] = useState<FilterKey>("all");
  const [cuts, setCuts] = useState<Cut[]>([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const urlCache = useRef<Record<string, Promise<string | null>>>({});

  const candidates = useMemo(
    () => files.filter((f) => f.file_type === "video" && f.l1_status === "ready"),
    [files]
  );
  const filesById = useMemo(() => {
    const m: Record<string, FileRecord> = {};
    for (const f of files) m[f.id] = f;
    return m;
  }, [files]);
  const candidateIds = useMemo(() => candidates.map((f) => f.id), [candidates]);
  const candidateKey = candidateIds.join(",");

  useEffect(() => {
    if (!token || candidateIds.length === 0) {
      setCuts([]);
      return;
    }
    let cancelled = false;
    setLoading(true);
    getCutsFeed(candidateIds, token)
      .then((r) => {
        if (cancelled) return;
        setCuts(r.cuts ?? []);
        setLoading(false);
      })
      .catch(() => {
        if (cancelled) return;
        setCuts([]);
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, candidateKey]);

  const getUrl = useCallback(
    (fileId: string): Promise<string | null> => {
      if (!token) return Promise.resolve(null);
      if (!urlCache.current[fileId]) {
        urlCache.current[fileId] = getFilePlaybackUrl(fileId, token)
          .then((r) => r.url)
          .catch(() => null);
      }
      return urlCache.current[fileId];
    },
    [token]
  );

  // Group by file -> per-video row, each row's cuts in source order.
  const rows = useMemo(() => {
    const present = cuts.filter((c) => filesById[c.file_id]);
    const byFile: Record<string, Cut[]> = {};
    for (const c of present) (byFile[c.file_id] ??= []).push(c);
    return Object.entries(byFile)
      .map(([fileId, list]) => ({
        fileId,
        fileName: filesById[fileId]?.name ?? fileId,
        cuts: [...list].sort((a, b) => a.src_in_ms - b.src_in_ms),
      }))
      .sort((a, b) => a.fileName.localeCompare(b.fileName));
  }, [cuts, filesById]);

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const f of FILTERS) {
      if (f.key === "all") continue;
      c[f.key] = cuts.filter((cut) => includesTag(cut, f.key)).length;
    }
    return c;
  }, [cuts]);

  const toggle = useCallback((key: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const setRowSelection = useCallback((rowCuts: Cut[], on: boolean) => {
    setSelected((prev) => {
      const next = new Set(prev);
      for (const c of rowCuts) {
        if (on) next.add(cutKey(c));
        else next.delete(cutKey(c));
      }
      return next;
    });
  }, []);

  const totalVisible = rows.reduce(
    (n, r) => n + r.cuts.filter((c) => includesTag(c, filter)).length,
    0
  );

  return (
    <div>
      {/* Tag filter (replaces the old channel tabs) + Edit pinned right. */}
      <div className="mb-6 flex items-center justify-between gap-6">
        <TagDropdown value={filter} counts={counts} total={cuts.length} onChange={setFilter} />
        <EditButton />
      </div>

      {loading && (
        <p className="py-12 text-center text-sm" style={{ color: "var(--muted)" }}>
          Partitioning footage…
        </p>
      )}

      {!loading && totalVisible === 0 && (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <Scissors size={34} style={{ color: "var(--accent)" }} />
          <p className="mt-4 text-lg font-semibold">No cuts yet</p>
          <p className="mt-1 max-w-sm text-sm" style={{ color: "var(--muted)" }}>
            Upload video. Once analyzed, each clip appears here as a row of
            non-overlapping cuts you can scroll through and pick.
          </p>
        </div>
      )}

      {!loading && totalVisible > 0 && (
        <div className="flex flex-col gap-10">
          {rows.map((row) => {
            const visible = row.cuts.filter((c) => includesTag(c, filter));
            if (visible.length === 0) return null;
            const allSelected = visible.every((c) => selected.has(cutKey(c)));
            const total = visible.reduce((n, c) => n + c.duration_ms, 0);
            return (
              <div key={row.fileId}>
                <div className="mb-3 flex items-center gap-3">
                  <span className="shrink-0 truncate text-sm font-medium">{row.fileName}</span>
                  <span className="shrink-0 text-xs" style={{ color: "var(--muted)" }}>
                    {visible.length} cuts · {fmtDur(total)}
                  </span>
                  <div className="h-px flex-1" style={{ background: "var(--border)" }} />
                  <button
                    onClick={() => setRowSelection(visible, !allSelected)}
                    className="shrink-0 text-xs transition-colors hover:text-[var(--foreground)]"
                    style={{ color: "var(--muted)" }}
                  >
                    {allSelected ? "Clear" : "Select all"}
                  </button>
                </div>
                <div className="-mx-1 flex overflow-x-auto px-1 pb-2">
                  {visible.map((c, i) => {
                    const prev = visible[i - 1];
                    const next = visible[i + 1];
                    const isSel = selected.has(cutKey(c));
                    // Weld only when adjacent cards are BOTH selected AND
                    // source-contiguous (they will play seamlessly).
                    const weldLeft =
                      isSel && !!prev && selected.has(cutKey(prev)) && prev.src_out_ms === c.src_in_ms;
                    const weldRight =
                      isSel && !!next && selected.has(cutKey(next)) && c.src_out_ms === next.src_in_ms;
                    return (
                      <CutCard
                        key={cutKey(c)}
                        file={filesById[c.file_id]!}
                        cut={c}
                        getUrl={getUrl}
                        selected={isSel}
                        weldLeft={weldLeft}
                        weldRight={weldRight}
                        onToggle={() => toggle(cutKey(c))}
                        isActive={activeKey === cutKey(c)}
                        onActivate={() => setActiveKey(cutKey(c))}
                        onDeactivate={() =>
                          setActiveKey((k) => (k === cutKey(c) ? null : k))
                        }
                      />
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function TagDropdown({
  value,
  counts,
  total,
  onChange,
}: {
  value: FilterKey;
  counts: Record<string, number>;
  total: number;
  onChange: (v: FilterKey) => void;
}) {
  const [open, setOpen] = useState(false);
  const current = FILTERS.find((f) => f.key === value) ?? FILTERS[0];
  const n = value === "all" ? total : counts[value] ?? 0;
  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors hover:bg-[var(--sidebar)]"
        style={{ borderColor: "var(--border)", color: "var(--foreground)" }}
      >
        {current.label}
        <span className="text-xs" style={{ color: "var(--muted)" }}>
          {n}
        </span>
        <ChevronDown size={15} style={{ color: "var(--muted)" }} />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} />
          <div
            className="absolute left-0 z-40 mt-1.5 min-w-[180px] overflow-hidden rounded-xl border shadow-xl"
            style={{ background: "var(--background)", borderColor: "var(--border)" }}
          >
            {FILTERS.map((f) => {
              const cn2 = f.key === "all" ? total : counts[f.key] ?? 0;
              return (
                <button
                  key={f.key}
                  onClick={() => {
                    onChange(f.key);
                    setOpen(false);
                  }}
                  className="flex w-full items-center justify-between gap-6 px-3.5 py-2 text-sm transition-colors hover:bg-[var(--sidebar)]"
                  style={{ color: value === f.key ? "var(--foreground)" : "var(--muted)" }}
                >
                  <span className="flex items-center gap-2">
                    {f.label}
                    <span className="text-xs" style={{ color: "var(--muted)" }}>
                      {cn2}
                    </span>
                  </span>
                  {value === f.key && <Check size={14} />}
                </button>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

function CutCard({
  file,
  cut,
  getUrl,
  selected,
  weldLeft,
  weldRight,
  onToggle,
  isActive,
  onActivate,
  onDeactivate,
}: {
  file: FileRecord;
  cut: Cut;
  getUrl: (fileId: string) => Promise<string | null>;
  selected: boolean;
  weldLeft: boolean;
  weldRight: boolean;
  onToggle: () => void;
  isActive: boolean;
  onActivate: () => void;
  onDeactivate: () => void;
}) {
  const [playUrl, setPlayUrl] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const inSec = cut.src_in_ms / 1000;
  const outSec = cut.src_out_ms / 1000;
  const peakSec = cut.peak_ms / 1000;

  async function ensureUrl() {
    if (playUrl) return;
    const url = await getUrl(file.id);
    if (url) setPlayUrl(url);
  }

  // Hover -> play the cut span (looping, muted); leave -> park on the peak still.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !playUrl) return;
    if (isActive) {
      try {
        if (v.currentTime < inSec || v.currentTime >= outSec) v.currentTime = inSec;
      } catch {
        /* ignore */
      }
      v.muted = true;
      v.play().catch(() => {});
    } else {
      v.pause();
      try {
        v.currentTime = peakSec;
      } catch {
        /* ignore */
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive, playUrl]);

  function onLoadedMetadata() {
    const v = videoRef.current;
    if (!v) return;
    try {
      v.currentTime = peakSec;
    } catch {
      /* ignore */
    }
  }

  function onTimeUpdate() {
    const v = videoRef.current;
    if (!v || !isActive) return;
    if (v.currentTime >= outSec - 0.02 || v.currentTime < inSec - 0.3) {
      try {
        v.currentTime = inSec;
      } catch {
        /* ignore */
      }
    }
  }

  function onDragStart(e: React.DragEvent) {
    const payload = JSON.stringify({
      kind: "hero",
      file_id: file.id,
      file_name: file.name,
      in_ms: cut.src_in_ms,
      out_ms: cut.src_out_ms,
      content: cut.label,
      speaker: cut.speaker,
    });
    e.dataTransfer.setData("application/x-hero-cut", payload);
    e.dataTransfer.setData("text/plain", payload);
    e.dataTransfer.effectAllowed = "copy";
  }

  return (
    <div
      style={{ width: CARD_W, marginLeft: weldLeft ? 0 : 8 }}
      className="shrink-0 first:ml-0"
    >
      <div
        onClick={onToggle}
        onMouseEnter={() => {
          onActivate();
          ensureUrl();
        }}
        onMouseLeave={onDeactivate}
        draggable
        onDragStart={onDragStart}
        className={cn(
          "group relative flex aspect-video cursor-pointer items-center justify-center overflow-hidden border transition-colors",
          !weldLeft && "rounded-l-lg",
          !weldRight && "rounded-r-lg"
        )}
        style={{
          background: "#000",
          borderColor: selected ? "var(--accent)" : "var(--border)",
          borderLeftColor: weldLeft ? "transparent" : selected ? "var(--accent)" : "var(--border)",
          borderRightColor: weldRight ? "transparent" : selected ? "var(--accent)" : "var(--border)",
        }}
        title={cut.label}
      >
        {playUrl && (
          <video
            ref={videoRef}
            src={`${playUrl}#t=${peakSec.toFixed(2)}`}
            playsInline
            preload="metadata"
            muted
            draggable={false}
            onLoadedMetadata={onLoadedMetadata}
            onTimeUpdate={onTimeUpdate}
            className="h-full w-full bg-black object-cover"
          />
        )}

        {/* Tag badges (top-left): primary first, then any extra tags. Neutral
            chips -- the palette stays b/w/grey; the words carry the meaning. */}
        <div className="absolute left-1.5 top-1.5 z-20 flex items-center gap-1">
          {cut.tags.map((t) => (
            <span
              key={t}
              className="rounded px-1.5 py-0.5 text-[10px] font-semibold capitalize"
              style={{
                background: t === cut.primary ? "rgba(255,255,255,0.92)" : "rgba(0,0,0,0.6)",
                color: t === cut.primary ? "#000" : "#fff",
              }}
            >
              {t}
            </span>
          ))}
        </div>

        {/* Selected tick (top-right). */}
        {selected && (
          <span
            className="absolute right-1.5 top-1.5 z-20 flex h-5 w-5 items-center justify-center rounded-full"
            style={{ background: "var(--accent)", color: "var(--background)" }}
          >
            <Check size={13} />
          </span>
        )}

        {/* Hover play hint. */}
        <span
          className="pointer-events-none absolute left-1/2 top-1/2 z-10 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full opacity-100 shadow-lg transition-opacity group-hover:opacity-0"
          style={{ background: "var(--accent)" }}
        >
          <Play size={16} className="ml-0.5" fill="currentColor" style={{ color: "var(--background)" }} />
        </span>

        {/* Duration (bottom-right) + speaker (bottom-left). */}
        <span className="absolute bottom-1.5 right-1.5 z-10 rounded bg-black/70 px-1.5 py-0.5 text-[10px] font-medium text-white">
          {fmtDur(cut.duration_ms)}
        </span>
        {cut.speaker && (
          <span className="absolute bottom-1.5 left-1.5 z-10 rounded bg-black/60 px-1.5 py-0.5 text-[10px] font-medium text-white">
            {cut.speaker}
          </span>
        )}
      </div>
      <p className="mt-1.5 line-clamp-2 px-0.5 text-[11px] leading-snug" style={{ minHeight: "2.4em" }}>
        {cut.label || <em style={{ color: "var(--muted)" }}>(no label)</em>}
      </p>
    </div>
  );
}
