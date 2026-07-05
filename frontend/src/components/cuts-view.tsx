"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAuthStore } from "@/stores/auth-store";
import { useDriveStore } from "@/stores/drive-store";
import { getCutsFeed, getFilePlaybackUrl, type Cut, type FileRecord } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Scissors, Play, ChevronDown, Check, GripVertical, Volume2, VolumeX } from "lucide-react";
import { EditButton } from "./search-edit-bar";

const ENERGY_LABELS = ["Broad", "Calm", "Balanced", "Tight", "Sharp"];
const energyLabel = (e: number) => ENERGY_LABELS[Math.min(4, Math.round(e * 4))];

// Cuts v2 (see cuts_v2.plan.md). One video = one horizontal row of its
// deterministic, NON-OVERLAPPING partition. Every cut carries >=1 tag
// (said/done/shown); a cut appears under a filter when it INCLUDES that tag.
// No energy ladder, no take stacking, no channel tabs -- a contiguous filmstrip.

type FilterKey = "all" | "said" | "done" | "shown";

const FILTERS: { key: FilterKey; label: string }[] = [
  { key: "all", label: "All" },
  { key: "said", label: "Said" },
  { key: "done", label: "Done" },
];

// Uniform tile width per orientation -- large, matching the old hero tiles.
const CARD_W = { landscape: 340, portrait: 232 } as const;

const ROW_DND = "application/x-cut-row";

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
  const [energy, setEnergy] = useState(0.5);
  const [orientation, setOrientation] = useState<"landscape" | "portrait">("landscape");
  const [fit, setFit] = useState<"adjusted" | "original">("adjusted");
  const [cuts, setCuts] = useState<Cut[]>([]);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [order, setOrder] = useState<string[]>([]);
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
    const t = setTimeout(() => {
      getCutsFeed(candidateIds, energy, token)
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
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, candidateKey, energy]);

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

  // All cuts stay in the feed (incl. shown) -- we just never WRITE the word
  // "shown" on a tile (no shown badge, no shown label); see CutCard.
  const displayCuts = cuts;

  // Group by file -> per-video row, each row's cuts in source order.
  const rows = useMemo(() => {
    const present = displayCuts.filter((c) => filesById[c.file_id]);
    const byFile: Record<string, Cut[]> = {};
    for (const c of present) (byFile[c.file_id] ??= []).push(c);
    return Object.entries(byFile)
      .map(([fileId, list]) => ({
        fileId,
        fileName: filesById[fileId]?.name ?? fileId,
        cuts: [...list].sort((a, b) => a.src_in_ms - b.src_in_ms),
      }))
      .sort((a, b) => a.fileName.localeCompare(b.fileName));
  }, [displayCuts, filesById]);

  // Keep a user-reorderable order of the rows: preserve manual moves, append
  // any newly-arrived videos at the end, drop ones that vanished.
  useEffect(() => {
    setOrder((prev) => {
      const ids = rows.map((r) => r.fileId);
      const kept = prev.filter((id) => ids.includes(id));
      const added = ids.filter((id) => !kept.includes(id));
      const next = [...kept, ...added];
      return next.length === prev.length && next.every((id, i) => id === prev[i])
        ? prev
        : next;
    });
  }, [rows]);

  const orderedRows = useMemo(() => {
    const byId: Record<string, (typeof rows)[number]> = {};
    for (const r of rows) byId[r.fileId] = r;
    return order.map((id) => byId[id]).filter(Boolean);
  }, [order, rows]);

  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const f of FILTERS) {
      if (f.key === "all") continue;
      c[f.key] = displayCuts.filter((cut) => includesTag(cut, f.key)).length;
    }
    return c;
  }, [displayCuts]);

  const toggle = useCallback((key: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  // --- Row reorder (drag the grip handle only, so card DnD is untouched) ---
  const dragRowId = useRef<string | null>(null);
  const onRowDragStart = useCallback((e: React.DragEvent, fileId: string) => {
    dragRowId.current = fileId;
    e.dataTransfer.setData(ROW_DND, fileId);
    e.dataTransfer.effectAllowed = "move";
  }, []);
  const onRowDragOver = useCallback((e: React.DragEvent, overId: string) => {
    const dragged = dragRowId.current;
    if (!dragged || dragged === overId) return;
    if (!e.dataTransfer.types.includes(ROW_DND)) return; // ignore card drags
    e.preventDefault();
    setOrder((prev) => {
      const from = prev.indexOf(dragged);
      const to = prev.indexOf(overId);
      if (from < 0 || to < 0 || from === to) return prev;
      const next = [...prev];
      next.splice(from, 1);
      next.splice(to, 0, dragged);
      return next;
    });
  }, []);
  const onRowDragEnd = useCallback(() => {
    dragRowId.current = null;
  }, []);

  const totalVisible = rows.reduce(
    (n, r) => n + r.cuts.filter((c) => includesTag(c, filter)).length,
    0
  );

  return (
    <div>
      {/* Dropdowns: category (left) + framing (orientation / fit); Edit right. */}
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2.5">
          <TagDropdown value={filter} counts={counts} total={displayCuts.length} onChange={setFilter} />
          <PillDropdown
            options={["Landscape", "Portrait"]}
            value={orientation === "landscape" ? "Landscape" : "Portrait"}
            onChange={(v) => setOrientation(v === "Portrait" ? "portrait" : "landscape")}
          />
          <PillDropdown
            options={["Frame Adjusted", "Original"]}
            value={fit === "adjusted" ? "Frame Adjusted" : "Original"}
            onChange={(v) => setFit(v === "Original" ? "original" : "adjusted")}
          />
        </div>
        <EditButton />
      </div>

      {/* Energy dial: tightness for every cut + windup/payoff split for
          done/shown at the high (Tight/Sharp) end. */}
      <div className="mb-7">
        <EnergyBar value={energy} onChange={setEnergy} />
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
          {orderedRows.map((row) => {
            const visible = row.cuts.filter((c) => includesTag(c, filter));
            if (visible.length === 0) return null;
            const total = visible.reduce((n, c) => n + c.duration_ms, 0);
            return (
              <div
                key={row.fileId}
                onDragOver={(e) => onRowDragOver(e, row.fileId)}
                onDrop={(e) => {
                  if (e.dataTransfer.types.includes(ROW_DND)) e.preventDefault();
                }}
              >
                <div className="mb-3 flex items-center gap-2.5">
                  <span
                    draggable
                    onDragStart={(e) => onRowDragStart(e, row.fileId)}
                    onDragEnd={onRowDragEnd}
                    className="shrink-0 cursor-grab active:cursor-grabbing"
                    style={{ color: "var(--muted)" }}
                    title="Drag to reorder"
                  >
                    <GripVertical size={15} />
                  </span>
                  <span
                    className="shrink-0 truncate text-xs"
                    style={{ color: "var(--muted)" }}
                  >
                    {row.fileName}
                  </span>
                  <span className="shrink-0 text-xs" style={{ color: "var(--muted)", opacity: 0.7 }}>
                    {visible.length} cuts · {fmtDur(total)}
                  </span>
                  <div className="h-px flex-1" style={{ background: "var(--border)" }} />
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
                        orientation={orientation}
                        fit={fit}
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
        style={{ borderColor: "rgba(255,255,255,0.4)", color: "var(--foreground)" }}
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

function EnergyBar({ value, onChange }: { value: number; onChange: (v: number) => void }) {
  const trackRef = useRef<HTMLDivElement>(null);
  const valueRef = useRef(value);
  valueRef.current = value;

  const apply = useCallback(
    (clientX: number, isClick: boolean) => {
      const el = trackRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const t = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      let snapped = Math.round(t * 4) / 4;
      const cur = valueRef.current;
      if (isClick && snapped === cur) {
        if (t > cur) snapped = Math.min(1, cur + 0.25);
        else if (t < cur) snapped = Math.max(0, cur - 0.25);
      }
      if (snapped !== cur) onChange(snapped);
    },
    [onChange]
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      apply(e.clientX, true);
      const move = (ev: PointerEvent) => apply(ev.clientX, false);
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    },
    [apply]
  );

  return (
    <div className="mx-auto flex w-3/4 items-center gap-4">
      <span className="shrink-0 pl-1 text-sm font-medium" style={{ color: "var(--foreground)" }}>
        Energy
      </span>
      <div
        ref={trackRef}
        onPointerDown={handlePointerDown}
        className="relative flex-1 cursor-pointer select-none py-4"
        style={{ touchAction: "none" }}
      >
        <div className="h-px w-full rounded-full" style={{ background: "rgba(255,255,255,0.16)" }} />
        <div
          className="absolute left-0 top-1/2 h-px -translate-y-1/2 rounded-full"
          style={{
            width: `${value * 100}%`,
            background: "var(--foreground)",
            transition: "width 0.35s cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        />
        <div
          className="absolute top-1/2 h-3.5 w-[3px] -translate-y-1/2 rounded-full"
          style={{
            left: `calc(${value * 100}% - 1.5px)`,
            background: "var(--foreground)",
            transition: "left 0.35s cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        />
      </div>
      <span
        className="inline-flex min-w-[74px] shrink-0 items-center justify-center rounded-md px-3 py-1 text-xs font-semibold"
        style={{ background: "var(--accent)", color: "var(--background)" }}
      >
        {energyLabel(value)}
      </span>
    </div>
  );
}

function PillDropdown({
  options,
  value,
  onChange,
}: {
  options: string[];
  value?: string;
  onChange?: (v: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [internal, setInternal] = useState(options[0]);
  const selected = value ?? internal;
  const select = (opt: string) => {
    setInternal(opt);
    onChange?.(opt);
  };
  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors hover:bg-[var(--sidebar)]"
        style={{ borderColor: "rgba(255,255,255,0.4)", color: "var(--foreground)" }}
      >
        {selected}
        <ChevronDown size={15} style={{ color: "var(--muted)" }} />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} />
          <div
            className="absolute left-0 z-40 mt-1.5 min-w-[170px] overflow-hidden rounded-xl border shadow-xl"
            style={{ background: "var(--background)", borderColor: "var(--border)" }}
          >
            {options.map((opt) => (
              <button
                key={opt}
                onClick={() => {
                  select(opt);
                  setOpen(false);
                }}
                className="flex w-full items-center justify-between px-3.5 py-2 text-sm transition-colors hover:bg-[var(--sidebar)]"
                style={{ color: selected === opt ? "var(--foreground)" : "var(--muted)" }}
              >
                {opt}
                {selected === opt && <Check size={14} />}
              </button>
            ))}
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
  orientation,
  fit,
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
  orientation: "landscape" | "portrait";
  fit: "adjusted" | "original";
  selected: boolean;
  weldLeft: boolean;
  weldRight: boolean;
  onToggle: () => void;
  isActive: boolean;
  onActivate: () => void;
  onDeactivate: () => void;
}) {
  const [playUrl, setPlayUrl] = useState<string | null>(null);
  const [muted, setMuted] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const inSec = cut.src_in_ms / 1000;
  const outSec = cut.src_out_ms / 1000;
  const peakSec = cut.peak_ms / 1000;
  const objectFit = fit === "adjusted" ? "cover" : "contain";

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
      v.muted = muted;
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
  }, [isActive, playUrl, muted]);

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
      style={{ width: CARD_W[orientation], marginLeft: weldLeft ? 0 : 10 }}
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
          "group relative flex cursor-pointer items-center justify-center overflow-hidden border transition-colors",
          orientation === "portrait" ? "aspect-[9/16]" : "aspect-video",
          !weldLeft && "rounded-l-xl",
          !weldRight && "rounded-r-xl"
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
            muted={muted}
            draggable={false}
            onLoadedMetadata={onLoadedMetadata}
            onTimeUpdate={onTimeUpdate}
            className="h-full w-full bg-black"
            style={{ objectFit }}
          />
        )}

        {/* Tag badges (top-left): primary first, then any extra tags. "shown"
            is never written on a tile. Neutral chips -- the palette stays
            b/w/grey; the words carry the meaning. */}
        <div className="absolute left-2 top-2 z-20 flex items-center gap-1">
          {cut.tags
            .filter((t) => t !== "shown")
            .map((t) => (
              <span
                key={t}
                className="rounded px-1.5 py-0.5 text-[11px] font-semibold capitalize"
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
            className="absolute right-2 top-2 z-20 flex h-6 w-6 items-center justify-center rounded-full"
            style={{ background: "var(--accent)", color: "var(--background)" }}
          >
            <Check size={14} />
          </span>
        )}

        {/* Mute toggle -- cards play with sound by default; sits left of the
            selected tick so they never overlap. */}
        {playUrl && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setMuted((m) => !m);
            }}
            className="absolute top-2 z-20 flex items-center justify-center rounded-full p-1.5 text-white transition-colors hover:bg-black/40"
            style={{ right: selected ? 36 : 8, background: "rgba(0,0,0,0.55)" }}
            title={muted ? "Unmute" : "Mute"}
          >
            {muted ? <VolumeX size={14} /> : <Volume2 size={14} />}
          </button>
        )}

        {/* Hover play hint. */}
        <span
          className="pointer-events-none absolute left-1/2 top-1/2 z-10 flex h-11 w-11 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full opacity-100 shadow-lg transition-opacity group-hover:opacity-0"
          style={{ background: "var(--accent)" }}
        >
          <Play size={20} className="ml-0.5" fill="currentColor" style={{ color: "var(--background)" }} />
        </span>

        {/* Duration (bottom-right) + speaker (bottom-left). */}
        <span className="absolute bottom-2 right-2 z-10 rounded bg-black/70 px-1.5 py-0.5 text-[11px] font-medium text-white">
          {fmtDur(cut.duration_ms)}
        </span>
        {cut.speaker && (
          <span className="absolute bottom-2 left-2 z-10 rounded bg-black/60 px-1.5 py-0.5 text-[11px] font-medium text-white">
            {cut.speaker}
          </span>
        )}
      </div>
      <p className="mt-2 line-clamp-2 px-0.5 text-sm leading-snug" style={{ minHeight: "2.5em" }}>
        {cut.primary === "shown" ? "" : cut.label}
      </p>
    </div>
  );
}
