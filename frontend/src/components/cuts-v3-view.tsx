"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAuthStore } from "@/stores/auth-store";
import { useDriveStore } from "@/stores/drive-store";
import {
  createProject,
  kickIngest,
  getCutsV3,
  getFilePlaybackUrl,
  type CutRecord,
  type CutsV3Response,
  type FileRecord,
  type IngestStatus,
} from "@/lib/api";
import { cn } from "@/lib/utils";
import {
  Sparkles,
  Play,
  ChevronDown,
  Check,
  GripVertical,
  Volume2,
  VolumeX,
  Layers,
  Bug,
  Loader2,
  Eye,
  EyeOff,
  Zap,
} from "lucide-react";
import { EditButton } from "./search-edit-bar";

// Cuts v3 (see cuts_v3.plan.md). One LLM ingest pass per project decides the
// final speech/video grouping, cross-clip takes, and every per-cut judgment
// (framing/look/summary). This view is READ-ONLY over `cut_records` -- it
// never groups/tightens/filters content; the dial's view-math (section 9)
// lands separately. Additive to v2 (`cuts-view.tsx`): that surface, its
// `/api/files/.../cuts` endpoint, and its data are untouched by this file.

type Aspect = "landscape" | "portrait" | "square";
const ASPECT_LABEL: Record<Aspect, string> = {
  landscape: "Landscape",
  portrait: "Portrait",
  square: "Square",
};
const ASPECT_CLASS: Record<Aspect, string> = {
  landscape: "aspect-video",
  portrait: "aspect-[9/16]",
  square: "aspect-square",
};
const CARD_W: Record<Aspect, number> = { landscape: 340, portrait: 232, square: 260 };

const ROW_DND = "application/x-cut-row";

const STATUS_LABEL: Record<IngestStatus, string> = {
  pending: "Queued…",
  pass1: "Reading transcripts + footage…",
  images: "Selecting frames…",
  pass2: "Judging every cut…",
  post: "Assembling…",
  ready: "Ready",
  failed: "Failed",
};

// The energy dial as pure view-math (cuts_v3_boundaries_v2.plan.md §D). Trims
// the played span INWARD toward the cut's anchor (hero_ts_ms) as energy rises --
// "negative padding". energy 0 -> the full grounded span; energy 1 -> pace.min_ms
// (the anchor-protected floor computed in post.py, so the payoff frame is never
// trimmed away). Speech cuts have min_ms == natural, so they don't tighten.
function tightenedSpan(cut: CutRecord, energy: number): { inMs: number; outMs: number } {
  const inMs0 = cut.src_in_ms;
  const outMs0 = cut.src_out_ms;
  const naturalDur = outMs0 - inMs0;
  const minDur = Math.min(cut.pace?.min_ms ?? naturalDur, naturalDur);
  const targetDur = Math.round(naturalDur - energy * (naturalDur - minDur));
  if (targetDur >= naturalDur || targetDur <= 0) return { inMs: inMs0, outMs: outMs0 };
  const hero = cut.hero_ts_ms ?? (inMs0 + outMs0) / 2;
  let inMs = Math.round(hero - targetDur / 2);
  let outMs = inMs + targetDur;
  if (inMs < inMs0) { inMs = inMs0; outMs = inMs + targetDur; }
  if (outMs > outMs0) { outMs = outMs0; inMs = outMs - targetDur; }
  return { inMs, outMs };
}

function fmtDur(ms: number): string {
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.round(s - m * 60)).padStart(2, "0")}`;
}

function cutKey(c: CutRecord): string {
  return c.id;
}

// Rect [x,y,w,h] (normalized, source-frame coords) -> the same
// object-position(focus) + transform(rotate,scale) convention already used
// by the render preview (`use-program-player.ts`'s applyFrameStyle) -- a
// preview approximation, not the pixel-exact export-time crop.
function cropStyle(
  box: [number, number, number, number] | null | undefined,
  rotationDeg: number | undefined
): { objectPosition: string; transform: string } {
  if (!box) {
    return { objectPosition: "center", transform: rotationDeg ? `rotate(${rotationDeg}deg)` : "" };
  }
  const [x, y, w, h] = box;
  const cx = Math.round(Math.min(1, Math.max(0, x + w / 2)) * 100);
  const cy = Math.round(Math.min(1, Math.max(0, y + h / 2)) * 100);
  const zoom = w > 0 && h > 0 ? Math.min(4, 1 / Math.max(w, h)) : 1;
  const parts: string[] = [];
  if (rotationDeg) parts.push(`rotate(${rotationDeg}deg)`);
  if (zoom > 1.02) parts.push(`scale(${zoom.toFixed(3)})`);
  return { objectPosition: `${cx}% ${cy}%`, transform: parts.join(" ") };
}

function cropForAspect(cut: CutRecord, aspect: Aspect): [number, number, number, number] | null | undefined {
  if (aspect === "portrait") return cut.framing?.crop_9x16;
  if (aspect === "square") return cut.framing?.crop_1x1;
  return cut.framing?.crop_16x9;
}

export function CutsV3View() {
  const token = useAuthStore((s) => s.session?.access_token);
  const files = useDriveStore((s) => s.files);
  const [aspect, setAspect] = useState<Aspect>("landscape");
  const [debugMode, setDebugMode] = useState(false);
  const [showDiscarded, setShowDiscarded] = useState(false);
  // Energy dial (cuts_v3_boundaries_v2.plan.md §D). 0 = full grounded span,
  // 1 = tightest (negative padding toward the anchor). Pure view-math over the
  // stored pace envelope -- never re-fetches or re-ingests.
  const [energy, setEnergy] = useState(0);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [data, setData] = useState<CutsV3Response | null>(null);
  const [loading, setLoading] = useState(false);
  const [kicking, setKicking] = useState(false);
  const [pollGen, setPollGen] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
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

  // Find-or-create the backend project for this exact file set.
  useEffect(() => {
    if (!token || candidateIds.length === 0) {
      setProjectId(null);
      setData(null);
      return;
    }
    let cancelled = false;
    createProject(candidateIds, token)
      .then((r) => {
        if (!cancelled) setProjectId(r.project_id);
      })
      .catch(() => {
        if (!cancelled) setProjectId(null);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, candidateKey]);

  // Fetch cuts-v3 + poll while the ingest run is in a non-terminal state.
  useEffect(() => {
    if (!token || !projectId) {
      setData(null);
      return;
    }
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;

    async function tick() {
      try {
        const r = await getCutsV3(projectId!, token!);
        if (cancelled) return;
        setData(r);
        setLoading(false);
        const status = r.ingest_run?.status;
        if (status && status !== "ready" && status !== "failed") {
          timer = setTimeout(tick, 3000);
        }
      } catch {
        if (!cancelled) setLoading(false);
      }
    }
    setLoading(true);
    tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [token, projectId, pollGen]);

  // A finished run stops polling, so a freshly-completed re-ingest wouldn't be
  // picked up while this view stays mounted. Refetch when the tab regains focus
  // (i.e. you switch back to check) so you always see the latest run.
  useEffect(() => {
    const onFocus = () => setPollGen((g) => g + 1);
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, []);

  const handleKickIngest = useCallback(async () => {
    if (!token || !projectId) return;
    setKicking(true);
    try {
      await kickIngest(projectId, token);
    } finally {
      setKicking(false);
      setPollGen((g) => g + 1);
    }
  }, [token, projectId]);

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

  const cuts = data?.cuts ?? [];

  // take_group_id -> every cut in that group, project-wide (a group can span
  // multiple clips -- "near-identical spoken lines recurring across clips").
  const takeGroups = useMemo(() => {
    const byGroup: Record<string, CutRecord[]> = {};
    for (const c of cuts) {
      if (c.take_group_id) (byGroup[c.take_group_id] ??= []).push(c);
    }
    return byGroup;
  }, [cuts]);

  // Non-winner "take" siblings never get their own tile -- they only appear
  // fanned out from their winner's badge. Outlooks keep their own tile (they
  // are different content, per the plan).
  const hiddenAsSibling = useMemo(() => {
    const hidden = new Set<string>();
    for (const group of Object.values(takeGroups)) {
      if (!group.some((c) => c.take_role === "winner")) continue;
      for (const c of group) if (c.take_role === "take") hidden.add(c.id);
    }
    return hidden;
  }, [takeGroups]);

  // High-confidence junk (camera cues, pre-roll, dead air) is hidden by
  // default -- "if in doubt, show", so low/doubtful junk stays visible inline.
  // "Show discarded" reveals the hidden ones.
  const hiddenJunkCount = useMemo(
    () => cuts.filter((c) => c.junk && c.junk_confidence === "high" && filesById[c.file_id]).length,
    [cuts, filesById]
  );

  const rows = useMemo(() => {
    const present = cuts.filter(
      (c) =>
        filesById[c.file_id] &&
        !hiddenAsSibling.has(c.id) &&
        (showDiscarded || !(c.junk && c.junk_confidence === "high"))
    );
    const byFile: Record<string, CutRecord[]> = {};
    for (const c of present) (byFile[c.file_id] ??= []).push(c);
    return Object.entries(byFile)
      .map(([fileId, list]) => ({
        fileId,
        fileName: filesById[fileId]?.name ?? fileId,
        cuts: [...list].sort((a, b) => a.src_in_ms - b.src_in_ms),
      }))
      .sort((a, b) => a.fileName.localeCompare(b.fileName));
  }, [cuts, filesById, hiddenAsSibling, showDiscarded]);

  useEffect(() => {
    setOrder((prev) => {
      const ids = rows.map((r) => r.fileId);
      const kept = prev.filter((id) => ids.includes(id));
      const added = ids.filter((id) => !kept.includes(id));
      const next = [...kept, ...added];
      return next.length === prev.length && next.every((id, i) => id === prev[i]) ? prev : next;
    });
  }, [rows]);

  const orderedRows = useMemo(() => {
    const byId: Record<string, (typeof rows)[number]> = {};
    for (const r of rows) byId[r.fileId] = r;
    return order.map((id) => byId[id]).filter(Boolean);
  }, [order, rows]);

  const toggle = useCallback((key: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const toggleGroup = useCallback((groupId: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  }, []);

  const dragRowId = useRef<string | null>(null);
  const onRowDragStart = useCallback((e: React.DragEvent, fileId: string) => {
    dragRowId.current = fileId;
    e.dataTransfer.setData(ROW_DND, fileId);
    e.dataTransfer.effectAllowed = "move";
  }, []);
  const onRowDragOver = useCallback((e: React.DragEvent, overId: string) => {
    const dragged = dragRowId.current;
    if (!dragged || dragged === overId) return;
    if (!e.dataTransfer.types.includes(ROW_DND)) return;
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

  const totalVisible = rows.reduce((n, r) => n + r.cuts.length, 0);
  const run = data?.ingest_run ?? null;
  const isProcessing = !!run && run.status !== "ready" && run.status !== "failed";

  return (
    <div>
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-2.5">
          <PillDropdown
            options={(["Landscape", "Portrait", "Square"] as const).map((l) => l)}
            value={ASPECT_LABEL[aspect]}
            onChange={(v) => {
              const found = (Object.keys(ASPECT_LABEL) as Aspect[]).find((k) => ASPECT_LABEL[k] === v);
              if (found) setAspect(found);
            }}
          />
          <button
            onClick={() => setDebugMode((v) => !v)}
            className="flex items-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-medium transition-colors hover:bg-[var(--sidebar)]"
            style={{
              borderColor: debugMode ? "var(--accent)" : "rgba(255,255,255,0.4)",
              color: debugMode ? "var(--accent)" : "var(--foreground)",
            }}
            title="Show boundary debug info (kind, word span / atom ids)"
          >
            <Bug size={13} />
            Debug
          </button>
          <button
            onClick={() => setShowDiscarded((v) => !v)}
            className="flex items-center gap-1.5 rounded-lg border px-3 py-2 text-xs font-medium transition-colors hover:bg-[var(--sidebar)]"
            style={{
              borderColor: showDiscarded ? "var(--accent)" : "rgba(255,255,255,0.4)",
              color: showDiscarded ? "var(--accent)" : "var(--foreground)",
            }}
            title="Show high-confidence junk (camera cues, pre-roll, dead air) hidden by default"
          >
            {showDiscarded ? <Eye size={13} /> : <EyeOff size={13} />}
            {showDiscarded ? "Hiding" : "Discarded"}
            {hiddenJunkCount > 0 ? ` (${hiddenJunkCount})` : ""}
          </button>
          <div
            className="flex items-center gap-2 rounded-lg border px-3 py-2"
            style={{ borderColor: "rgba(255,255,255,0.4)" }}
            title="Energy: tightens each cut inward toward its peak (speech is left intact)"
          >
            <Zap size={13} style={{ color: energy > 0 ? "var(--accent)" : "var(--muted)" }} />
            <input
              type="range"
              min={0}
              max={100}
              value={Math.round(energy * 100)}
              onChange={(e) => setEnergy(Number(e.target.value) / 100)}
              className="cuts-v3-energy h-1 w-24 cursor-pointer appearance-none rounded-full"
              style={{ accentColor: "var(--accent)" }}
              aria-label="Energy"
            />
            <span className="w-7 text-right text-xs tabular-nums" style={{ color: "var(--muted)" }}>
              {Math.round(energy * 100)}
            </span>
          </div>
        </div>
        <EditButton />
      </div>

      <IngestBanner
        run={run}
        loading={loading}
        kicking={kicking}
        hasProject={!!projectId}
        onKick={handleKickIngest}
      />

      {loading && !data && (
        <p className="py-12 text-center text-sm" style={{ color: "var(--muted)" }}>
          Loading…
        </p>
      )}

      {!loading && candidateIds.length === 0 && (
        <EmptyState
          title="No footage yet"
          body="Upload video. Once analyzed, you can run the cuts-v3 ingest here."
        />
      )}

      {!loading && candidateIds.length > 0 && data && !isProcessing && totalVisible === 0 && run?.status === "ready" && (
        <EmptyState title="No cuts" body="The ingest completed but produced no cuts for this project." />
      )}

      {totalVisible > 0 && (
        <div className="flex flex-col gap-10">
          {orderedRows.map((row) => {
            const visible = row.cuts;
            const total = visible.reduce((n, c) => n + (c.src_out_ms - c.src_in_ms), 0);
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
                  <span className="shrink-0 truncate text-xs" style={{ color: "var(--muted)" }}>
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
                    const weldableNeighbor = (n?: CutRecord) => !!n && !n.junk && !c.junk;
                    const weldLeft =
                      isSel &&
                      weldableNeighbor(prev) &&
                      selected.has(cutKey(prev)) &&
                      prev.src_out_ms === c.src_in_ms;
                    const weldRight =
                      isSel &&
                      weldableNeighbor(next) &&
                      selected.has(cutKey(next)) &&
                      c.src_out_ms === next.src_in_ms;
                    const siblingCount = c.take_group_id
                      ? (takeGroups[c.take_group_id] ?? []).filter((g) => g.take_role === "take").length
                      : 0;
                    const expanded = c.take_group_id ? expandedGroups.has(c.take_group_id) : false;
                    const siblings =
                      expanded && c.take_group_id
                        ? (takeGroups[c.take_group_id] ?? []).filter((g) => g.take_role === "take")
                        : [];
                    return (
                      <div key={cutKey(c)} className="flex shrink-0">
                        {c.junk ? (
                          <JunkStrip cut={c} width={CARD_W[aspect]} />
                        ) : (
                          <CutCardV3
                            file={filesById[c.file_id]!}
                            cut={c}
                            energy={energy}
                            getUrl={getUrl}
                            aspect={aspect}
                            debugMode={debugMode}
                            selected={isSel}
                            weldLeft={weldLeft}
                            weldRight={weldRight}
                            onToggle={() => toggle(cutKey(c))}
                            isActive={activeKey === cutKey(c)}
                            onActivate={() => setActiveKey(cutKey(c))}
                            onDeactivate={() => setActiveKey((k) => (k === cutKey(c) ? null : k))}
                            takeCount={siblingCount}
                            takeExpanded={expanded}
                            onToggleTakes={
                              siblingCount > 0 ? () => toggleGroup(c.take_group_id!) : undefined
                            }
                          />
                        )}
                        {siblings.map((s) => (
                          <CutCardV3
                            key={cutKey(s)}
                            file={filesById[s.file_id]}
                            cut={s}
                            energy={energy}
                            getUrl={getUrl}
                            aspect={aspect}
                            debugMode={debugMode}
                            selected={selected.has(cutKey(s))}
                            weldLeft={false}
                            weldRight={false}
                            onToggle={() => toggle(cutKey(s))}
                            isActive={activeKey === cutKey(s)}
                            onActivate={() => setActiveKey(cutKey(s))}
                            onDeactivate={() => setActiveKey((k) => (k === cutKey(s) ? null : k))}
                            takeCount={0}
                            takeExpanded={false}
                            dimmed
                          />
                        ))}
                      </div>
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

function IngestBanner({
  run,
  loading,
  kicking,
  hasProject,
  onKick,
}: {
  run: CutsV3Response["ingest_run"];
  loading: boolean;
  kicking: boolean;
  hasProject: boolean;
  onKick: () => void;
}) {
  if (!hasProject) return null;

  if (!loading && !run) {
    return (
      <div
        className="mb-6 flex items-center justify-between gap-3 rounded-xl border px-4 py-3"
        style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
      >
        <div className="flex items-center gap-2 text-sm">
          <Sparkles size={15} style={{ color: "var(--accent)" }} />
          <span>Not yet ingested for cuts-v3.</span>
        </div>
        <KickButton kicking={kicking} onClick={onKick} label="Run ingest" />
      </div>
    );
  }

  if (!run) return null;

  if (run.status === "failed") {
    return (
      <div
        className="mb-6 flex items-center justify-between gap-3 rounded-xl border px-4 py-3"
        style={{ borderColor: "#b91c1c", background: "rgba(185,28,28,0.08)" }}
      >
        <div className="text-sm">
          <span className="font-semibold" style={{ color: "#f87171" }}>
            Ingest failed
          </span>
          {run.error && (
            <span className="ml-2" style={{ color: "var(--muted)" }}>
              {run.error}
            </span>
          )}
        </div>
        <KickButton kicking={kicking} onClick={onKick} label="Retry" />
      </div>
    );
  }

  if (run.status !== "ready") {
    return (
      <div
        className="mb-6 flex items-center gap-3 rounded-xl border px-4 py-3 text-sm"
        style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
      >
        <Loader2 size={15} className="animate-spin" style={{ color: "var(--accent)" }} />
        <span>{STATUS_LABEL[run.status]}</span>
        {run.cost_usd != null && (
          <span style={{ color: "var(--muted)" }}>· ${run.cost_usd.toFixed(2)} so far</span>
        )}
      </div>
    );
  }

  return (
    <div className="mb-7 rounded-xl border px-4 py-3.5" style={{ borderColor: "var(--border)" }}>
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Sparkles size={15} style={{ color: "var(--accent)" }} />
          <span className="text-sm font-semibold">Project summary</span>
        </div>
        <div className="flex items-center gap-3 text-xs" style={{ color: "var(--muted)" }}>
          {run.cost_usd != null && <span>${run.cost_usd.toFixed(2)}</span>}
          <button
            onClick={onKick}
            disabled={kicking}
            className="rounded px-2 py-1 font-medium transition-colors hover:bg-[var(--sidebar)] disabled:opacity-50"
            style={{ color: "var(--foreground)" }}
          >
            {kicking ? "Re-running…" : "Re-run ingest"}
          </button>
        </div>
      </div>
      {run.project_summary && (
        <p className="mt-1.5 text-sm" style={{ color: "var(--muted)" }}>
          {run.project_summary}
        </p>
      )}
    </div>
  );
}

function KickButton({ kicking, onClick, label }: { kicking: boolean; onClick: () => void; label: string }) {
  return (
    <button
      onClick={onClick}
      disabled={kicking}
      className="flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-semibold transition-colors disabled:opacity-60"
      style={{ background: "var(--accent)", color: "var(--background)" }}
    >
      {kicking && <Loader2 size={12} className="animate-spin" />}
      {kicking ? "Starting…" : label}
    </button>
  );
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <Sparkles size={34} style={{ color: "var(--accent)" }} />
      <p className="mt-4 text-lg font-semibold">{title}</p>
      <p className="mt-1 max-w-sm text-sm" style={{ color: "var(--muted)" }}>
        {body}
      </p>
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

function JunkStrip({ cut, width }: { cut: CutRecord; width: number }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <div style={{ width: expanded ? width : 44 }} className="ml-2.5 shrink-0 transition-all">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="flex h-full min-h-[38px] w-full items-center justify-center overflow-hidden rounded-lg border px-2 text-left"
        style={{ borderColor: "var(--border)", background: "var(--sidebar)", opacity: 0.55 }}
        title={cut.junk_reason || "Junk"}
      >
        {expanded ? (
          <span className="truncate text-[11px]" style={{ color: "var(--muted)" }}>
            Junk{cut.junk_reason ? `: ${cut.junk_reason}` : ""} · {fmtDur(cut.src_out_ms - cut.src_in_ms)}
          </span>
        ) : (
          <span className="text-[10px] uppercase tracking-wide" style={{ color: "var(--muted)" }}>
            junk
          </span>
        )}
      </button>
    </div>
  );
}

function CutCardV3({
  file,
  cut,
  energy,
  getUrl,
  aspect,
  debugMode,
  selected,
  weldLeft,
  weldRight,
  onToggle,
  isActive,
  onActivate,
  onDeactivate,
  takeCount,
  takeExpanded,
  onToggleTakes,
  dimmed,
}: {
  file?: FileRecord;
  cut: CutRecord;
  energy: number;
  getUrl: (fileId: string) => Promise<string | null>;
  aspect: Aspect;
  debugMode: boolean;
  selected: boolean;
  weldLeft: boolean;
  weldRight: boolean;
  onToggle: () => void;
  isActive: boolean;
  onActivate: () => void;
  onDeactivate: () => void;
  takeCount: number;
  takeExpanded: boolean;
  onToggleTakes?: () => void;
  dimmed?: boolean;
}) {
  const [playUrl, setPlayUrl] = useState<string | null>(null);
  const [muted, setMuted] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const { inMs, outMs } = tightenedSpan(cut, energy);
  const inSec = inMs / 1000;
  const outSec = outMs / 1000;
  const heroSec = (cut.hero_ts_ms ?? (cut.src_in_ms + cut.src_out_ms) / 2) / 1000;
  const crop = cropForAspect(cut, aspect);
  const { objectPosition, transform } = cropStyle(crop, cut.framing?.rotation_deg);

  async function ensureUrl() {
    if (playUrl || !file) return;
    const url = await getUrl(file.id);
    if (url) setPlayUrl(url);
  }

  // Keep a live ref of isActive so async seek callbacks (which fire after a
  // hover has possibly already ended) never act on stale state.
  const isActiveRef = useRef(isActive);
  useEffect(() => {
    isActiveRef.current = isActive;
  }, [isActive]);

  // Robust "play from the in-point". The old code set currentTime=inSec then
  // called play() synchronously; on a second hover the previous seek-to-hero
  // was still settling, the browser coalesced the seeks, and play() resumed
  // from heroSec (mid-clip) instead of inSec. Fix: only start playback once
  // the seek to inSec has actually landed (the `seeked` event). The `#t=`
  // media fragment was also removed from the <video src> for the same reason
  // (it instructs the browser to start at heroSec, fighting the JS seek).
  const startPlayback = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    v.muted = muted;
    const play = () => v.play().catch(() => {});
    if (Math.abs(v.currentTime - inSec) < 0.05) {
      play();
      return;
    }
    const onSeeked = () => {
      v.removeEventListener("seeked", onSeeked);
      if (isActiveRef.current) play();
    };
    v.addEventListener("seeked", onSeeked);
    try {
      v.currentTime = inSec;
    } catch {
      v.removeEventListener("seeked", onSeeked);
      play();
    }
  }, [inSec, muted]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !playUrl) return;
    if (isActive) {
      startPlayback();
    } else {
      v.pause();
      try {
        v.currentTime = heroSec;
      } catch {
        /* ignore */
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive, playUrl]);

  useEffect(() => {
    const v = videoRef.current;
    if (v) v.muted = muted;
  }, [muted]);

  function onLoadedMetadata() {
    const v = videoRef.current;
    if (!v) return;
    // Respect the current hover state: if the user is already hovering when
    // metadata arrives, go straight to the in-point + play; otherwise park on
    // the hero frame as the poster.
    if (isActiveRef.current) {
      startPlayback();
    } else {
      try {
        v.currentTime = heroSec;
      } catch {
        /* ignore */
      }
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
    if (!file) return;
    const payload = JSON.stringify({
      kind: "hero",
      file_id: file.id,
      file_name: file.name,
      in_ms: inMs,
      out_ms: outMs,
      content: cut.label,
      speaker: cut.speaker,
    });
    e.dataTransfer.setData("application/x-hero-cut", payload);
    e.dataTransfer.setData("text/plain", payload);
    e.dataTransfer.effectAllowed = "copy";
  }

  return (
    <div
      style={{ width: CARD_W[aspect], marginLeft: weldLeft ? 0 : 10, opacity: dimmed ? 0.6 : 1 }}
      className="shrink-0 first:ml-0"
    >
      <div
        onClick={onToggle}
        onMouseEnter={() => {
          onActivate();
          ensureUrl();
        }}
        onMouseLeave={onDeactivate}
        draggable={!!file}
        onDragStart={onDragStart}
        className={cn(
          "group relative flex cursor-pointer items-center justify-center overflow-hidden border transition-colors",
          ASPECT_CLASS[aspect],
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
            src={playUrl}
            playsInline
            preload="metadata"
            muted={muted}
            draggable={false}
            onLoadedMetadata={onLoadedMetadata}
            onTimeUpdate={onTimeUpdate}
            className="h-full w-full bg-black"
            style={{ objectFit: "cover", objectPosition, transform }}
          />
        )}

        {/* take-role badge (top-left) */}
        {(cut.take_role || takeCount > 0) && (
          <div className="absolute left-2 top-2 z-20 flex items-center gap-1">
            {cut.take_role === "winner" && takeCount > 0 && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleTakes?.();
                }}
                className="flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-semibold"
                style={{ background: "rgba(255,255,255,0.92)", color: "#000" }}
              >
                <Layers size={11} />
                {takeCount + 1} takes
                <ChevronDown
                  size={11}
                  style={{ transform: takeExpanded ? "rotate(180deg)" : undefined }}
                />
              </button>
            )}
            {cut.take_role === "outlook" && (
              <span
                className="rounded px-1.5 py-0.5 text-[11px] font-semibold"
                style={{ background: "rgba(0,0,0,0.6)", color: "#fff" }}
              >
                outlook
              </span>
            )}
          </div>
        )}

        {selected && (
          <span
            className="absolute right-2 top-2 z-20 flex h-6 w-6 items-center justify-center rounded-full"
            style={{ background: "var(--accent)", color: "var(--background)" }}
          >
            <Check size={14} />
          </span>
        )}

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

        <span
          className="pointer-events-none absolute left-1/2 top-1/2 z-10 flex h-11 w-11 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full opacity-100 shadow-lg transition-opacity group-hover:opacity-0"
          style={{ background: "var(--accent)" }}
        >
          <Play size={20} className="ml-0.5" fill="currentColor" style={{ color: "var(--background)" }} />
        </span>

        <span className="absolute bottom-2 right-2 z-10 rounded bg-black/70 px-1.5 py-0.5 text-[11px] font-medium text-white">
          {fmtDur(outMs - inMs)}
        </span>
        <div className="absolute bottom-2 left-2 z-10 flex items-center gap-1">
          {cut.speaker && (
            <span className="rounded bg-black/60 px-1.5 py-0.5 text-[11px] font-medium text-white">
              {cut.speaker}
            </span>
          )}
          {cut.on_camera != null && (
            <span
              className="rounded bg-black/60 px-1.5 py-0.5 text-[11px] font-medium text-white"
              title={cut.on_camera ? "Speaker visible on camera" : "Speaker not on camera"}
            >
              {cut.on_camera ? "on cam" : "off cam"}
            </span>
          )}
        </div>
      </div>

      {/* label + summary line */}
      <div className="mt-2 px-0.5">
        <p className="truncate text-xs font-semibold" style={{ color: "var(--foreground)" }}>
          {cut.label || "—"}
        </p>
        {cut.summary && (
          <p className="mt-0.5 line-clamp-2 text-[11px]" style={{ color: "var(--muted)" }}>
            {cut.summary}
          </p>
        )}
        {debugMode && (
          <p className="mt-1 text-[10px] uppercase tracking-wide" style={{ color: "var(--muted)" }}>
            {cut.kind}
            {cut.kind === "speech" && cut.word_span ? ` [${cut.word_span[0]}-${cut.word_span[1]}]` : ""}
            {cut.kind === "video" && cut.atom_ids ? ` atoms[${cut.atom_ids.join(",")}]` : ""}
            {cut.take_role ? ` · ${cut.take_role}` : ""}
          </p>
        )}
      </div>
    </div>
  );
}
