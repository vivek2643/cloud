"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useDriveStore, type AiTimelineData } from "@/stores/drive-store";
import { useAuthStore } from "@/stores/auth-store";
import { getFilePlaybackUrl, getFileShots, ensureProject, commitEdl, type CommitClip } from "@/lib/api";
import { Film, ChevronDown, ChevronUp, Play, Pause, SkipBack, ZoomIn, ZoomOut, Trash2, Check, Loader2, AlertCircle } from "lucide-react";

const MIN_CLIP_MS = 200;
const EPS_MS = 45;
const RULER_H = 14;
const TRACK_H = 40;
const SHADES = ["#f97316", "#fb923c", "#ea580c", "#fdba74", "#c2410c"];

interface DockClip {
  id: string;
  shot_id: string | null;
  file_id: string | null;
  file_name: string | null;
  source_url: string | null;
  source_in_ms: number;
  source_out_ms: number;
  timeline_in_ms: number;
  timeline_out_ms: number;
  duration_ms: number;
}

type SaveState = "idle" | "saving" | "saved" | "error" | "unsaveable";

type DragState =
  | { mode: "scrub" }
  | { mode: "trim-in" | "trim-out"; id: string; startX: number; origIn: number; origOut: number }
  | { mode: "move"; id: string; idx: number; startX: number; dx: number; targetIdx: number; moved: boolean };

function uid(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `c_${Date.now()}_${Math.random().toString(36).slice(2)}`;
}

function fmtMs(ms: number): string {
  const total = Math.max(0, Math.round(ms));
  const m = Math.floor(total / 60000);
  const s = Math.floor((total % 60000) / 1000);
  const tenths = Math.floor((total % 1000) / 100);
  return `${m}:${s.toString().padStart(2, "0")}.${tenths}`;
}

function fmtTc(ms: number): string {
  const total = Math.max(0, Math.round(ms));
  const m = Math.floor(total / 60000);
  const s = Math.floor((total % 60000) / 1000);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function recompute(clips: DockClip[]): DockClip[] {
  let cursor = 0;
  return clips.map((c) => {
    const dur = Math.max(MIN_CLIP_MS, c.source_out_ms - c.source_in_ms);
    const out = { ...c, timeline_in_ms: cursor, timeline_out_ms: cursor + dur, duration_ms: dur };
    cursor += dur;
    return out;
  });
}

// Merge back-to-back clips that come from the same file and are contiguous in
// source time (within ~1 frame). A continuous span that the analyzer split
// into multiple shots becomes a SINGLE clip -- one ffmpeg segment, no concat
// seam -- so playback stays smooth. Adjacent-but-not-contiguous ranges (jump
// cuts) are left alone.
const COALESCE_EPS_MS = 40;
function coalesce(clips: DockClip[]): DockClip[] {
  if (clips.length <= 1) return clips;
  const out: DockClip[] = [];
  for (const c of clips) {
    const prev = out[out.length - 1];
    if (
      prev &&
      prev.file_id &&
      c.file_id &&
      prev.file_id === c.file_id &&
      Math.abs(c.source_in_ms - prev.source_out_ms) <= COALESCE_EPS_MS
    ) {
      out[out.length - 1] = { ...prev, source_out_ms: c.source_out_ms };
    } else {
      out.push({ ...c });
    }
  }
  return out;
}

function signature(t: AiTimelineData | null): string {
  if (!t) return "";
  return t.clips.map((c) => `${c.file_id}:${c.source_in_ms}-${c.source_out_ms}`).join("|");
}

export function AiTimelineDock() {
  const aiTimeline = useDriveStore((s) => s.aiTimeline);
  const visible = useDriveStore((s) => s.aiTimelineVisible);
  const show = useDriveStore((s) => s.showAiTimeline);
  const hide = useDriveStore((s) => s.hideAiTimeline);
  const aiPanelOpen = useDriveStore((s) => s.aiPanelOpen);
  const aiScopeFileIds = useDriveStore((s) => s.aiScopeFileIds);
  const previewVideoEl = useDriveStore((s) => s.previewVideoEl);
  const token = useAuthStore((s) => s.session?.access_token);

  const [clips, setClips] = useState<DockClip[]>([]);
  const [playheadMs, setPlayheadMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [zoom, setZoom] = useState(1);
  const [laneW, setLaneW] = useState(800);
  const [saveState, setSaveState] = useState<SaveState>("idle");

  // The project to commit into. An AI cut carries one; for an edit started from
  // an empty timeline we lazily find-or-create one on first save (see commitNow).
  const projectIdRef = useRef<string | null>(aiTimeline?.projectId ?? null);
  const baseVersionRef = useRef<string | null>(aiTimeline?.baseVersionId ?? null);
  const saveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const clipsRef = useRef<DockClip[]>([]);
  const playheadRef = useRef(0);
  const videoElRef = useRef<HTMLVideoElement | null>(null);
  const activeIdxRef = useRef(0);
  const activeSrcRef = useRef<string | null>(null);
  const loadTokenRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const urlCacheRef = useRef<Map<string, string>>(new Map());
  const wrapRef = useRef<HTMLDivElement | null>(null);

  const setEditorHasClips = useDriveStore((s) => s.setEditorHasClips);
  useEffect(() => { clipsRef.current = clips; }, [clips]);
  useEffect(() => { setEditorHasClips(clips.length > 0); }, [clips.length, setEditorHasClips]);
  useEffect(() => { playheadRef.current = playheadMs; }, [playheadMs]);
  useEffect(() => { videoElRef.current = previewVideoEl; activeSrcRef.current = null; }, [previewVideoEl]);

  const totalMs = clips.length ? clips[clips.length - 1].timeline_out_ms : 0;

  // Resolve presigned source URLs for any clips that still need one.
  const resolveUrls = useCallback(
    async (cs: DockClip[]) => {
      if (!token) return;
      const ids = Array.from(new Set(cs.map((c) => c.file_id).filter((x): x is string => !!x)));
      for (const fid of ids) {
        if (urlCacheRef.current.has(fid)) continue;
        try {
          const { url } = await getFilePlaybackUrl(fid, token);
          urlCacheRef.current.set(fid, url);
        } catch {
          /* ignore -- clip stays unplayable */
        }
      }
      setClips((prev) =>
        prev.map((c) =>
          c.source_url || !c.file_id ? c : { ...c, source_url: urlCacheRef.current.get(c.file_id) ?? null },
        ),
      );
    },
    [token],
  );

  // Rebuild the working timeline whenever the AI produces a new cut.
  const sig = signature(aiTimeline);
  useEffect(() => {
    if (!aiTimeline || aiTimeline.clips.length === 0) {
      setClips([]);
      setPlayheadMs(0);
      return;
    }
    const mapped: DockClip[] = aiTimeline.clips.map((c) => ({
      id: uid(),
      shot_id: c.shot_id ?? null,
      file_id: c.file_id ?? null,
      file_name: c.file_name ?? null,
      source_url: c.file_id ? urlCacheRef.current.get(c.file_id) ?? null : null,
      source_in_ms: c.source_in_ms,
      source_out_ms: c.source_out_ms,
      timeline_in_ms: 0,
      timeline_out_ms: 0,
      duration_ms: 0,
    }));
    // Collapse contiguous same-file shots into single clips up front.
    const base = recompute(coalesce(mapped));
    setClips(base);
    setPlayheadMs(0);
    activeSrcRef.current = null;
    // A fresh AI cut becomes the new base version to chain manual edits off.
    projectIdRef.current = aiTimeline.projectId ?? projectIdRef.current;
    baseVersionRef.current = aiTimeline.baseVersionId ?? null;
    if (saveTimerRef.current) { clearTimeout(saveTimerRef.current); saveTimerRef.current = null; }
    setSaveState("idle");
    void resolveUrls(base);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sig]);

  // Track available width to fit the whole sequence on screen.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setLaneW(e.contentRect.width);
    });
    ro.observe(el);
    setLaneW(el.clientWidth);
    return () => ro.disconnect();
  }, []);

  const fitPxPerSec = totalMs > 0 ? Math.max(6, Math.min(240, (laneW - 24) / (totalMs / 1000))) : 40;
  const pxPerSec = fitPxPerSec * zoom;

  // --- Playback driving the registered monitor element ---

  const fileMsForSeq = useCallback((seqMs: number) => {
    const cs = clipsRef.current;
    if (cs.length === 0) return null;
    let idx = cs.findIndex((c) => seqMs < c.timeline_out_ms);
    if (idx === -1) idx = cs.length - 1;
    const c = cs[idx];
    return { idx, clip: c, fileMs: c.source_in_ms + Math.max(0, seqMs - c.timeline_in_ms) };
  }, []);

  const seekTo = useCallback(
    async (seqMs: number, autoplay: boolean) => {
      const v = videoElRef.current;
      const hit = fileMsForSeq(seqMs);
      if (!v || !hit) return;
      const { idx, clip, fileMs } = hit;
      activeIdxRef.current = idx;
      if (!clip.source_url) return;
      const myToken = ++loadTokenRef.current;
      if (activeSrcRef.current !== clip.source_url) {
        activeSrcRef.current = clip.source_url;
        v.src = clip.source_url;
        await new Promise<void>((res) => {
          const done = () => { v.removeEventListener("loadedmetadata", done); res(); };
          v.addEventListener("loadedmetadata", done);
        });
        if (myToken !== loadTokenRef.current) return;
      }
      try { v.currentTime = Math.max(0, fileMs / 1000); } catch { /* pre-metadata */ }
      if (autoplay) { try { await v.play(); } catch { /* blocked */ } }
    },
    [fileMsForSeq],
  );

  const tick = useCallback(() => {
    const v = videoElRef.current;
    const cs = clipsRef.current;
    const idx = activeIdxRef.current;
    const c = cs[idx];
    if (!v || !c) { setPlaying(false); return; }
    const posMs = v.currentTime * 1000;
    if (posMs >= c.source_out_ms - EPS_MS) {
      const next = idx + 1;
      if (next >= cs.length) {
        setPlayheadMs(cs[cs.length - 1].timeline_out_ms);
        setPlaying(false);
        return;
      }
      activeIdxRef.current = next;
      void seekTo(cs[next].timeline_in_ms, true);
    } else {
      setPlayheadMs(c.timeline_in_ms + (posMs - c.source_in_ms));
    }
    rafRef.current = requestAnimationFrame(tick);
  }, [seekTo]);

  useEffect(() => {
    if (!playing) {
      // Paused -- either the user hit pause or tick() reached the end of the
      // sequence and called setPlaying(false). Stop the RAF loop AND the
      // underlying <video>. The pause() must be UNCONDITIONAL: React runs the
      // previous render's cleanup (which nulls rafRef) before this body, so
      // gating pause() on `rafRef.current != null` would skip it and the
      // monitor would keep playing past the clip / end of the timeline.
      if (rafRef.current != null) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
      videoElRef.current?.pause();
      return;
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current != null) { cancelAnimationFrame(rafRef.current); rafRef.current = null; }
    };
  }, [playing, tick]);

  async function togglePlay() {
    if (clips.length === 0) return;
    if (playing) { setPlaying(false); return; }
    let start = playheadRef.current;
    if (start >= totalMs - 1) start = 0;
    setPlayheadMs(start);
    await seekTo(start, true);
    setPlaying(true);
  }

  const scrubTo = useCallback(
    (seqMs: number) => {
      const clamped = Math.max(0, Math.min(seqMs, totalMs));
      setPlayheadMs(clamped);
      if (!playing) void seekTo(clamped, false);
    },
    [totalMs, playing, seekTo],
  );

  // Refresh the monitor frame after edits while paused.
  useEffect(() => {
    if (playing || clips.length === 0) return;
    const ms = Math.min(playheadRef.current, totalMs);
    void seekTo(ms, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clips, playing, previewVideoEl]);

  // --- Auto-persist: a manual edit is the new latest version, no save click ---

  const commitNow = useCallback(
    async (cs: DockClip[]) => {
      if (!token) return;
      if (cs.length === 0) return;
      // Save the coalesced form so contiguous same-file spans persist (and
      // render) as a single segment.
      const merged = coalesce(cs);
      // Every clip must map to a shot to be committable. Drag-added raw files
      // (no shot_id) can't persist yet -- flag that rather than failing.
      if (merged.some((c) => !c.shot_id)) {
        setSaveState("unsaveable");
        return;
      }
      setSaveState("saving");
      // Resolve a project to commit into. Edits started from an empty timeline
      // have none yet -- find-or-create one keyed on the editor's source set so
      // a later chat turn lands on the same project.
      let pid = projectIdRef.current;
      if (!pid) {
        const fileIds = aiScopeFileIds.length
          ? aiScopeFileIds
          : Array.from(new Set(merged.map((c) => c.file_id).filter((x): x is string => !!x)));
        try {
          const proj = await ensureProject(fileIds, token);
          pid = proj.id;
          projectIdRef.current = pid;
        } catch {
          setSaveState("error");
          return;
        }
      }
      const payload: CommitClip[] = merged.map((c) => ({
        id: c.id,
        shot_id: c.shot_id as string,
        source_in_ms: Math.round(c.source_in_ms),
        source_out_ms: Math.round(c.source_out_ms),
      }));
      try {
        const res = await commitEdl(pid, payload, token, {
          authorKind: "user",
          parentId: baseVersionRef.current,
          commitMsg: "Manual timeline edit",
        });
        baseVersionRef.current = res.edl_version_id;
        setSaveState("saved");
      } catch {
        setSaveState("error");
      }
    },
    [token, aiScopeFileIds],
  );

  const scheduleSave = useCallback(
    (cs: DockClip[]) => {
      if (saveTimerRef.current) clearTimeout(saveTimerRef.current);
      saveTimerRef.current = setTimeout(() => {
        saveTimerRef.current = null;
        void commitNow(cs);
      }, 900);
    },
    [commitNow],
  );

  useEffect(() => () => { if (saveTimerRef.current) clearTimeout(saveTimerRef.current); }, []);

  // The dock stays mounted across panel open/close. When the editor closes, flush
  // any pending debounced save (so the last edit persists) and then clear all
  // working state, so reopening for a different file set never inherits a prior
  // session's clips or commits into its project.
  useEffect(() => {
    if (aiPanelOpen) return;
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current);
      saveTimerRef.current = null;
      void commitNow(clipsRef.current);
    }
    setClips([]);
    setPlayheadMs(0);
    setPlaying(false);
    setSelectedId(null);
    setSaveState("idle");
    projectIdRef.current = null;
    baseVersionRef.current = null;
    activeSrcRef.current = null;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aiPanelOpen]);

  // --- Editing ops (local; drives the live preview + auto-persists) ---

  function markEdited(next: DockClip[]) {
    const recomputed = recompute(next);
    setClips(recomputed);
    scheduleSave(recomputed);
  }

  function deleteById(id: string) {
    markEdited(clips.filter((c) => c.id !== id));
    if (selectedId === id) setSelectedId(null);
  }

  function reorderClip(from: number, to: number) {
    const arr = [...clips];
    const [moved] = arr.splice(from, 1);
    arr.splice(to, 0, moved);
    markEdited(arr);
  }

  function setTrimById(id: string, edge: "in" | "out", absMs: number) {
    markEdited(
      clips.map((c) => {
        if (c.id !== id) return c;
        if (edge === "in") {
          const ni = Math.max(0, Math.min(absMs, c.source_out_ms - MIN_CLIP_MS));
          return { ...c, source_in_ms: ni };
        }
        const no = Math.max(c.source_in_ms + MIN_CLIP_MS, absMs);
        return { ...c, source_out_ms: no };
      }),
    );
  }

  async function dropMedia(payload: { file_id: string; file_name?: string; duration_seconds?: number }, atMs: number) {
    let url = urlCacheRef.current.get(payload.file_id) ?? null;
    if (!url && token) {
      try {
        const r = await getFilePlaybackUrl(payload.file_id, token);
        url = r.url;
        urlCacheRef.current.set(payload.file_id, url);
      } catch { /* unplayable */ }
    }

    // A raw whole file has no shot_id of its own, but the EDL is shot-keyed, so a
    // bare file clip can't be committed. Expand the file into its real shots and
    // let coalesce() merge the contiguous span back into one continuous clip that
    // carries a genuine shot_id (committable) and renders as a single segment.
    let inserted: DockClip[] = [];
    if (token) {
      try {
        const res = await getFileShots(payload.file_id, token);
        const shots = (res.shots ?? [])
          .filter((s) => s.start_ms != null && s.end_ms != null && (s.end_ms as number) > (s.start_ms as number))
          .sort((a, b) => (a.start_ms as number) - (b.start_ms as number));
        inserted = shots.map((s) => ({
          id: uid(),
          shot_id: s.shot_id,
          file_id: payload.file_id,
          file_name: payload.file_name ?? null,
          source_url: url,
          source_in_ms: s.start_ms as number,
          source_out_ms: s.end_ms as number,
          timeline_in_ms: 0,
          timeline_out_ms: 0,
          duration_ms: 0,
        }));
      } catch { /* fall through to single-clip fallback below */ }
    }

    // Fallback: no shots available -> insert a single bare-file clip covering the
    // full duration. It plays fine; auto-save will flag it as unsaveable until a
    // shot id can be resolved.
    if (inserted.length === 0) {
      const durMs = Math.max(MIN_CLIP_MS, Math.round((payload.duration_seconds ?? 5) * 1000));
      inserted = [{
        id: uid(),
        shot_id: null,
        file_id: payload.file_id,
        file_name: payload.file_name ?? null,
        source_url: url,
        source_in_ms: 0,
        source_out_ms: durMs,
        timeline_in_ms: 0,
        timeline_out_ms: 0,
        duration_ms: durMs,
      }];
    }

    const cs = clipsRef.current;
    let idx = cs.findIndex((c) => atMs < (c.timeline_in_ms + c.timeline_out_ms) / 2);
    if (idx === -1) idx = cs.length;
    const arr = [...cs];
    arr.splice(idx, 0, ...inserted);
    markEdited(coalesce(arr));
    setSelectedId(inserted[0].id);
  }

  // Show whenever the editor is open -- even before any AI cut -- so a raw file
  // can be dragged onto an empty timeline to start a manual edit from scratch.
  if (!aiPanelOpen) return null;

  return (
    <div ref={wrapRef} className="shrink-0 border-t" style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}>
      {/* Transport / toggle row */}
      <div className="flex items-center gap-2 px-3 py-1.5">
        <button
          onClick={visible ? hide : show}
          className="flex items-center gap-1.5 text-xs font-medium transition-colors hover:opacity-80"
          style={{ color: "var(--foreground)" }}
          title={visible ? "Hide timeline" : "Show timeline"}
        >
          <Film size={13} style={{ color: "var(--accent)" }} />
          {visible ? "Hide timeline" : "Show timeline"}
          {visible ? <ChevronDown size={13} /> : <ChevronUp size={13} />}
        </button>

        {visible && (
          <>
            <div className="mx-1 h-4 w-px" style={{ background: "var(--border)" }} />
            <button onClick={() => scrubTo(0)} className="rounded p-1 transition-colors hover:bg-black/10" title="Go to start" style={{ color: "var(--muted)" }}>
              <SkipBack size={14} />
            </button>
            <button
              onClick={togglePlay}
              className="flex h-6 w-6 items-center justify-center rounded-full text-white"
              style={{ background: "var(--accent)" }}
              title={playing ? "Pause" : "Play"}
            >
              {playing ? <Pause size={13} fill="white" /> : <Play size={13} className="ml-0.5" fill="white" />}
            </button>
            <span className="font-mono text-[11px]" style={{ color: "var(--muted)" }}>
              {fmtMs(playheadMs)} / {fmtMs(totalMs)}
            </span>
            <span className="ml-auto text-[11px]" style={{ color: "var(--muted)" }}>
              {clips.length} clip{clips.length === 1 ? "" : "s"}
            </span>
            <SaveBadge state={saveState} />
            <button onClick={() => setZoom((z) => Math.max(0.25, z / 1.5))} className="rounded p-1 transition-colors hover:bg-black/10" title="Zoom out" style={{ color: "var(--muted)" }}>
              <ZoomOut size={14} />
            </button>
            <button onClick={() => setZoom((z) => Math.min(8, z * 1.5))} className="rounded p-1 transition-colors hover:bg-black/10" title="Zoom in" style={{ color: "var(--muted)" }}>
              <ZoomIn size={14} />
            </button>
          </>
        )}
      </div>

      {visible && (
        <DockTrack
          clips={clips}
          pxPerSec={pxPerSec}
          totalMs={totalMs}
          playheadMs={playheadMs}
          selectedId={selectedId}
          onScrub={scrubTo}
          onSelect={setSelectedId}
          onReorder={reorderClip}
          onTrim={setTrimById}
          onDelete={deleteById}
          onDropMedia={dropMedia}
        />
      )}
    </div>
  );
}

function SaveBadge({ state }: { state: SaveState }) {
  if (state === "idle") return null;
  const map = {
    saving: { icon: <Loader2 size={11} className="animate-spin" />, text: "Saving…", color: "var(--muted)" },
    saved: { icon: <Check size={11} />, text: "Saved", color: "var(--accent)" },
    error: { icon: <AlertCircle size={11} />, text: "Save failed", color: "var(--danger)" },
    unsaveable: { icon: <AlertCircle size={11} />, text: "Drag-added clip not saved", color: "var(--danger)" },
  } as const;
  const m = map[state];
  return (
    <span className="flex items-center gap-1 text-[11px]" style={{ color: m.color }} title={m.text}>
      {m.icon}
      {m.text}
    </span>
  );
}

function DockTrack({
  clips,
  pxPerSec,
  totalMs,
  playheadMs,
  selectedId,
  onScrub,
  onSelect,
  onReorder,
  onTrim,
  onDelete,
  onDropMedia,
}: {
  clips: DockClip[];
  pxPerSec: number;
  totalMs: number;
  playheadMs: number;
  selectedId: string | null;
  onScrub: (ms: number) => void;
  onSelect: (id: string) => void;
  onReorder: (from: number, to: number) => void;
  onTrim: (id: string, edge: "in" | "out", absMs: number) => void;
  onDelete: (id: string) => void;
  onDropMedia: (payload: { file_id: string; file_name?: string; duration_seconds?: number }, atMs: number) => void;
}) {
  const laneRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);
  const [dropHint, setDropHint] = useState(false);

  const pxPerMs = pxPerSec / 1000;
  const contentW = Math.max(320, totalMs * pxPerMs + 24);

  const xToMs = useCallback(
    (clientX: number) => {
      const lane = laneRef.current;
      if (!lane) return 0;
      const rect = lane.getBoundingClientRect();
      return (clientX - rect.left + lane.scrollLeft) / pxPerMs;
    },
    [pxPerMs],
  );

  const tickSec = useMemo(() => {
    const raw = 80 / pxPerSec;
    const steps = [0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300];
    return steps.find((s) => s >= raw) ?? 300;
  }, [pxPerSec]);
  const ticks = useMemo(() => {
    const out: number[] = [];
    for (let s = 0; s * 1000 <= totalMs + tickSec * 1000; s += tickSec) out.push(s);
    return out;
  }, [totalMs, tickSec]);

  useEffect(() => {
    function onMove(e: PointerEvent) {
      const st = dragRef.current;
      if (!st) return;
      if (st.mode === "scrub") {
        onScrub(xToMs(e.clientX));
      } else if (st.mode === "trim-in" || st.mode === "trim-out") {
        const deltaMs = (e.clientX - st.startX) / pxPerMs;
        if (st.mode === "trim-in") onTrim(st.id, "in", st.origIn + deltaMs);
        else onTrim(st.id, "out", st.origOut + deltaMs);
      } else if (st.mode === "move") {
        const dx = e.clientX - st.startX;
        const moved = st.moved || Math.abs(dx) > 4;
        const ms = xToMs(e.clientX);
        let targetIdx = clips.findIndex((c) => ms < (c.timeline_in_ms + c.timeline_out_ms) / 2);
        if (targetIdx === -1) targetIdx = clips.length - 1;
        const nextSt: DragState = { ...st, dx, moved, targetIdx };
        dragRef.current = nextSt;
        setDrag(nextSt);
      }
    }
    function onUp() {
      const st = dragRef.current;
      if (st && st.mode === "move" && st.moved && st.targetIdx !== st.idx) onReorder(st.idx, st.targetIdx);
      dragRef.current = null;
      setDrag(null);
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    }
    if (drag) {
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    }
    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerup", onUp);
    };
  }, [drag, clips, pxPerMs, xToMs, onScrub, onTrim, onReorder]);

  function startScrub(e: React.PointerEvent) {
    const st: DragState = { mode: "scrub" };
    dragRef.current = st;
    setDrag(st);
    onScrub(xToMs(e.clientX));
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setDropHint(false);
    const raw = e.dataTransfer.getData("application/edso-clip") || e.dataTransfer.getData("text/plain");
    if (!raw) return;
    try {
      const payload = JSON.parse(raw);
      if (payload?.file_id) onDropMedia(payload, xToMs(e.clientX));
    } catch { /* ignore malformed payload */ }
  }

  return (
    <div
      className="overflow-x-auto"
      onDragOver={(e) => { e.preventDefault(); setDropHint(true); }}
      onDragLeave={() => setDropHint(false)}
      onDrop={handleDrop}
      style={{ boxShadow: dropHint ? "inset 0 0 0 2px var(--accent)" : undefined }}
    >
      <div ref={laneRef} className="relative select-none" style={{ width: contentW, height: RULER_H + TRACK_H + 12 }}>
        {/* Ruler */}
        <div className="absolute left-0 top-0 w-full cursor-text" style={{ height: RULER_H }} onPointerDown={startScrub}>
          {ticks.map((s) => (
            <div key={s} className="absolute top-0 h-full" style={{ left: s * 1000 * pxPerMs }}>
              <div className="h-1.5 w-px" style={{ background: "var(--border)" }} />
              <span className="absolute left-0.5 top-0 text-[8px]" style={{ color: "var(--muted)" }}>{fmtTc(s * 1000)}</span>
            </div>
          ))}
        </div>

        {/* Clip lane */}
        <div
          className="absolute left-0"
          style={{ top: RULER_H, height: TRACK_H, width: contentW }}
          onPointerDown={(e) => { if (e.target === e.currentTarget) startScrub(e); }}
        >
          {clips.length === 0 && (
            <div className="flex h-full items-center justify-center text-[11px]" style={{ color: "var(--muted)" }}>
              Drag a video here to start your cut
            </div>
          )}
          {clips.map((c, idx) => {
            const left = c.timeline_in_ms * pxPerMs;
            const w = Math.max(8, c.duration_ms * pxPerMs);
            const isSel = c.id === selectedId;
            const isDragging = drag?.mode === "move" && drag.id === c.id;
            const translate = isDragging ? drag.dx : 0;
            return (
              <div
                key={c.id}
                className="absolute top-1 overflow-hidden rounded-md"
                style={{
                  left,
                  width: w,
                  height: TRACK_H - 8,
                  border: isSel ? "2px solid var(--accent)" : "1px solid rgba(0,0,0,0.2)",
                  transform: translate ? `translateX(${translate}px)` : undefined,
                  opacity: isDragging ? 0.85 : 1,
                  zIndex: isDragging ? 30 : isSel ? 20 : 10,
                  background: SHADES[idx % SHADES.length],
                  cursor: "grab",
                  boxShadow: isDragging ? "0 6px 20px rgba(0,0,0,0.35)" : undefined,
                }}
                onPointerDown={(e) => {
                  e.stopPropagation();
                  onSelect(c.id);
                  const st: DragState = { mode: "move", id: c.id, idx, startX: e.clientX, dx: 0, targetIdx: idx, moved: false };
                  dragRef.current = st;
                  setDrag(st);
                }}
              >
                {w > 40 && (
                  <div className="absolute inset-x-1 bottom-0.5 truncate text-[9px] font-medium text-white drop-shadow">
                    {c.file_name || "clip"}
                  </div>
                )}
                {w > 64 && (
                  <div className="absolute left-1 top-0.5 rounded bg-black/40 px-1 text-[8px] text-white">
                    {(c.duration_ms / 1000).toFixed(1)}s
                  </div>
                )}
                {isSel && (
                  <>
                    <div
                      className="absolute left-0 top-0 h-full w-1.5 cursor-ew-resize bg-white/70"
                      title="Trim head"
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        const st: DragState = { mode: "trim-in", id: c.id, startX: e.clientX, origIn: c.source_in_ms, origOut: c.source_out_ms };
                        dragRef.current = st; setDrag(st);
                      }}
                    />
                    <div
                      className="absolute right-0 top-0 h-full w-1.5 cursor-ew-resize bg-white/70"
                      title="Trim tail"
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        const st: DragState = { mode: "trim-out", id: c.id, startX: e.clientX, origIn: c.source_in_ms, origOut: c.source_out_ms };
                        dragRef.current = st; setDrag(st);
                      }}
                    />
                    <button
                      onClick={(e) => { e.stopPropagation(); onDelete(c.id); }}
                      onPointerDown={(e) => e.stopPropagation()}
                      className="absolute right-2 top-0.5 rounded bg-black/60 p-0.5 text-white hover:bg-black/80"
                      title="Delete clip"
                    >
                      <Trash2 size={10} />
                    </button>
                  </>
                )}
              </div>
            );
          })}

          {drag?.mode === "move" && drag.moved && (() => {
            const ti = drag.targetIdx;
            const xMs = ti < clips.length ? clips[ti].timeline_in_ms : totalMs;
            return <div className="absolute top-0 h-full w-0.5" style={{ left: xMs * pxPerMs, background: "var(--accent)", zIndex: 40 }} />;
          })()}
        </div>

        {/* Playhead */}
        <div className="pointer-events-none absolute top-0" style={{ left: playheadMs * pxPerMs, height: RULER_H + TRACK_H }}>
          <div className="h-full w-0.5" style={{ background: "#ef4444" }} />
          <div className="absolute -left-1 top-0 h-0 w-0" style={{ borderLeft: "4px solid transparent", borderRight: "4px solid transparent", borderTop: "5px solid #ef4444" }} />
        </div>
      </div>
    </div>
  );
}
