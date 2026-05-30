"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Image from "next/image";
import { useAuthStore } from "@/stores/auth-store";
import {
  getLatestEdl,
  getEdlVersion,
  listEdlVersions,
  commitEdl,
  searchShotsForProject,
  pollRenderUntilDone,
  startEdlAgent,
  streamChatTurn,
  cancelChatTurn,
  type EnrichedClip,
  type EdlVersionMeta,
  type SearchShot,
  type AgentResult,
} from "@/lib/api";
import {
  Trash2,
  Plus,
  Search,
  Save,
  Download,
  Loader2,
  History,
  ChevronLeft,
  Film,
  Scissors,
  X,
  Sparkles,
  Wand2,
  Play,
  Pause,
  SkipBack,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

type WorkingClip = EnrichedClip;

const MIN_CLIP_MS = 200;
const EPS_MS = 45;

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

// Sequentially re-stack clips so timeline positions follow the cut order.
function recompute(clips: WorkingClip[]): WorkingClip[] {
  let cursor = 0;
  return clips.map((c) => {
    const dur = Math.max(MIN_CLIP_MS, c.source_out_ms - c.source_in_ms);
    const out = { ...c, timeline_in_ms: cursor, timeline_out_ms: cursor + dur, duration_ms: dur };
    cursor += dur;
    return out;
  });
}

function recoverProjectIdFromChat(): string | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem("edso_edit_chat_v1");
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    const turns: unknown[] = Array.isArray(parsed?.turns) ? parsed.turns : [];
    for (let i = turns.length - 1; i >= 0; i--) {
      const t = turns[i] as { role?: string; project_id?: string };
      if (t?.role === "assistant" && t.project_id) return t.project_id;
    }
  } catch {
    // ignore
  }
  return null;
}

export default function TimelineEditorPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const session = useAuthStore((s) => s.session);
  const token = session?.access_token;

  const [projectId, setProjectId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [clips, setClips] = useState<WorkingClip[]>([]);
  const [baseVersionId, setBaseVersionId] = useState<string | null>(null);
  const [loadedVersion, setLoadedVersion] = useState<EdlVersionMeta | null>(null);
  const [dirty, setDirty] = useState(false);
  const [fps, setFps] = useState(30);

  const [versions, setVersions] = useState<EdlVersionMeta[]>([]);
  const [showHistory, setShowHistory] = useState(false);

  const [saving, setSaving] = useState(false);
  const [renderStatus, setRenderStatus] = useState<string | null>(null);
  const [renderPct, setRenderPct] = useState(0);
  const [renderUrl, setRenderUrl] = useState<string | null>(null);
  const [renderError, setRenderError] = useState<string | null>(null);

  const [showSearch, setShowSearch] = useState(false);
  const [searchQ, setSearchQ] = useState("");
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<SearchShot[]>([]);

  // AI agent (Phase 3)
  const [showAgent, setShowAgent] = useState(false);
  const [agentInstruction, setAgentInstruction] = useState("");
  const [agentRunning, setAgentRunning] = useState(false);
  const [agentTurnId, setAgentTurnId] = useState<string | null>(null);
  const [agentProgress, setAgentProgress] = useState<{ label: string; pct: number } | null>(null);
  const [proposal, setProposal] = useState<AgentResult | null>(null);
  const [applying, setApplying] = useState(false);
  const agentAbortRef = useRef<AbortController | null>(null);

  const renderAbortRef = useRef<AbortController | null>(null);

  // NLE state
  const [pxPerSec, setPxPerSec] = useState(90);
  const [playheadMs, setPlayheadMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const activeIdxRef = useRef(0);
  const activeSrcRef = useRef<string | null>(null);
  const loadTokenRef = useRef(0);
  const clipsRef = useRef<WorkingClip[]>([]);
  const playheadRef = useRef(0);
  useEffect(() => {
    clipsRef.current = clips;
  }, [clips]);
  useEffect(() => {
    playheadRef.current = playheadMs;
  }, [playheadMs]);

  // Resolve project id (URL first, then chat recovery).
  useEffect(() => {
    const fromUrl = searchParams.get("project_id");
    setProjectId(fromUrl || recoverProjectIdFromChat());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const applyEdl = useCallback(
    (edl: { clips: EnrichedClip[]; version: EdlVersionMeta; fps: number }) => {
      setClips(recompute(edl.clips.map((c) => ({ ...c }))));
      setBaseVersionId(edl.version.id);
      setLoadedVersion(edl.version);
      setFps(edl.fps || 30);
      setDirty(false);
      setPlayheadMs(0);
      setPlaying(false);
      activeSrcRef.current = null;
      activeIdxRef.current = 0;
    },
    [],
  );

  const hydrate = useCallback(async () => {
    if (!projectId || !token) return;
    setLoading(true);
    setError(null);
    try {
      const [edl, vers] = await Promise.all([
        getLatestEdl(projectId, token),
        listEdlVersions(projectId, token),
      ]);
      setVersions(vers);
      if (edl) {
        applyEdl(edl);
      } else {
        setClips([]);
        setBaseVersionId(null);
        setLoadedVersion(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load timeline");
    } finally {
      setLoading(false);
    }
  }, [projectId, token, applyEdl]);

  useEffect(() => {
    if (projectId === null) {
      setLoading(false);
      return;
    }
    void hydrate();
  }, [projectId, hydrate]);

  const totalMs = useMemo(
    () => (clips.length ? clips[clips.length - 1].timeline_out_ms : 0),
    [clips],
  );
  const selectedClip = useMemo(
    () => clips.find((c) => c.id === selectedId) ?? null,
    [clips, selectedId],
  );

  // --- Editing ops (local; committed on Save) ---

  function markDirty(next: WorkingClip[]) {
    setClips(recompute(next));
    setDirty(true);
    // We intentionally do NOT reset activeSrcRef here: seekTo only reloads the
    // <video> src when the source file actually changes, so trims (same file)
    // stay smooth and only re-seek currentTime.
  }

  function deleteById(id: string) {
    markDirty(clips.filter((c) => c.id !== id));
    if (selectedId === id) setSelectedId(null);
  }

  function reorderClip(fromIdx: number, toIdx: number) {
    if (fromIdx === toIdx || fromIdx < 0 || toIdx < 0) return;
    const next = [...clips];
    const [moved] = next.splice(fromIdx, 1);
    next.splice(toIdx, 0, moved);
    markDirty(next);
  }

  function setTrimById(id: string, edge: "in" | "out", absMs: number) {
    const next = clips.map((c) => {
      if (c.id !== id) return c;
      let inMs = c.source_in_ms;
      let outMs = c.source_out_ms;
      if (edge === "in") {
        inMs = Math.max(0, Math.min(absMs, outMs - MIN_CLIP_MS));
      } else {
        const upper = c.file_duration_ms ?? Number.MAX_SAFE_INTEGER;
        outMs = Math.min(upper, Math.max(absMs, inMs + MIN_CLIP_MS));
      }
      return { ...c, source_in_ms: Math.round(inMs), source_out_ms: Math.round(outMs) };
    });
    markDirty(next);
  }

  function insertShot(shot: SearchShot) {
    const clip: WorkingClip = {
      id: uid(),
      shot_id: shot.shot_id,
      file_id: shot.file_id,
      file_name: shot.file_name,
      source_in_ms: shot.start_ms,
      source_out_ms: shot.end_ms,
      timeline_in_ms: 0,
      timeline_out_ms: 0,
      duration_ms: shot.duration_ms,
      shot_start_ms: shot.start_ms,
      shot_end_ms: shot.end_ms,
      file_duration_ms: null,
      thumbnail_url: shot.thumbnail_url,
      transcript_text: shot.transcript_text,
      source_url: null,
    };
    markDirty([...clips, clip]);
  }

  // --- Preview monitor playback ---

  const fileMsForSeq = useCallback((seqMs: number) => {
    const cs = clipsRef.current;
    if (cs.length === 0) return null;
    let idx = cs.findIndex((c) => seqMs < c.timeline_out_ms);
    if (idx === -1) idx = cs.length - 1;
    const c = cs[idx];
    const fileMs = c.source_in_ms + Math.max(0, seqMs - c.timeline_in_ms);
    return { idx, clip: c, fileMs };
  }, []);

  const seekTo = useCallback(
    async (seqMs: number, autoplay: boolean) => {
      const v = videoRef.current;
      const hit = fileMsForSeq(seqMs);
      if (!v || !hit) return;
      const { idx, clip, fileMs } = hit;
      activeIdxRef.current = idx;
      const myToken = ++loadTokenRef.current;
      if (clip.source_url && activeSrcRef.current !== clip.source_url) {
        activeSrcRef.current = clip.source_url;
        v.src = clip.source_url;
        await new Promise<void>((res) => {
          const done = () => {
            v.removeEventListener("loadedmetadata", done);
            res();
          };
          v.addEventListener("loadedmetadata", done);
        });
        if (myToken !== loadTokenRef.current) return;
      }
      if (!clip.source_url) return;
      try {
        v.currentTime = Math.max(0, fileMs / 1000);
      } catch {
        // seeking before metadata; ignore
      }
      if (autoplay) {
        try {
          await v.play();
        } catch {
          // autoplay may be blocked; ignore
        }
      }
    },
    [fileMsForSeq],
  );

  const tick = useCallback(() => {
    const v = videoRef.current;
    const cs = clipsRef.current;
    const idx = activeIdxRef.current;
    const c = cs[idx];
    if (!v || !c) {
      setPlaying(false);
      return;
    }
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
    if (playing) {
      rafRef.current = requestAnimationFrame(tick);
    } else if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      videoRef.current?.pause();
    }
    return () => {
      if (rafRef.current != null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [playing, tick]);

  async function togglePlay() {
    if (clips.length === 0) return;
    if (playing) {
      setPlaying(false);
      return;
    }
    let start = playheadRef.current;
    if (start >= totalMs - 1) start = 0; // restart from top if at end
    setPlayheadMs(start);
    await seekTo(start, true);
    setPlaying(true);
  }

  // Scrub: move playhead and show the corresponding frame (paused).
  const scrubTo = useCallback(
    (seqMs: number) => {
      const clamped = Math.max(0, Math.min(seqMs, totalMs));
      setPlayheadMs(clamped);
      if (!playing) void seekTo(clamped, false);
    },
    [totalMs, playing, seekTo],
  );

  // Keep the monitor showing the frame under the playhead after edits / load.
  useEffect(() => {
    if (playing || clips.length === 0) return;
    const ms = Math.min(playheadRef.current, totalMs);
    if (ms !== playheadRef.current) setPlayheadMs(ms);
    void seekTo(ms, false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clips, playing, seekTo]);

  // --- Search / versions / save / agent (unchanged wiring) ---

  async function doSearch() {
    if (!projectId || !token || !searchQ.trim()) return;
    setSearching(true);
    try {
      const res = await searchShotsForProject(projectId, searchQ.trim(), token, 24);
      setSearchResults(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search failed");
    } finally {
      setSearching(false);
    }
  }

  async function loadVersion(versionId: string) {
    if (!token) return;
    if (dirty && !window.confirm("Discard unsaved changes and load this version?")) return;
    try {
      const edl = await getEdlVersion(versionId, token);
      applyEdl(edl);
      setShowHistory(false);
      setRenderUrl(null);
      setRenderStatus(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load version");
    }
  }

  async function saveAndRender() {
    if (!projectId || !token || clips.length === 0 || saving) return;
    setSaving(true);
    setError(null);
    setRenderError(null);
    setRenderUrl(null);
    setRenderStatus("queued");
    setRenderPct(0);
    try {
      const commit = await commitEdl(
        projectId,
        clips.map((c) => ({
          id: c.id,
          shot_id: c.shot_id,
          source_in_ms: c.source_in_ms,
          source_out_ms: c.source_out_ms,
        })),
        token,
        { commitMsg: "Manual edit", parentId: baseVersionId, fps },
      );
      setBaseVersionId(commit.edl_version_id);
      setDirty(false);
      void listEdlVersions(projectId, token).then(setVersions).catch(() => {});

      if (commit.render_id) {
        const abort = new AbortController();
        renderAbortRef.current = abort;
        const row = await pollRenderUntilDone(commit.render_id, token, {
          intervalMs: 1500,
          abortSignal: abort.signal,
          onUpdate: (r) => {
            setRenderStatus(r.status);
            setRenderPct(r.progress_pct);
          },
        });
        if (row.status === "done") setRenderUrl(row.output_url ?? null);
        else if (row.status === "failed") setRenderError(row.error || "Render failed");
      } else {
        setRenderStatus(null);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
      setRenderStatus(null);
    } finally {
      setSaving(false);
      renderAbortRef.current = null;
    }
  }

  async function runAgent() {
    if (!projectId || !token || !agentInstruction.trim() || agentRunning) return;
    setAgentRunning(true);
    setProposal(null);
    setError(null);
    setAgentProgress({ label: "Starting", pct: 0 });
    try {
      const { turn_id } = await startEdlAgent(projectId, agentInstruction.trim(), token, baseVersionId);
      setAgentTurnId(turn_id);
      const abort = new AbortController();
      agentAbortRef.current = abort;
      let settled = false;
      await streamChatTurn(
        turn_id,
        token,
        (evt) => {
          if (evt.type === "phase") {
            setAgentProgress({ label: evt.label || "Working", pct: evt.pct ?? 0 });
          } else if (evt.type === "done") {
            settled = true;
            setProposal(evt.result as unknown as AgentResult);
          } else if (evt.type === "cancelled") {
            settled = true;
            setError("AI edit cancelled.");
          } else if (evt.type === "error") {
            settled = true;
            setError(evt.message);
          }
        },
        abort.signal,
      );
      if (!settled) setError("AI edit ended unexpectedly. Try again.");
    } catch (e) {
      if (e instanceof DOMException && e.name === "AbortError") {
        // cancelled
      } else {
        setError(e instanceof Error ? e.message : "AI edit failed");
      }
    } finally {
      setAgentRunning(false);
      setAgentProgress(null);
      setAgentTurnId(null);
      agentAbortRef.current = null;
    }
  }

  async function cancelAgent() {
    if (!agentTurnId || !token) return;
    try {
      await cancelChatTurn(agentTurnId, token);
    } catch {
      // the stream's cancelled event settles things
    }
  }

  async function applyProposal() {
    if (!projectId || !token || !proposal || applying) return;
    setApplying(true);
    setError(null);
    try {
      const commit = await commitEdl(
        projectId,
        proposal.proposed_clips.map((c) => ({
          id: c.id,
          shot_id: c.shot_id,
          source_in_ms: c.source_in_ms,
          source_out_ms: c.source_out_ms,
        })),
        token,
        {
          authorKind: "claude",
          commitMsg: proposal.summary?.slice(0, 120) || "AI edit",
          parentId: proposal.base_version_id ?? baseVersionId,
        },
      );
      setProposal(null);
      setShowAgent(false);
      const edl = await getEdlVersion(commit.edl_version_id, token);
      applyEdl(edl);
      void listEdlVersions(projectId, token).then(setVersions).catch(() => {});
      if (commit.render_id) {
        setRenderUrl(null);
        setRenderStatus("queued");
        setRenderPct(0);
        const row = await pollRenderUntilDone(commit.render_id, token, {
          intervalMs: 1500,
          onUpdate: (r) => {
            setRenderStatus(r.status);
            setRenderPct(r.progress_pct);
          },
        });
        if (row.status === "done") setRenderUrl(row.output_url ?? null);
        else if (row.status === "failed") setRenderError(row.error || "Render failed");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to apply AI edit");
    } finally {
      setApplying(false);
    }
  }

  function handleDownloadMp4() {
    if (!renderUrl) return;
    const a = document.createElement("a");
    a.href = renderUrl;
    a.download = "timeline.mp4";
    a.target = "_blank";
    a.rel = "noopener";
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  function zoom(dir: -1 | 1) {
    setPxPerSec((p) => Math.max(20, Math.min(400, Math.round(p * (dir === 1 ? 1.25 : 0.8)))));
  }

  // --- Empty / no-project states ---

  if (!projectId && !loading) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 px-6 text-center">
        <Film size={40} style={{ color: "var(--muted)" }} />
        <h1 className="text-lg font-semibold">No project to edit</h1>
        <p className="max-w-sm text-sm" style={{ color: "var(--muted)" }}>
          Open the timeline editor from a chat that produced an edit, or generate one first.
        </p>
        <button
          onClick={() => router.push("/edit")}
          className="mt-2 flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm transition-colors hover:opacity-80"
          style={{ borderColor: "var(--border)" }}
        >
          <Sparkles size={14} /> Go to AI Editor
        </button>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div
        className="flex flex-wrap items-center justify-between gap-2 border-b px-6 py-3"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex items-center gap-2">
          <button
            onClick={() => router.push("/edit")}
            className="flex items-center gap-1 rounded-lg border px-2 py-1.5 text-xs transition-colors hover:opacity-80"
            style={{ borderColor: "var(--border)" }}
            title="Back to AI Editor"
          >
            <ChevronLeft size={14} /> Chat
          </button>
          <Scissors size={18} style={{ color: "var(--accent)" }} />
          <div>
            <h1 className="text-base font-semibold leading-tight">Timeline Editor</h1>
            <div className="text-xs" style={{ color: "var(--muted)" }}>
              {clips.length} clip{clips.length === 1 ? "" : "s"} · {fmtMs(totalMs)}
              {dirty && <span style={{ color: "var(--accent)" }}> · unsaved</span>}
              {loadedVersion && (
                <span>
                  {" "}
                  · base{" "}
                  <span style={{ color: loadedVersion.author_kind === "user" ? "var(--accent)" : "var(--muted)" }}>
                    {loadedVersion.author_kind}
                  </span>
                </span>
              )}
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              setShowAgent((v) => !v);
              setShowSearch(false);
              setShowHistory(false);
            }}
            className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs transition-colors hover:opacity-80"
            style={{
              borderColor: showAgent ? "var(--accent)" : "var(--border)",
              color: showAgent ? "var(--accent)" : "inherit",
            }}
            title="Ask AI to edit this timeline"
          >
            <Sparkles size={14} /> Ask AI
          </button>
          <button
            onClick={() => setShowSearch((v) => !v)}
            className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs transition-colors hover:opacity-80"
            style={{ borderColor: "var(--border)" }}
            title="Insert a clip from your footage"
          >
            <Plus size={14} /> Insert
          </button>
          <button
            onClick={() => setShowHistory((v) => !v)}
            className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs transition-colors hover:opacity-80"
            style={{ borderColor: "var(--border)" }}
            title="Version history"
          >
            <History size={14} /> History
          </button>
          <button
            onClick={handleDownloadMp4}
            disabled={!renderUrl}
            className="flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs transition-colors hover:opacity-80 disabled:opacity-40"
            style={{ borderColor: "var(--border)" }}
          >
            <Download size={14} /> MP4
          </button>
          <button
            onClick={saveAndRender}
            disabled={saving || clips.length === 0 || !dirty}
            className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium text-white transition-opacity hover:opacity-90 disabled:opacity-40"
            style={{ background: "var(--accent)" }}
            title={dirty ? "Commit a new version and render" : "No changes to save"}
          >
            {saving ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
            Save &amp; Render
          </button>
        </div>
      </div>

      {error && (
        <div
          className="border-b px-6 py-2 text-sm"
          style={{ borderColor: "#ef4444", color: "#ef4444", background: "rgba(239,68,68,0.05)" }}
        >
          {error}
        </div>
      )}

      <div className="flex flex-1 overflow-hidden">
        {/* Main editing column: monitor (top) + timeline (bottom) */}
        <div className="flex min-w-0 flex-1 flex-col">
          {proposal && (
            <div className="border-b px-6 py-3" style={{ borderColor: "var(--border)" }}>
              <DiffOverlay
                proposal={proposal}
                applying={applying}
                onApply={applyProposal}
                onDiscard={() => setProposal(null)}
              />
            </div>
          )}

          {loading ? (
            <div className="flex flex-1 items-center justify-center gap-2 text-sm" style={{ color: "var(--muted)" }}>
              <Loader2 size={16} className="animate-spin" /> Loading timeline…
            </div>
          ) : clips.length === 0 ? (
            <div className="flex flex-1 items-center justify-center">
              <div
                className="max-w-md rounded-xl border border-dashed p-8 text-center"
                style={{ borderColor: "var(--border)" }}
              >
                <Film size={36} className="mx-auto mb-3" style={{ color: "var(--muted)" }} />
                <p className="text-sm font-medium">This timeline is empty</p>
                <p className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
                  Click <strong>Insert</strong> to add clips from your footage.
                </p>
              </div>
            </div>
          ) : (
            <>
              {/* Program monitor */}
              <div
                className="flex min-h-0 flex-1 items-center justify-center p-4"
                style={{ background: "#0a0a0a" }}
              >
                <video
                  ref={videoRef}
                  playsInline
                  className="max-h-full max-w-full rounded-lg"
                  style={{ background: "#000" }}
                  onClick={() => void togglePlay()}
                />
              </div>

              {/* Transport */}
              <div
                className="flex items-center gap-3 border-t px-4 py-2"
                style={{ borderColor: "var(--border)" }}
              >
                <button
                  onClick={() => scrubTo(0)}
                  className="rounded-md p-1.5 opacity-80 hover:opacity-100"
                  title="Go to start"
                >
                  <SkipBack size={16} />
                </button>
                <button
                  onClick={() => void togglePlay()}
                  className="flex items-center justify-center rounded-full p-2 text-white"
                  style={{ background: "var(--accent)" }}
                  title={playing ? "Pause" : "Play"}
                >
                  {playing ? <Pause size={16} /> : <Play size={16} />}
                </button>
                <span className="font-mono text-xs tabular-nums" style={{ color: "var(--muted)" }}>
                  {fmtMs(playheadMs)} / {fmtMs(totalMs)}
                </span>
                <div className="ml-auto flex items-center gap-1">
                  <button onClick={() => zoom(-1)} className="rounded-md border p-1.5 hover:opacity-80" style={{ borderColor: "var(--border)" }} title="Zoom out">
                    <ZoomOut size={14} />
                  </button>
                  <button onClick={() => zoom(1)} className="rounded-md border p-1.5 hover:opacity-80" style={{ borderColor: "var(--border)" }} title="Zoom in">
                    <ZoomIn size={14} />
                  </button>
                </div>
              </div>

              {/* Timeline track */}
              <TimelineTrack
                clips={clips}
                pxPerSec={pxPerSec}
                totalMs={totalMs}
                playheadMs={playheadMs}
                selectedId={selectedId}
                onScrub={scrubTo}
                onSelect={(id) => setSelectedId(id)}
                onReorder={reorderClip}
                onTrim={setTrimById}
                onDelete={deleteById}
              />
            </>
          )}

          {/* Final render strip (export output) */}
          {(renderUrl || renderStatus) && (
            <div
              className="flex items-center gap-3 border-t px-4 py-2 text-xs"
              style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
            >
              {renderUrl ? (
                <>
                  <span className="font-medium" style={{ color: "var(--accent)" }}>
                    Export ready
                  </span>
                  <button onClick={handleDownloadMp4} className="flex items-center gap-1 underline opacity-80 hover:opacity-100">
                    <Download size={12} /> Download MP4
                  </button>
                </>
              ) : (
                <span className="flex items-center gap-2" style={{ color: "var(--muted)" }}>
                  <Loader2 size={12} className="animate-spin" />
                  {renderError ? (
                    <span style={{ color: "#ef4444" }}>{renderError}</span>
                  ) : (
                    <span>Rendering export… {renderPct}%</span>
                  )}
                </span>
              )}
            </div>
          )}
        </div>

        {/* AI agent panel */}
        {showAgent && (
          <div className="flex w-80 shrink-0 flex-col border-l" style={{ borderColor: "var(--border)", background: "var(--background)" }}>
            <div className="flex items-center justify-between border-b px-3 py-2" style={{ borderColor: "var(--border)" }}>
              <span className="flex items-center gap-1.5 text-sm font-medium">
                <Sparkles size={14} style={{ color: "var(--accent)" }} /> Ask AI to edit
              </span>
              <button onClick={() => setShowAgent(false)} className="opacity-60 hover:opacity-100">
                <X size={16} />
              </button>
            </div>
            <div className="flex flex-col gap-2 p-3">
              <textarea
                value={agentInstruction}
                onChange={(e) => setAgentInstruction(e.target.value)}
                rows={4}
                placeholder="e.g. Remove the filler words, tighten the pauses, and put the product shot first."
                className="w-full resize-none rounded-md border bg-transparent px-2 py-1.5 text-sm outline-none"
                style={{ borderColor: "var(--border)" }}
                disabled={agentRunning}
              />
              {agentRunning ? (
                <div className="flex flex-col gap-2">
                  <div className="flex items-center justify-between text-xs" style={{ color: "var(--muted)" }}>
                    <span className="flex items-center gap-1.5">
                      <Loader2 size={12} className="animate-spin" />
                      {agentProgress?.label || "Working"}
                    </span>
                    <button onClick={cancelAgent} className="flex items-center gap-1 opacity-70 hover:opacity-100">
                      <X size={12} /> Cancel
                    </button>
                  </div>
                  <div className="h-1 w-full overflow-hidden rounded-full" style={{ background: "var(--sidebar)" }}>
                    <div className="h-full rounded-full transition-all duration-300" style={{ width: `${agentProgress?.pct ?? 0}%`, background: "var(--accent)" }} />
                  </div>
                </div>
              ) : (
                <button
                  onClick={runAgent}
                  disabled={!agentInstruction.trim() || clips.length === 0}
                  className="flex items-center justify-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium text-white transition-opacity hover:opacity-90 disabled:opacity-40"
                  style={{ background: "var(--accent)" }}
                >
                  <Wand2 size={14} /> Propose edits
                </button>
              )}
              <p className="text-[11px]" style={{ color: "var(--muted)" }}>
                The AI proposes cut-only changes (reorder, trim, delete, insert). You review a diff before anything is applied.
              </p>
            </div>
          </div>
        )}

        {/* Insert / search panel */}
        {showSearch && (
          <div className="flex w-80 shrink-0 flex-col border-l" style={{ borderColor: "var(--border)", background: "var(--background)" }}>
            <div className="flex items-center justify-between border-b px-3 py-2" style={{ borderColor: "var(--border)" }}>
              <span className="text-sm font-medium">Insert from footage</span>
              <button onClick={() => setShowSearch(false)} className="opacity-60 hover:opacity-100">
                <X size={16} />
              </button>
            </div>
            <div className="flex items-center gap-2 border-b px-3 py-2" style={{ borderColor: "var(--border)" }}>
              <input
                value={searchQ}
                onChange={(e) => setSearchQ(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && doSearch()}
                placeholder="e.g. person smiling, wide shot…"
                className="flex-1 rounded-md border bg-transparent px-2 py-1.5 text-sm outline-none"
                style={{ borderColor: "var(--border)" }}
              />
              <button
                onClick={doSearch}
                disabled={searching || !searchQ.trim()}
                className="rounded-md border p-1.5 transition-colors hover:opacity-80 disabled:opacity-40"
                style={{ borderColor: "var(--border)" }}
              >
                {searching ? <Loader2 size={14} className="animate-spin" /> : <Search size={14} />}
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-3">
              {searchResults.length === 0 ? (
                <p className="text-xs" style={{ color: "var(--muted)" }}>
                  Search your footage semantically, then click a result to append it.
                </p>
              ) : (
                <div className="space-y-2">
                  {searchResults.map((s) => (
                    <button
                      key={s.shot_id}
                      onClick={() => insertShot(s)}
                      className="flex w-full items-center gap-2 rounded-lg border p-2 text-left transition-colors hover:opacity-80"
                      style={{ borderColor: "var(--border)" }}
                      title="Append to timeline"
                    >
                      <Thumb url={s.thumbnail_url} w={56} h={32} />
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-xs font-medium">{s.file_name}</div>
                        <div className="text-[11px]" style={{ color: "var(--muted)" }}>
                          {fmtMs(s.start_ms)}–{fmtMs(s.end_ms)} · {(s.duration_ms / 1000).toFixed(1)}s
                        </div>
                      </div>
                      <Plus size={14} style={{ color: "var(--accent)" }} />
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Version history panel */}
        {showHistory && (
          <div className="flex w-80 shrink-0 flex-col border-l" style={{ borderColor: "var(--border)", background: "var(--background)" }}>
            <div className="flex items-center justify-between border-b px-3 py-2" style={{ borderColor: "var(--border)" }}>
              <span className="text-sm font-medium">Version history</span>
              <button onClick={() => setShowHistory(false)} className="opacity-60 hover:opacity-100">
                <X size={16} />
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-3">
              {versions.length === 0 ? (
                <p className="text-xs" style={{ color: "var(--muted)" }}>No versions yet.</p>
              ) : (
                <div className="space-y-1.5">
                  {versions.map((v) => {
                    const isBase = v.id === baseVersionId;
                    return (
                      <button
                        key={v.id}
                        onClick={() => loadVersion(v.id)}
                        className="flex w-full items-start gap-2 rounded-lg border p-2 text-left transition-colors hover:opacity-80"
                        style={{
                          borderColor: isBase ? "var(--accent)" : "var(--border)",
                          background: isBase ? "rgba(59,130,246,0.06)" : "transparent",
                        }}
                      >
                        {v.author_kind === "user" ? (
                          <Scissors size={14} className="mt-0.5 shrink-0" style={{ color: "var(--accent)" }} />
                        ) : (
                          <Wand2 size={14} className="mt-0.5 shrink-0" style={{ color: "var(--muted)" }} />
                        )}
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-xs font-medium">
                            {v.commit_msg || (v.author_kind === "claude" ? "AI edit" : "Edit")}
                          </div>
                          <div className="text-[11px]" style={{ color: "var(--muted)" }}>
                            {v.author_kind} · {v.clip_count} clips
                            {v.created_at ? ` · ${new Date(v.created_at).toLocaleTimeString()}` : ""}
                          </div>
                        </div>
                        {isBase && (
                          <span className="text-[10px]" style={{ color: "var(--accent)" }}>
                            base
                          </span>
                        )}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Selected-clip inspector strip */}
      {selectedClip && (
        <div
          className="flex items-center gap-3 border-t px-4 py-2 text-xs"
          style={{ borderColor: "var(--border)", background: "var(--background)" }}
        >
          <Thumb url={selectedClip.thumbnail_url} w={48} h={28} />
          <span className="truncate font-medium">{selectedClip.file_name || selectedClip.shot_id.slice(0, 8)}</span>
          <span style={{ color: "var(--muted)" }}>
            src {fmtMs(selectedClip.source_in_ms)}–{fmtMs(selectedClip.source_out_ms)} ·{" "}
            {(selectedClip.duration_ms / 1000).toFixed(1)}s
          </span>
          {selectedClip.transcript_text && (
            <span className="min-w-0 flex-1 truncate italic" style={{ color: "var(--muted)" }}>
              “{selectedClip.transcript_text.slice(0, 90)}”
            </span>
          )}
          <button
            onClick={() => deleteById(selectedClip.id)}
            className="ml-auto flex items-center gap-1 rounded-md border px-2 py-1 hover:opacity-100"
            style={{ borderColor: "var(--border)", color: "#ef4444" }}
          >
            <Trash2 size={12} /> Delete
          </button>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Timeline track (horizontal NLE lane)
// ---------------------------------------------------------------------------

type DragState =
  | { mode: "scrub" }
  | { mode: "trim-in" | "trim-out"; id: string; startX: number; origIn: number; origOut: number }
  | { mode: "move"; id: string; idx: number; startX: number; dx: number; targetIdx: number; moved: boolean };

const RULER_H = 22;
const TRACK_H = 76;

function TimelineTrack({
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
}: {
  clips: WorkingClip[];
  pxPerSec: number;
  totalMs: number;
  playheadMs: number;
  selectedId: string | null;
  onScrub: (ms: number) => void;
  onSelect: (id: string) => void;
  onReorder: (from: number, to: number) => void;
  onTrim: (id: string, edge: "in" | "out", absMs: number) => void;
  onDelete: (id: string) => void;
}) {
  const laneRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<DragState | null>(null);
  const [drag, setDrag] = useState<DragState | null>(null);

  const pxPerMs = pxPerSec / 1000;
  const contentW = Math.max(320, totalMs * pxPerMs + 24);

  const xToMs = useCallback(
    (clientX: number) => {
      const lane = laneRef.current;
      if (!lane) return 0;
      const rect = lane.getBoundingClientRect();
      const x = clientX - rect.left + lane.scrollLeft;
      return x / pxPerMs;
    },
    [pxPerMs],
  );

  // Ruler tick interval (seconds) chosen so labels don't crowd.
  const tickSec = useMemo(() => {
    const targetPx = 80;
    const raw = targetPx / pxPerSec;
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
        // Determine target index from pointer position over clip centers.
        const ms = xToMs(e.clientX);
        let targetIdx = clips.findIndex((c) => ms < (c.timeline_in_ms + c.timeline_out_ms) / 2);
        if (targetIdx === -1) targetIdx = clips.length - 1;
        const nextSt: DragState = { ...st, dx, moved, targetIdx };
        dragRef.current = nextSt;
        setDrag(nextSt);
        return;
      }
    }
    function onUp() {
      const st = dragRef.current;
      if (st && st.mode === "move" && st.moved && st.targetIdx !== st.idx) {
        onReorder(st.idx, st.targetIdx);
      }
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

  return (
    <div
      className="shrink-0 overflow-x-auto border-t"
      style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
    >
      <div ref={laneRef} className="relative select-none" style={{ width: contentW, height: RULER_H + TRACK_H + 16 }}>
        {/* Ruler */}
        <div
          className="absolute left-0 top-0 w-full cursor-text"
          style={{ height: RULER_H }}
          onPointerDown={startScrub}
        >
          {ticks.map((s) => (
            <div key={s} className="absolute top-0 h-full" style={{ left: s * 1000 * pxPerMs }}>
              <div className="h-2 w-px" style={{ background: "var(--border)" }} />
              <span className="absolute left-1 top-1 text-[9px]" style={{ color: "var(--muted)" }}>
                {fmtTc(s * 1000)}
              </span>
            </div>
          ))}
        </div>

        {/* Clip lane */}
        <div
          className="absolute left-0"
          style={{ top: RULER_H, height: TRACK_H, width: contentW }}
          onPointerDown={(e) => {
            // Clicking empty lane scrubs.
            if (e.target === e.currentTarget) startScrub(e);
          }}
        >
          {clips.map((c, idx) => {
            const left = c.timeline_in_ms * pxPerMs;
            const w = Math.max(8, c.duration_ms * pxPerMs);
            const isSel = c.id === selectedId;
            const isDragging = drag?.mode === "move" && drag.id === c.id;
            const translate = isDragging ? drag.dx : 0;
            return (
              <div
                key={c.id}
                className="absolute top-1 overflow-hidden rounded-md border"
                style={{
                  left,
                  width: w,
                  height: TRACK_H - 8,
                  borderColor: isSel ? "var(--accent)" : "var(--border)",
                  borderWidth: isSel ? 2 : 1,
                  transform: translate ? `translateX(${translate}px)` : undefined,
                  opacity: isDragging ? 0.85 : 1,
                  zIndex: isDragging ? 30 : isSel ? 20 : 10,
                  background: "var(--background)",
                  cursor: "grab",
                  boxShadow: isDragging ? "0 6px 20px rgba(0,0,0,0.35)" : undefined,
                }}
                onPointerDown={(e) => {
                  e.stopPropagation();
                  onSelect(c.id);
                  const st: DragState = {
                    mode: "move",
                    id: c.id,
                    idx,
                    startX: e.clientX,
                    dx: 0,
                    targetIdx: idx,
                    moved: false,
                  };
                  dragRef.current = st;
                  setDrag(st);
                }}
                onClick={() => onSelect(c.id)}
              >
                {c.thumbnail_url ? (
                  <Image
                    src={c.thumbnail_url}
                    alt=""
                    fill
                    unoptimized
                    sizes="200px"
                    className="object-cover opacity-60"
                  />
                ) : null}
                <div className="absolute inset-0" style={{ background: "linear-gradient(180deg, rgba(0,0,0,0.05), rgba(0,0,0,0.45))" }} />
                {w > 46 && (
                  <div className="absolute inset-x-1 bottom-1 truncate text-[10px] font-medium text-white drop-shadow">
                    {c.file_name || c.shot_id.slice(0, 6)}
                  </div>
                )}
                {w > 70 && (
                  <div className="absolute left-1 top-1 rounded bg-black/50 px-1 text-[9px] text-white">
                    {(c.duration_ms / 1000).toFixed(1)}s
                  </div>
                )}

                {/* Trim handles (selected clip only) */}
                {isSel && (
                  <>
                    <div
                      className="absolute left-0 top-0 h-full w-2 cursor-ew-resize"
                      style={{ background: "var(--accent)" }}
                      title="Trim head"
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        const st: DragState = {
                          mode: "trim-in",
                          id: c.id,
                          startX: e.clientX,
                          origIn: c.source_in_ms,
                          origOut: c.source_out_ms,
                        };
                        dragRef.current = st;
                        setDrag(st);
                      }}
                    />
                    <div
                      className="absolute right-0 top-0 h-full w-2 cursor-ew-resize"
                      style={{ background: "var(--accent)" }}
                      title="Trim tail"
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        const st: DragState = {
                          mode: "trim-out",
                          id: c.id,
                          startX: e.clientX,
                          origIn: c.source_in_ms,
                          origOut: c.source_out_ms,
                        };
                        dragRef.current = st;
                        setDrag(st);
                      }}
                    />
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        onDelete(c.id);
                      }}
                      className="absolute right-3 top-1 rounded bg-black/60 p-0.5 text-white hover:bg-black/80"
                      title="Delete clip"
                    >
                      <Trash2 size={11} />
                    </button>
                  </>
                )}
              </div>
            );
          })}

          {/* Insertion indicator while reordering */}
          {drag?.mode === "move" && drag.moved && (() => {
            const ti = drag.targetIdx;
            const xMs = ti < clips.length ? clips[ti].timeline_in_ms : totalMs;
            return (
              <div
                className="absolute top-0 h-full w-0.5"
                style={{ left: xMs * pxPerMs, background: "var(--accent)", zIndex: 40 }}
              />
            );
          })()}
        </div>

        {/* Playhead */}
        <div
          className="pointer-events-none absolute top-0"
          style={{ left: playheadMs * pxPerMs, height: RULER_H + TRACK_H }}
        >
          <div className="h-full w-0.5" style={{ background: "#ef4444" }} />
          <div
            className="absolute -left-1 top-0 h-0 w-0"
            style={{
              borderLeft: "5px solid transparent",
              borderRight: "5px solid transparent",
              borderTop: "6px solid #ef4444",
            }}
          />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function DiffOverlay({
  proposal,
  applying,
  onApply,
  onDiscard,
}: {
  proposal: AgentResult;
  applying: boolean;
  onApply: () => void;
  onDiscard: () => void;
}) {
  const diff = proposal.diff;
  const addedIds = new Set(diff.added.map((c) => c.id));
  const trimmedIds = new Set(diff.trimmed.map((t) => t.clip_id));
  const movedIds = new Set(diff.moved.map((m) => m.clip_id));

  const badge = (id: string) => {
    if (addedIds.has(id)) return { t: "NEW", c: "#22c55e" };
    if (trimmedIds.has(id)) return { t: "TRIMMED", c: "#f59e0b" };
    if (movedIds.has(id)) return { t: "MOVED", c: "#3b82f6" };
    return null;
  };

  return (
    <div className="mx-auto max-w-3xl rounded-xl border p-4" style={{ borderColor: "var(--accent)", background: "rgba(59,130,246,0.05)" }}>
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-start gap-2">
          <Sparkles size={16} className="mt-0.5 shrink-0" style={{ color: "var(--accent)" }} />
          <div>
            <div className="text-sm font-semibold">Proposed changes</div>
            <p className="mt-0.5 text-xs" style={{ color: "var(--muted)" }}>
              {proposal.summary}
            </p>
          </div>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <button
            onClick={onDiscard}
            disabled={applying}
            className="rounded-lg border px-3 py-1.5 text-xs transition-colors hover:opacity-80 disabled:opacity-50"
            style={{ borderColor: "var(--border)" }}
          >
            Discard
          </button>
          <button
            onClick={onApply}
            disabled={applying || !diff.changed}
            className="flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-medium text-white transition-opacity hover:opacity-90 disabled:opacity-40"
            style={{ background: "var(--accent)" }}
          >
            {applying ? <Loader2 size={14} className="animate-spin" /> : <Save size={14} />}
            Apply &amp; Render
          </button>
        </div>
      </div>

      {!diff.changed ? (
        <p className="mt-3 text-xs" style={{ color: "var(--muted)" }}>
          The AI didn&apos;t change anything.
        </p>
      ) : (
        <div className="mt-3 space-y-1.5">
          {proposal.proposed_enriched.map((c, idx) => {
            const b = badge(c.id);
            const trim = diff.trimmed.find((t) => t.clip_id === c.id);
            return (
              <div key={c.id} className="flex items-center gap-2 rounded-lg border p-2" style={{ borderColor: "var(--border)", background: "var(--background)" }}>
                <span className="w-5 text-center text-[10px]" style={{ color: "var(--muted)" }}>
                  {idx + 1}
                </span>
                <Thumb url={c.thumbnail_url} w={56} h={32} />
                <div className="min-w-0 flex-1">
                  <div className="truncate text-xs font-medium">{c.file_name || c.shot_id.slice(0, 8)}</div>
                  <div className="text-[11px]" style={{ color: "var(--muted)" }}>
                    {fmtMs(c.source_in_ms)}–{fmtMs(c.source_out_ms)} · {(c.duration_ms / 1000).toFixed(1)}s
                    {trim && (
                      <span style={{ color: "#f59e0b" }}>
                        {" "}
                        (was {fmtMs(trim.from.source_in_ms)}–{fmtMs(trim.from.source_out_ms)})
                      </span>
                    )}
                  </div>
                </div>
                {b && (
                  <span className="shrink-0 rounded-full px-2 py-0.5 text-[9px] font-semibold tracking-wide text-white" style={{ background: b.c }}>
                    {b.t}
                  </span>
                )}
              </div>
            );
          })}
          {diff.removed.length > 0 && (
            <div className="pt-1 text-[11px]" style={{ color: "#ef4444" }}>
              {diff.removed.length} clip{diff.removed.length === 1 ? "" : "s"} removed
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function Thumb({ url, w, h }: { url?: string | null; w: number; h: number }) {
  if (!url) {
    return (
      <div className="flex shrink-0 items-center justify-center rounded" style={{ width: w, height: h, background: "var(--sidebar)" }}>
        <Film size={14} style={{ color: "var(--muted)" }} />
      </div>
    );
  }
  return (
    <Image src={url} alt="" width={w} height={h} unoptimized className="shrink-0 rounded object-cover" style={{ width: w, height: h }} />
  );
}
