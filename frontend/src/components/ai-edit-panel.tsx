"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/auth-store";
import { useDriveStore } from "@/stores/drive-store";
import {
  pollRenderUntilDone,
  startEditChatTurn,
  streamChatTurn,
  cancelChatTurn,
  getChatTurn,
  type ChatMessage,
  type ChatResponse,
  type ChatTimelineClip,
  type RenderRow,
} from "@/lib/api";
import {
  Sparkles,
  Send,
  Loader2,
  RotateCcw,
  AlertTriangle,
  Film,
  User as UserIcon,
  X,
} from "lucide-react";

// ---------------------------------------------------------------------------
// Persistent session model
// ---------------------------------------------------------------------------

const STORAGE_KEY = "edso_edit_chat_v1";

type AssistantMessage = {
  role: "assistant";
  reasoning: string;
  warnings: string[];
  timeline: ChatTimelineClip[];
  total_duration_ms: number;
  fcp7_xml: string;
  catalog_size: number;
  created_at: number;
  project_id?: string;
  edl_version_id?: string;
  render_id?: string;
  render_status?: RenderRow["status"];
  render_progress?: number;
  render_url?: string;
  render_error?: string;
};

type UserMessage = {
  role: "user";
  content: string;
  created_at: number;
};

type Turn = UserMessage | AssistantMessage;

type Session = {
  id: string;
  created_at: number;
  sequence_name: string;
  turns: Turn[];
};

function newSession(): Session {
  return {
    id: typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `${Date.now()}`,
    created_at: Date.now(),
    sequence_name: "AI Rough Cut",
    turns: [],
  };
}

function loadSession(): Session {
  if (typeof window === "undefined") return newSession();
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return newSession();
    const parsed = JSON.parse(raw);
    if (!parsed || !Array.isArray(parsed.turns)) return newSession();
    return parsed as Session;
  } catch {
    return newSession();
  }
}

function saveSession(s: Session) {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
  } catch {
    // Quota or serialization error -- ignore.
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatMs(ms: number): string {
  const total = Math.round(ms / 1000);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${m}:${s.toString().padStart(2, "0")}`;
}

function clipDurationMs(c: ChatTimelineClip): number {
  return Math.max(0, c.source_out_ms - c.source_in_ms);
}

function turnsToMessages(turns: Turn[]): ChatMessage[] {
  return turns.map((t) =>
    t.role === "user"
      ? { role: "user", content: t.content }
      : {
          role: "assistant",
          reasoning: t.reasoning,
          timeline: t.timeline.map((c) => ({
            shot_id: c.shot_id,
            source_in_ms: c.source_in_ms,
            source_out_ms: c.source_out_ms,
            role_in_edit: c.role_in_edit ?? null,
            why: c.why ?? null,
          })),
        },
  );
}

function responseToAssistant(res: ChatResponse): AssistantMessage {
  const merged: ChatTimelineClip[] = res.timeline.map((t) => ({
    shot_id: t.shot_id || "",
    source_in_ms: t.source_in_ms,
    source_out_ms: t.source_out_ms,
    role_in_edit: t.role_in_edit ?? null,
    why: t.why ?? null,
    file_id: t.file_id,
    file_name: t.file_name,
    timeline_start_ms: t.timeline_start_ms,
    timeline_end_ms: t.timeline_end_ms,
  }));
  return {
    role: "assistant",
    reasoning: res.reasoning,
    warnings: res.warnings || [],
    timeline: merged,
    total_duration_ms: res.total_duration_ms,
    fcp7_xml: res.fcp7_xml,
    catalog_size: res.catalog_size,
    created_at: Date.now(),
    project_id: res.project_id ?? undefined,
    edl_version_id: res.edl_version_id ?? undefined,
    render_id: res.render_id ?? undefined,
    render_status: res.render_id ? "queued" : undefined,
    render_progress: 0,
  };
}

const SAMPLE_PROMPTS = [
  "Cut me a 30-second trailer of the most dramatic moments",
  "Stitch the demo into a 60-second walkthrough in story order",
  "Make a 20-second teaser ending on the loudest reaction",
];

type ScopeFile = { id: string; name: string; file_type?: string };

type Scope = {
  fileIds: string[];
  files: ScopeFile[];
};

function loadScope(ids: string[]): Scope {
  const clean = ids.map((s) => s.trim()).filter(Boolean);
  if (clean.length === 0) return { fileIds: [], files: [] };
  let stashed: ScopeFile[] = [];
  if (typeof window !== "undefined") {
    try {
      const raw = window.sessionStorage.getItem("edso_edit_scope_v1");
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed?.files)) {
          stashed = parsed.files.filter(
            (f: ScopeFile) => f && typeof f.id === "string" && clean.includes(f.id),
          );
        }
      }
    } catch {
      // ignore
    }
  }
  const byId = new Map(stashed.map((f) => [f.id, f]));
  const files: ScopeFile[] = clean.map((id) => byId.get(id) || { id, name: id.slice(0, 8) });
  return { fileIds: clean, files };
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

// Tracks whether we've already initialized a chat session for THIS page load.
// On the first mount we start a fresh conversation (so users always "come in"
// to a clean editor); subsequent mounts in the same load (e.g. closing and
// reopening the panel) restore the in-progress session.
let sessionInitializedThisLoad = false;

export function AiEditPanel({
  fileIds,
  onClose,
  embedded = false,
}: {
  fileIds: string[];
  onClose?: () => void;
  embedded?: boolean;
}) {
  const router = useRouter();
  const session = useAuthStore((s) => s.session);
  const setAiTimeline = useDriveStore((s) => s.setAiTimeline);
  const showAiTimeline = useDriveStore((s) => s.showAiTimeline);
  const [sessionState, setSessionState] = useState<Session>(() => newSession());
  const [scope, setScope] = useState<Scope>(() => ({ fileIds: [], files: [] }));
  const [hydrated, setHydrated] = useState(false);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ phase: string; pct: number; label: string } | null>(null);
  const [activeTurnId, setActiveTurnId] = useState<string | null>(null);
  const streamAbortRef = useRef<AbortController | null>(null);
  const scrollerRef = useRef<HTMLDivElement | null>(null);

  // Hydrate chat history once. First visit of a page load starts fresh.
  useEffect(() => {
    if (!sessionInitializedThisLoad) {
      sessionInitializedThisLoad = true;
      const fresh = newSession();
      setSessionState(fresh);
      saveSession(fresh);
    } else {
      setSessionState(loadSession());
    }
    setHydrated(true);
  }, []);

  // Keep scope in sync with the incoming file ids (lets the panel re-scope
  // when reopened with a different selection).
  const fileIdsKey = fileIds.join(",");
  useEffect(() => {
    setScope(loadScope(fileIds));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileIdsKey]);

  useEffect(() => {
    if (hydrated) saveSession(sessionState);
  }, [sessionState, hydrated]);

  function clearScope() {
    setScope({ fileIds: [], files: [] });
    try {
      window.sessionStorage.removeItem("edso_edit_scope_v1");
    } catch {
      // ignore
    }
    if (!embedded) router.replace("/edit");
  }

  useEffect(() => {
    if (scrollerRef.current) {
      scrollerRef.current.scrollTop = scrollerRef.current.scrollHeight;
    }
  }, [sessionState.turns.length, sending]);

  const pollingRef = useRef<Set<string>>(new Set());
  useEffect(() => {
    if (!hydrated || !session?.access_token) return;
    const token = session.access_token;
    sessionState.turns.forEach((turn, idx) => {
      if (turn.role !== "assistant") return;
      if (!turn.render_id) return;
      if (turn.render_status === "done" || turn.render_status === "failed" || turn.render_status === "cancelled") {
        return;
      }
      if (pollingRef.current.has(turn.render_id)) return;
      pollingRef.current.add(turn.render_id);
      const renderId = turn.render_id;
      (async () => {
        try {
          await pollRenderUntilDone(renderId, token, {
            intervalMs: 1500,
            onUpdate: (row) => {
              setSessionState((prev) => {
                const turns = prev.turns.map((t, i) => {
                  if (i !== idx) return t;
                  if (t.role !== "assistant") return t;
                  return {
                    ...t,
                    render_status: row.status,
                    render_progress: row.progress_pct,
                    render_url: row.output_url ?? undefined,
                    render_error: row.error ?? undefined,
                  };
                });
                return { ...prev, turns };
              });
            },
          });
        } catch (e) {
          const msg = e instanceof Error ? e.message : "Render polling failed";
          setSessionState((prev) => {
            const turns = prev.turns.map((t, i) => {
              if (i !== idx) return t;
              if (t.role !== "assistant") return t;
              return { ...t, render_status: "failed" as const, render_error: msg };
            });
            return { ...prev, turns };
          });
        } finally {
          pollingRef.current.delete(renderId);
        }
      })();
    });
  }, [sessionState.turns, hydrated, session?.access_token]);

  const lastAssistant: AssistantMessage | null = useMemo(() => {
    for (let i = sessionState.turns.length - 1; i >= 0; i--) {
      const t = sessionState.turns[i];
      if (t.role === "assistant") return t;
    }
    return null;
  }, [sessionState.turns]);

  // Mirror the latest cut into the shared store so the bottom timeline dock
  // can render it (it lives outside this panel, left of the chat).
  useEffect(() => {
    if (!hydrated) return;
    if (!lastAssistant || lastAssistant.timeline.length === 0) {
      setAiTimeline(null);
      return;
    }
    setAiTimeline({
      clips: lastAssistant.timeline.map((c) => ({
        shot_id: c.shot_id ?? null,
        file_id: c.file_id ?? null,
        file_name: c.file_name,
        source_in_ms: c.source_in_ms,
        source_out_ms: c.source_out_ms,
        role_in_edit: c.role_in_edit ?? null,
        why: c.why ?? null,
      })),
      totalMs: lastAssistant.total_duration_ms,
      renderStatus: lastAssistant.render_status ?? null,
      renderUrl: lastAssistant.render_url ?? null,
      projectId: lastAssistant.project_id ?? null,
      baseVersionId: lastAssistant.edl_version_id ?? null,
    });
  }, [lastAssistant, hydrated, setAiTimeline]);

  // Auto-open the timeline when a new edit lands and again once it renders.
  useEffect(() => {
    if (lastAssistant && lastAssistant.timeline.length > 0) showAiTimeline();
  }, [lastAssistant?.created_at, showAiTimeline]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (lastAssistant?.render_url) showAiTimeline();
  }, [lastAssistant?.render_url, showAiTimeline]);

  async function handleSend(prompt?: string) {
    const text = (prompt ?? input).trim();
    if (!text || !session?.access_token || sending) return;
    const token = session.access_token;
    setError(null);
    setInput("");

    const userTurn: UserMessage = { role: "user", content: text, created_at: Date.now() };
    const turnsWithUser: Turn[] = [...sessionState.turns, userTurn];
    setSessionState((prev) => ({ ...prev, turns: turnsWithUser }));
    setSending(true);
    setProgress({ phase: "queued", pct: 0, label: "Starting" });

    try {
      const { turn_id } = await startEditChatTurn(
        {
          messages: turnsToMessages(turnsWithUser),
          sequence_name: sessionState.sequence_name,
          file_ids: scope.fileIds.length > 0 ? scope.fileIds : null,
        },
        token,
      );
      setActiveTurnId(turn_id);

      const abort = new AbortController();
      streamAbortRef.current = abort;
      let settled = false;

      await streamChatTurn(
        turn_id,
        token,
        (evt) => {
          if (evt.type === "phase") {
            setProgress({
              phase: evt.phase || "working",
              pct: evt.pct ?? 0,
              label: evt.label || "",
            });
          } else if (evt.type === "done") {
            settled = true;
            const assistant = responseToAssistant(evt.result);
            setSessionState((prev) => ({ ...prev, turns: [...turnsWithUser, assistant] }));
          } else if (evt.type === "cancelled") {
            settled = true;
            setError("Turn cancelled.");
          } else if (evt.type === "error") {
            settled = true;
            setError(evt.message);
          }
        },
        abort.signal,
      );

      if (!settled) {
        const row = await getChatTurn(turn_id, token);
        if (row.status === "done" && row.result) {
          const assistant = responseToAssistant(row.result);
          setSessionState((prev) => ({ ...prev, turns: [...turnsWithUser, assistant] }));
        } else if (row.status === "failed") {
          setError(row.error || "Turn failed.");
        } else if (row.status === "cancelled") {
          setError("Turn cancelled.");
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") {
        // handled via cancel flow
      } else {
        const msg = err instanceof Error ? err.message : "Edit request failed";
        setError(msg);
      }
    } finally {
      setSending(false);
      setProgress(null);
      setActiveTurnId(null);
      streamAbortRef.current = null;
    }
  }

  async function handleCancel() {
    const token = session?.access_token;
    const turnId = activeTurnId;
    if (!turnId || !token) return;
    try {
      await cancelChatTurn(turnId, token);
    } catch {
      // ignore
    }
  }

  function handleNewSession() {
    if (sessionState.turns.length > 0 && !window.confirm("Start a fresh chat? Your current conversation will be cleared.")) {
      return;
    }
    const fresh = newSession();
    setSessionState(fresh);
    setError(null);
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  // In the docked panel we keep paddings tighter than the full-page view.
  const padX = embedded ? "px-3" : "px-6";
  const bodyMax = embedded ? "max-w-none" : "max-w-3xl";

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div
        className={`flex flex-wrap items-center justify-between gap-2 border-b ${padX} py-3`}
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex min-w-0 items-center gap-2">
          <Sparkles size={18} style={{ color: "var(--accent)" }} />
          <div className="min-w-0">
            <h1 className="text-base font-semibold leading-tight">AI Editor</h1>
            <div className="truncate text-xs" style={{ color: "var(--muted)" }}>
              {sessionState.turns.length === 0
                ? "Tell me what to cut."
                : `${sessionState.turns.length} message${sessionState.turns.length === 1 ? "" : "s"} in this session`}
            </div>
          </div>
          <ScopeChip scope={scope} onClear={clearScope} />
        </div>
        <div className="flex items-center gap-1.5">
          <button
            onClick={handleNewSession}
            className="flex items-center justify-center rounded-lg border p-1.5 transition-colors hover:opacity-80"
            style={{ borderColor: "var(--border)" }}
            title="Clear conversation and start over"
          >
            <RotateCcw size={14} />
          </button>
          {onClose && (
            <button
              onClick={onClose}
              className="flex items-center justify-center rounded-lg border p-1.5 transition-colors hover:opacity-80"
              style={{ borderColor: "var(--border)" }}
              title="Close the AI editor"
            >
              <X size={14} />
            </button>
          )}
        </div>
      </div>

      {/* Split the remaining space: top half is the black render stage, bottom
          half is the conversation + composer. */}
      <div className="flex min-h-0 flex-1 flex-col">
        <RenderStage assistant={lastAssistant} />

        <div className="flex h-1/2 min-h-0 flex-col">
          {/* Message list */}
          <div ref={scrollerRef} className={`flex-1 overflow-y-auto overflow-x-hidden ${padX} py-4`}>
            <div className={`mx-auto w-full ${bodyMax} space-y-3`}>
              {sessionState.turns.length === 0 && <EmptyState onPick={(p) => void handleSend(p)} />}

              {sessionState.turns.map((t, idx) =>
                t.role === "user" ? (
                  <UserBubble key={idx} content={t.content} />
                ) : (
                  <AssistantBubble key={idx} message={t} />
                ),
              )}

              {sending && <ThinkingBubble progress={progress} onCancel={activeTurnId ? handleCancel : undefined} />}
              {error && (
                <div
                  className="rounded-lg border px-4 py-3 text-sm"
                  style={{ borderColor: "var(--danger)", color: "var(--danger)", background: "var(--accent-soft)" }}
                >
                  {error}
                </div>
              )}
            </div>
          </div>

          {/* Composer */}
          <div className={`border-t ${padX} py-3`} style={{ borderColor: "var(--border)" }}>
            <div className={`mx-auto flex ${bodyMax} items-end gap-2`}>
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  sessionState.turns.length === 0
                    ? "e.g. Make a 30-second trailer"
                    : "Tell me what to change..."
                }
                rows={2}
                className="flex-1 resize-none rounded-lg border px-3 py-2 text-sm focus:outline-none"
                style={{ background: "var(--background)", borderColor: "var(--border)", color: "var(--foreground)" }}
              />
              <button
                onClick={() => void handleSend()}
                disabled={sending || !input.trim()}
                className="flex shrink-0 items-center gap-1.5 rounded-lg px-4 py-2 text-sm font-medium text-white transition-colors disabled:opacity-50"
                style={{ background: "var(--accent)" }}
              >
                {sending ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                {sending ? "..." : "Send"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

// The top half of the panel: the program monitor. The <video> element is
// registered in the store so the docked timeline can drive playback of the
// assembled sequence (scrub / play / clip transitions). A placeholder shows
// only while there are no clips yet.
function RenderStage({ assistant }: { assistant: AssistantMessage | null }) {
  const setPreviewVideoEl = useDriveStore((s) => s.setPreviewVideoEl);
  const hasClips = useDriveStore(
    (s) => (s.aiTimeline?.clips?.length ?? 0) > 0 || s.editorHasClips,
  );
  const status = assistant?.render_status;
  const rendering = !!assistant?.render_id && (status === "queued" || status === "running" || !status);
  const failed = status === "failed";

  return (
    <div
      className="relative flex h-1/2 shrink-0 items-center justify-center overflow-hidden border-b"
      style={{ borderColor: "var(--border)", background: "#000" }}
    >
      <video ref={setPreviewVideoEl} playsInline className="h-full w-full bg-black object-contain" />

      {!hasClips && (
        <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center gap-2 px-4 text-center">
          {failed ? (
            <>
              <Film size={26} style={{ color: "var(--danger)" }} />
              <span className="text-sm" style={{ color: "var(--danger)" }}>Render failed</span>
            </>
          ) : rendering ? (
            <>
              <Loader2 size={26} className="animate-spin text-white/80" />
              <span className="text-sm text-white/80">Working on your edit…</span>
            </>
          ) : (
            <>
              <Film size={28} className="text-white/30" />
              <span className="text-sm text-white/40">Your edited video will appear here</span>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function ScopeChip({ scope, onClear }: { scope: Scope; onClear: () => void }) {
  if (scope.fileIds.length === 0) {
    return (
      <span
        className="ml-2 hidden rounded-full border px-2 py-0.5 text-[11px] sm:inline"
        style={{ borderColor: "var(--border)", color: "var(--muted)" }}
        title="The editor will draw from every indexed video you own. Select specific videos on the Drive to scope this chat."
      >
        scope: all videos
      </span>
    );
  }
  const names = scope.files.map((f) => f.name).filter(Boolean).slice(0, 2).join(", ");
  const more = scope.files.length > 2 ? ` +${scope.files.length - 2}` : "";
  const label = names ? `${names}${more}` : `${scope.fileIds.length} video${scope.fileIds.length === 1 ? "" : "s"}`;
  return (
    <span
      className="ml-2 inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px]"
      style={{ borderColor: "var(--accent)", color: "var(--accent)", background: "var(--accent-soft)" }}
      title={scope.files.map((f) => f.name).join("\n") || scope.fileIds.join("\n")}
    >
      <span className="font-medium">scope:</span>
      <span className="max-w-[160px] truncate">{label}</span>
      <button onClick={onClear} className="ml-0.5 rounded-full p-0.5 hover:bg-black/5" title="Clear scope and use all videos">
        <X size={11} />
      </button>
    </span>
  );
}

function EmptyState({ onPick }: { onPick: (p: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-12 text-center">
      <div className="flex h-12 w-12 items-center justify-center rounded-full" style={{ background: "var(--sidebar)" }}>
        <Sparkles size={20} style={{ color: "var(--accent)" }} />
      </div>
      <div>
        <div className="text-sm font-medium">Chat with your editor</div>
        <p className="mt-1 max-w-md text-xs" style={{ color: "var(--muted)" }}>
          Describe what you want, then keep refining. I&apos;ll remember every prior cut and adjust accordingly.
        </p>
      </div>
      <div className="flex flex-wrap justify-center gap-1.5">
        {SAMPLE_PROMPTS.map((p) => (
          <button
            key={p}
            onClick={() => onPick(p)}
            className="rounded-full border px-3 py-1 text-xs transition-colors hover:opacity-80"
            style={{ borderColor: "var(--border)", color: "var(--muted)" }}
          >
            {p}
          </button>
        ))}
      </div>
    </div>
  );
}

function UserBubble({ content }: { content: string }) {
  return (
    <div className="flex justify-end gap-3">
      <div className="max-w-[85%] break-words rounded-2xl rounded-tr-sm px-4 py-2.5 text-sm" style={{ background: "var(--accent)", color: "white" }}>
        {content}
      </div>
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full" style={{ background: "var(--sidebar)" }}>
        <UserIcon size={14} style={{ color: "var(--muted)" }} />
      </div>
    </div>
  );
}

function ThinkingBubble({
  progress,
  onCancel,
}: {
  progress?: { phase: string; pct: number; label: string } | null;
  onCancel?: () => void;
}) {
  const label = progress?.label || "Thinking through the edit...";
  const pct = Math.max(0, Math.min(100, progress?.pct ?? 0));
  return (
    <div className="flex items-start gap-3">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full" style={{ background: "var(--sidebar)" }}>
        <Sparkles size={14} style={{ color: "var(--accent)" }} />
      </div>
      <div
        className="w-full max-w-[80%] rounded-2xl rounded-tl-sm border px-4 py-2.5 text-sm"
        style={{ borderColor: "var(--border)", background: "var(--background)", color: "var(--muted)" }}
      >
        <div className="flex items-center justify-between gap-3">
          <span className="inline-flex items-center gap-2">
            <Loader2 size={14} className="animate-spin" />
            {label}
          </span>
          {onCancel && (
            <button
              onClick={onCancel}
              className="flex items-center gap-1 rounded-md border px-2 py-1 text-[11px] transition-colors hover:opacity-80"
              style={{ borderColor: "var(--border)" }}
              title="Cancel this turn"
            >
              <X size={12} />
              Cancel
            </button>
          )}
        </div>
        {progress && (
          <div className="mt-2 h-1 w-full overflow-hidden rounded-full" style={{ background: "var(--sidebar)" }}>
            <div className="h-full rounded-full transition-all duration-300" style={{ width: `${pct}%`, background: "var(--accent)" }} />
          </div>
        )}
      </div>
    </div>
  );
}

function AssistantBubble({ message }: { message: AssistantMessage }) {
  return (
    <div className="flex items-start gap-3">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full" style={{ background: "var(--sidebar)" }}>
        <Sparkles size={14} style={{ color: "var(--accent)" }} />
      </div>
      <div className="min-w-0 flex-1 space-y-3">
        {message.reasoning && (
          <div className="w-full rounded-2xl rounded-tl-sm border px-4 py-3 text-sm" style={{ borderColor: "var(--border)", background: "var(--background)" }}>
            <p className="whitespace-pre-wrap break-words leading-relaxed">{message.reasoning}</p>
          </div>
        )}

        {message.warnings && message.warnings.length > 0 && (
          <div
            className="w-full rounded-lg border px-3 py-2 text-xs"
            style={{ borderColor: "rgba(234,179,8,0.4)", background: "rgba(234,179,8,0.06)", color: "#a16207" }}
          >
            <div className="mb-1 flex items-center gap-1.5 font-medium">
              <AlertTriangle size={12} />
              Notes from the editor
            </div>
            <ul className="ml-4 list-disc space-y-0.5 break-words">
              {message.warnings.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </div>
        )}

        {message.timeline.length > 0 ? (
          <TimelineCard message={message} />
        ) : (
          <div className="w-full rounded-lg border px-3 py-2 text-xs" style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
            (No clips in this turn -- try adjusting the brief.)
          </div>
        )}
      </div>
    </div>
  );
}

function TimelineCard({ message }: { message: AssistantMessage }) {
  const totalDur = message.total_duration_ms;
  return (
    <div className="w-full rounded-lg border" style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}>
      <div className="flex items-center justify-between border-b px-3 py-2 text-xs" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-1.5 font-medium">
          <Film size={12} />
          Timeline
        </div>
        <div style={{ color: "var(--muted)" }}>
          {message.timeline.length} clip{message.timeline.length === 1 ? "" : "s"} · {formatMs(totalDur)} · catalog {message.catalog_size}
        </div>
      </div>

      <div className="px-3 py-3">
        <div className="flex h-7 overflow-hidden rounded" style={{ background: "var(--background)" }}>
          {message.timeline.map((c, i) => {
            const dur = clipDurationMs(c);
            const pct = totalDur > 0 ? (dur / totalDur) * 100 : 100 / message.timeline.length;
            const hue = (i * 53) % 360;
            return (
              <div
                key={i}
                title={`Clip ${i + 1} · ${(dur / 1000).toFixed(1)}s${c.role_in_edit ? ` · ${c.role_in_edit}` : ""}${c.why ? `\n${c.why}` : ""}`}
                style={{ width: `${pct}%`, background: `hsl(${hue}, 60%, 55%)` }}
                className="border-r last:border-r-0"
              />
            );
          })}
        </div>
      </div>

      <ol className="space-y-0 border-t text-xs" style={{ borderColor: "var(--border)" }}>
        {message.timeline.map((c, i) => (
          <li key={i} className="flex items-baseline gap-3 border-b px-3 py-1.5 last:border-b-0" style={{ borderColor: "var(--border)" }}>
            <span className="w-5 shrink-0 text-right font-mono" style={{ color: "var(--muted)" }}>
              {i + 1}
            </span>
            <div className="min-w-0 flex-1 truncate">
              <span className="font-medium">{c.file_name || c.file_id || "(unknown)"}</span>
              {c.role_in_edit && (
                <span className="ml-2 rounded-full border px-1.5 py-0 text-[10px] uppercase tracking-wide" style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
                  {c.role_in_edit}
                </span>
              )}
              {c.why && (
                <div className="mt-0.5 truncate" style={{ color: "var(--muted)" }} title={c.why}>
                  {c.why}
                </div>
              )}
            </div>
            <span className="shrink-0 font-mono text-[11px]" style={{ color: "var(--muted)" }}>
              {formatMs(c.source_in_ms)}–{formatMs(c.source_out_ms)} ({(clipDurationMs(c) / 1000).toFixed(1)}s)
            </span>
          </li>
        ))}
      </ol>
    </div>
  );
}
