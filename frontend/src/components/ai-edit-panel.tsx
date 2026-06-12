"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Sparkles,
  X,
  Send,
  Plus,
  Play,
  Pause,
  Volume2,
  VolumeX,
  SkipBack,
  SkipForward,
  Loader2,
  AlertCircle,
  Film,
} from "lucide-react";
import { useDriveStore } from "@/stores/drive-store";
import { useAuthStore } from "@/stores/auth-store";
import {
  createEditThread,
  getEditThread,
  sendEditMessage,
  getFilePlaybackUrl,
  type EditThread,
  type EditThreadStatus,
  type EditSegment,
} from "@/lib/api";

const POLL_MS = 2000;

function scopeKey(ids: string[]) {
  return [...ids].sort().join(",");
}

// --- localStorage helpers (thread id + the user's typed turns per scope) ---

function loadThreadId(scope: string): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(`edit-thread:${scope}`);
}
function saveThreadId(scope: string, id: string) {
  window.localStorage.setItem(`edit-thread:${scope}`, id);
}
function clearThreadId(scope: string) {
  window.localStorage.removeItem(`edit-thread:${scope}`);
}
function loadTurns(threadId: string): string[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(window.localStorage.getItem(`edit-turns:${threadId}`) || "[]");
  } catch {
    return [];
  }
}
function saveTurns(threadId: string, turns: string[]) {
  window.localStorage.setItem(`edit-turns:${threadId}`, JSON.stringify(turns));
}

const STATUS_LABEL: Record<EditThreadStatus, string> = {
  drafting: "Drafting…",
  awaiting_user: "Needs your input",
  ready: "Ready",
  failed: "Failed",
};

function StatusBadge({ status }: { status: EditThreadStatus }) {
  const drafting = status === "drafting";
  const color =
    status === "ready"
      ? "var(--accent)"
      : status === "failed"
      ? "var(--danger)"
      : "var(--muted)";
  return (
    <span
      className="flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium"
      style={{ background: "var(--accent-soft)", color }}
    >
      {drafting && <Loader2 size={11} className="animate-spin" />}
      {STATUS_LABEL[status]}
    </span>
  );
}

export function AiEditPanel() {
  const { aiPanelOpen, aiScopeFileIds, closeAiPanel } = useDriveStore();
  const session = useAuthStore((s) => s.session);
  const token = session?.access_token;

  const scope = useMemo(() => scopeKey(aiScopeFileIds), [aiScopeFileIds]);

  const [threadId, setThreadId] = useState<string | null>(null);
  const [thread, setThread] = useState<EditThread | null>(null);
  const [userTurns, setUserTurns] = useState<string[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const refresh = useCallback(
    async (id: string) => {
      if (!token) return;
      try {
        const t = await getEditThread(id, token);
        setThread(t);
        if (t.status !== "drafting") {
          stopPolling();
          setBusy(false);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load edit");
        stopPolling();
        setBusy(false);
      }
    },
    [token, stopPolling]
  );

  const startPolling = useCallback(
    (id: string) => {
      stopPolling();
      pollRef.current = setInterval(() => refresh(id), POLL_MS);
    },
    [refresh, stopPolling]
  );

  // When opened (or scope changes), hydrate from any persisted thread.
  useEffect(() => {
    if (!aiPanelOpen) return;
    setError(null);
    const existing = loadThreadId(scope);
    setThreadId(existing);
    setThread(null);
    if (existing) {
      setUserTurns(loadTurns(existing));
      refresh(existing).then(() => {
        // resume polling if it was mid-draft
        getEditThread(existing, token || "").then((t) => {
          if (t.status === "drafting") startPolling(existing);
        }).catch(() => {});
      });
    } else {
      setUserTurns([]);
    }
    return stopPolling;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aiPanelOpen, scope]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [thread, userTurns, busy]);

  async function handleSend() {
    const text = input.trim();
    if (!text || !token || busy) return;
    setInput("");
    setError(null);
    setBusy(true);
    const nextTurns = [...userTurns, text];
    setUserTurns(nextTurns);
    try {
      if (!threadId) {
        const { thread_id } = await createEditThread(aiScopeFileIds, text, token);
        setThreadId(thread_id);
        saveThreadId(scope, thread_id);
        saveTurns(thread_id, nextTurns);
        startPolling(thread_id);
        await refresh(thread_id);
      } else {
        saveTurns(threadId, nextTurns);
        await sendEditMessage(threadId, { text }, token);
        startPolling(threadId);
        await refresh(threadId);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "The edit run failed.");
      setBusy(false);
    }
  }

  async function handleAnswers(answers: Record<string, string>, note?: string) {
    if (!threadId || !token || busy) return;
    setError(null);
    setBusy(true);
    const label =
      "Answered: " +
      Object.values(answers).join(" · ") +
      (note ? ` — ${note}` : "");
    const nextTurns = [...userTurns, label];
    setUserTurns(nextTurns);
    saveTurns(threadId, nextTurns);
    try {
      await sendEditMessage(threadId, { answers, text: note }, token);
      startPolling(threadId);
      await refresh(threadId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "The edit run failed.");
      setBusy(false);
    }
  }

  function handleNewThread() {
    stopPolling();
    clearThreadId(scope);
    setThreadId(null);
    setThread(null);
    setUserTurns([]);
    setInput("");
    setError(null);
    setBusy(false);
  }

  if (!aiPanelOpen) return null;

  const doc = thread?.document ?? null;
  const status = thread?.status;
  const questions =
    status === "awaiting_user" ? thread?.open_questions ?? [] : [];

  return (
    <aside
      className="flex h-full w-[460px] shrink-0 flex-col border-l"
      style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between gap-2 border-b px-4 py-3"
        style={{ borderColor: "var(--border)" }}
      >
        <div className="flex min-w-0 items-center gap-2">
          <Sparkles size={17} style={{ color: "var(--accent)" }} />
          <span className="truncate text-sm font-semibold">AI Edit</span>
          {status && <StatusBadge status={status} />}
        </div>
        <div className="flex items-center gap-1">
          <span
            className="rounded-full px-2 py-0.5 text-xs"
            style={{ background: "var(--accent-soft)", color: "var(--muted)" }}
            title={`${aiScopeFileIds.length} clip(s) in scope`}
          >
            {aiScopeFileIds.length} clip{aiScopeFileIds.length === 1 ? "" : "s"}
          </span>
          <button
            onClick={handleNewThread}
            className="rounded-lg p-1.5 transition-colors hover:bg-[var(--accent-soft)]"
            title="Start a fresh edit"
          >
            <Plus size={16} />
          </button>
          <button
            onClick={closeAiPanel}
            className="rounded-lg p-1.5 transition-colors hover:bg-[var(--accent-soft)]"
            title="Close"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Program monitor */}
      <EditPreview timeline={doc?.timeline ?? []} token={token} />

      {/* Conversation */}
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {!threadId && (
          <EmptyState />
        )}

        {userTurns.map((t, i) => (
          <Bubble key={`u${i}`} role="user">
            {t}
          </Bubble>
        ))}

        {doc && <DocumentView doc={doc} version={thread?.document_version ?? null} />}

        {busy && status === "drafting" && (
          <div
            className="flex items-center gap-2 text-sm"
            style={{ color: "var(--muted)" }}
          >
            <Loader2 size={14} className="animate-spin" />
            Planning the cut…
          </div>
        )}

        {questions.length > 0 && (
          <QuestionForm questions={questions} onSubmit={handleAnswers} disabled={busy} />
        )}

        {error && (
          <div
            className="flex items-start gap-2 rounded-lg border p-3 text-sm"
            style={{ borderColor: "var(--danger)", color: "var(--danger)" }}
          >
            <AlertCircle size={15} className="mt-0.5 shrink-0" />
            <span>{error}</span>
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="border-t px-3 py-3" style={{ borderColor: "var(--border)" }}>
        <div
          className="flex items-end gap-2 rounded-xl border px-3 py-2"
          style={{ borderColor: "var(--border)", background: "var(--background)" }}
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            rows={1}
            placeholder={
              threadId
                ? "Refine the edit, or answer above…"
                : "Describe the edit you want… (e.g. a punchy 60s pitch)"
            }
            className="max-h-32 min-h-[24px] flex-1 resize-none bg-transparent text-sm outline-none"
          />
          <button
            onClick={handleSend}
            disabled={!input.trim() || busy}
            className="rounded-lg p-1.5 text-white transition-opacity disabled:opacity-30"
            style={{ background: "var(--accent)" }}
            title="Send"
          >
            <Send size={15} />
          </button>
        </div>
      </div>
    </aside>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center py-10 text-center">
      <Sparkles size={30} style={{ color: "var(--accent)" }} />
      <p className="mt-3 text-sm font-semibold">Describe your edit</p>
      <p className="mt-1 max-w-[18rem] text-xs" style={{ color: "var(--muted)" }}>
        Tell the editor what you want — length, tone, the story to tell. It reads
        your footage, drafts a cut, and asks when it needs a decision.
      </p>
    </div>
  );
}

function Bubble({ role, children }: { role: "user" | "assistant"; children: React.ReactNode }) {
  const isUser = role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className="max-w-[85%] whitespace-pre-wrap rounded-2xl px-3 py-2 text-sm"
        style={
          isUser
            ? { background: "var(--accent)", color: "#fff" }
            : { background: "var(--background)", border: "1px solid var(--border)" }
        }
      >
        {children}
      </div>
    </div>
  );
}

function fmtClock(ms: number) {
  const s = Math.max(0, Math.round(ms / 1000));
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, "0")}`;
}

function DocumentView({
  doc,
  version,
}: {
  doc: NonNullable<EditThread["document"]>;
  version: number | null;
}) {
  const timeline = doc.timeline ?? [];
  const totalMs = timeline.reduce((a, s) => a + (s.out_ms - s.in_ms), 0);

  return (
    <div
      className="space-y-3 rounded-2xl border p-3 text-sm"
      style={{ background: "var(--background)", borderColor: "var(--border)" }}
    >
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5 font-semibold">
          <Film size={14} style={{ color: "var(--accent)" }} />
          Edit plan
          {version != null && (
            <span className="text-xs font-normal" style={{ color: "var(--muted)" }}>
              v{version}
            </span>
          )}
        </span>
        <span className="text-xs" style={{ color: "var(--muted)" }}>
          {timeline.length} cut{timeline.length === 1 ? "" : "s"} · {fmtClock(totalMs)}
        </span>
      </div>

      {doc.brief?.goal && (
        <p style={{ color: "var(--muted)" }}>{doc.brief.goal}</p>
      )}

      {doc.spine?.regions && doc.spine.regions.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs" style={{ color: "var(--muted)" }}>
            Spine
          </span>
          {doc.spine.regions.map((r, i) => (
            <span
              key={i}
              title={r.rationale ?? undefined}
              className="rounded-full border px-2 py-0.5 text-xs"
              style={{ borderColor: "var(--border)", color: "var(--accent)" }}
            >
              {r.kind === "other" && r.label ? r.label : r.kind}
              {r.locked_channels && r.locked_channels.length > 0
                ? ` · 🔒${r.locked_channels.map((c) => c[0].toUpperCase()).join("")}`
                : ""}
            </span>
          ))}
        </div>
      )}

      {doc.summary && <p className="whitespace-pre-wrap">{doc.summary}</p>}

      {timeline.length > 0 && (
        <ol className="space-y-1.5">
          {timeline.map((s, i) => (
            <li key={s.seg_id} className="flex gap-2">
              <span
                className="mt-0.5 shrink-0 text-xs tabular-nums"
                style={{ color: "var(--muted)" }}
              >
                {String(i + 1).padStart(2, "0")}
              </span>
              <div className="min-w-0">
                <div className="text-xs" style={{ color: "var(--muted)" }}>
                  {fmtClock(s.in_ms)}–{fmtClock(s.out_ms)}
                  {s.beat_id ? ` · ${s.beat_id}` : ""}
                </div>
                {s.content && <div className="truncate">{s.content}</div>}
              </div>
            </li>
          ))}
        </ol>
      )}

      {doc.notes && doc.notes.length > 0 && (
        <ul className="space-y-1 border-t pt-2 text-xs" style={{ borderColor: "var(--border)", color: "var(--muted)" }}>
          {doc.notes.map((n, i) => (
            <li key={i}>• {n}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function QuestionForm({
  questions,
  onSubmit,
  disabled,
}: {
  questions: NonNullable<EditThread["open_questions"]>;
  onSubmit: (answers: Record<string, string>, note?: string) => void;
  disabled: boolean;
}) {
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [note, setNote] = useState("");

  const allAnswered = questions.every((q) => answers[q.q_id]);

  return (
    <div
      className="space-y-3 rounded-2xl border p-3"
      style={{ background: "var(--accent-soft)", borderColor: "var(--accent)" }}
    >
      <p className="text-sm font-semibold">A couple of decisions</p>
      {questions.map((q) => (
        <div key={q.q_id} className="space-y-1.5">
          <p className="text-sm">{q.question}</p>
          {q.why && (
            <p className="text-xs" style={{ color: "var(--muted)" }}>
              {q.why}
            </p>
          )}
          <div className="flex flex-wrap gap-1.5">
            {(q.options && q.options.length > 0 ? q.options : [q.default]).map((opt) => {
              const active = answers[q.q_id] === opt;
              return (
                <button
                  key={opt}
                  onClick={() => setAnswers((a) => ({ ...a, [q.q_id]: opt }))}
                  className="rounded-full border px-2.5 py-1 text-xs transition-colors"
                  style={{
                    background: active ? "var(--accent)" : "var(--background)",
                    color: active ? "#fff" : "var(--foreground)",
                    borderColor: active ? "var(--accent)" : "var(--border)",
                  }}
                >
                  {opt}
                </button>
              );
            })}
          </div>
        </div>
      ))}
      <input
        type="text"
        value={note}
        onChange={(e) => setNote(e.target.value)}
        placeholder="Add a note (optional)…"
        className="w-full rounded-lg border bg-transparent px-2.5 py-1.5 text-sm outline-none"
        style={{ borderColor: "var(--border)" }}
      />
      <button
        onClick={() => onSubmit(answers, note.trim() || undefined)}
        disabled={disabled || !allAnswered}
        className="w-full rounded-lg py-2 text-sm font-medium text-white transition-opacity disabled:opacity-40"
        style={{ background: "var(--accent)" }}
      >
        Submit answers
      </button>
    </div>
  );
}

// --- Program monitor: sequential EDL playback of source proxies ---

function EditPreview({
  timeline,
  token,
}: {
  timeline: EditSegment[];
  token: string | undefined;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [urls, setUrls] = useState<Record<string, string>>({});
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);

  const fileIds = useMemo(
    () => Array.from(new Set(timeline.map((s) => s.file_id))),
    [timeline]
  );

  // Resolve playback URLs for every source clip in the timeline.
  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    (async () => {
      const entries = await Promise.all(
        fileIds.map(async (id) => {
          if (urls[id]) return [id, urls[id]] as const;
          try {
            const { url } = await getFilePlaybackUrl(id, token);
            return [id, url] as const;
          } catch {
            return [id, ""] as const;
          }
        })
      );
      if (!cancelled) {
        setUrls((prev) => {
          const next = { ...prev };
          for (const [id, url] of entries) if (url) next[id] = url;
          return next;
        });
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileIds.join(","), token]);

  // Reset to the start whenever the plan changes shape.
  useEffect(() => {
    setIndex(0);
    setPlaying(false);
  }, [timeline.length]);

  const seg = timeline[index];

  // Load + seek the active segment.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !seg) return;
    const src = urls[seg.file_id];
    if (!src) return;
    if (!v.src.startsWith(src.split("?")[0]) && v.src !== src) {
      v.src = src;
    }
    const seek = () => {
      v.currentTime = seg.in_ms / 1000;
      if (playing) v.play().catch(() => {});
    };
    if (v.readyState >= 1) seek();
    else v.addEventListener("loadedmetadata", seek, { once: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [index, seg?.seg_id, urls[seg?.file_id ?? ""]]);

  function onTimeUpdate() {
    const v = videoRef.current;
    if (!v || !seg) return;
    if (v.currentTime * 1000 >= seg.out_ms) {
      if (index + 1 < timeline.length) {
        setIndex((i) => i + 1);
      } else {
        v.pause();
        setPlaying(false);
        setIndex(0);
      }
    }
  }

  function togglePlay() {
    const v = videoRef.current;
    if (!v || !seg) return;
    if (playing) {
      v.pause();
      setPlaying(false);
    } else {
      if (v.currentTime * 1000 < seg.in_ms || v.currentTime * 1000 >= seg.out_ms) {
        v.currentTime = seg.in_ms / 1000;
      }
      v.play().catch(() => {});
      setPlaying(true);
    }
  }

  const hasTimeline = timeline.length > 0;

  return (
    <div
      className="border-b px-4 py-3"
      style={{ borderColor: "var(--border)" }}
    >
      <div
        className="relative aspect-video w-full overflow-hidden rounded-lg"
        style={{ background: "#000" }}
      >
        {hasTimeline ? (
          <video
            ref={videoRef}
            className="h-full w-full"
            muted={muted}
            playsInline
            onTimeUpdate={onTimeUpdate}
            onEnded={onTimeUpdate}
          />
        ) : (
          <div className="flex h-full w-full items-center justify-center text-xs" style={{ color: "#666" }}>
            <Film size={28} />
          </div>
        )}
      </div>

      {hasTimeline && (
        <div className="mt-2 flex items-center gap-2">
          <button
            onClick={() => setIndex((i) => Math.max(0, i - 1))}
            className="rounded p-1 transition-colors hover:bg-[var(--accent-soft)]"
            title="Previous cut"
          >
            <SkipBack size={15} />
          </button>
          <button
            onClick={togglePlay}
            className="rounded-full p-1.5 text-white"
            style={{ background: "var(--accent)" }}
            title={playing ? "Pause" : "Play"}
          >
            {playing ? <Pause size={15} /> : <Play size={15} />}
          </button>
          <button
            onClick={() => setIndex((i) => Math.min(timeline.length - 1, i + 1))}
            className="rounded p-1 transition-colors hover:bg-[var(--accent-soft)]"
            title="Next cut"
          >
            <SkipForward size={15} />
          </button>
          <button
            onClick={() => setMuted((m) => !m)}
            className="rounded p-1 transition-colors hover:bg-[var(--accent-soft)]"
            title={muted ? "Unmute" : "Mute"}
          >
            {muted ? <VolumeX size={15} /> : <Volume2 size={15} />}
          </button>
          <span className="ml-auto text-xs tabular-nums" style={{ color: "var(--muted)" }}>
            Cut {index + 1}/{timeline.length}
          </span>
        </div>
      )}
    </div>
  );
}
