"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  Sparkles,
  X,
  Send,
  Plus,
  Loader2,
  AlertCircle,
  Film,
  Layers,
  Music,
  Scissors,
  SlidersHorizontal,
} from "lucide-react";
import { useDriveStore } from "@/stores/drive-store";
import { RenderBar } from "@/components/render-bar";
import { TimelineEditor } from "@/components/timeline-editor";
import { CompositePreview } from "@/components/preview/composite-preview";
import { useEditDocStore } from "@/stores/edit-doc-store";
import { useTransport } from "@/stores/transport-store";
import { useAuthStore } from "@/stores/auth-store";
import {
  createEditThread,
  getEditThread,
  sendThreadMessage,
  type ThreadQuestion,
  type EditThread,
  type EditThreadStatus,
  type EditOperation,
} from "@/lib/api";

const POLL_MS = 2000;

type ChatMsg = { role: "user" | "assistant"; text: string };

function scopeKey(ids: string[]) {
  return [...ids].sort().join(",");
}

// --- localStorage helpers (thread id + the chat transcript per thread) ---

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
/** Wipe EVERY persisted edit thread (all scopes) + their transcripts, so nothing
 * auto-loads and a fresh start is truly empty. */
function clearAllPersistedThreads() {
  if (typeof window === "undefined") return;
  const keys: string[] = [];
  for (let i = 0; i < window.localStorage.length; i++) {
    const k = window.localStorage.key(i);
    if (k && (k.startsWith("edit-thread:") || k.startsWith("edit-msgs:") || k.startsWith("edit-turns:"))) {
      keys.push(k);
    }
  }
  keys.forEach((k) => window.localStorage.removeItem(k));
}
function loadMsgs(threadId: string): ChatMsg[] {
  if (typeof window === "undefined") return [];
  try {
    return JSON.parse(window.localStorage.getItem(`edit-msgs:${threadId}`) || "[]");
  } catch {
    return [];
  }
}
function saveMsgs(threadId: string, msgs: ChatMsg[]) {
  window.localStorage.setItem(`edit-msgs:${threadId}`, JSON.stringify(msgs));
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
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  // The editor asked for a user-owned decision on its latest turn -> show pickable
  // options. Answering is just sending the picked option as the next message.
  const [questions, setQuestions] = useState<ThreadQuestion[]>([]);
  const [error, setError] = useState<string | null>(null);
  // Portal target for the bottom editor dock (program monitor + timeline) that
  // lives in the main area, pro-editor style.
  const [dockEl, setDockEl] = useState<HTMLElement | null>(null);

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
      setMessages(loadMsgs(existing));
      refresh(existing).then(() => {
        // resume polling if it was mid-draft
        getEditThread(existing, token || "").then((t) => {
          if (t.status === "drafting") startPolling(existing);
        }).catch(() => {});
      });
    } else {
      // No saved thread for this scope -> ensure the preview/timeline are empty.
      setMessages([]);
      useTransport.getState().reset();
      clearDoc();
      seededRef.current = "";
    }
    return stopPolling;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [aiPanelOpen, scope]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [thread, messages, busy]);

  // Seed the live working-document store whenever the authoritative version
  // changes (agent wrote a new version, or we loaded a thread). Keyed on
  // version so 2s polls don't wipe in-progress manual edits.
  const seedDoc = useEditDocStore((s) => s.seed);
  const clearDoc = useEditDocStore((s) => s.clear);
  const seededRef = useRef<string>("");
  useEffect(() => {
    if (!thread?.id) return;
    const key = `${thread.id}:${thread.document_version ?? 0}`;
    if (seededRef.current === key) return;
    seededRef.current = key;
    seedDoc(thread.id, thread.document_version ?? 0, thread.document);
  }, [thread?.id, thread?.document_version, thread?.document, seedDoc]);

  // Find the bottom dock slot rendered by the drive layout so we can portal the
  // program monitor + timeline into the main area.
  useEffect(() => {
    setDockEl(document.getElementById("ai-editor-dock"));
  }, [aiPanelOpen]);

  async function handleSend(override?: string) {
    const text = (override ?? input).trim();
    if (!text || !token || busy) return;
    if (override === undefined) setInput("");
    setError(null);
    setBusy(true);
    setQuestions([]);
    // Optimistically show the user's message right away.
    const withUser: ChatMsg[] = [...messages, { role: "user", text }];
    setMessages(withUser);

    try {
      // Chat-first: ensure a thread exists (creating one drafts nothing), then
      // send the message into the conversation.
      let id = threadId;
      if (!id) {
        const created = await createEditThread(aiScopeFileIds, "", token);
        id = created.thread_id;
        setThreadId(id);
        saveThreadId(scope, id);
      }

      const res = await sendThreadMessage(id, text, token);
      const withReply: ChatMsg[] = [...withUser, { role: "assistant", text: res.reply }];
      setMessages(withReply);
      saveMsgs(id, withReply);
      // Agentic: the editor already APPLIED any edit during its turn. If the edit
      // changed, pull the new document version so the timeline reflects it.
      if (res.changed) {
        await refresh(id);
      }
      // If it paused to ask, surface the options; answering is the next message.
      if (res.awaiting_user && res.questions?.length) {
        setQuestions(res.questions);
      }
      setBusy(false);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Message failed.");
      setBusy(false);
    }
  }

  // Ensure an edit session exists WITHOUT seeding the working doc (so manually
  // dragged-in cuts aren't wiped). Used by the timeline's save + first drop.
  const ensureThread = useCallback(async (): Promise<string | null> => {
    if (threadId) return threadId;
    if (!token) return null;
    try {
      const created = await createEditThread(aiScopeFileIds, "", token);
      setThreadId(created.thread_id);
      saveThreadId(scope, created.thread_id);
      return created.thread_id;
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not start an edit.");
      return null;
    }
  }, [threadId, token, aiScopeFileIds, scope]);

  function handleNewThread() {
    stopPolling();
    // A fresh start must also empty the LIVE preview/timeline + stop playback,
    // otherwise the previous edit keeps showing (and its audio keeps playing).
    useTransport.getState().reset();
    clearDoc();
    seededRef.current = "";
    clearAllPersistedThreads();
    setThreadId(null);
    setThread(null);
    setMessages([]);
    setInput("");
    setError(null);
    setBusy(false);
    setQuestions([]);
  }

  function handleSavedEdit(newVersion: number, newDoc: NonNullable<EditThread["document"]>) {
    setThread((prev) =>
      prev ? { ...prev, document: newDoc, document_version: newVersion } : prev
    );
  }

  if (!aiPanelOpen) return null;

  const doc = thread?.document ?? null;
  const status = thread?.status;

  return (
    <>
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

      {/* Program monitor — reads the live working doc from the edit store */}
      <CompositePreview token={token} />

      {/* Export / render */}
      {threadId && (
        <RenderBar
          threadId={threadId}
          version={thread?.document_version ?? null}
          token={token}
          disabled={!doc?.timeline?.length}
        />
      )}

      {/* Conversation */}
      <div ref={scrollRef} className="flex-1 space-y-3 overflow-y-auto px-4 py-4">
        {messages.length === 0 && !busy && <EmptyState />}

        {messages.map((m, i) => (
          <Bubble key={`m${i}`} role={m.role}>
            {m.text}
          </Bubble>
        ))}

        {questions.length > 0 && !busy && (
          <QuestionCard questions={questions} onPick={(text) => handleSend(text)} />
        )}

        {doc && <DocumentView doc={doc} version={thread?.document_version ?? null} />}

        {busy && (
          <div
            className="flex items-center gap-2 text-sm"
            style={{ color: "var(--muted)" }}
          >
            <Loader2 size={14} className="animate-spin" />
            {status === "drafting" ? "Editing…" : "Thinking…"}
          </div>
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
            placeholder="Message EDSO — ask anything, or describe an edit…"
            className="max-h-32 min-h-[24px] flex-1 resize-none bg-transparent text-sm outline-none"
          />
          <button
            onClick={() => handleSend()}
            disabled={!input.trim() || busy}
            className="rounded-lg p-1.5 transition-opacity disabled:opacity-30"
            style={{ background: "var(--accent)", color: "var(--background)" }}
            title="Send"
          >
            <Send size={15} />
          </button>
        </div>
      </div>
    </aside>

    {/* Editable timeline dock, pinned full-width to the bottom of the main area
        (left of the chat), pro-editor style. Always shown in edit mode so users
        can drag clips in to build a timeline from scratch — not only after the
        AI produces cuts. The program monitor stays in the panel. */}
    {dockEl &&
      createPortal(
        <div
          className="max-h-[40vh] min-h-[200px] w-full overflow-y-auto border-t p-3"
          style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
        >
          <TimelineEditor
            threadId={threadId}
            ensureThread={ensureThread}
            token={token}
            onSaved={handleSavedEdit}
          />
        </div>,
        dockEl
      )}
    </>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center py-10 text-center">
      <Sparkles size={30} style={{ color: "var(--accent)" }} />
      <p className="mt-3 text-sm font-semibold">Chat about your footage</p>
      <p className="mt-1 max-w-[18rem] text-xs" style={{ color: "var(--muted)" }}>
        Ask about your clips, talk through ideas, or describe an edit. EDSO
        edits the timeline directly as it works -- you can always undo.
      </p>
    </div>
  );
}

function QuestionCard({
  questions,
  onPick,
}: {
  questions: ThreadQuestion[];
  onPick: (text: string) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      {questions.map((q) => (
        <div
          key={q.id}
          className="flex flex-col gap-2 rounded-2xl border px-3 py-3"
          style={{ borderColor: "var(--accent)", background: "var(--accent-soft)" }}
        >
          <p className="text-sm font-medium">{q.prompt}</p>
          <div className="flex flex-wrap gap-2">
            {q.options.map((opt) => (
              <button
                key={opt}
                onClick={() => onPick(opt)}
                className="rounded-lg border px-3 py-1.5 text-sm font-medium transition-colors hover:bg-[var(--background)]"
                style={{ borderColor: "var(--border)" }}
              >
                {opt}
              </button>
            ))}
          </div>
          <p className="text-xs" style={{ color: "var(--muted)" }}>
            …or just type your own answer below.
          </p>
        </div>
      ))}
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
            ? { background: "var(--accent)", color: "var(--background)" }
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

  const resolved = doc.resolved;
  const coveragePct =
    resolved && resolved.duration_ms
      ? Math.round(
          (1000 *
            resolved.video_layers
              .filter((v) => v.kind === "coverage")
              .reduce((a, v) => a + (v.prog_end_ms - v.prog_start_ms), 0)) /
            resolved.duration_ms
        ) / 10
      : 0;

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

      {doc.operations && doc.operations.length > 0 && (
        <OperationsView operations={doc.operations} coveragePct={coveragePct} />
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

function opVisual(op: EditOperation): {
  icon: React.ReactNode;
  label: string;
  detail: string;
} {
  const span =
    op.from_ms != null && op.to_ms != null
      ? `${fmtClock(op.from_ms)}–${fmtClock(op.to_ms)}`
      : "";
  const src = op.source_file_id ? op.source_file_id.slice(0, 6) : "";
  switch (op.type) {
    case "place_video":
      return {
        icon: <Layers size={13} />,
        label: "Coverage",
        detail: [span, src && `clip ${src}`].filter(Boolean).join(" · "),
      };
    case "place_audio": {
      const role =
        op.audio_kind === "replace"
          ? "Audio replace"
          : op.role === "music"
          ? "Music bed"
          : op.role === "sfx"
          ? "SFX"
          : "Audio";
      const mix = [
        op.gain_db ? `${op.gain_db > 0 ? "+" : ""}${op.gain_db}dB` : "",
        op.duck_db ? `duck ${op.duck_db}dB` : "",
      ]
        .filter(Boolean)
        .join(" · ");
      return {
        icon: <Music size={13} />,
        label: role,
        detail: [span, mix].filter(Boolean).join(" · "),
      };
    }
    case "split_edit":
      return {
        icon: <Scissors size={13} />,
        label: op.kind || "Split edit",
        detail: `audio ${(op.audio_offset_ms ?? 0) > 0 ? "+" : ""}${op.audio_offset_ms ?? 0}ms`,
      };
    case "level":
      return {
        icon: <SlidersHorizontal size={13} />,
        label: "Level",
        detail: [span, op.mute ? "mute" : op.gain_db != null ? `${op.gain_db}dB` : "", op.role]
          .filter(Boolean)
          .join(" · "),
      };
    default:
      return { icon: <Layers size={13} />, label: op.type, detail: span };
  }
}

function OperationsView({
  operations,
  coveragePct,
}: {
  operations: EditOperation[];
  coveragePct: number;
}) {
  return (
    <div className="border-t pt-2" style={{ borderColor: "var(--border)" }}>
      <div className="mb-1.5 flex items-center justify-between">
        <span className="text-xs font-semibold" style={{ color: "var(--muted)" }}>
          A/V layers · {operations.length}
        </span>
        {coveragePct > 0 && (
          <span className="text-xs" style={{ color: "var(--muted)" }}>
            {coveragePct}% covered
          </span>
        )}
      </div>
      <ul className="space-y-1">
        {operations.map((op) => {
          const v = opVisual(op);
          return (
            <li
              key={op.op_id}
              title={op.rationale ?? undefined}
              className="flex items-center gap-2 text-xs"
            >
              <span style={{ color: "var(--accent)" }}>{v.icon}</span>
              <span className="font-medium">{v.label}</span>
              {v.detail && (
                <span style={{ color: "var(--muted)" }}>{v.detail}</span>
              )}
              {op.warnings && op.warnings.length > 0 && (
                <AlertCircle size={11} style={{ color: "var(--danger)" }} />
              )}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

