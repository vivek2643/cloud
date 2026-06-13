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
  Layers,
  Music,
  Scissors,
  SlidersHorizontal,
  Pencil,
} from "lucide-react";
import { useDriveStore } from "@/stores/drive-store";
import { RenderBar } from "@/components/render-bar";
import { TimelineEditor } from "@/components/timeline-editor";
import { useAuthStore } from "@/stores/auth-store";
import {
  createEditThread,
  getEditThread,
  sendEditMessage,
  getFilePlaybackUrl,
  type EditThread,
  type EditThreadStatus,
  type EditSegment,
  type EditOperation,
  type ResolvedTimeline,
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
  const [editing, setEditing] = useState(false);

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
    setEditing(false);
  }

  function handleSavedEdit(newVersion: number, newDoc: NonNullable<EditThread["document"]>) {
    setThread((prev) =>
      prev ? { ...prev, document: newDoc, document_version: newVersion } : prev
    );
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
          {doc?.timeline && doc.timeline.length > 0 && (
            <button
              onClick={() => setEditing((v) => !v)}
              className="rounded-lg p-1.5 transition-colors hover:bg-[var(--accent-soft)]"
              style={editing ? { color: "var(--accent)" } : undefined}
              title={editing ? "Back to plan" : "Edit timeline"}
            >
              <Pencil size={16} />
            </button>
          )}
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
      <CompositePreview
        resolved={
          doc?.resolved ??
          (doc?.timeline?.length ? spineResolved(doc.timeline) : null)
        }
        token={token}
        layered={(doc?.operations?.length ?? 0) > 0}
      />

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
        {!threadId && (
          <EmptyState />
        )}

        {userTurns.map((t, i) => (
          <Bubble key={`u${i}`} role="user">
            {t}
          </Bubble>
        ))}

        {doc && editing && threadId ? (
          <TimelineEditor
            threadId={threadId}
            doc={doc}
            version={thread?.document_version ?? 0}
            token={token}
            onSaved={handleSavedEdit}
          />
        ) : (
          doc && <DocumentView doc={doc} version={thread?.document_version ?? null} />
        )}

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

// --- Program monitor: a real compositor over the resolved layer set ---
//
// One muted <video> shows the top video layer at the program instant; one
// <audio> per audio layer is mixed by gain+duck. Everything is slaved to a
// performance.now() master clock, with drift-correcting seeks so picture and
// the (possibly decoupled) audio beds stay aligned.

const SYNC_DRIFT_S = 0.18; // re-seek a media element if it drifts past this

function spineResolved(timeline: EditSegment[]): ResolvedTimeline {
  let t = 0;
  const video: ResolvedTimeline["video_layers"] = [];
  const audio: ResolvedTimeline["audio_layers"] = [];
  for (const seg of timeline) {
    const dur = Math.max(0, seg.out_ms - seg.in_ms);
    video.push({
      layer_id: `v_${seg.seg_id}`,
      source_file_id: seg.file_id,
      src_in_ms: seg.in_ms,
      src_out_ms: seg.out_ms,
      prog_start_ms: t,
      prog_end_ms: t + dur,
      z: 0,
      layout: "full_frame",
      opacity: 1,
      kind: "spine",
    });
    audio.push({
      layer_id: `a_${seg.seg_id}`,
      role: "dialogue",
      source_file_id: seg.file_id,
      src_in_ms: seg.in_ms,
      src_out_ms: seg.out_ms,
      prog_start_ms: t,
      prog_end_ms: t + dur,
      gain_db: 0,
      duck_db: 0,
      kind: "spine",
    });
    t += dur;
  }
  return { duration_ms: t, video_layers: video, audio_layers: audio };
}

function dbToGain(db: number): number {
  if (db <= -119) return 0;
  return Math.min(1, Math.pow(10, db / 20));
}

function srcMatches(el: HTMLMediaElement, url: string): boolean {
  if (!el.src || !url) return false;
  return el.src === url || el.src.split("?")[0] === url.split("?")[0];
}

function CompositePreview({
  resolved,
  token,
  layered = false,
}: {
  resolved: ResolvedTimeline | null;
  token: string | undefined;
  layered?: boolean;
}) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const audioEls = useRef<Map<string, HTMLAudioElement>>(new Map());
  const [urls, setUrls] = useState<Record<string, string>>({});
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [progMs, setProgMs] = useState(0);

  const progRef = useRef(0);
  const playingRef = useRef(false);
  const mutedRef = useRef(false);
  const lastTsRef = useRef(0);
  const lastDisplayRef = useRef(0);
  const rafRef = useRef<number | null>(null);
  const curVideoSrcRef = useRef<string>("");

  const duration = resolved?.duration_ms ?? 0;
  const hasTimeline = !!resolved && resolved.video_layers.length > 0;

  const fileIds = useMemo(() => {
    if (!resolved) return [];
    const s = new Set<string>();
    resolved.video_layers.forEach((v) => s.add(v.source_file_id));
    resolved.audio_layers.forEach((a) => s.add(a.source_file_id));
    return Array.from(s);
  }, [resolved]);

  // Resolve playback URLs for every source clip referenced by any layer.
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

  // Place every media element at the right source position for program time t.
  const syncTo = useCallback(
    (t: number, opts: { hardSeek?: boolean } = {}) => {
      if (!resolved) return;
      const hard = opts.hardSeek ?? false;

      // VIDEO — top z layer covering t.
      const v = videoRef.current;
      if (v) {
        // React's `muted` JSX prop is unreliable; enforce it on the node so the
        // picture's own audio track never doubles the audio bus (comb-filtering).
        if (!v.muted) v.muted = true;
        let top: ResolvedTimeline["video_layers"][number] | null = null;
        for (const layer of resolved.video_layers) {
          if (layer.prog_start_ms <= t && t < layer.prog_end_ms) {
            if (!top || layer.z > top.z) top = layer;
          }
        }
        if (top) {
          const url = urls[top.source_file_id];
          if (url) {
            const want = (top.src_in_ms + (t - top.prog_start_ms)) / 1000;
            if (curVideoSrcRef.current !== top.source_file_id || !srcMatches(v, url)) {
              curVideoSrcRef.current = top.source_file_id;
              v.src = url;
              const onMeta = () => {
                v.currentTime = want;
                if (playingRef.current) v.play().catch(() => {});
              };
              if (v.readyState >= 1) onMeta();
              else v.addEventListener("loadedmetadata", onMeta, { once: true });
            } else if (hard || Math.abs(v.currentTime - want) > SYNC_DRIFT_S) {
              v.currentTime = want;
            }
          }
        }
      }

      // AUDIO — every layer sounding at t plays; the rest pause.
      for (const a of resolved.audio_layers) {
        const el = audioEls.current.get(a.layer_id);
        if (!el) continue;
        const active = a.prog_start_ms <= t && t < a.prog_end_ms;
        if (active) {
          el.volume = mutedRef.current ? 0 : dbToGain(a.gain_db + a.duck_db);
          const want = (a.src_in_ms + (t - a.prog_start_ms)) / 1000;
          if (hard || Math.abs(el.currentTime - want) > SYNC_DRIFT_S) {
            try {
              el.currentTime = want;
            } catch {
              /* not seekable yet */
            }
          }
          if (playingRef.current && el.paused) el.play().catch(() => {});
        } else if (!el.paused) {
          el.pause();
        }
      }
    },
    [resolved, urls]
  );

  const stopRaf = useCallback(() => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const pauseAll = useCallback(() => {
    playingRef.current = false;
    setPlaying(false);
    videoRef.current?.pause();
    audioEls.current.forEach((el) => el.pause());
    stopRaf();
  }, [stopRaf]);

  // Master clock: advance program time in real time, nudging media into sync.
  const loop = useCallback(
    (ts: number) => {
      if (lastTsRef.current) {
        const dt = ts - lastTsRef.current;
        let t = progRef.current + dt;
        if (t >= duration) {
          t = duration;
          progRef.current = t;
          syncTo(t);
          pauseAll();
          progRef.current = 0;
          lastDisplayRef.current = 0;
          setProgMs(0);
          return;
        }
        progRef.current = t;
        syncTo(t);
        // Throttle the React re-render (~10fps) so 60fps state churn doesn't
        // starve audio/video decoding and cause stutter.
        if (Math.abs(t - lastDisplayRef.current) >= 100) {
          lastDisplayRef.current = t;
          setProgMs(t);
        }
      }
      lastTsRef.current = ts;
      rafRef.current = requestAnimationFrame(loop);
    },
    [duration, syncTo, pauseAll]
  );

  function togglePlay() {
    if (!hasTimeline) return;
    if (playingRef.current) {
      pauseAll();
      return;
    }
    playingRef.current = true;
    setPlaying(true);
    lastTsRef.current = 0;
    syncTo(progRef.current, { hardSeek: true });
    videoRef.current?.play().catch(() => {});
    stopRaf();
    rafRef.current = requestAnimationFrame(loop);
  }

  function scrub(t: number) {
    progRef.current = t;
    lastDisplayRef.current = t;
    setProgMs(t);
    lastTsRef.current = 0;
    syncTo(t, { hardSeek: true });
  }

  // Keep mute reactive without restarting the clock.
  useEffect(() => {
    mutedRef.current = muted;
    if (resolved) {
      resolved.audio_layers.forEach((a) => {
        const el = audioEls.current.get(a.layer_id);
        if (el) el.volume = muted ? 0 : dbToGain(a.gain_db + a.duck_db);
      });
    }
  }, [muted, resolved]);

  // Reset to the start whenever the resolved plan changes shape.
  useEffect(() => {
    pauseAll();
    progRef.current = 0;
    lastDisplayRef.current = 0;
    setProgMs(0);
    curVideoSrcRef.current = "";
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resolved]);

  useEffect(() => stopRaf, [stopRaf]);

  const activeVideo = useMemo(() => {
    if (!resolved) return null;
    let top: ResolvedTimeline["video_layers"][number] | null = null;
    for (const layer of resolved.video_layers) {
      if (layer.prog_start_ms <= progMs && progMs < layer.prog_end_ms) {
        if (!top || layer.z > top.z) top = layer;
      }
    }
    return top;
  }, [resolved, progMs]);

  const activeBeds = resolved
    ? resolved.audio_layers.filter(
        (a) =>
          a.kind !== "spine" &&
          a.prog_start_ms <= progMs &&
          progMs < a.prog_end_ms
      ).length
    : 0;

  return (
    <div className="border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
      <div
        className="relative aspect-video w-full overflow-hidden rounded-lg"
        style={{ background: "#000" }}
      >
        {hasTimeline ? (
          <video ref={videoRef} className="h-full w-full" muted playsInline />
        ) : (
          <div
            className="flex h-full w-full items-center justify-center text-xs"
            style={{ color: "#666" }}
          >
            <Film size={28} />
          </div>
        )}

        {/* hidden audio bus — one element per audio layer, mixed live */}
        {resolved?.audio_layers.map((a) => (
          <audio
            key={a.layer_id}
            ref={(el) => {
              if (el) audioEls.current.set(a.layer_id, el);
              else audioEls.current.delete(a.layer_id);
            }}
            src={urls[a.source_file_id] || undefined}
            preload="auto"
          />
        ))}

        {/* live badges for what's overriding the spine right now */}
        {hasTimeline && (activeVideo?.kind === "coverage" || activeBeds > 0) && (
          <div className="absolute left-2 top-2 flex gap-1">
            {activeVideo?.kind === "coverage" && (
              <span className="flex items-center gap-1 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white">
                <Layers size={10} /> coverage
              </span>
            )}
            {activeBeds > 0 && (
              <span className="flex items-center gap-1 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white">
                <Music size={10} /> {activeBeds} bed{activeBeds === 1 ? "" : "s"}
              </span>
            )}
          </div>
        )}
      </div>

      {hasTimeline && (
        <>
          <input
            type="range"
            min={0}
            max={Math.max(1, duration)}
            value={Math.min(progMs, duration)}
            onChange={(e) => scrub(Number(e.target.value))}
            className="mt-2 w-full accent-[var(--accent)]"
          />
          <div className="mt-1 flex items-center gap-2">
            <button
              onClick={() => scrub(Math.max(0, progMs - 5000))}
              className="rounded p-1 transition-colors hover:bg-[var(--accent-soft)]"
              title="Back 5s"
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
              onClick={() => scrub(Math.min(duration, progMs + 5000))}
              className="rounded p-1 transition-colors hover:bg-[var(--accent-soft)]"
              title="Forward 5s"
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
            <span
              className="ml-auto text-xs tabular-nums"
              style={{ color: "var(--muted)" }}
            >
              {fmtClock(progMs)} / {fmtClock(duration)}
            </span>
          </div>
          {layered && (
            <p className="mt-1 text-[11px]" style={{ color: "var(--muted)" }}>
              Composited preview — beds &amp; coverage mixed live; split-edit drift
              within ~0.2s.
            </p>
          )}
        </>
      )}
    </div>
  );
}
