"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAuthStore } from "@/stores/auth-store";
import { useDriveStore } from "@/stores/drive-store";
import {
  getDialogues,
  getFilePlaybackUrl,
  type DialogueSegment,
  type FileRecord,
} from "@/lib/api";
import { MessageSquare, Play, AlertTriangle, Volume2, GripVertical } from "lucide-react";

type Level = "sentence" | "topic";

type FileDialogues = { sentence: DialogueSegment[]; topic: DialogueSegment[] };

const SPEAKER_COLORS = ["#6366f1", "#10b981", "#f59e0b", "#ec4899", "#06b6d4", "#8b5cf6"];

function speakerColor(spk: string | null): string {
  if (!spk) return "var(--muted)";
  const n = parseInt(spk.replace(/\D/g, ""), 10);
  return SPEAKER_COLORS[(isNaN(n) ? 0 : n) % SPEAKER_COLORS.length];
}

function fmtDur(ms: number): string {
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.round(s - m * 60)).padStart(2, "0")}`;
}

function fmtTime(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  return `${m}:${String(s - m * 60).padStart(2, "0")}`;
}

export function DialoguesView() {
  const token = useAuthStore((s) => s.session?.access_token);
  const files = useDriveStore((s) => s.files);
  const [level, setLevel] = useState<Level>("sentence");
  const [data, setData] = useState<Record<string, FileDialogues>>({});
  const [loading, setLoading] = useState(false);
  const [playing, setPlaying] = useState<string | null>(null);
  const urlCache = useRef<Record<string, string>>({});

  const candidates = useMemo(
    () =>
      files.filter(
        (f) => (f.file_type === "video" || f.file_type === "audio") && f.l1_status === "ready"
      ),
    [files]
  );

  useEffect(() => {
    if (!token || candidates.length === 0) {
      setData({});
      return;
    }
    let cancelled = false;
    setLoading(true);
    Promise.allSettled(candidates.map((f) => getDialogues(f.id, token))).then((results) => {
      if (cancelled) return;
      const next: Record<string, FileDialogues> = {};
      results.forEach((r, i) => {
        if (r.status === "fulfilled") {
          next[candidates[i].id] = {
            sentence: r.value.sentence ?? [],
            topic: r.value.topic ?? [],
          };
        }
      });
      setData(next);
      setLoading(false);
    });
    return () => {
      cancelled = true;
    };
  }, [token, candidates]);

  const playClip = useCallback(
    async (fileId: string, seg: DialogueSegment) => {
      if (!token) return;
      if (!urlCache.current[fileId]) {
        try {
          const { url } = await getFilePlaybackUrl(fileId, token);
          urlCache.current[fileId] = url;
        } catch {
          return;
        }
      }
      setPlaying(seg.seg_id);
    },
    [token]
  );

  const sectionsWithClips = candidates
    .map((f) => ({ file: f, segs: data[f.id]?.[level] ?? [] }))
    .filter((s) => s.segs.length > 0);

  const totalClips = sectionsWithClips.reduce((n, s) => n + s.segs.length, 0);

  return (
    <div>
      {/* Level switch */}
      <div className="mb-5 flex items-center justify-between">
        <div
          className="inline-flex rounded-lg p-0.5"
          style={{ background: "var(--accent-soft)" }}
        >
          {(["sentence", "topic"] as Level[]).map((lv) => (
            <button
              key={lv}
              onClick={() => {
                setLevel(lv);
                setPlaying(null);
              }}
              className="rounded-md px-4 py-1.5 text-sm font-medium capitalize transition-colors"
              style={{
                background: level === lv ? "var(--accent)" : "transparent",
                color: level === lv ? "#fff" : "var(--accent)",
              }}
            >
              {lv}
            </button>
          ))}
        </div>
        {totalClips > 0 && (
          <span className="text-sm" style={{ color: "var(--muted)" }}>
            {totalClips} {level} clip{totalClips === 1 ? "" : "s"}
          </span>
        )}
      </div>

      {loading && (
        <p className="py-12 text-center text-sm" style={{ color: "var(--muted)" }}>
          Loading dialogue selects…
        </p>
      )}

      {!loading && sectionsWithClips.length === 0 && (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <MessageSquare size={36} style={{ color: "var(--accent)" }} />
          <p className="mt-4 text-lg font-semibold">No dialogue selects yet</p>
          <p className="mt-1 max-w-sm text-sm" style={{ color: "var(--muted)" }}>
            Upload footage with speech. Once analyzed, clean per-line clips will
            appear here, ready to drop into a timeline.
          </p>
        </div>
      )}

      {!loading &&
        sectionsWithClips.map(({ file, segs }) => (
          <div key={file.id} className="mb-8">
            <h3 className="mb-3 truncate text-sm font-semibold" style={{ color: "var(--muted)" }}>
              {file.name}
            </h3>
            <div className="space-y-2">
              {segs.map((seg) => (
                <ClipTile
                  key={seg.seg_id}
                  file={file}
                  seg={seg}
                  playing={playing === seg.seg_id}
                  playbackUrl={urlCache.current[file.id]}
                  onPlay={() => playClip(file.id, seg)}
                  onStop={() => setPlaying(null)}
                />
              ))}
            </div>
          </div>
        ))}
    </div>
  );
}

function ClipTile({
  file,
  seg,
  playing,
  playbackUrl,
  onPlay,
  onStop,
}: {
  file: FileRecord;
  seg: DialogueSegment;
  playing: boolean;
  playbackUrl?: string;
  onPlay: () => void;
  onStop: () => void;
}) {
  const dur = seg.src_out_ms - seg.src_in_ms;
  const hasOverlap = seg.flags.includes("overlap");
  const isNoisy = seg.flags.includes("noisy");
  const isBackchannel = seg.flags.includes("backchannel");

  function onDragStart(e: React.DragEvent) {
    // Integration seam for "drop to timeline": EditSegment-shaped payload.
    const payload = {
      kind: "dialogue",
      file_id: file.id,
      in_ms: seg.src_in_ms,
      out_ms: seg.src_out_ms,
      content: seg.text,
      speaker: seg.speaker,
    };
    e.dataTransfer.setData("application/x-dialogue-segment", JSON.stringify(payload));
    e.dataTransfer.effectAllowed = "copy";
  }

  return (
    <div
      draggable
      onDragStart={onDragStart}
      className="group rounded-lg border p-3 transition-colors hover:border-[var(--accent)]"
      style={{ borderColor: "var(--border)", background: "var(--card)" }}
    >
      <div className="flex items-start gap-3">
        <GripVertical
          size={16}
          className="mt-0.5 shrink-0 cursor-grab opacity-30 group-hover:opacity-60"
        />

        {seg.speaker && (
          <span
            className="mt-0.5 shrink-0 rounded px-1.5 py-0.5 text-xs font-semibold text-white"
            style={{ background: speakerColor(seg.speaker) }}
          >
            {seg.speaker}
          </span>
        )}

        <p className="flex-1 text-sm leading-snug">{seg.text || <em style={{ color: "var(--muted)" }}>(no text)</em>}</p>

        <div className="flex shrink-0 items-center gap-2">
          {hasOverlap && (
            <span title="Cross-talk: dropping both overlapping clips will collide" className="flex items-center gap-1 text-xs" style={{ color: "#f59e0b" }}>
              <AlertTriangle size={13} /> overlap
            </span>
          )}
          {isNoisy && (
            <span title="No clean silence at the cut; used a fixed handle" className="flex items-center gap-1 text-xs" style={{ color: "var(--muted)" }}>
              <Volume2 size={13} /> noisy
            </span>
          )}
          {isBackchannel && (
            <span title="Short acknowledgement (mhm/yeah)" className="text-xs" style={{ color: "var(--muted)" }}>
              backchannel
            </span>
          )}
          <span className="font-mono text-xs tabular-nums" style={{ color: "var(--muted)" }}>
            {fmtTime(seg.src_in_ms)} · {fmtDur(dur)}
          </span>
          <button
            onClick={playing ? onStop : onPlay}
            className="rounded-md p-1.5 transition-colors hover:bg-[var(--accent-soft)]"
            title={playing ? "Stop" : "Preview"}
          >
            <Play size={15} style={{ color: "var(--accent)" }} />
          </button>
        </div>
      </div>

      {playing && playbackUrl && (
        <video
          key={seg.seg_id}
          className="mt-3 w-full max-w-md rounded-md"
          src={`${playbackUrl}#t=${(seg.src_in_ms / 1000).toFixed(2)},${(seg.src_out_ms / 1000).toFixed(2)}`}
          controls
          autoPlay
          onEnded={onStop}
        />
      )}
    </div>
  );
}
