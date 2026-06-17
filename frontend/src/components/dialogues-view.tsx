"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAuthStore } from "@/stores/auth-store";
import { useDriveStore } from "@/stores/drive-store";
import { FileIcon } from "./file-icon";
import {
  getDialogues,
  getFilePlaybackUrl,
  type DialogueSegment,
  type FileRecord,
} from "@/lib/api";
import { MessageSquare, Play, Pause, Volume2, VolumeX, AlertTriangle } from "lucide-react";

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

export function DialoguesView() {
  const token = useAuthStore((s) => s.session?.access_token);
  const files = useDriveStore((s) => s.files);
  const [level, setLevel] = useState<Level>("sentence");
  const [data, setData] = useState<Record<string, FileDialogues>>({});
  const [loading, setLoading] = useState(false);
  const urlCache = useRef<Record<string, Promise<string | null>>>({});

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

  // Lazy, de-duped presigned playback URL per file (shared across its clips).
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

  const sectionsWithClips = candidates
    .map((f) => ({ file: f, segs: data[f.id]?.[level] ?? [] }))
    .filter((s) => s.segs.length > 0);

  const totalClips = sectionsWithClips.reduce((n, s) => n + s.segs.length, 0);

  return (
    <div>
      <div className="mb-5 flex items-center justify-between">
        <div className="inline-flex rounded-lg p-0.5" style={{ background: "var(--accent-soft)" }}>
          {(["sentence", "topic"] as Level[]).map((lv) => (
            <button
              key={lv}
              onClick={() => setLevel(lv)}
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
          <section key={file.id} className="mb-8">
            <h3 className="mb-3 truncate text-xs font-semibold uppercase tracking-wider" style={{ color: "var(--muted)" }}>
              {file.name}
            </h3>
            <div className="grid grid-cols-2 gap-4 md:grid-cols-3 2xl:grid-cols-4">
              {segs.map((seg) => (
                <DialogueClipCard key={seg.seg_id} file={file} seg={seg} getUrl={getUrl} />
              ))}
            </div>
          </section>
        ))}
    </div>
  );
}

function DialogueClipCard({
  file,
  seg,
  getUrl,
}: {
  file: FileRecord;
  seg: DialogueSegment;
  getUrl: (fileId: string) => Promise<string | null>;
}) {
  const [playUrl, setPlayUrl] = useState<string | null>(null);
  const [muted, setMuted] = useState(true);
  const [desiredPlaying, setDesiredPlaying] = useState(false);
  const [playing, setPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const pinnedRef = useRef(false);

  const inSec = seg.src_in_ms / 1000;
  const outSec = seg.src_out_ms / 1000;
  const dur = seg.src_out_ms - seg.src_in_ms;
  const isVideo = file.file_type === "video";
  const hasOverlap = seg.flags.includes("overlap");
  const isNoisy = seg.flags.includes("noisy");
  const isBackchannel = seg.flags.includes("backchannel");

  async function ensureUrl() {
    if (playUrl) return;
    const url = await getUrl(file.id);
    if (url) setPlayUrl(url);
  }

  useEffect(() => {
    if (videoRef.current) videoRef.current.muted = muted;
  }, [muted, playUrl]);

  // Start at the clip's in-point and play; loop within [in, out].
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !playUrl) return;
    if (desiredPlaying) {
      try {
        if (v.currentTime < inSec || v.currentTime >= outSec) v.currentTime = inSec;
      } catch {
        /* ignore */
      }
      v.muted = muted;
      v.play().then(() => setPlaying(true)).catch(() => {});
    } else {
      v.pause();
      setPlaying(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [desiredPlaying, playUrl]);

  function onLoadedMetadata() {
    const v = videoRef.current;
    if (!v) return;
    try {
      v.currentTime = inSec; // show the clip's first frame as the poster
    } catch {
      /* ignore */
    }
  }

  function onTimeUpdate() {
    const v = videoRef.current;
    if (!v) return;
    if (v.currentTime >= outSec - 0.02 || v.currentTime < inSec - 0.3) {
      try {
        v.currentTime = inSec; // loop the segment
      } catch {
        /* ignore */
      }
    }
  }

  async function handleEnter() {
    if (pinnedRef.current) return;
    await ensureUrl();
    setDesiredPlaying(true);
  }

  function handleLeave() {
    if (pinnedRef.current) return;
    setDesiredPlaying(false);
    const v = videoRef.current;
    if (v) {
      try {
        v.currentTime = inSec;
      } catch {
        /* ignore */
      }
    }
  }

  async function handleCenterToggle(e: React.MouseEvent) {
    e.stopPropagation();
    if (desiredPlaying) {
      pinnedRef.current = false;
      setDesiredPlaying(false);
    } else {
      pinnedRef.current = true;
      await ensureUrl();
      setDesiredPlaying(true);
    }
  }

  function onDragStart(e: React.DragEvent) {
    const payload = JSON.stringify({
      kind: "dialogue",
      file_id: file.id,
      file_name: file.name,
      in_ms: seg.src_in_ms,
      out_ms: seg.src_out_ms,
      content: seg.text,
      speaker: seg.speaker,
    });
    e.dataTransfer.setData("application/x-dialogue-segment", payload);
    e.dataTransfer.setData("text/plain", payload);
    e.dataTransfer.effectAllowed = "copy";
  }

  return (
    <div
      className="group relative flex flex-col overflow-hidden rounded-xl border transition-colors hover:border-[var(--accent)]"
      style={{ borderColor: "var(--border)", background: "var(--background)" }}
    >
      <div
        onMouseEnter={handleEnter}
        onMouseLeave={handleLeave}
        onClick={handleCenterToggle}
        draggable
        onDragStart={onDragStart}
        className="relative flex aspect-video cursor-pointer items-center justify-center overflow-hidden"
        style={{ background: "#000" }}
        title={seg.text}
      >
        {playUrl && (
          <video
            ref={videoRef}
            src={`${playUrl}#t=${inSec.toFixed(2)}`}
            playsInline
            preload="metadata"
            muted={muted}
            onLoadedMetadata={onLoadedMetadata}
            onTimeUpdate={onTimeUpdate}
            className="h-full w-full bg-black object-contain"
          />
        )}
        {!playUrl && <FileIcon type={(isVideo ? "video" : "audio") as "video"} size={32} />}

        {/* Speaker badge (top-left). */}
        {seg.speaker && (
          <span
            className="absolute left-2 top-2 z-20 rounded px-1.5 py-0.5 text-[11px] font-semibold text-white"
            style={{ background: speakerColor(seg.speaker) }}
          >
            {seg.speaker}
          </span>
        )}

        {/* Center play / pause. */}
        <button
          onClick={handleCenterToggle}
          className={`absolute left-1/2 top-1/2 z-20 flex h-11 w-11 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full shadow-lg transition-all ${
            playing ? "opacity-0 group-hover:opacity-100" : "opacity-100"
          }`}
          style={{ background: "var(--accent)" }}
          title={playing ? "Pause" : "Play clip"}
        >
          {playing ? (
            <Pause size={20} className="text-white" fill="white" />
          ) : (
            <Play size={20} className="ml-0.5 text-white" fill="white" />
          )}
        </button>

        {/* Mute toggle (top-right). */}
        {playUrl && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              setMuted((m) => !m);
            }}
            className="absolute right-2 top-2 z-20 flex items-center justify-center rounded-full p-1.5 text-white transition-colors hover:bg-black/40"
            style={{ background: "rgba(0,0,0,0.55)" }}
            title={muted ? "Unmute" : "Mute"}
          >
            {muted ? <VolumeX size={14} /> : <Volume2 size={14} />}
          </button>
        )}

        {/* Duration badge (bottom-right). */}
        <span className="absolute bottom-2 right-2 z-10 rounded bg-black/70 px-1.5 py-0.5 text-[11px] font-medium text-white">
          {fmtDur(dur)}
        </span>

        {/* Overlap warning (bottom-left). */}
        {hasOverlap && (
          <span
            className="absolute bottom-2 left-2 z-10 flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium text-white"
            style={{ background: "rgba(245,158,11,0.85)" }}
            title="Cross-talk: dropping both overlapping clips will collide"
          >
            <AlertTriangle size={11} /> overlap
          </span>
        )}
      </div>

      <div className="p-2.5">
        <p className="line-clamp-2 text-sm leading-snug" style={{ minHeight: "2.5em" }}>
          {seg.text || <em style={{ color: "var(--muted)" }}>(no speech)</em>}
        </p>
        {(isNoisy || isBackchannel) && (
          <div className="mt-1.5 flex flex-wrap gap-1">
            {isNoisy && (
              <span className="rounded px-1.5 py-0.5 text-[10px]" style={{ background: "var(--accent-soft)", color: "var(--muted)" }} title="No clean silence at the cut; used a fixed handle">
                noisy cut
              </span>
            )}
            {isBackchannel && (
              <span className="rounded px-1.5 py-0.5 text-[10px]" style={{ background: "var(--accent-soft)", color: "var(--muted)" }} title="Short acknowledgement (mhm/yeah)">
                backchannel
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
