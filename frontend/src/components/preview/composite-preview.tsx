"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type CSSProperties } from "react";
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Volume2,
  VolumeX,
  Film,
  Layers,
  Music,
} from "lucide-react";
import { getFile, getFilePlaybackUrl, type ResolvedTimeline } from "@/lib/api";
import { resolveTimeline, sampleMotion } from "@/lib/resolve-timeline";
import { useEditDocStore } from "@/stores/edit-doc-store";
import { useTransport, FRAME_MS, formatTimecode } from "@/stores/transport-store";
import { useAudioEngine } from "./use-audio-engine";
import { useVideoPicture } from "./use-video-picture";

/**
 * Program monitor. Reads the LIVE working document from the edit store, resolves
 * it client-side (same logic as the backend renderer), and plays it back with a
 * WebAudio mixer + double-buffered video. Any timeline edit re-resolves and the
 * preview reflects it on the next frame — no save round-trip.
 */
export function CompositePreview({ token }: { token: string | undefined }) {
  const timeline = useEditDocStore((s) => s.timeline);
  const operations = useEditDocStore((s) => s.operations);
  const durations = useEditDocStore((s) => s.durations);
  const aspect = useEditDocStore((s) => s.aspect);
  const mergeDurations = useEditDocStore((s) => s.mergeDurations);

  const resolved: ResolvedTimeline | null = useMemo(
    () =>
      timeline.length
        ? resolveTimeline({ timeline, operations, format: { aspect } }, durations)
        : null,
    [timeline, operations, durations, aspect]
  );

  // Frame box ratio for the program monitor, matching the delivery aspect.
  const frameRatio =
    aspect === "portrait" ? "9 / 16" : aspect === "square" ? "1 / 1" : "16 / 9";

  const [urls, setUrls] = useState<Record<string, string>>({});
  const [muted, setMuted] = useState(false);

  // Shared transport: the single source of truth for program time + play state,
  // read by both this monitor and the timeline so they stay in lockstep.
  const playing = useTransport((s) => s.playing);
  const progMs = useTransport((s) => s.progMs);
  const seekSeq = useTransport((s) => s.seekSeq);
  const seekTargetMs = useTransport((s) => s.seekTargetMs);
  const setPlaying = useTransport((s) => s.setPlaying);
  const togglePlaying = useTransport((s) => s.togglePlaying);
  const publish = useTransport((s) => s.publish);
  const seekTo = useTransport((s) => s.seek);
  const setDuration = useTransport((s) => s.setDuration);
  const resetTransport = useTransport((s) => s.reset);

  // Engine internals. The AUDIO ENGINE is the master clock (sample-accurate);
  // the rAF loop just reads it to drive the picture + publish a frame-snapped
  // program time. `progRef` mirrors the clock for seeks/resumes.
  const progRef = useRef(0);
  const lastFrameRef = useRef(-1);
  const rafRef = useRef<number | null>(null);

  const duration = resolved?.duration_ms ?? 0;
  const hasTimeline = !!resolved && resolved.video_layers.length > 0;

  const fileIds = useMemo(() => {
    if (!resolved) return [] as string[];
    const s = new Set<string>();
    resolved.video_layers.forEach((v) => s.add(v.source_file_id));
    resolved.audio_layers.forEach((a) => s.add(a.source_file_id));
    return Array.from(s).filter(Boolean);
  }, [resolved]);

  const engine = useAudioEngine(resolved, urls);
  const picture = useVideoPicture(resolved, urls);

  // Resolve playback URLs + source durations for every referenced clip.
  useEffect(() => {
    if (!token) return;
    let cancelled = false;
    (async () => {
      const newDur: Record<string, number> = {};
      const entries = await Promise.all(
        fileIds.map(async (id) => {
          if (urls[id]) return [id, urls[id]] as const;
          try {
            const { url } = await getFilePlaybackUrl(id, token);
            if (durations[id] == null) {
              try {
                const f = await getFile(id, token);
                if (f.duration_seconds) newDur[id] = Math.round(f.duration_seconds * 1000);
              } catch {
                /* duration is best-effort */
              }
            }
            return [id, url] as const;
          } catch {
            return [id, ""] as const;
          }
        })
      );
      if (cancelled) return;
      if (Object.keys(newDur).length) mergeDurations(newDur);
      setUrls((prev) => {
        const next = { ...prev };
        for (const [id, url] of entries) if (url) next[id] = url;
        return next;
      });
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fileIds.join(","), token]);

  const stopRaf = useCallback(() => {
    if (rafRef.current != null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  // Publish program duration so the transport can clamp seeks.
  useEffect(() => {
    setDuration(duration);
  }, [duration, setDuration]);

  // The rAF loop: read the audio master clock, drive the picture, and publish a
  // frame-snapped time only when the frame index changes (≈fps Hz). No audio is
  // touched here — it's already scheduled on the AudioContext.
  const loop = useCallback(() => {
    const t = engine.nowMs();
    if (duration > 0 && t >= duration) {
      engine.pause();
      engine.seek(0, false);
      progRef.current = 0;
      lastFrameRef.current = 0;
      picture.sync(0, false);
      publish(0);
      setPlaying(false); // the [playing] effect tears down the loop
      return;
    }
    progRef.current = t;
    picture.sync(t, true);
    const frame = Math.round(t / FRAME_MS);
    if (frame !== lastFrameRef.current) {
      lastFrameRef.current = frame;
      publish(frame * FRAME_MS);
    }
    rafRef.current = requestAnimationFrame(loop);
  }, [duration, engine, picture, publish, setPlaying]);

  // Start/stop purely from the shared `playing` flag, so the play button here
  // and spacebar/scrub on the timeline drive the same machine.
  useEffect(() => {
    if (!playing || !hasTimeline) {
      engine.pause();
      picture.stop();
      stopRaf();
      return;
    }
    engine.play(progRef.current);
    picture.sync(progRef.current, true);
    stopRaf();
    rafRef.current = requestAnimationFrame(loop);
    return stopRaf;
  }, [playing, hasTimeline, loop, engine, picture, stopRaf]);

  // External seek (timeline click/scrub, step buttons, monitor scrubber): move
  // the master clock + reschedule audio + park the picture on the new frame.
  useEffect(() => {
    progRef.current = seekTargetMs;
    lastFrameRef.current = Math.round(seekTargetMs / FRAME_MS);
    const playingNow = useTransport.getState().playing;
    engine.seek(seekTargetMs, playingNow);
    picture.sync(seekTargetMs, playingNow);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seekSeq]);

  function handleTogglePlay() {
    if (!hasTimeline) return;
    togglePlaying();
  }

  useEffect(() => {
    engine.setMuted(muted);
  }, [muted, engine]);

  // Reset to the start whenever the plan changes shape (segment count / ids).
  const shapeKey = useMemo(
    () => (resolved ? resolved.video_layers.map((v) => v.layer_id).join(",") : ""),
    [resolved]
  );
  useEffect(() => {
    resetTransport();
    engine.stop();
    progRef.current = 0;
    lastFrameRef.current = 0;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shapeKey]);

  // Edits re-resolve the timeline: decode any new files and reschedule audio
  // from the current playhead so the change is heard immediately. While paused
  // we just re-park the picture; the audio re-arms on the next play.
  useEffect(() => {
    engine.prepare();
    const playingNow = useTransport.getState().playing;
    engine.seek(progRef.current, playingNow);
    picture.sync(progRef.current, playingNow);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resolved, urls]);

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
        (a) => a.kind !== "spine" && a.prog_start_ms <= progMs && progMs < a.prog_end_ms
      ).length
    : 0;

  // CSS framing for the visible picture, mirroring the render's transform chain
  // (rotate -> fit -> zoom). cover/contain + anchor are exact; rotate/zoom are
  // best-effort (the automatic Phase-1 path emits neither).
  const videoStyle: CSSProperties = useMemo(() => {
    const t = activeVideo?.transform;
    // Motion (push-in/follow) fills a cover base; show a stable REPRESENTATIVE
    // frame (the path's midpoint) so the preview indicates the zoom/track. The
    // render animates the full path frame-by-frame (it is authoritative).
    const mid = t?.motion ? sampleMotion(t.motion, t.motion.dur_ms / 2) : null;
    const fit = mid ? "cover" : t?.fit ?? (aspect === "landscape" ? "contain" : "cover");
    const anchor = t?.anchor ?? "center";
    const focusPoint = mid ? { cx: mid.cx, cy: mid.cy } : t?.focus ?? null;
    // A focus point wins over the anchor enum: place it via object-position so
    // cover-crop keeps the subject in frame. This is the CSS approximation of the
    // render's focus-centered crop (the render is authoritative).
    let objectPosition: string;
    if (focusPoint) {
      const px = Math.round(Math.min(1, Math.max(0, focusPoint.cx)) * 100);
      const py = Math.round(Math.min(1, Math.max(0, focusPoint.cy)) * 100);
      objectPosition = `${px}% ${py}%`;
    } else {
      objectPosition =
        anchor === "left"
          ? "left center"
          : anchor === "right"
            ? "right center"
            : anchor === "top"
              ? "center top"
              : anchor === "bottom"
                ? "center bottom"
                : "center";
    }
    const tf: string[] = [];
    if (t?.rotate) tf.push(`rotate(${t.rotate}deg)`);
    const scale = mid ? mid.scale : t?.zoom && t.zoom > 1 ? t.zoom : 1;
    if (scale > 1) tf.push(`scale(${scale})`);
    return {
      transition: "opacity 60ms linear",
      objectFit: fit,
      objectPosition,
      ...(tf.length ? { transform: tf.join(" ") } : {}),
    };
  }, [activeVideo, aspect]);

  return (
    <div className="border-b px-4 py-3" style={{ borderColor: "var(--border)" }}>
      <div className="flex w-full justify-center">
      <div
        className="relative overflow-hidden rounded-lg"
        style={{
          background: "#000",
          aspectRatio: frameRatio,
          width: aspect === "landscape" ? "100%" : "auto",
          height: aspect === "landscape" ? undefined : "min(70vh, 460px)",
          maxWidth: "100%",
        }}
      >
        {hasTimeline ? (
          <>
            <video
              ref={picture.attachA}
              className="absolute inset-0 h-full w-full"
              style={videoStyle}
              muted
              playsInline
            />
            <video
              ref={picture.attachB}
              className="absolute inset-0 h-full w-full"
              style={videoStyle}
              muted
              playsInline
            />
          </>
        ) : (
          <div
            className="flex h-full w-full items-center justify-center text-xs"
            style={{ color: "#666" }}
          >
            <Film size={28} />
          </div>
        )}

        {hasTimeline && ((activeVideo && activeVideo.z > 0) || activeBeds > 0) && (
          <div className="absolute left-2 top-2 flex gap-1">
            {activeVideo && activeVideo.z > 0 && (
              <span className="flex items-center gap-1 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white">
                <Layers size={10} /> overlay
              </span>
            )}
            {activeBeds > 0 && (
              <span className="flex items-center gap-1 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white">
                <Music size={10} /> {activeBeds} audio
              </span>
            )}
          </div>
        )}
      </div>
      </div>

      {hasTimeline && (
        <>
          <input
            type="range"
            min={0}
            max={Math.max(1, duration)}
            step={FRAME_MS}
            value={Math.min(progMs, duration)}
            onChange={(e) => seekTo(Number(e.target.value))}
            className="mt-2 w-full accent-[var(--accent)]"
          />
          <div className="mt-1 flex items-center gap-2">
            <button
              onClick={() => seekTo(progMs - 5000)}
              className="rounded p-1 transition-colors hover:bg-[var(--accent-soft)]"
              title="Back 5s"
            >
              <SkipBack size={15} />
            </button>
            <button
              onClick={handleTogglePlay}
              className="rounded-full p-1.5"
              style={{ background: "var(--accent)", color: "var(--background)" }}
              title={playing ? "Pause" : "Play"}
            >
              {playing ? <Pause size={15} /> : <Play size={15} />}
            </button>
            <button
              onClick={() => seekTo(progMs + 5000)}
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
            <span className="ml-auto text-xs tabular-nums" style={{ color: "var(--muted)" }}>
              {formatTimecode(progMs)} / {formatTimecode(duration)}
            </span>
          </div>
        </>
      )}
    </div>
  );
}
