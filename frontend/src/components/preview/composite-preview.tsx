"use client";

import { forwardRef, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Film, Layers, Music } from "lucide-react";
import { getFile, getFilePlaybackUrl, type ResolvedTimeline } from "@/lib/api";
import { resolveTimeline } from "@/lib/resolve-timeline";
import { documentToProject } from "@/lib/edit-project";
import { useEditDocStore } from "@/stores/edit-doc-store";
import { useTimelineView, type TrackMeta } from "@/stores/timeline-view";
import { useTransport, FRAME_MS } from "@/stores/transport-store";
import { useProgramPlayer } from "./use-program-player";
import { RenderBar } from "@/components/render-bar";

interface CompositePreviewProps {
  token: string | undefined;
  /** Export overlay (SS2.3) needs the thread/version to render into; omitted
   * (null) hides the overlay entirely (nothing to export into yet). */
  threadId: string | null;
  version: number | null;
  /** Hover over the frame drives the external hover-morphing "Chat" bar
   * (SS2.5) -- this component only reports the boolean, the controls
   * themselves live in ai-edit-panel.tsx. */
  onHoverChange?: (hovering: boolean) => void;
}

/**
 * Preview-ONLY track mute, applied AFTER `resolveTimeline` so the
 * backend-parity resolver itself never sees this ephemeral, session-only UI
 * state (mute lives in stores/timeline-view.ts, never the document).
 * Video-track mute hides that layer outright ("view-only" per the plan);
 * audio-track mute already flows through as a real gain_db on the document
 * (timeline-editor.tsx's toggleTrackMute), so there's nothing more to do
 * for audio here.
 */
function applyTrackMeta(
  resolved: ResolvedTimeline,
  trackMeta: Record<string, TrackMeta>,
  project: ReturnType<typeof documentToProject>,
  gradeBypass: boolean
): ResolvedTimeline {
  const anyMeta = Object.keys(trackMeta).length > 0;
  if (!anyMeta && !gradeBypass) return resolved;

  const videoTrackIdByZ = new Map<number, string>();
  for (const t of project.tracks) {
    if (t.kind === "video") videoTrackIdByZ.set(t.z, t.id);
  }

  let video_layers = resolved.video_layers;
  if (anyMeta) {
    video_layers = video_layers.filter((v) => {
      const trackId = videoTrackIdByZ.get(v.z);
      return !(trackId && trackMeta[trackId]?.mute);
    });
  }
  // Before/after (SS12): strip every layer's grade so the preview shows the
  // ungraded picture, without touching the document at all.
  if (gradeBypass) {
    video_layers = video_layers.map((v) => (v.grade ? { ...v, grade: undefined } : v));
  }

  return { ...resolved, video_layers };
}

/**
 * Program monitor. Reads the LIVE working document from the edit store, resolves
 * it client-side (same logic as the backend renderer), and plays it back with a
 * pooled-media-element program player (see use-program-player): Web Audio mixes
 * the audio, a slot pool shows/pre-warms the picture, and the AudioContext is
 * the master clock. Any timeline edit re-resolves and the preview reflects it on
 * the next frame — no save round-trip.
 *
 * Playback CONTROLS live outside this component now (editor_ui.plan.md
 * SS2.5's hover-morphing "Chat" bar in ai-edit-panel.tsx) -- this component
 * only hosts the picture, the Export overlay (SS2.3), and hover/fullscreen
 * plumbing for that external control surface. `frameRef` is forwarded onto
 * the frame div itself so the parent can call `requestFullscreen()` on it.
 */
export const CompositePreview = forwardRef<HTMLDivElement, CompositePreviewProps>(
  function CompositePreview({ token, threadId, version, onHoverChange }, frameRef) {
  const timeline = useEditDocStore((s) => s.timeline);
  const operations = useEditDocStore((s) => s.operations);
  const layoutRegions = useEditDocStore((s) => s.layoutRegions);
  const durations = useEditDocStore((s) => s.durations);
  const aspect = useEditDocStore((s) => s.aspect);
  const look = useEditDocStore((s) => s.look);
  const mergeDurations = useEditDocStore((s) => s.mergeDurations);

  const trackMeta = useTimelineView((s) => s.trackMeta);
  const gradeBypass = useTimelineView((s) => s.gradeBypass);
  const project = useMemo(
    () => documentToProject(timeline, operations, aspect),
    [timeline, operations, aspect]
  );

  const resolved: ResolvedTimeline | null = useMemo(() => {
    if (!timeline.length) return null;
    const r = resolveTimeline(
      { timeline, operations, layout_regions: layoutRegions, format: { aspect }, look },
      durations
    );
    return applyTrackMeta(r, trackMeta, project, gradeBypass);
  }, [timeline, operations, layoutRegions, durations, aspect, look, trackMeta, gradeBypass, project]);

  // Frame box ratio for the program monitor, matching the delivery aspect.
  const frameRatio =
    aspect === "portrait" ? "9 / 16" : aspect === "square" ? "1 / 1" : "16 / 9";

  const [urls, setUrls] = useState<Record<string, string>>({});

  // Shared transport: the single source of truth for program time + play state,
  // read by both this monitor and the timeline so they stay in lockstep.
  const playing = useTransport((s) => s.playing);
  const progMs = useTransport((s) => s.progMs);
  const seekSeq = useTransport((s) => s.seekSeq);
  const seekTargetMs = useTransport((s) => s.seekTargetMs);
  const setPlaying = useTransport((s) => s.setPlaying);
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

  const player = useProgramPlayer(resolved, urls);

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

  // The rAF loop: read the master clock, drive the slot pool (picture + audio
  // gains), and publish a frame-snapped time only when the frame index changes
  // (≈fps Hz). All media work happens inside player.sync.
  const loop = useCallback(() => {
    const t = player.nowMs();
    if (duration > 0 && t >= duration) {
      player.pause();
      player.seek(0, false);
      player.sync(0, false);
      progRef.current = 0;
      lastFrameRef.current = 0;
      publish(0);
      setPlaying(false); // the [playing] effect tears down the loop
      return;
    }
    progRef.current = t;
    player.sync(t, true);
    const frame = Math.round(t / FRAME_MS);
    if (frame !== lastFrameRef.current) {
      lastFrameRef.current = frame;
      publish(frame * FRAME_MS);
    }
    rafRef.current = requestAnimationFrame(loop);
  }, [duration, player, publish, setPlaying]);

  // Start/stop purely from the shared `playing` flag, so the play button here
  // and spacebar/scrub on the timeline drive the same machine.
  useEffect(() => {
    if (!playing || !hasTimeline) {
      player.pause();
      stopRaf();
      return;
    }
    player.play(progRef.current);
    stopRaf();
    rafRef.current = requestAnimationFrame(loop);
    return stopRaf;
  }, [playing, hasTimeline, loop, player, stopRaf]);

  // External seek (timeline click/scrub, step buttons, monitor scrubber): move
  // the master clock + reposition every slot to the new frame.
  useEffect(() => {
    progRef.current = seekTargetMs;
    lastFrameRef.current = Math.round(seekTargetMs / FRAME_MS);
    const playingNow = useTransport.getState().playing;
    player.seek(seekTargetMs, playingNow);
    player.sync(seekTargetMs, playingNow);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [seekSeq]);

  // Reset to the start whenever the plan changes shape (segment count / ids).
  const shapeKey = useMemo(
    () => (resolved ? resolved.video_layers.map((v) => v.layer_id).join(",") : ""),
    [resolved]
  );
  useEffect(() => {
    resetTransport();
    player.stop();
    progRef.current = 0;
    lastFrameRef.current = 0;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shapeKey]);

  // Edits re-resolve the timeline: re-park every slot from the current playhead
  // so the change is reflected immediately (audio re-arms on the next frame).
  useEffect(() => {
    player.prepare();
    const playingNow = useTransport.getState().playing;
    player.seek(progRef.current, playingNow);
    player.sync(progRef.current, playingNow);
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

  return (
    <div className="px-2 pt-2">
      <div className="flex w-full justify-center">
      <div
        ref={frameRef}
        onMouseEnter={() => onHoverChange?.(true)}
        onMouseLeave={() => onHoverChange?.(false)}
        className="relative overflow-hidden rounded-lg"
        style={{
          background: "#000",
          aspectRatio: frameRatio,
          width: aspect === "landscape" ? "100%" : "auto",
          height: aspect === "landscape" ? undefined : "min(75vh, 560px)",
          maxWidth: "100%",
        }}
      >
        {/* The program player appends its pooled <video> elements here. */}
        <div ref={player.attachContainer} className="absolute inset-0 h-full w-full" />
        {!hasTimeline && (
          <div
            className="absolute inset-0 flex h-full w-full items-center justify-center text-xs"
            style={{ color: "#666" }}
          >
            <Film size={28} />
          </div>
        )}

        {hasTimeline && ((activeVideo && activeVideo.z > 0) || activeBeds > 0) && (
          <div className="absolute left-2 top-2 flex gap-1">
            {activeVideo && activeVideo.z > 0 && (
              <span className="flex items-center gap-1 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white">
                <Layers size={10} /> V2
              </span>
            )}
            {activeBeds > 0 && (
              <span className="flex items-center gap-1 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white">
                <Music size={10} /> {activeBeds} audio
              </span>
            )}
          </div>
        )}

        {/* Export -- relocated on top of the monitor (editor_ui.plan.md SS2.3) */}
        {threadId && (
          <div className="absolute right-2 top-2">
            <RenderBar threadId={threadId} version={version} token={token} disabled={!hasTimeline} />
          </div>
        )}
      </div>
      </div>
    </div>
  );
});

CompositePreview.displayName = "CompositePreview";
