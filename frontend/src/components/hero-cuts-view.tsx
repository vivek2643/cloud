"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useAuthStore } from "@/stores/auth-store";
import { useDriveStore } from "@/stores/drive-store";
import { FileIcon } from "./file-icon";
import {
  getHeroCutsFeed,
  getFilePlaybackUrl,
  type HeroCut,
  type HeroModality,
  type FileRecord,
} from "@/lib/api";
import { Star, Play, Volume2, VolumeX, Layers, Scissors, ChevronDown, Check } from "lucide-react";
import { EditButton } from "./search-edit-bar";

const MODALITY_STYLE: Record<HeroModality, { color: string; label: string }> = {
  speech: { color: "#6366f1", label: "speech" },
  action: { color: "#f59e0b", label: "action" },
  visual: { color: "#06b6d4", label: "visual" },
  moment: { color: "#10b981", label: "dialogue + action" },
  reaction: { color: "#ec4899", label: "reaction" },
  broll: { color: "#06b6d4", label: "b-roll" },
  insert: { color: "#a78bfa", label: "insert" },
};

type FilterKey = HeroModality | "all";

// Filter chips over the ONE feed -- this is how every edit style (soundbites,
// action beats, cutaways) is served without a separate pipeline.
const FILTERS: { key: FilterKey; label: string }[] = [
  { key: "all", label: "All" },
  { key: "moment", label: "Moments" },
  { key: "speech", label: "Speech" },
  { key: "reaction", label: "Reactions" },
  { key: "broll", label: "B-roll" },
];

function fmtDur(ms: number): string {
  const s = ms / 1000;
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  return `${m}:${String(Math.round(s - m * 60)).padStart(2, "0")}`;
}

const ENERGY_LABELS = ["Broad", "Calm", "Balanced", "Tight", "Sharp"];

function energyLabel(e: number): string {
  return ENERGY_LABELS[Math.min(ENERGY_LABELS.length - 1, Math.round(e * 4))];
}

export function HeroCutsView() {
  const token = useAuthStore((s) => s.session?.access_token);
  const files = useDriveStore((s) => s.files);
  const [energy, setEnergy] = useState(0.5);
  const [filter, setFilter] = useState<FilterKey>("all");
  // Display framing of the tiles (pure preview reframing, no backend change):
  // orientation picks the tile aspect, fit decides reframe-to-fill vs full frame.
  const [orientation, setOrientation] = useState<"landscape" | "portrait">("landscape");
  const [fit, setFit] = useState<"adjusted" | "original">("adjusted");
  const [heroes, setHeroes] = useState<HeroCut[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeHeroId, setActiveHeroId] = useState<string | null>(null);
  const urlCache = useRef<Record<string, Promise<string | null>>>({});

  const candidates = useMemo(
    () =>
      files.filter(
        (f) => (f.file_type === "video" || f.file_type === "audio") && f.l1_status === "ready"
      ),
    [files]
  );
  const filesById = useMemo(() => {
    const m: Record<string, FileRecord> = {};
    for (const f of files) m[f.id] = f;
    return m;
  }, [files]);

  // One combined feed across every ready clip, so repeated takes of the same
  // content stack across files. Refetch on clip-set / (debounced) energy change.
  const candidateIds = useMemo(() => candidates.map((f) => f.id), [candidates]);
  const candidateKey = candidateIds.join(",");

  useEffect(() => {
    if (!token || candidateIds.length === 0) {
      setHeroes([]);
      return;
    }
    let cancelled = false;
    setLoading(true);
    const t = setTimeout(() => {
      getHeroCutsFeed(candidateIds, energy, token)
        .then((r) => {
          if (cancelled) return;
          setHeroes(r.heroes ?? []);
          setLoading(false);
        })
        .catch(() => {
          if (cancelled) return;
          setHeroes([]);
          setLoading(false);
        });
    }, 250);
    return () => {
      cancelled = true;
      clearTimeout(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, candidateKey, energy]);

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

  const present = useMemo(() => heroes.filter((h) => filesById[h.file_id]), [heroes, filesById]);
  const counts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const h of present) c[h.modality] = (c[h.modality] ?? 0) + 1;
    return c;
  }, [present]);
  const visible =
    filter === "all" ? present : present.filter((h) => h.modality === filter);
  const totalClips = visible.length;

  return (
    <div>
      {/* Takes / framing / format dropdowns */}
      <div className="mb-6 flex flex-wrap items-center gap-2.5">
        <PillDropdown options={["Best Takes", "All takes"]} />
        <PillDropdown
          options={["Landscape", "Portrait"]}
          value={orientation === "landscape" ? "Landscape" : "Portrait"}
          onChange={(v) => setOrientation(v === "Portrait" ? "portrait" : "landscape")}
        />
        <PillDropdown
          options={["Frame Adjusted", "Original"]}
          value={fit === "adjusted" ? "Frame Adjusted" : "Original"}
          onChange={(v) => setFit(v === "Original" ? "original" : "adjusted")}
        />
      </div>

      {/* Energy bar — narrower, centered, thin track with a draggable scroller. */}
      <div className="mb-7">
        <EnergyBar value={energy} onChange={setEnergy} />
      </div>

      {/* Filters grouped close on the left, highlighted Edit pinned right. */}
      <div className="mb-6 flex items-center justify-between gap-6">
        <div className="flex items-center gap-2">
          {FILTERS.map((f) => {
            const active = filter === f.key;
            const n = f.key === "all" ? present.length : counts[f.key] ?? 0;
            return (
              <button
                key={f.key}
                onClick={() => setFilter(f.key)}
                className="flex items-center gap-1.5 rounded-full px-4 py-1.5 text-sm font-medium transition-colors"
                style={{
                  background: active ? "var(--accent)" : "transparent",
                  color: active ? "var(--background)" : "var(--foreground)",
                }}
              >
                {f.label}
                <span className="text-xs" style={{ opacity: 0.55 }}>
                  {n}
                </span>
              </button>
            );
          })}
        </div>
        <EditButton />
      </div>

      {loading && (
        <p className="py-12 text-center text-sm" style={{ color: "var(--muted)" }}>
          Assembling hero cuts…
        </p>
      )}

      {!loading && totalClips === 0 && (
        <div className="flex flex-col items-center justify-center py-24 text-center">
          <Star size={36} style={{ color: "var(--accent)" }} />
          <p className="mt-4 text-lg font-semibold">No hero cuts yet</p>
          <p className="mt-1 max-w-sm text-sm" style={{ color: "var(--muted)" }}>
            Upload footage. Once analyzed, the most usable moments will surface
            here as ready-to-drop clips, ranked best-first.
          </p>
        </div>
      )}

      {!loading && totalClips > 0 && (
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 2xl:grid-cols-4">
          {visible.map((h) => (
            <HeroClipCard
              key={h.hero_id}
              file={filesById[h.file_id]!}
              hero={h}
              getUrl={getUrl}
              orientation={orientation}
              fit={fit}
              isActive={activeHeroId === h.hero_id}
              onActivate={() => setActiveHeroId(h.hero_id)}
              onDeactivate={() =>
                setActiveHeroId((id) => (id === h.hero_id ? null : id))
              }
            />
          ))}
        </div>
      )}
    </div>
  );
}

function PillDropdown({
  options,
  value,
  onChange,
}: {
  options: string[];
  value?: string;
  onChange?: (v: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [internal, setInternal] = useState(options[0]);
  // Controlled when `value`/`onChange` are supplied; otherwise self-managed.
  const selected = value ?? internal;
  const select = (opt: string) => {
    setInternal(opt);
    onChange?.(opt);
  };

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-2 rounded-lg border px-4 py-2 text-sm font-medium transition-colors hover:bg-[var(--sidebar)]"
        style={{ borderColor: "rgba(255,255,255,0.4)", color: "var(--foreground)" }}
      >
        {selected}
        <ChevronDown size={15} style={{ color: "var(--muted)" }} />
      </button>
      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} />
          <div
            className="absolute left-0 z-40 mt-1.5 min-w-[170px] overflow-hidden rounded-xl border shadow-xl"
            style={{ background: "var(--background)", borderColor: "var(--border)" }}
          >
            {options.map((opt) => (
              <button
                key={opt}
                onClick={() => {
                  select(opt);
                  setOpen(false);
                }}
                className="flex w-full items-center justify-between px-3.5 py-2 text-sm transition-colors hover:bg-[var(--sidebar)]"
                style={{ color: selected === opt ? "var(--foreground)" : "var(--muted)" }}
              >
                {opt}
                {selected === opt && <Check size={14} />}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function EnergyBar({
  value,
  onChange,
}: {
  value: number;
  onChange: (v: number) => void;
}) {
  const trackRef = useRef<HTMLDivElement>(null);
  const valueRef = useRef(value);
  valueRef.current = value;

  // Map a click/drag x to one of 5 stops. On a discrete click we guarantee the
  // handle moves at least one stop toward the click, so it never feels stuck.
  const apply = useCallback(
    (clientX: number, isClick: boolean) => {
      const el = trackRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const t = Math.min(1, Math.max(0, (clientX - rect.left) / rect.width));
      let snapped = Math.round(t * 4) / 4;
      const cur = valueRef.current;
      if (isClick && snapped === cur) {
        if (t > cur) snapped = Math.min(1, cur + 0.25);
        else if (t < cur) snapped = Math.max(0, cur - 0.25);
      }
      if (snapped !== cur) onChange(snapped);
    },
    [onChange]
  );

  const handlePointerDown = useCallback(
    (e: React.PointerEvent) => {
      e.preventDefault();
      apply(e.clientX, true);
      const move = (ev: PointerEvent) => apply(ev.clientX, false);
      const up = () => {
        window.removeEventListener("pointermove", move);
        window.removeEventListener("pointerup", up);
      };
      window.addEventListener("pointermove", move);
      window.addEventListener("pointerup", up);
    },
    [apply]
  );

  return (
    <div className="mx-auto flex w-3/4 items-center gap-4">
      <span className="shrink-0 pl-1 text-sm font-medium" style={{ color: "var(--foreground)" }}>
        Energy
      </span>
      <div
        ref={trackRef}
        onPointerDown={handlePointerDown}
        className="relative flex-1 cursor-pointer select-none py-4"
        style={{ touchAction: "none" }}
      >
        <div
          className="h-px w-full rounded-full"
          style={{ background: "rgba(255,255,255,0.16)" }}
        />
        {/* Bright white filled portion to the left of the handle. */}
        <div
          className="absolute left-0 top-1/2 h-px -translate-y-1/2 rounded-full"
          style={{
            width: `${value * 100}%`,
            background: "var(--foreground)",
            transition: "width 0.35s cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        />
        <div
          className="absolute top-1/2 h-3.5 w-[3px] -translate-y-1/2 rounded-full"
          style={{
            left: `calc(${value * 100}% - 1.5px)`,
            background: "var(--foreground)",
            transition: "left 0.35s cubic-bezier(0.22, 1, 0.36, 1)",
          }}
        />
      </div>
      <span
        className="inline-flex min-w-[74px] shrink-0 items-center justify-center rounded-md px-3 py-1 text-xs font-semibold"
        style={{ background: "var(--accent)", color: "var(--background)" }}
      >
        {energyLabel(value)}
      </span>
    </div>
  );
}

function HeroClipCard({
  file,
  hero,
  getUrl,
  orientation,
  fit,
  isActive,
  onActivate,
  onDeactivate,
}: {
  file: FileRecord;
  hero: HeroCut;
  getUrl: (fileId: string) => Promise<string | null>;
  orientation: "landscape" | "portrait";
  fit: "adjusted" | "original";
  isActive: boolean;
  onActivate: () => void;
  onDeactivate: () => void;
}) {
  const [playUrl, setPlayUrl] = useState<string | null>(null);
  const [muted, setMuted] = useState(false);
  const [playing, setPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Kept spans in seconds. With a breath-removal edit-list (Sharp band) the
  // preview plays each kept run and jumps the excised gaps; otherwise it's the
  // single [src_in, src_out] span.
  const segs = useMemo<[number, number][]>(() => {
    if (hero.keep_spans && hero.keep_spans.length > 0) {
      return hero.keep_spans.map((k) => [k.in_ms / 1000, k.out_ms / 1000]);
    }
    return [[hero.src_in_ms / 1000, hero.src_out_ms / 1000]];
  }, [hero.keep_spans, hero.src_in_ms, hero.src_out_ms]);
  const inSec = segs[0][0];
  const outSec = segs[segs.length - 1][1];
  const isVideo = file.file_type === "video";
  const modality = MODALITY_STYLE[hero.modality] ?? MODALITY_STYLE.speech;

  // "Frame Adjusted" reframes the clip to fill the chosen tile (center-crop);
  // "Original" letterboxes the full source frame. The proxy is already baked
  // upright at ingest, so no client-side rotation is needed.
  const objectFit = fit === "adjusted" ? "cover" : "contain";

  async function ensureUrl() {
    if (playUrl) return;
    const url = await getUrl(file.id);
    if (url) setPlayUrl(url);
  }

  useEffect(() => {
    if (videoRef.current) videoRef.current.muted = muted;
  }, [muted, playUrl]);

  useEffect(() => {
    const v = videoRef.current;
    if (!v || !playUrl) return;
    if (isActive) {
      try {
        if (v.currentTime < inSec || v.currentTime >= outSec) v.currentTime = inSec;
      } catch {
        /* ignore */
      }
      v.muted = muted;
      v.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
    } else {
      v.pause();
      setPlaying(false);
      try {
        v.currentTime = inSec;
      } catch {
        /* ignore */
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isActive, playUrl, muted]);

  function onLoadedMetadata() {
    const v = videoRef.current;
    if (!v) return;
    try {
      v.currentTime = inSec;
    } catch {
      /* ignore */
    }
  }

  function onTimeUpdate() {
    const v = videoRef.current;
    if (!v) return;
    const t = v.currentTime;
    const seek = (to: number) => {
      try {
        v.currentTime = to;
      } catch {
        /* ignore */
      }
    };
    // Past the end (or rewound before the start): loop to the first kept span.
    if (t >= outSec - 0.02 || t < inSec - 0.3) {
      seek(inSec);
      return;
    }
    // Inside an excised breath: jump straight to the next kept span's start.
    for (let i = 0; i < segs.length - 1; i++) {
      if (t >= segs[i][1] - 0.02 && t < segs[i + 1][0]) {
        seek(segs[i + 1][0]);
        return;
      }
    }
  }

  async function handleEnter() {
    onActivate();
    await ensureUrl();
  }

  function handleLeave() {
    onDeactivate();
  }

  function onDragStart(e: React.DragEvent) {
    const payload = JSON.stringify({
      kind: "hero",
      modality: hero.modality,
      file_id: file.id,
      file_name: file.name,
      in_ms: hero.src_in_ms,
      out_ms: hero.src_out_ms,
      content: hero.label,
      speaker: hero.speaker,
    });
    e.dataTransfer.setData("application/x-hero-cut", payload);
    e.dataTransfer.setData("text/plain", payload);
    e.dataTransfer.effectAllowed = "copy";
  }

  return (
    <div className="relative">
      {/* Stacked-take pile: best in front, alternates peeking out behind. */}
      {hero.take_count > 1 && (
        <>
          <div
            aria-hidden
            className="pointer-events-none absolute inset-0 translate-x-1.5 translate-y-1.5 rounded-xl border"
            style={{ borderColor: "var(--border)", background: "var(--background)", zIndex: 0 }}
          />
          {hero.take_count > 2 && (
            <div
              aria-hidden
              className="pointer-events-none absolute inset-0 translate-x-3 translate-y-3 rounded-xl border"
              style={{ borderColor: "var(--border)", background: "var(--background)", opacity: 0.6, zIndex: 0 }}
            />
          )}
        </>
      )}
      <div
        className="group relative z-[1] flex flex-col overflow-hidden rounded-xl border transition-colors hover:border-[var(--accent)]"
        style={{ borderColor: "var(--border)", background: "var(--background)" }}
      >
      <div
        onMouseEnter={handleEnter}
        onMouseLeave={handleLeave}
        draggable
        onDragStart={onDragStart}
        className={`relative flex cursor-pointer items-center justify-center overflow-hidden ${
          orientation === "portrait" ? "aspect-[9/16]" : "aspect-video"
        }`}
        style={{ background: "#000" }}
        title={hero.label}
      >
        {playUrl && (
          <video
            ref={videoRef}
            src={`${playUrl}#t=${inSec.toFixed(2)}`}
            playsInline
            preload="metadata"
            muted={muted}
            draggable={false}
            onLoadedMetadata={onLoadedMetadata}
            onTimeUpdate={onTimeUpdate}
            className="h-full w-full bg-black"
            style={{ objectFit }}
          />
        )}
        {!playUrl && <FileIcon type={(isVideo ? "video" : "audio") as "video"} size={32} />}

        {/* Modality badge (top-left). */}
        <span
          className="absolute left-2 top-2 z-20 flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-semibold capitalize text-white"
          style={{ background: modality.color }}
        >
          {modality.label}
        </span>

        {/* Take-stack badge (top-left, below the modality badge) when repeats exist. */}
        {hero.take_count > 1 && (
          <span
            className="absolute left-2 top-9 z-20 flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-semibold text-white"
            style={{ background: "rgba(0,0,0,0.6)" }}
            title={`${hero.take_count} takes of this content \u2014 best shown`}
          >
            <Layers size={11} /> {hero.take_count} takes
          </span>
        )}

        {/* Hover preview hint — playback is hover-driven only. */}
        <span
          className={`pointer-events-none absolute left-1/2 top-1/2 z-20 flex h-11 w-11 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full shadow-lg transition-opacity ${
            playing ? "opacity-0" : "opacity-100 group-hover:opacity-0"
          }`}
          style={{ background: "var(--accent)" }}
        >
          <Play size={20} className="ml-0.5" fill="currentColor" style={{ color: "var(--background)" }} />
        </span>

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

        {/* Duration badge (bottom-right). When breaths are excised (Sharp band),
            show the played length and a jump-cut marker; tooltip notes the cuts. */}
        {hero.keep_spans && hero.keep_spans.length > 1 ? (
          <span
            className="absolute bottom-2 right-2 z-10 flex items-center gap-1 rounded bg-black/70 px-1.5 py-0.5 text-[11px] font-medium text-white"
            title={`${hero.keep_spans.length - 1} breath${
              hero.keep_spans.length > 2 ? "s" : ""
            } removed \u2014 ${fmtDur(hero.duration_ms)} of source plays in ${fmtDur(hero.play_ms)}`}
          >
            <Scissors size={10} /> {fmtDur(hero.play_ms)}
          </span>
        ) : (
          <span className="absolute bottom-2 right-2 z-10 rounded bg-black/70 px-1.5 py-0.5 text-[11px] font-medium text-white">
            {fmtDur(hero.duration_ms)}
          </span>
        )}

        {/* Score badge (bottom-left). */}
        <span
          className="absolute bottom-2 left-2 z-10 flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] font-medium text-white"
          style={{ background: "rgba(0,0,0,0.6)" }}
          title="Rank score"
        >
          <Star size={10} fill="white" /> {Math.round(hero.score * 100)}
        </span>
      </div>

        <div className="p-2.5">
          <p className="line-clamp-2 text-sm leading-snug" style={{ minHeight: "2.5em" }}>
            {hero.label || <em style={{ color: "var(--muted)" }}>(no label)</em>}
          </p>
          <p className="mt-1 truncate text-[11px]" style={{ color: "var(--muted)" }}>
            {file.name}
          </p>
        </div>
      </div>
    </div>
  );
}
