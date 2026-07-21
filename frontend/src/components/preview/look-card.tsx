"use client";

/**
 * frontend_look_gallery.plan.md: the picker's per-look card -- a live
 * thumbnail (`look-thumbnail.ts`'s singleton WebGL renderer) plus label,
 * active ring, and (for the film family) a texture badge.
 */
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import type { GradePresetSummary } from "@/lib/api";
import { getCachedLookThumbnail, requestLookThumbnail } from "./look-thumbnail";

/** Renders the cached bitmap for one engine look into an `<img>`, kicking
 * off the render on mount if it isn't cached yet. A neutral `--border`
 * placeholder shows while pending; on any failure (no WebGL2, cube 404) it
 * settles into a flat `--sidebar` swatch -- never a broken image. */
function LookThumbnail({ look }: { look: GradePresetSummary }) {
  const [url, setUrl] = useState<string | null>(() => getCachedLookThumbnail(look.look_id ?? ""));
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (url || !look.look_id) return;
    let cancelled = false;
    requestLookThumbnail(look.look_id, look.look_params).then((result) => {
      if (cancelled) return;
      if (result) setUrl(result);
      else setFailed(true);
    });
    return () => {
      cancelled = true;
    };
  }, [look.look_id, look.look_params, url]);

  if (url) {
    // eslint-disable-next-line @next/next/no-img-element -- a small, dynamically-rendered data URL, not a static asset
    return <img src={url} alt="" className="h-full w-full object-cover" draggable={false} />;
  }
  return (
    <div
      className="h-full w-full"
      style={{ background: failed ? "var(--sidebar)" : "var(--border)" }}
    />
  );
}

export function LookCard({
  look,
  active,
  onSelect,
}: {
  look: GradePresetSummary;
  active: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      title={look.description}
      className="group flex flex-col gap-1 text-left"
    >
      <div
        className="relative aspect-video overflow-hidden rounded-lg border transition-colors"
        style={{ borderColor: active ? "var(--accent)" : "var(--border)" }}
      >
        <LookThumbnail look={look} />
        {look.family === "film" && (
          <span
            className="absolute right-1 top-1 rounded border px-1 py-0.5 text-[9px] font-medium uppercase leading-none"
            style={{ borderColor: "var(--border)", background: "var(--background)", color: "var(--muted)" }}
            title="Adds film grain + halation (needs film texture on)"
          >
            grain
          </span>
        )}
      </div>
      <span
        className={cn("truncate text-[11px]", active ? "font-semibold" : "font-medium")}
        style={{ color: active ? "var(--foreground)" : "var(--muted)" }}
      >
        {look.label}
      </span>
    </button>
  );
}
