"use client";

import { useState } from "react";
import { DriveContent } from "@/components/drive-content";
import { HeroCutsView } from "@/components/hero-cuts-view";
import { CutsView } from "@/components/cuts-view";
import { useDriveStore } from "@/stores/drive-store";
import { cn } from "@/lib/utils";

const STAGE_LABELS: Record<string, string> = {
  color: "Colour grading",
  captions: "Captions",
};

/**
 * Renders the active project stage selected in the left sidebar. "Media" lists
 * every video in the project; "Cuts" surfaces the cut feed. Remaining stages
 * are placeholders for now.
 */
export function ProjectLenses() {
  const projectStage = useDriveStore((s) => s.projectStage);

  if (projectStage === "media") return <DriveContent />;
  if (projectStage === "cuts") return <CutsStage />;
  return <ComingSoon label={STAGE_LABELS[projectStage] ?? "Coming soon"} />;
}

/**
 * The Cuts stage runs the new deterministic non-overlapping partition (v2)
 * ALONGSIDE the old energy-laddered hero-cuts (v1). A small toggle flips
 * between them so v2 can be validated on real footage before v1 is retired
 * (see cuts_v2.plan.md, Phase R). Defaults to v2.
 */
function CutsStage() {
  const [version, setVersion] = useState<"v2" | "v1">("v2");
  return (
    <div>
      <div className="mb-5 flex items-center gap-1 text-sm">
        {(["v2", "v1"] as const).map((v) => (
          <button
            key={v}
            onClick={() => setVersion(v)}
            className={cn(
              "rounded-full px-3 py-1 font-medium transition-colors",
              version === v ? "" : "hover:text-[var(--foreground)]"
            )}
            style={{
              background: version === v ? "var(--accent-soft)" : "transparent",
              color: version === v ? "var(--foreground)" : "var(--muted)",
            }}
          >
            {v === "v2" ? "Cuts" : "Hero cuts (v1)"}
          </button>
        ))}
      </div>
      {version === "v2" ? <CutsView /> : <HeroCutsView />}
    </div>
  );
}

function ComingSoon({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      <p className="text-lg font-semibold">{label}</p>
      <p className="mt-1 max-w-sm text-sm" style={{ color: "var(--muted)" }}>
        This stage will be available here soon.
      </p>
      <span
        className="mt-4 rounded-full px-3 py-1 text-xs font-medium"
        style={{ background: "var(--accent-soft)", color: "var(--foreground)" }}
      >
        Coming soon
      </span>
    </div>
  );
}
