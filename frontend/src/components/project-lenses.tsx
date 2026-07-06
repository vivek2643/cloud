"use client";

import { DriveContent } from "@/components/drive-content";
import { CutsView } from "@/components/cuts-view";
import { CutsV3View } from "@/components/cuts-v3-view";
import { useDriveStore } from "@/stores/drive-store";

const STAGE_LABELS: Record<string, string> = {
  color: "Colour grading",
  captions: "Captions",
};

/**
 * Renders the active project stage selected in the left sidebar. "Media" lists
 * every video in the project; "Cuts" surfaces the deterministic non-overlapping
 * partition (cuts v2); "Cuts v3" surfaces the LLM-grouped ingest pipeline
 * (cuts_v3.plan.md) -- additive, side by side with v2 until cutover (plan
 * step H). Remaining stages are placeholders for now.
 *
 * The old energy-laddered hero-cuts (v1) view still exists in the codebase
 * (`hero-cuts-view.tsx`) but is no longer wired here -- hidden for now.
 */
export function ProjectLenses() {
  const projectStage = useDriveStore((s) => s.projectStage);

  if (projectStage === "media") return <DriveContent />;
  if (projectStage === "cuts") return <CutsView />;
  if (projectStage === "cuts-v3") return <CutsV3View />;
  return <ComingSoon label={STAGE_LABELS[projectStage] ?? "Coming soon"} />;
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
