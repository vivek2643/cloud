"use client";

import { DriveContent } from "@/components/drive-content";
import { CutsView } from "@/components/cuts-view";
import { ColorGradeView } from "@/components/color-grade-view";
import { CaptionsView } from "@/components/captions-view";
import { ExportView } from "@/components/export-view";
import { useDriveStore } from "@/stores/drive-store";

/**
 * Renders the active project stage selected in the left sidebar. "Media" lists
 * every video in the project; "Cuts" surfaces the deterministic V4 ingest
 * pipeline (cuts_v4_only.plan.md) -- the sole Cuts surface now that the older
 * views have been retired. "Colour grading" is color_grading.plan.md's Grade
 * panel. "Captions" is captions.plan.md's two-tier gallery. "Export" is
 * export_options.plan.md's deliverable picker (finished video / rough cut /
 * subtitles). Remaining stages are placeholders for now.
 */
export function ProjectLenses() {
  const projectStage = useDriveStore((s) => s.projectStage);

  if (projectStage === "media") return <DriveContent />;
  if (projectStage === "cuts") return <CutsView />;
  if (projectStage === "color") return <ColorGradeView />;
  if (projectStage === "captions") return <CaptionsView />;
  if (projectStage === "export") return <ExportView />;
  return <ComingSoon label="Coming soon" />;
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
