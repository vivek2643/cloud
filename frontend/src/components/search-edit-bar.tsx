"use client";

import { Search, Sparkles, X } from "lucide-react";
import { useDriveStore } from "@/stores/drive-store";

/**
 * Search input + "AI Edit" launcher, shown above the drive/folder content.
 * AI Edit opens the L3 editor panel scoped to the current selection (or, if
 * nothing is selected, every ready video in the current view).
 */
export function SearchEditBar() {
  const { searchQuery, setSearchQuery, files, selectedIds, openAiPanel } =
    useDriveStore();

  const readyVideos = files.filter(
    (f) => f.file_type === "video" && f.status === "ready"
  );
  const selectedVideoIds = readyVideos
    .filter((f) => selectedIds.has(f.id))
    .map((f) => f.id);
  const scopeIds =
    selectedVideoIds.length > 0 ? selectedVideoIds : readyVideos.map((f) => f.id);
  const canEdit = scopeIds.length > 0;

  return (
    <div className="mb-5 flex items-center gap-3">
      <div className="relative flex-1">
        <Search
          size={16}
          className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2"
          style={{ color: "var(--muted)" }}
        />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search your footage…"
          className="w-full rounded-lg border bg-transparent py-2 pl-9 pr-9 text-sm outline-none transition-colors focus:border-[var(--accent)]"
          style={{ borderColor: "var(--border)" }}
        />
        {searchQuery && (
          <button
            onClick={() => setSearchQuery("")}
            className="absolute right-2.5 top-1/2 -translate-y-1/2 rounded p-0.5 transition-colors hover:opacity-70"
            style={{ color: "var(--muted)" }}
            title="Clear search"
          >
            <X size={15} />
          </button>
        )}
      </div>

      <button
        onClick={() => canEdit && openAiPanel(scopeIds)}
        disabled={!canEdit}
        className="flex shrink-0 items-center gap-1.5 rounded-lg px-3.5 py-2 text-sm font-medium text-white transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
        style={{ background: "var(--accent)" }}
        title={
          canEdit
            ? selectedVideoIds.length > 0
              ? `AI Edit ${selectedVideoIds.length} selected clip(s)`
              : `AI Edit all ${scopeIds.length} clip(s) here`
            : "No ready videos to edit"
        }
      >
        <Sparkles size={16} />
        AI Edit
        {selectedVideoIds.length > 0 && (
          <span className="rounded-full bg-white/25 px-1.5 text-xs">
            {selectedVideoIds.length}
          </span>
        )}
      </button>
    </div>
  );
}
