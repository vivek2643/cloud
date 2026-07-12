"use client";

import { Search, Sparkles, Waves, X } from "lucide-react";
import { useDriveStore } from "@/stores/drive-store";

/**
 * Footage search box, shown centered above the project content.
 */
export function SearchEditBar() {
  const { searchQuery, setSearchQuery } = useDriveStore();

  return (
    <div className="relative mb-6 flex justify-center">
      <div className="relative w-full max-w-xl">
        <Search
          size={18}
          className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2"
          style={{ color: "var(--muted)" }}
        />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search your footage…"
          className="w-full rounded-full border bg-transparent py-3 pl-12 pr-11 text-[15px] outline-none transition-colors focus:border-[var(--accent)]"
          style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
        />
        {searchQuery && (
          <button
            onClick={() => setSearchQuery("")}
            className="absolute right-3.5 top-1/2 -translate-y-1/2 rounded p-0.5 transition-colors hover:opacity-70"
            style={{ color: "var(--muted)" }}
            title="Clear search"
          >
            <X size={16} />
          </button>
        )}
      </div>

      {/* Highlighted Edit launcher, parallel to the search bar on the right. */}
      <div className="absolute right-0 top-1/2 -translate-y-1/2 flex items-center gap-2">
        <SyncButton />
        <EditButton />
      </div>
    </div>
  );
}

/**
 * Outlook-group launcher: only appears once 2+ video/audio files are selected
 * -- "select the alternate angles, hit Outlook". Opens the declare panel
 * scoped to the current selection; unlike Edit, there is no "all clips here"
 * fallback -- an outlook group is meaningless without an explicit
 * multi-selection.
 */
export function SyncButton() {
  const { files, selectedIds, openSyncPanel } = useDriveStore();

  const selectable = files.filter(
    (f) => (f.file_type === "video" || f.file_type === "audio") && f.status === "ready"
  );
  const selectedSyncIds = selectable.filter((f) => selectedIds.has(f.id)).map((f) => f.id);
  const canSync = selectedSyncIds.length >= 2;

  if (selectedSyncIds.length === 0) return null;

  return (
    <button
      onClick={() => canSync && openSyncPanel(selectedSyncIds)}
      disabled={!canSync}
      className="flex items-center gap-1.5 rounded-lg border px-4 py-2.5 text-sm font-semibold transition-colors disabled:cursor-not-allowed disabled:opacity-40"
      style={{ borderColor: "var(--border)", color: "var(--foreground)" }}
      title={
        canSync
          ? `Group ${selectedSyncIds.length} selected file(s) as alternate-angle outlooks`
          : "Select 2+ video/audio files to group as outlooks"
      }
    >
      <Waves size={16} />
      OUTLOOK
      <span className="rounded-full px-1.5 text-xs" style={{ background: "var(--accent-soft)" }}>
        {selectedSyncIds.length}
      </span>
    </button>
  );
}

/**
 * The highlighted (orange) Edit launcher. Opens the L3 editor panel scoped to
 * the current selection, or every ready video in the project if none selected.
 */
export function EditButton() {
  const { files, selectedIds, openAiPanel } = useDriveStore();

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
    <button
      onClick={() => canEdit && openAiPanel(scopeIds)}
      disabled={!canEdit}
      className="flex items-center gap-1.5 rounded-lg px-5 py-2.5 text-sm font-semibold transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
      style={{ background: "#ed5b00", color: "#fff" }}
      title={
        canEdit
          ? selectedVideoIds.length > 0
            ? `Edit ${selectedVideoIds.length} selected clip(s)`
            : `Edit all ${scopeIds.length} clip(s) here`
          : "No ready videos to edit"
      }
    >
      <Sparkles size={16} />
      EDIT
      {selectedVideoIds.length > 0 && (
        <span className="rounded-full bg-white/25 px-1.5 text-xs">
          {selectedVideoIds.length}
        </span>
      )}
    </button>
  );
}
