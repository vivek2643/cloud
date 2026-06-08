"use client";

import { Search, Sparkles } from "lucide-react";
import { useDriveStore } from "@/stores/drive-store";

/**
 * Shared search input + "AI Edit" button. Used on the root drive view and inside
 * folders so search and the conversational editor are reachable everywhere.
 *
 * The AI Edit button docks the conversational editor on the right. If videos are
 * selected, the session is scoped to them; otherwise it draws from all videos in
 * the current view.
 */
export function SearchEditBar() {
  const {
    searchQuery,
    setSearchQuery,
    selectedIds,
    files,
    openAiPanel,
    aiPanelOpen,
  } = useDriveStore();

  function handleAiEdit() {
    const ids = Array.from(selectedIds);
    // Stash names so the panel's scope chip can render them without a refetch.
    const idToFile = new Map(files.map((f) => [f.id, f]));
    const payload = ids
      .map((id) => idToFile.get(id))
      .filter((f): f is NonNullable<typeof f> => !!f)
      .map((f) => ({ id: f.id, name: f.name, file_type: f.file_type }));
    try {
      window.sessionStorage.setItem(
        "edso_edit_scope_v1",
        JSON.stringify({ files: payload, ts: Date.now() }),
      );
    } catch {
      // non-fatal
    }
    openAiPanel(ids);
  }

  const selectedCount = selectedIds.size;

  return (
    <div className="mb-4 flex items-center gap-3">
      <div className="relative flex-1">
        <Search
          size={18}
          className="absolute left-3.5 top-1/2 -translate-y-1/2"
          style={{ color: "var(--muted)" }}
        />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search your videos…"
          className="w-full rounded-xl border py-2.5 pl-11 pr-4 text-sm outline-none transition-colors focus:border-[var(--accent)] focus:ring-2 focus:ring-[var(--accent-soft)]"
          style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
        />
      </div>
      <button
        onClick={handleAiEdit}
        className="flex shrink-0 items-center gap-2 rounded-xl px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition-transform hover:scale-[1.02] active:scale-95"
        style={{
          background: "var(--accent)",
          boxShadow: aiPanelOpen ? "0 0 0 2px var(--accent-soft)" : undefined,
        }}
        title={selectedCount > 0 ? "Edit selected videos with AI" : "Open the AI editor"}
      >
        <Sparkles size={16} />
        AI Edit{selectedCount > 0 ? ` (${selectedCount})` : ""}
      </button>
    </div>
  );
}
