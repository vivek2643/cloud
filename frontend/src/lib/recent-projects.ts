// Client-side "recently opened projects" (frontend_project_ux.plan.md Stage
// 4.2). localStorage only, by design -- Folder.updated_at tracks
// modification, not opening, and there is no backend concept of "opened."
// A `last_opened_at` column would be a different, larger feature.

export interface RecentProject {
  id: string;
  name: string;
}

const STORAGE_KEY = "edso.recentProjects";
const MAX_RECENTS = 8;

export function readRecentProjects(): RecentProject[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (e): e is RecentProject => !!e && typeof e.id === "string" && typeof e.name === "string"
    );
  } catch {
    return [];
  }
}

/** De-dupes by id, unshifts to the front, caps at MAX_RECENTS. */
export function recordRecentProject(id: string, name: string): void {
  if (typeof window === "undefined") return;
  try {
    const rest = readRecentProjects().filter((e) => e.id !== id);
    const next = [{ id, name }, ...rest].slice(0, MAX_RECENTS);
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Private browsing / quota exceeded -- recents just don't persist.
  }
}
