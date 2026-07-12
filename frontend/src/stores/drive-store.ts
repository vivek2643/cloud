import { create } from "zustand";
import type { Folder, FileRecord } from "@/lib/api";

export type ViewMode = "grid" | "list";

// The project workspace stages shown in the left sidebar.
export type ProjectStage = "media" | "cuts-v3" | "color" | "captions";

interface UploadItem {
  id: string;
  file: File;
  progress: number;
  status: "pending" | "uploading" | "complete" | "error";
  error?: string;
  fileId?: string;
}

interface DriveState {
  currentFolderId: string | null;
  folders: Folder[];
  files: FileRecord[];
  loading: boolean;
  viewMode: ViewMode;
  projectStage: ProjectStage;
  selectedIds: Set<string>;
  searchQuery: string;
  uploads: UploadItem[];

  // AI Edit (L3) panel
  aiPanelOpen: boolean;
  aiScopeFileIds: string[];
  openAiPanel: (fileIds: string[]) => void;
  closeAiPanel: () => void;

  // Multicam sync (audio_sync.plan.md SS10)
  syncPanelOpen: boolean;
  syncScopeFileIds: string[];
  openSyncPanel: (fileIds: string[]) => void;
  closeSyncPanel: () => void;

  setCurrentFolder: (id: string | null) => void;
  setFolders: (folders: Folder[]) => void;
  setFiles: (files: FileRecord[]) => void;
  removeFile: (id: string) => void;
  setLoading: (loading: boolean) => void;
  setViewMode: (mode: ViewMode) => void;
  setProjectStage: (stage: ProjectStage) => void;
  setSearchQuery: (q: string) => void;
  toggleSelected: (id: string) => void;
  clearSelection: () => void;
  addUpload: (item: UploadItem) => void;
  updateUpload: (id: string, patch: Partial<UploadItem>) => void;
  removeUpload: (id: string) => void;
}

export const useDriveStore = create<DriveState>((set) => ({
  currentFolderId: null,
  folders: [],
  files: [],
  loading: false,
  viewMode: "grid",
  projectStage: "media",
  selectedIds: new Set(),
  searchQuery: "",
  uploads: [],

  aiPanelOpen: false,
  aiScopeFileIds: [],
  openAiPanel: (fileIds) => set({ aiPanelOpen: true, aiScopeFileIds: fileIds }),
  closeAiPanel: () => set({ aiPanelOpen: false }),

  syncPanelOpen: false,
  syncScopeFileIds: [],
  openSyncPanel: (fileIds) => set({ syncPanelOpen: true, syncScopeFileIds: fileIds }),
  closeSyncPanel: () => set({ syncPanelOpen: false }),

  setCurrentFolder: (id) => set({ currentFolderId: id, selectedIds: new Set() }),
  setFolders: (folders) => set({ folders }),
  setFiles: (files) => set({ files }),
  removeFile: (id) =>
    set((state) => {
      const next = new Set(state.selectedIds);
      next.delete(id);
      return { files: state.files.filter((f) => f.id !== id), selectedIds: next };
    }),
  setLoading: (loading) => set({ loading }),
  setViewMode: (mode) => set({ viewMode: mode }),
  setProjectStage: (stage) => set({ projectStage: stage }),
  setSearchQuery: (q) => set({ searchQuery: q }),
  toggleSelected: (id) =>
    set((state) => {
      const next = new Set(state.selectedIds);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return { selectedIds: next };
    }),
  clearSelection: () => set({ selectedIds: new Set() }),
  addUpload: (item) => set((state) => ({ uploads: [...state.uploads, item] })),
  updateUpload: (id, patch) =>
    set((state) => ({
      uploads: state.uploads.map((u) => (u.id === id ? { ...u, ...patch } : u)),
    })),
  removeUpload: (id) =>
    set((state) => ({ uploads: state.uploads.filter((u) => u.id !== id) })),
}));
