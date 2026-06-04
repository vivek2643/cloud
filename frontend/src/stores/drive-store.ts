import { create } from "zustand";
import type { Folder, FileRecord } from "@/lib/api";

export type ViewMode = "grid" | "list";

interface UploadItem {
  id: string;
  file: File;
  progress: number;
  status: "pending" | "uploading" | "complete" | "error";
  error?: string;
  fileId?: string;
}

export interface AiTimelineClip {
  shot_id?: string | null;
  file_id?: string | null;
  file_name?: string | null;
  source_in_ms: number;
  source_out_ms: number;
  role_in_edit?: string | null;
  why?: string | null;
}

export interface AiTimelineData {
  clips: AiTimelineClip[];
  totalMs: number;
  renderStatus?: string | null;
  renderUrl?: string | null;
  // Server-side EDL lineage so manual edits can auto-commit new versions.
  projectId?: string | null;
  baseVersionId?: string | null;
}

interface DriveState {
  currentFolderId: string | null;
  folders: Folder[];
  files: FileRecord[];
  loading: boolean;
  viewMode: ViewMode;
  selectedIds: Set<string>;
  searchQuery: string;
  uploads: UploadItem[];
  aiPanelOpen: boolean;
  aiScopeFileIds: string[];
  aiTimeline: AiTimelineData | null;
  aiTimelineVisible: boolean;
  // Who owns the working timeline: a live chat cut, or a saved edit opened
  // from the Edits library. Keeps the chat panel from clearing a loaded
  // project just because its own session is empty.
  aiTimelineSource: "chat" | "project";
  // True when the working timeline has clips (AI cut or manually dropped),
  // so the preview monitor can drop its empty-state placeholder.
  editorHasClips: boolean;
  // The chat panel registers its monitor <video> here so the docked timeline
  // can drive playback/scrubbing of the assembled sequence.
  previewVideoEl: HTMLVideoElement | null;

  setCurrentFolder: (id: string | null) => void;
  setFolders: (folders: Folder[]) => void;
  setFiles: (files: FileRecord[]) => void;
  setLoading: (loading: boolean) => void;
  setViewMode: (mode: ViewMode) => void;
  setSearchQuery: (q: string) => void;
  openAiPanel: (fileIds: string[]) => void;
  closeAiPanel: () => void;
  setAiTimeline: (t: AiTimelineData | null) => void;
  setAiTimelineSource: (src: "chat" | "project") => void;
  // Open a saved edit into the docked editor (left media + bottom timeline,
  // right render/chat). Hydrates the dock and opens the AI panel scoped to the
  // edit's source files.
  openSavedEdit: (args: { fileIds: string[]; timeline: AiTimelineData }) => void;
  showAiTimeline: () => void;
  hideAiTimeline: () => void;
  setEditorHasClips: (has: boolean) => void;
  setPreviewVideoEl: (el: HTMLVideoElement | null) => void;
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
  selectedIds: new Set(),
  searchQuery: "",
  uploads: [],
  aiPanelOpen: false,
  aiScopeFileIds: [],
  aiTimeline: null,
  aiTimelineVisible: true,
  aiTimelineSource: "chat",
  editorHasClips: false,
  previewVideoEl: null,

  setCurrentFolder: (id) => set({ currentFolderId: id, selectedIds: new Set() }),
  setFolders: (folders) => set({ folders }),
  setFiles: (files) => set({ files }),
  setLoading: (loading) => set({ loading }),
  setViewMode: (mode) => set({ viewMode: mode }),
  setSearchQuery: (q) => set({ searchQuery: q }),
  openAiPanel: (fileIds) => set({ aiPanelOpen: true, aiScopeFileIds: fileIds }),
  closeAiPanel: () => set({ aiPanelOpen: false, editorHasClips: false }),
  setAiTimeline: (t) => set({ aiTimeline: t }),
  setAiTimelineSource: (src) => set({ aiTimelineSource: src }),
  openSavedEdit: ({ fileIds, timeline }) =>
    set({
      aiPanelOpen: true,
      aiScopeFileIds: fileIds,
      aiTimeline: timeline,
      aiTimelineSource: "project",
      aiTimelineVisible: true,
    }),
  showAiTimeline: () => set({ aiTimelineVisible: true }),
  hideAiTimeline: () => set({ aiTimelineVisible: false }),
  setEditorHasClips: (has) => set({ editorHasClips: has }),
  setPreviewVideoEl: (el) => set({ previewVideoEl: el }),
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
