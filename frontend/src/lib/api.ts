const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function request<T>(
  path: string,
  options: RequestInit & { token?: string } = {}
): Promise<T> {
  const { token, ...fetchOptions } = options;
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...(options.headers as Record<string, string>),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(`${API_URL}${path}`, { ...fetchOptions, headers });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `API error ${res.status}`);
  }
  return res.json();
}

// --- Folders ---

export interface Folder {
  id: string;
  user_id: string;
  name: string;
  parent_id: string | null;
  created_at: string;
  updated_at: string;
}

export function getFolders(parentId: string | null, token: string) {
  const q = parentId ? `?parent_id=${parentId}` : "?root=true";
  return request<Folder[]>(`/api/folders${q}`, { token });
}

export function createFolder(name: string, parentId: string | null, token: string) {
  return request<Folder>("/api/folders", {
    method: "POST",
    body: JSON.stringify({ name, parent_id: parentId }),
    token,
  });
}

export function renameFolder(id: string, name: string, token: string) {
  return request<Folder>(`/api/folders/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ name }),
    token,
  });
}

export function deleteFolder(id: string, token: string) {
  return request<void>(`/api/folders/${id}`, { method: "DELETE", token });
}

// --- Files ---

export interface FileRecord {
  id: string;
  user_id: string;
  folder_id: string | null;
  name: string;
  filename: string;
  mime_type: string;
  file_size: number;
  file_type: "video" | "image" | "audio" | "document" | "other";
  r2_key: string;
  r2_proxy_key: string | null;
  r2_thumbnail_key: string | null;
  duration_seconds: number | null;
  width: number | null;
  height: number | null;
  status: "uploading" | "processing" | "ready" | "failed";
  l1_status?: "pending" | "running" | "ready" | "failed" | "skipped" | null;
  l2_status?: "running" | "ready" | "partial" | "failed" | null;
  created_at: string;
  updated_at: string;
}

export function getFiles(folderId: string | null, token: string) {
  const q = folderId ? `?folder_id=${folderId}` : "?root=true";
  return request<FileRecord[]>(`/api/files${q}`, { token });
}

export function getFile(id: string, token: string) {
  return request<FileRecord>(`/api/files/${id}`, { token });
}

export function renameFile(id: string, name: string, token: string) {
  return request<FileRecord>(`/api/files/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ name }),
    token,
  });
}

export function moveFile(id: string, folderId: string | null, token: string) {
  return request<FileRecord>(`/api/files/${id}/move`, {
    method: "POST",
    body: JSON.stringify({ folder_id: folderId }),
    token,
  });
}

export function deleteFile(id: string, token: string) {
  return request<void>(`/api/files/${id}`, { method: "DELETE", token });
}

export function getFilePlaybackUrl(id: string, token: string) {
  return request<{ url: string }>(`/api/files/${id}/playback`, { token });
}

export function getFileDownloadUrl(id: string, token: string) {
  return request<{ url: string }>(`/api/files/${id}/download`, { token });
}

export interface FileShot {
  shot_id: string;
  shot_index: number | null;
  start_ms: number | null;
  end_ms: number | null;
}

export interface FileShotsResponse {
  file_id: string;
  duration_ms: number | null;
  shots: FileShot[];
}

export function getFileShots(id: string, token: string) {
  return request<FileShotsResponse>(`/api/files/${id}/shots`, { token });
}

// --- Upload ---

export interface PresignResponse {
  file_id: string;
  upload_url: string;
  upload_id?: string;
  part_urls?: string[];
}

export function presignUpload(
  filename: string,
  contentType: string,
  fileSize: number,
  folderId: string | null,
  token: string
) {
  return request<PresignResponse>("/api/upload/presign", {
    method: "POST",
    body: JSON.stringify({
      filename,
      content_type: contentType,
      file_size: fileSize,
      folder_id: folderId,
    }),
    token,
  });
}

export function completeUpload(fileId: string, token: string) {
  return request<FileRecord>(`/api/upload/${fileId}/complete`, {
    method: "POST",
    token,
  });
}

// --- L3 Edit requests ---

export interface EditRequestBody {
  prompt: string;
  folder_id?: string | null;
  candidate_limit?: number;
  fps?: number;
  sequence_name?: string;
}

export interface TimelineClipOut {
  file_id: string;
  file_name: string;
  source_in_ms: number;
  source_out_ms: number;
  timeline_start_ms: number;
  timeline_end_ms: number;
  score: number;
}

export interface CandidateShotOut {
  shot_id: string;
  file_id: string;
  file_name: string;
  shot_index: number;
  start_ms: number;
  end_ms: number;
  score: number;
  keyframe_r2_key: string | null;
}

export interface EditRequestResponse {
  query: Record<string, unknown>;
  candidates: CandidateShotOut[];
  timeline: TimelineClipOut[];
  fcp7_xml: string;
  total_duration_ms: number;
}

export function createEditRequest(body: EditRequestBody, token: string) {
  return request<EditRequestResponse>("/api/edit-request", {
    method: "POST",
    body: JSON.stringify(body),
    token,
  });
}

export async function downloadEditXml(body: EditRequestBody, token: string): Promise<Blob> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API_URL}/api/edit-request/download`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const errBody = await res.json().catch(() => ({}));
    throw new Error(errBody.detail || `XML download failed (${res.status})`);
  }
  return res.blob();
}

// --- L1 debug ---

export interface L1Index {
  file: FileRecord;
  transcript: {
    language: string;
    text: string;
    segments: unknown;
    fillers: unknown;
  } | null;
  audio_features: Record<string, unknown> | null;
  shots: Array<Record<string, unknown>>;
  processing_jobs: Array<Record<string, unknown>>;
}

export function getL1Index(fileId: string, token: string) {
  return request<L1Index>(`/api/files/${fileId}/l1`, { token });
}

// --- Semantic search ---

export interface SearchHit {
  shot_id: string;
  shot_index: number;
  start_ms: number;
  end_ms: number;
  keyframe_r2_key: string | null;
  file_id: string;
  file_name: string;
  duration_seconds: number | null;
  thumb_key: string | null;
  score: number;
}

export interface SearchResponse {
  query: string;
  results: SearchHit[];
}

export function semanticSearch(q: string, token: string, folderId?: string | null) {
  const params = new URLSearchParams({ q });
  if (folderId) params.set("folder_id", folderId);
  return request<SearchResponse>(`/api/search?${params.toString()}`, { token });
}

// --- L2 enrichment trigger ---

export function enqueueFileL2(fileId: string, token: string) {
  return request<{ ok: boolean; l2_status: string }>(
    `/api/files/${fileId}/l2`,
    { method: "POST", token },
  );
}

// --- Edit preview render ---

export function renderEditPreview(body: EditRequestBody, token: string) {
  return request<{ preview_url: string; clip_count: number }>(
    `/api/edit-request/preview`,
    { method: "POST", body: JSON.stringify(body), token },
  );
}

// --- Chat-mode edit (multi-turn) ---

// Compact representation of a clip the assistant returned, kept in chat history.
// We store the *Claude-level* fields (shot_id, etc.) so the backend can
// re-render the prior timeline into context on the next turn.
export interface ChatTimelineClip {
  shot_id: string;
  source_in_ms: number;
  source_out_ms: number;
  role_in_edit?: string | null;
  why?: string | null;
  // The frontend also stores the file-level fields it needs to render previews
  // and download XML. These are NOT sent back to the editor (the LLM only sees
  // shot_id + timestamps).
  file_id?: string;
  file_name?: string;
  timeline_start_ms?: number;
  timeline_end_ms?: number;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content?: string; // user
  reasoning?: string; // assistant
  timeline?: ChatTimelineClip[]; // assistant
}

export interface ChatRequestBody {
  messages: ChatMessage[];
  // Explicit file selection. When set (and non-empty), the editor only draws
  // from these files. Takes precedence over folder_id.
  file_ids?: string[] | null;
  folder_id?: string | null;
  sequence_name?: string;
  fps?: number;
  catalog_size?: number;
  duration_target_s?: number | null;
}

// Chat response: backend returns both file-level fields (for the UI) and
// editor-level fields (shot_id / role_in_edit / why) so the next turn's
// history can re-feed them to Claude.
export interface ChatTimelineClipOut {
  file_id: string;
  file_name: string;
  source_in_ms: number;
  source_out_ms: number;
  timeline_start_ms: number;
  timeline_end_ms: number;
  score: number;
  shot_id?: string | null;
  role_in_edit?: string | null;
  why?: string | null;
}

export interface ChatResponse {
  timeline: ChatTimelineClipOut[];
  fcp7_xml: string;
  total_duration_ms: number;
  reasoning: string;
  warnings: string[];
  catalog_size: number;
  // Phase 1: every chat turn persists a new EDL version and auto-enqueues a
  // render. The frontend polls /api/renders/:id to swap in the rendered MP4.
  project_id?: string | null;
  edl_version_id?: string | null;
  render_id?: string | null;
}

export function postEditChat(body: ChatRequestBody, token: string) {
  return request<ChatResponse>("/api/edit-request/chat", {
    method: "POST",
    body: JSON.stringify(body),
    token,
  });
}

// --- Phase 1.5: async chat turns (SSE progress + cancel) ---

export interface StartTurnResponse {
  turn_id: string;
  status: string;
}

export interface TurnStatusResponse {
  id: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  phase?: string | null;
  progress_pct: number;
  error?: string | null;
  project_id?: string | null;
  edl_version_id?: string | null;
  render_id?: string | null;
  result?: ChatResponse | null;
}

// One decoded SSE event from the turn stream.
export type ChatTurnEvent =
  | { type: "phase"; phase: string | null; pct: number | null; label: string | null }
  | { type: "warning"; message: string }
  | { type: "done"; result: ChatResponse }
  | { type: "error"; message: string }
  | { type: "cancelled"; message: string };

export function startEditChatTurn(body: ChatRequestBody, token: string) {
  return request<StartTurnResponse>("/api/edit-request/chat/async", {
    method: "POST",
    body: JSON.stringify(body),
    token,
  });
}

export function getChatTurn(turnId: string, token: string) {
  return request<TurnStatusResponse>(`/api/edit-request/chat/turn/${turnId}`, { token });
}

export function cancelChatTurn(turnId: string, token: string) {
  return request<TurnStatusResponse>(`/api/edit-request/chat/turn/${turnId}/cancel`, {
    method: "POST",
    token,
  });
}

/**
 * Open the SSE stream for a turn and invoke onEvent for each decoded event.
 * Uses fetch + ReadableStream (not native EventSource) so we can send the
 * Authorization header. Resolves when the stream closes (terminal event or
 * connection end). Pass an AbortSignal to stop reading (e.g. on unmount).
 */
export async function streamChatTurn(
  turnId: string,
  token: string,
  onEvent: (evt: ChatTurnEvent) => void,
  abortSignal?: AbortSignal,
): Promise<void> {
  const res = await fetch(
    `${API_URL}/api/edit-request/chat/turn/${turnId}/stream`,
    {
      headers: { Authorization: `Bearer ${token}` },
      signal: abortSignal,
    },
  );
  if (!res.ok || !res.body) {
    throw new Error(`Stream failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  // SSE frames are separated by a blank line. Each frame may have an
  // `event:` line and a `data:` line.
  const flushFrame = (frame: string) => {
    let eventType = "message";
    const dataLines: string[] = [];
    for (const raw of frame.split("\n")) {
      const line = raw.replace(/\r$/, "");
      if (line.startsWith(":")) continue; // keepalive comment
      if (line.startsWith("event:")) eventType = line.slice(6).trim();
      else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
    }
    if (dataLines.length === 0) return;
    let data: Record<string, unknown> = {};
    try {
      data = JSON.parse(dataLines.join("\n"));
    } catch {
      return;
    }
    switch (eventType) {
      case "phase":
        onEvent({
          type: "phase",
          phase: (data.phase as string) ?? null,
          pct: (data.pct as number) ?? null,
          label: (data.label as string) ?? null,
        });
        break;
      case "warning":
        onEvent({ type: "warning", message: (data.message as string) || "" });
        break;
      case "done":
        onEvent({ type: "done", result: data.result as ChatResponse });
        break;
      case "error":
        onEvent({ type: "error", message: (data.message as string) || "Turn failed" });
        break;
      case "cancelled":
        onEvent({ type: "cancelled", message: (data.message as string) || "Cancelled" });
        break;
      default:
        break;
    }
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        flushFrame(frame);
      }
    }
  } finally {
    try {
      reader.releaseLock();
    } catch {
      // ignore
    }
  }
}

// Render a timeline that's already been built (no LLM round-trip).
export interface TimelineClipForRender {
  file_id: string;
  file_name?: string;
  source_in_ms: number;
  source_out_ms: number;
  timeline_start_ms?: number;
  timeline_end_ms?: number;
  score?: number;
}

export function renderPreviewFromTimeline(
  timeline: TimelineClipForRender[],
  token: string,
) {
  return request<{ preview_url: string; clip_count: number }>(
    "/api/edit-request/preview-from-timeline",
    {
      method: "POST",
      body: JSON.stringify({ timeline }),
      token,
    },
  );
}

// --- Phase 1: EDL projects + renders ---

export interface RenderRow {
  id: string;
  edl_version_id: string;
  preset: string;
  status: "queued" | "running" | "done" | "failed" | "cancelled";
  progress_pct: number;
  output_r2_key?: string | null;
  output_url?: string | null;
  duration_ms?: number | null;
  error?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export function getRender(renderId: string, token: string) {
  return request<RenderRow>(`/api/renders/${renderId}`, { token });
}

export function createRender(
  edlVersionId: string,
  preset: "preview" | "export_landscape",
  token: string,
) {
  return request<RenderRow>("/api/renders", {
    method: "POST",
    body: JSON.stringify({ edl_version_id: edlVersionId, preset }),
    token,
  });
}

/**
 * Poll /api/renders/:id every `intervalMs` until the render reaches a
 * terminal state. Resolves with the final RenderRow.
 */
export async function pollRenderUntilDone(
  renderId: string,
  token: string,
  opts: {
    intervalMs?: number;
    maxAttempts?: number;
    onUpdate?: (row: RenderRow) => void;
    abortSignal?: AbortSignal;
  } = {},
): Promise<RenderRow> {
  const intervalMs = opts.intervalMs ?? 1500;
  const maxAttempts = opts.maxAttempts ?? 240; // 6 min @ 1.5s
  for (let i = 0; i < maxAttempts; i++) {
    if (opts.abortSignal?.aborted) {
      throw new Error("Render polling aborted");
    }
    const row = await getRender(renderId, token);
    opts.onUpdate?.(row);
    if (row.status === "done" || row.status === "failed" || row.status === "cancelled") {
      return row;
    }
    await new Promise((r) => setTimeout(r, intervalMs));
  }
  throw new Error("Render polling timed out");
}

// --- Phase 2: EDL manual editing surface ---

export interface EnrichedClip {
  id: string;
  shot_id: string;
  file_id?: string | null;
  file_name?: string | null;
  source_in_ms: number;
  source_out_ms: number;
  timeline_in_ms: number;
  timeline_out_ms: number;
  duration_ms: number;
  shot_start_ms?: number | null;
  shot_end_ms?: number | null;
  file_duration_ms?: number | null;
  thumbnail_url?: string | null;
  transcript_text?: string | null;
  source_url?: string | null;
}

export interface EdlVersionMeta {
  id: string;
  parent_id?: string | null;
  author_kind: "user" | "claude" | "system";
  commit_msg?: string | null;
  created_at?: string | null;
  clip_count: number;
}

export interface EnrichedEdl {
  project_id: string;
  version: EdlVersionMeta;
  fps: number;
  resolution: number[];
  clips: EnrichedClip[];
  total_duration_ms: number;
}

export interface CommitClip {
  id?: string;
  shot_id: string;
  source_in_ms: number;
  source_out_ms: number;
}

export interface CommitEdlResponse {
  project_id: string;
  edl_version_id: string;
  render_id?: string | null;
  clip_count: number;
  total_duration_ms: number;
}

export interface SearchShot {
  shot_id: string;
  file_id: string;
  file_name: string;
  start_ms: number;
  end_ms: number;
  duration_ms: number;
  score: number;
  thumbnail_url?: string | null;
  transcript_text?: string | null;
}

export interface ProjectMeta {
  id: string;
  name: string;
  source_file_ids: string[];
}

export function ensureProject(sourceFileIds: string[], token: string, name = "Untitled") {
  return request<ProjectMeta>(`/api/edl/projects/ensure`, {
    method: "POST",
    body: JSON.stringify({ source_file_ids: sourceFileIds, name }),
    token,
  });
}

export interface ProjectSummary {
  id: string;
  name: string;
  source_file_ids: string[];
  updated_at?: string | null;
  clip_count: number;
  duration_ms: number;
  author_kind: "user" | "claude" | "system";
  version_count: number;
  thumbnail_url?: string | null;
}

export function listProjects(token: string) {
  return request<ProjectSummary[]>(`/api/edl/projects`, { token });
}

export function renameProject(projectId: string, name: string, token: string) {
  return request<ProjectMeta>(`/api/edl/projects/${projectId}`, {
    method: "PATCH",
    body: JSON.stringify({ name }),
    token,
  });
}

export function deleteProject(projectId: string, token: string) {
  return request<{ ok: boolean }>(`/api/edl/projects/${projectId}`, {
    method: "DELETE",
    token,
  });
}

export function getLatestEdl(projectId: string, token: string) {
  return request<EnrichedEdl | null>(`/api/edl/projects/${projectId}/latest`, { token });
}

export function getEdlVersion(versionId: string, token: string) {
  return request<EnrichedEdl>(`/api/edl/versions/${versionId}`, { token });
}

export function listEdlVersions(projectId: string, token: string) {
  return request<EdlVersionMeta[]>(`/api/edl/projects/${projectId}/versions`, { token });
}

export function commitEdl(
  projectId: string,
  clips: CommitClip[],
  token: string,
  opts: {
    commitMsg?: string;
    parentId?: string | null;
    fps?: number;
    authorKind?: "user" | "claude" | "system";
  } = {},
) {
  return request<CommitEdlResponse>(`/api/edl/projects/${projectId}/commit`, {
    method: "POST",
    body: JSON.stringify({
      clips,
      commit_msg: opts.commitMsg,
      parent_id: opts.parentId ?? null,
      fps: opts.fps ?? 30,
      author_kind: opts.authorKind ?? "user",
    }),
    token,
  });
}

// --- Phase 3: cut-only EDL agent ---

export interface AgentProposedClip {
  id: string;
  shot_id: string;
  source_in_ms: number;
  source_out_ms: number;
}

export interface AgentDiff {
  added: AgentProposedClip[];
  removed: AgentProposedClip[];
  trimmed: {
    clip_id: string;
    from: { source_in_ms: number; source_out_ms: number };
    to: { source_in_ms: number; source_out_ms: number };
  }[];
  moved: { clip_id: string; from_index: number | null; to_index: number | null }[];
  changed: boolean;
}

export interface AgentResult {
  project_id: string;
  base_version_id: string | null;
  instruction: string;
  summary: string;
  reasoning: string;
  proposed_clips: AgentProposedClip[];
  proposed_enriched: EnrichedClip[];
  diff: AgentDiff;
  tool_log: { tool: string; input: Record<string, unknown>; result_keys: string[] }[];
}

export function startEdlAgent(
  projectId: string,
  instruction: string,
  token: string,
  baseVersionId?: string | null,
) {
  return request<{ turn_id: string; status: string }>(
    `/api/edl/projects/${projectId}/agent`,
    {
      method: "POST",
      body: JSON.stringify({ instruction, base_version_id: baseVersionId ?? null }),
      token,
    },
  );
}

export function searchShotsForProject(projectId: string, q: string, token: string, k = 24) {
  return request<SearchShot[]>(
    `/api/edl/projects/${projectId}/search-shots?q=${encodeURIComponent(q)}&k=${k}`,
    { token },
  );
}

// --- Breadcrumb ---

export interface BreadcrumbItem {
  id: string | null;
  name: string;
}

export function getBreadcrumb(folderId: string, token: string) {
  return request<BreadcrumbItem[]>(`/api/folders/${folderId}/breadcrumb`, { token });
}

// --- Audit logs ---

export interface EditLogListItem {
  id: string;
  prompt: string;
  status: "started" | "ok" | "failed" | "unknown";
  duration_target_s: number | null;
  actual_duration_s: number | null;
  logged_at: string;
}

export interface L1LogListItem {
  file_id: string;
  size_bytes: number;
  modified_at: string;
}

export function listEditLogs(token: string, limit = 50) {
  return request<{ items: EditLogListItem[] }>(`/api/logs/edits?limit=${limit}`, { token });
}

export function getEditLog(id: string, token: string) {
  return request<Record<string, unknown>>(`/api/logs/edits/${id}`, { token });
}

export function listL1Logs(token: string) {
  return request<{ items: L1LogListItem[] }>(`/api/logs/l1`, { token });
}

export function getL1Log(fileId: string, token: string) {
  return request<Record<string, unknown>>(`/api/logs/l1/${fileId}`, { token });
}
