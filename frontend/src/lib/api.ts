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

export function getFolderCovers(folderId: string, token: string, limit = 3) {
  return request<{ urls: string[] }>(
    `/api/folders/${folderId}/covers?limit=${limit}`,
    { token },
  );
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

// --- Dialogues lens ---

export interface DialogueSegment {
  seg_id: string;
  level: "sentence" | "topic";
  order: number;
  speaker: string | null;
  text: string;
  src_in_ms: number;
  src_out_ms: number;
  raw_in_ms: number;
  raw_out_ms: number;
  fade_in_ms: number;
  fade_out_ms: number;
  topic_id: number | null;
  child_seg_ids: string[];
  flags: string[];
  confidence: number;
}

export interface DialoguesResponse {
  sentence?: DialogueSegment[];
  topic?: DialogueSegment[];
  ready: boolean;
}

export function getDialogues(fileId: string, token: string) {
  return request<DialoguesResponse>(`/api/files/${fileId}/dialogues`, { token });
}

// --- Hero Cuts lens ---

// The closed editing vocabulary (backend l3/vocab.py). Exactly five affordances.
export type HeroModality =
  | "speech"
  | "action"
  | "reaction"
  | "broll"
  | "insert";

// cuts-v2 capture CHANNEL (backend l3/vocab.py): the honest substrate the tabs
// key on. Heard is detected but never surfaced.
export type HeroChannel = "said" | "done" | "shown" | "heard";

// Orthogonal SUBJECT tag riding on the video channels.
export type HeroSubject = "person" | "place" | "object" | "graphic";

// One typed, directional edge from this cut to another (mapped from the VLM's
// relation graph). `dir` is 'out' when this cut is the source, 'in' when target.
export interface HeroRelation {
  type: string;       // responds_to | illustrates | leads_into | answers | ...
  dir: "out" | "in";
  other: string;      // hero_id of the connected cut
  note?: string | null;
}

export interface HeroTake {
  file_id: string;
  src_in_ms: number;
  src_out_ms: number;
  score: number;
}

export interface HeroCut {
  hero_id: string;
  file_id: string;
  modality: HeroModality;
  label: string;
  src_in_ms: number;
  src_out_ms: number;
  duration_ms: number;
  // On-screen duration after progressive breath removal (Sharp band). Equals
  // duration_ms unless keep_spans is set; then it's the kept (spoken) time.
  play_ms: number;
  // Jump-cut edit-list: spoken runs to KEEP inside [src_in_ms, src_out_ms];
  // the gaps between them are excised breaths. null = play contiguously.
  keep_spans?: { in_ms: number; out_ms: number }[] | null;
  score: number;
  speaker: string | null;
  flags: string[];
  // All editorial uses this cut serves (filter keys / tabs).
  affordances: string[];
  // The intrinsic capture substrate (person/action/place/object/graphic/speech)
  // -- what the camera/mic actually caught, beneath the editorial affordance.
  primitives?: string[];
  // cuts-v2: the capture CHANNEL this cut delivers on (said|done|shown) -- what
  // the tabs filter by -- and the orthogonal SUBJECT tag (person|place|object|
  // graphic). For legacy cuts the channel is derived from the affordance.
  channel?: HeroChannel;
  subject?: HeroSubject | null;
  // For an information-dense graphic / insert cut, the gist of what it CONVEYS
  // (the VLM's read, not OCR) -- e.g. "User selects video files for upload".
  summary?: string | null;
  // Typed edges to other cuts (a reaction responds_to a line, b-roll
  // illustrates a topic). Flat model -- the cut stays its own card; this is how
  // it CONNECTS to others.
  relations?: HeroRelation[] | null;
  // The connected-cluster id this cut shares with the cuts it forms a moment
  // with. null for a standalone cut. The Moments view groups by this.
  moment_id?: string | null;
  // Narrative intent (hook/answer/cta/establishing/climax/listener), when the
  // VLM marked one. null for ordinary middle content.
  role?: string | null;
  // True when this cut belongs to a multi-cut moment cluster (has a moment_id).
  is_moment?: boolean;
  take_count: number;
  alt_takes: HeroTake[];
}

export interface HeroCutsResponse {
  heroes: HeroCut[];
  energy: number;
  ready: boolean;
}

export function getHeroCuts(fileId: string, energy: number, token: string) {
  return request<HeroCutsResponse>(
    `/api/files/${fileId}/hero-cuts?energy=${energy}`,
    { token }
  );
}

// One combined feed across many clips: repeated takes of the same content stack
// across files (best in front), instead of one isolated feed per file.
export function getHeroCutsFeed(fileIds: string[], energy: number, token: string) {
  return request<HeroCutsResponse>(`/api/files/hero-cuts`, {
    method: "POST",
    token,
    body: JSON.stringify({ file_ids: fileIds, energy }),
  });
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

// --- Multipart upload (files larger than R2's 5 GiB single-PUT limit) ---

export interface MultipartCreateResponse {
  file_id: string;
  r2_key: string;
  upload_id: string;
  part_size: number;
  part_urls: string[];
}

export function createMultipartUpload(
  filename: string,
  contentType: string,
  fileSize: number,
  folderId: string | null,
  token: string
) {
  return request<MultipartCreateResponse>("/api/upload/multipart/create", {
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

export function completeMultipartUpload(
  fileId: string,
  uploadId: string,
  token: string
) {
  return request<FileRecord>("/api/upload/multipart/complete", {
    method: "POST",
    body: JSON.stringify({ file_id: fileId, upload_id: uploadId }),
    token,
  });
}

export function abortMultipartUpload(
  fileId: string,
  uploadId: string,
  token: string
) {
  return request<{ ok: boolean }>("/api/upload/multipart/abort", {
    method: "POST",
    body: JSON.stringify({ file_id: fileId, upload_id: uploadId }),
    token,
  });
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
  processing_jobs: Array<Record<string, unknown>>;
}

export function getL1Index(fileId: string, token: string) {
  return request<L1Index>(`/api/files/${fileId}/l1`, { token });
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

export interface L1LogListItem {
  file_id: string;
  name: string;
  l1_status: string;
  duration_seconds: number | null;
  l1_seconds: number | null;
  analyzed_at: string | null;
}

export function listL1Logs(token: string) {
  return request<{ items: L1LogListItem[] }>(`/api/logs/l1`, { token });
}

export function getL1Log(fileId: string, token: string) {
  return request<Record<string, unknown>>(`/api/logs/l1/${fileId}`, { token });
}

// --- L3: AI edit orchestrator ---

export type EditThreadStatus = "drafting" | "awaiting_user" | "ready" | "failed";

export type EditAspect = "landscape" | "portrait" | "square";

export interface EditBrief {
  goal?: string;
  target_duration_s?: number;
  tone?: string;
  platform?: string;
  aspect?: EditAspect;
  constraints?: string[];
  assumptions?: string[];
}

export type LayerFit = "cover" | "contain";
export type LayerAnchor = "center" | "left" | "right" | "top" | "bottom";

/** Normalized point (0..1, origin top-left) to keep centered when reframing. */
export interface LayerFocus {
  cx: number;
  cy: number;
}

export type MotionStyle = "static" | "punch_in" | "push_in" | "follow";
export type MotionFeel = "snappy" | "glide";
export type MotionEase = "linear" | "smooth";

export interface MotionPoint {
  scale: number;
  cx: number;
  cy: number;
}

/** Animated scale+focus over a layer span (from->to, eased); mirrors backend. */
export interface LayerMotion {
  from: MotionPoint;
  to: MotionPoint;
  ease?: MotionEase;
  dur_ms: number;
}

/** Per-layer geometric framing; mirrors backend `layers.solve_transform`. */
export interface LayerTransform {
  rotate?: 0 | 90 | 180 | 270;
  fit?: LayerFit;
  anchor?: LayerAnchor;
  zoom?: number;
  focus?: LayerFocus;
  motion?: LayerMotion;
  dest?: "full";
}

export interface EditFormat {
  aspect?: EditAspect;
  fit?: LayerFit;
  motion_style?: MotionStyle;
  motion_feel?: MotionFeel;
}

export interface EditBeat {
  beat_id: string;
  purpose: string;
  intent: string;
  target_s?: number;
}

export type SpineKind = "dialogue" | "music" | "visual" | "sync" | "other";

export interface EditSpineRegion {
  kind: SpineKind;
  label?: string;
  locked_channels?: ("video" | "audio")[];
  source_file_ids?: string[];
  protected_windows?: { file_id: string; start_ms: number; end_ms: number; reason?: string }[];
  rationale?: string;
}

export interface EditSpine {
  regions?: EditSpineRegion[];
}

export interface EditSegment {
  seg_id: string;
  file_id: string;
  in_ms: number;
  out_ms: number;
  axis?: string;
  beat_id?: string | null;
  content?: string | null;
  rationale?: string | null;
  priority?: number;
  cut_in_cost?: number;
  cut_out_cost?: number;
  warnings?: string[];
  transform?: LayerTransform;
}

export interface EditQuestion {
  q_id: string;
  question: string;
  options?: string[];
  default: string;
  why?: string;
}

export interface EditDiagnostics {
  segment_count?: number;
  total_ms?: number;
  total_s?: number;
  mean_seam_cost?: number;
  max_seam_cost?: number;
  warnings?: string[];
  guardrail?: string;
  [k: string]: unknown;
}

export type EditOperationType =
  | "place_video"
  | "pick_angle"
  | "place_audio"
  | "split_edit"
  | "level";

export interface EditOperation {
  op_id: string;
  type: EditOperationType;
  rationale?: string | null;
  warnings?: string[];
  // place_video / pick_angle / place_audio
  source_file_id?: string;
  src_in_ms?: number;
  src_out_ms?: number;
  from_ms?: number;
  to_ms?: number;
  // place_video
  layout?: string;
  z?: number;
  opacity?: number;
  transform?: LayerTransform;
  // place_audio
  role?: string;
  audio_kind?: string;
  gain_db?: number | null;
  duck_db?: number;
  // split_edit
  seam_seg_id?: string;
  audio_offset_ms?: number;
  kind?: string;
  // level
  mute?: boolean;
}

export interface ResolvedVideoLayer {
  layer_id: string;
  source_file_id: string;
  src_in_ms: number;
  src_out_ms: number;
  prog_start_ms: number;
  prog_end_ms: number;
  z: number;
  layout: string;
  opacity: number;
  kind: string;
  op_id?: string | null;
  transform?: LayerTransform;
}

export interface ResolvedAudioLayer {
  layer_id: string;
  role: string;
  source_file_id: string;
  src_in_ms: number;
  src_out_ms: number;
  prog_start_ms: number;
  prog_end_ms: number;
  gain_db: number;
  duck_db: number;
  kind: string;
  op_id?: string | null;
}

export interface ResolvedTimeline {
  duration_ms: number;
  video_layers: ResolvedVideoLayer[];
  audio_layers: ResolvedAudioLayer[];
  aspect?: EditAspect;
}

export interface EditDocument {
  brief?: EditBrief;
  format?: EditFormat;
  spine?: EditSpine | null;
  operations?: EditOperation[];
  resolved?: ResolvedTimeline | null;
  outline?: EditBeat[];
  timeline?: EditSegment[];
  open_questions?: EditQuestion[];
  summary?: string;
  notes?: string[];
  diagnostics?: EditDiagnostics;
}

export interface EditThread {
  id: string;
  user_id: string;
  title: string | null;
  file_ids: string[];
  brief: string;
  status: EditThreadStatus;
  created_at: string;
  updated_at: string;
  document: EditDocument | null;
  document_version: number | null;
  open_questions: EditQuestion[];
  usage: { input_tokens?: number; output_tokens?: number };
}

export interface EditThreadListItem {
  id: string;
  title: string | null;
  status: EditThreadStatus;
  created_at: string;
  clip_count: number;
  latest_version: number | null;
}

export interface EditVersionListItem {
  version: number;
  created_by?: string;
  created_at: string;
}

export function createEditThread(
  fileIds: string[],
  brief: string,
  token: string
) {
  return request<{ thread_id: string; status: EditThreadStatus; mode: string }>(
    "/api/edit/threads",
    {
      method: "POST",
      body: JSON.stringify({ file_ids: fileIds, brief }),
      token,
    }
  );
}

export interface ThreadMessageResult {
  reply: string;
  // The assistant proposed a cut (its cut list was harvested deterministically).
  // The client shows a Confirm button; nothing is applied until the user says yes.
  proposal: boolean;
  proposal_count: number;
}

export function sendThreadMessage(id: string, text: string, token: string) {
  return request<ThreadMessageResult>(`/api/edit/threads/${id}/messages`, {
    method: "POST",
    body: JSON.stringify({ text }),
    token,
  });
}

export interface ApplyEditResult {
  applied: boolean;
  version: number;
  cuts: number;
}

export function applyThreadEdit(id: string, token: string) {
  return request<ApplyEditResult>(`/api/edit/threads/${id}/apply`, {
    method: "POST",
    token,
  });
}

export function listEditThreads(token: string) {
  return request<{ threads: EditThreadListItem[] }>("/api/edit/threads", { token });
}

export function getEditThread(id: string, token: string) {
  return request<EditThread>(`/api/edit/threads/${id}`, { token });
}

export function listEditVersions(id: string, token: string) {
  return request<{ versions: EditVersionListItem[] }>(
    `/api/edit/threads/${id}/versions`,
    { token }
  );
}

export function getEditVersion(id: string, version: number, token: string) {
  return request<{ version: number; document: EditDocument }>(
    `/api/edit/threads/${id}/versions/${version}`,
    { token }
  );
}

// --- Editable timeline (human edits -> new user version) ---

export interface SaveEditBody {
  base_version: number;
  timeline: EditSegment[];
  operations?: EditOperation[];
  summary?: string;
  notes?: string[];
}

export function saveEditDocument(id: string, body: SaveEditBody, token: string) {
  return request<{ version: number; document: EditDocument }>(
    `/api/edit/threads/${id}/document`,
    { method: "PUT", body: JSON.stringify(body), token }
  );
}

// --- Render / export ---

export type RenderStatus = "queued" | "running" | "done" | "failed" | "cancelled";
export type RenderPreset = "preview" | "export";

export interface RenderJob {
  id: string;
  thread_id: string;
  document_version: number;
  preset: RenderPreset | string;
  status: RenderStatus;
  progress_pct: number;
  resolved_hash?: string | null;
  output_r2_key?: string | null;
  output_url?: string | null;
  duration_ms?: number | null;
  error?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
}

export function createRender(
  threadId: string,
  preset: RenderPreset,
  token: string,
  version?: number
) {
  return request<RenderJob>(`/api/edit/threads/${threadId}/render`, {
    method: "POST",
    body: JSON.stringify({ preset, version }),
    token,
  });
}

export function getRender(renderId: string, token: string) {
  return request<RenderJob>(`/api/renders/${renderId}`, { token });
}

export function listRenders(threadId: string, token: string) {
  return request<{ renders: RenderJob[] }>(
    `/api/edit/threads/${threadId}/renders`,
    { token }
  );
}
