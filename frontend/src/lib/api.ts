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
  r2_proxy_a_key?: string | null;
  r2_proxy_b_key?: string | null;
  r2_thumbnail_key: string | null;
  duration_seconds: number | null;
  width: number | null;
  height: number | null;
  status: "uploading" | "processing" | "ready" | "failed";
  l1_status?: "pending" | "running" | "ready" | "failed" | "skipped" | null;
  // Coarse analysis progress (0..1) + short phase label from the server.
  analysis_progress?: number;
  analysis_phase?: string;
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

// --- Cuts (LLM-grouped ingest over the deterministic lattice) ---
// See cuts_v3.plan.md. The sole Cuts surface -- a project-scoped ingest
// pipeline (the deterministic v2 file-scoped endpoint has been retired).

export interface Pace {
  min_ms: number;
  natural_ms: number;
  max_ms: number;
  levels: number[];
  energy_grade: string;
  natural_sound: boolean;
  // Removable dead-air + filler spans across a speech cut ([start_ms, end_ms],
  // absolute source ms), for the dial to shave -- edge silence/fillers, interior
  // disfluencies, pause-excess. Absent on pre-feature runs -> nothing to shave.
  remove_spans?: [number, number][];
}

export type ShotSize =
  | "extreme_close_up" | "close_up" | "medium_close_up" | "medium"
  | "medium_wide" | "wide" | "extreme_wide" | "unsure";

// Technical shot stability (perception_upgrade.plan.md Part C2) -- a single
// still often can't judge this; absent on runs before the field existed.
export type ShotQuality =
  | "stable" | "shaky" | "whip" | "soft_focus" | "racking_focus" | "exposure_shift" | "unsure";

export interface Framing {
  subject_box?: [number, number, number, number] | null;
  crop_16x9?: [number, number, number, number] | null;
  crop_9x16?: [number, number, number, number] | null;
  crop_1x1?: [number, number, number, number] | null;
  rotation_deg?: number;
  // How tight the frame is on the subject (pass 2 image judgment). Feeds the
  // visual half of total_quality. Absent on pre-migration runs.
  shot_size?: ShotSize;
  shot_quality?: ShotQuality;
}

// One person visible in a cut, described well enough to recognise across cuts
// (pass 2 image LLM). Appearance only -- never a name or a score.
export interface PersonLook {
  description: string;
  position?: string | null;
  speaking?: boolean | null;
}

export interface Look {
  graded?: boolean;
  palette?: string[];
  exposure_flags?: string[];
}

export type TakeRole = "take" | "outlook" | "winner";

// Cuts v3 continuity (cuts_v3_continuity.plan.md): this cut's position among
// ALL cuts on its clip (incl. junk -- a gap in cut_no is the signal a junk
// beat sits there) + whether each neighbor is a weldable continuation of the
// same shot (seam.classify_seam, computed once at ingest). Absent/empty ({})
// on a pre-migration run -> no continuity to show, never fabricated.
export interface Continuity {
  clip?: string;
  cut_no?: number;
  of?: number;
  prev_contiguous?: boolean;
  next_contiguous?: boolean;
  seam_reason_prev?: string | null;
  seam_reason_next?: string | null;
}

// v4_cluster_tree_cuts.plan.md section 3: one salient event inside a video
// cut's cluster. onset_ms/settle_ms are the raw, content-derived reach (not
// floor-clamped) -- used by resolveCluster to size each event's own window.
export interface SalienceEvent {
  peak_ms: number;
  score: number;
  kind: "point" | "span" | "none";
  onset_ms: number;
  settle_ms: number;
  span_ms?: [number, number] | null;
}

export interface CutRecord {
  id: string;
  file_id: string;
  src_in_ms: number;
  src_out_ms: number;
  kind: "speech" | "video";
  word_span: [number, number] | null;
  atom_ids: number[] | null;
  label: string;
  summary: string;
  // voice_first_identity.plan.md: all code-derived, never LLM-echoed.
  // voice_ids = the global voice(s) heard; speaker_person = the global
  // person id the voice was confidently bound to (null = unknown owner,
  // e.g. narration); visible_persons = every global person id on screen
  // in this cut. Absent/[] on a pre-migration run.
  voice_ids?: string[];
  speaker_person: string | null;
  visible_persons?: string[];
  on_camera: boolean | null;
  take_group_id: string | null;
  take_role: TakeRole | null;
  // Delivery channel (LLM-owned category): "said" (spoken), "done" (an action
  // performed/demonstrated), "shown" (b-roll/display). Optional for older runs
  // ingested before the field existed -- the view falls back to kind.
  channel?: "said" | "done" | "shown" | null;
  junk: boolean;
  junk_reason: string | null;
  framing: Framing;
  look: Look;
  caption_zones: [number, number, number, number][];
  pace: Pace;
  hero_ts_ms: number | null;
  hero_key: string | null;
  transition_in: string | null;
  transition_out: string | null;
  continuity: Continuity;
  // Deterministic 0..1 quality scores (post.py). speech_quality is delivery-
  // only (null for cuts with no speech); total_quality blends speech + visual
  // and crowns the winner within a same-setting take cluster. Absent/0 on
  // pre-migration runs.
  speech_quality?: number | null;
  total_quality?: number | null;
  // Per-person appearance fingerprints from the pass 2 image LLM.
  characteristics?: PersonLook[];
  // A plain camera-move phrase for the shot ("static", "pan left"/"pan right",
  // "tilt up"/"tilt down", "zoom in"/"zoom out", "follow subject", "shaky").
  // Deterministic from L1 camera velocity; "unknown" on pre-migration runs.
  camera?: string;
  // On-screen text/graphics (title, lower-third, slide, UI) read off the
  // pixels by the pass 2 image LLM. "" / absent when there is none, or on a
  // pre-migration run.
  screen_text?: string;
  // This cut's single strongest INSTANT, code-computed (post._salience) --
  // distinct from hero_ts_ms (the best STILL for display). null/absent on a
  // pre-migration run or a cut with no usable signal. kind/span_ms/shape
  // (cuts_v4_segmentation.plan.md) are present only on a V4-ingested video
  // cut -- kind absent/null means this is a V3 cut (or has no signal),
  // and the dial keeps the original hero_ts_ms-centered symmetric shrink.
  // events/primary/density (v4_cluster_tree_cuts.plan.md): a video cut is a
  // CLUSTER carrying every salient event inside it; peak_ms/score/kind/
  // span_ms above are events[primary]'s own fields, broadcast to the top
  // level for backward compat. events.length <= 1 is the degenerate,
  // single-window case (today's behavior); > 1 is a genuine multi-event
  // cluster the dial resolves into multiple pieces (see resolveCluster).
  salience?: {
    peak_ms: number;
    score: number;
    kind?: "point" | "span" | "none" | null;
    span_ms?: [number, number] | null;
    shape?: "before" | "after" | "both" | "center" | "none" | null;
    events?: SalienceEvent[];
    primary?: number;
    density?: number;
  } | null;
  // av_coupling_authoritative.plan.md: this cut's baked AUTHORITATIVE audio
  // source, decided once at ingest (never re-derived lazily at render time).
  // audio_file_id defaults to this cut's own file_id (same-source, offset 0)
  // for the ~90% common case; a synced multicam cut couples to the group's
  // authoritative file with a per-cut refined offset_ms. confidence is
  // null for same-source cuts or when the refinement's guard fell back to
  // the group's unrefined global delta.
  audio_file_id?: string;
  audio_offset_ms?: number;
  audio_align_confidence?: number | null;
}

export type IngestStatus = "pending" | "pass1" | "images" | "pass2" | "post" | "ready" | "failed";

export interface IngestRun {
  id: string;
  status: IngestStatus;
  pass1_model: string | null;
  pass2_model: string | null;
  input_tokens: number;
  output_tokens: number;
  cache_read_tokens: number;
  cache_write_tokens: number;
  cost_usd: number | null;
  project_summary: string | null;
  error: string | null;
  created_at: string | null;
  updated_at: string | null;
}

export interface CutsV3Response {
  project_id: string;
  name: string;
  ingest_run: IngestRun | null;
  cuts: CutRecord[];
}

export function createProject(fileIds: string[], token: string) {
  return request<{ project_id: string }>("/api/projects", {
    method: "POST",
    token,
    body: JSON.stringify({ file_ids: fileIds }),
  });
}

export function kickIngest(projectId: string, token: string) {
  return request<{ project_id: string; status: string }>(`/api/projects/${projectId}/ingest`, {
    method: "POST",
    token,
  });
}

export function getCutsV3(projectId: string, token: string) {
  // no-store: this is read after a re-ingest, so a cached body would show
  // stale cuts (the exact "nothing changed" trap). Always hit the server.
  return request<CutsV3Response>(`/api/projects/${projectId}/cuts-v3`, {
    token,
    cache: "no-store",
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

// --- Client analysis proxies (see client_proxy.plan.md) ---
// The desktop app uploads two tiny proxies (A: 480p@1fps+audio, B: 160x90@10fps)
// so analysis starts in seconds instead of waiting on the multi-GB raw.

export interface AnalysisProxyPresignResponse {
  proxy_a_url: string;
  proxy_a_key: string;
  proxy_b_url: string;
  proxy_b_key: string;
}

export function presignAnalysisProxies(fileId: string, token: string) {
  return request<AnalysisProxyPresignResponse>(
    `/api/upload/${fileId}/analysis-proxies/presign`,
    { method: "POST", token }
  );
}

export function completeAnalysisProxies(fileId: string, token: string) {
  return request<FileRecord>(
    `/api/upload/${fileId}/analysis-proxies/complete`,
    { method: "POST", token }
  );
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

/** A split/PiP cell: a normalized sub-rect of the canvas (0..1). */
export interface DestRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Per-layer geometric framing; mirrors backend `layers.solve_transform`. */
export interface LayerTransform {
  rotate?: 0 | 90 | 180 | 270;
  fit?: LayerFit;
  anchor?: LayerAnchor;
  zoom?: number;
  focus?: LayerFocus;
  motion?: LayerMotion;
  dest?: "full" | DestRect;
}

/** ASC CDL -- the steerable, round-trippable grade spine; mirrors backend
 * `grade.cdl.Grade`. See color_grading.plan.md SS2.1. */
export interface CdlGrade {
  slope?: [number, number, number];
  offset?: [number, number, number];
  power?: [number, number, number];
  sat?: number;
}

/** Per-timeline-item grade override (SS2.4) -- an explicit nudge on top of
 * whatever the resolver would otherwise compute; same CDL shape as the
 * resolved grade below. */
export type ClipGrade = CdlGrade;

/** Fully resolved per-clip grade descriptor; mirrors backend
 * `grade.resolver.resolve_clip_grade`'s output. `grade_hash` is the
 * BACKEND's cache key for the baked `.cube` -- present when this came from
 * a server resolve, absent on a purely local (frontend) resolve, since the
 * frontend never computes it itself (see resolve-timeline.ts): the cube
 * endpoint takes the raw `cdl`/`creative_lut_ref`/`working_space` values
 * and hashes/caches server-side, so two independent hash implementations
 * never have to agree. */
/** Soft-local vignette descriptor (SS9) -- see backend `grade/softlocal.py`
 * for why this is an approximate-parity effect, unlike the CDL/LUT itself. */
export interface VignetteDescriptor {
  cx: number;
  cy: number;
  strength: number;
}

export interface SoftLocalDescriptor {
  vignette?: VignetteDescriptor | null;
}

export interface ResolvedGrade {
  cdl: Required<CdlGrade>;
  creative_lut_ref?: string | null;
  working_space: string;
  soft_local?: SoftLocalDescriptor | null;
  grade_hash?: string;
}

/** Sequence-level look selection (SS2.4/SS7): one of three input modes
 * (preset / reference-image / .cube upload) collapsing into the same CDL
 * spine, plus the user's single arc intensity dial (SS8). */
export type LookMode = "preset" | "reference" | "lut";

export interface SequenceLook {
  mode?: LookMode;
  preset_id?: string | null;
  reference_image_ref?: string | null;
  /** Pre-computed mean/std of the reference image (see backend
   * `grade.reference_transfer.compute_image_stats`), cached here once at
   * upload time so the resolver never has to re-decode the image. */
  reference_stats?: { rgb_mean: [number, number, number]; rgb_std: [number, number, number] } | null;
  lut_ref?: string | null;
  match_strength?: number;     // reference-image mode: 0..1
  arc_intensity?: number;      // 0 = flat, 1 = full arc (SS8)
  vignette_strength?: number;  // 0 = off (default), 1 = strongest (SS9)
}

/** A time-scoped spatial layout (split-screen / PiP): mirrors backend
 * `document.layout_regions`. Cells map a template slot to a layer selector
 * ("spine" for the main line, or a place_video op_id). */
export type LayoutTemplate = "split_h" | "split_v" | "pip";

export interface LayoutRegion {
  region_id: string;
  from_ms: number;
  to_ms: number;
  template: LayoutTemplate;
  cells: Record<string, { layer: string }>;
}

export interface EditFormat {
  aspect?: EditAspect;
  fit?: LayerFit;
  motion_style?: MotionStyle;
  motion_feel?: MotionFeel;
}

/** Per-project sequence-level color grade selection (SS2.4). */
export type EditLook = SequenceLook;

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
  grade?: ClipGrade;
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
  | "place_audio"
  | "split_edit"
  | "level";

export interface EditOperation {
  op_id: string;
  type: EditOperationType;
  rationale?: string | null;
  warnings?: string[];
  // place_video / place_audio
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
  grade?: ClipGrade;
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
  grade?: ResolvedGrade;
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

// --- Captions (captions.plan.md SS3/SS5) -----------------------------------

export type CaptionAnimationPreset = "fade" | "pop" | "karaoke" | "slide";
export type CaptionEmphasisMode = "semantic" | "loudness" | "none";
export type CaptionPlacementAnchor = "lower_third" | "center" | "top" | "dynamic" | "speaker";
export type CaptionColourSource = "white" | "black_box" | "match_grade" | "palette_accent" | "high_contrast";

export interface CaptionFont {
  font_id: string;
  family: string;
  weight: number;
  fallback_stack: string;
  case: "as-is" | "upper";
  tracking: number;
  max_lines: number;
  max_chars_per_line: number;
}

export interface CaptionAnimation {
  preset: CaptionAnimationPreset;
  intensity: number;
  beat_sync: boolean;
  emphasis: CaptionEmphasisMode;
}

export interface CaptionPlacement {
  anchor: CaptionPlacementAnchor;
  safe_area: boolean;
  stability_ms: number;
}

export interface CaptionColour {
  source: CaptionColourSource;
  fill: string;
  emphasis_fill: string;
  outline: string;
  shadow: string;
  box?: string | null;
  /** Set once a style has been contrast-resolved against real footage
   * (resolved.captions / the suggestions endpoint's tiles) -- absent on the
   * raw, footage-independent Standards catalog entries. */
  strong_outline?: boolean;
}

/** A caption style bundle -- the thing one gallery tile represents (SS3). */
export interface CaptionStyle {
  style_id: string;
  label: string;
  tier: "suggested" | "standard";
  font: CaptionFont;
  animation: CaptionAnimation;
  placement: CaptionPlacement;
  colour: CaptionColour;
  rationale?: string | null;
}

/** Persisted selection on the document (SS3), mirrors `look`'s shape. */
export interface EditCaptions {
  style_id?: string | null;
  enabled: boolean;
  /** A full style snapshot at selection time, so a Suggested pick stays
   * resolvable even after the ephemeral suggestion cache regenerates. */
  base_style?: CaptionStyle | null;
  overrides?: Record<string, unknown> | null;
}

export interface CaptionWord {
  text: string;
  t_in_ms: number;
  t_out_ms: number;
  emphasized: boolean;
}

export interface CaptionLine {
  words: CaptionWord[];
}

/** One resolved caption "card" -- `resolved.captions[]`, the ONE track both
 * the preview overlay and the ASS export read (SS3/SS4). */
export interface ResolvedCaptionEvent {
  prog_start_ms: number;
  prog_end_ms: number;
  lines: CaptionLine[];
  box: [number, number, number, number];
  style_ref: string;
  style: CaptionStyle;
  anim: CaptionAnimation;
}

export interface ResolvedTimeline {
  duration_ms: number;
  video_layers: ResolvedVideoLayer[];
  audio_layers: ResolvedAudioLayer[];
  aspect?: EditAspect;
  captions?: ResolvedCaptionEvent[];
}

export interface EditDocument {
  brief?: EditBrief;
  format?: EditFormat;
  look?: EditLook;
  captions?: EditCaptions;
  spine?: EditSpine | null;
  operations?: EditOperation[];
  layout_regions?: LayoutRegion[];
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

export interface ThreadQuestion {
  id: string;
  prompt: string;
  options: string[];
  allow_multiple?: boolean;
  // The editor's suggested pick (one of `options`) + a one-line reason, and
  // optionally a preview of what it'll do if picked -- an enrichment of the
  // ask, not a new confirm round-trip (interactive_ask_and_salience.plan.md).
  recommended?: string;
  why?: string;
  preview?: string;
}

export interface ThreadMessageResult {
  reply: string;
  // The agentic editor applies edits DIRECTLY during the turn (no confirm). When
  // the turn changed the edit, `changed` is true and `document_version` is the
  // new version to refresh the timeline from.
  changed: boolean;
  document_version: number | null;
  // When the editor needs a user-owned decision (ask_user), the turn PAUSES:
  // `awaiting_user` is true and `questions` are shown as pickable options; the
  // user's answer is just their next message.
  awaiting_user: boolean;
  questions: ThreadQuestion[];
}

export function sendThreadMessage(id: string, text: string, token: string) {
  return request<ThreadMessageResult>(`/api/edit/threads/${id}/messages`, {
    method: "POST",
    body: JSON.stringify({ text }),
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
  look?: EditLook;
  captions?: EditCaptions;
}

export function saveEditDocument(id: string, body: SaveEditBody, token: string) {
  return request<{ version: number; document: EditDocument }>(
    `/api/edit/threads/${id}/document`,
    { method: "PUT", body: JSON.stringify(body), token }
  );
}

// --- Color grading ---

export interface GradePresetSummary {
  preset_id: string;
  label: string;
  description: string;
}

export function getGradePresets(token: string) {
  return request<GradePresetSummary[]>("/api/grade/presets", { token });
}

// --- Captions (captions.plan.md SS13) ---

export interface CaptionFontSummary {
  font_id: string;
  family: string;
  weight: number;
  archetype: string;
  fallback_stack: string;
  license: string;
}

export interface CaptionCatalog {
  fonts: CaptionFontSummary[];
  standards: CaptionStyle[];
}

export function getCaptionCatalog(token: string) {
  return request<CaptionCatalog>("/api/captions/catalog", { token });
}

export interface CaptionRepresentativeFrame {
  url: string | null;
  hero_ts_ms: number | null;
  caption_zone: [number, number, number, number] | null;
  subject_box: [number, number, number, number] | null;
}

export interface CaptionSuggestionsResponse {
  suggestions: CaptionStyle[];
  representative_frame: CaptionRepresentativeFrame | null;
  sample_words: CaptionWord[];
}

export function getCaptionSuggestions(
  threadId: string, token: string, opts: { version?: number; reshuffleSeed?: number } = {}
) {
  const params = new URLSearchParams({ thread_id: threadId });
  if (opts.version != null) params.set("version", String(opts.version));
  if (opts.reshuffleSeed != null) params.set("reshuffle_seed", String(opts.reshuffleSeed));
  return request<CaptionSuggestionsResponse>(`/api/captions/suggestions?${params}`, { token });
}

// --- Multicam sync (audio_sync.plan.md SS10) ---

export type SyncRole = "video_angle" | "audio";
export type SyncAlignedBy = "auto" | "manual";

export interface SyncDetectMember {
  file_id: string;
  offset_ms: number;
  confidence: number;
  role: SyncRole;
  aligned_by: SyncAlignedBy;
  high_confidence: boolean;
}

export interface SyncDetectGroup {
  members: SyncDetectMember[];
  suggested_authoritative_file_id: string | null;
}

export interface SyncDetectResult {
  // The selected files partitioned into same-audio groups (all-pairs overlap).
  // May be more than one group (e.g. two camera pairs split by a recording
  // break) or empty (none of the files share audio).
  groups: SyncDetectGroup[];
  // Usable files that overlapped nobody -- they use their own audio.
  ungrouped_file_ids: string[];
  unusable_file_ids: string[];
}

export function detectSync(fileIds: string[], token: string, roles?: Record<string, SyncRole>) {
  return request<SyncDetectResult>("/api/sync/detect", {
    method: "POST",
    body: JSON.stringify({ file_ids: fileIds, roles }),
    token,
  });
}

export interface SyncGroupMember {
  file_id: string;
  offset_ms: number;
  role: SyncRole;
  confidence: number | null;
  aligned_by: SyncAlignedBy;
}

export interface SyncGroup {
  id: string;
  project_id: string;
  authoritative_audio_file_id: string | null;
  created_by: string;
  created_at: string;
  members: SyncGroupMember[];
}

export interface CreateSyncGroupMember {
  file_id: string;
  offset_ms: number;
  role: SyncRole;
  confidence?: number | null;
  aligned_by: SyncAlignedBy;
}

export function createSyncGroup(
  members: CreateSyncGroupMember[], token: string, authoritativeAudioFileId?: string | null
) {
  return request<SyncGroup>("/api/sync/groups", {
    method: "POST",
    body: JSON.stringify({ members, authoritative_audio_file_id: authoritativeAudioFileId }),
    token,
  });
}

export function listSyncGroups(projectId: string, token: string) {
  return request<SyncGroup[]>(`/api/sync/groups?project_id=${projectId}`, { token });
}

export function setSyncAuthoritative(groupId: string, fileId: string, token: string) {
  return request<SyncGroup>(`/api/sync/groups/${groupId}/authoritative`, {
    method: "PATCH", body: JSON.stringify({ file_id: fileId }), token,
  });
}

export function nudgeSyncOffset(groupId: string, fileId: string, offsetMs: number, token: string) {
  return request<SyncGroup>(`/api/sync/groups/${groupId}/members/${fileId}`, {
    method: "PATCH", body: JSON.stringify({ offset_ms: offsetMs }), token,
  });
}

export function deleteSyncGroup(groupId: string, token: string) {
  return request<{ deleted: boolean }>(`/api/sync/groups/${groupId}`, { method: "DELETE", token });
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
