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
