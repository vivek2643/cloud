"use client";

import {
  Film,
  Image as ImageIcon,
  Music,
  FileText,
  File,
  FolderClosed,
} from "lucide-react";

const TYPE_ICONS = {
  video: Film,
  image: ImageIcon,
  audio: Music,
  document: FileText,
  other: File,
  folder: FolderClosed,
} as const;

const TYPE_COLORS = {
  video: "#8b5cf6",
  image: "#ec4899",
  audio: "#f59e0b",
  document: "#3b82f6",
  other: "#6b7280",
  folder: "#3b82f6",
};

export function FileIcon({
  type,
  size = 20,
}: {
  type: keyof typeof TYPE_ICONS;
  size?: number;
}) {
  const Icon = TYPE_ICONS[type];
  const color = TYPE_COLORS[type];
  return <Icon size={size} style={{ color }} />;
}
