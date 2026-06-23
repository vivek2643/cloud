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

export function FileIcon({
  type,
  size = 20,
}: {
  type: keyof typeof TYPE_ICONS;
  size?: number;
}) {
  const Icon = TYPE_ICONS[type];
  return <Icon size={size} style={{ color: "var(--foreground)" }} />;
}
