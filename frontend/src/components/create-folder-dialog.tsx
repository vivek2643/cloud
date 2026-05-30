"use client";

import { useState, useRef, useEffect } from "react";
import { X } from "lucide-react";

interface Props {
  open: boolean;
  onClose: () => void;
  onCreate: (name: string) => void;
}

export function CreateFolderDialog({ open, onClose, onCreate }: Props) {
  const [name, setName] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open) {
      setName("");
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [open]);

  if (!open) return null;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const trimmed = name.trim();
    if (!trimmed) return;
    onCreate(trimmed);
    onClose();
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />
      <div
        className="relative w-full max-w-sm rounded-xl border p-6 shadow-xl"
        style={{ background: "var(--background)", borderColor: "var(--border)" }}
      >
        <button
          onClick={onClose}
          className="absolute right-4 top-4 rounded p-1 transition-colors hover:opacity-70"
          style={{ color: "var(--muted)" }}
        >
          <X size={16} />
        </button>

        <h2 className="text-lg font-semibold">New Folder</h2>
        <form onSubmit={handleSubmit} className="mt-4 space-y-4">
          <input
            ref={inputRef}
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Folder name"
            className="w-full rounded-lg border px-3 py-2 text-sm outline-none focus:ring-2"
            style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
          />
          <div className="flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-lg border px-4 py-2 text-sm transition-colors hover:opacity-80"
              style={{ borderColor: "var(--border)" }}
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!name.trim()}
              className="rounded-lg px-4 py-2 text-sm font-medium text-white transition-colors disabled:opacity-50"
              style={{ background: "var(--accent)" }}
            >
              Create
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
