"use client";

import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase";
import { useAuthStore } from "@/stores/auth-store";
import { useDriveStore, type ViewMode } from "@/stores/drive-store";
import { LogOut, Grid3X3, List, Search, HardDrive } from "lucide-react";

export function Navbar() {
  const router = useRouter();
  const user = useAuthStore((s) => s.user);
  const viewMode = useDriveStore((s) => s.viewMode);
  const setViewMode = useDriveStore((s) => s.setViewMode);

  async function handleLogout() {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/login");
  }

  return (
    <header
      className="flex h-14 shrink-0 items-center gap-4 border-b px-4"
      style={{ borderColor: "var(--border)" }}
    >
      <div className="flex items-center gap-2 font-semibold">
        <HardDrive size={20} style={{ color: "var(--accent)" }} />
        <span>AeroDrive</span>
      </div>

      <div className="relative mx-auto w-full max-w-md">
        <Search
          size={16}
          className="absolute left-3 top-1/2 -translate-y-1/2"
          style={{ color: "var(--muted)" }}
        />
        <input
          type="text"
          placeholder="Search files..."
          className="w-full rounded-lg border py-1.5 pl-9 pr-3 text-sm outline-none focus:ring-2"
          style={{
            borderColor: "var(--border)",
            background: "var(--sidebar)",
          }}
        />
      </div>

      <div className="flex items-center gap-1">
        <ViewToggle mode="grid" current={viewMode} onChange={setViewMode} />
        <ViewToggle mode="list" current={viewMode} onChange={setViewMode} />
      </div>

      {user && (
        <button
          onClick={handleLogout}
          className="flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm transition-colors hover:opacity-80"
          style={{ color: "var(--muted)" }}
          title="Sign out"
        >
          <LogOut size={16} />
        </button>
      )}
    </header>
  );
}

function ViewToggle({
  mode,
  current,
  onChange,
}: {
  mode: ViewMode;
  current: ViewMode;
  onChange: (m: ViewMode) => void;
}) {
  const Icon = mode === "grid" ? Grid3X3 : List;
  const isActive = mode === current;
  return (
    <button
      onClick={() => onChange(mode)}
      className="rounded-md p-1.5 transition-colors"
      style={{
        background: isActive ? "var(--border)" : "transparent",
        color: isActive ? "var(--foreground)" : "var(--muted)",
      }}
      title={mode === "grid" ? "Grid view" : "List view"}
    >
      <Icon size={16} />
    </button>
  );
}
