"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { FolderClosed, Clock, Trash2, Star, Sparkles, ScrollText, Clapperboard } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { label: "My Drive", href: "/drive", icon: FolderClosed },
  { label: "AI Rough Cut", href: "/edit", icon: Sparkles },
  { label: "Edits", href: "/edits", icon: Clapperboard },
  { label: "Logs", href: "/logs", icon: ScrollText },
  { label: "Recent", href: "/drive/recent", icon: Clock },
  { label: "Starred", href: "/drive/starred", icon: Star },
  { label: "Trash", href: "/drive/trash", icon: Trash2 },
];

export function Sidebar({ collapsed = false }: { collapsed?: boolean }) {
  const pathname = usePathname();

  return (
    <aside
      className={cn(
        "flex shrink-0 flex-col border-r py-4 transition-[width] duration-200",
        collapsed ? "w-16" : "w-56",
      )}
      style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
    >
      <nav className={cn("flex flex-col gap-0.5", collapsed ? "px-2" : "px-3")}>
        {NAV_ITEMS.map((item) => {
          const active =
            item.href === "/drive"
              ? pathname === "/drive" || pathname.startsWith("/drive/folder")
              : pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              title={collapsed ? item.label : undefined}
              className={cn(
                "flex items-center rounded-lg py-2 text-sm transition-colors",
                collapsed ? "justify-center px-0" : "gap-2.5 px-3",
                active ? "font-medium" : "",
              )}
              style={{
                background: active ? "var(--border)" : "transparent",
                color: active ? "var(--foreground)" : "var(--muted)",
              }}
            >
              <item.icon size={16} />
              {!collapsed && item.label}
            </Link>
          );
        })}
      </nav>

      {!collapsed && (
        <div className="mt-auto px-3 pt-4">
          <div className="rounded-lg p-3 text-xs" style={{ background: "var(--background)" }}>
            <div className="mb-1 font-medium">Storage</div>
            <div className="h-1.5 overflow-hidden rounded-full" style={{ background: "var(--border)" }}>
              <div className="h-full rounded-full" style={{ width: "12%", background: "var(--accent)" }} />
            </div>
            <div className="mt-1" style={{ color: "var(--muted)" }}>
              1.2 GB of 10 GB used
            </div>
          </div>
        </div>
      )}
    </aside>
  );
}
