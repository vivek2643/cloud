"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { FolderClosed, Clock, Trash2, Star, Sparkles, ScrollText } from "lucide-react";
import { cn } from "@/lib/utils";

const NAV_ITEMS = [
  { label: "My Drive", href: "/drive", icon: FolderClosed },
  { label: "AI Rough Cut", href: "/edit", icon: Sparkles },
  { label: "Logs", href: "/logs", icon: ScrollText },
  { label: "Recent", href: "/drive/recent", icon: Clock },
  { label: "Starred", href: "/drive/starred", icon: Star },
  { label: "Trash", href: "/drive/trash", icon: Trash2 },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside
      className="flex w-56 shrink-0 flex-col border-r py-4"
      style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
    >
      <nav className="flex flex-col gap-0.5 px-3">
        {NAV_ITEMS.map((item) => {
          const active =
            item.href === "/drive"
              ? pathname === "/drive" || pathname.startsWith("/drive/folder")
              : pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors",
                active ? "font-medium" : ""
              )}
              style={{
                background: active ? "var(--border)" : "transparent",
                color: active ? "var(--foreground)" : "var(--muted)",
              }}
            >
              <item.icon size={16} />
              {item.label}
            </Link>
          );
        })}
      </nav>

      <div className="mt-auto px-3 pt-4">
        <div className="rounded-lg p-3 text-xs" style={{ background: "var(--background)" }}>
          <div className="mb-1 font-medium">Storage</div>
          <div className="h-1.5 overflow-hidden rounded-full" style={{ background: "var(--border)" }}>
            <div
              className="h-full rounded-full"
              style={{ width: "12%", background: "var(--accent)" }}
            />
          </div>
          <div className="mt-1" style={{ color: "var(--muted)" }}>
            1.2 GB of 10 GB used
          </div>
        </div>
      </div>
    </aside>
  );
}
