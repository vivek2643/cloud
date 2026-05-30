"use client";

import Link from "next/link";
import { ChevronRight, Home } from "lucide-react";
import type { BreadcrumbItem } from "@/lib/api";

interface Props {
  items: BreadcrumbItem[];
}

export function Breadcrumb({ items }: Props) {
  return (
    <nav className="flex items-center gap-1 text-sm">
      <Link
        href="/drive"
        className="flex items-center gap-1 rounded px-1.5 py-1 transition-colors hover:opacity-70"
        style={{ color: "var(--muted)" }}
      >
        <Home size={14} />
      </Link>

      {items.map((item, i) => {
        const isLast = i === items.length - 1;
        return (
          <span key={item.id ?? "root"} className="flex items-center gap-1">
            <ChevronRight size={14} style={{ color: "var(--muted)" }} />
            {isLast ? (
              <span className="rounded px-1.5 py-1 font-medium">{item.name}</span>
            ) : (
              <Link
                href={item.id ? `/drive/folder/${item.id}` : "/drive"}
                className="rounded px-1.5 py-1 transition-colors hover:opacity-70"
                style={{ color: "var(--muted)" }}
              >
                {item.name}
              </Link>
            )}
          </span>
        );
      })}
    </nav>
  );
}
