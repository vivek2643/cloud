"use client";

import Image from "next/image";
import { Film, Palette, Captions, Sparkles, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { useDriveStore, type ProjectStage } from "@/stores/drive-store";

type NavItem = {
  label: string;
  stage: ProjectStage;
  icon?: LucideIcon;
  logo?: boolean;
};

const NAV_ITEMS: NavItem[] = [
  { label: "Media", stage: "media", icon: Film },
  { label: "Cuts", stage: "cuts", logo: true },
  { label: "Cuts v3", stage: "cuts-v3", icon: Sparkles },
  { label: "Colour grading", stage: "color", icon: Palette },
  { label: "Captions", stage: "captions", icon: Captions },
];

export function Sidebar() {
  const projectStage = useDriveStore((s) => s.projectStage);
  const setProjectStage = useDriveStore((s) => s.setProjectStage);

  return (
    <aside
      className="flex w-20 shrink-0 flex-col border-r py-3"
      style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
    >
      <nav className="flex flex-col gap-1 px-1.5">
        {NAV_ITEMS.map((item) => (
          <RailItem
            key={item.stage}
            item={item}
            active={projectStage === item.stage}
            onClick={() => setProjectStage(item.stage)}
          />
        ))}
      </nav>
    </aside>
  );
}

function RailItem({
  item,
  active,
  onClick,
}: {
  item: NavItem;
  active: boolean;
  onClick: () => void;
}) {
  const { label, icon: Icon, logo } = item;
  return (
    <button
      type="button"
      onClick={onClick}
      title={label}
      className={cn(
        "flex w-full flex-col items-center gap-1.5 rounded-xl py-2.5 transition-colors",
        !active && "hover:bg-[var(--border)]",
      )}
      style={{
        background: active ? "var(--accent-soft)" : undefined,
        color: active ? "var(--foreground)" : "var(--muted)",
      }}
    >
      <span className="flex h-6 items-center justify-center">
        {logo ? (
          <Image
            src="/edso-logo.png"
            alt={label}
            width={233}
            height={283}
            className="h-5 w-auto"
            style={{ filter: "grayscale(1)", opacity: active ? 1 : 0.7 }}
          />
        ) : (
          Icon && <Icon size={20} />
        )}
      </span>
      <span className="max-w-full text-center text-[11px] leading-tight">
        {label}
      </span>
    </button>
  );
}
