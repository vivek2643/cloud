"use client";

import Image from "next/image";
import { usePathname } from "next/navigation";
import { Film, Captions, Download, Folder, Clock, type LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { useDriveStore, type ProjectStage, type HomeView } from "@/stores/drive-store";

type RailEntry = {
  label: string;
  icon?: LucideIcon;
  logo?: boolean;
};

const STAGE_ITEMS: (RailEntry & { stage: ProjectStage })[] = [
  { label: "Media", stage: "media", icon: Film },
  { label: "Cuts", stage: "cuts", logo: true },
  // Colour grading is temporarily hidden — re-enable when ready.
  // { label: "Colour grading", stage: "color", icon: Palette },
  { label: "Captions", stage: "captions", icon: Captions },
  { label: "Export", stage: "export", icon: Download },
];

const HOME_ITEMS: (RailEntry & { view: HomeView })[] = [
  { label: "Projects", view: "projects", icon: Folder },
  { label: "Recents", view: "recents", icon: Clock },
];

// The rail is always present, but what it offers depends on where you are:
// the stages above act on one open project, so at the project list they are
// replaced by the two lenses that list actually has.
export function Sidebar() {
  const pathname = usePathname();
  const isHome = pathname === "/drive";

  return (
    <aside
      className="flex w-20 shrink-0 flex-col border-r py-3"
      style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
    >
      <nav className="flex flex-col gap-1 px-1.5">
        {isHome ? <HomeNav /> : <StageNav />}
      </nav>
    </aside>
  );
}

function HomeNav() {
  const homeView = useDriveStore((s) => s.homeView);
  const setHomeView = useDriveStore((s) => s.setHomeView);

  return (
    <>
      {HOME_ITEMS.map((item) => (
        <RailItem
          key={item.view}
          item={item}
          active={homeView === item.view}
          onClick={() => setHomeView(item.view)}
        />
      ))}
    </>
  );
}

function StageNav() {
  const projectStage = useDriveStore((s) => s.projectStage);
  const setProjectStage = useDriveStore((s) => s.setProjectStage);

  return (
    <>
      {STAGE_ITEMS.map((item) => (
        <RailItem
          key={item.stage}
          item={item}
          active={projectStage === item.stage}
          onClick={() => setProjectStage(item.stage)}
        />
      ))}
    </>
  );
}

function RailItem({
  item,
  active,
  onClick,
}: {
  item: RailEntry;
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
