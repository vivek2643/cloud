"use client";

import { useState } from "react";
import { DriveContent } from "@/components/drive-content";
import { DialoguesView } from "@/components/dialogues-view";
import { HeroCutsView } from "@/components/hero-cuts-view";
import { Layers, Crosshair, Star, MessageSquare } from "lucide-react";

type Tab = "versioned" | "hero" | "dialogues" | "crux";

const TABS: { id: Tab; label: string; icon: typeof Layers }[] = [
  { id: "versioned", label: "Versioned", icon: Layers },
  { id: "hero", label: "Hero Cuts", icon: Star },
  { id: "dialogues", label: "Dialogues", icon: MessageSquare },
  { id: "crux", label: "Crux", icon: Crosshair },
];

/**
 * The per-project lens switcher. A folder is one project, so the lenses
 * (Versioned / Dialogues / Crux / Highlights) operate on the files inside the
 * current folder. The drive root deliberately has no lenses.
 */
export function ProjectLenses() {
  const [activeTab, setActiveTab] = useState<Tab>("versioned");

  return (
    <div>
      <div className="mb-6 flex flex-wrap items-center gap-1.5 border-b pb-3" style={{ borderColor: "var(--border)" }}>
        {TABS.map((t) => {
          const active = t.id === activeTab;
          const Icon = t.icon;
          return (
            <button
              key={t.id}
              onClick={() => setActiveTab(t.id)}
              className="flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-sm font-medium transition-colors"
              style={{
                background: active ? "var(--accent)" : "transparent",
                color: active ? "#fff" : "var(--muted)",
                border: active ? "1px solid var(--accent)" : "1px solid var(--border)",
              }}
            >
              <Icon size={15} />
              {t.label}
            </button>
          );
        })}
      </div>

      {activeTab === "versioned" ? (
        <DriveContent />
      ) : activeTab === "hero" ? (
        <HeroCutsView />
      ) : activeTab === "dialogues" ? (
        <DialoguesView />
      ) : (
        <ComingSoon tab={activeTab} />
      )}
    </div>
  );
}

function ComingSoon({ tab }: { tab: Tab }) {
  const copy = { icon: <Crosshair size={36} style={{ color: "var(--accent)" }} />, title: "Crux", body: "The key moments in this project's footage will surface here." };
  void tab;
  return (
    <div className="flex flex-col items-center justify-center py-24 text-center">
      {copy.icon}
      <p className="mt-4 text-lg font-semibold">{copy.title}</p>
      <p className="mt-1 max-w-sm text-sm" style={{ color: "var(--muted)" }}>
        {copy.body}
      </p>
      <span
        className="mt-4 rounded-full px-3 py-1 text-xs font-medium"
        style={{ background: "var(--accent-soft)", color: "var(--accent)" }}
      >
        Coming soon
      </span>
    </div>
  );
}
