"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase";
import { useAuthStore } from "@/stores/auth-store";
import { LogOut, Settings, CreditCard, Home } from "lucide-react";

export function Navbar() {
  return (
    <header
      className="flex h-14 shrink-0 items-center gap-4 border-b pl-3 pr-4"
      style={{ borderColor: "var(--border)" }}
    >
      {/* Wordmark: the gap between the logo and "Edso" matches the gap to the
          left of the logo (pl-3 == gap-3) so the mark + name read as one unit.
          Links home since the sidebar no longer carries a Projects item. */}
      <Link href="/drive" className="ml-2 flex items-center gap-3">
        <Image
          src="/edso-logo.png"
          alt="Edso"
          width={233}
          height={283}
          priority
          className="h-4 w-auto"
        />
        <span
          className="relative text-[21px] font-semibold leading-none"
          style={{ fontFamily: "var(--font-brand)", letterSpacing: "-0.01em", top: "2px" }}
        >
          Edso
        </span>
      </Link>

      <div className="ml-auto flex items-center gap-2">
        <Link
          href="/drive"
          title="Home"
          className="flex h-8 w-8 items-center justify-center rounded-lg transition-colors hover:bg-[var(--sidebar)]"
          style={{ color: "var(--muted)" }}
        >
          <Home size={18} />
        </Link>
        <AccountMenu />
      </div>
    </header>
  );
}

function AccountMenu() {
  const router = useRouter();
  const user = useAuthStore((s) => s.user);
  const [open, setOpen] = useState(false);

  if (!user) return null;

  const meta = (user.user_metadata ?? {}) as Record<string, string>;
  const name = meta.full_name || meta.name || user.email?.split("@")[0] || "Account";
  const email = user.email ?? "";
  const initials = (name.match(/\b\w/g) ?? []).slice(0, 2).join("").toUpperCase() || "?";

  async function handleLogout() {
    const supabase = createClient();
    await supabase.auth.signOut();
    router.push("/login");
  }

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex h-8 w-8 items-center justify-center rounded-full text-xs font-medium transition-opacity hover:opacity-80"
        style={{ background: "#ed5b00", color: "#fff" }}
        title={name}
      >
        {initials}
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setOpen(false)} />
          <div
            className="absolute right-0 z-50 mt-2 w-60 overflow-hidden rounded-xl border shadow-xl"
            style={{ background: "var(--background)", borderColor: "var(--border)" }}
          >
            <div className="border-b px-3 py-3" style={{ borderColor: "var(--border)" }}>
              <div className="truncate text-sm font-medium">{name}</div>
              {email && (
                <div className="truncate text-xs" style={{ color: "var(--muted)" }}>
                  {email}
                </div>
              )}
            </div>
            <div className="py-1">
              <MenuItem
                icon={Settings}
                label="Settings"
                onClick={() => {
                  setOpen(false);
                  router.push("/settings");
                }}
              />
              <MenuItem
                icon={CreditCard}
                label="Plans and billing"
                onClick={() => {
                  setOpen(false);
                  router.push("/settings/billing");
                }}
              />
            </div>
            <div className="border-t py-1" style={{ borderColor: "var(--border)" }}>
              <MenuItem icon={LogOut} label="Sign out" onClick={handleLogout} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function MenuItem({
  icon: Icon,
  label,
  onClick,
}: {
  icon: typeof Settings;
  label: string;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className="flex w-full items-center gap-2.5 px-3 py-2 text-sm transition-colors hover:bg-[var(--sidebar)]"
      style={{ color: "var(--foreground)" }}
    >
      <Icon size={16} style={{ color: "var(--muted)" }} />
      {label}
    </button>
  );
}
