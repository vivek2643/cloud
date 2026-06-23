"use client";

export default function SettingsPage() {
  return (
    <div className="mx-auto w-full max-w-2xl px-6 py-10">
      <h1 className="text-xl font-semibold tracking-tight">Settings</h1>
      <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>
        Manage your account and workspace preferences.
      </p>

      <div className="mt-8 flex flex-col gap-px overflow-hidden rounded-xl border" style={{ borderColor: "var(--border)" }}>
        <SettingRow label="Account" hint="Profile and sign-in details" />
        <SettingRow label="Appearance" hint="Theme and display options" />
        <SettingRow label="Storage" hint="Usage and limits" />
      </div>
    </div>
  );
}

function SettingRow({ label, hint }: { label: string; hint: string }) {
  return (
    <div
      className="flex items-center justify-between px-4 py-3.5"
      style={{ background: "var(--sidebar)" }}
    >
      <div>
        <div className="text-sm font-medium">{label}</div>
        <div className="text-xs" style={{ color: "var(--muted)" }}>
          {hint}
        </div>
      </div>
      <span className="text-xs" style={{ color: "var(--muted)" }}>
        Coming soon
      </span>
    </div>
  );
}
