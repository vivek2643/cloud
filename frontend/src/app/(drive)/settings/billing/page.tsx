"use client";

export default function BillingPage() {
  return (
    <div className="mx-auto w-full max-w-2xl px-6 py-10">
      <h1 className="text-xl font-semibold tracking-tight">Plans and billing</h1>
      <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>
        Manage your plan, payment method, and invoices.
      </p>

      <div
        className="mt-8 rounded-xl border p-5"
        style={{ borderColor: "var(--border)", background: "var(--sidebar)" }}
      >
        <div className="text-sm font-medium">Current plan</div>
        <div className="mt-1 text-xs" style={{ color: "var(--muted)" }}>
          Free — billing isn’t set up yet.
        </div>
      </div>
    </div>
  );
}
