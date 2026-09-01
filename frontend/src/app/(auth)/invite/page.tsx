"use client";

import { Suspense, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import { createClient } from "@/lib/supabase";

const INVITE_COOKIE = "edso_invite";
const COOKIE_MAX_AGE = 60 * 60 * 24 * 30; // 30 days

function InviteForm() {
  const router = useRouter();
  const params = useSearchParams();
  const [code, setCode] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const supabase = createClient();
      const { data, error: rpcError } = await supabase.rpc("redeem_invite_code", {
        p_code: code,
      });

      if (rpcError) {
        setError(rpcError.message);
        return;
      }

      if (!data?.ok) {
        setError("That invite code is not valid or has already been used up.");
        return;
      }

      document.cookie = `${INVITE_COOKIE}=ok; path=/; max-age=${COOKIE_MAX_AGE}; samesite=lax`;

      // Only ever send them to an in-app path -- never reflect an absolute URL
      // from the query string back into a redirect.
      const next = params.get("next");
      router.push(next && next.startsWith("/") ? next : "/signup");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="w-full max-w-sm space-y-6">
      <div className="text-center">
        <h1 className="text-2xl font-bold tracking-tight">Edso</h1>
        <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>
          Enter your invite code to continue
        </p>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div
            className="rounded-lg px-3 py-2 text-sm"
            style={{ background: "rgba(220,38,38,0.1)", color: "var(--danger)" }}
          >
            {error}
          </div>
        )}

        <div>
          <label htmlFor="code" className="mb-1 block text-sm font-medium">
            Invite code
          </label>
          <input
            id="code"
            type="text"
            value={code}
            onChange={(e) => setCode(e.target.value)}
            required
            autoFocus
            placeholder="EDSO-XXXX"
            className="w-full rounded-lg border px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-[var(--accent)]"
            style={{ borderColor: "var(--border)", background: "var(--background)" }}
          />
        </div>

        <button
          type="submit"
          disabled={loading}
          className="w-full rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:opacity-50"
          style={{ background: "var(--accent)", color: "var(--background)" }}
        >
          {loading ? "Checking..." : "Continue"}
        </button>
      </form>

      <p className="text-center text-sm" style={{ color: "var(--muted)" }}>
        Already have an account?{" "}
        <Link href="/login" className="font-medium" style={{ color: "var(--accent)" }}>
          Sign in
        </Link>
      </p>

      <p className="text-center text-sm" style={{ color: "var(--muted)" }}>
        Edso is in a private beta right now &mdash; reach out and we&apos;ll add
        you to the waitlist.
      </p>
    </div>
  );
}

export default function InvitePage() {
  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      {/* useSearchParams needs a Suspense boundary to keep this route static. */}
      <Suspense fallback={null}>
        <InviteForm />
      </Suspense>
    </div>
  );
}
