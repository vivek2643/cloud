"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { createClient } from "@/lib/supabase";

export default function SignupPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [inviteCode, setInviteCode] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [emailSent, setEmailSent] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      const supabase = createClient();
      const { data, error: authError } = await supabase.auth.signUp({
        email,
        password,
        options: { data: { invite_code: inviteCode.trim().toUpperCase() } },
      });

      if (authError) {
        setError(authError.message);
        setLoading(false);
        return;
      }

      if (data.session) {
        router.push("/drive?intro=1");
      } else {
        setEmailSent(true);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  if (emailSent) {
    return (
      <div className="flex min-h-screen items-center justify-center px-4">
        <div className="w-full max-w-sm space-y-4 text-center">
          <div className="text-5xl">✉️</div>
          <h1 className="text-xl font-bold">Check your email</h1>
          <p className="text-sm" style={{ color: "var(--muted)" }}>
            We sent a confirmation link to <strong>{email}</strong>.
            Click the link to activate your account.
          </p>
          <Link
            href="/login"
            className="mt-4 inline-block text-sm font-medium"
            style={{ color: "var(--accent)" }}
          >
            Back to sign in
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-screen items-center justify-center px-4">
      <div className="w-full max-w-sm space-y-6">
        <div className="text-center">
          <h1 className="text-2xl font-bold tracking-tight">Edso</h1>
          <p className="mt-1 text-sm" style={{ color: "var(--muted)" }}>
            Create your account
          </p>
        </div>

        {/* invite_codes.plan.md Stage 5: the Google button is hidden on this
            page only -- new social accounts are rejected by the before-user-
            created hook (it can't carry an invite code), so leaving it would
            offer a path that always fails. login/page.tsx is untouched:
            existing Google users must still be able to sign in there. */}

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
            <label htmlFor="inviteCode" className="mb-1 block text-sm font-medium">
              Invite code
            </label>
            <input
              id="inviteCode"
              type="text"
              value={inviteCode}
              onChange={(e) => setInviteCode(e.target.value)}
              required
              className="w-full rounded-lg border px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-[var(--accent)]"
              style={{ borderColor: "var(--border)", background: "var(--background)" }}
            />
          </div>

          <div>
            <label htmlFor="email" className="mb-1 block text-sm font-medium">
              Email
            </label>
            <input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full rounded-lg border px-3 py-2 text-sm outline-none focus:ring-2 focus:ring-[var(--accent)]"
              style={{ borderColor: "var(--border)", background: "var(--background)" }}
            />
          </div>

          <div>
            <label htmlFor="password" className="mb-1 block text-sm font-medium">
              Password
            </label>
            <input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={6}
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
            {loading ? "Creating account..." : "Create account"}
          </button>
        </form>

        <p className="text-center text-sm" style={{ color: "var(--muted)" }}>
          Already have an account?{" "}
          <Link href="/login" className="font-medium" style={{ color: "var(--accent)" }}>
            Sign in
          </Link>
        </p>

        <p className="text-center text-sm" style={{ color: "var(--muted)" }}>
          Don&apos;t have an invite code? Edso is in a private beta right now
          &mdash; reach out and we&apos;ll add you to the waitlist.
        </p>
      </div>
    </div>
  );
}
