import Link from "next/link";

// Public landing page. "Try Beta" and "Sign in" are the only entry points into
// the app; both route into the (drive) group, which gates on auth. Monochrome,
// minimalist — accent is the near-white primary button only.
export default function Landing() {
  return (
    <main
      className="relative flex min-h-screen flex-col"
      style={{ background: "var(--background)", color: "var(--foreground)" }}
    >
      <header className="flex items-center justify-between px-6 py-5 sm:px-10">
        <span
          className="text-lg font-bold tracking-tight"
          style={{ fontFamily: "var(--font-brand)" }}
        >
          EDSO
        </span>
        <Link
          href="/login"
          className="text-sm transition-colors hover:text-[var(--foreground)]"
          style={{ color: "var(--muted)" }}
        >
          Sign in
        </Link>
      </header>

      <section className="flex flex-1 flex-col items-center justify-center px-6 pb-16 text-center">
        <span
          className="mb-7 rounded-full border px-3 py-1 text-xs font-medium"
          style={{ borderColor: "var(--border)", color: "var(--muted)" }}
        >
          Now in private beta
        </span>
        <h1 className="max-w-3xl text-4xl font-bold leading-[1.1] tracking-tight sm:text-6xl">
          The AI editor that turns raw footage into a finished cut.
        </h1>
        <p
          className="mt-6 max-w-xl text-base leading-relaxed sm:text-lg"
          style={{ color: "var(--muted)" }}
        >
          Upload your clips. EDSO finds the moments, builds the timeline, and
          hands you an edit you can ship — no timeline wrangling required.
        </p>
        <div className="mt-10 flex items-center gap-3">
          <Link
            href="/signup"
            className="rounded-lg px-6 py-3 text-sm font-semibold transition-opacity hover:opacity-90"
            style={{ background: "var(--accent)", color: "var(--background)" }}
          >
            Try Beta
          </Link>
          <Link
            href="/login"
            className="rounded-lg border px-6 py-3 text-sm font-medium transition-colors hover:bg-[var(--sidebar)]"
            style={{ borderColor: "var(--border)", color: "var(--foreground)" }}
          >
            Sign in
          </Link>
        </div>
      </section>

      <footer
        className="px-6 py-6 text-center text-xs"
        style={{ color: "var(--muted)" }}
      >
        © {new Date().getFullYear()} EDSO
      </footer>
    </main>
  );
}
