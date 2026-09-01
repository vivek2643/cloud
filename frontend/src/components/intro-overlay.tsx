"use client";

import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";

// Plays the brand logo animation (public/logo_animation.mp4) ONLY when you
// arrive at /drive from an entry point (sign-in, sign-up, OAuth callback, or a
// home-page CTA). Those redirects tag the URL with `?intro=1`; this overlay
// plays once, then strips the tag. Navigating within the app (e.g. a folder ->
// drive) never carries the tag, so the logo doesn't replay.
//
// It never blocks the app: any autoplay/decode failure dismisses immediately,
// and a safety timeout always tears it down.
const MAX_MS = 7000;

function Intro() {
  // Read via useSearchParams, not window.location in an effect. An effect runs
  // after the first paint, so the app was painting first and the logo appeared
  // on top of it a moment later. useSearchParams is available while the server
  // renders, so the overlay ships in the initial HTML and is the first thing
  // drawn.
  const searchParams = useSearchParams();
  const [show, setShow] = useState(searchParams.get("intro") === "1");
  const [leaving, setLeaving] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (!show) return;
    // Strip the tag so a refresh / back-forward doesn't replay it. Safe to do
    // after paint: it only rewrites the URL, it doesn't gate what is drawn.
    const params = new URLSearchParams(window.location.search);
    if (params.get("intro") !== "1") return;
    params.delete("intro");
    const qs = params.toString();
    window.history.replaceState(
      null,
      "",
      window.location.pathname + (qs ? `?${qs}` : "")
    );
  }, [show]);

  useEffect(() => {
    if (!show) return;
    const v = videoRef.current;
    v?.play().catch(() => dismiss()); // autoplay blocked -> don't block the app
    const safety = setTimeout(dismiss, MAX_MS);
    return () => clearTimeout(safety);
  }, [show]);

  function dismiss() {
    setLeaving(true);
    setTimeout(() => setShow(false), 350);
  }

  if (!show) return null;

  return (
    <div
      onClick={dismiss}
      className="fixed inset-0 z-[100] flex items-center justify-center transition-opacity duration-300"
      style={{ background: "var(--background)", opacity: leaving ? 0 : 1 }}
    >
      <video
        ref={videoRef}
        src="/logo_animation.mp4"
        muted
        playsInline
        autoPlay
        onEnded={dismiss}
        onError={dismiss}
        className="max-h-[70vh] max-w-[70vw] object-contain"
      />
      <button
        onClick={dismiss}
        className="absolute bottom-8 right-8 text-xs font-medium transition-colors hover:opacity-100"
        style={{ color: "var(--muted)", opacity: 0.7 }}
      >
        Skip
      </button>
    </div>
  );
}

export function IntroOverlay() {
  // useSearchParams needs a Suspense boundary; without one it would opt the
  // whole drive tree into client-only rendering, which is the very thing that
  // caused the flash.
  return (
    <Suspense fallback={null}>
      <Intro />
    </Suspense>
  );
}
