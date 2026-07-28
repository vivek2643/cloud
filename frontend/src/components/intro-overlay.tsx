"use client";

import { useEffect, useRef, useState } from "react";

// Plays the brand logo animation (public/logo_animation.mp4) ONLY when you
// arrive at /drive from an entry point (sign-in, sign-up, OAuth callback, or a
// home-page CTA). Those redirects tag the URL with `?intro=1`; this overlay
// plays once, then strips the tag. Navigating within the app (e.g. a folder ->
// drive) never carries the tag, so the logo doesn't replay.
//
// It never blocks the app: any autoplay/decode failure dismisses immediately,
// and a safety timeout always tears it down.
const MAX_MS = 7000;

export function IntroOverlay() {
  const [show, setShow] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const params = new URLSearchParams(window.location.search);
    if (params.get("intro") !== "1") return;
    // Strip the tag so a refresh / back-forward doesn't replay it.
    params.delete("intro");
    const qs = params.toString();
    window.history.replaceState(
      null,
      "",
      window.location.pathname + (qs ? `?${qs}` : "")
    );
    setShow(true);
  }, []);

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
