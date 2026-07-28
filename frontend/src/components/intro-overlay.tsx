"use client";

import { useEffect, useRef, useState } from "react";

// Plays the brand logo animation (public/logo_animation.mp4) as a full-screen
// intro the FIRST time you enter the platform in a browser session. It fades
// out on end / click / a safety timeout, and never blocks the app: any autoplay
// or decode failure dismisses it immediately. Session-scoped (sessionStorage)
// so it plays once on entry, not on every in-app navigation.
const SEEN_KEY = "edso-intro-seen";
const MAX_MS = 7000;

export function IntroOverlay() {
  const [show, setShow] = useState(false);
  const [leaving, setLeaving] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (window.sessionStorage.getItem(SEEN_KEY)) return;
    window.sessionStorage.setItem(SEEN_KEY, "1");
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
