"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";

// Brand logo animation, shown once on the way in from sign-in, sign-up or the
// OAuth callback.
//
// This is a route rather than an overlay on /drive. As an overlay it could
// never win: it needed the ?intro=1 query, which only reaches a client
// component, so the drive shell shipped in the server HTML and painted first
// and the logo landed on top of an app the user could already see. Here there
// is no shell to lose the race against -- /drive is not mounted until the
// video is done.
//
// It never traps anyone: blocked autoplay, a decode error, a click, or the
// safety timeout all move on immediately.
const MAX_MS = 7000;

export default function WelcomePage() {
  const router = useRouter();
  const videoRef = useRef<HTMLVideoElement>(null);
  const [leaving, setLeaving] = useState(false);
  // Guards every exit path -- onEnded, onError, the timeout and a click can
  // all fire for one visit, and each would otherwise push its own navigation.
  const done = useRef(false);

  const enter = useCallback(() => {
    if (done.current) return;
    done.current = true;
    setLeaving(true);
    // replace, not push: the back button should leave the app, not replay this.
    setTimeout(() => router.replace("/drive"), 300);
  }, [router]);

  useEffect(() => {
    // Warm /drive while the logo plays, so the wait buys something.
    router.prefetch("/drive");

    videoRef.current?.play().catch(enter); // autoplay blocked -> don't stall
    const safety = setTimeout(enter, MAX_MS);
    return () => clearTimeout(safety);
  }, [router, enter]);

  return (
    <div
      onClick={enter}
      className="fixed inset-0 z-50 flex items-center justify-center transition-opacity duration-300"
      style={{ background: "var(--background)", opacity: leaving ? 0 : 1 }}
    >
      <video
        ref={videoRef}
        src="/logo_animation.mp4"
        muted
        playsInline
        autoPlay
        onEnded={enter}
        onError={enter}
        className="max-h-[70vh] max-w-[70vw] object-contain"
      />
      <button
        onClick={enter}
        className="absolute bottom-8 right-8 text-xs font-medium transition-colors hover:opacity-100"
        style={{ color: "var(--muted)", opacity: 0.7 }}
      >
        Skip
      </button>
    </div>
  );
}
