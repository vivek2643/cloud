#!/usr/bin/env python3
"""
Benchmark the agentic director on the SAME edit across LLM providers.

Phase 4 deliverable: decide the backbone (Claude vs Gemini) with data, not vibes.
Runs `director.direct_edit` once per provider on identical footage + brief and
prints wall-clock latency plus the perception telemetry already collected by the
director (rounds, view_frames calls, images pulled, token counts).

This DOES hit the real DB and the real provider APIs, so set the relevant keys:
  ANTHROPIC_API_KEY (for anthropic), GEMINI_API_KEY + `pip install google-genai`
  (for gemini), plus the usual SUPABASE/R2/DATABASE_URL.

Run from backend/:
  .venv/bin/python scripts/bench_llm_provider.py \
      --file-id <uuid> --brief "tight 30s highlight" --duration 30 \
      --providers anthropic gemini
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import List, Optional

from app.config import get_settings
from app.services.l3 import director as director_mod


def _run_once(provider: str, *, file_ids: Optional[List[str]], folder_id: Optional[str],
              brief: str, duration: Optional[int]) -> dict:
    settings = get_settings()
    settings.llm_provider = provider  # mutate the cached settings for this run

    messages = [{"role": "user", "content": brief}]
    t0 = time.time()
    res = director_mod.direct_edit(
        user_id=settings.dev_user_id,
        messages=messages,
        file_ids=file_ids,
        folder_id=folder_id,
        duration_target_s=duration,
    )
    elapsed = time.time() - t0

    perception = res.raw.get("perception", {}) if res.raw else {}
    has_av = bool(res.edl and (res.edl.get("video_track") or res.edl.get("audio_track")))
    return {
        "provider": provider,
        "elapsed_s": round(elapsed, 1),
        "built_timeline": has_av,
        "clip_count": len(res.timeline),
        "total_s": round(res.total_ms / 1000, 1),
        "images_used": perception.get("images_used"),
        "max_images": perception.get("max_images"),
        "calls": perception.get("calls"),
        "warnings": res.warnings[:5],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-id", nargs="*", default=None)
    ap.add_argument("--folder-id", default=None)
    ap.add_argument("--brief", required=True)
    ap.add_argument("--duration", type=int, default=None)
    ap.add_argument("--providers", nargs="+", default=["anthropic"])
    args = ap.parse_args()

    if not args.file_id and not args.folder_id:
        print("Provide --file-id <uuid...> or --folder-id <uuid>", file=sys.stderr)
        return 2

    results = []
    for prov in args.providers:
        print(f"\n=== running provider={prov} ===", file=sys.stderr)
        try:
            results.append(_run_once(
                prov, file_ids=args.file_id, folder_id=args.folder_id,
                brief=args.brief, duration=args.duration,
            ))
        except Exception as e:  # noqa: BLE001
            results.append({"provider": prov, "error": f"{type(e).__name__}: {e}"})

    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
