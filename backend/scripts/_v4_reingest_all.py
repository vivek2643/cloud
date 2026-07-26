"""One-off: re-ingest the remaining projects under the V4 segmenter, printing a
clear per-project marker as each finishes (so the session can relay progress).
Run with CUTS_SEGMENTER=v4. Concurrency kept modest -- real paid API calls.

Resilient to Anthropic's sustained "Overloaded" windows: the client already
retries a single call with backoff, but a multi-minute outage can outlast that
budget, so each PROJECT is also retried a few times with a longer cooldown."""
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

from app.services.l3 import ingest, pass1

# Outer per-project retry for sustained API overload (beyond the client's own
# per-call backoff). Cooldown is long enough to sit out a real overload window.
_PROJECT_RETRIES = 5
_PROJECT_COOLDOWN_S = 120.0

REMAINING = [
    "72d87ca9-bb3e-4fc6-a270-fa86a249fc08",
    "642e9587-e3a4-43ad-8342-c58ae58c04ca",
    "f48da65f-0428-4480-80d6-742d1375bf9e",
    "f52e7ee1-5e33-454a-94cd-3749f1d3614e",
    "7ef4663d-8807-42f8-b114-1cb044b1c580",
    "94f92040-b2a1-46fd-9bae-09818ccc19bd",
    "a294f9da-ada2-4941-a23e-be3ff37c79b9",
    "a596ea5f-edfc-4d0b-8a22-bdab9e0458b0",
    "41fb01fc-08b3-4fbc-973d-995aabb52e4e",
    "91688328-272c-46d2-9879-15c560fd8a62",
    "8621c012-58c1-4bd5-8897-b8e8e4f24dca",
    "5cd8f004-13c7-43f8-a1ed-d2f7e646fae7",
    "57b689b3-39db-4cb4-8385-9e87a996fe9a",
]


def _counts(rid: str):
    with pass1._pg_conn() as conn:
        return conn.execute(
            "select count(*), count(*) filter (where kind='video'), "
            "count(*) filter (where kind='speech') from cut_records where ingest_run_id=%s",
            (rid,),
        ).fetchone()


def _is_overload(exc: Exception) -> bool:
    if isinstance(exc, anthropic.APIStatusError):
        body = getattr(exc, "body", None)
        etype = (body or {}).get("error", {}).get("type") if isinstance(body, dict) else None
        return etype in {"overloaded_error", "api_error", "rate_limit_error"} or \
            getattr(exc, "status_code", None) in {429, 500, 502, 503, 504, 529}
    return isinstance(exc, (anthropic.APIConnectionError, anthropic.APITimeoutError,
                            anthropic.InternalServerError, anthropic.RateLimitError))


def _run(pid: str):
    last: Exception | None = None
    for attempt in range(_PROJECT_RETRIES + 1):
        try:
            rid = ingest.run_ingest(pid)
            return pid, rid, _counts(rid)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if not _is_overload(exc) or attempt == _PROJECT_RETRIES:
                raise
            print(f"PROJECT_RETRY {pid} attempt {attempt + 1}/{_PROJECT_RETRIES} "
                  f"after API overload; cooling down {_PROJECT_COOLDOWN_S:.0f}s", flush=True)
            time.sleep(_PROJECT_COOLDOWN_S)
    raise last  # unreachable


def main():
    done = 0
    total = len(REMAINING)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(_run, pid): pid for pid in REMAINING}
        for fut in as_completed(futures):
            pid = futures[fut]
            done += 1
            try:
                pid, rid, counts = fut.result()
                print(f"PROJECT_DONE [{done}/{total}] {pid} run={rid} "
                      f"total/video/speech={counts}", flush=True)
            except Exception as e:
                print(f"PROJECT_FAIL [{done}/{total}] {pid} error={e!r}", flush=True)
    print("ALL_PROJECTS_FINISHED", flush=True)


if __name__ == "__main__":
    main()
