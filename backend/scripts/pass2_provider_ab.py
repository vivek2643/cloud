"""
gemini_pass2.plan.md Phase 5 -- the A/B merge-gate harness. For each given
project, runs Cuts v3 ingest TWICE (once per pass-2 provider), compares the
resulting cut_records on the plan's quality metrics, prints token cost at
both providers' rates, and cleans up its own throwaway ingest_runs so the
project's real "latest" run is restored when this finishes.

THIS SCRIPT SPENDS REAL MONEY -- two full ingest runs per project (one
Anthropic, one Gemini), each a real pass-1 call plus one pass-2 call per
batch. Never run it from an automated pipeline; it is the explicit, manual
merge-gate check the plan calls for, kept on the branch (not a throwaway)
so the next A/B re-run doesn't have to be rebuilt.

Usage:
    .venv/bin/python scripts/pass2_provider_ab.py <project_id> [<project_id> ...]
        [--anthropic-model claude-sonnet-5] [--gemini-model gemini-3.1-flash-lite]

Run it on >= 3 projects of different types (podcast/outlook, a reel, a
b-roll/food reel) before treating the merge gate (plan section 6) as
satisfied -- one project alone is not enough signal.

Not instrumented (disclosed gap, not silently omitted): per-batch re-ask
rate and schema-fallback count aren't persisted anywhere today (a batch's
``Completion.attempts``/fallback path is discarded once its usage is
summed into ``ingest_runs``), so this harness reports run-level success/
failure only, not a per-batch breakdown. Add that counter to
``client.py``/``ingest_gemini.py`` first if the merge gate needs the finer
number.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

import psycopg  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.l3 import cuts_v3_read, ingest, ingest_store as store  # noqa: E402

# Approx USD per MILLION tokens (input / output / cache-read). Only used for
# the printed cost estimate -- raw token counts (in ingest_runs) are the
# ground truth for the actual merge-gate cost comparison; update these if
# list pricing changes.
_RATES: Dict[str, Dict[str, float]] = {
    "anthropic": {"input": 3.0, "output": 15.0, "cache_read": 0.30},
    "gemini": {"input": 0.10, "output": 0.40, "cache_read": 0.025},
}


def _pg_conn():
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _latest_run_id(project_id: str) -> Optional[str]:
    with _pg_conn() as conn:
        row = conn.execute(
            "select id::text from ingest_runs where project_id = %s order by created_at desc limit 1",
            (project_id,),
        ).fetchone()
    return row[0] if row else None


def _run_usage(run_id: str) -> Dict[str, Any]:
    with _pg_conn() as conn:
        row = conn.execute(
            "select input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, status, error "
            "from ingest_runs where id = %s",
            (run_id,),
        ).fetchone()
    if not row:
        return {}
    keys = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "status", "error")
    return dict(zip(keys, row))


def _delete_run(run_id: str) -> None:
    """Tear down a throwaway run: its cut_records first (FK), then the
    ingest_runs row itself -- deleting the row (not just its cuts) is what
    lets the project's real latest run become "latest" again by
    created_at ordering (see cuts_v3_read.load_cuts_v3's own query)."""
    store.delete_cut_records_for_run(run_id)
    with _pg_conn() as conn:
        conn.execute("delete from ingest_runs where id = %s", (run_id,))


def _cost_usd(usage: Dict[str, Any], provider: str) -> float:
    rates = _RATES[provider]
    inp = max(int(usage.get("input_tokens") or 0) - int(usage.get("cache_read_tokens") or 0), 0)
    return (
        inp / 1e6 * rates["input"]
        + int(usage.get("output_tokens") or 0) / 1e6 * rates["output"]
        + int(usage.get("cache_read_tokens") or 0) / 1e6 * rates["cache_read"]
    )


def _metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    by_kind: Dict[str, int] = {}
    channel: Dict[str, int] = {}
    take_roles: Dict[str, int] = {}
    for r in records:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        ch = r.get("channel")
        channel[ch] = channel.get(ch, 0) + 1
        role = r.get("take_role")
        if role:
            take_roles[role] = take_roles.get(role, 0) + 1

    speech = [r for r in records if r["kind"] == "speech"]
    on_camera_true = sum(1 for r in speech if r.get("on_camera") is True)
    on_camera_false = sum(1 for r in speech if r.get("on_camera") is False)

    outlook_groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        if r.get("take_role") == "outlook" and r.get("take_group_id"):
            outlook_groups.setdefault(r["take_group_id"], []).append(r)
    clean_1on1off = 0
    for members in outlook_groups.values():
        oncam = sum(1 for m in members if m.get("on_camera") is True)
        offcam = sum(1 for m in members if m.get("on_camera") is False)
        if len(members) >= 2 and oncam == 1 and offcam == len(members) - 1:
            clean_1on1off += 1

    has_subject_box = sum(1 for r in records if (r.get("framing") or {}).get("subject_box"))
    known_shot_size = sum(
        1 for r in records if (r.get("framing") or {}).get("shot_size") not in (None, "unsure"))
    has_people = sum(1 for r in records if r.get("characteristics"))

    return {
        "total": total,
        "by_kind": by_kind,
        "channel": channel,
        "on_camera_true": on_camera_true,
        "on_camera_false": on_camera_false,
        "outlook_group_count": len(outlook_groups),
        "clean_1on1off": clean_1on1off,
        "pct_subject_box": (has_subject_box / total) if total else 0.0,
        "pct_known_shot_size": (known_shot_size / total) if total else 0.0,
        "pct_with_people": (has_people / total) if total else 0.0,
        "take_roles": take_roles,
    }


def run_ab_for_project(
    project_id: str, *, anthropic_model: str, gemini_model: str,
) -> Dict[str, Any]:
    settings = get_settings()
    orig_provider = settings.ingest_pass2_provider
    orig_model = settings.ingest_pass2_model
    original_latest = _latest_run_id(project_id)
    print(f"[{project_id[:8]}] original latest run: {original_latest}")

    results: Dict[str, Any] = {}
    try:
        for provider in ("anthropic", "gemini"):
            settings.ingest_pass2_provider = provider
            settings.ingest_pass2_model = anthropic_model if provider == "anthropic" else gemini_model
            print(f"[{project_id[:8]}] running ingest -- provider={provider} model={settings.ingest_pass2_model} ...")
            t0 = time.time()
            run_id: Optional[str] = None
            status = "ok"
            error: Optional[str] = None
            try:
                run_id = ingest.run_ingest(project_id)
            except Exception as e:  # noqa: BLE001 -- report every failure mode, never crash the harness
                run_id = _latest_run_id(project_id)
                if run_id == original_latest:
                    run_id = None  # nothing new was created (failed before create_ingest_run)
                status = type(e).__name__
                error = str(e)
            elapsed = time.time() - t0

            usage = _run_usage(run_id) if run_id else {}
            records = cuts_v3_read.rows_for_run(run_id) if (run_id and status == "ok") else []
            metrics = _metrics(records)
            cost = _cost_usd(usage, provider) if usage else 0.0
            results[provider] = {
                "run_id": run_id, "status": status, "error": error, "elapsed_s": elapsed,
                "usage": usage, "metrics": metrics, "cost_usd": cost,
            }
            print(f"[{project_id[:8]}] {provider}: status={status} elapsed={elapsed:.1f}s "
                 f"cuts={metrics['total']} cost=${cost:.4f}")
    finally:
        settings.ingest_pass2_provider = orig_provider
        settings.ingest_pass2_model = orig_model

    # Clean up BOTH throwaway runs -- never leave an experimental run as
    # "latest" (a 0-cut "ready" run can otherwise hijack what the
    # frontend/brain reads next).
    for provider, res in results.items():
        rid = res.get("run_id")
        if rid and rid != original_latest:
            print(f"[{project_id[:8]}] cleaning up {provider} run {rid}")
            _delete_run(rid)

    restored = _latest_run_id(project_id)
    cleanup_ok = restored == original_latest
    print(f"[{project_id[:8]}] latest run restored: {cleanup_ok} "
         f"(expected {original_latest}, got {restored})")

    return {"project_id": project_id, "original_latest": original_latest,
           "cleanup_ok": cleanup_ok, "results": results}


def _print_report(ab: Dict[str, Any]) -> None:
    pid = ab["project_id"]
    a, g = ab["results"].get("anthropic", {}), ab["results"].get("gemini", {})
    am, gm = a.get("metrics", {}), g.get("metrics", {})
    print(f"\n=== {pid} ===")
    print(f"{'':22} {'anthropic':>15} {'gemini':>15}")
    print(f"{'status':22} {a.get('status', '-'):>15} {g.get('status', '-'):>15}")
    print(f"{'total cuts':22} {am.get('total', 0):>15} {gm.get('total', 0):>15}")
    print(f"{'by_kind':22} {str(am.get('by_kind')):>15} {str(gm.get('by_kind')):>15}")
    print(f"{'channel':22} {str(am.get('channel')):>15} {str(gm.get('channel')):>15}")
    a_oncam = f"{am.get('on_camera_true', 0)}/{am.get('on_camera_false', 0)}"
    g_oncam = f"{gm.get('on_camera_true', 0)}/{gm.get('on_camera_false', 0)}"
    print(f"{'on_camera T/F':22} {a_oncam:>15} {g_oncam:>15}")
    print(f"{'outlook groups':22} {am.get('outlook_group_count', 0):>15} {gm.get('outlook_group_count', 0):>15}")
    print(f"{'clean 1-on/1-off':22} {am.get('clean_1on1off', 0):>15} {gm.get('clean_1on1off', 0):>15}")
    print(f"{'% subject_box':22} {am.get('pct_subject_box', 0):>14.0%} {gm.get('pct_subject_box', 0):>14.0%}")
    print(f"{'% known shot_size':22} {am.get('pct_known_shot_size', 0):>14.0%} {gm.get('pct_known_shot_size', 0):>14.0%}")
    print(f"{'% with people':22} {am.get('pct_with_people', 0):>14.0%} {gm.get('pct_with_people', 0):>14.0%}")
    print(f"{'take_roles':22} {str(am.get('take_roles')):>15} {str(gm.get('take_roles')):>15}")
    print(f"{'cost (this run)':22} {'$' + format(a.get('cost_usd', 0), '.4f'):>15} "
         f"{'$' + format(g.get('cost_usd', 0), '.4f'):>15}")
    if am.get("total"):
        delta_pct = (gm.get("total", 0) - am["total"]) / am["total"]
        print(f"{'cut-count delta':22} {'':>15} {delta_pct:>14.1%}  (gate: within +/-10%)")
    if a.get("cost_usd"):
        cheaper_by = a["cost_usd"] / g["cost_usd"] if g.get("cost_usd") else float("inf")
        print(f"{'gemini is cheaper by':22} {'':>15} {cheaper_by:>13.1f}x  (gate: >= ~5x with caching)")
    print(f"cleanup_ok: {ab['cleanup_ok']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project_ids", nargs="+", help="project id(s) to A/B")
    parser.add_argument("--anthropic-model", default="claude-sonnet-5")
    parser.add_argument("--gemini-model", default="gemini-3.1-flash-lite")
    args = parser.parse_args()

    print("THIS RUN SPENDS REAL MONEY -- two ingest runs (anthropic + gemini) "
         f"per project, for {len(args.project_ids)} project(s).")

    all_results = []
    for pid in args.project_ids:
        ab = run_ab_for_project(pid, anthropic_model=args.anthropic_model, gemini_model=args.gemini_model)
        all_results.append(ab)
        _print_report(ab)

    print("\n=== SUMMARY ===")
    for ab in all_results:
        cleanup = "OK" if ab["cleanup_ok"] else "*** CLEANUP FAILED -- CHECK MANUALLY ***"
        print(f"{ab['project_id']}: cleanup={cleanup}")


if __name__ == "__main__":
    main()
