#!/usr/bin/env python3
"""
Edit-quality eval harness.

Turns "does this edit feel broken?" into numbers, using the mechanical critic
(l3/critic.py) as a repeatable scorer over assembled EDLs. Two modes:

  --audit            Score EVERY committed edit in the user's library (the latest
                     version per project) against the critic. No LLM calls, no
                     cost -- a quality x-ray of what's already there.

  --run --brief ...  Generate fresh edits for one or more briefs via the director
                     (LLM cost) and score them. Use to A/B a prompt/recipe change.

Reports per-edit defects (issue code + severity) and an aggregate defect-rate
table so you can see, e.g., "40% of edits trip frantic_pacing" before and after a
change.

Examples:
  .venv/bin/python scripts/eval_edits.py --user-id <uuid> --audit
  .venv/bin/python scripts/eval_edits.py --user-id <uuid> --run \
      --brief "a punchy 20s highlight reel" --brief "a calm 30s b-roll mood" \
      --duration 25
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg  # noqa: E402
from psycopg.rows import dict_row  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.services.l3.critic import critique_edl  # noqa: E402
from app.services.l3.primitives.loader import load_file_analyses  # noqa: E402


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


def _section_styles(edl: Dict[str, Any]) -> List[str]:
    return [str(s.get("style") or "") for s in (edl.get("sections") or [])]


def _edl_target_s(edl: Dict[str, Any]) -> Optional[int]:
    # Committed EDLs don't store the brief's target; infer none (duration checks
    # are skipped). Run-mode passes the real target explicitly.
    return None


def _score(label: str, edl: Dict[str, Any], analyses, target_s: Optional[int]) -> Dict[str, Any]:
    res = critique_edl(edl, analyses, target_s=target_s, section_styles=_section_styles(edl))
    print(f"\n=== {label} ===")
    st = res.stats
    print(f"  {st.get('video_clips',0)} clips, {st.get('total_ms',0)/1000:.1f}s, "
          f"avg {st.get('avg_clip_ms',0)/1000:.2f}s "
          f"(shortest {st.get('shortest_clip_ms',0)/1000:.2f}s, longest {st.get('longest_clip_ms',0)/1000:.2f}s)")
    if not res.issues:
        print("  clean -- no mechanical defects")
    for i in res.issues:
        print(f"  [{i.severity}] {i.code}: {i.detail}")
    return {"label": label, "ok": res.ok, "issues": res.issues, "stats": res.stats}


def audit(user_id: str) -> List[Dict[str, Any]]:
    with _pg() as conn:
        rows = conn.execute(
            """
            select p.id, p.name, p.source_file_ids,
                   v.edl_json
              from projects p
              join lateral (
                  select edl_json from edl_versions
                   where project_id = p.id order by created_at desc limit 1
              ) v on true
             where p.user_id = %s
             order by p.updated_at desc
             limit 100
            """,
            (user_id,),
        ).fetchall()
    if not rows:
        print(f"No committed edits for user {user_id}")
        return []

    results: List[Dict[str, Any]] = []
    for r in rows:
        edl = r["edl_json"] or {}
        file_ids = [str(x) for x in (r["source_file_ids"] or [])]
        analyses = load_file_analyses(user_id, file_ids) if file_ids else {}
        results.append(_score(f"{r['name']} ({str(r['id'])[:8]})", edl, analyses, _edl_target_s(edl)))
    return results


def run_briefs(user_id: str, briefs: List[str], file_ids: Optional[List[str]],
               folder_id: Optional[str], duration_s: Optional[int]) -> List[Dict[str, Any]]:
    from app.services.l3 import director

    results: List[Dict[str, Any]] = []
    for brief in briefs:
        print(f"\n>>> generating: {brief!r}")
        dr = director.direct_edit(
            user_id=user_id,
            messages=[{"role": "user", "content": brief}],
            file_ids=file_ids,
            folder_id=folder_id,
            duration_target_s=duration_s,
        )
        if not dr.edl or not (dr.edl.get("video_track") or dr.edl.get("clips")):
            print(f"  (no timeline produced; warnings: {dr.warnings})")
            continue
        # Reload analyses for scoring (director doesn't return them).
        scope = file_ids or []
        if not scope:
            scope = list({str(c.get('file_id')) for c in (dr.edl.get('video_track') or []) if c.get('file_id')})
        analyses = load_file_analyses(user_id, scope) if scope else {}
        results.append(_score(brief, dr.edl, analyses, duration_s))
    return results


def _report(results: List[Dict[str, Any]]) -> None:
    if not results:
        return
    n = len(results)
    clean = sum(1 for r in results if not r["issues"])
    errored = sum(1 for r in results if not r["ok"])
    code_counts: Counter = Counter()
    for r in results:
        for i in r["issues"]:
            code_counts[i.code] += 1

    print("\n" + "=" * 60)
    print("AGGREGATE")
    print("=" * 60)
    print(f"edits scored:        {n}")
    print(f"clean (no defects):  {clean} ({100*clean/n:.0f}%)")
    print(f"with hard errors:    {errored} ({100*errored/n:.0f}%)")
    if code_counts:
        print("\ndefect rate by code (share of edits tripping it):")
        for code, cnt in code_counts.most_common():
            print(f"  {code:20} {cnt:>3}/{n}  ({100*cnt/n:.0f}%)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Edit-quality eval via the mechanical critic")
    ap.add_argument("--user-id", required=True)
    ap.add_argument("--audit", action="store_true", help="score existing committed edits (no LLM)")
    ap.add_argument("--run", action="store_true", help="generate edits for --brief(s) via the director")
    ap.add_argument("--brief", action="append", default=[], help="brief to generate (repeatable; with --run)")
    ap.add_argument("--file-id", action="append", default=[], help="restrict generation to these files")
    ap.add_argument("--folder-id")
    ap.add_argument("--duration", type=int, help="duration target seconds (run mode)")
    args = ap.parse_args()

    if not args.audit and not args.run:
        args.audit = True  # safe default

    results: List[Dict[str, Any]] = []
    if args.audit:
        results += audit(args.user_id)
    if args.run:
        if not args.brief:
            ap.error("--run requires at least one --brief")
        results += run_briefs(args.user_id, args.brief, args.file_id or None,
                              args.folder_id, args.duration)

    _report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
