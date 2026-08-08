"""
Cuts, Part 3 Pass B (cut_structure_and_scene_specificity.plan.md): targeted
vision over a taxonomy cluster's own frames + the middle layer's own
questions for it -- answers them with a SHORT, SPECIFIC one-line description
per cut ("pheras -- couple circling the fire", "facing a steel shaft on a
CNC lathe"), plus the closed-set label when one applies. Never re-describes
a cut from scratch; only resolves what the generic Pass A summary left
ambiguous.

One frame per cut (its already-computed ``hero_ts_ms`` -- the same still
already chosen as that cut's best representative), re-extracted fresh from
the R2 proxy (never stored). One call per taxonomy cluster (its member cuts'
frames + its own shared questions), cached (short-TTL, deleted on
completion) so a cluster with several questions doesn't re-send the same
frames per question.
"""
from __future__ import annotations

import logging
import tempfile
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.l3 import cuts_read, pass1
from app.services.l3.frames import ProxyCache, extract_for_planned_frames
from app.services.l3.image_plan import PlannedFrame
from app.services.llm.base import image_block, text_block
from app.services.llm.ingest_gemini import complete_gemini, create_pass2_cache, delete_pass2_cache

logger = logging.getLogger(__name__)

REASON_SCENE_SPECIFICITY = "scene_specificity"
# A cluster call answers a handful of cuts against 1-3 questions -- small on
# purpose (Part 3's own cost-reducer is triage via needs_pass_b, not
# shrinking any one call further).
_MAX_TOKENS = 4096


class CutSpecific(BaseModel):
    cut_id: str
    specific: str = ""
    label: str = ""    # closed-set taxonomy id, "other", or "" when N/A


class ClusterAnswer(BaseModel):
    cuts: List[CutSpecific] = Field(default_factory=list)


_SYSTEM = (
    "You are looking at specific frames from a video project whose domain "
    "and a set of targeted questions have already been worked out by an "
    "earlier pass. For EACH cut listed below, answer the questions using "
    "ONLY what you can see in its own frame(s), and write ONE short, "
    "SPECIFIC line (a few words) naming exactly what it shows -- not a "
    "re-description of the whole cut, just the sharp specific the "
    "questions were written to resolve (e.g. 'pheras -- couple circling "
    "the fire', 'facing a steel shaft on a CNC lathe'). When a closed-set "
    "taxonomy is given below and one of its entries applies, also set "
    "`label` to that entry's id (or 'other' if none fit); leave `label` "
    "empty when no closed set was given. If the frame genuinely does not "
    "resolve the question for a cut, write your best still-CONCRETE guess "
    "rather than a generic restatement -- never just repeat the question "
    "back, and never invent a specific the pixels don't support. Emit "
    "exactly one entry per cut_id you were shown."
)


def _hero_planned_frames(rows: List[Dict[str, Any]]) -> List[PlannedFrame]:
    return [
        PlannedFrame(file_id=r["file_id"], ts_ms=int(r.get("hero_ts_ms") or 0),
                    reason=REASON_SCENE_SPECIFICITY, ref=r["id"])
        for r in rows
    ]


def _proxy_keys_for_files(file_ids: List[str]) -> Dict[str, str]:
    if not file_ids:
        return {}
    with pass1._pg_conn() as conn:
        rows = conn.execute(
            "select id::text, r2_proxy_key from files where id = any(%s::uuid[])", (file_ids,),
        ).fetchall()
    return {fid: key for fid, key in rows if key}


def _cluster_blocks(
    cluster_rows: List[Dict[str, Any]], questions: List[str],
    images_b64: Dict[Any, str], taxonomy_ids: List[str],
) -> List[Dict[str, Any]]:
    lines = ["QUESTIONS to answer for every cut below:"]
    for q in questions:
        lines.append(f"- {q}")
    if taxonomy_ids:
        lines.append("\nCLOSED-SET LABELS (use `label` when one applies, else 'other'): "
                     + ", ".join(taxonomy_ids))
    lines.append("\nCUTS:")
    blocks: List[Dict[str, Any]] = [text_block("\n".join(lines))]
    for r in cluster_rows:
        b64 = images_b64.get((r["file_id"], int(r.get("hero_ts_ms") or 0)))
        cut_line = f"cut_id={r['id']} label={r.get('label')!r} summary={r.get('summary')!r}"
        blocks.append(text_block(cut_line))
        if b64:
            blocks.append(image_block(b64))
        else:
            blocks.append(text_block("(no frame resolved for this cut -- answer from the summary above)"))
    return blocks


def run_pass_b(ingest_run_id: str, taxonomy: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Run Pass B over every ``taxonomy["clusters"]`` entry, restricted to
    cuts also present in ``taxonomy["needs_pass_b"]`` (a cluster may cover
    MORE cuts than actually need the targeted pass -- some are already
    specific enough; needs_pass_b is the triage). Returns {cut_id:
    {"specific", "label"}} for every cut Pass B actually answered -- exactly
    the shape persisted onto that cut's own ``scene_specifics`` column.
    Never raises for a single cluster's failure (logged, skipped) -- a
    partial enrichment is still a strict improvement over none."""
    needs = set(taxonomy.get("needs_pass_b") or [])
    clusters = taxonomy.get("clusters") or []
    if not needs or not clusters:
        return {}

    all_cut_ids = sorted({cid for c in clusters for cid in (c.get("cut_refs") or []) if cid in needs})
    if not all_cut_ids:
        return {}
    rows_by_id = {r["id"]: r for r in cuts_read.rows_for_run(ingest_run_id) if r["id"] in all_cut_ids}
    if not rows_by_id:
        return {}

    file_ids = sorted({r["file_id"] for r in rows_by_id.values()})
    proxy_keys = _proxy_keys_for_files(file_ids)
    planned = _hero_planned_frames(list(rows_by_id.values()))

    settings = get_settings()
    taxonomy_ids = [t.get("id") for t in (taxonomy.get("taxonomy") or []) if t.get("id")]
    domain = taxonomy.get("domain") or "unknown/mixed"
    gist_blocks = [text_block(
        f"PROJECT DOMAIN: {domain} (confidence={taxonomy.get('confidence', 'low')})\n"
        + ("EVIDENCE: " + "; ".join(taxonomy.get("evidence") or []) + "\n" if taxonomy.get("evidence") else "")
    )]
    cache_name: Optional[str] = None
    results: Dict[str, Dict[str, str]] = {}

    with tempfile.TemporaryDirectory(prefix="edso_scene_b_") as tmp:
        cache = ProxyCache(tmp, proxy_keys)
        try:
            images_b64 = extract_for_planned_frames(planned, proxy_keys, cache=cache)
            cache_name = create_pass2_cache(
                _SYSTEM, gist_blocks, model=settings.ingest_pass_b_model,
                ttl_seconds=settings.ingest_scene_cache_ttl_seconds,
            )
            for cluster in clusters:
                cut_ids = [cid for cid in (cluster.get("cut_refs") or []) if cid in needs and cid in rows_by_id]
                questions = cluster.get("questions") or []
                if not cut_ids or not questions:
                    continue
                cluster_rows = [rows_by_id[cid] for cid in cut_ids]
                blocks = _cluster_blocks(cluster_rows, questions, images_b64, taxonomy_ids)
                try:
                    completion = complete_gemini(
                        _SYSTEM, blocks, ClusterAnswer,
                        max_tokens=_MAX_TOKENS, model=settings.ingest_pass_b_model,
                        thinking=settings.ingest_pass_b_thinking,
                        cached_content=cache_name,
                    )
                    answer = ClusterAnswer.model_validate(completion.data)
                except Exception:
                    logger.exception("scene_specificity: cluster (%d cuts) failed -- skipping",
                                    len(cut_ids))
                    continue
                for c in answer.cuts:
                    if c.cut_id in rows_by_id and (c.specific or "").strip():
                        results[c.cut_id] = {"specific": c.specific.strip(), "label": c.label or ""}
        finally:
            delete_pass2_cache(cache_name)

    logger.info("scene_specificity: run %s answered %d/%d triaged cut(s)",
               ingest_run_id, len(results), len(all_cut_ids))
    return results
