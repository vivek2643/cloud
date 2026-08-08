"""
Cuts, Part 3 middle text layer (cut_structure_and_scene_specificity.plan.md):
one TEXT-only, no-frames, capable-Gemini call that turns Pass A's generic
per-cut summaries into (a) a guess at the project's DOMAIN, (b) a closed-set
TAXONOMY only where one genuinely exists, and (c) targeted, footage-derived
QUESTIONS clustered cuts share. Mechanism = question generation, not
classification: you cannot enumerate a correct taxonomy from generic
summaries, but you CAN write the right question ("what part/product, what
operation?"). `needs_pass_b` then TRIAGES which cuts get the costlier
targeted vision pass -- a cost reducer, not just a router; a cut whose
summary is already specific enough is simply skipped.

Runs from the background `l3_scene_enrich` task (ingest.py), strictly AFTER
cuts are shown to the user -- never on the ingest critical path. No re-ask
loop of its own beyond what `complete_gemini` already does (one re-ask on a
schema violation) -- this is a single, low-stakes call.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.l3 import cuts_read, pass1
from app.services.llm.base import text_block
from app.services.llm.ingest_gemini import complete_gemini

logger = logging.getLogger(__name__)

# Keep the prompt small regardless of project size: a long transcript is
# truncated (the model needs domain SIGNAL here, not verbatim recall -- Pass
# B re-reads the actual frames + this cluster's own questions later).
_MAX_TRANSCRIPT_CHARS_PER_FILE = 4000
# Hard ceiling on distinct label/summary groups sent to the model -- the
# single largest groups win (they're what actually keeps a 200-cut reel
# cheap); a long tail of genuinely one-off cuts still gets represented up to
# the cap.
_MAX_GROUPS = 120


class TaxonomyEntry(BaseModel):
    id: str
    definition: str = ""


class SceneClusterOut(BaseModel):
    """The model's own output shape -- references groups by their PROMPT-
    LOCAL id (g0, g1, ...); ``build_scene_taxonomy`` expands these back to
    real ``cut_records`` ids before returning, so nothing downstream ever
    needs to know the dedup happened."""
    group_refs: List[str] = Field(default_factory=list)
    questions: List[str] = Field(default_factory=list)


class SceneTaxonomyOut(BaseModel):
    domain: str = "unknown/mixed"
    confidence: str = "low"
    evidence: List[str] = Field(default_factory=list)
    taxonomy: List[TaxonomyEntry] = Field(default_factory=list)
    clusters: List[SceneClusterOut] = Field(default_factory=list)
    needs_pass_b_groups: List[str] = Field(default_factory=list)


_SYSTEM = (
    "You are looking at the DISTILLED per-cut descriptions from a first, "
    "generic vision pass over one video project's footage -- not the pixels "
    "themselves. Each cut's label/summary already names whatever concrete, "
    "observable specifics (objects, on-screen text, part numbers, the "
    "literal physical action) the first pass could see, even where it could "
    "not interpret them. Your job is NOT to classify these into a taxonomy "
    "from the generic text alone -- generic text cannot support that. "
    "Instead:\n\n"
    "1. Infer the project's DOMAIN from the cumulative evidence (repeated "
    "objects/actions/on-screen text/transcript vocabulary) -- a short "
    "phrase (e.g. 'Indian wedding (Gujarati)', 'CNC machine shop', 'cooking "
    "tutorial'), or 'unknown/mixed' when the evidence genuinely doesn't "
    "converge. Set confidence honestly (high/med/low) and list the concrete "
    "evidence that grounds it.\n\n"
    "2. ONLY when a genuinely closed, well-known answer set exists for this "
    "domain (e.g. named wedding rituals, named sports plays/positions, "
    "named manufacturing operations) -- write it as `taxonomy`: a short "
    "list of {id, definition}, always including an 'other' and an 'unsure' "
    "entry. Leave `taxonomy` EMPTY when no such closed set genuinely "
    "exists -- never invent categories to fill it.\n\n"
    "3. For each group of cuts (or several groups together) that would "
    "benefit from a closer look, write a `cluster`: which group ids it "
    "covers (`group_refs`) and 1-3 SHARP, FOOTAGE-DERIVED questions a "
    "second vision pass should answer by looking at the actual frames "
    "again -- questions grounded in what THIS footage's own hooks suggest, "
    "never generic ('what is happening here?' is useless; 'what part is "
    "being machined and what operation is being performed on it?' or "
    "'which ritual step is this, and who is performing it?' is the kind of "
    "question that gets a specific answer). A cluster with no useful "
    "question to ask (the summary is already about as specific as it can "
    "get) should simply be omitted.\n\n"
    "4. List `needs_pass_b_groups`: the group ids that should actually get "
    "the second vision pass -- this TRIAGES cost, so be selective: skip a "
    "group whose summary is already specific and unambiguous, and skip a "
    "group with no real closed-set/domain question to answer. Only include "
    "groups where a targeted look would plausibly sharpen the description.\n\n"
    "If the evidence is too mixed/sparse for a domain to emerge at all, set "
    "domain='unknown/mixed', leave taxonomy empty, and either omit clusters "
    "entirely or write GENERIC clusters using a plain moment rubric "
    "(establishing/action/reaction/detail/dialogue/transition) instead of a "
    "domain-specific one -- never force a wrong domain onto ambiguous "
    "footage."
)


def _normalize_key(label: str, summary: str) -> str:
    key = f"{label} {summary}".strip().lower()
    return re.sub(r"\s+", " ", key)


def _group_cuts(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Dedupe near-identical cuts by normalized (label, summary) text -- a
    factory reel of 200 "machine on a line" cuts collapses to a handful of
    groups before the prompt is even built. Returns {group_id: [member
    rows]}, largest groups first, capped at _MAX_GROUPS."""
    by_key: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for r in rows:
        key = _normalize_key(r.get("label") or "", r.get("summary") or "")
        if key not in by_key:
            by_key[key] = []
            order.append(key)
        by_key[key].append(r)
    order.sort(key=lambda k: len(by_key[k]), reverse=True)
    return {f"g{i}": by_key[key] for i, key in enumerate(order[:_MAX_GROUPS])}


def _fetch_transcripts(file_ids: List[str]) -> Dict[str, str]:
    if not file_ids:
        return {}
    with pass1._pg_conn() as conn:
        rows = conn.execute(
            "select file_id::text, text from transcripts where file_id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()
    return {fid: (text or "") for fid, text in rows}


def _build_prompt_blocks(groups: Dict[str, List[Dict[str, Any]]], transcripts: Dict[str, str]) -> List[Dict[str, Any]]:
    lines = ["DISTINCT CUT GROUPS (deduped by near-identical label/summary; "
             "count = how many actual cuts share it):"]
    for gid, members in groups.items():
        rep = members[0]
        people = ", ".join(
            p.get("description", "") for p in (rep.get("characteristics") or []) if isinstance(p, dict)
        ) or "none"
        screen = (rep.get("screen_text") or "").strip()
        lines.append(
            f"- {gid} (x{len(members)}, channel={rep.get('channel')}): "
            f"label={rep.get('label')!r} summary={rep.get('summary')!r} "
            f"people=[{people}] screen_text={screen!r}"
        )
    transcript_section = ""
    parts = []
    for fid, text in transcripts.items():
        snippet = (text or "").strip()
        if len(snippet) > _MAX_TRANSCRIPT_CHARS_PER_FILE:
            snippet = snippet[:_MAX_TRANSCRIPT_CHARS_PER_FILE] + " ...[truncated]"
        if snippet:
            parts.append(f"[{fid[:8]}] {snippet}")
    if parts:
        transcript_section = "\n\nTRANSCRIPT EXCERPTS:\n" + "\n\n".join(parts)
    return [text_block("\n".join(lines) + transcript_section)]


def build_scene_taxonomy(ingest_run_id: str) -> Dict[str, Any]:
    """Load every cut this run produced, dedupe/cluster them, and run the
    middle text layer. Returns the FINAL persisted shape: {"domain",
    "confidence", "evidence", "taxonomy", "clusters": [{"cut_refs": [cut_id,
    ...], "questions": [...]}], "needs_pass_b": [cut_id, ...]} -- group ids
    already expanded back to real cut_records ids, so nothing downstream
    needs to know the dedup happened. The unknown/mixed default (everything
    else empty) when the run has no cuts at all."""
    rows = cuts_read.rows_for_run(ingest_run_id)
    if not rows:
        empty = SceneTaxonomyOut()
        return {"domain": empty.domain, "confidence": empty.confidence, "evidence": [],
                "taxonomy": [], "clusters": [], "needs_pass_b": []}

    groups = _group_cuts(rows)
    file_ids = sorted({r["file_id"] for r in rows})
    transcripts = _fetch_transcripts(file_ids)
    blocks = _build_prompt_blocks(groups, transcripts)

    settings = get_settings()
    completion = complete_gemini(
        _SYSTEM, blocks, SceneTaxonomyOut,
        model=settings.ingest_scene_text_model, thinking="low",
    )
    parsed = SceneTaxonomyOut.model_validate(completion.data)

    def _expand(gids: List[str]) -> List[str]:
        seen: List[str] = []
        for gid in gids:
            for row in groups.get(gid, []):
                cid = row["id"]
                if cid not in seen:
                    seen.append(cid)
        return seen

    clusters = [
        {"cut_refs": _expand(c.group_refs), "questions": list(c.questions)}
        for c in parsed.clusters if c.group_refs and c.questions
    ]
    clusters = [c for c in clusters if c["cut_refs"]]
    logger.info("scene_taxonomy: run %s domain=%r confidence=%s groups=%d clusters=%d needs_pass_b_groups=%d",
               ingest_run_id, parsed.domain, parsed.confidence, len(groups), len(clusters),
               len(parsed.needs_pass_b_groups))
    return {
        "domain": parsed.domain,
        "confidence": parsed.confidence,
        "evidence": list(parsed.evidence),
        "taxonomy": [{"id": t.id, "definition": t.definition} for t in parsed.taxonomy],
        "clusters": clusters,
        "needs_pass_b": _expand(parsed.needs_pass_b_groups),
    }
