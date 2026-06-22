"""
L3 v2 -- prompt-driven auto-editor (deterministic 3-call pipeline).

The agentic L3 (``orchestrator.py``) is a many-turn Claude tool loop. This is a
much simpler, opinionated alternative that turns a one-line brief into the SAME
Edit Document in three bounded LLM calls, then resolves it deterministically so
the existing preview / timeline / render read it unchanged:

  1. DIRECTOR  -- reads per-clip summaries, guesses the energy dial, the
                  delivery aspect, a rough target length, and a beat outline.
  2. (engine)  -- build_hero_cuts(file_ids, energy) -> the full cut feed at that
                  energy. The cut LABELS are the transcript at that granularity,
                  so the Editor sees the real selectable content, not a summary.
  3. EDITOR    -- selects + ORDERS the spine cuts (the rough cut) from the feed.
  4. COVERAGE  -- (when the spine frees video) lays B-roll / reaction / insert
                  overlays over spine ranges, and trims to the target length.

The model only ever emits hero_ids, an energy float, and program ranges -- never
source timestamps. Every id is validated against the feed (hallucinations are
dropped). Sharp-band speech cuts carry a breath-removal edit-list (``keep_spans``);
each kept span becomes its own back-to-back timeline segment, so the jump-cuts
survive into the render.

Provider/model route through the neutral ``get_llm`` factory (defaults to OpenAI
via ``autoedit_*`` settings), and ``make_edit`` takes an injectable ``llm`` so it
is testable with a fake client and trivial to switch models.

Fails OPEN: any call failure degrades to a deterministic fallback (top cuts in
score order at the guessed energy), so a thread always lands a watchable draft.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from procrastinate import RetryStrategy

from app.config import get_settings
from app.services.jobs import app
from app.services.l3 import framing, hero_store, layers
from app.services.l3.energy import energy_band
from app.services.llm import LLMClient, get_llm, user_message

logger = logging.getLogger(__name__)

# Which modalities form the SPINE (the A-roll the cut is built on) vs. COVERAGE
# (overlay material laid on top). Speech/action/performance carry the story;
# b-roll/reaction/insert decorate it.
_SPINE_MODALITIES = ("speech", "action", "performance")
_COVERAGE_MODALITIES = ("broll", "reaction", "insert")

# Coverage overlays are only legal when the spine frees its picture.
_VIDEO_FREE_SPINES = ("dialogue", "music")

_ASPECTS = ("landscape", "portrait", "square")
_SPINE_KINDS = ("dialogue", "music", "visual", "sync", "other")

# Per-call reasoning effort. Only the Editor's selection + ordering benefits from
# deep reasoning; the Director (energy/aspect guess) and Coverage (overlay
# placement) are shallow calls, so they run cheap/fast. The Editor falls through
# to settings.autoedit_effort (default "high").
_DIRECTOR_EFFORT = "low"
_COVERAGE_EFFORT = "low"


# --------------------------------------------------------------------------
# Result shapes
# --------------------------------------------------------------------------

@dataclass
class Plan:
    """The Director's read of the brief."""
    energy: float = 0.5
    aspect: str = "landscape"
    spine_kind: str = "dialogue"
    target_duration_ms: Optional[int] = None
    intent: str = ""
    beats: List[Dict[str, str]] = field(default_factory=list)
    rationale: str = ""


@dataclass
class AutoEditResult:
    document: dict
    plan: Plan
    feed_size: int
    selected: int


# --------------------------------------------------------------------------
# Loading per-clip context
# --------------------------------------------------------------------------

def _pg_conn() -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, autocommit=True)


def _as_doc(v) -> Optional[dict]:
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return None
    return None


def _clip_cards(file_ids: List[str]) -> Dict[str, dict]:
    """Per-clip context cards for the Director: the editorially-useful summary
    fields from L2 perception + duration. Keyed by file_id, order-stable."""
    if not file_ids:
        return {}
    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select f.id::text, f.name, coalesce(f.duration_seconds, 0), cp.perception
              from files f
              left join clip_perception cp on cp.file_id = f.id
             where f.id = any(%s::uuid[])
            """,
            (file_ids,),
        ).fetchall()

    cards: Dict[str, dict] = {}
    for fid, name, dur_s, perception in rows:
        p = _as_doc(perception) or {}
        edit = p.get("editability") or {}
        setting = p.get("setting") or {}
        look = p.get("look") or {}
        persons = p.get("persons") or []
        roles = [pp.get("role") for pp in persons if pp.get("role")]
        cards[fid] = {
            "file_id": fid,
            "name": name or fid,
            "duration_ms": int(float(dur_s) * 1000),
            "content_type": p.get("content_type"),
            "primary_axis": edit.get("primary_axis"),
            "cut_sensitivity": edit.get("cut_sensitivity"),
            "best_use": edit.get("best_use") or [],
            "logline": p.get("logline"),
            "synopsis": p.get("synopsis"),
            "topics": p.get("topics") or [],
            "location": setting.get("location"),
            "mood": look.get("mood"),
            "people": roles,
            "notes": p.get("notes"),
        }
    # Preserve the caller's order so prompts are deterministic.
    return {fid: cards[fid] for fid in file_ids if fid in cards}


# --------------------------------------------------------------------------
# JSON helpers
# --------------------------------------------------------------------------

def _parse_json(text: str) -> Optional[dict]:
    """Best-effort JSON object parse (handles fenced / prose-wrapped output)."""
    if not text:
        return None
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _run_json(llm: LLMClient, system: str, prompt: str,
              effort: Optional[str] = None) -> Optional[dict]:
    settings = get_settings()
    resp = llm.run(
        system=system,
        messages=[user_message(prompt)],
        max_tokens=settings.autoedit_max_output_tokens,
        effort=effort or (settings.autoedit_effort or None),
    )
    return _parse_json(resp.text)


# --------------------------------------------------------------------------
# Call 1 -- DIRECTOR
# --------------------------------------------------------------------------

_DIRECTOR_SYSTEM = (
    "You are the DIRECTOR of an automated video editor. You read short text "
    "summaries of every source clip in a project and a one-line brief, then make "
    "the high-level creative calls that frame the whole edit. You cannot watch the "
    "footage; work only from the summaries.\n\n"
    "Decide:\n"
    "1. ENERGY (0.0-1.0): the single pacing dial. It sets how the clips are cut.\n"
    "   - 0.0-0.2 Broad: long, whole-answer blocks; calm, contemplative, cinematic.\n"
    "   - 0.3-0.4 Calm: multi-point clusters; relaxed.\n"
    "   - 0.5     Balanced: one complete thought per cut; standard.\n"
    "   - 0.6-0.8 Tight: per-sentence cuts; punchy, social.\n"
    "   - 0.9-1.0 Sharp: per-sentence with dead air removed; fast, hype, viral.\n"
    "   Pick from the brief's TONE (a 'calm recap' is low; a 'punchy hype reel' is high).\n"
    "2. ASPECT: 'portrait' (9:16) for any reel/short/tiktok/story/vertical brief, "
    "'square' (1:1) if asked, else 'landscape' (16:9).\n"
    "3. SPINE_KIND: what carries the edit -- 'dialogue' (interview/talking-head/VO; "
    "the words lead, picture is free for b-roll), 'music' (a music bed leads), "
    "'visual' (a demo/performance/product where the picture is the point), or 'sync' "
    "(A and V locked together).\n"
    "4. TARGET_DURATION_MS: if the brief implies a length ('30 second', 'under a "
    "minute'), the target in ms; else null.\n"
    "5. INTENT: one sentence on the story/throughline you'll build.\n"
    "6. BEATS: 2-6 ordered beats, each {\"purpose\": short, \"intent\": one line}. "
    "Hook first when the brief implies an audience.\n\n"
    "Return ONLY JSON of this exact shape (no prose):\n"
    '{"energy": 0.6, "aspect": "portrait", "spine_kind": "dialogue", '
    '"target_duration_ms": 30000, "intent": "...", '
    '"beats": [{"purpose": "hook", "intent": "..."}], "rationale": "..."}'
)


def _director_prompt(brief: str, cards: Dict[str, dict]) -> str:
    lines: List[str] = [f"BRIEF: {brief.strip() or '(none given -- infer a sensible edit)'}", ""]
    lines.append(f"PROJECT: {len(cards)} clip(s).")
    for c in cards.values():
        dur_s = round(c["duration_ms"] / 1000.0, 1)
        bits = [f'CLIP "{c["name"]}" ({dur_s}s)']
        meta = []
        if c.get("content_type"):
            meta.append(f"type: {c['content_type']}")
        if c.get("primary_axis"):
            meta.append(f"axis: {c['primary_axis']}")
        if c.get("cut_sensitivity"):
            meta.append(f"cut_sensitivity: {c['cut_sensitivity']}")
        if meta:
            bits.append("  " + " | ".join(meta))
        if c.get("logline"):
            bits.append(f"  logline: {c['logline']}")
        if c.get("synopsis"):
            bits.append(f"  synopsis: {c['synopsis']}")
        extras = []
        if c.get("topics"):
            extras.append("topics: " + ", ".join(map(str, c["topics"][:6])))
        if c.get("best_use"):
            extras.append("best_use: " + ", ".join(map(str, c["best_use"][:5])))
        if extras:
            bits.append("  " + " | ".join(extras))
        ctx = []
        if c.get("location"):
            ctx.append(f"setting: {c['location']}")
        if c.get("people"):
            ctx.append("people: " + ", ".join(map(str, c["people"][:4])))
        if c.get("mood"):
            ctx.append(f"mood: {c['mood']}")
        if ctx:
            bits.append("  " + " | ".join(ctx))
        if c.get("notes"):
            bits.append(f"  notes: {c['notes']}")
        lines.append("\n".join(bits))
    return "\n".join(lines)


def _coerce_plan(doc: Optional[dict]) -> Plan:
    plan = Plan()
    if not doc:
        return plan
    try:
        plan.energy = _clamp01(float(doc.get("energy", 0.5)))
    except (TypeError, ValueError):
        plan.energy = 0.5
    a = str(doc.get("aspect") or "").lower()
    plan.aspect = a if a in _ASPECTS else "landscape"
    s = str(doc.get("spine_kind") or "").lower()
    plan.spine_kind = s if s in _SPINE_KINDS else "dialogue"
    tdm = doc.get("target_duration_ms")
    try:
        plan.target_duration_ms = int(tdm) if tdm not in (None, "", 0) else None
    except (TypeError, ValueError):
        plan.target_duration_ms = None
    plan.intent = str(doc.get("intent") or "").strip()
    plan.rationale = str(doc.get("rationale") or "").strip()
    beats = []
    for b in doc.get("beats") or []:
        if isinstance(b, dict) and (b.get("purpose") or b.get("intent")):
            beats.append({"purpose": str(b.get("purpose") or "").strip(),
                          "intent": str(b.get("intent") or "").strip()})
    plan.beats = beats
    return plan


def _director(brief: str, cards: Dict[str, dict], llm: LLMClient) -> Plan:
    doc = _run_json(llm, _DIRECTOR_SYSTEM, _director_prompt(brief, cards),
                    effort=_DIRECTOR_EFFORT)
    return _coerce_plan(doc)


# --------------------------------------------------------------------------
# Call 3 -- EDITOR (select + order the spine)
# --------------------------------------------------------------------------

_EDITOR_SYSTEM = (
    "You are the EDITOR. You are given a brief, the director's intent + beat "
    "outline, and the FULL pool of usable cuts already extracted from the footage "
    "(each cut's text is the actual spoken line or a description of the moment). "
    "Select the cuts that tell the story and put them in the FINAL ORDER they "
    "should play -- this is the rough cut's spine.\n\n"
    "Rules:\n"
    "- Refer to cuts ONLY by their bracketed [hero_id].\n"
    "- Order for narrative: hook first when there's an audience; build; end strong.\n"
    "- Cut filler, false starts, redundant restatements, and anything off-brief.\n"
    "- Never include two cuts that say the same thing (keep the stronger).\n"
    "- Respect the target length if given (sum of play_ms ~ target); fewer, "
    "stronger cuts beat a long flabby list.\n"
    "- You may map each pick to one of the director's beats by its purpose.\n\n"
    "Return ONLY JSON (no prose):\n"
    '{"picks": [{"hero_id": "<id>", "beat": "<beat purpose or null>", '
    '"reason": "<short why>"}]}'
)


def _cut_row(c: dict) -> str:
    secs = round(c.get("play_ms", c.get("duration_ms", 0)) / 1000.0, 1)
    spk = c.get("speaker") or "-"
    label = (c.get("label") or "").replace("\n", " ").strip()
    return (f'[{c["hero_id"]}] {c["modality"]} · {spk} · {secs}s · '
            f'score {round(float(c.get("score", 0)), 2)} · "{label}"')


def _editor_prompt(brief: str, plan: Plan, spine_pool: List[dict]) -> str:
    lines = [f"BRIEF: {brief.strip() or '(none)'}",
             f"INTENT: {plan.intent or '(none)'}",
             f"ENERGY: {plan.energy:.2f}  ASPECT: {plan.aspect}  SPINE: {plan.spine_kind}"]
    if plan.target_duration_ms:
        lines.append(f"TARGET LENGTH: {round(plan.target_duration_ms / 1000.0, 1)}s")
    if plan.beats:
        lines.append("BEATS: " + "; ".join(
            f'{b["purpose"]} ({b["intent"]})' for b in plan.beats))
    lines.append("")
    lines.append(f"CUT POOL ({len(spine_pool)} cuts) -- pick + order by [hero_id]:")
    cur = None
    for c in spine_pool:
        if c.get("file_id") != cur:
            cur = c.get("file_id")
        lines.append(_cut_row(c))
    return "\n".join(lines)


def _editor(brief: str, plan: Plan, spine_pool: List[dict],
            llm: LLMClient) -> List[dict]:
    """Return ordered picks: [{hero_id, beat, reason}], validated + de-duped."""
    by_id = {c["hero_id"]: c for c in spine_pool}
    doc = _run_json(llm, _EDITOR_SYSTEM, _editor_prompt(brief, plan, spine_pool))
    picks: List[dict] = []
    seen = set()
    for p in (doc or {}).get("picks", []) or []:
        if not isinstance(p, dict):
            continue
        hid = str(p.get("hero_id") or "").strip()
        if hid in by_id and hid not in seen:
            seen.add(hid)
            picks.append({
                "hero_id": hid,
                "beat": (str(p.get("beat")).strip() if p.get("beat") else None),
                "reason": str(p.get("reason") or "").strip(),
            })
    return picks


# --------------------------------------------------------------------------
# Call 4 -- COVERAGE / CRITIQUE (overlays + fit)
# --------------------------------------------------------------------------

_COVERAGE_SYSTEM = (
    "You are the COVERAGE editor. The spine (A-roll) is locked and laid on a "
    "program clock (times in ms from 0). You are given the spine segments with "
    "their program ranges and a pool of overlay cuts (b-roll / reactions / "
    "inserts). Lay overlays over spine ranges where they ADD value -- illustrate a "
    "point, cover a jump, punctuate a reaction -- and trim the cut to length.\n\n"
    "Rules:\n"
    "- Refer to overlay cuts by [hero_id] and spine segments by their seg_id.\n"
    "- An overlay covers a program range [from_ms, to_ms]; keep it shorter than "
    "the cut's own length and inside the spine's total length.\n"
    "- Do NOT cover the hook or moments better seen than illustrated. Be sparing; "
    "no wall-to-wall b-roll.\n"
    "- If a target length is given and the spine overshoots, list seg_ids to DROP "
    "(weakest / most redundant first).\n\n"
    "Return ONLY JSON (no prose):\n"
    '{"overlays": [{"hero_id": "<id>", "from_ms": 0, "to_ms": 0, "reason": "..."}], '
    '"trims": [{"seg_id": "<id>"}], "notes": "..."}'
)


def _coverage_prompt(brief: str, plan: Plan, spine_view: List[dict],
                     coverage_pool: List[dict], total_ms: int) -> str:
    lines = [f"BRIEF: {brief.strip() or '(none)'}",
             f"SPINE LENGTH: {round(total_ms / 1000.0, 1)}s"]
    if plan.target_duration_ms:
        lines.append(f"TARGET LENGTH: {round(plan.target_duration_ms / 1000.0, 1)}s")
    lines.append("")
    lines.append("SPINE SEGMENTS (seg_id · program range · text):")
    for s in spine_view:
        label = (s.get("content") or "").replace("\n", " ").strip()
        lines.append(f'{s["seg_id"]} · {s["prog_start_ms"]}-{s["prog_end_ms"]}ms · "{label}"')
    lines.append("")
    lines.append(f"OVERLAY POOL ({len(coverage_pool)} cuts):")
    for c in coverage_pool:
        lines.append(_cut_row(c))
    return "\n".join(lines)


def _coverage(brief: str, plan: Plan, spine_view: List[dict],
              coverage_pool: List[dict], total_ms: int, llm: LLMClient) -> dict:
    doc = _run_json(llm, _COVERAGE_SYSTEM,
                    _coverage_prompt(brief, plan, spine_view, coverage_pool, total_ms),
                    effort=_COVERAGE_EFFORT)
    return doc or {}


# --------------------------------------------------------------------------
# Resolve picks -> document
# --------------------------------------------------------------------------

def _segments_from_picks(picks: List[dict], by_id: Dict[str, dict]) -> List[dict]:
    """Ordered picks -> timeline segments. A cut with a breath-removal edit-list
    (``keep_spans``) expands into one back-to-back segment per kept span, so the
    jump-cuts survive into the render."""
    segments: List[dict] = []
    for i, p in enumerate(picks):
        c = by_id[p["hero_id"]]
        spans = c.get("keep_spans") or [
            {"in_ms": c["src_in_ms"], "out_ms": c["src_out_ms"]}]
        for j, sp in enumerate(spans):
            in_ms, out_ms = int(sp["in_ms"]), int(sp["out_ms"])
            if out_ms <= in_ms:
                continue
            segments.append({
                "seg_id": f"a{i:03d}_{j}",
                "file_id": c["file_id"],
                "in_ms": in_ms,
                "out_ms": out_ms,
                "axis": "speech" if c["modality"] == "speech" else "any",
                "beat_id": p.get("beat"),
                "content": c.get("label"),
                "rationale": p.get("reason"),
                "priority": 3,
                "cut_in_cost": 0.0,
                "cut_out_cost": 0.0,
                "warnings": [],
                "hero_id": p["hero_id"],
            })
    return segments


def _apply_trims(segments: List[dict], coverage: dict,
                 plan: Plan) -> List[dict]:
    """Drop segments the coverage pass flagged; deterministic fallback trims the
    lowest-score tail until within ~10% of the target when still overshooting."""
    drop = {str(t.get("seg_id")) for t in (coverage.get("trims") or [])
            if isinstance(t, dict) and t.get("seg_id")}
    kept = [s for s in segments if s["seg_id"] not in drop]
    if not plan.target_duration_ms or not kept:
        return kept
    target = plan.target_duration_ms
    total = sum(s["out_ms"] - s["in_ms"] for s in kept)
    # Deterministic safety net: if still > ~110% target, drop the least-protected
    # cuts first (priority is 1=most protected .. higher=more droppable).
    while kept and total > target * 1.1 and len(kept) > 1:
        worst = max(kept, key=lambda s: s.get("priority", 3))
        kept.remove(worst)
        total -= (worst["out_ms"] - worst["in_ms"])
    return kept


def _operations_from_coverage(coverage: dict, by_id: Dict[str, dict],
                              total_ms: int) -> List[dict]:
    ops: List[dict] = []
    for o in (coverage.get("overlays") or []):
        if not isinstance(o, dict):
            continue
        c = by_id.get(str(o.get("hero_id") or "").strip())
        if c is None:
            continue
        from_ms = max(0, min(int(o.get("from_ms", 0)), total_ms))
        to_ms = max(0, min(int(o.get("to_ms", 0)), total_ms))
        if to_ms - from_ms < 200:
            continue
        cut_len = c["src_out_ms"] - c["src_in_ms"]
        span = min(to_ms - from_ms, cut_len)
        ops.append({
            "op_id": f"cov_{uuid.uuid4().hex[:6]}",
            "type": "place_video",
            "source_file_id": c["file_id"],
            "src_in_ms": int(c["src_in_ms"]),
            "src_out_ms": int(c["src_in_ms"] + span),
            "from_ms": from_ms,
            "to_ms": from_ms + span,
            "layout": layers.DEFAULT_LAYOUT,
            "z": layers.Z_COVERAGE,
            "opacity": 1.0,
            "rationale": str(o.get("reason") or "").strip() or None,
            "warnings": [],
        })
    return ops


def _spine_regions(plan: Plan, file_ids: List[str]) -> List[dict]:
    locked = (["audio"] if plan.spine_kind in ("dialogue", "music")
              else (["video", "audio"] if plan.spine_kind == "sync" else ["video"]))
    return [{
        "kind": plan.spine_kind,
        "label": plan.intent or None,
        "locked_channels": locked,
        "source_file_ids": file_ids,
        "rationale": plan.rationale or None,
    }]


def _build_document(brief: str, plan: Plan, segments: List[dict],
                    operations: List[dict], file_ids: List[str],
                    summary: str, notes: List[str]) -> dict:
    document = {
        "brief": {
            "goal": brief.strip() or None,
            "aspect": plan.aspect,
            "target_duration_s": (round(plan.target_duration_ms / 1000.0, 1)
                                  if plan.target_duration_ms else None),
            "assumptions": [plan.rationale] if plan.rationale else [],
        },
        "format": {"aspect": plan.aspect},
        "spine": {"regions": _spine_regions(plan, file_ids)},
        "outline": [
            {"beat_id": f"b{i}", "purpose": b["purpose"], "intent": b["intent"]}
            for i, b in enumerate(plan.beats)
        ],
        "timeline": segments,
        "operations": operations,
        "open_questions": [],
        "summary": summary,
        "notes": notes,
        "diagnostics": {
            "engine": "auto_edit_v1",
            "energy": plan.energy,
            "band": energy_band(plan.energy),
            "spine_kind": plan.spine_kind,
        },
    }
    # Bake the automatic reframe transform so the preview and render agree, then
    # resolve the flat layer set the player/render read.
    try:
        framing.annotate_document(document)
    except Exception:
        logger.exception("auto_edit: framing annotation failed (continuing)")
    durations = {fid: 0 for fid in file_ids}
    try:
        from app.services.render.tasks import _durations as _render_durations
        durations = _render_durations(file_ids)
    except Exception:
        logger.exception("auto_edit: duration lookup failed (continuing)")
    document["resolved"] = layers.resolve(document, durations).to_dict()
    return document


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def _fallback_picks(spine_pool: List[dict], plan: Plan) -> List[dict]:
    """Deterministic draft when the Editor call fails: top cuts by score, in
    chronological order, capped to the target (or a sane default)."""
    target = plan.target_duration_ms or 60000
    ranked = sorted(spine_pool, key=lambda c: float(c.get("score", 0)), reverse=True)
    chosen, total = [], 0
    for c in ranked:
        chosen.append(c)
        total += c.get("play_ms", c.get("duration_ms", 0))
        if total >= target:
            break
    chosen.sort(key=lambda c: (c.get("file_id"), c.get("src_in_ms", 0)))
    return [{"hero_id": c["hero_id"], "beat": None,
             "reason": "auto (fallback: top score)"} for c in chosen]


def make_edit(file_ids: List[str], brief: str,
              *, llm: Optional[LLMClient] = None) -> AutoEditResult:
    """The full pipeline: brief + clips -> resolved Edit Document.

    ``llm`` is injectable for testing / model-swapping; defaults to the
    configured auto-edit provider (OpenAI)."""
    settings = get_settings()
    if llm is None:
        llm = get_llm(provider=settings.autoedit_provider or None,
                      model=settings.autoedit_model or None)

    cards = _clip_cards(file_ids)

    # Call 1: Director (energy + plan) from the clip summary cards alone -- the
    # cards' content_type / primary_axis / best_use already distinguish a
    # talking-head from an action reel, so we don't pay for a throwaway feed
    # build just to count modalities.
    plan = _director(brief, cards, llm)

    # Engine: the full cut feed at the director's energy, served from the
    # precompute cache (snaps to the nearest band; lazily backfills a miss).
    feed = hero_store.get_hero_feed(file_ids, energy=plan.energy)
    by_id = {c["hero_id"]: c for c in feed}
    spine_pool = [c for c in feed if c.get("modality") in _SPINE_MODALITIES]
    coverage_pool = [c for c in feed if c.get("modality") in _COVERAGE_MODALITIES]

    # Call 3: Editor (select + order the spine).
    picks: List[dict] = []
    try:
        picks = _editor(brief, plan, spine_pool, llm)
    except Exception:
        logger.exception("auto_edit: editor call failed; using fallback picks")
    if not picks:
        picks = _fallback_picks(spine_pool, plan)

    segments = _segments_from_picks(picks, by_id)

    # Call 4: Coverage (overlays + fit), only when the spine frees its picture.
    coverage: dict = {}
    if plan.spine_kind in _VIDEO_FREE_SPINES and coverage_pool and segments:
        spans, total = layers.spine_spans(segments)
        spine_view = [{
            "seg_id": s.seg["seg_id"],
            "prog_start_ms": s.prog_start_ms,
            "prog_end_ms": s.prog_end_ms,
            "content": s.seg.get("content"),
        } for s in spans]
        try:
            coverage = _coverage(brief, plan, spine_view, coverage_pool, total, llm)
        except Exception:
            logger.exception("auto_edit: coverage call failed (continuing)")

    segments = _apply_trims(segments, coverage, plan)
    _, total_ms = layers.spine_spans(segments)
    operations = _operations_from_coverage(coverage, by_id, total_ms)

    notes = []
    if isinstance(coverage.get("notes"), str) and coverage["notes"].strip():
        notes.append(coverage["notes"].strip())
    summary = plan.intent or "Auto-assembled edit."

    document = _build_document(brief, plan, segments, operations,
                               file_ids, summary, notes)
    return AutoEditResult(document=document, plan=plan,
                          feed_size=len(feed), selected=len(segments))


# --------------------------------------------------------------------------
# Thread run path (mirrors orchestrator.py; writes the same Edit Document)
# --------------------------------------------------------------------------

def run_thread(thread_id: str) -> None:
    """Run the auto-editor for a thread end-to-end and persist the document.
    Single-shot (no agent loop): a thread reaches `ready` (or `failed`)."""
    from app.services.l3 import store

    thread = store.get_thread(thread_id)
    if thread is None:
        logger.info("auto_edit: thread %s gone; skipping.", thread_id)
        return
    store.set_thread_status(thread_id, "drafting")

    try:
        result = make_edit(thread["file_ids"], thread.get("brief") or "")
    except Exception:
        logger.exception("auto_edit: make_edit failed for %s", thread_id)
        store.set_thread_status(thread_id, "failed")
        return

    version = store.save_document(thread_id, result.document, created_by="auto")
    store.set_thread_status(thread_id, "ready")
    logger.info(
        "auto_edit: thread %s ready (v%d, energy %.2f, %d/%d cuts, %d ops)",
        thread_id, version, result.plan.energy, result.selected,
        result.feed_size, len(result.document.get("operations", [])),
    )


@app.task(name="l3_auto_edit_turn", queue="l3",
          retry=RetryStrategy(max_attempts=2, exponential_wait=5))
def l3_auto_edit_turn(thread_id: str) -> None:
    try:
        run_thread(thread_id)
    except Exception:
        logger.exception("auto_edit run failed for thread %s", thread_id)
        try:
            from app.services.l3 import store
            store.set_thread_status(thread_id, "failed")
        except Exception:
            pass
        raise


def _defer_run(thread_id: str) -> None:
    """Enqueue on the existing `l3` queue (workers already serve it)."""
    from procrastinate import App, PsycopgConnector

    enqueue_app = App(connector=PsycopgConnector(
        conninfo=get_settings().database_url, min_size=1, max_size=2))
    with enqueue_app.open():
        enqueue_app.configure_task("l3_auto_edit_turn", queue="l3").defer(thread_id=thread_id)


def start_thread(user_id: str, file_ids: List[str], brief: str) -> str:
    """Create an auto-edit thread, seed the brief turn, enqueue the pipeline."""
    from app.services.l3 import store

    thread_id = store.create_thread(user_id, file_ids, brief)
    store.append_turn(thread_id, "user", brief or "Auto-edit the best cut from these clips.")
    _defer_run(thread_id)
    return thread_id
