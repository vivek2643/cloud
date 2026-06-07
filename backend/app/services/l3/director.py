"""
The director: plan -> fill -> critique -> re-plan loop.

This is the new editorial brain. It builds editorial units from L1/L2 data,
profiles the footage, asks the LLM for a SYMBOLIC plan (per-section styles +
unit selections), assembles a precise A/V timeline deterministically via the
composer/recipes, then asks the LLM to critique a text summary and revises a
bounded number of times.

LLM decides WHAT (style, sections, which units). Code decides HOW (cuts snapped
to real boundaries, A/V tracks, stitching).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings
from app.services import prompts as prompts_mod
from app.services.l3 import claude_editor
from app.services.l3.composer import ComposeResult, compose, summarize_for_critique
from app.services.l3.critic import critique_edl
from app.services.l3.primitives.loader import load_file_analyses
from app.services.l3.primitives.units import EditUnit, build_units
from app.services.l3.recipes import RecipeContext, SectionPlan
from app.services.l3.recipes.registry import RECIPES
from app.services.l3.router import profile_footage

logger = logging.getLogger(__name__)

# Bound the work: how many files to pull into a single edit, and how many units
# to show the planner.
MAX_SCOPE_FILES = 16
MAX_CATALOG_UNITS = 180
MAX_CRITIQUE_PASSES = 2


@dataclass
class DirectorResult:
    edl: Dict[str, Any]                       # v2 EDL (may be empty)
    sections: List[Dict[str, Any]] = field(default_factory=list)
    reasoning: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)
    critiques: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    vertical: bool = False
    total_ms: int = 0
    catalog_text: str = ""
    timeline: List[Dict[str, Any]] = field(default_factory=list)  # frontend video-track view
    raw: Dict[str, Any] = field(default_factory=dict)


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


EmitFn = Callable[[str, int, str], None]


def _noop(phase: str, pct: int, label: str) -> None:  # pragma: no cover
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def direct_edit(
    *,
    user_id: str,
    messages: List[Dict[str, Any]],
    file_ids: Optional[List[str]] = None,
    folder_id: Optional[str] = None,
    duration_target_s: Optional[int] = None,
    fps: int = 30,
    emit: EmitFn = _noop,
) -> DirectorResult:
    brief = ""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            brief = str(m.get("content") or "").strip()
            break

    emit("profiling", 12, "Profiling footage")
    scope = _scope_file_ids(user_id, file_ids, folder_id)
    if not scope:
        return DirectorResult(edl={}, warnings=["No indexed footage in scope."])

    analyses = load_file_analyses(user_id, scope)
    if not analyses:
        return DirectorResult(edl={}, warnings=["No analyzable footage in scope."])

    units_by_file: Dict[str, List[EditUnit]] = {fid: build_units(fa) for fid, fa in analyses.items()}
    all_units: List[EditUnit] = [u for us in units_by_file.values() for u in us]
    if not all_units:
        return DirectorResult(edl={}, warnings=["Footage has no usable editorial units yet."])

    profile = profile_footage(analyses, units_by_file)
    ctx = RecipeContext(analyses=analyses, units=all_units, profile=profile)

    catalog_text, label_to_id = _build_unit_catalog(all_units, analyses)
    profile_text = _profile_text(profile, analyses)

    # ---- Pass A: plan ----
    emit("planning", 30, f"Planning ({profile.suggested_style})")
    plan = _plan(messages, brief, profile_text, catalog_text, duration_target_s)
    if not plan.get("sections"):
        return DirectorResult(
            edl={}, plan=plan,
            reasoning=str(plan.get("reasoning") or ""),
            warnings=["Planner returned no sections."],
            catalog_text=catalog_text,
        )

    sections = _sections_from_plan(plan, label_to_id, duration_target_s)
    critiques: List[Dict[str, Any]] = []

    # ---- Pass B/C: fill + critique, bounded re-plan ----
    result: Optional[ComposeResult] = None
    for attempt in range(MAX_CRITIQUE_PASSES + 1):
        emit("assembling", 55 + attempt * 10, f"Assembling timeline (pass {attempt + 1})")
        result = compose(sections, ctx, fps=fps)
        if not (result.edl.get("video_track") or result.edl.get("audio_track")):
            break
        if attempt >= MAX_CRITIQUE_PASSES:
            break
        emit("critiquing", 70 + attempt * 8, "Reviewing the cut")

        # Mechanical critic FIRST: deterministic defects (flicker cuts, blurry/
        # black frames, shot reuse, duration miss, missing music bed). This is
        # the authoritative gate -- repeatable and pixel-grounded.
        styles = [str(s.get("style") or "") for s in result.sections]
        mech = critique_edl(
            result.edl, ctx.analyses,
            target_s=int(duration_target_s) if duration_target_s else None,
            section_styles=styles,
        )
        critiques.append(mech.to_dict())

        # LLM critic SECOND: editorial taste over a text summary of the cut.
        summary = summarize_for_critique(result, ctx)
        llm = _critique(brief, summary)
        critiques.append(llm)

        warn_count = sum(1 for i in mech.issues if i.severity == "warn")
        # Revise on any hard defect, an unhappy taste critic, or a pile-up of
        # softer warnings. The loop is bounded by MAX_CRITIQUE_PASSES regardless.
        should_revise = (not mech.ok) or (warn_count >= 2) or (not llm.get("ok", True))
        if not should_revise:
            break

        guidance = mech.guidance if (not mech.ok or warn_count) else ""
        if not llm.get("ok", True):
            extra = str(llm.get("guidance") or "").strip()
            if extra:
                guidance = (guidance + "\n\nEditorial notes:\n" + extra).strip()
        combined = {"ok": False, "guidance": guidance,
                    "mechanical": mech.to_dict(), "taste": llm}

        emit("revising", 78, "Revising the cut")
        plan = _replan(messages, brief, profile_text, catalog_text, duration_target_s, plan, combined)
        if not plan.get("sections"):
            break
        sections = _sections_from_plan(plan, label_to_id, duration_target_s)

    if result is None:
        return DirectorResult(edl={}, plan=plan, catalog_text=catalog_text,
                              warnings=["Assembly produced nothing."])

    reasoning = str(plan.get("reasoning") or "").strip()
    warnings = list(result.warnings)
    timeline = _frontend_timeline(result.edl, analyses)
    return DirectorResult(
        edl=result.edl,
        sections=result.sections,
        reasoning=reasoning or "(no reasoning)",
        plan=plan,
        critiques=critiques,
        warnings=warnings,
        vertical=result.vertical,
        total_ms=result.total_ms,
        catalog_text=catalog_text,
        timeline=timeline,
        raw={"plan": plan, "critiques": critiques, "profile": profile_text},
    )


def _frontend_timeline(edl: Dict[str, Any], analyses) -> List[Dict[str, Any]]:
    """Flatten the v2 video track into the clip shape the existing dock reads."""
    out: List[Dict[str, Any]] = []
    for c in sorted(edl.get("video_track") or [], key=lambda x: x["timeline_in_ms"]):
        fa = analyses.get(str(c.get("file_id")))
        out.append({
            "file_id": c.get("file_id"),
            "file_name": fa.name if fa else "",
            "source_in_ms": int(c["source_in_ms"]),
            "source_out_ms": int(c["source_out_ms"]),
            "timeline_start_ms": int(c["timeline_in_ms"]),
            "timeline_end_ms": int(c["timeline_out_ms"]),
            "score": 1.0,
            "shot_id": c.get("shot_id"),
            "role_in_edit": c.get("role_in_edit"),
            "why": c.get("why"),
            "section": c.get("section"),
        })
    return out


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

def _scope_file_ids(user_id: str, file_ids: Optional[List[str]], folder_id: Optional[str]) -> List[str]:
    if file_ids:
        return [str(f) for f in dict.fromkeys(file_ids) if f]
    sql = "select id from files where user_id = %s and l1_status = 'ready'"
    params: List[Any] = [user_id]
    if folder_id:
        sql += " and folder_id = %s"
        params.append(folder_id)
    sql += " order by created_at desc limit %s"
    params.append(MAX_SCOPE_FILES)
    with _pg() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [str(r["id"]) for r in rows]


# ---------------------------------------------------------------------------
# Catalog + profile rendering
# ---------------------------------------------------------------------------

def _fmt_tc(ms: int) -> str:
    s = max(0, int(ms)) // 1000
    return f"{s // 60}:{s % 60:02d}"


def _build_unit_catalog(
    units: List[EditUnit],
    analyses,
) -> tuple[str, Dict[str, str]]:
    """Chronological, labelled (U0..) catalog + {label -> unit_id} map."""
    ordered = sorted(units, key=lambda u: (u.file_name or "", u.in_ms))
    if len(ordered) > MAX_CATALOG_UNITS:
        # Keep the highest-quality units but preserve chronological display.
        keep = sorted(ordered, key=lambda u: u.quality, reverse=True)[:MAX_CATALOG_UNITS]
        keep_ids = {u.id for u in keep}
        ordered = [u for u in ordered if u.id in keep_ids]

    label_to_id: Dict[str, str] = {}
    blocks: List[str] = []
    for i, u in enumerate(ordered):
        label = f"U{i}"
        label_to_id[label] = u.id
        text = (u.text or "").replace("\n", " ").strip()
        if len(text) > 160:
            text = text[:157] + "..."
        blocks.append(
            f"{label} [{u.modality}] file={u.file_name!r} t={_fmt_tc(u.in_ms)} "
            f"dur={u.duration_ms / 1000:.1f}s q={u.quality:.2f} "
            f"role={u.narrative_role or '-'}"
            + (f"\n   text: {text!r}" if text else "")
        )
    return "\n".join(blocks), label_to_id


def _profile_text(profile, analyses) -> str:
    lines = [f"Corpus dominant modality: {profile.dominant_modality}; "
             f"suggested style: {profile.suggested_style}; "
             f"music available: {profile.has_musical}"]
    for fid, fp in profile.per_file.items():
        lines.append(
            f"- {fp.file_name!r}: {fp.dominant_modality} "
            f"(speech {fp.speech_fraction:.0%}, motion {fp.mean_motion:.1f}, "
            f"beats={'yes' if fp.has_beats else 'no'})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM passes
# ---------------------------------------------------------------------------

def _plan(messages, brief, profile_text, catalog_text, duration_target_s) -> Dict[str, Any]:
    system = prompts_mod.load("editor_plan")
    history = claude_editor._render_history(messages[:-1]) if len(messages) > 1 else "(no prior turns)"
    dur = f"DURATION TARGET: {duration_target_s} seconds" if duration_target_s else "DURATION TARGET: (none -- choose a tight length)"
    user = "\n\n".join([
        "CONVERSATION HISTORY (oldest first):", history,
        "FOOTAGE PROFILE:", profile_text,
        "BRIEF:", brief, dur,
        "UNIT CATALOG (chronological):", catalog_text,
        "Return the plan JSON now.",
    ])
    try:
        return claude_editor._call_claude(system, user, max_tokens=2000)
    except Exception as e:
        logger.exception("plan pass failed")
        return {"reasoning": f"(planner failed: {e})", "sections": []}


def _replan(messages, brief, profile_text, catalog_text, duration_target_s, prev_plan, critique) -> Dict[str, Any]:
    system = prompts_mod.load("editor_plan")
    dur = f"DURATION TARGET: {duration_target_s} seconds" if duration_target_s else "DURATION TARGET: (none)"
    user = "\n\n".join([
        "FOOTAGE PROFILE:", profile_text,
        "BRIEF:", brief, dur,
        "YOUR PREVIOUS PLAN:", json.dumps(prev_plan)[:3000],
        "CRITIC FEEDBACK (address this):", json.dumps(critique)[:1500],
        "UNIT CATALOG (chronological):", catalog_text,
        "Return a REVISED plan JSON that fixes the critic's points.",
    ])
    try:
        return claude_editor._call_claude(system, user, max_tokens=2000)
    except Exception as e:
        logger.exception("replan pass failed")
        return prev_plan


def _critique(brief: str, summary: str) -> Dict[str, Any]:
    system = prompts_mod.load("editor_critic")
    user = "\n\n".join(["BRIEF:", brief, "ASSEMBLED CUT SUMMARY:", summary, "Return the critique JSON now."])
    try:
        out = claude_editor._call_claude(system, user, max_tokens=800)
        if not isinstance(out, dict):
            return {"ok": True, "issues": [], "guidance": ""}
        return out
    except Exception as e:
        logger.exception("critique pass failed")
        return {"ok": True, "issues": [], "guidance": f"(critique skipped: {e})"}


# ---------------------------------------------------------------------------
# Plan -> SectionPlan
# ---------------------------------------------------------------------------

def _sections_from_plan(plan, label_to_id, duration_target_s) -> List[SectionPlan]:
    out: List[SectionPlan] = []
    for s in plan.get("sections") or []:
        if not isinstance(s, dict):
            continue
        style = str(s.get("style") or "").strip()
        if style not in RECIPES:
            style = "vlog"
        labels = s.get("units") or []
        unit_ids = [label_to_id[str(l)] for l in labels if str(l) in label_to_id]
        tgt = s.get("target_duration_s")
        out.append(SectionPlan(
            style=style,
            intent=str(s.get("intent") or "")[:200],
            target_duration_s=float(tgt) if isinstance(tgt, (int, float)) else None,
            file_ids=None,          # sections scope by their selected unit_ids
            unit_ids=unit_ids,
        ))
    return out
