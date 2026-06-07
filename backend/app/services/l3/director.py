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
from app.services.l3 import frame_service
from app.services.l3.composer import ComposeResult, compose, summarize_for_critique
from app.services.l3.critic import critique_edl
from app.services.l3.primitives.loader import load_file_analyses
from app.services.l3.primitives.units import EditUnit, build_units
from app.services.l3.recipes import RecipeContext, SectionPlan
from app.services.l3.recipes.registry import RECIPES
from app.services.l3.router import profile_footage
from app.services.llm import get_llm, text_block, tool_result_block, tool_spec, user_message

logger = logging.getLogger(__name__)

# Bound the work: how many files to pull into a single edit, and how many units
# to show the planner.
MAX_SCOPE_FILES = 16
MAX_CATALOG_UNITS = 180
MAX_CRITIQUE_PASSES = 2


@dataclass
class _Budget:
    """Shared keyframe-image budget for one direct_edit run (planning +
    verification draw from the same pool)."""

    max_images: int
    used: int = 0

    def remaining(self) -> int:
        return max(0, self.max_images - self.used)


# Terminal + read tools for the agentic planner/reviewer. view_frames comes from
# frame_service; these are the rest.
GET_UNIT_DETAILS_TOOL = tool_spec(
    name="get_unit_details",
    description=(
        "Return the full untruncated transcript text and all metadata for the "
        "given unit labels (U0, U7, ...). Use when the catalog summary is not "
        "enough to judge a unit."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "labels": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["labels"],
    },
)

_SECTION_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "style": {"type": "string"},
        "intent": {"type": "string"},
        "target_duration_s": {"type": ["number", "null"]},
        "params": {"type": "object"},
        "units": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["style", "units"],
}

SUBMIT_PLAN_TOOL = tool_spec(
    name="submit_plan",
    description=(
        "Submit your FINAL plan. Calling this ends your planning turn, so only "
        "call it once you are confident. Provide reasoning and 1-4 sections."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "sections": {"type": "array", "items": _SECTION_ITEM_SCHEMA},
        },
        "required": ["sections"],
    },
)

SUBMIT_REVIEW_TOOL = tool_spec(
    name="submit_review",
    description=(
        "Submit your FINAL review of the assembled cut. Calling this ends your "
        "review. ok=true ships the cut; ok=false requests a revision with "
        "specific, actionable guidance."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "ok": {"type": "boolean"},
            "guidance": {"type": "string"},
        },
        "required": ["ok"],
    },
)


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

    settings = get_settings()
    budget = _Budget(max_images=int(settings.editor_perception_max_images))
    perception_log: List[Dict[str, Any]] = []

    # Which shots can actually be shown, and how many frames each has. Powers the
    # "frames=N" catalog annotation and the view_frames pool.
    all_shot_ids = sorted({sid for u in all_units for sid in u.shot_ids})
    try:
        frames_by_shot = frame_service.available_frames(all_shot_ids)
    except Exception:
        logger.exception("available_frames failed; planner runs without frame counts")
        frames_by_shot = {}

    catalog_text, label_to_id = _build_unit_catalog(all_units, analyses, frames_by_shot)
    profile_text = _profile_text(profile, analyses)

    # ---- Pass A: agentic plan (blind editor pulls frames on demand) ----
    emit("planning", 30, f"Planning ({profile.suggested_style})")
    plan, plan_telem = _agentic_plan(
        messages=messages, brief=brief, profile_text=profile_text,
        catalog_text=catalog_text, units=all_units, label_to_id=label_to_id,
        duration_target_s=duration_target_s, budget=budget, emit=emit,
    )
    perception_log.append(plan_telem)
    if not plan.get("sections"):
        # Robustness: fall back to the text-only planner if the agentic loop
        # produced nothing usable (e.g. provider tool-use hiccup).
        logger.warning("agentic planner returned no sections; falling back to text plan")
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

        # LLM critic SECOND: a VISION verification pass -- the editor looks at the
        # actual frames of the assembled clips and approves or asks for a fix.
        emit("reviewing", 72 + attempt * 8, "Looking at the cut")
        llm, verify_telem = _verify_cut(
            brief=brief, result=result, units=all_units, label_to_id=label_to_id,
            budget=budget, emit=emit,
        )
        perception_log.append(verify_telem)
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
        prev_plan = plan
        plan, replan_telem = _agentic_plan(
            messages=messages, brief=brief, profile_text=profile_text,
            catalog_text=catalog_text, units=all_units, label_to_id=label_to_id,
            duration_target_s=duration_target_s, budget=budget, emit=emit,
            prev_plan=prev_plan, critique=combined,
        )
        perception_log.append(replan_telem)
        if not plan.get("sections"):
            plan = _replan(messages, brief, profile_text, catalog_text,
                           duration_target_s, prev_plan, combined)
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
        raw={
            "plan": plan,
            "critiques": critiques,
            "profile": profile_text,
            "perception": {
                "max_images": budget.max_images,
                "images_used": budget.used,
                "calls": perception_log,
            },
        },
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
    frames_by_shot: Optional[Dict[str, List]] = None,
) -> tuple[str, Dict[str, str]]:
    """Chronological, labelled (U0..) catalog + {label -> unit_id} map.

    The blind editor only has this text up front, so we 'gather everything':
    modality, quality, role, framing, motion, duration, transcript, and how many
    keyframes each unit can show via view_frames (frames=N)."""
    frames_by_shot = frames_by_shot or {}
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
        nframes = sum(len(frames_by_shot.get(sid, [])) for sid in u.shot_ids)
        blocks.append(
            f"{label} [{u.modality}] file={u.file_name!r} t={_fmt_tc(u.in_ms)} "
            f"dur={u.duration_ms / 1000:.1f}s q={u.quality:.2f} "
            f"role={u.narrative_role or '-'} framing={u.framing_scale or '-'} "
            f"motion={u.motion:.1f} frames={nframes}"
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
# Agentic perception loop (blind editor pulls frames on demand)
# ---------------------------------------------------------------------------

def _label_index(label: str) -> int:
    body = label[1:] if label[:1] in ("U", "u") else label
    return int(body) if body.isdigit() else 10**9


def _planning_context(units: List[EditUnit], label_to_id: Dict[str, str]):
    """Build the helpers the tool loop needs: unit lookup, target resolver
    (label|shot_id -> shot_ids), and a per-frame caption that re-attaches the
    unit label so the model can map a picture back to its catalog row."""
    unit_by_id = {u.id: u for u in units}
    valid_shot_ids = {sid for u in units for sid in u.shot_ids}

    shot_to_label: Dict[str, str] = {}
    for lbl in sorted(label_to_id, key=_label_index):
        u = unit_by_id.get(label_to_id[lbl])
        if not u:
            continue
        for sid in u.shot_ids:
            shot_to_label.setdefault(sid, lbl)

    def resolve(target: str) -> List[str]:
        t = str(target).strip()
        uid = label_to_id.get(t)
        if uid:
            u = unit_by_id.get(uid)
            return list(u.shot_ids) if u else []
        # A bare unit-looking label that isn't in the catalog -> unresolved.
        if t[:1] in ("U", "u") and t[1:].isdigit():
            return []
        return [t] if t in valid_shot_ids else []

    def caption(ref: "frame_service.FrameRef") -> str:
        lbl = shot_to_label.get(ref.shot_id)
        head = f"unit={lbl} | " if lbl else ""
        return f"[{head}shot={ref.shot_id} | t={_fmt_tc(ref.ts_ms)} | {ref.kind}]"

    return unit_by_id, resolve, caption


def _unit_details_text(labels: List[str], label_to_id, unit_by_id) -> str:
    lines: List[str] = []
    for lbl in labels or []:
        uid = label_to_id.get(str(lbl))
        u = unit_by_id.get(uid) if uid else None
        if not u:
            lines.append(f"{lbl}: (unknown label)")
            continue
        text = (u.text or "").replace("\n", " ").strip()
        lines.append(
            f"{lbl} [{u.modality}] file={u.file_name!r} "
            f"t={_fmt_tc(u.in_ms)}-{_fmt_tc(u.out_ms)} dur={u.duration_ms / 1000:.1f}s "
            f"q={u.quality:.2f} role={u.narrative_role or '-'} "
            f"framing={u.framing_scale or '-'} motion={u.motion:.1f} "
            f"valence={u.valence if u.valence is not None else '-'} "
            f"shots={','.join(u.shot_ids) or '-'}"
            + (f"\n   text: {text!r}" if text else "")
        )
    return "\n".join(lines) or "(no units requested)"


def _prune_old_images(msgs: List[Dict[str, Any]], keep_turns: int) -> None:
    """Strip image blocks from all but the last `keep_turns` view_frames result
    turns, leaving their captions. Bounds accumulating multimodal context (the
    real cost driver) while preserving what the model already reasoned about."""
    if keep_turns < 0:
        return
    image_turn_idxs: List[int] = []
    for i, m in enumerate(msgs):
        content = m.get("content")
        if not isinstance(content, list):
            continue
        has_image = any(
            b.get("type") == "tool_result"
            and any(ib.get("type") == "image" for ib in b.get("content", []))
            for b in content
        )
        if has_image:
            image_turn_idxs.append(i)

    prune_before = image_turn_idxs[:-keep_turns] if keep_turns > 0 else image_turn_idxs
    for i in prune_before:
        for b in msgs[i]["content"]:
            if b.get("type") != "tool_result":
                continue
            kept = []
            dropped = 0
            for ib in b.get("content", []):
                if ib.get("type") == "image":
                    dropped += 1
                else:
                    kept.append(ib)
            if dropped:
                kept.append({"type": "text", "text": f"[{dropped} frame(s) pruned from context]"})
                b["content"] = kept


def _run_view_frames_call(args, resolve, caption, budget, per_shot_default) -> tuple[List[Dict[str, Any]], int, str]:
    targets = args.get("targets") or []
    req = args.get("max_per_shot")
    cap = int(req) if isinstance(req, (int, float)) and req > 0 else per_shot_default
    cap = min(cap, per_shot_default)
    blocks, n, note = frame_service.run_view_frames(
        targets=[str(t) for t in targets],
        resolve_target=resolve,
        caption_for=caption,
        per_shot_max=cap,
        max_total=budget.remaining(),
    )
    budget.used += n
    return blocks, n, note


def _agentic_plan(
    *,
    messages: List[Dict[str, Any]],
    brief: str,
    profile_text: str,
    catalog_text: str,
    units: List[EditUnit],
    label_to_id: Dict[str, str],
    duration_target_s: Optional[int],
    budget: _Budget,
    emit: EmitFn,
    prev_plan: Optional[Dict[str, Any]] = None,
    critique: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Bounded tool loop: the planner gets all text up front and pulls keyframes
    via view_frames until confident, then calls submit_plan. Returns
    (plan_dict, telemetry). Never raises -- returns empty sections on failure so
    the caller can fall back."""
    settings = get_settings()
    telem: Dict[str, Any] = {
        "phase": "replan" if prev_plan else "plan",
        "rounds": 0, "view_frames_calls": 0, "images": 0,
        "tokens_in": 0, "tokens_out": 0,
    }
    try:
        unit_by_id, resolve, caption = _planning_context(units, label_to_id)
        system = prompts_mod.load("editor_director")
        history = (
            claude_editor._render_history(messages[:-1])
            if len(messages) > 1 else "(no prior turns)"
        )
        dur = (
            f"DURATION TARGET: {duration_target_s} seconds"
            if duration_target_s else "DURATION TARGET: (none -- choose a tight length)"
        )
        parts = [
            "CONVERSATION HISTORY (oldest first):", history,
            "FOOTAGE PROFILE:", profile_text,
            "BRIEF:", brief, dur,
        ]
        if prev_plan is not None and critique is not None:
            parts += [
                "YOUR PREVIOUS PLAN:", json.dumps(prev_plan)[:3000],
                "CRITIC FEEDBACK (address this):", json.dumps(critique)[:1500],
            ]
        parts += [
            "UNIT CATALOG (chronological):", catalog_text,
            "Plan the cut. Use view_frames / get_unit_details as needed, then "
            "call submit_plan exactly once.",
        ]
        msgs: List[Dict[str, Any]] = [user_message([text_block("\n\n".join(parts))])]

        llm = get_llm()
        tools = [frame_service.VIEW_FRAMES_TOOL, GET_UNIT_DETAILS_TOOL, SUBMIT_PLAN_TOOL]
        max_rounds = int(settings.editor_perception_max_rounds)
        per_shot_default = int(settings.view_frames_per_shot_max)
        keep_turns = int(settings.editor_perception_keep_image_turns)

        plan: Optional[Dict[str, Any]] = None
        for rnd in range(max_rounds):
            _prune_old_images(msgs, keep_turns)
            resp = llm.run(system=system, messages=msgs, tools=tools,
                           max_tokens=2500, cache_system=True)
            telem["rounds"] += 1
            telem["tokens_in"] += resp.usage.get("input_tokens", 0)
            telem["tokens_out"] += resp.usage.get("output_tokens", 0)
            msgs.append(resp.assistant_message)

            if resp.stop_reason != "tool_use" or not resp.tool_calls:
                if rnd < max_rounds - 1:
                    msgs.append(user_message([text_block(
                        "Call submit_plan with your final sections now.")]))
                    continue
                break

            tool_results: List[Dict[str, Any]] = []
            for tc in resp.tool_calls:
                args = tc.input if isinstance(tc.input, dict) else {}
                if tc.name == "submit_plan":
                    plan = args
                    tool_results.append(tool_result_block(tc.id, "Plan received."))
                elif tc.name == "view_frames":
                    blocks, n, note = _run_view_frames_call(
                        args, resolve, caption, budget, per_shot_default)
                    telem["view_frames_calls"] += 1
                    telem["images"] += n
                    emit("looking", min(34 + rnd * 2, 50),
                         f"Looking: {str(args.get('reason') or 'frames')[:38]}")
                    tool_results.append(tool_result_block(tc.id, [text_block(note)] + blocks))
                elif tc.name == "get_unit_details":
                    txt = _unit_details_text(args.get("labels") or [], label_to_id, unit_by_id)
                    tool_results.append(tool_result_block(tc.id, txt))
                else:
                    tool_results.append(tool_result_block(tc.id, f"Unknown tool {tc.name}."))

            msgs.append(user_message(tool_results))
            if plan is not None:
                break
            if budget.remaining() <= 0:
                msgs.append(user_message([text_block(
                    "Image budget exhausted. Submit your plan now with submit_plan.")]))

        if isinstance(plan, dict) and plan.get("sections"):
            return plan, telem
        return {"reasoning": str((plan or {}).get("reasoning") or ""), "sections": []}, telem
    except Exception as e:
        logger.exception("agentic planner failed")
        telem["error"] = str(e)
        return {"reasoning": f"(agentic planner failed: {e})", "sections": []}, telem


def _verify_cut(
    *,
    brief: str,
    result: ComposeResult,
    units: List[EditUnit],
    label_to_id: Dict[str, str],
    budget: _Budget,
    emit: EmitFn,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Vision verification: the editor looks at the assembled clips' frames and
    approves or requests a revision. Returns ({ok, guidance}, telemetry). Never
    raises -- defaults to ok=true so a flaky review never blocks shipping."""
    settings = get_settings()
    telem: Dict[str, Any] = {
        "phase": "verify", "rounds": 0, "view_frames_calls": 0, "images": 0,
        "tokens_in": 0, "tokens_out": 0,
    }
    try:
        clips = sorted(
            result.edl.get("video_track") or [],
            key=lambda c: int(c.get("timeline_in_ms", 0)),
        )
        if not clips:
            return {"ok": True, "issues": [], "guidance": ""}, telem

        _unit_by_id, resolve, caption = _planning_context(units, label_to_id)
        clip_shot_by_index: Dict[str, str] = {}
        lines: List[str] = []
        for i, c in enumerate(clips):
            sid = str(c.get("shot_id") or "")
            clip_shot_by_index[f"C{i}"] = sid
            why = str(c.get("why") or "").replace("\n", " ")[:80]
            lines.append(
                f"C{i} section={c.get('section') or '-'} role={c.get('role_in_edit') or '-'} "
                f"shot={sid or '-'} t={_fmt_tc(int(c.get('source_in_ms', 0)))} "
                + (f"why={why!r}" if why else "")
            )

        def resolve_review(target: str) -> List[str]:
            t = str(target).strip()
            if t in clip_shot_by_index:
                sid = clip_shot_by_index[t]
                return [sid] if sid else []
            return resolve(t)

        system = prompts_mod.load("editor_verify")
        initial = "\n\n".join([
            "BRIEF:", brief,
            "ASSEMBLED CLIPS (playback order):", "\n".join(lines),
            "Review the cut. view_frames on the clips that matter (targets may be "
            "clip ids like C0, shot ids, or unit labels), then call submit_review.",
        ])
        msgs: List[Dict[str, Any]] = [user_message([text_block(initial)])]

        llm = get_llm()
        tools = [frame_service.VIEW_FRAMES_TOOL, SUBMIT_REVIEW_TOOL]
        max_rounds = max(2, min(int(settings.editor_perception_max_rounds), 6))
        per_shot_default = int(settings.view_frames_per_shot_max)
        keep_turns = int(settings.editor_perception_keep_image_turns)

        review: Optional[Dict[str, Any]] = None
        for rnd in range(max_rounds):
            _prune_old_images(msgs, keep_turns)
            resp = llm.run(system=system, messages=msgs, tools=tools,
                           max_tokens=1200, cache_system=True)
            telem["rounds"] += 1
            telem["tokens_in"] += resp.usage.get("input_tokens", 0)
            telem["tokens_out"] += resp.usage.get("output_tokens", 0)
            msgs.append(resp.assistant_message)

            if resp.stop_reason != "tool_use" or not resp.tool_calls:
                if rnd < max_rounds - 1:
                    msgs.append(user_message([text_block(
                        "Call submit_review now.")]))
                    continue
                break

            tool_results: List[Dict[str, Any]] = []
            for tc in resp.tool_calls:
                args = tc.input if isinstance(tc.input, dict) else {}
                if tc.name == "submit_review":
                    review = {"ok": bool(args.get("ok", True)),
                              "guidance": str(args.get("guidance") or "")}
                    tool_results.append(tool_result_block(tc.id, "Review received."))
                elif tc.name == "view_frames":
                    blocks, n, note = _run_view_frames_call(
                        args, resolve_review, caption, budget, per_shot_default)
                    telem["view_frames_calls"] += 1
                    telem["images"] += n
                    emit("reviewing", 74, f"Reviewing: {str(args.get('reason') or 'frames')[:34]}")
                    tool_results.append(tool_result_block(tc.id, [text_block(note)] + blocks))
                else:
                    tool_results.append(tool_result_block(tc.id, f"Unknown tool {tc.name}."))

            msgs.append(user_message(tool_results))
            if review is not None:
                break

        if isinstance(review, dict):
            return review, telem
        return {"ok": True, "issues": [], "guidance": ""}, telem
    except Exception as e:
        logger.exception("vision verification failed")
        telem["error"] = str(e)
        return {"ok": True, "issues": [], "guidance": f"(verify skipped: {e})"}, telem


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
        raw_params = s.get("params")
        params = dict(raw_params) if isinstance(raw_params, dict) else {}
        out.append(SectionPlan(
            style=style,
            intent=str(s.get("intent") or "")[:200],
            target_duration_s=float(tgt) if isinstance(tgt, (int, float)) else None,
            file_ids=None,          # sections scope by their selected unit_ids
            unit_ids=unit_ids,
            params=params,
        ))
    return out
