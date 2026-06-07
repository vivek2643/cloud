"""
The smart-edit brain: hand the candidate catalog to Claude and let it
reason about which clips to pick, in what order, and how to trim them.

This replaces the deterministic SQL-filter + score-sort + duration-fill
pipeline in `edit_logic_basic` for the default ("smart") path. The old
fast path is still available as opt-in for cases where the user wants
"top-K, no thinking" speed.

Pipeline:
  1. Load full transcripts for every file represented in the candidates
     so we can hand Claude exact spoken text per shot.
  2. Pull L2 metadata (narrative_role, valence, framing, audio_tags)
     for the candidates from Postgres -- run_query / retrieve_top_k
     don't include these by default; we want them in the catalog.
  3. Format a compact catalog (one block per shot).
  4. Call Claude (Sonnet 4.5 today) with:
       - the editor prompt (prompts/editor.md)
       - the user's brief, duration target, and the catalog
  5. Validate Claude's JSON response:
       - shot_id must exist in catalog
       - source_in/out must be inside the shot's bounds
       - clip length >= 500 ms
       - sort by playback order
  6. Return a structured EditorResult with reasoning, validated timeline,
     and the raw response (for the audit log).
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
from app.services.l3 import keyframe_select as kf_select
from app.services.l3 import vision as vision_mod
from app.services.l3.anthropic_client import _client as _anthropic_client
from app.services.l3.edit_logic_basic import TimelineClip
from app.services.l3.query_executor import CandidateShot

logger = logging.getLogger(__name__)

# Per-clip minimum length below which we drop a clip post-validation.
MIN_CLIP_MS = 500
# How many catalog shots to send to Claude. The retriever already narrows
# down (chronologically, bucketed when over budget); we cap again here so we
# never blow the context window even on extreme cases.
MAX_CATALOG_SHOTS = 140


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

@dataclass
class EditorResult:
    reasoning: str
    timeline: List[TimelineClip]
    post_processing: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    raw_response: Dict[str, Any] = field(default_factory=dict)
    catalog_text: str = ""
    user_text: str = ""


# ---------------------------------------------------------------------------
# DB helpers (single-trip metadata + transcript fetch)
# ---------------------------------------------------------------------------

def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


def _fetch_shot_metadata(shot_ids: List[str]) -> Dict[str, dict]:
    """Pull L2 fields for the candidate shots in a single query."""
    if not shot_ids:
        return {}
    with _pg() as conn:
        cur = conn.execute(
            """
            select s.id, s.framing_scale, s.camera_dynamics,
                   s.narrative_role, s.emotional_valence, s.narrative_description
              from shots s
             where s.id = any(%s::uuid[])
            """,
            (shot_ids,),
        )
        return {str(r["id"]): r for r in cur.fetchall()}


def _fetch_audio_tags(file_ids: List[str]) -> Dict[str, List[str]]:
    if not file_ids:
        return {}
    with _pg() as conn:
        cur = conn.execute(
            "select file_id, acoustic_tags from audio_features where file_id = any(%s::uuid[])",
            (file_ids,),
        )
        return {str(r["file_id"]): list(r["acoustic_tags"] or []) for r in cur.fetchall()}


def _fetch_transcript_segments(file_ids: List[str]) -> Dict[str, list]:
    """Per-file transcript segments. Used to slice exact spoken text per shot."""
    if not file_ids:
        return {}
    with _pg() as conn:
        cur = conn.execute(
            "select file_id, segments, text from transcripts where file_id = any(%s::uuid[])",
            (file_ids,),
        )
        return {str(r["file_id"]): {"segments": r["segments"] or [], "text": r["text"] or ""} for r in cur.fetchall()}


def _slice_transcript(segments: list, start_ms: int, end_ms: int) -> str:
    parts: List[str] = []
    for seg in segments:
        if seg.get("end_ms", 0) < start_ms or seg.get("start_ms", 0) > end_ms:
            continue
        t = (seg.get("text") or "").strip()
        if t:
            parts.append(t)
    return " ".join(parts).strip()


# ---------------------------------------------------------------------------
# Catalog formatter
# ---------------------------------------------------------------------------

def _fmt_optional(v, prec: int = 2) -> str:
    if v is None:
        return "null"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def _fmt_tc(ms: int) -> str:
    """Source timecode as m:ss so Claude can reason about where in each file a
    shot sits without doing millisecond math."""
    total_s = max(0, int(ms)) // 1000
    return f"{total_s // 60}:{total_s % 60:02d}"


def _build_catalog(
    candidates: List[CandidateShot],
) -> tuple[str, Dict[str, CandidateShot]]:
    """
    Render the catalog Claude will read, and return a {shot_id -> candidate}
    map for downstream validation. Truncated to MAX_CATALOG_SHOTS.

    Shots are presented in CHRONOLOGICAL order (by file, then in-file time) so
    the editor reads the footage as a story rather than a similarity-ranked
    bag. Each block is labelled with its source file and source timecode.
    """
    cands = sorted(candidates, key=lambda c: (c.file_name or "", c.start_ms))[:MAX_CATALOG_SHOTS]
    cand_by_id: Dict[str, CandidateShot] = {c.shot_id: c for c in cands}

    meta = _fetch_shot_metadata([c.shot_id for c in cands])
    tx_by_file = _fetch_transcript_segments(list({c.file_id for c in cands}))
    audio_by_file = _fetch_audio_tags(list({c.file_id for c in cands}))

    blocks: List[str] = []
    for i, c in enumerate(cands):
        m = meta.get(c.shot_id) or {}
        tx_for_file = tx_by_file.get(c.file_id, {"segments": [], "text": ""})
        spoken = _slice_transcript(tx_for_file["segments"], c.start_ms, c.end_ms)
        audio_tags = audio_by_file.get(c.file_id) or []

        visual = (m.get("narrative_description") or "").strip().replace("\n", " ")
        # Trim very long descriptions; Claude has them but keep tokens reasonable.
        if len(visual) > 300:
            visual = visual[:297] + "..."

        spoken_repr = json.dumps(spoken) if spoken else '""'
        audio_repr = ", ".join(audio_tags) if audio_tags else "null"
        block = (
            f"SHOT {i}  id={c.shot_id}  file={c.file_name!r}  "
            f"t={_fmt_tc(c.start_ms)}  start={c.start_ms}ms  end={c.end_ms}ms  "
            f"duration={(c.end_ms - c.start_ms) / 1000:.1f}s\n"
            f"  visual:    {visual or '(no description)'}\n"
            f"  framing:   {_fmt_optional(m.get('framing_scale'))}          "
            f"camera: {_fmt_optional(m.get('camera_dynamics'))}\n"
            f"  role:      {_fmt_optional(m.get('narrative_role'))}   "
            f"valence: {_fmt_optional(m.get('emotional_valence'))}\n"
            f"  blur_min:  {_fmt_optional(c.blur_min)}                   "
            f"intra_var: {_fmt_optional(c.intra_shot_variance, 4)}\n"
            f"  audio:     {audio_repr}\n"
            f"  transcript: {spoken_repr}"
        )
        blocks.append(block)

    catalog_text = "\n\n".join(blocks)

    # Add full file-level transcripts at top so Claude has narrative context
    transcripts_section_parts: List[str] = []
    seen_files: set[str] = set()
    for c in cands:
        if c.file_id in seen_files:
            continue
        seen_files.add(c.file_id)
        full = (tx_by_file.get(c.file_id) or {}).get("text", "").strip()
        if full:
            transcripts_section_parts.append(
                f"--- Full transcript of {c.file_name} ---\n{full}"
            )
    full_transcripts_text = "\n\n".join(transcripts_section_parts)

    if full_transcripts_text:
        catalog_text = f"{full_transcripts_text}\n\n========================================\nCATALOG\n========================================\n\n{catalog_text}"

    return catalog_text, cand_by_id


# ---------------------------------------------------------------------------
# Claude call + validation
# ---------------------------------------------------------------------------

def _build_vision(
    candidates: List[CandidateShot],
    brief: str,
) -> tuple[List[Dict[str, Any]], str]:
    """Layer C: pick keyframes for this brief and turn them into Anthropic image
    blocks + a preamble. Never raises -- returns ([], "") to fall back to a
    text-only edit. Honors editor_vision_max_images (0 disables)."""
    settings = get_settings()
    budget = int(getattr(settings, "editor_vision_max_images", 0) or 0)
    if budget <= 0 or not candidates:
        return [], ""
    try:
        frames = kf_select.select_frames_for_edit(
            shot_ids=[c.shot_id for c in candidates],
            brief=brief,
            budget=budget,
            per_shot_max=int(getattr(settings, "editor_vision_per_shot_max", 1) or 1),
        )
        blocks = vision_mod.build_image_blocks(frames)
        n = len([b for b in blocks if b.get("type") == "image"])
        if n == 0:
            return [], ""
        return blocks, vision_mod.vision_preamble(n)
    except Exception:
        logger.exception("Vision attach failed; editor will run text-only")
        return [], ""


def _build_user_message(
    brief: str,
    duration_target_s: Optional[int],
    catalog_text: str,
) -> str:
    parts: List[str] = [
        f"BRIEF:\n{brief.strip()}",
    ]
    if duration_target_s is not None:
        parts.append(f"DURATION TARGET: {duration_target_s} seconds")
    else:
        parts.append("DURATION TARGET: (none -- you choose a tight defensible length)")
    parts.append(catalog_text)
    parts.append(
        "Respond with the JSON object specified in your instructions. JSON only, no other text."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Pass A: planning (beat sheet) -- decide the SHAPE before picking clips
# ---------------------------------------------------------------------------

_PLAN_SYSTEM = (
    "You are the lead editor planning a cut BEFORE touching the timeline. "
    "You are given the user's brief and a chronological catalog of available "
    "shots (with file, source timecode, visual description, and spoken text). "
    "Do NOT pick exact clips yet. Instead write a short BEAT SHEET: the shape "
    "of the edit as 3-7 beats. For each beat give:\n"
    "  - a one-line intent (what this beat accomplishes for the viewer),\n"
    "  - which footage region(s) it should draw from (reference files / rough "
    "timecodes, e.g. \"clip2 around 0:30-0:50\"),\n"
    "  - the emotional or narrative function (hook / build / reveal / payoff / "
    "breather / outro).\n"
    "Respect the brief literally first, then add editorial taste. Keep the "
    "footage in a coherent order (usually chronological within a file) unless "
    "the brief calls for a non-linear structure. Be concise: plain text, no "
    "JSON, no clip ids. This plan will guide the clip-selection pass."
)


def _plan_beats(
    history: List[Dict[str, Any]],
    catalog_text: str,
    duration_target_s: Optional[int],
) -> str:
    """Pass A: ask Claude for a short beat sheet. Returns plain text (best
    effort) -- never raises; on failure we return "" and the fill pass runs
    without an explicit plan."""
    latest = ""
    for m in reversed(history):
        if m.get("role") == "user":
            latest = str(m.get("content") or "").strip()
            break
    if not latest:
        return ""
    dur = (
        f"DURATION TARGET: {duration_target_s} seconds"
        if duration_target_s is not None
        else "DURATION TARGET: (none -- choose a tight, defensible length)"
    )
    user_message = "\n\n".join([f"BRIEF:\n{latest}", dur, catalog_text,
                                "Write the beat sheet now."])
    try:
        settings = get_settings()
        client = _anthropic_client()
        msg = client.messages.create(
            model=settings.anthropic_model,
            max_tokens=900,
            system=_PLAN_SYSTEM,
            messages=[{"role": "user", "content": user_message}],
        )
        return "".join(
            b.text for b in msg.content if getattr(b, "type", None) == "text"
        ).strip()
    except Exception:
        logger.exception("Plan pass failed; continuing without a beat sheet")
        return ""


def _call_claude(
    system_prompt: str,
    user_message: str,
    max_tokens: int = 4096,
    image_blocks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    settings = get_settings()
    client = _anthropic_client()
    if image_blocks:
        # Text first (the catalog/brief), then the labeled keyframe images.
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_message}]
        content.extend(image_blocks)
    else:
        content = user_message  # type: ignore[assignment]
    msg = client.messages.create(
        model=settings.anthropic_model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": content}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
    if not text:
        raise ValueError("Empty response from Claude editor")
    if text.startswith("```"):
        lines = text.splitlines()
        # drop the first fence and the last fence if present
        text = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        # Surface a helpful diagnostic; the caller's audit log will record
        # the raw text via the exception message.
        snippet = text[:500]
        raise ValueError(f"Claude editor returned non-JSON: {e}; first 500 chars: {snippet}") from e


def _validate_and_build_timeline(
    response: Dict[str, Any],
    cand_by_id: Dict[str, CandidateShot],
    warnings_out: List[str],
) -> List[TimelineClip]:
    raw_timeline = response.get("timeline") or []
    if not isinstance(raw_timeline, list):
        warnings_out.append("Claude returned a non-list 'timeline'; treated as empty.")
        return []

    out: List[TimelineClip] = []
    cursor_ms = 0
    for i, entry in enumerate(raw_timeline):
        if not isinstance(entry, dict):
            warnings_out.append(f"Timeline entry {i} was not a JSON object; dropped.")
            continue
        sid = entry.get("shot_id")
        c = cand_by_id.get(str(sid)) if sid else None
        if not c:
            warnings_out.append(
                f"Claude referenced shot_id={sid!r} which is not in the catalog; dropped."
            )
            continue

        try:
            s_in = int(entry.get("source_in_ms", c.start_ms))
            s_out = int(entry.get("source_out_ms", c.end_ms))
        except (TypeError, ValueError):
            warnings_out.append(
                f"Timeline entry {i} had non-integer timestamps; dropped."
            )
            continue

        # Clamp into the shot's actual bounds.
        s_in = max(c.start_ms, min(s_in, c.end_ms))
        s_out = max(c.start_ms, min(s_out, c.end_ms))
        if s_out - s_in < MIN_CLIP_MS:
            warnings_out.append(
                f"Timeline entry {i} (shot {c.shot_index}) was {s_out - s_in}ms after clamping; "
                f"below minimum {MIN_CLIP_MS}ms, dropped."
            )
            continue

        clip_len = s_out - s_in
        role = entry.get("role_in_edit")
        why = entry.get("why")
        out.append(TimelineClip(
            file_id=c.file_id,
            file_name=c.file_name,
            file_r2_key=c.file_r2_key,
            file_r2_proxy_key=c.file_r2_proxy_key,
            source_in_ms=s_in,
            source_out_ms=s_out,
            timeline_start_ms=cursor_ms,
            timeline_end_ms=cursor_ms + clip_len,
            score=float(c.score or 1.0),
            trimmed_around_ms=None,  # Claude trimmed directly; not "around peak motion"
            shot_id=str(sid),
            role_in_edit=str(role) if role else None,
            why=str(why) if why else None,
        ))
        cursor_ms += clip_len
    return out


def _enforce_duration_cap(
    timeline: List[TimelineClip],
    target_s: Optional[int],
) -> List[TimelineClip]:
    """If Claude overshoots the target by more than 15%, hard-cap by trimming
    the tail of the last clip. We never *extend* a timeline that fell short."""
    if target_s is None or not timeline:
        return timeline
    target_ms = target_s * 1000
    total = timeline[-1].timeline_end_ms
    if total <= target_ms:
        return timeline
    overshoot = total - target_ms
    last = timeline[-1]
    last_len = last.source_out_ms - last.source_in_ms
    if last_len - overshoot < MIN_CLIP_MS:
        # Drop the last clip entirely if we can't trim it sensibly.
        return timeline[:-1]
    last.source_out_ms -= overshoot
    last.timeline_end_ms -= overshoot
    return timeline


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compile_timeline_chat(
    history: List[Dict[str, Any]],
    candidates: List[CandidateShot],
    duration_target_s: Optional[int] = None,
    emit: Optional[Callable[[str, int, str], None]] = None,
) -> EditorResult:
    """
    Multi-turn variant. `history` is the entire conversation so far, ordered
    chronologically. Each item is either:
      {"role": "user", "content": "<text>"}
      {"role": "assistant", "reasoning": "<text>", "timeline": [
          {"shot_id": "...", "source_in_ms": int, "source_out_ms": int,
           "role_in_edit": "...", "why": "..."},
          ...
      ]}
    The LATEST item must be from the user. We render the history into a
    text block prepended to the standard catalog and ask Claude to produce
    the next full timeline. Same JSON schema as single-shot mode.
    """
    if not history or history[-1].get("role") != "user":
        return EditorResult(
            reasoning="No latest user message in conversation history.",
            timeline=[],
            warnings=["Empty or malformed conversation history."],
        )
    latest_brief = str(history[-1].get("content") or "").strip()
    if not latest_brief:
        return EditorResult(
            reasoning="Latest user message was empty.",
            timeline=[],
            warnings=["Empty user message."],
        )
    if not candidates:
        return EditorResult(
            reasoning="The catalog was empty so no timeline could be built.",
            timeline=[],
            warnings=["No candidate shots were found for this brief."],
        )

    catalog_text, cand_by_id = _build_catalog(candidates)

    # Pass A: plan the shape (beat sheet) before selecting clips. Streamed to
    # the UI so the user sees the editorial intent as it forms.
    plan_text = _plan_beats(history, catalog_text, duration_target_s)
    if plan_text and emit:
        emit("planning", 55, plan_text[:280])

    image_blocks, vision_note = _build_vision(candidates, latest_brief)
    history_text = _render_history(history[:-1])  # everything BEFORE the latest user msg
    user_message = _build_chat_user_message(
        history_text=history_text,
        latest_brief=latest_brief,
        duration_target_s=duration_target_s,
        catalog_text=catalog_text + vision_note,
        plan_text=plan_text,
    )
    system_prompt = prompts_mod.load("editor_chat")

    warnings: List[str] = []
    raw: Dict[str, Any] = {}
    try:
        raw = _call_claude(system_prompt, user_message, image_blocks=image_blocks)
    except Exception as e:
        logger.exception("Claude chat editor call failed")
        warnings.append(f"Claude editor call failed: {type(e).__name__}: {e}")
        return EditorResult(
            reasoning="Claude editor was unreachable or returned invalid JSON.",
            timeline=[],
            warnings=warnings,
            raw_response={"error": str(e)},
            catalog_text=catalog_text,
            user_text=user_message,
        )

    reasoning = str(raw.get("reasoning") or "").strip()
    timeline = _validate_and_build_timeline(raw, cand_by_id, warnings)
    timeline = _enforce_duration_cap(timeline, duration_target_s)

    pp = raw.get("post_processing") or {}
    if not isinstance(pp, dict):
        pp = {}
    claude_warnings = raw.get("warnings") or []
    if isinstance(claude_warnings, list):
        warnings.extend(str(w) for w in claude_warnings)
    if not timeline and not warnings:
        warnings.append("Claude returned an empty timeline with no explanation.")

    return EditorResult(
        reasoning=reasoning or "(Claude returned no reasoning.)",
        timeline=timeline,
        post_processing=pp,
        warnings=warnings,
        raw_response=raw,
        catalog_text=catalog_text,
        user_text=user_message,
    )


def _render_history(history: List[Dict[str, Any]]) -> str:
    """Format prior turns into a compact text block for the system to read."""
    if not history:
        return "(no prior turns yet)"
    lines: List[str] = []
    for i, msg in enumerate(history, start=1):
        role = msg.get("role")
        if role == "user":
            content = str(msg.get("content") or "").strip()
            lines.append(f"--- Turn {i} (user) ---\n{content}")
        elif role == "assistant":
            reasoning = str(msg.get("reasoning") or "").strip()
            timeline = msg.get("timeline") or []
            tl_lines: List[str] = []
            for j, c in enumerate(timeline):
                if not isinstance(c, dict):
                    continue
                tl_lines.append(
                    f"  {j+1}. shot_id={c.get('shot_id')}  "
                    f"in={c.get('source_in_ms')}ms  out={c.get('source_out_ms')}ms  "
                    f"role={c.get('role_in_edit') or '-'}  "
                    f"why={(c.get('why') or '').strip()[:120]}"
                )
            tl_block = "\n".join(tl_lines) or "  (empty timeline)"
            lines.append(
                f"--- Turn {i} (you, the editor) ---\n"
                f"reasoning: {reasoning}\n"
                f"timeline:\n{tl_block}"
            )
    return "\n\n".join(lines)


def _build_chat_user_message(
    *,
    history_text: str,
    latest_brief: str,
    duration_target_s: Optional[int],
    catalog_text: str,
    plan_text: str = "",
) -> str:
    parts: List[str] = [
        "CONVERSATION HISTORY (oldest first):",
        history_text,
        "",
        "LATEST USER MESSAGE:",
        latest_brief,
    ]
    if plan_text:
        parts.append(
            "\nEDITORIAL PLAN (your own beat sheet from the planning pass -- "
            "follow it, refining as needed against the exact catalog):\n"
            + plan_text
        )
    if duration_target_s is not None:
        parts.append(f"\nDURATION TARGET (this turn): {duration_target_s} seconds")
    else:
        parts.append("\nDURATION TARGET (this turn): (not specified -- preserve prior length unless the user asked otherwise)")
    parts.append(f"\n{catalog_text}")
    parts.append(
        "\nReturn the JSON object specified in your instructions. JSON only, no other text."
    )
    return "\n".join(parts)


def compile_timeline(
    brief: str,
    candidates: List[CandidateShot],
    duration_target_s: Optional[int] = None,
) -> EditorResult:
    """
    Smart-mode timeline compilation. Returns a validated EditorResult; never
    raises if Claude misbehaves -- instead surfaces the issue via warnings
    and returns whatever subset of clips was usable.
    """
    if not candidates:
        return EditorResult(
            reasoning="The catalog was empty so no timeline could be built.",
            timeline=[],
            warnings=["No candidate shots were found for this brief."],
        )

    catalog_text, cand_by_id = _build_catalog(candidates)
    image_blocks, vision_note = _build_vision(candidates, brief)
    user_message = _build_user_message(brief, duration_target_s, catalog_text + vision_note)
    system_prompt = prompts_mod.load("editor")

    warnings: List[str] = []
    raw: Dict[str, Any] = {}
    try:
        raw = _call_claude(system_prompt, user_message, image_blocks=image_blocks)
    except Exception as e:
        logger.exception("Claude editor call failed")
        warnings.append(f"Claude editor call failed: {type(e).__name__}: {e}")
        return EditorResult(
            reasoning="Claude editor was unreachable or returned invalid JSON.",
            timeline=[],
            warnings=warnings,
            raw_response={"error": str(e)},
            catalog_text=catalog_text,
            user_text=user_message,
        )

    reasoning = str(raw.get("reasoning") or "").strip()
    timeline = _validate_and_build_timeline(raw, cand_by_id, warnings)
    timeline = _enforce_duration_cap(timeline, duration_target_s)

    pp = raw.get("post_processing") or {}
    if not isinstance(pp, dict):
        pp = {}
    claude_warnings = raw.get("warnings") or []
    if isinstance(claude_warnings, list):
        warnings.extend(str(w) for w in claude_warnings)

    if not timeline and not warnings:
        warnings.append("Claude returned an empty timeline with no explanation.")

    return EditorResult(
        reasoning=reasoning or "(Claude returned no reasoning.)",
        timeline=timeline,
        post_processing=pp,
        warnings=warnings,
        raw_response=raw,
        catalog_text=catalog_text,
        user_text=user_message,
    )
