"""
Clip catalog: the compact per-clip index the orchestrator plans over.

Context strategy (the "don't load the whole repo" rule): Opus gets ONE short
paragraph per clip up front -- enough to decide which clips are even relevant.
Full detail (the entire L2 footage log, seam lists) stays behind tools
(`read_clip`, `query_seams`) the agent calls only for clips it is actually
considering. This keeps a multi-clip project inside a stable, cacheable prompt
prefix regardless of how rich the underlying analysis is.

Built once per thread from data that is already in the DB (clip_perception +
cut grids); no model calls, no video access.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import get_settings

logger = logging.getLogger(__name__)


def _pg_conn():
    import psycopg  # lazy: keeps pure helpers importable without the driver
    settings = get_settings()
    return psycopg.connect(settings.database_url, autocommit=True)


@dataclass
class ClipSummary:
    file_id: str
    name: str
    duration_s: float
    l1_status: Optional[str]
    l2_status: Optional[str]
    # From L2 perception (None/empty when L2 hasn't run or was skipped).
    content_type: Optional[str] = None
    logline: Optional[str] = None
    primary_axis: Optional[str] = None
    cut_sensitivity: Optional[str] = None
    time_of_day: Optional[str] = None
    location: Optional[str] = None
    topics: List[str] = field(default_factory=list)
    persons: List[str] = field(default_factory=list)  # rendered one-liners
    event_count: int = 0
    reaction_count: int = 0
    # From L1 grids: which channels exist + how many discrete seam candidates.
    has_transcript: bool = False
    dialogue_seams: int = 0
    beat_seams: int = 0
    action_seams: int = 0
    has_camera_grid: bool = False


def _person_line(p: dict) -> str:
    """'p1: man in 30s, beard, dark jacket (voice S0)' -- id + identikit + voice."""
    bits: List[str] = []
    desc = p.get("canonical_description") or p.get("role")
    if desc:
        bits.append(desc)
    voice = p.get("voice_speaker_id")
    tail = f" (voice {voice})" if voice else ""
    return f"{p.get('local_id', 'p?')}: {', '.join(bits) or 'person'}{tail}"


def build_catalog(file_ids: List[str]) -> List[ClipSummary]:
    """One ClipSummary per existing file, in the order given."""
    if not file_ids:
        return []
    out: Dict[str, ClipSummary] = {}

    with _pg_conn() as conn:
        rows = conn.execute(
            """
            select f.id::text, f.name, coalesce(f.duration_seconds, 0),
                   f.l1_status, f.l2_status,
                   cp.perception,
                   t.segments is not null as has_transcript,
                   coalesce(jsonb_array_length(af.dialogue_cut_points), 0),
                   coalesce(jsonb_array_length(af.beat_cut_points), 0),
                   coalesce(jsonb_array_length(md.action_points), 0),
                   md.file_id is not null as has_camera_grid
              from files f
              left join clip_perception cp on cp.file_id = f.id
              left join transcripts t      on t.file_id  = f.id
              left join audio_features af  on af.file_id = f.id
              left join motion_dynamics md on md.file_id = f.id
             where f.id = any(%s::uuid[])
            """,
            (file_ids,),
        ).fetchall()

    for (
        fid, name, duration, l1_status, l2_status, perception,
        has_transcript, dlg_seams, beat_seams, action_seams, has_camera,
    ) in rows:
        summary = ClipSummary(
            file_id=fid,
            name=name,
            duration_s=float(duration),
            l1_status=l1_status,
            l2_status=l2_status,
            has_transcript=bool(has_transcript),
            dialogue_seams=int(dlg_seams),
            beat_seams=int(beat_seams),
            action_seams=int(action_seams),
            has_camera_grid=bool(has_camera),
        )

        doc = perception if isinstance(perception, dict) else (
            json.loads(perception) if perception else None
        )
        if doc and not doc.get("_parse_error"):
            summary.content_type = doc.get("content_type")
            summary.logline = doc.get("logline")
            edit = doc.get("editability") or {}
            summary.primary_axis = edit.get("primary_axis")
            summary.cut_sensitivity = edit.get("cut_sensitivity")
            look = doc.get("look") or {}
            summary.time_of_day = look.get("time_of_day")
            setting = doc.get("setting") or {}
            summary.location = setting.get("location")
            summary.topics = list(doc.get("topics") or [])[:5]
            summary.persons = [_person_line(p) for p in (doc.get("persons") or [])]
            summary.event_count = len(doc.get("events") or [])
            summary.reaction_count = len(doc.get("reactions") or [])

        out[fid] = summary

    # Preserve caller order; silently drop ids whose files no longer exist.
    return [out[fid] for fid in file_ids if fid in out]


def load_perceptions(file_ids: List[str]) -> Dict[str, dict]:
    """file_id -> parsed L2 perception dict (skips missing/unparseable). Shared
    by the people roster and the angle menu so they read one source."""
    if not file_ids:
        return {}
    out: Dict[str, dict] = {}
    with _pg_conn() as conn:
        rows = conn.execute(
            "select file_id::text, perception from clip_perception where file_id = any(%s::uuid[])",
            (file_ids,),
        ).fetchall()
    for fid, perception in rows:
        doc = perception if isinstance(perception, dict) else (
            json.loads(perception) if perception else None
        )
        if doc and not doc.get("_parse_error"):
            out[fid] = doc
    return out


def render_catalog_text(clips: List[ClipSummary]) -> str:
    """The prompt-facing rendering: one compact block per clip."""
    if not clips:
        return "(no clips in scope)"
    blocks: List[str] = []
    for c in clips:
        lines = [f"CLIP {c.file_id} \"{c.name}\" -- {c.duration_s:.1f}s"]

        if c.l2_status == "ready" and c.logline:
            desc = f"  {c.content_type or 'video'}: {c.logline}"
            ctx_bits = [b for b in (c.time_of_day, c.location) if b]
            if ctx_bits:
                desc += f" [{', '.join(ctx_bits)}]"
            lines.append(desc)
        else:
            lines.append(f"  (no deep perception: l2_status={c.l2_status or 'none'})")

        if c.persons:
            lines.append("  people: " + "; ".join(c.persons))
        if c.topics:
            lines.append("  topics: " + ", ".join(str(t) for t in c.topics))

        profile = []
        if c.primary_axis:
            profile.append(f"axis={c.primary_axis}")
        if c.cut_sensitivity:
            profile.append(f"cut_sensitivity={c.cut_sensitivity}")
        profile.append(f"events={c.event_count}")
        if c.reaction_count:
            profile.append(f"reactions={c.reaction_count}")
        lines.append("  edit profile: " + ", ".join(profile))

        seams = []
        seams.append(f"dialogue={c.dialogue_seams}" if c.has_transcript else "dialogue=n/a(silent)")
        seams.append(f"beat={c.beat_seams}")
        seams.append(f"action={c.action_seams}")
        seams.append("camera=yes" if c.has_camera_grid else "camera=no")
        lines.append("  seam candidates: " + ", ".join(seams))

        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)
