"""
Synced-group people roster.

For a multicam SYNC GROUP (same moment, several angles), we hand Opus a compact
roster of who appears in each angle -- `canonical_description` + the strongest
DURABLE traits + the within-clip voice link -- so it can infer cross-clip
identity itself, plus the matching RULE that keeps that inference safe.

Why no matching pipeline: the same person's `local_id` (p1) and diarization
label (S0) are CLIP-LOCAL, so they don't carry across angles. The durable traits
are the cross-video key by design (see l2/schema.py), and the group is already
time-aligned -- enough for Opus to match. Low-confidence matches simply stay
unmade (the failure mode is a missing link, never a hallucinated one).
"""
from __future__ import annotations

from typing import List

from app.services.l3.catalog import load_perceptions
from app.services.l3.sync import SyncGroup

# Durable traits, strongest re-id signal first; we surface a few per person.
_DURABLE_ORDER = (
    "distinctive_marks",
    "age_band",
    "gender_presentation",
    "hair_color",
    "build",
    "skin_tone",
    "face_shape",
)

MATCHING_RULE = (
    "MATCHING RULE: the angles in a sync group are the SAME moment, so the same "
    "person appears in several of them under DIFFERENT clip-local ids. Match "
    "people across angles by their DURABLE appearance traits and by the fact "
    "that the group is time-aligned -- NEVER by the voice label or p-id (S0/p1 "
    "in one clip is not S0/p1 in another). If traits don't clearly agree, leave "
    "them unmatched rather than guess."
)


def _durable_bits(person: dict) -> str:
    durable = person.get("durable") or {}
    bits: List[str] = []
    for key in _DURABLE_ORDER:
        val = durable.get(key)
        if not val:
            continue
        if isinstance(val, list):
            val = ", ".join(str(x) for x in val if x)
            if not val:
                continue
        bits.append(str(val))
        if len(bits) >= 4:
            break
    return "; ".join(bits)


def _person_line(person: dict) -> str:
    pid = person.get("local_id", "p?")
    desc = person.get("canonical_description") or person.get("role") or "person"
    durable = _durable_bits(person)
    voice = person.get("voice_speaker_id")
    parts = [desc]
    if durable:
        parts.append(f"traits: {durable}")
    line = f"    {pid}: " + " | ".join(parts)
    if voice:
        line += f" (voice {voice} in-clip)"
    return line


def render_roster_text(groups: List[SyncGroup]) -> str:
    """Compact roster block for the synced groups, or '' when there are none."""
    if not groups:
        return ""
    member_ids = [m.file_id for g in groups for m in g.members]
    perceptions = load_perceptions(member_ids)

    lines: List[str] = ["SYNC-GROUP PEOPLE ROSTER (match identities across angles yourself):"]
    for g in groups:
        lines.append(f"  {g.group_id}:")
        for m in g.members:
            persons = (perceptions.get(m.file_id) or {}).get("persons") or []
            tag = " [hero audio]" if m.file_id == g.hero_file_id else ""
            if not persons:
                lines.append(f"   angle {m.file_id}{tag}: (no people logged)")
                continue
            lines.append(f"   angle {m.file_id}{tag}:")
            for p in persons:
                lines.append(_person_line(p))
    lines.append("  " + MATCHING_RULE)
    return "\n".join(lines)
