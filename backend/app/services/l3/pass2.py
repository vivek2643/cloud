"""
Cuts v3, Pass 2: the MERGED final per-cut record (see cuts_v3.plan.md sec.
5) -- ``Pass2Cut``/``Pass2Output`` are what ``post.py`` consumes.

The actual LLM calling is split across two sibling modules, decided after
real ingest runs showed the model getting unreliable past ~15-40 cuts of
output in one call:

  - ``pass2a.py`` -- IDENTITY + take resolution (needs cross-cut visual
    comparison for take-vs-outlook, so shards stay take-group-aware).
  - ``pass2b.py`` -- VISUAL judgment only (framing/look/captions/taste;
    no cross-cut dependency at all, so batches are pure chunking and run
    with far more parallelism).

This module has no model-calling logic of its own -- just the merged data
shape both halves assemble into, and the merge itself.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field

from app.services.l3.pass1 import Pass1Output
from app.services.l3.pass2a import IdentityOutput
from app.services.l3.pass2b import Framing, Look, TasteFences, VisualJudgment

__all__ = ["Framing", "Look", "TasteFences", "Pass2Cut", "Pass2Output",
           "merge_identity_and_visual", "apply_junk_suspects"]

logger = logging.getLogger(__name__)


class Pass2Cut(BaseModel):
    source_ref: str                 # e.g. "speech_cut[2]" / "video_group[0]" -- joins back to pass 1
    kind: str                        # "speech" | "video"
    file_id: str
    word_span: Tuple[int, int] | None = None    # speech cuts only
    atom_ids: List[int] | None = None           # video cuts only
    label: str
    summary: str
    speaker: str | None = None
    on_camera: bool | None = None
    junk: bool = False
    junk_reason: str = ""
    framing: Framing = Field(default_factory=Framing)
    look: Look = Field(default_factory=Look)
    caption_zones: List[Tuple[float, float, float, float]] = Field(default_factory=list)
    taste_fences: TasteFences = Field(default_factory=TasteFences)
    readability_ms: int = 0
    natural_sound: bool = False
    take_group_id: str | None = None
    take_role: str | None = None    # "take" | "outlook" | "winner"
    channel: str | None = None      # "said" | "done" | "shown" (video: done|shown)


class Pass2Output(BaseModel):
    cuts: List[Pass2Cut] = Field(default_factory=list)


def merge_identity_and_visual(
    identity_output: IdentityOutput, visual_by_index: Dict[int, VisualJudgment],
) -> Pass2Output:
    """Combine pass 2a's identity/take resolution with pass 2b's visual
    judgment into the final per-cut record post.py consumes. Raises
    ``ValueError`` if any confirmed cut is missing its visual judgment --
    a partial merge is exactly the kind of silent gap the plan's "no
    fallback" rule exists to prevent, and simply defaulting Framing()/Look()
    for a skipped cut would be indistinguishable from a genuine judgment."""
    missing = [i for i in range(len(identity_output.cuts)) if i not in visual_by_index]
    if missing:
        raise ValueError(f"pass 2b did not return a visual judgment for cut index(es) {missing}")

    cuts: List[Pass2Cut] = []
    for i, identity in enumerate(identity_output.cuts):
        visual = visual_by_index[i]
        cuts.append(Pass2Cut(
            source_ref=identity.source_ref, kind=identity.kind, file_id=identity.file_id,
            word_span=identity.word_span, atom_ids=identity.atom_ids,
            label=identity.label, summary=identity.summary, speaker=identity.speaker,
            on_camera=identity.on_camera, junk=identity.junk, junk_reason=identity.junk_reason,
            framing=visual.framing, look=visual.look, caption_zones=visual.caption_zones,
            taste_fences=visual.taste_fences, readability_ms=visual.readability_ms,
            natural_sound=identity.natural_sound,
            take_group_id=identity.take_group_id, take_role=identity.take_role,
            channel=identity.channel,
        ))
    return Pass2Output(cuts=cuts)


def apply_junk_suspects(pass2: Pass2Output, pass1: Pass1Output) -> Pass2Output:
    """Deterministically carry pass 1's semantic junk calls onto the final
    cuts. A cut fully contained in a pass-1 junk_suspect span (a leading
    camera cue that the coverage-fill surfaced as its own recovered cut, dead
    air, etc.) is marked junk -- rather than trusting pass 2a to re-flag it
    from a single still. Junk is binary and RECOVERABLE (hidden into the
    Discarded tray, never deleted), so nothing is lost. Conservative: only
    EXACT containment, never a partial overlap (which might clip real
    content)."""
    speech_susp: Dict[str, List[Tuple[int, int, str]]] = {}
    video_susp: Dict[str, List[Tuple[set, str]]] = {}
    for js in pass1.junk_suspects:
        if js.word_span is not None:
            speech_susp.setdefault(js.file_id, []).append((js.word_span[0], js.word_span[1], js.reason))
        elif js.atom_ids:
            video_susp.setdefault(js.file_id, []).append((set(js.atom_ids), js.reason))

    n = 0
    for c in pass2.cuts:
        if c.junk:
            continue
        hit = None
        if c.kind == "speech" and c.word_span is not None:
            a, b = c.word_span
            hit = next((r for (sa, sb, r) in speech_susp.get(c.file_id, []) if sa <= a and b <= sb), None)
        elif c.kind == "video" and c.atom_ids:
            ids = set(c.atom_ids)
            hit = next((r for (sids, r) in video_susp.get(c.file_id, []) if ids <= sids), None)
        if hit is not None:
            c.junk = True
            c.junk_reason = c.junk_reason or hit
            n += 1
    if n:
        logger.info("pass2: %d cut(s) marked junk from pass-1 suspects", n)
    return pass2
