"""
Cuts v3, Pass 2b: VISUAL judgment (framing/look/caption_zones/taste_fences/
readability_ms) for cuts pass 2a has already confirmed. No cross-cut
dependency at all -- unlike take resolution, judging one cut's crop or grade
never needs another cut's pixels -- so batches are pure chunking (no
co-location constraint) and can run with much more parallelism than pass
2a's take-aware shards.

Addressing: pass 2a may split one pass-1 video_tentative_group into several
final cuts (all sharing the same source_ref), so "source_ref" alone can't
address an individual pass-2a output cut. Instead, every pass-2a cut gets a
stable ``cut_index`` (its position in ``IdentityOutput.cuts``) once pass 2a
finishes; pass 2b's prompt renders the confirmed-cut list against those
indices, and its output references cuts by the same index.

Images: pass 2b needs the same pixels pass 2a saw (crops/grade/captions are
just as visual a judgment as take comparison), so it re-associates each
confirmed cut with the frames ``image_plan.py`` originally planned for its
source_ref -- if pass 2a split a group, the pieces share the group's
originally-planned frames (image_plan's economy is already per-group, not
per eventual sub-cut, so this matches its own granularity).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, ConfigDict, Field

from app.services.l3.image_plan import PlannedFrame
from app.services.l3.pass2a import IdentityCut, IdentityOutput
from app.services.llm import client as ic
from app.services.llm.base import image_block, text_block


# --------------------------------------------------------------------------
# Output schema -- the visual half of the complete per-cut record (see
# cuts_v3.plan.md sec. 5; pass2.py holds the MERGED Pass2Cut both pass 2a
# and pass 2b assemble into)
# --------------------------------------------------------------------------

class Framing(BaseModel):
    subject_box: Tuple[float, float, float, float] | None = None   # normalized x,y,w,h
    crop_16x9: Tuple[float, float, float, float] | None = None
    crop_9x16: Tuple[float, float, float, float] | None = None
    crop_1x1: Tuple[float, float, float, float] | None = None
    rotation_deg: float = 0.0


class Look(BaseModel):
    graded: bool = False
    palette: List[str] = Field(default_factory=list)
    exposure_flags: List[str] = Field(default_factory=list)


class TasteFences(BaseModel):
    max_tasteful_speed: float = 1.0
    min_tasteful_speed: float = 1.0


class VisualJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cut_index: int                  # position in the IdentityOutput.cuts list this judges
    framing: Framing = Field(default_factory=Framing)
    look: Look = Field(default_factory=Look)
    caption_zones: List[Tuple[float, float, float, float]] = Field(default_factory=list)
    taste_fences: TasteFences = Field(default_factory=TasteFences)
    readability_ms: int = 0


class VisualOutput(BaseModel):
    # Same rationale as IdentityOutput/Pass1Output: a single defaulted field
    # means a response wrapped under an unexpected top-level key would
    # otherwise "validate" as an empty result instead of failing loud.
    model_config = ConfigDict(extra="forbid")

    judgments: List[VisualJudgment] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Prompt
# --------------------------------------------------------------------------

_SYSTEM = (
    "Pass 1 grouped this project's footage; pass 2a confirmed each cut's "
    "identity (label, summary, junk, take resolution) -- repeated below as "
    "a numbered CUT list. Now you see the actual pixels for a batch of "
    "those confirmed cuts, captioned with which CUT number and timestamp "
    "each image belongs to.\n\n"
    "This is a VISUAL-JUDGMENT-ONLY pass -- do not re-judge label, summary, "
    "junk, or take role; that's already decided. For EVERY cut in this "
    "batch, judge purely from the pixels: framing (subject_box + three "
    "crops -- 16:9, 9:16, 1:1 -- + rotation_deg for orientation/horizon "
    "fixes), look (graded vs log/flat, palette, exposure flags), "
    "caption_zones (normalized boxes clear of the subject across every "
    "image you were shown for that cut), taste fences (max/min tasteful "
    "playback speed for this content), and readability_ms (how long a "
    "viewer needs to read this frame if it holds as a still).\n\n"
    "Reference every judgment by cut_index, the integer given in the CUT "
    "list and image captions -- one judgment per cut in this batch, no "
    "more, no fewer."
)


def render_identity_output(identity_output: IdentityOutput) -> str:
    """The confirmed-cut list, numbered by position -- pass 2b's addressing
    scheme (cut_index) and its only source of non-visual context."""
    lines = ["=== CONFIRMED CUTS (pass 2a) ==="]
    for i, cut in enumerate(identity_output.cuts):
        locator = f"words[{cut.word_span[0]}-{cut.word_span[1]}]" if cut.kind == "speech" \
            else f"atoms{cut.atom_ids}"
        lines.append(f"CUT {i}: {cut.kind} file={cut.file_id} {locator} "
                    f"label={cut.label!r} summary={cut.summary!r} junk={cut.junk}")
    return "\n".join(lines)


def _images_for_cut(
    cut: IdentityCut, planned_frames: List[PlannedFrame], images_b64: Dict[Tuple[str, int], str],
) -> List[Tuple[int, str]]:
    """(ts_ms, b64) pairs for every planned frame belonging to this cut's
    source_ref -- see module docstring for why source_ref (not the final
    cut) is the right join key."""
    out = []
    for f in planned_frames:
        if f.file_id == cut.file_id and f.ref == cut.source_ref:
            b64 = images_b64.get((f.file_id, f.ts_ms))
            if b64 is not None:
                out.append((f.ts_ms, b64))
    return sorted(out)


def build_visual_batches(identity_output: IdentityOutput, max_per_batch: int) -> List[List[int]]:
    """Cut indices chunked into batches -- no co-location constraint (see
    module docstring), so this is pure, order-preserving chunking."""
    indices = list(range(len(identity_output.cuts)))
    return [indices[i:i + max_per_batch] for i in range(0, len(indices), max_per_batch)]


def build_visual_batch_blocks(
    identity_output: IdentityOutput, batch_indices: List[int],
    planned_frames: List[PlannedFrame], images_b64: Dict[Tuple[str, int], str],
) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    for idx in batch_indices:
        cut = identity_output.cuts[idx]
        for ts_ms, b64 in _images_for_cut(cut, planned_frames, images_b64):
            blocks.append(text_block(f"IMG for CUT {idx} @ {ts_ms / 1000:.1f}s"))
            blocks.append(image_block(b64))
    return blocks


def run_visual_batch(
    identity_output: IdentityOutput, batch_indices: List[int],
    planned_frames: List[PlannedFrame], images_b64: Dict[Tuple[str, int], str],
) -> ic.Completion:
    """One pass-2b batch call. ``identity_output`` renders the SAME cached
    prefix on every batch in a run (identical blocks -> cache hits after
    the first); this batch's images are appended uncached. Raises
    ``ValueError`` if the batch has no resolvable images."""
    cached_blocks = [text_block(render_identity_output(identity_output))]
    image_blocks = build_visual_batch_blocks(identity_output, batch_indices, planned_frames, images_b64)
    if not image_blocks:
        raise ValueError("run_visual_batch: no images resolved for this batch")
    return ic.complete("pass2", _SYSTEM, cached_blocks, VisualOutput,
                       extra_blocks=image_blocks, max_tokens=16000)
