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

# Ordinal shot-size vocabulary, tightest -> widest (+ "unsure"). A CATEGORY
# the model owns (deterministic-keep rule); code turns it into the ordinal
# tightness term of total_quality (post._shot_tightness). Kept as a closed set
# so that ranking is well-defined; anything off-list reads as "unsure".
SHOT_SIZES = (
    "extreme_close_up", "close_up", "medium_close_up", "medium",
    "medium_wide", "wide", "extreme_wide", "unsure",
)


class Framing(BaseModel):
    subject_box: Tuple[float, float, float, float] | None = None   # normalized x,y,w,h
    crop_16x9: Tuple[float, float, float, float] | None = None
    crop_9x16: Tuple[float, float, float, float] | None = None
    crop_1x1: Tuple[float, float, float, float] | None = None
    rotation_deg: float = 0.0
    # How tight the framing is on the subject -- one of SHOT_SIZES. Purely a
    # category from the pixels; code (not the model) maps it to a number.
    shot_size: str = "unsure"


class WhiteReference(BaseModel):
    """A candidate neutral object/region for white-balance anchoring
    (color_grading.plan.md SS2.3). The model only PROPOSES this from the
    pixels -- deterministic code in the correct layer verifies it actually
    reads as neutral (low Lab a*/b* cast) before ever trusting it, same
    "model proposes, code decides" split as `shot_size`."""
    present: bool = False
    region: Tuple[float, float, float, float] | None = None   # normalized x,y,w,h
    object: str | None = None   # brief description, e.g. "white wall", "grey card", "white shirt"


class Look(BaseModel):
    graded: bool = False
    palette: List[str] = Field(default_factory=list)
    exposure_flags: List[str] = Field(default_factory=list)
    white_reference: WhiteReference = Field(default_factory=WhiteReference)


class TasteFences(BaseModel):
    max_tasteful_speed: float = 1.0
    min_tasteful_speed: float = 1.0


class Appearance(BaseModel):
    """Structured, STABLE-ONLY identity traits (identity_map.plan.md Phase 0):
    categorical fields code can match exactly, instead of parsing free prose.
    Deliberately excludes anything volatile (clothing, pose, current action)
    -- those change shot to shot and would poison the cross-file fingerprint
    `identity/reconcile.py` clusters on. Every field is optional/"unsure";
    the model states what it can actually see, never guesses to fill a slot."""
    model_config = ConfigDict(extra="forbid")

    apparent_gender: str | None = None    # "male" | "female" | "unsure"
    apparent_age_band: str | None = None  # "child"|"teen"|"20s"|"30s"|"40s"|"50s"|"60s+"|"unsure"
    hair: str | None = None               # "bald"|"very_short"|"short"|"medium"|"long"|"unsure"
    hair_color: str | None = None         # "black"|"brown"|"blonde"|"grey"|"white"|"red"|"unsure"
    facial_hair: str | None = None        # "none"|"stubble"|"moustache"|"beard"|"goatee"|"unsure"
    glasses: str | None = None            # "yes"|"no"|"unsure"
    skin_tone: str | None = None          # "light"|"medium"|"tan"|"dark"|"unsure"
    build: str | None = None              # "slim"|"average"|"heavy"|"unsure"


class PersonLook(BaseModel):
    """A concise visual fingerprint of one person visible in the cut -- enough
    to recognise the same person across cuts by eye, never an identity claim.
    Categorical/descriptive only (the model owns appearance; it never assigns
    scores or ids)."""
    model_config = ConfigDict(extra="forbid")

    description: str                       # e.g. "man, short dark hair, beard, grey hoodie"
    appearance: Appearance = Field(default_factory=Appearance)  # stable-trait fingerprint, see Appearance
    position: str | None = None            # rough frame position: "left" | "center" | "right"
    speaking: bool | None = None           # mouth visibly moving in these frames


class VisualJudgment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cut_index: int                  # position in the IdentityOutput.cuts list this judges
    framing: Framing = Field(default_factory=Framing)
    look: Look = Field(default_factory=Look)
    caption_zones: List[Tuple[float, float, float, float]] = Field(default_factory=list)
    taste_fences: TasteFences = Field(default_factory=TasteFences)
    readability_ms: int = 0
    # Every person visible in this cut, described well enough to re-identify by
    # eye across cuts (for take/outlook grouping + "show the speaker" arrange
    # decisions). Empty for a cut with no people on screen.
    people: List[PersonLook] = Field(default_factory=list)


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
    "batch, judge purely from the pixels: framing -- subject_box, plus the "
    "best crop for each delivery shape (crop_16x9, crop_9x16, crop_1x1), each "
    "recomposed to keep the subject and eyeline in frame for that aspect (not "
    "a centre-crop of the landscape), rotation_deg only for a visibly "
    "tilted shot (else 0), and shot_size -- how tight the frame is on the "
    "subject, exactly one of: extreme_close_up, close_up, medium_close_up, "
    "medium, medium_wide, wide, extreme_wide (use unsure only if there is no "
    "clear subject). Look -- graded vs log/flat, palette, exposure_flags, "
    "and its white_reference, which is a field NESTED INSIDE look "
    "(look.white_reference, NOT a top-level field of the judgment): if some "
    "object in frame is genuinely neutral-colored (a white/grey wall, a grey "
    "card, white paper, a plain white garment -- NOT skin, NOT anything "
    "colored or patterned) and evenly lit, set look.white_reference.present="
    "true with its region (normalized x,y,w,h) and a short object "
    "description; otherwise present=false and leave region/object empty. Only "
    "propose it when genuinely confident -- this is a candidate the code will "
    "verify, not a guess to force. "
    "caption_zones (normalized boxes clear of the subject across every "
    "image you were shown for that cut), taste fences (max/min tasteful "
    "playback speed for this content), and readability_ms (how long a "
    "viewer needs to read this frame if it holds as a still).\n\n"
    "Also list `people`: every person visible in the cut with a concise "
    "description that would let someone recognise them again across cuts "
    "(apparent gender/age, hair, facial hair, clothing/colour, anything "
    "distinctive), their rough frame position (left/center/right), and "
    "whether their mouth is visibly moving (speaking). No people on screen "
    "-> empty list. Describe appearance only; never guess names or assign "
    "any score.\n\n"
    "Each person ALSO gets a structured `appearance` (nested inside that "
    "person, alongside description/position/speaking): apparent_gender, "
    "apparent_age_band, hair, hair_color, facial_hair, glasses, skin_tone, "
    "build -- each one of its listed categories, or omitted/\"unsure\" if not "
    "clearly visible. These are for matching the SAME PERSON across "
    "different cuts and different camera angles, so use ONLY traits that "
    "stay stable shot to shot: never clothing, never pose, never what they "
    "are doing right now -- those belong in `description`/`position`, not "
    "`appearance`. Leave a field unset rather than guess.\n\n"
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
                       extra_blocks=image_blocks, max_tokens=24000)
