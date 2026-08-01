"""
speech_cuts_pipeline.plan.md section 11 -- Stage 6: minimal frame analysis
for on-camera speech cuts. Reuses vcut's own closed question bank
(app.services.vcut.questions) and structured-call pattern (app.services.
llm.ingest_gemini.complete_gemini) -- the SAME machinery the video
channel's pass2.py rides -- but its OWN small, separate, UNCACHED batched
call (section 13: "separate from the video Pass-2 cache, since speech
frames are a small, different set"; no CachedContent lifecycle needed for
a handful of images -- cost-first, section 6, favors the simpler path).

Cost-first cut selection (section 11): only WINNING cuts (+ their outlook
angles) that are ON-CAMERA get a frame at all -- voiceover/off-camera cuts
get zero. "Visually changing" (the trigger for a 2nd frame on a long cut)
is read off the ALREADY-COMPUTED seam action_energy signal (app.services.
seam), not scene_cuts.composition_points/shot_points -- a deliberate
simplification: the video channel already computes seam signals for every
file in the SAME run_vcut_ingest call, so reusing that array costs nothing
extra, whereas composition_points/shot_points would be a new signal to load
just for this one 2-vs-1-frame decision.

THIS MODULE SPENDS REAL MONEY once invoked (one Gemini call per project,
skipped entirely when there are zero on-camera cuts).
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

from app.services.vcut import questions
from app.services.vcut.speech.inputs import FaceTrackLite
from app.services.vcut.speech.params import SPEECH_FRAME_BATCH, SPEECH_FRAME_LONG_MS, SPEECH_FRAME_MAX
from app.services.vcut.speech.store import ExpandedCut, VisualFields, cut_key, is_on_camera

_ANGLE_TYPES = ("wide", "medium", "close", "ots", "other")

_SYSTEM_PROMPT = f"""You are looking at one or two frames from each of several on-camera speech
cuts. For each cut_id, answer:

  "angle_type": one of {list(_ANGLE_TYPES)} -- the camera's framing of the subject.
  "on_camera": true if a person is genuinely visible speaking/present on
    camera in these frames, false if the frame(s) turn out NOT to actually
    show anyone (e.g. an empty room, a cutaway) despite being sampled from
    a nominally on-camera span.
  A few short scene fields from this closed bank -- answer ONLY the ones
  that clearly apply, leave the rest at their default (empty/null):

{questions.bank_prompt_lines()}

Answer for every cut_id shown -- never skip one. Never invent a cut_id.
"""


class _CutAnswerOut(BaseModel):
    cut_id: str
    angle_type: Literal[_ANGLE_TYPES] = "other"
    on_camera: bool = True
    subject: str = ""
    action: str = ""
    moment_type: str = ""
    setting: str = ""
    on_screen_text: str = ""
    notable_object: str = ""
    count: Optional[int] = None
    motion_quality: str = ""
    continuity_cue: str = ""


class FrameSchema(BaseModel):
    answers: List[_CutAnswerOut] = []


def _check_nonempty(parsed: FrameSchema) -> Optional[str]:
    if not parsed.answers:
        return "answers must not be empty -- answer for every cut_id shown."
    return None


def _has_interior_motion_peak(action_energy: List[float], hop_ms: int, in_ms: int, out_ms: int) -> bool:
    """A crude "visually changing" proxy (module docstring): true if some
    hop strictly inside (in_ms, out_ms) -- excluding the edges -- has
    action_energy at or above the clip's own median. Cheap, and reuses a
    signal already computed for the video channel in the same run."""
    if not action_energy or hop_ms <= 0:
        return False
    lo_i = max(0, in_ms // hop_ms + 1)
    hi_i = min(len(action_energy) - 1, out_ms // hop_ms - 1)
    if lo_i > hi_i:
        return False
    clip_vals = sorted(action_energy)
    median = clip_vals[len(clip_vals) // 2] if clip_vals else 0.0
    return any(action_energy[i] >= median for i in range(lo_i, hi_i + 1))


def frame_timestamps_for_cut(
    in_ms: int, out_ms: int, face_tracks: List[FaceTrackLite],
    action_energy: Optional[List[float]] = None, hop_ms: int = 0,
) -> List[int]:
    """1 timestamp by default (the clearest overlapping speaking face crop,
    or the beat midpoint with no face data); 2 only when the cut is BOTH
    longer than SPEECH_FRAME_LONG_MS AND visually changing -- the second
    frame placed at the detected motion peak nearest the cut's midpoint,
    never at the raw start/end (section 11's own instruction)."""
    from app.services.vcut.speech.store import hero_ts_for_span

    first = hero_ts_for_span(face_tracks, in_ms, out_ms)
    duration_ms = out_ms - in_ms
    if duration_ms <= SPEECH_FRAME_LONG_MS or not action_energy:
        return [first]
    if not _has_interior_motion_peak(action_energy, hop_ms, in_ms, out_ms):
        return [first]

    lo_i = max(0, in_ms // hop_ms + 1)
    hi_i = min(len(action_energy) - 1, out_ms // hop_ms - 1)
    peak_i = max(range(lo_i, hi_i + 1), key=lambda i: action_energy[i])
    second = peak_i * hop_ms
    ts = sorted({first, second})
    return ts[:SPEECH_FRAME_MAX]


def select_cuts_needing_frames(cuts: List[ExpandedCut], face_tracks_by_file: Dict[str, List[FaceTrackLite]]) -> List[ExpandedCut]:
    """Cost-first filter (section 11): winning, on-camera cuts only.
    Alternates and voiceover/off-camera cuts get zero frames."""
    out = []
    for ec in cuts:
        if ec.resolved_beat.take_role != "winner":
            continue
        tracks = face_tracks_by_file.get(ec.file_id, [])
        if is_on_camera(tracks, ec.in_ms, ec.out_ms):
            out.append(ec)
    return out


def _nonempty_fields(ans: "_CutAnswerOut") -> Dict[str, object]:
    """Only the fields the model actually answered -- an empty string means
    "left at default" (not answered) for every text field; for "count"
    (the one integer field) 0 is a genuine, meaningful answer and must
    survive, so only None (the documented "not applicable") is excluded."""
    out: Dict[str, object] = {}
    for qid in questions.ALL_IDS:
        val = getattr(ans, qid, None)
        if qid == "count":
            if val is not None:
                out[qid] = val
        elif val:
            out[qid] = val
    return out


def run_frame_analysis(
    cuts: List[ExpandedCut],
    face_tracks_by_file: Dict[str, List[FaceTrackLite]],
    action_energy_by_file: Optional[Dict[str, Tuple[List[float], int]]] = None,
) -> Tuple[Dict[str, VisualFields], Dict[str, int]]:
    """Returns ({cut_key(beat_id, file_id): VisualFields}, usage). Empty
    dicts (no call at all) when there are zero on-camera winning cuts.

    The image-bearing cuts are chunked into batches of SPEECH_FRAME_BATCH at
    CUT granularity (a cut's frames never split across calls) and each batch
    is ONE complete_gemini call with its own known_ids -- mirroring l3
    Pass-2's size-based batching (pass2.build_pass2_batches). A single call
    that had to emit one answer per cut overran flash-lite's output budget at
    multicam fan-out scale (~30 outlook-angle cuts per winning beat) and
    returned zero JSON; bounding the answers per call fixes that structurally.
    Usage is summed across batches so cost accounting stays honest. A project
    that fits in one batch is byte-for-byte identical to the pre-batch call.
    NO FALLBACK: a batch that still fails after complete_gemini's own retries
    propagates and fails the whole run."""
    from app.config import get_settings
    from app.services.l3.frames import extract_stills_from_r2
    from app.services.llm.base import image_block, text_block
    from app.services.llm.ingest_gemini import _sum_usage, complete_gemini

    targets = select_cuts_needing_frames(cuts, face_tracks_by_file)
    if not targets:
        return {}, {}

    action_energy_by_file = action_energy_by_file or {}
    ts_by_cut: Dict[str, List[int]] = {}
    for ec in targets:
        ae, hop_ms = action_energy_by_file.get(ec.file_id, ([], 0))
        key = cut_key(ec.resolved_beat.beat.id, ec.file_id)
        ts_by_cut[key] = frame_timestamps_for_cut(
            ec.in_ms, ec.out_ms, face_tracks_by_file.get(ec.file_id, []), ae, hop_ms)

    proxy_keys = _proxy_keys_for_files({ec.file_id for ec in targets})
    stills_by_file: Dict[str, Dict[int, str]] = {}
    for ec in targets:
        if ec.file_id in stills_by_file:
            continue
        proxy_key = proxy_keys.get(ec.file_id)
        if not proxy_key:
            continue
        all_ts = sorted({ts for k, ts_list in ts_by_cut.items() for ts in ts_list
                         if k.endswith(f"::{ec.file_id}")})
        stills_by_file[ec.file_id] = extract_stills_from_r2(proxy_key, all_ts)

    # Assemble each cut's own [caption, image...] block group ONCE, keeping
    # only cuts that resolved to at least one still (imageless cuts were never
    # sent and never became a known_id). Batching below is over these whole
    # groups, so a single cut's frames can never be split across calls.
    cut_blocks: List[Tuple[str, List]] = []
    for ec in targets:
        key = cut_key(ec.resolved_beat.beat.id, ec.file_id)
        stills = stills_by_file.get(ec.file_id) or {}
        imgs = [stills[ts] for ts in ts_by_cut.get(key, []) if ts in stills]
        if not imgs:
            continue
        group: List = [text_block(f"cut_id={key} meaning={ec.resolved_beat.beat.gist!r}")]
        for b64 in imgs:
            group.append(image_block(b64))
        cut_blocks.append((key, group))

    if not cut_blocks:
        return {}, {}

    settings = get_settings()
    visual_by_key: Dict[str, VisualFields] = {}
    usage: Dict[str, int] = {}
    for start in range(0, len(cut_blocks), SPEECH_FRAME_BATCH):
        batch = cut_blocks[start:start + SPEECH_FRAME_BATCH]
        known_ids = {key for key, _group in batch}
        blocks = [b for _key, group in batch for b in group]
        completion = complete_gemini(
            _SYSTEM_PROMPT, blocks, FrameSchema,
            model=settings.vcut_pass2_model, max_tokens=8000, thinking="low",
            extra_check=_check_nonempty,
        )
        parsed = FrameSchema.model_validate(completion.data)
        for ans in parsed.answers:
            if ans.cut_id not in known_ids:
                continue
            visual_by_key[ans.cut_id] = VisualFields(
                on_camera=ans.on_camera, angle_type=ans.angle_type,
                scene_specifics=_nonempty_fields(ans),
            )
        usage = _sum_usage(usage, completion.usage)
    return visual_by_key, usage


def _proxy_keys_for_files(file_ids) -> Dict[str, str]:
    from app.services import db
    if not file_ids:
        return {}
    with db.connection_dict_row() as conn:
        rows = conn.execute(
            "select id::text as id, coalesce(r2_proxy_key, r2_key) as proxy_key "
            "from files where id = any(%s::uuid[])",
            (list(file_ids),),
        ).fetchall()
    return {r["id"]: r["proxy_key"] for r in rows if r["proxy_key"]}
