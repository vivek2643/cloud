"""
vcut_moment_energy.plan.md section 2 -- Pass 1: one VLM call PER FILE
(section 2.5) over that file's sampled non-speech frames -> MomentPlan
(flat, per-file moment FLAGS only), merged across files. Pass 1 plants
semantics (where the payoff is, its shape, what it is) and NEVER geometry
-- no boundaries, no grouping, no filtering, no "good ones" selection, no
question ids. Geometry is entirely resolve.py's job (a pure, deterministic
function of the flags + S(t) + energy). Model gemini-3.1-flash-lite by
default (cost first) via the SAME provider-neutral Gemini adapter the old
pipeline's gemini pass 2 uses (app.services.llm.ingest_gemini/base) --
generic LLM adapter layer, not L3 business logic, so reusing it does not
violate principle 7's isolation constraint.

Per-file fan-out (section 2.5), not one call over all files: the flag-only
prompt asks for "as MANY moments as the footage holds," so a single call's
output scales with TOTAL footage -- on a real 44-file project this blew
past even a 48000-token budget and truncated (twice), silently dropping
real cuts. A file's flags never depend on any other file, so per-file
output is bounded (one file's moments, never a project's) and truncation
is structurally impossible regardless of project size. The N per-file
calls run concurrently on a bounded thread pool and merge in file_rows
order. NO FALLBACK: a single file's persistent failure propagates and
aborts the whole project -- a broken file quietly contributing 0 flags
would hide missing cuts, so we refuse to mask it.

Each per-file call sends ONLY that file's sampled frames INLINE (via
build_frame_blocks scoped to a single file), with the task text as that
call's real system instruction. This bounds OUTPUT per file AND keeps
TOTAL input tokens the same as one uncached all-files call (each file's
frames are sent exactly once). Pass 1 therefore does NOT ride the shared
frame CachedContent (vcut_pass2_rich.plan.md section 3): referencing an
all-files cache from every per-file call would re-read every file's frames
N times -- strictly MORE input than sending each file's frames once. The
``cached_content`` parameter is kept on run_pass1 for call-site
compatibility but is IGNORED for Pass 1's own content; the shared cache is
retained solely for Pass 2 / enrich, which is unchanged.

pass1_video_input.plan.md, Phase 1: when a per-file ``VideoHandle`` is
given (``video_by_file``, built by orchestrate.py from subclip.py's
ffmpeg-cut, uploaded non-speech clip), that file's call sends video instead
of sampled frames -- the model sees actual motion (build/settle, camera
moves) over the file's non-speech footage only (speech regions are cut out
before upload), at ``vcut_video_fps``/``vcut_video_media_resolution``. The
model's returned ``t_ms`` is in SUB-CLIP time; a join-artifact/out-of-range
check plus ``subclip.map_sub_to_orig`` remap it back to real file ms before
the existing ``_clamp_moment`` safety net runs (unchanged). A file absent
from ``video_by_file`` uses the frames path instead -- this is how "frames"
input mode works (video mode off). In "video" mode orchestrate.py now
guarantees EVERY file has a handle before calling run_pass1 (a file whose
cut/upload failed to yield a handle aborts the run there); there is no
per-file frames substitution for a failed video prep any more.

vcut_pass2_video_specifics.plan.md section 3.2/13 step 4: when that
VideoHandle also carries a ``cache_name`` (a per-file video CachedContent,
created once in orchestrate._prepare_video_inputs and shared with Pass 2),
Pass 1 rides that cache instead of sending the video inline -- only the
task text goes per-call, cached-input rate for the (paid-once) video.
``cache_name`` is None when creation failed or the clip was below the
model's min-cache-token floor; that file's call falls back to inline video,
a cost regression only, never a correctness one.

THIS MODULE SPENDS REAL MONEY once invoked (one Gemini call per file per
project), same disclosure as l3/ingest.py's own docstring.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

from app.services.llm.base import Block, image_block, text_block, video_file_block
from app.services.vcut.params import JOIN_GUARD_MS
from app.services.vcut.resolve import FilePlan, MomentFlag, MomentPlan
from app.services.vcut.subclip import is_near_join, map_sub_to_orig

logger = logging.getLogger(__name__)

# section 2.5: bounded pool so a large project's per-file calls run
# concurrently instead of serializing one-by-one.
_PASS1_MAX_WORKERS = 6
# A sane PER-FILE budget (not a project's) -- one file's moments are a
# small, bounded amount of JSON even for a busy clip, so the old project-
# wide 48000 is vastly more than any single file needs.
_PER_FILE_MAX_TOKENS = 8000

# vcut_pass2_rich.plan.md section 3.2: the shared cache's OWN system,
# baked in once at creation time (orchestrate._create_vcut_cache). Public
# (not ``_``-prefixed) so orchestrate.py creates the cache with this EXACT
# string. The shared cache is now used by Pass 2 / enrich ONLY -- Pass 1
# sends its per-file frames inline (see module docstring) -- but this
# constant stays here because orchestrate still builds the cache from
# build_frame_blocks + NEUTRAL_SYSTEM in this module.
NEUTRAL_SYSTEM = (
    "You are a video-analysis assistant. Follow the task in each turn "
    "exactly and answer only in the required schema."
)


@dataclass
class VideoHandle:
    """What pass1.py needs to send a file's uploaded non-speech clip as
    video input (pass1_video_input.plan.md section 5) -- a minimal
    projection of subclip.SubClip + the Files API upload result, built by
    orchestrate.py. ``segments`` is SubClip.segments verbatim (needed to
    remap the model's sub-clip-time t_ms back to original-file ms).
    ``cache_name`` (vcut_pass2_video_specifics.plan.md section 3.2): the
    per-file video CachedContent's resource name, or None when cache
    creation failed / the clip was below the model's min-cache-token floor
    -- Pass 1 (and later Pass 2) then send the video inline via
    ``file_uri`` instead, uncached (a cost regression, never a correctness
    one)."""
    file_uri: str
    segments: List[Tuple[int, int, int]]
    cache_name: Optional[str] = None


def _task_text(file_id: str, filename: str, *, video_mode: bool = False) -> str:
    if video_mode:
        intro = (
            "You are watching a continuous video clip assembled from this file's own\n"
            "raw footage, so an editor can later turn it into finished videos.\n\n"
            f"You are working on exactly ONE file right now: FILE {file_id} ({filename!r}).\n"
            "You are shown ONE CONTINUOUS CLIP assembled from this file's own footage; "
            "give each moment's t_ms as milliseconds from the START of THIS clip."
        )
    else:
        intro = (
            "You are watching sampled frames from raw video footage, in time order, so an\n"
            "editor can later turn it into finished videos.\n\n"
            f"You are working on exactly ONE file right now: FILE {file_id} ({filename!r})."
        )
    return intro + """

Your only job: mark every distinct MOMENT worth landing on -- a single thing
happening that an editor might want to cut to. For each moment, give:
  - t_ms: the timestamp of the HEART of the moment -- the single frame the cut
    would land on (the payoff).
  - shape: how the interest sits around that frame:
      build  -- it builds UP TO this frame (the run-up matters; land on this)
      settle -- this frame IS the peak, then it eases off (the after matters)
      both   -- interesting roughly equally before and after
  - summary: one short sentence naming what the moment is.

Mark as MANY moments as this file's footage holds -- do NOT group them, filter
them, or judge which are good; that happens later. If an action repeats
(several reps of the same thing), mark EACH occurrence as its own moment. Do
not describe boundaries, camera work, counts, or per-frame detail -- only each
moment's frame, its shape, and its one-line summary.
"""


class _MomentOut(BaseModel):
    t_ms: int
    shape: Literal["build", "settle", "both"] = "both"
    summary: str = ""


class _Pass1FileSchema(BaseModel):
    """The per-file wire schema (section 2.5): each call is scoped to ONE
    file (named in the prompt, not echoed by the model), so the response is
    a flat list of that file's moments -- no ``files``/``file_id`` wrapper.
    The merge sets ``FilePlan.file_id`` itself from the loop."""
    moments: List[_MomentOut] = []


def _find_enclosing_span(
    center_ms: float, spans: List[Tuple[int, int]],
) -> Optional[Tuple[int, int]]:
    for s, e in spans:
        if s <= center_ms <= e:
            return (s, e)
    if not spans:
        return None

    def _dist(sp: Tuple[int, int]) -> float:
        s, e = sp
        if center_ms < s:
            return s - center_ms
        if center_ms > e:
            return center_ms - e
        return 0.0

    return min(spans, key=_dist)


def _clamp_moment(m: _MomentOut, spans: List[Tuple[int, int]]) -> Optional[MomentFlag]:
    """Defensive clamp (unchanged idea from the old _clamp_loose_cut): a
    model-proposed t_ms that doesn't fall in any non-speech span for its
    file snaps to the nearest one -- VLM timestamps are approximate and
    occasionally drift a little outside the frames actually shown. A file
    with no non-speech spans at all (shouldn't happen; every flag came from
    a file that WAS sampled) drops the moment rather than guessing."""
    span = _find_enclosing_span(m.t_ms, spans)
    if span is None:
        return None
    s, e = span
    return MomentFlag(t_ms=int(max(s, min(e, m.t_ms))), shape=m.shape, summary=m.summary)


def build_frame_blocks(
    file_rows: List[Tuple[str, str, int]],
    non_speech_by_file: Dict[str, List[Tuple[int, int]]],
    images_by_key: Dict[Tuple[str, int], str],
) -> List[Block]:
    """The sampled-frame content, file by file (labeled header + one text/
    image pair per timestamp) -- used BOTH as the shared CachedContent's
    body (orchestrate.run_vcut_ingest, created once, for Pass 2 / enrich)
    and as this module's own per-call blocks (scoped to a single file).
    Public (not ``_``-prefixed) so orchestrate.py can build the exact same
    content for cache creation."""
    blocks: List[Block] = []
    for file_id, filename, duration_ms in file_rows:
        spans = non_speech_by_file.get(file_id) or []
        ts_list = sorted(ts for (fid, ts) in images_by_key if fid == file_id)
        blocks.append(text_block(
            f"=== FILE {file_id} ({filename!r}, duration {duration_ms}ms) ===\n"
            f"Non-speech spans (ms): {spans}\n"
            f"{len(ts_list)} sampled frame(s) follow, in order:"
        ))
        for ts in ts_list:
            blocks.append(text_block(f"t={ts}ms"))
            blocks.append(image_block(images_by_key[(file_id, ts)]))
    return blocks


def _moments_from_video(
    moments: List["_MomentOut"], video_handle: VideoHandle, spans: List[Tuple[int, int]],
) -> List[MomentFlag]:
    """Post-process a video-mode call's raw moments (section 5): drop any
    landing on a concat seam (splice artifact, not real content), remap the
    survivors from sub-clip ms to original-file ms, drop any that map to
    nowhere (shouldn't happen for an in-range t, but defensive), then run
    the SAME _clamp_moment safety net the frames path already uses."""
    flags: List[MomentFlag] = []
    for m in moments:
        if is_near_join(m.t_ms, video_handle.segments, JOIN_GUARD_MS):
            continue
        orig_ms = map_sub_to_orig(m.t_ms, video_handle.segments)
        if orig_ms is None:
            continue
        flag = _clamp_moment(_MomentOut(t_ms=orig_ms, shape=m.shape, summary=m.summary), spans)
        if flag is not None:
            flags.append(flag)
    return flags


def _run_pass1_for_file(
    file_id: str,
    filename: str,
    duration_ms: int,
    non_speech_by_file: Dict[str, List[Tuple[int, int]]],
    images_by_key: Dict[Tuple[str, int], str],
    *,
    model: str,
    video_handle: Optional[VideoHandle] = None,
) -> Tuple[FilePlan, Dict[str, int]]:
    """One file's Pass 1 call (section 2.5) -> (FilePlan, usage). Sends
    video when ``video_handle`` is given (pass1_video_input.plan.md Phase
    1) -- riding that file's per-file video CachedContent
    (vcut_pass2_video_specifics.plan.md section 3.2/13 step 4) when
    ``video_handle.cache_name`` is set (shared with Pass 2: only the task
    text goes per-call, the cache already carries the video + a neutral
    system), else sending the video inline via ``video_file_block`` (no
    cache -- creation failed or the clip was below the model's min-cache-
    token floor). Otherwise sends THIS file's sampled frames inline. Runs
    on a worker thread from ``run_pass1``'s pool -- raises on persistent
    failure (schema validation failed twice, or any other exception),
    which now propagates through the caller and aborts the whole run (no
    per-file masking). A file that legitimately yields zero moments returns
    an empty FilePlan (nonempty is NOT required per call)."""
    from app.config import get_settings
    from app.services.llm.ingest_gemini import complete_gemini

    spans = non_speech_by_file.get(file_id) or []

    if video_handle is not None:
        settings = get_settings()
        task_text = _task_text(file_id, filename, video_mode=True)
        if video_handle.cache_name:
            system = NEUTRAL_SYSTEM
            video_blocks: List[Block] = [text_block(task_text)]
        else:
            system = task_text
            video_blocks = [video_file_block(video_handle.file_uri, fps=settings.vcut_video_fps)]
        completion = complete_gemini(
            system, video_blocks, _Pass1FileSchema,
            model=model, max_tokens=_PER_FILE_MAX_TOKENS, thinking="low",
            media_resolution=settings.vcut_video_media_resolution,
            cached_content=video_handle.cache_name,
        )
        parsed = _Pass1FileSchema.model_validate(completion.data)
        flags = _moments_from_video(parsed.moments, video_handle, spans)
        return FilePlan(file_id=file_id, flags=flags), completion.usage

    blocks = build_frame_blocks(
        [(file_id, filename, duration_ms)],
        {file_id: spans},
        {(fid, ts): img for (fid, ts), img in images_by_key.items() if fid == file_id},
    )
    completion = complete_gemini(
        _task_text(file_id, filename), blocks, _Pass1FileSchema,
        model=model, max_tokens=_PER_FILE_MAX_TOKENS, thinking="low",
    )
    parsed = _Pass1FileSchema.model_validate(completion.data)
    flags = [flag for flag in (
        _clamp_moment(m, spans) for m in parsed.moments
    ) if flag is not None]
    return FilePlan(file_id=file_id, flags=flags), completion.usage


def run_pass1(
    file_rows: List[Tuple[str, str, int]],
    non_speech_by_file: Dict[str, List[Tuple[int, int]]],
    images_by_key: Dict[Tuple[str, int], str],
    *,
    cached_content: Optional[str] = None,
    video_by_file: Optional[Dict[str, VideoHandle]] = None,
) -> Tuple[MomentPlan, Dict[str, int]]:
    """``file_rows``: (file_id, filename, duration_ms) for every file in the
    project. Fans out ONE complete_gemini call PER FILE (section 2.5) on a
    bounded thread pool -- independent calls, no reason to serialize. A file
    present in ``video_by_file`` sends that file's uploaded non-speech clip
    (pass1_video_input.plan.md Phase 1); every other file sends its sampled
    frames inline (the "frames"-mode path). NO FALLBACK: a file whose call
    fails persistently (retries exhausted) propagates and aborts the whole
    run -- a broken file contributing an empty FilePlan would silently hide
    missing cuts. A file that legitimately yields zero moments still
    contributes an empty FilePlan (that is not an error). Results are merged
    in ``file_rows`` order regardless of completion order.

    ``cached_content`` (the shared frame CachedContent's resource name) is
    accepted for call-site compatibility but IGNORED for Pass 1's own
    content: Pass 1 sends per-file frames/video inline (module docstring),
    and the shared cache is retained solely for Pass 2 / enrich.

    Returns (MomentPlan, usage) -- usage is the raw dict
    ingest_store.accumulate_pass2_usage already expects (input_tokens/
    output_tokens/cache_read_input_tokens/cache_creation_input_tokens),
    summed across every file's call (missing keys treated as 0)."""
    from app.config import get_settings

    settings = get_settings()
    model = settings.vcut_pass1_model
    video_by_file = video_by_file or {}

    plans_by_file: Dict[str, FilePlan] = {}
    usages: List[Dict[str, int]] = []
    with ThreadPoolExecutor(max_workers=min(_PASS1_MAX_WORKERS, len(file_rows)) or 1) as pool:
        future_to_file = {
            pool.submit(
                _run_pass1_for_file, file_id, filename, duration_ms,
                non_speech_by_file, images_by_key, model=model,
                video_handle=video_by_file.get(file_id),
            ): file_id
            for file_id, filename, duration_ms in file_rows
        }
        for future, file_id in future_to_file.items():
            # NO FALLBACK: a file's persistent failure (retries exhausted)
            # propagates and aborts the whole run. A broken file contributing
            # 0 flags would silently hide missing cuts, so we refuse it.
            plan, usage = future.result()
            usages.append(usage)
            plans_by_file[file_id] = plan

    files = [plans_by_file[file_id] for file_id, _name, _dur in file_rows]
    total_usage: Dict[str, int] = {}
    for usage in usages:
        for key, value in usage.items():
            total_usage[key] = total_usage.get(key, 0) + value

    return MomentPlan(files=files), total_usage
