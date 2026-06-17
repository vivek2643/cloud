"""
The orchestrator's tool belt: neutral tool specs + the executor that binds them
to one editing session.

Contract (the two-brain split): every tool here is deterministic. The model
chooses WHICH tool with WHAT intent; the engine decides exact frames and
reports objective numbers back. Tool results are compact JSON strings -- they
re-enter the model's context every iteration, so brevity is a cost feature.

`ask_user` and `finalize` are *terminal* tools: the loop runner watches for
them and ends the run (pausing or completing the thread) instead of looping.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from app.services.l3 import angle_menu, engine, focus, framing, layers, principles, score_span
from app.services.l3 import sync as sync_mod
from app.services.l3.catalog import ClipSummary
from app.services.l3.content import ClipContent, OverlapPair, content_coverage
from app.services.l3.engine import ClipGrids

logger = logging.getLogger(__name__)

TERMINAL_TOOLS = {"ask_user", "finalize"}


# --------------------------------------------------------------------------
# Session: the working state one agent run mutates
# --------------------------------------------------------------------------

@dataclass
class EditSession:
    thread_id: str
    file_ids: List[str]
    catalog: List[ClipSummary]
    document: Dict[str, Any] = field(default_factory=dict)
    content: List[ClipContent] = field(default_factory=list)
    overlap: List[OverlapPair] = field(default_factory=list)
    synced: List[Any] = field(default_factory=list)   # sync.VerifiedAngle[]
    _grids: Dict[str, ClipGrids] = field(default_factory=dict)
    _align_cache: Dict[tuple, Any] = field(default_factory=dict)
    _perceptions: Optional[Dict[str, dict]] = None

    def __post_init__(self) -> None:
        self.document.setdefault("brief", {})
        self.document.setdefault("format", {"aspect": layers.DEFAULT_ASPECT})
        self.document.setdefault("spine", None)
        self.document.setdefault("principles", [])
        self.document.setdefault("outline", [])
        self.document.setdefault("timeline", [])
        self.document.setdefault("operations", [])
        self.document.setdefault("open_questions", [])
        self.document.setdefault("diagnostics", {})

    def grids(self, file_id: str) -> ClipGrids:
        if file_id not in self._grids:
            self._grids[file_id] = engine.load_grids(file_id)
        return self._grids[file_id]

    def aligned(self, spine_fid: str, angle_fid: str):
        """Verified full-clip alignment of an angle to a spine clip (memoized).
        Returns an AlignResult (offset = the angle's start in the spine's clock)
        or None when they are not a reliable synced pair. The offset is global
        for the pair, so one verification serves every pick_angle on it."""
        key = (spine_fid, angle_fid)
        if key not in self._align_cache:
            self._align_cache[key] = sync_mod.align_clips(spine_fid, angle_fid)
        return self._align_cache[key]

    def perceptions_map(self) -> Dict[str, dict]:
        """L2 perception per clip in scope (loaded once, shared by the angle menu)."""
        if self._perceptions is None:
            from app.services.l3.catalog import load_perceptions
            self._perceptions = load_perceptions(self.file_ids)
        return self._perceptions

    def angle_offsets(self, spine_fid: str, angle_fids: List[str]) -> Dict[str, int]:
        """For each candidate angle, its verified offset in the spine's clock
        (angle start in spine time), dropping any pair that doesn't lock."""
        out: Dict[str, int] = {}
        for fid in angle_fids:
            if fid == spine_fid:
                continue
            al = self.aligned(spine_fid, fid)
            if al is not None:
                out[fid] = al.offset_ms
        return out

    def synced_with(self, spine_fid: str) -> List[str]:
        """The other member of every verified synced pair that includes this clip."""
        out: List[str] = []
        for v in self.synced:
            if v.file_a == spine_fid:
                out.append(v.file_b)
            elif v.file_b == spine_fid:
                out.append(v.file_a)
        return out

    def grids_by_file(self) -> Dict[str, ClipGrids]:
        for fid in {s["file_id"] for s in self.document["timeline"]}:
            self.grids(fid)
        return self._grids

    def durations(self) -> Dict[str, int]:
        """file_id -> source duration (ms), used to clamp split-edit audio
        extensions to real footage."""
        out: Dict[str, int] = {}
        for fid in self.file_ids:
            try:
                out[fid] = self.grids(fid).duration_ms
            except Exception:
                pass
        return out

    def resolved(self) -> layers.ResolvedTimeline:
        return layers.resolve(self.document, self.durations())

    def find_segment(self, seg_id: str) -> Optional[dict]:
        for s in self.document["timeline"]:
            if s["seg_id"] == seg_id:
                return s
        return None


# --------------------------------------------------------------------------
# Tool specs (neutral; match Anthropic's tool shape)
# --------------------------------------------------------------------------

def _spec(name: str, description: str, properties: dict, required: List[str]) -> dict:
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


_AXIS = {
    "type": "string",
    "enum": ["speech", "action", "music", "visual", "any"],
    "description": "which cut-cost channels dominate at this seam",
}

TOOL_SPECS: List[dict] = [
    _spec(
        "read_clip",
        "Full footage log for one clip: the L2 perception document (events, "
        "persons, reactions, gaze, camera craft, speaking spans) plus duration. "
        "Call this before using a clip's material; the catalog is only a teaser.",
        {"file_id": {"type": "string"}},
        ["file_id"],
    ),
    _spec(
        "query_seams",
        "Ranked clean cut candidates near a timestamp in one clip, from the "
        "deterministic cost grids (0=ideal..1=forbidden, dirty>0.45). Use to "
        "scout where a beat can start/end cleanly before adding a segment.",
        {
            "file_id": {"type": "string"},
            "around_ms": {"type": "integer"},
            "axis": _AXIS,
            "window_ms": {"type": "integer", "description": "search radius, default 2000"},
        },
        ["file_id", "around_ms"],
    ),
    _spec(
        "set_brief",
        "Record your interpretation of the user's brief (goal, target duration, "
        "tone, platform, constraints, and delivery ASPECT). Defaults you assumed "
        "belong in `assumptions` so they can become questions. Set `aspect` to "
        "the DELIVERY frame shape: 'landscape' (16:9, default), 'portrait' (9:16 "
        "-- reels / shorts / tiktok / vertical), or 'square' (1:1). Infer it from "
        "the brief (words like reel/short/vertical/9:16 => portrait); if the user "
        "didn't say and it materially changes the edit, ask_user instead of "
        "guessing.",
        {
            "goal": {"type": "string"},
            "target_duration_s": {"type": "number"},
            "tone": {"type": "string"},
            "platform": {"type": "string"},
            "aspect": {
                "type": "string",
                "enum": ["landscape", "portrait", "square"],
                "description": "delivery frame shape (default landscape)",
            },
            "motion_style": {
                "type": "string",
                "enum": ["static", "punch_in", "push_in", "follow"],
                "description": "how the frame ZOOMS/MOVES. Default 'static' (no "
                "zoom). 'punch_in' = held slightly tighter; 'push_in' = slow zoom "
                "in over each shot; 'follow' = stay tight and pan to keep the "
                "subject in frame. This is a user CHOICE -- ask_user, don't assume.",
            },
            "motion_feel": {
                "type": "string",
                "enum": ["snappy", "glide"],
                "description": "easing for motion_style: 'snappy' (linear) or "
                "'glide' (smooth ease). Only matters when motion_style moves.",
            },
            "constraints": {"type": "array", "items": {"type": "string"}},
            "assumptions": {"type": "array", "items": {"type": "string"}},
        },
        ["goal"],
    ),
    _spec(
        "set_spine",
        "Declare the EDIT SPINE before building the timeline: the load-bearing "
        "through-line every other choice serves. It names, per time-ordered "
        "region, which channel is LOCKED (irreplaceable) vs FREE (coverable / "
        "scoreable). Decoupling A/V is the privileged move -- default to "
        "kind='sync' (both locked, atomic) unless the brief or footage justifies "
        "freeing a channel. kinds: dialogue (audio locked, video free -> B-roll "
        "covers picture, cut on dialogue seams); music (music bed locked, video "
        "free -> coverage cut to beats/sections); visual (VIDEO locked -- "
        "on-screen text / demo / reveal / performance -- audio free to score, cut "
        "on action/visual); sync (BOTH locked -- punchline+face, sync-sound; do "
        "not split); other (escape hatch -- set label + locked_channels). Mark "
        "do-not-cover spans (on-screen text, key reveals) as protected_windows. "
        "One region for most edits; multiple only when the edit shifts mode "
        "(e.g. montage hook -> testimonial).",
        {
            "regions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "enum": ["dialogue", "music", "visual", "sync", "other"],
                        },
                        "label": {
                            "type": "string",
                            "description": "human label; REQUIRED when kind='other'",
                        },
                        "locked_channels": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["video", "audio"]},
                            "description": "irreplaceable channels. dialogue/music=[audio]; "
                                           "visual=[video]; sync=[video,audio]",
                        },
                        "source_file_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "clip(s)/track forming this region's spine "
                                           "(the VO/interview clip, or the music file)",
                        },
                        "protected_windows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_id": {"type": "string"},
                                    "start_ms": {"type": "integer"},
                                    "end_ms": {"type": "integer"},
                                    "reason": {"type": "string"},
                                },
                                "required": ["file_id", "start_ms", "end_ms"],
                            },
                            "description": "do-not-cover spans inside an otherwise-free region",
                        },
                        "rationale": {
                            "type": "string",
                            "description": "why this spine, citing the brief + footage evidence",
                        },
                    },
                    "required": ["kind", "locked_channels", "rationale"],
                },
            }
        },
        ["regions"],
    ),
    _spec(
        "set_principles",
        "Declare the editorial PRINCIPLES -- weighted style tendencies (NOT "
        "rules) that bias how the cut is made within the spine's freedom. Each "
        "is {id, weight 0..1, scope}. scope='global' (default) or a spine-region "
        "label to override there. Set only the knobs you want to push; the rest "
        "run at sensible defaults. Common: favor_speaker, reward_reaction, "
        "shot_variety, pace, anti_metronome, hook_first, tighten_dead_air.",
        {
            "principles": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "a known principle id"},
                        "weight": {"type": "number", "description": "tendency strength, 0..1"},
                        "scope": {"type": "string", "description": "'global' or a spine-region label"},
                    },
                    "required": ["id", "weight"],
                },
            }
        },
        ["principles"],
    ),
    _spec(
        "set_outline",
        "Replace the beat outline (the narrative skeleton). Keep 2-6 beats; "
        "each maps to >=1 timeline segments later.",
        {
            "beats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "beat_id": {"type": "string"},
                        "purpose": {"type": "string", "description": "hook/setup/payoff/outro/..."},
                        "intent": {"type": "string"},
                        "target_s": {"type": "number"},
                    },
                    "required": ["beat_id", "purpose", "intent"],
                },
            }
        },
        ["beats"],
    ),
    _spec(
        "add_segment",
        "Add a span of one clip to the timeline. You give rough in/out; the "
        "engine snaps both ends to the cleanest nearby seams and returns exact "
        "frames + costs. position=-1 appends.",
        {
            "file_id": {"type": "string"},
            "in_ms": {"type": "integer", "description": "rough in point; will be snapped"},
            "out_ms": {"type": "integer", "description": "rough out point; will be snapped"},
            "axis": _AXIS,
            "beat_id": {"type": "string"},
            "content": {"type": "string", "description": "what is on screen in this span"},
            "rationale": {"type": "string", "description": "why this material, here"},
            "priority": {"type": "integer", "description": "1=never trim .. 5=trim first (default 3)"},
            "position": {"type": "integer", "description": "index in timeline; -1/omit = append"},
        },
        ["file_id", "in_ms", "out_ms"],
    ),
    _spec(
        "update_segment",
        "Re-cut an existing segment (new rough in/out get re-snapped) and/or "
        "update its metadata.",
        {
            "seg_id": {"type": "string"},
            "in_ms": {"type": "integer"},
            "out_ms": {"type": "integer"},
            "content": {"type": "string"},
            "rationale": {"type": "string"},
            "priority": {"type": "integer"},
            "beat_id": {"type": "string"},
        },
        ["seg_id"],
    ),
    _spec(
        "remove_segment",
        "Delete a segment from the timeline.",
        {"seg_id": {"type": "string"}},
        ["seg_id"],
    ),
    _spec(
        "move_segment",
        "Reorder: move a segment to a new index in the timeline.",
        {"seg_id": {"type": "string"}, "new_index": {"type": "integer"}},
        ["seg_id", "new_index"],
    ),
    _spec(
        "place_video",
        "Lay another clip's PICTURE over a program-time range while the spine "
        "AUDIO keeps playing underneath (B-roll / cutaway over an UNSYNCED clip). "
        "You give the covering clip + a rough source point and the rough program "
        "range to cover; the engine snaps the clip's entry to its own visual "
        "grid, snaps the range edges to the spine's audio seams, and refuses if "
        "the spine locks the video there or it hits a protected window. Requires "
        "a spine that frees the video channel (dialogue/music). Picture is "
        "full-frame (split-screen/PiP is a later layout).",
        {
            "file_id": {"type": "string", "description": "the covering clip"},
            "src_in_ms": {"type": "integer", "description": "rough entry point in the covering clip"},
            "from_ms": {"type": "integer", "description": "program time the coverage starts"},
            "to_ms": {"type": "integer", "description": "program time the coverage ends"},
            "rationale": {"type": "string", "description": "why cut away here, to this"},
        },
        ["file_id", "src_in_ms", "from_ms", "to_ms"],
    ),
    _spec(
        "pick_angle",
        "Cut the spine PICTURE to a SYNCED SECOND ANGLE over a program range -- a "
        "NORMAL picture cut (e.g. cut to the other camera in a 2-camera interview: "
        "the speaker, or the listener reacting / asking). NOT B-roll. The chosen "
        "clip must be a VERIFIED synced angle of the spine clip under from_ms -- "
        "the engine calls align_clips itself, derives the angle's source time from "
        "the exact offset (frame-aligned), keeps the spine AUDIO playing "
        "underneath, and snaps the range to spine seams. Refused (use place_video) "
        "if the pair isn't a reliable sync. Legal on ANY spine kind.",
        {
            "file_id": {"type": "string", "description": "the synced angle to cut to"},
            "from_ms": {"type": "integer", "description": "program time the angle starts"},
            "to_ms": {"type": "integer", "description": "program time the angle ends"},
            "rationale": {"type": "string", "description": "why this angle here (speaker / reaction / variety)"},
        },
        ["file_id", "from_ms", "to_ms"],
    ),
    _spec(
        "place_audio",
        "Lay an audio layer over a program-time range: a music bed, ambience, or "
        "an SFX (role='music'|'sfx'), or REPLACE the spine dialogue with cleaner "
        "audio (role='dialogue', kind='replace'). Beds duck under live dialogue "
        "automatically when you pass a negative duck_db. The engine snaps a "
        "musical source's entry to its beat/phrase grid and the range edges to "
        "the spine seams. Replacing locked spine audio is refused.",
        {
            "file_id": {"type": "string", "description": "the audio source (e.g. a music clip)"},
            "role": {"type": "string", "enum": ["music", "sfx", "dialogue"]},
            "from_ms": {"type": "integer"},
            "to_ms": {"type": "integer"},
            "src_in_ms": {"type": "integer", "description": "rough entry in the source; default 0"},
            "gain_db": {"type": "number", "description": "layer gain in dB (default 0)"},
            "duck_db": {"type": "number", "description": "how much to duck this bed under dialogue, e.g. -12"},
            "kind": {"type": "string", "enum": ["bed", "replace"], "description": "default bed"},
            "rationale": {"type": "string"},
        },
        ["file_id", "role", "from_ms", "to_ms"],
    ),
    _spec(
        "split_edit",
        "Make a J- or L-cut at a spine seam: offset the AUDIO cut from the VIDEO "
        "cut. Positive audio_offset_ms = L-cut (the previous clip's audio lingers "
        "over the next clip's picture); negative = J-cut (the next clip's audio "
        "leads in under the current picture). The audio boundary snaps to a "
        "dialogue seam. Refused on a sync-locked spine.",
        {
            "seam_seg_id": {"type": "string", "description": "the LATER segment at the seam (cut is at its start)"},
            "audio_offset_ms": {"type": "integer", "description": "+L-cut / -J-cut, in ms"},
            "rationale": {"type": "string"},
        },
        ["seam_seg_id", "audio_offset_ms"],
    ),
    _spec(
        "set_level",
        "Mix automation: set a gain (dB) or mute on an audio role over a program "
        "range. Use to balance a bed against dialogue beyond the auto-duck, or "
        "silence a channel.",
        {
            "role": {"type": "string", "enum": ["dialogue", "music", "sfx"]},
            "from_ms": {"type": "integer"},
            "to_ms": {"type": "integer"},
            "gain_db": {"type": "number"},
            "mute": {"type": "boolean"},
        },
        ["role", "from_ms", "to_ms"],
    ),
    _spec(
        "remove_operation",
        "Delete a previously added layer operation (coverage / audio / split / "
        "level) by its op_id.",
        {"op_id": {"type": "string"}},
        ["op_id"],
    ),
    _spec(
        "timeline_status",
        "Objective health report of the current timeline: total duration, "
        "per-segment durations and seam costs, jump-cut/short-segment warnings, "
        "plus the resolved A/V layers (coverage %, audio roles, split edits). "
        "Call after edits and before finalize.",
        {},
        [],
    ),
    _spec(
        "fit_duration",
        "Deterministically trim the timeline to a target duration: shrinks "
        "lowest-priority segments first, moving out-points onto clean seams. "
        "Only shrinks -- if under target, add material yourself.",
        {
            "target_s": {"type": "number"},
            "tolerance_ms": {"type": "integer", "description": "default 500"},
        },
        ["target_s"],
    ),
    _spec(
        "score_span",
        "Get an objective quality scorecard for a specific span of one clip: "
        "metrics (word pace, fillers, pauses, gaze-to-camera, loudness) and the "
        "perception's localized quality notes (energy/fluency/naturalness/"
        "technical). Use when the CONTENT OVERLAP map shows two clips cover the "
        "same content (competing takes) and you want to choose the better "
        "delivery: score each candidate span, decide on the transcript FIRST "
        "(most complete, correct, fewest fillers/stumbles), then weigh the "
        "metrics by the brief before add_segment-ing the winner.",
        {
            "file_id": {"type": "string", "description": "the clip to score"},
            "in_ms": {"type": "integer", "description": "span start in the clip"},
            "out_ms": {"type": "integer", "description": "span end in the clip"},
        },
        ["file_id", "in_ms", "out_ms"],
    ),
    _spec(
        "align_clips",
        "Check whether two clips are the SAME MOMENT recorded simultaneously "
        "(multicam / a separate recorder) by comparing their audio, and if so "
        "return their exact time offset. Use to verify before treating two clips "
        "as interchangeable angles, or before cutting between them as one "
        "continuous moment. Returns matched=true with offset_ms (file_b's start "
        "in file_a's clock), a confidence (0..1), and the overlap length; returns "
        "matched=false when they are NOT simultaneous, one clip is silent, or the "
        "shared audio is too short to be sure -- in which case they are distinct "
        "takes/material, not synced angles.",
        {
            "file_a": {"type": "string"},
            "file_b": {"type": "string"},
            "from_ms": {"type": "integer", "description": "optional: restrict the check to a region of file_a"},
            "to_ms": {"type": "integer", "description": "optional: end of that file_a region"},
        },
        ["file_a", "file_b"],
    ),
    _spec(
        "read_angles",
        "The per-moment ANGLE MENU for a synced multicam region: for each focus "
        "interval (speaker turn / action beat / music section of the spine clip) "
        "it reports what EACH verified angle shows -- whether the visible person "
        "is the SPEAKER or a LISTENER, the shot size, the strongest reaction, and "
        "gaze. Use this BEFORE pick_angle so your cuts follow the focus (cut to "
        "whoever is speaking; cut to the listener on a genuine reaction or a long "
        "hold) instead of riding one camera or cutting for blind variety. Times "
        "are in the spine clip's own clock. Pass the spine clip + the region; "
        "angle_file_ids is optional (defaults to every verified angle of the "
        "spine clip). Only verified synced angles appear.",
        {
            "file_id": {"type": "string", "description": "the spine clip (the one whose audio plays)"},
            "from_ms": {"type": "integer", "description": "region start in the spine clip's clock"},
            "to_ms": {"type": "integer", "description": "region end in the spine clip's clock"},
            "angle_file_ids": {
                "type": "array", "items": {"type": "string"},
                "description": "optional: which angles to compare; default = all verified angles",
            },
        },
        ["file_id", "from_ms", "to_ms"],
    ),
    _spec(
        "read_framing",
        "What the AUTOMATIC reframe will do for a clip (or a range of it) at the "
        "current delivery aspect: whether it crops to fill (only non-landscape "
        "aspects reframe), the orthogonal rotation to set it upright, and the "
        "FOCUS point it will keep in frame (from who is speaking / where the "
        "action is / the main subject). Read-only facts -- the framing is applied "
        "automatically; call this only to see/sanity-check it. Clips logged "
        "before spatial perception have no focus and stay centered.",
        {
            "file_id": {"type": "string"},
            "from_ms": {"type": "integer", "description": "optional range start (clip clock); default 0"},
            "to_ms": {"type": "integer", "description": "optional range end; default the clip end"},
        },
        ["file_id"],
    ),
    _spec(
        "ask_user",
        "Pause and ask the user. Use for genuine forks (length, ending, tone, "
        "include/exclude) -- not for things the footage answers. ALWAYS have a "
        "complete draft on the timeline before calling this; every question "
        "needs a default so the draft stands if the user never answers.",
        {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "q_id": {"type": "string"},
                        "question": {"type": "string"},
                        "options": {"type": "array", "items": {"type": "string"}},
                        "default": {"type": "string"},
                        "why": {"type": "string", "description": "what hinges on this"},
                    },
                    "required": ["q_id", "question", "default"],
                },
            }
        },
        ["questions"],
    ),
    _spec(
        "finalize",
        "Complete this run: attach a human-readable summary of the plan and "
        "what you chose/assumed. The current timeline becomes the new document "
        "version shown to the user.",
        {
            "summary": {"type": "string"},
            "notes": {"type": "array", "items": {"type": "string"},
                      "description": "caveats, weak spots, suggested next tweaks"},
        },
        ["summary"],
    ),
]


# --------------------------------------------------------------------------
# Executor
# --------------------------------------------------------------------------

def _j(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _read_clip(session: EditSession, file_id: str) -> str:
    from app.services.l3.catalog import _pg_conn

    if file_id not in session.file_ids:
        return _j({"error": "file_id not in this thread's scope"})
    with _pg_conn() as conn:
        row = conn.execute(
            """
            select cp.perception, coalesce(f.duration_seconds, 0), f.name
              from files f
              left join clip_perception cp on cp.file_id = f.id
             where f.id = %s
            """,
            (file_id,),
        ).fetchone()
    if not row:
        return _j({"error": "clip not found"})
    perception, duration_s, name = row
    doc = perception if isinstance(perception, dict) else (
        json.loads(perception) if perception else None
    )
    if not doc:
        return _j({
            "file_id": file_id, "name": name, "duration_s": float(duration_s),
            "note": "no L2 perception for this clip; only cost grids available",
        })

    # Strip nulls/empties recursively: the full doc is verbose and re-enters
    # context on every later iteration.
    def slim(v):
        if isinstance(v, dict):
            return {k: slim(x) for k, x in v.items() if x not in (None, [], {}, "")}
        if isinstance(v, list):
            return [slim(x) for x in v]
        return v

    return _j({
        "file_id": file_id, "name": name, "duration_s": float(duration_s),
        "perception": slim(doc),
    })


def _op_id() -> str:
    return f"op{uuid.uuid4().hex[:6]}"


def _spine_kind_for(doc: Dict[str, Any], file_id: str) -> Optional[str]:
    """The set_spine kind of the region this clip anchors (so the angle menu's
    focus matches the declared spine). None when no spine names this clip."""
    spine = doc.get("spine") or {}
    for region in spine.get("regions") or []:
        if file_id in (region.get("source_file_ids") or []):
            return region.get("kind")
    # A single-region spine governs everything even if it didn't list the file.
    regions = spine.get("regions") or []
    if len(regions) == 1:
        return regions[0].get("kind")
    return None


def _snap_prog_to_spine(
    session: EditSession,
    spans: List["layers.SpineSpan"],
    prog_ms: int,
    axis_override: Optional[str] = None,
) -> tuple[int, Optional[float]]:
    """Snap a program-time instant to the nearest clean seam on the spine clip
    underneath it (its dialogue/beat grid). Returns (program_ms, seam_cost)."""
    m = layers.prog_to_source(spans, prog_ms)
    if not m:
        return prog_ms, None
    seg, src, span = m
    grids = session.grids(seg["file_id"])
    axis = axis_override or seg.get("axis", "any")
    snapped = engine.snap_cut(grids, src, axis)
    new_prog = span.prog_start_ms + (snapped["ts_ms"] - int(seg["in_ms"]))
    new_prog = max(span.prog_start_ms, min(span.prog_end_ms, new_prog))
    return new_prog, snapped.get("cost")


def _full_status(session: EditSession) -> Dict[str, Any]:
    """Spine health report (engine) + the resolved A/V layer summary."""
    status = engine.timeline_status(session.document["timeline"])
    r = session.resolved()
    cover_ms = sum(
        v.prog_end_ms - v.prog_start_ms for v in r.video_layers if v.kind == "coverage"
    )
    angle_ms = sum(
        v.prog_end_ms - v.prog_start_ms for v in r.video_layers if v.kind == "angle"
    )
    beds = [a for a in r.audio_layers if a.kind in ("bed", "sfx")]
    ops = session.document.get("operations", [])
    status["layers"] = {
        "program_ms": r.duration_ms,
        "coverage_ms": cover_ms,
        "coverage_pct": round(100.0 * cover_ms / r.duration_ms, 1) if r.duration_ms else 0.0,
        # Angle switches re-point the spine picture; tracked apart from coverage.
        "angle_switches": sum(1 for o in ops if o.get("type") == "pick_angle"),
        "angle_ms": angle_ms,
        "angle_pct": round(100.0 * angle_ms / r.duration_ms, 1) if r.duration_ms else 0.0,
        "video_layer_count": len(r.video_layers),
        "audio_roles": sorted({a.role for a in r.audio_layers}),
        "audio_bed_count": len(beds),
        "split_edits": sum(1 for o in ops if o.get("type") == "split_edit"),
        "operations": len(ops),
    }
    if session.content:
        status["content_coverage"] = content_coverage(
            session.content, session.overlap, session.document
        )
    return status


def execute_tool(session: EditSession, name: str, args: Dict[str, Any]) -> str:
    """Run one tool against the session; returns the JSON string fed back to
    the model. Raises nothing: errors return as structured results so the
    agent can correct course."""
    try:
        return _execute(session, name, args)
    except Exception as e:  # noqa: BLE001 - the agent handles its own errors
        logger.exception("L3 tool %s failed", name)
        return _j({"error": f"{type(e).__name__}: {e}"})


def _execute(session: EditSession, name: str, args: Dict[str, Any]) -> str:
    doc = session.document

    if name == "read_clip":
        return _read_clip(session, args["file_id"])

    if name == "query_seams":
        grids = session.grids(args["file_id"])
        seams = engine.query_seams(
            grids, int(args["around_ms"]),
            args.get("axis", "any"), int(args.get("window_ms", 2000)),
        )
        return _j({"seams": seams})

    if name == "set_brief":
        doc["brief"] = {k: v for k, v in args.items() if v is not None}
        # The delivery aspect lives on the document `format` (where resolve +
        # render + preview read it), mirrored from the brief.
        aspect = args.get("aspect")
        if aspect in layers.ASPECTS:
            doc.setdefault("format", {})["aspect"] = aspect
        # Motion style/feel (the one user-chosen zoom knob) also live on `format`,
        # where the framing pass reads them to build per-segment motion.
        style = args.get("motion_style")
        if style in framing.MOTION_STYLES:
            doc.setdefault("format", {})["motion_style"] = style
        feel = args.get("motion_feel")
        if feel in framing.MOTION_FEELS:
            doc.setdefault("format", {})["motion_feel"] = feel
        return _j({"ok": True, "brief": doc["brief"],
                   "aspect": layers.aspect_of(doc),
                   "motion_style": (doc.get("format") or {}).get("motion_style", "static")})

    if name == "set_spine":
        regions = args.get("regions") or []
        doc["spine"] = {"regions": regions}
        return _j({
            "ok": True,
            "region_count": len(regions),
            "spine": [
                {"kind": r.get("kind"), "locked": r.get("locked_channels", [])}
                for r in regions
            ],
        })

    if name == "set_principles":
        clean, errors = principles.normalize(args.get("principles") or [])
        doc["principles"] = clean
        result: Dict[str, Any] = {"ok": True, "principles": clean}
        if errors:
            result["ignored"] = errors
            result["known"] = sorted(principles.KNOWN)
        return _j(result)

    if name == "set_outline":
        doc["outline"] = args["beats"]
        return _j({"ok": True, "beat_count": len(doc["outline"])})

    if name == "add_segment":
        grids = session.grids(args["file_id"])
        seg = engine.make_segment(
            grids,
            int(args["in_ms"]), int(args["out_ms"]),
            args.get("axis", "any"),
            beat_id=args.get("beat_id"),
            content=args.get("content"),
            rationale=args.get("rationale"),
            priority=int(args.get("priority", 3)),
        )
        pos = int(args.get("position", -1))
        if pos < 0 or pos >= len(doc["timeline"]):
            doc["timeline"].append(seg)
        else:
            doc["timeline"].insert(pos, seg)
        return _j({"segment": seg, "total_s": engine.timeline_status(doc["timeline"])["total_s"]})

    if name == "update_segment":
        seg = session.find_segment(args["seg_id"])
        if seg is None:
            return _j({"error": "unknown seg_id"})
        grids = session.grids(seg["file_id"])
        axis = seg.get("axis", "any")
        if "in_ms" in args and args["in_ms"] is not None:
            snapped = engine.snap_cut(grids, int(args["in_ms"]), axis)
            seg["in_ms"], seg["cut_in_cost"] = snapped["ts_ms"], snapped["cost"]
        if "out_ms" in args and args["out_ms"] is not None:
            snapped = engine.snap_cut(grids, int(args["out_ms"]), axis)
            seg["out_ms"], seg["cut_out_cost"] = snapped["ts_ms"], snapped["cost"]
        for k in ("content", "rationale", "priority", "beat_id"):
            if args.get(k) is not None:
                seg[k] = args[k]
        return _j({"segment": seg})

    if name == "remove_segment":
        before = len(doc["timeline"])
        doc["timeline"] = [s for s in doc["timeline"] if s["seg_id"] != args["seg_id"]]
        if len(doc["timeline"]) == before:
            return _j({"error": "unknown seg_id"})
        return _j({"ok": True, "remaining": len(doc["timeline"])})

    if name == "move_segment":
        seg = session.find_segment(args["seg_id"])
        if seg is None:
            return _j({"error": "unknown seg_id"})
        doc["timeline"].remove(seg)
        idx = max(0, min(int(args["new_index"]), len(doc["timeline"])))
        doc["timeline"].insert(idx, seg)
        return _j({"ok": True, "order": [s["seg_id"] for s in doc["timeline"]]})

    if name == "place_video":
        fid = args["file_id"]
        if fid not in session.file_ids:
            return _j({"error": "file_id not in this thread's scope"})
        spans, total = layers.spine_spans(doc["timeline"])
        if not spans:
            return _j({"error": "no spine yet; add_segment the base timeline before covering it"})
        from_ms = max(0, min(int(args["from_ms"]), total))
        to_ms = max(0, min(int(args["to_ms"]), total))
        if to_ms <= from_ms:
            return _j({"error": "to_ms must be after from_ms"})
        reason = layers.coverage_conflicts(doc, spans, from_ms, to_ms)
        if reason:
            return _j({"error": reason})
        from_s, _ = _snap_prog_to_spine(session, spans, from_ms)
        to_s, _ = _snap_prog_to_spine(session, spans, to_ms)
        if to_s <= from_s:
            to_s = min(total, from_s + engine.MIN_SEGMENT_MS)
        prog_dur = to_s - from_s
        cg = session.grids(fid)
        warnings = []
        rough_src = args.get("src_in_ms")
        if rough_src is None:
            return _j({"error": "src_in_ms required (the rough entry point in the covering clip)"})
        snap_in = engine.snap_cut(cg, int(rough_src), "visual")
        src_in = snap_in["ts_ms"]
        src_out = src_in + prog_dur
        if snap_in.get("warning"):
            warnings.append(snap_in["warning"])
        if cg.duration_ms and src_out > cg.duration_ms:
            avail = cg.duration_ms - src_in
            src_out = cg.duration_ms
            to_s = from_s + max(0, avail)
            warnings.append("covering clip too short for the range; coverage shortened")
        if to_s - from_s < engine.MIN_SEGMENT_MS:
            warnings.append(f"coverage is only {to_s - from_s}ms after snapping")
        op = {
            "op_id": _op_id(), "type": "place_video",
            "source_file_id": fid, "src_in_ms": src_in, "src_out_ms": src_out,
            "from_ms": from_s, "to_ms": to_s,
            "layout": layers.DEFAULT_LAYOUT, "z": layers.Z_COVERAGE, "opacity": 1.0,
            "rationale": args.get("rationale"),
            "cut_in_cost": snap_in["cost"], "warnings": warnings,
        }
        doc["operations"].append(op)
        return _j({"operation": op, "covers_pct": round(100.0 * (to_s - from_s) / total, 1) if total else 0.0})

    if name == "pick_angle":
        fid = args["file_id"]
        if fid not in session.file_ids:
            return _j({"error": "file_id not in this thread's scope"})
        spans, total = layers.spine_spans(doc["timeline"])
        if not spans:
            return _j({"error": "no spine yet; add_segment the base timeline before picking angles"})
        from_ms = max(0, min(int(args["from_ms"]), total))
        to_ms = max(0, min(int(args["to_ms"]), total))
        if to_ms <= from_ms:
            return _j({"error": "to_ms must be after from_ms"})
        m = layers.prog_to_source(spans, from_ms)
        if not m:
            return _j({"error": "could not map program time to the spine here"})
        spine_fid = m[0]["file_id"]
        if spine_fid == fid:
            return _j({"error": "that clip is already the spine picture here"})
        # Verify the pair is a synced angle (airtight, on-demand, per-pair).
        al = session.aligned(spine_fid, fid)
        if al is None:
            return _j({"error": (
                f"{fid} is not a verified synced angle of the spine clip here -- "
                "align_clips found no reliable sync (different content, not "
                "simultaneous, or one clip is silent; lip-sync fallback is not "
                "available yet). Use place_video for unsynced coverage."
            )})
        reason = layers.angle_conflicts(doc, spans, from_ms, to_ms)
        if reason:
            return _j({"error": reason})
        from_s, _ = _snap_prog_to_spine(session, spans, from_ms)
        to_s, _ = _snap_prog_to_spine(session, spans, to_ms)
        if to_s <= from_s:
            to_s = min(total, from_s + engine.MIN_SEGMENT_MS)
        prog_dur = to_s - from_s
        m2 = layers.prog_to_source(spans, from_s)
        if not m2:
            return _j({"error": "could not map program time to the spine here"})
        _seg, src_spine, _span = m2
        # align_clips offset = the angle's start in the spine's clock, so the
        # same instant in the angle is (spine source - offset). Frame-aligned.
        rough_src = src_spine - al.offset_ms
        cg = session.grids(fid)
        if rough_src < 0 or (cg.duration_ms and rough_src > cg.duration_ms):
            return _j({"error": "the angle clip doesn't cover this moment (the sync "
                                "offset puts it outside the angle's footage)"})
        snap_in = engine.snap_cut(cg, int(rough_src), "visual")
        src_in = snap_in["ts_ms"]
        src_out = src_in + prog_dur
        warnings = []
        if snap_in.get("warning"):
            warnings.append(snap_in["warning"])
        if cg.duration_ms and src_out > cg.duration_ms:
            avail = cg.duration_ms - src_in
            src_out = cg.duration_ms
            to_s = from_s + max(0, avail)
            warnings.append("angle clip too short for the range; shortened")
        if to_s - from_s < engine.MIN_SEGMENT_MS:
            warnings.append(f"angle is only {to_s - from_s}ms after snapping")
        if al.confidence < 0.6:
            warnings.append(f"sync confidence {al.confidence:.2f}; angle may drift slightly")
        op = {
            "op_id": _op_id(), "type": "pick_angle",
            "source_file_id": fid, "src_in_ms": src_in, "src_out_ms": src_out,
            "from_ms": from_s, "to_ms": to_s,
            "layout": layers.DEFAULT_LAYOUT, "z": layers.Z_ANGLE,
            "sync_offset_ms": al.offset_ms, "sync_confidence": round(al.confidence, 3),
            "rationale": args.get("rationale"),
            "cut_in_cost": snap_in["cost"], "warnings": warnings,
        }
        doc["operations"].append(op)
        return _j({
            "operation": op,
            "angle_pct": round(100.0 * (to_s - from_s) / total, 1) if total else 0.0,
            "verified_offset_ms": al.offset_ms,
            "sync_confidence": round(al.confidence, 3),
        })

    if name == "place_audio":
        fid = args["file_id"]
        if fid not in session.file_ids:
            return _j({"error": "file_id not in this thread's scope"})
        role = args.get("role", "music")
        kind = args.get("kind", "bed")
        if kind == "replace" and not layers.audio_is_free(doc):
            return _j({"error": "spine locks the audio channel; cannot replace it (a bed is fine)"})
        spans, total = layers.spine_spans(doc["timeline"])
        if not spans:
            return _j({"error": "no spine yet; build the base timeline first"})
        from_ms = max(0, min(int(args["from_ms"]), total))
        to_ms = max(0, min(int(args["to_ms"]), total))
        if to_ms <= from_ms:
            return _j({"error": "to_ms must be after from_ms"})
        from_s, _ = _snap_prog_to_spine(session, spans, from_ms)
        to_s, _ = _snap_prog_to_spine(session, spans, to_ms)
        cg = session.grids(fid)
        # A musical source snaps its entry to the beat grid; otherwise as-is.
        snap = engine.snap_cut(cg, int(args.get("src_in_ms", 0)), "music")
        src_in = snap["ts_ms"]
        src_out = src_in + (to_s - from_s)
        warnings = []
        if cg.duration_ms and src_out > cg.duration_ms:
            src_out = cg.duration_ms
            warnings.append("audio source shorter than the range; not looped (loop is a later feature)")
        op = {
            "op_id": _op_id(), "type": "place_audio",
            "source_file_id": fid, "role": role, "audio_kind": kind,
            "from_ms": from_s, "to_ms": to_s,
            "src_in_ms": src_in, "src_out_ms": src_out,
            "gain_db": float(args.get("gain_db", 0.0)),
            "duck_db": float(args.get("duck_db", 0.0)),
            "rationale": args.get("rationale"), "warnings": warnings,
        }
        doc["operations"].append(op)
        return _j({"operation": op})

    if name == "split_edit":
        if layers.av_is_coupled(doc):
            return _j({"error": "spine keeps A/V coupled (sync / no spine); no split edits"})
        spans, _ = layers.spine_spans(doc["timeline"])
        idx = next((i for i, s in enumerate(spans) if s.seg["seg_id"] == args["seam_seg_id"]), None)
        if idx is None:
            return _j({"error": "unknown seam_seg_id"})
        if idx == 0:
            return _j({"error": "no seam before the first segment"})
        video_boundary = spans[idx].prog_start_ms
        audio_boundary, cost = _snap_prog_to_spine(
            session, spans, video_boundary + int(args["audio_offset_ms"]), axis_override="speech"
        )
        offset = audio_boundary - video_boundary
        op = {
            "op_id": _op_id(), "type": "split_edit",
            "seam_seg_id": args["seam_seg_id"], "audio_offset_ms": offset,
            "kind": "L-cut" if offset > 0 else "J-cut",
            "rationale": args.get("rationale"), "cut_cost": cost,
        }
        doc["operations"].append(op)
        return _j({"operation": op})

    if name == "set_level":
        op = {
            "op_id": _op_id(), "type": "level",
            "role": args["role"],
            "from_ms": int(args["from_ms"]), "to_ms": int(args["to_ms"]),
            "gain_db": args.get("gain_db"), "mute": bool(args.get("mute", False)),
        }
        doc["operations"].append(op)
        return _j({"operation": op})

    if name == "remove_operation":
        before = len(doc["operations"])
        doc["operations"] = [o for o in doc["operations"] if o.get("op_id") != args["op_id"]]
        if len(doc["operations"]) == before:
            return _j({"error": "unknown op_id"})
        return _j({"ok": True, "remaining": len(doc["operations"])})

    if name == "score_span":
        fid = args["file_id"]
        if fid not in session.file_ids:
            return _j({"error": "file_id not in this thread's scope"})
        in_ms = int(args["in_ms"])
        out_ms = int(args["out_ms"])
        if out_ms <= in_ms:
            return _j({"error": "out_ms must be after in_ms"})
        src = score_span.load_sources([fid]).get(fid)
        if not src:
            return _j({"error": "no analysis available for this clip"})
        return _j({
            "file_id": fid,
            "in_ms": in_ms,
            "out_ms": out_ms,
            "metrics": score_span.score_span(src, in_ms, out_ms),
            "quality_notes": score_span.quality_events_in(src, in_ms, out_ms),
        })

    if name == "align_clips":
        fa, fb = args["file_a"], args["file_b"]
        if fa not in session.file_ids or fb not in session.file_ids:
            return _j({"error": "file_a and file_b must both be in this thread's scope"})
        if fa == fb:
            return _j({"error": "file_a and file_b must be different clips"})
        span = None
        if args.get("from_ms") is not None and args.get("to_ms") is not None:
            span = (int(args["from_ms"]), int(args["to_ms"]))
        res = sync_mod.align_clips(fa, fb, span)
        if res is None:
            return _j({
                "file_a": fa, "file_b": fb, "matched": False,
                "note": "not simultaneous (or one clip is silent / shared audio too "
                        "short to be sure); treat as distinct takes/material, not synced angles",
            })
        out = {"file_a": fa, "file_b": fb, "matched": True, **res.to_dict()}
        return _j(out)

    if name == "read_angles":
        spine = args["file_id"]
        if spine not in session.file_ids:
            return _j({"error": "file_id not in this thread's scope"})
        from_ms = max(0, int(args["from_ms"]))
        to_ms = int(args["to_ms"])
        if to_ms <= from_ms:
            return _j({"error": "to_ms must be after from_ms"})
        requested = args.get("angle_file_ids") or session.synced_with(spine)
        if not requested:
            return _j({
                "spine": spine, "matched_angles": [],
                "note": "no verified synced angle for this clip; this is single-camera "
                        "here. Use place_video for unsynced coverage, not pick_angle.",
            })
        offsets = session.angle_offsets(spine, list(dict.fromkeys(requested)))
        if not offsets:
            return _j({
                "spine": spine, "matched_angles": [],
                "note": "the requested clips did not verify as synced angles of this "
                        "clip (align_clips found no reliable sync).",
            })
        sig = focus.load_focus_signals(spine)
        kind = focus.focus_for_spine_kind(_spine_kind_for(doc, spine), sig)
        intervals = focus.focus_timeline(kind, sig, from_ms, to_ms)
        rows = angle_menu.build_angle_menu(spine, offsets, intervals, session.perceptions_map())
        return _j({
            "spine": spine,
            "focus_kind": kind,
            "matched_angles": {fid: off for fid, off in offsets.items()},
            "menu": angle_menu.render_angle_menu_text(spine, rows),
        })

    if name == "read_framing":
        from app.services.l3 import framing

        fid = args["file_id"]
        if fid not in session.file_ids:
            return _j({"error": "file_id not in this thread's scope"})
        from_ms = max(0, int(args.get("from_ms", 0)))
        to_ms = int(args.get("to_ms") or 0)
        if to_ms <= from_ms:
            try:
                to_ms = session.grids(fid).duration_ms
            except Exception:
                to_ms = from_ms + 1
        perc = session.perceptions_map().get(fid)
        focus = framing.focus_for_range(perc, framing._load_motion_centroids(fid), from_ms, to_ms)
        rotate = framing.orientation_rotate(perc)
        auto = layers.solve_transform(doc)
        reframes = auto["fit"] == "cover"
        return _j({
            "file_id": fid,
            "aspect": layers.aspect_of(doc),
            "fit": auto["fit"],
            "reframes": reframes,
            "orientation_rotate_deg": rotate,
            "focus": focus,
            "note": (
                ("centered (no spatial perception logged for this clip)"
                 if focus is None else
                 f"will keep {focus['source']} in frame ({focus['evidence']})")
                if reframes else
                "landscape delivery letterboxes; no crop, so focus is unused"
            ),
        })

    if name == "timeline_status":
        return _j(_full_status(session))

    if name == "fit_duration":
        fitted, report = engine.fit_duration(
            doc["timeline"], session.grids_by_file(),
            int(float(args["target_s"]) * 1000),
            int(args.get("tolerance_ms", 500)),
        )
        doc["timeline"] = fitted
        return _j({"report": report, "status": engine.timeline_status(fitted)})

    if name == "ask_user":
        doc["open_questions"] = args["questions"]
        return _j({"ok": True, "paused": True})

    if name == "finalize":
        doc["summary"] = args["summary"]
        doc["notes"] = args.get("notes", [])
        doc["open_questions"] = doc.get("open_questions", [])
        doc["diagnostics"] = _full_status(session)
        # Persist the resolved A/V layers alongside the authoritative spine +
        # operations, so consumers (export/preview) don't re-resolve.
        doc["resolved"] = session.resolved().to_dict()
        return _j({"ok": True, "finalized": True})

    return _j({"error": f"unknown tool {name!r}"})
