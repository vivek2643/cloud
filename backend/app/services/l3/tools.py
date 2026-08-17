"""
The editing TOOL LOOP: the brain's agentic turn.

This is the Cursor-over-a-repo model applied to an edit. Instead of one shot that
returns a cut list, the brain runs a bounded perceive -> act -> re-perceive loop:
it calls OBSERVE tools (its deterministic senses -- ``observe.py``) to read the
edit and ACT tools (its verbs -- ``act.py``) to change it, then ends the turn
with a prose reply to the user. Every act mutates a WORKING copy of the Edit
Document; the caller persists the result once the loop ends.

No VLM in the loop, ever -- the senses are free projections of the document +
the per-turn context. Tool calls are native (Anthropic/Gemini function-calling
via the neutral ``LLMClient``); a provider without tools simply gets no tool
calls and the loop degrades to a single prose turn.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from app.services.l3 import act, observe
from app.services.l3.grade.arc import ARC_INTENTS
from app.services.l3.observe import EditContext
from app.services.llm import LLMClient, text_block, tool_result_block, tool_spec, user_message

logger = logging.getLogger(__name__)

# edso_think_act_check.plan.md change 5: modest bump from 12 -- a compound ask
# (a length AND a split screen) needs room for both the spine and the added
# feature, plus the done-gate's own overhead (up to _STRUCT_MAX_TRIES fix
# turns, one length reconcile, one advisory review). The real churn reduction
# is change 1 (think-first) removing place-then-remove trial-and-error, not
# this cap -- don't over-raise it (every turn costs latency + tokens).
_MAX_TURNS = 18
# Snap sovereignty: the seam-snapper may move a split_screen raw-window edge at
# most this far. Further than this, the brain's edge is kept and the seam is
# only SUGGESTED.
_SNAP_CAP_MS = 400


@dataclass
class LoopResult:
    reply: str
    document: dict
    changed: bool = False
    steps: List[str] = field(default_factory=list)   # tool names called, in order
    # Ordered per-turn AUDIT LOG, in true loop order. Two entry kinds share this
    # one list (each carries a "kind"):
    #   kind="reasoning": the assistant's per-step natural-language content --
    #     {turn, kind, reasoning, thinking?} -- so a turn's WHY (how it planned/
    #     decided, not just what it called) is auditable after the fact. Captured
    #     for EVERY step, including a step that makes no tool call.
    #   kind="tool":      a tool call -- {turn, kind, name, args, applied, result}.
    # Persisted verbatim onto edit_turns.trace (jsonb); a reasoning entry sits
    # right before the tool entries of the same step (the prose that motivated
    # them). See run_edit_loop.
    trace: List[dict] = field(default_factory=list)
    # When the brain called ask_user, the turn PAUSES: these are the questions to
    # surface, and the user's next message is their answer (the loop resumes).
    questions: List[dict] = field(default_factory=list)
    awaiting_user: bool = False


# --------------------------------------------------------------------------
# Tool declarations (neutral schema -> Anthropic tools / Gemini functions)
# --------------------------------------------------------------------------

def _specs() -> List[Dict[str, Any]]:
    S = tool_spec
    obj = lambda props, required=None: {  # noqa: E731
        "type": "object", "properties": props, "required": required or []}
    return [
        # --- OBSERVE (read-only senses) ---
        S("read_state", "Returns the current edit: ordered cuts (pos, seg_id, ref, "
          "duration, channel, speaker, muted, text, and a plain-language grade summary "
          "when a cut has one), channels in use, total length, a feel narration, and "
          "the z-stack (any V2/coverage layers + layout, beyond the main-line cuts) -- "
          "each audio layer beyond the main line also carries its own source loudness "
          "(loudness_lufs) when known. Also audio_gaps (stretches with NO audible audio "
          "layer at all) and audio_candidates (this user's unused audio files, each with "
          "is_musical/bpm so you can tell usable music from junk) when either applies. "
          "Pass seg_id to ALSO resolve that one cut's spoken words down to program-time "
          "offsets -- e.g. to land an overlay precisely on a line; omit it for a plain look.",
          obj({"seg_id": {"type": "string"}})),
        S("inspect_cut", "Returns the rich per-cut signal detail behind a Beat Index "
          "line's sig: breadcrumb: a downsampled action-energy curve + hit offsets, a "
          "downsampled loudness envelope + rise/fall change offsets + silence-gap "
          "offsets, and internal shot/composition-cut offsets -- all measured from the "
          "cut's own start. Also returns specifics: the COMPLETE scene_specifics for "
          "this cut when the vision model has enriched it -- every answered field "
          "(count, notable_object, continuity_cue, setting, custom-probe answers, and "
          "the full moments shot-list for a merged loose cut), beyond what the "
          "beat line's compact spec: tag shows. Omitted when the cut isn't enriched yet. "
          "Pass ref (a Beat Index moment id, e.g. from a sig: line) to "
          "inspect a cut not yet placed, or seg_id for an already-placed one.",
          obj({"ref": {"type": "string"}, "seg_id": {"type": "string"}})),
        S("predict", "Returns the program LENGTH under a proposed change without "
          "applying it: set_level re-takes every main-line cut at that level, drop "
          "removes those seg_ids, add appends [{ref, level}] cuts. Gives "
          "current_ms, projected_ms, delta_ms.",
          obj({"set_level": {"type": "string", "enum": list(observe._LEVELS)},
               "drop": {"type": "array", "items": {"type": "string"}},
               "add": {"type": "array", "items": {"type": "object", "properties": {
                   "ref": {"type": "string"}, "level": {"type": "string"}}}}})),
        S("validate", "Returns STRUCTURAL problems in the edit (spans out of range, "
          "empty cuts, malformed V2 cutaways/layouts). Empty result means clean.", obj({})),
        S("diagnose", "Returns editorial findings computed from the edit (same-speaker "
          "runs, low-energy runs, jump-cuts -- same clip placed out of "
          "source order, distance from any target length, same-beat takes "
          "that are both on the main line). Observations only.",
          obj({})),
        S("affordances", "Returns what is POSSIBLE: per cut the retake levels "
          "(tighter/wider), alternate takes, whether audio can toggle, the pace "
          "steps, and the program window for placing a bed there; globally the "
          "channels in use, addable channels, layout templates, unused audio "
          "asset count, and the video moments not currently on the main line.", obj({})),
        S("audio_state", "Returns the audio digest: placed beds (role, program "
          "window, gain/duck, and asset-vs-window length so a shortfall is "
          "visible), continuous outlook-authoritative runs (one shared audio "
          "source across angle switches -- a fact, not a lock), this user's "
          "uploaded audio files not yet placed (each with is_musical/bpm to tell "
          "usable music from junk), and -- when a musical source is in play -- the "
          "beat_grid (bpm + onset positions in program time per source, plus "
          "sections and drop_ms when detected: coarse phrase boundaries and the "
          "single strongest musical moment, a prior worth checking, not a fact).",
          obj({})),
        S("read_transcript", "Returns the COMPLETE, unbounded program-order transcript "
          "-- every placed segment's verbatim spoken words, including incidental "
          "speech under a picture cut (never hidden). The always-present mirror "
          "already carries a short per-cut gist every turn; call this only when you "
          "need the exact full wording (e.g. to quote precisely or check phrasing "
          "before placing a cut).", obj({})),
        S("review", "Returns the ASSEMBLED program read back, per cut in order: "
          "played_text (the verbatim words actually spoken over that cut's PLAYED "
          "span, after any trim) and word-level program-time offsets, plus "
          "total/target length and `flags` -- presented facts (never a prescribed "
          "fix), checked in order against your ask first (a feature you named -- "
          "split screen, a music bed -- that isn't actually in the edit), then the "
          "guidance (a rough head/tail -- wrong-speaker lead-in, filler/backchannel, "
          "leftover dead air -- or a V2 overlay that overruns/underfills the beat "
          "it sits over), then specific craft sharpeners (an audio gap with no "
          "sound at all, or a layer whose loudness sits well off the program's "
          "median). Call this to double-check what you actually built before "
          "finishing.", obj({})),
        # --- ACT (edit verbs; each mutates the working document) ---
        S("place", "Adds a cut by its ref. channel 'V1' inserts on the main line "
          "(picture+sound) at index `at` (default append); 'V2' lays a silent video "
          "layer over the ongoing audio at program `from_ms` (audio:'keep' plays its "
          "own sound). A moment holding several beats (shown in the Beat Index as a "
          "range: line) can be taken three ways: (1) WHOLE -- omit `piece` and use "
          "level 'broad' for the full continuous stretch; (2) TIGHTENED to a level "
          "'broad'\u2192'sharp' via `level` -- tightening keeps only the strongest "
          "beats and drops the weaker/connective ones (so 'sharp' yields fewer, "
          "tighter beats than 'broad'); (3) a SINGLE beat via `piece` = the 1-based "
          "position shown in the Beat Index/read_state, which places just that one "
          "beat on its own EVEN IF tightening would have dropped it (every listed "
          "beat stays reachable). `level` is ignored when `piece` is set.",
          obj({"ref": {"type": "string"}, "level": {"type": "string", "enum": list(observe._LEVELS)},
               "channel": {"type": "string", "enum": ["V1", "V2"]},
               "at": {"type": "integer"}, "from_ms": {"type": "integer"},
               "audio": {"type": "string", "enum": ["keep", "mute"]},
               "reason": {"type": "string"},
               "piece": {"type": "integer",
                         "description": "1-based position of one beat within a "
                         "multi-beat moment (from the Beat Index range: list); places "
                         "just that beat, reachable even when tightening would drop "
                         "it. Omit to place the whole moment / a level take."}}, ["ref"])),
        S("trim", "Changes a cut's SOURCE in/out. Absolute (in_ms/out_ms) or relative "
          "(delta_in_ms/delta_out_ms; delta_in_ms:200 starts 200ms later). Targets a "
          "main-line seg_id, a V2 place_video op_id, or an A2 place_audio op_id "
          "(trims which part of the bed's source plays; its program window shifts "
          "to match). Shortens the cut by the removed source span -- but internal "
          "keep_spans (jump-cuts already excised) mean the exact resulting length "
          "isn't purely linear; read it back from read_state/the Program Map rather "
          "than computing it. For a V2/A2 op only: snap:'beat' lands the RESULTING "
          "program edge (trim only ever moves that op's own program END) on the "
          "nearest beat/onset (audio_state's beat_grid) if a musical source is in play.",
          obj({"target_id": {"type": "string"},
               "in_ms": {"type": "integer"}, "out_ms": {"type": "integer"},
               "delta_in_ms": {"type": "integer"}, "delta_out_ms": {"type": "integer"},
               "snap": {"type": "string", "enum": ["beat"],
                        "description": "snaps the resulting program edge to the "
                        "nearest beat/onset; op targets only (place_video/place_audio)"}},
              ["target_id"])),
        S("remove", "Removes a main-line cut (seg_id) or an operation (op_id) -- "
          "including a place_audio bed.",
          obj({"target_id": {"type": "string"}}, ["target_id"])),
        S("place_audio", "Places an audio bed on program window [from_ms,to_ms]. "
          "role music|voiceover|sfx; source is an audio (or video) asset by file "
          "id; gain_db sets its level; duck_db (<=0) lowers it under overlapping "
          "dialogue, 0 = no duck. src_in_ms/src_out_ms pick which part of the "
          "source plays (default: from the start, up to the window or the "
          "asset's own length, whichever is shorter).",
          obj({"source": {"type": "string"},
               "role": {"type": "string", "enum": list(act._AUDIO_ROLES)},
               "from_ms": {"type": "integer"}, "to_ms": {"type": "integer"},
               "src_in_ms": {"type": "integer"}, "src_out_ms": {"type": "integer"},
               "gain_db": {"type": "number"}, "duck_db": {"type": "number"},
               "kind": {"type": "string", "enum": list(act._AUDIO_KINDS)},
               "snap": {"type": "string", "enum": ["beat"],
                        "description": "snaps from_ms to the nearest beat/onset "
                        "(audio_state's beat_grid) if a musical source is in play"},
               "reason": {"type": "string"}},
              ["source", "role", "from_ms", "to_ms"])),
        S("set_gain", "Sets a layer's OWN level in dB -- a main-line seg's coupled "
          "audio, or an A2 place_audio bed. Separate from duck (a side-chain "
          "reduction only where a bed overlaps dialogue); this is the base level.",
          obj({"target_id": {"type": "string"}, "gain_db": {"type": "number"}},
              ["target_id", "gain_db"])),
        S("duck", "Sets an A2 bed's explicit duck in dB (typically <=0), applied "
          "only where it overlaps live dialogue; 0 clears it. There is no "
          "auto-duck -- a bed ducks only by what this or place_audio sets. "
          "Targets a place_audio op_id only.",
          obj({"target_id": {"type": "string"}, "amount_db": {"type": "number"}},
              ["target_id", "amount_db"])),
        S("fade_audio", "Sets a fade envelope (ms) on a layer's own edges -- a "
          "main-line seg's coupled audio, or an A2 bed. in_ms/out_ms are fade "
          "durations (0 clears that edge; an omitted edge is left as-is). Hard "
          "start/stop by default -- nothing fades unless this sets it.",
          obj({"target_id": {"type": "string"},
               "in_ms": {"type": "integer"}, "out_ms": {"type": "integer"}},
              ["target_id"])),
        S("crossfade", "Cross-dissolves the spine AUDIO across the seam just "
          "before main-line cut seam_seg_id: the previous and next cuts' audio "
          "overlap by ms (split evenly) and fade across that overlap. One "
          "crossfade per seam (re-issuing replaces); ms=0 clears.",
          obj({"seam_seg_id": {"type": "string"}, "ms": {"type": "integer"}},
              ["seam_seg_id", "ms"])),
        S("replace_audio", "Overrides a main-line cut's coupled audio source with "
          "an explicit file span -- the escape hatch for outlook authoritative "
          "routing (this wins over the auto-computed route), or just swapping a "
          "cut's sound to any other file's span.",
          obj({"target_id": {"type": "string"}, "source": {"type": "string"},
               "src_in_ms": {"type": "integer"}, "src_out_ms": {"type": "integer"}},
              ["target_id", "source", "src_in_ms", "src_out_ms"])),
        S("move", "Reorders a main-line cut to a new 0-based index (to_index), OR "
          "repositions a placed op -- a V2 place_video or A2 place_audio -- to start "
          "at program time to_ms, keeping its duration. Give to_index for a seg_id, "
          "to_ms for an op_id. Shift-to-align (op_id only): instead of naming to_ms "
          "directly, give align_onset_ms (a CURRENT onset of this op's own source, "
          "e.g. audio_state's beat_grid or drop_ms, in this op's OWN program time) "
          "and align_to_ms (the program moment it should land on, e.g. a beat you "
          "chose for the climax) -- the op shifts by (align_to_ms - align_onset_ms), "
          "keeping its own duration; combine with snap:'beat' to land the final "
          "position exactly on the grid.",
          obj({"target_id": {"type": "string"}, "to_index": {"type": "integer"},
               "to_ms": {"type": "integer"},
               "align_onset_ms": {"type": "integer",
                                  "description": "a current onset of this op's own "
                                  "source, in program time, to use as the anchor"},
               "align_to_ms": {"type": "integer",
                               "description": "the program moment align_onset_ms "
                               "should land on"},
               "snap": {"type": "string", "enum": ["beat"],
                        "description": "snaps to_ms to the nearest beat/onset "
                        "(audio_state's beat_grid) if a musical source is in play"}},
              ["target_id"])),
        S("set_audio", "Mutes or unmutes a cut's SOURCE audio, keeping its picture "
          "(mute:true silences, mute:false plays its sound).",
          obj({"target_id": {"type": "string"}, "mute": {"type": "boolean"}},
              ["target_id", "mute"])),
        S("tag_arc_intent", "Tags a cut's position in the color arc: calm, build, peak, "
          "or resolve. A deterministic table turns this into a color nudge, scaled by "
          "the user's arc intensity dial (0 = no visible effect regardless of tags).",
          obj({"target_id": {"type": "string"},
               "intent": {"type": "string", "enum": list(ARC_INTENTS)}},
              ["target_id", "intent"])),
        S("set_grade", "Nudges color via named dials, each -1..1 (0/omitted = no "
          "change): warmth (+warmer/-cooler), tint (+magenta/-green), brightness, "
          "contrast, saturation. A deterministic mapping turns the dials into the "
          "actual color numbers. With target_id -> that cut only; without -> every "
          "main-line cut. Stacks onto whatever grade that cut already has.",
          obj({"target_id": {"type": "string"},
               "warmth": {"type": "number"}, "tint": {"type": "number"},
               "brightness": {"type": "number"}, "contrast": {"type": "number"},
               "saturation": {"type": "number"}}, [])),
        S("split_edit", "Decouples the AUDIO edge from the VIDEO edge at the seam just "
          "before main-line cut `seam_seg_id` (J/L cut). audio_offset_ms < 0: the "
          "incoming cut's audio leads under the previous picture; > 0: the previous "
          "cut's audio lingers over the new picture; 0 clears the split. One split "
          "per seam (re-issuing replaces).",
          obj({"seam_seg_id": {"type": "string"},
               "audio_offset_ms": {"type": "integer"}},
              ["seam_seg_id", "audio_offset_ms"])),
        S("tighten", "Re-takes main-line cut(s) at a different energy `level` = how "
          "much of the beat is kept around its peak (broad = the full run-up, sharp = "
          "just the core). With seg_id -> that cut; without -> every cut that has "
          "that level.",
          obj({"seg_id": {"type": "string"}, "level": {"type": "string", "enum": list(observe._LEVELS)}},
              ["level"])),
        S("retime", "Sets a cut's PLAYBACK PACE. A VIDEO cut plays at that speed "
          "(levels are normalized across clips so a step stays consistent between "
          "neighbours; 'natural'~=1x, so length scales roughly length/pace); this is "
          "RECORDED and shown in read_state but the render does not bake speed into "
          "the export length yet. A SPEECH cut is never pitched/sped: 'faster'/"
          "'much_faster' shave removable dead-air + fillers (also shortens by "
          "roughly the removed budget, not a pace multiplier), 'natural'/'slower' "
          "keep every pause. Either way, read the exact resulting length back from "
          "read_state/the Program Map -- don't compute it. With seg_id -> that cut; "
          "without -> the whole main line.",
          obj({"seg_id": {"type": "string"},
               "pace": {"type": "string", "enum": list(act._PACE_STEPS)}},
              ["pace"])),
        S("split_screen", "Shows the MAIN LINE and a second source at once over the "
          "window [from_ms, to_ms] (program ms): template 'split_h' (side-by-side), "
          "'split_v' (stacked), or 'pip' (inset over the main line). The second cell "
          "source is either a map `ref` or a raw window `file`+`in_ms`+`out_ms` "
          "(seam-snapped to the nearest clean boundary). The second cell is silent "
          "unless audio:'keep'. Its window [from_ms, to_ms] is pinned to program "
          "time and does NOT follow later trims/removes -- so add split/PiP LAST, "
          "once the main line's cuts and length are settled; if the main line "
          "changes afterward the window goes stale (check it in the Program Map) "
          "and you must re-lay it.",
          obj({"ref": {"type": "string"},
               "file": {"type": "string"},
               "in_ms": {"type": "integer"}, "out_ms": {"type": "integer"},
               "template": {"type": "string", "enum": ["split_h", "split_v", "pip"]},
               "from_ms": {"type": "integer"}, "to_ms": {"type": "integer"},
               "level": {"type": "string", "enum": list(observe._LEVELS)},
               "audio": {"type": "string", "enum": ["keep", "mute"]},
               "snap": {"type": "string", "enum": ["off"]},
               "reason": {"type": "string"}},
              ["template", "from_ms", "to_ms"])),
        # --- ASK (pause the turn for a user-owned decision) ---
        S("ask_user", "Pauses the turn and asks the user one or more multiple-choice "
          "questions (each needs 2+ concrete options; they can also type their own). "
          "SUGGEST, don't just ask: set `recommended` to your pick and `why` to one "
          "short reason, when you have one. Calling this ENDS your turn; you resume "
          "when they answer.",
          obj({"questions": {"type": "array", "items": {"type": "object", "properties": {
              "prompt": {"type": "string"},
              "options": {"type": "array", "items": {"type": "string"}},
              "allow_multiple": {"type": "boolean"},
              "recommended": {"type": "string", "description":
                  "Your suggested pick -- must be one of `options`."},
              "why": {"type": "string", "description":
                  "One short line: why you'd go with `recommended`."},
              "preview": {"type": "string", "description":
                  "Optional: one short line on what you'll do if they pick `recommended`."}},
              "required": ["prompt", "options"]}}}, ["questions"])),
        # --- PLAN (the durable plan artifact, brain_plan_mechanism.plan.md) ---
        S("set_plan", "Write (or rewrite) your PLAN for this edit -- the ordered intent "
          "you reasoned to before building, per PLAN BEFORE YOU BUILD. It is echoed back "
          "to you every turn as the PLAN mirror and survives across turns, so build "
          "against it and re-call this to UPDATE it (rather than drifting) when reading "
          "changes your mind. Full replace each time: pass the whole plan. Keep it tight.",
          obj({"purpose": {"type": "string",
                           "description": "what the piece is FOR, in one line"},
               "carries": {"type": "array", "items": {"type": "string"},
                           "description": "what LEADS moment to moment (a line, the "
                           "picture, music, a mix) -- one entry, or one per section"},
               "structure": {"type": "array",
                             "description": "the ORDERED BEATS -- one per section, "
                             "at INTENT level (a JOB like 'hook -- founder's "
                             "strongest intro', 'differentiator -- shown through "
                             "the demo'), in play order. NEVER a clip id: which "
                             "clip carries a beat is a tactic you discover at the "
                             "timeline, not part of the plan.",
                             "items": {"type": "object",
                                       "properties": {
                                           "beat": {"type": "string",
                                                    "description": "the beat's job "
                                                    "-- what this section must "
                                                    "accomplish, at intent level"},
                                           "need": {"type": "string",
                                                    "enum": ["required", "optional"],
                                                    "description": "required = "
                                                    "non-negotiable (if its tactic "
                                                    "fails, find another route or "
                                                    "surface the ceiling -- never "
                                                    "silently drop it); optional = "
                                                    "include only if the material "
                                                    "supports it. Defaults to "
                                                    "required."}},
                                       "required": ["beat"]}},
               "watch": {"type": "array", "items": {"type": "string"},
                         "description": "the hard spots to watch (a jump risk, a thin "
                         "stretch, a take choice)"},
               "note": {"type": "string",
                        "description": "optional: one line on what changed your mind, "
                        "when this call is an update"}})),
        S("wrap_up", "Write the user-facing WRAP-UP for this edit: `summary` -- "
          "ONE short, clean recap for the user of what you built and, when you "
          "made a compromise (a required beat you couldn't land cleanly, a ceiling "
          "the material or an asset forced), what it was and why. `open_questions` "
          "-- anything you need the user to decide; `notes` -- anything they should "
          "know. This is the ONLY channel the user actually SEES besides your reply, "
          "so when you finish with a known compromise it MUST be stated here -- "
          "never finish a compromised edit with an empty summary. Keep it tight and "
          "human; it is NOT a place to dump internal craft reasoning.",
          obj({"summary": {"type": "string"},
               "open_questions": {"type": "array", "items": {"type": "string"}},
               "notes": {"type": "array", "items": {"type": "string"}}})),
    ]


def _resolve_file(ctx: EditContext, ref: Any) -> str:
    """Resolve a brain-supplied clip id to a full file_id. Accepts a full id or
    the 8-char 'CLIP <file8>' prefix shown in the beat index. Falls back to the
    raw value (validate/act will no-op on a bad id)."""
    s = str(ref or "").strip()
    if not s:
        return ""
    for fid in ctx.file_ids:
        if fid == s or fid.startswith(s):
            return fid
    return s


def _resolve_audio_source(ctx: EditContext, ref: Any) -> str:
    """Resolve a brain-supplied audio source id against both this turn's video
    file_ids (a clip's own coupled sound) and this user's uploaded audio assets
    (a music/SFX file that was never a video source) -- same full-id or 8-char-
    prefix matching as `_resolve_file`. Falls back to the raw value
    (`place_audio` no-ops on an id that resolves to nothing real)."""
    s = str(ref or "").strip()
    if not s:
        return ""
    candidates = list(ctx.file_ids) + [a["file_id"] for a in ctx.audio_assets]
    for fid in candidates:
        if fid == s or fid.startswith(s):
            return fid
    return s


def _beat_grid_ms(document: dict, ctx: EditContext) -> List[int]:
    """Every onset (program ms) across whatever musical sources are currently
    placed, flattened for `observe.snap_to_beats` -- empty when no musical
    bed/clip is in play (audio_brain.plan.md 2b)."""
    return [ms for entry in observe._beat_grid(document, ctx)
           for ms in entry.get("onsets_ms") or []]


def _edit_fingerprint(doc: dict) -> tuple:
    """A CONTENT fingerprint of the edit's meaningful state -- the ordered spine
    plus the ops -- with the random per-place seg_id/op_id EXCLUDED, so a revert
    to an arrangement already held (place->remove->place the same ref; A->B->A)
    hashes IDENTICAL even though ids are freshly minted each place. Spine ORDER
    matters (a move is a real change); op order does not (resolve sorts them), so
    ops are sorted. Pure; used only for churn detection (brain_loop_convergence.
    plan.md Part 1)."""
    spine = tuple(
        (s.get("file_id"), int(s.get("in_ms") or 0), int(s.get("out_ms") or 0),
         s.get("ref"), s.get("level"), bool(s.get("mute")),
         s.get("audio_override") and tuple(sorted(s["audio_override"].items())))
        for s in (doc.get("timeline") or []))
    ops = tuple(sorted(
        (o.get("type"), o.get("source_file_id"), o.get("seam_seg_id"),
         int(o.get("from_ms") or 0), int(o.get("to_ms") or 0),
         int(o.get("src_in_ms") or 0), int(o.get("src_out_ms") or 0),
         o.get("role"), o.get("audio_kind"),
         float(o.get("gain_db") or 0.0), float(o.get("duck_db") or 0.0),
         int(o.get("audio_offset_ms") or 0), int(o.get("ms") or 0))
        for o in (doc.get("operations") or [])))
    return (spine, ops)


def _snap_trim_to_beat(doc: dict, ctx: EditContext, target_id: str,
                       in_ms: Any, out_ms: Any, delta_in_ms: Any, delta_out_ms: Any):
    """Beat-snap a `trim`'s resulting program-time edge for a placed OP (a V2
    place_video cutaway or an A2 place_audio bed -- audio_and_audit.plan.md
    Phase 3.1, mirroring move/place_audio's existing pattern, which only ever
    snaps a placed op's program anchor). `trim` moves an op's SOURCE span;
    since its program START (`from_ms`) never moves via trim, whichever
    source edge the brain names, there is exactly one derived program-time
    quantity to snap: the resulting program END. Returns (new_in_ms,
    new_out_ms, snap_info) as ABSOLUTE edges once computed, or None when
    there's no grid/target op/valid span to snap against -- the caller then
    leaves the original args untouched and act.trim's own validation still
    applies."""
    grid = _beat_grid_ms(doc, ctx)
    if not grid:
        return None
    op = next((o for o in doc.get("operations") or [] if o.get("op_id") == target_id
              and o.get("type") in ("place_video", "place_audio")), None)
    if op is None:
        return None
    cur_in, cur_out = int(op["src_in_ms"]), int(op["src_out_ms"])
    anchor = int(op.get("from_ms", 0))
    new_in = int(in_ms) if in_ms is not None else cur_in + int(delta_in_ms or 0)
    new_out = int(out_ms) if out_ms is not None else cur_out + int(delta_out_ms or 0)
    if new_out <= new_in:
        return None
    naive_end = anchor + (new_out - new_in)
    snap = observe.snap_to_beats(grid, naive_end, max_move_ms=_SNAP_CAP_MS)
    if not snap.get("snapped") or "suggested_ms" in snap:
        return None
    delta = int(snap["ms"]) - naive_end
    if out_ms is not None or delta_out_ms is not None:
        return new_in, new_out + delta, snap
    return new_in - delta, new_out, snap


def _trim_domain_reason(doc, ctx, target_id, in_ms, out_ms, delta_in_ms, delta_out_ms):
    """Part 3 (brain_loop_convergence.plan.md): reject a trim whose RESULTING
    absolute source edge lands well past the source's own length -- the
    fingerprint of an absolute source timestamp passed where a relative delta
    was meant (footgun b). Cheap + generic; returns a plain reason or None.
    Fail-open: unknown target/duration -> None (let the verb's own validation
    run)."""
    op = next((o for o in doc.get("operations") or []
               if o.get("op_id") == target_id
               and o.get("type") in ("place_video", "place_audio")), None)
    seg = None if op else next((s for s in doc.get("timeline") or []
                                if s.get("seg_id") == target_id), None)
    src = (op or {}).get("source_file_id") or (seg or {}).get("file_id")
    if not src:
        return None
    dur = ctx.durations.get(src)
    if dur is None:
        dur = next((a["dur_ms"] for a in getattr(ctx, "audio_assets", []) or []
                    if a["file_id"] == src), None)
    if not dur:
        return None
    cur_out = int((op or seg or {}).get("src_out_ms", (seg or {}).get("out_ms", 0)))
    new_out = int(out_ms) if out_ms is not None else cur_out + int(delta_out_ms or 0)
    if new_out > int(dur) + 1000:     # >1s past the source end -> almost certainly a mix-up
        return (f"trim's resulting source out ({new_out}ms) is past this source's "
                f"length ({int(dur)}ms). in_ms/out_ms/delta_* are SOURCE times -- "
                "did you pass an absolute timestamp where a relative delta_* was "
                "meant, or vice versa?")
    return None


def _normalize_questions(args: Dict[str, Any]) -> List[dict]:
    """Coerce the brain's ask_user payload into surfaced questions: each needs a
    prompt + >= 2 concrete options (bad ones dropped). `recommended`/`why`/
    `preview` are an enrichment (interactive_ask_and_salience.plan.md WS1) --
    the brain SUGGESTS rather than just asking. `recommended` is kept only
    when it names one of the surfaced `options` (never a dangling default);
    `why`/`preview` are kept only alongside a valid `recommended` (a reason
    with nothing to recommend is noise)."""
    out: List[dict] = []
    for i, q in enumerate(args.get("questions") or []):
        if not isinstance(q, dict):
            continue
        prompt = str(q.get("prompt") or "").strip()
        opts = [str(o).strip() for o in (q.get("options") or []) if str(o).strip()]
        if not prompt or len(opts) < 2:
            continue
        item = {"id": f"q{i}", "prompt": prompt, "options": opts,
                "allow_multiple": bool(q.get("allow_multiple"))}
        recommended = str(q.get("recommended") or "").strip()
        if recommended and recommended in opts:
            item["recommended"] = recommended
            why = str(q.get("why") or "").strip()
            if why:
                item["why"] = why
            preview = str(q.get("preview") or "").strip()
            if preview:
                item["preview"] = preview
        out.append(item)
    return out


# --------------------------------------------------------------------------
# Dispatch: a tool call -> (observation text, new working doc, changed?)
# --------------------------------------------------------------------------

def _dispatch(name: str, args: Dict[str, Any], ctx: EditContext,
              doc: dict, user_ask: str = "") -> Tuple[str, dict, bool]:
    snap_info: Dict[str, Any] = {}
    try:
        # OBSERVE (read-only)
        if name == "read_state":
            return _json(observe.read_state(doc, ctx, seg_id=args.get("seg_id"))), doc, False
        if name == "inspect_cut":
            return _json(observe.inspect_cut(
                ctx, ref=args.get("ref"), seg_id=args.get("seg_id"), document=doc)), doc, False
        if name == "predict":
            return _json(observe.predict(doc, ctx, set_level=args.get("set_level"),
                                         drop=args.get("drop"), add=args.get("add"))), doc, False
        if name == "validate":
            return _json({"issues": observe.validate(doc, ctx)}), doc, False
        if name == "diagnose":
            return _json({"findings": observe.diagnose(doc, ctx)}), doc, False
        if name == "affordances":
            return _json(observe.affordances(doc, ctx)), doc, False
        if name == "audio_state":
            return _json(observe.audio_state(doc, ctx)), doc, False
        if name == "review":
            return _json(observe.review(doc, ctx, user_ask=user_ask)), doc, False
        if name == "read_transcript":
            return _json(observe.read_transcript(doc, ctx)), doc, False

        # ACT (mutate)
        if name == "place":
            new = act.place(doc, ctx.index, args["ref"], level=args.get("level", "balanced"),
                            channel=args.get("channel", "V1"), at=args.get("at"),
                            from_ms=args.get("from_ms"),
                            audio=args.get("audio"), reason=args.get("reason", ""),
                            piece=args.get("piece"))
        elif name == "trim":
            tr_in, tr_out = args.get("in_ms"), args.get("out_ms")
            tr_din, tr_dout = args.get("delta_in_ms"), args.get("delta_out_ms")
            # Part 3 domain guard (brain_loop_convergence.plan.md): an ABSOLUTE
            # source edge past the source's own length almost always means an
            # absolute value was passed where a relative delta was meant. Reject
            # loudly (the verb can't see the source length; _dispatch can).
            # Fail-open: no known duration -> skip.
            trim_reason = _trim_domain_reason(doc, ctx, args["target_id"],
                                              tr_in, tr_out, tr_din, tr_dout)
            if trim_reason:
                return _json({"applied": False, "reason": trim_reason}), doc, False
            if args.get("snap") == "beat":
                snapped = _snap_trim_to_beat(doc, ctx, args["target_id"],
                                             tr_in, tr_out, tr_din, tr_dout)
                if snapped is not None:
                    tr_in, tr_out, snap_info = snapped
                    tr_din = tr_dout = None
            new = act.trim(doc, args["target_id"], in_ms=tr_in, out_ms=tr_out,
                           delta_in_ms=tr_din, delta_out_ms=tr_dout)
        elif name == "remove":
            new = act.remove(doc, args["target_id"])
        elif name == "place_audio":
            src = _resolve_audio_source(ctx, args.get("source"))
            # `ctx.durations` only covers this turn's file_ids (video sources);
            # an uploaded audio-only asset lives in `ctx.audio_assets` instead.
            asset_dur = ctx.durations.get(src)
            if asset_dur is None:
                asset_dur = next((a["dur_ms"] for a in ctx.audio_assets if a["file_id"] == src), None)
            pa_from, pa_to = args.get("from_ms"), args.get("to_ms")
            if args.get("snap") == "beat" and pa_from is not None and pa_to is not None:
                grid = _beat_grid_ms(doc, ctx)
                snap_info = observe.snap_to_beats(grid, pa_from, max_move_ms=_SNAP_CAP_MS)
                if snap_info.get("snapped") and "suggested_ms" not in snap_info:
                    shift = snap_info["ms"] - int(pa_from)
                    pa_from, pa_to = snap_info["ms"], int(pa_to) + shift
            new = act.place_audio(
                doc, source_file_id=src, role=args.get("role", ""),
                from_ms=pa_from, to_ms=pa_to,
                src_in_ms=args.get("src_in_ms", 0), src_out_ms=args.get("src_out_ms"),
                gain_db=args.get("gain_db", 0.0), duck_db=args.get("duck_db", 0.0),
                audio_kind=args.get("kind", "bed"), asset_dur_ms=asset_dur,
                reason=args.get("reason", ""),
            )
        elif name == "set_gain":
            new = act.set_gain(doc, args["target_id"], gain_db=args["gain_db"])
        elif name == "duck":
            new = act.duck(doc, args["target_id"], amount_db=args["amount_db"])
        elif name == "fade_audio":
            new = act.fade_audio(doc, args["target_id"], in_ms=args.get("in_ms"), out_ms=args.get("out_ms"))
        elif name == "crossfade":
            new = act.crossfade(doc, args["seam_seg_id"], ms=args["ms"])
        elif name == "replace_audio":
            src = _resolve_audio_source(ctx, args.get("source"))
            new = act.replace_audio(doc, args["target_id"], source_file_id=src,
                                    src_in_ms=args["src_in_ms"], src_out_ms=args["src_out_ms"])
        elif name == "move":
            mv_to_ms = args.get("to_ms")
            align_onset, align_to = args.get("align_onset_ms"), args.get("align_to_ms")
            if align_onset is not None and align_to is not None:
                op = next((o for o in doc.get("operations") or []
                          if o.get("op_id") == args["target_id"]
                          and o.get("type") in ("place_video", "place_audio")), None)
                if op is not None:
                    offset = int(align_to) - int(align_onset)
                    mv_to_ms = int(op.get("from_ms", 0)) + offset
            if args.get("snap") == "beat" and mv_to_ms is not None:
                grid = _beat_grid_ms(doc, ctx)
                snap_info = observe.snap_to_beats(grid, mv_to_ms, max_move_ms=_SNAP_CAP_MS)
                if snap_info.get("snapped") and "suggested_ms" not in snap_info:
                    mv_to_ms = snap_info["ms"]
            new = act.move(doc, args["target_id"], to_index=args.get("to_index"), to_ms=mv_to_ms)
        elif name == "set_audio":
            new = act.set_audio(doc, args["target_id"], mute=bool(args.get("mute")))
        elif name == "tag_arc_intent":
            new = act.set_arc_intent(doc, args["target_id"], intent=args.get("intent", ""))
        elif name == "set_grade":
            new = act.set_grade(
                doc, args.get("target_id"),
                warmth=args.get("warmth"), tint=args.get("tint"),
                brightness=args.get("brightness"), contrast=args.get("contrast"),
                saturation=args.get("saturation"),
            )
        elif name == "split_edit":
            new = act.split_edit(doc, args["seam_seg_id"],
                                 audio_offset_ms=args.get("audio_offset_ms", 0))
        elif name == "tighten":
            new = act.tighten(doc, ctx.index, seg_id=args.get("seg_id"), level=args.get("level", "tight"))
        elif name == "retime":
            new = act.retime(doc, ctx.index, seg_id=args.get("seg_id"), pace=args.get("pace", "natural"))
        elif name == "split_screen":
            # A cell source is a map ref OR a raw (file, in, out) window. The
            # window path seam-snaps to the clean cut-boundary points from
            # cut_records (v3-native -- see observe._seams_for_file) so a
            # nominated cell lands on a clean edge; the ref path is already a
            # minted cut.
            sc_file = _resolve_file(ctx, args.get("file")) if args.get("file") else None
            sc_in, sc_out = args.get("in_ms"), args.get("out_ms")
            if (sc_file and sc_in is not None and sc_out is not None
                    and args.get("snap") != "off"):
                points = observe._seams_for_file(ctx, sc_file)
                snap_info = observe.snap_span_to_seams(points, sc_in, sc_out, max_move_ms=_SNAP_CAP_MS)
                if snap_info.get("snapped"):
                    sc_in, sc_out = snap_info["in_ms"], snap_info["out_ms"]
            new = act.split_screen(doc, ctx.index, args.get("ref"),
                                   file=sc_file, in_ms=sc_in, out_ms=sc_out,
                                   template=args.get("template", "split_h"),
                                   from_ms=args.get("from_ms"), to_ms=args.get("to_ms"),
                                   level=args.get("level", "balanced"),
                                   audio=args.get("audio"), reason=args.get("reason", ""))
        elif name == "set_plan":
            # brain_plan_mechanism.plan.md §3.2: echo the rendered plan (not
            # read_state) so the tool result is a tight confirmation, not a
            # full state dump -- returns directly, bypassing the shared
            # changed/read_state tail below.
            new = act.set_plan(doc, purpose=args.get("purpose"),
                               carries=args.get("carries"), structure=args.get("structure"),
                               watch=args.get("watch"), note=args.get("note"))
            changed = new is not doc
            return _json({"applied": changed,
                          "plan": (new.get("plan") if changed else None)}), new, changed
        elif name == "wrap_up":
            # brain_plan_conformance.plan.md §3.2: same direct-return shape as
            # set_plan -- echo the written surface, not a full read_state dump.
            new = act.wrap_up(doc, summary=args.get("summary"),
                              open_questions=args.get("open_questions"),
                              notes=args.get("notes"))
            changed = new is not doc
            return _json({"applied": changed,
                          "surface": {"summary": new.get("summary"),
                                      "open_questions": new.get("open_questions"),
                                      "notes": new.get("notes")} if changed else None}), new, changed
        else:
            return _json({"error": f"unknown tool {name}"}), doc, False

        changed = new is not doc
        # Echo the resulting state so the model SEES the effect of its edit.
        result = {"applied": changed, "state": observe.read_state(new, ctx)}
        if not changed:
            result["note"] = "no-op (unknown id or illegal argument)"
        # Tell the brain how far a split_screen window's edges were seam-snapped
        # (+ the quality of the boundary it landed on), a place_audio/move edge
        # was beat-snapped, or which snap was SUGGESTED when an edge was kept
        # under the sovereignty cap, so it can judge/adjust.
        if (changed and snap_info.get("snapped")
                and (snap_info.get("in_delta_ms") or snap_info.get("out_delta_ms")
                     or snap_info.get("delta_ms")
                     or "in_suggested_ms" in snap_info or "out_suggested_ms" in snap_info
                     or "suggested_ms" in snap_info)):
            result["snap"] = snap_info
        return _json(result), new, changed
    except act.EditReject as r:
        # brain_loop_convergence.plan.md Part 3: a verb rejected the request with
        # a specific reason -- surface it immediately as applied=false so the brain
        # corrects THIS turn, not later via review.
        return _json({"applied": False, "reason": r.reason}), doc, False
    except Exception as e:  # a bad tool call must never crash the turn
        logger.exception("tools: %s failed", name)
        return _json({"error": f"{type(e).__name__}: {e}"}), doc, False


def _json(obj: Any) -> str:
    return json.dumps(obj, default=str)[:12000]


# Generous per-step cap for captured reasoning: NOT the tiny result[:600] used
# for tool results -- reasoning is the whole point here, so we keep it in full
# up to this bound. Only pathological runaway prose is trimmed, purely to bound
# a single jsonb row's size. Comfortably larger than a normal edit-turn reply.
_REASONING_CAP = 8000


def _reasoning_from_response(resp: Any) -> Tuple[str, str]:
    """Pull this loop step's natural-language reasoning out of an LLMResponse:
    the assistant's TEXT content (its prose reasoning, emitted alongside/between
    tool calls) and, when a provider returns them, its extended-THINKING blocks.

    Text is read from ``resp.text`` (the neutral, already-joined assistant text).
    Thinking is best-effort: today the Anthropic edit-loop client neither enables
    extended thinking nor surfaces thinking blocks, so this is a no-op for
    thinking now -- but we scan both the neutral ``assistant_message`` content and
    the provider-native ``raw`` completion for ``thinking``/``redacted_thinking``
    blocks so the capture starts working automatically if thinking is ever turned
    on, without another code change. Returns (text, thinking); either may be ''."""
    text = (getattr(resp, "text", "") or "").strip()
    thinking_parts: List[str] = []
    msg = getattr(resp, "assistant_message", None)
    content = msg.get("content") if isinstance(msg, dict) else None
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and b.get("type") in ("thinking", "redacted_thinking"):
                t = b.get("thinking") or b.get("text") or b.get("data") or ""
                if t:
                    thinking_parts.append(str(t))
    if not thinking_parts:  # provider-native fallback (e.g. an Anthropic completion)
        raw_content = getattr(getattr(resp, "raw", None), "content", None)
        if isinstance(raw_content, list):
            for blk in raw_content:
                if getattr(blk, "type", None) in ("thinking", "redacted_thinking"):
                    t = getattr(blk, "thinking", None) or getattr(blk, "data", None) or ""
                    if t:
                        thinking_parts.append(str(t))
    return text, "\n".join(thinking_parts).strip()


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

_STRUCT_MAX_TRIES = 3
# brain_plan_conformance.plan.md Part A: bounds how many times the surface-
# non-empty block can re-fire for one finish attempt -- unlike the once-fired
# advisory stages, Part A must actually GATE (re-block until filled), so it
# needs a cap of its own to guarantee the loop still terminates.
_SURFACE_MAX_BLOCKS = 2


def _latest_user_text(messages: List[dict]) -> str:
    """The newest user message as plain text (content may be a bare string or
    a list of blocks -- see store.load_messages). Used only to detect an
    explicitly requested feature (edso_think_act_check.plan.md change 4) for
    the done-gate's audit. Mirrors converse._latest_user_text -- duplicated,
    not imported, since converse.py already imports this module (a cycle)."""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            return " ".join(b.get("text", "") for b in c
                            if isinstance(b, dict) and b.get("type") == "text")
    return ""


# --------------------------------------------------------------------------
# brain_plan_conformance.plan.md: Part A (surface-non-empty) + Part B
# (plan-conformance accountability) -- see _verify_before_finish below for
# where these slot into the done-gate ladder.
# --------------------------------------------------------------------------

_SYNC_WORDS_RE = re.compile(r"\b(beat|grid|snap|sync|on the beat|downbeat|onset)\b", re.IGNORECASE)
_PUNCHY_WORDS_RE = re.compile(r"\b(punch\w*|quick|fast|snappy|rapid|energetic|tight cut\w*|"
                              r"quick succession|fast[- ]?paced)\b", re.IGNORECASE)
_LONG_CUT_MS = 4000          # a "long" main-line cut for a piece that declared punch
_FAST_PACE = {"faster", "much_faster"}   # act._PACE_STEPS toward quick


def _plan_text(plan: dict) -> str:
    """All free-form plan prose the brain wrote -- purpose + carries + watch +
    beat texts -- lowercased into one blob for keyword hints."""
    parts = [str(plan.get("purpose") or "")]
    parts += [str(x) for x in (plan.get("carries") or [])]
    parts += [str(x) for x in (plan.get("watch") or [])]
    parts += [observe._beat_view(e)[0] for e in (plan.get("structure") or [])]
    return " ".join(parts).lower()


def _hint_beatsync_declared_but_absent(working, ctx, plan) -> str | None:
    """§4.5.1: plan text declares beat-sync intent AND a musical grid exists
    AND no placed op edge actually lands on it. Ops record no snap flag
    (§1.4), so this measures op edges against the real grid directly."""
    if not _SYNC_WORDS_RE.search(_plan_text(plan)):
        return None
    grid = _beat_grid_ms(working, ctx)
    if not grid:
        return None                              # no music grid -> nothing to sync TO
    ops = [o for o in working.get("operations") or []
           if o.get("type") in ("place_video", "place_audio")]
    if not ops:
        return None
    TOL = 60                                      # ms; well inside _SNAP_CAP_MS=400

    def _on_grid(ms):
        s = observe.snap_to_beats(grid, ms, max_move_ms=_SNAP_CAP_MS)
        return s.get("snapped") and abs(int(s.get("ms", ms)) - int(ms)) <= TOL

    aligned = any(_on_grid(o.get("from_ms")) or (o.get("to_ms") is not None and _on_grid(o["to_ms"]))
                  for o in ops)
    if aligned:
        return None
    return ("plan/watch calls for beat-sync but no placed overlay/bed edge lands "
            "on the music grid -- either snap the ones that should hit the beat, "
            "or surface that you chose not to.")


def _hint_punchy_declared_but_flat(working, ctx, plan) -> str | None:
    """§4.5.2: plan text declares punchy/quick pacing AND the main line is
    mostly long cuts at natural pace (no pace_level toward faster)."""
    if not _PUNCHY_WORDS_RE.search(_plan_text(plan)):
        return None
    try:
        cuts = observe.read_state(working, ctx).get("cuts") or []
    except Exception:
        return None
    main = [c for c in cuts if c.get("dur_ms") is not None]
    if len(main) < 3:
        return None                              # too short to judge pace
    long_cuts = [c for c in main if int(c.get("dur_ms") or 0) >= _LONG_CUT_MS]
    any_fast = any((c.get("pace_level") in _FAST_PACE) for c in main)
    if any_fast or len(long_cuts) < (len(main) + 1) // 2:   # <half are long -> fine
        return None
    return (f"plan calls for punchy/quick pacing but {len(long_cuts)} of "
            f"{len(main)} main-line cuts run long ({_LONG_CUT_MS}ms+) at natural "
            "pace -- tighten/retime the ones that drag, or surface that the "
            "material can't go faster.")


def _conformance_hints(working: dict, ctx: EditContext, plan: dict | None) -> List[str]:
    """The cheap, robust advisory signals fed to the conformance checkpoint
    (brain_plan_conformance.plan.md §4.5). Few by design; each fail-open."""
    if not plan:
        return []
    out: List[str] = []
    for fn in (_hint_beatsync_declared_but_absent, _hint_punchy_declared_but_flat):
        try:
            h = fn(working, ctx, plan)
            if h:
                out.append(h)
        except Exception:
            logger.exception("conformance hint %s failed (skipping)", getattr(fn, "__name__", "?"))
    return out


def _surface_is_empty(working: dict) -> bool:
    """True when NONE of the user-facing surface fields carry content
    (brain_plan_conformance.plan.md Part A). summary is the primary channel;
    open_questions/notes also count as surfaced."""
    if (working.get("summary") or "").strip():
        return False
    if [q for q in (working.get("open_questions") or []) if str(q).strip()]:
        return False
    if [n for n in (working.get("notes") or []) if str(n).strip()]:
        return False
    return True


def _plan_conformance(working: dict, ctx: EditContext,
                      plan: dict | None, state: Dict[str, Any]) -> str | None:
    """Part B (brain_plan_conformance.plan.md): the finish-time plan-conformance
    accountability CHECKPOINT. Beats are intent-level prose with no clip ids, so
    this is an LLM self-accountability stage -- it PRESENTS the written plan
    (beats + carries + watch) beside the built timeline and a few advisory HINTS,
    and requires the brain to account for each required beat + each declared craft
    intention (delivered / fixed / compromised-and-surfaced), mirroring the
    FIT-TO-CRAFT discipline. Fires at most once (state-tracked). Fail-open: any
    error -> None (the edit still finishes). Returns feedback or None."""
    if state.get("conformance_surfaced"):
        return None
    if not (plan and (plan.get("structure") or plan.get("carries") or plan.get("watch"))):
        return None                     # no plan of record -> nothing to conform to
    try:
        required = [b for b in (observe._beat_view(e) for e in plan.get("structure") or [])
                    if b[0] and b[1] != "optional"]           # (beat_text, need)
        declared = list(plan.get("carries") or []) + list(plan.get("watch") or [])
        hints = _conformance_hints(working, ctx, plan)        # advisory, few (§4.5)
    except Exception:
        logger.exception("_plan_conformance: presentation build failed (finishing)")
        return None
    if not required and not declared and not hints:
        return None
    state["conformance_surfaced"] = True
    lines = ["AUTOMATIC CHECK -- plan conformance: hold the edit you BUILT against "
             "the plan you WROTE. For EACH item below, account in one line: "
             "DELIVERED (it's in the cut), FIXED (you're about to build/repair it), "
             "or COMPROMISED (the material genuinely can't land it) -- and a "
             "COMPROMISE MUST be surfaced to the user via wrap_up (summary/"
             "open_questions), never left silent. Same rule as fit-to-craft: a gap "
             "may pass ONLY for one of two reasons -- the ask required it, or the "
             "material can't support better -- named plainly."]
    if required:
        lines.append("  required beats (each must be delivered, or its ceiling surfaced):")
        lines += [f"    - {b}" for b, _ in required]
    if declared:
        lines.append("  declared craft intentions (carries / watch -- honor or surface):")
        lines += [f"    - {d}" for d in declared]
    if hints:
        lines.append("  advisory signals (cheap checks -- confirm or refute, don't trust blindly):")
        lines += [f"    - {h}" for h in hints]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# brain_accountability_architecture.plan.md PART 1: the done-gate as a
# DECLARED ORDERED LIST of stages, not a chain of nested early-returns.
#
# Each stage is a pure (or fail-open) function of one _GateCtx and returns its
# OWN feedback string or None. A stage NEVER returns on behalf of a later
# stage -- the evaluator (_run_gate_stages) owns the ladder -- which is what
# makes a mandatory invariant structurally unskippable and makes it impossible
# to orphan a stage by adding a `return` above it.
#
#   kind="mandatory": an invariant. Runs on EVERY gate evaluation, on EVERY
#     termination path. Never filtered by the advisory-skip.
#   kind="advisory":  a redundant nudge. May be skipped when the brain already
#     self-reviewed this turn (the ONLY thing the old `reviewed` short-circuit
#     was ever meant to skip).
# --------------------------------------------------------------------------

MANDATORY, ADVISORY = "mandatory", "advisory"


@dataclass(frozen=True)
class _Stage:
    key: str        # the `state` key this stage OWNS (its once-fired bookkeeping)
    kind: str       # MANDATORY | ADVISORY
    label: str      # stable machine token, recorded in the trace
    fn: Any         # (_GateCtx) -> str | None


@dataclass
class _GateCtx:
    """Everything a stage may read, computed ONCE per gate evaluation.

    `findings` (observe.diagnose) and `review_out` (observe.review) are computed
    UNCONDITIONALLY -- the old code skipped review() whenever the brain had
    self-reviewed, which silently emptied Stage 1's ask-flags (Bypass 2b). A
    mandatory stage may never depend on an optimization for an advisory one."""
    working: dict
    ctx: EditContext
    state: Dict[str, Any]
    steps: List[str]
    user_ask: str
    advisory_skip: bool
    findings: List[dict] = field(default_factory=list)
    review_out: Dict[str, Any] | None = None
    # Per-evaluation AUDIT of the ladder: one entry per declared stage, in
    # declared order -- {stage, kind, fired, why?}. Recorded into the trace so
    # "did the mandatory invariant actually run?" is assertable from a session
    # test and diagnosable from a stored thread.
    fired: List[dict] = field(default_factory=list)


def _gate_ctx(working: dict, ctx: EditContext, state: Dict[str, Any],
              steps: List[str], user_ask: str = "") -> _GateCtx:
    """Build the per-evaluation context. Fail-open: a sense that raises yields
    an empty projection, never an aborted gate."""
    advisory_skip = bool(
        "diagnose" in steps or "validate" in steps or "review" in steps
        or state.get("reviewed"))
    try:
        findings = observe.diagnose(working, ctx)
    except Exception:
        logger.exception("_gate_ctx: diagnose failed (continuing with none)")
        findings = []
    try:
        review_out = observe.review(working, ctx, user_ask=user_ask)
    except Exception:
        logger.exception("_gate_ctx: review failed (continuing without it)")
        review_out = None
    return _GateCtx(working=working, ctx=ctx, state=state, steps=steps,
                    user_ask=user_ask, advisory_skip=advisory_skip,
                    findings=findings, review_out=review_out)


def _stage_structural(gc: _GateCtx) -> str | None:
    issues = observe.validate(gc.working, gc.ctx)
    if not issues or gc.state["struct_tries"] >= _STRUCT_MAX_TRIES:
        return None
    gc.state["struct_tries"] += 1
    body = "; ".join(f"{i.get('kind')} {i.get('id')}: {i.get('message')}"
                     for i in issues[:8])
    return ("AUTOMATIC CHECK -- structural problems that would break the render. "
            "Fix these before finishing:\n" + body)


def _stage_length(gc: _GateCtx) -> str | None:
    over = [f for f in gc.findings if "over target" in (f.get("message") or "")]
    if not over or gc.state["length_surfaced"]:
        return None
    gc.state["length_surfaced"] = True
    return ("AUTOMATIC CHECK -- length: " + (over[0].get("message") or "") +
            ". Either trim to the target, or say in one line why this length is "
            "right, then finish.")


def _stage_intent(gc: _GateCtx) -> str | None:
    """Stage 1, fit to the ask. MANDATORY -- and it now actually sees flags on a
    self-reviewed turn (Bypass 2b), because _gate_ctx always computes review."""
    ask_flags = [f for f in (gc.review_out or {}).get("flags", [])
                 if f.get("category") == "ask"]
    if not ask_flags or gc.state["intent_surfaced"]:
        return None
    gc.state["intent_surfaced"] = True
    body = "\n".join(f"- {f.get('message')}" for f in ask_flags[:8])
    return ("AUTOMATIC CHECK -- Stage 1, fit to your ask: a feature you named "
            "isn't actually in the edit:\n" + body +
            "\nAdd it, or finish only if the material genuinely can't support it.")


def _stage_craft(gc: _GateCtx) -> str | None:
    if not gc.working.get("timeline") or gc.state["craft_surfaced"]:
        return None
    gc.state["craft_surfaced"] = True
    return ("AUTOMATIC CHECK -- Stage 2, fit to craft: forget the ask entirely -- "
            "judge this edit the way you'd judge any finished video handed to you "
            "cold. Does it look and sound like a real, high-quality piece of work? "
            "Fix what falls short. If you finish anyway with a known flaw, name in "
            "ONE line which of exactly two reasons applies: (a) the user's ask "
            "required it, or (b) the material can't support better. Any other "
            "reason ('looks fine anyway') isn't enough -- fix it instead.")


def _stage_flags(gc: _GateCtx) -> str | None:
    """Stage 3, specific flags. The ONE advisory stage -- the only thing the old
    `reviewed` short-circuit was ever meant to skip. brain_mirror_readside.
    plan.md section 4.2: the former Stage 2.5 (a one-shot, reviewed-gate-
    bypassing continuity reveal) is RETIRED -- the always-present mirror
    already puts jump-cut/broken-line/incidental flags in front of the brain
    continuously, so this stage no longer needs a special *revealing* role for
    continuity. A jump-cut finding still reaches this stage (folded into
    `rest`, same as any other advisory flag) -- it's just never uniquely
    surfaced anymore."""
    rest = [f for f in gc.findings if "target" not in (f.get("message") or "")]
    rest += [f for f in (gc.review_out or {}).get("flags", []) if f.get("category") != "ask"]
    if not rest or gc.state["reviewed"]:
        return None
    gc.state["reviewed"] = True
    body = "\n".join(
        f"- [{f.get('severity') or 'info'}]"
        + (f" ({f['category']})" if f.get("category") else "") + " "
        + (f"{f['anchor']}: " if f.get("anchor") else "") + (f.get("message") or "")
        for f in rest[:12])
    return ("AUTOMATIC CHECK -- Stage 3, specific flags: advisory -- act on what "
            "serves the goal, ignore the rest, then finish:\n" + body)


def _stage_conformance(gc: _GateCtx) -> str | None:
    return _plan_conformance(gc.working, gc.ctx, gc.working.get("plan"), gc.state)


def _stage_surface(gc: _GateCtx) -> str | None:
    """Part A (brain_plan_conformance.plan.md): never finish a KNOWN compromise
    with an empty surface. "Compromise in play" = the conformance stage has
    run (an account was demanded) AND a live advisory hint. Bounded by
    surface_blocked so a stubborn brain still terminates."""
    compromise_in_play = gc.state.get("conformance_surfaced") and bool(
        _conformance_hints(gc.working, gc.ctx, gc.working.get("plan")))
    if not (gc.working.get("timeline") and compromise_in_play
            and _surface_is_empty(gc.working)
            and gc.state["surface_blocked"] < _SURFACE_MAX_BLOCKS):
        return None
    gc.state["surface_blocked"] += 1
    return ("AUTOMATIC CHECK -- surface: this edit carries a known compromise "
            "but nothing is surfaced to the user. Call wrap_up with a short, "
            "human `summary` that states plainly what you built and what was "
            "compromised and why (and open_questions if the user should weigh "
            "in). Don't finish a compromised edit silent.")


# THE LADDER. Order is data, not source order. A new stage is added HERE or it
# never runs (loud in review) -- it can no longer be orphaned by an earlier
# `return`. The MANDATORY set is asserted in the tests, so silently demoting an
# invariant to advisory fails CI.
_GATE_STAGES: Tuple[_Stage, ...] = (
    _Stage("struct_tries",         MANDATORY, "structural",  _stage_structural),
    _Stage("length_surfaced",      MANDATORY, "length",      _stage_length),
    _Stage("intent_surfaced",      MANDATORY, "intent",      _stage_intent),
    _Stage("craft_surfaced",       MANDATORY, "craft",       _stage_craft),
    _Stage("reviewed",             ADVISORY,  "flags",       _stage_flags),
    _Stage("conformance_surfaced", MANDATORY, "conformance", _stage_conformance),
    _Stage("surface_blocked",      MANDATORY, "surface",     _stage_surface),
)


def _run_gate_stages(gc: _GateCtx, *, kinds=(MANDATORY, ADVISORY),
                     collect: bool = False) -> Any:
    """Evaluate the declared ladder in order.

    Semantics, explicit and total:
      * a stage whose `kind` is not in `kinds` is not evaluated (recorded as
        skipped-by-scope);
      * an ADVISORY stage is skipped when `gc.advisory_skip` -- this is the ONLY
        skip mechanism, and it can never reach a MANDATORY stage;
      * a stage that raises is logged and treated as "nothing to say" (fail-open,
        house rule) -- it can never suppress the stages after it;
      * `collect=False` (interactive): return the FIRST stage's feedback, or None
        -> the loop asks one thing at a time, exactly as today;
      * `collect=True` (enforcement, used by the ONE finalizer): evaluate EVERY
        in-scope stage and return the list of all outstanding feedback.
    Every outcome is appended to `gc.fired` for the trace."""
    out: List[str] = []
    for st in _GATE_STAGES:
        if st.kind not in kinds:
            gc.fired.append({"stage": st.label, "kind": st.kind, "fired": False,
                             "why": "out-of-scope"})
            continue
        if st.kind == ADVISORY and gc.advisory_skip:
            gc.fired.append({"stage": st.label, "kind": st.kind, "fired": False,
                             "why": "advisory-skip: brain self-reviewed"})
            continue
        try:
            fb = st.fn(gc)
        except Exception:
            logger.exception("gate stage %s failed (fail-open)", st.label)
            gc.fired.append({"stage": st.label, "kind": st.kind, "fired": False,
                             "why": "error"})
            continue
        gc.fired.append({"stage": st.label, "kind": st.kind, "fired": fb is not None})
        if fb is None:
            continue
        if not collect:
            return fb
        out.append(fb)
    return out if collect else None


def _verify_before_finish(working: dict, ctx: EditContext,
                          state: Dict[str, Any], steps: List[str],
                          user_ask: str = "") -> str | None:
    """The done-gate, interactive mode: evaluate the DECLARED ladder
    (_GATE_STAGES) and return the first stage's feedback, or None to allow
    finishing. The ladder's order, and which stages are mandatory vs advisory,
    are declared data -- see _GATE_STAGES. This function no longer contains any
    stage logic and MUST NOT grow an early return."""
    return _run_gate_stages(_gate_ctx(working, ctx, state, steps, user_ask))


def _record_gate(trace: List[dict], *, exit_reason: str, gc: _GateCtx | None,
                 note: str = "") -> None:
    """The ladder's own outcome, once per gate evaluation:
    {kind:"gate", exit_reason, stages:[{stage, kind, fired, why?}], note?}.
    This is what makes "the mandatory invariant ACTUALLY fired" assertable from
    a session test and diagnosable from a stored thread -- the missing evidence
    that let two dead-code bugs ship 'verified'. Fail-open: a recording error
    must never alter the edit or the reply."""
    try:
        entry: Dict[str, Any] = {"kind": "gate", "exit_reason": exit_reason,
                                 "stages": gc.fired if gc is not None else []}
        if note:
            entry["note"] = note
        trace.append(entry)
    except Exception:
        logger.exception("tools: gate trace record failed (continuing)")


def _finalize_turn(llm, *, system, convo, ctx, working, tools, state, steps,
                   user_ask, trace, edit_moved, questions, exit_reason,
                   max_tokens) -> dict:
    """THE ONE FINALIZATION CHOKE POINT (PART 1). Every termination path lands
    here exactly once -- voluntary finish, cap exhaustion, last-turn finish,
    ask_user pause -- and the MANDATORY stages are evaluated in ENFORCEMENT mode
    (collect=True) regardless of which path arrived. There is no flag that can
    say "already fine": whether anything is outstanding is READ OFF the declared
    ladder and the state dict.

    `exit_reason` is recorded, never consulted for control flow -- except for the
    one legitimate case: a paused ask_user turn keeps its own question framing
    (the turn isn't over, the user is being asked something).

    Fail-open: any error leaves `working` untouched; the reply builder's
    last_text fallback is unchanged."""
    try:
        if questions:
            _record_gate(trace, exit_reason=exit_reason, gc=None,
                         note="paused for ask_user -- finalization deferred")
            return working
        if not edit_moved:                    # a plan-only turn is not an edit
            return working
        gc = _gate_ctx(working, ctx, state, steps, user_ask)
        outstanding = _run_gate_stages(gc, kinds=(MANDATORY,), collect=True)
        _record_gate(trace, exit_reason=exit_reason, gc=gc)
        if outstanding or _surface_is_empty(working):
            working = _finalize_after_loop(
                llm, system=system, convo=convo, ctx=ctx, working=working,
                tools=tools, verify=state, user_ask=user_ask, trace=trace,
                outstanding=outstanding, max_tokens=max_tokens)
        return working
    except Exception:
        logger.exception("tools: _finalize_turn failed (leaving working untouched)")
        return working


def _fallback_summary(working: dict, ctx: EditContext) -> str:
    """A plain, truthful one-liner built from the edit STATE (never leftover
    mid-action prose) for when the brain runs out of budget without a wrap_up.
    Names the size of what was built and any LIVE compromise the advisory hints
    already detected, so even the floor tells the user the truth. Pure/fail-open."""
    try:
        st = observe.read_state(working, ctx)      # returns cut_count + total_ms
        n = st.get("cut_count") or len(st.get("cuts") or [])
        total = st.get("total_ms")
    except Exception:
        n, total = len(working.get("timeline") or []), None
    dur = f", ~{int(total)//1000}s" if total else ""
    base = f"Assembled {n} cut{'s' if n != 1 else ''}{dur}."
    hints = _conformance_hints(working, ctx, working.get("plan"))
    if hints:
        base += (" Ran out of editing room before fully resolving: "
                 + "; ".join(h.split(" -- ")[0] for h in hints) + ".")
    else:
        base += " Ran out of editing room before a final self-review."
    return base


def _finalize_after_loop(llm, *, system, convo, ctx, working, tools, verify,
                         user_ask, trace, outstanding, max_tokens) -> dict:
    """PART 1 (brain_accountability_architecture.plan.md): make a NON-voluntary
    exit (the max_turns cap, a last-turn finish, or a self-reviewed turn that
    left a mandatory invariant outstanding) finalize as truthfully as a clean
    voluntary finish would. Runs ONE bounded finalization step: it instructs
    the brain to STOP editing, ACCEPT the best-available version of anything
    that didn't land, and call wrap_up with a real recap + any surfaced
    compromise. Any wrap_up (or tiny lock-in edit) it makes is applied. Returns
    the (possibly updated) working doc. Fail-open: any error -> working unchanged.

    `outstanding` is the list of feedback lines `_finalize_turn` already
    COLLECTED from the declared ladder (in enforcement mode) -- this function
    presents those lines rather than re-deriving them by calling
    `_plan_conformance` itself, which would find `conformance_surfaced` already
    True (set by that same collection pass) and silently return nothing: a
    double-fire hazard against the SAME state dict, not a double message.
    `trace` is accepted for the observability funnel that later work wires
    the finalize instruction through; unused here."""
    try:
        instruct = (
            "AUTOMATIC FINALIZE -- you've reached this turn's build budget. Stop "
            "opening new work. In ONE step: if a required beat or a declared "
            "intention didn't land, ACCEPT the best version you have (do NOT retry) "
            "and call wrap_up with a short, human `summary` of what you built and "
            "any compromise + why (open_questions if it's the user's to weigh). "
            "Make at most a single tiny edit only if it LOCKS the best-available "
            "version; otherwise just wrap_up.")
        if outstanding:
            instruct += "\n\n" + "\n\n".join(outstanding)
        convo.append(user_message(instruct))
        resp = llm.run(system=system, messages=convo, tools=tools,
                       max_tokens=max_tokens, cache_system=True)
        for tc in (resp.tool_calls or []):
            if tc.name == "ask_user":       # finalization never pauses for a question
                continue
            _obs, working, _did = _dispatch(tc.name, tc.input or {}, ctx, working, user_ask)
    except Exception:
        logger.exception("tools: forced finalization step failed (continuing)")
    # Deterministic floor: if the brain still left the surface empty, synthesize a
    # truthful recap from the built state so the reply is NEVER leftover reasoning.
    try:
        if working.get("timeline") and _surface_is_empty(working):
            working = act.wrap_up(working, summary=_fallback_summary(working, ctx))
    except Exception:
        logger.exception("tools: fallback summary failed (continuing)")
    return working


def _progress_note(turn: int, max_turns: int, *, churn: bool = False) -> str:
    """A per-turn CONVERGENCE note (brain_loop_convergence.plan.md Part 1),
    pushed beside the mirror. Deliberately framed around PROGRESS + converging,
    NOT a countdown to rush: budget visibility exists so the brain reserves room
    to finish and knows when to accept-and-surface a ceiling. Pure."""
    used, remaining = turn + 1, max_turns - (turn + 1)
    head = (f"PROGRESS: step {used} of {max_turns} this turn (~{remaining} left "
            "before it auto-finalizes).")
    body = (" This is room to CONVERGE, not a clock to beat: attempt each goal "
            "honestly, but if a goal genuinely won't land after a real attempt, "
            "that's a CEILING -- accept the best available version and SURFACE it "
            "with wrap_up rather than retrying. Reserve room to finish cleanly.")
    if remaining <= 4:
        body += (" You're near this turn's budget -- move to CONVERGE: lock the "
                 "best version you have, resolve or surface any open compromise, "
                 "and leave a wrap_up recap. Don't open new threads of work.")
    return head + body


def run_edit_loop(llm: LLMClient, *, system: str, messages: List[dict],
                  ctx: EditContext, document: dict,
                  max_turns: int = _MAX_TURNS,
                  max_tokens: int = 4096) -> LoopResult:
    """Run the bounded tool loop for one user turn. Returns the final prose reply,
    the (possibly mutated) working document, and whether it changed."""
    convo = list(messages)
    working = document
    changed = False
    steps: List[str] = []
    trace: List[dict] = []
    tools = _specs()
    last_text = ""
    questions: List[dict] = []
    verify = {"struct_tries": 0, "length_surfaced": False, "intent_surfaced": False,
             "craft_surfaced": False, "reviewed": False,
             "conformance_surfaced": False, "surface_blocked": 0}
    # brain_accountability_architecture.plan.md PART 1: recorded, never consulted
    # for control flow (except the ask_user pause, which keeps its own question
    # framing) -- the ONE finalizer (_finalize_turn) reads outstanding work off
    # the declared ladder + state, never off a boolean set at an exit site.
    exit_reason = "cap"
    # brain_loop_convergence.plan.md Part 1: fingerprints of every DISTINCT edit
    # state seen this turn (excluding random ids), and the states already flagged
    # as churn so the same revert isn't nagged every step. Generic no-progress
    # signal -- never a retry cap.
    seen_states: dict = {_edit_fingerprint(working): -1}   # fp -> turn first seen
    churned_states: set = set()
    # edso_think_act_check.plan.md change 4: the done-gate's audit checks a
    # NAMED feature (split screen, a music bed) is actually present -- needs
    # the user's own latest words, computed once here.
    user_ask = _latest_user_text(messages)

    for turn in range(max_turns):
        resp = llm.run(system=system, messages=convo, tools=tools,
                       max_tokens=max_tokens, cache_system=True)
        last_text = (resp.text or "").strip() or last_text
        convo.append(resp.assistant_message)
        # Capture this step's natural-language reasoning (+ thinking, if the
        # provider ever returns it) into the ordered trace, BEFORE the step's
        # tool entries so the prose sits with the actions it motivated. Runs
        # for every step -- including a no-tool step (a finish attempt / a
        # blocked done-gate turn) -- so no reasoning is lost. Additive and
        # fail-open: a capture error must never alter the edit or the reply.
        try:
            r_text, r_think = _reasoning_from_response(resp)
            if r_text or r_think:
                entry: Dict[str, Any] = {"turn": turn, "kind": "reasoning",
                                         "reasoning": r_text[:_REASONING_CAP]}
                if r_think:
                    entry["thinking"] = r_think[:_REASONING_CAP]
                trace.append(entry)
        except Exception:
            logger.exception("tools: reasoning capture failed (continuing)")
        if not resp.tool_calls:
            # Finish attempt. The gate now runs on EVERY finish attempt, the last
            # turn included (the old `turn < max_turns - 1` guard is what made a
            # last-turn finish skip the ladder entirely). On the last turn there
            # is no room to iterate, so feedback is not injected; the stages have
            # still marked `verify`, and _finalize_turn enforces whatever is left
            # outstanding.
            edit_moved = changed   # PART 4 will define edit_moved properly; until then an alias
            if edit_moved:
                feedback = _verify_before_finish(working, ctx, verify, steps, user_ask)
                if feedback is not None and turn < max_turns - 1:
                    convo.append(user_message(feedback))
                    continue
            exit_reason = "voluntary"
            break
        results = []
        asked = False
        for tc in resp.tool_calls:
            steps.append(tc.name)
            if tc.name == "ask_user":
                questions.extend(_normalize_questions(tc.input or {}))
                asked = True
                results.append(tool_result_block(tc.id, _json(
                    {"posed": True, "note": "Shown to the user; end your turn and wait for their answer."})))
                trace.append({"turn": turn, "kind": "tool", "name": tc.name,
                              "args": tc.input or {},
                              "applied": False, "result": "posed to user"})
                continue
            obs, working, did = _dispatch(tc.name, tc.input or {}, ctx, working, user_ask)
            changed = changed or did
            results.append(tool_result_block(tc.id, obs))
            trace.append({"turn": turn, "kind": "tool", "name": tc.name,
                          "args": tc.input or {},
                          "applied": bool(did), "result": obs[:600]})
        # brain_loop_convergence.plan.md Part 1: did this turn's edits land the
        # program back on an arrangement it already held at an EARLIER turn?
        # (Revert / net-zero swap / place-remove-place.) Generic, id-independent;
        # advisory only -- never stops the loop, just nudges toward accept+surface.
        churn_now = churn_hint = None
        try:
            if changed:
                fp = _edit_fingerprint(working)
                prior_turn = seen_states.get(fp)
                # A match to a state first seen strictly before the last step means
                # the edit left that state and came back -- churn. (A match to the
                # immediately prior step is a same-turn no-op, handled by Part 3's
                # fail-loud verbs.)
                if prior_turn is not None and prior_turn < turn - 1 and fp not in churned_states:
                    churned_states.add(fp)
                    churn_now = True
                    churn_hint = (
                        "CHURN: this edit has returned to an arrangement you already "
                        "built earlier -- you're cycling (revert / swap-back / place-"
                        "remove-place) without net progress. Treat this as a CEILING: "
                        "pick the best version you have and MOVE ON (accept + surface "
                        "the trade-off with wrap_up), rather than trying the same "
                        "swap again.")
                seen_states.setdefault(fp, turn)
        except Exception:
            logger.exception("tools: churn detection failed (continuing)")
        # brain_mirror_readside.plan.md section 4.2: push the always-present
        # mirror after EVERY tool call (not only when a sense is explicitly
        # called) -- a plain text block alongside this iteration's tool_result
        # blocks (a user-role message may legally mix both), so the brain
        # sees the CURRENT join/speech/flag state before its next move. Pure
        # + idempotent (observe.mirror), so pushing it every step can never
        # itself thrash the loop.
        mirror_text = observe.mirror_text(working, ctx)
        if mirror_text:
            results.append(text_block(mirror_text))
        # brain_plan_mechanism.plan.md §4.2: the PLAN mirror, pushed the same
        # way -- so a set_plan this round is reflected immediately, and building
        # WITHOUT a plan is nudged (building=changed makes the absence loud only
        # once the edit has actually moved this turn).
        plan_text = observe.plan_mirror_text(working, building=changed)
        if plan_text:
            results.append(text_block(plan_text))
        # brain_loop_convergence.plan.md Part 1: per-turn budget/progress +
        # convergence discipline, and a churn hint when the edit is not making
        # progress (returned to a prior state). Advisory text blocks, same shape
        # as the mirrors; pure + fail-open (never mutate the edit or reply).
        results.append(text_block(_progress_note(turn, max_turns, churn=churn_now)))
        if churn_hint:
            results.append(text_block(churn_hint))
        convo.append(user_message(results))
        # ask_user PAUSES the turn: the user's next message is the answer.
        if asked and questions:
            exit_reason = "asked"
            break
    else:
        logger.info("tools: hit max_turns=%d; finalizing", max_turns)

    # brain_accountability_architecture.plan.md PART 1: the ONE finalization
    # choke point. Unconditional on the exit path -- no `finished_clean`, no
    # `not questions and not …` guard chain deciding whether the invariants get
    # to run. Whether anything is outstanding is read off the declared ladder.
    edit_moved = changed   # PART 4 will define edit_moved properly; until then an alias
    working = _finalize_turn(
        llm, system=system, convo=convo, ctx=ctx, working=working, tools=tools,
        state=verify, steps=steps, user_ask=user_ask, trace=trace,
        edit_moved=edit_moved, questions=questions, exit_reason=exit_reason,
        max_tokens=max_tokens)

    awaiting = bool(questions)
    # brain_plan_conformance.plan.md Part A: the user-facing reply prefers the
    # brain's deliberate wrap-up (act.wrap_up -> document["summary"]) over the
    # last internal prose line, so a finished edit surfaces a clean recap (and
    # any compromise) instead of leftover craft reasoning. Falls back to
    # last_text (chat turns, no-op turns) so nothing regresses.
    summary = (working.get("summary") or "").strip()
    reply = (summary if summary and not awaiting else last_text) or (
        "Before I go further I need your call on a couple of things below."
        if awaiting else "Done.")
    return LoopResult(reply=reply, document=working, changed=changed, steps=steps,
                      trace=trace, questions=questions, awaiting_user=awaiting)
