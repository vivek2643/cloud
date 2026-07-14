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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from app.services.l3 import act, observe
from app.services.l3.grade.arc import ARC_INTENTS
from app.services.l3.observe import EditContext
from app.services.llm import LLMClient, tool_result_block, tool_spec, user_message

logger = logging.getLogger(__name__)

_MAX_TURNS = 12
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
    # Full per-call trace ({name, args, applied, result}) so a turn's REASONING
    # is auditable after the fact ("check the reasoning") without re-running it.
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
          "when a cut has one), channels in use, total length, and a feel narration.",
          obj({})),
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
          "runs, low-energy runs, distance from any target length, same-beat takes "
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
          "source across angle switches -- a fact, not a lock), and this user's "
          "uploaded audio files not yet placed.", obj({})),
        # --- ACT (edit verbs; each mutates the working document) ---
        S("place", "Adds a cut by its ref. channel 'V1' inserts on the main line "
          "(picture+sound) at index `at` (default append); 'V2' lays a silent video "
          "layer over the ongoing audio at program `from_ms` (audio:'keep' plays its "
          "own sound). `level` selects the energy take.",
          obj({"ref": {"type": "string"}, "level": {"type": "string", "enum": list(observe._LEVELS)},
               "channel": {"type": "string", "enum": ["V1", "V2"]},
               "at": {"type": "integer"}, "from_ms": {"type": "integer"},
               "audio": {"type": "string", "enum": ["keep", "mute"]},
               "reason": {"type": "string"}}, ["ref"])),
        S("trim", "Changes a cut's SOURCE in/out. Absolute (in_ms/out_ms) or relative "
          "(delta_in_ms/delta_out_ms; delta_in_ms:200 starts 200ms later). Targets a "
          "main-line seg_id, a V2 place_video op_id, or an A2 place_audio op_id "
          "(trims which part of the bed's source plays; its program window shifts "
          "to match).",
          obj({"target_id": {"type": "string"},
               "in_ms": {"type": "integer"}, "out_ms": {"type": "integer"},
               "delta_in_ms": {"type": "integer"}, "delta_out_ms": {"type": "integer"}},
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
          "to_ms for an op_id.",
          obj({"target_id": {"type": "string"}, "to_index": {"type": "integer"},
               "to_ms": {"type": "integer"},
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
          "neighbours; 'natural'~=1x); this is RECORDED and shown in read_state but "
          "the render does not bake speed into the export length yet. A SPEECH cut "
          "is never pitched/sped: 'faster'/'much_faster' shave removable dead-air + "
          "fillers, 'natural'/'slower' keep every pause. With seg_id -> that cut; "
          "without -> the whole main line.",
          obj({"seg_id": {"type": "string"},
               "pace": {"type": "string", "enum": list(act._PACE_STEPS)}},
              ["pace"])),
        S("split_screen", "Shows the MAIN LINE and a second source at once over the "
          "window [from_ms, to_ms] (program ms): template 'split_h' (side-by-side), "
          "'split_v' (stacked), or 'pip' (inset over the main line). The second cell "
          "source is either a map `ref` or a raw window `file`+`in_ms`+`out_ms` "
          "(seam-snapped to the nearest clean boundary). The second cell is silent "
          "unless audio:'keep'.",
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
              doc: dict) -> Tuple[str, dict, bool]:
    snap_info: Dict[str, Any] = {}
    try:
        # OBSERVE (read-only)
        if name == "read_state":
            return _json(observe.read_state(doc, ctx)), doc, False
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

        # ACT (mutate)
        if name == "place":
            new = act.place(doc, ctx.index, args["ref"], level=args.get("level", "balanced"),
                            channel=args.get("channel", "V1"), at=args.get("at"),
                            from_ms=args.get("from_ms"),
                            audio=args.get("audio"), reason=args.get("reason", ""))
        elif name == "trim":
            new = act.trim(doc, args["target_id"], in_ms=args.get("in_ms"), out_ms=args.get("out_ms"),
                           delta_in_ms=args.get("delta_in_ms"), delta_out_ms=args.get("delta_out_ms"))
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
    except Exception as e:  # a bad tool call must never crash the turn
        logger.exception("tools: %s failed", name)
        return _json({"error": f"{type(e).__name__}: {e}"}), doc, False


def _json(obj: Any) -> str:
    return json.dumps(obj, default=str)[:12000]


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

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

    for turn in range(max_turns):
        resp = llm.run(system=system, messages=convo, tools=tools,
                       max_tokens=max_tokens, cache_system=True)
        last_text = (resp.text or "").strip() or last_text
        convo.append(resp.assistant_message)
        if not resp.tool_calls:
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
                trace.append({"turn": turn, "name": tc.name, "args": tc.input or {},
                              "applied": False, "result": "posed to user"})
                continue
            obs, working, did = _dispatch(tc.name, tc.input or {}, ctx, working)
            changed = changed or did
            results.append(tool_result_block(tc.id, obs))
            trace.append({"turn": turn, "name": tc.name, "args": tc.input or {},
                          "applied": bool(did), "result": obs[:600]})
        convo.append(user_message(results))
        # ask_user PAUSES the turn: the user's next message is the answer.
        if asked and questions:
            break
    else:
        logger.info("tools: hit max_turns=%d; wrapping up", max_turns)

    awaiting = bool(questions)
    reply = last_text or (
        "Before I go further I need your call on a couple of things below."
        if awaiting else "Done.")
    return LoopResult(reply=reply, document=working, changed=changed, steps=steps,
                      trace=trace, questions=questions, awaiting_user=awaiting)
