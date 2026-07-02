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
from app.services.l3.observe import EditContext
from app.services.llm import LLMClient, tool_result_block, tool_spec, user_message

logger = logging.getLogger(__name__)

_MAX_TURNS = 12


@dataclass
class LoopResult:
    reply: str
    document: dict
    changed: bool = False
    steps: List[str] = field(default_factory=list)   # tool names called, in order
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
        S("read_state", "Look at the current edit: ordered cuts (pos, seg_id, ref, "
          "duration, channel, speaker, muted, text), channels in use, total length, "
          "and a feel narration. Call this first, and again after edits, to see "
          "what you changed.", obj({})),
        S("predict", "Project the program LENGTH under a proposed change WITHOUT "
          "applying it -- e.g. how long the edit would run if every cut were taken "
          "at 'tight', or after dropping/adding cuts. Use to hit a target length.",
          obj({"set_level": {"type": "string", "enum": list(observe._LEVELS)},
               "drop": {"type": "array", "items": {"type": "string"}},
               "add": {"type": "array", "items": {"type": "object", "properties": {
                   "ref": {"type": "string"}, "level": {"type": "string"}}}}})),
        S("validate", "Check the edit for STRUCTURAL problems (spans out of range, "
          "empty cuts, bad overlays). Empty result means clean.", obj({})),
        S("diagnose", "Editorial problems worth fixing (jump-cut risk, energy sags, "
          "over/under target length, redundant same-beat takes). Advice, not orders.",
          obj({})),
        S("affordances", "The menu of what you CAN do and to what: per-cut retake "
          "levels (tighter/wider), alternate takes, audio toggles, channels in use, "
          "and a pool of video moments you could add as cutaways.", obj({})),
        S("source_awareness", "See the clips as a CONTINUOUS, fully-addressable "
          "source: change-point lanes (who is present / speaking on camera / gaze / "
          "shot size / action over the whole clock), the cleanest seams, the "
          "impact/reveal peaks, and a scored cut INDEX per clip. Use this when you "
          "need a span that ISN'T a pre-baked cut -- e.g. a person's silent reaction "
          "-- then place it with place_span. Each clip is headed 'CLIP <file8>'.",
          obj({})),
        S("scan_source", "Query one clip's continuous timeline for spans matching a "
          "facet: `lane` (e.g. 'presence:p2', 'speaking', 'shot', 'speech', 'action') "
          "+ optional `match` (e.g. {state:'on'} or {subject:'p1'}). Each hit carries "
          "the full facets at its midpoint, so ask 'where is p2 on screen?' with lane "
          "'presence:p2' match {state:'on'} and read each hit's facets to see if they "
          "are also silent. `file` is the 'CLIP <file8>' id.",
          obj({"file": {"type": "string"}, "lane": {"type": "string"},
               "match": {"type": "object"}}, ["file", "lane"])),
        # --- ACT (edit verbs; each mutates the working document) ---
        S("place", "Add a cut from a map ref. channel 'V1' inserts on the MAIN LINE "
          "(picture+sound) at index `at` (default append); 'V2' lays a SILENT video "
          "cutaway over the ongoing audio at `from_ms` (add audio:'keep' to play its "
          "own sound). `level` sets the energy take.",
          obj({"ref": {"type": "string"}, "level": {"type": "string", "enum": list(observe._LEVELS)},
               "channel": {"type": "string", "enum": ["V1", "V2"]},
               "at": {"type": "integer"}, "from_ms": {"type": "integer"},
               "audio": {"type": "string", "enum": ["keep", "mute"]},
               "reason": {"type": "string"}}, ["ref"])),
        S("place_span", "Place an ARBITRARY source span (from source_awareness) -- "
          "not just a pre-baked cut. `file` is the 'CLIP <file8>' id; in_ms/out_ms "
          "are SOURCE ms into that clip. channel 'V1' = main line (picture+sound) at "
          "index `at`; 'V2' = video cutaway at program `from_ms`. axis:'speech' marks "
          "it audio-load-bearing; audio 'keep'/'mute' sets its sound. Use this to lift "
          "a silent reaction or any window the lanes revealed.",
          obj({"file": {"type": "string"}, "in_ms": {"type": "integer"},
               "out_ms": {"type": "integer"},
               "channel": {"type": "string", "enum": ["V1", "V2"]},
               "at": {"type": "integer"}, "from_ms": {"type": "integer"},
               "axis": {"type": "string", "enum": ["speech", "any"]},
               "audio": {"type": "string", "enum": ["keep", "mute"]},
               "content": {"type": "string"}, "reason": {"type": "string"}},
              ["file", "in_ms", "out_ms"])),
        S("trim", "Nudge a cut's SOURCE in/out. Absolute (in_ms/out_ms) or relative "
          "(delta_in_ms/delta_out_ms, e.g. delta_in_ms:200 starts 200ms later). "
          "Targets a main-line seg_id or a V2 op_id.",
          obj({"target_id": {"type": "string"},
               "in_ms": {"type": "integer"}, "out_ms": {"type": "integer"},
               "delta_in_ms": {"type": "integer"}, "delta_out_ms": {"type": "integer"}},
              ["target_id"])),
        S("remove", "Drop a main-line cut (seg_id) or an operation (op_id).",
          obj({"target_id": {"type": "string"}}, ["target_id"])),
        S("move", "Reorder a main-line cut to a new 0-based index.",
          obj({"seg_id": {"type": "string"}, "to_index": {"type": "integer"}},
              ["seg_id", "to_index"])),
        S("set_audio", "Mute or unmute a cut's SOURCE audio, keeping its picture. "
          "mute:true silences (e.g. a b-roll cutaway); mute:false plays its sound.",
          obj({"target_id": {"type": "string"}, "mute": {"type": "boolean"}},
              ["target_id", "mute"])),
        S("tighten", "Re-take main-line cut(s) at a different energy `level` for "
          "pacing. With seg_id -> just that cut; without -> every cut that has that "
          "level. Tighter = shorter/punchier.",
          obj({"seg_id": {"type": "string"}, "level": {"type": "string", "enum": list(observe._LEVELS)}},
              ["level"])),
        S("split_screen", "Show the MAIN LINE and a second source `ref` at the same "
          "time over the window [from_ms, to_ms]: template 'split_h' (side-by-side), "
          "'split_v' (stacked), or 'pip' (ref inset over the main line). The added "
          "cell is silent unless audio:'keep'. This is a user-owned look -- ask_user "
          "first. from_ms/to_ms are PROGRAM ms (see read_state).",
          obj({"ref": {"type": "string"},
               "template": {"type": "string", "enum": ["split_h", "split_v", "pip"]},
               "from_ms": {"type": "integer"}, "to_ms": {"type": "integer"},
               "level": {"type": "string", "enum": list(observe._LEVELS)},
               "audio": {"type": "string", "enum": ["keep", "mute"]},
               "reason": {"type": "string"}},
              ["ref", "template", "from_ms", "to_ms"])),
        # --- ASK (pause the turn for a user-owned decision) ---
        S("ask_user", "Pause and ask the user when a choice is genuinely THEIRS -- "
          "a split-screen/PiP layout, the delivery aspect/framing, or a big pacing "
          "tradeoff. Give each question 2+ CONCRETE options (they can also type "
          "their own). Calling this ENDS your turn; you resume when they answer. "
          "Don't ask about things you can reasonably decide yourself.",
          obj({"questions": {"type": "array", "items": {"type": "object", "properties": {
              "prompt": {"type": "string"},
              "options": {"type": "array", "items": {"type": "string"}},
              "allow_multiple": {"type": "boolean"}},
              "required": ["prompt", "options"]}}}, ["questions"])),
    ]


def _resolve_file(ctx: EditContext, ref: Any) -> str:
    """Resolve a brain-supplied clip id to a full file_id. Accepts a full id or
    the 8-char 'CLIP <file8>' prefix shown in source_awareness. Falls back to the
    raw value (validate/act will no-op on a bad id)."""
    s = str(ref or "").strip()
    if not s:
        return ""
    for fid in ctx.file_ids:
        if fid == s or fid.startswith(s):
            return fid
    return s


def _normalize_questions(args: Dict[str, Any]) -> List[dict]:
    """Coerce the brain's ask_user payload into surfaced questions: each needs a
    prompt + >= 2 concrete options (bad ones dropped)."""
    out: List[dict] = []
    for i, q in enumerate(args.get("questions") or []):
        if not isinstance(q, dict):
            continue
        prompt = str(q.get("prompt") or "").strip()
        opts = [str(o).strip() for o in (q.get("options") or []) if str(o).strip()]
        if not prompt or len(opts) < 2:
            continue
        out.append({"id": f"q{i}", "prompt": prompt, "options": opts,
                    "allow_multiple": bool(q.get("allow_multiple"))})
    return out


# --------------------------------------------------------------------------
# Dispatch: a tool call -> (observation text, new working doc, changed?)
# --------------------------------------------------------------------------

def _dispatch(name: str, args: Dict[str, Any], ctx: EditContext,
              doc: dict) -> Tuple[str, dict, bool]:
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
        if name == "source_awareness":
            return observe.source_awareness(ctx)[:12000], doc, False
        if name == "scan_source":
            return _json(observe.scan_source(ctx, args.get("file"), args.get("lane"),
                                             args.get("match"))), doc, False

        # ACT (mutate)
        if name == "place":
            new = act.place(doc, ctx.index, args["ref"], level=args.get("level", "balanced"),
                            channel=args.get("channel", "V1"), at=args.get("at"),
                            from_ms=args.get("from_ms"), audio=args.get("audio"),
                            reason=args.get("reason", ""))
        elif name == "place_span":
            fid = _resolve_file(ctx, args.get("file"))
            new = act.place_span(doc, fid, in_ms=args.get("in_ms"), out_ms=args.get("out_ms"),
                                 channel=args.get("channel", "V1"), at=args.get("at"),
                                 from_ms=args.get("from_ms"), audio=args.get("audio"),
                                 axis=args.get("axis", "any"), content=args.get("content", ""),
                                 reason=args.get("reason", ""))
        elif name == "trim":
            new = act.trim(doc, args["target_id"], in_ms=args.get("in_ms"), out_ms=args.get("out_ms"),
                           delta_in_ms=args.get("delta_in_ms"), delta_out_ms=args.get("delta_out_ms"))
        elif name == "remove":
            new = act.remove(doc, args["target_id"])
        elif name == "move":
            new = act.move(doc, args["seg_id"], args["to_index"])
        elif name == "set_audio":
            new = act.set_audio(doc, args["target_id"], mute=bool(args.get("mute")))
        elif name == "tighten":
            new = act.tighten(doc, ctx.index, seg_id=args.get("seg_id"), level=args.get("level", "tight"))
        elif name == "split_screen":
            new = act.split_screen(doc, ctx.index, args["ref"],
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
    tools = _specs()
    last_text = ""
    questions: List[dict] = []

    for _ in range(max_turns):
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
                continue
            obs, working, did = _dispatch(tc.name, tc.input or {}, ctx, working)
            changed = changed or did
            results.append(tool_result_block(tc.id, obs))
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
                      questions=questions, awaiting_user=awaiting)
