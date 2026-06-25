"""
Conversational L3: a chat-first assistant over a project's footage + current edit.

Default behaviour is plain chat -- answer questions about the clips and the
current timeline, discuss ideas, ask clarifying questions. It NEVER edits on its
own. When it senses the user wants a change, it PROPOSES the change in words and
asks for confirmation; only once the user clearly agrees does it signal an EDIT
with a concise, self-contained instruction, which the caller hands to the
deterministic arranger (``auto_edit.make_edit``) as a refinement of the cut.

One cheap LLM call per user message. Returns ``{reply, intent, brief}``; the
caller owns persistence and running the edit. Fails OPEN: a parse/LLM error
degrades to a plain chat reply so a turn never hard-fails.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from app.config import get_settings
from app.services.l3 import arrange, footage_map, store
from app.services.llm import LLMClient, get_llm, user_message

logger = logging.getLogger(__name__)

# The chat brain gets the WHOLE footage map (every clip + moment) so it has the
# same total awareness the arranger does -- like handing a coding agent the full
# repo map. The cap is only a runaway guard for pathologically large libraries
# (~35k chars for 7 clips / 263 moments; ~9k tokens), well within context.
_MAP_CHAR_CAP = 200_000


@dataclass
class ConverseResult:
    reply: str
    intent: str = "chat"      # "chat" | "edit"
    brief: str = ""           # the agreed edit instruction (only when intent="edit")
    # The chat brain IS the editor: on an edit it returns the actual cut list it
    # picked (moment ids + level + track), which the caller compiles directly --
    # no separate arranger re-guess. Empty -> caller falls back to the arranger.
    timeline: List[dict] = field(default_factory=list)
    aspect: str = ""          # "portrait" | "landscape" | "square" (optional)
    target_s: Optional[float] = None   # target length in seconds (optional)


_SYSTEM = (
    "You are EDSO, a friendly, sharp video-editing assistant -- AND the editor "
    "itself. You are CHATTING with the user about their project. You can see a MAP "
    "of all their footage (every clip and its usable moments, with the FULL line "
    "of what is said) and the CURRENT TIMELINE of the edit so far.\n\n"
    "Map notation (one moment per line):\n"
    "  m07 speech S1 .82 [0:14-0:21] \"the full line text\" · nrg:calm|balanced "
    "(+3 atoms) · dup:tg4*\n"
    "  - the id is <clip8>:<m##>; refer to a moment by its FULL id (e.g. "
    "ab12cd34:m07).\n"
    "  - nrg lists the energy LEVELS you may take it at: broad (whole answer) .. "
    "balanced (one thought) .. sharp (tightest). Pick the level that fits pacing.\n"
    "  - 'dup:tgN' = the same content as others in group tgN (another take/angle); "
    "the '*' is the best take. Use exactly ONE moment per dup group.\n\n"
    "How to behave:\n"
    "- By default, just TALK. Answer questions, explain what's in the footage, "
    "discuss ideas, give opinions, ask clarifying questions. Be concise and warm.\n"
    "- You do NOT edit on your own. Chatting never changes the timeline.\n"
    "- When the user seems to want a change, do NOT do it yet. First PROPOSE what "
    "you'd do in one or two sentences (you may name the exact moments) and ASK "
    "them to confirm (e.g. \"Want me to apply that?\"). Keep intent=\"chat\".\n"
    "- ONLY when the user has clearly confirmed in their LATEST message (\"yes\", "
    "\"do it\", \"go ahead\"): set intent=\"edit\" AND return the COMPLETE cut "
    "list you decided on in \"timeline\" -- this list is exactly what gets cut, so "
    "include every moment in play order. Also give a short \"brief\".\n"
    "- Build the timeline like a real editor: open strong, build, land; cut "
    "filler/slates/banter/off-brief; never repeat a line (one moment per dup "
    "group); choose the energy level per cut for pacing; respect any target "
    "length. Everything on track 0 is the main line and plays back-to-back; put "
    "b-roll/cutaways on track 1 (it covers the picture while the main audio "
    "continues) with a from_ms anchor.\n"
    "- Use ONLY moment ids that appear in the MAP. If you are unsure whether they "
    "confirmed, ASK -- do not edit.\n\n"
    "Return ONLY JSON (no prose around it):\n"
    "{\"reply\": \"<what you say to the user>\", "
    "\"intent\": \"chat\"|\"edit\", "
    "\"brief\": \"<one-line edit instruction; only when intent=edit>\", "
    "\"aspect\": \"portrait\"|\"landscape\"|\"square\", "
    "\"target_s\": <target length in seconds or null>, "
    "\"timeline\": [{\"ref\": \"<full moment id>\", \"level\": \"balanced\", "
    "\"track\": 0, \"from_ms\": null, \"reason\": \"<short why>\"}]}\n"
    "Leave \"timeline\" empty ([]) for a plain chat turn or a proposal; fill it "
    "ONLY on the confirmed edit turn."
)


# One internal pass on a COMMITTED edit only: the brain critiques its own cut
# list before it lands (draft -> critique -> final), the same self-check a real
# editor does on a first assembly. Plain chat turns never trigger this.
_CRITIQUE_PROMPT = (
    "Before this is applied, review your own cut list as a tough editor, then "
    "return the FINAL version in the SAME JSON schema. Check, and fix if needed:\n"
    "- ARC: does it open on the strongest moment, build, and land? Reorder if a "
    "weak cut opens or a strong one is buried.\n"
    "- DEDUP: is any line or content repeated? Keep exactly ONE moment per dup "
    "group (the best take) and drop the rest.\n"
    "- FILLER: cut anything off-brief, slates, dead air, rambling tails; tighten "
    "each cut to the energy level that carries the point.\n"
    "- DELIVERY: for each kept moment, is the chosen level right for the pacing? "
    "Adjust levels.\n"
    "- LENGTH: if there is a target, hit it -- trim the weakest cuts to fit.\n"
    "Return ONLY the JSON (reply, intent, brief, aspect, target_s, timeline), "
    "keeping intent=\"edit\". If the draft was already right, return it unchanged."
)


def _context_block(file_ids: List[str], document: Optional[dict]) -> str:
    parts: List[str] = []
    try:
        text = (footage_map.assemble_map(file_ids).get("text") or "") if file_ids else ""
        if len(text) > _MAP_CHAR_CAP:
            text = text[:_MAP_CHAR_CAP] + "\n…(truncated)"
        if text:
            parts.append("FOOTAGE MAP:\n" + text)
    except Exception:
        logger.exception("converse: map build failed (continuing without it)")
    overlay = arrange.timeline_overlay(document)
    parts.append("CURRENT TIMELINE:\n" + overlay if overlay
                 else "CURRENT TIMELINE: (empty -- no edit drafted yet)")
    return "\n\n".join(parts)


def respond(thread_id: str, *, llm: Optional[LLMClient] = None) -> ConverseResult:
    """Produce the assistant's reply to the latest user turn on a thread.

    Reads the thread's message history + footage map + current timeline and
    returns what to say plus whether the user has confirmed an edit (and, if so,
    the instruction to run). The caller appends the reply turn and, on an EDIT,
    runs the arranger."""
    settings = get_settings()
    if llm is None:
        llm = get_llm(provider=settings.autoedit_provider or None,
                      model=settings.autoedit_model or None)

    thread = store.get_thread(thread_id)
    file_ids = (thread or {}).get("file_ids") or []
    document, _ = store.latest_document(thread_id)
    messages = store.load_messages(thread_id)
    if not messages:
        return ConverseResult(reply="Tell me what you'd like to do with these clips.")

    system = _SYSTEM + "\n\n" + _context_block(file_ids, document)
    max_tokens = settings.autoedit_max_output_tokens
    # cache_system keeps the resident map cached across turns (Anthropic/Gemini;
    # a no-op on OpenAI) so multi-turn chat stays cheap and fast.
    try:
        resp = llm.run(system=system, messages=messages,
                       max_tokens=max_tokens, cache_system=True)
    except Exception:
        logger.exception("converse: llm call failed for thread %s", thread_id)
        return ConverseResult(reply="Sorry -- I hit an error there. Mind trying again?")

    result = _to_result(resp.text)

    # Reasoning loop -- ONLY on a committed edit that produced a cut list: have
    # the brain critique its own draft and return the final timeline before it
    # lands. One extra call on edit turns; plain chat stays single-call/fast.
    if result.intent == "edit" and result.timeline:
        try:
            crit_messages = list(messages) + [
                resp.assistant_message, user_message(_CRITIQUE_PROMPT)]
            resp2 = llm.run(system=system, messages=crit_messages,
                            max_tokens=max_tokens, cache_system=True)
            refined = _to_result(resp2.text)
            if refined.intent == "edit" and refined.timeline:
                result = refined
        except Exception:
            logger.exception("converse: critique pass failed; keeping draft")

    return result


def _to_result(text: Optional[str]) -> ConverseResult:
    """Parse one model turn into a ConverseResult. A non-JSON answer degrades to
    a plain chat reply so a turn never hard-fails."""
    data = _parse(text)
    if not data:
        return ConverseResult(reply=(text or "").strip() or "…")

    intent = str(data.get("intent") or "chat").strip().lower()
    if intent not in ("chat", "edit"):
        intent = "chat"
    brief = str(data.get("brief") or "").strip()
    reply = str(data.get("reply") or "").strip()
    timeline = _coerce_timeline(data.get("timeline"))
    aspect = str(data.get("aspect") or "").strip().lower()
    if aspect not in ("portrait", "landscape", "square"):
        aspect = ""
    target_s = data.get("target_s")
    try:
        target_s = float(target_s) if target_s is not None else None
    except (TypeError, ValueError):
        target_s = None
    if intent == "edit" and not brief:
        intent = "chat"            # no instruction -> nothing to run; keep talking
    if not reply:
        reply = "On it -- applying that now." if intent == "edit" else "…"
    return ConverseResult(reply=reply, intent=intent, brief=brief,
                          timeline=timeline, aspect=aspect, target_s=target_s)


def _coerce_timeline(raw: Any) -> List[dict]:
    """Keep only well-formed cut entries; validation against the map happens in
    the compiler (``arrange._coerce_placements``), so here we just shape them."""
    if not isinstance(raw, list):
        return []
    out: List[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("ref") or "").strip()
        if not ref:
            continue
        entry: dict = {"ref": ref}
        if item.get("level"):
            entry["level"] = str(item["level"]).strip().lower()
        if item.get("track") is not None:
            try:
                entry["track"] = int(item["track"])
            except (TypeError, ValueError):
                pass
        if item.get("from_ms") is not None:
            try:
                entry["from_ms"] = int(item["from_ms"])
            except (TypeError, ValueError):
                pass
        if item.get("reason"):
            entry["reason"] = str(item["reason"]).strip()
        out.append(entry)
    return out


def _parse(text: Optional[str]) -> Optional[dict]:
    if not text:
        return None
    raw = text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
