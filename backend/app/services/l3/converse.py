"""
Conversational L3: a chat-first assistant over a project's footage + current edit.

Design note (why it's shaped this way): a strong model edits best with a NEAR-
EMPTY frame and room to think in prose -- the same way it does when you just hand
it the footage and ask. Heavy rule-lists, a controlled vocabulary and a "return
JSON" demand turn an editor into a form-filler and cap quality. So the brain runs
in TWO steps:

  1) THINK -- a tiny frame (you're an editor, here's the footage with full lines,
     here's the timeline) and the model replies naturally in PROSE: it chats,
     proposes, asks, or -- once the user confirms -- applies an edit and ends with
     its final cut list (moments by id). This prose is what the user sees.
  2) EXTRACT -- a separate, dumb call with no taste that reads that reply and
     turns a CONFIRMED edit into data (ordered ids + level/track). Thinking never
     competes with formatting, and the model's nuance survives the hand-off.

Returns ``{reply, intent, brief, timeline, aspect, target_s}``; the caller owns
persistence and compiling the timeline. Fails OPEN: any parse/LLM error degrades
to a plain chat reply so a turn never hard-fails.
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


# THINK frame: deliberately minimal. Give the editor the footage + the way to
# refer to it, the chat etiquette (propose -> confirm -> apply), and then get out
# of its way. No taste rules, no arc lecture, no JSON demand -- those narrow a
# strong model and lower quality (verified). It thinks and replies in prose.
_THINK_SYSTEM = (
    "You are EDSO, a sharp video editor talking with someone about their footage. "
    "Below you can see the WHOLE shoot -- every clip and every usable moment, with "
    "the COMPLETE line of what is said -- and the CURRENT TIMELINE of the edit so "
    "far.\n\n"
    "Each moment reads like: clip8:m07 speech .82 [0:14-0:21] \"the full line\" · "
    "nrg:calm|balanced · dup:tg4*\n"
    "  - refer to a moment by its full id (e.g. ab12cd34:m07).\n"
    "  - the quoted text is the complete line -- judge from it.\n"
    "  - nrg = the energy LEVELS you can take it at (broad = the whole thing .. "
    "sharp = tightest); pick what fits.\n"
    "  - dup:tgN = the same content as others in that group (another take/angle); "
    "use only one of them.\n\n"
    "Just talk by default -- answer, discuss, give your honest editorial opinion. "
    "Don't change the edit on your own. When they want a change, propose it in a "
    "sentence or two and ask if they'd like it applied. When they confirm, apply "
    "it: edit like a real editor who watched everything, and END your message with "
    "your final cut list -- the moments in play order, each by id, with a word on "
    "why. Use only ids that appear below. (By default each cut plays in sequence; "
    "if you specifically want a silent video cutaway laid over the ongoing audio, "
    "just say so and where.)"
)

# EXTRACT step: a dumb parser with NO taste. It only turns a CONFIRMED, applied
# edit into data; otherwise it's a chat turn. Kept tiny + map-free so it's cheap.
_EXTRACT_SYSTEM = "You convert an editor's decision into data. Return ONLY JSON, nothing else."

_EXTRACT_INSTR = (
    "Below is a chat between a user and a video editor, ending with the editor's "
    "latest message. Decide the intent and, if an edit was just APPLIED, extract "
    "it.\n"
    "- intent = \"edit\" ONLY if the user CONFIRMED a change AND the editor's "
    "latest message commits to a concrete cut list. A proposal/question, or any "
    "turn the user hasn't confirmed, is intent = \"chat\".\n"
    "- When intent = \"edit\", extract the editor's FINAL cut list IN ORDER, each "
    "moment by the exact id it used. For each: copy the energy level if the editor "
    "named one (broad/calm/balanced/tight/sharp), set track (0 = main line, which "
    "is the default; >=1 only if the editor described an overlay) and from_ms if "
    "it gave an overlay anchor, and a short reason.\n"
    "- aspect: portrait/landscape/square if mentioned, else \"\". target_s: the "
    "target length in seconds if mentioned, else null. brief: one line summarising "
    "the applied edit.\n"
    "Return ONLY this JSON:\n"
    "{\"intent\": \"chat\"|\"edit\", \"brief\": \"\", \"aspect\": \"\", "
    "\"target_s\": null, \"timeline\": [{\"ref\": \"id\", \"level\": \"\", "
    "\"track\": 0, \"from_ms\": null, \"reason\": \"\"}]}\n\n"
    "CHAT:\n"
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

    Step 1 THINK: the editor reads the footage + timeline and replies in prose
    (chat / propose / apply). Step 2 EXTRACT: a separate dumb call turns a
    confirmed, applied edit into the structured cut list the caller compiles.
    The prose reply is what the user sees; the caller persists it and, on an
    edit, compiles the timeline."""
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

    system = _THINK_SYSTEM + "\n\n" + _context_block(file_ids, document)
    max_tokens = settings.autoedit_max_output_tokens
    # cache_system keeps the resident map cached across turns (Anthropic/Gemini;
    # a no-op on OpenAI) so multi-turn chat stays cheap and fast.
    try:
        resp = llm.run(system=system, messages=messages,
                       max_tokens=max_tokens, cache_system=True)
    except Exception:
        logger.exception("converse: think call failed for thread %s", thread_id)
        return ConverseResult(reply="Sorry -- I hit an error there. Mind trying again?")

    reply = (resp.text or "").strip() or "…"
    return _extract_decision(llm, messages, reply)


def _extract_decision(llm: LLMClient, messages: List[dict], reply: str) -> ConverseResult:
    """Turn the editor's prose reply into a routed result. The extractor only
    fires intent=edit when the user confirmed and a concrete cut list was applied;
    any failure degrades to a plain chat reply (the prose still reaches the user)."""
    convo = _render_convo(messages, reply)
    try:
        r = llm.run(system=_EXTRACT_SYSTEM,
                    messages=[user_message(_EXTRACT_INSTR + convo)],
                    max_tokens=2000)
        data = _parse(r.text) or {}
    except Exception:
        logger.exception("converse: extract call failed; treating as chat")
        data = {}

    intent = str(data.get("intent") or "chat").strip().lower()
    if intent not in ("chat", "edit"):
        intent = "chat"
    timeline = _coerce_timeline(data.get("timeline"))
    brief = str(data.get("brief") or "").strip()
    aspect = str(data.get("aspect") or "").strip().lower()
    if aspect not in ("portrait", "landscape", "square"):
        aspect = ""
    target_s = data.get("target_s")
    try:
        target_s = float(target_s) if target_s is not None else None
    except (TypeError, ValueError):
        target_s = None
    # An edit must carry an actual cut list, else it's just talk.
    if intent == "edit" and not timeline:
        intent = "chat"
    if intent == "edit" and not brief:
        brief = "applied edit"
    return ConverseResult(reply=reply, intent=intent, brief=brief,
                          timeline=timeline, aspect=aspect, target_s=target_s)


def _render_convo(messages: List[dict], reply: str) -> str:
    """Compact transcript for the extractor: the last few turns + the editor's
    just-made reply. Map-free, so this call stays cheap."""
    lines: List[str] = []
    for m in messages[-6:]:
        role = "USER" if m.get("role") == "user" else "EDITOR"
        text = str(m.get("content") or "").strip()
        if text:
            lines.append(f"{role}: {text}")
    lines.append(f"EDITOR (just now): {reply}")
    return "\n".join(lines)


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
