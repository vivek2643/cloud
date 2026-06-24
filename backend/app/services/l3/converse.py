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
from dataclasses import dataclass
from typing import List, Optional

from app.config import get_settings
from app.services.l3 import arrange, footage_map, store
from app.services.llm import LLMClient, get_llm

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


_SYSTEM = (
    "You are EDSO, a friendly, sharp video-editing assistant. You are CHATTING "
    "with the user about their project. You can see a MAP of all their footage "
    "(every clip and its usable moments) and the CURRENT TIMELINE of the edit so "
    "far.\n\n"
    "How to behave:\n"
    "- By default, just TALK. Answer questions, explain what's in the footage, "
    "discuss ideas, give opinions, ask clarifying questions. Be concise and warm.\n"
    "- You do NOT edit on your own. Chatting never changes the timeline.\n"
    "- When the user seems to want a change to the edit, do NOT do it yet. First "
    "PROPOSE what you'd do in one or two sentences and ASK them to confirm (e.g. "
    "\"Want me to apply that?\"). Keep intent = \"chat\" for a proposal.\n"
    "- ONLY when the user has clearly confirmed in their LATEST message (\"yes\", "
    "\"do it\", \"go ahead\", etc.) do you set intent = \"edit\" and put a clear, "
    "self-contained instruction in \"brief\" describing exactly the edit to make "
    "(what to keep / cut / reorder, the pacing, the length). Your reply should "
    "tell them you're applying it.\n"
    "- If you are unsure whether they confirmed, ASK -- do not edit.\n\n"
    "Return ONLY JSON (no prose around it): {\"reply\": \"<what you say to the "
    "user>\", \"intent\": \"chat\"|\"edit\", \"brief\": \"<edit instruction; only "
    "when intent=edit, else empty>\"}"
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
    try:
        resp = llm.run(system=system, messages=messages,
                       max_tokens=settings.autoedit_max_output_tokens)
    except Exception:
        logger.exception("converse: llm call failed for thread %s", thread_id)
        return ConverseResult(reply="Sorry -- I hit an error there. Mind trying again?")

    data = _parse(resp.text)
    if not data:
        # Model answered in prose rather than JSON -> treat it as a chat reply.
        return ConverseResult(reply=(resp.text or "").strip() or "…")

    intent = str(data.get("intent") or "chat").strip().lower()
    if intent not in ("chat", "edit"):
        intent = "chat"
    brief = str(data.get("brief") or "").strip()
    reply = str(data.get("reply") or "").strip()
    if intent == "edit" and not brief:
        intent = "chat"            # no instruction -> nothing to run; keep talking
    if not reply:
        reply = "On it -- applying that now." if intent == "edit" else "…"
    return ConverseResult(reply=reply, intent=intent, brief=brief)


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
