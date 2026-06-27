"""
Conversational L3: a chat-first assistant over a project's footage + current edit.

Design note (why it's shaped this way): a strong model edits best with a NEAR-
EMPTY frame and room to think in prose -- the same way it does when you just hand
it the footage and ask. Heavy rule-lists, a controlled vocabulary and a "return
JSON" demand turn an editor into a form-filler and cap quality. So the brain just
THINKS in prose:

  THINK -- a tiny frame (you're an editor, here's the footage with full lines,
  here's the timeline) and the model replies naturally in PROSE: it chats,
  answers, or PROPOSES a cut and ends with its cut list (moments by id, one per
  line, in play order). This prose is what the user sees.

We then HARVEST that cut list DETERMINISTICALLY out of the reply (no second LLM
guessing intent): every map-valid moment id the brain named, in order. A non-empty
harvest is a PROPOSAL -- the caller shows the user a Confirm button and only
applies (re-harvesting the same reply) once they say yes.

Returns ``{reply, proposal, brief}``; the caller owns persistence and, on
confirm, compiling the proposal. Fails OPEN: any LLM/parse error degrades to a
plain chat reply with no proposal, so a turn never hard-fails.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
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
    # The chat brain IS the editor. When it proposes a cut we HARVEST its cut list
    # deterministically out of its own reply (every map-valid moment id it named,
    # in play order) -- no second LLM guessing intent. A non-empty proposal means
    # the user is shown a Confirm button; nothing is applied until they say yes.
    proposal: List[dict] = field(default_factory=list)
    brief: str = ""           # one-line label for the proposed edit


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
    "Don't change the edit on your own. When they want a change (or ask you to "
    "build the edit), edit like a real editor who watched everything and PROPOSE "
    "the cut: a sentence on the idea, then END your message with your cut list -- "
    "the moments in PLAY ORDER, each on its OWN LINE led by its id, with a word on "
    "why. The user gets a Confirm button to apply it, so present the cut as a "
    "proposal -- don't claim it's already applied. Use only ids that appear below. "
    "(By default each cut plays in sequence; if you specifically want a silent "
    "video cutaway laid over the ongoing audio, just say so and where.)"
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

    THINK: the editor reads the footage + timeline and replies in prose -- it
    chats, answers, or proposes a cut (ending with its cut list). We then HARVEST
    that cut list deterministically out of the reply (``harvest_cut_list``); a
    non-empty harvest is a PROPOSAL the user confirms before anything is applied.
    No second LLM guesses intent -- the Confirm button is the intent."""
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
    proposal = harvest_cut_list(reply, file_ids)
    brief = _latest_user_text(messages) if proposal else ""
    return ConverseResult(reply=reply, proposal=proposal, brief=brief)


# --------------------------------------------------------------------------
# Deterministic cut-list harvest
# --------------------------------------------------------------------------
# A moment id is <fid8>:m## (see footage_map). Match it generously and validate
# every hit against the live map index, so prose can never inject a bogus cut.
_ID_RE = re.compile(r"\b[0-9A-Za-z]{4,}:m\d{1,3}\b")
# A "cut-list line" leads with an id after an optional number/bullet + markdown
# bold -- exactly the shape the brain is told to end with (one moment per line).
_LINE_ID_RE = re.compile(r"^\s*(?:\d{1,3}[.)]|[-*•])?\s*\*{0,2}\s*([0-9A-Za-z]{4,}:m\d{1,3})\b")
_LEVELS = ("broad", "calm", "balanced", "tight", "sharp")


def harvest_cut_list(reply: str, file_ids: List[str]) -> List[dict]:
    """Pull the editor's proposed cut list out of its own prose, deterministically.

    Strategy (verified the brain emits valid ids reliably): prefer the brain's
    FINAL list -- lines that LEAD with a moment id -- since that's the authoritative
    play order; fall back to every id mentioned, in first-mention order. Each id is
    validated against the live footage map (bogus/hallucinated ids are dropped),
    deduped, and given the energy level the brain named on that line if any (else
    the compiler defaults to balanced). What the user saw the brain pick is exactly
    what gets cut -- no LLM re-guess. Returns [] on any failure (treated as chat)."""
    if not reply or not file_ids:
        return []
    try:
        # assemble_map returns a wrapper; the clip/moment tree the index walks
        # lives under "struct".
        struct = footage_map.assemble_map(file_ids).get("struct") or {}
        index = arrange._MapIndex(struct)
    except Exception:
        logger.exception("converse: map build failed during harvest")
        return []

    candidates = _list_line_candidates(reply) or _inline_candidates(reply)
    out: List[dict] = []
    seen: set[str] = set()
    for ref, context in candidates:
        if ref in seen or not index.has(ref):
            continue
        seen.add(ref)
        entry: dict = {"ref": ref}
        ctx = context.lower()
        level = next((lv for lv in _LEVELS if lv in ctx and index.level_ok(ref, lv)), "")
        if level:
            entry["level"] = level
        out.append(entry)
    return out


def _list_line_candidates(reply: str) -> List[tuple[str, str]]:
    """(id, rest-of-line) for every line that LEADS with a moment id."""
    pairs: List[tuple[str, str]] = []
    for line in reply.splitlines():
        m = _LINE_ID_RE.match(line)
        if m:
            pairs.append((m.group(1), line[m.end():]))
    return pairs


def _inline_candidates(reply: str) -> List[tuple[str, str]]:
    """(id, short trailing window) for every id mentioned anywhere in the reply."""
    pairs: List[tuple[str, str]] = []
    for m in _ID_RE.finditer(reply):
        pairs.append((m.group(0), reply[m.end():m.end() + 32]))
    return pairs


def _latest_user_text(messages: List[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            text = m.get("content")
            if isinstance(text, str) and text.strip():
                return text.strip()[:200]
    return "edit"
