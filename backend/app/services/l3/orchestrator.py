"""
L3 orchestrator: the Claude Opus agentic loop that produces Edit Documents.

Run shape (the Cursor pattern):
  inner loop   reason -> call tool -> observe -> reason ... until the agent
               calls a terminal tool (`finalize` -> document ready,
               `ask_user` -> pause for the human) or trips a guardrail.
  outer loop   every pause/finish returns control to the user; their answer
               or feedback is appended as a user turn and the SAME context is
               resumed (messages are persisted neutrally in edit_turns).

Cost containment: the system prompt + tool specs + clip catalog form a stable,
cacheable prefix (Anthropic prompt caching), so each loop iteration only pays
incremental tokens. Guardrails bound the iteration count.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from procrastinate import RetryStrategy

from app.config import get_settings
from app.services.jobs import app
from app.services.l3 import store
from app.services.l3.catalog import build_catalog, render_catalog_text
from app.services.l3.takes import build_take_groups, render_take_groups_text
from app.services.l3.tools import TERMINAL_TOOLS, TOOL_SPECS, EditSession, execute_tool
from app.services.llm.anthropic_client import AnthropicClient
from app.services.llm.base import tool_result_block, user_message

logger = logging.getLogger(__name__)


SYSTEM_PROMPT_TEMPLATE = """You are an expert video editor agent. You plan edits over a library of \
single-take clips that have already been deeply analyzed; you work entirely from that text \
analysis -- you cannot watch the video. Your deliverable is an EDIT DOCUMENT: an ordered \
timeline of segments with exact, machine-snapped cut points, plus the reasoning behind every \
choice.

THE CONTRACT (two brains):
- YOU decide what material to use, in what order, with what intent, and roughly where beats \
start and end. You are the only source of creativity, story, and taste here. Be decisive and \
opinionated; bland safe choices make bad edits.
- THE ENGINE (your tools) decides exact frames. Cut-cost grids (0=ideal seam .. 1=forbidden) \
were computed mechanically per channel (dialogue/beat/action/camera). When you add_segment with \
rough times, both ends are snapped to the cleanest nearby seams and you get back exact \
timestamps + costs. Trust the snap; never fight it by a few ms. If a snap comes back dirty \
(cost > 0.45), the moment you wanted has no clean exit -- pick a different moment, don't force it.

THE MATERIAL:
- Every clip is ONE continuous take (no internal cuts). Its footage log is chronological: \
events (with actor p-ids), persons (with appearance + voice links), reactions, gaze, camera \
craft spans, speaking spans.
- The catalog below is a teaser. Before using any clip's material, read_clip it and ground \
every in/out you propose in its logged event timestamps. Never invent timestamps; never exceed \
a clip's duration; only use clips in scope.

WORKFLOW (each run):
1. Interpret the brief -> set_brief. Record every default you assumed in `assumptions`.
2. Skim the catalog; read_clip the promising clips.
3. set_outline: 2-6 beats with purpose + intent (hook first when the brief implies an audience).
4. Build the timeline beat by beat: query_seams to scout, add_segment with rough times, \
content + rationale on every segment. Set priority (1=core, 5=expendable filler).
5. timeline_status; fix warnings that matter (jump cuts, dirty seams, micro-segments).
6. fit_duration to the target. If it can't fit by trimming, drop a whole segment yourself.
7. finalize with an honest summary (what you chose, what's weak, what you'd tweak next).

ASKING THE USER (ask_user):
- Draft FIRST, ask second: you must have a complete, watchable timeline before asking anything. \
Apply your best default, then ask only the genuine forks where the answer would change the cut \
(target length, ending choice, include/exclude a moment, tone). 0-4 questions, each with a \
default. Never ask what the footage already answers.
- When the user replies, apply their answers with minimal disruption: re-cut only affected \
segments, then finalize again.

STYLE OF THE CUT:
- Respect the grain of the footage: cut dialogue at sentence/turn seams, action on impacts, \
music on beats. Enter scenes as late as possible, leave as early as possible. Vary segment \
lengths; monotony reads as machine-made.
- TAKES: when the same content was delivered more than once (a retake, or a flub-then-retry \
inside one clip), they appear as TAKE GROUPS below. Call compare_takes to get an objective \
scorecard for each alternative, weight it by the brief (polished -> fluency/clean; raw -> \
energy/authenticity), pick the strongest span, and say why in the rationale. Don't eyeball it \
from the logline.

CLIP CATALOG (scope for this thread):
{catalog}

{take_groups}
"""


def _render_system_prompt(catalog, take_groups) -> str:
    tg_text = render_take_groups_text(take_groups)
    return SYSTEM_PROMPT_TEMPLATE.format(
        catalog=render_catalog_text(catalog),
        take_groups=tg_text or "(no repeated-content take groups detected)",
    )


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------

def run_thread(thread_id: str) -> None:
    """Run the agent until finalize / ask_user / guardrail. Safe to call again
    after appending a user turn -- it resumes from persisted context."""
    settings = get_settings()
    thread = store.get_thread(thread_id)
    if thread is None:
        logger.info("L3: thread %s gone; skipping.", thread_id)
        return

    store.set_thread_status(thread_id, "drafting")

    # Built once from the DB; shared by the (cacheable) system prompt and the
    # tool session so the catalog/take-groups are computed a single time.
    catalog = build_catalog(thread["file_ids"])
    take_groups = build_take_groups(thread["file_ids"])

    system = _render_system_prompt(catalog, take_groups)
    messages = store.load_messages(thread_id)
    if not messages:
        logger.error("L3: thread %s has no messages; nothing to do.", thread_id)
        store.set_thread_status(thread_id, "failed")
        return

    # Resume the working document from the latest snapshot (empty on first run).
    doc, _version = store.latest_document(thread_id)
    session = EditSession(
        thread_id=thread_id,
        file_ids=thread["file_ids"],
        catalog=catalog,
        take_groups=take_groups,
        document=doc or {},
    )

    client = AnthropicClient(model=settings.l3_model)
    terminal: Optional[str] = None

    for iteration in range(settings.l3_max_iterations):
        resp = client.run(
            system=system,
            messages=messages,
            tools=TOOL_SPECS,
            max_tokens=settings.l3_max_output_tokens,
            cache_system=True,
            thinking_budget=settings.l3_thinking_budget_tokens,
            effort=settings.l3_effort,
        )
        messages.append(resp.assistant_message)
        store.append_turn(thread_id, "assistant", resp.assistant_message["content"], resp.usage)

        if not resp.tool_calls:
            # The model stopped talking without finalize: surface what it said,
            # snapshot the document, and hand control to the user.
            logger.info("L3: %s ended without terminal tool (iter %d)", thread_id, iteration)
            terminal = "no_tool"
            break

        result_blocks = []
        for call in resp.tool_calls:
            logger.info("L3 %s iter %d: %s(%s)", thread_id, iteration, call.name,
                        str(call.input)[:200])
            result = execute_tool(session, call.name, call.input)
            result_blocks.append(tool_result_block(call.id, result))
            if call.name in TERMINAL_TOOLS:
                terminal = call.name

        tool_msg = user_message(result_blocks)
        messages.append(tool_msg)
        store.append_turn(thread_id, "user", result_blocks)

        if terminal:
            break

    version = store.save_document(thread_id, session.document)

    if terminal == "finalize":
        store.set_thread_status(thread_id, "ready")
        logger.info("L3: thread %s ready (document v%d, %d segments)",
                    thread_id, version, len(session.document.get("timeline", [])))
    elif terminal == "ask_user":
        store.set_thread_status(thread_id, "awaiting_user")
        logger.info("L3: thread %s awaiting user (%d question(s), document v%d)",
                    thread_id, len(session.document.get("open_questions", [])), version)
    elif terminal == "no_tool":
        store.set_thread_status(thread_id, "awaiting_user")
    else:
        # Guardrail tripped: iterations exhausted mid-flight.
        session.document.setdefault("diagnostics", {})["guardrail"] = (
            f"stopped after {settings.l3_max_iterations} iterations without finalize"
        )
        store.save_document(thread_id, session.document)
        store.set_thread_status(thread_id, "awaiting_user")
        logger.warning("L3: thread %s hit iteration guardrail", thread_id)


# --------------------------------------------------------------------------
# Procrastinate task + entry points
# --------------------------------------------------------------------------

# Own queue: workers running older code (no l3 task registered) must never
# fetch these jobs -- procrastinate permanently fails unknown tasks. Run a
# worker with WORKER_QUEUES=gpu,l3 (or all queues) to serve them.
@app.task(name="l3_edit_turn", queue="l3", retry=RetryStrategy(max_attempts=2, exponential_wait=5))
def l3_edit_turn(thread_id: str) -> None:
    try:
        run_thread(thread_id)
    except Exception:
        logger.exception("L3 run failed for thread %s", thread_id)
        try:
            store.set_thread_status(thread_id, "failed")
        except Exception:
            pass
        raise


def _defer_run(thread_id: str) -> None:
    """Enqueue from the API process via a short-lived per-call App + task name
    (same pattern as upload.py: the shared global app isn't open in the API,
    and a per-call connector can't be raced shut by another thread)."""
    from procrastinate import App, PsycopgConnector

    enqueue_app = App(connector=PsycopgConnector(
        conninfo=get_settings().database_url, min_size=1, max_size=2))
    with enqueue_app.open():
        enqueue_app.configure_task("l3_edit_turn", queue="l3").defer(thread_id=thread_id)


def start_thread(user_id: str, file_ids: List[str], brief: str) -> str:
    """Create a thread, seed the first user turn, enqueue the agent."""
    thread_id = store.create_thread(user_id, file_ids, brief)
    store.append_turn(thread_id, "user", brief or "Plan the best edit from these clips.")
    _defer_run(thread_id)
    return thread_id


def send_user_message(thread_id: str, text: str) -> None:
    """Answers to open questions, or any feedback on the current plan; resumes
    the agent with full context."""
    store.append_turn(thread_id, "user", text)
    store.set_thread_status(thread_id, "drafting")
    _defer_run(thread_id)
