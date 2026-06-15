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
from app.services.l3 import sync as sync_mod
from app.services.l3.angle_menu import render_synced_angles_text
from app.services.l3.catalog import build_catalog, render_catalog_text
from app.services.l3.content import (
    build_content_map,
    build_overlap_index,
    render_content_map_text,
    render_overlap_text,
)
from app.services.l3.principles import render_principles_text
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

THE SPINE (decide this FIRST, before any segment):
Every edit has a load-bearing through-line -- the SPINE -- that everything else serves. \
Declaring it fixes which channel is LOCKED (irreplaceable) and which is FREE (coverable / \
scoreable):
- dialogue: the audio storyline (interview / VO) is locked; the VIDEO is free -> B-roll covers \
the picture while the spine audio runs underneath. Cut on dialogue seams.
- music: an uploaded music bed is locked; the VIDEO is free -> clips are coverage cut onto \
beats / sections / the energy curve. Cut on the beat.
- visual: the PICTURE is locked (on-screen text, a demo / tutorial, a product reveal, a \
performance) -- never cover it; the audio is free to score. Cut on action / visual.
- sync: BOTH are locked -- the A and V are one unit (a punchline with the face, a sync-sound \
hit). Atomic; do not split.
- other: anything else -- set a `label` and `locked_channels` explicitly.
Inside a region, mark do-not-cover spans (on-screen text, key reveals) as `protected_windows`.
SAFE DEFAULT: decoupling A/V is the privileged move. When unsure, choose 'sync' (keep them \
together) or ask -- never silently cover a clip whose picture was the point. An edit may hold \
multiple time-ordered regions if it shifts mode (montage hook -> testimonial). Background music, \
multicam, and B-roll are ways material ATTACHES to the spine, not spine kinds.

WORKFLOW (each run):
1. Interpret the brief -> set_brief. Record every default you assumed in `assumptions`.
2. Decide the SPINE -> set_spine (see THE SPINE above). This frames every later choice.
2b. (optional) set_principles to bias the cut's STYLE -- weighted tendencies the \
engine honors (favor_speaker, reward_reaction, shot_variety, pace, anti_metronome, \
hook_first, tighten_dead_air, ...). Set only the knobs you want to push; the rest run \
at sensible defaults.
3. Survey the CONTENT MAP -- it shows what EVERY clip contains. The CONTENT OVERLAP \
block marks which clips are duplicate takes/angles (same content) vs. distinct material. \
read_clip the promising clips for exact timestamps. Distinct-content clips are not \
interchangeable: don't build the whole edit from one and silently drop the rest.
4. set_outline: 2-6 beats with purpose + intent (hook first when the brief implies an audience).
5. Build the timeline beat by beat: query_seams to scout, add_segment with rough times, \
content + rationale on every segment. Set priority (1=core, 5=expendable filler). If the \
SYNCED ANGLES block lists a second camera for this material, call read_angles over the beat \
to see who is speaking vs. listening per moment, then pick_angle to follow the focus -- don't \
ride one camera through a whole conversation.
6. timeline_status; fix warnings that matter (jump cuts, dirty seams, micro-segments). \
Check its content_coverage: any clip listed there has UNIQUE content used nowhere -- \
confirm each omission is intentional (off-topic / redundant / weaker take) or add it.
7. fit_duration to the target. If it can't fit by trimming, drop a whole segment yourself.
8. finalize with an honest summary (what you chose, what's weak, what you'd tweak next).

ASKING THE USER (ask_user):
- Draft FIRST, ask second: you must have a complete, watchable timeline before asking anything. \
Apply your best default, then ask only the genuine forks where the answer would change the cut \
(target length, ending choice, include/exclude a moment, tone). 0-4 questions, each with a \
default. Never ask what the footage already answers.
- When the user replies, apply their answers with minimal disruption: re-cut only affected \
segments, then finalize again.

A/V LAYERS (decoupling video from audio):
- The spine is the base, coupled layer (each segment's picture + its own audio). On top of it \
you place LAYER OPERATIONS, checked against the spine's locks so you can't express an illegal edit:
  * place_video -- lay another clip's PICTURE over a range while the spine AUDIO keeps playing: \
B-roll / cutaway over an UNSYNCED clip. Needs a spine that frees video (dialogue/music).
  * pick_angle -- cut the picture to a VERIFIED synced SECOND ANGLE of the spine clip (see \
MULTICAM below). A normal picture cut, not B-roll.
  * place_audio -- lay a music bed / ambience / SFX under a range (duck it under dialogue), or \
replace spine audio with a cleaner source.
  * split_edit -- J/L cut: offset the audio cut from the video cut at a seam.
  * set_level -- gain/duck/mute a role over a range.
- These are layers, not spine edits: the spine still owns the clock. Build a solid spine FIRST. \
For decorative B-roll / beds the default is coupled -- don't add them just to show off.

MULTICAM / ANGLES (two cameras of the SAME moment -- e.g. an interview shot from two angles):
- The SYNCED ANGLES block (below, when present) lists pairs already VERIFIED as the same moment \
from two cameras, with their exact offset -- these are real second angles, not B-roll. (If a pair \
you suspect isn't listed, align_clips checks any two clips on demand.)
- DRIVE the cut from the facts, not from a quota. For a synced beat, call read_angles over the \
range: it returns, per moment, which camera shows the SPEAKER vs. a LISTENER, the shot size, and \
any reaction. Then pick_angle to FOLLOW THE FOCUS: stay on the speaker by default; cut to the \
listener when an answer runs long (a breathing/listening beat) or when there's a genuine reaction \
(a real laugh/nod/surprise the menu flags). That is the difference from B-roll -- you are tracking \
who/what matters, not filling time. Riding one camera through a whole conversation is the #1 \
failure; cutting metronomically for blind variety is the #2. Cut on dialogue seams; let your \
principles (favor_speaker, reward_reaction, shot_variety) bias the balance.
- pick_angle is for VERIFIED synced angles only; for unsynced coverage / B-roll use place_video. \
The same read_angles logic generalizes to action (cut to the camera framing the action beat) and \
music (cut on sections) when those are the spine.

STYLE OF THE CUT:
- Respect the grain of the footage: cut dialogue at sentence/turn seams, action on impacts, \
music on beats. Enter scenes as late as possible, leave as early as possible. Vary segment \
lengths; monotony reads as machine-made.
- TAKES: when the same content was delivered more than once (a retake, or a flub-then-retry \
inside one clip), keep EXACTLY ONE take in the timeline -- never let two takes of the same \
content both survive, and never splice across takes mid-sentence. Pick the best take on the \
TEXT first: prefer the words that are the most complete and correct delivery of the line -- no \
false starts, no cut-off or missing words, fewest filler words ("um", "uh", "like"), no \
repeated/stumbled phrasing, the full intended sentence present. Use score_span to get objective \
metrics + quality notes for each candidate span; only after the text is clean do you weight the \
remaining performance factors by the brief (polished -> fluency/stability; raw -> \
energy/authenticity). Name the winning take and the textual reason in the rationale.

CLIP CATALOG (scope for this thread):
{catalog}

{content_map}

{content_overlap}

{synced_angles}

{principles}
"""


def _render_system_prompt(catalog, content_map=None, overlap=None,
                          synced=None, document=None) -> str:
    synced_block = render_synced_angles_text(synced or [])
    return SYSTEM_PROMPT_TEMPLATE.format(
        catalog=render_catalog_text(catalog),
        content_map=render_content_map_text(content_map or []),
        content_overlap=render_overlap_text(overlap or []),
        synced_angles=synced_block or "SYNCED ANGLES: none detected (single-camera / "
                                      "no simultaneously-recorded clips in scope).",
        principles=render_principles_text(document or {}),
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
    # tool session so the catalog is computed a single time. The content map
    # surveys every clip's actual material so the model can't build from a
    # subset and silently drop unique footage.
    catalog = build_catalog(thread["file_ids"])
    content = build_content_map(thread["file_ids"])
    overlap = build_overlap_index(content)
    # Audio-energy pre-screen + airtight verification surfaces REAL second
    # angles (genre-general: works for action/music multicam that share no
    # dialogue), so the model follows the focus instead of riding one camera.
    synced = sync_mod.discover_synced_angles(thread["file_ids"])

    # Resume the working document from the latest snapshot (empty on first run).
    # Loaded before the prompt so a resumed run reflects any principles set
    # earlier (the rest of the prefix stays stable for caching within a run).
    doc, _version = store.latest_document(thread_id)

    system = _render_system_prompt(catalog, content, overlap, synced, doc or {})
    messages = store.load_messages(thread_id)
    if not messages:
        logger.error("L3: thread %s has no messages; nothing to do.", thread_id)
        store.set_thread_status(thread_id, "failed")
        return

    session = EditSession(
        thread_id=thread_id,
        file_ids=thread["file_ids"],
        catalog=catalog,
        content=content,
        overlap=overlap,
        synced=synced,
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
