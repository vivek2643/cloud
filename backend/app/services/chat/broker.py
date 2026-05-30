"""
In-memory broker for live chat-turn progress events + cooperative cancel.

The runner executes in a background thread (blocking work: psycopg, the
Anthropic HTTP call). It pushes events into a thread-safe queue here. The SSE
endpoint (async, in the event loop) drains that queue and writes Server-Sent
Events to the client.

This is process-local. It is the LOW-LATENCY path: a client connected to the
same API process that's running the turn gets real-time updates. Durability
and reconnect-after-the-fact come from the chat_turns table, not from here.

Cancellation is dual-channel:
  * In-process: a threading.Event the runner polls (instant).
  * Durable: the chat_turns.cancel_requested flag (survives restart, works
    even if cancel hits a different worker than the one running the turn).
The runner checks both.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# How long to retain a finished turn's state so a slightly-late SSE connection
# can still replay the terminal event before we evict it.
_RETAIN_AFTER_TERMINAL_S = 120


@dataclass
class _Event:
    type: str  # phase | warning | done | error | cancelled
    payload: Dict[str, Any]


@dataclass
class TurnState:
    turn_id: str
    events: "queue.Queue[_Event]" = field(default_factory=queue.Queue)
    cancel: threading.Event = field(default_factory=threading.Event)
    terminal: bool = False
    terminal_at: Optional[float] = None
    # Latest coarse snapshot so a freshly-connected SSE stream can show
    # current state immediately instead of waiting for the next event.
    snapshot: Dict[str, Any] = field(
        default_factory=lambda: {"phase": "queued", "pct": 0, "label": "Queued"}
    )


class TurnBroker:
    def __init__(self) -> None:
        self._turns: Dict[str, TurnState] = {}
        self._lock = threading.Lock()

    def create(self, turn_id: str) -> TurnState:
        with self._lock:
            self._evict_stale_locked()
            st = TurnState(turn_id=turn_id)
            self._turns[turn_id] = st
            return st

    def get(self, turn_id: str) -> Optional[TurnState]:
        with self._lock:
            return self._turns.get(turn_id)

    def emit(self, turn_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        st = self.get(turn_id)
        if st is None:
            return
        if event_type == "phase":
            st.snapshot = {
                "phase": payload.get("phase"),
                "pct": payload.get("pct"),
                "label": payload.get("label"),
            }
        st.events.put(_Event(type=event_type, payload=payload))
        if event_type in ("done", "error", "cancelled"):
            st.terminal = True
            st.terminal_at = time.time()

    def request_cancel(self, turn_id: str) -> bool:
        st = self.get(turn_id)
        if st is None:
            return False
        st.cancel.set()
        return True

    def is_cancelled(self, turn_id: str) -> bool:
        st = self.get(turn_id)
        return bool(st and st.cancel.is_set())

    def _evict_stale_locked(self) -> None:
        now = time.time()
        stale = [
            tid
            for tid, st in self._turns.items()
            if st.terminal and st.terminal_at and (now - st.terminal_at) > _RETAIN_AFTER_TERMINAL_S
        ]
        for tid in stale:
            self._turns.pop(tid, None)


# Single process-local broker.
broker = TurnBroker()
