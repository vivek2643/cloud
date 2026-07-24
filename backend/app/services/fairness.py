"""
Per-user fair scheduling (scale_architecture.plan.md Pillar 6).

Procrastinate has no built-in per-user concept -- only a global `priority`
(higher runs first, ties broken by insertion order), `lock`, and
`queueing_lock`. Fairness is built entirely in application code on top of
those: a busy user's new job gets a LOWER priority, so a light user's job on
a contended queue doesn't wait behind someone else's Nth job. L3 ingest (a
real, costed API call the user directly triggers) additionally gets a hard
per-user in-flight cap -- unlike L1 (upload-triggered, not a discrete user
"go" action each time), rejecting outright when a user is already at their
cap is normal, expected API behavior, not a UX regression.

Queues stay resource-typed (gpu/ingest/render/export); a genuine `cpu` queue
split is pipeline_parallelism.plan.md territory (L1 CPU/GPU stage split) --
adding the queue name here with nothing routed to it yet would be a no-op,
not a fairness improvement.
"""
from __future__ import annotations

from app.config import get_settings

# Clamp so one pathological user can't drive priority to -infinity and
# starve normal default-priority (0) jobs indefinitely once they stop being
# the busiest user.
_MAX_PRIORITY_PENALTY = 20


class CapacityExceeded(Exception):
    """A user already has too many in-flight jobs of a kind that enforces
    a hard cap (currently: L3 ingest runs)."""

    def __init__(self, user_id: str, in_flight: int, max_inflight: int):
        self.user_id = user_id
        self.in_flight = in_flight
        self.max_inflight = max_inflight
        super().__init__(
            f"user {user_id} already has {in_flight} ingest run(s) in flight (max {max_inflight})"
        )


def _pg():
    from app.services import db
    return db.connection()


def priority_for(in_flight: int) -> int:
    """Procrastinate priority for a user's new job given how many they
    already have in flight -- lower (more negative) the busier they are."""
    return -min(max(in_flight, 0), _MAX_PRIORITY_PENALTY)


def count_inflight_ingest_runs(user_id: str) -> int:
    """L3 ingest runs for this user's projects that haven't reached a
    terminal status (ingest_runs.status: pending/pass1/images/pass2/post are
    all non-terminal; ready/failed are)."""
    with _pg() as conn:
        row = conn.execute(
            """
            select count(*) from ingest_runs ir
            join projects p on p.id = ir.project_id
            where p.user_id = %s and ir.status not in ('ready', 'failed')
            """,
            (user_id,),
        ).fetchone()
    return int(row[0]) if row else 0


def count_inflight_l1(user_id: str) -> int:
    """Files for this user whose L1 pass hasn't reached a terminal
    l1_status."""
    with _pg() as conn:
        row = conn.execute(
            "select count(*) from files where user_id = %s and l1_status in ('pending', 'running')",
            (user_id,),
        ).fetchone()
    return int(row[0]) if row else 0


def check_ingest_capacity(user_id: str) -> int:
    """Returns the Procrastinate priority to defer a new ingest run with.
    Raises CapacityExceeded if this user is already at the configured cap."""
    in_flight = count_inflight_ingest_runs(user_id)
    max_inflight = get_settings().max_inflight_ingest_runs_per_user
    if in_flight >= max_inflight:
        raise CapacityExceeded(user_id, in_flight, max_inflight)
    return priority_for(in_flight)


def priority_for_l1(user_id: str) -> int:
    """No hard cap for L1 -- upload already succeeded and the file needs
    SOME orchestrator run eventually; this only affects ordering under
    contention."""
    return priority_for(count_inflight_l1(user_id))
