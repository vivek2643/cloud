# Local-Dev Queue Isolation — share the substrate, isolate the execution lane

Status: IMPLEMENTED on branch `local-dev-isolation` (uncommitted, for review).
Production (`main`) is untouched: every change below is a **no-op when
`QUEUE_PREFIX` is empty**, which is the production default.

Goal: make local development iterate against the **real** system — real
projects, real media, real L1/GPU compute — while guaranteeing a local worker
running in-development branch code can **never** pick up a real user's
production job, and production workers never pick up a local test job.

---

## 0. The design decision (settled — this documents it, doesn't re-litigate)

The previous approach fully isolated local dev with a separate Postgres schema
(`DB_SCHEMA=dev`) and an R2 key prefix (`R2_KEY_PREFIX=dev`). That produced an
empty, disconnected sandbox — the wrong model. It is replaced.

**New model: share the substrate, isolate only the execution lane.**

- **SHARE** with production (local points at the same things):
  - Postgres DATA — prod's `public` schema. Local develops against real projects.
  - R2 media — no key prefix; bare, shared object keys.
  - L1 / RunPod GPU compute — local runs **no** `gpu` worker; prod's GPU workers
    do all L1 and write results back to the shared DB, visible to local.
  - Environment / secrets — one `.env`.
- **ISOLATE** (the only real boundary):
  - The Procrastinate **job queue** for the "dev-owned" stages, so a local
    worker (branch code) and a prod worker never share a queue.

### Why an execution boundary, not a data boundary

The dev-schema/R2-prefix model drew the boundary at the **data** layer: local
got its own empty tables and its own empty keyspace, so it could see none of
the real projects and none of the real media. That is safe but useless — you
cannot develop the cuts/render/export/grade pipelines against nothing.

The real risk during local development is **not** reading production data — it
is a **half-finished branch worker executing a real user's job** (or prod
executing your test job). That risk lives entirely in the **queue → worker**
hop. So we move the boundary there: local and prod read/write the same rows and
objects, but their workers pull from disjoint queues. L1 is the deliberate
exception — it is expensive GPU compute we explicitly want to keep shared and
identical, so its queue stays bare and only prod's RunPod fleet ever runs it.

Residual risk (honest): shared data means **local writes touch real rows**. A
destructive local experiment (re-ingest, delete, render overwrite) hits real
projects. Mitigation: **use throwaway test projects** for destructive
experiments; treat real projects as read-mostly locally. This is an accepted
trade-off of the "develop against reality" decision, not a bug.

---

## 1. Mechanism: `QUEUE_PREFIX` queue-name namespacing

A single new env var `QUEUE_PREFIX` (default empty). Local sets `QUEUE_PREFIX=dev`.

- **Dev-owned queues** (everything local iterates on) get the prefix:
  `ingest` → `dev-ingest`, `grade` → `dev-grade`, `render` → `dev-render`,
  `export` → `dev-export` (plus `cpu`/`l3` if ever used — see §2).
- **The `gpu` / L1 queue KEEPS ITS BARE NAME** — never prefixed. Local simply
  does not run a `gpu` worker, so every L1 job flows to prod's RunPod workers
  and lands in the shared DB, visible to local. This realizes "L1 same for prod
  and local, everything else separate."
- **Enqueue sites** derive their queue name through a prefix-aware helper
  (`Settings.effective_queue`), **except** the `gpu`/L1 enqueues, which stay
  literally bare.
- **Worker startup** derives the queues it listens on through the same helper
  (`Settings.worker_queues`): local (`QUEUE_PREFIX=dev`) listens on `dev-*` and
  **not** `gpu`; prod (no prefix) listens on the bare names including `gpu`.
- **Empty prefix ⇒ byte-for-byte identical to today.** `effective_queue` is the
  identity function and `worker_queues` passes its argument straight through
  (`None` still means "all queues"). This is the production guarantee.

### Helper design (`backend/app/config.py`)

```python
DEV_OWNED_QUEUES = ("ingest", "grade", "render", "export", "cpu", "l3")
SHARED_QUEUE = "gpu"   # L1 GPU compute: shared with prod, never prefixed

# normalized prefix: "dev" and "dev-" both -> "dev"
def _queue_prefix(self) -> str: return self.queue_prefix.strip().rstrip("-")

def effective_queue(self, base) -> str:
    # prod (no prefix) or the shared gpu queue -> identity
    if not self._queue_prefix or base == SHARED_QUEUE: return base
    return f"{self._queue_prefix}-{base}"       # e.g. "dev-ingest"

def worker_queues(self, requested):
    # prod: pass through unchanged (None == all queues, exactly as today)
    if not self._queue_prefix: return requested
    # local: map every base to its dev-* name AND drop the shared gpu queue,
    # so a dev worker can NEVER pull a real user's L1 (or any bare prod) job.
    # nothing requested under a prefix -> the dev-owned set (NOT "all queues",
    # which would include prod's bare queues).
    bases = requested or list(DEV_OWNED_QUEUES)
    return [self.effective_queue(b) for b in bases if b != SHARED_QUEUE] or None
```

Key safety property: under a prefix, `worker_queues` **never** returns a bare
queue name — a local worker is structurally incapable of listening on `gpu` or
on any unprefixed prod queue, even if `WORKER_QUEUES` is unset or lists `gpu`.

---

## 2. Enqueue sites + queue names (verified against the code)

Every `configure_task(..., queue=X).defer()` call site and every
`@app.task(..., queue=X)` definition:

| Task | Queue (base) | Enqueue site | Prefixed? |
|---|---|---|---|
| `build_export` | `export` | `routers/exports.py::_enqueue` | ✅ dev-export |
| `render_edit` | `render` | `routers/renders.py::_enqueue` | ✅ dev-render |
| `run_grade_job` | `grade` | `services/l3/grade/job.py::maybe_enqueue` | ✅ dev-grade |
| `l3_cuts_ingest` | `ingest` | `services/l3/ingest.py::defer_ingest` | ✅ dev-ingest |
| `l3_scene_enrich` | `ingest` | `services/l3/ingest.py::defer_scene_enrich` | ✅ dev-ingest |
| `vcut_ingest` | `ingest` | `services/vcut/orchestrate.py::defer_vcut_ingest` | ✅ dev-ingest |
| `vcut_enrich` | `ingest` | `services/vcut/orchestrate.py::defer_vcut_enrich` | ✅ dev-ingest |
| `l1_orchestrate` | `gpu` | `routers/upload.py::_enqueue_task` | ❌ BARE (L1) |
| `l1_editing_proxy` | `gpu` | `routers/upload.py::_enqueue_task` | ❌ BARE (L1) |
| `l1_active_speaker` | `gpu` | `services/l1/pipeline.py` (chained x2) | ❌ BARE (L1) |

Notes:
- Routing is decided at **defer** time by `configure_task(queue=...)`, which
  overrides the `@app.task(queue=...)` default. So only the enqueue call sites
  (and the worker's listen set) need to change; the `@app.task` decorator
  defaults are left bare as the logical-queue documentation and are never used
  for routing here (every defer sets `queue=` explicitly).
- **Discrepancy vs the recalled list**: the recalled dev-owned set was
  `ingest, l3, render, export, grade, cpu`. Following the code, the queues
  actually **enqueued to** are `ingest, grade, render, export` (+ bare `gpu`).
  `cpu` and `l3` are **listened to** by the worker fleet (`run_workers.sh`:
  `WORKER_QUEUES="cpu,l3,render,export"`) but **nothing enqueues to them** today
  — they are legacy/aspirational lanes. They are kept in `DEV_OWNED_QUEUES` so a
  local fleet started without `WORKER_QUEUES` mirrors prod's fleet (minus `gpu`)
  and so any future `cpu`/`l3` enqueue is auto-prefixed and stays isolated.

---

## 3. Worker-startup change (`backend/worker.py`)

Today:
```python
queues_env = os.getenv("WORKER_QUEUES", "").strip()
queues = [q.strip() for q in queues_env.split(",") if q.strip()] or None
```
After:
```python
requested = [q.strip() for q in queues_env.split(",") if q.strip()] or None
queues = get_settings().worker_queues(requested)   # prefix-aware; drops gpu locally
```
- Prod (`QUEUE_PREFIX` unset): `queues == requested`, identical to today; `None`
  still means "all queues" and a `gpu` dispatcher still listens on `gpu`.
- Local (`QUEUE_PREFIX=dev`): `queues` is the `dev-*` set with `gpu` removed. The
  existing Whisper-warmup gate (`"gpu" in queues`) then correctly skips warmup.

---

## 4. Revert of the schema / R2 isolation

- `.env`: remove `DB_SCHEMA=dev` and `R2_KEY_PREFIX=dev` (and their comment
  block). Net effect: local points back at shared prod `public` + bare R2 keys.
- `backend/app/config.py::pg_options`: revert the uncommitted `,extensions`
  addition back to its original `-c search_path={schema},public`. It only ever
  mattered for the (now unused) dev-schema path. With no `DB_SCHEMA` set,
  `pg_options` returns `""` (guarded by `if not schema: return ""`), so callers
  pass no libpq options and behave byte-for-byte like production.
- The `db_schema` / `r2_key_prefix` / `pg_connect_kwargs` plumbing is **left
  intact** in config/jobs/worker — it is harmless when unset (returns `""`/`{}`)
  and keeps the opt-in escape hatch available without touching prod behavior.

---

## 5. Exact local `.env` / launch-env

`.env` (shared, one file) — isolation block becomes:
```dotenv
# Local-dev QUEUE isolation. We SHARE prod data (public schema), R2 media, and
# L1/GPU compute; we ISOLATE only the Procrastinate job queue for dev-owned
# stages via queue-name namespacing. QUEUE_PREFIX=dev => local enqueues/worker
# use dev-ingest/dev-grade/dev-render/dev-export; the gpu/L1 queue stays bare
# (local runs no gpu worker, so all L1 flows to prod's RunPod workers).
# UNSET this in production (main) so behavior is byte-for-byte today's.
QUEUE_PREFIX=dev
# (DB_SCHEMA / R2_KEY_PREFIX intentionally UNSET: local shares prod public + R2)
```

Local launch:
```bash
# API (already how it runs locally; MIGRATION_GUARD=off since the shared public
# ledger is authoritative for prod, and QUEUE_PREFIX comes from .env):
cd backend && MIGRATION_GUARD=off .venv/bin/uvicorn app.main:app --port 8000 --reload

# Local worker (dev-* lanes only, NEVER gpu). WORKER_QUEUES optional — with a
# prefix set, an empty WORKER_QUEUES defaults to the dev-owned set:
cd backend && GPU_EXECUTION=local .venv/bin/python worker.py
#   -> listens on: dev-ingest, dev-grade, dev-render, dev-export, dev-cpu, dev-l3
```
Prod is unaffected: it runs with no `QUEUE_PREFIX`, so `run_workers.sh`
continues to listen on bare `gpu`, `ingest`, `cpu,l3,render,export`.

---

## 6. Residual risks / follow-ups

1. **Shared data — local writes touch real rows.** Mitigation: throwaway test
   projects for destructive experiments (re-ingest / delete / render). Accepted
   trade-off of "develop against reality."
2. **L1 → autostart-ingest carries prod's (empty) prefix.** When a file's L1
   finishes, `_maybe_autostart_cuts` (inside the L1 task) calls `defer_ingest`.
   In the shared model L1 runs on **prod's** GPU/RunPod worker, whose process
   has `QUEUE_PREFIX` empty — so that follow-up ingest is enqueued on the
   **bare** `ingest` queue and is picked up by **prod's** ingest worker (prod
   code), not the local `dev-ingest` worker. So a freshly-uploaded local file's
   auto-ingest runs on prod. To exercise the **dev** ingest/vcut code, kick
   ingest **manually via the local API** (`POST /projects/{id}/ingest`) — that
   runs in the local API process (`QUEUE_PREFIX=dev`) → `dev-ingest` → local
   worker. This is inherent to a process-global prefix over shared data (there
   is no per-file "owner environment" marker, by design), not a fixable bug.
3. **Bare per-call connectors are fine.** Every enqueue opens its own short-lived
   `PsycopgConnector` on `DATABASE_URL`; none of them pin a schema, so they all
   hit shared `public`. No connector change needed for this plan.
4. **`@app.task` decorator queues stay bare.** Harmless (overridden at defer),
   but if a future defer omits `configure_task(queue=...)` it would fall back to
   the bare decorator queue and escape isolation. Convention to preserve: always
   set `queue=self.effective_queue(...)` at the defer site for dev-owned tasks.

---

## 7. Verification checklist

- [x] `pyflakes` clean for all touched files (config, worker, 5 enqueue sites).
- [x] Backend tests pass, fully mocked, zero real spend / zero real jobs — full
      suite `pass=78 fail=0` (incl. `scripts/test_projects_router.py`).
- [x] `effective_queue`/`worker_queues` with `QUEUE_PREFIX=""` produce the exact
      original bare names, and `worker_queues(None)` stays `None` (prod-identical)
      — unit-proven.
- [x] `effective_queue`/`worker_queues` with `QUEUE_PREFIX=dev` produce `dev-*`,
      keep `gpu` bare at the enqueue helper, and DROP `gpu` from the worker set
      (even when `WORKER_QUEUES` lists it) — unit-proven.
- [x] Local API boots cleanly with no `DB_SCHEMA`, `QUEUE_PREFIX=dev`
      (`/health` = 200); connects to shared `public` — table counts confirmed
      projects=22, files=168, cut_records=1705 (all 22 owned by the dev user),
      and `/api/folders` returns the 12 real root folders; startup guard did not
      crash (`MIGRATION_GUARD=off` warning only).
