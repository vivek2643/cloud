# Plan: A real migration runner + a fail-loud drift guard

## Goal
Close the gap that let `042`/`043` sit unapplied against production while the
code that depends on them shipped anyway: give every deploy an explicit,
idempotent step that actually applies pending migrations (**the applier**),
and give every process a read-only startup check that refuses to boot if the
live schema has drifted from what `migrations/` says it should be (**the
guard**). Two different jobs, two different safety properties — build both,
per the audit's own conclusion: the applier is what fixes it, the guard is
what catches it the next time something bypasses the applier.

Non-negotiable: **never auto-apply migrations from multiple concurrent
processes.** This deployment already forks N worker processes from one
entrypoint (`run_workers.sh`); a naive "apply on every process start" would
race all N of them against the same SQL. The applier must be a single gated
step, protected by a Postgres advisory lock as a second line of defense.

## Current state (grounded, so steps are concrete)
- **No tracking table exists.** Migrations are 44 numbered, hand-written
  `.sql` files in `backend/migrations/`, applied by a human pasting each one
  into the Supabase SQL Editor (`README.md`'s own documented process). There
  is no record anywhere of which files have actually been run.
- **This already caused a real bug.** A full audit (this session) found `042`
  and `043` unapplied against production — `042` is an ACTIVE mismatch:
  `pipeline.py::_stage5_audio`'s INSERT and `observe.py::_fetch_audio_features`'s
  SELECT both reference columns (`sections`, `drop_ms`) that didn't exist,
  meaning every ingest touching audio and every audio-aware edit turn was
  failing. `018`'s cleanup drops were also found only *partially* applied
  (2 of 7 target columns gone, 5 not) — direct evidence the apply process has
  been manual and non-atomic more than once, not a one-off slip.
- **The deploy pipeline never touches the database**, confirmed by reading
  the actual files, not assumed: `.github/workflows/build-worker.yml` builds
  and pushes a Docker image; `backend/Dockerfile.worker` only installs deps
  and copies code; `deploy/aws/docker-compose.yml` runs
  `bash run_workers.sh`. Nothing in that chain runs SQL.
- **The one concrete production entrypoint** is `backend/run_workers.sh`: it
  runs ONCE per container start, then forks `GPU_WORKERS` + 1 (ingest) +
  `CPU_WORKERS` separate `python worker.py` processes on the SAME box (see
  its own header comment). This is exactly the shape the "no naive
  auto-apply" warning is about — and exactly why a single step at the TOP of
  this script, before any forking, is the natural gated-step location.
- **`worker.py`** (`backend/worker.py::main()`) is also run directly in local
  dev (per its own docstring), bypassing `run_workers.sh` entirely.
- **The FastAPI app** (`backend/app/main.py`) has no startup hooks today (no
  `@app.on_event`/lifespan) and is not part of the Docker/compose setup above
  — wherever/however it's actually deployed, it's out of scope for the
  applier (never auto-apply from an API process) but IS in scope for the
  guard (it should refuse to serve traffic against a drifted schema too).
- **The current 44 files are already fully reconciled against live schema**
  as of this plan (see the completed audit) — `045`'s bootstrap step (below)
  can mark all of them "applied" without re-running a single one, which
  matters a lot given at least one of them (`012`'s drop of `public.projects`)
  would now be actively destructive if blindly re-executed.

---

## Step 1 — `schema_migrations` tracking table + one-time backfill
**Files:** new `backend/migrations/045_schema_migrations.sql`, a one-off
backfill (run once by hand, same way `042`/`043`/`044` were applied this
session — see Finish), new `.gitattributes` entry.

1. ```sql
   create table if not exists public.schema_migrations (
       filename    text primary key,        -- e.g. '046_something.sql'
       checksum    text not null,           -- sha256 of the file's bytes at apply time
       applied_at  timestamptz not null default now()
   );
   ```
   `checksum` exists specifically because of what the audit found: a file
   edited (or partially applied) after the fact is a real, observed failure
   mode here, not a theoretical one. A mismatch is what the guard flags as
   drift later (Step 4) — it should never be silently ignored.
2. **Backfill, once, by hand: 45 rows, `001_initial.sql` .. `045_schema_migrations.sql`
   inclusive.** `045` backfills *itself* — by the time you're inserting rows
   you've already run `045`'s `create table` (that's what you're inserting
   into), so it belongs in its own baseline the same as any other already-live
   file. This is what makes the acceptance criterion below literally true
   instead of off-by-one. This is a deliberate, manual "I have verified these
   are already live" act (backed by this session's audit), NOT something the
   runner should ever infer on its own — the runner must never assume an
   already-existing table means "everything's fine," only an explicit
   backfill should establish that baseline.
3. **Checksum normalization** (`.gitattributes`): add `*.sql text eol=lf` so
   git never lets a CRLF checkout silently flip a file's bytes. Belt-and-
   suspenders on top of that, `checksum()` itself (Step 2) normalizes before
   hashing — strip trailing whitespace per line and ensure a single trailing
   newline — so an innocuous re-save of an already-applied file (a linter
   touching whitespace, e.g.) doesn't flip its hash and trip the guard on
   every process boot fleet-wide. Without this, checksum drift becomes a
   deploy-blocking false positive on nothing but formatting — exactly the
   kind of friction that gets a fail-loud guard quietly disabled.

**Acceptance:** `schema_migrations` has exactly 45 rows (`001`..`045`, `045`
included) immediately after backfill; running the Step 2 applier right after
is a genuine no-op (nothing pending, nothing drifted) — not "44 rows, then
the applier runs once more," which would make the very first `apply` call
post-backfill non-trivial and contradict "clean no-op."

## Step 2 — the runner (`backend/scripts/migrate.py`)
**Files:** new `backend/scripts/migrate.py`, new
`backend/app/services/db_migrations.py` (the shared, importable logic both
the CLI script and the Step 4 guard call — keeps the guard from having to
shell out to a script).

1. `db_migrations.py`:
   - `list_migration_files() -> list[Path]`: `backend/migrations/*.sql`,
     sorted by filename (the existing `NNN_name.sql` convention already sorts
     correctly).
   - `checksum(path) -> str`: sha256 hex of the file's *normalized* text —
     trailing whitespace stripped per line, single trailing newline enforced
     — not the raw bytes (see Step 1.3: this is what keeps innocuous
     formatting changes on old files from flipping the hash).
   - `fetch_applied(conn) -> dict[str, str]`: filename -> checksum from
     `schema_migrations`. Returns `{}` if the table itself doesn't exist yet
     (never an error — that's the pre-bootstrap state).
   - `pending(conn) -> tuple[list[Path], list[str]]`: `(files not yet
     applied, filenames whose live checksum no longer matches the tracked
     one)`. The second list is the drift signal for Step 4.
   - `apply_pending(conn) -> list[str]`: for each pending file IN ORDER —
     unconditionally `create table if not exists public.schema_migrations`
     first (the bootstrap: safe/idempotent even once `045` itself is a
     tracked row) — then, per file, apply it and record it. A file that
     fails leaves everything after it un-applied and exits non-zero with the
     failing filename named explicitly — never skip ahead.

   **Transactional vs. non-transactional files.** By default, a whole file's
   text is sent through **one** `cur.execute(sql_text)` call on a connection
   with `autocommit=False`. This is deliberate, not incidental: Postgres's
   simple-query protocol wraps *all* statements in a single multi-statement
   message into one implicit transaction — this is a server-side behavior,
   independent of the client's autocommit setting — which is exactly why
   sending the whole file as one `execute()` call inside an explicit
   `BEGIN`/`COMMIT` gives true all-or-nothing apply (confirmed empirically:
   this is the same mechanism that made `042`/`043`/`044` apply correctly
   this session as single multi-statement transactions).

   The problem: a handful of legitimate Postgres statements — `CREATE INDEX
   CONCURRENTLY`, `ALTER TYPE ... ADD VALUE` pre-12, `VACUUM` — are
   physically incapable of running inside a transaction block, transactional
   or not, and will hard-fail with "cannot run inside a transaction block."
   `CREATE INDEX CONCURRENTLY` in particular is a realistic near-term need
   (an index on a large, hot table like `cut_records` or `audio_features`
   without locking it). Deciding this now rather than deferring:

   - A file whose **first line** is the literal comment
     `-- migrate:no-transaction` is run differently: the connection is
     switched to `autocommit=True` and the file's text is still sent as a
     single `execute()` call, but the migration and its `schema_migrations`
     insert are now two separate statements/commits, not one transaction —
     there is no atomicity between "the DDL ran" and "it's recorded as
     applied." **Convention (documented in `backend/migrations/README` or
     equivalent, not mechanically enforced by a SQL parser — see below):
     a no-transaction file should contain exactly one statement.** This is
     mostly self-enforcing: if a no-transaction file bundles more than one
     genuinely non-transactional statement, Postgres's own multi-statement
     grouping in the simple-query protocol still applies to a single
     `execute()` call regardless of autocommit, so it fails with the same
     clear "cannot run inside a transaction block" error, immediately telling
     the author to split it into separate files. Because apply+record isn't
     atomic here, a no-transaction migration should also be written
     idempotently where Postgres supports it (`create index concurrently if
     not exists ...`) so a retry after a crash between "DDL ran" and "row
     inserted" is safe rather than double-applying.
   - Everything else (the common case, and all 45 existing files) stays
     fully transactional as described above — no opt-in needed, no parser
     required to split statements.

2. `migrate.py` (CLI):
   - `python backend/scripts/migrate.py apply` — takes a Postgres advisory
     lock (`pg_advisory_lock(<fixed constant>)`; pick one that doesn't
     collide with Procrastinate's own advisory-lock usage — check its source
     at implementation time), calls `apply_pending`, releases the lock in a
     `finally`. Prints each newly-applied filename. A concurrent second
     invocation blocks on the lock, then finds nothing pending and exits
     cleanly — this is what makes it safe to call from every one of
     `run_workers.sh`'s eventual N forks if that ever changes, not just the
     single call site Step 3 actually wires up.
   - `python backend/scripts/migrate.py check` — read-only, prints pending +
     drifted filenames, exits non-zero if either is non-empty. (What Step 4
     calls internally, exposed as its own command for CI/manual use.)
   - `python backend/scripts/migrate.py reconcile <filename>` — updates the
     tracked checksum for one filename to match the file's current on-disk
     contents WITHOUT re-running it. The deliberate, explicit escape hatch
     for "this old file's text changed for a legitimate reason (a comment
     fix, e.g.) and I've confirmed the DB itself needs no change" — never
     used silently, never used automatically.

**Acceptance:** on a DB with `schema_migrations` already backfilled (Step 1),
`apply` is a no-op. On a DB missing a later file, `apply` runs exactly that
file and records it. Two concurrent `apply` invocations never both try to run
the same file (verify via a test that holds the lock and asserts the second
call blocks, then proceeds cleanly once released).

## Step 3 — wire the applier into the one real deploy entrypoint
**Files:** `backend/run_workers.sh`.

1. At the very top of `run_workers.sh`, before GPU detection / before any
   `worker.py` fork: `python migrate.py apply || exit 1`. This makes the
   applier a genuinely single, gated step for the deployment shape that
   actually exists today (one container, `run_workers.sh` runs once, forks
   happen after) — the advisory lock from Step 2 is then defense-in-depth
   for if/when this ever scales to multiple concurrent container replicas,
   not the thing doing the work in the common case.
2. Do **not** call the applier from `worker.py` or `app/main.py` directly —
   only the guard (Step 4) goes there. Keeps "who's allowed to write schema"
   to exactly one call site.
3. **Open question to confirm at implementation time, not assumed here:**
   whether `app/main.py` (the FastAPI app) deploys/boots independently of the
   worker container — this repo has no visible Dockerfile/render.yaml/
   fly.toml/Procfile for it, so its deploy path is currently unknown. If it
   does deploy independently, a migration-bearing release makes the API
   crashloop (via Step 4's guard) until *some* worker container happens to
   run `run_workers.sh` and apply the pending files. That's arguably the
   right fail-safe behavior (refuse to serve rather than serve a stale
   schema) but it should be a conscious decision, not a surprise — if
   confirmed independent, either give the API's own deploy path a
   `migrate.py apply` pre-start step too, or introduce a single shared
   pre-deploy job both entrypoints depend on, before Step 3 ships.

**Acceptance:** starting the fleet (`docker compose up` / running
`run_workers.sh` locally) against a DB with pending migrations applies them
once, before any worker forks; the log shows exactly one `migrate.py apply`
line, not N.

## Step 4 — the guard: fail loud on startup, everywhere a process boots
**Files:** `backend/worker.py`, `backend/app/main.py`.

1. `db_migrations.py` gains `assert_up_to_date(conn)`: calls `pending()`; if
   either list is non-empty, raises `SchemaDriftError` with a message naming
   every pending/drifted filename explicitly (this is the whole point — a
   vague "schema out of date" is not good enough, the fix has to be obvious
   from the error alone).
2. `worker.py::main()`: call `assert_up_to_date` first thing, before
   `register_tasks()`. An uncaught `SchemaDriftError` here should crash the
   process with a clear log line — exactly the "won't start until fixed"
   property, and it catches the local-dev path (`python worker.py` direct)
   that Step 3's `run_workers.sh` wiring never touches.
3. `app/main.py`: add a `lifespan` (or `@app.on_event("startup")`) handler
   that calls the same `assert_up_to_date`. A drifted schema should fail
   FastAPI's startup, not serve a single request against it.
4. Read-only, cheap (one query), safe to run from every process
   unconditionally — this is the guard, not the applier; it never writes.
5. **Bypass for local dev:** if `os.environ.get("MIGRATION_GUARD") == "off"`,
   `assert_up_to_date` logs a loud, impossible-to-miss warning
   (`"MIGRATION_GUARD=off — schema drift checks DISABLED, do not use in
   production"`, printed even above normal log level) and returns
   immediately instead of checking. Purpose: a dev with a deliberately
   divergent local DB (mid-migration-authoring, e.g.) needs a *sanctioned*
   way to boot anyway. Without one, the realistic outcome is someone quietly
   comments out the `assert_up_to_date` call itself the first time it's
   inconvenient — and then it's gone, silently, for everyone who touches that
   code next. A named, loud, env-gated flag is worse-case-bounded (it only
   ever affects the process that set it) and self-documenting in a way a
   commented-out call is not.

**Acceptance:** manually add an unrecorded `.sql` file to `migrations/` and
confirm both `worker.py` and the FastAPI app refuse to start, each naming the
file in the error; removing it (or backfilling it) lets both start clean.
Setting `MIGRATION_GUARD=off` lets both start anyway, with the warning
visible in the log.

### What the guard does — and does not — catch
State this plainly so it's never mistaken for more than it is: the guard is
a **filename + checksum ledger**. A green guard proves *"every file in
`migrations/` has either never run, or ran once through this system and
hasn't been edited since."* It does **not** prove *"the live database
currently reflects what these files say."* Those are different claims, and
the gap between them is exactly what caused the `018` incident that
motivated this plan — `018`'s file was untouched and would checksum-match
today even though only 2 of its 7 column drops actually landed, because it
was applied by hand, outside any transaction, before this system existed.

Two futures, not one, matter here:
- **Going forward**, this gap closes for anything applied *through*
  `migrate.py`: Step 2's per-file transaction means a (transactional)
  migration is now genuinely all-or-nothing, so "recorded as applied" really
  does imply "DB reflects it," for every file from `045` onward.
- **The guard still cannot detect** (a) a migration applied through this
  system that partially failed mid-flight for a *no-transaction* file (an
  inherent trade-off of those, called out in Step 2 — mitigate by keeping
  them idempotent and single-statement), or (b) anyone running DDL directly
  against the database outside `migrate.py` entirely (a hand-pasted
  `ALTER TABLE` in the Supabase SQL Editor, the exact workflow this plan
  replaces). Case (b) is a process risk, not a code gap, and this plan
  doesn't attempt to close it at the database-permissions level (e.g.
  revoking direct DDL grants from the app's connection role) — that's a
  heavier, separate decision, worth naming explicitly as future scope rather
  than silently out of scope.

---

## Tests
- `db_migrations.py` unit tests (`backend/scripts/test_db_migrations.py`,
  no real DB — mock the connection like the rest of this suite does for
  `grade/job.py`): `pending()` correctly separates "never applied" from
  "checksum drifted"; `apply_pending` stops at the first failure and never
  runs files after it; `assert_up_to_date` raises with every drifted/pending
  filename named, not just the first.
- `checksum()` normalization: a file with trailing whitespace/CRLF added
  produces the *same* checksum as its normalized form, but a genuine content
  change still produces a different one.
- A `-- migrate:no-transaction` file is detected and routed to the
  autocommit path (assert the mocked connection's `autocommit` is toggled
  and `apply`/`insert` happen as separate calls, not one wrapped
  transaction); a normal file is not.
- `assert_up_to_date` with `MIGRATION_GUARD=off` set returns without
  querying the connection at all, and logs the warning.
- A concurrency test: two threads/processes both call `apply_pending`
  against the same (real or fully-mocked-lock) target; assert the migration
  set only ever gets applied once (no duplicate-key errors from a race).
- An end-to-end smoke test (manual, per Step 4's acceptance): the "add an
  unrecorded file, confirm both entrypoints refuse to start" check.

## Order & risk
Build order: **1 (table + backfill) → 2 (runner) → 3 (wire the applier) → 4
(wire the guard)**. Land 1+2 and run the backfill FIRST, in isolation, and
confirm `migrate.py check` reports clean against production before touching
either entrypoint — Steps 3/4 are inert until that's true, and getting the
backfill wrong (e.g. missing a file, or backfilling before double-checking
`044` landed) would make the guard fail-loud immediately and legitimately
block every future deploy. Step 4 is the one with the most blast radius
(every process boot, everywhere) — land it last, after 1-3 have been running
clean for at least one real deploy cycle.

Every existing dev's local `.env`-pointed DB also needs the Step 1 backfill
run once after this lands, or their local `worker.py`/API process will
correctly (if surprisingly) refuse to start — call this out explicitly when
landing, not just in this plan.

## Finish
Run the new test suite, apply `045` + the one-time backfill against
production the same careful, verified way `042`/`043`/`044` were applied
this session (explicit transaction, verify before/after, never blind), then
**commit and push** — but hold on the `run_workers.sh`/`worker.py`/
`app/main.py` wiring (Steps 3-4) until the backfill's been confirmed clean
against production for real, per Order & risk above.
