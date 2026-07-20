# Edso

AI-powered video drive and collaboration hub for the creator economy.

## Architecture

- **Frontend:** Next.js (App Router) + Tailwind CSS + Zustand
- **Backend:** Python + FastAPI + FFmpeg (background tasks)
- **Storage:** Cloudflare R2 (S3-compatible, zero egress fees)
- **Database & Auth:** Supabase (PostgreSQL + Auth)

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.9+
- FFmpeg installed locally (`brew install ffmpeg` on macOS)
- [Supabase](https://supabase.com/) project
- [Cloudflare R2](https://developers.cloudflare.com/r2/) bucket

### Setup

```bash
cp .env.example .env
# Fill in your Supabase and R2 credentials in .env
```

### Database

Migrations live in `backend/migrations/*.sql` and are applied via the runner, not by hand:

```bash
cd backend
.venv/bin/python scripts/migrate.py apply   # applies every pending migration, in order
.venv/bin/python scripts/migrate.py check   # read-only: reports pending/drifted files, exits non-zero if any
```

`apply` takes a Postgres advisory lock, so it's safe to run from multiple
processes/deploys concurrently -- only one actually applies, the rest find
nothing pending. Every migration file runs in its own transaction (fully
atomic) unless its first line is the literal comment `-- migrate:no-transaction`,
which a handful of statements (`CREATE INDEX CONCURRENTLY`, `ALTER TYPE ...
ADD VALUE` pre-Postgres-12, `VACUUM`) require -- those run in autocommit
mode instead and should contain exactly one statement each. See
`migration_runner.plan.md` for the full design, including what the
startup drift guard (`app/services/db_migrations.py::assert_up_to_date`,
bypassable locally via `MIGRATION_GUARD=off`) does and does not catch.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## How It Works

1. User uploads a video via drag-and-drop (goes directly to Cloudflare R2)
2. Backend creates a file record and triggers a background task
3. FFmpeg generates a 1080p proxy, extracts a thumbnail, and probes metadata
4. Proxy and thumbnail are uploaded to R2, database is updated to "ready"
5. User can browse, play, and manage files in the drive UI
