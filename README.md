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

Run the migration in `backend/migrations/001_initial.sql` against your Supabase project via the SQL Editor.

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
