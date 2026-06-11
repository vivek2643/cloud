-- =============================================
-- 014: L3 edit orchestrator -- threads, documents, turns.
--
-- One THREAD per editing conversation ("make me a 30s reel from these clips").
-- The agent's deliverable is a versioned edit DOCUMENT (append-only history:
-- every finalize/ask_user snapshot is a new version, so undo/branching is
-- free). TURNS hold the raw agent message log (reasoning, tool calls/results)
-- so a paused thread can resume with full context -- and double as the
-- correction log for future fine-tuning.
--
-- No RLS: these are only touched by the backend with the service key, like
-- clip_perception (013).
-- =============================================

create table if not exists public.edit_threads (
    id          uuid primary key default uuid_generate_v4(),
    -- No FK to auth.users: dev mode runs with a synthetic user id that has no
    -- auth.users row (same loose ownership the rest of the app relies on).
    user_id     uuid not null,
    title       text,
    -- The clips in scope for this edit (no FK: clips can be deleted later
    -- without tearing down the thread; the agent re-validates at run time).
    file_ids    uuid[] not null default '{}',
    brief       text,
    -- drafting | awaiting_user | ready | failed
    status      text not null default 'drafting',
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create index if not exists idx_edit_threads_user on public.edit_threads(user_id);

create trigger edit_threads_updated_at
    before update on public.edit_threads
    for each row execute function public.set_updated_at();

create table if not exists public.edit_documents (
    id          uuid primary key default uuid_generate_v4(),
    thread_id   uuid not null references public.edit_threads(id) on delete cascade,
    version     int  not null,
    -- The whole Edit Document: brief, outline, timeline (snapped segments with
    -- costs + rationale), open_questions, diagnostics.
    document    jsonb not null default '{}'::jsonb,
    -- agent | user (manual tweak)
    created_by  text not null default 'agent',
    created_at  timestamptz not null default now(),
    unique (thread_id, version)
);

create index if not exists idx_edit_documents_thread on public.edit_documents(thread_id);

create table if not exists public.edit_turns (
    id          bigint generated always as identity primary key,
    thread_id   uuid not null references public.edit_threads(id) on delete cascade,
    -- Position in the agent conversation (monotonic within a thread).
    seq         int  not null,
    -- user | assistant | tool  (mirrors the Anthropic message roles we replay
    -- on resume; tool results are stored as their own rows for inspectability)
    role        text not null,
    content     jsonb not null default '{}'::jsonb,
    -- Per-call token accounting for cost tracking ({input,output,cache_read...}).
    usage       jsonb,
    created_at  timestamptz not null default now(),
    unique (thread_id, seq)
);

create index if not exists idx_edit_turns_thread on public.edit_turns(thread_id);

comment on table public.edit_threads   is 'L3: one editing conversation (scope + brief + lifecycle).';
comment on table public.edit_documents is 'L3: versioned Edit Document snapshots (append-only).';
comment on table public.edit_turns     is 'L3: raw agent message log for resume + future training data.';
