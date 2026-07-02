-- =============================================
-- 021: per-turn TOOL TRACE on edit_turns.
--
-- The agentic loop (l3.tools.run_edit_loop) already returns a full trace of the
-- turn's tool calls -- {turn, name, args, applied, result} per call. Persisting
-- it on the assistant turn makes "check the reasoning" a read instead of a
-- re-run: the exact observe/act sequence, its args, and each result are
-- inspectable after the fact (and double as correction/training data).
--
-- Nullable + defaulted so existing rows and pure-chat turns (no tools) are fine.
-- =============================================

alter table public.edit_turns
    add column if not exists trace jsonb;

comment on column public.edit_turns.trace is
    'L3: ordered tool-call trace for an assistant turn ({turn,name,args,applied,result}).';
