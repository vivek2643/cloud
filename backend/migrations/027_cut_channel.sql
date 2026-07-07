-- =============================================
-- Cuts v3: delivery channel (said / done / shown) on cut_records.
--
-- The channel is a SEMANTIC category the LLM owns (deterministic-keep rule:
-- code owns numbers, the model owns categories). Speech cuts are always
-- "said"; a video cut is "done" (an action is performed/demonstrated on
-- screen) or "shown" (b-roll / object / scenery / display, no performed
-- action). Drives the All/Said/Done/Shown category filter in the cuts view.
--
-- Additive; existing rows default to 'shown' (the conservative video default).
-- Re-ingest repopulates it with the model's real call.
-- =============================================

alter table public.cut_records
    add column if not exists channel text not null default 'shown'
        check (channel in ('said', 'done', 'shown'));

comment on column public.cut_records.channel is
    'Cuts v3 delivery channel: "said" (spoken), "done" (action performed/demonstrated), "shown" (b-roll/display). LLM-owned category; speech is always "said".';
