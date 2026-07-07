-- =============================================
-- Cuts v3, editorial pass (section D): conservative removal.
--
-- Adds junk_confidence to cut_records so the frontend can HIDE clearly
-- unusable footage (camera cues, pre-roll, dead air) by default while keeping
-- doubtful cuts visible inline -- "if in doubt, show". Additive; existing rows
-- default to 'low' (visible), matching the pre-existing show-everything
-- behavior. See cuts_v3_editorial.plan.md section D.
-- =============================================

alter table public.cut_records
    add column if not exists junk_confidence text not null default 'low'
        check (junk_confidence in ('high', 'low'));

comment on column public.cut_records.junk_confidence is
    'Cuts v3 editorial pass: "high" (clearly unusable -> hidden by default) vs "low"/doubtful (kept visible). Only meaningful when junk = true.';
