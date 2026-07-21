-- color_skin_vibrance.plan.md Phase 0: adds mean per-pixel Lab chroma to
-- color_stats -- the "how colorful is this footage" signal the vibrance
-- normalization feature boosts toward a target. `color_stats` is a real
-- relational table with named columns (unlike cut_color_stats, which is a
-- jsonb blob keyed by schema_version), so a Python SCHEMA_VERSION bump alone
-- does not make this field queryable -- it needs an actual column.

alter table public.color_stats add column if not exists chroma_mean real not null default 0;

comment on column public.color_stats.chroma_mean is
    'Mean per-pixel Lab chroma sqrt(a*^2+b*^2) across every sampled pixel -- how colorful the footage is; vibrance normalization boosts sat toward a target when this is low.';
