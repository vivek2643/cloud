-- Adds per-channel RGB standard deviation to color_stats -- needed for a
-- proper Reinhard-style reference-image color transfer (color_grading.plan.md
-- SS7 mode 2), which matches both mean AND spread, not just mean (a mean-only
-- shift is a flat color-cast nudge, not real style transfer).

alter table public.color_stats add column if not exists rgb_std jsonb not null default '[]'::jsonb;

comment on column public.color_stats.rgb_std is
    'Per-channel [r,g,b] standard deviation (0..1) across every sampled pixel -- the spread half of a mean+std (Reinhard-style) color transfer.';
