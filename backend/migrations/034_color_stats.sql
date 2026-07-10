-- L1 color_stats: deterministic per-file color measurement -- the foundation
-- the color-grading system (color_grading.plan.md) measures every layer off.
-- One row per file, computed off the same analysis proxy motion_dynamics /
-- scene_cuts already decode (see backend/app/services/l1/color_stats.py).

create table if not exists public.color_stats (
    file_id             uuid primary key references public.files(id) on delete cascade,
    schema_version      int     not null default 1,
    frames_sampled      int     not null default 0,
    luma_hist           jsonb   not null default '[]'::jsonb,
    black_point         real    not null default 0,
    white_point         real    not null default 1,
    mid_gray            real    not null default 0.5,
    rgb_mean            jsonb   not null default '[]'::jsonb,
    rgb_median          jsonb   not null default '[]'::jsonb,
    lab_ab_cast         jsonb   not null default '[]'::jsonb,
    wb_gray_world       jsonb   not null default '[]'::jsonb,
    wb_white_patch      jsonb   not null default '[]'::jsonb,
    clip_shadow_pct     real    not null default 0,
    clip_highlight_pct  real    not null default 0,
    is_log_flat         boolean not null default false,
    skin_lab            jsonb,
    palette             jsonb   not null default '[]'::jsonb,
    created_at          timestamptz not null default now()
);

comment on table public.color_stats is
    'L1 deterministic color measurement from N evenly-spaced sampled frames over the proxy: exposure/contrast/cast/WB/clipping/log-flat/skin/palette. Foundation for the correct/match/look/arc grading layers -- see color_grading.plan.md SS2.2.';
comment on column public.color_stats.luma_hist is
    '32-bin, L1-normalized luma histogram (0..1 range) over every sampled pixel.';
comment on column public.color_stats.black_point is
    'p0.5 luma percentile (0..1) across sampled frames -- the effective shadow floor.';
comment on column public.color_stats.white_point is
    'p99.5 luma percentile (0..1) across sampled frames -- the effective highlight ceiling.';
comment on column public.color_stats.mid_gray is
    'Median luma (0..1) across sampled frames.';
comment on column public.color_stats.rgb_mean is
    'Mean per-channel [r,g,b] (0..1) across every sampled pixel.';
comment on column public.color_stats.rgb_median is
    'Median per-channel [r,g,b] (0..1) across every sampled pixel.';
comment on column public.color_stats.lab_ab_cast is
    'Mean CIE Lab [a*, b*] (color cast) across every sampled pixel; near [0,0] = neutral.';
comment on column public.color_stats.wb_gray_world is
    'Per-channel [r,g,b] multipliers that would neutralize the frame under the gray-world assumption (mean scene reflectance is neutral gray).';
comment on column public.color_stats.wb_white_patch is
    'Per-channel [r,g,b] multipliers derived from the brightest ~1% of pixels (white-patch WB estimate); a second, independent WB candidate.';
comment on column public.color_stats.clip_shadow_pct is
    'Fraction (0..1) of sampled pixels at or below the shadow-clip luma threshold.';
comment on column public.color_stats.clip_highlight_pct is
    'Fraction (0..1) of sampled pixels at or above the highlight-clip luma threshold.';
comment on column public.color_stats.is_log_flat is
    'Heuristic: true when luma spread + dynamic range both read as a compressed log/flat input curve rather than a already-contrasty rec709-like image.';
comment on column public.color_stats.skin_lab is
    'Mean CIE Lab [L*, a*, b*] over a center-weighted geometric proxy region (no face detector exists in this codebase, and L1 runs before any cut/VLM pass) -- null when the frame has no usable center region.';
comment on column public.color_stats.palette is
    'Up to 5 dominant colors as [r,g,b] (0..1), k-means over sampled pixels, ordered by cluster prevalence (most prevalent first).';
