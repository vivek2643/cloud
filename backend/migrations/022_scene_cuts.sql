-- =============================================
-- Cuts v2, Phase B1: scene/shot detection.
--
-- The one genuinely NEW signal cuts-v2 needs (L2 used to assume "one
-- continuous take" per clip). A dedicated per-file table, additive and easy to
-- drop -- mirrors motion_dynamics (011), not overloaded onto an existing table.
-- =============================================

create table if not exists public.scene_cuts (
    file_id             uuid primary key references public.files(id) on delete cascade,
    hop_ms              int   not null default 0,
    -- Hard shot-cut boundaries (a real edit point in the source).
    shot_points         jsonb not null default '[]'::jsonb,
    -- Softer within-shot composition-change candidates (reframe, subject
    -- entering/leaving) -- the same drift signal, a lower bar.
    composition_points  jsonb not null default '[]'::jsonb,
    schema_version      int   not null default 1,
    created_at          timestamptz not null default now()
);

comment on table public.scene_cuts is
    'L1 cuts-v2 signal: shot/composition-change detection from one frame-to-frame color-histogram-drift pass over the proxy (see l1.scene_cuts).';
comment on column public.scene_cuts.shot_points is
    'Hard shot-cut instants to split ON [{ts_ms,kind,score}]. kind: shot_cut.';
comment on column public.scene_cuts.composition_points is
    'Softer within-shot composition-change instants [{ts_ms,kind,score}]. kind: composition_change.';
comment on column public.scene_cuts.schema_version is
    'l1.scene_cuts.SCHEMA_VERSION at write time, so a detection-logic change can be told apart from a stale row.';
