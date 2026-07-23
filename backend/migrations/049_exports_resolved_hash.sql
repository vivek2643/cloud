-- =============================================
-- 049: Dedup column for exports (scale_architecture.plan.md Pillar 5).
--
-- Mirrors 016_renders.sql's resolved_hash: a fingerprint of the resolved
-- timeline + kind + quality + include_media, so the API can short-circuit
-- to an existing 'done' export of the identical (thread, version, kind,
-- quality, include_media) combination instead of re-rendering/re-bundling
-- for free (export_options.plan.md never added this -- renders got it,
-- exports didn't).
-- =============================================

alter table public.exports
    add column if not exists resolved_hash text;

create index if not exists idx_exports_dedup
    on public.exports(thread_id, document_version, kind, quality, include_media);

comment on column public.exports.resolved_hash is
    'scale_architecture.plan.md Pillar 5: fingerprint of (resolved timeline, kind, quality, include_media) for export dedup, mirroring renders.resolved_hash.';
