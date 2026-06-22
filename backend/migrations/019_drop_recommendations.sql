-- =============================================
-- 019: drop the recommendations table
--
-- The "recommendations" concept (an LLM filtration pass that flagged a curated
-- "Recommended" pool over the hero-cuts feed) has been removed end-to-end:
-- the l3.recommend module, its hero_cuts tagging, the API fields, and the
-- frontend "Recommended" filter are all gone. This table was created lazily at
-- runtime by the now-deleted module and nothing references it anymore.
--
-- Idempotent.
-- =============================================

drop table if exists public.recommendations;
