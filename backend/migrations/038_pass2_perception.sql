-- =============================================
-- Perception upgrade: two new per-cut fields (perception_upgrade.plan.md).
--
-- screen_text: on-screen text/graphics (slide, lower-third, UI, title) the
-- pass-2 image LLM read off the pixels -- unlocks tutorials/explainers/
-- screen-recordings/news. LLM-owned free text, "" when none.
--
-- salience: the cut's single strongest INSTANT ({peak_ms, score}), fused
-- deterministically at ingest (post._salience) from signals L1 already
-- computed (loudness, action_energy, onset/anchor proximity). Code-owned --
-- NOT the LLM's job -- and distinct from hero_ts_ms (the best STILL for
-- display; salience is the strongest EVENT moment).
--
-- shot_quality (technical stability: stable/shaky/whip/soft_focus/
-- racking_focus/exposure_shift/unsure) needs NO migration -- it rides
-- inside the existing `framing` jsonb column.
--
-- Idempotent, additive-safe.
-- =============================================

alter table cut_records add column if not exists screen_text text not null default '';
alter table cut_records add column if not exists salience jsonb not null default '{}'::jsonb;
