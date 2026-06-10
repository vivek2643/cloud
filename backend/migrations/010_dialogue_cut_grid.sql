-- =============================================
-- L1 derived signal: dialogue cut-cost grid (universal-editor phase 1)
--
-- A cheap, CPU-only derivation over signals L1 already stores (Whisper word
-- timings + fillers + diarization speakers + pause_map / rms_db). Tells the
-- editor where it is cheap vs forbidden to cut a dialogue clip.
--
-- Additive + nullable so existing audio_features rows keep working. Silent /
-- music-only files leave these empty (dialogue-only signal).
-- =============================================

alter table public.audio_features
    add column if not exists dialogue_cut_cost   jsonb not null default '[]'::jsonb,
    add column if not exists dialogue_cut_hop_ms int   not null default 0,
    add column if not exists dialogue_cut_points  jsonb not null default '[]'::jsonb;

comment on column public.audio_features.dialogue_cut_cost is
    'Dense per-hop cut cost (0=ideal seam .. 1=forbidden/mid-word) sampled every dialogue_cut_hop_ms. "Safe to cut" = 1 - cost.';
comment on column public.audio_features.dialogue_cut_hop_ms is
    'Sample hop (ms) for dialogue_cut_cost (default 100ms = minimum expected cut granularity).';
comment on column public.audio_features.dialogue_cut_points is
    'Discrete exact-timestamp seam candidates [{ts_ms,gap_start_ms,gap_end_ms,kind,score}]. kind: word_gap|sentence_end|speaker_change|pause|filler_edge.';
