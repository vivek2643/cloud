-- =============================================
-- 020: Client-generated analysis proxies.
--
-- The desktop web app now decodes the local file ONCE and emits two tiny
-- analysis proxies that upload (in seconds) while the multi-GB raw uploads in
-- the background -- so analysis is decoupled from the raw upload (see
-- client_proxy.plan.md). Two new key slots hold them:
--
--   r2_proxy_a_key -- 480p @ 1fps video + full audio (AAC). Feeds L2 (Gemini
--                     perception) AND, via a server-side WAV demux, the whole
--                     speech/audio L1 stack (transcript, audio_features,
--                     diarization, dialogue_segments).
--   r2_proxy_b_key -- 160x90 @ 10fps, video-only. Feeds motion_dynamics.
--
-- r2_proxy_key stays exactly as-is: the 1080p editing proxy for scrub/playback,
-- still generated server-side from the raw. These are optional -- when a client
-- can't produce them (codec/machine) the server regenerates every analysis
-- input from the raw exactly as before, so ingest is unchanged in the fallback.
-- =============================================

alter table public.files
    add column if not exists r2_proxy_a_key text,
    add column if not exists r2_proxy_b_key text;

comment on column public.files.r2_proxy_a_key is
    'Client analysis proxy A: 480p@1fps + audio. Source for L2 + (demuxed) the speech/audio L1 stack. Null => regenerate from raw.';
comment on column public.files.r2_proxy_b_key is
    'Client analysis proxy B: 160x90@10fps video-only. Source for motion_dynamics. Null => regenerate from raw.';
