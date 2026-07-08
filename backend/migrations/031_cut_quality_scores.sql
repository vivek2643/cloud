-- =============================================
-- Cuts v3: deterministic per-cut quality scores + person characteristics.
--
-- Two 0..1 scores, both computed in code (post.assemble_cut_records) from L1
-- signals + the pass-2 image judgment -- never an LLM number, never a magic
-- threshold on a raw measurement (every continuous term is normalised against
-- the clip's own spread):
--   * speech_quality -- delivery ONLY (fluency + loudness); camera-independent,
--     so simultaneous angles of one line score the same. NULL for a cut with
--     no speech (video-only cuts).
--   * total_quality  -- speech (if any) blended with visual presentation
--     (on-camera, shot tightness, sharpness, look). This is the number that
--     crowns the winner WITHIN a same-setting take cluster (outlook angles are
--     never crowned -- see post._enforce_take_winner). Always set.
--
-- characteristics -- per-person appearance fingerprints from the pass-2 image
-- LLM (list of {description, position, speaking}); lets take/outlook grouping
-- and "show the speaker" arrange logic recognise the same person across cuts.
--
-- shot_size rides inside the existing `framing` jsonb (pass2b.Framing) -- no
-- column of its own. All additive; existing rows default (scores NULL/0,
-- characteristics []) and are repopulated on re-ingest.
-- =============================================

alter table public.cut_records
    add column if not exists speech_quality  real,
    add column if not exists total_quality   real not null default 0,
    add column if not exists characteristics jsonb not null default '[]';

comment on column public.cut_records.speech_quality is
    'Cuts v3 deterministic delivery score 0..1 (fluency + clip-relative loudness). NULL for cuts with no speech. See post.compute_speech_quality.';
comment on column public.cut_records.total_quality is
    'Cuts v3 deterministic rank score 0..1 (speech blended with visual presentation). Crowns the winner within a same-setting take cluster. See post.compute_total_quality.';
comment on column public.cut_records.characteristics is
    'Cuts v3 per-person appearance fingerprints from the pass-2 image LLM: [{description, position, speaking}]. For cross-cut re-identification by eye.';
