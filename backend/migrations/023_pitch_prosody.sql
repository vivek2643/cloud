-- =============================================
-- Cuts v2, Phase C2: pitch/f0 prosody signal.
--
-- The one new L1 signal cuts_v2_boundaries.plan.md's speech-boundary grader
-- needs: a coarse pitch track, on the SAME hop as the existing rms_db
-- envelope, so a caller can read pitch + energy at matching instants. Folded
-- onto audio_features (not a new table) since it shares that table's hop and
-- lifecycle exactly -- additive, nullable/defaulted, easy to drop.
--
-- Existing rows read f0_hz = [] until a re-analyze recomputes them; the
-- speech-boundary grader degrades to RMS + gap-length alone when empty (see
-- l3.partition._prosody_bridges_gap).
-- =============================================

alter table public.audio_features
    add column if not exists f0_hz jsonb not null default '[]'::jsonb;

comment on column public.audio_features.f0_hz is
    'Coarse pitch/f0 track (Hz) via librosa pyin, same hop as rms_db (prosody_hop_ms). 0.0 = unvoiced/silent, not NaN. Empty until a re-analyze populates it (see l1.audio_features._compute_prosody).';
