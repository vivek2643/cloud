-- =============================================
-- Cuts v3: per-cut continuity (clip position + weldable-neighbor flags).
--
-- Computed ONCE at ingest (post.assemble_cut_records) from the same lattice/
-- atom signals pass 1's own word-gap seam test reads (seam.classify_seam),
-- generalized from a word-to-word seam to a CUT-to-cut seam on the same clip.
-- Read paths (brain + UI) just read the block -- no re-derivation.
--
-- {clip, cut_no, of, prev_contiguous, next_contiguous, seam_reason_prev,
--  seam_reason_next}. cut_no/of number ALL cuts on the clip INCLUDING junk
-- (a gap in the numbering is the signal a junk beat sits there). Additive;
-- existing rows default to '{}' and are backfilled on re-ingest.
-- See cuts_v3_continuity.plan.md.
-- =============================================

alter table public.cut_records
    add column if not exists continuity jsonb not null default '{}';

comment on column public.cut_records.continuity is
    'Cuts v3 continuity: {clip, cut_no, of, prev_contiguous, next_contiguous, seam_reason_prev, seam_reason_next}, computed once at ingest via seam.classify_seam. Numbering is over ALL cuts on the clip incl. junk. See cuts_v3_continuity.plan.md.';
