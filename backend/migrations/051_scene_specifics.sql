-- =============================================
-- 051: Scene specificity (cut_structure_and_scene_specificity.plan.md
-- Part 3) -- the two-pass, intent-conditioned vision enrichment that turns
-- a generic Pass-A summary ("a machine in a factory") into a specific one
-- ("CNC lathe turning a steel shaft") via a text middle layer (project
-- domain + targeted questions) followed by a small, targeted Pass B vision
-- call. Runs in the background AFTER cuts are shown; a run/cut with no
-- enrichment yet (or one that failed) simply has NULL here -- cuts remain
-- fully usable without it.
--
-- cut_records.scene_specifics: this ONE cut's Pass-B answer, e.g.
--   {"specific": "pheras -- couple circling the fire", "label": "pheras"}
-- "label" is the closed-set taxonomy id when one applies (or "other"),
-- omitted entirely on the generic-rubric fallback (domain unknown/mixed).
--
-- ingest_runs.scene_taxonomy: the project-level middle-layer output --
-- domain, confidence, evidence, the closed-set taxonomy (empty unless a
-- genuine one exists), and needs_pass_b for auditability. Lives on
-- ingest_runs (not a separate table) for the same reason pass1_output/
-- identity_map do: one run, one snapshot, read back by run id.
-- =============================================

alter table public.cut_records
    add column if not exists scene_specifics jsonb;

alter table public.ingest_runs
    add column if not exists scene_taxonomy jsonb;

comment on column public.cut_records.scene_specifics is
    'cut_structure_and_scene_specificity.plan.md Part 3: this cut''s Pass-B '
    'specific -- {"specific": <one-line>, "label": <closed-set id or "other">}. '
    'NULL until the background enrich job reaches this cut (or if it was not '
    'selected for Pass B -- Pass A''s own summary was specific enough).';

comment on column public.ingest_runs.scene_taxonomy is
    'cut_structure_and_scene_specificity.plan.md Part 3: the middle text '
    'layer''s project-level output -- {"domain", "confidence", "evidence", '
    '"taxonomy", "needs_pass_b"}. NULL until the background enrich job runs '
    '(or on a run predating this plan).';
