"""
speech_cuts_pipeline.plan.md -- vcut's own speech channel: transcript-first
segmentation, take/outlook detection, delivery scoring, and winner
selection, replacing the copy_prior_speech_cuts stopgap. See orchestrate.py
for the entry point (run_speech_channel), called from
app.services.vcut.orchestrate.run_vcut_ingest.
"""
from __future__ import annotations
