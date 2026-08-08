"""
seam_cut_pipeline.plan.md -- the seam-driven VLM cut pipeline. A separate
pipeline producing cut_records for the EXISTING frontend cuts channel; the
current L3 ingest (app.services.l3.ingest) stays untouched and keeps
working (settings.cuts_pipeline selects which one a project's ingest
dispatches to). See orchestrate.py for the two procrastinate tasks.
"""
from __future__ import annotations
