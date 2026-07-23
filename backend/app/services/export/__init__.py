"""Export subsystem (export_options.plan.md): turns a resolved edit document
into a deliverable other than the MP4 render itself -- SRT sidecar subtitles
(`srt.py`), an FCPXML rough cut (`fcpxml.py`), and the self-relinking ZIP that
bundles a rough cut for an NLE (`bundle.py`). All three, plus the existing MP4
render, read the SAME `render.tasks.resolve_document(...)` output so every
deliverable matches.
"""
