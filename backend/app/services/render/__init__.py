"""Server-side render/composite engine for L3 edit documents.

The single source of truth is the RESOLVED layer set (`layers.resolve`) -- the
same model the browser preview composites. `compositor.render_resolved` turns
that into a real MP4 via ffmpeg: a cheap concat fast-path for a pure spine, and
a layer-aware `filter_complex` graph (z-ordered video overlays + a gain/duck
audio mix) when operations add coverage, beds, or split edits.
"""
