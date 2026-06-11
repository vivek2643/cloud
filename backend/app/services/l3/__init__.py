"""L3: the edit orchestrator.

Two brains over the cached analysis layers:
  * Claude Opus (creative): reads the L1+L2 text analysis, decides what
    material to use, in what order, with what intent -- and asks the user
    when genuinely blocked.
  * A deterministic cut-engine (mechanical): snaps every proposed cut to the
    L1 cost grids, fits the total duration, and scores/validates assemblies.

The deliverable is a versioned Edit Document, not a render.
"""
