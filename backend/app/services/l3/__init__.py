"""L3: the auto-editor.

A prompt-driven, OpenAI-backed pipeline over the cached L1+L2 analysis layers:
  * builds a hero-cuts feed (the usable moments) at a guessed energy,
  * selects + orders the cuts into a story (Director -> Editor),
  * optionally lays light coverage,
then resolves the result into a layer set.

The deliverable is a versioned Edit Document, not a render.
"""
