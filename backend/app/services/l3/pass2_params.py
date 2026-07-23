"""
Tuning knobs for the cuts-v3 pass-2 vision call (``pass2.py``).
"""
from __future__ import annotations

STILL_WIDTH_PX = 768

# One call now carries identity + full visual judgment together (folded from
# the old split identity/visual passes -- see pass2_merge.plan.md), so each
# cut's output is larger than either half was alone. Start conservative,
# around the old identity-shard's cut cap rather than the larger visual-batch
# cap: past ~40-80 cuts in one call the model measurably starts
# stringifying/dropping fields under its own output complexity -- not a
# truncation, a reliability cliff in generating one very large nested
# structure, independent of image count. (No separate image-token cap here
# anymore either -- co-location is gone, so batching is pure size-based
# chunking; cut count was already the binding constraint in practice.)
# perception_upgrade.plan.md Part B: halved from 15 -- a cut can now carry
# TWO frames (early/late) instead of one, so image bytes per batch roughly
# double on dense clips. Cheap on Flash-Lite; keeps request size safe.
MAX_CUTS_PER_PASS2_BATCH = 7

# Batches only share a read-only cached prompt prefix, so running them
# concurrently instead of back-to-back is a pure wall-clock win (see
# pass2.build_pass2_batches / ingest.py). scale_architecture.plan.md Pillar 4:
# raised from 4 now that llm/client.complete() has a proactive per-provider
# in-flight limiter (Settings.ingest_llm_max_inflight_{anthropic,gemini}) --
# without that limiter this many concurrent batches was just a bigger retry
# storm against the provider's own rate limit, not a real wall-clock win.
MAX_PARALLEL_PASS2_BATCHES = 8
