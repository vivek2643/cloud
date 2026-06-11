"""L2: the VLM perception layer (Gemini).

A single multimodal pass over a short clip (gated by duration) that turns the
raw footage into a structured, single-take "footage log": clip-level look and
setting, durable person identities, an event timeline, and the semantic
cut-cost events L1's physical signals can't see (reactions, gaze, reveals).

The output is grounded against L1 (the Whisper transcript + diarization are fed
in as timing scaffolding) and is reconciled back to L1's 100 ms grid downstream.
"""
