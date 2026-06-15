"""
Regression test for airtight, universal audio alignment (l3.sync).

Pins the hardened cross-correlation against MEASURED truth on a real interview
dataset (envelopes frozen in scripts/fixtures/sync_envs_interview.json, so this
needs no DB and no video):

  * MVI_7749 and C0965 ARE the same moment from two cameras -> must lock with a
    small offset (~0.9 s) at the measured confidence, over the full ~700 s
    overlap.
  * C0962 / C0964 / C0963 / C0967 / MVI_7752 are DIFFERENT content -> must each
    be rejected (None), proving the gates (overlap floor + peak prominence +
    confidence) reject the coincidental short-tail matches the old code took.

If this fails after a threshold change, the change has loosened the gates enough
to risk re-introducing the false-multicam bug that dropped unique footage.

Run:  cd backend && python3 scripts/test_sync_align.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.services.l3 import sync  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sync_envs_interview.json")

# The one true synchronized pair and its measured alignment (sub-hop refined).
TRUE_PAIR = ("MVI_7749", "C0965")
EXPECTED_OFFSET_MS = 890
OFFSET_TOLERANCE_MS = 250          # comfortably inside one 500 ms hop
MIN_EXPECTED_CONFIDENCE = 0.5
MIN_EXPECTED_OVERLAP_MS = 600_000  # the real overlap is ~701 s
# These each cover distinct content and MUST be rejected against C0965.
REJECTERS = ["C0962", "C0964", "C0963", "C0967", "MVI_7752"]


def _load_envs() -> dict:
    raw = json.load(open(FIXTURE))
    return {name: (d["hop_ms"], d["sync_env"], d["lufs"]) for name, d in raw.items()}


def test_true_pair_locks() -> None:
    envs = _load_envs()
    a, b = TRUE_PAIR
    res = sync.align_envs(envs[a], envs[b])
    assert res is not None, f"{a}<->{b} should be detected as the same moment"
    assert abs(res.offset_ms - EXPECTED_OFFSET_MS) <= OFFSET_TOLERANCE_MS, (
        f"offset {res.offset_ms}ms drifted from measured {EXPECTED_OFFSET_MS}ms"
    )
    assert res.confidence >= MIN_EXPECTED_CONFIDENCE, f"confidence {res.confidence} too low"
    assert res.overlap_ms >= MIN_EXPECTED_OVERLAP_MS, (
        f"overlap {res.overlap_ms}ms -- a short-tail match slipped through the floor"
    )
    print(f"  OK  {a}<->{b} locked: {res.to_dict()}")


def test_distinct_clips_rejected() -> None:
    envs = _load_envs()
    ref = "C0965"
    for name in REJECTERS:
        res = sync.align_envs(envs[name], envs[ref])
        assert res is None, (
            f"{name}<->{ref} is distinct content but was accepted as synced: "
            f"{res.to_dict() if res else None}"
        )
        print(f"  OK  {name}<->{ref} correctly rejected")


def test_symmetry_and_silence() -> None:
    envs = _load_envs()
    a, b = TRUE_PAIR
    fwd = sync.align_envs(envs[a], envs[b])
    rev = sync.align_envs(envs[b], envs[a])
    assert fwd and rev, "true pair must align in both directions"
    # b-after-a and a-after-b should be near mirror images.
    assert abs(fwd.offset_ms + rev.offset_ms) <= OFFSET_TOLERANCE_MS, (
        f"asymmetric offsets: {fwd.offset_ms} vs {rev.offset_ms}"
    )
    # A flat/empty envelope is never simultaneous with anything.
    hop = envs[a][0]
    flat = (hop, [0.0] * len(envs[a][1]), None)
    assert sync.align_envs(flat, envs[b]) is None, "a flat envelope must not match"
    print("  OK  alignment is symmetric and rejects a flat envelope")


def main() -> None:
    print("sync alignment regression:")
    test_true_pair_locks()
    test_distinct_clips_rejected()
    test_symmetry_and_silence()
    print("ALL PASS")


if __name__ == "__main__":
    main()
