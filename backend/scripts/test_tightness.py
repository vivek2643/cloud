"""
Tests for cuts-v2 SAID tightness -- no DB.

Tightness here applies ONLY to said-primary cuts (breath excision into
keep_spans); it never re-scopes a cut's own [src_in_ms, src_out_ms] claim
boundaries. Video (done/shown) tightness + granularity moved into
partition.py's `_tighten_video` (cuts_v2_boundaries.plan.md Phase C1), so a
done/shown cut passes through this module UNCHANGED -- applying inset again
here would double-tighten it. Run:  .venv/bin/python scripts/test_tightness.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
if BACKEND not in sys.path:
    sys.path.insert(0, BACKEND)

from app.services.l3 import tightness as tt  # noqa: E402
from app.services.l3 import vocab  # noqa: E402
from app.services.l3.partition import Cut  # noqa: E402


def _cut(primary, in_ms, out_ms, peak_ms=None, tags=None):
    return Cut(
        file_id="f1", src_in_ms=in_ms, src_out_ms=out_ms,
        tags=tags or [primary], primary=primary, label="x",
        peak_ms=peak_ms if peak_ms is not None else (in_ms + out_ms) // 2,
    )


def _words(*spans):
    """Build a flat word list from (start_ms, end_ms) pairs, one word each."""
    return [{"start_ms": s, "end_ms": e, "text": "w"} for s, e in spans]


def test_said_breath_removal_at_sharp_excises_long_gap():
    """At Sharp energy, a long internal silence inside a said cut is excised
    into a jump-cut keep_spans edit-list."""
    cut = _cut(vocab.CHANNEL_SAID, 0, 5000)
    words = _words((0, 800), (900, 1800), (4000, 4800))  # a 2200ms gap in the middle
    out = tt.apply_tightness(cut, energy=0.9, words=words)
    assert out.keep_spans is not None, out.keep_spans
    assert len(out.keep_spans) == 2, out.keep_spans
    assert out.keep_spans[0][1] <= 1800 + 1, out.keep_spans
    assert out.keep_spans[1][0] >= 4000 - 1, out.keep_spans
    # Boundaries themselves are untouched by tightness (partition already set them).
    assert out.src_in_ms == 0 and out.src_out_ms == 5000
    print("ok  test_said_breath_removal_at_sharp_excises_long_gap")


def test_said_no_qualifying_gap_leaves_keep_spans_none():
    """No internal gap clears the (small, Sharp-band) breath threshold ->
    keep_spans stays None -- the cut just plays contiguously."""
    cut = _cut(vocab.CHANNEL_SAID, 0, 2000)
    words = _words((0, 500), (520, 1000), (1020, 1500))  # tiny natural word gaps only
    out = tt.apply_tightness(cut, energy=0.9, words=words)
    assert out.keep_spans is None, out.keep_spans
    print("ok  test_said_no_qualifying_gap_leaves_keep_spans_none")


def test_said_without_words_is_a_safe_noop():
    """Omitting `words` (the caller didn't load them) never crashes -- just no
    breath excision."""
    cut = _cut(vocab.CHANNEL_SAID, 0, 3000)
    out = tt.apply_tightness(cut, energy=0.9, words=None)
    assert out.keep_spans is None
    assert out.src_in_ms == 0 and out.src_out_ms == 3000
    print("ok  test_said_without_words_is_a_safe_noop")


def test_done_and_shown_cuts_pass_through_unchanged():
    """A done/shown cut is already tightened by partition_clip -- this module
    must leave it exactly as-is (no re-inset), at any energy."""
    done = _cut(vocab.CHANNEL_DONE, 1000, 5000, peak_ms=3000)
    shown = _cut(vocab.CHANNEL_SHOWN, 2000, 6000, peak_ms=4000)
    for energy in (0.1, 0.5, 0.95):
        out_done = tt.apply_tightness(done, energy=energy)
        out_shown = tt.apply_tightness(shown, energy=energy)
        assert (out_done.src_in_ms, out_done.src_out_ms) == (1000, 5000), (energy, out_done)
        assert (out_shown.src_in_ms, out_shown.src_out_ms) == (2000, 6000), (energy, out_shown)
    print("ok  test_done_and_shown_cuts_pass_through_unchanged")


def test_apply_tightness_does_not_mutate_the_input_cut():
    """Pure: the original Cut object is untouched."""
    cut = _cut(vocab.CHANNEL_SAID, 0, 5000)
    words = _words((0, 800), (900, 1800), (4000, 4800))
    before = (cut.src_in_ms, cut.src_out_ms, cut.keep_spans)
    tt.apply_tightness(cut, energy=0.95, words=words)
    after = (cut.src_in_ms, cut.src_out_ms, cut.keep_spans)
    assert before == after, (before, after)
    print("ok  test_apply_tightness_does_not_mutate_the_input_cut")


def test_apply_tightness_all_routes_words_by_file():
    """The batch helper looks up each said cut's words by its own file_id;
    the done cut in a different file passes through unchanged."""
    said = _cut(vocab.CHANNEL_SAID, 0, 5000)
    done = Cut(file_id="f2", src_in_ms=1000, src_out_ms=5000, tags=[vocab.CHANNEL_DONE],
              primary=vocab.CHANNEL_DONE, label="x", peak_ms=3000)
    words_by_file = {"f1": _words((0, 800), (900, 1800), (4000, 4800))}
    out = tt.apply_tightness_all([said, done], energy=0.9, words_by_file=words_by_file)
    said_out = next(c for c in out if c.file_id == "f1")
    done_out = next(c for c in out if c.file_id == "f2")
    assert said_out.keep_spans is not None
    assert (done_out.src_in_ms, done_out.src_out_ms) == (1000, 5000)
    print("ok  test_apply_tightness_all_routes_words_by_file")


def main():
    test_said_breath_removal_at_sharp_excises_long_gap()
    test_said_no_qualifying_gap_leaves_keep_spans_none()
    test_said_without_words_is_a_safe_noop()
    test_done_and_shown_cuts_pass_through_unchanged()
    test_apply_tightness_does_not_mutate_the_input_cut()
    test_apply_tightness_all_routes_words_by_file()
    print("\nall tightness tests passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
