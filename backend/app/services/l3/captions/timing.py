"""
Caption timing engine (captions.plan.md SS10): per spine (dialogue) layer,
turns a file's transcript words into program-time-mapped CAPTION EVENTS --
grouped into readable lines, held for a readability floor, with one
emphasised word per event and (optionally) beat-snapped.

Word source is TIME-BASED (SS16 resolved #4): each spine `VideoLayer`
(`kind="spine"`) already carries the exact `[src_in_ms, src_out_ms] ->
[prog_start_ms, prog_end_ms]` mapping (`layers.resolve`'s own spine spans),
so slicing a file's `transcripts.segments[].words[]` by that source window
and re-basing onto program time is a direct, retime-safe transform -- no
`word_span` index games (confirmed non-portable outside the ingest lattice;
see captions.plan.md SS2's own hedge). Filler words are dropped by default
(`is_filler`); a covering `cut_records.channel != "said"` beat is suppressed
entirely (SS10 "no captions on channel != said beats").

Pure function module: every signal (words, rms_db envelope, onsets, cut
records) is passed in already-fetched. No I/O here.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# A cheap, non-ML "semantic keyword" proxy (SS10 "semantic" emphasis mode):
# content words tend to be longer AND rarer than function words, so filtering
# common function words and ranking by length is a defensible zero-cost
# stand-in for a real keyword extractor. Deliberately short -- this only
# needs to catch the highest-frequency offenders, not be exhaustive.
_STOPWORDS = frozenset("""
a an the and or but if so to of in on at is are was were be been being
it its this that these those i you he she we they my your his her our
their as for with from by about into over under again then than not
just really very okay ok yeah well like know think mean got get going
do did does can could would should will up down out no yes
""".split())

# Minimum caption hold time (SS10/SS16 "readability_ms" -- the plan assumed a
# `cut_records.readability_ms` column, which turns out NOT to persist (it's
# folded into `pace.min_ms`, a CUT-trim floor, not a per-line reading floor;
# see captions.plan.md verification). Standard subtitle-industry reading-
# speed heuristic instead: a floor by character count plus an absolute floor,
# so a one-word line never flashes for 40ms.
READ_MS_PER_CHAR = 45          # ~22 chars/sec, a common (Netflix-ish) reading speed
MIN_EVENT_HOLD_MS = 700
MAX_EVENT_HOLD_MS = 7000
# A gap this long between two consecutive words reads as a natural sentence/
# beat break -- close the current event here even if there's char budget left.
SENTENCE_GAP_MS = 550
# Beat-sync snap window (SS10): only pull an emphasis onto a beat if it's
# already close -- never yank the pop far from where the word actually lands.
BEAT_SNAP_WINDOW_MS = 150


class Word(Dict[str, Any]):
    """Loosely-typed: {text, start_ms, end_ms, is_filler} as stored in
    `transcripts.segments[].words[]` (see l1.transcript.Word)."""


def words_in_source_window(
    segments: Sequence[Dict[str, Any]], src_in_ms: int, src_out_ms: int, *, drop_fillers: bool = True
) -> List[Dict[str, Any]]:
    """Every word from a file's transcript `segments` fully inside
    `[src_in_ms, src_out_ms]`, time-ordered. A word straddling the trim edge
    is dropped rather than clipped -- a half-word caption reads worse than a
    momentarily-early cut (SS2's own "prefer this over word_span" framing)."""
    out: List[Dict[str, Any]] = []
    for seg in segments or []:
        for w in seg.get("words") or []:
            if drop_fillers and w.get("is_filler"):
                continue
            s, e = int(w.get("start_ms", 0)), int(w.get("end_ms", 0))
            if s >= src_in_ms and e <= src_out_ms and e > s:
                out.append(w)
    out.sort(key=lambda w: w["start_ms"])
    return out


def _to_program(word: Dict[str, Any], src_in_ms: int, prog_start_ms: int) -> Tuple[int, int]:
    t_in = prog_start_ms + (int(word["start_ms"]) - src_in_ms)
    t_out = prog_start_ms + (int(word["end_ms"]) - src_in_ms)
    return t_in, t_out


def _display_text(text: str, case: str) -> str:
    t = text.strip()
    return t.upper() if case == "upper" else t


def _rms_at(rms_db: Optional[List[float]], hop_ms: int, ms: int) -> float:
    if not rms_db or hop_ms <= 0:
        return -60.0
    i = max(0, min(len(rms_db) - 1, ms // hop_ms))
    return float(rms_db[i])


def _pick_emphasis_word(
    words: List[Dict[str, Any]], mode: str, *, rms_db: Optional[List[float]], hop_ms: int
) -> Optional[int]:
    """Index (into `words`) of the ONE word to emphasise in a line, or None
    (`emphasis: "none"`). "loudness" ranks by the rms_db envelope sampled at
    the word's own FILE-absolute time window (SS10 "scaled by nrg/rms_db
    loudness -> the pop lands on the right word"; `rms_db` is indexed by
    absolute file-ms via `hop_ms`, same domain as the word's own
    `start_ms`/`end_ms` -- see l1 `transcript.Word`); "semantic" ranks by
    the stopword-filtered length heuristic above, tie-broken by loudness so
    it still lands on a plausible beat."""
    if mode == "none" or not words:
        return None
    candidates = [
        (i, w) for i, w in enumerate(words)
        if re.sub(r"[^a-zA-Z']", "", w.get("text", "")).lower() not in _STOPWORDS
    ] or list(enumerate(words))

    def loudness(i_w: Tuple[int, Dict[str, Any]]) -> float:
        _, w = i_w
        mid = (int(w["start_ms"]) + int(w["end_ms"])) // 2
        return _rms_at(rms_db, hop_ms, mid)

    if mode == "semantic":
        best = max(
            candidates,
            key=lambda iw: (len(re.sub(r"[^a-zA-Z']", "", iw[1].get("text", ""))), loudness(iw)),
        )
        return best[0]
    # "loudness" (default)
    best = max(candidates, key=loudness)
    return best[0]


def _snap_to_beat(ms: int, onsets_ms: Optional[List[int]]) -> int:
    if not onsets_ms:
        return ms
    nearest = min(onsets_ms, key=lambda o: abs(o - ms))
    return nearest if abs(nearest - ms) <= BEAT_SNAP_WINDOW_MS else ms


def _line_break(
    words: List[Dict[str, Any]], max_chars: int, max_lines: int
) -> List[List[Dict[str, Any]]]:
    """Greedy word-wrap into <= `max_lines` lines of <= `max_chars` each.
    Never drops a word: if `max_lines` is exceeded the caller (`build_events`)
    closes the event and starts a new one instead of truncating text."""
    lines: List[List[Dict[str, Any]]] = [[]]
    cur_len = 0
    for w in words:
        text = w.get("text", "")
        add_len = len(text) + (1 if cur_len else 0)
        if cur_len + add_len > max_chars and cur_len > 0:
            lines.append([])
            cur_len = 0
            add_len = len(text)
        lines[-1].append(w)
        cur_len += add_len
    return lines


def build_events(
    words: List[Dict[str, Any]],
    *,
    src_in_ms: int,
    prog_start_ms: int,
    layer_prog_end_ms: int,
    max_chars_per_line: int,
    max_lines: int,
    case: str,
    emphasis_mode: str,
    beat_sync: bool,
    rms_db: Optional[List[float]] = None,
    rms_hop_ms: int = 0,
    onsets_ms: Optional[List[int]] = None,
    is_musical: bool = False,
    next_event_start_ms: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """One spine layer's words -> a list of caption EVENT dicts:
    `{prog_start_ms, prog_end_ms, lines: [{words: [{text,t_in_ms,t_out_ms,
    emphasized}]}]}`, program-time already mapped, readability-held, one
    emphasised word per event, optionally beat-snapped.

    Segmentation: greedy-fill lines to `max_chars_per_line`; once a line
    count would exceed `max_lines`, OR a gap to the next word exceeds
    SENTENCE_GAP_MS (a natural beat break), the event closes and a new one
    starts. `next_event_start_ms` (this layer's own successor, if any) caps
    how far the LAST event's readability hold may extend so two captions
    never visually overlap.
    """
    if not words:
        return []

    # First pass: chunk the word stream into event-sized runs.
    runs: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = [words[0]]
    for w in words[1:]:
        gap = int(w["start_ms"]) - int(cur[-1]["end_ms"])
        trial_lines = _line_break(cur + [w], max_chars_per_line, max_lines)
        if gap > SENTENCE_GAP_MS or len(trial_lines) > max_lines:
            runs.append(cur)
            cur = [w]
        else:
            cur = cur + [w]
    runs.append(cur)

    events: List[Dict[str, Any]] = []
    for idx, run in enumerate(runs):
        lines_words = _line_break(run, max_chars_per_line, max_lines)
        first_in, _ = _to_program(run[0], src_in_ms, prog_start_ms)
        _, last_out = _to_program(run[-1], src_in_ms, prog_start_ms)

        char_count = sum(len(w.get("text", "")) for w in run)
        read_floor = min(MAX_EVENT_HOLD_MS, max(MIN_EVENT_HOLD_MS, char_count * READ_MS_PER_CHAR))
        held_end = max(last_out, first_in + read_floor)
        # Clamp against whatever comes next (the following event in this same
        # run-set, else the caller-supplied successor layer's own start) so
        # extending for readability never overlaps the next caption. No cap
        # at all for the very last event of the very last layer (nothing
        # follows it to overlap).
        if idx + 1 < len(runs):
            cap = runs[idx + 1][0]["start_ms"] - src_in_ms + prog_start_ms
        else:
            cap = next_event_start_ms
        if cap is not None:
            held_end = min(held_end, max(last_out, cap - 1))

        emph_idx = _pick_emphasis_word(run, emphasis_mode, rms_db=rms_db, hop_ms=rms_hop_ms)
        emph_word = run[emph_idx] if emph_idx is not None else None

        lines_out = []
        for line in lines_words:
            words_out = []
            for w in line:
                t_in, t_out = _to_program(w, src_in_ms, prog_start_ms)
                if beat_sync and is_musical and onsets_ms and w is emph_word:
                    t_in = _snap_to_beat(t_in, onsets_ms)
                words_out.append({
                    "text": _display_text(w.get("text", ""), case),
                    "t_in_ms": t_in, "t_out_ms": t_out,
                    "emphasized": w is emph_word,
                })
            lines_out.append({"words": words_out})

        events.append({
            "prog_start_ms": first_in, "prog_end_ms": held_end,
            "lines": lines_out,
        })
    return events
