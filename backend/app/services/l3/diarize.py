"""
Shared diarization-turn loader.

L1 stores per-word speaker labels on `transcripts.segments[].words[].speaker`
(clip-local diarization ids: S0, S1, ...). Several consumers need the same
thing -- the merged speaker TURNS -- so the merge lives here once:

  * L2 perception: turns drive the audio<->visual fusion (`_fuse_speakers`) and
    the transcript scaffolding in the Gemini prompt.
  * L3 angle routing: the hero clip's turns are the "floor speaker" timeline a
    multicam baseline cuts against.

This module is a LEAF: it imports only config + psycopg, so anything (L1/L2/L3)
can import it without a cycle. Speaker labels are CLIP-LOCAL -- never compare
them across files.
"""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

from app.config import get_settings

# Merge consecutive same-speaker words into one turn unless the silence between
# them exceeds this (likely a real handover or pause).
DEFAULT_TURN_GAP_MS = 800
# Default cap so a chatty multi-minute clip can't blow up a prompt.
DEFAULT_MAX_TRANSCRIPT_CHARS = 24000

# (start_ms, end_ms, speaker)
Turn = Tuple[int, int, str]


def _pg_conn():
    import psycopg

    return psycopg.connect(get_settings().database_url, autocommit=True)


def merge_words_to_turns(
    words: List[dict], turn_gap_ms: int = DEFAULT_TURN_GAP_MS
) -> Tuple[List[Turn], List[str]]:
    """Merge time-ordered, filler-free words into (turns, transcript_lines).

    `words` must already be sorted by start_ms and have fillers removed.
    Returns the turn tuples plus one `[start-end] SPK: text` line per turn.
    """
    turns: List[Turn] = []
    lines: List[str] = []
    cur_start: Optional[int] = None
    cur_end: Optional[int] = None
    cur_spk: Optional[str] = None
    cur_text: List[str] = []

    def _flush() -> None:
        if cur_start is None:
            return
        spk = cur_spk or "S?"
        turns.append((cur_start, cur_end if cur_end is not None else cur_start, spk))
        lines.append(f"[{cur_start}-{cur_end}] {spk}: {' '.join(cur_text).strip()}")

    for w in words:
        spk = w.get("speaker") or "S?"
        start = int(w.get("start_ms", 0))
        end = int(w.get("end_ms", start))
        text = (w.get("text") or "").strip()
        if cur_spk == spk and cur_end is not None and start - cur_end <= turn_gap_ms:
            cur_end = end
            cur_text.append(text)
        else:
            _flush()
            cur_start, cur_end, cur_spk, cur_text = start, end, spk, [text]
    _flush()
    return turns, lines


def load_turns(
    file_id: str,
    turn_gap_ms: int = DEFAULT_TURN_GAP_MS,
    max_chars: int = DEFAULT_MAX_TRANSCRIPT_CHARS,
) -> Tuple[Optional[str], List[str], List[Turn]]:
    """Return (transcript_text, speaker_ids, turns) for one file.

    transcript_text is the joined turn lines (truncated to `max_chars`);
    speaker_ids are the distinct non-unknown labels; turns are the merged spans.
    Empty/missing transcript -> (None, [], []).
    """
    with _pg_conn() as conn:
        row = conn.execute(
            "select segments from transcripts where file_id = %s", (file_id,)
        ).fetchone()
    if not row or not row[0]:
        return None, [], []

    segments = row[0] if isinstance(row[0], list) else json.loads(row[0])

    words: List[dict] = []
    for seg in segments:
        for w in seg.get("words") or []:
            if w.get("is_filler"):
                continue
            words.append(w)
    words.sort(key=lambda w: w.get("start_ms", 0))
    if not words:
        return None, [], []

    turns, lines = merge_words_to_turns(words, turn_gap_ms)

    transcript_text = "\n".join(lines)
    if len(transcript_text) > max_chars:
        transcript_text = transcript_text[:max_chars] + "\n... [truncated]"

    speaker_ids = sorted({spk for _, _, spk in turns if spk != "S?"})
    return transcript_text, speaker_ids, turns
