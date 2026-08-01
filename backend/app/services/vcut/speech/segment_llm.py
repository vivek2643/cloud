"""
speech_cuts_pipeline.plan.md section 8 -- Stage 3: ONE Gemini-pro TEXT-ONLY
call over the collapsed take-instance transcripts -> beats (word-index
spans, never timestamps) + retake clustering. Reuses the generic,
provider-neutral app.services.llm.ingest_gemini/base adapter (not L3
business logic -- principle 4/7's isolation).

Schema note (a deliberate deviation from the plan's own JSON sketch): the
plan's §8 example nests `fluency`/`flags` as DICTS keyed by beat_id inside
each take_group. A bare Dict[str, X] field doesn't survive Gemini's
structured-output schema conversion reliably (no `additionalProperties`
support once app.services.llm.ingest_gemini.gemini_schema's sanitizer strips
it -- the SAME "invented shorthand ids" trap already hit and fixed in
pass1.py's question_ids field this session). So `fluency`/`flags` live
directly ON each beat instead (flat, Literal-typed where possible) and
`take_groups` is JUST a clustering of beat ids -- strictly simpler, and
avoids a known Gemini structured-output failure mode rather than risking it.

THIS MODULE SPENDS REAL MONEY once invoked (one Gemini call per project).
"""
from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel

from app.services.vcut.speech.inputs import FileSpeechInputs, Word

_SYSTEM_PROMPT = """You are segmenting spoken-word transcripts into coherent speech beats and
identifying retakes.

You are shown every file's WORD-INDEXED transcript (diarized where speaker
changes are marked, filler words flagged). For each file, propose one or
more "beats" -- coherent spoken stretches worth keeping as an edit (a
complete thought/sentence/line).

Each beat MUST use WORD INDICES, never timestamps:
  "id": a short unique string you assign (e.g. "b0", "b1", ...), unique
    across the WHOLE response, not just within one file.
  "file_id": which file this beat is from -- must be one of the files shown.
  "word_span": [i, j] -- an inclusive word-index range (i <= j), using the
    numbers shown for THAT file. Do not include leading/trailing filler
    words unless they are genuinely part of the natural line. Beats within
    one file must not overlap.
  "gist": one short phrase describing what is said.
  "fluency": 0..1 -- how complete, cleanly phrased, and well-worded this
    beat is (judge the TEXT only, not delivery/energy/tone).
  "flags": zero or more of "incomplete" (the line doesn't finish) /
    "false_start" (a stumble/restart at the beginning) -- these mark a take
    as DISQUALIFIED outright, not just lower-scoring; only use them for a
    genuine problem, never for ordinary stylistic imperfection.
  "is_speech": true for genuine spoken content by a person on the recording; set it
    to FALSE only when the words are clearly NOT that -- e.g. song lyrics from
    background music, dialogue bleeding from a TV/other playback, or boilerplate the
    transcriber invented over ambient noise ("thanks for watching", "please
    subscribe"). When unsure, leave it true. A non-speech beat still gets an id,
    file_id, word_span, and gist; just mark is_speech=false. Do not put it in any
    take_group.

Then, across ALL beats (any file), group any that are RETAKES of the same
line/content -- the speaker re-performing the same material, whether in the
same file or a different one. Each take_group is just {"beat_ids": [...]}
(2 or more beat ids that are retakes of each other). A beat with no retake
should simply not appear in any take_group -- do not create singleton groups.

Word spans must not overlap within a file. Never invent a word index outside
what was shown for that file. Never invent a file_id you were not shown.
"""


class _BeatOut(BaseModel):
    id: str
    file_id: str
    word_span: Tuple[int, int]
    gist: str = ""
    fluency: float = 0.5
    flags: List[Literal["incomplete", "false_start"]] = []
    # speech_noise_gate.plan.md Improvement A: text-only judgment of whether
    # this beat is the subject's genuine spoken performance -- one of THREE
    # orthogonal non-speech signals (alongside the L1 segment confidence gate
    # in transcript.py/inputs.py and the per-beat energy-floor gate in
    # delivery.py/orchestrate.py). This is the only one that reads the words
    # themselves, so it's the only one that can catch loud-but-not-speech
    # content (song lyrics, TV bleed, invented boilerplate) the acoustic
    # gates can't see. Default True = fail-open: the model must affirmatively
    # judge a beat non-speech to gate it.
    is_speech: bool = True


class _TakeGroupOut(BaseModel):
    beat_ids: List[str] = []


class SegmentSchema(BaseModel):
    beats: List[_BeatOut] = []
    take_groups: List[_TakeGroupOut] = []


def _check_nonempty(parsed: SegmentSchema) -> Optional[str]:
    if not parsed.beats:
        return "beats must not be empty -- emit at least one beat for the shown transcripts."
    return None


def _format_words(words: List[Word]) -> str:
    lines = []
    last_speaker = None
    for w in words:
        parts = [f"{w.idx}: {w.text}"]
        if w.is_filler:
            parts.append("[filler]")
        if w.speaker and w.speaker != last_speaker:
            parts.append(f"(speaker {w.speaker})")
            last_speaker = w.speaker
        lines.append(" ".join(parts))
    return "\n".join(lines)


def build_task_text(inputs_by_file: Dict[str, FileSpeechInputs]) -> str:
    sections = []
    for file_id, fi in inputs_by_file.items():
        sections.append(f"=== FILE {file_id} ({len(fi.words)} words) ===\n{_format_words(fi.words)}")
    return "\n\n".join(sections)


def _validate_beats(
    beats: List[_BeatOut], inputs_by_file: Dict[str, FileSpeechInputs],
) -> List[_BeatOut]:
    """Section 8's own validation requirement: word spans must exist and
    must not overlap within a file. Reimplemented locally (not imported
    from l3.lattice's own validation, per this module's isolation). Overlap
    resolution is deterministic: sort by span start, first-claimed wins."""
    by_file: Dict[str, List[_BeatOut]] = {}
    for b in beats:
        by_file.setdefault(b.file_id, []).append(b)

    out: List[_BeatOut] = []
    for file_id, file_beats in by_file.items():
        fi = inputs_by_file.get(file_id)
        if fi is None:
            continue
        n = len(fi.words)
        in_range = [b for b in file_beats if 0 <= b.word_span[0] <= b.word_span[1] < n]
        in_range.sort(key=lambda b: b.word_span[0])
        claimed_until = -1
        for b in in_range:
            if b.word_span[0] <= claimed_until:
                continue
            out.append(b)
            claimed_until = b.word_span[1]
    return out


def build_take_groups(beats: List[_BeatOut], raw_groups: List[_TakeGroupOut]) -> List[List[_BeatOut]]:
    """Beat-id clusters -> lists of the actual (validated) beats, one list
    per take group. Dangling ids (referencing an already-dropped/invalid
    beat) are silently skipped; an empty group after that is skipped
    entirely; every beat not claimed by any group becomes its own singleton
    group (section 8: "a beat with no sibling is its own singleton group,
    trivially the winner")."""
    by_id = {b.id: b for b in beats}
    grouped_ids: set = set()
    groups: List[List[_BeatOut]] = []
    for g in raw_groups:
        members = [by_id[bid] for bid in g.beat_ids if bid in by_id and bid not in grouped_ids]
        if not members:
            continue
        groups.append(members)
        grouped_ids.update(m.id for m in members)
    for b in beats:
        if b.id not in grouped_ids:
            groups.append([b])
            grouped_ids.add(b.id)
    return groups


def run_segment_llm(inputs_by_file: Dict[str, FileSpeechInputs]) -> Tuple[List[List[_BeatOut]], Dict[str, int]]:
    """Returns (take_groups, usage) -- take_groups is a list of beat-id
    clusters (validated, singleton-completed); usage is the raw dict
    ingest_store.accumulate_pass2_usage already expects."""
    from app.config import get_settings
    from app.services.llm.base import text_block
    from app.services.llm.ingest_gemini import complete_gemini

    task_text = build_task_text(inputs_by_file)
    settings = get_settings()
    completion = complete_gemini(
        _SYSTEM_PROMPT, [text_block(task_text)], SegmentSchema,
        model=settings.vcut_speech_model, max_tokens=16000, thinking="low",
        extra_check=_check_nonempty,
    )
    parsed = SegmentSchema.model_validate(completion.data)
    beats = _validate_beats(parsed.beats, inputs_by_file)
    take_groups = build_take_groups(beats, parsed.take_groups)
    return take_groups, completion.usage
