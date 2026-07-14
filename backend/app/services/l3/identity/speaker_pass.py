"""
Voice -> face binding pass (voice_first_identity.plan.md Phase F): a SMALL,
fixed Gemini vision call, one per voice, that looks at the close-burst
frames identity/speaker_frames.py planned and answers one question per
window -- does any visible candidate's mouth actively move, and if so which
one? Code aggregates the per-window votes into voice -> owner + confidence
(majority + margin, else unbound). Keeps "model perceives, code decides":
the model only ever names which VISIBLE FACE is mouthing, never assigns an
id, a score, or resolves a tie itself.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from app.config import get_settings
from app.services.l3.identity.speaker_frames import Burst
from app.services.llm.base import image_block, text_block

logger = logging.getLogger(__name__)

# Binding requires the winning candidate to take a clear MAJORITY of a
# voice's windows, with a real margin over the runner-up -- a close or
# ambiguous vote leaves the voice unbound (owner unknown) rather than
# guessing. Conservative on purpose: a wrong bind is worse than an honest
# "don't know" (identity_map.plan.md's founding stance, carried forward).
MIN_VOTE_SHARE = 0.5
MIN_VOTE_MARGIN = 1


class WindowVote(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window_id: str
    speaking_person: Optional[str] = None   # one of the window's candidate person_ids, or unset


class SpeakerPassOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    votes: List[WindowVote] = Field(default_factory=list)


_SYSTEM = (
    "You are shown a series of WINDOWS, each a short moment where ONE voice "
    "was talking. Each window shows one or more CANDIDATE people (labelled by "
    "id and a short description), each as a tiny burst of close, consecutive "
    "frames at the SAME instants -- read a candidate's burst as one "
    "continuous glimpse, not separate stills, and judge whether their mouth "
    "is ACTIVELY MOVING across it (open/closing, articulating) versus "
    "still/closed/not clearly visible.\n\n"
    "For each window, answer: does exactly one candidate's mouth clearly "
    "move? If yes, set speaking_person to that candidate's id, verbatim. If "
    "no candidate's mouth clearly moves, more than one seems to move, or the "
    "mouth isn't visible/legible in any candidate's frames, leave "
    "speaking_person unset -- never guess. You are only asked to say WHICH "
    "VISIBLE FACE is mouthing, if any; never invent a candidate id, never "
    "assign a score, never answer for a window you were not shown.\n\n"
    "Emit exactly one vote per window shown, using the SAME window_id given "
    "in its label."
)


def build_speaker_pass_blocks(
    bursts: List[Burst], images_b64: Dict[Tuple[str, int], str], persons: Dict[str, dict],
) -> List[Any]:
    """One voice's full prompt content: every window (grouped), each
    candidate's burst captioned with its person id + display description
    (identity/reconcile.py Phase D) so the model can NAME who it means, not
    just point at a position. A candidate frame with no extracted image is
    skipped (never sent blank); a candidate that resolves zero images
    contributes nothing (never captioned with no pixels behind it)."""
    by_window: Dict[str, List[Burst]] = {}
    for b in bursts:
        by_window.setdefault(b.window_id, []).append(b)

    blocks: List[Any] = []
    for window_id in sorted(by_window.keys()):
        for b in sorted(by_window[window_id], key=lambda x: x.candidate_person or ""):
            frame_blocks = [image_block(b64) for ts in b.ts_ms
                            if (b64 := images_b64.get((b.file_id, ts))) is not None]
            if not frame_blocks:
                continue
            display = (persons.get(b.candidate_person) or {}).get("display", b.candidate_person)
            blocks.append(text_block(f"WINDOW {window_id}, candidate {b.candidate_person} ({display}):"))
            blocks.extend(frame_blocks)
    return blocks


def run_speaker_pass(
    bursts: List[Burst], images_b64: Dict[Tuple[str, int], str], persons: Dict[str, dict],
) -> List[WindowVote]:
    """One Gemini call for ONE voice's bursts. Returns the raw per-window
    votes -- empty when there are no resolvable images to show at all
    (never a fabricated vote)."""
    blocks = build_speaker_pass_blocks(bursts, images_b64, persons)
    if not blocks:
        return []
    from app.services.llm import ingest_gemini as ig
    settings = get_settings()
    completion = ig.complete_gemini(
        _SYSTEM, blocks, SpeakerPassOutput, max_tokens=2048,
        model=settings.identity_speaker_pass_model, thinking="low",
    )
    return SpeakerPassOutput.model_validate(completion.data).votes


def aggregate_votes(votes: List[WindowVote]) -> Optional[str]:
    """Majority + margin over a SINGLE voice's window votes -> the winning
    candidate person id, or None (unbound) when the vote is close, split,
    or empty. Requires the winner to hold a clear majority share
    (`MIN_VOTE_SHARE`) AND a real margin (`MIN_VOTE_MARGIN`) over the
    runner-up -- a 2-2 split or an unconvincing 1-vote edge both stay
    unbound rather than guessing."""
    total = len(votes)
    if total == 0:
        return None
    counts: Dict[str, int] = {}
    for v in votes:
        if v.speaking_person:
            counts[v.speaking_person] = counts.get(v.speaking_person, 0) + 1
    if not counts:
        return None
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    winner, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0
    if top / total < MIN_VOTE_SHARE:
        return None
    if top - second < MIN_VOTE_MARGIN:
        return None
    return winner


def bind_voices(
    bursts: List[Burst], images_b64: Dict[Tuple[str, int], str], persons: Dict[str, dict],
) -> Dict[str, Optional[str]]:
    """Run the binding pass once per voice (bursts grouped by `.voice`),
    aggregate each voice's votes, and return `{voice: owner_person_id_or_
    None}` for every voice that had at least one burst to show. A voice
    with no bursts at all (identity/speaker_frames.py already flagged it
    off-camera, no visible-person candidate anywhere) is simply absent here
    -- the caller already has that from `plan_bursts`'s own off_camera set."""
    by_voice: Dict[str, List[Burst]] = {}
    for b in bursts:
        by_voice.setdefault(b.voice, []).append(b)
    out: Dict[str, Optional[str]] = {}
    for voice, voice_bursts in sorted(by_voice.items()):
        votes = run_speaker_pass(voice_bursts, images_b64, persons)
        out[voice] = aggregate_votes(votes)
        if out[voice] is None:
            logger.info("speaker_pass: voice %s unbound (%d window vote(s), no confident majority)",
                       voice, len(votes))
    return out
