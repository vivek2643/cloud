"""
speech_cuts_pipeline.plan.md section 10 -- Stage 5 (part 2): flag gates +
fluency/delivery fusion -> winner. PURE (no I/O). Same shape as the seam
curve: flags are multiplicative GATES, fluency/delivery are additive
ATTRACTORS. Because everything upstream (delivery.py) and this module are
pure and the LLM output (fluency/flags/clustering) is persisted, winners can
be re-picked without re-calling the model -- retune params.py, instant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from app.services.vcut.speech.params import W_DELIVERY, W_FLUENCY


@dataclass
class TakeCandidate:
    beat_id: str
    fluency_llm: float          # 0..1, from segment_llm.py
    flags: List[str] = field(default_factory=list)   # e.g. ["incomplete", "false_start"]
    delivery: float = 0.0       # group_delivery_scores' own output for this candidate


@dataclass
class TakeGroupResult:
    winner_beat_id: str
    roles: Dict[str, str]       # beat_id -> "winner" | "take" (cut_records.take_role's real
                                 # check constraint -- ('take', 'outlook', 'winner'), NOT
                                 # "alternate"; the plan's own prose uses "alternate" loosely,
                                 # but the DB schema (024_cuts_v3.sql) doesn't have that value)
    finals: Dict[str, float]    # beat_id -> final fused (gated) score


def _raw_score(c: TakeCandidate) -> float:
    return W_FLUENCY * c.fluency_llm + W_DELIVERY * c.delivery


def select_winner(candidates: List[TakeCandidate]) -> TakeGroupResult:
    """A single candidate (section 8: "a beat with no sibling is its own
    singleton group") always wins, flags notwithstanding -- there's no
    alternative to prefer instead. Otherwise: gate (0 if flagged) times the
    fluency+delivery fusion, argmax wins. If EVERY candidate in a group is
    flagged (gated to 0), fall back to ranking by the raw (ungated) score --
    a take group should never end up winner-less just because every attempt
    had some flaw; something still has to ship."""
    if not candidates:
        raise ValueError("select_winner: empty take group")
    if len(candidates) == 1:
        only = candidates[0].beat_id
        return TakeGroupResult(winner_beat_id=only, roles={only: "winner"},
                              finals={only: _raw_score(candidates[0])})

    finals = {c.beat_id: (0.0 if c.flags else 1.0) * _raw_score(c) for c in candidates}
    if all(v == 0.0 for v in finals.values()):
        raw = {c.beat_id: _raw_score(c) for c in candidates}
        winner_id = max(raw, key=raw.get)
    else:
        winner_id = max(finals, key=finals.get)

    roles = {c.beat_id: ("winner" if c.beat_id == winner_id else "take") for c in candidates}
    return TakeGroupResult(winner_beat_id=winner_id, roles=roles, finals=finals)
