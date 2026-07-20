"""
Semantic scene grouping (color_grading_upgrade.plan.md Step 3.2): groups
timeline shots by MEANING (the same setup/scene), not merely RGB proximity
-- the reliability edge over pixel-only tools, which mis-match unrelated
shots that happen to share a palette and FAIL to match related shots whose
RGB drifted for an incidental reason (a bright object entering frame, a
light flicker).

Feeds `grade.match.solve_sequence_match`'s optional `groups` param, taking
over from its default RGB-based `group_neighbors` when semantic signals are
available (`settings.grade_semantic`).

Signals used, most-to-least trusted (see the plan's own framing: Pass 2's
label/summary is only a coarse 2-frame VLM read, so it's NEVER the sole
signal):
  - same `file_id`: a continuous single-camera take is one scene by
    construction.
  - same `speaker_person` (a structural ASD/identity fact, not a VLM guess):
    two cuts of the SAME person are very likely the same setup.
  - same `on_camera`-ness PLUS coarse label/summary word-overlap -- the
    weakest signal, only ever consulted as a tiebreak alongside another
    corroborating fact, never alone.

Chain-linked to the immediately preceding shot only (mirrors `match.
group_neighbors`'s discipline): two DISTANT shots, however similar, are
never grouped -- gradual continuity within one scene survives, but nothing
drags unrelated shots together.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ShotSceneMeta:
    """Structural + coarse-semantic facts about one shot, for GROUPING
    only -- never fed into the color math itself (that stays span-stat
    driven; see `match.ShotStats`)."""
    key: str
    file_id: str
    speaker_person: Optional[str] = None
    on_camera: Optional[bool] = None
    label: str = ""
    summary: str = ""


# Words this short are too generic (articles, prepositions) to mean
# anything as a topical overlap signal.
_MIN_WORD_LEN = 4


def _label_overlap(a: ShotSceneMeta, b: ShotSceneMeta) -> bool:
    words_a = {w for w in (a.label + " " + a.summary).lower().split() if len(w) >= _MIN_WORD_LEN}
    words_b = {w for w in (b.label + " " + b.summary).lower().split() if len(w) >= _MIN_WORD_LEN}
    return bool(words_a & words_b)


def group_shots_semantically(ordered_shots: List[ShotSceneMeta]) -> List[List[int]]:
    """Chain-link adjacent shots into groups (as ordered-list INDICES) using,
    in order of trust: same `file_id` -> same `speaker_person` (when both
    known) -> same `on_camera`-ness AND label/summary overlap (the weakest
    signal, only ever a tiebreak). A shot with no qualifying predecessor
    starts its own group of 1."""
    groups: List[List[int]] = []
    for i, shot in enumerate(ordered_shots):
        if groups:
            prev = ordered_shots[groups[-1][-1]]
            same_file = bool(shot.file_id) and shot.file_id == prev.file_id
            same_speaker = shot.speaker_person is not None and shot.speaker_person == prev.speaker_person
            weak_tiebreak = (
                shot.on_camera is not None and shot.on_camera == prev.on_camera
                and _label_overlap(shot, prev)
            )
            if same_file or same_speaker or weak_tiebreak:
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups
