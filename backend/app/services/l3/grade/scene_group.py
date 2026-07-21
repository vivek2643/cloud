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

Signals used, most-to-least trusted (color_scene_grouping.plan.md extended
this beyond the original 3 -- see that plan's DB-verified population rates:
`label`/`summary` 100%, `on_camera` ~55%, `speaker_person` ~13%,
`take_group_id` ~32%, `sync_group_id` ~25% -- so the structural ids are
strong SIGNALS when present, not universal ones):
  - same `file_id`: a continuous single-camera take is one scene by
    construction.
  - same `sync_group_id` (multicam outlook: the same moment from another
    camera, sharing authoritative audio) -- definitively the same scene.
  - same `take_group_id` (retakes of the same content) -- same setup/lighting.
  - same `speaker_person` (a structural ASD/identity fact, not a VLM guess),
    OR overlapping `voice_ids` (a coarser but still structural voice-cluster
    signal) -- two cuts of the SAME person are very likely the same setup.
  - same `on_camera`-ness PLUS coarse label/summary word-overlap -- the
    weakest signal, only ever consulted as a tiebreak alongside another
    corroborating fact, never alone.
  - `rgb_close` (the graceful BASE, always available from `measure_span`):
    span `rgb_mean` within `SCENE_RGB_DIST_MAX` of the previous shot -- the
    same test `match.group_neighbors` uses, folded in here so semantic
    grouping itself never degrades to all-singletons when the structural/
    label signals are genuinely absent (the common real-data case on a
    montage reel, verified: label/summary present but no speaker/on_camera/
    take/sync link).

Chain-linked to the immediately preceding shot only (mirrors `match.
group_neighbors`'s discipline): two DISTANT shots, however similar, are
never grouped -- gradual continuity within one scene survives, but nothing
drags unrelated shots together.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
    # color_scene_grouping.plan.md: stronger structural scene signals joined
    # from cut_records (grade.scene_meta.lookup_shot_cut_meta), and the span
    # RGB used as the graceful grouping base. All default to empty/None, so
    # a caller that omits them (or the pre-this-plan test fixtures) behaves
    # exactly as before -- every new `or` clause below is simply `False`.
    voice_ids: List[str] = field(default_factory=list)
    take_group_id: Optional[str] = None
    sync_group_id: Optional[str] = None
    rgb_mean: Optional[List[float]] = None


# Words this short are too generic (articles, prepositions) to mean
# anything as a topical overlap signal.
_MIN_WORD_LEN = 4

# Same distance family as match.SPAN_RGB_DIST_MAX (per-SPAN rgb_mean); the
# graceful base so semantic grouping NEVER returns all-singletons when the
# metadata is genuinely absent.
SCENE_RGB_DIST_MAX = 0.12


def _label_overlap(a: ShotSceneMeta, b: ShotSceneMeta) -> bool:
    words_a = {w for w in (a.label + " " + a.summary).lower().split() if len(w) >= _MIN_WORD_LEN}
    words_b = {w for w in (b.label + " " + b.summary).lower().split() if len(w) >= _MIN_WORD_LEN}
    return bool(words_a & words_b)


def _rgb_dist(a: List[float], b: List[float]) -> float:
    return sum((a[i] - b[i]) ** 2 for i in range(3)) ** 0.5


def _voice_overlap(a: ShotSceneMeta, b: ShotSceneMeta) -> bool:
    return bool(set(a.voice_ids) & set(b.voice_ids))


def group_shots_semantically(ordered_shots: List[ShotSceneMeta]) -> List[List[int]]:
    """Chain-link adjacent shots into groups (as ordered-list INDICES) using,
    in order of trust: same `file_id` -> same `sync_group_id` -> same
    `take_group_id` -> same `speaker_person` or overlapping `voice_ids` ->
    same `on_camera`-ness AND label/summary overlap (weakest, only ever a
    tiebreak) -> `rgb_close` (the graceful base, see module docstring). A
    shot with no qualifying predecessor starts its own group of 1."""
    groups: List[List[int]] = []
    for i, shot in enumerate(ordered_shots):
        if groups:
            prev = ordered_shots[groups[-1][-1]]
            same_file = bool(shot.file_id) and shot.file_id == prev.file_id
            same_sync = shot.sync_group_id is not None and shot.sync_group_id == prev.sync_group_id
            same_take = shot.take_group_id is not None and shot.take_group_id == prev.take_group_id
            same_speaker = (shot.speaker_person is not None and shot.speaker_person == prev.speaker_person) \
                or _voice_overlap(shot, prev)
            weak_tiebreak = (
                shot.on_camera is not None and shot.on_camera == prev.on_camera
                and _label_overlap(shot, prev)
            )
            rgb_close = (
                shot.rgb_mean is not None and prev.rgb_mean is not None
                and _rgb_dist(shot.rgb_mean, prev.rgb_mean) < SCENE_RGB_DIST_MAX
            )
            if same_file or same_sync or same_take or same_speaker or weak_tiebreak or rgb_close:
                groups[-1].append(i)
                continue
        groups.append([i])
    return groups
