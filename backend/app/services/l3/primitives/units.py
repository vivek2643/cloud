"""
Editorial units: the primitive the editor actually assembles with.

Instead of handing recipes raw shots (a visual-only primitive that knows
nothing about sentences), we build two kinds of unit, both snapped to real
boundaries:

  - SpeechUnit: one utterance/sentence, trimmed to its first/last non-filler
    word. Carries the exact spoken text.
  - VisualUnit: one shot (a visual beat), carrying motion/quality/L2 tags.

A talking edit assembles SpeechUnits (with VisualUnits as cutaways); a montage
assembles VisualUnits. Units are computed deterministically from the loaded
analysis -- no DB access here, so this stays pure + testable.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Optional

from app.services.l3.primitives.boundaries import (
    SENTENCE_END_CHARS,
    SENTENCE_GAP_MS,
)
from app.services.l3.primitives.loader import FileAnalysis, ShotRow, WordTok
from app.services.l3.primitives.quality import (
    energy_score,
    speech_quality,
    take_quality,
    visual_quality,
)

# Drop utterances shorter than this -- too short to be a usable clip.
MIN_SPEECH_UNIT_MS = 600
MIN_VISUAL_UNIT_MS = 500


@dataclass
class EditUnit:
    id: str
    file_id: str
    file_name: str
    modality: str            # "speech" | "visual"
    in_ms: int
    out_ms: int
    quality: float
    lane: str = "spine"      # "spine" (primary narrative) | "coverage" (overlay/b-roll)
    text: str = ""           # spoken text (speech units)
    shot_id: Optional[str] = None
    shot_ids: List[str] = field(default_factory=list)
    motion: float = 0.0
    motion_dx: float = 0.0   # dominant screen-space motion direction (px/frame)
    motion_dy: float = 0.0
    valence: Optional[float] = None
    narrative_role: Optional[str] = None
    framing_scale: Optional[str] = None
    characters: List[str] = field(default_factory=list)
    speaker_id: Optional[str] = None   # reserved for future diarization
    keyframe_r2_key: Optional[str] = None

    @property
    def duration_ms(self) -> int:
        return max(0, self.out_ms - self.in_ms)


def build_units(fa: FileAnalysis) -> List[EditUnit]:
    """All editorial units for one file: speech utterances + visual shots."""
    units: List[EditUnit] = []
    units.extend(_build_speech_units(fa))
    units.extend(_build_visual_units(fa))
    units.sort(key=lambda u: (u.in_ms, u.out_ms))
    return units


def assign_lanes(units: List[EditUnit]) -> None:
    """Corpus-level spine/coverage assignment (in place).

    Speech is the spine and visual units are coverage (b-roll/cutaways) whenever
    the corpus has any speech. For a purely visual corpus (montage, no speech),
    the visual units ARE the timeline, so they become the spine.
    """
    has_speech = any(u.modality == "speech" for u in units)
    if has_speech:
        return
    for u in units:
        if u.modality == "visual":
            u.lane = "spine"


# ---------------------------------------------------------------------------
# Speech units
# ---------------------------------------------------------------------------

def _split_utterances(words: List[WordTok]) -> List[List[WordTok]]:
    """Group consecutive words into utterances on big gaps / sentence-final
    punctuation."""
    groups: List[List[WordTok]] = []
    cur: List[WordTok] = []
    for i, w in enumerate(words):
        cur.append(w)
        end_here = w.text[-1:] in SENTENCE_END_CHARS if w.text else False
        if i < len(words) - 1:
            gap = words[i + 1].start_ms - w.end_ms
            if gap >= SENTENCE_GAP_MS:
                end_here = True
        if end_here:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _build_speech_units(fa: FileAnalysis) -> List[EditUnit]:
    tx = fa.transcript
    if not tx or not tx.words:
        return []
    out: List[EditUnit] = []
    for group in _split_utterances(tx.words):
        # Trim leading/trailing fillers for the clip bounds.
        real = [w for w in group if not w.is_filler and w.text]
        if not real:
            continue
        in_ms = real[0].start_ms
        out_ms = real[-1].end_ms
        if out_ms - in_ms < MIN_SPEECH_UNIT_MS:
            continue
        text = " ".join(w.text for w in real).strip()
        q = speech_quality(group)
        speaker_id = _majority_speaker(real)
        shot = _shot_covering(fa.shots, (in_ms + out_ms) // 2)
        out.append(
            EditUnit(
                id=str(uuid.uuid4()),
                file_id=fa.file_id,
                file_name=fa.name,
                modality="speech",
                in_ms=in_ms,
                out_ms=out_ms,
                quality=q,
                lane="spine",
                text=text,
                shot_id=shot.shot_id if shot else None,
                shot_ids=[shot.shot_id] if shot else [],
                motion=shot.motion_magnitude or 0.0 if shot else 0.0,
                motion_dx=(shot.motion_dx or 0.0) if shot else 0.0,
                motion_dy=(shot.motion_dy or 0.0) if shot else 0.0,
                valence=shot.emotional_valence if shot else None,
                narrative_role=shot.narrative_role if shot else None,
                framing_scale=shot.framing_scale if shot else None,
                characters=list(shot.tracked_character_ids) if shot else [],
                speaker_id=speaker_id,
                keyframe_r2_key=shot.keyframe_r2_key if shot else None,
            )
        )
    return out


def _majority_speaker(words: List[WordTok]) -> Optional[str]:
    """The speaker label covering the most words in an utterance (diarization
    is per-word; an utterance is almost always one speaker)."""
    counts: dict = {}
    for w in words:
        if w.speaker_id:
            counts[w.speaker_id] = counts.get(w.speaker_id, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# Visual units
# ---------------------------------------------------------------------------

def _build_visual_units(fa: FileAnalysis) -> List[EditUnit]:
    out: List[EditUnit] = []
    for s in fa.shots:
        if s.duration_ms < MIN_VISUAL_UNIT_MS:
            continue
        vq = visual_quality(s)
        eng = energy_score(s)
        q = take_quality(visual=vq, energy=eng, valence=s.emotional_valence,
                         weights={"visual": 0.55, "energy": 0.3, "valence": 0.15, "speech": 0.0})
        out.append(
            EditUnit(
                id=str(uuid.uuid4()),
                file_id=fa.file_id,
                file_name=fa.name,
                modality="visual",
                in_ms=s.start_ms,
                out_ms=s.end_ms,
                quality=q,
                lane="coverage",
                text=(s.narrative_description or "").strip(),
                shot_id=s.shot_id,
                shot_ids=[s.shot_id],
                motion=s.motion_magnitude or 0.0,
                motion_dx=s.motion_dx or 0.0,
                motion_dy=s.motion_dy or 0.0,
                valence=s.emotional_valence,
                narrative_role=s.narrative_role,
                framing_scale=s.framing_scale,
                characters=list(s.tracked_character_ids),
                keyframe_r2_key=s.keyframe_r2_key,
            )
        )
    return out


def _shot_covering(shots: List[ShotRow], ms: int) -> Optional[ShotRow]:
    for s in shots:
        if s.start_ms <= ms <= s.end_ms:
            return s
    return None
