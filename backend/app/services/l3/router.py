"""
Content router: profile footage by modality so recipes adapt and the planner
can suggest a sensible default style (and split a mixed edit into sections).

Modalities
  TALKING  - speech-dominant (interviews, talking-head, vlogs)
  ACTION   - high motion, little speech (sports, b-roll movement)
  SCENIC   - low motion, little speech (landscapes, slow establishing)
  MUSICAL  - the file has a usable music bed / beat grid
  MIXED    - none clearly dominates

The profile is computed deterministically from the loaded analysis + the
editorial units. It is advisory: the LLM planner can override the suggestion,
but the suggestion makes the common case one-click-right.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from app.services.l3.primitives.loader import FileAnalysis
from app.services.l3.primitives.units import EditUnit

TALKING = "talking"
ACTION = "action"
SCENIC = "scenic"
MUSICAL = "musical"
MIXED = "mixed"

# Modality -> default recipe style key (must match recipes registry keys).
MODALITY_DEFAULT_STYLE = {
    TALKING: "talking_head",
    ACTION: "highlight",
    SCENIC: "cinematic_broll",
    MUSICAL: "beat_sync",
    MIXED: "vlog",
}

# Tuning thresholds (fractions of file duration / normalized motion).
SPEECH_TALKING_FRAC = 0.40
MOTION_ACTION_REF = 4.0     # mean motion_magnitude above this => action-ish
MOTION_SCENIC_REF = 1.5     # mean motion_magnitude below this => scenic-ish


@dataclass
class FileProfile:
    file_id: str
    file_name: str
    dominant_modality: str
    speech_fraction: float
    mean_motion: float
    has_beats: bool
    duration_ms: int
    fractions: Dict[str, float] = field(default_factory=dict)


@dataclass
class FootageProfile:
    per_file: Dict[str, FileProfile]
    dominant_modality: str
    has_musical: bool
    total_duration_ms: int
    suggested_style: str

    def file_modality(self, file_id: str) -> str:
        fp = self.per_file.get(file_id)
        return fp.dominant_modality if fp else MIXED


def profile_footage(
    analyses: Dict[str, FileAnalysis],
    units_by_file: Dict[str, List[EditUnit]],
) -> FootageProfile:
    per_file: Dict[str, FileProfile] = {}
    total_dur = 0
    modality_dur: Dict[str, int] = {TALKING: 0, ACTION: 0, SCENIC: 0, MUSICAL: 0, MIXED: 0}
    has_musical = False

    for fid, fa in analyses.items():
        fp = _profile_one(fa, units_by_file.get(fid, []))
        per_file[fid] = fp
        total_dur += fp.duration_ms
        modality_dur[fp.dominant_modality] = modality_dur.get(fp.dominant_modality, 0) + fp.duration_ms
        if fp.has_beats:
            has_musical = True

    dominant = max(modality_dur, key=lambda k: modality_dur[k]) if total_dur > 0 else MIXED
    # If multiple modalities each hold a meaningful share, call the corpus MIXED.
    if total_dur > 0:
        shares = {k: v / total_dur for k, v in modality_dur.items()}
        big = [k for k, v in shares.items() if v >= 0.30]
        if len(big) >= 2:
            dominant = MIXED

    suggested = MODALITY_DEFAULT_STYLE.get(dominant, "vlog")
    return FootageProfile(
        per_file=per_file,
        dominant_modality=dominant,
        has_musical=has_musical,
        total_duration_ms=total_dur,
        suggested_style=suggested,
    )


def _profile_one(fa: FileAnalysis, units: List[EditUnit]) -> FileProfile:
    duration_ms = int((fa.duration_seconds or 0) * 1000)
    if duration_ms <= 0 and fa.shots:
        duration_ms = max(s.end_ms for s in fa.shots)

    speech_ms = sum(u.duration_ms for u in units if u.modality == "speech")
    speech_fraction = (speech_ms / duration_ms) if duration_ms > 0 else 0.0

    motions = [s.motion_magnitude for s in fa.shots if s.motion_magnitude is not None]
    mean_motion = sum(motions) / len(motions) if motions else 0.0

    has_beats = bool(fa.audio and fa.audio.is_musical and fa.audio.onsets_ms)

    # Decide dominant modality.
    if has_beats and speech_fraction < SPEECH_TALKING_FRAC:
        dominant = MUSICAL
    elif speech_fraction >= SPEECH_TALKING_FRAC:
        dominant = TALKING
    elif mean_motion >= MOTION_ACTION_REF:
        dominant = ACTION
    elif mean_motion <= MOTION_SCENIC_REF:
        dominant = SCENIC
    else:
        dominant = MIXED

    fractions = {
        TALKING: round(speech_fraction, 3),
        ACTION: round(min(1.0, mean_motion / MOTION_ACTION_REF), 3),
        MUSICAL: 1.0 if has_beats else 0.0,
    }

    return FileProfile(
        file_id=fa.file_id,
        file_name=fa.name,
        dominant_modality=dominant,
        speech_fraction=round(speech_fraction, 3),
        mean_motion=round(mean_motion, 3),
        has_beats=has_beats,
        duration_ms=duration_ms,
        fractions=fractions,
    )
