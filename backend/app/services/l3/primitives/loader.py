"""
Load L1/L2 analysis for a set of files into in-memory dataclasses.

Everything downstream (boundaries, units, quality, recipes) operates on these
structures rather than hitting Postgres directly, so the heavy queries run
once per turn and the rest of the pipeline stays pure + unit-testable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row

from app.config import get_settings

logger = logging.getLogger(__name__)


def _pg():
    return psycopg.connect(get_settings().database_url, autocommit=True, row_factory=dict_row)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WordTok:
    start_ms: int
    end_ms: int
    text: str
    is_filler: bool = False
    speaker_id: Optional[str] = None   # per-file diarization label ("S0", ...)


@dataclass
class TranscriptData:
    language: Optional[str]
    text: str
    segments: List[dict]              # raw JSONB segments
    words: List[WordTok] = field(default_factory=list)  # flattened, chronological


@dataclass
class AudioData:
    is_musical: bool = False
    bpm: float = 0.0
    onsets_ms: List[int] = field(default_factory=list)
    silence_intervals: List[dict] = field(default_factory=list)  # {start_ms,end_ms}
    acoustic_tags: List[str] = field(default_factory=list)
    integrated_lufs: Optional[float] = None
    # Prosody / rhythm (cut-timing): emphasis peaks + natural pauses.
    energy_peaks_ms: List[int] = field(default_factory=list)
    pause_map: List[dict] = field(default_factory=list)
    # Cross-file simultaneity (multicam): fixed-hop normalized energy envelope.
    sync_env: List[float] = field(default_factory=list)
    sync_hop_ms: int = 0
    # Dialogue cut-cost grid (0=ideal seam .. 1=forbidden/mid-word) + discrete
    # exact-timestamp seam candidates. "Safe to cut" = 1 - cost.
    dialogue_cut_cost: List[float] = field(default_factory=list)
    dialogue_cut_hop_ms: int = 0
    dialogue_cut_points: List[dict] = field(default_factory=list)
    # Beat/music cut grid (0=on a beat/ideal .. 1=off-beat). Empty if not musical.
    beat_cut_cost: List[float] = field(default_factory=list)
    beat_cut_hop_ms: int = 0
    beat_cut_points: List[dict] = field(default_factory=list)


@dataclass
class MotionData:
    """Video-derived motion cut grids (action + camera/distortion)."""
    hop_ms: int = 0
    action_energy: List[float] = field(default_factory=list)
    camera_motion: List[float] = field(default_factory=list)
    # Camera-motion quality: coherence (rigid global move) + stability (sustained
    # vs jerky). High coherence + high stability + sustained magnitude = a
    # deliberate camera move (cuttable), not the kind of motion to avoid.
    camera_coherence: List[float] = field(default_factory=list)
    camera_stability: List[float] = field(default_factory=list)
    blur: List[float] = field(default_factory=list)
    # Cut-cost channels (0=ideal seam .. 1=avoid).
    action_cut_cost: List[float] = field(default_factory=list)
    camera_cut_cost: List[float] = field(default_factory=list)
    action_points: List[dict] = field(default_factory=list)


@dataclass
class ShotRow:
    shot_id: str
    file_id: str
    shot_index: int
    start_ms: int
    end_ms: int
    motion_magnitude: Optional[float] = None
    motion_dx: Optional[float] = None
    motion_dy: Optional[float] = None
    peak_motion_ms: Optional[int] = None
    blur_min: Optional[float] = None
    focus_score: Optional[float] = None
    brightness: Optional[float] = None
    intra_shot_variance: Optional[float] = None
    framing_scale: Optional[str] = None
    camera_dynamics: Optional[str] = None
    narrative_role: Optional[str] = None
    emotional_valence: Optional[float] = None
    narrative_description: Optional[str] = None
    tracked_character_ids: List[str] = field(default_factory=list)
    keyframe_r2_key: Optional[str] = None

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)


@dataclass
class FileAnalysis:
    file_id: str
    name: str
    r2_key: str
    r2_proxy_key: Optional[str]
    duration_seconds: Optional[float]
    shots: List[ShotRow] = field(default_factory=list)
    transcript: Optional[TranscriptData] = None
    audio: Optional[AudioData] = None
    motion: Optional["MotionData"] = None

    @property
    def has_speech(self) -> bool:
        return bool(self.transcript and self.transcript.words)

    @property
    def is_musical(self) -> bool:
        return bool(self.audio and self.audio.is_musical)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_file_analyses(
    user_id: str,
    file_ids: List[str],
) -> Dict[str, FileAnalysis]:
    """
    Return {file_id -> FileAnalysis} for the given files owned by ``user_id``.

    Files that aren't owned by the user, or that have no shots yet, are simply
    omitted from the result. Three batched queries (files+shots, transcripts,
    audio) keep this to constant round-trips regardless of shot count.
    """
    file_ids = [str(f) for f in dict.fromkeys(file_ids) if f]
    if not file_ids:
        return {}

    out: Dict[str, FileAnalysis] = {}
    with _pg() as conn:
        # --- files ---
        rows = conn.execute(
            """
            select id, name, r2_key, r2_proxy_key, duration_seconds
            from files
            where id = any(%s::uuid[]) and user_id = %s
            """,
            (file_ids, user_id),
        ).fetchall()
        for r in rows:
            fid = str(r["id"])
            out[fid] = FileAnalysis(
                file_id=fid,
                name=r["name"],
                r2_key=r["r2_key"],
                r2_proxy_key=r.get("r2_proxy_key"),
                duration_seconds=r.get("duration_seconds"),
            )
        if not out:
            return {}

        owned_ids = list(out.keys())

        # --- shots ---
        shot_rows = conn.execute(
            """
            select id, file_id, shot_index, start_ms, end_ms,
                   motion_magnitude, motion_dx, motion_dy, peak_motion_ms,
                   blur_min, focus_score,
                   brightness, intra_shot_variance, framing_scale,
                   camera_dynamics, narrative_role, emotional_valence,
                   narrative_description, tracked_character_ids, keyframe_r2_key
            from shots
            where file_id = any(%s::uuid[])
            order by file_id, shot_index
            """,
            (owned_ids,),
        ).fetchall()
        for r in shot_rows:
            fid = str(r["file_id"])
            fa = out.get(fid)
            if fa is None:
                continue
            fa.shots.append(
                ShotRow(
                    shot_id=str(r["id"]),
                    file_id=fid,
                    shot_index=int(r["shot_index"]),
                    start_ms=int(r["start_ms"]),
                    end_ms=int(r["end_ms"]),
                    motion_magnitude=_f(r.get("motion_magnitude")),
                    motion_dx=_f(r.get("motion_dx")),
                    motion_dy=_f(r.get("motion_dy")),
                    peak_motion_ms=_i(r.get("peak_motion_ms")),
                    blur_min=_f(r.get("blur_min")),
                    focus_score=_f(r.get("focus_score")),
                    brightness=_f(r.get("brightness")),
                    intra_shot_variance=_f(r.get("intra_shot_variance")),
                    framing_scale=r.get("framing_scale"),
                    camera_dynamics=r.get("camera_dynamics"),
                    narrative_role=r.get("narrative_role"),
                    emotional_valence=_f(r.get("emotional_valence")),
                    narrative_description=r.get("narrative_description"),
                    tracked_character_ids=[str(x) for x in (r.get("tracked_character_ids") or [])],
                    keyframe_r2_key=r.get("keyframe_r2_key"),
                )
            )

        # --- transcripts ---
        tx_rows = conn.execute(
            "select file_id, language, text, segments from transcripts where file_id = any(%s::uuid[])",
            (owned_ids,),
        ).fetchall()
        for r in tx_rows:
            fid = str(r["file_id"])
            fa = out.get(fid)
            if fa is None:
                continue
            segments = r.get("segments") or []
            fa.transcript = TranscriptData(
                language=r.get("language"),
                text=r.get("text") or "",
                segments=segments,
                words=_flatten_words(segments),
            )

        # --- audio features ---
        af_rows = conn.execute(
            """
            select file_id, is_musical, bpm, onsets_ms, silence_intervals,
                   acoustic_tags, integrated_lufs, energy_peaks_ms, pause_map,
                   sync_env, sync_hop_ms,
                   dialogue_cut_cost, dialogue_cut_hop_ms, dialogue_cut_points,
                   beat_cut_cost, beat_cut_hop_ms, beat_cut_points
            from audio_features where file_id = any(%s::uuid[])
            """,
            (owned_ids,),
        ).fetchall()
        for r in af_rows:
            fid = str(r["file_id"])
            fa = out.get(fid)
            if fa is None:
                continue
            fa.audio = AudioData(
                is_musical=bool(r.get("is_musical")),
                bpm=_f(r.get("bpm")) or 0.0,
                onsets_ms=[int(x) for x in (r.get("onsets_ms") or [])],
                silence_intervals=list(r.get("silence_intervals") or []),
                acoustic_tags=[str(x) for x in (r.get("acoustic_tags") or [])],
                integrated_lufs=_f(r.get("integrated_lufs")),
                energy_peaks_ms=[int(x) for x in (r.get("energy_peaks_ms") or [])],
                pause_map=list(r.get("pause_map") or []),
                sync_env=[float(x) for x in (r.get("sync_env") or [])],
                sync_hop_ms=int(r.get("sync_hop_ms") or 0),
                dialogue_cut_cost=[float(x) for x in (r.get("dialogue_cut_cost") or [])],
                dialogue_cut_hop_ms=int(r.get("dialogue_cut_hop_ms") or 0),
                dialogue_cut_points=list(r.get("dialogue_cut_points") or []),
                beat_cut_cost=[float(x) for x in (r.get("beat_cut_cost") or [])],
                beat_cut_hop_ms=int(r.get("beat_cut_hop_ms") or 0),
                beat_cut_points=list(r.get("beat_cut_points") or []),
            )

        # --- motion dynamics (action + camera/distortion) ---
        md_rows = conn.execute(
            """
            select file_id, hop_ms, action_energy, camera_motion, camera_coherence,
                   camera_stability, blur, action_cut_cost, camera_cut_cost,
                   action_points
            from motion_dynamics where file_id = any(%s::uuid[])
            """,
            (owned_ids,),
        ).fetchall()
        for r in md_rows:
            fid = str(r["file_id"])
            fa = out.get(fid)
            if fa is None:
                continue
            fa.motion = MotionData(
                hop_ms=int(r.get("hop_ms") or 0),
                action_energy=[float(x) for x in (r.get("action_energy") or [])],
                camera_motion=[float(x) for x in (r.get("camera_motion") or [])],
                camera_coherence=[float(x) for x in (r.get("camera_coherence") or [])],
                camera_stability=[float(x) for x in (r.get("camera_stability") or [])],
                blur=[float(x) for x in (r.get("blur") or [])],
                action_cut_cost=[float(x) for x in (r.get("action_cut_cost") or [])],
                camera_cut_cost=[float(x) for x in (r.get("camera_cut_cost") or [])],
                action_points=list(r.get("action_points") or []),
            )

    return out


def _flatten_words(segments: List[dict]) -> List[WordTok]:
    words: List[WordTok] = []
    for seg in segments or []:
        for w in seg.get("words") or []:
            try:
                spk = w.get("speaker")
                words.append(
                    WordTok(
                        start_ms=int(w.get("start_ms", 0)),
                        end_ms=int(w.get("end_ms", 0)),
                        text=str(w.get("text", "")).strip(),
                        is_filler=bool(w.get("is_filler", False)),
                        speaker_id=str(spk) if spk else None,
                    )
                )
            except (TypeError, ValueError):
                continue
    words.sort(key=lambda x: (x.start_ms, x.end_ms))
    return words


def _f(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _i(v: Any) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None
