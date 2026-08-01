"""
speech_cuts_pipeline.plan.md section 6 -- Stage 1: load the per-file speech
signals already persisted by L1 (words, RMS, silence, face tracks). Direct
minimal SELECTs (mirrors app.services.vcut.spans' own pattern), keeping vcut
isolated from L3 business logic (principle 4/7).

Whisper per-word confidence (`probability`) is captured by L1's transcript
stage going forward, but is NOT yet backfilled on older transcripts, so a
Word built here may carry ``probability=None``. Rather than carry a neutral
fallback for a not-yet-universal signal, delivery.py simply does NOT score
ASR confidence in v1 -- the field is read here for the day it's universal and
the term is re-added cleanly (see delivery.py / params.py W_ASR notes).

Noise gate (params.py): a Whisper segment the model itself judges non-speech
(`no_speech_prob` high) or decoded with very low `avg_logprob` is a
hallucination over ambient noise -- its words are dropped WHOLE here, before
they can seed a beat. Fail-open when those signals are absent (older
transcripts), so no real speech is ever lost to a missing signal.

The per-beat energy gate's floor reference (`rms_floor_db`, consumed by
delivery.is_voiced_beat) is calibrated from this file's OWN detected silence
frames (the true noise floor) when there's enough silence to measure, falling
back to a blind 15th-percentile otherwise (speech_noise_gate.plan.md
Improvement B) -- see `_energy_refs`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from app.services.vcut.speech.params import (
    RMS_FLOOR_PCTL, RMS_VOICED_PCTL, SEG_MIN_AVG_LOGPROB, SEG_NO_SPEECH_MAX,
    SILENCE_FLOOR_MIN_FRAMES,
)


@dataclass
class Word:
    idx: int
    start_ms: int
    end_ms: int
    text: str
    is_filler: bool = False
    speaker: Optional[str] = None
    probability: Optional[float] = None  # always None today -- see module docstring


@dataclass
class FaceTrackLite:
    track_id: int
    best_crop_ms: int
    speaking: List[Tuple[int, int, float]] = field(default_factory=list)  # (start_ms, end_ms, score)


@dataclass
class FileSpeechInputs:
    file_id: str
    duration_ms: int
    words: List[Word] = field(default_factory=list)
    rms_db: List[float] = field(default_factory=list)
    rms_hop_ms: int = 0
    silences: List[Tuple[int, int]] = field(default_factory=list)
    face_tracks: List[FaceTrackLite] = field(default_factory=list)
    # Per-file energy references for the beat-level noise gate (orchestrate.py):
    # a "typical voiced/speech" level and a "quiet/noise floor", read off this
    # file's own rms_db distribution. 0.0/0.0 => not enough data to gate.
    rms_voiced_db: float = 0.0
    rms_floor_db: float = 0.0


def _pg_conn():
    from app.services import db
    return db.connection_dict_row()


def _segment_is_non_speech(seg: dict) -> bool:
    """A Whisper segment that is a hallucination over non-speech audio, per
    Whisper's OWN self-confidence. Fail-open (False) when the signals aren't
    present -- older transcripts predate this capture, and a missing signal
    must never drop real speech."""
    nsp = seg.get("no_speech_prob")
    if nsp is None:
        return False  # signal not captured -> cannot judge -> keep
    if float(nsp) >= SEG_NO_SPEECH_MAX:
        return True
    alp = seg.get("avg_logprob")
    if alp is not None and float(alp) < SEG_MIN_AVG_LOGPROB:
        return True
    return False


def _percentile(sorted_vals: List[float], pct: float) -> float:
    if not sorted_vals:
        return 0.0
    i = int(round((pct / 100.0) * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(len(sorted_vals) - 1, i))]


def _median(values: List[float]) -> float:
    xs = sorted(values)
    n = len(xs)
    if n == 0:
        return 0.0
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def _energy_refs(rms_db: List[float], hop_ms: int, silences: List[Tuple[int, int]]) -> Tuple[float, float]:
    """(voiced_ref_db, floor_ref_db) from a file's rms_db -- the beat gate's
    self-calibration. voiced_ref = 75th pct of all rms (typical speech
    level). floor_ref = MEDIAN rms over frames inside detected silence (the
    file's true noise floor, speech_noise_gate.plan.md Improvement B) when
    there is enough silence to measure; otherwise the 15th-pct fallback
    (today's original behavior). 0.0/0.0 when there isn't enough rms data to
    gate at all (is_voiced_beat then no-ops via MIN_VOICED_SPREAD_DB)."""
    vals = [float(v) for v in rms_db if v is not None]
    if len(vals) < 20:
        return 0.0, 0.0
    s = sorted(vals)
    voiced = _percentile(s, RMS_VOICED_PCTL)
    floor = None
    if hop_ms and hop_ms > 0 and silences:
        sil_frames = []
        for a, b in silences:
            lo, hi = a // hop_ms, min(len(rms_db) - 1, b // hop_ms)
            sil_frames.extend(float(rms_db[i]) for i in range(max(0, lo), hi + 1)
                              if rms_db[i] is not None)
        if len(sil_frames) >= SILENCE_FLOOR_MIN_FRAMES:
            floor = _median(sil_frames)
    if floor is None:
        floor = _percentile(s, RMS_FLOOR_PCTL)
    return voiced, floor


def _words_for_file(file_id: str) -> List[Word]:
    """Every word across every segment, flattened and sequentially indexed
    -- the numbering segment_llm.py shows the model and word_span refers to.
    Filler words are KEPT (not dropped) so the model can see natural speech
    (fillers/restarts) when judging fluency/flags; is_filler still rides
    along per word for delivery.py's own hesitation/pace math."""
    with _pg_conn() as conn:
        row = conn.execute(
            "select segments from transcripts where file_id = %s", (file_id,)
        ).fetchone()
    if not row or not row["segments"]:
        return []
    words: List[Word] = []
    idx = 0
    for seg in row["segments"]:
        if not isinstance(seg, dict):
            continue
        if _segment_is_non_speech(seg):
            continue  # hallucinated non-speech segment -> its words never enter
        for w in seg.get("words") or []:
            if not isinstance(w, dict) or "start_ms" not in w or "end_ms" not in w:
                continue
            words.append(Word(
                idx=idx, start_ms=int(w["start_ms"]), end_ms=int(w["end_ms"]),
                text=w.get("text") or "", is_filler=bool(w.get("is_filler")),
                speaker=w.get("speaker"), probability=w.get("probability"),
            ))
            idx += 1
    return words


def _rms_for_file(file_id: str) -> Tuple[List[float], int]:
    with _pg_conn() as conn:
        row = conn.execute(
            "select rms_db, prosody_hop_ms from audio_features where file_id = %s", (file_id,)
        ).fetchone()
    if not row:
        return [], 0
    return list(row["rms_db"] or []), int(row["prosody_hop_ms"] or 0)


def _silences_for_file(file_id: str) -> List[Tuple[int, int]]:
    with _pg_conn() as conn:
        row = conn.execute(
            "select silence_intervals from audio_features where file_id = %s", (file_id,)
        ).fetchone()
    if not row or not row["silence_intervals"]:
        return []
    return [
        (int(s["start_ms"]), int(s["end_ms"]))
        for s in row["silence_intervals"]
        if isinstance(s, dict) and "start_ms" in s and "end_ms" in s
    ]


def _face_tracks_for_file(file_id: str) -> List[FaceTrackLite]:
    with _pg_conn() as conn:
        row = conn.execute(
            "select tracks from face_tracks where file_id = %s", (file_id,)
        ).fetchone()
    if not row or not row["tracks"]:
        return []
    out: List[FaceTrackLite] = []
    for t in row["tracks"]:
        if not isinstance(t, dict):
            continue
        speaking = [
            (int(s["start_ms"]), int(s["end_ms"]), float(s.get("score") or 0.0))
            for s in (t.get("speaking") or []) if isinstance(s, dict) and "start_ms" in s and "end_ms" in s
        ]
        out.append(FaceTrackLite(
            track_id=int(t.get("track_id") or 0), best_crop_ms=int(t.get("best_crop_ms") or 0),
            speaking=speaking,
        ))
    return out


def load_face_tracks_for_files(file_ids: List[str]) -> Dict[str, List[FaceTrackLite]]:
    """{file_id: [FaceTrackLite, ...]} for ANY file, independent of whether
    it has a transcript -- an outlook angle camera can lack usable audio
    (no transcript, no take instance of its own) yet still need on-camera
    detection when store.py fans a winning take out to it."""
    return {fid: _face_tracks_for_file(fid) for fid in file_ids}


def load_file_inputs(file_id: str, duration_ms: int) -> Optional[FileSpeechInputs]:
    """None when the file has no transcript at all -- it contributes no
    speech cuts (the non-speech channel's domain, section 6)."""
    words = _words_for_file(file_id)
    if not words:
        return None
    rms_db, rms_hop_ms = _rms_for_file(file_id)
    silences = _silences_for_file(file_id)
    voiced_db, floor_db = _energy_refs(rms_db, rms_hop_ms, silences)
    return FileSpeechInputs(
        file_id=file_id, duration_ms=duration_ms, words=words,
        rms_db=rms_db, rms_hop_ms=rms_hop_ms,
        silences=silences, face_tracks=_face_tracks_for_file(file_id),
        rms_voiced_db=voiced_db, rms_floor_db=floor_db,
    )


def load_project_inputs(file_rows: List[Tuple[str, str, int]]) -> Dict[str, FileSpeechInputs]:
    """``file_rows``: (file_id, filename, duration_ms). Returns {file_id:
    FileSpeechInputs} for every file WITH a transcript -- a file with none
    is simply absent from the result."""
    out: Dict[str, FileSpeechInputs] = {}
    for file_id, _name, duration_ms in file_rows:
        fi = load_file_inputs(file_id, duration_ms)
        if fi is not None:
            out[file_id] = fi
    return out
