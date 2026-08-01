"""
speech_cuts_pipeline.plan.md section 12 -- Stage 7: build kind='speech'
CutRecords (word_span/take/sync/audio/visual) for insertion via the
EXISTING ingest_store.insert_cut_records. Reuses post.CutRecord and vcut.
store's own flat (never-speed-ramped) pace helper -- speech always plays at
native speed, matching the old pipeline's own contract for kind="speech"
(post.compute_pace_envelope: "Speech always plays at native speed").

Audio-offset semantics (verified against l3.sync.av_couple.authoritative_for,
NOT imported -- read for reference only): for an outlook angle's cut,
audio_offset_ms = angle_member.offset_ms - hero_member.offset_ms (the GLOBAL
group-level delta, taken directly from sync_group_members per section 7's
own wording -- no per-cut cross-correlation refinement, that's L3-specific
machinery this module deliberately doesn't reuse).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from app.services.l3.post import CutRecord
from app.services.vcut.speech.inputs import FaceTrackLite
from app.services.vcut.speech.outlooks import SyncGroup, angle_ms, video_angle_files
from app.services.vcut.speech.segment_llm import _BeatOut
from app.services.vcut.store import _pace_for


@dataclass
class ResolvedBeat:
    """One beat, already boundary-snapped and take-scored -- the join point
    between select.py's winner decision and this module's CutRecord
    assembly."""
    beat: _BeatOut
    in_ms: int
    out_ms: int
    take_group_key: str     # any stable string shared by every beat in the cluster
    take_role: str           # "winner" | "take" (cut_records.take_role's real check
                              # constraint value for a non-winning candidate; a junk
                              # beat is always "take", never "winner")
    speech_quality: float    # the gated final score (select.TakeGroupResult.finals[...])
    # speech_noise_gate.plan.md Improvement C: a beat the noise gate caught
    # (LLM- or energy-flagged) is marked junk instead of silently dropped --
    # auditable/recoverable, hidden from the default cuts view same as video
    # junk. junk_reason: "non_speech_llm" | "non_speech_energy" | "" (not junk).
    junk: bool = False
    junk_reason: str = ""


@dataclass
class VisualFields:
    on_camera: Optional[bool] = None
    framing: Dict[str, Any] = field(default_factory=dict)
    angle_type: str = ""
    scene_specifics: Dict[str, Any] = field(default_factory=dict)


def cut_key(beat_id: str, file_id: str) -> str:
    """The join key between frames.py's visual analysis output and this
    module's CutRecord assembly -- one beat can produce several cuts (one
    per outlook angle), each potentially answered separately."""
    return f"{beat_id}::{file_id}"


def _beat_midpoint_ms(in_ms: int, out_ms: int) -> int:
    return (in_ms + out_ms) // 2


def hero_ts_for_span(face_tracks: List[FaceTrackLite], in_ms: int, out_ms: int) -> int:
    """The clearest, most-confidently-speaking face crop overlapping
    [in_ms, out_ms] -- falls back to the beat's own midpoint when no face
    data overlaps at all (true voiceover, or no face_tracks for this file)."""
    best: Optional[Tuple[float, int]] = None
    for t in face_tracks:
        for s, e, score in t.speaking:
            if e > in_ms and s < out_ms and (best is None or score > best[0]):
                best = (score, t.best_crop_ms)
    if best is None:
        return _beat_midpoint_ms(in_ms, out_ms)
    crop_ms = best[1]
    return crop_ms if in_ms <= crop_ms <= out_ms else _beat_midpoint_ms(in_ms, out_ms)


def is_on_camera(face_tracks: List[FaceTrackLite], in_ms: int, out_ms: int) -> bool:
    """True iff some face track has a speaking interval (any positive
    score) overlapping [in_ms, out_ms] -- the basis for both hero_ts_ms and
    frames.py's own "which cuts need a frame" decision."""
    return any(
        e > in_ms and s < out_ms and score > 0
        for t in face_tracks for s, e, score in t.speaking
    )


def _short_label(gist: str, max_words: int = 8) -> str:
    words = (gist or "").split()
    label = " ".join(words[:max_words])
    return label or "speech"


@dataclass
class ExpandedCut:
    """One (beat, angle_file, ms-window) this run will actually emit as a
    CutRecord -- the fan-out decision, factored out of build_speech_cut_
    records so frames.py can decide which cuts need a frame BEFORE this
    module builds the final records (both need the identical fan-out)."""
    resolved_beat: ResolvedBeat
    file_id: str
    in_ms: int
    out_ms: int
    sync_group_id: Optional[str]
    audio_file_id: str
    audio_offset_ms: int
    audio_align_confidence: Optional[float]
    # When a winning take fans out across a synced group, every emitted angle
    # is an equal camera OUTLOOK of the same audio -- not a retake to choose
    # among -- so it carries take_role="outlook" (never "winner"). This keeps
    # the frontend "Best takes" filter collapsing only real retakes while every
    # outlook angle stays visible. None -> use the beat's own take_role (an
    # ungrouped winner stays "winner", a losing take stays "take").
    take_role_override: Optional[str] = None


def expand_outlook_cuts(
    resolved_beats: List[ResolvedBeat], sync_groups: Dict[str, SyncGroup],
) -> List[ExpandedCut]:
    """A winning take whose take-instance file is a synced group's
    authoritative (hero) file fans out to one ExpandedCut per video-angle
    member (section 12); everything else (an ungrouped file, or an
    alternate -- alternates never fan out even inside a sync group) is
    exactly one ExpandedCut on its own take-instance file, no audio
    routing."""
    out: List[ExpandedCut] = []
    for rb in resolved_beats:
        beat = rb.beat
        group = sync_groups.get(_group_id_for_hero(sync_groups, beat.file_id))
        if rb.take_role == "winner" and group is not None and group.authoritative_file_id == beat.file_id:
            hero_member = group.members[group.authoritative_file_id]
            for angle_file_id in video_angle_files(group):
                angle_member = group.members[angle_file_id]
                a_in = angle_ms(rb.in_ms, hero_member.offset_ms, angle_member.offset_ms)
                a_out = angle_ms(rb.out_ms, hero_member.offset_ms, angle_member.offset_ms)
                out.append(ExpandedCut(
                    resolved_beat=rb, file_id=angle_file_id, in_ms=a_in, out_ms=a_out,
                    sync_group_id=group.group_id, audio_file_id=group.authoritative_file_id,
                    audio_offset_ms=angle_member.offset_ms - hero_member.offset_ms,
                    audio_align_confidence=angle_member.confidence,
                    take_role_override="outlook",
                ))
        else:
            out.append(ExpandedCut(
                resolved_beat=rb, file_id=beat.file_id, in_ms=rb.in_ms, out_ms=rb.out_ms,
                sync_group_id=None, audio_file_id="", audio_offset_ms=0, audio_align_confidence=None,
            ))
    return out


def _build_record(
    file_id: str, in_ms: int, out_ms: int, rb: ResolvedBeat,
    face_tracks: List[FaceTrackLite], visual: Optional[VisualFields],
    sync_group_id: Optional[str], audio_file_id: str, audio_offset_ms: int,
    audio_align_confidence: Optional[float], take_role: str,
    junk: bool = False, junk_reason: str = "",
) -> Tuple[CutRecord, Optional[Dict[str, Any]]]:
    """Returns (record, scene_specifics_or_None). CutRecord itself has NO
    scene_specifics field (verified: insert_cut_records' own INSERT column
    list omits it entirely) -- it's only ever set via a separate post-insert
    update_cut_scene_specifics call, same contract as vcut's video pipeline
    (pass2.py/vcut_enrich). angle_type rides inside scene_specifics
    (section 15's own instruction), never as its own column."""
    v = visual or VisualFields()
    duration_ms = max(1, out_ms - in_ms)
    pace = _pace_for(duration_ms)
    gist = rb.beat.gist

    record = CutRecord(
        file_id=file_id, src_in_ms=in_ms, src_out_ms=out_ms,
        kind="speech", word_span=tuple(rb.beat.word_span), atom_ids=None,
        label=_short_label(gist), summary=gist,
        on_camera=v.on_camera, junk=junk, junk_reason=junk_reason,
        framing=v.framing, look={}, caption_zones=[],
        hero_ts_ms=hero_ts_for_span(face_tracks, in_ms, out_ms),
        pace=pace, take_group_id=rb.take_group_key, take_role=take_role,
        channel="said", speech_quality=rb.speech_quality, total_quality=rb.speech_quality,
        sync_group_id=sync_group_id, audio_file_id=audio_file_id,
        audio_offset_ms=audio_offset_ms, audio_align_confidence=audio_align_confidence,
    )

    specifics: Dict[str, Any] = dict(v.scene_specifics)
    if v.angle_type:
        specifics["angle_type"] = v.angle_type
    return record, (specifics or None)


def build_speech_cut_records(
    resolved_beats: List[ResolvedBeat],
    sync_groups: Dict[str, SyncGroup],
    face_tracks_by_file: Dict[str, List[FaceTrackLite]],
    visual_by_key: Dict[str, VisualFields],
) -> Tuple[List[CutRecord], List[Optional[Dict[str, Any]]]]:
    """(records, scene_specifics_per_record) -- the second list is parallel
    (same order/length) so a caller zips it against insert_cut_records'
    returned ids to write scene_specifics afterward.

    One CutRecord per resolved beat, or one PER OUTLOOK ANGLE for a winning
    take whose take-instance file is a synced group's authoritative (hero)
    file (section 12). Alternates never fan out -- exactly one cut, on
    their own take-instance file, no audio routing.

    ``visual_by_key``: {cut_key(beat.id, file_id): VisualFields} -- frames.py's
    output, winners (+ their angles) only; an alternate or any cut missing
    from this map gets VisualFields()'s defaults (on_camera=None, no scene_
    specifics) -- cost-first (section 6): frame analysis only pays for what's
    actually shown."""
    records: List[CutRecord] = []
    specifics: List[Optional[Dict[str, Any]]] = []
    for ec in expand_outlook_cuts(resolved_beats, sync_groups):
        record, spec = _build_record(
            ec.file_id, ec.in_ms, ec.out_ms, ec.resolved_beat,
            face_tracks_by_file.get(ec.file_id, []),
            visual_by_key.get(cut_key(ec.resolved_beat.beat.id, ec.file_id)),
            sync_group_id=ec.sync_group_id, audio_file_id=ec.audio_file_id,
            audio_offset_ms=ec.audio_offset_ms, audio_align_confidence=ec.audio_align_confidence,
            take_role=ec.take_role_override or ec.resolved_beat.take_role,
            # A winner is never junk by construction (junk beats are always
            # take_role="take", never fanned out to outlook angles) -- so an
            # outlook's take_role_override branch here always carries junk=False.
            junk=ec.resolved_beat.junk, junk_reason=ec.resolved_beat.junk_reason,
        )
        records.append(record)
        specifics.append(spec)
    return records, specifics


def _group_id_for_hero(sync_groups: Dict[str, SyncGroup], file_id: str) -> Optional[str]:
    for gid, g in sync_groups.items():
        if g.authoritative_file_id == file_id:
            return gid
    return None
