"""
Voice -> face binding via Active Speaker Detection (asd_identity.plan.md),
replacing the Gemini clip-judging pass (identity/voice_id.py, deleted)
entirely. No model call anywhere in this module -- ASD (app.services.l1.
active_speaker) already produced each face track's deterministic `speaking`
timeline at L1; this module's whole job is INTERSECTING that with
diarization turns and voting.

Two-level aggregation:
  1. Per (file_id, local_speaker): sum ASD-speaking overlap-ms, across ALL
     of that local speaker's turns in that file, per candidate person (via
     `track_to_person`) -- the person with the most total overlap is this
     (file, speaker)'s one vote. A local speaker with zero overlap against
     any track (pure narration, no face ever tracks its speech) casts no
     vote at all.
  2. Per GLOBAL voice (`voice_of` may map several (file, local_speaker)
     pairs -- e.g. the same person diarized separately in two clips, or an
     outlook group's mirrored turns -- onto one voice): majority + margin
     across those (file, speaker) votes, same contract identity/speaker_
     pass.aggregate_votes proved (MIN_VOTE_SHARE/MIN_VOTE_MARGIN, ported
     here unchanged). A genuine conflict (different cameras' tracks
     disagreeing about who's on screen for the same voice) stays unbound
     rather than guessing -- the exact failure mode this whole plan exists
     to resolve deterministically instead of via a stochastic model call.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.services.l1.active_speaker import FaceTrack

# Same majority+margin contract this codebase has used for every voice bind
# so far (identity/speaker_pass.aggregate_votes, then identity/voice_id.
# bind_from_verdicts) -- share computed over OPINIONATED (file, speaker)
# votes only; a genuine conflict (two people each drawing votes) blocks a
# bind, but a single "nobody overlaps" case never dilutes an otherwise-clear
# winner (there's nothing to dilute -- it simply casts no vote).
MIN_VOTE_SHARE = 0.5
MIN_VOTE_MARGIN = 1


def _overlap_ms(a_s: int, a_e: int, b_s: int, b_e: int) -> int:
    return max(0, min(a_e, b_e) - max(a_s, b_s))


def _speaking_overlap_ms(track: FaceTrack, turn_s: int, turn_e: int) -> int:
    """Total ms of `track`'s ASD-speaking intervals that fall inside
    [turn_s, turn_e)."""
    return sum(_overlap_ms(turn_s, turn_e, iv.start_ms, iv.end_ms) for iv in track.speaking)


def _local_speaker_owner(
    file_id: str,
    turns: List[Tuple[int, int]],
    face_tracks_by_file: Dict[str, List[FaceTrack]],
    track_to_person: Dict[Tuple[str, int], str],
) -> Optional[str]:
    """The person whose track's ASD-speaking intervals overlap this local
    speaker's turns the MOST, summed across every turn in `turns` -- None
    when no track overlaps at all (the speaker never coincides with any
    tracked, speaking face -- honest off-camera/narration)."""
    totals: Dict[str, int] = {}
    for track in face_tracks_by_file.get(file_id, []):
        person = track_to_person.get((file_id, track.track_id))
        if person is None:
            continue
        overlap = sum(_speaking_overlap_ms(track, s, e) for s, e in turns)
        if overlap > 0:
            totals[person] = totals.get(person, 0) + overlap
    if not totals:
        return None
    return max(totals.items(), key=lambda kv: kv[1])[0]


def bind(
    turns_by_file: Dict[str, List[Tuple[int, int, str]]],
    voice_of: Dict[Tuple[str, str], str],
    face_tracks_by_file: Dict[str, List[FaceTrack]],
    track_to_person: Dict[Tuple[str, int], str],
) -> Tuple[Dict[str, Optional[str]], set]:
    """The whole binding pipeline. Returns `(owner_by_voice, off_camera_
    voices)`. `owner_by_voice` covers every voice that had at least one
    (file, local_speaker) resolve to SOME person (bound -> person id,
    unbound -> None); a voice with zero resolving (file, speaker) pairs at
    all is NOT in this dict -- the caller (identity/apply.py) reconciles
    that against the full voice roster, exactly like the pre-ASD contract."""
    turns_by_speaker: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for fid, turns in turns_by_file.items():
        for start_ms, end_ms, local in turns:
            if not local:
                continue
            turns_by_speaker.setdefault((fid, local), []).append((int(start_ms), int(end_ms)))

    votes_by_voice: Dict[str, List[str]] = {}
    for (fid, local), turns in sorted(turns_by_speaker.items()):
        voice = voice_of.get((fid, local))
        if voice is None:
            continue
        owner = _local_speaker_owner(fid, turns, face_tracks_by_file, track_to_person)
        if owner is not None:
            votes_by_voice.setdefault(voice, []).append(owner)

    owner_by_voice: Dict[str, Optional[str]] = {}
    off_camera: set = set()
    for voice, votes in sorted(votes_by_voice.items()):
        counts: Dict[str, int] = {}
        for p in votes:
            counts[p] = counts.get(p, 0) + 1
        opinionated = sum(counts.values())
        ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        winner, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0
        if top / opinionated < MIN_VOTE_SHARE or top - second < MIN_VOTE_MARGIN:
            owner_by_voice[voice] = None
            off_camera.add(voice)
        else:
            owner_by_voice[voice] = winner
    return owner_by_voice, off_camera
