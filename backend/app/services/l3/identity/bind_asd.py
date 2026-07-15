"""
Voice -> face binding via Active Speaker Detection (asd_identity.plan.md),
replacing the Gemini clip-judging pass (identity/voice_id.py, deleted)
entirely. No model call anywhere in this module -- ASD (app.services.l1.
active_speaker) already produced each face track's deterministic `speaking`
timeline at L1; this module's whole job is INTERSECTING that with
diarization turns and picking, per voice, the person whose tracked face was
speaking during that voice's turns.

Magnitude aggregation (NOT an equal per-file vote). For each global voice
(`voice_of` maps every (file, local_speaker) onto a voice) we sum, in
milliseconds, how much each candidate person's tracks were ASD-speaking
inside that voice's turns, across EVERY file the voice appears in. The person
with the most cumulative overlap wins the voice, provided the win clears a
margin over the runner-up.

Why magnitude and not one-vote-per-file: in a multi-cam podcast each camera
frames ONE person but its mic records EVERYONE, so a single-face clip's lone
face overlaps BOTH voices' turns (a listener's jaw/reaction motion
coincidentally tracks the loud dual-speaker audio envelope). An equal
per-(file, speaker) vote lets those muddy clips -- where the on-camera person
barely edges the wrong voice -- cancel out the high-signal clips where the
discrimination is decisive (one real clip: 173s of overlap for the true voice
vs 20s for the other), leaving every voice a 2-2 tie and unbound. Summing ms
lets the confident clips dominate, which is exactly what recovers the true
speaker. A genuine cross-camera conflict (two people each drawing comparable
overlap for one voice) still fails the margin and stays unbound rather than
guessing -- the exact failure mode this plan exists to resolve deterministically
instead of via a stochastic model call.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.services.l1.active_speaker import FaceTrack

# A voice needs at least this much cumulative ASD-speaking overlap (summed ms,
# across all files) with its winning person before it binds -- below it there's
# no real signal (stray reaction motion), so the voice stays unbound/off-camera.
MIN_OVERLAP_MS = 1000
# The winning person's overlap must beat the runner-up person's by this ratio,
# else the voice is a genuine cross-camera conflict and stays unbound (never
# guess a speaker). Conservative on purpose: a WRONG bind stamps the wrong face
# as the speaker (a visible error), whereas leaving it unbound just yields an
# honest id-less cut. Tunable once more real footage validates it (asd_identity.
# plan.md SS12's open question).
MIN_MARGIN_RATIO = 1.2


def _overlap_ms(a_s: int, a_e: int, b_s: int, b_e: int) -> int:
    return max(0, min(a_e, b_e) - max(a_s, b_s))


def _speaking_overlap_ms(track: FaceTrack, turn_s: int, turn_e: int) -> int:
    """Total ms of `track`'s ASD-speaking intervals that fall inside
    [turn_s, turn_e)."""
    return sum(_overlap_ms(turn_s, turn_e, iv.start_ms, iv.end_ms) for iv in track.speaking)


def bind(
    turns_by_file: Dict[str, List[Tuple[int, int, str]]],
    voice_of: Dict[Tuple[str, str], str],
    face_tracks_by_file: Dict[str, List[FaceTrack]],
    track_to_person: Dict[Tuple[str, int], str],
) -> Tuple[Dict[str, Optional[str]], set]:
    """The whole binding pipeline. Returns `(owner_by_voice, off_camera_
    voices)`. `owner_by_voice` covers every voice that had ANY track overlap
    at all (bound -> person id, unbound -> None); a voice no tracked face ever
    spoke over is NOT in this dict -- the caller (identity/apply.py) reconciles
    that against the full voice roster, exactly like the pre-ASD contract."""
    turns_by_speaker: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    for fid, turns in turns_by_file.items():
        for start_ms, end_ms, local in turns:
            if not local:
                continue
            turns_by_speaker.setdefault((fid, local), []).append((int(start_ms), int(end_ms)))

    # voice -> person -> cumulative ASD-speaking overlap (ms), summed over files.
    overlap_by_voice: Dict[str, Dict[str, int]] = {}
    for (fid, local), turns in sorted(turns_by_speaker.items()):
        voice = voice_of.get((fid, local))
        if voice is None:
            continue
        for track in face_tracks_by_file.get(fid, []):
            person = track_to_person.get((fid, track.track_id))
            if person is None:
                continue
            ov = sum(_speaking_overlap_ms(track, s, e) for s, e in turns)
            if ov > 0:
                per = overlap_by_voice.setdefault(voice, {})
                per[person] = per.get(person, 0) + ov

    owner_by_voice: Dict[str, Optional[str]] = {}
    off_camera: set = set()
    for voice, per_person in sorted(overlap_by_voice.items()):
        # Rank persons by overlap desc, person id as a stable tiebreak.
        ranked = sorted(per_person.items(), key=lambda kv: (-kv[1], kv[0]))
        winner, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0
        if top < MIN_OVERLAP_MS or (second > 0 and top < second * MIN_MARGIN_RATIO):
            owner_by_voice[voice] = None
            off_camera.add(voice)
        else:
            owner_by_voice[voice] = winner
    return owner_by_voice, off_camera
