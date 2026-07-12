"""
Authoritative speech lattice merge: for one angle file in an OUTLOOK group
(clips a user declared to be alternate cameras of the same moment), replace
its own (independently transcribed, per-camera-drifting) words/turns with the
group's AUTHORITATIVE source's own words/turns, re-based onto this angle's own
file-clock. Atoms (the video side) are left untouched -- they're genuinely
per-angle (different framing/movement is real content, not duplication).

This fixes the underlying bug: each camera's own transcript (different mic,
different STT run) produces MINUTELY-DIFFERENT word timestamps and filler flags
for the literal same spoken words -- pass 1 sees N near-but-not-quite-matching
transcripts and has to *guess* (token overlap) which clips are the same moment.
After this swap, every angle in a group carries the byte-identical (re-based)
word list, so pass 1's cross-clip grouping has an EXACT match to work with
instead of an approximate one -- see also `outlook_hint_line` for the explicit
membership hint fed alongside.
"""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from app.services.l3.diarize import Turn
from app.services.l3.lattice import Lattice, build_atoms, load_motion_scene


def _retime_words(words: List[dict], delta_ms: int) -> List[dict]:
    return [
        {**w, "start_ms": int(w["start_ms"]) + delta_ms, "end_ms": int(w["end_ms"]) + delta_ms}
        for w in words
    ]


def _retime_turns(turns: List[Turn], delta_ms: int) -> List[Turn]:
    return [(int(s) + delta_ms, int(e) + delta_ms, spk) for s, e, spk in turns]


def _clip_words_to_footage(words: List[dict], duration_ms: int) -> List[dict]:
    """The (start-sorted, already re-based) authoritative words restricted to
    this angle's own footage ``[0, duration_ms]``. A shorter or
    differently-started angle simply wasn't rolling for part of the
    conversation, so those authoritative words have no picture on it.

    Only the TRAILING out-of-footage words are dropped (start at/after the end
    of footage), which leaves every kept word's INDEX unchanged -- so a beat
    that lands fully inside this angle's footage keeps the exact word-index span
    it has on a full-length angle, which is what lets the outlook grouping match
    the same beat across angles. A kept word that merely runs a hair past the
    edge, or a head word that re-based to before 0 (a later-starting angle), is
    clamped into footage rather than dropped, so indices stay intact."""
    cutoff = len(words)
    for i, w in enumerate(words):
        if int(w["start_ms"]) >= duration_ms:
            cutoff = i
            break
    clipped: List[dict] = []
    for w in words[:cutoff]:
        s = max(0, int(w["start_ms"]))
        e = min(duration_ms, max(int(w["end_ms"]), s))
        clipped.append({**w, "start_ms": s, "end_ms": e})
    return clipped


def _clip_turns_to_footage(turns: List[Turn], duration_ms: int) -> List[Turn]:
    """``_clip_words_to_footage`` for diarization turns -- drop turns that start
    beyond footage, clamp the rest into ``[0, duration_ms]`` (so ``build_atoms``
    never carves against a turn this angle didn't film)."""
    out: List[Turn] = []
    for s, e, spk in turns:
        if int(s) >= duration_ms:
            continue
        s2 = max(0, int(s))
        out.append((s2, min(duration_ms, max(int(e), s2)), spk))
    return out


def authoritative_view(angle_lattice: Lattice, authoritative: Lattice, delta_ms: int) -> Lattice:
    """`angle_lattice` with its speech side replaced by `authoritative`'s
    words/turns/hints, shifted by `delta_ms` onto `angle_lattice`'s own clock.
    `delta_ms` = `authoritative_offset_ms - angle_offset_ms` (the group-clock
    term cancels: `group_ms = auth_ms + auth_offset == angle_ms + angle_offset`
    => `angle_ms = auth_ms + (auth_offset - angle_offset)`; see
    `sync/detect.py`'s offset convention). `hints` are copied UNCHANGED:
    `speech_hints` text only ever references word INDEX positions and
    relative gap durations, both invariant under a constant ms shift.

    The re-based words/turns are CLIPPED to this angle's own footage
    (`_clip_words_to_footage`): an angle that is shorter than, or started later
    than, the authoritative source only filmed part of the conversation, so the
    authoritative words outside its `[0, duration]` window have no picture on it
    and must not appear in its lattice (else the image plan asks for a frame the
    clip doesn't have, and coverage-fill recovers a cut past the clip's end).
    Clipping drops only trailing out-of-footage words, so every kept word keeps
    its index -- an interior beat matches the same word-index span across angles.

    Atoms are REBUILT (not carried over) against those clipped words, using THIS
    angle's own motion/scene. That keeps them genuinely per-angle (framing/
    movement is real per-angle content, sourced from this angle's video) while
    restoring `build_atoms`'s structural guarantee that a video atom never
    overlaps speech: the original atoms were carved against this angle's OWN
    (now-discarded) word timings, so after the swap a (shifted) authoritative
    word span could straddle a stale atom -- which breaks the speech/atom
    partition `enforce_lattice_partition` asserts (`_no_speech_cut_swallows_
    atoms`). Re-deriving them against the words they now coexist with makes the
    partition hold by construction again."""
    dur = angle_lattice.duration_ms
    words = _clip_words_to_footage(_retime_words(authoritative.words, delta_ms), dur)
    turns = _clip_turns_to_footage(_retime_turns(authoritative.turns, delta_ms), dur)
    motion, scene = load_motion_scene(angle_lattice.file_id)
    atoms = build_atoms(angle_lattice.file_id, dur, motion, scene, turns, words=words)
    return Lattice(
        file_id=angle_lattice.file_id,
        duration_ms=angle_lattice.duration_ms,
        words=words,
        turns=turns,
        hints=list(authoritative.hints),
        atoms=atoms,
    )


def outlook_groups(sync_by_file: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """`sync.store.sync_groups_for_files` collapsed to `{group_id: {"auth":
    authoritative_file_id, "members": {ALL member file_ids}}}`, keeping only
    groups with >= 2 members (a lone member is a no-op group).

    Every member of a user-declared group is an OUTLOOK of the others -- a
    different camera on the same moment -- by DEFINITION, not by how well its
    audio happened to cross-correlate. Alignment confidence only affects how
    tightly a member's picture lines up (and drives the nudge UI); it never
    demotes a member out of the group. That demotion was the bug: an excluded
    member fell back to an ordinary clip, so pass 1 then mis-grouped it as a
    TAKE (with its own, non-authoritative audio) instead of an outlook. Takes
    and outlooks are distinct, so a declared group's members are never takes of
    each other."""
    groups: Dict[str, Dict[str, Any]] = {}
    for fid, g in sync_by_file.items():
        gid = g["group_id"]
        grp = groups.setdefault(gid, {"auth": g["authoritative_audio_file_id"], "members": set()})
        grp["members"].add(fid)
    return {gid: grp for gid, grp in groups.items() if len(grp["members"]) >= 2}


def apply_outlook_groups(
    file_rows: List[Tuple[str, str, int, Lattice]],
    sync_by_file: Dict[str, Dict[str, Any]],
    groups: Dict[str, Dict[str, Any]],
) -> Tuple[List[Tuple[str, str, int, Lattice]], Dict[str, str], Set[str]]:
    """`file_rows` (as `pass1.load_project_file_rows` returns) with every
    outlook-group angle's Lattice speech-swapped onto its group's authoritative
    source, plus the pass-1 outlook-membership hint per angle and the set of
    grouped file_ids (threaded into `enforce_lattice_partition`/
    `post.assemble_cut_records` so a shot boundary never splits a shared beat).

    ALL members of a `groups` group are swapped -- each carries the
    authoritative words re-based onto its clock and clipped to its own footage
    (`authoritative_view`), so a member that only filmed part of the moment
    carries the matching PREFIX of the authoritative words (shared indices).
    This is correct regardless of a member's alignment confidence; only the ms
    offset it re-bases onto is as good as that member's alignment. `sync_by_file`
    supplies the per-member offsets. A file in no group is untouched (a project
    with no declared groups behaves byte-identical to before)."""
    if not groups:
        return file_rows, {}, set()

    fid_to_group = {fid: gid for gid, grp in groups.items() for fid in grp["members"]}
    lattices_by_id = {fid: lat for fid, _, _, lat in file_rows}
    hints: Dict[str, str] = {}
    grouped_ids: Set[str] = set()
    out: List[Tuple[str, str, int, Lattice]] = []

    for fid, name, dur, lattice in file_rows:
        gid = fid_to_group.get(fid)
        if gid is None:
            out.append((fid, name, dur, lattice))
            continue
        grp = groups[gid]
        auth_fid = grp["auth"]
        auth_lattice = lattices_by_id.get(auth_fid)
        members = sync_by_file[fid]["members"]
        other_ids = sorted(m for m in grp["members"] if m != fid)
        grouped_ids.add(fid)
        hints[fid] = outlook_hint_line(fid, other_ids)
        if auth_lattice is None or auth_fid not in members or fid not in members:
            # Authoritative source not (yet) loadable, or a malformed group
            # row -- fail open: keep this angle's own lattice untouched
            # rather than block ingest on a grouping-data problem.
            out.append((fid, name, dur, lattice))
            continue
        delta_ms = int(members[auth_fid]["offset_ms"]) - int(members[fid]["offset_ms"])
        out.append((fid, name, dur, authoritative_view(lattice, auth_lattice, delta_ms)))

    return out, hints, grouped_ids


def outlook_hint_line(angle_file_id: str, other_angle_ids: List[str]) -> str:
    """A factual, non-prescriptive prompt hint (matches footage_map.py's own
    `·alt-PIC` framing convention -- "never a verdict, just where else this
    lives") fed to pass 1 alongside an angle's clip block. Lists OTHER members
    only, never itself. Tells the model these clips are alternate ANGLES of one
    moment (outlooks), so it must NOT propose a take-candidate among them -- an
    outlook is not a retake, and code owns the outlook grouping directly."""
    others = ", ".join(fid[:8] for fid in other_angle_ids if fid != angle_file_id)
    return (
        f"OUTLOOK: this clip is an alternate camera angle of the same moment as "
        f"{others} -- a simultaneous outlook, NOT a separate take. Its transcript "
        "below is shared verbatim (re-based) across all of them; do not propose a "
        "take-candidate among these, they are alternate angles the code groups."
    )
