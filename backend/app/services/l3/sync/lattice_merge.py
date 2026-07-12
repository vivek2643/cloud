"""
Authoritative speech lattice merge (audio_sync.plan.md SS7.1): for one
"angle" file in a synced group, replace its own (independently transcribed,
per-camera-drifting) words/turns with the group's AUTHORITATIVE source's own
words/turns, re-based onto this angle's own file-clock. Atoms (the video
side) are left untouched -- they're genuinely per-angle (SS7.4: different
framing/movement is real content, not duplication).

This is the concrete fix for SS0.1's stated bug: today each camera's own
transcript (different mic, different STT run) produces MINUTELY-DIFFERENT
word timestamps and filler flags for the literal same spoken words -- pass 1
sees N near-but-not-quite-matching transcripts and has to *guess* (token
overlap) which clips are the same moment. After this swap, every angle in a
group carries the byte-identical (re-based) word list, so pass 1's existing
cross-clip grouping has an EXACT match to work with instead of an
approximate one -- see also `sync_hint_line` for the explicit membership
hint fed alongside (SS7.3).
"""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from app.services.l3.diarize import Turn
from app.services.l3.lattice import Lattice, build_atoms, load_motion_scene
from app.services.l3.sync.detect import HIGH_CONFIDENCE_THRESHOLD


def _retime_words(words: List[dict], delta_ms: int) -> List[dict]:
    return [
        {**w, "start_ms": int(w["start_ms"]) + delta_ms, "end_ms": int(w["end_ms"]) + delta_ms}
        for w in words
    ]


def _retime_turns(turns: List[Turn], delta_ms: int) -> List[Turn]:
    return [(int(s) + delta_ms, int(e) + delta_ms, spk) for s, e, spk in turns]


def authoritative_view(angle_lattice: Lattice, authoritative: Lattice, delta_ms: int) -> Lattice:
    """`angle_lattice` with its speech side replaced by `authoritative`'s
    words/turns/hints, shifted by `delta_ms` onto `angle_lattice`'s own clock.
    `delta_ms` = `authoritative_offset_ms - angle_offset_ms` (the group-clock
    term cancels: `group_ms = auth_ms + auth_offset == angle_ms + angle_offset`
    => `angle_ms = auth_ms + (auth_offset - angle_offset)`; see
    `sync/detect.py`'s offset convention). `hints` are copied UNCHANGED:
    `speech_hints` text only ever references word INDEX positions and
    relative gap durations, both invariant under a constant ms shift.

    Atoms are REBUILT (not carried over) against the re-based authoritative
    words, using THIS angle's own motion/scene. That keeps them genuinely
    per-angle (SS7.4: framing/movement is real per-angle content, sourced from
    this angle's video) while restoring `build_atoms`'s structural guarantee
    that a video atom never overlaps speech: the original atoms were carved
    against this angle's OWN (now-discarded) word timings, so after the swap a
    (shifted) authoritative word span could straddle a stale atom -- which
    breaks the speech/atom partition `enforce_lattice_partition` asserts
    (`_no_speech_cut_swallows_atoms`). Re-deriving them against the words they
    now coexist with makes the partition hold by construction again."""
    words = _retime_words(authoritative.words, delta_ms)
    turns = _retime_turns(authoritative.turns, delta_ms)
    motion, scene = load_motion_scene(angle_lattice.file_id)
    atoms = build_atoms(angle_lattice.file_id, angle_lattice.duration_ms,
                        motion, scene, turns, words=words)
    return Lattice(
        file_id=angle_lattice.file_id,
        duration_ms=angle_lattice.duration_ms,
        words=words,
        turns=turns,
        hints=list(authoritative.hints),
        atoms=atoms,
    )


def _is_high_conf(member: Dict[str, Any]) -> bool:
    """A member we trust enough to fold into an outlook group. A committed
    manual nudge is trusted outright (a human aligned it); an auto alignment
    must clear the cross-correlation confidence gate. Below the gate the member
    is left as an ordinary independent clip -- SS7's "never fabricate a
    misaligned outlook"."""
    if member.get("aligned_by") == "manual":
        return True
    c = member.get("confidence")
    return c is not None and float(c) >= HIGH_CONFIDENCE_THRESHOLD


def high_conf_groups(sync_by_file: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """`sync.store.sync_groups_for_files` collapsed to the groups that actually
    have >= 2 confidently-aligned members -- the ONLY ones we treat as synced
    (speech-swap, mirror beats, form outlooks). Shape: `{group_id: {"auth":
    authoritative_file_id, "members": {high-conf file_ids}}}`. A group with
    fewer than two high-confidence members is dropped whole (its files behave
    as ordinary, independent clips)."""
    groups: Dict[str, Dict[str, Any]] = {}
    for fid, g in sync_by_file.items():
        gid = g["group_id"]
        grp = groups.setdefault(gid, {"auth": g["authoritative_audio_file_id"], "members": set()})
        if _is_high_conf(g["members"][fid]):
            grp["members"].add(fid)
    return {gid: grp for gid, grp in groups.items() if len(grp["members"]) >= 2}


def apply_sync_groups(
    file_rows: List[Tuple[str, str, int, Lattice]],
    sync_by_file: Dict[str, Dict[str, Any]],
    groups_hi: Dict[str, Dict[str, Any]],
) -> Tuple[List[Tuple[str, str, int, Lattice]], Dict[str, str], Set[str]]:
    """`file_rows` (as `pass1.load_project_file_rows` returns) with every
    high-confidence synced angle's Lattice speech-swapped onto its group's
    authoritative source (SS7.1/7.2), plus the pass-1 sync-hint text per synced
    file (SS7.3) and the set of synced file_ids (SS7.6's `synced_file_ids`,
    threaded into `enforce_lattice_partition`/`post.assemble_cut_records`).

    Only files in a `groups_hi` group are touched: a low-confidence member (or
    a group with fewer than two confident members) is dropped by
    `high_conf_groups` and passes through untouched -- both the no-declared-sync
    no-op guarantee (SS2) and the "never fabricate a misaligned outlook"
    fail-safe (SS7). `sync_by_file` is still consulted for the per-member
    offsets used to compute the re-basing delta."""
    if not groups_hi:
        return file_rows, {}, set()

    hi_to_group = {fid: gid for gid, grp in groups_hi.items() for fid in grp["members"]}
    lattices_by_id = {fid: lat for fid, _, _, lat in file_rows}
    hints: Dict[str, str] = {}
    synced_ids: Set[str] = set()
    out: List[Tuple[str, str, int, Lattice]] = []

    for fid, name, dur, lattice in file_rows:
        gid = hi_to_group.get(fid)
        if gid is None:
            out.append((fid, name, dur, lattice))
            continue
        grp = groups_hi[gid]
        auth_fid = grp["auth"]
        auth_lattice = lattices_by_id.get(auth_fid)
        members = sync_by_file[fid]["members"]
        other_ids = sorted(m for m in grp["members"] if m != fid)
        synced_ids.add(fid)
        hints[fid] = sync_hint_line(fid, other_ids)
        if auth_lattice is None or auth_fid not in members or fid not in members:
            # Authoritative source not (yet) loadable, or a malformed group
            # row -- fail open: keep this angle's own lattice untouched
            # rather than block ingest on a sync-data problem.
            out.append((fid, name, dur, lattice))
            continue
        delta_ms = int(members[auth_fid]["offset_ms"]) - int(members[fid]["offset_ms"])
        out.append((fid, name, dur, authoritative_view(lattice, auth_lattice, delta_ms)))

    return out, hints, synced_ids


def sync_hint_line(angle_file_id: str, other_angle_ids: List[str]) -> str:
    """A factual, non-prescriptive prompt hint (matches footage_map.py's own
    `·alt-PIC` framing convention -- "never a verdict, just where else this
    lives") to feed pass 1 alongside a synced angle's clip block (SS7.3
    "feed pass1 the known simultaneity ... so it does NOT emit
    TakeCandidates for them"). Lists OTHER members only, never itself."""
    others = ", ".join(fid[:8] for fid in other_angle_ids if fid != angle_file_id)
    return (
        f"SYNC: this clip is audio-synced with camera angle(s) {others} -- "
        "same recorded moment, not a separate take. Its transcript below is "
        "shared verbatim (re-based) across all of them; do not propose a "
        "take-candidate for these, they are already known to be simultaneous."
    )
