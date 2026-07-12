"""
Derive `on_camera`, persist the identity map (identity_map.plan.md Phase 3).
Called from `ingest.run_ingest` after `pass2.*`, before `post.assemble_
cut_records` -- `on_camera` also feeds `total_quality` there, so the
rewrite must land before assembly for the derived fact to actually count.

The single entry point, `run(...)`, does the whole bind -> reconcile ->
rewrite pipeline and returns `(new_pass2_output, persisted_payload)`. Never
fabricates: a file with no confident binding keeps pass 2a's original
per-still `on_camera` guess untouched.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from app.services.l3.identity.bind import Binding, bind_file, bind_outlook_group
from app.services.l3.identity.reconcile import reconcile
from app.services.l3.lattice import Lattice
from app.services.l3.pass2 import Pass2Cut, Pass2Output


def _bind_all(
    lattices: Dict[str, Lattice],
    motion_by_file: Dict[str, dict],
    groups: Dict[str, Dict[str, Any]],
) -> Dict[str, Binding]:
    fid_to_group = {fid: gid for gid, grp in groups.items() for fid in grp["members"]}
    bindings: Dict[str, Binding] = {}

    for grp in groups.values():
        member_ids = sorted(grp["members"])
        turns_by_file = {fid: lattices[fid].turns for fid in member_ids if fid in lattices}
        energy_by_file = {fid: (motion_by_file.get(fid) or {}).get("action_energy") or [] for fid in member_ids}
        hop_by_file = {fid: int((motion_by_file.get(fid) or {}).get("hop_ms") or 0) for fid in member_ids}
        bindings.update(bind_outlook_group(member_ids, turns_by_file, energy_by_file, hop_by_file))

    for fid, lattice in lattices.items():
        if fid in fid_to_group:
            continue
        m = motion_by_file.get(fid) or {}
        bindings[fid] = bind_file(lattice.turns, m.get("action_energy") or [], int(m.get("hop_ms") or 0))

    return bindings


def _cuts_by_file(cuts: List[Pass2Cut]) -> Dict[str, List[Pass2Cut]]:
    out: Dict[str, List[Pass2Cut]] = {}
    for c in cuts:
        out.setdefault(c.file_id, []).append(c)
    return out


def _multi_person_lone_files(cuts_by_file: Dict[str, List[Pass2Cut]], grouped_ids: Set[str]) -> Set[str]:
    """Phase 1's documented edge case: a single (non-grouped) camera framing
    >1 person has no motion basis to say WHICH of them is talking --
    whole-frame `action_energy` can't separate them. Detected via >1
    distinct people `position` across the file's own cuts. Grouped files are
    exempt: the whole outlook premise is one camera per person, so a
    genuinely multi-person angle there is out of scope, not this guard's job."""
    flagged: Set[str] = set()
    for fid, cuts in cuts_by_file.items():
        if fid in grouped_ids:
            continue
        positions = {
            p.get("position") for c in cuts for p in (c.people or []) if p.get("position")
        }
        if len(positions) > 1:
            flagged.add(fid)
    return flagged


def _people_by_file(cuts_by_file: Dict[str, List[Pass2Cut]]) -> tuple[Dict[str, List[dict]], Dict[str, List[str]]]:
    appearances: Dict[str, List[dict]] = {}
    descriptions: Dict[str, List[str]] = {}
    for fid, cuts in cuts_by_file.items():
        for c in cuts:
            for p in (c.people or []):
                appearances.setdefault(fid, []).append(p.get("appearance") or {})
                desc = p.get("description")
                if desc:
                    descriptions.setdefault(fid, []).append(desc)
    return appearances, descriptions


def _rewrite_on_camera(cuts: List[Pass2Cut], bound_voice_by_file: Dict[str, Binding]) -> List[Pass2Cut]:
    out: List[Pass2Cut] = []
    for cut in cuts:
        if cut.kind != "speech" or not cut.speaker:
            out.append(cut)  # video cut, or no speaker at all -- unchanged
            continue
        binding = bound_voice_by_file.get(cut.file_id)
        voice = binding.voice if binding else None
        if voice is None:
            out.append(cut)  # no confident binding -- keep pass 2a's per-still guess
            continue
        speakers = [s.strip() for s in cut.speaker.split(",") if s.strip()]
        out.append(cut.model_copy(update={"on_camera": voice in speakers}))
    return out


def run(
    pass2_output: Pass2Output,
    lattices: Dict[str, Lattice],
    motion_by_file: Dict[str, dict],
    groups: Dict[str, Dict[str, Any]],
) -> tuple[Pass2Output, Dict[str, Any]]:
    """The whole Phase 1-3 pipeline. `groups` is `sync.lattice_merge.
    outlook_groups`'s output (`{group_id: {"auth", "members"}}`) -- the SAME
    grouping the speech lattice merge already used, so binding and identity
    share one notion of "these cameras are one moment"."""
    cuts_by_file = _cuts_by_file(pass2_output.cuts)
    grouped_ids = {fid for grp in groups.values() for fid in grp["members"]}

    bindings = _bind_all(lattices, motion_by_file, groups)
    for fid in _multi_person_lone_files(cuts_by_file, grouped_ids):
        bindings[fid] = Binding(voice=None, confidence=0.0)

    appearances_by_file, descriptions_by_file = _people_by_file(cuts_by_file)
    bound_voice_by_file: Dict[str, Optional[str]] = {fid: b.voice for fid, b in bindings.items()}
    identity = reconcile(appearances_by_file, descriptions_by_file, bound_voice_by_file, groups)

    new_cuts = _rewrite_on_camera(pass2_output.cuts, bindings)

    payload = {
        "persons": identity["persons"],
        "file_person": identity["file_person"],
        "bound_voice": {fid: v for fid, v in bound_voice_by_file.items() if v},
        "oncam": identity["oncam"],
        "alias": {f"{fid}|{voice}": display for (fid, voice), display in identity["alias"].items()},
    }
    return Pass2Output(cuts=new_cuts), payload
