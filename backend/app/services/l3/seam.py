"""
Seam significance: is the gap between two kept spans on ONE clip a weldable
CONTINUATION (play as one continuous unit) or a HARD cut (keep them separate)?

Deterministic + clip-relative, with NO tuned constants (deterministic-keep:
code owns quantitative/structural calls, never a hand-set threshold). One
function, two callers:

  * NOW -- ingest beat-bridge (``pass1.enforce_lattice_partition``): a spoken
    beat absorbs a brief wordless moment (a pause, a demonstrated action) into
    ONE continuous cut only when the seam between the lines is weldable.
  * FUTURE (documented hook, not wired here) -- timeline weld: when the editor
    drops two cuts adjacent, the SAME rule decides weld vs hard cut. Cross-clip
    is always hard (different footage), which is why ``same_clip`` is an
    explicit input rather than assumed.

A seam is HARD (a genuine break) when ANY of:
  1. cross-clip (different footage),
  2. a speaker change across it (two speakers are not one continuous beat),
  3. a shot/scene boundary or wipe/degenerate transition inside the gap -- read
     off the atoms' OWN boundary reasons; an energy-regime (R_ACTION) edge is
     NOT a break, it's continuous footage that merely gets livelier,
  4. a flagged production break inside the gap (a pass-1 junk suspect -- cue,
     reset, dead air),
  5. MAGNITUDE BACKSTOP: the gap is longer than the speech it would bridge
     (``gap_ms > bridged_speech_ms``) -- a structural 1:1 (absorb only when
     there is at least as much speech as connective tissue), never a knob.

Otherwise it is one continuous take -> WELDABLE.

Boundary reasons that count as a real break. These mirror the ``R_*`` reason
strings ``lattice.py`` stamps on an atom's ``state_in``/``state_out``; kept as
literals here so this module has no import cycle back into the lattice.
"""
from __future__ import annotations

from dataclasses import dataclass

# The atom boundary-reason strings that mark a real break: a hard shot cut
# (base_cuts.R_SHOT) or a transition (lattice.R_WIPE / R_DEGENERATE). An
# energy-regime edge (R_ACTION) or a speech/clip edge is NOT a break -- the
# footage is continuous across them. Kept as literals (matching those R_*
# values) so this module stays import-cycle-free; pass 1 matches atom
# state_in/state_out against this set.
BREAK_BOUNDARY_REASONS = frozenset({"shot_cut", "wipe", "degenerate"})


@dataclass(frozen=True)
class Seam:
    """Everything needed to judge one seam. All quantitative fields are
    milliseconds measured by code; the categorical fields come from signals
    (diarization, atom boundary reasons, pass-1 junk), never an LLM number."""
    same_clip: bool
    same_speaker: bool
    gap_ms: int                     # wordless span between the two kept spans
    bridged_speech_ms: int          # left + right speech wall-clock being bridged
    has_scene_or_transition: bool   # a break-type atom boundary lands inside the gap
    has_flagged_break: bool         # a pass-1 junk suspect overlaps the gap


@dataclass(frozen=True)
class SeamVerdict:
    weldable: bool
    reason: str


def classify_seam(seam: Seam) -> SeamVerdict:
    """Weldable (one continuous unit) or hard (keep separate), plus the reason.
    Pure -- trivially unit-tested (see ``scripts/test_seam.py``)."""
    if not seam.same_clip:
        return SeamVerdict(False, "cross-clip (different footage)")
    if not seam.same_speaker:
        return SeamVerdict(False, "speaker change across the seam")
    if seam.has_scene_or_transition:
        return SeamVerdict(False, "shot/scene boundary or transition inside the gap")
    if seam.has_flagged_break:
        return SeamVerdict(False, "a flagged production break (cue/reset/dead air) in the gap")
    if seam.gap_ms > seam.bridged_speech_ms:
        return SeamVerdict(False, "gap is longer than the speech it would bridge")
    return SeamVerdict(True, "continuous take")
